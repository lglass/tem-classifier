"""
Run a trained CryoFormerLite model over a set of particles and write out
predicted labels + class probabilities.

Particles can come from a plain CSV (--input_csv) or be pulled live from a
CryoSPARC job (--job_id / --project_id). See models.py for the architecture.

Usage
-----
    python inference_cryoclassifier.py --model_dir models/exp_gt/ --input_csv particles.csv
    python inference_cryoclassifier.py --model_dir models/exp_gt/ --job_id J123 --project_id P4
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import CryoEMDataset
from label_utils import resolve_label_map
from models import CryoFormerLite
from utils_cs import initialize_cryosparc, load_env_variables

DEFAULT_IMAGE_SIZE = 128
DEFAULT_MODEL_DIR = Path("models/out")
DEFAULT_PROJECT_ID = "P1"
DEFAULT_PIXEL_SIZE_A = 2.053125
LABEL_MAP_FILENAME = "particle2label.json"


def find_checkpoint(model_dir: Path) -> Path:
    """Return the most recently modified `cryo_model*.pt(h)` checkpoint in `model_dir`."""
    patterns = ["cryo_model*.pt", "cryo_model*.pth"]
    candidates = [p for pattern in patterns for p in model_dir.glob(pattern)]
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint found in {model_dir}. Expected one of: {patterns}"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_model(model_path: Path, num_classes: int, img_size: int, pixel_size_A: float, device="cpu"):
    print(f"Loading model {model_path}")
    model = CryoFormerLite(num_classes=num_classes, img_size=img_size, pixel_size_A=pixel_size_A)
    model = model.to(device)

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = _unwrap_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=True)

    model.eval()
    return model


def _unwrap_state_dict(checkpoint) -> dict:
    """Accept a bare state_dict, or one nested under a "model"/"model_state_dict" key."""
    if isinstance(checkpoint, dict):
        for key in ("model", "model_state_dict"):
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint


def load_particles_from_cryosparc(job_id: str, project_id: str) -> pd.DataFrame:
    cs = initialize_cryosparc(load_env_variables(".env"))
    project = cs.find_project(project_id)
    particles = project.find_job(job_id).load_output("particles")
    return pd.DataFrame(particles.rows())


@torch.no_grad()
def predict(model, dataloader, device):
    """Run `model` over every batch in `dataloader`. Returns (pred_labels, pred_probs) as numpy arrays."""
    all_preds = []
    all_probs = []
    for batch in dataloader:
        # dataloader may return (images, labels) or images only
        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        images = images.to(device)

        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        all_preds.append(preds.cpu())
        all_probs.append(probs.cpu())

    return torch.cat(all_preds).numpy(), torch.cat(all_probs).numpy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_classes",
        type=int,
        default=None,
        help="Number of particle classes. Derived from --label_map if provided.",
    )
    parser.add_argument(
        "--label_map",
        type=Path,
        default=None,
        help=f"Path to label map JSON file. Defaults to <model_dir>/{LABEL_MAP_FILENAME}.",
    )
    parser.add_argument(
        "--input_csv",
        type=Path,
        default=None,
        help="Path to the input CSV file containing particle data.",
    )
    parser.add_argument(
        "--job_id",
        type=str,
        default=None,
        help="CryoSPARC job id that contains the particle data.",
    )
    parser.add_argument(
        "--project_id",
        type=str,
        default=DEFAULT_PROJECT_ID,
        help="CryoSPARC project id that contains the particle data.",
    )
    parser.add_argument("--output_csv", type=Path, default=None)
    parser.add_argument(
        "--project_dir",
        type=Path,
        default=None,
        help="Base directory for relative blob/path or fpath values.",
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Directory containing the trained model checkpoint.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="Image size in pixels (default: 128).",
    )
    parser.add_argument(
        "--pixel_size",
        type=float,
        default=DEFAULT_PIXEL_SIZE_A,
        help=f"Pixel size in Ångströms (default: {DEFAULT_PIXEL_SIZE_A}).",
    )
    return parser.parse_args()


def load_particles(args: argparse.Namespace) -> pd.DataFrame:
    if args.job_id:
        return load_particles_from_cryosparc(args.job_id, args.project_id)
    return pd.read_csv(args.input_csv)


def report_predictions(particles_df: pd.DataFrame, elapsed_seconds: float) -> None:
    num_images = len(particles_df)
    per_image_ms = (elapsed_seconds / max(1, num_images)) * 1000.0
    print(f"Inference time: {elapsed_seconds:.4f} s for {num_images} images ({per_image_ms:.2f} ms/image)")

    counts_column = "predicted_particle" if "predicted_particle" in particles_df else "predicted_label"
    print(particles_df[counts_column].value_counts().to_dict())


def main():
    args = parse_args()

    label_map_path = args.label_map or _default_label_map_path(args.model_dir)
    label_map, num_classes = resolve_label_map(label_map_path, args.num_classes)
    idx_to_name = {v: k for k, v in label_map.items()} if label_map else None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = find_checkpoint(args.model_dir)
    model = load_model(model_path, num_classes, img_size=args.image_size, pixel_size_A=args.pixel_size, device=device)

    particles_df = load_particles(args)
    particles_ds = CryoEMDataset(
        proj_dir=args.project_dir,
        df=particles_df,
        train=False,
        crop_size=args.image_size,
    )
    test_loader = DataLoader(
        particles_ds,
        batch_size=256,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        prefetch_factor=1,
        persistent_workers=False,
    )

    start_time = time.perf_counter()
    predicted_labels, predicted_probs = predict(model, test_loader, device)
    elapsed = time.perf_counter() - start_time

    particles_df["predicted_label"] = predicted_labels
    if idx_to_name is not None:
        particles_df["predicted_particle"] = particles_df["predicted_label"].map(idx_to_name)
    report_predictions(particles_df, elapsed)

    output_csv = args.output_csv or (args.model_dir / "predictions.csv")
    output_probs = output_csv.with_suffix(".probs.npy")
    particles_df.to_csv(output_csv, index=False)
    np.save(output_probs, predicted_probs)


def _default_label_map_path(model_dir: Path) -> Path | None:
    default_path = model_dir / LABEL_MAP_FILENAME
    return default_path if default_path.exists() else None


if __name__ == "__main__":
    main()
