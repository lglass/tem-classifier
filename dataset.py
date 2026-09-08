import json
from pathlib import Path
from typing import Optional

import mrcfile
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2


class NormalizePerImage(object):
    """
    Per-image z-score standardisation: subtracts the mean and divides by the
    standard deviation, so each image ends up roughly zero-mean, unit-variance.

    Args:
        eps (float): A small value to avoid division by zero if the image is
            (near-)constant valued.
    """

    def __init__(self, eps=1e-8):
        self.eps = eps

    def __call__(self, tensor):
        # Ensure input is a tensor
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Input should be a torch.Tensor. Got {type(tensor)}")

        im_mean = tensor.mean()
        im_std = tensor.std()

        if im_std > self.eps:
            return (tensor - im_mean) / (im_std + self.eps)

        # Handle constant-valued images (e.g., all black or all white)
        return torch.zeros_like(tensor)

    def __repr__(self):
        return self.__class__.__name__ + "()"


class CircularMask(object):
    """
    Applies a circular mask to a torch Tensor, replacing pixels outside the
    circle with a fill value. Not currently used in CryoEMAugmentation's
    pipeline, but available to add back in if edge artefacts (e.g. from
    neighbouring particles at the box corners) turn out to hurt training.

    Args:
        fill (str): "mean" to fill with the image's own mean value, anything
                    else to use `fill_value` directly.
        fill_value (float): The pixel value for the masked area when
                             fill != "mean" (default: 0.0).
        radius_ratio (float): Radius of the circle as a fraction of the
                              shortest side (default: 0.5 for full edge-to-edge).
    """

    def __init__(self, fill="mean", fill_value=0.0, radius_ratio=0.5):
        self.fill = fill
        self.fill_value = fill_value
        self.radius_ratio = radius_ratio

    def __call__(self, img):
        # Ensure we are working with a tensor
        if not isinstance(img, torch.Tensor):
            raise TypeError(f"Input should be a torch.Tensor. Got {type(img)}")

        # Get dimensions (Assumes C, H, W)
        c, h, w = img.shape

        # Create a grid of coordinates - clone to avoid memory aliasing issues
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij"
        )
        yy = yy.clone()
        xx = xx.clone()

        # Calculate distance from center
        # We adjust for aspect ratio if the image isn't square
        aspect_ratio = w / h
        if w > h:
            xx *= aspect_ratio
        else:
            yy /= aspect_ratio

        dist = torch.sqrt(xx**2 + yy**2)

        # "mean" fill is computed per-call (not cached on self) so repeated
        # use of the same CircularMask instance doesn't leak one image's
        # mean into the next.
        fill_value = img.mean().item() if self.fill == "mean" else self.fill_value

        # Create mask (1 within radius, 0 outside)
        sharpness = 10.0  # Higher value = sharper edge; lower value = softer edge
        cutoff = (
            self.radius_ratio * 2
        )  # Normalize cutoff to [0, 2] since dist is normalized
        mask = torch.sigmoid(sharpness * (cutoff - dist))
        mask = mask.to(img.dtype)

        # Apply mask - use where() to avoid aliasing
        result = img * mask + fill_value * (1 - mask)

        return result.contiguous()

    def __repr__(self):
        return f"{self.__class__.__name__}(fill={self.fill!r}, fill_value={self.fill_value}, radius_ratio={self.radius_ratio})"


class CryoEMAugmentation:
    """
    Data augmentation for cryo-EM particle images using torchvision v2
    transforms.

    Training pipeline: normalise, then randomly rotate (orientational
    invariance) and randomly crop within a slightly larger center crop
    (translational robustness). Evaluation pipeline: normalise and
    center-crop only, no randomness.

    Other transforms worth trying if training needs more regularisation:
    CircularMask (mask out box corners) and v2.GaussianNoise (noise-level
    jitter) — both defined/available but not wired in by default.
    """

    def __init__(self, crop_size: int = 128, train: bool = True):
        base_transforms = [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=False),
            NormalizePerImage(),
        ]
        if train:
            transforms = base_transforms + [
                v2.RandomRotation(degrees=360),
                v2.CenterCrop(crop_size + 10),
                v2.RandomCrop(crop_size),
            ]
        else:
            transforms = base_transforms + [v2.CenterCrop(crop_size)]

        self.transform = v2.Compose(transforms)

    def __call__(self, x) -> torch.Tensor:
        return self.transform(x)


class CryoEMDataset(Dataset):
    """Labelled particle images."""

    def __init__(
        self,
        proj_dir: Path,
        df: pd.DataFrame,
        train: bool = True,
        crop_size: int = 128,
    ):
        """
        Arguments:
            proj_dir: Base directory for relative paths.
            df: DataFrame containing label information.
            train: If True, apply augmentation transforms; otherwise, center-crop only.
        """
        self.base_dir = proj_dir
        self.data = df
        self.has_labels = "label" in self.data.columns

        # "blob_path" holds paths already resolved to absolute (as
        # setup_dataflow does before constructing this dataset); "blob/path"
        # is the raw CryoSPARC column of paths relative to proj_dir, used
        # when a dataset is built directly from a CryoSPARC-exported CSV.
        if "blob_path" in self.data.columns:
            self.paths = [str(Path(p)) for p in self.data["blob_path"].tolist()]
        else:
            self.paths = [
                str(self.base_dir / p) for p in self.data["blob/path"].tolist()
            ]

        self.idxs = self.data["blob/idx"].astype(int).tolist()
        self.labels = (
            self.data["label"].astype(int).tolist() if self.has_labels else None
        )

        # Build transform pipeline
        self.transform = CryoEMAugmentation(crop_size=crop_size, train=train)

        # Per-worker mmap cache: key=path, val=mrcfile mmap handle
        self._mrc_handles = {}
        self._max_open_files = 32

    def __len__(self):
        return len(self.paths)

    def _prune_handles_if_needed(self):
        if len(self._mrc_handles) < self._max_open_files:
            return
        for h in self._mrc_handles.values():
            try:
                h.close()
            except Exception:
                pass
        self._mrc_handles.clear()

    def _get_mrc_data(self, path: str) -> np.ndarray:
        handle = self._mrc_handles.get(path)
        if handle is None:
            self._prune_handles_if_needed()
            # read-only mmap; r+ can add needless overhead/risk
            handle = mrcfile.mmap(path, permissive=True, mode="r")
            self._mrc_handles[path] = handle
        return handle.data

    def __getitem__(self, idx: int):
        blob_path = self.paths[idx]
        blob_idx = self.idxs[idx]
        data = self._get_mrc_data(blob_path)
        image = data[blob_idx].copy()

        if self.transform:
            image = self.transform(image)

        if self.has_labels:
            return image, self.labels[idx]

        return image

    def __del__(self):
        for h in self._mrc_handles.values():
            try:
                h.close()
            except Exception:
                pass
        self._mrc_handles.clear()


def load_label_map(path: str) -> dict:
    """Load a label map JSON file.

    Format: {"ParticleName": model_index, ...}
    e.g. {"HSP60": 0, "Apoferritin": 1, "Empty": 2}
    Returns dict[str, int].
    """
    with open(path) as f:
        raw = json.load(f)
    return {k: int(v) for k, v in raw.items()}


def _make_loader(ds: Dataset, batch_size: int = 128, shuffle: bool = False) -> DataLoader:
    """DataLoader with the pinned-memory / persistent-worker settings shared
    by the train and validation loaders."""
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=2,
        prefetch_factor=2,
        persistent_workers=True,
    )


def setup_dataflow(
    project_dir: Path = Path("/mnt/data/CS-protein-mixture"),
    csv_path: str = "training.csv",
    test_size: float = 0.1,
    random_state: Optional[int] = None,
    label_map: Optional[dict] = None,
    crop_size: int = 128,
):
    """Load a CryoSPARC particles CSV, split it into stratified train/val
    sets, and wrap each half in a CryoEMDataset + DataLoader."""
    particles_df = pd.read_csv(csv_path)
    particles_df["blob_path"] = project_dir / particles_df["blob/path"]
    if label_map is not None:
        particles_df = particles_df[particles_df["particle"].isin(label_map.keys())].copy()
        particles_df["label"] = particles_df["particle"].map(label_map)

    train_idx, val_idx = train_test_split(
        particles_df.index,
        test_size=test_size,
        random_state=random_state,
        stratify=particles_df["label"],
    )
    train_df = particles_df.loc[train_idx].reset_index(drop=True)
    val_df = particles_df.loc[val_idx].reset_index(drop=True)

    train_ds = CryoEMDataset(proj_dir=project_dir, df=train_df, train=True, crop_size=crop_size)
    val_ds = CryoEMDataset(proj_dir=project_dir, df=val_df, train=False, crop_size=crop_size)

    train_loader = _make_loader(train_ds, batch_size=128)
    val_loader = _make_loader(val_ds, batch_size=128)
    return train_loader, val_loader
