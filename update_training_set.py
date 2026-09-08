"""
Build a semi-supervised "seeded" training set: for each particle class, top
up (or replace) the original labeled rows with high-confidence self-labeled
predictions from a previous inference run, keeping each class the same size
as it was originally.

Usage
-----
    python update_training_set.py --training_path training.csv \\
        --predictions_path predictions.csv --probs_path predictions.probs.npy \\
        --label_map particle2label.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROB_COLUMN = "prob"


def build_seeded_training(
    training_path: Path = "training.csv",
    predictions_path: Path = "predictions.csv",
    probs_path: Path = "predictions.probs.npy",
    output_path: Path = "training_seeded.csv",
    label_map: dict = None,
    quantile: float = 0.75,
    seed: int = None,
) -> Path:
    """
    Write a seeded training CSV to `output_path` and return that path.

    For each particle class, the output rows are:
      - up to as many high-confidence predictions (top-`quantile` fraction
        by predicted probability) as the class originally had, and
      - the original training rows, sampled to fill any remaining slots
        when there aren't enough high-confidence predictions.

    This keeps per-class counts unchanged while replacing some original
    examples with the model's own confident predictions.
    """
    if not label_map:
        raise ValueError("A label map must be provided to map predicted_particle to label.")

    train_df = pd.read_csv(training_path)
    pred_df = pd.read_csv(predictions_path)
    probs = np.load(probs_path)
    _validate_predictions(pred_df, probs, predictions_path, probs_path)

    pred_df[PROB_COLUMN] = probs.max(axis=1)

    seeded_frames = []
    for particle in pred_df["predicted_particle"].dropna().unique():
        train_sub = train_df[train_df["particle"] == particle]
        pred_sub = pred_df[pred_df["predicted_particle"] == particle]
        seeded_frames.extend(
            _seed_one_class(train_sub, pred_sub, quantile, label_map, seed)
        )

    df_new = pd.concat(seeded_frames, ignore_index=True)
    df_new.to_csv(output_path, index=False)
    return output_path


def _validate_predictions(pred_df: pd.DataFrame, probs: np.ndarray, predictions_path: Path, probs_path: Path) -> None:
    if len(pred_df) != len(probs):
        raise ValueError(
            f"Row mismatch: {predictions_path.name} has {len(pred_df)} rows, "
            f"but {probs_path.name} has shape {probs.shape}."
        )
    if "predicted_particle" not in pred_df.columns:
        raise KeyError("predictions.csv must contain a 'predicted_particle' column.")
    if probs.ndim != 2:
        raise ValueError(f"Expected a 2D probability array, got shape {probs.shape}.")


def _high_confidence_predictions(pred_sub: pd.DataFrame, quantile: float) -> pd.DataFrame:
    """Rows within this class whose top predicted probability exceeds the given quantile."""
    threshold = pred_sub[PROB_COLUMN].quantile(quantile)
    return pred_sub[pred_sub[PROB_COLUMN] > threshold]


def _seed_one_class(
    train_sub: pd.DataFrame,
    pred_sub: pd.DataFrame,
    quantile: float,
    label_map: dict,
    seed: int,
) -> list[pd.DataFrame]:
    """
    Build the seeded rows for one particle class: a random sample of its
    high-confidence predictions, topped up with original training rows so
    the class size matches `train_sub`.
    """
    high_conf = _high_confidence_predictions(pred_sub, quantile)
    high_conf = high_conf.drop(columns=["particle"], errors="ignore")
    high_conf = high_conf.rename(columns={"predicted_particle": "particle"})
    high_conf["label"] = high_conf["particle"].map(label_map)

    n_from_predictions = min(len(train_sub), len(high_conf))
    n_from_original = len(train_sub) - n_from_predictions

    frames = [
        high_conf.sample(n=n_from_predictions, replace=False, random_state=seed)
        .drop(columns=[PROB_COLUMN])
    ]
    if n_from_original > 0:
        frames.append(train_sub.sample(n=n_from_original, replace=False, random_state=seed))
    return frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a seeded training CSV by adding high-confidence predictions "
            "above each class median."
        )
    )
    parser.add_argument(
        "--training_path",
        default="training.csv",
        help="Filename of the base training CSV inside input_dir.",
    )
    parser.add_argument(
        "--predictions_path",
        default="predictions.csv",
        help="Filename of the predictions CSV inside input_dir.",
    )
    parser.add_argument(
        "--probs_path",
        default="predictions.probs.npy",
        help="Filename of the prediction probabilities NumPy array inside input_dir.",
    )
    parser.add_argument(
        "--label_map",
        default=None,
        help="Path to the label map JSON file.",
    )
    parser.add_argument(
        "--output_path",
        default="training_seeded.csv",
        help="Filename for the generated CSV inside input_dir.",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.75,
        help="Quantile of prediction probabilities to use as the threshold for high-confidence rows.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed used when sampling rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    label_map = None
    if args.label_map:
        with open(args.label_map, "r") as f:
            label_map = json.load(f)

    output_path = build_seeded_training(
        training_path=args.training_path,
        predictions_path=args.predictions_path,
        probs_path=args.probs_path,
        output_path=args.output_path,
        quantile=args.quantile,
        seed=args.seed,
        label_map=label_map,
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
