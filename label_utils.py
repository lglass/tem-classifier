"""Shared CLI helper for resolving a particle-name -> class-index label map,
used by both the training and inference entry points."""

from pathlib import Path
from typing import Optional, Tuple

from dataset import load_label_map


def resolve_label_map(
    label_map_path: Optional[Path] = None,
    num_classes: Optional[int] = None,
) -> Tuple[Optional[dict], int]:
    """
    Resolve a (label_map, num_classes) pair from CLI-style arguments.

    - If `label_map_path` is given, it is loaded and takes precedence; if
      `num_classes` is also given, it must agree with the map's size.
    - If only `num_classes` is given, there is no label map — predictions
      are reported as bare class indices.

    Raises:
        ValueError: neither argument is usable, or they disagree.
    """
    if label_map_path is not None:
        label_map = load_label_map(label_map_path)
        resolved_classes = len(label_map)
        if num_classes is not None and num_classes != resolved_classes:
            raise ValueError(
                f"--num_classes {num_classes} conflicts with label_map size {resolved_classes}"
            )
        return label_map, resolved_classes

    if num_classes is not None:
        return None, num_classes

    raise ValueError("Provide --num_classes or --label_map.")
