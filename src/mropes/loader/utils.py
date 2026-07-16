from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms.functional as Fvt


def build_label_map(
    root_path: Path, splits: Tuple[str, ...] = ("train", "test", "val")
):
    """
    Build a label map from a root path.

    Arguments:
        root_path -- root path
        splits -- splits to include (default: {"train", "test"})

    Returns:
        Label map
    """
    all_classes = set()

    for split in splits:
        split_path = Path(root_path).joinpath(split)

        if split_path.exists():
            all_classes.update(
                [d.name for d in split_path.iterdir() if d.is_dir()]
            )

    sorted_classes = sorted(all_classes)

    return {cls_name: idx for idx, cls_name in enumerate(sorted_classes)}


def video_dropout(x: torch.Tensor, p: float = 0.3) -> torch.Tensor:
    """
    Dropout video frames.

    Arguments:
        x -- video tensor
        p -- dropout probability

    Returns:
        Dropped video tensor
    """
    if torch.rand(1) <= p:
        return torch.zeros(x.shape)

    return x


def video_hflip(x: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    """
    Horizontally flip video frames.

    Arguments:
        x -- video tensor
        p -- flip probability

    Returns:
        Flipped video tensor
    """
    if torch.rand(1) <= p:
        return Fvt.hflip(x)

    return x


def pad_along_axis(mat: np.array, pad_size: int, dim: int = 0) -> np.array:
    """
    Pad over only a single axis.

    Arguments:
        mat -- input numpy array
        pad_size -- padding size

    Keyword Arguments:
        dim -- padding axis (default: {0})

    Returns:
        Padded array
    """
    size = list(mat.shape)
    size[dim] = pad_size

    pad_values = torch.zeros(tuple(size))
    return torch.concatenate((mat, pad_values), dim)
