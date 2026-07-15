import math
from dataclasses import dataclass, replace
from glob import glob
from pathlib import Path
from typing import Optional, Tuple

import torch
from PIL import Image, ImageFile
from torch.nn import functional as F
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandAugment,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

from .utils import build_label_map

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


@dataclass
class ImageConfig:
    resolution: int = (224,)
    channels_last: bool = True


class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path: Path,
        split: str = "train",
        augmentation: Optional[bool] = True,
        image: Optional[ImageConfig] = None,
        **kwargs,
    ):
        image = image or ImageConfig()

        overrides = {k: v for k, v in kwargs.items() if hasattr(image, k)}

        # Replace dataclass instead of mutating (in-place)
        self.image = replace(image, **overrides)

        path = Path(path) / f"{split}"
        assert path.exists(), "Invalid filepath"

        # labels are the folders on the first level
        self.files = glob(str(path / "**/*"))

        self.label_map = build_label_map(path.parent)

        # increase by 10%
        resize_shape = (
            math.ceil(self.image.resolution * 1.1),
            math.ceil(self.image.resolution * 1.1),
        )

        if split == "train" and augmentation:
            self.transform = Compose(
                [
                    Resize(resize_shape),
                    RandomResizedCrop(self.image.resolution, scale=(0.2, 1.0)),
                    RandomHorizontalFlip(),
                    RandAugment(num_ops=4, magnitude=5),
                    ToTensor(),
                    Normalize(mean=MEAN, std=STD),
                ]
            )

        else:
            self.transform = Compose(
                [
                    Resize(resize_shape),
                    CenterCrop(self.image.resolution),
                    ToTensor(),
                    Normalize(mean=MEAN, std=STD),
                ]
            )

        self.n_classes = len(self.label_map)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filepath = self.files[idx]
        label = filepath.split("/")[-2]

        Y = torch.tensor(self.label_map[label])
        Y = F.one_hot(Y, num_classes=self.n_classes).float()

        X = get_image(filepath)

        X = self.transform(X)

        if self.image.channels_last:
            X = X.permute((1, 2, 0))

        return X, Y

    def output_shape(self) -> Tuple[int]:
        if self.image.channels_last:
            return torch.Size((self.image.resolution, self.image.resolution, 3))

        return torch.Size((3, self.image.resolution, self.image.resolution))


def get_image(path: str) -> ImageFile:
    """
    Load an image from disk and convert it to RGB.

    Arguments:
        path -- path to the image file to load

    Returns:
        The loaded image, converted to RGB color mode (3 channels),
        as a PIL ImageFile
    """
    img = Image.open(path).convert("RGB")
    return img
