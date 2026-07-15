import math
import os
from dataclasses import dataclass, field, replace
from glob import glob
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from einops import rearrange
from torch.nn import functional as F
from torchvision.transforms import (
    CenterCrop,
    Compose,
    ConvertImageDtype,
    Lambda,
    Normalize,
    Resize,
)

from .image import MEAN, STD, get_image
from .utils import build_label_map, video_dropout, video_hflip


@dataclass
class VideoConfig:
    resolution: int = 224
    n_samples: int = 32
    downsample: List[int] = field(default_factory=lambda: [2, 8, 8])
    extension: str = "mp4"


class VideoDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        path: Path,
        split: str = "train",
        video: Optional[VideoConfig] = None,
        **kwargs,
    ):
        video = video or VideoConfig()

        video_overrides = {k: v for k, v in kwargs.items() if hasattr(video, k)}

        # Replace dataclass instead of mutating (in-place)
        self.video = replace(video, **video_overrides)

        path = Path(path) / f"{split}"
        assert path.exists(), "Invalid filepath"

        # labels are the folders on the first level
        self.files = glob(str(path / f"**/*.{self.video.extension}"))

        self.label_map = {x: i for i, x in enumerate(os.listdir(path))}
        self.n_classes = len(self.label_map)

        self.label_map = build_label_map(path.parent)

        # video properties
        assert len(self.video.downsample) == 3, "invalid downsample config."
        # self.dt, self.dh, self.dw = self.video.downsample

        bypass = Lambda(lambda x: x)  # helper transform

        # self.resolution = self.video.resolution

        # increase by 10%
        resize_shape = (
            math.ceil(self.video.resolution * 1.1),
            math.ceil(self.video.resolution * 1.1),
        )

        self.v_transform = Compose(
            [
                ConvertImageDtype(torch.float32),
                Resize(resize_shape),
                CenterCrop(self.video.resolution),
                Normalize(mean=MEAN, std=STD),
                Lambda(
                    lambda x: rearrange(
                        x,
                        "(t dt) c (h dh) (w dw) -> t (dt dh dw) c h w",
                        dt=self.video.downsample[0],
                        dh=self.video.downsample[1],
                        dw=self.video.downsample[2],
                    )
                ),
                (
                    Lambda(lambda x: video_hflip(x, p=0.5))
                    if split == "train"
                    else bypass
                ),
                (
                    Lambda(lambda x: video_dropout(x, p=0.2))
                    if split == "train"
                    else bypass
                ),
        

        super().__init__()

    def output_shape(self) -> Tuple[int, ...]:
        t_out = self.video.n_samples // self.video.downsample[0]
        h_out = self.video.resolution // self.video.downsample[1]
        w_out = self.video.resolution // self.video.downsample[2]

        patch_dim = math.prod(self.video.downsample) * 3  # c * p

        return torch.Size((t_out, h_out, w_out, patch_dim))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        filepath = Path(self.files[idx])

        if not filepath.exists():
            raise FileNotFoundError("File not found.")

        label = filepath.parent.name  # parent folder defines the label

        Y = torch.tensor(self.label_map[label])
        Y = F.one_hot(Y, num_classes=self.n_classes).float()

        x_video = get_video(filepath, n_frames=self.video.n_samples)

        if self.v_transform is not None:
            x_video = self.v_transform(x_video)

        return x_video, Y


def get_video(path: Path, n_frames: int = 32) -> torch.Tensor:
    """
    Read video frames from a folder with already pre-processed
    data. Videos should be pre-processed with ffmpeg to extract
    frames every second.


    Arguments:
        path -- path to video file, directory with frames should be in the same
                location

    Returns:
        Tensor with the packed frames.
    """

    frames_path = path.parent / f"{path.stem}/*"
    frames_path = sorted(glob(frames_path.as_posix()))

    frames = [get_image(x) for x in frames_path]

    assert (
        len(frames) >= n_frames
    ), f"Invalid amount of frames ({len(frames)}/{n_frames})"

    frames = np.stack(frames)
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # channels first

    return frames
