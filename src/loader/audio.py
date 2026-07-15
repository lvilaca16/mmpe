import json
import math
import random
from dataclasses import dataclass, replace
from glob import glob
from pathlib import Path
from typing import Optional, Tuple

import torch
import torchaudio
import torchaudio.transforms as T
from einops import rearrange
from torch.nn import functional as Fn
from torchvision.transforms import Compose, Lambda

from .utils import pad_along_axis


@dataclass
class AudioConfig:
    dim: int = 128
    extension: str = ".wav"
    hop_length: int = 512
    length: int = 10
    n_fft: int = 1024
    n_mels: int = 64
    preprocessing: str = "spec"
    sr: int = 48000


class AudioDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path: Path,
        split: str = "train",
        audio: Optional[AudioConfig] = None,
        augment: Optional[bool] = False,
        n_classes: Optional[int] = 527,
        **kwargs,
    ):
        # audio properties
        audio = audio or AudioConfig()

        # Replace dataclass instead of mutating (in-place)
        overrides = {k: v for k, v in kwargs.items() if hasattr(audio, k)}
        self.audio = replace(audio, **overrides)

        path = Path(path) / f"{split}"
        assert path.exists(), "Invalid filepath"

        self.files = glob(str(path / f"*{self.audio.extension}"))

        self.n_classes = n_classes
        self.search_ext = self.audio.extension

        self.augment = augment and split == "train"

        if self.audio.preprocessing == "spec":
            self.transform = Compose(
                [
                    T.MelSpectrogram(
                        sample_rate=self.audio.sr,
                        n_fft=self.audio.n_fft,
                        hop_length=self.audio.hop_length,
                        n_mels=self.audio.n_mels,
                    ),
                    T.AmplitudeToDB(),
                    Lambda(lambda x: rearrange(x, "c f t -> t (c f)")),
                ]
            )

        elif self.audio.preprocessing == "raw":
            self.transform = Lambda(
                lambda x: rearrange(x, "(t ds) -> t ds", ds=self.audio.dim)
            )

        else:
            raise ValueError(
                f"Invalid audio preprocessing method ({self.audio.preprocessing})"
            )

        super().__init__()

    def output_shape(self) -> Tuple[int, ...]:
        if self.audio.preprocessing == "raw":
            t = self.audio.length * self.audio.sr // self.audio.dim
            return torch.Size((t, self.audio.dim))

        else:
            t = math.ceil(
                self.audio.length * self.audio.sr / self.audio.hop_length
            )
            return torch.Size((t, self.audio.n_mels))

    def time_masking(
        self, audio: torch.Tensor, length: float = 0.1, p: float = 0.5
    ) -> torch.Tensor:

        if random.randint(0, 1) <= p:
            mask_length = int(audio.shape[0] * random.uniform(0.0, length))
            mask_start = random.randint(0, audio.shape[0] - mask_length)

            audio[mask_start : mask_start + mask_length, ::] = 0

        return audio

    def apply_augmentations(self, audio: torch.Tensor) -> torch.Tensor:
        augmentations = [self.time_masking]

        return random.choice(augmentations)(audio)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filepath = Path(self.files[idx])

        with open(filepath.with_suffix(".json"), "r+") as fp:
            metadata = json.loads(fp.read())
            labels = metadata["label_id"]

        Y = torch.tensor(labels, dtype=torch.int64)
        Y = Fn.one_hot(Y, num_classes=self.n_classes).float()

        if len(Y) > 1:  # multi-label
            Y = Y.sum(0, keepdim=True)

        A = get_audio(filepath, self.audio.sr, self.audio.length)

        # Pre-processing
        A = self.transform(A)

        # Apply augmentations
        if self.augment:
            A = self.apply_augmentations(A)

        return A, Y.squeeze()


def get_audio(path: Path, sr: int, max_length: int) -> torch.Tensor:
    """
    Load an audio file, resample it to the target sample rate if needed,
    and pad or truncate it along the time axis to a fixed length.

    Arguments:
        path -- path to the audio file to load
        sr -- target sample rate
        max_length -- target number of seconds along the time axis.

    Returns:
        Audio tensor of shape (channels, max_length), resampled and
        padded/truncated to a fixed length
    """
    assert path.suffix == ".wav", "Only .wav files are accepted."

    # Load and resample audio
    audio, read_sr = torchaudio.load(path)

    if read_sr != sr:
        resampler = T.Resample(read_sr, sr, dtype=audio.dtype)
        audio = resampler(audio)

    # Fit channel size to max length
    max_length_samples = max_length * sr
    length = audio.shape[-1]

    if length < max_length_samples:
        pad_size = max_length_samples - length
        audio = pad_along_axis(audio, pad_size, dim=-1)

    elif length > max_length_samples:
        audio = audio[:, : -(length - max_length_samples)]

    return audio
