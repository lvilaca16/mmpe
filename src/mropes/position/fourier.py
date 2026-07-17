from typing import Tuple

import torch
from einops import repeat
from torch import nn


class FourierPE(nn.Module):
    """
    Fourier positional encoding for arbitrary-dimensional grid inputs.

    Generates sinusoidal positional features independently for each spatial
    axis of the input (e.g. height, width, depth, time), normalizing each
    axis to the range [-1, 1] and modulating it across a fixed number of
    frequency bands. Each axis is allocated its own full set of n_bands
    frequencies (concatenated rather than channel-split), so the encoding
    width grows linearly with the number of axes.
    """

    def __init__(
        self,
        dim: int,
        n_bands: int = 64,
        stack: bool = False,
        add_position: bool = True,
    ):
        super().__init__()

        self.dim = dim
        self.n_bands = n_bands
        self.stack = stack
        self.add_position = add_position

    def _get_positions(self, shape, v_min=0, v_max=1.0, **kwargs):
        coords = [
            torch.linspace(v_min, v_max, steps=s, **kwargs) for s in shape
        ]
        return torch.stack(
            torch.meshgrid(*coords, indexing="ij"), dim=len(shape)
        )

    def _get_frequencies(self, v_max, **kwargs):
        return torch.linspace(1.0, v_max / 2.0, self.n_bands, **kwargs)

    def forward(self, x: torch.Tensor, return_frequency: bool = False):
        b, *axes, _ = x.shape
        args = {"device": x.device, "dtype": x.dtype}

        # list of positions for each axis [-1 to 1]
        pos = self._get_positions(axes, v_min=0, v_max=1, **args)

        freq = [self._get_frequencies(ax, **args) for ax in axes]
        freq = torch.stack(freq, -1)

        # Get frequencies
        frequencies = []

        for i, frequency in enumerate(freq.T):
            frequencies.append(pos[..., i : i + 1] * frequency[None, ...])

        if return_frequency:
            return torch.stack(frequencies)

        encodings = []

        encodings.extend(
            [
                torch.sin(torch.pi * frequency_grid)
                for frequency_grid in frequencies
            ]
        )
        encodings.extend(
            [
                torch.cos(torch.pi * frequency_grid)
                for frequency_grid in frequencies
            ]
        )

        encodings = torch.cat(encodings, dim=-1)

        if self.add_position:
            encodings = torch.cat((pos, encodings), dim=-1)

        encodings = repeat(encodings, "... -> b ...", b=b)

        if self.stack:
            return torch.cat((x, encodings), dim=-1)

        return encodings

    def output_shape(self, shape: Tuple[int]) -> Tuple[int]:
        channel_dim = len(shape) * (2 * self.n_bands) + len(shape)

        if self.stack:
            channel_dim += self.dim

        return torch.Size((*shape, channel_dim))
