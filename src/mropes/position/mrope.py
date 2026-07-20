import torch
from einops import rearrange
from torch import nn


class MRoPE(nn.Module):
    """
    Multi-axis extension of RoPE using contiguous channel-splitting (the
    baseline design popularized by Qwen2-VL). The dim channels are divided
    evenly among the n_axes positional axes, with each axis getting its own
    contiguous, non-overlapping block of frequency pairs -- e.g. for 2 axes,
    axis 0 gets the first half of the frequency spectrum and axis 1 gets the
    second half.

    Output dimension stays fixed at dim regardless of how many axes are
    present, but each axis only sees dim / n_axes of the total frequency
    spectrum, which coarsens the frequency resolution available to any
    single axis as the number of axes grows. Contrast with MRoPEInterleave,
    which keeps the same per-axis frequency budget but spreads it across
    the full spectrum (round-robin) instead of one contiguous block.
    """

    def __init__(self, dim: int, base: int = 10000):
        super().__init__()

        self.base = base
        self.dim = dim

    @staticmethod
    def rotate_half(x):
        assert x.shape[-1] % 2 == 0, "Dimension should be even"

        x1, x2 = x.split(x.shape[-1] // 2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def get_positions(shape, **kwargs):
        coords = [torch.arange(0, s, **kwargs) for s in shape]
        return torch.stack(
            torch.meshgrid(*coords, indexing="ij"), dim=len(shape)
        )

    def forward(self, x: torch.Tensor, return_theta: bool = False):
        args = {"device": x.device, "dtype": x.dtype}
        b, *axes, d = x.shape
        n_axes = len(axes)

        assert d == self.dim, f"expected last dim {self.dim}, got {d}"
        assert d % (n_axes * 2) == 0, "Embedding dimension must be even"

        n_pairs = d // (n_axes * 2)

        pos = self.get_positions(axes, **args)

        pos_flat = rearrange(pos, "... d -> (...) d")  # (seq, n_axes)
        x_flat = rearrange(x, "b ... d -> b (...) d")  # (b, seq, d)

        seq_len = pos_flat.shape[0]
        cos = torch.zeros((n_axes, seq_len, d // n_axes), **args)
        sin = torch.zeros((n_axes, seq_len, d // n_axes), **args)

        if return_theta:
            thetas = []

        for i in range(len(axes)):
            freq_idx = torch.arange(i * n_pairs, (i + 1) * n_pairs, **args)
            theta = 1.0 / (self.base ** (2 * freq_idx / d))

            ang = torch.einsum("n,k->nk", pos_flat[:, i], theta)  # (seq, ki)

            if return_theta:
                thetas.append(ang)
                continue

            ang = torch.cat([ang, ang], dim=-1)  # (T, D)

            cos[i] = ang.cos()
            sin[i] = ang.sin()

        if return_theta:
            return torch.stack(thetas)

        cos = rearrange(cos, "n t d -> t (n d)")
        sin = rearrange(sin, "n t d -> t (n d)")

        x_rope = (x_flat * cos) + (self.rotate_half(x_flat) * sin)

        return x_rope.reshape(b, *axes, d)  # restore original shape

    def output_shape(self, shape) -> int:
        return torch.Size((*shape, self.dim))
