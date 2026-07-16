import torch
from einops import rearrange
from torch import nn


class MRoPEInterleave(nn.Module):
    """
    Multi-axis RoPE using an interleaved (round-robin) frequency-allocation
    strategy, based on the design proposed in https://arxiv.org/abs/2510.23095.

    Each frequency pair within a fixed dim is assigned to one axis via
    round-robin rather than contiguous blocks -- e.g. for 2 axes, pair 0
    goes to axis 0, pair 1 to axis 1, pair 2 back to axis 0, and so on.
    This lets every axis sample both low- and high-frequency bands (full
    spectrum coverage), at the cost of each axis getting roughly 1/n_axes
    of the resolution instead of a contiguous 1/n_axes chunk of the range,
    as in naive contiguous channel-splitting (see MRoPE). Output dimension
    stays fixed at dim regardless of how many axes are present.
    """

    def __init__(self, dim: int, base: int = 10000):
        super().__init__()

        assert dim % 2 == 0, "head_dim must be even"

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
        assert d % 2 == 0, "Embedding dimension must be even"

        n_pairs = d // 2

        # round-robin assignment of frequency pairs to axes:
        # axis 0 gets pairs [0, n_axes, 2*n_axes, ...]
        # axis 1 gets pairs [1, n_axes+1, ...], etc.
        pair_idx = torch.arange(0, n_pairs, **args)
        theta_all = 1.0 / (self.base ** (2 * pair_idx / d))

        pos = self.get_positions(axes, **args)

        pos_flat = rearrange(pos, "... d -> (...) d")  # (seq, n_axes)
        x_flat = rearrange(x, "b ... d -> b (...) d")  # (b, seq, d)

        seq_len = pos_flat.shape[0]
        cos_full = torch.zeros((seq_len, d), **args)
        sin_full = torch.zeros((seq_len, d), **args)

        if return_theta:
            thetas = []

        for i in range(n_axes):
            axis_pair_idx = pair_idx[pair_idx.long() % n_axes == i].long()
            theta_a = theta_all[axis_pair_idx]

            ang = torch.einsum("n,k->nk", pos_flat[:, i], theta_a)  # (seq, ki)

            if return_theta:
                thetas.append(ang)
                continue

            cos_a, sin_a = ang.cos(), ang.sin()

            # scatter into both halves of the channel dim (mirrored, as in
            # standard RoPE's torch.cat([ang, ang]) pattern)
            cos_full[:, axis_pair_idx] = cos_a
            cos_full[:, axis_pair_idx + n_pairs] = cos_a
            sin_full[:, axis_pair_idx] = sin_a
            sin_full[:, axis_pair_idx + n_pairs] = sin_a

        if return_theta:
            return torch.stack(thetas)

        x_rope = (x_flat * cos_full) + (self.rotate_half(x_flat) * sin_full)

        return x_rope.reshape(b, *axes, d)

    def output_shape(self, shape) -> int:
        return torch.Size((*shape, self.dim))
