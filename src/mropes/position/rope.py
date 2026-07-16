import torch
from torch import nn


class RoPE(nn.Module):
    """
    Rotary Positional Embedding (RoPE), as specified in
    https://arxiv.org/abs/2104.09864

    Applies a position-dependent rotation directly to the input tensor,
    rather than adding or concatenating a separate positional signal.
    Each pair of channels is treated as a 2D coordinate and rotated by an
    angle proportional to the token's sequence position, with the
    rotation frequency decaying geometrically across channel pairs
    (lower pairs rotate faster, higher pairs rotate slower). This gives
    attention dot products a relative-position property: the result
    depends only on the offset between two positions, not their absolute
    values.
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

        assert x.dim() == 3, (
            f"RoPE expects a 1D sequence input of shape (batch, seq_len, dim), "
            f"got shape {tuple(x.shape)}"
        )

        _, t, d = x.shape

        assert d == self.dim, f"expected last dim {self.dim}, got {d}"
        assert d % 2 == 0, "Embedding dimension must be even"

        # get phase and positions
        freq_idx = torch.arange(0, d, 2, **args)
        theta = 1.0 / (self.base ** (freq_idx / d))

        pos = self.get_positions([t], **args).squeeze(-1)

        ang = torch.einsum("n,d -> nd", pos, theta)

        if return_theta:
            return ang

        ang = torch.cat([ang, ang], dim=1)

        neg_half_x = self.rotate_half(x)

        x_rope = (x * ang.cos()) + (neg_half_x * ang.sin())

        return x_rope

    def output_shape(self, shape) -> int:
        return torch.Size((*shape, self.dim))
