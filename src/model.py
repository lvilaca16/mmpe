import math
from typing import Optional, List

import torch
import torch.autograd.profiler as profiler
import torch.nn as nn
from einops import rearrange, repeat

from .position import SUPPORT_TYPES, get_positional_encoding


class Attention(nn.Module):
    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        n_heads: int = 1,
        bias: Optional[bool] = True,
        eps: Optional[float] = 1e-6,
    ):
        super().__init__()

        if dim_q % n_heads != 0:
            raise ValueError("dim_q must be divisible by n_heads")

        # Attention
        self.norm_q = nn.RMSNorm(dim_q, eps=eps)
        self.norm_kv = nn.RMSNorm(dim_kv, eps=eps)

        self.q_proj = nn.Linear(dim_q, dim_q, bias=bias)
        self.kv_proj = nn.Linear(dim_kv, dim_q * 2, bias=bias)
        self.o_proj = nn.Linear(dim_q, dim_q, bias=bias)

        # Properties
        self.dim_kv = dim_kv
        self.dim_q = dim_q
        self.n_heads = n_heads

    def forward(
        self, x_q: torch.Tensor, x_kv: torch.Tensor, need_weights: bool = False
    ) -> torch.Tensor:
        h = self.n_heads

        # with autocast(device_type=x_q.device.type, enabled=False):
        q = self.norm_q(x_q.type(torch.float32)).type(x_q.dtype)
        kv = self.norm_kv(x_kv.type(torch.float32)).type(x_kv.dtype)

        # [batch, tokens, emb_size]
        k, v = self.kv_proj(kv).chunk(2, dim=-1)
        q = self.q_proj(q)

        # [(batch + n_heads), tokens, emb_size]
        q = rearrange(q, "b t (h d) -> (b h) t d", h=h)
        k = rearrange(k, "b t (h d) -> (b h) t d", h=h)
        v = rearrange(v, "b t (h d) -> (b h) t d", h=h)

        with profiler.record_function("attention/dot-product"):
            q_scaled = q * math.sqrt(1.0 / float(q.size(-1)))
            M = torch.bmm(q_scaled, k.transpose(-2, -1))

        with profiler.record_function("attention/softmax"):
            w_att = torch.softmax(M, -1)

        o = torch.bmm(w_att, v)
        o = rearrange(o, "(b h) i d -> b i (h d)", h=h)

        if need_weights:
            return self.o_proj(o), w_att

        return self.o_proj(o)


class Model(nn.Module):
    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        n_classes: int = 16,
        n_heads: int = 4,
        input_shape: List[int] = [1, 224, 224, 3],
        pe_type: str = "rope",
        **kwargs,
    ):
        super().__init__()

        self.latent_query = nn.Parameter(torch.randn(1, dim_q))

        assert (
            pe_type in SUPPORT_TYPES
        ), f"{pe_type} not supported ({SUPPORT_TYPES})"

        # Positional Encoding
        pe_args = {}

        stack = kwargs.get("stack", None)
        if stack is not None:
            pe_args["stack"] = stack

        n_bands = kwargs.get("n_bands", None)
        if n_bands is not None:
            pe_args["n_bands"] = n_bands

        # Configure resampling according to input shape
        _, *axes, dim = input_shape

        self.channel_projection = nn.Linear(dim, dim_kv)

        self.pos_enc = get_positional_encoding(pe_type, dim=dim_kv, **pe_args)

        dim_pos_enc = self.pos_enc.output_shape(axes)[-1]

        # Attention
        bias = kwargs.get("bias", True)
        eps = kwargs.get("eps", 1e-6)

        self.att = Attention(dim_q, dim_pos_enc, n_heads, bias, eps)

        # Classifier head
        self.to_logits = nn.Linear(dim_q, n_classes)

        with torch.no_grad():
            # As mentioned in https://arxiv.org/abs/2103.03206
            nn.init.trunc_normal_(self.latent_query, 0.0, 0.02, -2, 2)

    def forward(self, x_kv: torch.Tensor) -> torch.Tensor:
        b = x_kv.shape[0]

        # [batch, tokens, dim_kv]
        with profiler.record_function("attention/pre-processing"):
            kv = self.channel_projection(x_kv)

        # [batch, tokens, pe_size]
        with profiler.record_function("attention/positional-encoding"):
            kv = self.pos_enc(kv)
            kv = rearrange(kv, "b ... c -> b (...) c")  # to sequence

        # Get Latent array
        with profiler.record_function("latent-array"):
            x_latent = repeat(self.latent_query, "n d -> b n d", b=b)

        with profiler.record_function("attention"):
            x_attn = self.att(x_latent, kv).squeeze(dim=1)

        return self.to_logits(x_attn)
