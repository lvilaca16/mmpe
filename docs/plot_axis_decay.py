import torch
import matplotlib.pyplot as plt

from src.position import get_positional_encoding

max_dist = 128
dim = 192
base = 10000
n_axes = 3

axes_shape = [1] * n_axes

rope = get_positional_encoding("rope", dim=dim, base=base)
mrope = get_positional_encoding("mrope", dim=dim, base=base)
mrope_i = get_positional_encoding("mrope_i", dim=dim, base=base)

# --- reference RoPE decay (single axis, full spectrum) ---
x_1d = torch.zeros(1, max_dist, dim)
theta_rope = rope(x_1d, return_theta=True)  # (max_dist, d//2)
decay_rope = theta_rope.cos().mean(dim=-1)

fig, (ax_mrope, ax_mrope_i) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

for axis_idx in range(n_axes):
    shape = list(axes_shape)
    shape[axis_idx] = max_dist  # vary only this axis, others fixed at size 1

    x = torch.zeros(1, *shape, dim)

    theta_mrope = mrope(x, return_theta=True)  # (n_axes, seq, k)
    theta_mrope_i = mrope_i(x, return_theta=True)  # (n_axes, seq, k)

    decay_mrope = theta_mrope[axis_idx].cos().mean(dim=-1)
    decay_mrope_i = theta_mrope_i[axis_idx].cos().mean(dim=-1)

    ax_mrope.plot(decay_mrope.detach().numpy(), label=f"axis {axis_idx}")
    ax_mrope_i.plot(decay_mrope_i.detach().numpy(), label=f"axis {axis_idx}")

# reference curve, plotted last so it's drawn on top, styled distinctly
ax_mrope.plot(
    decay_rope.detach().numpy(),
    label="RoPE (reference)",
    color="black",
    linestyle=":",
    linewidth=2,
)
ax_mrope_i.plot(
    decay_rope.detach().numpy(),
    label="RoPE (reference)",
    color="black",
    linestyle=":",
    linewidth=2,
)

ax_mrope.set_title("MRoPE (contiguous split)")
ax_mrope.set_xlabel("Relative distance")
ax_mrope.set_ylabel("Normalized similarity bound")
ax_mrope.grid(alpha=0.3)
ax_mrope.legend()

ax_mrope_i.set_title("MRoPEInterleave (round-robin split)")
ax_mrope_i.set_xlabel("Relative distance")
ax_mrope_i.grid(alpha=0.3)
ax_mrope_i.legend()

fig.suptitle("Long-range decay by axis (RoPE as reference)")
plt.tight_layout()
plt.show()
