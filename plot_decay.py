import argparse
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from src.position import get_positional_encoding, SUPPORT_TYPES

METHOD_CONFIG = {
    "rope": {
        "module_kwargs": lambda args: {"dim": args.dim, "base": args.base},
        "input_shape": lambda args: (1, args.max_dist, args.dim),
        "return_kwarg": "return_theta",
    },
    "mrope": {
        "module_kwargs": lambda args: {"dim": args.dim, "base": args.base},
        "input_shape": lambda args: (
            1,
            args.max_dist,
            *([1] * (args.n_axes - 1)),
            args.dim,
        ),
        "return_kwarg": "return_theta",
    },
    "mrope_i": {
        "module_kwargs": lambda args: {"dim": args.dim, "base": args.base},
        "input_shape": lambda args: (
            1,
            args.max_dist,
            *([1] * (args.n_axes - 1)),
            args.dim,
        ),
        "return_kwarg": "return_theta",
    },
    "fourier": {
        "module_kwargs": lambda args: {
            "n_bands": args.dim // 2,
            "dim": args.dim,
        },
        "input_shape": lambda args: (
            1,
            args.max_dist,
            *([1] * (args.n_axes - 1)),
            args.dim,
        ),
        "return_kwarg": "return_frequency",
    },
}


def compute_decay(name: str, args: argparse.Namespace) -> torch.Tensor:
    if name not in METHOD_CONFIG:
        raise ValueError(f"No plotting config for method '{name}'")

    config = METHOD_CONFIG[name]

    module = get_positional_encoding(name, **config["module_kwargs"](args))

    shape = config["input_shape"](args)
    x = torch.zeros(*shape)

    raw = module(x, **{config["return_kwarg"]: True})

    # raw is either (seq, k) for single-axis methods, or (n_axes, ..., k)
    if raw.dim() == 2:
        decay = raw.cos().mean(dim=-1)
    else:
        axis_slice = raw[args.axis_idx]
        axis_slice = axis_slice.reshape(args.max_dist, -1)
        decay = axis_slice.cos().mean(dim=-1)

    return decay


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--methods",
        nargs="+",
        choices=SUPPORT_TYPES,
        default=SUPPORT_TYPES,
        help=f"Positional encoding methods to plot. Default: {SUPPORT_TYPES}.",
    )
    parser.add_argument(
        "--max-dist",
        type=int,
        default=128,
        help="Maximum relative distance to sweep over.",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=192,
        help="Feature dimension used to construct each module (must be even).",
    )
    parser.add_argument(
        "--base",
        type=int,
        default=10000,
        help="Rotary base for rope/mrope/mrope_i.",
    )
    parser.add_argument(
        "--n-axes",
        type=int,
        default=3,
        help="Number of positional axes for multi-axis methods "
        "(mrope, mrope_i, fourier). Ignored for rope. Axis 0 is the "
        "one swept over relative distance; remaining axes are fixed "
        "at size 1.",
    )
    parser.add_argument(
        "--axis-idx",
        type=int,
        default=0,
        help="Which axis's decay curve to plot for multi-axis methods.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="If set, save the figure to this path instead of showing it.",
    )

    args = parser.parse_args()

    if args.dim % 2 != 0:
        parser.error("--dim must be even")
    if args.max_dist < 1:
        parser.error("--max-dist must be >= 1")
    if args.n_axes < 1:
        parser.error("--n-axes must be >= 1")
    if not (0 <= args.axis_idx < args.n_axes):
        parser.error("--axis-idx must be in [0, n_axes)")

    plt.figure(figsize=(9, 5))

    for name in args.methods:
        try:
            decay = compute_decay(name, args)
        except ValueError as e:
            print(f"Skipping '{name}': {e}")
            continue

        plt.plot(decay.detach().numpy(), label=name)

    plt.xlabel("Relative distance")
    plt.ylabel("Normalized similarity bound")
    plt.title("Long-range decay across positional encoding methods")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=150)
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
