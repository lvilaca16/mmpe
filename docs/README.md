# docs

Analysis and benchmark results for the positional encoding variants in `mropes`.

## Decay Visualisations

[`decay.ipynb`](decay.ipynb) plots long-range attention similarity decay for each positional encoding (`rope`, `mrope`, `mrope_i`, `fourier`) as a function of relative distance.

This notebook provides the following analyses:

1. **Overall decay comparison** — one curve per encoding, showing normalised similarity bound vs relative distance.

    - `rope` and `mrope_i` plateau at a non-zero floor rather than decaying to zero: low-frequency channel pairs barely rotate with distance, so they keep contributing a persistent similarity floor even at long range. `mrope` and `fourier` drop and oscillate around/below zero at longer distances.

2. **Per-axis decay, MRoPE vs MRoPE-Interleave** — a side-by-side comparison isolating each positional axis's decay curve, with plain `rope` plotted as a reference.

    This reveals the core difference between the two multi-axis strategies:

    - **`mrope` (contiguous split)**: each axis gets an exclusive,non-overlapping block of the frequency spectrum. The axis assigned the lowest channel indices (highest frequencies) decays fastest, while the axis assigned the highest indices (lowest frequencies) barely decays at all. As such, contiguous splitting forces some axes into an inductive bias toward rapid decay while starving others of any effective positional signal.

    - **`mrope_i` (round-robin split)**: every axis samples the full frequency spectrum via interleaved channel assignment, so all axes track the reference `rope` decay curve closely instead of diverging.

## Benchmark Results

Using a model with a single attention resampling layer, we present benchmarking results for all positional encoding methods (see [`config/`](../config) for exact hyperparameters per run). We provide Top-1 accuracy (classification) / mAP (multi-label) on held-out validation splits, one model per positional encoding.

### Sidenotes

Fourier's dimension-expanding behaviour means its effective `dim_kv` going into attention differs from the rotary variants unless explicitly matched.

RoPE is excluded from these benchmarks because it only accepts flat 1D sequence inputs (`x.dim() == 3`) and cannot be applied to multi-axis inputs (e.g., video or image).

### Results

| Encoding   | ESC-50 (raw) | ESC-50 (spec) | AVE | CIFAR-100 |
|------------|:---:|:---:|:---:|:---:|
| `mrope` | 11.5 ± | 30.5 ± | 20.3 ± | 14.9 ± |
| `mrope_i` | 11.5 ± | 30.5 ± | 18.6 ± | 15.8 ± |
| `fourier` | 8.0 ± | 29.3 ± | 20.1 ± | 11.4 ± |


