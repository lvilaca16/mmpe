# mropes.position

Positional encoding layers for multimodal resampler models. All encodings
share a common interface:

```python
from mropes.position import get_positional_encoding

pos_enc = get_positional_encoding(pe_type, dim=dim, **pe_args)

pos_enc.output_shape(axes)   # -> shape after encoding, given input axis shape
kv = pos_enc(x)                # x: [batch, *axes, dim]
```

Supported types: `"rope"`, `"mrope"`, `"mrope_i"`, `"fourier"`.

---

### `rope`, `mrope`, `mrope_i` — rotary variants

These rotate the input in place rather than concatenating a positional
signal, so the feature dimension is always preserved: `output_shape(axes)[-1] == dim`.

- **`rope`** operates on a flat 1D sequence only (`x.dim() == 3`).
- **`mrope`** divides `dim` evenly across `n_axes`, giving each axis a
  contiguous, non-overlapping block of frequency pairs — coarser per-axis
  frequency resolution as the number of axes grows.
- **`mrope_i`** ([Multimodal RoPE variants paper](https://arxiv.org/abs/2510.23095))
  keeps the same per-axis frequency budget as `mrope` but assigns pairs
  round-robin instead of contiguous blocks, so every axis samples across the
  full frequency spectrum, rather than being confined to a single slice of it.

```python
mrope_i = get_positional_encoding("mrope_i", dim=192, base=10000)
kv = mrope_i(x)   # x: [batch, t, h, w, 192] -> same shape, rotated
```

---

### `fourier` — sinusoidal feature expansion

Unlike the rotary variants, `FourierPE` **concatenates** new sin/cos features
rather than rotating existing ones, so it expands the feature dimension. Each
axis gets its own full set of `n_bands` frequencies (not split across axes),
so output width grows linearly with the number of axes:

```python
fourier = get_positional_encoding("fourier", dim=192, n_bands=64, stack=False)

fourier.output_shape([32, 14, 14])
# -> torch.Size([32, 14, 14, len(axes) * 2 * n_bands])  (sin + cos per axis)
```

- `stack=False` (default): returns only the positional encoding —
  `len(shape) * 2 * n_bands` channels.
- `stack=True`: concatenates the encoding onto the original input —
  `dim + len(shape) * 2 * n_bands` channels.

Because `fourier` does not preserve `dim`, **always read the true post-encoding
size from `pos_enc.output_shape(axes)`** rather than assuming it matches the
input dimension — this is what feeds sizing decisions for any projection
layer that consumes the encoded output downstream.