# mropes

Dataloaders and positional encoding layers for multimodal resampling models.

`mropes` provides two building blocks for multimodal resampler architectures:

1. **Dataloaders** — audio, video, and image datasets with configurable
 preprocessing, downsampling, and augmentation, producing tensors ready for
 token-based models.
2. **Positional encoding layers** — pluggable, drop-in positional encoding
 schemes (`rope`, `mrope`, `mrope_i`, `fourier`) that operate over
 arbitrary axis shapes.

## Installation

```bash
git clone https://github.com/<your-username>/mropes.git
cd mropes
pip install -e .
```

## Dataloaders

All datasets share a common factory interface:

```python
from mropes.loader import get_dataset, get_reader

train_set = get_dataset("video", path="data/my_dataset", split="train")
read_fn = get_reader("video")   # -> get_video
```

Supported types: `"audio"`, `"video"`, `"image"`.

### VideoDataset

Loads pre-extracted video frames from `path/<split>/<class>/<video_name>/*`,
applies resize, centre-crop, normalisation, temporal-spatial downsampling, and
(on `train`) horizontal-flip and dropout augmentation.

```python
from mropes.loader.video import VideoDataset, VideoConfig

video_cfg = VideoConfig(
 resolution=224,
 n_samples=32,
 downsample=[2, 8, 8],   # [dt, dh, dw]
 extension="mp4",
)

train_set = VideoDataset(path="data/my_dataset", split="train", video=video_cfg)

x, y = train_set[0]        # x: [t, h, w, c*dt*dh*dw]
train_set.output_shape()   # torch.Size([t_out, h_out, w_out, patch_dim])
```

<!-- **Transform Pipeline**:
1. ConvertImageDtype
2. Resize (1.1x)
3. CenterCrop
4. Normalise
5. [train only] horizontal flip (p=0.5)
6. [train only] frame dropout (p=0.2)
7. Rearrange: (t dt) c (h dh) (w dw) -> t h w (c dt dh dw) -->

### AudioDataset

Loads `.wav` files (with a sibling `.json` containing `label_id`), resamples
and pads/truncates to a fixed length, then applies one of two preprocessing
modes via `AudioConfig.preprocessing`:

```python
from mropes.loader.audio import AudioDataset, AudioConfig

# Mel-spectrogram
audio_cfg = AudioConfig(preprocessing="spec", sr=48000, n_fft=1024, hop_length=512, n_mels=64, length=10)

# Raw waveform, reshaped into fixed-size chunks
audio_cfg = AudioConfig(preprocessing="raw", dim=128, sr=48000, length=10)

train_set = AudioDataset(path="data/my_audio_dataset", split="train", audio=audio_cfg, augment=True)

train_set.output_shape()
```

- `"spec"` → `MelSpectrogram → AmplitudeToDB → rearrange(c f t -> t (c f))`
 — output shape: `(ceil(length * sr / hop_length), n_mels)`
- `"raw"` → `rearrange(c (t ds) -> t c ds, ds=dim).squeeze(1)`
 — output shape: `(length * sr // dim, dim)`

When `augment=True` (train split only), a random time-masking augmentation is
applied per sample.

### ImageDataset

Loads images from `path/<split>/<class>/*`, with distinct train/eval transform
pipelines:

```python
from mropes.loader.image import ImageDataset, ImageConfig

image_cfg = ImageConfig(resolution=224, channels_last=True)

train_set = ImageDataset(path="data/my_image_dataset", split="train", image=image_cfg, augmentation=True)

x, y = train_set[0]        # x: [h, w, c] if channels_last else [c, h, w]
train_set.output_shape()
```

- **Train** (`augmentation=True`): `Resize (1.1x) → RandomResizedCrop → RandomHorizontalFlip → RandAugment → ToTensor → Normalize`
- **Eval / no augmentation**: `Resize (1.1x) → CenterCrop → ToTensor → Normalize`

All three datasets expose `output_shape()`, so positional encoding layers and
downstream models can be sized directly from real data rather than hardcoded
config.


## Positional Encoding Layers

All positional encodings share a common interface:

```python
from mropes.position import get_positional_encoding

pos_enc = get_positional_encoding(pe_type, dim=dim, **pe_args)

pos_enc.output_shape(axes)   # -> shape after encoding, given input axis shape
kv = pos_enc(x)                # x: [batch, *axes, dim]
```

| `pe_type` | Output dim         | Input shape                | Notes |
|------------|---------------------|------------------------------|-------|
| `rope` | `dim` (unchanged)   | `[batch, seq_len, dim]` (1D only) | Standard rotary embedding along a single sequence axis. |
| `mrope` | `dim` (unchanged)   | `[batch, *axes, dim]` | Splits `dim` into contiguous per-axis frequency blocks (Qwen2-VL style). |
| `mrope_i` | `dim` (unchanged)   | `[batch, *axes, dim]` | Round-robin interleaved frequency allocation across axes — full spectrum per axis. |
| `fourier` | expanded            | `[batch, *axes, dim]` | Fixed sinusoidal features, independent of `dim`; see below. |

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
 The full frequency spectrum, rather than being confined to a single slice of it.

```python
mrope_i = get_positional_encoding("mrope_i", dim=192, base=10000)
kv = mrope_i(x)   # x: [batch, t, h, w, 192] -> same shape, rotated
```

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

## Citations

If you find this repository is useful in your research or applications, please consider giving us a star 🌟

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

## Correspondence and Maintenance

Any feedback is appreciated. If you observed any issues, please get in touch with us. All the project-related problems and feature requests should be submitted through our GitHub Issues page.


<!-- Links -->
[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg