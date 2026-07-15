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

### Available Datasets

| Dataset | Config | Input format | Output shape | Notes |
|---------|--------|---------------|----------------|-------|
| `AudioDataset` | `AudioConfig` | `.wav` file + sibling `.json` (`label_id`) | `(t, n_mels)` (`spec`) or `(t, dim)` (`raw`) | Two preprocessing modes: mel-spectrogram (`spec`) or fixed-size raw waveform chunks (`raw`). Optional time-masking augmentation on `train`. |
| `VideoDataset` | `VideoConfig` | Pre-extracted frames in `path/<split>/<class>/<video_name>/*` | `(t_out, h_out, w_out, c*dt*dh*dw)` | Resize, center-crop, normalize, then temporal-spatial downsampling folds patches into the channel dim. Train-time horizontal flip and frame dropout. |
| `ImageDataset` | `ImageConfig` | `path/<split>/<class>/*` | `(h, w, 3)` or `(3, h, w)` (`channels_last`) | Distinct train (`RandomResizedCrop`, `RandomHorizontalFlip`, `RandAugment`) and eval (`CenterCrop`) pipelines. |

All three datasets expose `output_shape()`, so positional encoding layers and
downstream models can be sized directly from real data rather than hardcoded
config.

> **For more detail go to [src/loader](src/loader)**.

## Positional Encoding Layers

All positional encodings share a common interface:

```python
from mropes.position import get_positional_encoding

pos_enc = get_positional_encoding(pe_type, dim=dim, **pe_args)

pos_enc.output_shape(axes)   # -> shape after encoding, given input axis shape
kv = pos_enc(x)                # x: [batch, *axes, dim]
```

### Available Positional Encodings

| `pe_type` | Output dim         | Input shape                | Notes |
|------------|---------------------|------------------------------|-------|
| `rope` | `dim` (unchanged)   | `[batch, seq_len, dim]` (1D only) | Standard rotary embedding along a single sequence axis. |
| `mrope` | `dim` (unchanged)   | `[batch, *axes, dim]` | Splits `dim` into contiguous per-axis frequency blocks (Qwen2-VL style). |
| `mrope_i` | `dim` (unchanged)   | `[batch, *axes, dim]` | Round-robin interleaved frequency allocation across axes — full spectrum per axis. |
| `fourier` | expanded            | `[batch, *axes, dim]` | Fixed sinusoidal features, independent of `dim`; see below. |

> **For more detail go to [src/position](src/position)**.

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