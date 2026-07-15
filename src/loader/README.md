# mropes.loader

Dataloaders for audio, video, and image modalities, producing tensors ready
for token-based multimodal resampler models.

All datasets share a common factory interface:

```python
from mropes.loader import get_dataset, get_reader

train_set = get_dataset("video", path="data/my_dataset", split="train")
read_fn = get_reader("video")   # -> get_video
```

Supported types: `"audio"`, `"video"`, `"image"`.

---

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

**Transform pipeline:**
1. `ConvertImageDtype`
2. `Resize` (1.1x)
3. `CenterCrop`
4. `Normalize`
5. Rearrange: `(t dt) c (h dh) (w dw) -> t (dt dh dw) c h w`
6. [train only] horizontal flip (p=0.5)
7. [train only] frame dropout (p=0.2)
8. Rearrange: `t p c h w -> t h w (c p)`

---

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

---

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