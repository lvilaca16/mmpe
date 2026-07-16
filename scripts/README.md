# scripts

Pre-processing helpers for preparing raw audio and video files into the
formats expected by `mropes.loader` (`AudioDataset`, `VideoDataset`).

Both scripts are standalone Bash utilities that wrap `ffmpeg`/`ffprobe` and
are meant to be run once per file, ahead of time, over your raw dataset —
not at training/dataloading time.

## Requirements

- `ffmpeg` and `ffprobe` on `PATH`
- `bc` (for floating-point fps calculation in `video.sh`)

## `audio.sh`

Converts a `.wav` file **in place** to mono-channel, 16-bit PCM, 48kHz —
matching the format `AudioDataset`/`get_audio` expects before resampling.

```bash
./scripts/audio.sh path/to/sample.wav
```

- Accepts only `.wav` input.
- The original file is preserved as `<file>.wav.bak` before being overwritten,
  so a failed or unwanted conversion can be undone by restoring the backup.
- Conversion happens via a temp file (`<stem>_tmp.wav`) which only replaces
  the original after `ffmpeg` succeeds — the original is never partially
  overwritten by a failed run.

**Rollback if needed:**
```bash
mv path/to/sample.wav.bak path/to/sample.wav
```

## `video.sh`

Extracts `n_frames` (default: 32) evenly spaced frames from a video file and
writes them as `.jpg` images into a new directory named after the video's
stem, alongside it — matching the layout `VideoDataset`/`get_video` expects
(`path/<video_name>/*`).

```bash
./scripts/video.sh path/to/sample.mp4
```

- Accepts `mp4`, `mov`, `mkv`, `avi`, `webm` input.
- Refuses to run if the output directory already exists, to avoid silently
  mixing frames from separate runs.
- Frame spacing is computed from the video's actual duration
  (`fps = n_frames / duration`), so frame count may occasionally land at
  ±1 frame depending on the source video's native frame rate — verify the
  output directory has exactly `n_frames` files if exact counts matter for
  your use case.

## Batch usage

Both scripts operate on a single file. To process a whole directory:

```bash
find data/raw_audio -name "*.wav" -exec ./scripts/audio.sh {} \;
find data/raw_video -name "*.mp4" -exec ./scripts/video.sh {} \;
```

## Notes

- These scripts mutate files/directories on disk — always run against a copy
  or version-controlled/backed-up dataset first, especially for `audio.sh`'s
  in-place conversion.
- Neither script is part of the `mropes` Python package or its test suite —
  they're one-off data-prep tooling, run manually before training or before
  committing test fixtures.