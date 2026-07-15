import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.loader import get_dataset, VideoDataset

DATA_PATH = Path("data/video")
RESOLUTION = 224
PATCH_DOWNSAMPLE = [2, 8, 8]
N_MELS = 64


class TestVideoDataset(unittest.TestCase):

    def setUp(self):
        kwargs = {
            "path": DATA_PATH,
            "split": "test",
            "resolution": RESOLUTION,
            "downsample": PATCH_DOWNSAMPLE,
        }

        self.dataset = get_dataset("video", **kwargs)

    def test_non_empty_dataset(self):
        self.assertGreater(len(self.dataset), 0, "Dataset is empty")

    def test_dataloader_full_pass(self):
        loader = DataLoader(self.dataset, batch_size=1, drop_last=True)

        for X, Y in tqdm(loader):
            dt, dh, dw = PATCH_DOWNSAMPLE

            self.assertEqual(X.shape[0], 1)

            # video: "t h w (c p)" after downsample rearrange
            self.assertEqual(X.shape[-1] % (dt * dh * dw), 0)
            self.assertEqual(X.shape[1:], self.dataset.output_shape())

            self.assertEqual(Y.shape[0], 1)
            self.assertEqual(Y.shape[1], self.dataset.n_classes)

    def test_invalid_patch_downsample_raises(self):
        with self.assertRaises(AssertionError):
            VideoDataset(DATA_PATH, downsample=[2, 8])  # only 2 values

    def test_missing_file_raises(self):
        with self.assertRaises((FileNotFoundError, IndexError)):
            # force an out-of-range access if dataset supports it; otherwise
            # this documents expected behavior for a corrupted file list
            bad_dataset = VideoDataset(DATA_PATH, split="test")
            bad_dataset.files = ["nonexistent/path/video.mp4"]
            bad_dataset[0]

    def _assert_no_nan_or_inf(self, tensor, label):
        if torch.is_floating_point(tensor):
            self.assertFalse(
                torch.isnan(tensor).any(), f"NaN found in '{label}'"
            )
            self.assertFalse(
                torch.isinf(tensor).any(), f"Inf found in '{label}'"
            )

    def test_no_nan_or_inf(self):
        X, Y = self.dataset[0]

        with self.subTest(check="type", key="x_video"):
            self.assertIsInstance(X, torch.Tensor, "X is not a Tensor")

        with self.subTest(check="nan_inf", key="y_true"):
            self._assert_no_nan_or_inf(Y, "y_true")


if __name__ == "__main__":
    unittest.main()
