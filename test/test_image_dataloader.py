import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.loader import get_dataset, ImageDataset

DATA_PATH = Path("data/image")
RESOLUTION = 224


class TestImageFolder(unittest.TestCase):

    def setUp(self):
        kwargs = {"path": DATA_PATH, "split": "test", "resolution": RESOLUTION}
        self.dataset = get_dataset("image", **kwargs)

    def test_non_empty_dataset(self):
        self.assertGreater(len(self.dataset), 0, "Dataset is empty")

    def test_dataloader_full_pass(self):
        loader = DataLoader(self.dataset, batch_size=2, drop_last=True)

        for X, Y in tqdm(loader):
            # channels_last=True by default -> (B, H, W, C)
            self.assertEqual(X.shape[1:], self.dataset.output_shape())

            self.assertEqual(Y.shape[0], 2)
            self.assertEqual(Y.shape[1], self.dataset.n_classes)

    def test_channels_last_toggle(self):
        dataset = ImageDataset(
            DATA_PATH,
            split="test",
            resolution=RESOLUTION,
            channels_last=False,
        )
        X, _ = dataset[0]
        self.assertEqual(X.shape, (3, RESOLUTION, RESOLUTION))

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

        with self.subTest(check="type", key="X"):
            self.assertIsInstance(X, torch.Tensor, "'X' is not a Tensor")

        with self.subTest(check="nan_inf", key="X"):
            self._assert_no_nan_or_inf(X, "X")

        with self.subTest(check="type", key="Y"):
            self.assertIsInstance(Y, torch.Tensor, "'Y' is not a Tensor")

        with self.subTest(check="nan_inf", key="Y"):
            self._assert_no_nan_or_inf(Y, "Y")


if __name__ == "__main__":
    unittest.main()
