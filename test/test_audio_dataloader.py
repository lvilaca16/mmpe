import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.loader import get_dataset, AudioDataset

DATA_PATH = Path("data/audioset")
DIM = 128
N_MELS = 64


class TestAudioDatasetRaw(unittest.TestCase):

    def setUp(self):
        kwargs = {
            "path": DATA_PATH,
            "dim": DIM,
            "split": "train",
            "preprocessing": "raw",
        }

        self.dataset = get_dataset("audio", **kwargs)

    def test_non_empty_dataset(self):
        self.assertGreater(len(self.dataset), 0, "Dataset is empty")

    def test_dataloader_full_pass(self):
        loader = DataLoader(self.dataset, batch_size=1, drop_last=True)

        for A, Y in tqdm(loader):
            self.assertEqual(A.shape[1:], self.dataset.output_shape())
            self.assertEqual(Y.shape[-1], 527)

    def _assert_no_nan_or_inf(self, tensor, label):
        if torch.is_floating_point(tensor):
            self.assertFalse(
                torch.isnan(tensor).any(), f"NaN found in '{label}'"
            )
            self.assertFalse(
                torch.isinf(tensor).any(), f"Inf found in '{label}'"
            )

    def test_no_nan_or_inf(self):
        A, Y = self.dataset[0]

        with self.subTest(check="type", key="A"):
            self.assertIsInstance(A, torch.Tensor, "'A' is not a Tensor")

        with self.subTest(check="nan_inf", key="A"):
            self._assert_no_nan_or_inf(A, "A")

        with self.subTest(check="type", key="Y"):
            self.assertIsInstance(Y, torch.Tensor, "'Y' is not a Tensor")

        with self.subTest(check="nan_inf", key="Y"):
            self._assert_no_nan_or_inf(Y, "Y")


class TestAudioDatasetSpec(unittest.TestCase):

    def setUp(self):
        kwargs = {
            "path": DATA_PATH,
            "split": "train",
            "preprocessing": "spec",
            "n_mels": N_MELS,
        }

        self.dataset = get_dataset("audio", **kwargs)

    def test_dataloader_full_pass(self):
        loader = DataLoader(self.dataset, batch_size=2, drop_last=True)

        for A, Y in tqdm(loader):
            self.assertEqual(A.shape[-1] % N_MELS, 0)
            self.assertEqual(A.shape[1:], self.dataset.output_shape())

            self.assertEqual(Y.shape[-1], 527)

    def test_invalid_preprocessing_raises(self):
        with self.assertRaises(ValueError):
            AudioDataset(DATA_PATH, preprocessing="not_a_real_mode")


if __name__ == "__main__":
    unittest.main()
