import unittest

import torch

from mropes.position import FourierPE


class TestFourierPositionalEncoding(unittest.TestCase):

    def setUp(self):
        self.n_bands = 4
        self.module = FourierPE(3, n_bands=self.n_bands)

    def test_output_shape_matches_declared_output_shape(self):
        x = torch.rand(2, 3, 4, 5, 2)
        axes = x.shape[1:-1]

        out = self.module(x)
        expected = self.module.output_shape(axes)

        self.assertEqual(out.shape[0], x.shape[0])  # batch size
        self.assertEqual(out.shape[1:], expected)  # full

    def test_batch_dimension_matches_input(self):
        x = torch.rand(5, 6, 7, 3)
        out = self.module(x)
        self.assertEqual(out.shape[0], 5)

    def test_axes_dimensions_preserved(self):
        x = torch.rand(2, 6, 7, 3)
        out = self.module(x)
        self.assertEqual(out.shape[1], 6)
        self.assertEqual(out.shape[2], 7)

    def test_last_dim_is_2_times_bands_times_num_axes(self):
        x = torch.rand(2, 6, 7, 3)
        out = self.module(x)
        num_axes = len(x.shape[1:-1])
        self.assertEqual(out.shape[-1], num_axes * 2 * self.n_bands)

    def test_values_are_bounded(self):
        # Output is built entirely from sin/cos, so must lie in [-1, 1].
        x = torch.rand(2, 6, 7, 3)
        out = self.module(x)
        self.assertTrue(torch.all(out >= -1.0 - 1e-6))
        self.assertTrue(torch.all(out <= 1.0 + 1e-6))

    def test_batch_copies_are_identical(self):
        # Since positions don't depend on batch, every batch slice should
        # be identical (encodings are repeated, not computed per-sample).
        x = torch.rand(3, 4, 5, 3)
        out = self.module(x)
        self.assertTrue(torch.allclose(out[0], out[1], atol=1e-6))
        self.assertTrue(torch.allclose(out[1], out[2], atol=1e-6))


if __name__ == "__main__":
    unittest.main()
