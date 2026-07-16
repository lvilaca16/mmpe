import unittest

import torch

from mropes.position import MRoPE


class TestMRoPE(unittest.TestCase):

    def setUp(self):
        self.dim = 32
        self.module = MRoPE(self.dim)

    def test_output_shape_grows_with_num_axes(self):
        # MRoPE concatenates a full rotation per axis, so output width is
        # n_axes * dim, not dim.
        x = torch.rand(2, 4, 5, self.dim)
        out = self.module(x)
        self.assertEqual(out.shape, (2, 4, 5, self.dim))

    def test_output_shape_matches_output_shape_method(self):
        x = torch.rand(2, 5, 6, self.dim)
        axes = x.shape[1:-1]

        out = self.module(x)
        expected = self.module.output_shape(axes)

        self.assertEqual(out.shape[1:], expected)

    def test_requires_matching_last_dim(self):
        x = torch.rand(2, 4, 5, 6)  # last dim 6, module expects 8

        with self.assertRaises(AssertionError):
            self.module(x)

    def test_requires_even_last_dim(self):
        module = MRoPE(5)  # odd dim
        x = torch.rand(1, 3, 4, 5)

        with self.assertRaises(AssertionError):
            module(x)

    def test_dtype_and_device_preserved(self):
        x = torch.rand(1, 4, 5, self.dim, dtype=torch.float64)
        out = self.module(x)

        self.assertEqual(out.dtype, x.dtype)
        self.assertEqual(out.device, x.device)

    def test_position_zero_is_identity(self):
        axes = (4, 5)  # any sizes work now -- position 0 is always index 0
        x = torch.rand(1, *axes, self.dim)
        out = self.module(x)

        corner = x[:, 0, 0]
        out_corner = out[:, 0, 0]

        self.assertTrue(torch.allclose(out_corner, corner, atol=1e-6))

    def test_batch_independence(self):
        torch.manual_seed(0)
        x = torch.rand(3, 4, 5, self.dim)
        out = self.module(x)

        for i in range(3):
            single = self.module(x[i : i + 1])
            self.assertTrue(torch.allclose(out[i : i + 1], single, atol=1e-6))

    def test_single_axis_matches_two_axis_slice(self):
        # For a single-axis input, MRoPE should behave like a single full
        # rotation concatenated with itself once (n_axes=1).
        module = MRoPE(self.dim)
        x = torch.rand(1, 5, self.dim)  # one spatial axis
        out = module(x)

        self.assertEqual(out.shape, (1, 5, self.dim))  # 1 axis -> no growth


if __name__ == "__main__":
    unittest.main()
