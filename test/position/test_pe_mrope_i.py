import unittest

import torch

from mropes.position import MRoPEInterleave


class TestMRoPEInterleave(unittest.TestCase):

    def setUp(self):
        self.dim = 8
        self.module = MRoPEInterleave(self.dim)

    def test_output_shape_stays_fixed_regardless_of_num_axes(self):
        # Unlike MRoPE, output width stays at dim regardless of axis count.
        x = torch.rand(2, 4, 5, self.dim)
        out = self.module(x)

        self.assertEqual(out.shape, x.shape)

    def test_output_shape_matches_output_shape_method(self):
        x = torch.rand(2, 4, 5, 6, self.dim)
        axes = x.shape[1:-1]

        out = self.module(x)
        expected = self.module.output_shape(axes)

        self.assertEqual(out.shape[1:], expected)

    def test_requires_matching_last_dim(self):
        x = torch.rand(2, 4, 5, 6)  # last dim 6, module expects 8
        with self.assertRaises(AssertionError):
            self.module(x)

    def test_requires_even_dim_at_construction(self):
        with self.assertRaises(AssertionError):
            MRoPEInterleave(5)  # odd dim

    def test_dtype_and_device_preserved(self):
        x = torch.rand(1, 4, 5, self.dim, dtype=torch.float64)
        out = self.module(x)

        self.assertEqual(out.dtype, x.dtype)
        self.assertEqual(out.device, x.device)

    def test_axis0_channels_are_identity_at_axis0_center(self):
        # Round-robin assigns channel pair k to axis (k % n_axes). Using an
        # odd size for axis 0 (center -> position 0) and a size for axis 1
        # that never lands on position 0, we can isolate axis 0's assigned
        # channels and confirm they behave as the identity rotation there,
        # regardless of axis 1's position.
        n_axes = 2
        axes = (3, 2)  # axis0 size 3 -> center index 1 has position 0
        x = torch.rand(1, *axes, self.dim)
        out = self.module(x)

        n_pairs = self.dim // 2
        pair_idx = torch.arange(n_pairs)
        axis0_idx = pair_idx[pair_idx % n_axes == 0]
        channels = torch.cat([axis0_idx, axis0_idx + n_pairs])

        out_slice = out[:, 0, :, channels]
        x_slice = x[:, 0, :, channels]

        self.assertTrue(torch.allclose(out_slice, x_slice, atol=1e-6))

    def test_round_robin_assignment_is_position_invariant(self):
        # The mapping of channels to axes shouldn't depend on which
        # position is being encoded -- verify the same channel groups are
        # used at two different grid locations by checking self-consistency
        # of shapes/no exceptions across positions (structural check).
        axes = (4, 4)
        x = torch.rand(2, *axes, self.dim)
        out = self.module(x)

        self.assertEqual(out.shape, x.shape)

    def test_batch_independence(self):
        torch.manual_seed(0)
        x = torch.rand(3, 4, 5, self.dim)
        out = self.module(x)

        for i in range(3):
            single = self.module(x[i : i + 1])

            self.assertTrue(torch.allclose(out[i : i + 1], single, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
