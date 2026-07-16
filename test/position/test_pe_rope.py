import unittest

import torch

from mropes.position import RoPE


class TestRotaryPositionEmbedding(unittest.TestCase):

    def setUp(self):
        self.module = RoPE(8)

    def test_output_shape_matches_input(self):
        x = torch.rand(2, 5, 8)
        out = self.module(x)
        self.assertEqual(out.shape, x.shape)

    def test_requires_even_last_dim(self):
        x = torch.rand(1, 3, 5)  # odd last dim
        with self.assertRaises(AssertionError):
            self.module(x)

    def test_dtype_and_device_preserved(self):
        x = torch.rand(1, 4, 8, dtype=torch.float64)
        out = self.module(x)
        self.assertEqual(out.dtype, x.dtype)
        self.assertEqual(out.device, x.device)

    def test_position_zero_is_identity(self):
        # At t=0 the rotation angle is 0, so cos=1, sin=0 -> output == input
        # for the first position along the sequence dim.
        x = torch.rand(1, 1, 8)  # sequence length 1 -> only position 0
        out = self.module(x)
        self.assertTrue(torch.allclose(out, x, atol=1e-6))

    def test_relative_position_property(self):
        # Core RoPE property: <RoPE(q, m), RoPE(k, n)> depends only on (m - n).
        # We verify this indirectly by checking that shifting a pair of
        # positions by the same offset leaves their dot product unchanged.
        torch.manual_seed(0)
        d = 8
        q = torch.rand(1, 1, d)
        k = torch.rand(1, 1, d)

        def rotated_dot(pos_q, pos_k, q_vec, k_vec):
            seq_len = max(pos_q, pos_k) + 1
            q_pad = torch.zeros(1, seq_len, d)
            k_pad = torch.zeros(1, seq_len, d)
            q_pad[:, pos_q] = q_vec
            k_pad[:, pos_k] = k_vec
            q_rot = self.module(q_pad)[:, pos_q]
            k_rot = self.module(k_pad)[:, pos_k]
            return (q_rot * k_rot).sum()

        dot_a = rotated_dot(2, 5, q, k)  # relative offset -3
        dot_b = rotated_dot(10, 13, q, k)  # relative offset -3

        self.assertTrue(torch.allclose(dot_a, dot_b, atol=1e-4))

    def test_batch_independence(self):
        torch.manual_seed(0)
        x = torch.rand(3, 4, 8)
        out = self.module(x)

        for i in range(3):
            single = self.module(x[i : i + 1])
            self.assertTrue(torch.allclose(out[i : i + 1], single, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
