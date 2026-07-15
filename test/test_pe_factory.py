import unittest

from src.position import (
    FourierPE,
    get_positional_encoding,
    MRoPE,
    MRoPEInterleave,
    RoPE,
)


class TestFactory(unittest.TestCase):
    def setUp(self):
        self.cases = [
            ("fourier", {"dim": 16, "n_bands": 16}, FourierPE),
            ("mrope_i", {"dim": 16, "base": 500}, MRoPEInterleave),
            ("mrope", {"dim": 16, "base": 500}, MRoPE),
            ("rope", {"dim": 16, "base": 500}, RoPE),
        ]

    def test_returns_correct_class(self):

        for name, kwargs, expected_class in self.cases:
            with self.subTest(name=name):
                module = get_positional_encoding(name, **kwargs)
                self.assertTrue(isinstance(module, expected_class))

    def test_kwargs_are_forwarded(self):

        for name, kwargs, _ in self.cases:
            with self.subTest(name=name):
                module = get_positional_encoding(name, **kwargs)

                for key, value in kwargs.items():
                    self.assertEqual(getattr(module, key), value)

    def test_unknown_type_raises(self):
        with self.assertRaises(ValueError):
            get_positional_encoding("not_a_real_encoding")


if __name__ == "__main__":
    unittest.main()
