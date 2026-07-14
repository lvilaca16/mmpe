from .fourier import FourierPE
from .mrope import MRoPE
from .rope import RoPE

SUPPORT_TYPES = ["rope", "mrope", "fourier"]

_ENCODING_REGISTRY = {
    "fourier": FourierPE,
    "mrope": MRoPE,
    "rope": RoPE,
}


def get_positional_encoding(name: str, **kwargs):
    """
    Factory method to construct a positional encoding module by name.

    Args:
        name: one of SUPPORT_TYPES ("rope", "fourier").
        **kwargs: forwarded to the underlying module's constructor
                  (e.g. base=10000 for rope, n_bands=64 for fourier).

    Returns:
        An nn.Module instance implementing the requested positional encoding.
    """
    if name not in _ENCODING_REGISTRY:
        raise ValueError(
            f"Unknown positional encoding type '{name}'. "
            f"Supported types: {SUPPORT_TYPES}"
        )
    return _ENCODING_REGISTRY[name](**kwargs)
