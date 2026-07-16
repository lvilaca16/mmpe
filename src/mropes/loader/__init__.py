from .audio import AudioDataset, get_audio
from .image import ImageDataset, get_image
from .video import VideoDataset, get_video

SUPPORT_TYPES = ["audio", "video", "image"]

_DATASET_REGISTRY = {
    "audio": AudioDataset,
    "video": VideoDataset,
    "image": ImageDataset,
}

_READ_REGISTRY = {
    "audio": get_audio,
    "video": get_video,
    "image": get_image,
}


def get_dataset(name: str, **kwargs):
    """
    Factory method to construct a dataset by name.

    Args:
        name: one of SUPPORT_TYPES ("audio", "video", "image").
        **kwargs: forwarded to the underlying dataset's constructor.

    Returns:
        A torch.utils.data.Dataset instance implementing the requested dataset.
    """
    if name not in _DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset type '{name}'. "
            f"Supported types: {SUPPORT_TYPES}"
        )
    return _DATASET_REGISTRY[name](**kwargs)


def get_reader(name: str):
    """
    Factory method to retrieve the raw-data reading function by name.

    Args:
        name: one of SUPPORT_TYPES ("audio", "video", "image").

    Returns:
        The reader function associated with the given type (e.g.
        get_audio, get_video, get_image).
    """
    if name not in _READ_REGISTRY:
        raise ValueError(
            f"Unknown dataset type '{name}'. "
            f"Supported types: {SUPPORT_TYPES}"
        )
    return _READ_REGISTRY[name]
