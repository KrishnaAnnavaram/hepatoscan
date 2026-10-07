"""Deep segmenters (needs the ``torch`` extra). This package file does not import torch."""


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True
