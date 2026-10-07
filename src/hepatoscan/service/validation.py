"""Upload validation. The service accepts CT volumes only: ``.npz``, ``.nii`` or ``.nii.gz``.

The file name from the client is used only to read the extension. It is never
used as a path. The content must match the extension (magic bytes), and the
size must be below the limit.
"""
from __future__ import annotations

import gzip
import io
import struct

from ..volume import Volume, VolumeError, load_nifti_bytes, load_npz_bytes

ZIP_MAGIC = b"PK\x03\x04"
GZIP_MAGIC = b"\x1f\x8b"
NIFTI1_MAGICS = (b"n+1\x00", b"ni1\x00")


class UploadError(ValueError):
    """The upload is not an accepted CT volume. The message is safe to show to the client."""


def kind_from_name(filename: str | None) -> str:
    name = (filename or "").strip().lower()
    for suffix, kind in ((".nii.gz", "nii.gz"), (".nii", "nii"), (".npz", "npz")):
        if name.endswith(suffix):
            return kind
    raise UploadError("accepted file types: .npz, .nii, .nii.gz (CT volumes only; photos and screenshots are refused)")


def _is_nifti1(header: bytes) -> bool:
    if len(header) < 348:
        return False
    size_le = struct.unpack("<i", header[:4])[0]
    size_be = struct.unpack(">i", header[:4])[0]
    return 348 in (size_le, size_be) and header[344:348] in NIFTI1_MAGICS


def validate_upload(data: bytes, filename: str | None, max_bytes: int) -> Volume:
    if not data:
        raise UploadError("empty upload")
    if len(data) > max_bytes:
        raise UploadError(f"upload is larger than {max_bytes // (1024 * 1024)} MB")
    kind = kind_from_name(filename)
    try:
        if kind == "npz":
            if not data.startswith(ZIP_MAGIC):
                raise UploadError("the content is not an npz archive")
            return load_npz_bytes(data, "upload")
        if kind == "nii.gz":
            if not data.startswith(GZIP_MAGIC):
                raise UploadError("the content is not gzip-compressed")
            limit = 8 * max_bytes  # stops a compression bomb
            raw = gzip.GzipFile(fileobj=io.BytesIO(data)).read(limit + 1)
            if len(raw) > limit:
                raise UploadError("the decompressed volume is too large")
            if not _is_nifti1(raw[:348]):
                raise UploadError("the content is not a NIfTI-1 image")
            return load_nifti_bytes(raw, "upload")
        if not _is_nifti1(data[:348]):
            raise UploadError("the content is not a NIfTI-1 image")
        return load_nifti_bytes(data, "upload")
    except VolumeError as exc:
        raise UploadError(str(exc)) from exc
    except (OSError, EOFError) as exc:
        raise UploadError("the file is damaged or truncated") from exc
