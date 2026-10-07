import io
import struct

import numpy as np
import pytest

from hepatoscan.assistant.chat import Assistant
from hepatoscan.service.audit import AuditLog
from hepatoscan.service.handlers import AuthError, NotConfigured, Service, check_token
from hepatoscan.service.validation import UploadError, kind_from_name, validate_upload
from hepatoscan.volume import save_npz

MB = 1024 * 1024


def npz_bytes(vol, tmp_path):
    return save_npz(vol, tmp_path / "x.npz").read_bytes()


def test_kind_from_name_accepts_ct_formats_only():
    assert kind_from_name("scan.NII.GZ") == "nii.gz"
    assert kind_from_name("a.nii") == "nii" and kind_from_name("b.npz") == "npz"
    for bad in ("photo.png", "screenshot.jpg", "x.gif", "", None, "scan.nii.gz.exe"):
        with pytest.raises(UploadError):
            kind_from_name(bad)


def test_content_must_match_the_extension(tmp_path, phantoms):
    png = b"\x89PNG\r\n\x1a\n" + b"0" * 100
    with pytest.raises(UploadError):
        validate_upload(png, "scan.npz", MB)
    with pytest.raises(UploadError):
        validate_upload(png, "scan.nii", MB)
    with pytest.raises(UploadError):
        validate_upload(b"\x1f\x8b" + b"0" * 10, "scan.nii.gz", MB)
    with pytest.raises(UploadError):
        validate_upload(b"", "scan.npz", MB)


def test_size_limit(tmp_path, phantoms):
    data = npz_bytes(phantoms[0], tmp_path)
    with pytest.raises(UploadError):
        validate_upload(data, "scan.npz", len(data) - 1)
    assert validate_upload(data, "scan.npz", len(data)).image.shape == phantoms[0].image.shape


def test_nifti_header_check_without_nibabel():
    header = bytearray(352)
    header[:4] = struct.pack("<i", 348)
    header[344:348] = b"n+2\x00"  # NIfTI-2 magic is not accepted
    with pytest.raises(UploadError):
        validate_upload(bytes(header), "x.nii", MB)


def test_nifti_upload_round_trip(tmp_path, phantoms):
    nib = pytest.importorskip("nibabel")
    vol = phantoms[0]
    img = nib.Nifti1Image(np.transpose(vol.image, (2, 1, 0)), np.diag([*reversed(vol.spacing), 1.0]))
    img.header.set_zooms(tuple(reversed(vol.spacing)))
    buf = io.BytesIO(img.to_bytes())
    out = validate_upload(buf.getvalue(), "x.nii", 64 * MB)
    assert out.image.shape == vol.image.shape
    assert np.allclose(out.spacing, vol.spacing, atol=1e-5)


def test_token_rules():
    with pytest.raises(NotConfigured):
        check_token(None, "Bearer x")
    with pytest.raises(AuthError):
        check_token("secret", None)
    with pytest.raises(AuthError):
        check_token("secret", "Bearer wrong")
    check_token("secret", "Bearer secret")


@pytest.fixture
def service(fitted_baseline, tmp_path):
    return Service(fitted_baseline, Assistant(), AuditLog(tmp_path / "audit.jsonl"), "secret", 16 * MB)


def test_segment_upload_keeps_no_file_and_no_name(service, tmp_path, phantoms):
    data = npz_bytes(phantoms[6], tmp_path)
    service.audit.write("start")
    before = sorted(p.name for p in tmp_path.rglob("*"))
    out = service.segment_upload(data, "../../etc/passwd/patient_name.npz", "Bearer secret")
    assert out["summary"]["liver_volume_ml"] > 0 and "Not a diagnosis" in out["summary"]["note"]
    assert sorted(p.name for p in tmp_path.rglob("*")) == before  # the upload wrote no file
    log = (tmp_path / "audit.jsonl").read_text()
    assert "patient_name" not in log and "passwd" not in log and "segment_ok" in log


def test_rejected_upload_is_audited(service):
    with pytest.raises(UploadError):
        service.segment_upload(b"not a scan", "x.npz", "Bearer secret")
    assert service.audit.memory[-1]["event"] == "segment_rejected"


def test_segment_needs_the_token(service):
    with pytest.raises(AuthError):
        service.segment_upload(b"x", "x.npz", None)


def test_chat_through_the_service(service):
    out = service.chat({"question": "What does the lesion count mean?"}, "Bearer secret")
    assert out["citations"] and out["session_id"]
    again = service.chat({"question": "And the liver volume?", "session_id": out["session_id"]}, "Bearer secret")
    assert again["session_id"] == out["session_id"]
    assert service.end_chat(out["session_id"], "Bearer secret") == {"deleted": True}
