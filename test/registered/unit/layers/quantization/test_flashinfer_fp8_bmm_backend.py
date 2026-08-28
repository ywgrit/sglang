import sys
from types import SimpleNamespace

from sglang.srt.layers.quantization import fp8_utils


def test_sm90_uses_cudnn_for_flashinfer_fp8_bmm(monkeypatch):
    monkeypatch.setattr(fp8_utils, "get_device_sm", lambda: 90)
    monkeypatch.setitem(
        sys.modules,
        "flashinfer.gemm.gemm_base",
        SimpleNamespace(CUDNN_AVAILABLE=True),
    )

    select_backend = getattr(
        fp8_utils,
        "_select_flashinfer_fp8_bmm_backend",
        lambda: "missing",
    )
    if hasattr(select_backend, "cache_clear"):
        select_backend.cache_clear()

    assert select_backend() == "cudnn"


def test_non_sm90_keeps_cublas_for_flashinfer_fp8_bmm(monkeypatch):
    monkeypatch.setattr(fp8_utils, "get_device_sm", lambda: 100)

    select_backend = getattr(
        fp8_utils,
        "_select_flashinfer_fp8_bmm_backend",
        lambda: "missing",
    )
    if hasattr(select_backend, "cache_clear"):
        select_backend.cache_clear()

    assert select_backend() == "cublas"
