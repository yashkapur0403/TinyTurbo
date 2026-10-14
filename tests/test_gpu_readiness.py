"""Checks that only matter once a GPU is involved, but can be tested without one.

A GPU run has to be able to answer "which hardware produced this?" and "can these two
models work together at all?". Both questions can be exercised on CPU, which is what
this file does, so that the remote run is not the first time the code paths execute.
"""

from __future__ import annotations

import pytest

from tinyturbo.env import (
    _driver_version_from_report,
    compute_capability,
    default_dtype_name,
    hardware_fingerprint,
    resolve_attention,
    resolve_dtype,
)
from tinyturbo.models import ModelError, assert_compatible_vocabularies, vocab_size


class _TextConfig:
    def __init__(self, size: int) -> None:
        self.vocab_size = size


class _Config:
    def __init__(self, size: int | None) -> None:
        self._text = _TextConfig(size) if size is not None else None
        if size is not None:
            self.vocab_size = size

    def get_text_config(self):
        if self._text is None:
            raise AttributeError("no text config")
        return self._text


class _StubModel:
    def __init__(self, size: int | None) -> None:
        self.config = _Config(size)


# ---------------------------------------------------------------------------
# Hardware fingerprint
# ---------------------------------------------------------------------------


def test_cpu_fingerprint_invents_nothing():
    """Every GPU field must be explicitly None on a CPU run, not zero or a guess."""
    fingerprint = hardware_fingerprint("cpu")

    assert fingerprint["is_cuda"] is False
    for field in (
        "gpu_name",
        "gpu_compute_capability",
        "gpu_memory_total_mb",
        "gpu_driver_version",
        "nvidia_smi_report",
        "cuda_version",
    ):
        assert fingerprint[field] is None, f"{field} must be None on CPU, got {fingerprint[field]}"

    for field in ("python", "platform", "torch", "transformers", "device"):
        assert fingerprint[field], f"{field} should always be recorded"


def test_fingerprint_carries_the_fields_a_gpu_report_needs():
    """The keys a Step-5 hardware report reads must exist before the GPU run produces one."""
    fingerprint = hardware_fingerprint("cpu")
    for field in (
        "gpu_name",
        "gpu_compute_capability",
        "gpu_memory_total_mb",
        "gpu_driver_version",
        "cudnn_version",
        "cuda_version",
        "torch",
        "transformers",
        "python",
    ):
        assert field in fingerprint, f"missing fingerprint field: {field}"


def test_compute_capability_is_none_off_gpu():
    assert compute_capability("cpu") is None


def test_driver_version_is_parsed_from_an_nvidia_smi_line():
    # Shape of `nvidia-smi --query-gpu=driver_version,name,memory.total --format=csv,noheader,nounits`
    report = "580.65.06, Tesla T4, 15360"
    assert _driver_version_from_report(report) == "580.65.06"
    assert _driver_version_from_report("535.104.05, Tesla P100-PCIE-16GB, 16384") == "535.104.05"


def test_driver_version_parsing_is_safe_on_missing_input():
    assert _driver_version_from_report(None) is None
    assert _driver_version_from_report("") is None


# ---------------------------------------------------------------------------
# Hardware-dependent dtype and attention choices
# ---------------------------------------------------------------------------


def test_cpu_dtype_defaults_to_float32():
    assert default_dtype_name("cpu") == "float32"
    assert resolve_dtype("auto", "cpu") == "float32"


def test_float16_on_cpu_is_refused_with_an_explanation():
    with pytest.raises(Exception) as error:
        resolve_dtype("float16", "cpu")
    assert "float16" in str(error.value)


def test_cpu_attention_defaults_to_eager():
    assert resolve_attention("auto", "cpu") == "eager"


def test_flash_attention_on_cpu_is_refused():
    with pytest.raises(Exception) as error:
        resolve_attention("flash_attention_2", "cpu")
    assert "flash_attention_2" in str(error.value)


# ---------------------------------------------------------------------------
# Vocabulary compatibility
# ---------------------------------------------------------------------------


def test_vocab_size_reads_through_the_same_accessor_transformers_uses():
    assert vocab_size(_StubModel(49152)) == 49152


def test_vocab_size_falls_back_when_there_is_no_text_config():
    class Legacy:
        class config:  # noqa: N801 - mimics an older config object
            vocab_size = 32000

    assert vocab_size(Legacy()) == 32000


def test_matching_vocabularies_are_accepted():
    target, assistant = assert_compatible_vocabularies(_StubModel(49152), _StubModel(49152))
    assert (target, assistant) == (49152, 49152)


def test_mismatched_vocabularies_fail_before_any_generation():
    with pytest.raises(ModelError) as error:
        assert_compatible_vocabularies(_StubModel(49152), _StubModel(32000))
    message = str(error.value)
    assert "49152" in message and "32000" in message
    assert "same token space" in message


def test_unknown_vocabularies_do_not_produce_a_false_alarm():
    """If either size cannot be read, the guard must stay quiet rather than guess."""
    assert assert_compatible_vocabularies(_StubModel(None), _StubModel(49152)) == (None, 49152)
