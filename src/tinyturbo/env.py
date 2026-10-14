"""Device selection, dtype/attention resolution and the hardware fingerprint.

Two jobs:

  1. Turn a portable request ("auto") into a concrete choice for this machine. A Tesla
     T4 cannot use bfloat16 efficiently and cannot run FlashAttention-2 at all, so the
     working dtype and attention implementation are properties of the hardware, not
     constants in the code.
  2. Record what the machine actually was. A latency number with no fingerprint cannot
     be compared with anything and cannot be defended.
"""

from __future__ import annotations

import platform
import subprocess

import torch
import transformers


class DeviceError(RuntimeError):
    """Raised when the requested device, dtype or attention backend cannot work here."""


def detect_device(preferred: str = "auto") -> str:
    """Resolve the device string. 'auto' means cuda when available, otherwise cpu."""
    if preferred == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if preferred.startswith("cuda") and not torch.cuda.is_available():
        raise DeviceError(
            f"device '{preferred}' was requested but torch.cuda.is_available() is False "
            f"(torch {torch.__version__} was built for {_torch_build()})"
        )
    return preferred


def _torch_build() -> str:
    return "cuda" if torch.version.cuda else "cpu"


def compute_capability(device: str) -> tuple[int, int] | None:
    """The GPU's compute capability, e.g. (7, 5) for a Tesla T4. None on CPU."""
    if not device.startswith("cuda"):
        return None
    return tuple(torch.cuda.get_device_capability())  # type: ignore[return-value]


def default_dtype_name(device: str) -> str:
    """The dtype to use when the config says 'auto'."""
    if not device.startswith("cuda"):
        # The CPU path is for correctness checks only. float32 is the only dtype that
        # is both fast and well supported on CPU.
        return "float32"
    capability = compute_capability(device) or (0, 0)
    # bfloat16 is only fast from Ampere (sm80) onwards. On Turing/T4 use float16.
    return "bfloat16" if capability >= (8, 0) else "float16"


def resolve_dtype(requested: str, device: str) -> str:
    """Validate the requested dtype and resolve 'auto' for this device.

    float16 on CPU is rejected rather than warned about: torch emulates it, it is much
    slower than float32, and a latency measurement taken that way would be a
    measurement of the emulation, not of the model.
    """
    name = default_dtype_name(device) if requested == "auto" else requested
    if not device.startswith("cuda") and name == "float16":
        raise DeviceError(
            "dtype float16 is not supported on CPU in this project: it is emulated and "
            "misleadingly slow. Use float32 (or bfloat16) for CPU runs."
        )
    if name == "bfloat16" and device.startswith("cuda"):
        capability = compute_capability(device) or (0, 0)
        if capability < (8, 0):
            raise DeviceError(
                f"dtype bfloat16 requires compute capability >= 8.0, but this device is "
                f"{capability[0]}.{capability[1]}. Use float16 on this hardware."
            )
    return name


def resolve_attention(requested: str, device: str) -> str:
    """Validate the requested attention implementation and resolve 'auto'.

    FlashAttention-2 needs Ampere or newer. Requesting it on older hardware would fail
    deep inside the model, so it is caught here where the message can be useful.
    """
    if requested == "auto":
        # FlashAttention-2 is never chosen automatically: it needs a separate install and
        # Ampere or newer, and a silent fallback would mean two arms of the comparison
        # silently ran different attention kernels.
        return "sdpa" if device.startswith("cuda") else "eager"

    if requested == "flash_attention_2":
        if not device.startswith("cuda"):
            raise DeviceError("flash_attention_2 requires a CUDA device")
        capability = compute_capability(device) or (0, 0)
        if capability < (8, 0):
            raise DeviceError(
                f"flash_attention_2 requires compute capability >= 8.0, but this device "
                f"is {capability[0]}.{capability[1]}. Use sdpa on this hardware."
            )
    return requested


def dtype_from_name(name: str) -> torch.dtype:
    """Map a recorded dtype name back to the torch dtype object."""
    table = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in table:
        raise DeviceError(f"unknown dtype name {name!r}")
    return table[name]


def nvidia_smi_report() -> str | None:
    """The first GPU line reported by nvidia-smi, or None when it is unavailable.

    torch.cuda exposes the CUDA runtime, the GPU name and the cudnn version, but not the
    driver version, and the driver is part of what makes a GPU measurement reproducible.
    nvidia-smi is present on Colab and Kaggle images. If it is missing, or this is not a
    GPU machine at all, the caller records None rather than inventing a value.
    """
    command = [
        "nvidia-smi",
        "--query-gpu=driver_version,name,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=20, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return lines[0] if lines else None


def _driver_version_from_report(report: str | None) -> str | None:
    """Pull the driver version out of an nvidia-smi line: driver, name, memory total."""
    if not report:
        return None
    parts = [part.strip() for part in report.split(",")]
    return parts[0] or None


def hardware_fingerprint(device: str) -> dict:
    """A record of the machine and software stack, stored with every run.

    Every field is a measurement or is explicitly None. A GPU run records the device
    name, compute capability, total memory, CUDA runtime and cudnn versions and, when
    nvidia-smi is available, the driver version as well.
    """
    fingerprint = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "torch_build": _torch_build(),
        "transformers": transformers.__version__,
        "device": device,
        "is_cuda": device.startswith("cuda"),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_name": None,
        "gpu_compute_capability": None,
        "gpu_memory_total_mb": None,
        "gpu_driver_version": None,
        "nvidia_smi_report": None,
        "cuda_version": torch.version.cuda,
        "cudnn_version": None,
        "cpu_count": None,
        "torch_threads": torch.get_num_threads(),
    }

    try:
        import os

        fingerprint["cpu_count"] = os.cpu_count()
    except Exception:  # pragma: no cover - os.cpu_count() is best-effort
        pass

    if device.startswith("cuda"):
        capability = compute_capability(device) or (0, 0)
        fingerprint["gpu_name"] = torch.cuda.get_device_name(0)
        fingerprint["gpu_compute_capability"] = [capability[0], capability[1]]
        try:
            properties = torch.cuda.get_device_properties(0)
            fingerprint["gpu_memory_total_mb"] = properties.total_memory / (1024 * 1024)
        except Exception:  # pragma: no cover - defensive
            pass
        try:
            fingerprint["cudnn_version"] = torch.backends.cudnn.version()
        except Exception:  # pragma: no cover - not available on every build
            fingerprint["cudnn_version"] = None

        report = nvidia_smi_report()
        fingerprint["nvidia_smi_report"] = report
        fingerprint["gpu_driver_version"] = _driver_version_from_report(report)

    return fingerprint
