"""Instrumentation: counting forward passes, timing, and memory readings.

The number that explains assisted decoding is not the latency, it is how many tokens
the target model accepts per forward pass. The library does not expose that, so it is
measured directly by counting forward passes on both models.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from .metrics import median


@dataclass
class ForwardCounts:
    """How many times each model ran a forward pass during one trial."""

    target: int = 0
    assistant: int = 0

    def as_dict(self) -> dict:
        return {
            "target_forwards": self.target,
            "assistant_forwards": self.assistant,
        }


@contextmanager
def count_forward_calls(model, field_name: str, counts: ForwardCounts):
    """Count every forward pass through `model` inside the `with` block.

    How it works: `torch.nn.Module.__call__` looks up `self.forward` at call time, so
    assigning that attribute on the *instance* shadows the class method. This catches
    the call that matters - transformers' assisted decoding loop calls the target
    model as `self(**model_inputs)` - and also a direct `model.forward(...)`.

    The original attribute is restored in a `finally` block, including the case where
    the instance had no `forward` of its own, so the model is left exactly as found.
    """
    if field_name not in ("target", "assistant"):
        raise ValueError(f"unknown counter field {field_name!r}")

    had_instance_attribute = "forward" in model.__dict__
    previous = model.__dict__.get("forward")
    original = model.forward

    def counted_forward(*args, **kwargs):
        setattr(counts, field_name, getattr(counts, field_name) + 1)
        return original(*args, **kwargs)

    model.forward = counted_forward
    try:
        yield counts
    finally:
        if had_instance_attribute:
            model.forward = previous
        else:
            del model.forward


@contextmanager
def label_forward_calls(model, label: str):
    """Wrap every forward pass through `model` in a `torch.profiler` label.

    A trace on its own is a wall of identical kernels: nothing in it says which model a
    matmul belonged to. The assisted loop calls the target and the assistant through the
    same `__call__` path, so the only place the distinction can be introduced is here,
    around the instance's `forward`. Stage 4 uses this to attribute kernel time to
    drafting versus verification instead of guessing from the shape of the trace.

    The same instance-level shadowing trick as `count_forward_calls`, and the same
    guarantee: the model is left exactly as it was found. The two can be nested in
    either order; the counter counts the call and the label marks it.
    """
    import torch.profiler

    had_instance_attribute = "forward" in model.__dict__
    previous = model.__dict__.get("forward")
    original = model.forward

    def labelled_forward(*args, **kwargs):
        with torch.profiler.record_function(label):
            return original(*args, **kwargs)

    model.forward = labelled_forward
    try:
        yield model
    finally:
        if had_instance_attribute:
            model.forward = previous
        else:
            del model.forward


@contextmanager
def step_profiler_at_forward(model, profiler):
    """Advance a `torch.profiler` schedule once per forward pass through `model`.

    A `torch.profiler.schedule` only moves when someone calls `profiler.step()`, and a
    single `generate()` call is one opaque block from the profiler's point of view. The
    forward pass is the loop's natural iteration: it is what the schedule should count, and
    counting it here is what lets a bounded trace record a window of the decode loop
    instead of every step of it.

    One step per *completed* forward - a call that raises does not advance the schedule,
    and the run is over either way. Same instance-level shadowing as the labeller, and the
    same guarantee: the model is left exactly as found, even if the block raises. Nest
    this inside `label_forward_calls` so the step lands just after the labelled region
    closes, on the boundary the schedule expects.

    Nothing here is specific to a GPU: a schedule works the same on a CPU trace, which is
    why the bounded path can be exercised without one.
    """
    had_instance_attribute = "forward" in model.__dict__
    previous = model.__dict__.get("forward")
    original = model.forward

    def stepped_forward(*args, **kwargs):
        result = original(*args, **kwargs)
        profiler.step()
        return result

    model.forward = stepped_forward
    try:
        yield model
    finally:
        if had_instance_attribute:
            model.forward = previous
        else:
            del model.forward


class Stopwatch:
    """Wall-clock timing that understands CUDA's asynchrony.

    A CUDA kernel launch returns as soon as the work is queued, so timing without
    torch.cuda.synchronize() measures how fast the CPU could enqueue work rather than
    how long the GPU took to do it. Both `start()` and `stop()` synchronise first when
    the device is a GPU. On CPU synchronisation is not needed and is skipped.
    """

    def __init__(self, device: str):
        self._device = device
        self._started_at: float | None = None

    def _synchronise(self) -> None:
        if self._device.startswith("cuda"):
            torch.cuda.synchronize()

    def start(self) -> None:
        self._synchronise()
        self._started_at = time.perf_counter()

    def stop(self) -> float:
        """Elapsed time in milliseconds."""
        if self._started_at is None:
            raise RuntimeError("Stopwatch.stop() called before start()")
        self._synchronise()
        elapsed_ms = (time.perf_counter() - self._started_at) * 1000.0
        self._started_at = None
        return elapsed_ms


def reset_peak_memory(device: str) -> None:
    """Reset the peak-memory counters so each trial reports its own peak, not the run's."""
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()


def peak_memory_mb(device: str) -> dict:
    """Peak allocated and reserved GPU memory since the last reset, in MiB.

    On CPU there is no equivalent counter, so this returns None fields rather than a
    made-up number. The reporter treats missing memory as missing, not as zero.
    """
    if not device.startswith("cuda"):
        return {"peak_alloc_mb": None, "peak_reserved_mb": None}
    return {
        "peak_alloc_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / (1024 * 1024),
    }


def measure_prefill_ms(model, model_inputs: dict, device: str, repeats: int = 3) -> float:
    """Time a single forward pass over the whole prompt, with an empty cache.

    This is the part of the latency an assistant cannot help with: both arms pay it
    once per generation. Measuring it separately lets the report show how much of the
    end-to-end time is prompt processing rather than decoding.

    The median of a few repeats is used because the first call also pays one-off
    allocator and kernel-selection costs.
    """
    watch = Stopwatch(device)
    samples: list[float] = []
    with torch.inference_mode():
        for _ in range(max(1, repeats)):
            watch.start()
            model(**model_inputs, use_cache=True)
            samples.append(watch.stop())
    return median(samples)
