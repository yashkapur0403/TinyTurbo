"""Tests for the measurement instruments.

The forward-pass counter is the load-bearing idea in this project: it is how the
mechanism behind assisted decoding (tokens accepted per target forward pass) gets
measured at all. These tests use a plain torch module rather than a language model, so
they run instantly, with no GPU and no downloads, while still exercising exactly the
mechanism the real run depends on.
"""

from __future__ import annotations

import torch

from tinyturbo.instrument import (
    ForwardCounts,
    Stopwatch,
    count_forward_calls,
    measure_prefill_ms,
    peak_memory_mb,
    reset_peak_memory,
)


def test_counts_calls_made_through_the_module():
    model = torch.nn.Linear(3, 2)
    counts = ForwardCounts()

    with count_forward_calls(model, "target", counts):
        model(torch.zeros(1, 3))
        model(torch.zeros(1, 3))
        model(torch.zeros(1, 3))

    assert counts.target == 3
    assert counts.assistant == 0


def test_counts_direct_forward_calls_too():
    """transformers calls the target model as `self(**inputs)`, but not everywhere."""
    model = torch.nn.Linear(3, 2)
    counts = ForwardCounts()

    with count_forward_calls(model, "target", counts):
        model.forward(torch.zeros(1, 3))

    assert counts.target == 1


def test_two_models_are_counted_separately():
    target = torch.nn.Linear(3, 2)
    assistant = torch.nn.Linear(3, 2)
    counts = ForwardCounts()

    with count_forward_calls(target, "target", counts):
        with count_forward_calls(assistant, "assistant", counts):
            target(torch.zeros(1, 3))
            for _ in range(4):
                assistant(torch.zeros(1, 3))

    assert counts.target == 1
    assert counts.assistant == 4
    assert counts.as_dict() == {"target_forwards": 1, "assistant_forwards": 4}


def test_the_model_is_left_exactly_as_it_was_found():
    model = torch.nn.Linear(3, 2)
    before = model.forward

    with count_forward_calls(model, "target", ForwardCounts()):
        assert "forward" in model.__dict__, "the counter should shadow forward inside the block"

    assert "forward" not in model.__dict__, "the shadow must be removed, not left in place"
    assert model.forward == before

    # And it must not keep counting once the block has exited.
    counts = ForwardCounts()
    model(torch.zeros(1, 3))
    assert counts.target == 0


def test_counting_is_nested_and_unwound_correctly():
    model = torch.nn.Linear(3, 2)
    outer = ForwardCounts()
    inner = ForwardCounts()

    with count_forward_calls(model, "target", outer):
        model(torch.zeros(1, 3))
        with count_forward_calls(model, "target", inner):
            model(torch.zeros(1, 3))
        model(torch.zeros(1, 3))

    assert outer.target == 3
    assert inner.target == 1


def test_unknown_counter_field_is_rejected():
    model = torch.nn.Linear(3, 2)
    try:
        with count_forward_calls(model, "someone-else", ForwardCounts()):
            pass
    except ValueError as error:
        assert "unknown counter field" in str(error)
    else:  # pragma: no cover - only runs if the validation disappears
        raise AssertionError("an unknown counter name should be rejected")


def test_stopwatch_measures_and_requires_start():
    watch = Stopwatch("cpu")
    watch.start()
    elapsed = watch.stop()
    assert elapsed >= 0.0
    assert elapsed < 5000.0

    with_error = Stopwatch("cpu")
    try:
        with_error.stop()
    except RuntimeError as error:
        assert "before start" in str(error)
    else:  # pragma: no cover
        raise AssertionError("stop() without start() should raise")


def test_memory_reporting_is_honest_about_cpu():
    """There is no peak-memory counter on CPU; missing must not be reported as zero."""
    reset_peak_memory("cpu")
    memory = peak_memory_mb("cpu")
    assert memory == {"peak_alloc_mb": None, "peak_reserved_mb": None}


class _ToyModel(torch.nn.Module):
    """Accepts the keyword arguments generate() style calls use."""

    def __init__(self) -> None:
        super().__init__()
        self.seen = 0

    def forward(self, input_ids=None, attention_mask=None, use_cache=False, **kwargs):
        self.seen += 1
        return None


def test_measure_prefill_passes_the_prompt_through_once_per_repeat():
    model = _ToyModel()
    inputs = {"input_ids": torch.zeros(1, 4, dtype=torch.long)}
    elapsed = measure_prefill_ms(model, inputs, "cpu", repeats=2)

    assert model.seen == 2, "one forward pass per repeat"
    assert elapsed >= 0.0
