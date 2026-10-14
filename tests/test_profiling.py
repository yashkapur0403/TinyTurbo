"""Tests for the Stage 4 helpers.

The measured part of Stage 4 needs a GPU, but three things about it do not, and all three
have been wrong in profiler code before:

  * **which clock a total came from.** A CUDA trace has kernel time; a CPU-only trace has
    none. Attributing a CPU run against the device clock reports every millisecond as
    unattributed while the labelled host time sits unused, which reads as a finding about
    the loop rather than as a bug.
  * **what the residual is.** It is labelled time subtracted from measured wall clock, and
    nothing is split up beyond that.
  * **that the labeller gives the model back untouched.** It shadows `forward` on the
    instance, so a failure to restore would leave every later trial wrapped, including the
    ones in the sweep.
"""

from __future__ import annotations

import pytest
import torch

from conftest import REPO_ROOT

from tinyturbo.instrument import label_forward_calls, step_profiler_at_forward
from tinyturbo.profiling import (
    DEFAULT_CASES,
    LABEL_ASSISTANT,
    LABEL_TARGET,
    ProfiledCase,
    ProfilingError,
    TraceWindow,
    attributed_ms,
    cases_for,
    clock_key,
    profile_case,
    trace_clock,
)


def test_the_clock_follows_the_device():
    assert trace_clock("cuda") == "device"
    assert trace_clock("cuda:0") == "device"
    assert trace_clock("cpu") == "cpu"


def test_each_clock_reads_its_own_field():
    assert clock_key("device") == "self_device_ms"
    assert clock_key("cpu") == "self_cpu_ms"


def _labels(device_ms: float, cpu_ms: float) -> dict:
    return {
        LABEL_TARGET: {"count": 2, "self_device_ms": device_ms, "self_cpu_ms": cpu_ms},
        LABEL_ASSISTANT: {"count": 3, "self_device_ms": device_ms, "self_cpu_ms": cpu_ms},
    }


def test_attribution_adds_target_and_assistant_on_one_clock_only():
    labels = _labels(device_ms=10.0, cpu_ms=100.0)

    assert attributed_ms(labels, "device") == 20.0
    assert attributed_ms(labels, "cpu") == 200.0


def test_a_cpu_trace_is_not_attributed_against_the_empty_device_clock():
    """The bug this guards: a CPU trace has no device time, so 0.0 would be the whole story."""
    labels = _labels(device_ms=0.0, cpu_ms=100.0)

    assert attributed_ms(labels, "cpu") == 200.0
    assert attributed_ms(labels, "device") == 0.0


class _Stub(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, value=None):
        self.calls += 1
        return value


def test_the_labeller_gives_the_model_back_untouched():
    model = _Stub()
    original = model.forward

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]):
        with label_forward_calls(model, LABEL_TARGET):
            assert model.forward is not original, "the label must wrap the call it measures"
            model(value=1)
    assert model.calls == 1

    # Bound methods are rebuilt on every attribute access, so this is an equality check,
    # matching how tests/test_instrument.py checks the same restore on the counter.
    assert model.forward == original
    assert "forward" not in model.__dict__, "the temporary instance attribute must be removed"


# ---------------------------------------------------------------------------
# The bounded trace: a window instead of the whole generation
# ---------------------------------------------------------------------------


def test_a_window_must_record_something():
    with pytest.raises(ProfilingError) as error:
        TraceWindow(active=0)
    assert "at least one step" in str(error.value)

    assert TraceWindow(active=1).active == 1


def test_a_window_refuses_negative_skips_and_an_empty_repeat():
    with pytest.raises(ProfilingError):
        TraceWindow(active=4, wait=-1)
    with pytest.raises(ProfilingError):
        TraceWindow(active=4, repeat=0)


def test_a_window_records_what_it_means_and_the_step_it_counts():
    """The artifact has to say which loop iteration a step is, or the window is unreadable."""
    described = TraceWindow(active=4, wait=2, warmup=3).as_dict()

    assert described == {
        "active": 4,
        "wait": 2,
        "warmup": 3,
        "repeat": 1,
        "step": "one step per target forward pass",
    }


def test_a_window_stops_after_one_pass_by_default():
    """`repeat=1` is the point of a bound: a repeating schedule records forever."""
    assert TraceWindow(active=4).repeat == 1


def test_the_cases_can_be_narrowed_to_one_length_and_nothing_else():
    assert cases_for(None) == DEFAULT_CASES
    assert [case.max_new_tokens for case in cases_for(32)] == [32]
    assert [case.max_new_tokens for case in cases_for(256)] == [256]

    with pytest.raises(ProfilingError) as error:
        cases_for(7)
    assert "32" in str(error.value) and "256" in str(error.value)


class _Stepper(torch.nn.Module):
    """A model that counts its own forward passes and can be made to fail one."""

    def __init__(self, fail_after: int | None = None):
        super().__init__()
        self.calls = 0
        self.fail_after = fail_after

    def forward(self, *args, **kwargs):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("boom")
        return None


class _RecordingProfiler:
    """Just enough of a profiler for the stepping contract: it counts `step()` calls."""

    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1


def test_a_step_is_taken_once_per_completed_forward_pass():
    model = _Stepper()
    profiler = _RecordingProfiler()

    with step_profiler_at_forward(model, profiler) as returned:
        assert returned is model
        model()
        model(value=1)

    assert profiler.steps == 2
    assert model.calls == 2


def test_a_forward_pass_that_raises_does_not_advance_the_schedule():
    model = _Stepper(fail_after=1)
    profiler = _RecordingProfiler()

    with step_profiler_at_forward(model, profiler):
        model()
        with pytest.raises(RuntimeError):
            model()

    assert profiler.steps == 1


def test_the_stepper_gives_the_model_back_untouched_even_when_the_block_raises():
    model = _Stepper()
    original = model.forward

    try:
        with step_profiler_at_forward(model, _RecordingProfiler()):
            raise RuntimeError("boom")
    except RuntimeError:
        pass

    assert model.forward == original
    assert "forward" not in model.__dict__


# ---------------------------------------------------------------------------
# A window on a real (if tiny) trace, with no GPU
# ---------------------------------------------------------------------------


class _StubTokenizer:
    def __call__(self, text, return_tensors=None):
        return {"input_ids": torch.tensor([[1, 2, 3, 4]])}


class _StubOutput:
    def __init__(self, length: int):
        self.sequences = [list(range(length))]


class _StubModel(torch.nn.Module):
    """A model whose `generate` runs a fixed number of target passes, drafting alongside."""

    def __init__(self, iterations: int):
        super().__init__()
        self.iterations = iterations
        self.forward_calls = 0

    def forward(self, *args, **kwargs):
        self.forward_calls += 1
        return None

    def generate(self, **kwargs):
        assistant = kwargs.get("assistant_model")
        for _ in range(self.iterations):
            if assistant is not None:
                assistant()
            self()
        return _StubOutput(4 + self.iterations)


class _StubModels:
    def __init__(self, iterations: int, with_assistant: bool = True):
        self.tokenizer = _StubTokenizer()
        self.target = _StubModel(iterations)
        self.assistant = _StubModel(iterations) if with_assistant else None


class _StubConfig:
    def __init__(self, iterations: int):
        self.prompts_file = REPO_ROOT / "configs" / "prompts.jsonl"
        self.max_new_tokens = iterations
        self.num_assistant_tokens = 4


def _trace(iterations: int = 6, window: TraceWindow | None = None) -> dict:
    models = _StubModels(iterations)
    config = _StubConfig(iterations)
    case = ProfiledCase("s1-01-short-open", iterations, "test case")
    return profile_case(models, config, "cpu", case, window=window)["methods"]


def test_a_whole_generation_trace_records_every_step_of_the_loop():
    methods = _trace(iterations=6)

    assisted = methods["assisted"]
    assert assisted["covers_whole_generation"] is True
    assert assisted["labels"][LABEL_TARGET]["count"] == 6
    assert assisted["labels"][LABEL_ASSISTANT]["count"] == 6
    assert assisted["steps_recorded"] == 6
    assert assisted["attributed_ms"] is not None
    assert assisted["unattributed_ms"] is not None


def test_a_bounded_trace_records_a_window_and_runs_the_same_generation():
    """The bound is on what is *recorded*; the generation itself is untouched."""
    whole = _trace(iterations=12)["assisted"]
    bounded = _trace(iterations=12, window=TraceWindow(active=2, wait=1, warmup=1))["assisted"]

    assert bounded["tokens_generated"] == whole["tokens_generated"]
    assert bounded["labels"][LABEL_TARGET]["count"] < whole["labels"][LABEL_TARGET]["count"]
    assert bounded["labels"][LABEL_TARGET]["count"] <= 1 + 2  # warmup + the recorded window
    assert bounded["steps_recorded"] == bounded["labels"][LABEL_TARGET]["count"]


def test_a_clipped_whole_run_annotation_is_reported_as_unmeasurable_not_as_zero():
    """`record_function` around all of generate() does not survive a schedule's cycle end."""
    whole = _trace(iterations=12)["assisted"]
    bounded = _trace(iterations=12, window=TraceWindow(active=2))["assisted"]

    assert whole["labels"]["tinyturbo:generate"]["count"] == 1
    assert whole["whole_run_annotation_measurable"] is True

    assert bounded["labels"]["tinyturbo:generate"]["count"] is None
    assert bounded["whole_run_annotation_measurable"] is False


def test_a_window_that_never_opens_records_nothing_instead_of_crashing():
    """`--trace-wait` past the end of the run: the schedule never completes a cycle.

    `key_averages()` raises on a profiler that has no completed cycle, and the generation
    has already run by then, so this must come back as "nothing measured" rather than as an
    exception thrown part-way through writing an artifact.
    """
    bounded = _trace(iterations=3, window=TraceWindow(active=4, wait=50))["assisted"]

    assert bounded["steps_recorded"] == 0
    assert bounded["labels"][LABEL_TARGET]["count"] == 0
    assert bounded["means_from_trace"] == {"target_forward_ms": None, "assistant_forward_ms": None}
    assert bounded["covers_whole_generation"] is False
    assert bounded["attributed_ms"] is None
    assert bounded["tokens_generated"] == 3  # the generation itself still ran


def test_a_bounded_trace_does_not_pretend_to_cover_the_whole_run():
    """A window cannot carry a whole-run residual, so it does not carry one."""
    bounded = _trace(iterations=12, window=TraceWindow(active=2))["assisted"]

    assert bounded["covers_whole_generation"] is False
    assert bounded["attributed_ms"] is None
    assert bounded["unattributed_ms"] is None
    assert bounded["attributed_cpu_ms"] is None
    assert bounded["attributed_device_ms"] is None

    # What a window does measure: the per-forward means, on the window's own clock.
    assert bounded["clock"] == "cpu"
    assert bounded["means_from_trace"]["target_forward_ms"] is not None
    assert bounded["means_from_trace"]["assistant_forward_ms"] is not None


def test_the_labeller_restores_the_model_even_when_the_block_raises():
    model = _Stub()
    original = model.forward

    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]):
            with label_forward_calls(model, LABEL_TARGET):
                raise RuntimeError("boom")
    except RuntimeError:
        pass

    assert model.forward == original
    assert "forward" not in model.__dict__
