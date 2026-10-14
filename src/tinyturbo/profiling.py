"""Stage 4: where the per-step time actually goes, on two representative cases.

The calibration in `calibrate.py` times the primitives in isolation. It cannot see the
Python-level work between forward passes - the candidate loop, cache bookkeeping, the
per-step stopping criteria - and that gap is exactly what H3 is about. This module runs
the *real* `generate()` call under `torch.profiler` with the two models' forward passes
labelled, so kernel time can be attributed to drafting versus verification instead of
being inferred from a residual.

Two cases only, both taken from the sweep that already ran: the shortest generation
length, where fixed per-step overhead is a large share of a short run, and the longest,
where the decode loop dominates. Nothing here re-measures the benchmark; it re-runs two
of its points with a profiler attached and compares the attribution against the wall
clock that point already recorded.

Honesty rules this module follows:

  * every number is labelled with the clock it came from (GPU kernel time or CPU time),
    and which of the two is meaningful depends on the profiler activities that ran;
  * the labelled regions cover the forward passes only, so whatever wall-clock time they
    do not account for is reported as *unattributed*, never split up by assumption;
  * nothing is smoothed, scaled or fitted to make the total match.

The whole-generation trace needs about both models plus one profiler event per kernel in a
256-token assisted generation, which is more than a free Colab instance has (see
`docs/FINDINGS.md` §7 and §9). `TraceWindow` is the bounded alternative: a
`torch.profiler.schedule` that records a window of the decode loop and stops, so the event
buffer is a slice rather than the whole run. It is opt-in, and because a window is not the
whole generation the artifact says so explicitly - the windowed numbers describe the loop
during the window, and the whole-run residual is left unset rather than computed from a
part of the run it does not cover.
"""

from __future__ import annotations

import datetime as _datetime
from contextlib import nullcontext
from dataclasses import dataclass

import torch

from .env import hardware_fingerprint
from .instrument import Stopwatch, label_forward_calls, step_profiler_at_forward
from .methods import build_generate_kwargs
from .prompts import get_prompt, load_prompts


class ProfilingError(ValueError):
    """A Stage 4 request that cannot be honoured, as opposed to one that failed to run."""


@dataclass(frozen=True)
class TraceWindow:
    """A bounded slice of the decode loop, for `torch.profiler.schedule`.

    The profiler is stepped once per target forward pass, so a step is one iteration of
    the assisted loop. `wait` then `warmup` iterations pass unrecorded, `active` of them
    are recorded, and the schedule repeats `repeat` times - `repeat=1` stops recording
    after that one window, which is the point: the event buffer scales with the window
    rather than with the generation length, and on a long generation that is the
    difference between a trace that fits in the session's memory and one that does not.
    """

    active: int
    wait: int = 1
    warmup: int = 1
    repeat: int = 1

    def __post_init__(self) -> None:
        if self.active < 1:
            raise ProfilingError(f"a trace window must record at least one step, got {self.active}")
        if self.wait < 0 or self.warmup < 0:
            raise ProfilingError("wait and warmup cannot be negative")
        if self.repeat < 1:
            raise ProfilingError(f"repeat must be at least 1, got {self.repeat}")

    def as_dict(self) -> dict:
        return {
            "active": self.active,
            "wait": self.wait,
            "warmup": self.warmup,
            "repeat": self.repeat,
            "step": "one step per target forward pass",
        }

    def as_schedule(self) -> "torch.profiler.schedule":
        return torch.profiler.schedule(
            wait=self.wait, warmup=self.warmup, active=self.active, repeat=self.repeat
        )


LABEL_TARGET = "tinyturbo:target_forward"
LABEL_ASSISTANT = "tinyturbo:assistant_forward"
LABEL_GENERATE = "tinyturbo:generate"
LABELS = (LABEL_GENERATE, LABEL_TARGET, LABEL_ASSISTANT)


@dataclass(frozen=True)
class ProfiledCase:
    """One (prompt, generation length) pair to trace, and why that pair was chosen."""

    prompt_id: str
    max_new_tokens: int
    rationale: str

    def as_dict(self) -> dict:
        return {
            "prompt_id": self.prompt_id,
            "max_new_tokens": self.max_new_tokens,
            "rationale": self.rationale,
        }


DEFAULT_CASES = (
    ProfiledCase(
        "s1-01-short-open",
        32,
        "shortest generation length in the sweep: fixed per-step overhead is the "
        "largest share of the run here",
    ),
    ProfiledCase(
        "s2-01-long-open",
        256,
        "longest generation length in the sweep: the decode loop dominates, so the "
        "attribution is least sensitive to start-up effects",
    ),
)


def cases_for(max_new_tokens: int | None = None, cases=DEFAULT_CASES) -> tuple:
    """The cases to trace, optionally narrowed to one generation length.

    `--max-new-tokens` means the same thing here as it does for `run`: select the point
    rather than redefining it. A length that no case uses is an error rather than an
    empty trace, because a profiling artifact with no cases in it looks like a measurement
    that found nothing.
    """
    if max_new_tokens is None:
        return tuple(cases)
    selected = tuple(case for case in cases if case.max_new_tokens == max_new_tokens)
    if not selected:
        available = ", ".join(str(case.max_new_tokens) for case in cases)
        raise ProfilingError(
            f"no profiled case generates {max_new_tokens} tokens; the cases are "
            f"{available}"
        )
    return selected


def _profiler_activities(device: str) -> list:
    """CPU activity always; GPU activity only where there is a GPU to trace."""
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.startswith("cuda"):
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    return activities


def trace_clock(device: str) -> str:
    """Which clock the attribution is expressed in.

    A CUDA trace has kernel time and that is the one to attribute; a CPU-only trace has no
    device time at all, so attributing against it would report every millisecond as
    unattributed while the labelled host time sat unused. The chosen clock is recorded in
    the artifact next to the numbers, so a reader always knows which one a total came from.
    """
    return "device" if device.startswith("cuda") else "cpu"


def clock_key(clock: str) -> str:
    """The `key_averages()` field that carries time on the chosen clock."""
    return "self_device_ms" if clock == "device" else "self_cpu_ms"


def attributed_ms(labels: dict, clock: str) -> float:
    """Total labelled time for one method, on one clock.

    Only the forward-pass regions are labelled, so this is the part of the run the trace can
    actually attribute. Everything else is the caller's residual to report, not to divide.
    """
    key = clock_key(clock)
    return float(labels[LABEL_TARGET][key]) + float(labels[LABEL_ASSISTANT][key])


def _label_times(profiler) -> dict:
    """Self time per label, in milliseconds, split by clock.

    `self_device_time_total` on a CUDA trace is the kernel time; `self_cpu_time_total` is
    the host time spent inside the region. Both are reported because on a
    bandwidth-bound card they tell different stories, and because a CPU-only trace has
    no device time at all and must not be read as "zero GPU cost".

    A scheduled trace whose window never opened - `--trace-wait` or `--trace-warmup` larger
    than the number of forward passes the run took - never completes a cycle, and
    `key_averages()` on it raises rather than returning an empty list. The generation has
    already happened by then, so that is reported as nothing measured, not as a crash
    halfway through writing an artifact.
    """
    try:
        events = profiler.key_averages()
    except AssertionError:
        events = []

    collected: dict = {}
    for event in events:
        if event.key not in LABELS:
            continue
        collected[event.key] = {
            "count": int(event.count),
            "self_device_ms": float(getattr(event, "self_device_time_total", 0.0)) / 1000.0,
            "self_cpu_ms": float(event.self_cpu_time_total) / 1000.0,
        }
    for label in LABELS:
        collected.setdefault(
            label, {"count": 0, "self_device_ms": 0.0, "self_cpu_ms": 0.0}
        )
    return collected


def profile_case(
    models,
    config,
    device: str,
    case: ProfiledCase,
    methods: tuple[str, ...] = ("baseline", "assisted"),
    warmup_runs: int = 1,
    window: TraceWindow | None = None,
) -> dict:
    """Run one case's methods under the profiler and return the attribution.

    With `window` set the trace is bounded: only the scheduled window is recorded, the
    run otherwise proceeds exactly as before, and the parts of the result that only make
    sense for a whole-generation trace are left unset instead of being derived from the
    part the window covers.
    """
    prompts = load_prompts(config.prompts_file)
    prompt = get_prompt(prompts, case.prompt_id)
    tokenizer_inputs = models.tokenizer(prompt.text, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in tokenizer_inputs.items()}
    prompt_length = int(inputs["input_ids"].shape[1])

    results: dict = {}
    for method in methods:
        kwargs = build_generate_kwargs(method, config, models.assistant, case.max_new_tokens)

        # One unprofiled run first: kernel selection, allocator growth and the first
        # attention-mask build belong to start-up, not to the steady-state loop the
        # trace is supposed to describe.
        with torch.inference_mode():
            for _ in range(max(0, warmup_runs)):
                models.target.generate(**inputs, **kwargs)

        watch = Stopwatch(device)
        profiler_options = {
            "activities": _profiler_activities(device),
            "record_shapes": False,
            "with_stack": False,
        }
        if window is not None:
            profiler_options["schedule"] = window.as_schedule()

        with torch.profiler.profile(**profiler_options) as profiler:
            # The schedule only advances where it is stepped, and it is stepped once per
            # completed target forward pass - the assisted loop's iteration. This wraps the
            # label rather than replacing it, so a call is both labelled and counted.
            stepping = (
                step_profiler_at_forward(models.target, profiler)
                if window is not None
                else nullcontext()
            )
            with label_forward_calls(models.target, LABEL_TARGET):
                with stepping:
                    if method == "assisted" and models.assistant is not None:
                        with label_forward_calls(models.assistant, LABEL_ASSISTANT):
                            with torch.profiler.record_function(LABEL_GENERATE):
                                watch.start()
                                with torch.inference_mode():
                                    output = models.target.generate(**inputs, **kwargs)
                                e2e_ms = watch.stop()
                    else:
                        with torch.profiler.record_function(LABEL_GENERATE):
                            watch.start()
                            with torch.inference_mode():
                                output = models.target.generate(**inputs, **kwargs)
                            e2e_ms = watch.stop()

        sequence = output.sequences[0] if hasattr(output, "sequences") else output[0]
        tokens_generated = len(sequence) - prompt_length
        labels = _label_times(profiler)
        if window is not None:
            # The generate() annotation spans the whole run, so it crosses every step boundary
            # and the schedule ends its cycle out from under it: it comes back with a count of
            # zero on a bounded trace, which would read as "generation never ran". It is
            # unmeasurable under a window, not absent, and the artifact says which.
            labels[LABEL_GENERATE]["count"] = None

        # Only the forward passes are labelled, and every target forward pass is inside
        # one. Averaging the trace's total over the count gives the mean measured forward
        # cost, which is what the calibration predicts - so the two can be put side by side
        # without either one being derived from the other.
        clock = trace_clock(device)
        key = clock_key(clock)
        target = labels[LABEL_TARGET]
        assistant = labels[LABEL_ASSISTANT]
        shortfall = attributed_ms(labels, clock)

        # A window records a slice of the loop, so its labelled time is a slice of the
        # run's. Subtracting it from the whole run's wall clock would invent a residual
        # that is mostly "the rest of the generation", so on a bounded trace the residual
        # is left unset and only the per-forward means - which the window does measure -
        # are reported. The means are what the calibration is compared against anyway.
        whole_generation = window is None
        results[method] = {
            "e2e_ms": e2e_ms,
            "tokens_generated": int(tokens_generated),
            "clock": clock,
            "labels": labels,
            "means_from_trace": {
                "target_forward_ms": (target[key] / target["count"]) if target["count"] else None,
                "assistant_forward_ms": (assistant[key] / assistant["count"])
                if assistant["count"]
                else None,
            },
            "steps_recorded": labels[LABEL_TARGET]["count"],
            "covers_whole_generation": whole_generation,
            # The whole-run region can only be counted when nothing ends the trace early, so
            # the flag that explains a null count sits next to the count itself.
            "whole_run_annotation_measurable": whole_generation,
            "attributed_ms": shortfall if whole_generation else None,
            "attributed_device_ms": shortfall if (whole_generation and clock == "device") else None,
            "attributed_cpu_ms": shortfall if (whole_generation and clock == "cpu") else None,
            "unattributed_ms": max(0.0, e2e_ms - shortfall) if whole_generation else None,
        }

    return {
        "case": case.as_dict(),
        "prompt_tokens": prompt_length,
        "methods": results,
    }


def profile(
    models,
    config,
    device: str,
    dtype_name: str,
    attn_implementation: str,
    cases=DEFAULT_CASES,
    methods: tuple[str, ...] = ("baseline", "assisted"),
    warmup_runs: int = 1,
    window: TraceWindow | None = None,
    smoke: bool = False,
) -> dict:
    """Profile every case and return one artifact describing all of them."""
    if models.assistant is None and "assisted" in methods:
        raise ValueError("profiling the assisted arm needs an assistant model")
    cases = tuple(cases)
    if not cases:
        raise ProfilingError("no cases to profile")

    profiled = [
        profile_case(models, config, device, case, methods, warmup_runs, window) for case in cases
    ]
    return {
        "record_type": "profiling",
        "stage": "stage4",
        "created_utc": _datetime.datetime.now(_datetime.timezone.utc).isoformat(timespec="seconds"),
        "run_id": None,
        "config_fingerprint": None,
        # A profiling artifact from a smoke configuration is a check that the path runs, not
        # a measurement of the drafting loop, and it says so in its own record - the same
        # distinction the run manifest draws for `smoke`.
        "smoke": bool(smoke),
        "hardware": hardware_fingerprint(device),
        "resolved": {
            "device": device,
            "dtype": dtype_name,
            "attn_implementation": attn_implementation,
        },
        "settings": {
            "warmup_runs": int(warmup_runs),
            "methods": list(methods),
            "activities": [str(activity) for activity in _profiler_activities(device)],
            "trace_window": None if window is None else window.as_dict(),
            "covers_whole_generation": window is None,
        },
        "cases": profiled,
    }
