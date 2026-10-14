"""Stage 0: the latency primitives of the cost model, measured rather than inferred.

`docs/PLAN.md` section 5 prices assisted decoding as

    speedup_decode ~= b / (rho + kappa),  where

        rho   = tau_t^verify / tau_t   (how much the verification pass costs relative
                                        to a plain single-token target step)
        kappa = gamma * tau_d / tau_t  (the total drafting cost in one iteration,
                                        relative to the same single-token step)

and `b` is the accepted length the main sweep already measures. What the sweep did *not*
measure is the three tau values, so the model could only be checked by fitting a cost
ratio out of the very residual it was meant to explain. That is a consistency check, and
it is labelled as one everywhere it appears.

This module measures the primitives directly, with the same models, dtype, device and
attention implementation as the sweep:

    tau_t          one target forward pass over a single query token, cache populated
    tau_t^verify   one target forward pass over `gamma` drafted tokens plus the
                   correction position, cache populated
    tau_d          one assistant forward pass over a single query token, cache populated

Each is a median over repeated samples, with warm-up passes discarded first, and the
individual samples are kept in the artifact so the spread is auditable rather than
asserted.

What this module cannot see is the Python-level bookkeeping *between* forward passes.
A gap between a prediction and the observed sweep therefore has two candidate causes -
wrong rho or kappa, or unmodelled per-step overhead - and only a Stage 4 trace
(`profiling.py`) can separate them. This module does not claim to.
"""

from __future__ import annotations

import datetime as _datetime
from dataclasses import dataclass

import torch

from .env import hardware_fingerprint
from .metrics import interquartile_range, median


@dataclass(frozen=True)
class LatencyStats:
    """Repeated timings of one cached forward pass, reduced to robust statistics."""

    label: str
    query_tokens: int
    warmups: int
    values_ms: tuple[float, ...]

    @property
    def samples(self) -> int:
        return len(self.values_ms)

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "query_tokens": self.query_tokens,
            "warmups": self.warmups,
            "samples": self.samples,
            "median_ms": median(list(self.values_ms)),
            "iqr_ms": interquartile_range(list(self.values_ms)),
            "min_ms": min(self.values_ms),
            "max_ms": max(self.values_ms),
            # The raw samples travel with the summary: a median with no spread cannot be
            # audited, and a suspicious one cannot be re-examined.
            "values_ms": list(self.values_ms),
        }


def measure_cached_step(
    model,
    input_ids,
    query_tokens: int,
    device: str,
    warmups: int,
    samples: int,
    label: str,
) -> LatencyStats:
    """Time one cached forward pass over `query_tokens` positions, `samples` times.

    The cache is rebuilt before every timed call and the rebuild is *not* timed, so every
    sample measures exactly the same shape of work: a forward pass whose attention spans
    the same number of cached keys. Letting the cache grow across samples instead would
    make later samples slower for a reason that has nothing to do with the primitive.

    `query_tokens` is 1 for a plain decode step and `gamma + 1` for a verification pass
    over gamma drafted tokens plus the correction position.
    """
    if query_tokens < 1:
        raise ValueError(f"query_tokens must be >= 1, got {query_tokens}")
    if samples < 1:
        raise ValueError(f"samples must be >= 1, got {samples}")

    from .instrument import Stopwatch

    watch = Stopwatch(device)
    prompt_ids = input_ids.to(device)
    query_ids = prompt_ids[:, :query_tokens]
    if query_ids.shape[1] != query_tokens:
        raise ValueError(
            f"the prompt has {prompt_ids.shape[1]} tokens, too few for a {query_tokens}-token query"
        )

    values: list[float] = []
    with torch.inference_mode():
        for index in range(warmups + samples):
            # Untimed: build a populated cache for this iteration.
            prefill = model(input_ids=prompt_ids, use_cache=True)
            cache = prefill.past_key_values
            del prefill

            watch.start()
            model(input_ids=query_ids, past_key_values=cache, use_cache=True)
            elapsed = watch.stop()

            if index >= warmups:
                values.append(elapsed)

    return LatencyStats(
        label=label,
        query_tokens=query_tokens,
        warmups=warmups,
        values_ms=tuple(values),
    )


def calibrate(
    models,
    device: str,
    dtype_name: str,
    attn_implementation: str,
    gamma: int,
    prompt_ids,
    prompt_id: str,
    warmups: int = 5,
    samples: int = 20,
    config_fingerprint: str | None = None,
    run_id: str | None = None,
) -> dict:
    """Measure the three primitives and derive rho and kappa.

    `prompt_ids` is a real tokenised prompt from the shipped prompt set, so the cache
    length and the position indices match what the sweep sees. The assistant pass is
    measured over the same prompt, because its cache length is what makes a draft step
    cheap or expensive.
    """
    if models.assistant is None:
        raise ValueError("calibration needs an assistant model: tau_d must be measured, not assumed")

    target_single = measure_cached_step(
        models.target, prompt_ids, 1, device, warmups, samples, "tau_t (target, 1 query token)"
    )
    verify_tokens = int(gamma) + 1
    target_verify = measure_cached_step(
        models.target,
        prompt_ids,
        verify_tokens,
        device,
        warmups,
        samples,
        f"tau_t_verify (target, gamma+1 = {verify_tokens} query tokens)",
    )
    assistant_single = measure_cached_step(
        models.assistant, prompt_ids, 1, device, warmups, samples, "tau_d (assistant, 1 query token)"
    )

    tau_t = target_single.as_dict()["median_ms"]
    tau_verify = target_verify.as_dict()["median_ms"]
    tau_d = assistant_single.as_dict()["median_ms"]
    if tau_t <= 0:
        raise ValueError("tau_t measured as zero, which cannot be divided by")

    return {
        "record_type": "calibration",
        "stage": "stage0",
        "created_utc": _datetime.datetime.now(_datetime.timezone.utc).isoformat(timespec="seconds"),
        "run_id": run_id,
        "config_fingerprint": config_fingerprint,
        "hardware": hardware_fingerprint(device),
        "resolved": {
            "device": device,
            "dtype": dtype_name,
            "attn_implementation": attn_implementation,
        },
        "settings": {
            "gamma": int(gamma),
            "warmups": int(warmups),
            "samples": int(samples),
            "prompt_id": prompt_id,
            "prompt_tokens": int(prompt_ids.shape[1]),
        },
        "model_facts": models.as_dict(),
        "primitives": {
            "target_single": target_single.as_dict(),
            "target_verify": target_verify.as_dict(),
            "assistant_single": assistant_single.as_dict(),
        },
        "ratios": {
            "rho": tau_verify / tau_t,
            "tau_d_over_tau_t": tau_d / tau_t,
            "kappa": int(gamma) * tau_d / tau_t,
        },
    }


def calibration_is_usable(calibration: dict, manifest: dict) -> list[str]:
    """Reasons this calibration cannot be applied to a run. Empty means usable.

    Applying a CPU calibration, or one taken on a different card, driver or library
    version, to a GPU sweep would produce a predicted-vs-observed table that looks like
    evidence and is not. Every field that would have to match for the comparison to mean
    anything is checked here, and the reasons are returned rather than raised so the
    caller can report them instead of quietly proceeding.
    """
    differences: list[str] = []

    if calibration.get("record_type") != "calibration":
        return [f"not a calibration artifact (record_type={calibration.get('record_type')!r})"]

    if manifest.get("config_fingerprint") != calibration.get("config_fingerprint"):
        differences.append(
            f"different configuration ({calibration.get('config_fingerprint')} "
            f"vs {manifest.get('config_fingerprint')})"
        )

    for field in ("device", "dtype", "attn_implementation"):
        expected = (calibration.get("resolved") or {}).get(field)
        actual = (manifest.get("resolved") or {}).get(field)
        if expected != actual:
            differences.append(f"resolved {field}: calibration {expected!r}, run {actual!r}")

    calibration_hardware = calibration.get("hardware") or {}
    run_hardware = manifest.get("hardware") or {}
    for field in ("gpu_name", "gpu_driver_version", "cuda_version", "torch", "transformers"):
        expected = calibration_hardware.get(field)
        actual = run_hardware.get(field)
        if expected != actual:
            differences.append(f"{field}: calibration {expected!r}, run {actual!r}")

    gamma_run = (manifest.get("assistant_controls") or {}).get("num_assistant_tokens")
    gamma_calibration = (calibration.get("settings") or {}).get("gamma")
    if gamma_run is not None and gamma_calibration is not None and gamma_run != gamma_calibration:
        differences.append(
            f"lookahead: calibration gamma={gamma_calibration}, run gamma={gamma_run}"
        )

    return differences
