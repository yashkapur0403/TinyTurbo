"""Tests for Stage 0 (calibration), its scoring, and the figures that depend on it.

Two things are being protected here.

First, the calibration *protocol*: warm-ups discarded, a fresh cache per sample, and the
query width actually requested. A measurement of "a target pass over gamma + 1 tokens"
that quietly measured 1 token instead would misprice the whole cost model, and nothing
downstream would notice.

Second, the *guard*. A calibration taken on the wrong machine or with the wrong lookahead
must never be allowed to score a run. A predicted-vs-observed table built that way looks
like evidence and is not, which is the failure mode this whole project is written to
avoid.

No model is downloaded and no GPU is used: the protocol test drives a stub module, and
the scoring and figure tests use the synthetic records from `test_report.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from conftest import write_config

from test_report import make_trial, write_fixture

from tinyturbo.calibrate import (
    LatencyStats,
    calibration_is_usable,
    measure_cached_step,
)
from tinyturbo.config import config_fingerprint, load_config, run_id
from tinyturbo.report import (
    ReportError,
    build_report,
    build_rows,
    calibration_path_for,
    calibration_score,
    plot_accepted_len_vs_speedup,
    plot_time_attribution,
    profiling_path_for,
    raw_path_for,
    read_run,
    score_rows,
    time_attribution,
    write_json_artifact,
)


# ---------------------------------------------------------------------------
# The measurement protocol
# ---------------------------------------------------------------------------


class _FakeOutput:
    def __init__(self, cache):
        self.past_key_values = cache


class _FakeModel(torch.nn.Module):
    """Stands in for a causal LM: records the shape of every call it receives."""

    def __init__(self):
        super().__init__()
        self.calls: list[tuple[int, object]] = []

    def forward(self, input_ids=None, past_key_values=None, use_cache=False):
        self.calls.append((int(input_ids.shape[1]), past_key_values))
        return _FakeOutput(cache=f"cache-{len(self.calls)}")


def test_measure_cached_step_discards_warmups_and_samples_the_rest():
    model = _FakeModel()
    prompt = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])

    stats = measure_cached_step(model, prompt, 1, "cpu", warmups=2, samples=5, label="tau_t")

    # Each iteration is one untimed prefill plus one timed call, so warm-ups and samples
    # both contribute two calls.
    assert len(model.calls) == 2 * (2 + 5)
    assert stats.warmups == 2
    assert stats.samples == 5
    assert len(stats.values_ms) == 5
    assert all(value > 0 for value in stats.values_ms)


def test_every_timed_call_runs_against_a_populated_cache_of_the_same_length():
    """The point of the protocol: identical shape per sample, cache never empty."""
    model = _FakeModel()
    prompt = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])

    measure_cached_step(model, prompt, 6, "cpu", warmups=1, samples=3, label="tau_t_verify")

    timed = model.calls[1::2]
    assert [width for width, _ in timed] == [6, 6, 6, 6]
    assert all(cache is not None for _, cache in timed), "a timed call without a cache is a prefill, not a decode step"

    # And the cache is rebuilt from scratch before each timed call rather than growing.
    prefills = model.calls[0::2]
    assert all(width == 8 and cache is None for width, cache in prefills)


def test_a_query_longer_than_the_prompt_is_rejected():
    model = _FakeModel()
    prompt = torch.tensor([[1, 2, 3]])

    with pytest.raises(ValueError, match="too few"):
        measure_cached_step(model, prompt, 6, "cpu", warmups=0, samples=1, label="tau_t_verify")


def test_zero_samples_is_rejected_rather_than_reported_as_a_latency():
    model = _FakeModel()
    prompt = torch.tensor([[1, 2, 3]])

    with pytest.raises(ValueError, match="samples must be >= 1"):
        measure_cached_step(model, prompt, 1, "cpu", warmups=0, samples=0, label="tau_t")


def test_latency_stats_carry_their_raw_samples():
    stats = LatencyStats(label="tau_t", query_tokens=1, warmups=2, values_ms=(1.0, 2.0, 6.0))

    payload = stats.as_dict()
    assert payload["samples"] == 3
    assert payload["median_ms"] == 2.0
    assert payload["values_ms"] == [1.0, 2.0, 6.0], "a median with no samples cannot be audited"


# ---------------------------------------------------------------------------
# The guard: a calibration may only score the run it describes
# ---------------------------------------------------------------------------


def _calibration_for(config, **overrides) -> dict:
    calibration = {
        "record_type": "calibration",
        "stage": "stage0",
        "run_id": run_id(config),
        "config_fingerprint": config_fingerprint(config),
        "hardware": {
            "device": "cuda",
            "gpu_name": "Synthetic GPU",
            "is_cuda": True,
            "gpu_driver_version": None,
            "cuda_version": None,
            "torch": None,
            "transformers": None,
        },
        "resolved": {"device": "cuda", "dtype": "float16", "attn_implementation": "sdpa"},
        "settings": {"gamma": 4, "warmups": 5, "samples": 20, "prompt_id": "s1-01-short-open"},
        "primitives": {
            "target_single": {"median_ms": 20.0, "iqr_ms": 0.5, "samples": 20, "values_ms": [20.0]},
            "target_verify": {"median_ms": 24.0, "iqr_ms": 0.5, "samples": 20, "values_ms": [24.0]},
            "assistant_single": {"median_ms": 4.0, "iqr_ms": 0.2, "samples": 20, "values_ms": [4.0]},
        },
        "ratios": {"rho": 1.2, "tau_d_over_tau_t": 0.2, "kappa": 0.8},
    }
    calibration.update(overrides)
    return calibration


def test_a_matching_calibration_is_usable(config_path: Path):
    config = load_config(config_path)
    manifest, _ = read_run(write_fixture(config, [make_trial(config, "assisted", 0, 50.0)]))

    assert calibration_is_usable(_calibration_for(config), manifest) == []


@pytest.mark.parametrize(
    "override, expected",
    [
        ({"config_fingerprint": "a" * 64}, "different configuration"),
        ({"resolved": {"device": "cpu", "dtype": "float16", "attn_implementation": "sdpa"}}, "resolved device"),
        ({"settings": {"gamma": 7}}, "lookahead"),
        ({"hardware": {"device": "cuda", "gpu_name": "Tesla T4", "is_cuda": True}}, "gpu_name"),
    ],
)
def test_a_calibration_from_another_machine_or_lookahead_is_refused(config_path: Path, override, expected):
    config = load_config(config_path)
    manifest, _ = read_run(write_fixture(config, [make_trial(config, "assisted", 0, 50.0)]))

    differences = calibration_is_usable(_calibration_for(config, **override), manifest)
    assert differences, "a mismatch must produce a reason, not silence"
    assert any(expected in difference for difference in differences)


def test_something_that_is_not_a_calibration_is_refused_by_type(config_path: Path):
    config = load_config(config_path)
    manifest, _ = read_run(write_fixture(config, [make_trial(config, "assisted", 0, 50.0)]))

    differences = calibration_is_usable({"record_type": "profiling"}, manifest)
    assert differences and "not a calibration" in differences[0]


# ---------------------------------------------------------------------------
# Scoring: predicted versus observed
# ---------------------------------------------------------------------------


def test_the_prediction_is_the_plan_cost_model_and_the_residual_is_observed_minus_predicted(config_path: Path):
    config = load_config(config_path)
    # accepted length is 6 tokens over 2 target passes = 3.0; speedup is 100/50 = 2.0.
    manifest, trials = read_run(
        write_fixture(
            config,
            [make_trial(config, "baseline", 0, 100.0), make_trial(config, "assisted", 0, 50.0)],
        )
    )
    rows = score_rows(build_rows(manifest, trials))
    calibration = _calibration_for(config)

    scored = calibration_score(rows, calibration, manifest)

    assert scored["usable"] is True
    row = next(entry for entry in scored["rows"] if entry["method"] == "assisted")
    assert row["accepted_len"] == pytest.approx(3.0)
    assert row["predicted"] == pytest.approx(3.0 / (1.2 + 0.8))
    assert row["predicted_ideal_verification"] == pytest.approx(3.0 / (1.0 + 0.8))
    assert row["observed"] == pytest.approx(2.0)
    assert row["residual"] == pytest.approx(2.0 - 3.0 / (1.2 + 0.8))


def test_scoring_is_refused_when_the_calibration_does_not_describe_the_run(config_path: Path):
    config = load_config(config_path)
    manifest, trials = read_run(
        write_fixture(
            config,
            [make_trial(config, "baseline", 0, 100.0), make_trial(config, "assisted", 0, 50.0)],
        )
    )
    rows = score_rows(build_rows(manifest, trials))

    scored = calibration_score(rows, _calibration_for(config, config_fingerprint="b" * 64), manifest)

    assert scored["usable"] is False
    assert scored["rows"] == [], "no table is better than a table from the wrong machine"
    assert any("different configuration" in difference for difference in scored["differences"])


def test_time_attribution_uses_recorded_counts_times_measured_latency(config_path: Path):
    config = load_config(config_path)
    manifest, trials = read_run(
        write_fixture(
            config,
            [make_trial(config, "baseline", 0, 100.0), make_trial(config, "assisted", 0, 50.0)],
        )
    )
    rows = score_rows(build_rows(manifest, trials))

    attribution = time_attribution(rows, _calibration_for(config))
    baseline = next(entry for entry in attribution if entry["method"] == "baseline")
    assisted = next(entry for entry in attribution if entry["method"] == "assisted")

    # Fixture: 6 tokens, 10 ms prefill, baseline 6 target passes, assisted 2 target and
    # 4 assistant passes. tau_t = 20 ms, tau_d = 4 ms.
    assert baseline["components_ms_per_token"]["target"] == pytest.approx(6 * 20.0 / 6)
    assert assisted["components_ms_per_token"]["target"] == pytest.approx(2 * 20.0 / 6)
    assert assisted["components_ms_per_token"]["draft"] == pytest.approx(4 * 4.0 / 6)
    # Baseline is the reference, so its own relative total is exactly 1.0 by construction.
    assert baseline["relative_ms_per_token"] == pytest.approx(1.0)


def test_time_attribution_reports_a_residual_instead_of_forcing_a_match(config_path: Path):
    """The leftover is reported as one number; it is never split up by assumption."""
    config = load_config(config_path)
    manifest, trials = read_run(
        write_fixture(
            config,
            [make_trial(config, "baseline", 0, 100.0), make_trial(config, "assisted", 0, 50.0)],
        )
    )
    rows = score_rows(build_rows(manifest, trials))

    assisted = next(
        entry for entry in time_attribution(rows, _calibration_for(config)) if entry["method"] == "assisted"
    )
    attributed = sum(assisted["relative_components"].values())
    assert assisted["relative_unattributed"] == pytest.approx(
        assisted["relative_ms_per_token"] - attributed
    )


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


def test_writing_an_analysis_artifact_into_raw_records_is_refused(tmp_path: Path):
    with pytest.raises(ReportError, match="raw evidence"):
        write_json_artifact({"a": 1}, tmp_path / "raw" / "artifact.json")


def test_artifact_paths_name_the_run_and_do_not_collide(tmp_path: Path):
    config = load_config(write_config(tmp_path))
    calibration = calibration_path_for(config)
    profiling = profiling_path_for(config)

    assert calibration.name == f"{run_id(config)}-calibration.json"
    assert profiling.name == f"{run_id(config)}-profiling.json"
    assert calibration != profiling
    assert calibration.parent.name == "calibration"
    assert profiling.parent.name == "profiling"


# ---------------------------------------------------------------------------
# The figures that depend on a calibration
# ---------------------------------------------------------------------------


def _assisted_run(config):
    """Two paired trials at the single point this configuration declares.

    Only declared points are used: `validate_run` refuses to report a trial that ran
    somewhere the configuration does not name, and it is right to.
    """
    length = config.max_new_tokens
    prompt_id = config.prompt_id
    return write_fixture(
        config,
        [
            make_trial(config, "baseline", 0, 100.0, max_new_tokens=length, prompt_id=prompt_id),
            make_trial(config, "baseline", 1, 102.0, max_new_tokens=length, prompt_id=prompt_id),
            make_trial(config, "assisted", 0, 50.0, max_new_tokens=length, prompt_id=prompt_id),
            make_trial(config, "assisted", 1, 51.0, max_new_tokens=length, prompt_id=prompt_id),
        ],
    )


def _sweep_config(tmp_path: Path):
    """A two-length sweep, so the figures have more than one point to draw."""
    path = write_config(
        tmp_path,
        prompt_ids=["s1-01-short-open"],
        max_new_tokens=[8, 16],
    )
    return load_config(path)


def _sweep_run(config):
    records = []
    for index, length in enumerate(config.max_new_tokens_values):
        records += [
            make_trial(config, "baseline", 0, 100.0, max_new_tokens=length),
            make_trial(config, "baseline", 1, 102.0, max_new_tokens=length),
            make_trial(config, "assisted", 0, 50.0 + index, max_new_tokens=length),
            make_trial(config, "assisted", 1, 51.0 + index, max_new_tokens=length),
        ]
    return write_fixture(config, records)


def test_figure_two_is_drawn_without_a_calibration_and_says_the_model_is_missing(config_path: Path, tmp_path: Path):
    config = load_config(config_path)
    manifest, trials = read_run(_assisted_run(config))
    rows = score_rows(build_rows(manifest, trials))

    path = plot_accepted_len_vs_speedup(rows, manifest, tmp_path / "fig2.png", None)

    assert path is not None and path.stat().st_size > 0


def test_figure_two_draws_the_model_lines_only_when_a_calibration_is_supplied(config_path: Path, tmp_path: Path):
    config = load_config(config_path)
    manifest, trials = read_run(_assisted_run(config))
    rows = score_rows(build_rows(manifest, trials))

    without = plot_accepted_len_vs_speedup(rows, manifest, tmp_path / "fig2a.png", None)
    with_model = plot_accepted_len_vs_speedup(
        rows, manifest, tmp_path / "fig2b.png", _calibration_for(config)
    )

    assert without is not None and with_model is not None
    # The model overlay is extra ink, so the two renders cannot be the same image.
    assert without.read_bytes() != with_model.read_bytes()


def test_figure_three_needs_a_calibration_and_returns_nothing_without_one(config_path: Path, tmp_path: Path):
    config = load_config(config_path)
    manifest, trials = read_run(_assisted_run(config))
    rows = score_rows(build_rows(manifest, trials))

    assert plot_time_attribution(rows, manifest, tmp_path / "fig3a.png", {}) is None
    drawn = plot_time_attribution(rows, manifest, tmp_path / "fig3b.png", _calibration_for(config))
    assert drawn is not None and drawn.stat().st_size > 0


def test_report_warns_when_there_is_no_calibration_and_still_writes_figure_two(tmp_path: Path):
    config = _sweep_config(tmp_path)
    _sweep_run(config)

    result = build_report(config, verbose=False)

    assert result["calibration"] is None
    assert result["calibration_score"] is None
    assert result["plot_accepted_len_vs_speedup"].is_file()
    assert result["plot_time_attribution"] is None
    assert any("no Stage 0 calibration" in warning for warning in result["warnings"])


def test_report_uses_a_matching_calibration_and_produces_the_comparison(tmp_path: Path):
    config = _sweep_config(tmp_path)
    _sweep_run(config)
    write_json_artifact(_calibration_for(config), calibration_path_for(config))

    result = build_report(config, verbose=False)

    assert result["calibration_score"]["usable"] is True
    assert result["calibration_score"]["rows"], "a usable calibration must produce a scored row"
    assert result["plot_time_attribution"].is_file()
    assert not [w for w in result["warnings"] if "does not describe this run" in w]


def test_report_refuses_a_calibration_from_another_machine(tmp_path: Path):
    config = _sweep_config(tmp_path)
    _sweep_run(config)
    write_json_artifact(
        _calibration_for(config, resolved={"device": "cpu", "dtype": "float32", "attn_implementation": "eager"}),
        calibration_path_for(config),
    )

    result = build_report(config, verbose=False)

    assert result["calibration_score"] is None
    assert result["plot_time_attribution"] is None
    assert any("does not describe this run" in warning for warning in result["warnings"])
    # The point of refusing: no curve, and no table, from numbers about a different machine.
    assert result["plot_accepted_len_vs_speedup"].is_file()
