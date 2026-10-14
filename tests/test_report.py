"""Tests for result validation and reporting.

The raw records used here are synthetic fixtures, built in a temporary directory and
labelled as such in the manifest. Nothing in this file touches results/raw, so these
tests can never be mistaken for, or contaminated by, real measurements.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest

from conftest import write_config, write_jsonl

from tinyturbo.config import config_fingerprint, load_config, run_id
from tinyturbo.metrics import accepted_len_mean, draft_tokens_per_step, tokens_per_second
from tinyturbo.report import (
    CSV_COLUMNS,
    ReportError,
    build_report,
    build_rows,
    raw_path_for,
    read_run,
    rows_to_frame,
    score_rows,
    summarise,
    summary_label,
    validate_run,
)

SYNTHETIC_NOTE = "synthetic fixture written by tests/test_report.py; this is not a measurement"


def make_manifest(config, *, gpu_name="Synthetic GPU", smoke=None) -> dict:
    payload = config.as_dict()
    if smoke is not None:
        payload["smoke"] = smoke
    return {
        "record_type": "run",
        "run_id": run_id(config),
        "config_fingerprint": config_fingerprint(config),
        "created_utc": "2026-01-01T00:00:00+00:00",
        "config": payload,
        "hardware": {"device": "cuda", "gpu_name": gpu_name, "is_cuda": gpu_name is not None},
        "resolved": {"device": "cuda", "dtype": "float16", "attn_implementation": "sdpa"},
        "model_facts": {"target_parameters": 1000, "assistant_parameters": 100, "parameter_ratio": 0.1},
        "prompt_set_hash": "0" * 64,
        "prompt_count": 10,
        "run_scope": {"prompt_id": "s1-01-short-open", "is_slice": True, "methods": ["baseline", "assisted"]},
        "assistant_controls": {"num_assistant_tokens": 4},
        "git": {"git_available": False, "git_commit": None, "git_dirty": None},
        "command": "synthetic fixture",
        "note": SYNTHETIC_NOTE,
    }


def make_trial(
    config,
    method: str,
    trial_index: int,
    e2e_ms: float,
    *,
    prefill_ms: float = 10.0,
    tokens_generated: int = 6,
    generated_ids: tuple[int, ...] = (1, 2, 3, 4, 5, 6),
    is_warmup: bool = False,
    max_new_tokens: int = 8,
    prompt_id: str = "s1-01-short-open",
) -> dict:
    assisted = method == "assisted"
    target_forwards = 2 if assisted else tokens_generated
    assistant_forwards = 4 if assisted else 0
    decode_ms = max(0.0, e2e_ms - prefill_ms)
    decode_ms_per_token = decode_ms / tokens_generated if tokens_generated else None

    return {
        "record_type": "trial",
        "run_id": run_id(config),
        "config_fingerprint": config_fingerprint(config),
        "created_utc": "2026-01-01T00:00:00+00:00",
        "method": method,
        "trial": trial_index,
        "is_warmup": is_warmup,
        "hardware": {"device": "cuda", "gpu_name": "Synthetic GPU", "is_cuda": True},
        "resolved": {"dtype": "float16", "attn_implementation": "sdpa"},
        "assistant_controls": {"num_assistant_tokens": 4},
        "prompt": {"prompt_id": prompt_id, "stratum": "S1-short-open", "prompt_tokens": 9},
        "generation": {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "num_beams": 1,
            "use_cache": True,
        },
        "timing": {
            "prefill_ms": prefill_ms,
            "e2e_ms": e2e_ms,
            "decode_ms": decode_ms,
            "decode_ms_per_token": decode_ms_per_token,
        },
        "counts": {
            "target_forwards": target_forwards,
            "assistant_forwards": assistant_forwards,
            "tokens_generated": tokens_generated,
        },
        "derived": {
            "accepted_len_mean": accepted_len_mean(tokens_generated, target_forwards),
            "draft_tokens_per_step": draft_tokens_per_step(assistant_forwards, target_forwards),
            "tokens_per_second": tokens_per_second(tokens_generated, e2e_ms),
            "decode_tokens_per_second": tokens_per_second(tokens_generated, decode_ms),
        },
        "memory": {"peak_alloc_mb": 512.0, "peak_reserved_mb": 640.0},
        "output": {
            "hit_eos": False,
            "generated_token_ids": list(generated_ids),
            "generated_text_preview": "synthetic",
        },
        "notes": [],
    }


@pytest.fixture
def config(config_path: Path):
    return load_config(config_path)


def write_fixture(config, trials: list[dict], **manifest_kwargs) -> Path:
    return write_jsonl(raw_path_for(config), [make_manifest(config, **manifest_kwargs), *trials])


# ---------------------------------------------------------------------------
# Reading and validation
# ---------------------------------------------------------------------------


def test_one_baseline_and_one_assisted_trial_produce_a_speedup(config):
    write_fixture(
        config,
        [
            make_trial(config, "baseline", 0, 100.0),
            make_trial(config, "assisted", 0, 50.0),
        ],
    )
    manifest, trials = read_run(raw_path_for(config))
    validate_run(manifest, trials, raw_path_for(config))

    rows = score_rows(build_rows(manifest, trials))
    assisted = next(row for row in rows if row["method"] == "assisted")
    assert assisted["speedup_e2e"] == pytest.approx(2.0)
    assert assisted["baseline_e2e_ms"] == 100.0
    assert assisted["accepted_len_mean"] == pytest.approx(3.0)
    assert assisted["exact_match"] is True


def test_missing_raw_records_is_reported_not_guessed(config):
    with pytest.raises(ReportError, match="no raw records"):
        read_run(raw_path_for(config))


def test_a_file_without_a_manifest_is_refused(config):
    path = write_jsonl(raw_path_for(config), [make_trial(config, "baseline", 0, 100.0)])
    with pytest.raises(ReportError, match="no run manifest"):
        read_run(path)


def test_warmups_alone_are_not_a_result(config):
    path = write_jsonl(
        raw_path_for(config),
        [make_manifest(config), make_trial(config, "baseline", -1, 100.0, is_warmup=True)],
    )
    with pytest.raises(ReportError, match="no completed measured trials"):
        read_run(path)


def test_trials_from_a_foreign_configuration_are_refused(config):
    trial = make_trial(config, "assisted", 0, 50.0)
    trial["config_fingerprint"] = "different"
    path = write_jsonl(raw_path_for(config), [make_manifest(config), trial])

    manifest, trials = read_run(path)
    with pytest.raises(ReportError, match="different configuration"):
        validate_run(manifest, trials, path)


def test_incomplete_trial_records_are_refused(config):
    trial = make_trial(config, "baseline", 0, 100.0)
    del trial["timing"]
    path = write_jsonl(raw_path_for(config), [make_manifest(config), trial])

    manifest, trials = read_run(path)
    with pytest.raises(ReportError, match="missing 'timing'"):
        validate_run(manifest, trials, path)


def test_missing_measurement_is_refused(config):
    trial = make_trial(config, "baseline", 0, 100.0)
    trial["timing"]["e2e_ms"] = None
    path = write_jsonl(raw_path_for(config), [make_manifest(config), trial])

    manifest, trials = read_run(path)
    with pytest.raises(ReportError, match="no e2e_ms"):
        validate_run(manifest, trials, path)


def test_smoke_and_cpu_runs_are_flagged_in_warnings(config):
    path = write_fixture(
        config,
        [make_trial(config, "baseline", 0, 100.0), make_trial(config, "assisted", 0, 50.0)],
        gpu_name=None,
        smoke=True,
    )
    manifest, trials = read_run(path)
    warnings = " ".join(validate_run(manifest, trials, path))

    assert "ran on CPU" in warnings
    assert "smoke run" in warnings


def test_gpu_run_without_a_driver_version_is_flagged(config):
    path = write_fixture(config, [make_trial(config, "baseline", 0, 100.0)])
    manifest, trials = read_run(path)
    warnings = " ".join(validate_run(manifest, trials, path))
    assert "no GPU driver version" in warnings


def test_gpu_run_with_a_driver_version_is_not_flagged(config):
    path = write_fixture(config, [make_trial(config, "baseline", 0, 100.0)])
    manifest, trials = read_run(path)
    manifest["hardware"]["gpu_driver_version"] = "580.65.06"
    warnings = " ".join(validate_run(manifest, trials, path))
    assert "no GPU driver version" not in warnings


def test_resumed_run_with_a_different_trial_count_is_flagged(config):
    """`--trials` is not part of the config fingerprint, so a resumed run can exceed it."""
    path = write_fixture(config, [make_trial(config, "baseline", i, 100.0) for i in range(3)])
    manifest, trials = read_run(path)
    manifest["run_scope"]["measured_trials"] = 2

    warnings = " ".join(validate_run(manifest, trials, path))
    assert "declares 2 measured trial(s) but 3 'baseline' trials are present" in warnings


def test_a_run_matching_its_declared_scope_is_not_flagged(config):
    path = write_fixture(config, [make_trial(config, "baseline", i, 100.0) for i in range(3)])
    manifest, trials = read_run(path)
    manifest["run_scope"]["measured_trials"] = 3

    warnings = " ".join(validate_run(manifest, trials, path))
    assert "declares" not in warnings


def test_missing_forward_counts_produce_a_warning(config):
    trial = make_trial(config, "assisted", 0, 50.0)
    trial["counts"]["target_forwards"] = 0
    path = write_jsonl(raw_path_for(config), [make_manifest(config), trial])

    manifest, trials = read_run(path)
    warnings = " ".join(validate_run(manifest, trials, path))
    assert "0 target forward passes" in warnings


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_speedups_are_paired_per_trial_not_pooled(config):
    """Pooled averaging would give 200/75 = 2.67; pairing gives a median of 3.0."""
    write_fixture(
        config,
        [
            make_trial(config, "baseline", 0, 100.0),
            make_trial(config, "baseline", 1, 100.0),
            make_trial(config, "assisted", 0, 50.0),
            make_trial(config, "assisted", 1, 25.0),
        ],
    )
    manifest, trials = read_run(raw_path_for(config))
    rows = score_rows(build_rows(manifest, trials))

    speedups = sorted(row["speedup_e2e"] for row in rows if row["speedup_e2e"] is not None)
    assert speedups == pytest.approx([2.0, 4.0])
    assert summarise(rows)[0]["median_speedup_e2e"] == pytest.approx(3.0)


def test_a_trial_without_its_baseline_gets_no_speedup(config):
    write_fixture(
        config,
        [
            make_trial(config, "baseline", 0, 100.0),
            make_trial(config, "assisted", 0, 50.0),
            make_trial(config, "assisted", 7, 40.0),
        ],
    )
    manifest, trials = read_run(raw_path_for(config))
    rows = score_rows(build_rows(manifest, trials))

    unpaired = next(row for row in rows if row["trial"] == 7)
    assert unpaired["speedup_e2e"] is None, "a ratio with no baseline is not a result"
    assert unpaired["baseline_e2e_ms"] is None
    assert summarise(rows)[0]["trials"] == 1, "unpaired trials are excluded from the summary"


def test_output_disagreement_is_recorded_without_being_hidden(config):
    write_fixture(
        config,
        [
            make_trial(config, "baseline", 0, 100.0, generated_ids=(1, 2, 3, 4, 5, 6)),
            make_trial(config, "assisted", 0, 50.0, generated_ids=(1, 2, 9, 4, 5, 6)),
        ],
    )
    manifest, trials = read_run(raw_path_for(config))
    rows = score_rows(build_rows(manifest, trials))
    assisted = next(row for row in rows if row["method"] == "assisted")

    assert assisted["exact_match"] is False
    assert assisted["first_divergence_index"] == 2
    assert summarise(rows)[0]["exact_match"] is False


def test_summary_reports_spread_across_trials(config):
    write_fixture(
        config,
        [
            make_trial(config, "baseline", 0, 100.0),
            make_trial(config, "baseline", 1, 100.0),
            make_trial(config, "baseline", 2, 100.0),
            make_trial(config, "assisted", 0, 100.0),
            make_trial(config, "assisted", 1, 50.0),
            make_trial(config, "assisted", 2, 200.0),
        ],
    )
    manifest, trials = read_run(raw_path_for(config))
    summary = summarise(score_rows(build_rows(manifest, trials)))[0]

    assert summary["trials"] == 3
    assert summary["median_speedup_e2e"] == pytest.approx(1.0)
    assert summary["iqr_speedup_e2e"] > 0


# ---------------------------------------------------------------------------
# Output artifacts
# ---------------------------------------------------------------------------


def test_csv_has_exactly_the_published_columns(config):
    write_fixture(
        config,
        [make_trial(config, "baseline", 0, 100.0), make_trial(config, "assisted", 0, 50.0)],
    )
    manifest, trials = read_run(raw_path_for(config))
    frame = rows_to_frame(build_rows(manifest, trials))

    assert list(frame.columns) == CSV_COLUMNS
    assert not any(column.startswith("_") for column in frame.columns), "internal fields stay out"
    assert len(frame) == 2


def test_build_report_writes_a_csv_and_a_plot(config):
    write_fixture(
        config,
        [
            make_trial(config, "baseline", 0, 100.0),
            make_trial(config, "baseline", 1, 102.0),
            make_trial(config, "assisted", 0, 50.0),
            make_trial(config, "assisted", 1, 51.0),
        ],
    )
    result = build_report(config, verbose=False)

    assert result["csv"].is_file() and result["csv"].stat().st_size > 0
    assert result["plot"].is_file() and result["plot"].stat().st_size > 0
    assert result["summary"][0]["method"] == "assisted"
    assert result["summary"][0]["median_speedup_e2e"] == pytest.approx(2.0, rel=0.05)

    header = result["csv"].read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",") == CSV_COLUMNS


def test_a_baseline_only_run_produces_a_table_but_no_figure(config):
    """There is nothing to compare, so no chart is drawn that could imply an effect."""
    write_fixture(config, [make_trial(config, "baseline", 0, 100.0)])
    result = build_report(config, verbose=False)

    assert result["summary"] == []
    assert result["plot"] is None
    assert result["csv"].is_file()
    assert any("no figure" in warning for warning in result["warnings"])


# ---------------------------------------------------------------------------
# The generation-length axis
# ---------------------------------------------------------------------------


def test_a_sweep_pairs_and_summarises_per_generation_length(tmp_path: Path):
    """Each length must be scored against its own baseline, never across lengths."""
    config = load_config(write_config(tmp_path, max_new_tokens=[8, 16]))
    trials = [
        trial
        for length in (8, 16)
        for trial in (
            make_trial(config, "baseline", 0, 100.0 + length, max_new_tokens=length),
            make_trial(config, "assisted", 0, 50.0 + length, max_new_tokens=length),
            make_trial(config, "baseline", 1, 100.0 + length, max_new_tokens=length),
            make_trial(config, "assisted", 1, 50.0 + length, max_new_tokens=length),
        )
    ]
    write_fixture(config, trials)

    manifest, read_trials = read_run(raw_path_for(config))
    warnings = " ".join(validate_run(manifest, read_trials, raw_path_for(config)))
    assert "smoke run" in warnings and "declares" not in warnings

    rows = score_rows(build_rows(manifest, read_trials))
    summary = summarise(rows)

    assert [entry["max_new_tokens"] for entry in summary] == [8, 16]
    assert summary[0]["median_speedup_e2e"] == pytest.approx(108.0 / 58.0)
    assert summary[1]["median_speedup_e2e"] == pytest.approx(116.0 / 66.0)
    assert [entry["trials"] for entry in summary] == [2, 2]
    assert [summary_label(entry) for entry in summary] == ["assisted @ 8", "assisted @ 16"]
    assert all(entry["exact_match"] is True for entry in summary)


def test_arms_that_stop_at_different_lengths_are_flagged_separately(tmp_path: Path):
    """A truncated pair must not read as a losslessness failure."""
    config = load_config(write_config(tmp_path))
    path = write_fixture(
        config,
        [
            make_trial(config, "baseline", 0, 100.0, tokens_generated=6),
            make_trial(
                config,
                "assisted",
                0,
                50.0,
                tokens_generated=4,
                generated_ids=(1, 2, 3, 4),
            ),
        ],
    )
    manifest, trials = read_run(path)
    warnings = " ".join(validate_run(manifest, trials, path))

    assert "generated different lengths" in warnings
    assert "assisted 4, baseline 6" in warnings
    assert "hit_eos" in warnings


def test_arms_that_stop_together_are_not_flagged(tmp_path: Path):
    config = load_config(write_config(tmp_path))
    path = write_fixture(
        config,
        [make_trial(config, "baseline", 0, 100.0), make_trial(config, "assisted", 0, 50.0)],
    )
    manifest, trials = read_run(path)
    assert "generated different lengths" not in " ".join(validate_run(manifest, trials, path))


def test_the_declared_trial_count_is_compared_per_point_not_per_file(tmp_path: Path):
    """`--trials 1` declares one trial per point, however many points the grid has."""
    config = load_config(write_config(tmp_path, max_new_tokens=[8, 16]))
    trials = [
        make_trial(config, "baseline", 0, 100.0, max_new_tokens=length) for length in (8, 16)
    ]
    path = write_fixture(config, trials)
    manifest, read_trials = read_run(path)

    manifest["run_scope"]["measured_trials"] = 1
    assert [w for w in validate_run(manifest, read_trials, path) if "declares" in w] == []

    manifest["run_scope"]["measured_trials"] = 2
    declared = [w for w in validate_run(manifest, read_trials, path) if "declares" in w]
    assert declared and "across 2 points" in declared[0]


def test_a_trial_at_an_undeclared_generation_length_is_refused(tmp_path: Path):
    config = load_config(write_config(tmp_path, max_new_tokens=[8, 16]))
    path = write_fixture(
        config,
        [
            make_trial(config, "baseline", 0, 100.0, max_new_tokens=8),
            make_trial(config, "assisted", 0, 50.0, max_new_tokens=64),
        ],
    )
    manifest, trials = read_run(path)
    with pytest.raises(ReportError, match="not a point of this run's configuration"):
        validate_run(manifest, trials, path)


def test_a_trial_for_an_undeclared_prompt_is_refused(tmp_path: Path):
    config = load_config(write_config(tmp_path, max_new_tokens=[8]))
    path = write_fixture(
        config,
        [make_trial(config, "baseline", 0, 100.0, prompt_id="s4-01-repetitive")],
    )
    manifest, trials = read_run(path)
    with pytest.raises(ReportError, match="not a point of this run's configuration"):
        validate_run(manifest, trials, path)


def test_a_trial_without_a_recorded_length_is_refused(tmp_path: Path):
    """Tracing a trial needs the length, so its absence is an error and not a guess."""
    config = load_config(write_config(tmp_path, max_new_tokens=[8]))
    trial = make_trial(config, "baseline", 0, 100.0)
    del trial["generation"]["max_new_tokens"]
    path = write_jsonl(raw_path_for(config), [make_manifest(config), trial])

    manifest, trials = read_run(path)
    with pytest.raises(ReportError, match="does not record the generation length"):
        validate_run(manifest, trials, path)


def test_the_csv_carries_the_generation_length_of_every_row(tmp_path: Path):
    config = load_config(write_config(tmp_path, max_new_tokens=[8, 16]))
    write_fixture(
        config,
        [
            make_trial(config, "baseline", 0, 108.0, max_new_tokens=8),
            make_trial(config, "assisted", 0, 58.0, max_new_tokens=8),
            make_trial(config, "baseline", 0, 116.0, max_new_tokens=16),
            make_trial(config, "assisted", 0, 66.0, max_new_tokens=16),
        ],
    )
    result = build_report(config, verbose=False)
    text = result["csv"].read_text(encoding="utf-8")
    rows = list(csv.DictReader(io.StringIO(text)))

    assert text.splitlines()[0].split(",") == CSV_COLUMNS
    assert [row["max_new_tokens"] for row in rows] == ["8", "8", "16", "16"]
    assert len(result["summary"]) == 2


def test_plotting_nothing_is_an_error_even_if_called_directly(config):
    write_fixture(config, [make_trial(config, "baseline", 0, 100.0)])
    manifest, trials = read_run(raw_path_for(config))
    rows = score_rows(build_rows(manifest, trials))
    with pytest.raises(ReportError, match="no speedup could be computed"):
        from tinyturbo.report import plot_speedup

        plot_speedup(rows, manifest, config.plots_dir / "nope.png")
