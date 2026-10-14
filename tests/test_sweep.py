"""Tests for the sweep: the point grid, the run identity, and the resume key.

The properties here are the ones that decide whether a long remote run is trustworthy.
Changing the serialised form of a configuration would orphan published evidence, narrowing
a run with a command line flag must not let two different experiments share one file, and
a resumed run must neither duplicate a trial nor pool two machines into one average. None
of them needs a GPU or a model to check.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from conftest import REPO_ROOT, write_config

from tinyturbo.cli import build_parser, main
from tinyturbo.config import ConfigError, config_fingerprint, load_config, run_id
from tinyturbo.runner import (
    HARDWARE_IDENTITY_FIELDS,
    RunError,
    completed_trial_keys,
    hardware_differences,
    narrow,
    scope_for,
)

# The identity of the published T4 run. If a change to RunConfig breaks this, it has also
# broken the link between results/raw/slice-gpu-902bc285880d.jsonl and the configuration
# that a reader can re-run today.
PUBLISHED_GPU_FINGERPRINT = (
    "902bc285880d629929108308e6a4cf751f5cd42b6fa565df3f561973c3da8dd2"
)


# ---------------------------------------------------------------------------
# The grid, and the frozen form of a single-point configuration
# ---------------------------------------------------------------------------


def test_a_single_point_config_serialises_exactly_as_it_always_has():
    config = load_config(REPO_ROOT / "configs" / "slice_gpu.yaml")
    payload = config.as_dict()

    assert payload["prompt_id"] == "s2-01-long-open"
    assert payload["max_new_tokens"] == 64
    assert "prompt_ids" not in payload
    assert config.points == (("s2-01-long-open", 64),)
    assert config.is_sweep is False
    assert config_fingerprint(config) == PUBLISHED_GPU_FINGERPRINT


def test_the_shipped_sweep_is_the_ten_prompt_by_three_length_grid():
    config = load_config(REPO_ROOT / "configs" / "gpu_main.yaml")

    assert config.is_sweep is True
    assert len(config.prompt_ids) == 10
    assert config.max_new_tokens_values == (32, 128, 256)
    assert len(config.points) == 30
    assert config.methods == ("baseline", "assisted")
    assert config.models.target == "HuggingFaceTB/SmolLM-1.7B"
    assert config.models.assistant == "HuggingFaceTB/SmolLM-135M"
    assert config.num_assistant_tokens == 5


def test_a_sweep_announces_its_prompts_with_the_plural_key():
    payload = load_config(REPO_ROOT / "configs" / "gpu_main.yaml").as_dict()
    assert isinstance(payload["prompt_ids"], list)
    assert len(payload["prompt_ids"]) == 10
    assert "prompt_id" not in payload
    assert payload["max_new_tokens"] == [32, 128, 256]


def test_the_grid_is_the_product_of_the_two_axes():
    config = load_config(REPO_ROOT / "configs" / "gpu_main.yaml")
    assert set(config.points) == {
        (prompt_id, length) for prompt_id in config.prompt_ids for length in (32, 128, 256)
    }
    assert len(config.points) == 30


# ---------------------------------------------------------------------------
# Narrowing a sweep with a command line flag
# ---------------------------------------------------------------------------


def test_narrowing_a_sweep_changes_its_identity(tmp_path: Path):
    config = load_config(write_config(tmp_path, max_new_tokens=[8, 16]))

    untouched = narrow(config, None, None)
    assert config_fingerprint(untouched) == config_fingerprint(config)

    narrowed = narrow(config, prompt_id=None, max_new_tokens=8)
    assert narrowed.is_sweep is False
    assert config_fingerprint(narrowed) != config_fingerprint(config)
    assert run_id(narrowed) != run_id(config), "a narrowed run must not share a raw file"


def test_narrowing_to_something_the_configuration_does_not_contain_is_refused(tmp_path: Path):
    config = load_config(write_config(tmp_path, max_new_tokens=[8, 16]))

    with pytest.raises(RunError, match="is not in this configuration"):
        narrow(config, prompt_id=None, max_new_tokens=64)
    with pytest.raises(RunError, match="is not in this configuration"):
        narrow(config, prompt_id="s2-01-long-open", max_new_tokens=None)


def test_the_cli_exposes_the_narrowing_flags_and_refuses_unconfigured_values(tmp_path: Path, capsys):
    args = build_parser().parse_args(["run", "--config", "c.yaml", "--max-new-tokens", "8"])
    assert args.max_new_tokens == 8

    path = write_config(tmp_path, max_new_tokens=[8, 16])
    assert main(["run", "--config", str(path), "--max-new-tokens", "64"]) == 2
    assert "is not in this configuration" in capsys.readouterr().err
    assert main(["run", "--config", str(path), "--prompt-id", "s9-99-nope"]) == 2


# ---------------------------------------------------------------------------
# Resuming
# ---------------------------------------------------------------------------


def _trial(prompt_id: str, length: int | None, method: str, index: int) -> dict:
    generation = {} if length is None else {"max_new_tokens": length}
    return {
        "record_type": "trial",
        "prompt": {"prompt_id": prompt_id},
        "generation": generation,
        "method": method,
        "trial": index,
    }


def test_the_resume_key_carries_the_generation_length():
    """Without the length, trial 0 at 32 tokens would mark trial 0 at 256 tokens as done."""
    records = [
        _trial("p", 32, "baseline", 0),
        _trial("p", 256, "baseline", 0),
        _trial("p", 256, "assisted", 0),
    ]
    keys = completed_trial_keys(records)

    assert keys == {("p", 32, "baseline", 0), ("p", 256, "baseline", 0), ("p", 256, "assisted", 0)}


def test_the_resume_key_carries_the_prompt_and_the_method():
    records = [_trial("p1", 32, "baseline", 0), _trial("p2", 32, "baseline", 0)]
    assert len(completed_trial_keys(records)) == 2
    assert completed_trial_keys([{"record_type": "run"}]) == set()


def test_a_resumed_run_refuses_hardware_it_was_not_started_on():
    started = dict.fromkeys(HARDWARE_IDENTITY_FIELDS, "Tesla T4")
    same_machine = dict(started)

    assert hardware_differences(started, same_machine) == []

    different = hardware_differences(started, {**started, "gpu_driver_version": "535.104.05"})
    assert len(different) == 1 and different[0].startswith("gpu_driver_version")

    assert hardware_differences({**started, "device": "cuda"}, {**started, "device": "cpu"})
    assert hardware_differences(started, {}), "a machine that reports nothing is not the same one"


def test_session_noise_does_not_block_a_legitimate_resume():
    """Two sessions on one card differ in ways that say nothing about the measurements."""
    started = dict.fromkeys(HARDWARE_IDENTITY_FIELDS, "Tesla T4")
    noisier_session = {
        **started,
        "cpu_count": 2,
        "torch_threads": 1,
        "gpu_memory_total_mb": 14912.6875,
        "nvidia_smi_report": "580.82.07, Tesla T4, 15360",
        "python": "3.13.15",
    }
    assert hardware_differences(started, noisier_session) == []


# ---------------------------------------------------------------------------
# What the manifest declares
# ---------------------------------------------------------------------------


def test_scope_stays_flat_for_one_point_and_names_the_grid_for_a_sweep(tmp_path: Path):
    single = load_config(write_config(tmp_path / "single"))
    flat = scope_for(single, measured_trials=2)
    assert flat["is_slice"] is True
    assert flat["prompt_id"] == "s1-01-short-open"
    assert "prompt_ids" not in flat
    assert flat["measured_trials"] == 2

    sweep = load_config(
        write_config(
            tmp_path / "sweep",
            prompt_ids=["s1-01-short-open", "s4-01-repetitive"],
            max_new_tokens=[8, 16],
        )
    )
    grid = scope_for(sweep, measured_trials=1)
    assert grid["is_slice"] is False
    assert grid["prompt_ids"] == ["s1-01-short-open", "s4-01-repetitive"]
    assert grid["max_new_tokens"] == [8, 16]
    assert "4 points" in grid["note"]


# ---------------------------------------------------------------------------
# Configuration validation of the two axes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "override, message",
    [
        ({"prompt_ids": []}, "at least one prompt"),
        ({"prompt_ids": ["s1-01-short-open", "s1-01-short-open"]}, "duplicate prompt id"),
        ({"prompt_ids": ["s1-01-short-open", "nope"]}, "unknown prompt_id"),
        ({"prompt_ids": "s1-01-short-open"}, None),
        ({"max_new_tokens": []}, "at least one generation length"),
        ({"max_new_tokens": [8, 8]}, "duplicate generation length"),
        ({"max_new_tokens": [8, 0]}, "must be >= 1"),
        ({"max_new_tokens": [8, "sixteen"]}, "must be an integer"),
    ],
)
def test_invalid_sweep_axes_are_rejected(tmp_path: Path, override: dict, message: str | None):
    path = write_config(tmp_path, **override)
    if message is None:
        # A bare string is a one-prompt configuration, not a sweep, and must still load.
        assert load_config(path).points == (("s1-01-short-open", 8),)
        return
    with pytest.raises(ConfigError) as error:
        load_config(path)
    assert message in str(error.value)


def test_declaring_the_prompts_twice_is_a_contradiction(tmp_path: Path):
    path = write_config(tmp_path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["prompt_ids"] = ["s4-01-repetitive"]
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ConfigError, match="not both"):
        load_config(path)


def test_a_different_grid_is_a_different_run(tmp_path: Path):
    one = load_config(write_config(tmp_path / "one", max_new_tokens=[8]))
    two = load_config(write_config(tmp_path / "two", max_new_tokens=[8, 16]))
    assert config_fingerprint(one) != config_fingerprint(two)
