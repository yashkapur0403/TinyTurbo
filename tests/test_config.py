"""Configuration and prompt-set tests. No GPU, no network, no model loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import REPO_ROOT, write_config

from tinyturbo.config import (
    REPO_ROOT as CONFIG_REPO_ROOT,
    ConfigError,
    config_fingerprint,
    load_config,
    resolve_path,
    run_id,
)
from tinyturbo.prompts import PromptError, get_prompt, load_prompts, prompt_set_hash


def test_loads_the_shipped_cpu_config():
    config = load_config(REPO_ROOT / "configs" / "slice_cpu.yaml")
    assert config.name == "slice-cpu-smoke"
    assert config.smoke is True
    assert config.models.target == "HuggingFaceTB/SmolLM-135M"
    assert config.models.dtype == "float32", "float16 on CPU is rejected elsewhere; keep it out of the shipped config"
    assert config.methods == ("baseline", "assisted")
    assert config.raw_dir == REPO_ROOT / "results" / "raw"


def test_loads_the_shipped_gpu_config():
    config = load_config(REPO_ROOT / "configs" / "slice_gpu.yaml")
    assert config.smoke is False
    assert config.models.target == "HuggingFaceTB/SmolLM-1.7B"
    assert config.models.assistant == "HuggingFaceTB/SmolLM-135M"
    assert config.models.dtype == "float16"
    assert config.models.attn_implementation == "sdpa"


def test_repo_root_points_at_the_project():
    assert CONFIG_REPO_ROOT == REPO_ROOT
    assert (CONFIG_REPO_ROOT / "configs" / "prompts.jsonl").is_file()


def test_valid_config_round_trips(config_path: Path):
    config = load_config(config_path)
    assert config.methods == ("baseline", "assisted")
    assert config.num_assistant_tokens == 4


@pytest.mark.parametrize(
    "override, message",
    [
        ({"methods": ["baseline", "telepathy"]}, "unknown method"),
        ({"max_new_tokens": 0}, "max_new_tokens"),
        ({"max_new_tokens": True}, "max_new_tokens"),
        ({"measured_trials": 0}, "measured_trials"),
        ({"num_assistant_tokens": -3}, "num_assistant_tokens"),
        ({"prompt_id": "does-not-exist"}, "unknown prompt_id"),
    ],
)
def test_invalid_configs_are_rejected(tmp_path: Path, override: dict, message: str):
    path = write_config(tmp_path, **override)
    with pytest.raises(ConfigError) as error:
        load_config(path)
    assert message in str(error.value)


def test_assisted_method_requires_an_assistant(tmp_path: Path):
    path = write_config(
        tmp_path,
        models={
            "target": "org/target",
            "assistant": None,
            "dtype": "float32",
            "attn_implementation": "eager",
        },
    )
    with pytest.raises(ConfigError, match="needs models.assistant"):
        load_config(path)


def test_unknown_dtype_is_rejected(tmp_path: Path):
    path = write_config(
        tmp_path,
        models={
            "target": "org/target",
            "assistant": "org/assistant",
            "dtype": "float8",
            "attn_implementation": "eager",
        },
    )
    with pytest.raises(ConfigError, match="float8"):
        load_config(path)


def test_missing_prompt_file_is_reported(tmp_path: Path):
    path = write_config(tmp_path, prompts_file=str(tmp_path / "absent.jsonl"))
    with pytest.raises(ConfigError, match="prompt file not found"):
        load_config(path)


def test_missing_config_file_is_reported(tmp_path: Path):
    with pytest.raises(ConfigError, match="config file not found"):
        load_config(tmp_path / "nope.yaml")


def test_fingerprint_is_stable_and_sensitive(config_path: Path, tmp_path: Path):
    first = load_config(config_path)
    second = load_config(config_path)
    assert config_fingerprint(first) == config_fingerprint(second)

    changed = load_config(write_config(tmp_path / "other", max_new_tokens=9))
    assert config_fingerprint(changed) != config_fingerprint(first)


def test_run_id_is_readable_and_deterministic(config_path: Path):
    config = load_config(config_path)
    assert run_id(config).startswith("test-run-")
    assert run_id(config) == run_id(load_config(config_path))


def test_paths_are_recorded_relative_to_the_repo_root():
    config = load_config(REPO_ROOT / "configs" / "slice_cpu.yaml")
    assert config.as_dict()["prompts_file"] == "configs/prompts.jsonl"
    assert config.as_dict()["raw_dir"] == "results/raw"


def test_resolve_path_handles_absolute_and_relative(tmp_path: Path):
    assert resolve_path("configs/prompts.jsonl") == REPO_ROOT / "configs" / "prompts.jsonl"
    assert resolve_path(tmp_path / "x.csv") == tmp_path / "x.csv"


# ---------------------------------------------------------------------------
# Prompt set
# ---------------------------------------------------------------------------


def test_prompt_set_shape():
    prompts = load_prompts(REPO_ROOT / "configs" / "prompts.jsonl")
    assert len(prompts) == 10, "the plan fixes the prompt set at ten prompts"

    strata: dict[str, int] = {}
    for prompt in prompts:
        strata[prompt.stratum] = strata.get(prompt.stratum, 0) + 1
    assert set(strata) == {"S1-short-open", "S2-long-open", "S3-grounded", "S4-repetitive"}
    assert len({prompt.prompt_id for prompt in prompts}) == 10

    # The long strata must actually be longer than the short one, otherwise the
    # prefill comparisons they exist for are meaningless.
    shortest = max(len(prompt.text) for prompt in prompts if prompt.stratum == "S1-short-open")
    longest = min(len(prompt.text) for prompt in prompts if prompt.stratum == "S2-long-open")
    assert longest > shortest


def test_get_prompt_rejects_unknown_id():
    prompts = load_prompts(REPO_ROOT / "configs" / "prompts.jsonl")
    found = get_prompt(prompts, "s1-01-short-open")
    assert found.stratum == "S1-short-open"
    with pytest.raises(PromptError, match="unknown prompt_id"):
        get_prompt(prompts, "nope")


def test_prompt_errors_are_specific(tmp_path: Path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"prompt_id": "a", "stratum": "s", "text": "hello"}\n{"prompt_id": "a", "stratum": "s", "text": "again"}\n', encoding="utf-8")
    with pytest.raises(PromptError, match="duplicates prompt_id"):
        load_prompts(bad)

    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text("not json\n", encoding="utf-8")
    with pytest.raises(PromptError, match="not valid JSON"):
        load_prompts(malformed)

    incomplete = tmp_path / "incomplete.jsonl"
    incomplete.write_text('{"prompt_id": "a", "stratum": "s"}\n', encoding="utf-8")
    with pytest.raises(PromptError, match="missing"):
        load_prompts(incomplete)


def test_prompt_set_hash_changes_with_content(tmp_path: Path):
    prompts = load_prompts(REPO_ROOT / "configs" / "prompts.jsonl")
    assert prompt_set_hash(prompts) == prompt_set_hash(prompts)

    edited = tmp_path / "edited.jsonl"
    edited.write_text(
        '{"prompt_id": "s1-01-short-open", "stratum": "S1-short-open", "text": "different text"}\n',
        encoding="utf-8",
    )
    assert prompt_set_hash(load_prompts(edited)) != prompt_set_hash(prompts)
