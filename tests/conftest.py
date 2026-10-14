"""Shared fixtures for the test suite.

Everything here runs without a GPU and without network access. Configurations point
their output paths at a temporary directory, and the reporting tests build their own
clearly-labelled synthetic records in that same temporary directory. No test reads or
writes results/raw, so a test can never be confused with a measurement.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIPPED_PROMPTS = REPO_ROOT / "configs" / "prompts.jsonl"


def write_config(tmp_path: Path, **overrides) -> Path:
    """Write a small valid configuration into tmp_path and return its path."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": "test-run",
        "smoke": True,
        "device": "cpu",
        "models": {
            "target": "org/target-model",
            "assistant": "org/assistant-model",
            "dtype": "float32",
            "attn_implementation": "eager",
        },
        "prompt_id": "s1-01-short-open",
        "methods": ["baseline", "assisted"],
        "max_new_tokens": 8,
        "num_assistant_tokens": 4,
        "warmup_trials": 1,
        "measured_trials": 2,
        "seed": 0,
        "prompts_file": str(SHIPPED_PROMPTS),
        "raw_dir": str(tmp_path / "raw"),
        "results_csv": str(tmp_path / "benchmark_results.csv"),
        "plots_dir": str(tmp_path / "plots"),
    }
    payload.update(overrides)

    # A sweep names its prompts with the plural key; the singular one would be rejected as
    # a contradiction rather than quietly ignored.
    if "prompt_ids" in overrides:
        payload.pop("prompt_id", None)

    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    return write_config(tmp_path)


def write_jsonl(path: Path, records: list[dict]) -> Path:
    """Write records as JSON Lines, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return path
