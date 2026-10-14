"""Configuration loading, validation and fingerprinting.

A run is fully described by one YAML file. Loading it into an explicit object (rather
than passing a dictionary around) means three useful things:

  * typos and contradictions fail immediately, before any model is loaded;
  * every recorded trial can carry the exact settings that produced it;
  * `config_fingerprint()` gives those settings a short identity, so two result files
    can be compared without diffing them by eye.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .prompts import PromptError, load_prompts

# This file is <repo_root>/src/tinyturbo/config.py, so two parents up is the repo root.
# Relative paths in a config file are resolved against it, which means the commands in
# the README work regardless of which directory you happen to run them from.
REPO_ROOT = Path(__file__).resolve().parents[2]

KNOWN_METHODS = ("baseline", "assisted")
KNOWN_DTYPES = ("auto", "float32", "float16", "bfloat16")
KNOWN_ATTENTION = ("auto", "eager", "sdpa", "flash_attention_2")


class ConfigError(ValueError):
    """Raised when a configuration is incomplete or internally inconsistent."""


@dataclass(frozen=True)
class ModelSpec:
    """Which checkpoints to load, and how."""

    target: str
    assistant: str | None
    dtype: str
    attn_implementation: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "assistant": self.assistant,
            "dtype": self.dtype,
            "attn_implementation": self.attn_implementation,
        }


@dataclass(frozen=True)
class RunConfig:
    """Everything needed to reproduce one run.

    A run covers a grid of points: every prompt in `prompt_ids` at every length in
    `max_new_tokens_values`. A configuration that names a single prompt and a single
    length is a single point, which is what the vertical-slice configurations are; the
    plural fields are still the whole truth about the run, so the runner reads those and
    never a first element.
    """

    name: str
    models: ModelSpec
    prompt_ids: tuple[str, ...]
    methods: tuple[str, ...]
    max_new_tokens_values: tuple[int, ...]
    num_assistant_tokens: int
    warmup_trials: int
    measured_trials: int
    seed: int
    prompts_file: Path
    raw_dir: Path
    results_csv: Path
    plots_dir: Path
    device: str = "auto"
    smoke: bool = False
    notes: str = ""

    @property
    def prompt_id(self) -> str:
        """The single prompt of a single-point run.

        Exists so that code which only makes sense for one prompt (the figure caption,
        for instance) has something well-defined to read. For a sweep, read `points`.
        """
        return self.prompt_ids[0]

    @property
    def max_new_tokens(self) -> int:
        """The single generation length of a single-point run. See `points` for a sweep."""
        return self.max_new_tokens_values[0]

    @property
    def points(self) -> tuple[tuple[str, int], ...]:
        """Every (prompt_id, max_new_tokens) pair this configuration runs, in a fixed order."""
        return tuple(
            (prompt_id, length)
            for prompt_id in self.prompt_ids
            for length in self.max_new_tokens_values
        )

    @property
    def is_sweep(self) -> bool:
        """Whether this configuration covers more than one point."""
        return len(self.points) > 1

    def as_dict(self) -> dict[str, Any]:
        """A JSON-serialisable view, used both for hashing and for the run manifest.

        Keys are written in a fixed order and paths are stored relative to the
        repository root, so the same configuration always hashes to the same value on
        any machine.

        A single-point configuration serialises to exactly the mapping it always has:
        `prompt_id` as a string and `max_new_tokens` as an integer. That form is frozen,
        because published raw records and the run ids derived from them depend on it. A
        sweep adds the plural keys instead, so its fingerprint covers the whole grid.
        """
        prompt_entry: dict[str, Any] = (
            {"prompt_id": self.prompt_ids[0]}
            if len(self.prompt_ids) == 1
            else {"prompt_ids": list(self.prompt_ids)}
        )
        lengths: Any = (
            self.max_new_tokens_values[0]
            if len(self.max_new_tokens_values) == 1
            else list(self.max_new_tokens_values)
        )
        return {
            "name": self.name,
            "smoke": self.smoke,
            "device": self.device,
            "models": self.models.as_dict(),
            **prompt_entry,
            "methods": list(self.methods),
            "max_new_tokens": lengths,
            "num_assistant_tokens": self.num_assistant_tokens,
            "warmup_trials": self.warmup_trials,
            "measured_trials": self.measured_trials,
            "seed": self.seed,
            "prompts_file": _path_for_recording(self.prompts_file),
            "raw_dir": _path_for_recording(self.raw_dir),
            "results_csv": _path_for_recording(self.results_csv),
            "plots_dir": _path_for_recording(self.plots_dir),
            "notes": self.notes,
        }


def _path_for_recording(path: Path) -> str:
    """Store paths relative to the repo root when possible, so they travel with a clone."""
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def resolve_path(value: str | Path) -> Path:
    """Turn a configured path into an absolute one, resolved against the repo root."""
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path)


def load_config(path: str | Path) -> RunConfig:
    """Read, validate and resolve a YAML configuration file."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        try:
            raw = yaml.safe_load(handle)
        except yaml.YAMLError as exc:
            raise ConfigError(f"{path} is not valid YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a YAML mapping at the top level")

    name = _require_str(raw, "name", path)
    raw_models = raw.get("models")
    if not isinstance(raw_models, dict):
        raise ConfigError(f"{path}: 'models' must be a mapping")

    models = ModelSpec(
        target=_require_str(raw_models, "target", path, where="models"),
        assistant=_optional_str(raw_models, "assistant", path),
        dtype=_require_choice(raw_models, "dtype", KNOWN_DTYPES, path),
        attn_implementation=_require_choice(raw_models, "attn_implementation", KNOWN_ATTENTION, path),
    )

    methods_raw = raw.get("methods", ["baseline", "assisted"])
    if not isinstance(methods_raw, (list, tuple)) or not methods_raw:
        raise ConfigError(f"{path}: 'methods' must be a non-empty list")
    methods = tuple(str(method) for method in methods_raw)
    unknown = [method for method in methods if method not in KNOWN_METHODS]
    if unknown:
        raise ConfigError(
            f"{path}: unknown method(s) {unknown}. Known methods: {list(KNOWN_METHODS)}"
        )
    if len(set(methods)) != len(methods):
        raise ConfigError(f"{path}: 'methods' contains duplicates: {list(methods)}")
    if "assisted" in methods and not models.assistant:
        raise ConfigError(
            f"{path}: method 'assisted' needs models.assistant, but it is not set"
        )

    prompt_ids = _require_prompts(raw, path)
    prompts_file = resolve_path(raw.get("prompts_file", "configs/prompts.jsonl"))
    try:
        known_prompts = load_prompts(prompts_file)
    except PromptError as exc:
        raise ConfigError(str(exc)) from exc
    known_ids = {prompt.prompt_id for prompt in known_prompts}
    for prompt_id in prompt_ids:
        if prompt_id not in known_ids:
            available = ", ".join(prompt.prompt_id for prompt in known_prompts)
            raise ConfigError(f"{path}: unknown prompt_id '{prompt_id}'. Available: {available}")

    return RunConfig(
        name=name,
        models=models,
        prompt_ids=prompt_ids,
        methods=methods,
        max_new_tokens_values=_require_lengths(raw, path),
        num_assistant_tokens=_require_positive_int(raw, "num_assistant_tokens", path),
        warmup_trials=_require_int(raw, "warmup_trials", path, minimum=0, default=1),
        measured_trials=_require_int(raw, "measured_trials", path, minimum=1, default=3),
        seed=_require_int(raw, "seed", path, minimum=0, default=0),
        prompts_file=prompts_file,
        raw_dir=resolve_path(raw.get("raw_dir", "results/raw")),
        results_csv=resolve_path(raw.get("results_csv", "results/benchmark_results.csv")),
        plots_dir=resolve_path(raw.get("plots_dir", "plots")),
        device=str(raw.get("device", "auto")),
        smoke=bool(raw.get("smoke", False)),
        notes=str(raw.get("notes", "")),
    )


def config_fingerprint(config: RunConfig) -> str:
    """A sha256 over the resolved configuration.

    Sorted keys and compact separators make this stable across machines and Python
    versions, so a stored fingerprint is a reliable statement about what ran.
    """
    canonical = json.dumps(config.as_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def run_id(config: RunConfig) -> str:
    """A short human-readable identity for a run: name plus the first 12 hex characters."""
    return f"{config.name}-{config_fingerprint(config)[:12]}"


# ---------------------------------------------------------------------------
# Small validators. Each one names the file, the key and the problem, because a
# configuration error should be fixable without reading this module.
# ---------------------------------------------------------------------------


def _require_str(mapping: dict, key: str, path: Path, where: str = "") -> str:
    value = mapping.get(key)
    prefix = f"{where}." if where else ""
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{path}: '{prefix}{key}' is required and must be a non-empty string")
    return value


def _optional_str(mapping: dict, key: str, path: Path) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{path}: 'models.{key}' must be a non-empty string or null")
    return value


def _require_choice(mapping: dict, key: str, choices: tuple[str, ...], path: Path) -> str:
    value = mapping.get(key)
    if value not in choices:
        raise ConfigError(f"{path}: 'models.{key}' must be one of {list(choices)}, got {value!r}")
    return str(value)


def _require_positive_int(mapping: dict, key: str, path: Path) -> int:
    return _require_int(mapping, key, path, minimum=1)


def _require_prompts(raw: dict, path: Path) -> tuple[str, ...]:
    """`prompt_id: <str>` for one prompt, or a list for a sweep.

    Both spellings are accepted because both read naturally in YAML, and `prompt_ids`
    is how a ten-prompt sweep announces itself.
    """
    if "prompt_id" in raw and "prompt_ids" in raw:
        raise ConfigError(f"{path}: set either 'prompt_id' or 'prompt_ids', not both")
    value = raw.get("prompt_ids", raw.get("prompt_id"))

    if isinstance(value, str):
        items = [value]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        raise ConfigError(
            f"{path}: 'prompt_id' must be a prompt id or a list of prompt ids, got {value!r}"
        )

    if not items:
        raise ConfigError(f"{path}: 'prompt_id' must name at least one prompt")
    for item in items:
        if not isinstance(item, str) or not item.strip():
            raise ConfigError(f"{path}: prompt ids must be non-empty strings, got {item!r}")
    if len(set(items)) != len(items):
        duplicates = sorted({item for item in items if items.count(item) > 1})
        raise ConfigError(f"{path}: duplicate prompt id(s) {duplicates} would run the same point twice")
    return tuple(items)


def _require_lengths(raw: dict, path: Path) -> tuple[int, ...]:
    """`max_new_tokens: <int>` for one length, or a list of lengths for a sweep."""
    value = raw.get("max_new_tokens")
    items = list(value) if isinstance(value, (list, tuple)) else [value]

    if not items:
        raise ConfigError(f"{path}: 'max_new_tokens' must list at least one generation length")
    for item in items:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ConfigError(f"{path}: 'max_new_tokens' must be an integer, got {item!r}")
        if item < 1:
            raise ConfigError(f"{path}: 'max_new_tokens' must be >= 1, got {item}")
    if len(set(items)) != len(items):
        duplicates = sorted({item for item in items if items.count(item) > 1})
        raise ConfigError(
            f"{path}: duplicate generation length(s) {duplicates} would run the same point twice"
        )
    return tuple(items)


def _require_int(
    mapping: dict, key: str, path: Path, minimum: int, default: int | None = None
) -> int:
    value = mapping.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{path}: '{key}' must be an integer, got {value!r}")
    if value < minimum:
        raise ConfigError(f"{path}: '{key}' must be >= {minimum}, got {value}")
    return value
