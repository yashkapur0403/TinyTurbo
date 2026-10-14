"""Tests for the two methods and the assistant knobs they depend on.

These tests exist because the behaviour they pin down is surprising. In the installed
transformers, `num_assistant_tokens` passed to `target.generate(...)` never reaches the
assistant model, and the draft length silently stays at the library default. If a
future version fixes that, these tests fail and the comment in methods.py gets
rewritten deliberately rather than the experiment quietly changing meaning.
"""

from __future__ import annotations

import pytest
from transformers import GenerationConfig

from tinyturbo.config import ModelSpec
from tinyturbo.methods import (
    ASSISTED,
    BASELINE,
    apply_assistant_controls,
    build_generate_kwargs,
    generation_config_supports,
    json_safe_settings,
)


class FakeModel:
    """Stands in for a loaded assistant model: only generation_config is used."""

    def __init__(self) -> None:
        self.generation_config = GenerationConfig()


class FakeRunConfig:
    max_new_tokens = 16
    num_assistant_tokens = 5
    models = ModelSpec(
        target="org/target", assistant="org/assistant", dtype="float32", attn_implementation="eager"
    )


def test_library_default_lookahead_is_twenty():
    """Locks the default we found by reading the source, so a change cannot pass silently."""
    defaults = GenerationConfig()._get_default_generation_params()
    assert defaults["num_assistant_tokens"] == 20
    assert defaults["num_assistant_tokens_schedule"] == "constant"
    assert defaults["assistant_confidence_threshold"] == 0.4


def test_controls_are_applied_to_the_assistant_not_the_target():
    assistant = FakeModel()
    applied = apply_assistant_controls(assistant, 6)

    assert assistant.generation_config.num_assistant_tokens == 6
    assert assistant.generation_config.num_assistant_tokens_schedule == "constant"
    assert assistant.generation_config.assistant_confidence_threshold == 0.0
    assert applied["applied_to"] == "assistant.generation_config"
    assert applied["num_assistant_tokens"] == 6


def test_confidence_threshold_is_disabled_with_a_falsy_value_not_none():
    """None is overwritten by the library default (defaults_only=True), 0.0 is not."""
    assistant = FakeModel()
    apply_assistant_controls(assistant, 5)
    assert assistant.generation_config.assistant_confidence_threshold == 0.0
    assert not assistant.generation_config.assistant_confidence_threshold


def test_baseline_and_assisted_differ_only_by_the_assistant():
    config = FakeRunConfig()
    assistant = FakeModel()

    baseline = build_generate_kwargs(BASELINE, config)
    assisted = build_generate_kwargs(ASSISTED, config, assistant)

    assert "assistant_model" not in baseline
    assert assisted["assistant_model"] is assistant

    # Every other decoding setting must be identical, or the comparison is confounded.
    baseline.pop("assistant_model", None)
    assisted.pop("assistant_model")
    assert baseline == assisted


def test_decoding_settings_are_pinned_for_both_arms():
    config = FakeRunConfig()
    kwargs = build_generate_kwargs(BASELINE, config)
    assert kwargs["do_sample"] is False, "greedy, so the lossless output check is meaningful"
    assert kwargs["num_beams"] == 1
    assert kwargs["use_cache"] is True
    assert kwargs["max_new_tokens"] == config.max_new_tokens


def test_assisted_requires_an_assistant_model():
    with pytest.raises(ValueError, match="needs an assistant model"):
        build_generate_kwargs(ASSISTED, FakeRunConfig(), None)


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="unknown method"):
        build_generate_kwargs("telepathy", FakeRunConfig())


def test_compile_setting_is_guarded_and_recorded():
    """The guard must agree with the installed version rather than assume it."""
    config = FakeRunConfig()
    kwargs = build_generate_kwargs(BASELINE, config)
    assert generation_config_supports("disable_compile") == ("disable_compile" in kwargs)
    assert generation_config_supports("definitely_not_a_real_field") is False


def test_settings_are_recorded_without_model_objects():
    assistant = FakeModel()
    kwargs = build_generate_kwargs(ASSISTED, FakeRunConfig(), assistant)
    settings = json_safe_settings(kwargs)

    assert settings["do_sample"] is False
    assert settings["assistant_model"] == "<FakeModel>", "record that it was used, not the module"
    assert settings["max_new_tokens"] == 16
