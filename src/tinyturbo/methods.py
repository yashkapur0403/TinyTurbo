"""The methods under test, and the exact generate() arguments each one uses.

This module is the single source of truth for what "baseline" and "assisted" mean, so
the two arms cannot quietly drift apart as the project grows.
"""

from __future__ import annotations

from functools import lru_cache

from transformers import GenerationConfig

BASELINE = "baseline"
ASSISTED = "assisted"


@lru_cache(maxsize=None)
def generation_config_supports(field_name: str) -> bool:
    """Whether this installed transformers version knows a given GenerationConfig field.

    transformers ships breaking changes frequently, and a benchmark must not change
    meaning because a dependency moved. Guarding optional keyword arguments this way
    keeps the code working across minor versions, and whatever was actually applied
    gets recorded in the run record.
    """
    try:
        return hasattr(GenerationConfig(), field_name)
    except Exception:  # pragma: no cover - defensive
        return False


def apply_assistant_controls(assistant, num_assistant_tokens: int) -> dict:
    """Pin the knobs that control how the assistant drafts, and report what took effect.

    Everything here was read out of the installed transformers 5.3.0 source, because
    the library's real behaviour is not what the documentation implies:

    * The draft length is read from the ASSISTANT's own generation_config, not from
      the arguments passed to target.generate(). In generation/candidate_generator.py:

          self.num_assistant_tokens = self.assistant_generation_config.num_assistant_tokens

      and generation/utils.py copies the target's value across only for the
      "heuristic" schedule. So passing num_assistant_tokens=5 to generate() is a
      silent no-op: the draft length stays at the library default of 20.

    * Defaults are applied with defaults_only=True, so they only fill in fields that
      are None. Setting a field to None therefore cannot opt out of a default - only a
      concrete value can, which is why every field below is set explicitly.

    * assistant_confidence_threshold defaults to 0.4 and, when truthy and > 0, adds a
      ConfidenceCriteria stopping rule that lets the assistant stop drafting early.
      That is an interesting mechanism in its own right, but it is deliberately out of
      scope for this slice. Setting it to 0.0 disables it, so the assisted arm is plain
      constant-lookahead drafting and any measured difference is attributable to the
      draft-and-verify loop rather than to an adaptive schedule.
    """
    config = assistant.generation_config
    config.num_assistant_tokens = int(num_assistant_tokens)
    config.num_assistant_tokens_schedule = "constant"
    config.assistant_confidence_threshold = 0.0
    return {
        "num_assistant_tokens": int(config.num_assistant_tokens),
        "num_assistant_tokens_schedule": str(config.num_assistant_tokens_schedule),
        "assistant_confidence_threshold": float(config.assistant_confidence_threshold),
        "applied_to": "assistant.generation_config",
    }


def build_generate_kwargs(
    method: str, config, assistant=None, max_new_tokens: int | None = None
) -> dict:
    """The generate() keyword arguments for one method at one generation length.

    Both arms share the same decoding settings on purpose: the only difference between
    them is whether the target model gets an assistant to draft candidates. Anything
    else that differed would be a confound.

    `max_new_tokens` is the length of the point being run. A sweep configuration holds
    several lengths, so the runner always passes the one it is on; when it is omitted the
    configuration's own (single) value is used, which keeps single-point callers simple.
    """
    kwargs: dict = {
        "max_new_tokens": config.max_new_tokens if max_new_tokens is None else max_new_tokens,
        "do_sample": False,  # greedy: deterministic, and the lossless case we can check
        "num_beams": 1,
        "use_cache": True,  # without the key/value cache the comparison is meaningless
    }

    if generation_config_supports("disable_compile"):
        # transformers can compile parts of the forward pass automatically once certain
        # cache implementations are in use. A compiler decision changing between arms
        # would contaminate the comparison, so it is pinned and recorded.
        kwargs["disable_compile"] = True

    if method == ASSISTED:
        if assistant is None:
            raise ValueError("the assisted method needs an assistant model")
        kwargs["assistant_model"] = assistant
    elif method != BASELINE:
        raise ValueError(f"unknown method {method!r}")

    return kwargs


def json_safe_settings(kwargs: dict) -> dict:
    """The part of a generate() call worth recording: plain values, no model objects."""
    settings = {}
    for key, value in kwargs.items():
        if isinstance(value, (bool, int, float, str)) or value is None:
            settings[key] = value
        else:
            # e.g. assistant_model=<module>: record that it was present, not the module.
            settings[key] = f"<{type(value).__name__}>"
    return settings
