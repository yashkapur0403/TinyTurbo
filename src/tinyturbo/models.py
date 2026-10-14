"""Loading the tokenizer, the target model and (optionally) the assistant model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ModelSpec
from .env import dtype_from_name


class ModelError(RuntimeError):
    """Raised when a model pair cannot be used together for assisted decoding."""


@dataclass
class LoadedModels:
    """The objects the runner needs, plus the facts worth recording about them."""

    tokenizer: object
    target: object
    assistant: object | None
    target_parameters: int
    assistant_parameters: int | None
    target_vocab_size: int | None = None
    assistant_vocab_size: int | None = None

    @property
    def parameter_ratio(self) -> float | None:
        """Assistant size as a fraction of target size.

        This is the single most important structural number in assisted decoding: it
        sets how cheap a draft pass can be relative to a verification pass.
        """
        if not self.assistant_parameters or not self.target_parameters:
            return None
        return self.assistant_parameters / self.target_parameters

    def as_dict(self) -> dict:
        return {
            "target_parameters": self.target_parameters,
            "assistant_parameters": self.assistant_parameters,
            "parameter_ratio": self.parameter_ratio,
            "target_vocab_size": self.target_vocab_size,
            "assistant_vocab_size": self.assistant_vocab_size,
        }


def count_parameters(model) -> int:
    """Total number of parameters, in elements rather than bytes."""
    return sum(parameter.numel() for parameter in model.parameters())


def vocab_size(model) -> int | None:
    """The vocabulary size, read the same way transformers reads it for its own checks.

    transformers decides whether two models share a token space by comparing exactly this
    value (generation/utils.py: if vocab_size equal -> same tokenizer assumed, otherwise
    `tokenizer` and `assistant_tokenizer` become mandatory). Reading it through the same
    accessor keeps our guard consistent with the library's decision.
    """
    try:
        return int(model.config.get_text_config().vocab_size)
    except Exception:  # pragma: no cover - older config shapes
        value = getattr(model.config, "vocab_size", None)
        return int(value) if value is not None else None


def assert_compatible_vocabularies(target, assistant) -> tuple[int | None, int | None]:
    """Check that the two models can share a token space, and return both sizes.

    A mismatch does not silently produce slightly wrong output: transformers rejects the
    call unless you also pass two tokenizers for universal assisted decoding, which this
    project does not implement. Checking here replaces a deep library error, raised after
    both models are resident in GPU memory, with a clear message before the run starts.
    """
    target_vocab = vocab_size(target)
    assistant_vocab = vocab_size(assistant)
    if target_vocab is not None and assistant_vocab is not None and target_vocab != assistant_vocab:
        raise ModelError(
            f"target and assistant vocabularies differ ({target_vocab} vs {assistant_vocab}). "
            "Assisted generation requires the same token space; either pick an assistant "
            "trained on the target's tokenizer, or use universal assisted decoding "
            "(pass tokenizer and assistant_tokenizer), which this project does not support."
        )
    return target_vocab, assistant_vocab


def load_tokenizer(name: str):
    """Load a tokenizer and make sure it has a padding token.

    SmolLM tokenizers have no pad token. generate() only needs one for batched input,
    which this study does not use, but several utility paths warn or fail without it,
    so it is set to the EOS token.
    """
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_causal_lm(name: str, dtype_name: str, attn_implementation: str, device: str):
    """Load one causal language model onto the chosen device.

    Note on the keyword: transformers v5 calls this argument `dtype`. In v4 it was
    `torch_dtype`, and passing `torch_dtype` to v5 raises rather than being ignored,
    which is a useful failure mode - it means a version mismatch is loud instead of
    silently loading the wrong precision.
    """
    model = AutoModelForCausalLM.from_pretrained(
        name,
        dtype=dtype_from_name(dtype_name),
        attn_implementation=attn_implementation,
    )
    model.to(device)
    model.eval()
    return model


def load_models(spec: ModelSpec, dtype_name: str, attn_implementation: str, device: str) -> LoadedModels:
    """Load everything named in the configuration."""
    tokenizer = load_tokenizer(spec.target)
    target = load_causal_lm(spec.target, dtype_name, attn_implementation, device)
    target_parameters = count_parameters(target)

    assistant = None
    assistant_parameters = None
    if spec.assistant:
        assistant = load_causal_lm(spec.assistant, dtype_name, attn_implementation, device)
        assistant_parameters = count_parameters(assistant)

    return LoadedModels(
        tokenizer=tokenizer,
        target=target,
        assistant=assistant,
        target_parameters=target_parameters,
        assistant_parameters=assistant_parameters,
        target_vocab_size=vocab_size(target),
        assistant_vocab_size=vocab_size(assistant) if assistant is not None else None,
    )


def model_dtype_name(model) -> str:
    """Read back the dtype a model actually holds, as a plain string.

    Recorded rather than assumed: a checkpoint can arrive in one dtype and be cast to
    another, and the recorded value must be the one that was really used.
    """
    try:
        return str(model.dtype).replace("torch.", "")
    except AttributeError:  # pragma: no cover - defensive
        return torch.get_default_dtype().__str__().replace("torch.", "")
