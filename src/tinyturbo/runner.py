"""Running the sweep: the configured points, both methods, real measurements.

The runner is deliberately linear, because the whole point of this stage is that a
person can read it top to bottom and explain every number it produces:

  1. resolve the device, dtype and attention implementation for this machine
  2. load the models once
  3. measure the prompt prefill for every prompt in the grid, which both arms pay and
     neither can avoid
  4. run the arms interleaved, warm-ups first, over every (prompt, generation length)
     point, recording each trial
  5. append one JSON line per trial, flushed immediately

A configuration may name one prompt and one length - the vertical slice - or a grid of
them - the main sweep. This module runs the grid either way.

Interleaving matters. Running all the baseline trials and then all the assisted
trials would let a slow drift in the machine (thermal throttling, a noisy neighbour on
a shared cloud GPU) be attributed to the method. For the same reason the two arms of a
point run back to back, rather than a whole round of one method at a time.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import torch

from .config import RunConfig, config_fingerprint, run_id
from .env import detect_device, hardware_fingerprint, resolve_attention, resolve_dtype
from .instrument import (
    ForwardCounts,
    Stopwatch,
    count_forward_calls,
    measure_prefill_ms,
    peak_memory_mb,
    reset_peak_memory,
)
from .methods import (
    ASSISTED,
    apply_assistant_controls,
    build_generate_kwargs,
    json_safe_settings,
)
from .metrics import (
    accepted_len_mean,
    draft_tokens_per_step,
    generated_token_ids,
    tokens_per_second,
)
from .models import LoadedModels, assert_compatible_vocabularies, load_models, model_dtype_name
from .prompts import Prompt, get_prompt, load_prompts, prompt_set_hash

RUN_RECORD_TYPE = "run"
TRIAL_RECORD_TYPE = "trial"


class RunError(RuntimeError):
    """Raised when a run cannot start or cannot continue safely."""


@dataclass
class PromptContext:
    """Everything the trials of one prompt need, resolved once at the start of the run.

    A prompt's tokenizer inputs and its prefill measurement do not depend on how many
    tokens are generated, so they are measured once per prompt and shared by every
    length the sweep runs for it.
    """

    config: RunConfig
    identifier: str
    config_hash: str
    device: str
    dtype_name: str
    attn_implementation: str
    fingerprint: dict
    models: LoadedModels
    prompt: Prompt
    inputs: dict
    prompt_length: int
    prefill_ms: float
    eos_ids: list[int]
    assistant_controls: dict | None


def _log(message: str) -> None:
    """Print progress immediately, so a long run on a remote machine shows life."""
    print(message, flush=True)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_state() -> dict:
    """The commit that produced a run, when the project is under version control.

    Recorded but optional: results from a directory that is not a git repository are
    still usable, they are just harder to tie back to a version of the code.
    """
    unavailable = {"git_available": False, "git_commit": None, "git_dirty": None}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=False
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return unavailable
    if commit.returncode != 0:
        return unavailable
    return {
        "git_available": True,
        "git_commit": commit.stdout.strip(),
        "git_dirty": bool(status.stdout.strip()),
    }


def eos_token_ids(model) -> list[int]:
    """The model's EOS ids as a list. transformers v5 allows a list here, v4 only an int."""
    raw = getattr(model.generation_config, "eos_token_id", None)
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [int(value) for value in raw]
    return [int(raw)]


def extract_sequence(output) -> list[int]:
    """Pull the first generated sequence out of whatever generate() returned.

    generate() returns a plain tensor by default and a ModelOutput when asked for one,
    so both shapes are handled here rather than at every call site.
    """
    sequence = output.sequences[0] if hasattr(output, "sequences") else output[0]
    return [int(token) for token in sequence.tolist()]


def run_one_trial(
    context: PromptContext,
    method: str,
    trial_index: int,
    is_warmup: bool,
    max_new_tokens: int,
) -> dict:
    """Run one generation at one point and return its record."""
    config = context.config
    kwargs = build_generate_kwargs(method, config, context.models.assistant, max_new_tokens)

    torch.manual_seed(config.seed)
    reset_peak_memory(context.device)
    counts = ForwardCounts()
    watch = Stopwatch(context.device)

    with torch.inference_mode():
        with count_forward_calls(context.models.target, "target", counts):
            if method == ASSISTED and context.models.assistant is not None:
                # Nested context: the assistant runs a forward pass per drafted token,
                # so counting it separately is what makes the draft cost visible.
                with count_forward_calls(context.models.assistant, "assistant", counts):
                    watch.start()
                    output = context.models.target.generate(**context.inputs, **kwargs)
                    e2e_ms = watch.stop()
            else:
                watch.start()
                output = context.models.target.generate(**context.inputs, **kwargs)
                e2e_ms = watch.stop()

    memory = peak_memory_mb(context.device)

    sequence = extract_sequence(output)
    generated = generated_token_ids(sequence, context.prompt_length)
    tokens_generated = len(generated)

    # End-to-end time minus a separately measured prefill pass. EOS handling and the
    # generate() bookkeeping are inside the measured region but outside our prefill
    # measurement, so this is an approximation and is labelled as one below.
    decode_ms = max(0.0, e2e_ms - context.prefill_ms)
    decode_ms_per_token = decode_ms / tokens_generated if tokens_generated else None

    notes: list[str] = ["decode_ms is e2e_ms minus a separately measured prefill pass; treat it as an upper bound on decode time."]
    if counts.target == 0:
        notes.append("target_forwards is 0: forward-pass instrumentation did not catch the generation loop, so accepted_len_mean is not usable.")
    if method == ASSISTED and counts.assistant == 0:
        notes.append("assistant_forwards is 0: the assistant model appears not to have been used.")

    record = {
        "record_type": TRIAL_RECORD_TYPE,
        "run_id": context.identifier,
        "config_fingerprint": context.config_hash,
        "created_utc": utc_now_iso(),
        "method": method,
        "trial": trial_index,
        "is_warmup": is_warmup,
        "hardware": {
            "device": context.device,
            "gpu_name": context.fingerprint["gpu_name"],
            "is_cuda": context.fingerprint["is_cuda"],
        },
        "resolved": {
            "dtype": context.dtype_name,
            "attn_implementation": context.attn_implementation,
            "loaded_target_dtype": model_dtype_name(context.models.target),
        },
        "assistant_controls": context.assistant_controls,
        "prompt": {
            "prompt_id": context.prompt.prompt_id,
            "stratum": context.prompt.stratum,
            "prompt_tokens": context.prompt_length,
        },
        "generation": json_safe_settings(kwargs),
        "timing": {
            "prefill_ms": context.prefill_ms,
            "e2e_ms": e2e_ms,
            "decode_ms": decode_ms,
            "decode_ms_per_token": decode_ms_per_token,
        },
        "counts": counts.as_dict() | {"tokens_generated": tokens_generated},
        "derived": {
            "accepted_len_mean": accepted_len_mean(tokens_generated, counts.target),
            "draft_tokens_per_step": draft_tokens_per_step(counts.assistant, counts.target),
            "tokens_per_second": tokens_per_second(tokens_generated, e2e_ms),
            "decode_tokens_per_second": tokens_per_second(tokens_generated, decode_ms),
        },
        "memory": memory,
        "output": {
            "hit_eos": bool(generated and generated[-1] in context.eos_ids),
            "generated_token_ids": generated,
            "generated_text_preview": context.models.tokenizer.decode(
                generated, skip_special_tokens=True
            )[:200],
        },
        "notes": notes,
    }
    return record


def build_manifest(parts: dict) -> dict:
    """Assemble the run-level record that every trial in the file points back to.

    This record is what makes a result file self-explanatory six months later: it
    carries the full configuration, the machine fingerprint, the prompt set hash and
    the command line, so a trial can never be separated from the conditions that
    produced it.
    """
    config: RunConfig = parts["config"]
    return {
        "record_type": RUN_RECORD_TYPE,
        "run_id": parts["identifier"],
        "config_fingerprint": parts["config_hash"],
        "created_utc": utc_now_iso(),
        "config": config.as_dict(),
        "hardware": parts["fingerprint"],
        "resolved": {
            "device": parts["device"],
            "dtype": parts["dtype_name"],
            "attn_implementation": parts["attn_implementation"],
        },
        "model_facts": parts["model_facts"],
        "prompt_set_hash": parts["prompt_set_hash"],
        "prompt_count": parts["prompt_count"],
        "run_scope": parts["run_scope"],
        "assistant_controls": parts["assistant_controls"],
        "git": git_state(),
        "command": " ".join(sys.argv),
    }


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSON Lines file into a list of records."""
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RunError(f"{path}:{line_number} is not valid JSON: {exc}") from exc
    return records


def completed_trial_keys(records: list[dict]) -> set[tuple]:
    """Keys of trials already present, so a resumed run does not duplicate them.

    A key includes the generation length, because the same prompt is run at several of
    them: without it, trial 0 at 32 tokens would mark trial 0 at 256 tokens as done.
    """
    keys: set[tuple] = set()
    for record in records:
        if record.get("record_type") != TRIAL_RECORD_TYPE:
            continue
        keys.add(
            (
                record.get("prompt", {}).get("prompt_id"),
                record.get("generation", {}).get("max_new_tokens"),
                record.get("method"),
                record.get("trial"),
            )
        )
    return keys


# The fields that decide whether two runs happened on the same machine running the same
# stack. Deliberately not the whole fingerprint: the CPU count, the peak memory reading
# and the raw nvidia-smi line can differ between two sessions on one identical card, and
# refusing to resume over those would be noise rather than protection.
HARDWARE_IDENTITY_FIELDS = ("device", "gpu_name", "gpu_driver_version", "cuda_version", "torch")


def hardware_differences(previous: dict, current: dict) -> list[str]:
    """Fields that say two runs used different hardware or a different software stack."""
    return [
        f"{field}: {previous.get(field)!r} -> {current.get(field)!r}"
        for field in HARDWARE_IDENTITY_FIELDS
        if previous.get(field) != current.get(field)
    ]


def scope_for(config: RunConfig, measured_trials: int) -> dict:
    """The declared scope of a run, recorded once in its manifest.

    A single-point run keeps the exact shape it has always had, because published
    manifests are read against it. A sweep declares the whole grid, which is what lets
    `report` refuse a trial belonging to a point the configuration does not contain.
    """
    if not config.is_sweep:
        return {
            "prompt_id": config.prompt_id,
            "methods": list(config.methods),
            "warmup_trials": config.warmup_trials,
            "measured_trials": measured_trials,
            "is_slice": True,
            "note": "vertical slice: one prompt, not the full benchmark",
        }
    return {
        "prompt_ids": list(config.prompt_ids),
        "max_new_tokens": list(config.max_new_tokens_values),
        "methods": list(config.methods),
        "warmup_trials": config.warmup_trials,
        "measured_trials": measured_trials,
        "is_slice": False,
        "note": (
            f"sweep: {len(config.prompt_ids)} prompt(s) x "
            f"{len(config.max_new_tokens_values)} generation length(s) "
            f"= {len(config.points)} points, {len(config.methods)} method(s)"
        ),
    }


def narrow(
    config: RunConfig, prompt_id: str | None, max_new_tokens: int | None
) -> RunConfig:
    """Restrict a configuration to a subset of its points, identity included.

    A command line override that changes which points run changes the run itself, so it
    changes the fingerprint and therefore the run id and the raw file. A narrowed run
    gets its own file instead of appending to the file the full sweep owns, where its
    trials would otherwise be indistinguishable from the sweep's.
    """
    prompt_ids = config.prompt_ids
    if prompt_id is not None:
        if prompt_id not in config.prompt_ids:
            raise RunError(
                f"prompt '{prompt_id}' is not in this configuration, which runs: "
                f"{', '.join(config.prompt_ids)}"
            )
        prompt_ids = (prompt_id,)

    lengths = config.max_new_tokens_values
    if max_new_tokens is not None:
        if max_new_tokens not in config.max_new_tokens_values:
            raise RunError(
                f"max_new_tokens {max_new_tokens} is not in this configuration, which runs: "
                f"{', '.join(str(value) for value in config.max_new_tokens_values)}"
            )
        lengths = (max_new_tokens,)

    if prompt_ids == config.prompt_ids and lengths == config.max_new_tokens_values:
        return config
    return replace(config, prompt_ids=prompt_ids, max_new_tokens_values=lengths)


def run_sweep(
    config: RunConfig,
    prompt_id: str | None = None,
    max_new_tokens: int | None = None,
    trials: int | None = None,
    force: bool = False,
) -> dict:
    """Run every point of the configuration and return a small summary of what happened."""
    config = narrow(config, prompt_id, max_new_tokens)

    device = detect_device(config.device)
    dtype_name = resolve_dtype(config.models.dtype, device)
    attn_implementation = resolve_attention(config.models.attn_implementation, device)
    fingerprint = hardware_fingerprint(device)

    identifier = run_id(config)
    config_hash = config_fingerprint(config)
    measured_trials = trials if trials is not None else config.measured_trials
    if measured_trials < 1:
        raise RunError(f"measured trials must be >= 1, got {measured_trials}")

    prompts = load_prompts(config.prompts_file)

    raw_path = config.raw_dir / f"{identifier}.jsonl"
    config.raw_dir.mkdir(parents=True, exist_ok=True)
    if force and raw_path.exists():
        _log(f"removing existing raw file: {raw_path}")
        raw_path.unlink()

    existing = read_jsonl(raw_path) if raw_path.exists() else []
    if existing:
        previous_manifest = next(
            (record for record in existing if record.get("record_type") == RUN_RECORD_TYPE), None
        )
        if previous_manifest is None:
            raise RunError(f"{raw_path} exists but has no run manifest; refusing to append")
        if previous_manifest.get("config_fingerprint") != config_hash:
            raise RunError(
                f"{raw_path} was produced by a different configuration "
                f"({previous_manifest.get('config_fingerprint', 'unknown')} != {config_hash}). "
                f"Use --force to start a fresh file."
            )
        differences = hardware_differences(previous_manifest.get("hardware") or {}, fingerprint)
        if differences:
            # A resumed Colab session can land on a different card or driver. Pooling the
            # two would produce a speedup that describes no machine that exists.
            raise RunError(
                f"{raw_path} was produced on different hardware ({'; '.join(differences)}). "
                "Results from two machines are never pooled, so this run will not append to "
                "it. Give the run a different 'name:' in the config so this machine's trials "
                "get their own file, or pass --force if you accept losing the trials already "
                "recorded here."
            )
    done_keys = completed_trial_keys(existing)

    _log(f"run id           : {identifier}")
    _log(f"config           : {config.as_dict()['name']} ({config_hash[:12]})")
    _log(f"device / dtype   : {device} / {dtype_name} / attention={attn_implementation}")
    _log(f"gpu              : {fingerprint['gpu_name'] or 'none (CPU)'}")
    _log(f"target           : {config.models.target}")
    _log(f"assistant        : {config.models.assistant or 'none'}")
    _log(
        f"scope            : {len(config.points)} point(s) "
        f"({len(config.prompt_ids)} prompt(s) x {len(config.max_new_tokens_values)} length(s))"
    )

    models = load_models(config.models, dtype_name, attn_implementation, device)
    if models.assistant is not None and ASSISTED in config.methods:
        # transformers treats equal vocab_size as proof that the two models share a token
        # space and then uses the target's tokenizer for both. Checked here so a mismatch
        # fails before any generation rather than deep inside the generation loop.
        assert_compatible_vocabularies(models.target, models.assistant)

    assistant_controls = None
    if models.assistant is not None:
        assistant_controls = apply_assistant_controls(models.assistant, config.num_assistant_tokens)

    _log(
        f"parameters       : target={models.target_parameters:,}"
        + (f" assistant={models.assistant_parameters:,}" if models.assistant_parameters else "")
    )

    # One context per prompt. A prompt's tokens and its prefill cost do not depend on how
    # many tokens are generated, so a sweep measures a prompt's prefill once and every
    # length it runs for that prompt reuses the measurement.
    configured_prompts = [get_prompt(prompts, wanted) for wanted in config.prompt_ids]
    contexts: dict[str, PromptContext] = {}
    for configured_prompt in configured_prompts:
        tokenizer_inputs = models.tokenizer(configured_prompt.text, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in tokenizer_inputs.items()}
        prompt_length = int(inputs["input_ids"].shape[1])
        prefill_ms = measure_prefill_ms(models.target, inputs, device)
        contexts[configured_prompt.prompt_id] = PromptContext(
            config=config,
            identifier=identifier,
            config_hash=config_hash,
            device=device,
            dtype_name=dtype_name,
            attn_implementation=attn_implementation,
            fingerprint=fingerprint,
            models=models,
            prompt=configured_prompt,
            inputs=inputs,
            prompt_length=prompt_length,
            prefill_ms=prefill_ms,
            eos_ids=eos_token_ids(models.target),
            assistant_controls=assistant_controls,
        )
        _log(
            f"prompt           : {configured_prompt.prompt_id} ({configured_prompt.stratum})"
            f"  {prompt_length} tokens  prefill {prefill_ms:.1f} ms"
        )

    manifest = build_manifest(
        {
            "config": config,
            "identifier": identifier,
            "config_hash": config_hash,
            "fingerprint": fingerprint,
            "device": device,
            "dtype_name": dtype_name,
            "attn_implementation": attn_implementation,
            "model_facts": {
                **models.as_dict(),
                "target_dtype_loaded": model_dtype_name(models.target),
                "assistant_dtype_loaded": model_dtype_name(models.assistant) if models.assistant else None,
            },
            "prompt_set_hash": prompt_set_hash(prompts),
            "prompt_count": len(prompts),
            "run_scope": scope_for(config, measured_trials),
            "assistant_controls": assistant_controls,
        }
    )

    mode = "a" if existing else "w"
    written = 0
    skipped = 0
    with raw_path.open(mode, encoding="utf-8") as handle:
        if not existing:
            handle.write(json.dumps(manifest) + "\n")
            handle.flush()

        total_rounds = config.warmup_trials + measured_trials
        for round_index in range(total_rounds):
            is_warmup = round_index < config.warmup_trials
            trial_index = round_index - config.warmup_trials
            for configured_prompt in configured_prompts:
                context = contexts[configured_prompt.prompt_id]
                for length in config.max_new_tokens_values:
                    for method in config.methods:
                        # Warm-up rounds use trial index -1, -2, ... so they are keyed just
                        # like measured trials, and the key carries the generation length
                        # because every prompt is run at several of them. Skipping completed
                        # keys is what makes re-running a config idempotent instead of
                        # quietly piling up duplicate records.
                        key = (configured_prompt.prompt_id, length, method, trial_index)
                        if key in done_keys:
                            skipped += 1
                            continue

                        record = run_one_trial(
                            context, method, trial_index, is_warmup, length
                        )
                        handle.write(json.dumps(record) + "\n")
                        handle.flush()

                        where = f"{configured_prompt.prompt_id}@{length}"
                        if is_warmup:
                            _log(
                                f"  warm-up   {where:<22} {method:<9} "
                                f"e2e={record['timing']['e2e_ms']:.1f} ms"
                            )
                            continue

                        written += 1
                        timing = record["timing"]
                        derived = record["derived"]
                        accepted = derived["accepted_len_mean"]
                        accepted_text = f"{accepted:.2f}" if accepted is not None else "n/a"
                        rate = derived["tokens_per_second"]
                        rate_text = f"{rate:.1f}" if rate is not None else "n/a"
                        _log(
                            f"  trial {trial_index}  {where:<22} {method:<9} "
                            f"e2e={timing['e2e_ms']:8.1f} ms"
                            f"  {rate_text:>7} tok/s"
                            f"  target_forwards={record['counts']['target_forwards']:>3}"
                            f"  tokens_per_forward={accepted_text}"
                        )

    return {
        "raw_path": raw_path,
        "run_id": identifier,
        "written": written,
        "skipped": skipped,
        "points": len(config.points),
        "total_in_file": written + skipped + len(existing),
    }
