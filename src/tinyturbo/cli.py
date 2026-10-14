"""Command line interface.

    python -m tinyturbo run    --config configs/slice_cpu.yaml
    python -m tinyturbo report --config configs/slice_cpu.yaml
    python -m tinyturbo verify --config configs/slice_cpu.yaml

The same three commands run the main sweep, which covers every prompt at every
generation length the configuration names (30 points and 180 measured trials, 24
minutes on a T4 when it was recorded as run gpu-main-9e26c12945e6):

    python -m tinyturbo run    --config configs/gpu_main.yaml
    python -m tinyturbo verify --config configs/gpu_main.yaml
    python -m tinyturbo report --config configs/gpu_main.yaml

`run` produces raw records, `report` turns them into a table and figures, and
`verify` checks the records without writing anything. Exit code 0 means it worked,
1 means the records are unusable, 2 means the configuration or environment was wrong.

Two more commands cover the stages that score the cost model rather than measure a
speedup:

    python -m tinyturbo calibrate --config configs/gpu_main.yaml   # Stage 0: tau_t, tau_t_verify, tau_d
    python -m tinyturbo profile   --config configs/gpu_main.yaml   # Stage 4: two cases, traced

`calibrate` writes `results/calibration/<run_id>-calibration.json` and `profile` writes
`results/profiling/<run_id>-profiling.json`. Neither is a run's evidence - that is the raw
JSONL - and `report` refuses to score a run with a calibration that describes a different
machine or configuration.

`profile` traces two representative cases. Where the whole-generation trace does not fit in
the session's memory, `--trace-active` records a bounded window of the decode loop instead,
with one profiler step per target forward pass; the artifact then records that its numbers
describe a window and leaves the whole-run residual unset rather than computing it from a
part of the run the window does not cover. `--max-new-tokens` narrows the cases to one.

`--trials` is not part of the run's identity, so re-running with a larger value resumes
and appends. `--prompt-id` and `--max-new-tokens` are, because they change which points
ran: a narrowed run gets its own file rather than mixing with the sweep it came from.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .calibrate import calibrate
from .config import ConfigError, config_fingerprint, load_config, run_id
from .env import DeviceError, detect_device, resolve_attention, resolve_dtype
from .models import ModelError, load_models
from .profiling import ProfilingError, TraceWindow, cases_for, profile as profile_cases
from .prompts import PromptError, get_prompt, load_prompts
from .report import (
    ReportError,
    build_report,
    calibration_path_for,
    profiling_path_for,
    raw_path_for,
    read_run,
    validate_run,
    write_json_artifact,
)
from .runner import RunError, run_sweep


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tinyturbo",
        description="Measure and compare plain greedy generation against assisted generation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run the trials in a configuration")
    run_parser.add_argument("--config", required=True, help="path to a YAML configuration")
    run_parser.add_argument(
        "--prompt-id",
        default=None,
        help="run only this prompt from the configuration; it must be one of them",
    )
    run_parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="run only this generation length from the configuration; it must be one of them",
    )
    run_parser.add_argument(
        "--trials", type=int, default=None, help="override the number of measured trials"
    )
    run_parser.add_argument(
        "--force",
        action="store_true",
        help="delete existing raw records for this configuration and start again",
    )

    for name, help_text in (
        ("report", "build the CSV and figure from raw records"),
        ("verify", "check raw records without writing anything"),
    ):
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument("--config", required=True, help="path to a YAML configuration")

    # Stage 0. Measured on the same models, dtype, device and lookahead as the run the
    # calibration will be used to score, which is why it takes a run configuration rather
    # than its own set of knobs.
    calibrate_parser = subparsers.add_parser(
        "calibrate",
        help="Stage 0: measure tau_t, tau_t_verify(gamma) and tau_d on this machine",
    )
    calibrate_parser.add_argument("--config", required=True, help="path to a YAML configuration")
    calibrate_parser.add_argument(
        "--warmups", type=int, default=5, help="forward passes discarded before sampling"
    )
    calibrate_parser.add_argument(
        "--samples", type=int, default=20, help="timed forward passes per primitive"
    )
    calibrate_parser.add_argument(
        "--prompt-id",
        default=None,
        help="prompt whose token length to calibrate at; defaults to the config's first",
    )
    calibrate_parser.add_argument("--out", default=None, help="where to write the artifact")

    # Stage 4. Two representative cases from the sweep that already ran.
    profile_parser = subparsers.add_parser(
        "profile", help="Stage 4: trace two representative cases with labelled forward passes"
    )
    profile_parser.add_argument("--config", required=True, help="path to a YAML configuration")
    profile_parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="unprofiled generate() runs before each trace, to keep start-up out of it",
    )
    profile_parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="trace only the case with this generation length; it must be one of them",
    )
    # A bounded trace. The whole-generation trace is what the T4 could not fit in memory;
    # recording a window of the loop instead is the cheap route, and it is opt-in because a
    # window is not the whole generation and the artifact has to say which one it holds.
    profile_parser.add_argument(
        "--trace-active",
        type=int,
        default=None,
        help=(
            "record only this many target forward passes (a bounded trace) instead of the "
            "whole generation; the event buffer then scales with this number"
        ),
    )
    profile_parser.add_argument(
        "--trace-wait",
        type=int,
        default=1,
        help="iterations skipped before the recorded window starts (default 1)",
    )
    profile_parser.add_argument(
        "--trace-warmup",
        type=int,
        default=1,
        help="iterations traced but discarded before the recorded window (default 1)",
    )
    profile_parser.add_argument("--out", default=None, help="where to write the artifact")

    return parser


def _verify(config) -> int:
    """Check that a run's records are usable, and complain clearly when they are not."""
    source = raw_path_for(config)
    manifest, trials = read_run(source)
    warnings = validate_run(manifest, trials, source)

    # Forward-pass counts are the mechanism measurement for this whole study. Records
    # without them are structurally valid but useless for the question being asked, so
    # this is treated as a failure rather than a warning.
    missing_counts = [trial for trial in trials if not trial["counts"].get("target_forwards")]

    print(f"raw records : {source}")
    print(f"run id      : {manifest['run_id']}")
    print(f"measured    : {len(trials)} trial records")
    print(f"methods     : {', '.join(sorted({trial['method'] for trial in trials}))}")
    print(f"hardware    : {manifest['hardware'].get('gpu_name') or 'CPU'} / {manifest['resolved']['dtype']}")
    scope = manifest.get("run_scope") or {}
    if scope.get("prompt_ids"):
        lengths = scope.get("max_new_tokens") or []
        print(
            f"scope       : {len(scope['prompt_ids'])} prompt(s) x {len(lengths)} length(s) "
            f"= {len(scope['prompt_ids']) * len(lengths)} points"
        )
    for warning in warnings:
        print(f"WARNING: {warning}")

    if missing_counts:
        print(
            f"FAIL: {len(missing_counts)} trial(s) recorded no target forward passes, so "
            "tokens-per-forward cannot be computed. The instrumentation did not catch "
            "the generation loop."
        )
        return 1
    if any(trial.get("is_warmup") for trial in trials):
        print("FAIL: warm-up trials leaked into the measured set")
        return 1

    print("OK: records are complete and the forward-pass instrumentation worked.")
    return 0


def _resolve_and_load(config):
    """Device, dtype, attention and models, resolved exactly as `run` resolves them.

    Shared by `calibrate` and `profile` so a Stage 0 or Stage 4 measurement cannot end up
    describing a different precision or backend than the sweep it will be compared with.
    """
    device = detect_device(config.device)
    dtype_name = resolve_dtype(config.models.dtype, device)
    attn_implementation = resolve_attention(config.models.attn_implementation, device)
    models = load_models(config.models, dtype_name, attn_implementation, device)
    return device, dtype_name, attn_implementation, models


def _calibrate(config, args) -> int:
    """Stage 0: measure the three primitives and record rho and kappa."""
    device, dtype_name, attn_implementation, models = _resolve_and_load(config)
    prompts = load_prompts(config.prompts_file)
    prompt = get_prompt(prompts, args.prompt_id or config.prompt_ids[0])
    tokenizer_inputs = models.tokenizer(prompt.text, return_tensors="pt")
    prompt_tokens = int(tokenizer_inputs["input_ids"].shape[1])

    print(f"calibration : {config.as_dict()['name']} on {device} / {dtype_name} / attention={attn_implementation}")
    print(f"models      : {config.models.target} target, {config.models.assistant} assistant")
    print(f"prompt      : {prompt.prompt_id} ({prompt_tokens} tokens, cache populated as in a run)")
    print(f"protocol    : {args.warmups} warm-up + {args.samples} timed passes per primitive")
    print()

    payload = calibrate(
        models,
        device,
        dtype_name,
        attn_implementation,
        gamma=config.num_assistant_tokens,
        prompt_ids=tokenizer_inputs["input_ids"],
        prompt_id=prompt.prompt_id,
        warmups=args.warmups,
        samples=args.samples,
        config_fingerprint=config_fingerprint(config),
        run_id=run_id(config),
    )

    path = Path(args.out) if args.out else calibration_path_for(config)
    write_json_artifact(payload, path)

    primitives = payload["primitives"]
    for key in ("target_single", "target_verify", "assistant_single"):
        stats = primitives[key]
        print(
            f"  {stats['label']:<52} {stats['median_ms']:8.3f} ms"
            f"  (iqr {stats['iqr_ms']:6.3f}, n={stats['samples']})"
        )
    ratios = payload["ratios"]
    print()
    print(f"  rho   = tau_t_verify / tau_t            = {ratios['rho']:.3f}")
    print(f"  tau_d / tau_t                           = {ratios['tau_d_over_tau_t']:.3f}")
    print(f"  kappa = gamma * tau_d / tau_t           = {ratios['kappa']:.3f}")
    print()
    print(f"artifact    : {path}")
    print("next: python -m tinyturbo report --config " + args.config)
    return 0


def _profile(config, args) -> int:
    """Stage 4: trace two representative cases from the sweep."""
    cases = cases_for(args.max_new_tokens)
    window = (
        None
        if args.trace_active is None
        else TraceWindow(
            active=args.trace_active, wait=args.trace_wait, warmup=args.trace_warmup
        )
    )

    device, dtype_name, attn_implementation, models = _resolve_and_load(config)
    print(f"profiling   : {config.as_dict()['name']} on {device} / {dtype_name} / attention={attn_implementation}")
    print(f"protocol    : {args.warmup_runs} unprofiled generate() run per case, then one trace")
    print("scope       : labelled forward passes only; unlabelled time is reported as unattributed")
    if window is None:
        print("trace       : whole generation (every step recorded)")
    else:
        print(
            f"trace       : bounded - {window.wait} skipped + {window.warmup} warmed up, then "
            f"{window.active} step(s) recorded, one step per target forward pass"
        )
        print("              the residual is left unset: a window does not cover the whole run")
    if config.smoke:
        print("NOTE        : this configuration is a smoke run - the artifact is a plumbing check")
    print()

    payload = profile_cases(
        models,
        config,
        device,
        dtype_name,
        attn_implementation,
        cases=cases,
        warmup_runs=args.warmup_runs,
        window=window,
        smoke=config.smoke,
    )
    payload["config_fingerprint"] = config_fingerprint(config)
    payload["run_id"] = run_id(config)

    path = Path(args.out) if args.out else profiling_path_for(config)
    write_json_artifact(payload, path)

    for case in payload["cases"]:
        print(f"case        : {case['case']['prompt_id']} @ {case['case']['max_new_tokens']} tokens")
        for method, measured in case["methods"].items():
            labels = measured["labels"]
            attributed = measured["attributed_ms"]
            unattributed = measured["unattributed_ms"]
            print(
                f"  {method:<9} e2e {measured['e2e_ms']:9.1f} ms"
                f"  target passes {labels['tinyturbo:target_forward']['count']:>4}"
                f"  draft passes {labels['tinyturbo:assistant_forward']['count']:>4}"
                f"  labelled ({measured['clock']}) "
                + (f"{attributed:9.1f} ms" if attributed is not None else "  (window)")
                + (
                    f"  unattributed {unattributed:9.1f} ms"
                    if unattributed is not None
                    else "  residual not computed for a window"
                )
            )
    # A window that never opened is not a measurement of nothing - it is a protocol that did
    # not fit the run. It is reported here rather than inferred from a null in the artifact.
    empty_windows = [
        (case["case"], method)
        for case in payload["cases"]
        for method, measured in case["methods"].items()
        if window is not None and not measured["steps_recorded"]
    ]
    for case, method in empty_windows:
        print(
            f"WARNING: {method} on {case['prompt_id']} @ {case['max_new_tokens']} tokens "
            f"recorded no steps: the window opens after {window.wait + window.warmup} forward "
            "passes and this run did not take that many. Nothing about it was measured."
        )
    print()
    print(f"artifact    : {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        if args.command == "run":
            summary = run_sweep(
                config,
                prompt_id=args.prompt_id,
                max_new_tokens=args.max_new_tokens,
                trials=args.trials,
                force=args.force,
            )
            print()
            print(f"wrote {summary['written']} measured trial(s) to {summary['raw_path']}")
            if summary["skipped"]:
                print(f"skipped {summary['skipped']} trial(s) already present")
            print("next: python -m tinyturbo report --config " + args.config)
            return 0

        if args.command == "report":
            build_report(config)
            return 0

        if args.command == "verify":
            return _verify(config)

        if args.command == "calibrate":
            return _calibrate(config, args)

        if args.command == "profile":
            return _profile(config, args)

    except (
        ConfigError,
        PromptError,
        DeviceError,
        ModelError,
        RunError,
        ReportError,
        ProfilingError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    parser.error(f"unknown command {args.command!r}")
    return 2
