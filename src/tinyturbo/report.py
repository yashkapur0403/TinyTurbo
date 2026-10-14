"""Turning raw records into a table and a figure.

Two rules govern this module:

  * Nothing is invented. Every value here is computed from records that a real run
    wrote to disk, and if there are no usable records the report fails loudly rather
    than producing a plausible-looking empty table.
  * Comparisons live here, not in the raw data. Speedups need both arms of a
    comparison, so they are computed at report time from the paired records. A raw
    record can therefore never contain a speedup whose baseline does not exist.

Run it with `python -m tinyturbo report --config <config>`.
"""

from __future__ import annotations

from pathlib import Path

from typing import Any

import pandas as pd

from .config import RunConfig, run_id
from .metrics import exact_match, first_divergence_index, interquartile_range, median
from .runner import RUN_RECORD_TYPE, TRIAL_RECORD_TYPE, read_jsonl

# The exact columns of the derived CSV, in order. Listing them explicitly keeps the
# file stable and keeps internal fields (such as the raw token ids) out of it.
CSV_COLUMNS = [
    "run_id",
    "config_fingerprint",
    "config_name",
    "smoke",
    "device",
    "gpu_name",
    "dtype",
    "attn_implementation",
    "prompt_id",
    "stratum",
    "prompt_tokens",
    "max_new_tokens",
    "method",
    "trial",
    "target_forwards",
    "assistant_forwards",
    "tokens_generated",
    "prefill_ms",
    "e2e_ms",
    "decode_ms",
    "decode_ms_per_token",
    "tokens_per_second",
    "decode_tokens_per_second",
    "peak_alloc_mb",
    "peak_reserved_mb",
    "hit_eos",
    "accepted_len_mean",
    "draft_tokens_per_step",
    "baseline_e2e_ms",
    "speedup_e2e",
    "speedup_decode",
    "exact_match",
    "first_divergence_index",
]


class ReportError(RuntimeError):
    """Raised when a report cannot be produced honestly."""


def raw_path_for(config: RunConfig) -> Path:
    """Where a run's records live, derived from the same id the runner used."""
    return config.raw_dir / f"{run_id(config)}.jsonl"


def _as_list(value: Any) -> list[Any]:
    """A recorded configuration key is either one value or a list of them."""
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


def declared_points(manifest: dict) -> set[tuple[str, int | None]]:
    """The (prompt, generation length) points a manifest's own configuration declares.

    Read from the recorded configuration rather than from the scope note, so a trial can
    be traced in a file written by any version of the writer. A configuration that names
    a single prompt or a single length is one point, which is what the slice configs are.
    """
    config = manifest.get("config") or {}
    prompts = _as_list(config.get("prompt_ids") or config.get("prompt_id"))
    lengths = _as_list(config.get("max_new_tokens")) or [None]
    return {(prompt, length) for prompt in prompts for length in lengths}


def read_run(path: Path) -> tuple[dict, list[dict]]:
    """Read one raw file and split it into (manifest, measured trials)."""
    if not path.is_file():
        raise ReportError(f"no raw records at {path}. Run `python -m tinyturbo run` first.")

    records = read_jsonl(path)
    manifests = [record for record in records if record.get("record_type") == RUN_RECORD_TYPE]
    trials = [
        record
        for record in records
        if record.get("record_type") == TRIAL_RECORD_TYPE and not record.get("is_warmup")
    ]

    if not manifests:
        raise ReportError(f"{path} has no run manifest, so its records cannot be attributed")
    if len(manifests) > 1:
        raise ReportError(f"{path} contains {len(manifests)} run manifests; it should contain one")
    if not trials:
        raise ReportError(
            f"{path} has no completed measured trials (warm-ups do not count). Nothing to report."
        )
    return manifests[0], trials


def validate_run(manifest: dict, trials: list[dict], source: Path) -> list[str]:
    """Check that the records are internally consistent. Returns warnings.

    Errors are raised, warnings are returned. A trial that cannot be traced back to
    the manifest, or that is missing a measurement, is an error: silently dropping it
    would change the reported result.
    """
    warnings: list[str] = []

    for field in ("run_id", "config_fingerprint", "hardware", "config"):
        if not manifest.get(field):
            raise ReportError(f"{source}: run manifest is missing '{field}'")

    required = ("method", "trial", "timing", "counts", "prompt")
    points = declared_points(manifest)
    for index, trial in enumerate(trials):
        for field in required:
            if field not in trial:
                raise ReportError(f"{source}: trial {index} is missing '{field}'")
        if trial.get("config_fingerprint") != manifest["config_fingerprint"]:
            raise ReportError(
                f"{source}: trial {index} belongs to a different configuration. "
                "Refusing to mix records from separate runs."
            )
        if trial["timing"].get("e2e_ms") is None:
            raise ReportError(f"{source}: trial {index} has no e2e_ms measurement")
        # A trial that ran a point the configuration does not contain cannot be traced to
        # it, and dropping it silently would change the reported result.
        generation = trial.get("generation") or {}
        length = generation.get("max_new_tokens")
        prompt_id = trial["prompt"].get("prompt_id")
        if length is None:
            raise ReportError(
                f"{source}: trial {index} does not record the generation length it ran at"
            )
        if (prompt_id, length) not in points and (prompt_id, None) not in points:
            raise ReportError(
                f"{source}: trial {index} ran '{prompt_id}' at {length} tokens, which is not "
                "a point of this run's configuration. Refusing to report it as part of this run."
            )
        if not trial["counts"].get("target_forwards"):
            warnings.append(
                f"trial {index} ({trial['method']}) recorded 0 target forward passes: "
                "tokens-per-forward is not usable for this run."
            )

    # Declared scope against what is actually in the file. `--trials` is a command line
    # override and is deliberately not part of the configuration fingerprint, so a run that
    # was interrupted and resumed - the normal case on a Colab session - legitimately
    # contains more trials than the manifest's first invocation declared. Surfacing the
    # difference is the point; it must not be discovered by counting rows by hand.
    # Arms that stop at different lengths cannot be identical, and a length difference looks
    # exactly like a content difference in an `exact_match: False`. Surfaced separately so
    # the two are never confused: one is a truncated pair, the other is a losslessness bug.
    paired: dict[tuple, dict[str, int | None]] = {}
    for trial in trials:
        generation = trial.get("generation") or {}
        key = (
            trial["prompt"].get("prompt_id"),
            generation.get("max_new_tokens"),
            trial["trial"],
        )
        paired.setdefault(key, {})[trial["method"]] = trial["counts"].get("tokens_generated")
    for key, arms in sorted(paired.items(), key=lambda item: str(item[0])):
        counts = {count for count in arms.values() if count is not None}
        if len(arms) > 1 and len(counts) > 1:
            detail = ", ".join(f"{method} {count}" for method, count in sorted(arms.items()))
            warnings.append(
                f"'{key[0]}' at {key[1]} tokens, trial {key[2]}: the arms generated different "
                f"lengths ({detail}), so they cannot be identical past the shorter one. Check "
                "hit_eos before reading that mismatch as a losslessness failure."
            )

    # Per point, not per file: a sweep contributes one trial set per (prompt, length), so
    # the declared count is compared against what each method produced for one point.
    declared = (manifest.get("run_scope") or {}).get("measured_trials")
    if declared is not None:
        point_count = max(1, len(points))
        per_method: dict[str, int] = {}
        for trial in trials:
            per_method[trial["method"]] = per_method.get(trial["method"], 0) + 1
        for method, count in sorted(per_method.items()):
            if count != declared * point_count:
                scope_note = (
                    f" across {point_count} points ({declared} per point)"
                    if point_count > 1
                    else ""
                )
                warnings.append(
                    f"manifest declares {declared} measured trial(s) but {count} '{method}' "
                    f"trials are present{scope_note}. Expected if the run was resumed with a "
                    "different --trials value; the extra trials are included in the summary."
                )

    if manifest["hardware"].get("gpu_name") is None:
        warnings.append(
            "this run recorded no GPU name, so it ran on CPU. The numbers are a "
            "correctness check, not a performance measurement."
        )
    elif not manifest["hardware"].get("gpu_driver_version"):
        warnings.append(
            "no GPU driver version was recorded (nvidia-smi was unavailable), so this "
            "result cannot be tied to a specific driver."
        )
    if manifest.get("config", {}).get("smoke"):
        warnings.append(
            "this run is marked as a smoke run in its configuration; its numbers are "
            "not benchmark results."
        )
    return warnings


def build_rows(manifest: dict, trials: list[dict]) -> list[dict]:
    """Flatten raw records into report rows, keeping the token ids for comparison."""
    config = manifest.get("config", {})
    resolved = manifest.get("resolved", {})
    hardware = manifest.get("hardware", {})

    rows: list[dict] = []
    for trial in trials:
        timing = trial["timing"]
        counts = trial["counts"]
        derived = trial.get("derived", {})
        memory = trial.get("memory", {})
        prompt = trial["prompt"]
        output = trial.get("output", {})
        generation = trial.get("generation") or {}

        rows.append(
            {
                "run_id": trial["run_id"],
                "config_fingerprint": trial["config_fingerprint"],
                "config_name": config.get("name"),
                "smoke": bool(config.get("smoke", False)),
                "device": resolved.get("device"),
                "gpu_name": hardware.get("gpu_name"),
                "dtype": resolved.get("dtype"),
                "attn_implementation": resolved.get("attn_implementation"),
                "prompt_id": prompt.get("prompt_id"),
                "stratum": prompt.get("stratum"),
                "prompt_tokens": prompt.get("prompt_tokens"),
                "max_new_tokens": generation.get("max_new_tokens"),
                "method": trial["method"],
                "trial": trial["trial"],
                "target_forwards": counts.get("target_forwards"),
                "assistant_forwards": counts.get("assistant_forwards"),
                "tokens_generated": counts.get("tokens_generated"),
                "prefill_ms": timing.get("prefill_ms"),
                "e2e_ms": timing.get("e2e_ms"),
                "decode_ms": timing.get("decode_ms"),
                "decode_ms_per_token": timing.get("decode_ms_per_token"),
                "tokens_per_second": derived.get("tokens_per_second"),
                "decode_tokens_per_second": derived.get("decode_tokens_per_second"),
                "peak_alloc_mb": memory.get("peak_alloc_mb"),
                "peak_reserved_mb": memory.get("peak_reserved_mb"),
                "hit_eos": output.get("hit_eos"),
                "accepted_len_mean": derived.get("accepted_len_mean"),
                "draft_tokens_per_step": derived.get("draft_tokens_per_step"),
                # Filled in by score_rows():
                "baseline_e2e_ms": None,
                "speedup_e2e": None,
                "speedup_decode": None,
                "exact_match": None,
                "first_divergence_index": None,
                # Internal, deliberately not part of CSV_COLUMNS:
                "_generated_token_ids": output.get("generated_token_ids") or [],
            }
        )
    return rows


def score_rows(rows: list[dict]) -> list[dict]:
    """Pair each row with the baseline trial of the same prompt, length and trial index.

    Pairing this way is what makes the comparison fair: both arms saw the same input, at
    the same generation length, in the same process, with the same dtype, moments apart.
    Pooled averages across prompts would hide exactly the prompt-to-prompt variation this
    study is about - and pairing across lengths would compare a 32-token run against a
    256-token one, which is not a comparison at all.
    """
    baselines = {
        (row["prompt_id"], row["max_new_tokens"], row["trial"]): row
        for row in rows
        if row["method"] == "baseline"
    }

    for row in rows:
        baseline = baselines.get((row["prompt_id"], row["max_new_tokens"], row["trial"]))
        if baseline is None or row["method"] == "baseline":
            continue
        row["baseline_e2e_ms"] = baseline["e2e_ms"]
        if row["e2e_ms"]:
            row["speedup_e2e"] = baseline["e2e_ms"] / row["e2e_ms"]
        if row["decode_ms_per_token"] and baseline["decode_ms_per_token"]:
            row["speedup_decode"] = baseline["decode_ms_per_token"] / row["decode_ms_per_token"]
        left = baseline["_generated_token_ids"]
        right = row["_generated_token_ids"]
        row["exact_match"] = exact_match(left, right)
        row["first_divergence_index"] = first_divergence_index(left, right)
    return rows


def rows_to_frame(rows: list[dict]) -> pd.DataFrame:
    """A DataFrame with exactly the published columns."""
    if not rows:
        raise ReportError("there are no rows to tabulate")
    return pd.DataFrame([{column: row.get(column) for column in CSV_COLUMNS} for row in rows])


def summarise(rows: list[dict]) -> list[dict]:
    """Median and interquartile range of the speedups, per method per generation length.

    A run whose rows all share one generation length produces one entry per method, which
    is what a single-point run reports. A sweep produces one entry per (method, length):
    pooling the lengths would average a 32-token run into a 256-token one and hide the
    length dependence the sweep exists to measure.
    """
    methods = sorted({row["method"] for row in rows if row["method"] != "baseline"})
    lengths = sorted(
        {row["max_new_tokens"] for row in rows if row["max_new_tokens"] is not None}
    )
    groups: list[tuple[str, int | None]] = (
        [(method, None) for method in methods]
        if len(lengths) <= 1
        else [(method, length) for method in methods for length in lengths]
    )

    summary: list[dict] = []
    for method, length in groups:
        selected = [
            row
            for row in rows
            if row["method"] == method
            and (length is None or row["max_new_tokens"] == length)
        ]
        speedups = [row["speedup_e2e"] for row in selected if row["speedup_e2e"] is not None]
        if not speedups:
            continue

        decode_speedups = [
            row["speedup_decode"] for row in selected if row["speedup_decode"] is not None
        ]
        exact_values = [row["exact_match"] for row in selected]
        accepted = [
            row["accepted_len_mean"]
            for row in selected
            if row["accepted_len_mean"] is not None
        ]

        summary.append(
            {
                "method": method,
                "max_new_tokens": length,
                "trials": len(speedups),
                "median_speedup_e2e": median(speedups),
                "iqr_speedup_e2e": interquartile_range(speedups),
                "median_speedup_decode": median(decode_speedups) if decode_speedups else None,
                "median_accepted_len": median(accepted) if accepted else None,
                "exact_match": all(exact_values) if exact_values else None,
            }
        )
    return summary


def write_csv(rows: list[dict], path: Path) -> Path:
    """Write the derived table. This file is always regenerated, never hand-edited."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_to_frame(rows).to_csv(path, index=False)
    return path


def summary_label(entry: dict) -> str:
    """How one summary entry is named in the console output and on the figure.

    One row per method, or per (method, generation length) when the run swept lengths.
    """
    length = entry.get("max_new_tokens")
    return entry["method"] if length is None else f"{entry['method']} @ {length}"


def calibration_path_for(config: RunConfig) -> Path:
    """Where this configuration's Stage 0 calibration lives.

    Beside the results, not inside them: a calibration is not evidence about a run, it is
    a separate measurement of the machine that the run's cost model is scored against.
    The filename carries the same run id, so a calibration can never be applied to a
    different configuration by accident - and `calibration_is_usable` checks the rest.
    """
    return config.raw_dir.parent / "calibration" / f"{run_id(config)}-calibration.json"


def profiling_path_for(config: RunConfig) -> Path:
    """Where this configuration's Stage 4 trace summary lives."""
    return config.raw_dir.parent / "profiling" / f"{run_id(config)}-profiling.json"


def read_json_artifact(path: Path) -> dict | None:
    """Read an artifact, or return None when it has not been produced yet."""
    import json

    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ReportError(f"{path} is not valid JSON: {error}") from error


def write_json_artifact(payload: dict, path: Path) -> Path:
    """Write an artifact, creating its directory. Never overwrites raw evidence.

    The guard is deliberate: these artifacts live under results/, and results/raw/ is the
    only directory in this project that is allowed to accumulate run records. Writing
    here cannot touch a raw JSONL, and the check makes that a matter of code rather than
    of remembering.
    """
    import json

    resolved = path.resolve()
    raw_dir = (path.parent.parent / "raw").resolve()
    if resolved.parent == raw_dir:
        raise ReportError(f"refusing to write an analysis artifact into {raw_dir}: that is raw evidence")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path


def calibration_score(rows: list[dict], calibration: dict, manifest: dict) -> dict:
    """Predicted versus observed speedup, per generation length, with signed residuals.

    This is the scoring the plan asks for, and the direction of the dependency matters:
    `rho` and `kappa` come from Stage 0, which timed the primitives *outside* the
    generation loop, and `b` and the observed speedup come from the sweep. Neither side
    is fitted to the other, which is exactly what was missing before Stage 0 ran.

    Two predictions are reported. The first uses the measured verification ratio; the
    second sets it to 1, which is the plan's ideal-verification simplification. Printing
    both makes the cost of assuming rho = 1 visible instead of hiding it in a footnote.
    """
    from .calibrate import calibration_is_usable
    from .metrics import theoretical_speedup

    differences = calibration_is_usable(calibration, manifest)
    ratios = calibration.get("ratios") or {}
    rho = ratios.get("rho")
    kappa = ratios.get("kappa")
    if rho is None or kappa is None:
        differences.append("the calibration artifact has no rho/kappa")

    if differences:
        return {"usable": False, "differences": differences, "rows": []}

    scored: list[dict] = []
    for entry in summarise(rows):
        if entry["method"] == "baseline" or entry["median_accepted_len"] is None:
            continue
        accepted = entry["median_accepted_len"]
        predicted = theoretical_speedup(accepted, float(rho), float(kappa))
        predicted_ideal = theoretical_speedup(accepted, 1.0, float(kappa))
        observed = entry["median_speedup_e2e"]
        scored.append(
            {
                "method": entry["method"],
                "max_new_tokens": entry.get("max_new_tokens"),
                "n": entry["trials"],
                "accepted_len": accepted,
                "predicted": predicted,
                "predicted_ideal_verification": predicted_ideal,
                "observed": observed,
                "residual": observed - predicted,
            }
        )
    return {
        "usable": True,
        "differences": [],
        "rho": float(rho),
        "kappa": float(kappa),
        "rows": scored,
    }


def plot_speedup(rows: list[dict], manifest: dict, path: Path) -> Path:
    """One figure: end-to-end speedup against baseline, with spread across trials.

    The title states the hardware and whether the run is a smoke run, because a figure
    that travels without those labels is a figure that misleads.
    """
    import matplotlib

    matplotlib.use("Agg")  # no display on a server, in CI or in a notebook kernel
    import matplotlib.pyplot as plt

    summary = summarise(rows)
    if not summary:
        raise ReportError("no speedup could be computed: the assisted arm found no baseline to pair with")

    names = [summary_label(entry) for entry in summary]
    medians = [entry["median_speedup_e2e"] for entry in summary]
    spreads = [entry["iqr_speedup_e2e"] for entry in summary]

    figure, axes = plt.subplots(figsize=(6.0, 3.6))
    axes.barh(names, medians, xerr=spreads, color="#3b6ea5", ecolor="#1f3d5c", capsize=4)
    axes.axvline(1.0, linestyle="--", linewidth=1.0, color="#b03030")
    axes.set_xlabel("median end-to-end speedup vs baseline (higher is faster)")
    axes.text(1.0, -0.45, "1.0 = no change", color="#b03030", fontsize=8, ha="center")

    hardware = manifest.get("hardware", {}).get("gpu_name") or "CPU (no GPU)"
    resolved = manifest.get("resolved", {})
    run_scope = manifest.get("run_scope", {})
    title = f"{hardware}, {resolved.get('dtype')}, attention={resolved.get('attn_implementation')}"
    if manifest.get("config", {}).get("smoke"):
        title += "  [SMOKE RUN: not a benchmark result]"
    axes.set_title(title, fontsize=9)

    if run_scope.get("prompt_ids"):
        lengths = run_scope.get("max_new_tokens") or []
        caption = (
            f"{len(run_scope['prompt_ids'])} prompts x {len(lengths)} generation lengths, "
            f"{len(rows)} measured trials, error bars = interquartile range"
        )
    else:
        caption = (
            f"one prompt ({run_scope.get('prompt_id', '?')}), {len(rows)} measured trials, "
            "error bars = interquartile range"
        )
    figure.text(0.01, 0.01, caption, fontsize=7, color="#555555")

    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def per_point_medians(rows: list[dict]) -> list[dict]:
    """One assisted point per (prompt, generation length): median accepted length and
    median speedup over that point's measured trials.

    Trials, not lengths, are the replicate here: averaging the three trials of a point
    first is what makes the 30 points comparable, and it is the same aggregation the
    summary uses.
    """
    points: dict[tuple, dict] = {}
    for row in rows:
        if row["method"] == "baseline" or row.get("speedup_e2e") is None:
            continue
        key = (row["prompt_id"], row.get("max_new_tokens"))
        points.setdefault(key, {"accepted": [], "speedup": []})
        if row.get("accepted_len_mean") is not None:
            points[key]["accepted"].append(row["accepted_len_mean"])
        points[key]["speedup"].append(row["speedup_e2e"])

    collected = []
    for (prompt_id, length), values in points.items():
        if not values["accepted"] or not values["speedup"]:
            continue
        collected.append(
            {
                "prompt_id": prompt_id,
                "max_new_tokens": length,
                "accepted_len": median(values["accepted"]),
                "speedup": median(values["speedup"]),
                "trials": len(values["speedup"]),
            }
        )
    return collected


def plot_accepted_len_vs_speedup(
    rows: list[dict], manifest: dict, path: Path, calibration: dict | None = None
) -> Path | None:
    """Figure 2 of the plan: does accepted length explain the speedup, and how far is it
    from what the cost model says it should be?

    The scatter is measured. The two straight lines are the cost model `b / (rho + kappa)`
    and its ideal-verification simplification `b / (1 + kappa)`, and they are drawn *only*
    when Stage 0 has measured rho and kappa. Without a calibration there is no line, and
    the caption says so, because a curve drawn from the sweep's own residual would turn
    the figure into a restatement of the data it is supposed to test.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = per_point_medians(rows)
    if not points:
        return None

    lengths = sorted({point["max_new_tokens"] for point in points}, key=lambda v: (v is None, v))
    colours = {length: colour for length, colour in zip(lengths, ("#3b6ea5", "#c47f2a", "#4a8f5b", "#8b5ba8"))}

    figure, axes = plt.subplots(figsize=(6.0, 3.8))
    for length in lengths:
        subset = [point for point in points if point["max_new_tokens"] == length]
        axes.scatter(
            [point["accepted_len"] for point in subset],
            [point["speedup"] for point in subset],
            s=26,
            color=colours.get(length, "#555555"),
            label=f"{length} new tokens",
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )

    model_note = "cost model not drawn: run `tinyturbo calibrate` (Stage 0)"
    if calibration and calibration.get("ratios"):
        ratios = calibration["ratios"]
        rho = float(ratios["rho"])
        kappa = float(ratios["kappa"])
        highest = max(point["accepted_len"] for point in points)
        span = [0.0, highest * 1.05]
        axes.plot(span, [b / (rho + kappa) for b in span], color="#b03030", linewidth=1.2,
                  label=f"model b/(rho+kappa), rho={rho:.2f}, kappa={kappa:.1f}")
        axes.plot(span, [b / (1.0 + kappa) for b in span], color="#b03030", linewidth=0.9,
                  linestyle=":", label="ideal verification, b/(1+kappa)")
        model_note = (
            f"lines: rho and kappa measured in Stage 0 on this hardware, not fitted to these points"
        )

    axes.set_xlabel("median tokens generated per target forward pass (accepted length)")
    axes.set_ylabel("median end-to-end speedup")
    axes.axhline(1.0, linestyle="--", linewidth=1.0, color="#888888")

    hardware = manifest.get("hardware", {}).get("gpu_name") or "CPU (no GPU)"
    resolved = manifest.get("resolved", {})
    title = f"{hardware}, {resolved.get('dtype')} - one point per prompt x length"
    if manifest.get("config", {}).get("smoke"):
        title += "  [SMOKE RUN: not a benchmark result]"
    axes.set_title(title, fontsize=9)
    axes.legend(fontsize=7, loc="upper left")
    figure.text(0.01, 0.01, model_note, fontsize=7, color="#555555")

    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def time_attribution(rows: list[dict], calibration: dict) -> list[dict]:
    """Split each arm's measured time per generated token into measured components.

    Every component is a *measured quantity times a recorded count*, never a share
    apportioned by assumption:

        prefill        recorded prefill_ms / recorded tokens
        target passes  recorded target_forwards x measured tau_t / tokens
        draft passes   recorded assistant_forwards x measured tau_d / tokens

    Whatever the measured end-to-end time has left over is one honest residual column -
    the Python between forward passes, kernel launches and framework bookkeeping - which
    this module deliberately does not divide further. Only a trace can do that, which is
    what Stage 4 is for. The residual can come out slightly negative, because a single
    measured tau_t and tau_d need not reproduce the in-loop average exactly, and that is
    reported rather than clipped.
    """
    primitives = calibration.get("primitives") or {}
    tau_t = (primitives.get("target_single") or {}).get("median_ms")
    tau_d = (primitives.get("assistant_single") or {}).get("median_ms")
    if tau_t is None or tau_d is None:
        return []

    per_length: dict = {}
    for row in rows:
        key = (row["method"], row.get("max_new_tokens"))
        per_length.setdefault(key, []).append(row)

    aggregates: dict = {}
    for (method, length), group in per_length.items():
        tokens = median([row["tokens_generated"] for row in group])
        if not tokens:
            continue
        e2e = median([row["e2e_ms"] for row in group])
        per_token = e2e / tokens
        aggregates[(method, length)] = {
            "method": method,
            "max_new_tokens": length,
            "trials": len(group),
            "tokens": tokens,
            "ms_per_token": per_token,
            "components": {
                "prefill": median([row["prefill_ms"] for row in group]) / tokens,
                "target": median([row["target_forwards"] for row in group]) * tau_t / tokens,
                "draft": median([row["assistant_forwards"] for row in group]) * tau_d / tokens,
            },
        }

    # Everything is expressed relative to the baseline's own measured cost per token, so
    # one unit on the axis is "as expensive as plain greedy was at this length".
    output = []
    for key, entry in sorted(aggregates.items(), key=lambda item: (item[0][1] or 0, item[0][0])):
        baseline = aggregates.get(("baseline", entry["max_new_tokens"]))
        reference = baseline["ms_per_token"] if baseline else entry["ms_per_token"]
        if not reference:
            continue
        components = entry["components"]
        attributed = sum(components.values())
        output.append(
            {
                **{k: v for k, v in entry.items() if k != "components"},
                "components_ms_per_token": components,
                "relative_ms_per_token": entry["ms_per_token"] / reference,
                "relative_components": {name: value / reference for name, value in components.items()},
                "relative_unattributed": (entry["ms_per_token"] - attributed) / reference,
                "tau_t_ms": tau_t,
                "tau_d_ms": tau_d,
            }
        )
    return output


def plot_time_attribution(
    rows: list[dict], manifest: dict, path: Path, calibration: dict
) -> Path | None:
    """Figure 3 of the plan: where the per-step time goes, from measured components.

    Two stacked bars per generation length, baseline and assisted, normalised so the
    baseline is 1.0. Requires Stage 0: without measured tau_t and tau_d there is nothing
    to stack except the total, which is already figure 1.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    attribution = time_attribution(rows, calibration)
    if not attribution:
        return None

    lengths = sorted({entry["max_new_tokens"] for entry in attribution}, key=lambda v: (v is None, v))
    names: list[str] = []
    series = {
        "prefill (recorded)": [],
        "target passes: count x measured tau_t": [],
        "draft passes: count x measured tau_d": [],
        "unattributed: loop + framework overhead": [],
    }
    for length in lengths:
        for method in ("baseline", "assisted"):
            entry = next(
                (
                    item
                    for item in attribution
                    if item["method"] == method and item["max_new_tokens"] == length
                ),
                None,
            )
            if entry is None:
                continue
            names.append(f"{method}\n@ {length}")
            components = entry["relative_components"]
            series["prefill (recorded)"].append(components["prefill"])
            series["target passes: count x measured tau_t"].append(components["target"])
            series["draft passes: count x measured tau_d"].append(components["draft"])
            series["unattributed: loop + framework overhead"].append(entry["relative_unattributed"])

    figure, axes = plt.subplots(figsize=(6.6, 4.0))
    colours = ["#9dbcd8", "#3b6ea5", "#c47f2a", "#b9b9b9"]
    bottom = [0.0] * len(names)
    for (label, values), colour in zip(series.items(), colours):
        axes.bar(names, values, bottom=bottom, label=label, color=colour, width=0.6)
        bottom = [base + value for base, value in zip(bottom, values)]

    axes.axhline(1.0, linestyle="--", linewidth=1.0, color="#b03030")
    axes.set_ylabel("time per generated token,\nrelative to the baseline (1.0 = plain greedy)")

    tau_t = attribution[0]["tau_t_ms"]
    tau_d = attribution[0]["tau_d_ms"]
    hardware = manifest.get("hardware", {}).get("gpu_name") or "CPU (no GPU)"
    resolved = manifest.get("resolved", {})
    title = f"{hardware}, {resolved.get('dtype')} - measured tau_t={tau_t:.2f} ms, tau_d={tau_d:.2f} ms"
    if manifest.get("config", {}).get("smoke"):
        title += "  [SMOKE: not a benchmark result]"
    axes.set_title(title, fontsize=9)
    axes.legend(fontsize=7, loc="upper left")
    figure.text(
        0.01,
        0.01,
        "components are recorded counts multiplied by Stage 0's measured latencies; "
        "the grey band is whatever the measured end-to-end time has left over",
        fontsize=7,
        color="#555555",
    )

    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def build_report(config: RunConfig, verbose: bool = True) -> dict:
    """Full pipeline: raw records -> validated rows -> CSV -> figure."""
    source = raw_path_for(config)
    manifest, trials = read_run(source)
    warnings = validate_run(manifest, trials, source)
    rows = score_rows(build_rows(manifest, trials))
    csv_path = write_csv(rows, config.results_csv)

    summary = summarise(rows)
    # No comparison means no figure. A baseline-only run still gets its table, but no
    # chart is drawn that could be read as showing an effect.
    plot_path = None
    accepted_plot = None
    attribution_plot = None
    score = None

    # A calibration is only allowed to score this run when it describes the same machine,
    # configuration and lookahead. A CPU calibration applied to a T4 sweep would produce a
    # table that looks like evidence and is not, so it is rejected and said out loud.
    calibration = read_json_artifact(calibration_path_for(config))
    calibration_differences: list[str] = []
    if calibration is not None:
        from .calibrate import calibration_is_usable

        calibration_differences = calibration_is_usable(calibration, manifest)
    usable_calibration = (
        calibration if calibration is not None and not calibration_differences else None
    )

    if summary:
        plot_path = plot_speedup(
            rows, manifest, config.plots_dir / f"speedup_{manifest['run_id']}.png"
        )
        accepted_plot = plot_accepted_len_vs_speedup(
            rows,
            manifest,
            config.plots_dir / f"accepted_len_vs_speedup_{manifest['run_id']}.png",
            usable_calibration,
        )
        if usable_calibration is not None:
            score = calibration_score(rows, usable_calibration, manifest)
            attribution_plot = plot_time_attribution(
                rows,
                manifest,
                config.plots_dir / f"time_attribution_{manifest['run_id']}.png",
                usable_calibration,
            )
    else:
        warnings.append(
            "no non-baseline trials were found, so no figure was produced; the CSV was "
            "still written"
        )

    if calibration is None:
        warnings.append(
            "no Stage 0 calibration found at "
            f"{calibration_path_for(config)}: figure 2 is drawn without the cost model "
            "and figure 3 and the predicted-vs-observed table need one. Run "
            "`tinyturbo calibrate --config "
            f"{config.as_dict().get('name')}` on the run's hardware to produce it."
        )
    elif calibration_differences:
        warnings.append(
            "a Stage 0 calibration exists but does not describe this run, so it was not "
            "used: " + "; ".join(calibration_differences)
        )

    if verbose:
        print(f"raw records : {source}")
        print(f"manifest    : {manifest['run_id']} ({manifest['created_utc']})")
        hardware = manifest.get("hardware", {})
        print(f"hardware    : {hardware.get('gpu_name') or 'CPU'} / {manifest['resolved']['dtype']}")
        print(f"csv         : {csv_path}")
        print(f"plot        : {plot_path if plot_path else 'none produced'}")
        if accepted_plot:
            print(f"plot        : {accepted_plot}")
        if attribution_plot:
            print(f"plot        : {attribution_plot}")
        print()
        width = max(9, max((len(summary_label(entry)) for entry in summary), default=9))
        for entry in summary:
            parts = [
                f"{summary_label(entry):<{width}}",
                f"n={entry['trials']:<3}",
                f"speedup_e2e={entry['median_speedup_e2e']:.3f}",
                f"(iqr {entry['iqr_speedup_e2e']:.3f})",
            ]
            if entry["median_speedup_decode"] is not None:
                parts.append(f"speedup_decode={entry['median_speedup_decode']:.3f}")
            print("  ".join(parts))
            if entry["median_accepted_len"] is not None:
                print(
                    f"{'':<{width}} tokens per target forward pass: "
                    f"{entry['median_accepted_len']:.2f}"
                )
            print(f"{'':<{width}} output identical to baseline: {entry['exact_match']}")

        # The predicted-vs-observed comparison is printed with the residuals, not just
        # summarised, because the size of a residual is the finding H3 is about.
        if score and score["usable"] and score["rows"]:
            print()
            print(
                f"Stage 0 cost model, measured on this hardware: "
                f"rho={score['rho']:.3f} kappa={score['kappa']:.3f} "
                f"(rho = verification pass / single target step, kappa = gamma * tau_d / tau_t)"
            )
            print(f"{'length':>7}  {'b':>5}  {'predicted':>9}  {'observed':>8}  {'residual':>8}")
            for row in score["rows"]:
                length = row["max_new_tokens"]
                print(
                    f"{str(length):>7}  {row['accepted_len']:>5.2f}  {row['predicted']:>9.3f}  "
                    f"{row['observed']:>8.3f}  {row['residual']:>+8.3f}"
                )
        if warnings:
            print()
            for warning in warnings:
                print(f"WARNING: {warning}")

    return {
        "raw": source,
        "csv": csv_path,
        "plot": plot_path,
        "plot_accepted_len_vs_speedup": accepted_plot,
        "plot_time_attribution": attribution_plot,
        "calibration": calibration,
        "calibration_differences": calibration_differences,
        "calibration_score": score,
        "warnings": warnings,
        "summary": summary,
    }
