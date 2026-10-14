# Colab session artefacts — `gpu-main-9e26c12945e6`

Verbatim copies of what a real Colab session downloaded, kept so the numbers in the
derived CSV can be traced back to the session that produced them. Nothing in this
directory is edited by hand and nothing in it is a source of truth: the evidence is
`results/raw/gpu-main-9e26c12945e6.jsonl`, and the canonical figure is
`plots/speedup_gpu-main-9e26c12945e6.png`.

This is the **Stage 1 sweep** of `docs/PLAN.md` — the first run in this repository that
covers more than one prompt at more than one generation length.

| | |
|---|---|
| run_id | `gpu-main-9e26c12945e6` |
| config | `configs/gpu_main.yaml` (fingerprint `9e26c12945e6…`, recomputed from this checkout and matching) |
| command | `python -m tinyturbo run --config configs/gpu_main.yaml` |
| scope | 10 prompts × 3 generation lengths × 2 methods = 30 points, 60 trial sets |
| trials | 2 warm-ups + **3 measured** per trial set (the configured value, no `--trials` override) |
| records | 301 JSONL lines: 1 manifest + 300 trials (180 measured, 120 warm-up) |
| hardware | Tesla T4, compute capability 7.5, 14912 MiB, driver 580.82.07, CUDA 13.0, cudnn 92700 |
| software | Linux 6.6.122+, Python 3.13.15, torch 2.11.0+cu130, transformers 5.3.0, 2 vCPUs |
| resolved | device `cuda`, dtype `float16`, attention `sdpa`, γ = 5, confidence threshold `0.0` |
| models | `SmolLM-1.7B` (1 711 376 384 params) target, `SmolLM-135M` (134 515 008) assistant, ratio 0.0786 |
| created (UTC) | `2026-10-14T14:24:50+00:00`, first trial `14:24:51`, last trial `14:49:04` — **24 min 13 s** |
| session | `notebooks/colab_gpu_smoke.ipynb` cells 1–5, 7, 12–13, uploading `TinyTurbo.zip` |

The manifest records `"git_available": false`: the bundle is a zip of the working tree and
carries no `.git`, so the code identity for this run is the config fingerprint plus the
zipped tree, not a commit hash. The notebook recorded the same for the slice run.

## Files

| file | what it is | sha256 |
|---|---|---|
| `benchmark_results.csv` | the derived table as Colab produced it | `621ae5e67000f77e…` |
| `environment_fingerprint.json` | standalone hardware/scope fingerprint (the notebook writes it separately because it is not reproducible after the session ends) | `3b3247ae9af0dfd8…` |
| `verify_output.txt` | stdout of `tinyturbo verify`, captured before any number was read | `d412112d61722027…` |
| `figure.png` | the figure exactly as Colab's matplotlib encoded it | `d06aee9e635301e7…` |

The raw records are filed where the tooling expects them:
`results/raw/gpu-main-9e26c12945e6.jsonl` (sha256 `e146f85b67b07afb…`).

The figure exists in two places on purpose. `plots/speedup_gpu-main-9e26c12945e6.png`
(sha256 `f6f744a2f2a41ced…`, 37,755 bytes) is the file `tinyturbo report` regenerates, and
regenerating it twice is byte-identical, so that copy is reproducible from the raw records.
`figure.png` here (46,532 bytes) is Colab's encoder output for the same plot: 0 of 486,000
pixels differ between the two, so they are the same image, but they are not the same bytes.
Filing both means the session's own bytes survive any later re-report, while the derived
copy stays regenerable - the same split the CSV already uses.

`benchmark_results.csv` is duplicated here rather than left at `results/benchmark_results.csv`
because that path is shared by every config. See `results/README.md`.

## Stage 0 calibration (a later session, same run_id)

The cost model in `docs/FINDINGS.md` §5 is scored from an artifact this sweep's own session
never produced. `python -m tinyturbo calibrate --config configs/gpu_main.yaml` ran on a T4 in
a **separate, later session the same day** - its `created_utc` is `2026-10-14T17:22:16+00:00`
against the sweep's 14:24:51-14:49:04 - and its artifact is filed at
`results/calibration/gpu-main-9e26c12945e6-calibration.json`.

| | |
|---|---|
| file | `results/calibration/gpu-main-9e26c12945e6-calibration.json` (4,033 bytes) |
| sha256 | `1f434066d9fa9881…` |
| protocol | 5 discarded warm-ups + 20 timed passes per primitive, fresh populated cache per sample, raw samples retained |
| prompt | `s1-01-short-open`, 8 tokens |
| measured | τ_t **22.431 ms** (IQR 0.583), τ_t^verify **25.737 ms** (IQR 1.149), τ_d **35.250 ms** (IQR 1.379) |
| derived | ρ = **1.147**, τ_d/τ_t = **1.572**, κ = **7.858** |
| guard | every field `calibration_is_usable()` compares - configuration fingerprint, device, dtype, attention, GPU name, driver, CUDA, torch, transformers, γ - matches this sweep's manifest, so `report` applies it without a warning |

Two things about it are worth recording precisely.

**It is a second measurement, not a rerun of the first.** A first `calibrate` invocation
earlier the same day printed τ_t 22.310 ms, τ_t^verify 25.579 ms, τ_d 34.280 ms, hence ρ = 1.147
and κ = 7.683. That session's artifact was overwritten by the one filed here, so those numbers
survive only in the session log and are **not** cited as findings anywhere in this repository.
They are noted here for one reason: they bound the run-to-run spread of Stage 0 at about 3% on
τ_d and κ, while ρ printed as 1.147 in both. ρ is the more stable of the two ratios.

**What it changed downstream.** Figure 2 gained its cost-model lines (sha256
`53f71644a9f59cce` → `7be8b57ebcf630ae`) and figure 3 came into existence
(`plots/time_attribution_gpu-main-9e26c12945e6.png`, `e09f5fb9b1d10d50`). Both regenerate
byte-identically on a second `report` run, and moving the artifact aside reverts figure 2 to
`53f71644a9f59cce` and writes no figure 3 at all - which is what proves the change is
attributable to the calibration rather than to the code. The raw records, the CSV and
`speedup_<run_id>.png` are untouched by it, and the filed `figure.png` above stays the
session's own bytes.

**Stage 4 did not produce an artifact in this session.** `tinyturbo profile` was attempted in
the same later session and was killed by the host's memory limit - `/bin/bash: Killed`, exit
**137** - after both models had loaded and generation had begun, with no exception raised.
That attempt was the whole-generation trace, and it is still the protocol that has never
completed. Nothing in `report` reads a trace either.

**A bounded trace has since produced one, on different hardware.** On a Kaggle T4,
`--trace-active 8` traced both cases and wrote
`results/profiling/gpu-main-9e26c12945e6-profiling.json` (6 187 bytes, sha256 `8884e21fa4987264…`,
2026-10-14T20:06:13Z). It is keyed to this run id and uses the same models, dtype and
attention backend, but it ran on driver **580.178.04** / CUDA **12.8** / torch
**2.11.0+cu128** rather than this session's 580.82.07 / 13.0 / cu130 - which is exactly the
kind of difference `report` refuses to paper over. Its per-pass means came out 1.6–2.0× the
Stage 0 primitives, by a roughly constant ≈20 ms per pass, so it does **not** decompose the
per-step residual this sweep left. `docs/FINDINGS.md` §9 records the numbers, the two
candidate explanations it cannot separate, and what would settle it (the same window on the
calibration's own image).

## What was checked when these were filed

Run locally against the filed raw JSONL, on the machine that wrote the notebook:

* **`tinyturbo verify` exits 0** and reproduces `verify_output.txt` line for line, apart
  from the `raw records :` line, which names where the file is being read from.
* **The raw file's config fingerprint matches `configs/gpu_main.yaml` in this checkout**
  (`9e26c12945e6…`, recomputed with `config_fingerprint()`), so the records were produced
  by this configuration and not by a stale copy inside the bundle.
* **Every field of the CSV regenerates identically** from the raw records — 180 rows × 33
  columns, all 29 non-key fields compared row by row after joining on
  `(prompt_id, max_new_tokens, method, trial)`, with zero differences and the same row
  order. The byte difference is 181 bytes and is exactly the line endings: Colab's pandas
  writes LF, Windows pandas writes CRLF (181 lines, so 181 `\r\n`). The two files are
  byte-identical after newline normalisation.
* **The figure regenerates pixel-identically** — 0 of 486,000 pixels differ at 900×540
  between Colab's `figure.png` and the derived file in `plots/`. The byte sizes differ
  (37,755 local vs 46,532 from Colab), which is encoder variation rather than a different
  plot, and regenerating the same config twice locally is byte-identical - so the derived
  figure is both reproducible and pixel-equal to the one the session downloaded.

## What this run measures

The real model pair at float16 on a T4, every prompt at every generation length, three
measured trials per point. This is the first result here with a distribution behind it
rather than a single prompt.

| generation length | n | median e2e speedup | IQR | decode-only | tokens per target forward pass |
|---|---|---|---|---|---|
| 32 | 30 | **0.452** | 0.169 | 0.443 | 3.56 |
| 128 | 30 | **0.528** | 0.095 | 0.525 | 4.66 |
| 256 | 30 | **0.592** | 0.066 | 0.591 | 5.17 |
| all | 90 | **0.528** | 0.171 | 0.525 | — |

**Assisted decoding is slower than plain greedy at every one of the 30 points** — not one
point reached parity, and the per-point median speedup spans 0.303 (`s2-01-long-open` @ 32)
to 0.659 (`s3-01-grounded` @ 128). Pooled, assisted was ~1.9× slower.

Two things move in the direction the mechanism predicts:

* **Speedup rises with generation length** (0.452 → 0.592) and so does accepted length
  (3.56 → 5.17), because a longer continuation amortises the one-time prefill and gives the
  drafter more chances to land a long accepted run.
* **Speedup rises with accepted length across prompts and strata.** Spearman rank
  correlation between `accepted_len_mean` and `speedup_e2e` is 0.98 pooled over the 90
  measured trials, and 0.91 / 0.97 / 0.97 within each length separately — H4's ≥ 0.8
  threshold is met in all four groupings. Prompt choice matters: grounded prompts
  (S3, median 0.618) and repetitive ones (S4, 0.605) accept more than open-ended short ones
  (S1, 0.430), which is the workload-dependence the plan expected, seen here for the one
  method family that was run.

**Exactness holds everywhere.** `exact_match` is `True` for all 90 assisted measured
trials, with `first_divergence_index` null in all of them: three generation lengths, ten
prompts and 90 trials produced token-identical output to plain greedy at the same dtype.
H6 survives its widest test so far. No trial hit EOS (`hit_eos: false` throughout), so every
comparison is over the full requested length.

**Memory is not the constraint.** Median `peak_alloc_mb` is 3617.5 (baseline) vs 3635.4
(assisted), and `peak_reserved_mb` is 3820.0 for both — the 135M assistant and its KV cache
add under 20 MiB on a 15 GiB card.

**Where the time went, as a hypothesis.** Taking the baseline's own measured cost per target
forward pass (22.9–23.4 ms, prefill subtracted) and removing the assisted arm's target passes
from its decode time leaves 36.0–37.3 ms per assistant forward pass. That is a 135M draft
pass costing **1.57–1.61×** a 1.7B target pass, against the 0.079 parameter ratio — roughly
20× more per parameter than the target. Feeding that single implied ratio back into the §5
model as κ = γ·τ_d/τ_t ≈ 7.9 (ρ = 1) predicts 0.401 / 0.515 / 0.585 against 0.452 / 0.528 /
0.592 observed, residuals +0.051 / +0.012 / +0.007.

**That arithmetic is not an independent test.** τ_d was inferred from these same
observations, so the agreement says the sweep is *consistent* with the cost model once the
draft pass is priced as measured — it does not validate the model, and with ρ = 1 assumed
there is no residual left to attribute to verification overhead. Stage 0 calibration and a
Stage 4 trace are what would test it. Until they run, the drafting-loop cost stays a
candidate explanation, not a measured one, and nothing in this repository is a profiler
result.

## What this run does not settle

* **Only two methods ran.** Baseline and model-assisted. `prompt_lookup`, the adaptive
  schedules and ensemble verification have their own configurations and were not run, so
  H1, H2 and H7 are untouched by this file. See `docs/PLAN.md` — two methods is the scope
  of this stage, not a placeholder for six.
* **Prefill dilution (H5) is barely exercised.** Prefill is a flat 40.8 ms, which is 5.0%,
  1.3% and 0.7% of a baseline trial at 32/128/256 tokens. The e2e and decode-only speedups
  therefore differ by at most 0.010, and dilution does shrink with generation length as
  predicted. But because the assisted arm is *slower* here, dilution pulls the ratio toward
  1.0, so it understates the slowdown rather than the speedup — the sign H5 states for a
  positive effect. And the S2 prompts, designed at 300–1000 tokens to make prefill bite,
  actually tokenize to 174–188, so this sweep cannot test the magnitude either way.
* **The trial-to-trial spread is not small.** The IQR at 32 tokens (0.169) is comparable to
  the entire gap between the 32- and 256-token medians (0.140), so three trials per point is
  enough for the direction at each length and thin for comparing adjacent points.
* **No sm80+ hardware**, so H8 remains untested rather than inferred.
