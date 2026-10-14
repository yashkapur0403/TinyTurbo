# Run log

Every committed raw record belongs here, mapped to the machine that produced it and the
command that produced it. Results from different machines are never pooled: a CPU smoke
run and a T4 run are not comparable, and averaging them would hide that.

The derived files (`results/benchmark_results.csv`, everything in `plots/`) are always
regenerated from these raw records with `python -m tinyturbo report --config <config>`,
never edited by hand.

What these records mean is written up in `docs/FINDINGS.md`: the findings, the distinction
between measured and estimated quantities, and the hypotheses that remain untested.

| run_id | config | hardware | dtype | command | status |
|---|---|---|---|---|---|
| `slice-cpu-smoke-8b0f61630f43` | `configs/slice_cpu.yaml` | CPU, no GPU (Windows, Python 3.14.6, torch 2.10.0+cpu, transformers 5.3.0) | float32 | `python -m tinyturbo run --config configs/slice_cpu.yaml` | smoke run, completed |
| `slice-gpu-902bc285880d` | `configs/slice_gpu.yaml` | Tesla T4, CC 7.5, 14912 MiB, driver 580.82.07 (Colab, Linux, Python 3.13.15, torch 2.11.0+cu130, transformers 5.3.0) | float16 | `python -m tinyturbo run --config configs/slice_gpu.yaml --trials 2` | GPU slice, completed — **one prompt, two measured trials per arm** |
| `gpu-main-9e26c12945e6` | `configs/gpu_main.yaml` | Tesla T4, CC 7.5, 14912 MiB, driver 580.82.07 (Colab, Linux, Python 3.13.15, torch 2.11.0+cu130, transformers 5.3.0) | float16 | `python -m tinyturbo run --config configs/gpu_main.yaml` | **Stage 1 sweep, completed — 10 prompts × 3 lengths, two methods, three measured trials per point** |
| `gpu-main-9e26c12945e6` (Stage 0) | `configs/gpu_main.yaml` | Tesla T4, CC 7.5, 14912 MiB, driver 580.82.07 (Colab, Linux, Python 3.13.15, torch 2.11.0+cu130, transformers 5.3.0) | float16 | `python -m tinyturbo calibrate --config configs/gpu_main.yaml` | Stage 0 calibration, completed in a later session — **ρ = 1.147, κ = 7.858**; artifact in `results/calibration/`, no raw records |
| `gpu-main-9e26c12945e6` (Stage 4) | `configs/gpu_main.yaml` | Tesla T4, CC 7.5, 14911 MiB, driver 580.178.04, CUDA 12.8 (Kaggle, Linux, Python 3.13.15, torch 2.11.0+cu128, transformers 5.3.0, tokenizers 0.22.2) | float16 | `python -m tinyturbo profile --config configs/gpu_main.yaml --trace-active 8` | Stage 4 **bounded trace**, completed — 2 cases × 2 arms, an 8-step window each; artifact in `results/profiling/`, no raw records. Its per-pass means are 1.6–2.0× the Stage 0 primitives, on a different driver/torch build |
| `profile-smoke-cab8be82e4cd` (Stage 4) | `configs/profile_smoke.yaml` | CPU, no GPU (Windows, Python 3.14.6, torch 2.10.0+cpu, transformers 5.3.0) | float32 | `python -m tinyturbo profile --config configs/profile_smoke.yaml --trace-active 4 --max-new-tokens 32` | Stage 4 **bounded trace**, smoke run — a plumbing check of the windowed path, not a measurement of the drafting loop; artifact in `results/profiling/`, no raw records |

Every raw record is one JSONL file: `results/raw/<run_id>.jsonl`. Those files are the only
evidence in this repository.

## What the CPU run is, and is not

`slice-cpu-smoke-8b0f61630f43` is a **plumbing check**, not a measurement:

* it ran on CPU, where the draft model is as expensive as the target model;
* the target and the assistant are the *same* checkpoint (`SmolLM-135M`), so the
  parameter ratio is 1.0 and there is no size asymmetry to exploit;
* it covers one prompt and two measured trials.

It exists to answer one question: does the instrumented pipeline produce real,
self-consistent records end to end? It does. The numbers in it must not be quoted as a
result about assisted decoding.

## What the smoke profiling artifact is, and is not

`results/profiling/profile-smoke-cab8be82e4cd-profiling.json` is the Stage 4 counterpart of
that CPU run, and has the same status. It is the artifact of

```bash
python -m tinyturbo profile --config configs/profile_smoke.yaml --trace-active 4 --max-new-tokens 32
```

| | |
|---|---|
| file | `results/profiling/profile-smoke-cab8be82e4cd-profiling.json` (3,772 bytes) |
| sha256 | `abd84d0685ff0682…` |
| what it is | a **bounded trace**: `torch.profiler.schedule` recording 4 steps after 1 skipped and 1 warmed up, one step per target forward pass |
| window recorded | `active` 4, `wait` 1, `warmup` 1, `repeat` 1 — in `settings.trace_window` |
| measured | baseline 3 063.4 ms and assisted 4 960.6 ms end to end, over 8-token prompts at 32 tokens; 4 labelled target passes and 4 labelled draft passes inside the window |
| not measured | the whole-run residual. Every method records `covers_whole_generation: false`, `attributed_ms: null`, `unattributed_ms: null`, and a `null` count for the whole-run `generate` annotation, which a schedule's cycle end clips |
| also recorded | `smoke: true`, and the resolved `cpu` / `float32` / `eager` it ran as |

It is filed for one reason: Stage 4 is the stage a free Colab session could not finish, and
this shows the bounded route running end to end and writing an artifact that describes what it
does and does not cover. The same flag on the real pair is what produced
`results/profiling/gpu-main-9e26c12945e6-profiling.json` on a Kaggle T4 - a window that fits,
and numbers that do not reconcile with Stage 0 (`docs/FINDINGS.md` §9).
Nothing in `report` reads either file, and no number in them is a finding.

The bound itself was checked separately on the same pair, by counting
`torch.profiler` events: a 4-step window holds **130 515** events on a 48-token assisted
generation and **130 515** again on a 96-token one, against **371 423** for the whole 48-token
trace. The buffer follows the window, not the run, which is the claim the route rests on.

## What the sweep is, and is not

`gpu-main-9e26c12945e6` is the one run here whose numbers are a **distribution rather than
an anecdote**: the real model pair at float16 on a T4, all ten prompts across all three
generation lengths, three measured trials per point behind two discarded warm-ups, run in a
single 24-minute Colab session.

It is the Stage 1 sweep of `docs/PLAN.md`, and it answers the two-arm question the plan
frames: does model-assisted greedy decoding beat plain greedy here, and does accepted
length explain by how much. It does not answer the method-comparison questions — one
alternative method family (a second draft model) is not a method study. `prompt_lookup`,
the adaptive lookahead schedules and ensemble verification have their own configurations
and their own stages and were not run. **Two methods is the scope of this stage, not a
placeholder for six.**

Headline numbers, all computed by `report` from the raw file:

| generation length | n | median e2e speedup | IQR | decode-only | tokens per target forward pass |
|---|---|---|---|---|---|
| 32 | 30 | **0.452** | 0.169 | 0.443 | 3.56 |
| 128 | 30 | **0.528** | 0.095 | 0.525 | 4.66 |
| 256 | 30 | **0.592** | 0.066 | 0.591 | 5.17 |
| all | 90 | **0.528** | 0.171 | 0.525 | — |

* **Assisted decoding was slower at every one of the 30 points** — best per-point median
  0.659 (`s3-01-grounded` @ 128), worst 0.303 (`s2-01-long-open` @ 32). Pooled, ~1.9× slower.
* **Exactness held everywhere**: `exact_match` is `True` for all 90 assisted measured trials,
  token-identical to the paired baseline at the same dtype. H6's widest test so far.
* **Accepted length tracks speedup**: Spearman 0.98 pooled, and 0.91 / 0.97 / 0.97 within the
  three lengths separately, clearing H4's ≥ 0.8 threshold in all four groupings.
* **The cost sits in the drafting loop.** Removing the assisted arm's target passes from its
  decode time leaves 36.0–37.3 ms per assistant forward pass against 22.9–23.4 ms per target
  pass: the 135M draft pass costs 1.57–1.61× the 1.7B target pass. This is arithmetic over
  recorded counts, and it is a hypothesis, not a profiler result — see the session README.

The fingerprint, the checksums, what was checked when these were filed, and what the run
does *not* settle are in `results/sessions/gpu-main-9e26c12945e6/README.md`.

## What the T4 slice is, and is not

`slice-gpu-902bc285880d` was the first run whose numbers were evidence about assisted
decoding, and it is a **smoke run**: one prompt out of ten (`s2-01-long-open`), two measured
trials per arm instead of the five the configuration asks for, at one generation length
(64). The sweep above now covers its prompt and length, with three trials and a ten-prompt
distribution, so the slice is kept as the small end-to-end check that a session can lean on
rather than as the headline.

| | value |
|---|---|
| median end-to-end speedup vs baseline | **0.411** (IQR 0.002) — assisted was ~2.4× slower |
| median decode-only speedup | 0.402 |
| tokens per target forward pass (assisted) | 3.56 |
| exact match with the baseline's tokens | True, both trials |

The mechanism attribution, the checksums and the session's other downloads are in
`results/sessions/slice-gpu-902bc285880d/README.md`.

## Where the derived files are, and one caveat

`results/benchmark_results.csv` is written by `report` for **one config at a time**, and
every config points at that same path. It currently holds the sweep's 180 rows. Each run's
table, exactly as Colab produced it, is kept verbatim in
`results/sessions/<run_id>/benchmark_results.csv`, and running

```
python -m tinyturbo report --config configs/slice_cpu.yaml
python -m tinyturbo report --config configs/slice_gpu.yaml
python -m tinyturbo report --config configs/gpu_main.yaml
```

regenerates them field-for-field from the raw records. That was checked row by row, joining
on `(prompt_id, max_new_tokens, method, trial)`, for the sweep and for the GPU slice - the
two runs whose session tables were archived, 0 of 29 fields differing in either. The CPU
smoke run has no archived table to compare against, only its raw records, so for it the
claim is weaker: `report` exits 0 and produces its 4 rows, and regenerating the same config
twice yields an identical figure.

Running `report` overwrites whatever stood at `results/benchmark_results.csv` before; the
evidence is the raw JSONL, which is never overwritten, so nothing is lost but a derived
summary. It also writes up to three figures per run, all named for the run id so they cannot
collide with each other or with an archived copy:

| figure | path | needs |
|---|---|---|
| speedup summary | `plots/speedup_<run_id>.png` | nothing but the raw records |
| accepted length vs speedup (figure 2) | `plots/accepted_len_vs_speedup_<run_id>.png` | nothing; the cost-model overlay needs a calibration |
| per-step time attribution (figure 3) | `plots/time_attribution_<run_id>.png` | a matching Stage 0 calibration |

## Stage 0 and Stage 4 artifacts

Two more directories sit beside `raw/`, and neither is a run's evidence:

* `results/calibration/<run_id>-calibration.json` - `tinyturbo calibrate`: the measured
target and assistant latencies (`tau_t`, `tau_t_verify`, `tau_d`), their raw samples, and the
derived `rho` and `kappa`.
* `results/profiling/<run_id>-profiling.json` - `tinyturbo profile`: labelled kernel time for
two representative cases, split into target passes, assistant passes and unattributed time.

Both record the hardware and software fingerprint of the machine that produced them, and
`report` refuses to apply a calibration to a run whose configuration, device, dtype,
attention backend, driver, library versions or lookahead differ from its own. When it refuses
it prints the specific differences, draws figure 2 without the cost-model lines, and writes
no figure 3 and no predicted-vs-observed table - because a comparison across two machines
would look like evidence without being any.

**One GPU calibration now exists.** `results/calibration/gpu-main-9e26c12945e6-calibration.json`
is the sweep's Stage 0 artifact, measured on the same card in a later session
(2026-10-14T17:22:16Z) with the same models, dtype, attention backend and lookahead. It was
checked against the sweep's manifest field by field before it was used - configuration
fingerprint, device, dtype, attention, GPU name, driver, CUDA, torch, transformers and γ all
match, so `calibration_is_usable()` returns no differences and `report` applies it without a
warning.

| | value |
|---|---|
| τ_t (target, 1 query token) | **22.431 ms** (IQR 0.583, n=20) |
| τ_t^verify (target, 6 query tokens = γ+1) | **25.737 ms** (IQR 1.149, n=20) |
| τ_d (assistant, 1 query token) | **35.250 ms** (IQR 1.379, n=20) |
| ρ = τ_t^verify / τ_t | **1.147** |
| τ_d / τ_t | **1.572** |
| κ = γ·τ_d/τ_t | **7.858** |

Each primitive is a median over 20 timed passes behind 5 discarded warm-ups, with a fresh
populated cache per sample and every raw sample kept in the artifact. The consequence is that
the sweep's cost model is no longer arithmetic over recorded counts: it is scored against
independently measured primitives, and the artifact is what makes figure 2 carry the model
lines, figure 3 exist at all, and the predicted-vs-observed table printable. All three
regenerate from a clean clone with no GPU. The numbers, the residuals and the hypothesis the
result did not support are in `docs/FINDINGS.md` §4, §5 and §6.

No Stage 4 artifact exists. `tinyturbo profile` was attempted on that same T4 and killed by
the session's memory limit (exit 137) after both models had loaded and generation had begun;
the retry routes are in `docs/FINDINGS.md` §9. It is the only thing that would decompose the
per-step residual, and nothing in `report` reads it.

One derived-schema change to be aware of: the sweep added a `max_new_tokens` column, because
a table that pools three generation lengths cannot tell those rows apart without it. The
archived session CSV for the GPU slice was written before that column existed, so regenerating
it adds that one column, carrying `64` in every row, with every other field unchanged -
checked field by field, and its figure still regenerates pixel-for-pixel. The sweep's own CSV
was written after the change and regenerates with no schema difference at all.

## Pending

* **Stage 1 leaves three hypotheses untouched.** H1 (prompt_lookup on grounded/repetitive
  prompts), H2 (adaptive lookahead beating a fixed γ) and H7 (the lossy ensemble frontier)
  all need methods that `configs/gpu_main.yaml` deliberately does not run. They are not
  refuted and not supported - they were not tested.
* **H5 (prefill dilution) is weakly exercised.** Prefill is a flat 40.8 ms, 5.0% / 1.3% /
  0.7% of a trial at 32 / 128 / 256 tokens, so e2e and decode-only speedups differ by at most
  0.010. The S2 prompts, specified at 300-1000 tokens to make prefill bite, tokenize to
  174-188, so the magnitude is not testable from this run.
* **Stage 4 has produced a GPU artifact, and it is a bounded trace.** `--trace-active 8` on a
  Kaggle T4 traced both cases and wrote
  `results/profiling/gpu-main-9e26c12945e6-profiling.json`. The *whole-generation* protocol is
  still the one that has never completed - it was killed by the T4 session's memory limit
  (exit 137) and on a 2-core CPU ran past a nine-minute budget - so the drafting loop's
  *internal* split is still not decomposed. What the bounded trace adds is a discrepancy
  rather than a decomposition: its per-pass means are 1.6–2.0× the Stage 0 primitives, by a
  roughly constant ≈20 ms per pass, and it was taken on a different driver/torch build than
  the calibration. `docs/FINDINGS.md` §9 sets out what that does and does not license.
* **H3 is not supported.** H3 predicted a material gap between measured speedup and the
  cost-model bound with the residual dominated by per-step overhead. With ρ and κ measured the
  residual is +0.058 / +0.010 / +0.018 against IQRs of 0.169 / 0.095 / 0.066, and the
  verification premium is 1.6% of a modelled iteration, so both clauses of its falsification
  rule hold. There is no material gap to attribute.
* **A second GPU (sm80+) is still unavailable**, so H8 stays untested rather than inferred.
* **The calibration comes from a second session.** The Stage 0 artifact is keyed to the same run id and
  describes the same models, dtype, backend and γ, but it was measured in a *different Colab
  session* on the same day. `report` compares the recorded fingerprints field by field, which
  is what makes that safe, and it refuses to apply a calibration whose recorded driver or
  library versions differ.
