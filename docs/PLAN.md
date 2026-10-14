# TinyTurbo — When does assisted decoding actually pay?

**Status:** pre-registration, with a completion note. Written *before* any GPU run.
Hypotheses, metrics and analysis rules are fixed here so that later results can contradict
them rather than be fitted to them.

> **Where this stands now (2026-10-14).** Stage 1 has run: `gpu-main-9e26c12945e6`, two
> methods (base + `model_assisted`) × 10 prompts × 3 lengths × 3 measured trials, on a T4.
> What it found, what is measured versus estimated, and what remains untested is in
> `docs/FINDINGS.md` - read that first; this file is the plan the results are scored against,
> and it is deliberately left as written so the pre-registration can be checked.
>
> Stage 0 (calibration) has since run on the same card, so §5's model is **scored** rather
> than unscored: ρ = 1.147 and κ = 7.858 predict 0.395 / 0.517 / 0.574 against observed
> 0.452 / 0.528 / 0.592. Those residuals sit inside the trial noise and the verification
> premium is 1.6% of a modelled iteration, which is H3's falsification condition, so H3 is
> reported as **not supported** in `docs/FINDINGS.md` §6 - a worse outcome for the plan than
> a large residual, and reported as such. Stage 4 (profiling) is implemented as
> `tinyturbo profile`, was attempted on the same T4 and was killed by the session's memory
> limit (exit 137), so the per-step cost is measured in total but not decomposed. It has
> since gained a bounded trace (`--trace-active N`), which records a window of the loop
> rather than all of it and says so in the artifact, and that route has now produced a T4
> artifact - an 8-step window on a Kaggle session, filed at
> `results/profiling/gpu-main-9e26c12945e6-profiling.json`. Its per-pass means are 1.6-2.0x
> Stage 0's primitives on a different driver and torch build, so the per-step cost is still
> measured in total and not decomposed: the trace bought a discrepancy, and
> `docs/FINDINGS.md` §9 says what would settle it. Stage 2
> (adaptive lookahead) and Stage 3 (lossy frontier) are out of scope for this study: the
> two-method scope is final, not partial. §3.1's `S2-long-open` specification was not met -
> those prompts tokenize to 174-188 rather than 300-1000, which is documented in
> `docs/FINDINGS.md` §7 rather than repaired, because the prompt set is an input to a run
> that already happened.

**One-line thesis:** assisted decoding has a theoretical speedup bound that the framework
implementation rarely reaches; this study measures both, explains the gap with instrumented
forward-pass and acceptance counts, and reports the cases where assisted decoding is *not*
worth it.

**Non-goals:** we do not reimplement the algorithm, train anything, or build a serving stack.
We use `transformers` assisted decoding as-is and study its behaviour. Comparisons against a
production engine (vLLM / TensorRT-LLM) are explicitly future work.

---

## 1. Why this is not "standard vs. assisted"

A two-arm comparison answers "did it get faster", which a blog post has already answered.
The version with signal answers four questions:

1. **Mechanism** — how many tokens does the target model accept per forward pass, and does
   that single number predict the speedup?
2. **Bound** — how close is the measured speedup to the cost-model bound, and what accounts
   for the gap (verification overhead? draft cost? Python loop overhead?)
3. **Workload dependence** — for which prompt types does each method help, and where does the
   assistant's overhead exceed the savings?
4. **Exactness** — assisted *greedy* is provably lossless. Is it, on our runs? And what does
   the deliberately lossy variant (static ensemble verification) cost in distributional bias?

Transformers v5 exposes several mechanisms that share one harness:

| Method id | Mechanism | Mechanism-specific knobs | Lossless? |
|---|---|---|---|
| `baseline` | plain greedy | — | reference |
| `model_assisted` | small draft model drafts γ tokens | `num_assistant_tokens` | yes |
| `heuristic_schedule` | draft model + dynamic lookahead | `num_assistant_tokens_schedule` | yes |
| `confidence_threshold` | draft stops early if unsure | `assistant_confidence_threshold` | yes |
| `prompt_lookup` | n-gram match inside the context, no second model | `prompt_lookup_num_tokens`, `max_matching_ngram_size` | yes |
| `ensemble_w070` | verify against `w·p_target + (1−w)·q_draft` | `assistant_ensemble_weight` | **no** |

Documented but **out of scope on ≤16 GB hardware**: `use_mtp=True` (requires DeepSeek-V3 /
GLM-4.5-class checkpoints), `assistant_early_exit` (requires a LayerSkip checkpoint),
`tokenizer` + `assistant_tokenizer` universal decoding (needs a mismatched pair; possible as a
follow-up, not part of the main sweep). `speculation_type="dflash"` is undocumented — see §11.

---

## 2. Pre-registered hypotheses

Each hypothesis has an explicit falsification condition. The write-up must state which
hypotheses failed; a failed hypothesis is a result, not an embarrassment.

**H1 — Workload dependence.** `prompt_lookup` (zero extra parameters, zero extra memory)
will beat `model_assisted` on grounded/repetitive prompts (strata S3–S4) and lose to it on
open-ended prompts (S1–S2).
*Falsified if:* prompt-lookup's relative median speedup vs. baseline is lower than
model-assisted's in S3–S4.

**H2 — Lookahead beats fixed lookahead.** `heuristic_schedule` and `confidence_threshold`
will both outperform `model_assisted` at its best fixed `num_assistant_tokens`, because the
optimal γ differs between prompts and within a single generation.
*Falsified if:* the best fixed γ matches or beats both adaptive variants on median decode
speedup.

**H3 — The bound is not reached.** Measured decode speedup will be materially below the
cost-model bound (§5), and the residual will be dominated by per-step framework overhead
rather than by verification cost (`ρ − 1`).
*Falsified if:* the residual is within measurement noise, or verification overhead dominates.

**H4 — Acceptance length is the mechanism variable.** Measured mean accepted length explains
the majority of between-config variance in speedup. Operationally: rank correlation between
accepted length and speedup ≥ 0.8 within the model-assisted family.
*Falsified if:* correlation < 0.8, implying an unmodelled factor dominates.

**H5 — Prefill dilutes.** Because prompt prefill cost is identical across arms, end-to-end
speedup will systematically understate decode speedup, and the dilution will grow with prompt
length and shrink with generation length.
*Falsified if:* end-to-end and decode-only speedups agree within noise.

**H6 — Exactness holds for every lossless method.** For every `(prompt, max_new_tokens, γ)`
configuration, assisted greedy produces token-identical output to plain greedy **at the same
dtype**. Any mismatch is a bug to investigate, not a result to report.
*Falsified if:* mismatches persist after ruling out dtype, compile and cache differences.

**H7 — Lossiness is measurable and monotone.** Lowering `assistant_ensemble_weight` w raises
acceptance length (faster) and raises the target-model NLL of the generated text (worse
distributional fidelity), monotonically in w.
*Falsified if:* either relationship is non-monotone, or the NLL penalty is within noise.

**H8 — Hardware dependence.** The T4-class result (fp16, SDPA, no FlashAttention-2, no bf16)
will show a smaller speedup than a ≥sm80 result at identical settings. If no second GPU is
available this hypothesis stays **untested** and must be reported as such.

---

## 3. Design space

### 3.1 Prompt set (fixed, committed, offline)

`configs/prompts.jsonl`, 10 hand-written prompts, explicit text (no dataset download, so runs
are reproducible without network access). Four strata chosen so that the prompt-lookup
mechanism has something to exploit:

| Stratum | n | Description | Why it is here |
|---|---|---|---|
| `S1-short-open` | 3 | 1–40 token prompts, open-ended continuation | Worst case for lookup, easy for a drafter |
| `S2-long-open` | 3 | 300–1000 token prompts, open-ended | Tests prefill dilution (H5) |
| `S3-grounded` | 2 | long source text + instruction to summarise/quote | Heavy n-gram overlap → lookup should win |
| `S4-repetitive` | 2 | structured text (config-like, enumerations) | Near-best case for lookup |

Every record carries a stable `prompt_id`; strata are never pooled without being reported
separately first.

### 3.2 Factors

| Factor | Levels |
|---|---|
| `method` | the 6 ids in §1 |
| `max_new_tokens` | 32, 128, 256 |
| `num_assistant_tokens` (γ) | 2, 5, 10 |
| `num_assistant_tokens_schedule` | `constant`, `heuristic` |
| `assistant_confidence_threshold` | 0.4 (v5 adjusts it internally) |
| `assistant_ensemble_weight` | 0.5, 0.7, 0.9 |
| `prompt_lookup_num_tokens` | 5, 15 |
| `cache_implementation` | default (`dynamic`), `static` + `max_cache_len` |
| `dtype` | `float16` (GPU), `float32` (CPU smoke) |

Full factorial is ~10⁴ runs; not attempted. The sweep is staged and each stage is independently
runnable and resumable:

- **Stage 0 — calibration (no generation).** §5 primitives: τ_t, τ_t^verify(γ), τ_d, per
  prompt length and γ.
- **Stage 1 — main sweep.** 6 methods × 10 prompts × 3 generation lengths, γ = 5 fixed for
  model-based methods, 3 trials. → H1, H4, H5, H6.
- **Stage 2 — lookahead.** {`model_assisted`, `heuristic_schedule`, `confidence_threshold`} ×
  γ ∈ {2, 5, 10} × 5 prompts (one per stratum + 1) × 3 trials. → H2.
- **Stage 3 — lossy frontier.** ω ∈ {0.5, 0.7, 0.9} × 10 prompts × 3 trials, sampled decoding
  with fixed seeds. → H7.
- **Stage 4 — attribution.** `torch.profiler` traces on 3 representative (method, prompt)
  pairs. → H3.

---

## 4. Metric definitions

Ambiguity here is what makes benchmark numbers unreproducible, so each metric is defined
operationally.

**Timing.** All wall-clock regions are bracketed by `torch.cuda.synchronize()` on GPU.
- `prefill_ms` — one forward pass over the full prompt with an empty cache, measured
  immediately before the timed `generate` call. Identical across arms by construction; reported
  so it can be subtracted.
- `e2e_ms` — the timed `generate` call only.
- `decode_ms = e2e_ms − prefill_ms`; `decode_ms_per_token = decode_ms / tokens_generated`.
- `tok_per_s` — decode-only. End-to-end throughput is reported separately when needed so the
  two are never confused.

**Mechanism (instrumented, not inferred).** A context manager wraps `target.forward` and
`assistant.forward` and counts invocations:
- `target_forwards` — number of decoding iterations.
- `assistant_forwards` — total draft passes.
- `accepted_len_mean = tokens_generated / target_forwards`. This is the tokens emitted per
  target forward pass (accepted prefix + the target's own correction token), i.e. the quantity
  `b` in §5. It is *not* an acceptance rate.
- `draft_per_step = assistant_forwards / target_forwards`. If this is far below γ, the
  requested γ is not being honoured — a correctness signal, not a curiosity (§11).

**Comparisons.** Speedup is always computed as a **paired ratio per prompt** against that
prompt's own baseline in the same dtype, then aggregated by median with IQR across trials:
`speedup_e2e`, `speedup_decode = baseline_decode_per_token / method_decode_per_token`.

**Exactness.** `exact_match` compares the full generated token sequence to the baseline's, out
to `max_new_tokens` or the first EOS. Reported per config; H6 requires it everywhere for
lossless methods.

**Bias (lossy arm only).** `target_mean_nll` — mean per-token negative log-likelihood of the
generated continuation under the target model, computed in a separate scoring pass. This is the
fidelity cost of ensemble verification; lower is better and the lossless arm is the reference.

**Memory.** `peak_alloc_mb` and `peak_reserved_mb` from `torch.cuda.max_memory_allocated /
reserved`, reset before each trial; on-GPU used memory from NVML at trial end when available.

**Trial protocol.** Per (config, prompt): 2 discarded warm-ups, then 5 measured trials. Arms
are **interleaved** (`baseline, method, baseline, method, …`) rather than blocked, so thermal
and clock drift on a shared cloud GPU cannot masquerade as a treatment effect. Clock and
temperature are sampled via NVML where available and stored with each record. Aggregate is
median + IQR, never mean ± sd.

**Hygiene flags recorded per run:** dtype, `attn_implementation`, `disable_compile`, cache
implementation, `max_cache_len`, `use_cache`, `do_sample=False`, `num_beams=1`,
`renormalize_logits` untouched, EOS ids, `hit_eos`, whether the run ended by length or EOS.
If any arm terminates early, comparisons truncate to the shortest common generated length.

---

## 5. Cost model (the part to be scored)

`transformers` automatic compilation and cache policy can silently change results, so
`disable_compile` and `cache_implementation` are pinned and recorded rather than left to
defaults.

Let τ_t = time of a target forward pass over **1** query token with a populated cache, and
τ_d the same for the draft model. Let ρ = τ_t^verify / τ_t, where τ_t^verify is the target pass
over the γ drafted tokens **plus** the correction position. Let κ = γ · τ_d / τ_t. Per decoding
iteration the model spends `τ_t^verify + γ·τ_d` and emits `b` tokens, while the baseline emits
1 token in τ_t, so

```
speedup_decode ≈ b / (ρ + κ)          (full model)
speedup_decode ≈ b / (1 + κ)          (ideal-verification model, ρ = 1)
```

The physics that makes this work at batch 1 is that a decode pass is dominated by reading
weights, so verifying γ+1 positions costs nearly the same as verifying 1 — i.e. ρ is close to 1.
Whether it *is* on our hardware is measured, not assumed.

Scoring: Stage 0 measures τ_t, τ_t^verify, τ_d; Stage 1 measures b and the observed speedup.
We then report, per config, **predicted vs. observed speedup and the signed residual**, and
attribute the residual to (a) ρ − 1 and (b) unmodelled per-step overhead, the latter quantified
by Stage 4 profiling. A model that predicts 8 of 11 configurations and fails on 3, with the
failures explained, is the deliverable — not a model with a good average.

---

## 6. Repository layout

```
tinyturbo/
├── README.md                  # the recruiter surface: headline, figure, surprises, repro
├── pyproject.toml             # package metadata + ruff/pytest config
├── requirements.txt           # exact pins, Colab-installable in one line
├── configs/
│   ├── prompts.jsonl          # the fixed prompt set
│   ├── smoke.yaml             # CPU, tiny model, seconds
│   ├── gpu_main.yaml          # Stage 1
│   ├── gpu_lookahead.yaml     # Stage 2
│   ├── gpu_lossy.yaml         # Stage 3
│   └── gpu_profile.yaml       # Stage 4
├── src/tinyturbo/
│   ├── config.py              # YAML -> validated dataclasses, run_id hashing
│   ├── env.py                 # fingerprint: GPU/CC/driver/CUDA/torch/transformers/git sha
│   ├── models.py              # loading, dtype + attention selection per compute capability
│   ├── instrument.py          # forward counters, timers, NVML sampling, profiler hooks
│   ├── methods.py             # method id -> generate kwargs (single source of truth)
│   ├── runner.py              # sharded, resumable sweep execution
│   ├── metrics.py             # accepted length, exactness, NLL, percentiles
│   ├── report.py              # raw JSONL -> CSV + figures + report.md
│   └── cli.py                 # run | calibrate | report | verify
├── tests/                     # CPU-only, run in CI
├── results/
│   ├── README.md              # run_id -> hardware map, and what produced what
│   ├── raw/*.jsonl            # the only committed evidence
│   └── benchmark_results.csv  # derived from raw, never hand-edited
├── plots/
├── notebooks/colab.ipynb      # installs pinned versions, runs one resumable shard
└── docs/
    ├── PLAN.md                # this file
    └── FINDINGS.md            # written after the run: predicted vs observed
```

Rationale for `src/` over a single `benchmark.py`: the sweep is a
(method × knob × workload × trial) product with resumability requirements. One file became
unmaintainable the moment the third method was added.

---

## 7. Evidence chain (anti-fabrication, enforced not promised)

- One JSONL record per (config, prompt, trial) under `results/raw/`, containing the resolved
  config, the environment fingerprint, raw counts and raw timings.
- **Raw records hold observations only.** All comparisons — speedups, ratios, aggregates — are
  computed in `report.py` and live only in the derived CSV. Nothing pair-dependent is stored
  raw, so a speedup cannot exist without both of its arms.
- `run_id = sha256(canonical resolved config + code commit)`. Reruns are idempotent; resume
  skips completed records only when the fingerprint matches.
- `tinyturbo verify` (and a test) fails if: a derived row has no raw record, a raw record lacks
  a hardware fingerprint, the CSV is newer than the raw data it claims to summarise, or a raw
  record's config hash does not match its run_id.
- `results/README.md` maps every committed run to the GPU it ran on and the command that
  produced it. Results produced on different GPUs are never merged into one aggregate.
- GPU-only artifacts (fp16 raw runs) are committed only when a real run produced them. The
  repository must be meaningful and honest with `results/raw/` empty except the CPU smoke run.

---

## 8. Hardware-agnostic execution

`hardware: auto` resolves to a profile from the device's compute capability; every resolved
value is written into the fingerprint.

| Profile | dtype | attention | Notes |
|---|---|---|---|
| CPU (smoke only) | float32 | `eager`/`sdpa` | tiny models only; fp16 is not meaningful here |
| sm60/sm70 (P100) | float16 | `sdpa` | no FlashAttention-2 |
| sm75 (T4, Colab free) | float16 | `sdpa` | **no bf16, no FA2** — batch-1, bandwidth-bound |
| sm80+ (A100/4090) | bfloat16 | `flash_attention_2` if installed | needed for H8 |

Practical consequences to design for: Colab free sessions disconnect, so runs are **sharded**
(`--shard i/n`) and **resumable** from `results/raw/`; the local dev environment is Python
3.14.6 / torch 2.10.0+cpu / transformers 5.3.0 while cloud images differ, so `requirements.txt`
pins exact versions and the notebook installs *that* file rather than whatever pip resolves;
the fingerprint records the commit of `transformers` so a weekly v5 minor release cannot
silently change the API under a rerun.

---

## 9. Figures (three, each carrying one message)

1. **"Speedup is not one number."** Paired per-prompt speedup vs. `max_new_tokens`, one series
   per method, with decode-only speedup as a faint overlay. Message: which methods help, how
   much prefill dilutes (H5), and the spread across prompts.
2. **"Accepted length explains it."** Observed speedup vs. measured `accepted_len_mean` per
   config, with the §5 curve overlaid; marker shape = method, labelled outliers. Message:
   mechanism + where the model breaks (H3, H4).
3. **"Where the time actually goes."** Stacked per-step time attribution (target verify /
   draft passes / loop overhead) per method, normalised to one baseline target step. Message:
   the gap to the bound is framework overhead, quantified (H3).
4. *(optional, Stage 3)* **"The lossless–lossy frontier."** Speedup vs. `target_mean_nll`
   across w. Message: what exactness costs (H7).

Plots are regenerated from raw by `tinyturbo report`; no figure is ever hand-edited.

---

## 10. Phased build, with exit criteria

| Phase | Deliverable | Exit criterion |
|---|---|---|
| 0 | package skeleton, config schema, fingerprint, instrumentation, CPU smoke config, CI | `pytest` green on CPU; one smoke run yields valid JSONL and a regenerated CSV + figure |
| 1 | Stage 1 methods + exactness test + report pipeline | one **real GPU run**, ≥10 prompts, H6 holds, CSV and figures 1–2 regenerated from raw |
| 2 | Stage 0 calibration + Stage 2 lookahead | predicted-vs-observed table for ≥11 configs (H1–H4 scored) |
| 3 | Stage 4 profiling + Stage 3 lossy frontier | `docs/FINDINGS.md` answers *why*, not just *what* (H3, H7) |
| 4 | README rewrite, Colab notebook, `FINDINGS.md` | a stranger can reproduce one figure from a clean clone + one command |

Nothing in Phase 4 is allowed to sacrifice Phase 1–3 validity.

---

## 11. Known risks and unknowns

| Risk | Mitigation / stance |
|---|---|
| `speculation_type="dflash"` appears in v5 but is undocumented — unknown mechanism | investigate in Phase 2; do **not** promise results from it |
| `num_assistant_tokens` was once not propagated to the assistant's own `generation_config` (reported against 4.56.0 as GH #40739, closed via PR #40740; believed fixed in 5.3.0, not yet re-verified here). A wrong γ would silently invalidate the lookahead sweep | keep the `draft_per_step ≈ γ` assertion per record as a **regression guard**, not because failure is expected; treat a violation as a blocker, not a footnote |
| Speculative decoding is batch-1 only in `generate`; v5 continuous batching is a separate API path | out of scope, stated as such; throughput-vs-batch-size claims are avoided rather than approximated |
| Automatic compilation can silently alter results | `disable_compile` and cache policy pinned and recorded |
| fp16 non-determinism means exactness must be asserted *within* a dtype | baseline and method always run at the same dtype; H6 compares like with like |
| Colab/Kaggle session limits mid-sweep | sharding + resume keyed on run_id; partial shards are still reports |
| No second GPU available → H8 untestable | reported as untested, never inferred from someone else's numbers |
| Result could plausibly be "assisted decoding does not help here" | this is a legitimate, publishable finding under the §1 framing; the write-up commits to reporting it plainly |

---

## 12. What a reader should be able to get in 90 seconds

`README.md` structure, in order:

1. The headline finding in one sentence, with the number and the GPU it came from.
2. Figure 2 above the fold.
3. "Three things I expected and what actually happened" — including refuted hypotheses.
4. Method in a paragraph, with the model pair, prompt strata and version pins.
5. One command to reproduce, and the hardware fingerprint of the recorded run.
6. Limitations, stated by the author, not waiting to be found by a reader.
7. Next steps: batch-size effects via continuous batching, a production-engine comparison,
   a fused acceptance step, an sm80 rerun.
