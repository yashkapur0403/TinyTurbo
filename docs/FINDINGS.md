# Findings — the two-method study, and what is still open

Written after the Stage 1 sweep ran to completion, and revised once Stage 0 ran on the same
card. Every number below is produced by `python -m tinyturbo report` from the records in
`results/raw/` and the calibration artifact in `results/calibration/`, and each one names the
run it came from. Where a number is arithmetic over recorded counts, or an inference rather
than a measurement of the quantity itself, it says so.

| | |
|---|---|
| Run | `gpu-main-9e26c12945e6`, `configs/gpu_main.yaml` |
| What ran | 10 prompts × 3 generation lengths × 2 methods = 30 points, 2 discarded warm-ups + 3 measured trials per point |
| Hardware | Tesla T4, CC 7.5, 14912 MiB, driver 580.82.07, CUDA 13.0, cudnn 92700 (Colab free, 2 vCPUs) |
| Software | Linux 6.6.122+, Python 3.13.15, torch 2.11.0+cu130, transformers 5.3.0 |
| Models | `SmolLM-1.7B` target (1 711 376 384 params), `SmolLM-135M` assistant (134 515 008), ratio 0.0786 |
| Resolved | `cuda`, `float16`, attention `sdpa`, γ = 5, `assistant_confidence_threshold` 0.0 |
| Session | 24 min 13 s, 2026-10-14 14:24:51 → 14:49:04 UTC, 301 records (1 manifest + 300 trials) |
| Evidence | `results/raw/gpu-main-9e26c12945e6.jsonl`; `verify` exits 0 on it; CSV and figures regenerate from it |
| Stage 0 | `results/calibration/gpu-main-9e26c12945e6-calibration.json` - the same models, dtype and attention backend measured on the same T4, in a later session on the same day (2026-10-14T17:22:16Z) |
| Stage 4 | `results/profiling/gpu-main-9e26c12945e6-profiling.json` - an **8-step bounded trace** of two cases on a T4, on a *different image* (driver 580.178.04 / CUDA 12.8 / torch 2.11.0+cu128), Kaggle, 2026-10-14T20:06:13Z; its per-pass means do not reconcile with Stage 0 (§9) |

## 1. The headline

**Model-assisted greedy decoding was slower than plain greedy at every one of the 30
points.** The best point reached 0.659× baseline; the worst 0.303×.

| generation length | n | median e2e speedup | IQR | decode-only | tokens per target forward pass | draft passes per target pass |
|---|---|---|---|---|---|---|
| 32 | 30 | **0.452** | 0.169 | 0.443 | 3.56 | 4.79 |
| 128 | 30 | **0.528** | 0.095 | 0.525 | 4.66 | 4.93 |
| 256 | 30 | **0.592** | 0.066 | 0.591 | 5.17 | 4.94 |
| all | 90 | **0.528** | 0.171 | 0.525 | — | — |

Pooled, assisted decoding took about **1.9× longer** than the baseline it was paired with.
This is not a marginal result that a different seed would flip: the pooled median is 0.528,
no point's median reaches 1.0, and the largest of the three interquartile ranges (0.169 at 32
tokens) does not come close to spanning it.

`draft passes per target pass` sits at 4.79–4.94 against a configured γ = 5 throughout. That
is the plan's regression guard from §11: the requested lookahead reached the assistant's own
`generation_config` and was honoured, so the negative result is not an artifact of a draft
length that silently defaulted to something else.

## 2. The mechanism moved exactly as predicted, and still lost

Two relationships in the data are in the direction the mechanism predicts, and they are the
reason this is an interesting negative result rather than a broken one.

**Speedup rises with generation length.** 0.452 → 0.528 → 0.592 as the continuation grows
from 32 to 256 tokens, and accepted length rises with it (3.56 → 4.66 → 5.17 tokens per
target pass). A longer continuation amortises the fixed prefill and gives the drafter more
chances to land a long accepted run, so both curves move as the theory says they should. The
assisted arm gets *relatively* better as generations get longer, and is still 1.7× slower at
the longest length tested.

**Speedup tracks accepted length across prompts.** Spearman rank correlation between
`accepted_len_mean` and `speedup_e2e` is **0.98** pooled over the 90 measured trials, and
0.91 / 0.97 / 0.97 within the three lengths separately. H4's pre-registered threshold was
≥ 0.8, so it is cleared in all four groupings: the mechanism variable the plan named does
explain the ordering of the results, even though the level is below 1.0.

Workload dependence shows up the same way, in the one method family that was run:

| stratum | points | median speedup | tokens per target pass | median prompt tokens |
|---|---|---|---|---|
| S1-short-open | 9 | 0.430 | 3.56 | 12 |
| S2-long-open | 9 | 0.475 | 4.00 | 183 |
| S3-grounded | 6 | 0.618 | 5.33 | 231 |
| S4-repetitive | 6 | 0.605 | 5.23 | 241 |

Prompts whose text invites repetition - grounded summaries and structured enumerations -
accept considerably more per target pass than open-ended continuation does, and are
correspondingly less bad. H1 predicted this *ordering* between strata for `prompt_lookup`
against `model_assisted`; what is measured here is only the accepted-length side of it, in a
single method family.

## 3. Exactness held everywhere it was tested (H6)

`exact_match` is `True` for **90 of 90** assisted measured trials, with
`first_divergence_index` null in every one: ten prompts, three generation lengths, 90 trials
at float16 with `sdpa` attention, each producing token-identical output to its paired
baseline at the same dtype. No trial hit EOS, so every comparison covers the full requested
length rather than truncating early.

This is the widest test of the plan's losslessness claim so far, and it passed. It is worth
being precise about what it does not show: it is exactness at γ = 5, one dtype and one
attention backend. The claim that fp16 `sdpa` numerics never break it is supported over
these 90 trials, not proven.

## 4. Where the time went: measured first, arithmetic second

Both readings of the same run agree that the cost is in the drafting loop, not in
verification, and the first of them is now a **measurement** rather than an inference.
Neither is a profiler result - Stage 4 was attempted on the same card and blocked by the
session's memory limit (§7) - so nothing here attributes time *within* one decoding
iteration. That is still the open end of this section.

**(a) Stage 0 measured the primitives, and the drafter is the cost.** `tinyturbo calibrate`
timed the three quantities §5 needs, on the same models, dtype, device and attention backend
as the sweep, against a real prompt's cache, discarding 5 warm-up passes before 20 timed
ones:

| primitive | median | IQR |
|---|---|---|
| τ_t (target, 1 query token) | **22.431 ms** | 0.583 |
| τ_t^verify (target, 6 query tokens = γ+1) | **25.737 ms** | 1.149 |
| τ_d (assistant, 1 query token) | **35.250 ms** | 1.379 |

hence **ρ = 1.147** and **κ = γ·τ_d/τ_t = 7.858**, with τ_d/τ_t = 1.572. A 135M draft pass
costs **1.57×** what a 1.7B target pass costs, against a parameter ratio of 0.0786: about
20× more per parameter.

Two consequences for H3 live in those numbers, and they point the same way:

* **Verification is not the cost.** ρ − 1 = 0.147 against a modelled iteration of
  ρ + κ = 9.005. A verification pass over γ+1 positions costs 14.7% more than a single-token
  pass, and that premium is **1.6%** of the per-iteration cost. Substituting the plan's ideal
  verification (ρ = 1) for the measured value moves the predicted speedup by at most
  **0.009**, so §5's bandwidth assumption holds on this card.
* **Drafting is the whole cost.** κ = 7.858 means the five draft passes outweigh the
  verification pass by 5.4×: one assisted iteration costs 9.0 target steps to emit 3.6–5.2
  tokens.

**(b) The sweep's own arithmetic, which never reads Stage 0, lands on the same numbers.**
Taking the baseline's measured cost per target forward pass (22.90 / 23.24 / 23.40 ms at
32 / 128 / 256 tokens, prefill subtracted) and removing the assisted arm's target passes from
its decode time leaves 35.99 / 37.33 / 36.73 ms per **assistant** forward pass *in the loop*,
against the 35.25 ms measured *outside* it. That is the same quantity to within 2–6%, reached
by two procedures that share no inputs: one times the assistant alone against a populated
cache, the other subtracts recorded target passes from recorded decode time.

That agreement is what makes §4(a) a cross-check rather than the consistency check it used to
be, and it also fixes the sign of §5's residuals: the isolated measurement is the lower of the
two, so a prediction built from it over-prices the iteration slightly and under-predicts the
speedup, which is what the small positive residuals in §5 show. What it cannot separate is
"the assistant's kernel time really is 1.6× the target's" from "some other per-step cost
scales with the number of draft passes" - and with those residuals now inside the trial noise
(§5), there may be little left for a trace to attribute.

**(c) The prefill is not the problem.** Prefill is a flat 40.8 ms in this configuration - 5.0%
of a baseline trial at 32 tokens, 1.3% at 128 and 0.7% at 256. The end-to-end and decode-only
speedups therefore differ by at most 0.010, and the dilution shrinks with generation length
exactly as H5 predicts. Because the assisted arm is *slower* here, that dilution pulls the
ratio toward 1.0, so it understates the slowdown rather than the speedup - the opposite sign
to the one H5 is written for. Either way the effect is negligible at these prompt sizes.

**(d) Memory is not a constraint.** Median `peak_alloc_mb` is 3617.5 MiB for the baseline and
3635.4 MiB for the assisted arm; `peak_reserved_mb` is 3820.0 for both. The 135M assistant
and its KV cache add under 20 MiB on a 15 GiB card, so this result is not a memory-pressure
artifact and a larger assistant would not be excluded by memory at this batch size.

## 5. What is measured, what is derived, and what is still open

This is the distinction that matters most for reading the rest of this repository.

| quantity | status |
|---|---|
| accepted length, forward-pass counts, prefill, e2e, memory, exactness | **measured**, in `results/raw/` |
| §5's τ_t, τ_t^verify(γ), τ_d | **measured on a T4**: 22.431 / 25.737 / 35.250 ms as medians over 20 samples behind 5 discarded warm-ups, with every raw sample kept in `results/calibration/gpu-main-9e26c12945e6-calibration.json` |
| ρ, κ | **measured**: ρ = 1.147, κ = 7.858. The sweep's own arithmetic had implied κ ≈ 7.9 (§4b); the two agree closely because they price the same hardware, not because either was fitted to the other |
| the draft-pass cost ratio (1.57–1.61×) | **measured as 1.572×** (τ_d/τ_t), and independently implied as 1.57–1.61× by the sweep's arithmetic |
| per-step framework overhead | **still not measured.** With ρ and κ measured, the model's residual is at most 0.048 and inside the trial noise, so this is a small quantity rather than a large unexplained one - but no trace has decomposed it, and Stage 4 is what would |

### The scored cost model

`speedup_decode ≈ b / (ρ + κ)`, with ρ and κ from Stage 0 and `b` measured in the sweep. This
is what `python -m tinyturbo report --config configs/gpu_main.yaml` prints:

| generation length | b | predicted | observed e2e | residual |
|---|---|---|---|---|
| 32 | 3.56 | 0.395 | 0.452 | +0.058 |
| 128 | 4.66 | 0.517 | 0.528 | +0.010 |
| 256 | 5.17 | 0.574 | 0.592 | +0.018 |

The model predicts *decode-only* speedup while the table's `observed` column is the
end-to-end one `report` prints, so prefill dilution sits inside those residuals as assumption
6 below says. Measured against the decode-only speedups (0.443 / 0.525 / 0.591) the residuals
are **+0.048 / +0.008 / +0.017**. Either way every residual is positive - the model slightly
*under*-predicts the speedup - and every one is smaller than the trial spread at its own
length (IQR 0.169 / 0.095 / 0.066).

### The cost model's assumptions, stated plainly

`speedup_decode ≈ b / (ρ + κ)` is a model, and every one of these is an assumption that this
hardware or this library could violate:

1. **One iteration emits `b` tokens for one verification pass plus γ draft passes.** Assumes
the loop runs its full lookahead every step with no early stopping. Measured support: the
constant schedule is pinned, `assistant_confidence_threshold` is 0.0, and draft passes per
target pass came out at 4.79–4.94 against γ = 5.
2. **ρ is a constant, independent of γ and of the cached prefix length.** This is the whole
physics of the method: a decode pass at batch 1 is dominated by reading weights, so
verifying γ+1 positions should cost about what verifying 1 costs. If this card is not
bandwidth-bound for these shapes, ρ > 1 and the model over-predicts. Stage 0 measures it
rather than assuming it, which is exactly why it was not estimated from the sweep. Measured
at γ = 5: **ρ = 1.147**, so this card is close to bandwidth-bound but not exactly, and the
difference is worth 0.009 of predicted speedup at `b` = 4.66.
3. **κ assumes each draft pass costs one single-token assistant decode step.** It ignores any
difference between the assistant's cache state in the loop and in isolation, and treats the
cost as linear in γ. Measured τ_d = 35.250 ms in isolation against 35.99–36.73 ms implied in
the loop (§4b): they differ by 2–6%, and that difference is the sign of the residuals above.
4. **Only model time is in the model.** Python bookkeeping between steps, cache reallocation,
the stopping criteria, masking and sampling are not in it. The plan attributes those to a
separate residual term, which only a trace can quantify — which is Stage 4. Figure 3 now
exists and stacks *measured* components, but it shows the residual rather than decomposing it,
and Stage 4 did not complete (§7).
5. **`b` is treated as a property of the models, not of the timing.** Accepted length is
measured in the loop and used as an input to the prediction; if drafting cost influenced
acceptance, the prediction would be circular.
6. **It predicts decode-only speedup.** Prefill is identical across arms and excluded, so
   comparing it against `speedup_e2e` imports prefill dilution - bounded here at 0.010 (§4c).
   The table `report` prints uses the e2e observed value, so that dilution is inside its
   residuals; the scored-model table above gives both.

What that leaves open: the residuals are +0.058 / +0.010 / +0.018, so they are not uniform
across generation lengths, and the model contains no term that would make them
length-dependent. They are also all positive and all smaller than the trial spread at their
own length. The honest reading is that assumption 3 is mildly wrong in a direction that costs
nothing here - the in-loop draft pass is a little cheaper than the isolated one (§4b) - and
that the residual is now too small to carry an explanation. Stage 4 would have measured the
in-loop mean cost of a target and an assistant forward pass directly, which is what could
separate "expensive kernels" from "per-step overhead"; with the residual inside the noise it
would have confirmed a small number rather than explained a large one.

A CPU run of `tinyturbo calibrate` against the CPU smoke configuration confirms the
measurement path works end to end: it returns τ_t = 131.7 ms, τ_t^verify = 162.3 ms,
τ_d = 159.9 ms, hence ρ = 1.232 and κ = 4.856, at float32 on a machine with no GPU, with the
same checkpoint on both sides. Those are **CPU plumbing numbers and must never be quoted as
the calibration** - on CPU the draft model is exactly as expensive as the verifier, which is
why τ_d/τ_t ≈ 1.2 rather than ≈ 0.2. They are recorded here only as evidence that the
protocol runs: warm-ups discarded, a fresh populated cache per sample, raw samples retained.

`tinyturbo profile` was likewise verified to complete and to write its artifact, on CPU with
a small case (4 new tokens: 693 ms end to end, 181 ms inside labelled forward passes, 512 ms
unattributed). Its two default cases are sized for a GPU - 32 and 256 new tokens, whole
generation - and they have been attempted on both kinds of machine this study has had and
completed on neither: on a 2-core CPU the profiler's own host-side overhead ran past a
nine-minute budget, and on the T4 the session's memory limit killed the process (exit 137,
§7). The *whole-generation* protocol is therefore still unverified as an artifact-producing
command on a real pair. What has since produced an artifact is the **bounded** route: an
8-step window on a Kaggle T4 wrote `results/profiling/gpu-main-9e26c12945e6-profiling.json`
(§9), and `configs/profile_smoke.yaml --trace-active 4` writes one on CPU in about 15 s with
no GPU at all. Both are windows, so they verify the path and not the decomposition - and the
T4 one disagrees with Stage 0 by ±20 ms per pass, which is itself the most interesting thing
Stage 4 has produced. Figure 3 and the predicted-vs-observed table do not depend on any of
it, which is why the scoring above is complete without it.

The guard that keeps that distinction honest is code, not convention.
`calibration_is_usable()` compares the calibration's configuration fingerprint, device, dtype,
attention backend, GPU name, driver, CUDA version, torch and transformers versions, and the
lookahead γ against the run's manifest. If anything differs, `report` refuses the calibration:
it prints the specific differences as warnings, draws figure 2 **without** the cost-model
lines, and writes no figure 3 and no predicted-vs-observed table. The failure mode being
protected against is a table built from a measurement of a different machine that looks like
evidence.

## 6. Hypotheses: what the sweep did and did not settle

| | hypothesis | status |
|---|---|---|
| H1 | `prompt_lookup` beats `model_assisted` on grounded/repetitive prompts | **untested** - `prompt_lookup` was not run. The accepted-length ordering across strata is consistent with the mechanism H1 is about, and is not a test of H1. |
| H2 | adaptive lookahead beats the best fixed γ | **untested** - no adaptive schedule was run; γ = 5 constant by design |
| H3 | measured speedup is materially below the cost-model bound, residual dominated by per-step overhead | **not supported.** With ρ and κ measured in Stage 0, the residuals are +0.058 / +0.010 / +0.018 (e2e) and sit inside the trial noise at every length, and verification overhead is 1.6% of the modelled iteration rather than the dominant term. Both clauses of H3's falsification rule hold: the residual is within measurement noise, and verification cost does not dominate. Measured speedup is *not* materially below the bound - the bound itself is low, because the drafter is expensive. |
| H4 | accepted length explains the majority of between-config variance | **supported** - Spearman 0.98 pooled, 0.91–0.97 within each length, threshold was 0.8 |
| H5 | prefill dilutes e2e relative to decode, growing with prompt length | **magnitude untestable here.** Prefill is 0.7–5.0% of a trial and e2e vs decode differ by ≤ 0.010. The designed test - 300–1000 token prompts - was not achieved; see §7. |
| H6 | assisted greedy is token-identical to plain greedy for lossless methods | **supported** - 90/90 trials, 3 lengths, 10 prompts |
| H7 | lossiness is measurable and monotone in ensemble weight | **untested** - the lossy arm was not run |
| H8 | a T4-class result is worse than an ≥sm80 result | **untested** - no second GPU has been available |

The bounded Stage 4 trace does not reopen H3. It does show in-loop per-pass costs well above
the isolated primitives (§9), which is the *direction* H3's second clause points, but those
numbers were taken under the profiler on an image that differs from the calibration's, so they
cannot be used as evidence for or against it. H3 stands as not supported on the Stage 0
scoring, which needed no trace.

Three hypotheses were never in scope for this stage, and the two-method scope is deliberate:
one alternative method family is not a method study. `prompt_lookup`, the adaptive schedules
and ensemble verification have their own configurations and their own stages in
`docs/PLAN.md`.

## 7. Defects, and the decision not to paper over them

**The S2 prompts are too short for the job they were designed for.** `docs/PLAN.md` §3.1
specifies `S2-long-open` as 300–1000 tokens, chosen so that prefill dilution (H5) has
something to bite on. As tokenised by the shipped `configs/prompts.jsonl` and recorded in the
raw records, S2 is **174–188 tokens** (median 183), and the other strata are S1 8–21,
S3 213–249, S4 114–369. Every S2 row in the sweep is therefore a valid measurement at a
length the design did not intend for it.

**This was left as documentation rather than repaired, deliberately.** Fixing it means editing
the prompt set, and the prompt set is an input to a run that has already happened: the raw
records store `prompt_id` and `prompt_tokens` but not the prompt text, so changing
`configs/prompts.jsonl` would sever the recorded run from the inputs that produced it, and
would require re-running every affected point to say anything about longer prompts. That is
an expansion of the benchmark, not a repair of it. The consequence is stated plainly instead:
**H5's magnitude is not testable from this run**, and the S2 results above are results about
174–188-token prompts.

**The cost model is scored, and the gap it was written to explain is not there.** Stage 0 ran
on the T4 and the predicted-vs-observed table exists (§5). Measured ρ = 1.147 and κ = 7.858
predict 0.395 / 0.517 / 0.574 against observed 0.452 / 0.528 / 0.592: a residual inside the
trial noise at every length. For H3 that is a worse outcome than a large residual would have
been, because H3 predicted a material gap and the gap is too small to attribute. The
two-stage structure of the plan is still what makes that statement possible: before Stage 0,
κ was inferred from the same trials it predicted.

**Stage 4 could not complete, and the reason is the host rather than the code.**
`tinyturbo profile` runs both cases in one process that loads the two models and then attaches
a CUDA trace to a whole generation; on a free Colab instance that process was killed by the
platform's memory limit - `/bin/bash: Killed`, exit **137** (128 + SIGKILL) - after both
models had loaded and generation had begun, with no exception raised. A kill rather than a
traceback is also why the first symptom was a step that appeared to hang: a child's stdout is
block-buffered into the capture pipe, so on a kill its header and any traceback are lost
while stderr has already been flushed. The defect is recorded, not worked around: shrinking
the 256-token case would change the case's stated rationale, and the honest alternatives are a
runtime with more memory or a bounded profiler schedule (both in §9).

**Figure coverage.** `plots/speedup_<run_id>.png` (per-length speedup summary) and
`plots/accepted_len_vs_speedup_<run_id>.png` (figure 2: accepted length against speedup, one
point per prompt × length) exist and regenerate from the raw records. The cost-model lines on
figure 2, the predicted-vs-observed table, and figure 3 (stacked per-step attribution) were
gated on a matching Stage 0 calibration, and all three now exist: figure 2 carries the model
lines, figure 3 exists as `plots/time_attribution_gpu-main-9e26c12945e6.png`, and the table is
printed by `report`. Removing the calibration artifact is what proves they depended on it -
with it, figure 2 hashes to `7be8b57ebcf630ae` and figure 3 exists; without it, figure 2
reverts to `53f71644a9f59cce` and no figure 3 is written. A baseline run and a benchmark
figure are different things, and drawing §5's curve from the sweep's own residual would make
figure 2 a restatement of its own data; the committed calibration is what prevents that.

## 8. Limitations

* **Two methods, one card, one session.** Baseline and `model_assisted` at γ = 5, on a T4, in
  one 24-minute Colab run. Three measured trials per point gives a direction per length and is
  thin for comparing adjacent points: the IQR at 32 tokens (0.169) exceeds the whole gap
  between the 32- and 256-token medians (0.140).
* **`decode_ms` is derived,** not measured: it is `e2e_ms` minus a separately measured prefill,
  so it is an upper bound on decode time. Every record says so in its `notes`.
* **The prompt set is ten hand-written prompts,** not a corpus, and the S2 defect above means
  the prompt-length axis is narrower than designed.
* **Hardware generality is unknown.** Everything here is one Turing-class card at float16 with
  `sdpa`. H8 is untested, and the draft-pass cost ratio that drives the whole result is a
  property of this card, this dtype and this library version as much as of the method.
* **The library is not a constant.** The result depends on transformers 5.3.0's candidate
  generator, including the two behaviours documented in `README.md` (`num_assistant_tokens`
  read from the assistant's own config, and `assistant_confidence_threshold` defaulting to
  0.4). A minor release that changes the drafting loop would change these numbers, which is
  why both are pinned in code, recorded per trial, and locked by tests.

## 9. Reproducing this, and what is still open

Everything already recorded reproduces with no GPU:

```bash
python -m pytest                                                     # 158 tests, no GPU, no network
python -m tinyturbo verify --config configs/gpu_main.yaml            # exits 0 on the recorded sweep
python -m tinyturbo report --config configs/gpu_main.yaml            # CSV + figures 1, 2 and 3 + the table
```

That third command now produces the whole set, because the T4's Stage 0 artifact is committed
beside the raw records. What that depends on is worth being explicit about, since it is the
one place where a derived figure is not merely derived: with
`results/calibration/gpu-main-9e26c12945e6-calibration.json` present, `report` writes figure 2
with the cost-model lines (sha256 `7be8b57ebcf630ae`) and figure 3
(`e09f5fb9b1d10d50`), and both regenerate byte-identically on a second run. Move the artifact
aside and the same command writes figure 2 without those lines (`53f71644a9f59cce`), writes no
figure 3, and prints the warning that says why. No GPU and no network are needed for any of
it, because the scoring reads a fingerprint rather than a device.

### Stage 4 has now produced an artifact, and it is a bounded one

On 2026-10-14T20:06:13Z, on a Kaggle notebook with a Tesla T4, `tinyturbo profile` traced both
cases with an 8-step window and wrote
`results/profiling/gpu-main-9e26c12945e6-profiling.json` (6 187 bytes, sha256
`8884e21fa4987264…`). The whole-generation trace that a free Colab session could not finish
(§7) is still not what produced it; the window is. What it recorded:

| case | arm | e2e ms | tokens | loop avg ms/token | target mean ms | ÷ τ_t | draft mean ms | ÷ τ_d | draft/target |
|---|---|---|---|---|---|---|---|---|---|
| 32 | baseline | 1359.3 | 32 | 42.48 | 39.60 | 1.77 | — | — | — |
| 32 | assisted | 2783.0 | 32 | 86.97 | 44.53 | 1.99 | 57.64 | 1.64 | 1.294 |
| 256 | baseline | 7207.5 | 256 | 28.15 | 41.91 | 1.87 | — | — | — |
| 256 | assisted | 14028.2 | 256 | 54.80 | 41.71 | 1.86 | 55.07 | 1.56 | 1.320 |

Every mean there is over **8 target forward passes** - one profiler step each, after 1 skipped
and 1 warmed up - so it is a window, not the run. The artifact says so itself:
`covers_whole_generation: false`, `attributed_ms` and `unattributed_ms` `null`, and a `null`
count for the whole-run `generate` annotation that the schedule's cycle end clips.

**It does not reconcile with Stage 0, and the artifact's own fingerprint says why it may not.**
Every labelled pass came out **1.6-2.0× the calibration's isolated primitive**, and the
mismatch is close to a *constant* added per pass rather than a proportional inflation: the
offsets are +17.2, +18.8, +19.5 ms on the target clock and +16.0, +22.4, +19.8 ms on the draft
and assisted-target clocks. The ratio says the same thing independently - the trace prices a
draft pass at **1.29-1.32×** a target pass where the calibration measured 1.572, and a single
additive offset of about 20 ms per pass is exactly what turns 1.572 into 1.29-1.32.

Two candidate causes are not separable from this artifact, and both are stated here rather
than picked between:

* **the profiler's own cost falls inside the measured region.** The window is precisely when
events are being collected (≈130k events for 4 steps on the CPU pair, route 3 below), and the
one number a slower card cannot explain points that way: the 256-token baseline's whole-loop
average is **28.2 ms/token, below** the 41.9 ms mean of its 8 recorded steps. The 32-token case
points the other way - a 42.5 ms/token loop average against a 39.6 ms window mean - so the
direction is not settled even by this internal check.
* **a different image.** This session ran driver 580.178.04 / CUDA 12.8 / torch 2.11.0+cu128,
where the calibration ran 580.82.07 / CUDA 13.0 / torch 2.11.0+cu130. `report`'s guard would
refuse to apply the calibration to these records, and on that basis alone the two sets of
numbers are not a measurement of one machine against a measurement of the same machine.

So the trace **does not decompose the residual** and none of its per-pass means are quoted as
evidence about the drafting loop. What it does establish on its own terms: the stage runs on
the real pair at float16/`sdpa`, both cases complete without the memory limit that killed the
whole-generation attempt, in-loop per-pass costs are tens of milliseconds, the assisted arm's
target pass is no cheaper than the baseline's, and this run's end-to-end ratio - assisted
**2.05×** and **1.95×** baseline - sits inside the sweep's range (§1).

### Closing it properly, and what is left

The remaining step is not "run Stage 4" any more, it is "run it on the same image as the
calibration". From a shell with the project present:

```bash
python -m tinyturbo profile --config configs/gpu_main.yaml --trace-active 8
python -m tinyturbo report  --config configs/gpu_main.yaml   # unchanged: it never reads the trace
```

Three routes, cheapest first, and the first is now the cheap retry:

1. **Same image as the calibration, windowed.** A session whose driver and torch build match
   the recorded sweep makes the trace comparable to Stage 0; the window is what makes that
   affordable, and route 2 already showed it fits. This is the one experiment that would turn
   the ±20 ms discrepancy above into a measurement of either the profiler or the image.
2. **Bound the window the profiler records.** `--trace-active N` traces the loop under
   `torch.profiler.schedule` and records `N` steps after `--trace-wait` skipped and
   `--trace-warmup` discarded ones, with one profiler step per target forward pass, so the
   event buffer scales with `N` rather than with the generation length:

   ```bash
   python -m tinyturbo profile --config configs/gpu_main.yaml --trace-active 4
   ```

   Measured on the CPU smoke pair of route 3, a 48-token assisted generation: a 4-step window
   holds **130 515** events where the whole trace holds **371 423**, and the same 4-step
   window on a **96**-token generation holds **130 515** again - the same number, because the
   buffer is set by the window and not by the run. That is the property the route rests on.
   A window is not the whole generation, so the artifact says so: it records the window it
   used and `covers_whole_generation: false`, and leaves the whole-run residual **unset**
   rather than computing it from the part of the run the window does not cover. The
   per-forward means it does measure - mean target cost, mean draft cost - are the two the
   calibration is built to predict. They are now measured, and they disagree with it for the
   two reasons listed above; that disagreement is the result, not a gap in the table.
3. **Exercise the path with no GPU at all.** `configs/profile_smoke.yaml` is a smoke
   configuration - tiny pair, CPU, one case - and the artifact it writes is a check that the
   stage runs end to end and describes itself correctly, not a result about the drafting loop.
   It is the same status as the CPU run in `results/raw/`; see `results/README.md`.

Stage 4 remains deliberately optional: it is the only thing that could decompose the per-step
residual, and that residual is now inside the trial noise (§5), so it was never needed to score
the model. It has now run in its bounded form, and what that bought is the ±20 ms discrepancy
above rather than a decomposition. Nothing in `report` reads it either - figure 3 and the
predicted-vs-observed table come from Stage 0 and the recorded counts.

Both `calibrate` and `profile` write under `results/calibration/` and `results/profiling/`,
never into `results/raw/`, and both record their own hardware fingerprint. Run `calibrate` on
the same image as the sweep (driver 580.82.07, torch 2.11.0+cu130) or `report` will -
correctly - refuse to apply the calibration and say why. That guard is code, not convention,
and the filed artifact above was checked against the sweep's manifest field by field before
any of its numbers were used.

## 10. What a reader should take away

Assisted decoding lost by about 2× on this hardware, at every prompt and every generation
length, and the reason is legible: accepted length behaved exactly as the mechanism predicts
and the drafter cost about 1.6× what the target costs per forward pass, roughly 20× what the
parameter ratio suggests. The model was right about *what* drives the speedup and wrong about
the *level*, because the level is set by a drafting pass that is far more expensive than its
size implies. Stage 0 has now measured that drafting pass (1.572× a target pass) and priced
the verification premium (1.6% of an iteration), and the result is a prediction that matches
the sweep within its own noise - so the study's answer to "why" is a measured ratio rather
than a hypothesis. What it does not have is a decomposition of the tiny residual that remains,
which is what Stage 4 would have given and which the host's memory limit blocked (§7).
