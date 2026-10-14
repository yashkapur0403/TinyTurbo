# Colab session artefacts — `slice-gpu-902bc285880d`

Verbatim copies of what a real Colab session downloaded, kept so the numbers in the
derived CSV can be traced back to the session that produced them. Nothing in this
directory is edited by hand and nothing in it is a source of truth: the evidence is
`results/raw/slice-gpu-902bc285880d.jsonl`, and the canonical figure is
`plots/speedup_slice-gpu-902bc285880d.png`.

| | |
|---|---|
| run_id | `slice-gpu-902bc285880d` |
| config | `configs/slice_gpu.yaml` (fingerprint `902bc285880d6299…`, verified to match this checkout) |
| command | `python -m tinyturbo run --config configs/slice_gpu.yaml --trials 2` |
| hardware | Tesla T4, compute capability 7.5, 14912 MiB, driver 580.82.07, CUDA 13.0, cudnn 92700 |
| software | Linux, Python 3.13.15, torch 2.11.0+cu130, transformers 5.3.0 |
| resolved | device `cuda`, dtype `float16`, attention `sdpa` |
| created (UTC) | 2026-10-14T11:59:10 |
| session | `notebooks/colab_gpu_smoke.ipynb`, browser workflow, uploaded `TinyTurbo.zip` |

## Files

| file | what it is | sha256 |
|---|---|---|
| `benchmark_results.csv` | the derived table as Colab produced it | `2fc0821b22083ffd…` |
| `environment_fingerprint.json` | standalone hardware/scope fingerprint (the notebook writes it separately because it is not reproducible after the session ends) | `4107e6956bf5c664…` |
| `verify_output.txt` | stdout of `tinyturbo verify`, captured before any number was read | `4c3f7e33344b5a9a…` |
| `figure.png` | the figure exactly as Colab's matplotlib encoded it (see the note below) | `b7158da1d9259b0b…` |

The other session download is filed where the tooling expects it:
`results/raw/slice-gpu-902bc285880d.jsonl` (sha256 `77b8f6bf7ed57508…`).

**Why the figure is filed twice, in two different forms.** `plots/speedup_slice-gpu-902bc285880d.png`
is the path `tinyturbo report` regenerates, so it is a *derived* file: running the documented
report command rewrites it with whatever this machine's matplotlib produces, which is
pixel-identical but not byte-identical to Colab's encoder output. The checksum above therefore
describes `figure.png`, the copy in this directory, which nothing regenerates into and which is
the only place this session's bytes survive a re-report. Treat `plots/` as reproducible output
and `results/sessions/<run_id>/` as the filed record; that split is the same one the CSV already
uses, and it is why this directory is not written to by any command in the toolchain.

`benchmark_results.csv` is duplicated here rather than left at `results/benchmark_results.csv`
because that path is shared by every config: regenerating the GPU report overwrites the CPU
smoke run's table. See `results/README.md`.

## What was checked when these were filed

Run locally against the filed raw JSONL, on the machine that wrote the notebook:

* **`tinyturbo verify` exits 0** and reproduces `verify_output.txt` line for line, apart
  from the `raw records :` line, which names where the file is being read from.
* **The raw file's config fingerprint matches `configs/slice_gpu.yaml` in this checkout**,
  so the records were produced by this configuration and not a stale copy in the bundle.
* **Every field of the CSV regenerates identically** from the raw records. Two things about
  the bytes are worth knowing. Colab's pandas writes LF where Windows pandas writes CRLF,
  which is a platform artefact and not a difference in content. And the derived schema later
  gained a `max_new_tokens` column, added because one sweep now covers three generation
  lengths and their rows would otherwise be indistinguishable: regenerating this file today
  therefore adds that single column, with the value `64` in every row and every other field
  unchanged, field by field. The figure still regenerates pixel-for-pixel.
* **The figure regenerates pixel-identically** — 0 of 486,000 pixels differ at 900×540.
  The PNG byte sizes differ (29,444 local vs 35,548 from Colab), which is encoder variation,
  not a different plot. `figure.png` in this directory holds Colab's bytes, so the sha256
  above keeps describing the filed figure even after `tinyturbo report` has rewritten the
  derived copy in `plots/`; the two are pixel-for-pixel the same image either way.

## What this run measures

One prompt (`s2-01-long-open`, 183 tokens), 2 discarded warm-ups and 2 measured trials per
arm, 64 new tokens each. This is a smoke run of the real model pair: it is enough to check
that the pipeline works on CUDA and to see the sign of the effect, not enough to estimate
its size.

| | baseline | assisted |
|---|---|---|
| tokens per target forward pass | 1.00 (by definition) | 3.56 |
| draft passes per target forward pass | 0.00 | 4.83 (γ = 5) |
| e2e per trial | 1639.5, 1658.3 ms | 4007.3, 4009.6 ms |
| peak allocated | 3610.1 MiB | 3620.2 MiB |

**Median end-to-end speedup: 0.411 (IQR 0.002)** — assisted decoding was ~2.4× *slower*
than plain greedy here. Decode-only speedup is 0.402, so this is not prefill dilution.

**Exactness holds**: `exact_match: True` for both measured trials, so the lossless property
the plan asserts (H6) survived contact with fp16 on a T4.

**Where the time went, as a hypothesis.** Subtracting the baseline's 24.9 ms per target
forward pass from the assisted arm's 3942 ms of decode leaves ~3494 ms spread over 87 draft
passes — about 40 ms each, i.e. a 135M draft pass apparently costing ~1.6× a 1.7B target
pass, roughly 20× more than the 0.079 parameter ratio would suggest. Putting that implied
ratio into the cost model of `docs/PLAN.md` §5 gives `b / (1 + κ) = 3.56 / (1 + 8.0) = 0.40`
against the 0.411 observed, and it points at the drafting loop rather than at verification.

**This is arithmetic over recorded counts, and it is a hypothesis, not a finding.** Neither
the cost model nor a profiler has been run: Stage 0 calibration and a Stage 4 trace are what
would test it, and until they do, the drafting-loop cost stays a candidate explanation rather
than a measured one. Nothing in this repository should be read as a profiler result.
