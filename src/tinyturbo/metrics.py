"""Pure functions over measurements: no torch, no files, no side effects.

Every formula the report relies on lives here, so the arithmetic can be tested on a
laptop with no GPU. If a number in the CSV is wrong, the bug is in this file.
"""

from __future__ import annotations

import math


def generated_token_ids(sequence: list[int], prompt_length: int) -> list[int]:
    """The tokens the model produced, with the prompt removed."""
    if prompt_length < 0 or prompt_length > len(sequence):
        raise ValueError(f"prompt_length {prompt_length} is outside the sequence (len {len(sequence)})")
    return sequence[prompt_length:]


def exact_match(first: list[int], second: list[int]) -> bool:
    """Whether two generated token sequences are identical.

    Assisted greedy decoding is provably lossless: with the same prompt, the same
    dtype and greedy decoding, the output must be token-for-token identical to plain
    greedy. A False here is a bug to investigate, not a result to report.
    """
    return list(first) == list(second)


def first_divergence_index(first: list[int], second: list[int]) -> int | None:
    """Index of the first token where two sequences differ, or None if identical.

    If one sequence is a prefix of the other, the index just past the shorter one is
    returned: that is where they start to disagree about what comes next.
    """
    for index, (left, right) in enumerate(zip(first, second)):
        if left != right:
            return index
    if len(first) != len(second):
        return min(len(first), len(second))
    return None


def accepted_len_mean(tokens_generated: int, target_forwards: int) -> float | None:
    """Tokens produced per target-model forward pass.

    This is the mechanism behind assisted decoding, and it is deliberately not called
    an acceptance rate. One target pass that validates `n` drafted tokens and then
    corrects the first wrong one emits n + 1 tokens, so the value is >= 1 whenever
    anything was generated at all. For plain greedy decoding it is 1.0 by definition,
    which makes it a useful sanity check on both arms.
    """
    if target_forwards <= 0:
        return None
    return tokens_generated / target_forwards


def draft_tokens_per_step(assistant_forwards: int, target_forwards: int) -> float | None:
    """Assistant forward passes per target forward pass.

    Recorded as an observation, not an assertion. It is expected to be close to the
    requested lookahead, and a much smaller value means either that the drafting loop
    stopped early or that the requested lookahead never reached the assistant at all.
    """
    if target_forwards <= 0:
        return None
    return assistant_forwards / target_forwards


def tokens_per_second(tokens_generated: int, elapsed_ms: float) -> float | None:
    """Throughput over an interval, in tokens per second."""
    if elapsed_ms <= 0:
        return None
    return tokens_generated / (elapsed_ms / 1000.0)


def theoretical_speedup(
    accepted_len: float, verification_ratio: float = 1.0, draft_cost_ratio: float = 0.0
) -> float:
    """The speedup the draft-and-verify loop should give, from first principles.

    In one iteration the target model emits `accepted_len` tokens while spending one
    verification pass plus the drafts that fed it, so

        speedup ~= accepted_len / (verification_ratio + draft_cost_ratio)

    where `verification_ratio` is the cost of verifying several drafted tokens relative
    to one single-token target pass, and `draft_cost_ratio` is the total cost of all
    draft passes in that iteration relative to the same single-token pass. On
    memory-bandwidth-bound hardware at batch size 1 the verification ratio is close to
    1, which is the whole reason the method works.

    This is a model, not a law. The report compares it against measured speedups and
    reports the residual, because the gap is the interesting part.
    """
    denominator = verification_ratio + draft_cost_ratio
    if denominator <= 0:
        raise ValueError("verification_ratio + draft_cost_ratio must be positive")
    return accepted_len / denominator


def median(values: list[float]) -> float:
    """The 50th percentile."""
    return percentile(values, 50.0)


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile, matching the common (numpy) convention.

    Implemented explicitly rather than imported so the definition is visible: with
    very few trials the interpolation choice changes the reported number.
    """
    if not values:
        raise ValueError("percentile() needs at least one value")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (q / 100.0)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def interquartile_range(values: list[float]) -> float:
    """Spread of the middle half of the measurements (75th minus 25th percentile).

    Used instead of a standard deviation because latency distributions are skewed:
    a single slow trial can move a mean a long way and a median almost not at all.
    """
    return percentile(values, 75.0) - percentile(values, 25.0)
