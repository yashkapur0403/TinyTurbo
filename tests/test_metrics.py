"""Tests for the arithmetic. Pure functions, so these are fast and need nothing."""

from __future__ import annotations

import pytest

from tinyturbo.metrics import (
    accepted_len_mean,
    draft_tokens_per_step,
    exact_match,
    first_divergence_index,
    generated_token_ids,
    interquartile_range,
    median,
    percentile,
    theoretical_speedup,
    tokens_per_second,
)


def test_generated_token_ids_strips_the_prompt():
    assert generated_token_ids([10, 11, 12, 13], 2) == [12, 13]
    assert generated_token_ids([10, 11], 2) == []


def test_generated_token_ids_rejects_impossible_prompt_length():
    with pytest.raises(ValueError, match="outside the sequence"):
        generated_token_ids([1, 2, 3], 4)


def test_exact_match():
    assert exact_match([1, 2, 3], [1, 2, 3]) is True
    assert exact_match([1, 2, 3], [1, 2, 4]) is False
    assert exact_match([1, 2, 3], [1, 2, 3, 4]) is False, "different lengths cannot be a match"


def test_first_divergence_index():
    assert first_divergence_index([1, 2, 3], [1, 2, 3]) is None
    assert first_divergence_index([1, 2, 3], [1, 9, 3]) == 1
    assert first_divergence_index([1, 2], [1, 2, 3]) == 2


def test_accepted_len_is_tokens_per_target_forward_pass():
    # 12 tokens emitted by 4 target forward passes: 3 tokens per verification pass.
    assert accepted_len_mean(12, 4) == 3.0
    # Plain greedy decoding emits exactly one token per pass.
    assert accepted_len_mean(5, 5) == 1.0
    assert accepted_len_mean(5, 0) is None, "no counts means no value, not zero"


def test_draft_tokens_per_step():
    assert draft_tokens_per_step(20, 5) == 4.0
    assert draft_tokens_per_step(0, 5) == 0.0
    assert draft_tokens_per_step(5, 0) is None


def test_tokens_per_second():
    assert tokens_per_second(100, 2000.0) == 50.0
    assert tokens_per_second(0, 100.0) == 0.0
    assert tokens_per_second(10, 0.0) is None


def test_theoretical_speedup_matches_the_model():
    # 4 tokens per pass, one verification pass worth one baseline pass, no draft cost.
    assert theoretical_speedup(4.0) == pytest.approx(4.0)
    # A draft costing half a target pass per iteration: 4 / (1 + 0.5).
    assert theoretical_speedup(4.0, draft_cost_ratio=0.5) == pytest.approx(4.0 / 1.5)
    # Verifying more tokens than one is not free.
    assert theoretical_speedup(4.0, verification_ratio=1.5) == pytest.approx(4.0 / 1.5)
    with pytest.raises(ValueError, match="must be positive"):
        theoretical_speedup(4.0, verification_ratio=0.0, draft_cost_ratio=0.0)


def test_theoretical_speedup_is_increasing_in_accepted_length():
    values = [theoretical_speedup(length, draft_cost_ratio=0.25) for length in (1, 2, 4, 8)]
    assert values == sorted(values)


def test_percentile_matches_the_interpolated_convention():
    values = [1.0, 2.0, 3.0, 4.0]
    assert percentile(values, 0) == 1.0
    assert percentile(values, 100) == 4.0
    assert percentile(values, 50) == pytest.approx(2.5)
    assert percentile(values, 25) == pytest.approx(1.75)
    assert percentile([7.0], 90) == 7.0
    with pytest.raises(ValueError):
        percentile([], 50)


def test_median_and_iqr_are_robust_to_one_slow_trial():
    typical = [10.0, 10.5, 11.0, 11.5, 12.0]
    with_outlier = typical + [500.0]
    # One stalled trial moves the mean enormously and the median hardly at all, which
    # is why the report uses medians.
    assert median(typical) == 11.0
    assert median(with_outlier) == pytest.approx(11.25)
    assert interquartile_range(typical) > 0
    assert interquartile_range(typical) < interquartile_range(with_outlier)
