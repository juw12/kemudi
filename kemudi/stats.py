"""Small statistics used by the measurements. Pure Python, no model needed."""

from __future__ import annotations

import math
import random


def auc(low: list[float], high: list[float]) -> float:
    """Chance that a random `high` value is above a random `low` value (ties count half)."""
    wins = sum(1.0 if h > l else 0.5 if h == l else 0.0 for h in high for l in low)
    return wins / (len(high) * len(low))


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def bootstrap_ci(values: list[float], seed: int = 0, draws: int = 2000) -> tuple[float, float]:
    """95% interval of the mean, resampling the values with replacement."""
    picker = random.Random(seed)
    count = len(values)
    means = sorted(mean([values[picker.randrange(count)] for _ in range(count)]) for _ in range(draws))
    return means[int(0.025 * draws)], means[int(0.975 * draws) - 1]


def spearman(xs: list[float], ys: list[float]) -> float:
    """Rank correlation: 1 means y always rises with x."""
    rank_x, rank_y = _ranks(xs), _ranks(ys)
    mean_x, mean_y = mean(rank_x), mean(rank_y)
    covariance = sum((a - mean_x) * (b - mean_y) for a, b in zip(rank_x, rank_y))
    spread = math.sqrt(sum((a - mean_x) ** 2 for a in rank_x) * sum((b - mean_y) ** 2 for b in rank_y))
    return covariance / spread if spread else 0.0


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        for position in range(start, end + 1):
            ranks[order[position]] = (start + end) / 2 + 1
        start = end + 1
    return ranks


def distinct_2(text: str) -> float:
    """Share of word pairs that are different from each other. Low values mean the text repeats."""
    words = text.lower().split()
    pairs = list(zip(words, words[1:]))
    return len(set(pairs)) / len(pairs) if pairs else 1.0


QUOTE_PAIRS = {'"': '"', "\u201c": "\u201d"}


def quoted_share(text: str) -> float:
    """Share of characters inside double quotation marks. An unclosed quote runs to the end."""
    inside = 0
    closer = None
    for character in text:
        if closer is None:
            closer = QUOTE_PAIRS.get(character)
        elif character == closer:
            closer = None
        else:
            inside += 1
    return inside / len(text) if text else 0.0


def usable_range(
    doses: list[float], perplexity_ratio: list[float], distinct_ratio: list[float], max_perplexity_ratio: float, min_distinct_ratio: float
) -> tuple[float, float]:
    """Widest run of doses around zero where the text stays fluent and does not start repeating.

    The run grows outward from dose 0 one step at a time and stops at the first dose that breaks
    either limit, so a dose that passes beyond a failing one is not counted.
    """
    order = sorted(range(len(doses)), key=lambda index: doses[index])
    zero = next(position for position, index in enumerate(order) if doses[index] == 0)

    def passes(index: int) -> bool:
        return perplexity_ratio[index] <= max_perplexity_ratio and distinct_ratio[index] >= min_distinct_ratio

    low = high = zero
    while low > 0 and passes(order[low - 1]):
        low -= 1
    while high < len(order) - 1 and passes(order[high + 1]):
        high += 1
    return doses[order[low]], doses[order[high]]
