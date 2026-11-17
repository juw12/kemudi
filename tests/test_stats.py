import pytest

from kemudi.stats import auc, bootstrap_ci, distinct_2, quoted_share, spearman, usable_range


def test_auc_separated_mixed_and_tied():
    assert auc([0.1, 0.2], [0.8, 0.9]) == 1.0
    assert auc([0.8, 0.9], [0.1, 0.2]) == 0.0
    assert auc([0.5], [0.5]) == 0.5
    assert auc([0.1, 0.6], [0.4, 0.9]) == 0.75


def test_bootstrap_interval_contains_the_mean_and_is_repeatable():
    values = [0.1, 0.4, 0.5, 0.9, 0.3, 0.7]
    low, high = bootstrap_ci(values)
    assert low < sum(values) / len(values) < high
    assert (low, high) == bootstrap_ci(values)
    assert bootstrap_ci([0.5] * 5) == (0.5, 0.5)


def test_spearman_follows_order_not_size():
    assert spearman([1, 2, 3, 4], [10, 20, 400, 9000]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 2, 3], [5, 5, 5]) == 0.0
    assert spearman([1, 1, 2, 2], [1, 2, 3, 4]) == pytest.approx(0.894, abs=1e-3)


def test_distinct_2_drops_when_text_repeats():
    assert distinct_2("the cat sat on a mat") == 1.0
    assert distinct_2("go go go go go") == 0.25
    assert distinct_2("one") == 1.0


def test_quoted_share_counts_characters_inside_quotes():
    assert quoted_share("") == 0.0
    assert quoted_share("no speech here") == 0.0
    assert quoted_share('"abcd"') == pytest.approx(4 / 6)
    assert quoted_share("\u201cabcd\u201d xx") == pytest.approx(4 / 9)
    assert quoted_share('ab "cd') == pytest.approx(2 / 6)


def test_usable_range_stops_at_the_first_failing_dose():
    doses = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
    perplexity = [1.2, 1.9, 1.1, 1.0, 1.2, 1.4, 2.5]
    distinct = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    assert usable_range(doses, perplexity, distinct, 1.5, 0.85) == (-0.1, 0.2)


def test_usable_range_also_stops_when_text_starts_repeating():
    doses = [0.0, 0.1, 0.2]
    assert usable_range(doses, [1.0, 1.1, 1.1], [1.0, 0.9, 0.5], 1.5, 0.85) == (0.0, 0.1)
    assert usable_range(doses, [1.0, 9.0, 1.0], [1.0, 1.0, 1.0], 1.5, 0.85) == (0.0, 0.0)
