import numpy as np
import pytest

from synthdet import eval as E


def test_perfect_separation():
    s = np.r_[np.zeros(20), np.ones(20)]
    y = np.r_[np.zeros(20, int), np.ones(20, int)]
    assert E.eer(s, y) == 0.0
    assert E.auc(s, y) == 1.0


def test_inverted_scores():
    s = np.r_[np.ones(20), np.zeros(20)]
    y = np.r_[np.zeros(20, int), np.ones(20, int)]
    assert E.auc(s, y) == 0.0
    assert E.eer(s, y) == 1.0


def test_chance_level_with_ties():
    y = np.r_[np.zeros(50, int), np.ones(50, int)]
    s = np.zeros(100)
    assert E.auc(s, y) == 0.5
    assert E.eer(s, y) == pytest.approx(0.5)


def test_eer_known_case():
    # one real above the fake cluster and one fake below the real cluster
    s = np.array([0.1, 0.2, 0.3, 0.9, 0.4, 0.8, 0.85, 0.95])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    e, thr = E.eer_and_threshold(s, y)
    assert e == pytest.approx(0.25)
    assert 0.3 <= thr <= 0.9


def test_auc_matches_brute_force():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    s = rng.normal(y * 0.7, 1.0)
    pairs = [(a > b) + 0.5 * (a == b) for a in s[y == 1] for b in s[y == 0]]
    assert E.auc(s, y) == pytest.approx(np.mean(pairs))


def test_single_class_raises():
    with pytest.raises(ValueError):
        E.auc(np.arange(5.0), np.zeros(5, int))


def test_bootstrap_interval_contains_point_and_narrows_with_n():
    rng = np.random.default_rng(1)

    def make(n):
        y = np.r_[np.zeros(n, int), np.ones(n, int)]
        return rng.normal(y * 1.0, 1.0), y

    s, y = make(50)
    ci = E.bootstrap_ci(E.auc, s, y, n_boot=300)
    assert ci["lo"] <= ci["value"] <= ci["hi"]
    s2, y2 = make(500)
    ci2 = E.bootstrap_ci(E.auc, s2, y2, n_boot=300)
    assert ci2["hi"] - ci2["lo"] < ci["hi"] - ci["lo"]


def test_bootstrap_is_seeded():
    s = np.random.default_rng(2).normal(size=60)
    y = np.r_[np.zeros(30, int), np.ones(30, int)]
    assert E.bootstrap_ci(E.eer, s, y, 100, seed=3) == E.bootstrap_ci(E.eer, s, y, 100, seed=3)


def test_far_and_frr():
    assert E.false_accept_rate([0.1, 0.5, 0.9], 0.5) == pytest.approx(1 / 3)
    assert E.false_reject_rate([0.1, 0.5, 0.9], 0.5) == pytest.approx(2 / 3)
