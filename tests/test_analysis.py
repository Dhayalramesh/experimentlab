"""
Validates the core statistical functions three ways:
1. Sanity checks against hand-computable or scipy-equivalent values.
2. A Monte Carlo check that the z-test actually holds its nominal 5%
   false-positive rate when there is no true effect (the single most
   important property a hypothesis test must have).
3. Power-analysis formula checks against known relationships (larger MDE
   needs fewer samples; higher target power needs more samples).
"""
import numpy as np
import pytest

from experimentlab.analysis import two_proportion_ztest, welch_ttest, correct_multiple_tests
from experimentlab.power_analysis import required_sample_size, achieved_power
from experimentlab.simulate import simulate_experiment
from experimentlab.segment_analysis import analyze_by_segment, check_simpsons_paradox


def test_two_proportion_ztest_detects_no_effect_when_identical():
    rng = np.random.default_rng(0)
    control = (rng.random(5000) < 0.10).astype(int)
    treatment = (rng.random(5000) < 0.10).astype(int)
    result = two_proportion_ztest(control, treatment)
    assert abs(result.absolute_lift) < 0.03
    assert result.p_value > 0.01  # should not spuriously reject with this much data at same rate


def test_two_proportion_ztest_detects_real_effect():
    rng = np.random.default_rng(1)
    control = (rng.random(5000) < 0.10).astype(int)
    treatment = (rng.random(5000) < 0.20).astype(int)  # large, obvious effect
    result = two_proportion_ztest(control, treatment)
    assert result.absolute_lift > 0.05
    assert result.p_value < 0.001


def test_ztest_false_positive_rate_matches_nominal_alpha():
    """Monte Carlo check: under the null (no real effect), the fraction of
    simulated experiments with p < 0.05 should be close to 5%."""
    rng = np.random.default_rng(123)
    n_sims = 600
    false_positives = 0
    for _ in range(n_sims):
        control = (rng.random(800) < 0.10).astype(int)
        treatment = (rng.random(800) < 0.10).astype(int)
        result = two_proportion_ztest(control, treatment)
        if result.p_value < 0.05:
            false_positives += 1
    fpr = false_positives / n_sims
    # allow a generous band since this is a finite Monte Carlo estimate
    assert 0.02 < fpr < 0.09, f"false-positive rate {fpr:.3f} too far from nominal 0.05"


def test_welch_ttest_detects_mean_shift():
    rng = np.random.default_rng(2)
    control = rng.normal(420, 35, 2000)
    treatment = rng.normal(440, 35, 2000)
    result = welch_ttest(control, treatment)
    assert result.diff == pytest.approx(treatment.mean() - control.mean())
    assert result.p_value < 0.001


def test_correct_multiple_tests_is_more_conservative_than_raw():
    p_values = [0.01, 0.04, 0.20]
    out = correct_multiple_tests(p_values, alpha=0.05)
    assert len(out["p_adjusted"]) == 3
    # BH-adjusted p-values are never smaller than the raw p-values
    assert all(adj >= raw - 1e-9 for adj, raw in zip(out["p_adjusted"], p_values))


def test_required_sample_size_decreases_with_larger_mde():
    n_small_effect = required_sample_size(baseline_rate=0.10, mde=0.01)
    n_large_effect = required_sample_size(baseline_rate=0.10, mde=0.05)
    assert n_large_effect < n_small_effect


def test_required_sample_size_increases_with_higher_power_target():
    n_80 = required_sample_size(baseline_rate=0.10, mde=0.02, power=0.80)
    n_95 = required_sample_size(baseline_rate=0.10, mde=0.02, power=0.95)
    assert n_95 > n_80


def test_achieved_power_increases_with_sample_size():
    low = achieved_power(baseline_rate=0.10, mde=0.01, n_per_arm=500)
    high = achieved_power(baseline_rate=0.10, mde=0.01, n_per_arm=20000)
    assert high > low
    assert 0 <= low <= 1 and 0 <= high <= 1


def test_simulate_experiment_shape_and_balance():
    df = simulate_experiment()
    assert set(df["variant"].unique()) == {"control", "treatment"}
    assert set(df["segment"].unique()) == {"new", "returning"}
    counts = df["variant"].value_counts()
    # roughly balanced random assignment
    ratio = counts.min() / counts.max()
    assert ratio > 0.9


def test_segment_analysis_runs_and_paradox_check_is_consistent():
    df = simulate_experiment(seed=5)
    seg_df = analyze_by_segment(df)
    assert set(seg_df["segment"]) == {"new", "returning"}

    control = df.loc[df.variant == "control", "converted"].to_numpy()
    treatment = df.loc[df.variant == "treatment", "converted"].to_numpy()
    pooled_lift = treatment.mean() - control.mean()

    result = check_simpsons_paradox(pooled_lift, seg_df)
    assert result["pooled_sign"] in {"positive", "negative", "zero"}
    assert isinstance(result["heterogeneous_effect_detected"], bool)
