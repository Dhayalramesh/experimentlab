"""
Validates the peeking-problem demonstration itself: naive repeated peeking
must inflate the false-positive rate well above nominal alpha, while
fixed-horizon testing stays close to nominal, and Bonferroni-corrected
peeking stays at or below nominal (conservative by construction).
"""
from experimentlab.peeking import simulate_false_positive_rate


def test_naive_peeking_inflates_false_positive_rate():
    result = simulate_false_positive_rate(
        n_simulations=800, n_looks=10, max_n_per_arm=1500, alpha=0.05, seed=11
    )
    # naive peeking should be meaningfully above the nominal 5%
    assert result["naive_peeking_fpr"] > result["nominal_alpha"] * 1.5


def test_fixed_horizon_stays_near_nominal_alpha():
    result = simulate_false_positive_rate(
        n_simulations=800, n_looks=10, max_n_per_arm=1500, alpha=0.05, seed=11
    )
    assert abs(result["fixed_horizon_fpr"] - result["nominal_alpha"]) < 0.04


def test_bonferroni_correction_is_conservative():
    result = simulate_false_positive_rate(
        n_simulations=800, n_looks=10, max_n_per_arm=1500, alpha=0.05, seed=11
    )
    assert result["bonferroni_corrected_fpr"] <= result["naive_peeking_fpr"]
    assert result["bonferroni_corrected_fpr"] <= result["nominal_alpha"] + 0.03


def test_more_looks_without_correction_means_more_inflation():
    few_looks = simulate_false_positive_rate(
        n_simulations=600, n_looks=3, max_n_per_arm=1500, seed=22
    )
    many_looks = simulate_false_positive_rate(
        n_simulations=600, n_looks=15, max_n_per_arm=1500, seed=22
    )
    assert many_looks["naive_peeking_fpr"] >= few_looks["naive_peeking_fpr"] - 0.03
