"""
Demonstrates the "peeking problem": checking a test's p-value repeatedly
as data comes in, and stopping as soon as it crosses 0.05, inflates the
true false-positive rate far above the nominal 5% -- even when there is
NO real effect at all. This is one of the most common and most costly
statistical mistakes in product experimentation, and a data scientist
needs to be able to both demonstrate it and correct for it.

Two corrections are compared against the naive approach:
1. Fixed-horizon testing: decide the sample size in advance (via power
   analysis) and look exactly once, at the end. Simple, but requires
   discipline and sacrifices the ability to stop early.
2. Bonferroni-corrected sequential testing: if you insist on looking K
   times, test each look at alpha/K instead of alpha. Conservative, but
   valid at any look.
"""
import numpy as np
from scipy import stats


def _p_value_at_n(control: np.ndarray, treatment: np.ndarray) -> float:
    n1, n2 = len(control), len(treatment)
    if n1 == 0 or n2 == 0:
        return 1.0
    p1, p2 = control.mean(), treatment.mean()
    p_pool = (control.sum() + treatment.sum()) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = (p2 - p1) / se
    return 2 * (1 - stats.norm.cdf(abs(z)))


def simulate_false_positive_rate(
    n_simulations: int,
    n_looks: int,
    max_n_per_arm: int,
    true_rate: float = 0.10,
    alpha: float = 0.05,
    seed: int = 7,
) -> dict:
    """
    Runs `n_simulations` experiments where control and treatment have the
    IDENTICAL true conversion rate (no real effect), "peeking" at the
    p-value `n_looks` times as data accrues. Returns the empirical
    false-positive rate under three stopping rules: naive peeking (stop
    at the first look where p < alpha), fixed-horizon (look only at the
    end), and Bonferroni-corrected peeking (stop at the first look where
    p < alpha / n_looks).
    """
    rng = np.random.default_rng(seed)
    look_points = np.linspace(max_n_per_arm / n_looks, max_n_per_arm, n_looks).astype(int)

    naive_false_positives = 0
    fixed_horizon_false_positives = 0
    corrected_false_positives = 0
    bonferroni_alpha = alpha / n_looks

    for _ in range(n_simulations):
        control = rng.random(max_n_per_arm) < true_rate
        treatment = rng.random(max_n_per_arm) < true_rate  # same true rate: null is true

        naive_stopped = False
        corrected_stopped = False
        for n in look_points:
            p = _p_value_at_n(control[:n], treatment[:n])
            if not naive_stopped and p < alpha:
                naive_false_positives += 1
                naive_stopped = True
            if not corrected_stopped and p < bonferroni_alpha:
                corrected_false_positives += 1
                corrected_stopped = True

        final_p = _p_value_at_n(control, treatment)
        if final_p < alpha:
            fixed_horizon_false_positives += 1

    return {
        "n_simulations": n_simulations,
        "n_looks": n_looks,
        "nominal_alpha": alpha,
        "naive_peeking_fpr": naive_false_positives / n_simulations,
        "fixed_horizon_fpr": fixed_horizon_false_positives / n_simulations,
        "bonferroni_corrected_fpr": corrected_false_positives / n_simulations,
    }


if __name__ == "__main__":
    result = simulate_false_positive_rate(n_simulations=2000, n_looks=10, max_n_per_arm=2000)
    for k, v in result.items():
        print(f"{k}: {v}")
