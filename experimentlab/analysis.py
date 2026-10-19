"""
Core hypothesis-testing and estimation functions for analyzing a completed
experiment: two-proportion z-test for the primary conversion metric,
Welch's t-test for a continuous guardrail metric, confidence intervals for
both, and Benjamini-Hochberg correction for testing multiple metrics at
once (primary + guardrails) without inflating the overall false-positive
rate.
"""
from dataclasses import dataclass

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests


@dataclass
class ProportionTestResult:
    control_rate: float
    treatment_rate: float
    absolute_lift: float
    relative_lift: float
    z_stat: float
    p_value: float
    ci_low: float
    ci_high: float


def two_proportion_ztest(control: np.ndarray, treatment: np.ndarray) -> ProportionTestResult:
    """control/treatment: 0/1 arrays of per-user conversion outcomes."""
    n1, n2 = len(control), len(treatment)
    p1, p2 = control.mean(), treatment.mean()

    p_pool = (control.sum() + treatment.sum()) / (n1 + n2)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se_pool if se_pool > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    se_diff = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    diff = p2 - p1
    ci_low = diff - 1.96 * se_diff
    ci_high = diff + 1.96 * se_diff

    return ProportionTestResult(
        control_rate=p1,
        treatment_rate=p2,
        absolute_lift=diff,
        relative_lift=(diff / p1) if p1 > 0 else float("nan"),
        z_stat=z,
        p_value=p_value,
        ci_low=ci_low,
        ci_high=ci_high,
    )


@dataclass
class MeanTestResult:
    control_mean: float
    treatment_mean: float
    diff: float
    t_stat: float
    p_value: float
    ci_low: float
    ci_high: float


def welch_ttest(control: np.ndarray, treatment: np.ndarray) -> MeanTestResult:
    """Welch's t-test (unequal variance) for a continuous guardrail metric."""
    t_stat, p_value = stats.ttest_ind(treatment, control, equal_var=False)
    diff = treatment.mean() - control.mean()

    n1, n2 = len(control), len(treatment)
    se = np.sqrt(control.var(ddof=1) / n1 + treatment.var(ddof=1) / n2)
    ci_low = diff - 1.96 * se
    ci_high = diff + 1.96 * se

    return MeanTestResult(
        control_mean=control.mean(),
        treatment_mean=treatment.mean(),
        diff=diff,
        t_stat=t_stat,
        p_value=p_value,
        ci_low=ci_low,
        ci_high=ci_high,
    )


def correct_multiple_tests(p_values: list[float], alpha: float = 0.05) -> dict:
    """
    Benjamini-Hochberg FDR correction. Experiments rarely test one metric
    in isolation (primary + guardrails), and testing N metrics at
    alpha=0.05 each inflates the chance of at least one false positive
    well above 5% -- this corrects for that.
    """
    reject, p_adj, _, _ = multipletests(p_values, alpha=alpha, method="fdr_bh")
    return {"reject": list(reject), "p_adjusted": list(p_adj)}
