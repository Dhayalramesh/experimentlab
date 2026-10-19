"""
Pre-experiment power analysis: how many users per arm does the experiment
actually need to reliably detect a given minimum detectable effect (MDE)?

Running an experiment without this step is how teams end up either
wasting traffic on an over-powered test, or -- worse -- shipping a
"no significant difference" conclusion that was never adequately powered
to detect the effect size that would have mattered commercially.
"""
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize


def required_sample_size(
    baseline_rate: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """
    Sample size required PER ARM to detect an absolute lift of `mde` on
    top of `baseline_rate`, at the given significance level and power,
    using a two-proportion z-test.
    """
    effect_size = proportion_effectsize(baseline_rate + mde, baseline_rate)
    analysis = NormalIndPower()
    n = analysis.solve_power(
        effect_size=abs(effect_size),
        alpha=alpha,
        power=power,
        ratio=1.0,
        alternative="two-sided",
    )
    return n


def achieved_power(
    baseline_rate: float,
    mde: float,
    n_per_arm: float,
    alpha: float = 0.05,
) -> float:
    """The inverse question: given a sample size we actually have, what
    power does that give us to detect this MDE?"""
    effect_size = proportion_effectsize(baseline_rate + mde, baseline_rate)
    analysis = NormalIndPower()
    return analysis.power(
        effect_size=abs(effect_size),
        nobs1=n_per_arm,
        alpha=alpha,
        ratio=1.0,
        alternative="two-sided",
    )


if __name__ == "__main__":
    n = required_sample_size(baseline_rate=0.10, mde=0.01)
    print(f"Need ~{n:.0f} users per arm to detect a 1pp lift on a 10% baseline "
          f"(alpha=0.05, power=0.80).")
