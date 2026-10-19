"""
Checks the experiment result by segment, not just pooled -- and flags a
Simpson's-paradox-style reversal: a case where the pooled effect and a
per-segment effect disagree in sign or in a commercially meaningful way,
usually because segment mix differs between arms or the true effect is
genuinely heterogeneous across segments.

This is the check that catches "the experiment looked flat overall, but
it was actually working great for returning users and roughly neutral
for new users" -- a very different (and more useful) conclusion than
"ship it" or "kill it" based on the pooled number alone.
"""
import pandas as pd

from experimentlab.analysis import two_proportion_ztest


def analyze_by_segment(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for segment, g in df.groupby("segment"):
        control = g.loc[g.variant == "control", "converted"].to_numpy()
        treatment = g.loc[g.variant == "treatment", "converted"].to_numpy()
        result = two_proportion_ztest(control, treatment)
        rows.append({
            "segment": segment,
            "n_control": len(control),
            "n_treatment": len(treatment),
            "control_rate": result.control_rate,
            "treatment_rate": result.treatment_rate,
            "absolute_lift": result.absolute_lift,
            "p_value": result.p_value,
        })
    return pd.DataFrame(rows)


def check_simpsons_paradox(pooled_lift: float, segment_df: pd.DataFrame) -> dict:
    """
    Flags whether the pooled lift's sign disagrees with any individual
    segment's lift sign, or whether the segments disagree sharply with
    each other (a sign that reporting only the pooled number would
    mislead a reader into thinking the effect is uniform).
    """
    signs = set(
        "positive" if lift > 0 else ("negative" if lift < 0 else "zero")
        for lift in segment_df["absolute_lift"]
    )
    pooled_sign = "positive" if pooled_lift > 0 else ("negative" if pooled_lift < 0 else "zero")

    sign_disagreement = pooled_sign not in signs or len(signs) > 1

    return {
        "pooled_sign": pooled_sign,
        "segment_signs": sorted(signs),
        "heterogeneous_effect_detected": sign_disagreement,
    }
