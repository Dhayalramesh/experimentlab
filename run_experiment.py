"""
End-to-end experiment analysis workflow:
  1. Pre-experiment power analysis (how much traffic do we need?)
  2. Simulate the experiment (standing in for pulling real logged data)
  3. Primary metric test (conversion rate) + guardrail metric test (page load)
  4. Multiple-testing correction across primary + guardrail
  5. Peeking-problem demonstration (why looking repeatedly without
     correction inflates false positives)
  6. Segment-level breakdown + Simpson's-paradox check

Prints a plain-English summary a stakeholder could actually read, mirroring
what a data scientist would hand back after analyzing a real experiment.
"""
from experimentlab.simulate import simulate_experiment
from experimentlab.power_analysis import required_sample_size, achieved_power
from experimentlab.analysis import two_proportion_ztest, welch_ttest, correct_multiple_tests
from experimentlab.peeking import simulate_false_positive_rate
from experimentlab.segment_analysis import analyze_by_segment, check_simpsons_paradox


def main():
    print("=" * 70)
    print("1. PRE-EXPERIMENT POWER ANALYSIS")
    print("=" * 70)
    baseline = 0.10
    mde = 0.01
    n_needed = required_sample_size(baseline_rate=baseline, mde=mde)
    print(f"To detect a {mde*100:.1f}pp lift on a {baseline*100:.0f}% baseline "
          f"(alpha=0.05, power=0.80): need ~{n_needed:.0f} users per arm.")

    print()
    print("=" * 70)
    print("2. SIMULATING EXPERIMENT DATA")
    print("=" * 70)
    df = simulate_experiment()
    print(f"{len(df)} users logged across {df['day'].nunique()} days.")
    print(df.groupby("variant").size().to_string())

    print()
    print("=" * 70)
    print("3. PRIMARY METRIC: CONVERSION RATE (two-proportion z-test)")
    print("=" * 70)
    control = df.loc[df.variant == "control", "converted"].to_numpy()
    treatment = df.loc[df.variant == "treatment", "converted"].to_numpy()
    primary = two_proportion_ztest(control, treatment)
    achieved = achieved_power(baseline_rate=primary.control_rate, mde=primary.absolute_lift,
                               n_per_arm=len(control))
    print(f"Control:   {primary.control_rate:.4f}  (n={len(control)})")
    print(f"Treatment: {primary.treatment_rate:.4f}  (n={len(treatment)})")
    print(f"Absolute lift: {primary.absolute_lift:+.4f}  "
          f"(95% CI: [{primary.ci_low:+.4f}, {primary.ci_high:+.4f}])")
    print(f"Relative lift: {primary.relative_lift*100:+.2f}%")
    print(f"p-value: {primary.p_value:.4f}  |  achieved power at this n: {achieved:.2f}")

    print()
    print("=" * 70)
    print("4. GUARDRAIL METRIC: PAGE LOAD TIME (Welch's t-test)")
    print("=" * 70)
    c_load = df.loc[df.variant == "control", "page_load_ms"].to_numpy()
    t_load = df.loc[df.variant == "treatment", "page_load_ms"].to_numpy()
    guardrail = welch_ttest(c_load, t_load)
    print(f"Control mean:   {guardrail.control_mean:.1f} ms")
    print(f"Treatment mean: {guardrail.treatment_mean:.1f} ms")
    print(f"Diff: {guardrail.diff:+.1f} ms  (95% CI: [{guardrail.ci_low:+.1f}, {guardrail.ci_high:+.1f}])")
    print(f"p-value: {guardrail.p_value:.4f}")

    print()
    print("=" * 70)
    print("5. MULTIPLE-TESTING CORRECTION (primary + guardrail, BH-FDR)")
    print("=" * 70)
    corrected = correct_multiple_tests([primary.p_value, guardrail.p_value])
    print(f"Raw p-values:      conversion={primary.p_value:.4f}, page_load={guardrail.p_value:.4f}")
    print(f"Adjusted p-values: conversion={corrected['p_adjusted'][0]:.4f}, "
          f"page_load={corrected['p_adjusted'][1]:.4f}")
    print(f"Reject null after correction: conversion={corrected['reject'][0]}, "
          f"page_load={corrected['reject'][1]}")

    print()
    print("=" * 70)
    print("6. THE PEEKING PROBLEM (why you can't just check p-value daily)")
    print("=" * 70)
    peek_result = simulate_false_positive_rate(n_simulations=1500, n_looks=10, max_n_per_arm=2000)
    print(f"Simulated {peek_result['n_simulations']} experiments with NO true effect, "
          f"checked {peek_result['n_looks']} times each:")
    print(f"  Naive peeking (stop at first p<0.05):         "
          f"{peek_result['naive_peeking_fpr']*100:.1f}% false-positive rate "
          f"(nominal alpha={peek_result['nominal_alpha']*100:.0f}%)")
    print(f"  Fixed-horizon (look once, at the end):        "
          f"{peek_result['fixed_horizon_fpr']*100:.1f}% false-positive rate")
    print(f"  Bonferroni-corrected peeking (alpha/n_looks):  "
          f"{peek_result['bonferroni_corrected_fpr']*100:.1f}% false-positive rate")

    print()
    print("=" * 70)
    print("7. SEGMENT BREAKDOWN & SIMPSON'S PARADOX CHECK")
    print("=" * 70)
    seg_df = analyze_by_segment(df)
    print(seg_df.to_string(index=False))
    paradox = check_simpsons_paradox(primary.absolute_lift, seg_df)
    print(f"\nPooled lift sign: {paradox['pooled_sign']}  |  "
          f"Segment lift signs: {paradox['segment_signs']}  |  "
          f"Heterogeneous effect flagged: {paradox['heterogeneous_effect_detected']}")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    verdict = "SHIP" if corrected["reject"][0] and primary.absolute_lift > 0 else "DO NOT SHIP"
    print(f"Primary metric significant after correction: {corrected['reject'][0]}")
    print(f"Guardrail regressed significantly: {corrected['reject'][1]}")
    print(f"Recommendation: {verdict} (pending guardrail review), with effect "
          f"concentrated in the 'returning' user segment -- see section 7.")


if __name__ == "__main__":
    main()
