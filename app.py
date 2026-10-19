"""
ExperimentLab dashboard: an interactive front-end over the experimentlab
analysis engine, for exploring an A/B test the way a data scientist
actually would -- plan it, run it, check for the peeking trap, and look
past the pooled number into segments.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from experimentlab.analysis import two_proportion_ztest, welch_ttest, correct_multiple_tests
from experimentlab.power_analysis import required_sample_size, achieved_power
from experimentlab.peeking import simulate_false_positive_rate
from experimentlab.segment_analysis import analyze_by_segment, check_simpsons_paradox
from experimentlab.simulate import simulate_experiment

st.set_page_config(page_title="ExperimentLab", layout="wide")
st.title("ExperimentLab — A/B Testing & Experiment Analysis Toolkit")
st.caption(
    "Power analysis, hypothesis testing, multiple-testing correction, "
    "the peeking problem, and segment-level effect checks — all backed "
    "by a real (simulated) checkout-conversion experiment."
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Power Analysis", "Experiment Results", "The Peeking Problem", "Segments"]
)

# ---------------------------------------------------------------- Tab 1 ---
with tab1:
    st.subheader("How much traffic do you need before you even start?")
    col1, col2, col3 = st.columns(3)
    baseline = col1.slider("Baseline conversion rate", 0.01, 0.50, 0.10, 0.01)
    mde = col2.slider("Minimum detectable lift (absolute)", 0.001, 0.10, 0.01, 0.001)
    power_target = col3.slider("Target power", 0.50, 0.99, 0.80, 0.01)

    n_needed = required_sample_size(baseline_rate=baseline, mde=mde, power=power_target)
    st.metric("Required sample size per arm", f"{n_needed:,.0f}")

    st.write("#### Power curve: how power grows with sample size")
    sizes = np.linspace(max(n_needed * 0.05, 50), n_needed * 2, 60)
    powers = [achieved_power(baseline_rate=baseline, mde=mde, n_per_arm=n) for n in sizes]
    fig, ax = plt.subplots()
    ax.plot(sizes, powers)
    ax.axhline(power_target, color="gray", linestyle="--", label=f"target power={power_target:.2f}")
    ax.axvline(n_needed, color="red", linestyle="--", label=f"required n={n_needed:.0f}")
    ax.set_xlabel("Sample size per arm")
    ax.set_ylabel("Statistical power")
    ax.legend()
    st.pyplot(fig)

# ---------------------------------------------------------------- Tab 2 ---
with tab2:
    st.subheader("Run the (simulated) experiment and analyze the results")
    if st.button("Simulate a new experiment run"):
        st.session_state["df"] = simulate_experiment(seed=np.random.randint(0, 1_000_000))
    if "df" not in st.session_state:
        st.session_state["df"] = simulate_experiment()

    df = st.session_state["df"]
    st.write(f"{len(df):,} users across {df['day'].nunique()} days.")

    control = df.loc[df.variant == "control", "converted"].to_numpy()
    treatment = df.loc[df.variant == "treatment", "converted"].to_numpy()
    primary = two_proportion_ztest(control, treatment)

    c_load = df.loc[df.variant == "control", "page_load_ms"].to_numpy()
    t_load = df.loc[df.variant == "treatment", "page_load_ms"].to_numpy()
    guardrail = welch_ttest(c_load, t_load)

    corrected = correct_multiple_tests([primary.p_value, guardrail.p_value])

    col1, col2 = st.columns(2)
    with col1:
        st.write("##### Primary metric: conversion rate")
        st.metric(
            "Absolute lift",
            f"{primary.absolute_lift*100:+.2f}pp",
            f"p={primary.p_value:.4f}",
        )
        st.write(f"Control: {primary.control_rate*100:.2f}% | Treatment: {primary.treatment_rate*100:.2f}%")
        st.write(f"95% CI: [{primary.ci_low*100:+.2f}pp, {primary.ci_high*100:+.2f}pp]")
        st.write(f"Significant after BH correction: **{corrected['reject'][0]}**")

    with col2:
        st.write("##### Guardrail metric: page load time")
        st.metric(
            "Diff (treatment - control)",
            f"{guardrail.diff:+.1f} ms",
            f"p={guardrail.p_value:.4f}",
        )
        st.write(f"Control mean: {guardrail.control_mean:.1f}ms | Treatment mean: {guardrail.treatment_mean:.1f}ms")
        st.write(f"95% CI: [{guardrail.ci_low:+.1f}ms, {guardrail.ci_high:+.1f}ms]")
        st.write(f"Regressed significantly after BH correction: **{corrected['reject'][1]}**")

    verdict = "SHIP" if corrected["reject"][0] and primary.absolute_lift > 0 else "DO NOT SHIP"
    st.info(f"Recommendation: **{verdict}** (pending guardrail review)")

    with st.expander("Raw daily conversion rate by variant"):
        daily = df.groupby(["day", "variant"])["converted"].mean().unstack()
        st.line_chart(daily)

# ---------------------------------------------------------------- Tab 3 ---
with tab3:
    st.subheader("Why checking your p-value every day is a trap")
    st.write(
        "Simulating experiments where control and treatment have the "
        "**identical true rate** (no real effect), then 'peeking' at the "
        "p-value repeatedly as data accrues."
    )
    col1, col2 = st.columns(2)
    n_sims = col1.slider("Number of simulated null experiments", 200, 2000, 800, 100)
    n_looks = col2.slider("Number of peeks per experiment", 2, 20, 10, 1)

    if st.button("Run peeking simulation"):
        result = simulate_false_positive_rate(n_simulations=n_sims, n_looks=n_looks, max_n_per_arm=1500)
        st.session_state["peek_result"] = result
    if "peek_result" not in st.session_state:
        st.session_state["peek_result"] = simulate_false_positive_rate(
            n_simulations=800, n_looks=10, max_n_per_arm=1500
        )

    r = st.session_state["peek_result"]
    fig, ax = plt.subplots()
    labels = ["Naive peeking", "Fixed-horizon", "Bonferroni-corrected"]
    values = [r["naive_peeking_fpr"], r["fixed_horizon_fpr"], r["bonferroni_corrected_fpr"]]
    colors = ["#d62728", "#2ca02c", "#1f77b4"]
    ax.bar(labels, values, color=colors)
    ax.axhline(r["nominal_alpha"], color="gray", linestyle="--", label=f"nominal alpha={r['nominal_alpha']}")
    ax.set_ylabel("Empirical false-positive rate")
    ax.legend()
    st.pyplot(fig)

    st.write(
        f"With **{r['n_looks']} peeks** and no real effect, naive stopping gives a "
        f"**{r['naive_peeking_fpr']*100:.1f}%** false-positive rate — "
        f"{r['naive_peeking_fpr']/r['nominal_alpha']:.1f}x the nominal {r['nominal_alpha']*100:.0f}%."
    )

# ---------------------------------------------------------------- Tab 4 ---
with tab4:
    st.subheader("Don't trust the pooled number alone")
    df = st.session_state.get("df", simulate_experiment())
    seg_df = analyze_by_segment(df)
    st.dataframe(seg_df.style.format({
        "control_rate": "{:.4f}", "treatment_rate": "{:.4f}",
        "absolute_lift": "{:+.4f}", "p_value": "{:.4f}",
    }))

    fig, ax = plt.subplots()
    ax.bar(seg_df["segment"], seg_df["absolute_lift"] * 100)
    ax.set_ylabel("Absolute lift (pp)")
    ax.axhline(0, color="black", linewidth=0.8)
    st.pyplot(fig)

    control = df.loc[df.variant == "control", "converted"].to_numpy()
    treatment = df.loc[df.variant == "treatment", "converted"].to_numpy()
    pooled_lift = treatment.mean() - control.mean()
    paradox = check_simpsons_paradox(pooled_lift, seg_df)

    if paradox["heterogeneous_effect_detected"]:
        st.warning(
            f"Pooled lift is **{paradox['pooled_sign']}**, but segments disagree: "
            f"{paradox['segment_signs']}. Reporting only the pooled number would mislead."
        )
    else:
        st.success(
            f"Pooled lift ({paradox['pooled_sign']}) agrees in direction with all segments "
            f"({paradox['segment_signs']}) — no Simpson's-paradox-style reversal detected. "
            f"Effect size still varies by segment though — check the table above."
        )
