# ExperimentLab

**Live demo:** https://experimentlab-jl3kjupstzhvfeabqb3zso.streamlit.app

A/B-testing and experimentation-statistics toolkit, with an interactive
dashboard on top: power analysis, hypothesis testing with confidence
intervals, multiple-testing correction, a live demonstration of the
"peeking problem," and segment-level / Simpson's-paradox checking.

Built to go beyond "run a t-test and report the p-value" — the kind of
experimentation rigor a data scientist is actually expected to bring to a
real product experiment, not just an offline model-evaluation notebook.

## Architecture
Streamlit dashboard (app.py)
|
v
experimentlab/ (analysis engine, plain Python + numpy/scipy/statsmodels)
|- simulate.py synthetic experiment data generator
|- power_analysis.py pre-experiment sample-size / power calculations
|- analysis.py z-test, Welch’s t-test, BH-FDR correction
|- peeking.py Monte Carlo demo of the peeking problem
|- segment_analysis.py per-segment effects + Simpson’s-paradox check
|
v
tests/ (pytest, 14 tests, including Monte Carlo validation of the
tests’ own false-positive rate under the null)

## What it does

- **Power analysis**: given a baseline rate and a minimum detectable
  effect, computes the sample size needed per arm (and the inverse:
  given a sample size, what power does that actually buy you).
- **Simulated experiment**: a synthetic checkout-flow A/B test with a
  real (but modest) treatment effect, a decaying novelty effect, and a
  heterogeneous effect across user segments — so the analysis has
  something non-trivial to catch.
- **Primary + guardrail testing**: two-proportion z-test for conversion
  rate, Welch's t-test for a guardrail metric (page load time) that
  treatment should not regress, both with 95% confidence intervals.
- **Multiple-testing correction**: Benjamini-Hochberg FDR correction
  across the primary and guardrail metrics, since testing multiple
  metrics at alpha=0.05 each inflates the overall false-positive rate.
- **The peeking problem**: simulates many experiments with *no true
  effect* and shows that naively stopping the first time p < 0.05
  gives a real false-positive rate well above the nominal 5% — then
  shows fixed-horizon testing and Bonferroni-corrected sequential
  testing both correct for it.
- **Segment analysis**: breaks the result out by user segment and flags
  disagreement between the pooled effect and per-segment effects (a
  Simpson's-paradox-style check), since a pooled number alone can hide
  a very different story in each segment.

## Running it

```bash
pip install -r requirements.txt
python run_experiment.py          # full analysis, printed to the terminal
streamlit run app.py              # interactive dashboard
pytest tests/ -v                  # 14 tests, including Monte Carlo validation
```

## Design notes

- **The peeking demonstration is the centerpiece.** It's not just
  claimed in a comment — `tests/test_peeking.py` actually asserts that
  naive peeking inflates the false-positive rate well above nominal,
  and that fixed-horizon and Bonferroni-corrected approaches don't.
- **The statistical tests are self-validated.**
  `test_ztest_false_positive_rate_matches_nominal_alpha` runs 600
  simulated null experiments and checks the z-test's own false-positive
  rate lands near 5% — a test testing the test.
- **Guardrail metrics are treated as first-class**, not an afterthought:
  the dashboard explicitly warns if a guardrail regresses significantly
  even when the primary metric improves, since "ship it" should depend
  on both.
- **Simpson's-paradox check is a real check, not a canned finding.** On
  a given simulation run, segments may or may not disagree in sign —
  the dashboard reports whatever the data actually shows.
