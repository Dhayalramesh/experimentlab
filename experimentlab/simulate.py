"""
Generates synthetic A/B test data for a checkout-flow experiment, standing
in for the kind of product experiment a data scientist would actually
analyze (e.g. "does the new checkout page increase conversion?").

The simulation deliberately includes three realistic complications so the
downstream analysis has something real to catch, rather than testing
against a suspiciously clean dataset:

1. A genuine (but modest) treatment effect on conversion rate.
2. A novelty effect that decays over the first few days (common in real
   product experiments: a redesign gets a short-lived bump from curiosity,
   not from being actually better).
3. Heterogeneous treatment effects by user segment (new vs. returning
   users), strong enough that the pooled effect can look different from
   either segment's effect on its own -- the setup for a Simpson's
   paradox-style sanity check.
"""
import numpy as np
import pandas as pd

RNG_SEED = 42

BASE_CONVERSION = {
    "new": 0.080,
    "returning": 0.150,
}

# True treatment effect (additive, on conversion probability), by segment.
TRUE_EFFECT = {
    "new": 0.006,        # +0.6pp for new users
    "returning": 0.018,  # +1.8pp for returning users -- effect is real but concentrated
}

NOVELTY_BONUS = 0.015   # extra lift on day 1, decaying to 0 by day 5
NOVELTY_HALF_LIFE_DAYS = 1.5

N_DAYS = 14
USERS_PER_DAY = 1200
SEGMENT_SHARE = {"new": 0.55, "returning": 0.45}


def _novelty_decay(day: int) -> float:
    return NOVELTY_BONUS * (0.5 ** (day / NOVELTY_HALF_LIFE_DAYS))


def simulate_experiment(seed: int = RNG_SEED) -> pd.DataFrame:
    """
    Returns one row per simulated user: day, segment, variant (control/
    treatment), converted (0/1), and a guardrail metric (page_load_ms)
    that treatment should NOT regress.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for day in range(N_DAYS):
        novelty = _novelty_decay(day)
        for _ in range(USERS_PER_DAY):
            segment = rng.choice(
                list(SEGMENT_SHARE.keys()), p=list(SEGMENT_SHARE.values())
            )
            variant = rng.choice(["control", "treatment"])

            base_p = BASE_CONVERSION[segment]
            if variant == "treatment":
                p = base_p + TRUE_EFFECT[segment] + novelty
            else:
                p = base_p
            p = min(max(p, 0.0), 1.0)
            converted = rng.random() < p

            # Guardrail metric: treatment is *slightly* slower to render
            # (a real cost of the redesign) but not enough to be a problem.
            base_load = 420 if variant == "control" else 428
            page_load_ms = rng.normal(base_load, 35)

            rows.append({
                "day": day,
                "segment": segment,
                "variant": variant,
                "converted": int(converted),
                "page_load_ms": round(max(page_load_ms, 50), 1),
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = simulate_experiment()
    df.to_csv("experiment_data.csv", index=False)
    print(f"wrote {len(df)} rows -> experiment_data.csv")
