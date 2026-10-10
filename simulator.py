"""
WattSense AI - Step 1: Household power simulator
------------------------------------------------
Generates one day of 1-second smart-meter data for a household.

What you get (a pandas DataFrame, one row per second):
  total_p, total_q   -> what the meter "sees" (active power W, reactive power VAR)
  <name>_p, <name>_q -> the TRUE contribution of each appliance (answer key)

Run:  python simulator.py
Output: data/sim_day.csv  and  data/sim_day.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib

if "--no-show" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

SECONDS_PER_DAY = 24 * 3600

# --------------------------------------------------------------------------
# Appliance electrical "fingerprints": steady-state P (W) and Q (VAR)
# Note: microwave and kettle have similar WATTS but different Q -> look-alikes
# --------------------------------------------------------------------------
APPLIANCES = {
    "fridge":    {"p": 150.0,  "q": 120.0, "inrush": 3.0},  # motor: high Q, startup spike
    "ac":        {"p": 1500.0, "q": 900.0, "inrush": 1.8},  # big motor, slow ramp
    "microwave": {"p": 1100.0, "q": 250.0, "inrush": 1.0},
    "kettle":    {"p": 1200.0, "q": 10.0,  "inrush": 1.0},  # purely resistive
}

BASELINE_P = 120.0   # lights, router, standby
BASELINE_Q = 30.0
NOISE_P = 4.0        # std dev of measurement noise (W)
NOISE_Q = 3.0


def hms(h, m=0, s=0):
    """Clock time -> second index in the day."""
    return int(h * 3600 + m * 60 + s)


def add_run(p_arr, q_arr, spec, start, duration, ramp=0, inrush_secs=2):
    """Put one ON period of an appliance into its power arrays."""
    end = min(start + duration, SECONDS_PER_DAY)
    if start >= SECONDS_PER_DAY or end <= start:
        return
    n = end - start
    p = np.full(n, spec["p"])
    q = np.full(n, spec["q"])

    # slow ramp up (e.g. AC compressor spinning up)
    if ramp > 0:
        r = min(ramp, n)
        p[:r] = np.linspace(spec["p"] * 0.3, spec["p"], r)
        q[:r] = np.linspace(spec["q"] * 0.3, spec["q"], r)

    # short startup spike (inrush current)
    if spec["inrush"] > 1.0 and ramp == 0:
        k = min(inrush_secs, n)
        p[:k] = spec["p"] * spec["inrush"]
        q[:k] = spec["q"] * spec["inrush"]

    p_arr[start:end] += p
    q_arr[start:end] += q


def cycling(rng, p_arr, q_arr, spec, t_from, t_to, on_mean, on_std,
            off_mean, off_std, ramp=0):
    """Appliance that switches ON/OFF repeatedly (fridge, AC thermostat)."""
    t = t_from + int(rng.integers(0, 600))
    while t < t_to:
        on = max(60, int(rng.normal(on_mean, on_std)))
        add_run(p_arr, q_arr, spec, t, on, ramp=ramp)
        t += on + max(60, int(rng.normal(off_mean, off_std)))


FAULTS = {
    None: "Healthy home",
    "fridge": "Fridge compressor wearing out (runs much longer each cycle)",
    "ac": "AC stuck running (barely switches off)",
}


def generate_day(seed=42, fault=None):
    """fault: None, "fridge" or "ac". Injects a problem so it can be detected later."""
    rng = np.random.default_rng(seed)
    p = {name: np.zeros(SECONDS_PER_DAY) for name in APPLIANCES}
    q = {name: np.zeros(SECONDS_PER_DAY) for name in APPLIANCES}

    # Fridge: all day, ~15 min on, ~30 min off (fault: ~38 min on, ~15 min off)
    if fault == "fridge":
        f_on, f_off = (38 * 60, 300), (15 * 60, 120)
    else:
        f_on, f_off = (15 * 60, 120), (30 * 60, 240)
    cycling(rng, p["fridge"], q["fridge"], APPLIANCES["fridge"],
            0, SECONDS_PER_DAY, f_on[0], f_on[1], f_off[0], f_off[1])

    # AC: afternoon + night, long runs with a 15 s ramp (fault: ~2 h runs, short breaks)
    if fault == "ac":
        a_on, a_off = (120 * 60, 600), (3 * 60, 30)
    else:
        a_on, a_off = (30 * 60, 300), (15 * 60, 180)
    cycling(rng, p["ac"], q["ac"], APPLIANCES["ac"],
            hms(13), hms(23), a_on[0], a_on[1], a_off[0], a_off[1], ramp=15)

    # Microwave: short bursts around meal times
    for meal_h in (7.5, 13.0, 19.5):
        for _ in range(int(rng.integers(1, 3))):
            start = hms(meal_h) + int(rng.integers(0, 40 * 60))
            add_run(p["microwave"], q["microwave"], APPLIANCES["microwave"],
                    start, int(rng.integers(30, 180)))

    # Kettle: a few 2-4 minute boils (look-alike of the microwave in watts)
    for h_lo, h_hi, count in ((6, 9, 2), (16, 19, 2), (20, 22, 1)):
        for _ in range(count):
            start = hms(h_lo) + int(rng.integers(0, (h_hi - h_lo) * 3600))
            add_run(p["kettle"], q["kettle"], APPLIANCES["kettle"],
                    start, int(rng.integers(120, 240)))

    # Baseline: slow drift + noise
    drift = np.cumsum(rng.normal(0, 0.05, SECONDS_PER_DAY))
    drift -= drift.mean()
    base_p = BASELINE_P + np.clip(drift, -15, 15)
    base_q = np.full(SECONDS_PER_DAY, BASELINE_Q)

    # Build the DataFrame: truth columns + noisy aggregate
    idx = pd.date_range("2026-01-01", periods=SECONDS_PER_DAY, freq="s")
    df = pd.DataFrame(index=idx)
    df["baseline_p"] = base_p
    df["baseline_q"] = base_q
    for name in APPLIANCES:
        df[f"{name}_p"] = p[name]
        df[f"{name}_q"] = q[name]

    p_cols = ["baseline_p"] + [f"{n}_p" for n in APPLIANCES]
    q_cols = ["baseline_q"] + [f"{n}_q" for n in APPLIANCES]
    df["total_p"] = df[p_cols].sum(axis=1) + rng.normal(0, NOISE_P, SECONDS_PER_DAY)
    df["total_q"] = df[q_cols].sum(axis=1) + rng.normal(0, NOISE_Q, SECONDS_PER_DAY)
    return df


def plot_day(df, path):
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))

    axes[0].plot(df.index, df["total_p"], lw=0.5, color="black")
    axes[0].set_title("Aggregate active power - full day (what the meter sees)")
    axes[0].set_ylabel("Watts")

    # zoom on lunch time, when several appliances overlap
    z = df.loc["2026-01-01 12:30":"2026-01-01 14:30"]
    axes[1].plot(z.index, z["total_p"], lw=0.8, color="black", label="total P")
    axes[1].set_title("Zoom: 12:30-14:30 (look for the jumps = events)")
    axes[1].set_ylabel("Watts")
    axes[1].legend(loc="upper right")

    axes[2].stackplot(
        z.index,
        z["baseline_p"], z["fridge_p"], z["ac_p"], z["microwave_p"], z["kettle_p"],
        labels=["baseline", "fridge", "ac", "microwave", "kettle"],
    )
    axes[2].set_title("Same window - TRUE appliance contributions (the answer key)")
    axes[2].set_ylabel("Watts")
    axes[2].legend(loc="upper right", ncol=5)

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    print(f"Saved plot -> {path}")
    if "--no-show" not in sys.argv:
        plt.show()


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_day(seed=42)
    df.to_csv("data/sim_day.csv")
    print(f"Saved {len(df):,} rows -> data/sim_day.csv")

    print("\nTrue energy per appliance today (kWh):")
    for name in ["baseline"] + list(APPLIANCES):
        kwh = df[f"{name}_p"].sum() / 3600 / 1000
        print(f"  {name:10s} {kwh:6.2f}")
    print(f"  {'TOTAL':10s} {df['total_p'].sum() / 3600 / 1000:6.2f}  (measured, includes noise)")

    plot_day(df, "data/sim_day.png")