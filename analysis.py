"""
WattSense AI - analysis helpers used by the dashboard.

- reconstruct_runs: turns detected ON/OFF events into appliance runs (start, end, watts, kWh)
- fault_report:     compares each cycling appliance's average run length with its normal length
- train_live_model: Random Forest used by the "Try the model" tab

No Streamlit code in here, so everything can be tested from a plain script.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

APPLIANCES = ["fridge", "ac", "microwave", "kettle"]
ON_FEATURES = ["delta_p", "delta_q", "inrush_ratio", "ramp_speed", "hour_of_day"]
OFF_FEATURES = ["delta_p", "delta_q", "hour_of_day"]

# Normal run length per cycle, taken from the healthy simulator settings (minutes)
NORMAL_RUN_MIN = {"fridge": 15.0, "ac": 30.0}

# An OFF closer than this to its ON is the tail of the startup spike, not a real switch-off
MIN_RUN_SECONDS = 20


def out_of_fold_predict(X, y, seed=42):
    """Predict each row with a model that never saw that row (cross-validation)."""
    counts = y.value_counts()
    k = int(min(5, counts.min())) if len(counts) else 0
    if k < 2:
        model = RandomForestClassifier(n_estimators=100, random_state=seed).fit(X, y)
        return model.predict(X)
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    model = RandomForestClassifier(n_estimators=100, random_state=seed)
    return cross_val_predict(model, X, y, cv=cv)


def reconstruct_runs(events_df, day_end=None):
    """
    Build appliance runs from detected events using only model predictions.
    Returns a DataFrame: appliance, start, end, minutes, watts, kwh.
    """
    ev = events_df.sort_values("timestamp").reset_index(drop=True).copy()
    ev["pred"] = "unknown"

    on_mask = ev["event_type"] == "ON"
    off_mask = ev["event_type"] == "OFF"
    if on_mask.sum() >= 4:
        ev.loc[on_mask, "pred"] = out_of_fold_predict(
            ev.loc[on_mask, ON_FEATURES], ev.loc[on_mask, "appliance_label"])
    if off_mask.sum() >= 4:
        ev.loc[off_mask, "pred"] = out_of_fold_predict(
            ev.loc[off_mask, OFF_FEATURES], ev.loc[off_mask, "appliance_label"])

    rows = []
    for app in APPLIANCES:
        offs = ev[(ev["pred"] == app) & off_mask]
        if offs.empty:
            continue
        typical_step = offs["delta_p"].abs().median()

        app_rows = []
        running, start = False, None
        for _, r in ev[ev["pred"] == app].iterrows():
            t = r["timestamp"]
            if r["event_type"] == "ON":
                # small ramp steps are not a real switch-on; a newer ON replaces an older one,
                # so one wrong prediction cannot stretch a run for hours
                if abs(r["delta_p"]) >= 0.2 * typical_step:
                    running, start = True, t
            elif running and (t - start).total_seconds() >= MIN_RUN_SECONDS:
                app_rows.append({"appliance": app, "start": start, "end": t,
                                 "watts": abs(float(r["delta_p"]))})
                running = False
        # a switch-on that never gets a matching switch-off is dropped: it cannot be checked,
        # and one wrong prediction would otherwise add hours of fake usage

        # one power figure per appliance: the median step of the switch-offs that closed a run
        closed = [r["watts"] for r in app_rows if not np.isnan(r["watts"])]
        power = float(np.median(closed)) if closed else float(typical_step)
        for r in app_rows:
            r["watts"] = power
        rows.extend(app_rows)

    runs = pd.DataFrame(rows)
    if runs.empty:
        return pd.DataFrame(columns=["appliance", "start", "end", "minutes", "watts", "kwh"])
    seconds = (runs["end"] - runs["start"]).dt.total_seconds()
    runs["minutes"] = seconds / 60.0
    runs["kwh"] = runs["watts"] * seconds / 3600.0 / 1000.0
    return runs


def usage_table(runs, sim_df):
    """Estimated vs true energy per appliance for the simulated day."""
    rows = []
    for app in APPLIANCES:
        true_kwh = sim_df[f"{app}_p"].sum() / 3600.0 / 1000.0
        est_kwh = runs.loc[runs["appliance"] == app, "kwh"].sum() if not runs.empty else 0.0
        n_runs = int((runs["appliance"] == app).sum()) if not runs.empty else 0
        err = (est_kwh - true_kwh) / true_kwh * 100.0 if true_kwh > 0 else np.nan
        rows.append({"appliance": app, "runs": n_runs, "estimated_kwh": est_kwh,
                     "true_kwh": true_kwh, "error_pct": err})
    return pd.DataFrame(rows)


def fault_report(runs, alert_ratio=1.5):
    """Average run length of the cycling appliances against their normal length."""
    rows = []
    for app, normal in NORMAL_RUN_MIN.items():
        sub = runs[runs["appliance"] == app] if not runs.empty else runs
        if sub.empty:
            continue
        # a run still going at the end of the day is cut short, so keep it but judge on the longest too
        avg = float(sub["minutes"].mean())
        longest = float(sub["minutes"].max())
        ratio = avg / normal
        rows.append({
            "appliance": app,
            "cycles": len(sub),
            "avg_run_min": avg,
            "longest_run_min": longest,
            "normal_run_min": normal,
            "ratio": ratio,
            "alert": ratio >= alert_ratio,
        })
    return pd.DataFrame(rows)


def train_live_model(events_df, seed=42):
    """Random Forest on all switch-on events, used for single-event predictions."""
    on = events_df[events_df["event_type"] == "ON"]
    model = RandomForestClassifier(n_estimators=150, random_state=seed)
    model.fit(on[ON_FEATURES], on["appliance_label"])
    return model


# ----------------------------------------------------------------------------
# Tamil Nadu slab bill
# ----------------------------------------------------------------------------
# (units in slab, rate in ₹ per unit); None = no upper limit. 1 unit = 1 kWh.
TN_SLABS_UP_TO_500 = [(200, 0.0), (200, 4.70), (100, 6.30)]
TN_SLABS_ABOVE_500 = [(100, 0.0), (100, 4.70), (200, 6.30), (100, 6.30),
                      (100, 8.40), (200, 9.45), (200, 10.50), (None, 11.55)]


def tn_bill(units):
    """Slab bill for one billing period. Returns (scheme_name, slab_table, total_rupees)."""
    units = max(0, int(round(units)))
    if units <= 500:
        scheme, slabs = "500 units or below (first 200 units free)", TN_SLABS_UP_TO_500
    else:
        scheme, slabs = "Above 500 units (first 100 units free)", TN_SLABS_ABOVE_500

    rows, lower, remaining = [], 0, units
    for width, rate in slabs:
        upper = None if width is None else lower + width
        take = remaining if width is None else min(remaining, width)
        if take > 0:
            label = f"{lower + 1} to {upper} units" if upper else f"Above {lower} units"
            rows.append({"slab": label, "units": take, "rate": rate, "amount": take * rate})
        remaining -= take
        lower = upper if upper else lower
        if remaining <= 0:
            break
    table = pd.DataFrame(rows, columns=["slab", "units", "rate", "amount"])
    return scheme, table, float(table["amount"].sum()) if not table.empty else 0.0


# ----------------------------------------------------------------------------
# "Try the model": every input and the hour of day affect the answer
# ----------------------------------------------------------------------------
SIG_FEATURES = ["delta_p", "delta_q", "inrush_ratio", "ramp_speed"]
FEATURE_RANGE = {"delta_p": (0.0, 2500.0), "delta_q": (-100.0, 1500.0),
                 "inrush_ratio": (1.0, 5.0), "ramp_speed": (-100.0, 100.0)}
BANDWIDTH_FLOOR = {"delta_p": 25.0, "delta_q": 15.0, "inrush_ratio": 0.12,
                   "ramp_speed": 3.0, "hour_of_day": 0.7}
UNUSUAL = "unusual"


def learn_hour_samples(generate_day, n_days=30):
    """Hours at which each appliance switches on, collected over several simulated healthy days."""
    hours = {a: [] for a in APPLIANCES}
    for seed in range(1000, 1000 + n_days):
        day = generate_day(seed=seed)
        for a in APPLIANCES:
            p = day[f"{a}_p"].to_numpy()
            starts = np.where((p[1:] > 1.0) & (p[:-1] < 1.0))[0] + 1
            hours[a].extend(day.index[starts].hour + day.index[starts].minute / 60.0)
    return {a: np.asarray(v, dtype=float) for a, v in hours.items()}


def _bandwidth(x, floor):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return floor
    spread = min(x.std(), (np.percentile(x, 75) - np.percentile(x, 25)) / 1.34)
    return max(floor, 0.9 * spread * len(x) ** -0.2)


class EvidenceModel:
    """
    Each appliance has a typical range for every input (a smoothed histogram learned from the
    detected events) and typical hours of use (learned from several simulated days).

    1. Appliances are compared input by input, and time of day counts as much as the
       electrical signature, so the hour changes the answer when two appliances look alike.
    2. Every input is also checked against the winner's usual range. An input far outside it
       lowers the probability and moves it to "unusual", so no input is ignored.
    """

    def __init__(self, on_events, hour_samples, sig_weight=0.35, hour_weight=2.0,
                 fit_hour_weight=4.0, floor=0.01):
        self.sig_weight, self.hour_weight, self.floor = sig_weight, hour_weight, floor
        self.fit_hour_weight = fit_hour_weight
        self.features = SIG_FEATURES + ["hour_of_day"]
        self.classes = [a for a in APPLIANCES
                        if (on_events["appliance_label"] == a).sum() >= 3 and len(hour_samples.get(a, [])) >= 5]
        rng = np.random.default_rng(0)
        self.samples, self.bw, self._typical = {}, {}, {}
        for c in self.classes:
            sub = on_events[on_events["appliance_label"] == c]
            self.samples[c] = {f: sub[f].to_numpy(dtype=float) for f in SIG_FEATURES}
            self.samples[c]["hour_of_day"] = np.asarray(hour_samples[c], dtype=float)
            self.bw[c] = {f: _bandwidth(v, BANDWIDTH_FLOOR[f]) for f, v in self.samples[c].items()}
        for c in self.classes:
            for f in self.features:
                pts = self.samples[c][f]
                if len(pts) > 300:
                    pts = rng.choice(pts, 300, replace=False)
                # density at the 5th-percentile training value: anything seen this often counts as a full match
                self._typical[(c, f)] = float(np.percentile(self._density(c, f, pts), 5))

    def _density(self, c, f, x):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        s, h = self.samples[c][f], self.bw[c][f]
        d = x[:, None] - s[None, :]
        if f == "hour_of_day":
            d = (d + 12.0) % 24.0 - 12.0
            flat = 1.0 / 24.0
        else:
            lo, hi = FEATURE_RANGE[f]
            flat = 1.0 / (hi - lo)
        kde = np.mean(np.exp(-0.5 * (d / h) ** 2) / (h * np.sqrt(2 * np.pi)), axis=1)
        return (1 - self.floor) * kde + self.floor * flat

    def predict(self, x):
        """
        x: dict with delta_p, delta_q, inrush_ratio, ramp_speed, hour_of_day.
        Returns (probabilities incl. "unusual", match table in %, fit per appliance in %).
        """
        logs, match = {}, {}
        for c in self.classes:
            dens = {f: float(self._density(c, f, x[f])[0]) for f in self.features}
            logs[c] = self.sig_weight * sum(np.log(dens[f]) for f in SIG_FEATURES) \
                + self.hour_weight * np.log(dens["hour_of_day"])
            match[c] = {f: 100.0 * min(1.0, dens[f] / self._typical[(c, f)]) for f in self.features}

        vals = np.array([logs[c] for c in self.classes])
        post = np.exp(vals - vals.max())
        post /= post.sum()

        match_df = pd.DataFrame(match).T[self.features]
        weights = np.array([1.0] * len(SIG_FEATURES) + [self.fit_hour_weight])
        fit = (match_df / 100.0).apply(
            lambda row: float(np.exp((np.log(row.clip(lower=1e-6)) * weights).sum() / weights.sum())), axis=1)

        probs = {c: float(p * fit[c]) for c, p in zip(self.classes, post)}
        probs[UNUSUAL] = max(0.0, 1.0 - sum(probs.values()))
        return probs, match_df, fit * 100.0


def usual_hours(hour_samples, keep=0.25):
    """Hour ranges (start, end) in which an appliance usually switches on."""
    counts = np.bincount(np.floor(np.asarray(hour_samples)).astype(int) % 24, minlength=24)
    on = counts >= keep * counts.max()
    ranges, start = [], None
    for h in range(24):
        if on[h] and start is None:
            start = h
        if not on[h] and start is not None:
            ranges.append((start, h))
            start = None
    if start is not None:
        ranges.append((start, 24))
    return ranges
