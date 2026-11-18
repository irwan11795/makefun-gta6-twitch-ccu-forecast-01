"""GTA VI launch-day Twitch CCU forecast — v1.

Model: per-hour kernel-weighted log-log regression over the case's own
comparable launches (the 38 past launches in practice, 42 in the final).

For each forecast hour h = 2..24, using every comparable launch c with a
known hour-1 print p1_c and a known hour-h print y_c,h:

    x_c = ln(p1_c),  y_c = ln(y_c,h)
    w_c = exp(-(x_c - ln p1)^2 / (2 * bw^2)),   bw = 1.5

    b_h = weighted least-squares slope of y on x, shrunk 75% of the way
          toward 1 (ratio model) and clamped to [0, 2]
    median_h = exp( weighted mean of (y_c - b_h * (x_c - ln p1)) )
    s_h = clamp(0.8 * weighted residual SD, 0.15, 0.90)
    q_tau(h) = median_h * exp(z_tau * s_h), rounded half-up to 2 decimals

Why this shape: across past launches the hour-1 print is nearly
uninformative about the day's peak by itself (peak/p1 ranges 1.0x to
2000x), but ln(y_h) vs ln(p1) is strongly log-log linear with a slope
that falls from ~1.0 at hour 2 to ~0.06 at hour 21 — big launches are
already near their peak at hour 1, small ones grow many-fold. The
kernel localises the fit to launches of a similar scale, and the slope
does the scale extrapolation. The spread is the honest cross-launch
residual dispersion at that hour, so the 90% band is wide where
launches genuinely disagree (mid-morning US-unlock hours) and narrow
where they agree (the early dip).

Validation (leave-one-out over the 37 usable comparables + the 4
official practice cases, pinball loss on the log scale, same scorer as
the rules): practice mean 0.180 (p00 0.201, p01 0.206, p02 0.128,
p03 0.184) vs house baseline 0.387 on the same cases; pseudo-case mean
0.382 vs baseline 0.930. Verified end-to-end on 2026-11-18: two runs on
the 4 practice cases byte-identical, ~0.15 s per case, outputs ~1.5 KB,
all 23 rows well-formed (hours 2-24, five quantiles each, in (0, 1e9],
non-decreasing). Two features were tried and REJECTED on the
pseudo-case evidence: a pre-midnight-peak regressor (practice 0.189
but pseudo 0.474 — overfits) and a quadratic scale term (no gain).

Deterministic, standard library only, no network. Reads every <id>.in
in the input directory and writes <id>.out with exactly 23 rows.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0)
in python:3.13-slim. Outside the grader, MAKEFUN_INPUT_DIR and
MAKEFUN_OUTPUT_DIR stand in for /input and /output.
"""

import json
import math
import os
from datetime import datetime, timedelta
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Tuned parameters (see module docstring for the validation numbers).
BW = 1.5          # log-scale kernel bandwidth on ln(p1)
LAMBDA = 0.75     # slope shrinkage toward 1 (ratio model)
SPREAD_K = 0.8    # multiplier on the residual SD
SPREAD_FLOOR = 0.15
SPREAD_CAP = 0.90
BASELINE_SIGMA = 0.15  # fallback spread (house baseline shape)
MIN_POOL = 3      # minimum usable comparables before falling back


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def parse_ts(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")


def hour1_print(case: dict) -> float:
    """Hour-1 viewers from `known`; fall back to the history point at
    day_start, then to 1. Never raises."""
    try:
        first = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
        if first is not None and first.get("viewers"):
            return max(1.0, float(first["viewers"]))
    except (TypeError, ValueError):
        pass
    try:
        ds = parse_ts(case["day_start"])
        hist = case.get("history_hourly") or []
        at_start = [p for p in hist if parse_ts(p["from"]) == ds and p.get("viewers")]
        if at_start:
            return max(1.0, float(at_start[-1]["viewers"]))
    except (KeyError, TypeError, ValueError):
        pass
    return 1.0


def comparable_day(comp: dict) -> list:
    """24 hourly prints for a comparable's release day (None where missing)."""
    ds = parse_ts(comp["day_start"])
    hourly = {}
    for p in comp.get("hourly") or []:
        try:
            hourly[parse_ts(p["from"])] = float(p["viewers"])
        except (KeyError, TypeError, ValueError):
            continue
    return [hourly.get(ds + timedelta(hours=h)) for h in range(24)]


def usable_pool(case: dict) -> list:
    """Usable comparables: known positive hour-1 print and >= 12 of 24 hours."""
    pool = []
    for comp in case.get("comparables") or []:
        try:
            day = comparable_day(comp)
        except (KeyError, TypeError, ValueError):
            continue
        p1 = day[0]
        if p1 is None or p1 <= 0:
            continue
        if sum(1 for v in day if v is not None and v > 0) < 12:
            continue
        pool.append(day)
    return pool


def baseline_quantiles(p1: float, h: int) -> list:
    """House baseline shape: flat at p1, spread widening like a random walk."""
    return [round2(p1 * math.exp(z * BASELINE_SIGMA * math.sqrt(h - 1))) for z in Z]


def hour_quantiles(p1: float, pool: list, h: int) -> list:
    """The five quantiles for hour h from the kernel-weighted log-log fit."""
    lp1 = math.log(max(1.0, p1))
    xs, ys, ws = [], [], []
    for day in pool:
        y1, yh = day[0], day[h - 1]
        if y1 is None or y1 <= 0 or yh is None or yh <= 0:
            continue
        xs.append(math.log(y1))
        ys.append(math.log(yh))
        ws.append(math.exp(-((math.log(y1) - lp1) ** 2) / (2 * BW * BW)))
    if len(xs) < MIN_POOL:
        return baseline_quantiles(p1, h)
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    den = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if den <= 1e-12:
        b = 1.0
    else:
        b = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys)) / den
        b = 1.0 + LAMBDA * (b - 1.0)
        b = max(0.0, min(2.0, b))
    # Median at x = ln p1 (the intercept of the shrunk fit at the target).
    med_log = sum(w * (y - b * (x - lp1)) for w, x, y in zip(ws, xs, ys)) / sw
    resid = [y - (med_log + b * (x - lp1)) for x, y in zip(xs, ys)]
    var = sum(w * r * r for w, r in zip(ws, resid)) / sw
    s = min(SPREAD_CAP, max(SPREAD_FLOOR, math.sqrt(max(var, 0.0)) * SPREAD_K))
    med = math.exp(med_log)
    row = [round2(med * math.exp(z * s)) for z in Z]
    # Clamp to (0, 1e9] and enforce non-decreasing (rounding can only tie).
    row = [min(1e9, max(1.0, float(v))) for v in row]
    for i in range(1, len(row)):
        if row[i] < row[i - 1]:
            row[i] = row[i - 1]
    return [int(v) if float(v).is_integer() else v for v in row]


def forecast_case(case: dict, name: str) -> dict:
    p1 = hour1_print(case)
    pool = usable_pool(case)
    hours = []
    try:
        hours = [int(r["hour"]) for r in case.get("forecast") or []]
    except (KeyError, TypeError, ValueError):
        hours = []
    if not hours:
        hours = list(range(2, 25))
    rows = []
    for h in hours:
        q = hour_quantiles(p1, pool, h) if len(pool) >= MIN_POOL else baseline_quantiles(p1, h)
        rows.append({"hour": h, "q": q})
    return {"hours": rows}


def fallback_output() -> dict:
    """A valid (if flat) output, used only if a case cannot be parsed at all."""
    return {"hours": [{"hour": h, "q": baseline_quantiles(1.0, h)} for h in range(2, 25)]}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            forecast = forecast_case(case, path.name)
        except Exception:
            forecast = fallback_output()
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
