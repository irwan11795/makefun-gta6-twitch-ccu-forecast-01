# GTA VI Launch-Day Twitch CCU Forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/irwan11795/makefun-gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/irwan11795/makefun-gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts hour-by-hour Twitch concurrent viewers in the **Grand Theft Auto VI**
category on launch day (2026-11-19 UTC), hours 2–24, as five quantiles
(5%, 25%, 50%, 75%, 95%) anchored on the hour-1 print. Built for the
[make.fun bounty](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
($250, top 20 beat the house).

## Model

Per-hour **kernel-weighted log-log regression** over the case's own comparable
launches (38 in practice, 42 in the final):

- For each hour h, regress `ln(viewers_h)` on `ln(hour-1 print)` across
  comparables, with Gaussian kernel weights on `ln(p1)` (bandwidth 1.5) so
  launches of a similar scale dominate.
- Slope shrunk 75% toward 1 (the pure ratio model) and clamped to [0, 2];
  the fitted slope falls from ~1.0 at hour 2 to ~0.06 at hour 21 — big
  launches are already near peak at hour 1, small ones grow many-fold.
- Quantiles are the fitted median times `exp(z_tau * s_h)`, where `s_h` is
  the honest cross-launch residual dispersion at that hour (×0.8, floored
  0.15, capped 0.90) — wide where launches genuinely disagree (the
  mid-morning US-unlock hours), narrow where they agree (the early dip).

Two candidate features were tried and **rejected on the evidence**: a
pre-midnight-peak regressor (practice 0.189 but leave-one-out 0.474 —
overfits) and a quadratic scale term (no gain).

## Validation (local, pinball loss on the log scale)

| Set | This model | House baseline |
| --- | --- | --- |
| Practice p00 Wolverine | 0.201 | 0.744 |
| Practice p01 RDR2 | 0.206 | 0.287 |
| Practice p02 God of War Ragnarök | 0.128 | 0.215 |
| Practice p03 Spider-Man 2 | 0.184 | 0.303 |
| **Practice mean** | **0.180** | **0.387** |
| Leave-one-out pseudo-cases (37 launches) | 0.382 | 0.930 |

Practice answers were read from SullyGnome's public monthly charts and
cross-checked against the hour history in each practice case file; the
house baseline scored on those answers reproduces the rules' "~0.387",
confirming the answers and the local scorer.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network, deterministic (two runs
byte-identical). Reads every `<id>.in`, writes `<id>.out` with exactly 23
rows (hours 2–24). Runtime ~0.15 s per case, output ~1.5 KB per case.

## Files

- `main.py` — the whole entry (model + I/O), self-contained.
- `LICENSE` — MIT.

---

<sub>Entered on [make.fun](https://make.fun/?ref=2wht1jqa&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
