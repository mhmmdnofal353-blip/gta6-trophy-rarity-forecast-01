"""Simulator validation for the pooled curve-extension forecaster.

The real final is one case whose answer nobody has until the target passes,
so the model is validated here on synthetic worlds instead: each world is a
full launch week of Exophase-style trophy pages generated from known ground
truth -- logistic earner accumulation per trophy over a growing tracked
player base, a periodic page-refresh cadence, rounding to 2 decimals and
observation noise. Each world is cut into a case of exactly the final's
shape (checkpoint at 59.5 h after release, target at 168 h) and scored with
the contest's exact pinball loss on the logit scale:

    loss(tau, y, q) = 2 * max(tau * d, (tau - 1) * d),  d = logit(y) - logit(q)

Three forecasters are scored on identical cases:

  * model        -- main.py's forecast_case
  * persistence  -- the house baseline: last rate, random-walk spread
  * curve        -- the house curve baseline: per-trophy least-squares line
                    of logit(rate) against ln(1 + hours since release) over
                    the last 24 h of reads, read at the target, spread the
                    random walk and the fit residual combined

Six regimes (base, fast, slow, dilution, weekend, noisy) x four seeds each,
plus a seventh, freeze: a base world whose page stops refreshing 23.5 hours
before the checkpoint, so the anchor page is stale and a whole day of
evolution is unobserved -- the shape of the launch-night practice failure,
where a frozen page's first refresh dropped every rate outside the band.
Unlike the six classic regimes (whose reads jitter independently), the
freeze world's page repeats exactly between refreshes, as the real page
does. The table reports each forecaster's loss and its 90% coverage (the
share of trophies whose true rate lands inside the 5-95% band; nominal
0.90). The numbers are simulator numbers: they compare forecasters on equal
footing, they are not a promise about the real launch week.

    python3 backtest.py
"""

import math
import random
import sys
import zlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import main as forecaster

LEVELS = forecaster.LEVELS
Z = forecaster.Z
RELEASE = datetime(2026, 11, 19, tzinfo=timezone.utc)
CHECKPOINT_H = 59.5
TARGET_H = 168.0
N_TROPHIES = 55


# --------------------------------------------------------------------------
# scoring


def pinball_mean(rows, truth):
    """Mean contest pinball loss of forecast rows against true rates."""
    total = 0.0
    count = 0
    for row, y in zip(rows, truth):
        ly = forecaster.logit(y)
        for tau, q in zip(LEVELS, row["q"]):
            d = ly - forecaster.logit(q)
            total += 2.0 * max(tau * d, (tau - 1.0) * d)
            count += 1
    return total / count


def coverage90(rows, truth):
    """Share of trophies whose true rate lands inside the 5-95% band."""
    inside = sum(1 for row, y in zip(rows, truth) if row["q"][0] <= y <= row["q"][4])
    return inside / len(truth)


# --------------------------------------------------------------------------
# house baselines, reimplemented from the rules' descriptions


def persistence_forecast(case):
    last = case["history"][-1]
    days = case["horizon_minutes"] / 1440.0
    sigma = 0.5 * math.sqrt(days)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = last["rates"][i]
        centre = forecaster.logit(rate if rate is not None else 0.0)
        qs = [forecaster.round4(forecaster.clamp_rate(
            forecaster.inv_logit(centre + z * sigma))) for z in Z]
        rows.append({"id": trophy["id"], "q": qs})
    return rows


def curve_forecast(case):
    """Per-trophy least-squares line over the last 24 h of reads."""
    rows = []
    target_h = TARGET_H
    days = case["horizon_minutes"] / 1440.0
    rw = 0.5 * math.sqrt(days)
    for i, trophy in enumerate(case["trophies"]):
        pts = []
        for read in case["history"]:
            h = (datetime.fromisoformat(read["at"].replace("Z", "+00:00")) - RELEASE).total_seconds() / 3600.0
            if h >= CHECKPOINT_H - 24.0 and read["rates"][i] is not None:
                pts.append((math.log1p(h), forecaster.logit(read["rates"][i])))
        if len(pts) < 3:
            rate = case["history"][-1]["rates"][i]
            centre = forecaster.logit(rate if rate is not None else 0.0)
            sigma = rw
        else:
            n = len(pts)
            mx = sum(p[0] for p in pts) / n
            my = sum(p[1] for p in pts) / n
            sxx = sum((p[0] - mx) ** 2 for p in pts)
            sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
            b = sxy / sxx if sxx > 1e-12 else 0.0
            a = my - b * mx
            rms = math.sqrt(sum((p[1] - (a + b * p[0])) ** 2 for p in pts) / n)
            centre = a + b * math.log1p(target_h)
            sigma = math.sqrt(rw * rw + rms * rms)
        qs = [forecaster.round4(forecaster.clamp_rate(
            forecaster.inv_logit(centre + z * sigma))) for z in Z]
        rows.append({"id": trophy["id"], "q": qs})
    return rows


# --------------------------------------------------------------------------
# synthetic worlds


def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


REGIMES = {
    # name: (player midpoint, player width, earner midpoint shift, noise, refresh h)
    "base":     (3.2, 0.90, 0.0, 0.06, 2.0),
    "fast":     (2.6, 0.85, -0.7, 0.06, 2.0),
    "slow":     (3.8, 0.95, +0.8, 0.06, 2.0),
    "dilution": (2.2, 0.60, +0.5, 0.06, 2.0),
    "weekend":  (3.2, 0.90, 0.0, 0.06, 2.0),
    "noisy":    (3.2, 0.90, 0.0, 0.25, 4.0),
}


def make_world(regime, seed):
    # a stable seed: the same worlds on every run and every machine
    rng = random.Random(zlib.crc32(f"{regime}:{seed}".encode()))
    p_mid, p_wid, e_shift, noise, refresh_h = REGIMES[regime]
    n_inf = rng.uniform(60000, 300000)

    players = lambda t: n_inf * sigmoid((math.log1p(t) - p_mid) / p_wid)

    trophies = []
    for _ in range(N_TROPHIES):
        trophies.append({
            "share": sigmoid(rng.uniform(-4.5, 1.5)),   # eventual earner share
            "mid": rng.uniform(1.2, 4.6) + e_shift,     # in ln(1 + hours)
            "wid": rng.uniform(0.35, 0.8),
        })

    def earn_progress(tr, t):
        # the weekend regime plays faster once the first weekend starts
        if regime == "weekend" and t > 96.0:
            t = 96.0 + (t - 96.0) * 1.8
        return sigmoid((math.log1p(t) - tr["mid"]) / tr["wid"])

    def true_rates(t):
        n = players(t)
        return [100.0 * tr["share"] * n_inf * earn_progress(tr, t) / n for tr in trophies]

    # reads every 30 minutes up to the checkpoint; the page only changes at
    # refresh times, so a read shows the truth as of the last refresh
    history = []
    step = 0.5
    t = 0.0
    while t <= CHECKPOINT_H + 1e-9:
        shown_t = math.floor(t / refresh_h) * refresh_h
        rates = []
        for r in true_rates(shown_t):
            noisy = r * math.exp(rng.gauss(0.0, noise)) if r > 0 else r
            rates.append(round(min(100.0, max(0.0, noisy)), 2))
        at = RELEASE + timedelta(hours=t)
        history.append({
            "slot": at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "at": at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "players_tracked": int(round(players(shown_t))),
            "trophies_earned": 0,
            "rates": rates,
        })
        t += step

    case = {
        "schema": "makefun/gta6-trophies-case/v1",
        "id": f"sim-{regime}-{seed}",
        "at": (RELEASE + timedelta(hours=CHECKPOINT_H)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "horizon_minutes": int((TARGET_H - CHECKPOINT_H) * 60),
        "target": (RELEASE + timedelta(hours=TARGET_H)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "levels": LEVELS,
        "game": {"name": "Grand Theft Auto VI", "platform": "PS5",
                 "release_date": "2026-11-19"},
        "trophies": [{"id": i + 1, "name": f"Sim Trophy {i + 1}",
                      "description": "", "tier": "bronze", "points": 15,
                      "secret": False} for i in range(N_TROPHIES)],
        "history": history,
        "notes": "synthetic",
    }
    truth = [round(min(100.0, max(0.0, r)), 2) for r in true_rates(TARGET_H)]
    return case, truth


FREEZE_AT_H = 36.0  # the freeze world's page stops refreshing here


def make_freeze_world(seed):
    """A base world whose page freezes at FREEZE_AT_H: every read after that
    repeats one page exactly (players and rates), while the truth keeps
    evolving underneath. The checkpoint therefore anchors on a page that is
    CHECKPOINT_H - FREEZE_AT_H hours stale."""
    rng = random.Random(zlib.crc32(f"freeze:{seed}".encode()))
    p_mid, p_wid, e_shift, noise, refresh_h = REGIMES["base"]
    n_inf = rng.uniform(60000, 300000)

    players = lambda t: n_inf * sigmoid((math.log1p(t) - p_mid) / p_wid)

    trophies = []
    for _ in range(N_TROPHIES):
        trophies.append({
            "share": sigmoid(rng.uniform(-4.5, 1.5)),
            "mid": rng.uniform(1.2, 4.6) + e_shift,
            "wid": rng.uniform(0.35, 0.8),
        })

    def true_rates(t):
        n = players(t)
        return [100.0 * tr["share"] * n_inf
                * sigmoid((math.log1p(t) - tr["mid"]) / tr["wid"]) / n
                for tr in trophies]

    history = []
    shown_cache = {}
    step = 0.5
    t = 0.0
    while t <= CHECKPOINT_H + 1e-9:
        shown_t = math.floor(min(t, FREEZE_AT_H) / refresh_h) * refresh_h
        if shown_t not in shown_cache:
            # one noisy page per refresh, then repeated exactly -- as the
            # real page repeats between refreshes
            rates = []
            for r in true_rates(shown_t):
                noisy = r * math.exp(rng.gauss(0.0, noise)) if r > 0 else r
                rates.append(round(min(100.0, max(0.0, noisy)), 2))
            shown_cache[shown_t] = (int(round(players(shown_t))), rates)
        players_shown, rates = shown_cache[shown_t]
        at = RELEASE + timedelta(hours=t)
        history.append({
            "slot": at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "at": at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "players_tracked": players_shown,
            "trophies_earned": 0,
            "rates": list(rates),
        })
        t += step

    case = {
        "schema": "makefun/gta6-trophies-case/v1",
        "id": f"sim-freeze-{seed}",
        "at": (RELEASE + timedelta(hours=CHECKPOINT_H)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "horizon_minutes": int((TARGET_H - CHECKPOINT_H) * 60),
        "target": (RELEASE + timedelta(hours=TARGET_H)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "levels": LEVELS,
        "game": {"name": "Grand Theft Auto VI", "platform": "PS5",
                 "release_date": "2026-11-19"},
        "trophies": [{"id": i + 1, "name": f"Sim Trophy {i + 1}",
                      "description": "", "tier": "bronze", "points": 15,
                      "secret": False} for i in range(N_TROPHIES)],
        "history": history,
        "notes": "synthetic freeze",
    }
    truth = [round(min(100.0, max(0.0, r)), 2) for r in true_rates(TARGET_H)]
    return case, truth


# --------------------------------------------------------------------------


def score_all(case, truth):
    """(loss, coverage90) for the model and both house baselines."""
    model = forecaster.forecast_case_safe(case)["trophies"]
    persist = persistence_forecast(case)
    curve = curve_forecast(case)
    return {
        "model": (pinball_mean(model, truth), coverage90(model, truth)),
        "persist": (pinball_mean(persist, truth), coverage90(persist, truth)),
        "curve": (pinball_mean(curve, truth), coverage90(curve, truth)),
    }


def run():
    print(f"{'regime':<10} {'model':>8} {'cov90':>6} {'persist':>8} {'cov90':>6} "
          f"{'curve':>8} {'cov90':>6}")
    totals = {"model": [], "persist": [], "curve": []}
    covs = {"model": [], "persist": [], "curve": []}

    def one(label, worlds):
        agg = {"model": [], "persist": [], "curve": []}
        agc = {"model": [], "persist": [], "curve": []}
        for case, truth in worlds:
            got = score_all(case, truth)
            for k in agg:
                agg[k].append(got[k][0])
                agc[k].append(got[k][1])
        means = {k: sum(v) / len(v) for k, v in agg.items()}
        cmeans = {k: sum(v) / len(v) for k, v in agc.items()}
        print(f"{label:<10} {means['model']:>8.3f} {cmeans['model']:>6.3f} "
              f"{means['persist']:>8.3f} {cmeans['persist']:>6.3f} "
              f"{means['curve']:>8.3f} {cmeans['curve']:>6.3f}")
        return agg, agc

    for regime in REGIMES:
        agg, agc = one(regime, [make_world(regime, seed) for seed in range(4)])
        for k in totals:
            totals[k].extend(agg[k])
            covs[k].extend(agc[k])
    print(f"{'MEAN':<10} {sum(totals['model'])/len(totals['model']):>8.3f} "
          f"{sum(covs['model'])/len(covs['model']):>6.3f} "
          f"{sum(totals['persist'])/len(totals['persist']):>8.3f} "
          f"{sum(covs['persist'])/len(covs['persist']):>6.3f} "
          f"{sum(totals['curve'])/len(totals['curve']):>8.3f} "
          f"{sum(covs['curve'])/len(covs['curve']):>6.3f}")
    one("freeze", [make_freeze_world(seed) for seed in range(4)])


if __name__ == "__main__":
    run()
