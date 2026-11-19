"""Simulated backtest for the trophy-rarity forecaster in main.py.

The kit ships no multi-day history -- its practice cases are a single frozen launch-night
page, and the final's history does not exist until the deadline -- so the honest local
validation is a simulator. A world here is written down explicitly: each of the 55
trophies' earn share follows a log-time adoption curve (a ceiling times a logistic in
ln(1 + hours since release), the parameters drawn by category -- story chapters in order,
named missions, one-off activities, the long grinds, the platinum last, the opening
mission pinned at everyone), the tracked-player count follows the launch's observed shape
(638 on the frozen launch-night page, 3,868 at the first refresh, decelerating growth
after), and the page the house reads only changes when Exophase refreshes it, every
20-110 minutes, with rates rounded to two decimals. The world is then cut into a case of
exactly the final's shape -- reads every ten minutes to a checkpoint 59.5 hours after
release, the target at 168 hours -- and scored with the contest's exact loss, the mean
pinball loss on the logit scale.

Six regimes vary the world: base, fast (every curve shifted earlier), slow, dilution (the
tracked count grows faster, dragging rates down), weekend (earning speeds up after the
first weekend begins) and noisy (each refresh also carries per-trophy page noise, and the
scored answer -- itself a page read in the target hour -- carries it too). The model under
test, the house persistence baseline (the kit starter's method) and the house curve
baseline (reimplemented from the rules: a per-trophy least-squares line of logit(rate)
against ln(1 + hours) over the last 24 hours of reads, the spread the random walk's and
the fit's residual combined) all read the same simulated cases; none of them sees the
world's parameters.

    python3 backtest.py

On the default seeds the model scores near 0.16 against the curve baseline's 0.25 and the
persistence baseline's 0.41, in every regime. These are simulator numbers -- evidence the
machinery beats the baselines on worlds shaped like the launch's first days, not a
measurement on real reads, which do not exist yet.
"""

import importlib.util
import math
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
EPOCH = datetime(2026, 11, 19, tzinfo=timezone.utc)
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
CHECKPOINT_H = 59.5
TARGET_H = 168.0


def load_model():
    spec = importlib.util.spec_from_file_location("entry_model", HERE / "main.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


model = load_model()


def clamp_rate(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def round4(x):
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    return whole / 10000


def pinball_loss(forecast, truth):
    """The contest's loss: mean pinball on the logit scale over trophies and levels."""
    total = 0.0
    count = 0
    for row in forecast["trophies"]:
        y = truth[row["id"]]
        for tau, q in zip(LEVELS, row["q"]):
            d = logit(y) - logit(q)
            total += 2 * max(tau * d, (tau - 1) * d)
            count += 1
    return total / count


# ---------------------------------------------------------------- the house baselines

def persistence_baseline(case):
    """The house persistence baseline, the kit starter's method: the median is the last
    read's rate, the spread the random walk's 0.5 per square-root day on the logit."""
    horizon_days = case["horizon_minutes"] / 1440.0
    spread = 0.5 * math.sqrt(horizon_days)
    last = case["history"][-1]["rates"]
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = last[i] if last[i] is not None else 0
        lg = logit(rate)
        rows.append({"id": trophy["id"], "q": [
            round4(min(100.0, max(0.0, 100.0 / (1 + math.exp(-(lg + z * spread))))))
            for z in Z]})
    return {"trophies": rows}


def curve_baseline(case):
    """The house curve baseline, from the rules: per trophy, a least-squares line of
    logit(rate) against ln(1 + hours since release) over the case's last 24 hours of
    reads, read at the target; the spread the random walk's and the fit's residual
    combined; under 3 reads, the persistence baseline's answer."""
    checkpoint = model.parse_time(case["at"])
    target = model.parse_time(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0
    rw = 0.5 * math.sqrt(horizon_days)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        pts = []
        for read in case["history"]:
            at = model.parse_time(read["at"])
            if (checkpoint - at).total_seconds() > 24 * 3600:
                continue
            rate = read["rates"][i]
            if rate is None:
                continue
            age_h = (at - EPOCH).total_seconds() / 3600.0
            pts.append((math.log(1 + max(age_h, -0.9)), logit(rate)))
        if len(pts) < 3:
            rate = case["history"][-1]["rates"][i]
            lg = logit(rate if rate is not None else 0)
            qs = [100.0 / (1 + math.exp(-(lg + z * rw))) for z in Z]
        else:
            n = float(len(pts))
            mu = sum(p[0] for p in pts) / n
            ml = sum(p[1] for p in pts) / n
            sxx = sum((p[0] - mu) ** 2 for p in pts)
            slope = (sum((p[0] - mu) * (p[1] - ml) for p in pts) / sxx) if sxx > 1e-12 else 0.0
            resid = math.sqrt(sum((p[1] - (ml + slope * (p[0] - mu))) ** 2 for p in pts) / n)
            target_h = (target - EPOCH).total_seconds() / 3600.0
            lg = ml + slope * (math.log(1 + target_h) - mu)
            spread = math.sqrt(rw ** 2 + resid ** 2)
            qs = [100.0 / (1 + math.exp(-(lg + z * spread))) for z in Z]
        rows.append({"id": trophy["id"],
                     "q": [round4(min(100.0, max(0.0, q))) for q in qs]})
    return {"trophies": rows}


# ---------------------------------------------------------------- the simulated worlds

def logistic(x):
    return 1.0 / (1.0 + math.exp(-x))


def world_params(rng, regime):
    """Per-trophy (ceiling, mu, s) for the 55 trophies, by the real list's categories."""
    shift = {"fast": -0.55, "slow": 0.55, "base": 0.0, "weekend": 0.1,
             "dilution": 0.15, "noisy": 0.0}[regime]
    params = []
    for tid in range(1, 56):
        if tid == 2:
            params.append((1.0, -6.0, 0.5))  # the opening mission: everyone tracked has it
        elif tid in (3, 4, 5, 6, 7):  # chapters 1..4 and the final mission
            k = tid - 3
            params.append((0.58 - 0.07 * k, 4.35 + 0.38 * k + shift + rng.uniform(-0.12, 0.12),
                           rng.uniform(0.8, 1.05)))
        elif tid in (8, 9, 10, 11, 12, 13):  # named story missions
            params.append((rng.uniform(0.3, 0.48), rng.uniform(4.3, 5.3) + shift,
                           rng.uniform(0.8, 1.1)))
        elif tid in (1, 14, 15, 17, 25, 30, 31, 32):  # platinum, 100%, the long grinds
            params.append((0.004 if tid == 1 else rng.uniform(0.01, 0.09),
                           rng.uniform(5.9, 7.3) + shift, rng.uniform(0.7, 1.0)))
        else:  # one-off activities and collectibles
            params.append((rng.uniform(0.12, 0.5), rng.uniform(3.6, 5.4) + shift,
                           rng.uniform(0.75, 1.15)))
    return params


def tracked_count(h, regime):
    """The tracked count's true path: the frozen launch-night page, the first refresh's
    3,868 at h = 4.5, then decelerating growth."""
    if h < 4.5:
        return 3868.0 * (0.165 + 0.835 * max(h + 0.167, 0.0) / 4.667)
    exponent = 1.08 if regime == "dilution" else 0.85
    return 3868.0 * ((h + 4.5) / 4.5) ** exponent


def earn_share(params, h, regime):
    ceiling, mu, s = params
    x = (math.log(h + 1.0) - mu) / s
    share = ceiling * logistic(x)
    if regime == "weekend" and h > 108:  # the first weekend: earning speeds up late
        share = ceiling * logistic(x + 0.35 * (1 - logistic(x)))
    return share


def simulate(seed, regime):
    """One world cut into a case of the final's shape, plus its answer at the target."""
    rng = random.Random(seed)
    params = world_params(rng, regime)
    trophies = [{"id": tid} for tid in range(1, 56)]
    refreshes = []
    t = -0.167
    while t < TARGET_H:
        refreshes.append(t)
        t += rng.uniform(20, 110) / 60.0
    noise = {}
    if regime == "noisy":
        nrng = random.Random(seed + 777)
        for ri in range(len(refreshes)):
            noise[ri] = [nrng.gauss(0, 0.18) for _ in trophies]

    def page_at(h):
        shown = max(r for r in refreshes if r <= h)
        ri = refreshes.index(shown)
        n = tracked_count(shown, regime)
        rates = []
        for i in range(55):
            share = earn_share(params[i], shown, regime)
            if regime == "noisy" and 0.0 < share < 1.0:
                lg = math.log(share / (1 - share)) + noise[ri][i]
                share = 1.0 / (1.0 + math.exp(-lg))
            earners = share * n
            rates.append(round(100.0 * earners / n, 2) if earners >= 0.5 else 0.0)
        return int(round(n)), rates

    history = []
    h = -0.167
    while h < CHECKPOINT_H:
        n, rates = page_at(h)
        at = EPOCH + timedelta(hours=h)
        history.append({
            "slot": at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "at": at.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "players_tracked": n,
            "trophies_earned": int(round(sum(rates) / 100.0 * n)),
            "rates": rates,
        })
        h += 10.0 / 60.0
    case = {
        "schema": "makefun/gta6-trophies-case/v1",
        "id": "f00",
        "at": (EPOCH + timedelta(hours=CHECKPOINT_H)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "horizon_minutes": int((TARGET_H - CHECKPOINT_H) * 60),
        "target": (EPOCH + timedelta(hours=TARGET_H)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "levels": LEVELS,
        "game": {"name": "Grand Theft Auto VI", "platform": "PS5",
                 "release_date": "2026-11-19",
                 "source": "https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/"},
        "trophies": trophies,
        "history": history,
        "notes": "simulated",
    }
    _n, rates_t = page_at(TARGET_H - 0.01)
    truth = {tid: (rates_t[tid - 1] if rates_t[tid - 1] > 0 else 0.01)
             for tid in range(1, 56)}
    return case, truth


def main():
    seeds = [int(a) for a in sys.argv[1:]] or [11, 23, 37, 41]
    regimes = ["base", "fast", "slow", "dilution", "weekend", "noisy"]
    totals = {"model": 0.0, "persistence": 0.0, "curve": 0.0}
    runs = 0
    for regime in regimes:
        sub = {"model": 0.0, "persistence": 0.0, "curve": 0.0}
        for seed in seeds:
            case, truth = simulate(seed * 1000 + len(regime), regime)
            sub["model"] += pinball_loss(model.forecast_case(case), truth)
            sub["persistence"] += pinball_loss(persistence_baseline(case), truth)
            sub["curve"] += pinball_loss(curve_baseline(case), truth)
            runs += 1
        for k in totals:
            totals[k] += sub[k]
        n = float(len(seeds))
        print(f"{regime:>9}: model {sub['model'] / n:.4f}  "
              f"persistence {sub['persistence'] / n:.4f}  curve {sub['curve'] / n:.4f}")
    print(f"\nmean over {runs} worlds: model {totals['model'] / runs:.4f}  "
          f"persistence {totals['persistence'] / runs:.4f}  "
          f"curve {totals['curve'] / runs:.4f}")


if __name__ == "__main__":
    main()
