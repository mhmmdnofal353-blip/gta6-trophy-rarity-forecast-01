"""Pooled curve-extension forecaster for GTA VI trophy earn rates.

Entry for make.fun's gta6-trophy-rarity-forecast-01. Pure Python 3 standard
library, offline, deterministic: the same input bytes always produce the same
output bytes.

For every case file (<id>.in) in the input directory the program writes
<id>.out to the output directory, holding five quantiles (5/25/50/75/95%) of
each trophy's Exophase earn rate (percent) at the case's target time.

Method, in outline:

  * The recorded page history is first compressed into distinct page states:
    consecutive reads that show the same players-tracked count and the same
    rates are one state, because Exophase refreshes its counts on its own
    schedule and the house's reads between refreshes repeat one page.
  * Each trophy's recent trajectory is a weighted least-squares line of
    logit(rate) against ln(1 + hours since release), fitted over the trailing
    30 hours of states. Working on the logit scale matches the scoring and
    keeps every forecast inside (0, 100).
  * A single trophy's slope is noisy, so it is shrunk toward the pooled slope
    of the trophies sitting at the same rate level (six anchor-logit bins,
    shrinkage constant 6): rare trophies borrow the typical rare-trophy
    climb, common ones the typical common-trophy climb.
  * The median is the shrunk line read at the target, damped as it nears the
    100% ceiling, and floored by earner conservation: the players who already
    hold a trophy cannot un-hold it, so the target rate is at least the last
    rate scaled by the (shrunk, capped) growth of the tracked player base.
  * The spread is 0.7 times the quadrature sum of a discounted random walk,
    the pooled fit residual, the within-bin slope dispersion projected over
    the extrapolation distance, and a counting-noise term 1/sqrt(earners).
    Added to that in quadrature, unscaled, is a random walk over the span
    the anchor page has been frozen: the evolution since its last refresh
    is unobserved, and it has already happened, so it is not discounted.
    A trophy whose history holds a single page state has never seen a
    refresh at all; its spread is floored at 0.85 logit, because a launch's
    first refresh arrives as a flood of new players that dilutes every
    rate at once (on launch night the tracked base jumped 6x between one
    read and the next, and every rate fell by more than the old band).
  * Two guards: a trophy with too little usable history falls back to
    persistence with a wide spread; and a page that has been frozen for at
    least 1.5 hours inside the first 36 hours after release is treated as
    stale -- Exophase is about to refresh it, historically with a flood of
    new players that dilutes every rate -- so the median is shifted down by
    0.8 logit and the spread is widened asymmetrically upward.
  * Any failure anywhere falls back to a plain persistence forecast, so the
    output is always a valid forecast file.

Run locally with MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR set; the grader's
/input and /output are the defaults.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader runs python3 -P, which leaves this file's directory off
# sys.path; put it back first so sibling modules would import.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

RATE_LO = 0.01
RATE_HI = 99.99

WINDOW_H = 30.0        # trailing fit window, hours of page states
HALF_LIFE_H = 11.0
SHRINK_K = 6.0         # slope shrinkage constant toward the bin pool
BIN_EDGES = [-3.9, -2.2, -0.85, 0.4, 1.6]   # anchor-logit edges -> 6 bins
STALE_MIN_H = 1.5      # a page frozen at least this long is stale ...
STALE_MAX_AGE_H = 36.0  # ... while the game is younger than this
STALE_SHIFT = 0.8      # logit shift applied to a stale page's median
GROWTH_SHRINK = 0.6    # shrink applied to forecast player-base growth
GROWTH_CAP = 4.0       # the player base is never forecast past 4x
SPREAD_SCALE = 0.7     # overall spread calibration
SINGLE_STATE_SIGMA = 0.85  # spread floor when no page refresh was ever seen
RW_DISCOUNT = 0.85     # discount on the random-walk spread component
RW_SIGMA = 0.5         # random-walk sigma per sqrt(day), logit units


# --------------------------------------------------------------------------
# small numeric helpers


def clamp_rate(x):
    return min(RATE_HI, max(RATE_LO, x))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(y):
    if y >= 0:
        return 100.0 / (1.0 + math.exp(-y))
    e = math.exp(y)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up; whole numbers come back as int so the
    JSON text matches what a JavaScript writer would emit."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def parse_time(text):
    """Parse an ISO-8601 timestamp into epoch seconds (float)."""
    s = text.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def release_epoch(case):
    """Release is the case's release_date at 00:00 UTC."""
    date = (case.get("game") or {}).get("release_date") or "2026-11-19"
    dt = datetime.fromisoformat(date).replace(tzinfo=timezone.utc)
    return dt.timestamp()


def hours_since(epoch, release):
    return max(0.0, (epoch - release) / 3600.0)


def u_of(hours):
    """The curve coordinate: ln(1 + hours since release)."""
    return math.log1p(max(0.0, hours))


def bin_index(anchor_logit):
    b = 0
    for edge in BIN_EDGES:
        if anchor_logit >= edge:
            b += 1
    return b


# --------------------------------------------------------------------------
# page states


def page_states(case):
    """Compress the read history into distinct page states.

    A state is a maximal run of consecutive reads with identical
    players_tracked and identical rates. Each state records the epoch of its
    first and last read, the players tracked and the rates tuple.
    """
    states = []
    for read in case["history"]:
        rates = tuple(read["rates"])
        players = read["players_tracked"]
        at = parse_time(read["at"])
        if states and states[-1]["players"] == players and states[-1]["rates"] == rates:
            states[-1]["last"] = at
            states[-1]["reads"] += 1
        else:
            states.append({
                "first": at,
                "last": at,
                "reads": 1,
                "players": players,
                "rates": rates,
            })
    return states


# --------------------------------------------------------------------------
# weighted line fit


def wls_line(xs, ys, ws):
    """Weighted least squares y ~ a + b x. Returns (a, b, rms_residual)."""
    sw = sum(ws)
    if sw <= 0:
        return None
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return (my, 0.0, 0.0)
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    a = my - b * mx
    rms = math.sqrt(sum(w * (y - (a + b * x)) ** 2 for x, y, w in zip(xs, ys, ws)) / sw)
    return (a, b, rms)


# --------------------------------------------------------------------------
# per-trophy raw fits


def trophy_fit(states, idx, release, checkpoint):
    """Fit one trophy's recent trajectory. Returns a dict, or None when the
    trophy has no usable rate anywhere in the history."""
    pts = []  # (hours since release, rate) over distinct states
    for st in states:
        rate = st["rates"][idx]
        if rate is None:
            continue
        pts.append((hours_since(st["last"], release), rate))
    if not pts:
        return None
    last_h, last_rate = pts[-1]
    anchor = logit(last_rate)

    cutoff = hours_since(checkpoint, release) - WINDOW_H
    win = [(h, r) for (h, r) in pts if h >= cutoff]
    fit = None
    if len(win) >= 3 and win[-1][0] - win[0][0] > 0.25:
        xs = [u_of(h) for h, _ in win]
        ys = [logit(r) for _, r in win]
        ws = [0.5 ** ((win[-1][0] - h) / HALF_LIFE_H) for h, _ in win]
        got = wls_line(xs, ys, ws)
        if got is not None:
            fit = {"slope": got[1], "rms": got[2], "n": len(win)}
    return {
        "last_h": last_h,
        "last_rate": last_rate,
        "anchor": anchor,
        "bin": bin_index(anchor),
        "fit": fit,
    }


# --------------------------------------------------------------------------
# player-base growth


def forecast_players(states, checkpoint, target):
    """Forecast the tracked player count at the target: log-linear growth
    over the distinct states, shrunk and capped. Returns (n_last, n_target)."""
    n_last = states[-1]["players"]
    series = [(st["last"], st["players"]) for st in states if st["players"]]
    ratio = 1.0
    if len(series) >= 2 and series[0][1] > 0 and series[-1][0] > series[0][0]:
        span_h = (series[-1][0] - series[0][0]) / 3600.0
        growth = math.log(series[-1][1] / series[0][1]) / span_h  # per hour
        ahead_h = max(0.0, (target - checkpoint) / 3600.0)
        ratio = math.exp(growth * ahead_h)
    ratio = min(GROWTH_CAP, max(1.0, ratio))
    ratio = 1.0 + GROWTH_SHRINK * (ratio - 1.0)
    return n_last, n_last * ratio


# --------------------------------------------------------------------------
# the forecast


def persistence_rows(case, extra_wide=0.0):
    """Last rate carried forward, with a random-walk spread."""
    last = case["history"][-1]
    days = max(0.0, case["horizon_minutes"] / 1440.0)
    sigma = RW_SIGMA * math.sqrt(days) + extra_wide
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = last["rates"][i]
        if rate is None:
            rate = 0.0
        centre = logit(rate)
        qs = []
        prev = RATE_LO
        for z in Z:
            q = max(clamp_rate(inv_logit(centre + z * sigma)), prev)
            prev = q
            qs.append(round4(q))
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def wide_row(trophy_id, centre_logit, sigma):
    qs = []
    prev = RATE_LO
    for z in Z:
        q = max(clamp_rate(inv_logit(centre_logit + z * sigma)), prev)
        prev = q
        qs.append(round4(q))
    return {"id": trophy_id, "q": qs}


def forecast_case(case):
    if case.get("levels") != LEVELS:
        raise ValueError("unexpected levels")
    history = case.get("history") or []
    if not history:
        raise ValueError("empty history")

    release = release_epoch(case)
    checkpoint = parse_time(case["at"])
    target = parse_time(case["target"])
    states = page_states(case)
    if not states:
        raise ValueError("no page states")

    trophies = case["trophies"]
    fits = [trophy_fit(states, i, release, checkpoint) for i in range(len(trophies))]

    # pooled statistics per anchor bin
    bin_slopes = {}
    all_rms = []
    for f in fits:
        if f and f["fit"]:
            bin_slopes.setdefault(f["bin"], []).append((f["fit"]["slope"], f["fit"]["n"]))
            all_rms.append(f["fit"]["rms"])
    every_slope = [s for vals in bin_slopes.values() for s, _ in vals]
    global_slope = (sum(every_slope) / len(every_slope)) if every_slope else 0.0
    pooled_slope = {}
    pooled_disp = {}
    for b, vals in bin_slopes.items():
        tot_n = sum(n for _, n in vals)
        pooled_slope[b] = sum(s * n for s, n in vals) / tot_n if tot_n else global_slope
        if len(vals) >= 2:
            mean = sum(s for s, _ in vals) / len(vals)
            pooled_disp[b] = math.sqrt(sum((s - mean) ** 2 for s, _ in vals) / len(vals))
        else:
            pooled_disp[b] = 0.35
    all_rms.sort()
    pooled_rms = all_rms[len(all_rms) // 2] if all_rms else 0.15

    n_last, n_target = forecast_players(states, checkpoint, target)

    # staleness of the page the checkpoint sees
    last_state = states[-1]
    frozen_h = (checkpoint - last_state["first"]) / 3600.0
    age_h = hours_since(checkpoint, release)
    stale = frozen_h >= STALE_MIN_H and age_h <= STALE_MAX_AGE_H

    days_ahead = max(0.0, (target - checkpoint) / 86400.0)
    x_target = u_of(hours_since(target, release))
    rw = RW_DISCOUNT * RW_SIGMA * math.sqrt(days_ahead)
    # The anchor page may be frozen: the evolution since its last refresh is
    # unobserved, so the random walk also runs over the frozen span.
    stale_term = RW_SIGMA * math.sqrt(max(0.0, frozen_h) / 24.0)
    single_state = len(states) == 1

    rows = []
    for i, trophy in enumerate(trophies):
        f = fits[i]
        if f is None:
            # never seen anywhere: near-zero centre, very wide spread
            sigma = 0.6 + 0.5 * math.sqrt(days_ahead)
            rows.append(wide_row(trophy["id"], logit(0.0), sigma))
            continue

        centre = f["anchor"]
        x_last = u_of(f["last_h"])
        du = max(0.0, x_target - x_last)

        if f["fit"] is None:
            # thin history: persistence median, wide spread
            sigma = math.sqrt(
                (0.35 + RW_SIGMA * math.sqrt(days_ahead)) ** 2 + stale_term ** 2
            )
            if single_state:
                # No refresh has ever been observed, so the first one is
                # still pending -- and a launch's first refresh arrives as a
                # flood of new players that dilutes every rate at once (on
                # launch night the tracked base jumped 6x between reads).
                sigma = max(sigma, SINGLE_STATE_SIGMA)
            disp_term = 0.35 * du
        else:
            pool = pooled_slope.get(f["bin"], global_slope)
            n = min(f["fit"]["n"], 24)
            slope = (n * f["fit"]["slope"] + SHRINK_K * pool) / (n + SHRINK_K)
            # keep the extrapolation sane: a fitted slope outside this band
            # is a two-point accident, not a trajectory
            slope = min(6.0, max(-3.0, slope))
            delta = slope * du
            if delta > 0:
                # saturation damping near the ceiling
                damp = min(1.0, (100.0 - clamp_rate(f["last_rate"])) / 25.0 + 0.15)
                delta *= damp
            centre = centre + delta
            disp_term = pooled_disp.get(f["bin"], 0.35) * du
            sigma = None  # from components below

        # earner-conservation floor on the median: earners cannot un-earn
        if n_target > 0:
            floor_rate = f["last_rate"] * n_last / n_target
            centre = max(centre, logit(floor_rate))

        earners = max(1.0, clamp_rate(f["last_rate"]) / 100.0 * max(n_last, 1))
        noise = 1.0 / math.sqrt(earners)
        if sigma is None:
            # the stale term is added unscaled: the walk over the frozen
            # span has already happened, it is not a forecast to discount
            sigma = math.sqrt(
                (SPREAD_SCALE * math.sqrt(
                    rw * rw + pooled_rms * pooled_rms + disp_term * disp_term
                    + noise * noise)) ** 2
                + stale_term * stale_term
            )
        sigma = min(2.5, max(0.05, sigma))

        sigma_lo = sigma
        sigma_hi = sigma
        if stale:
            centre -= STALE_SHIFT
            sigma_lo = sigma * 0.75
            sigma_hi = sigma * 1.55

        qs = []
        prev = RATE_LO
        for z in Z:
            s = sigma_hi if z > 0 else sigma_lo
            q = max(clamp_rate(inv_logit(centre + z * s)), prev)
            prev = q
            qs.append(round4(q))
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def forecast_case_safe(case):
    """The model, with a persistence safety net: whatever goes wrong, the
    grader still gets a valid forecast file for the case."""
    try:
        return forecast_case(case)
    except Exception:
        return persistence_rows(case)


# --------------------------------------------------------------------------


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        out = forecast_case_safe(case)
        text = json.dumps(out, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / (path.stem + ".out")).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
