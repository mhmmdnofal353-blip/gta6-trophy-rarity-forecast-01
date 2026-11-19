"""gta6-trophy-rarity-forecast-01 entry: a pooled curve-extension quantile forecaster.

The idea, in one paragraph: a trophy's earn rate on Exophase is a ratio -- the players who
have earned it over the players tracked -- and over a game's first week both sides of that
ratio move in log time. Each trophy's rate, read in logit against u = ln(1 + hours since
release), climbs its own curve: fast for the trophies the first hours hand out, slower for
the ones gated behind forty hours of story, flattening as it nears the share of players who
will ever earn it this week. No single trophy's two and a half days of reads pin its curve
down -- the reads arrive in steps (the page only changes when Exophase refreshes it), the
printed rates are rounded to two decimals, and the rarest trophies are a handful of players
-- so this program fits every trophy's curve and then shrinks each slope toward the pooled
slope of the trophies sitting at the same rate, the analog step: trophies at the same level
grow alike. The median is the shrunk curve read at the target, held above the floor the
earner counts impose (earners never un-earn, so a rate can only fall as fast as the tracked
count grows), and the spread is the house random walk's, discounted for the part the trend
explains, plus the fit's own residual, the slope dispersion inside the level bin and the
counting noise of the anchor itself.

Why this scores: the contest's loss is the pinball loss on the logit scale, whose optimal
forecast is the conditional quantiles of logit(rate) -- and a trophies-at-the-same-level
pooled curve is a far lower-variance estimate of each trophy's conditional median than
either house baseline's answer (the last rate held flat, or one trophy's own noisy line
extended alone). The spread is built from the same pooled quantities, so it widens exactly
where the data is thin: the rare trophies, the short histories, the trophies whose
neighbours disagree about the slope.

Two guards cover what the fits cannot see. If the page has not changed for hours when the
checkpoint falls early in the game's life, a refresh is pending whose first effect is
dilution (the tracked count jumps, the earners do not), so the median steps down and the
down side widens. And whatever happens, a trophy whose last rate is the page's ceiling or
floor is forecast at that bound, not extrapolated through it.

The program is deterministic, standard library only, and never touches the network: for
each case it reads only that case's `trophies` and `history`. If a case's history is too
thin to fit (a single unchanging page), it falls back to persistence with an honestly wide
spread, so a forecast file is always written and always valid.

Run locally the way the kit describes:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
The grader runs it as `python3 -P -s /bot/main.py`; the sys.path line below puts this
file's own directory back on the path, as the kit's starter notes explain.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module shipped beside
# this file would import (this entry is a single file, but the habit is the kit's own).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each output row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# The house random walk's spread, in logit units per square-root day: the yardstick every
# spread below is built around.
RW_SIGMA = 0.5
# How much of the random walk the fitted trend is trusted to explain away. Below 1: a
# fitted curve that is partly right should still carry most of the walk's uncertainty.
RW_DISCOUNT = 0.85
# A final calibration on the whole spread: the components above are each honest, but
# they are not independent (a trophy whose slope is off also fits worse), so their
# quadrature sum overstates the whole and is scaled back. The scale (and the fit window
# below) were set on simulated week-1 worlds -- logistic earn curves in log time, a
# decelerating tracked count, Exophase-style stale refreshes, 2-decimal rounding, in
# fast, slow, dilution-heavy, weekend-shifted and page-noisy variants -- where this
# configuration's loss ran near 0.16 against the curve baseline's 0.25 and the
# persistence baseline's 0.41; see backtest.py.
SIGMA_SCALE = 0.7
# The shrinkage strength of the analog step, in effective observations: a trophy's own
# slope counts for its number of distinct page states (capped), the pooled slope of its
# level bin for this many.
SHRINK_K = 6.0
# The fit's window, in hours before the checkpoint (None: every page the case holds).
# Launch night's frozen page and first-refresh cliff are a different regime from the one
# the forecast extends, so the fit is kept to the recent pages: the last thirty hours,
# the same span the house curve baseline reads, but pooled.
FIT_WINDOW_H = 30.0
# Recency weighting inside the fit: 0 weights every page by its duration alone, 1 tilts
# the weights from half at the window's start to one and a half at its end.
RECENCY = 0.0
# Slopes (logit per unit of ln(1 + hours)) are never trusted outside this band.
SLOPE_MIN, SLOPE_MAX = -1.0, 4.0
# A page frozen this long at the checkpoint may be hiding a refresh.
STALE_HOURS = 1.5
# The launch-window dilution prior applies only while the game is this young: a frozen
# page then means the tracked count is about to jump under a fixed set of earners.
LAUNCH_WINDOW_H = 36.0
STALE_DILUTION = 0.8
STALE_SIGMA_DOWN = 1.3
STALE_SIGMA_UP = 0.9
# With less than this in the history there is nothing to fit: persistence, widely spread.
MIN_STATES = 3
MIN_SPAN_H = 3.0
THIN_SIGMA = 1.1
# A page frozen for hours at any age hides whatever happened since it froze; outside the
# launch window there is no dilution prior to shift the median with, but the spread
# still widens to at least this.
STALE_SIGMA_LATE = 1.25


def clamp_rate(x):
    """A rate in percent clamped to the scorer's own band, 0.01..99.99."""
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def round4(x):
    """x rounded to 4 decimals, half up; whole numbers come back as int for compact JSON."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def parse_time(text):
    """An ISO 8601 UTC timestamp from a case file, as an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def weighted_median(pairs):
    """The weighted median of (value, weight) pairs."""
    ordered = sorted(pairs)
    total = sum(w for _, w in ordered)
    if total <= 0:
        return ordered[len(ordered) // 2][0]
    acc = 0.0
    for value, weight in ordered:
        acc += weight
        if acc >= total / 2:
            return value
    return ordered[-1][0]


def weighted_quantile(pairs, tau):
    """The tau quantile of (value, weight) pairs, linear in the cumulative weight."""
    ordered = sorted(pairs)
    total = sum(w for _, w in ordered)
    if total <= 0:
        return ordered[len(ordered) // 2][0]
    target = tau * total
    acc = 0.0
    for i, (value, weight) in enumerate(ordered):
        acc += weight
        if acc >= target:
            if i == 0:
                return value
            prev = ordered[i - 1][0]
            frac = (target - (acc - weight)) / weight if weight > 0 else 0.0
            return prev + (value - prev) * frac
    return ordered[-1][0]


class State:
    """One distinct page: a stretch of reads that all showed the same thing.

    The page only changes when Exophase refreshes it, so the reads between two changes
    are one observation, not many; a state is kept with the first read that showed it,
    the read that ended it, and how long it lasted.
    """

    __slots__ = ("at", "until", "players", "rates")

    def __init__(self, at, players, rates):
        self.at = at          # hours since release of the first read showing this page
        self.until = at       # hours since release of the last read showing it
        self.players = players
        self.rates = rates


def case_states(case, epoch):
    """The case's history compressed to its distinct pages, oldest first."""
    states = []
    for read in case.get("history") or []:
        try:
            at = (parse_time(read["at"]) - epoch).total_seconds() / 3600.0
            players = read.get("players_tracked")
            rates = read.get("rates") or []
        except (KeyError, TypeError, ValueError):
            continue
        if states and states[-1].players == players and states[-1].rates == list(rates):
            states[-1].until = at
        else:
            states.append(State(at, players, list(rates)))
    return states


def log_time(hours):
    """u = ln(1 + hours since release), the clock every curve is fitted against."""
    return math.log(1.0 + max(hours, -0.9))


class TrophyFit:
    """One trophy's fitted curve and everything the forecast needs from it."""

    __slots__ = ("index", "last_rate", "anchor", "anchor_u", "slope", "weight",
                 "resid_sd", "earners", "bin")

    def __init__(self, index):
        self.index = index
        self.last_rate = None
        self.anchor = 0.0
        self.anchor_u = 0.0
        self.slope = None
        self.weight = 0.0
        self.resid_sd = None
        self.earners = 0.0
        self.bin = 0


# Anchor-logit bin edges: trophies are pooled with the ones at their own level, because a
# trophy's growth in logit terms is set mostly by where it sits -- the rare ones have the
# whole curve ahead of them, the common ones are already bending toward their ceiling.
BIN_EDGES = [logit(0.3), logit(1.0), logit(3.0), logit(10.0), logit(30.0)]


def bin_of(anchor_logit):
    b = 0
    while b < len(BIN_EDGES) and anchor_logit > BIN_EDGES[b]:
        b += 1
    return b


def fit_trophy(index, states, checkpoint_h):
    """Fit one trophy's logit curve against log time over the case's distinct pages.

    The anchor is the duration-weighted level of the last few hours of pages (the level
    the forecast grows from); the slope is a weighted least-squares line through every
    page, weighted by the square root of how long each page lasted, so a launch-night
    plateau counts once and the recent pages count most.
    """
    fit = TrophyFit(index)
    points = []  # (u, logit, weight, hours, rate, players)
    for j, state in enumerate(states):
        rate = state.rates[index] if index < len(state.rates) else None
        if rate is None:
            continue
        end = states[j + 1].at if j + 1 < len(states) else checkpoint_h
        duration = max(end - state.at, 0.05)
        points.append((log_time(state.at), logit(rate), math.sqrt(min(duration, 6.0)),
                       state.at, float(rate), state.players))
        fit.last_rate = float(rate)
        if state.players:
            fit.earners = float(rate) / 100.0 * state.players
    if FIT_WINDOW_H is not None and len(points) > 3:
        recent = [p for p in points if p[3] >= checkpoint_h - FIT_WINDOW_H]
        if len(recent) >= 3:
            points = recent
    if RECENCY and len(points) > 1:
        h0 = min(p[3] for p in points)
        h1 = max(p[3] for p in points)
        if h1 > h0:
            points = [(u, l, w * (1.0 + RECENCY * ((h - h0) / (h1 - h0) - 0.5)), h, r, n)
                      for (u, l, w, h, r, n) in points]
    if not points:
        return fit
    # The anchor: the trailing pages' levels, weighted by how long each lasted inside
    # the last eight hours (the final page counts up to the checkpoint).
    window_start = checkpoint_h - 8.0
    num = 0.0
    den = 0.0
    for j, state in enumerate(states):
        rate = state.rates[index] if index < len(state.rates) else None
        if rate is None:
            continue
        end = states[j + 1].at if j + 1 < len(states) else checkpoint_h
        overlap = max(0.0, min(end, checkpoint_h) - max(state.at, window_start))
        if overlap > 0:
            num += overlap * logit(rate)
            den += overlap
    fit.anchor = num / den if den > 0 else points[-1][1]
    fit.anchor_u = log_time(checkpoint_h)
    if len(points) >= 3:
        sw = sum(p[2] for p in points)
        mu_u = sum(p[2] * p[0] for p in points) / sw
        mu_l = sum(p[2] * p[1] for p in points) / sw
        sxx = sum(p[2] * (p[0] - mu_u) ** 2 for p in points)
        if sxx > 1e-9:
            sxy = sum(p[2] * (p[0] - mu_u) * (p[1] - mu_l) for p in points)
            slope = sxy / sxx
            resid = [p[1] - (mu_l + slope * (p[0] - mu_u)) for p in points]
            fit.resid_sd = math.sqrt(sum(r * r for r in resid) / len(resid))
            fit.slope = min(SLOPE_MAX, max(SLOPE_MIN, slope))
            span_u = max(p[0] for p in points) - min(p[0] for p in points)
            fit.weight = min(len(points), 12) * min(span_u, 1.5)
    fit.bin = bin_of(fit.anchor)
    return fit


def fit_players(states):
    """The tracked count's own slope against log time, from the distinct pages."""
    points = [(log_time(s.at), math.log(s.players)) for s in states if s.players]
    if len(points) < 3:
        return 0.0
    sw = float(len(points))
    mu_u = sum(p[0] for p in points) / sw
    mu_l = sum(p[1] for p in points) / sw
    sxx = sum((p[0] - mu_u) ** 2 for p in points)
    if sxx <= 1e-9:
        return 0.0
    return sum((p[0] - mu_u) * (p[1] - mu_l) for p in points) / sxx


def pooled_slopes(fits):
    """Each level bin's pooled slope and slope dispersion, and the global pair.

    A bin with fewer than three fitted trophies borrows the global figures; a bin's
    dispersion is its slopes' interquartile range, robust to the one wild fit.
    """
    def pool(group):
        if not group:
            return None, 0.0
        pairs = [(f.slope, max(f.weight, 0.1)) for f in group]
        med = weighted_median(pairs)
        q25 = weighted_quantile(pairs, 0.25)
        q75 = weighted_quantile(pairs, 0.75)
        return med, (q75 - q25) / 1.349

    fitted = [f for f in fits if f.slope is not None]
    global_med, global_sd = pool(fitted)
    if global_med is None:
        global_med, global_sd = 0.0, 0.5
    bins = {}
    for b in range(len(BIN_EDGES) + 1):
        group = [f for f in fitted if f.bin == b]
        if len(group) >= 3:
            bins[b] = pool(group)
        else:
            bins[b] = (global_med, global_sd)
    return bins, (global_med, global_sd)


def saturation_damp(anchor_logit):
    """How much of a positive logit step survives near the ceiling.

    A trophy already held by a third of the tracked players cannot keep a rare trophy's
    pace: its remaining earners are the slow tail. The damp runs from ~1 for rare
    trophies to ~0.6 for common ones.
    """
    return 0.55 + 0.45 / (1.0 + math.exp((anchor_logit - 0.5) / 0.75))


def forecast_case(case):
    """One case's forecast object: {"trophies": [{"id": n, "q": [...]}, ...]}."""
    trophies = case.get("trophies") or []
    checkpoint = parse_time(case["at"])
    target = parse_time(case["target"])
    try:
        epoch = datetime.fromisoformat(str(case["game"]["release_date"])).replace(
            tzinfo=timezone.utc)
    except (KeyError, TypeError, ValueError):
        epoch = checkpoint
    checkpoint_h = (checkpoint - epoch).total_seconds() / 3600.0
    target_h = (target - epoch).total_seconds() / 3600.0
    horizon_days = max(case.get("horizon_minutes", 0) / 1440.0, 0.0)
    states = case_states(case, epoch)
    last = (case.get("history") or [{}])[-1]
    last_rates = last.get("rates") or []

    def last_rate(i):
        rate = last_rates[i] if i < len(last_rates) else None
        if rate is None:
            for state in reversed(states):
                if i < len(state.rates) and state.rates[i] is not None:
                    rate = state.rates[i]
                    break
        return float(rate) if rate is not None else 0.01

    rw = RW_SIGMA * math.sqrt(horizon_days)
    span_h = (states[-1].until - states[0].at) if states else 0.0
    stale_h = (checkpoint_h - states[-1].at) if states else 0.0

    # Too thin to fit: persistence, with a spread honest about a page that may be stale.
    if len(states) < MIN_STATES or span_h < MIN_SPAN_H:
        sigma = max(rw, THIN_SIGMA)
        launch_stale = stale_h >= STALE_HOURS and checkpoint_h <= LAUNCH_WINDOW_H
        if stale_h >= STALE_HOURS and not launch_stale:
            sigma = max(sigma, STALE_SIGMA_LATE)
        rows = []
        for i, trophy in enumerate(trophies):
            med = logit(last_rate(i))
            if launch_stale:
                med -= STALE_DILUTION
                qs = [100.0 * sigmoid(med + z * (STALE_SIGMA_DOWN if z < 0 else STALE_SIGMA_UP))
                      for z in Z]
            else:
                qs = [100.0 * sigmoid(med + z * sigma) for z in Z]
            rows.append({"id": trophy["id"], "q": finish_row(qs)})
        return {"trophies": rows}

    fits = [fit_trophy(i, states, checkpoint_h) for i in range(len(trophies))]
    bins, _global = pooled_slopes(fits)
    resids = [f.resid_sd for f in fits if f.resid_sd is not None]
    fit_sd = sorted(resids)[len(resids) // 2] if resids else 0.25
    # The tracked count at the target, grown at 60% of its fitted log-time slope and
    # capped: the floor it sets on every rate (earners never un-earn).
    players_last = states[-1].players or 0
    du = log_time(target_h) - log_time(checkpoint_h)
    growth = math.exp(0.6 * fit_players(states) * du) if players_last else 1.0
    growth = min(max(growth, 1.0), 4.0)
    stale = stale_h >= STALE_HOURS and checkpoint_h <= LAUNCH_WINDOW_H

    rows = []
    for i, trophy in enumerate(trophies):
        f = fits[i]
        rate = f.last_rate if f.last_rate is not None else last_rate(i)
        bin_med, bin_sd = bins[f.bin]
        if f.slope is None:
            slope = bin_med
        else:
            slope = (f.weight * f.slope + SHRINK_K * bin_med) / (f.weight + SHRINK_K)
        step = slope * du
        if step > 0:
            step *= saturation_damp(f.anchor)
        med = f.anchor + step
        # The counts floor: the earners already counted, spread over the grown tracked
        # count, is the lowest the median may sit.
        if players_last:
            floor_rate = rate * (1.0 / growth)
            med = max(med, logit(floor_rate))
        if stale:
            med -= STALE_DILUTION * 0.5
        meas = min(1.2, 1.0 / math.sqrt(max(f.earners, 1.0)))
        slope_sd = bin_sd * du
        sigma = SIGMA_SCALE * math.sqrt((RW_DISCOUNT * rw) ** 2 + fit_sd ** 2
                                        + slope_sd ** 2 + meas ** 2)
        if stale:
            qs = [100.0 * sigmoid(med + z * (max(sigma, STALE_SIGMA_DOWN) if z < 0
                                             else max(sigma, STALE_SIGMA_UP)))
                  for z in Z]
        else:
            qs = [100.0 * sigmoid(med + z * sigma) for z in Z]
        rows.append({"id": trophy["id"], "q": finish_row(qs)})
    return {"trophies": rows}


def finish_row(qs):
    """Quantiles in percent: clamped to the scorer's band, non-decreasing, rounded."""
    out = []
    prev = 0.01
    for q in qs:
        value = round4(min(99.99, max(0.01, q)))
        value = max(value, prev)
        out.append(value)
        prev = value
    return out


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case)
        except Exception:
            # Never fail a case: persistence at the last read beats an invalid one.
            forecast = fallback_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


def fallback_case(case):
    """The house persistence row for every trophy, from the case's last read."""
    last = (case.get("history") or [{}])[-1]
    rates = last.get("rates") or []
    horizon_days = max(case.get("horizon_minutes", 0) / 1440.0, 0.0)
    spread = RW_SIGMA * math.sqrt(horizon_days)
    rows = []
    for i, trophy in enumerate(case.get("trophies") or []):
        rate = rates[i] if i < len(rates) and rates[i] is not None else 0.01
        lg = logit(rate)
        rows.append({"id": trophy["id"],
                     "q": finish_row([100.0 * sigmoid(lg + z * spread) for z in Z])})
    return {"trophies": rows}


if __name__ == "__main__":
    main()
