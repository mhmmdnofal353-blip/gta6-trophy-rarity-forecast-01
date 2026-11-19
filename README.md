[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/mhmmdnofal353-blip/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/mhmmdnofal353-blip/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# GTA VI trophy rarity forecast — a pooled curve-extension quantile forecaster

An entry for make.fun's **gta6-trophy-rarity-forecast-01**: forecast, for each of Grand
Theft Auto VI's 55 PS5 trophies, five quantiles (5/25/50/75/95%) of the share of tracked
players who will hold it one week after release (2026-11-26 00:00 UTC), from the house's
own reads of the game's Exophase page. The program runs offline in about a tenth of a
second, standard library only, and writes one forecast file per case.

**Bounty:** [gta6-trophy-rarity-forecast-01 on make.fun](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

## The idea

A trophy's earn rate is a ratio — the players who have earned it over the players
Exophase tracks — and over a game's first week both sides of that ratio move in log
time. Each trophy's rate, read in logit against u = ln(1 + hours since release), climbs
its own curve: fast for the trophies the first hours hand out, slower for the ones gated
behind forty hours of story, flattening as it nears the share of players who will ever
earn it this week.

No single trophy's two and a half days of reads pin its curve down. The reads arrive in
steps (the page only changes when Exophase refreshes it), the printed rates are rounded
to two decimals, and the rarest trophies are a handful of players. So the program fits
every trophy's curve and then shrinks each slope toward the pooled slope of the trophies
sitting at the same rate — the analog step: trophies at the same level grow alike, the
rare ones with the whole curve ahead of them, the common ones already bending toward
their ceiling. The median is the shrunk curve read at the target, held above the floor
the earner counts impose (earners never un-earn, so a rate can only fall as fast as the
tracked count grows), and the spread is the house random walk's, discounted for the part
the trend explains, plus the fit's own residual, the slope dispersion inside the level
bin and the counting noise of the anchor itself.

Two guards cover what the fits cannot see. If the page has not changed for hours when
the checkpoint falls early in the game's life, a refresh is pending whose first effect
is dilution — the tracked count jumps, the earners do not — so the median steps down and
the down side widens; the kit's practice cases are exactly this, one frozen launch-night
page (638 players) whose scored answer is the refreshed page (3,868 players, every rate
but the pinned one lower). And a history too thin to fit at all falls back to persistence
with an honestly wide spread, so a forecast file is always written and always valid.

## Why this scores

The contest's loss is the pinball loss on the logit scale, whose optimal forecast is the
conditional quantiles of logit(rate) — and a trophies-at-the-same-level pooled curve is a
far lower-variance estimate of each trophy's conditional median than either house
baseline's answer (the last rate held flat, or one trophy's own noisy line extended
alone over the last 24 hours). The spread is built from the same pooled quantities, so
it widens exactly where the data is thin: the rare trophies, the short histories, the
trophies whose neighbours disagree about the slope.

## Validation

The kit ships no multi-day history, so the forecaster is validated on a simulator
(`backtest.py`): worlds whose earn curves and tracked-count growth are written down
explicitly, cut into cases of exactly the final's shape and scored with the contest's
exact loss. Six regimes — base, fast, slow, dilution-heavy, weekend-shifted and
page-noisy — four seeds each:

| | model | persistence baseline | curve baseline |
|---|---|---|---|
| mean loss, 24 worlds | **0.160** | 0.410 | 0.255 |

The model wins in every regime, the noisy one included (0.182 against the curve
baseline's 0.292). These are simulator numbers — evidence the machinery beats the
baselines on worlds shaped like the launch's first days, not a measurement on real
reads, which do not exist until the deadline. The two tuning choices the simulator set —
a thirty-hour fit window and a 0.7 calibration on the spread — are round numbers, kept
well off the simulator's own optimum.

## What is in the repo

- `main.py` — the entrypoint: the forecaster described above (pure Python 3.13 standard
  library, offline, deterministic; two runs write byte-identical output).
- `backtest.py` — the simulator, both house baselines reimplemented from the rules, and
  the contest's exact scorer.
- `LICENSE` — the kit starter's MIT license, kept.

## Run it locally

With the kit's practice input (or any directory of `*.in` case files):

```
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
cat out/p00.out
```

Each `X.in` case produces `X.out`: one row per trophy, `{"id": n, "q": [q05, q25, q50,
q75, q95]}` in percent, non-decreasing, inside the scorer's 0.01..99.99 band.

## Honest limits

The final's history is 2.5 days of one game's first week; nothing in it can show a
regime the week has not produced yet — a server outage freezing earn rates, a viral
shortcut collapsing a grind trophy's curve, the tracked population's mix shifting as
casual players arrive over the weekend. The spread carries a discounted random walk
precisely for that unmodelled remainder, and the simulator's weekend and noisy regimes
check the model does not fall apart when the world bends. What the entry claims is
narrower and, on the evidence above, true: given the reads the house provides, a pooled
curve is a better answer than a flat one or a lonely one.
