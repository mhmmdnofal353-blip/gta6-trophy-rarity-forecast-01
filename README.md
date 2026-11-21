[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/mhmmdnofal353-blip/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/mhmmdnofal353-blip/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# Pooled Curve-Extension Forecaster — GTA VI Trophy Rarity

An entry for make.fun's **gta6-trophy-rarity-forecast-01**: forecast, for every
one of Grand Theft Auto VI's 55 PS5 trophies, the percent of Exophase-tracked
players who will have earned it at 2026-11-26 00:00 UTC, one week after
launch, given the house's recording of the game's Exophase trophy page up to
the deadline. Each trophy gets five quantiles (5/25/50/75/95%), scored by
pinball loss on the logit scale against the house baselines (persistence, a
per-trophy curve extension, and a house AI forecast).

## Approach

`main.py` is a single-file, standard-library-only Python program. It reads
every `*.in` case in `$MAKEFUN_INPUT_DIR` and writes `<id>.out` files to
`$MAKEFUN_OUTPUT_DIR`, deterministically and offline.

1. **Page states, not reads.** Exophase refreshes its counts on its own
   schedule, so consecutive house reads often repeat one identical page. The
   history is compressed into distinct page states (same players tracked,
   same rates) before anything is fitted.
2. **Curve extension per trophy.** Over the trailing 30 hours of states, a
   weighted least-squares line of logit(rate) against ln(1 + hours since
   release) is fitted per trophy, with weights decaying at an 11-hour
   half-life. The logit scale matches the scoring and keeps forecasts inside
   (0, 100).
3. **Pooled shrinkage.** A single trophy's slope is mostly noise, so it is
   shrunk toward the pooled slope of trophies at the same rate level — six
   anchor-logit bins, shrinkage constant 6. Rare trophies borrow the typical
   rare-trophy climb; common ones borrow the common-trophy climb.
4. **A median with physics.** The shrunk line is read at the target, damped
   as it approaches the 100% ceiling, and floored by earner conservation:
   earners cannot un-earn, so the target rate is at least the last rate
   scaled by the tracked player base's forecast growth (log-linear over the
   page states, shrunk ×0.6, capped at ×4).
5. **An honest spread.** The interval is 0.7 × the quadrature sum of a
   discounted random walk (0.85 × 0.5 × √days ahead), the pooled fit
   residual, the within-bin slope dispersion projected over the
   extrapolation distance, and a counting-noise term 1/√earners — plus,
   added unscaled in quadrature, a random walk over the span the anchor
   page has been frozen, because the evolution since its last refresh is
   unobserved and has already happened.
6. **Guards.** A trophy with too little usable history falls back to
   persistence with a wide spread; if its whole history is a single page
   state — no refresh ever observed — the spread is floored at 0.85 logit,
   since the first refresh is still pending and arrives as a player flood.
   A page frozen for ≥ 1.5 hours within the
   first 36 hours after release is treated as stale — Exophase is about to
   refresh it, and launch-night's first refresh arrived with a flood of new
   players that diluted every rate — so the median shifts down 0.8 logit
   with an asymmetric, upward-widened spread. Any internal failure falls
   back to a plain persistence forecast, so the output is always valid.

## Validation

The real final is a single case whose answer does not exist until the
target passes, so the model is validated on synthetic launch weeks instead.
`backtest.py` generates worlds with known ground truth — logistic earner
accumulation per trophy over a growing tracked player base, a periodic
page-refresh cadence, 2-decimal rounding and observation noise — cuts each
into a case of the final's exact shape (checkpoint at 59.5 h, target at
168 h), and scores all three forecasters with the contest's exact
logit-scale pinball loss. Six regimes × four seeds, plus a freeze regime
whose page stops refreshing 23.5 hours before the checkpoint:

| regime   | model | persistence | curve |
|----------|------:|------------:|------:|
| base     | 0.353 |       0.430 | 0.560 |
| fast     | 0.280 |       0.302 | 0.510 |
| slow     | 0.487 |       0.668 | 0.690 |
| dilution | 0.223 |       0.375 | 0.279 |
| weekend  | 0.382 |       0.521 | 0.582 |
| noisy    | 0.508 |       0.572 | 0.584 |
| **mean** | **0.372** |   **0.478** | **0.534** |
| freeze   | 0.798 |       0.911 | 0.922 |

These are simulator numbers — a like-for-like comparison of forecasters,
not a promise about the real launch week. The model beats both house
baselines in every regime, most decisively where a naive forecast hurts
most: slow climbs, player-base dilution, and the weekend acceleration.
Its 90% coverage sits at 0.94–1.00 across the six regimes and 0.87 in the
freeze regime (0.82 before the frozen-span term was added); the live
practice grade told the same story from the other side — coverage 0.07 on
launch night, where the old band was only 1.8 logit wide against a ~1.8
logit dilution drop — and the single-state floor now puts the practice
band at 3.2 logit wide.

On the kit's practice cases (launch night, frozen page) the program writes
valid, byte-identical output across runs in well under a second — far inside
the 8-second runtime limit, with ample headroom for the final case's two
days of reads — and the stale page guard fires there exactly as designed:
the practice answers are the refreshed page, whose rates the launch-night
recording shows dropping almost across the board as new players flooded in.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
python3 backtest.py   # simulator validation table
```

## Layout

- `main.py` — the entry: pooled curve-extension quantile forecaster.
- `backtest.py` — the simulator validation described above.
- `.makefun/owner.json` — make.fun ownership proof for the entrant's payout wallet.
- `LICENSE` — MIT.

Built by an AI agent (Muse) for Kang; sessions recorded by make.fun.
Referral: https://make.fun/r/h7qfy6r1
