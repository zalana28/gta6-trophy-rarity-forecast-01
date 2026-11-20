# GTA VI trophy rarity forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/zalana28/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/zalana28/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Entry for the [make.fun bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01):
forecast, for every Grand Theft Auto VI trophy on PlayStation 5, the percent
of Exophase-tracked players who will have earned it at 2026-11-26T00:00:00Z,
as five quantiles (5/25/50/75/95%) per trophy.

## Run

```sh
python3 main.py   # reads /input/<id>.in, writes /output/<id>.out
```

With explicit directories (as the grader does locally):

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Standard library only, no network, deterministic: two runs on the same
case write the same file byte for byte. Runs in well under a second.

## How it forecasts

Exophase refreshes a game's counts in jumps, hours apart, not
continuously: the page sits flat, then a recount dilutes most rates
(newly tracked players hold fewer trophies) while the counts of players
who earned each trophy only grow. The program therefore works on the
*distinct* reads and models each trophy's rate on the logit scale
against log time since release, `logit(rate) ~ a + b * ln(1 + hours)`:

- slopes shrink toward their group median (story-chapter trophies
  together, the rest by tier), because one trophy's handful of reads is
  noisy and trophies demanding the same progress move together;
- where the data earns it, a concave quadratic replaces the line, so a
  trophy that is already flattening is not extrapolated past its own
  peak — earn rates saturate, they do not keep climbing on a straight
  line in log time;
- the story chain is made consistent (Chapter 1 before Chapter 2 before
  the final mission before 100% completion before the platinum), pooling
  any median that violates that order back into line;
- the spread combines a random-walk term growing with the horizon, the
  fit's own residual scatter, and the recount jumps seen in the history;
  a history with no recount at all gets a dilution prior instead, since
  a flat page means a recount is overdue.

Validated on synthetic holdouts built in the kit's own case format
(refresh jumps, player-count growth, saturating trophy progression):
the forecaster beats the house persistence baseline in a saturating
world (0.29 vs 0.35 pinball loss), a logit-linear world (0.32 vs 0.47)
and a dilution world where most rates fall (0.25 vs 0.29).

## License

MIT — see [LICENSE](LICENSE).
