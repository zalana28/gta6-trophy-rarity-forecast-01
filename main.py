#!/usr/bin/env python3
"""Forecast GTA VI trophy earn rates on Exophase at a case's target time.

For every trophy in the case, read the house's recorded reads of Exophase's
trophy page (earn rate per trophy plus the players-tracked count at each
read) and write five quantiles (5/25/50/75/95%) of that trophy's earn rate
at the target, in percent.

How it forecasts
---------------
Exophase does not refresh continuously: the page sits unchanged for hours,
then jumps when Exophase recounts. Each jump dilutes most rates, because
the newly tracked players hold fewer trophies than the early ones, while
the counts of players who have earned each trophy only ever grow. So the
program works on the *distinct* reads (consecutive identical reads carry
no new information), and models each trophy's rate on the logit scale
against log time since release:

    logit(rate) ~= a + b * ln(1 + hours since 2026-11-19T00:00Z)

The slope is shrunk toward the median slope of the trophy's group (story
chapter trophies together, other trophies grouped by tier), because a
single trophy's handful of reads is noisy and trophies that demand the
same kind of progress move together. Slopes are clamped to a sane band.

The story chain is then made consistent: completing Chapter 2 requires
Chapter 1, the final mission requires the chapters, 100% completion
requires everything, and the platinum requires all of it. Where the fitted
medians violate that order, they are pooled back into decreasing order
(pool adjacent violators on the logit scale).

The spread around each median combines three honest sources of doubt:
a random-walk term that grows with the horizon (0.5 * sqrt(days)), the
residual scatter of that trophy's own fit, and the refresh jumps seen in
the history (the typical logit jump at a recount, scaled by how many
recounts the horizon should hold). A history with no recount at all gets
a dilution prior instead: a flat page means a recount is overdue, and
the one recount the house has seen cut the average rate, so the median
is pulled down and the spread widened.

Everything is deterministic: no randomness, no network, no clock reads
beyond the case file itself. Standard library only.
"""

import json
import math
import os
import re
import sys

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard normal quantiles at the five levels.
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]
RELEASE_EPOCH = 1795123200  # 2026-11-19T00:00:00Z as a Unix timestamp.

CHAPTER_RE = re.compile(r"complete chapter (\d)", re.IGNORECASE)


def parse_time(text):
    """ISO 8601 UTC timestamp -> Unix seconds (float)."""
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    from datetime import datetime
    return datetime.fromisoformat(text).timestamp()


def clamp_rate(rate):
    """Keep a percent inside the range the logit can represent."""
    return min(99.99, max(0.01, rate))


def logit_of(rate):
    p = clamp_rate(rate) / 100.0
    return math.log(p / (1.0 - p))


def rate_of(logit):
    if logit > 20.0:
        return 99.99
    if logit < -20.0:
        return 0.01
    return 100.0 / (1.0 + math.exp(-logit))


def hours_since_release(stamp):
    return max(0.0, (parse_time(stamp) - RELEASE_EPOCH) / 3600.0)


def weighted_line(xs, ys, ws):
    """Weighted least squares for y ~ a + b*x.

    Returns (intercept, slope, residual_sd). With fewer than two distinct
    x values the slope is 0 and the level is the weighted mean.
    """
    n = len(xs)
    sw = sum(ws)
    if n == 0 or sw <= 0:
        return 0.0, 0.0, 0.0
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        resid = [y - my for y in ys]
        sd = math.sqrt(sum(r * r for r in resid) / max(1, n))
        return my, 0.0, sd
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    slope = sxy / sxx
    intercept = my - slope * mx
    resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    dof = max(1, n - 2)
    sd = math.sqrt(sum(r * r for r in resid) / dof)
    return intercept, slope, sd


def weighted_quad(xs, ys, ws):
    """Weighted least squares for y ~ a + b*x + c*x^2.

    Returns (a, b, c, residual_sd) or None when the system is degenerate.
    Solved with plain Gaussian elimination on the 3x3 normal equations.
    """
    n = len(xs)
    if n < 3:
        return None
    # Normal-equation sums.
    s = [0.0] * 5  # sum w*x^k for k = 0..4
    t = [0.0] * 3  # sum w*x^k*y for k = 0..2
    for x, y, w in zip(xs, ys, ws):
        xp = 1.0
        for k in range(5):
            s[k] += w * xp
            if k < 3:
                t[k] += w * xp * y
            xp *= x
    # Matrix rows: [s0 s1 s2 | t0], [s1 s2 s3 | t1], [s2 s3 s4 | t2].
    m = [[s[0], s[1], s[2], t[0]],
         [s[1], s[2], s[3], t[1]],
         [s[2], s[3], s[4], t[2]]]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            return None
        m[col], m[piv] = m[piv], m[col]
        for r in range(col + 1, 3):
            f = m[r][col] / m[col][col]
            for cc in range(col, 4):
                m[r][cc] -= f * m[col][cc]
    coef = [0.0] * 3
    for r in (2, 1, 0):
        coef[r] = (m[r][3] - sum(m[r][cc] * coef[cc] for cc in range(r + 1, 3))) / m[r][r]
    resid = [y - (coef[0] + coef[1] * x + coef[2] * x * x)
             for x, y in zip(xs, ys)]
    dof = max(1, n - 3)
    sd = math.sqrt(sum(r * r for r in resid) / dof)
    return coef[0], coef[1], coef[2], sd


def median(values):
    ordered = sorted(values)
    m = len(ordered)
    if m == 0:
        return 0.0
    if m % 2:
        return ordered[m // 2]
    return 0.5 * (ordered[m // 2 - 1] + ordered[m // 2])


def chain_position(trophy):
    """Order along the story chain, or None for non-story trophies.

    Chapter 1 < Chapter 2 < Chapter 3 < Chapter 4 < final mission
    < 100% completion < platinum. A trophy earlier in the chain can never
    be rarer than one later in it, because the later one requires the
    earlier one.
    """
    name = trophy.get("name", "")
    desc = trophy.get("description", "")
    tier = trophy.get("tier", "")
    if tier == "platinum":
        return 6
    if "100% completion" in desc:
        return 5
    if "final mission" in desc.lower():
        return 4
    match = CHAPTER_RE.search(desc)
    if match:
        return int(match.group(1)) - 1  # Chapter 1 -> 0, ... Chapter 4 -> 3
    return None


def group_key(trophy):
    """Shrinkage group: story trophies by chapter, the rest by tier."""
    pos = chain_position(trophy)
    if pos is not None:
        return ("chain", pos)
    return ("tier", trophy.get("tier", "bronze"))


def pool_decreasing(values):
    """Pool adjacent violators: return the closest decreasing sequence.

    `values` are logits in chain order; the result is non-increasing and
    as close as possible to the input in the least-squares sense.
    """
    blocks = [[v, 1] for v in values]  # [mean, weight]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] < blocks[i + 1][0]:
            total = blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]
            weight = blocks[i][1] + blocks[i + 1][1]
            blocks[i:i + 2] = [[total / weight, weight]]
            if i > 0:
                i -= 1
        else:
            i += 1
    out = []
    for mean, weight in blocks:
        out.extend([mean] * weight)
    return out


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history", [])
    target = case["target"]
    horizon_days = case.get("horizon_minutes", 0) / 1440.0
    horizon_hours = case.get("horizon_minutes", 0) / 60.0

    # --- distinct reads: drop consecutive repeats of the whole page. ---
    distinct = []
    for read in history:
        key = (read.get("players_tracked"), tuple(read.get("rates") or []))
        if not distinct or distinct[-1][0] != key:
            distinct.append((key, read))
    reads = [read for _, read in distinct]

    # --- refresh-jump scale: typical logit move of one trophy at a recount.
    jumps = []
    for prev, cur in zip(reads, reads[1:]):
        for a, b in zip(prev.get("rates") or [], cur.get("rates") or []):
            if a is None or b is None:
                continue
            jumps.append(abs(logit_of(b) - logit_of(a)))
    typical_jump = median(jumps) if jumps else 0.0
    gaps = []
    for prev, cur in zip(reads, reads[1:]):
        gaps.append(parse_time(cur["at"]) - parse_time(prev["at"]))
    median_gap_h = median(gaps) / 3600.0 if gaps else 0.0
    expected_recounts = 0.0
    if median_gap_h > 0:
        expected_recounts = min(12.0, horizon_hours / median_gap_h)

    flat_history = len(reads) <= 1

    x_target = math.log1p(hours_since_release(target))

    # --- per-trophy logit fit on log time (linear, plus a concave
    # quadratic where the data earns one: earn rates saturate, so a
    # straight line in log time over-extrapolates trophies that are
    # already flattening). ---
    fitted = {}  # id -> dict with the fit, the series and the last rate
    for idx, trophy in enumerate(trophies):
        xs, ys, ws = [], [], []
        for read in reads:
            rates = read.get("rates") or []
            if idx >= len(rates) or rates[idx] is None:
                continue
            n_players = read.get("players_tracked") or 1
            xs.append(math.log1p(hours_since_release(read["at"])))
            ys.append(logit_of(rates[idx]))
            ws.append(math.sqrt(max(1, n_players)))
        intercept, slope, resid = weighted_line(xs, ys, ws)
        quad = weighted_quad(xs, ys, ws) if len(xs) >= 5 else None
        last_rate = None
        for read in reversed(reads):
            rates = read.get("rates") or []
            if idx < len(rates) and rates[idx] is not None:
                last_rate = rates[idx]
                break
        fitted[trophy["id"]] = {
            "intercept": intercept, "slope": slope, "resid": resid,
            "quad": quad,
            "last_logit": logit_of(last_rate) if last_rate is not None else 0.0,
            "n": len(xs), "xs": xs, "ys": ys,
        }

    # --- shrink slopes toward their group median. ---
    group_slopes = {}
    for trophy in trophies:
        fit = fitted[trophy["id"]]
        if fit["n"] >= 2:
            group_slopes.setdefault(group_key(trophy), []).append(fit["slope"])
    group_median = {g: median(v) for g, v in group_slopes.items()}

    medians = {}
    spreads = {}
    for trophy in trophies:
        tid = trophy["id"]
        fit = fitted[tid]
        intercept, slope, resid = fit["intercept"], fit["slope"], fit["resid"]
        last_logit, n = fit["last_logit"], fit["n"]
        if n >= 2:
            prior = group_median.get(group_key(trophy), 0.0)
            slope = (n * slope + 2.0 * prior) / (n + 2.0)
        else:
            slope = 0.0
            intercept = last_logit
        slope = min(4.0, max(-1.0, slope))
        linear_logit = intercept + slope * x_target
        med_logit = linear_logit
        use_resid = resid
        # Concave quadratic: use it only when it fits the history
        # better than the line and bends down (saturation), and never
        # let it forecast past its own peak -- a saturating rate
        # plateaus, it does not turn back down.
        quad = fit["quad"]
        if quad is not None:
            qa, qb, qc, qresid = quad
            if qc < 0.0 and qresid < resid:
                x_vertex = -qb / (2.0 * qc)
                x_eval = min(x_target, x_vertex) if x_vertex > fit["xs"][-1] else x_target
                quad_logit = qa + qb * x_eval + qc * x_eval * x_eval
                med_logit = 0.7 * quad_logit + 0.3 * linear_logit
                use_resid = qresid
        med_logit = min(9.0, max(-9.0, med_logit))

        # --- spread: random walk + fit scatter + (damped) recount jumps.
        # The random-walk term already covers drift between recounts, so
        # the jump term enters at half weight instead of stacking fully.
        s_rw = 0.5 * math.sqrt(max(0.0, horizon_days))
        s_fit = min(0.6, max(0.10, use_resid)) if n >= 3 else 0.35
        s_ref = min(1.0, typical_jump * math.sqrt(expected_recounts)) if jumps else 0.0
        if flat_history:
            # No recount seen yet: one is overdue. The house's first
            # recount cut the average rate to roughly 0.7x, so pull the
            # median down and widen the band to cover a recount either
            # way; this branch only fires on a flat history. A trophy
            # every tracked player already holds is exempt: the house's
            # recount left the one such trophy at 100%, so its median
            # stays at the top and only the lower tail covers dilution.
            if last_logit < logit_of(99.0):
                med_logit = logit_of(rate_of(med_logit) * 0.7)
            spread = math.sqrt(s_rw ** 2 + 0.35 ** 2 + 0.8 ** 2)
        else:
            spread = math.sqrt(s_rw ** 2 + s_fit ** 2 + (0.5 * s_ref) ** 2)
        spread = min(2.5, max(0.15, spread))
        medians[tid] = med_logit
        spreads[tid] = spread

    # --- story-chain consistency on the medians. ---
    chain = [(chain_position(t), t["id"]) for t in trophies]
    chain = sorted([(p, i) for p, i in chain if p is not None])
    if len(chain) >= 2:
        fixed = pool_decreasing([medians[i] for _, i in chain])
        for (_, tid), value in zip(chain, fixed):
            medians[tid] = value

    rows = []
    for trophy in trophies:
        tid = trophy["id"]
        med = medians[tid]
        spread = spreads[tid]
        qs = []
        for z in Z:
            q = rate_of(med + z * spread)
            qs.append(round(min(99.99, max(0.01, q)), 4))
        # Guarantee non-decreasing after rounding.
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": tid, "q": qs})
    rows.sort(key=lambda r: r["id"])
    return {"trophies": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    if not names:
        print("no .in case files in %s" % in_dir, file=sys.stderr)
        return 1
    for name in names:
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        result = forecast_case(case)
        out_name = name[:-3] + ".out"
        with open(os.path.join(out_dir, out_name), "w", encoding="utf-8") as fh:
            json.dump(result, fh, separators=(",", ":"))
            fh.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
