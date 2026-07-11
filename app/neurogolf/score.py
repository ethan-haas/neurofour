"""Composite scoring, budget tiers, Pareto frontier, leaderboard (METRIC.md §5-8)."""
from __future__ import annotations

import math

from app.neurogolf.config import tier_for, TIER_RANK, FLOP_CAP, LATENCY_BUCKET_MS


def neurogolf_score(optimality: float, soundness: float, size_bytes: int) -> float:
    strength = optimality
    size_kb = size_bytes / 1024.0
    efficiency_pen = math.log2(1.0 + size_kb)
    raw = 100.0 * (0.85 * strength + 0.15 * soundness) / (1.0 + 0.15 * efficiency_pen)
    return round(raw, 3)


def _dominates(a: dict, b: dict) -> bool:
    """A dominates B iff strength>=, size<=, flops<=, with >=1 strict."""
    ge = (a["optimality"] >= b["optimality"] and
          a["size_bytes"] <= b["size_bytes"] and
          a["flops_per_move"] <= b["flops_per_move"])
    if not ge:
        return False
    strict = (a["optimality"] > b["optimality"] or
              a["size_bytes"] < b["size_bytes"] or
              a["flops_per_move"] < b["flops_per_move"])
    return strict


def build_leaderboard(records: list[dict], seed: int) -> dict:
    """`records`: one dict per agent with strength+cost+elo fields already merged."""
    agents = []
    for r in records:
        size_bytes = r["size_bytes"]
        tier = tier_for(size_bytes)
        qualifies_micro = (size_bytes <= 32_768) and (not r["over_budget"])
        agents.append({
            "name": r["name"],
            "kind": r["kind"],
            "optimality": round(r["optimality"], 6),
            "blunder_rate": round(r["blunder_rate"], 6),
            "soundness": round(r["soundness"], 6),
            "size_bytes": size_bytes,
            "params": r["params"],
            "flops_per_move": r["flops_per_move"],
            "flops_plausible": r["flops_plausible"],
            "latency_ms": round(r["latency_ms"], 3),
            "over_budget": r["over_budget"],
            "tier": tier,
            "qualifies_micro": qualifies_micro,
            "neurogolf_score": neurogolf_score(r["optimality"], r["soundness"], size_bytes),
            "elo": r["elo"],
            "per_outcome": r["per_outcome"],
        })

    # Pareto frontier (dominance on strength / size / flops)
    for a in agents:
        a["pareto"] = not any(_dominates(b, a) for b in agents if b is not a)

    # headline gate: max optimality among micro-qualifying agents, ties
    # broken with the SAME sec.5 ranking function used for tier crowns
    # (see `_rank_key` -- this used to be a second, divergent ordering that
    # dropped latency/name entirely and just took the first agent in
    # registration order on a tie; that is what let neurofour-net2 outrank
    # the tier-crown-correct neurofour-net4 on the headline path alone).
    micro_pool = [a for a in agents if a["qualifies_micro"]]
    if micro_pool:
        best = max(micro_pool, key=_rank_key)
        headline = {"metric": "optimality", "value": best["optimality"],
                    "agent": best["name"], "tier": best["tier"]}
    else:
        headline = {"metric": "optimality", "value": 0.0, "agent": None, "tier": None}

    # frontier points for plotting (ordered by size, then flops)
    frontier = sorted([a for a in agents if a["pareto"]],
                      key=lambda a: (a["size_bytes"], a["flops_per_move"], -a["optimality"]))
    frontier_by_size = [{"name": a["name"], "size_bytes": a["size_bytes"],
                         "optimality": a["optimality"], "elo": a["elo"]} for a in frontier]
    frontier_by_flops = [{"name": a["name"], "flops_per_move": a["flops_per_move"],
                          "optimality": a["optimality"], "elo": a["elo"]}
                         for a in sorted([a for a in agents if a["pareto"]],
                                         key=lambda a: (a["flops_per_move"], a["size_bytes"]))]

    auc = _auc_strength_logsize(frontier)

    # deterministic leaderboard ordering
    agents.sort(key=lambda a: (-a["neurogolf_score"], TIER_RANK[a["tier"]], a["name"]))

    return {
        "seed": seed,
        "headline": headline,
        "auc_strength_logsize": round(auc, 6),
        "tiers": {
            name: _best_in_tier(agents, name)
            for name in ("nano", "micro", "mini", "small", "open")
        },
        "agents": agents,
        "frontier": {"by_size": frontier_by_size, "by_flops": frontier_by_flops},
    }


def _rank_key(a: dict) -> tuple:
    """METRIC.md sec.5 tiebreak, used verbatim by BOTH the sec.8 headline
    gate and every sec.5 tier crown -- there is exactly ONE ranking function
    for "which agent wins this optimality tie", never two. Order:
    optimality desc, soundness desc, latency asc (bucketed per sec.9 --
    `config.LATENCY_BUCKET_MS` -- so wall-clock noise can't decide a real
    question), size_bytes asc, THEN flops_per_move asc (sec.7's own cost
    axis -- see below), and ONLY once every one of those sec.5/sec.7 keys is
    genuinely exhausted, agent name as a final deterministic fallback (so a
    result never depends on raw timing OR naming at all). Each key must
    never be reached before every key that precedes it in this tuple is
    checked -- that ordering is exactly why it's a tuple: it cannot override
    a real, earlier-listed difference this key order intends to decide.

    Why flops_per_move sits between size_bytes and name (DEFECT 2, gen-N+1
    postmortem): once optimality/soundness/bucketed-latency/size_bytes are
    ALL genuinely tied, the previous code fell straight through to `name`,
    i.e. `max()` picked whichever agent's name sorted lexicographically
    GREATEST. That is pure luck, not correctness -- proven by
    `test_dominated_agent_never_outranks_its_dominator`: renaming the exact
    same two agents (identical optimality/soundness/size_bytes/latency-
    bucket, genuinely different flops_per_move) so the Pareto-DOMINATED one
    (higher flops) sorts alphabetically after the dominator flips who wins.
    An agent that another agent strictly Pareto-dominates (sec.7: equal-or-
    better on every axis, strictly better on >=1) must never outrank its own
    dominator in any tier crown or the headline -- that is a metric-
    integrity invariant, not a style preference. flops_per_move is sec.7's
    own cost axis, so breaking a genuine remaining tie on it (ascending --
    fewer flops wins) can never contradict Pareto dominance the way `name`
    could: if A dominates B (A's flops <= B's flops, with a real difference
    left after every other key ties), this key always resolves in A's
    favor. `name` remains as the absolute last resort, purely so the result
    is fully deterministic (never depends on raw wall-clock timing) even in
    the vanishingly rare case flops_per_move is ALSO exactly equal.
    (gen-N postmortem: a headline-only path that quietly dropped the
    latency+name keys and fell back to Python's `max()`-keeps-first-on-tie
    behaviour picked whichever agent happened to be registered earlier,
    disagreeing with the tier crown -- which used this exact key -- on a
    genuine optimality/soundness/size_bytes tie. Both paths must call this
    one function so they can never diverge again.)"""
    return (
        a["optimality"], a["soundness"],
        -round(a["latency_ms"] / LATENCY_BUCKET_MS),
        -a["size_bytes"], -a["flops_per_move"], a["name"],
    )


def _best_in_tier(agents, tier_name):
    from app.neurogolf.config import TIERS
    cap = dict(TIERS)[tier_name]
    pool = [a for a in agents if a["size_bytes"] <= cap and
            (tier_name == "open" or not a["over_budget"])]
    if not pool:
        return None
    best = max(pool, key=_rank_key)
    return {"name": best["name"], "optimality": best["optimality"],
            "size_bytes": best["size_bytes"], "neurogolf_score": best["neurogolf_score"]}


def _auc_strength_logsize(frontier_by_size) -> float:
    """Trapezoidal area under strength vs log2(size+1) over the size-ordered frontier."""
    pts = [(math.log2(1.0 + a["size_bytes"] / 1024.0), a["optimality"])
           for a in frontier_by_size]
    pts = sorted(set(pts))
    if len(pts) < 2:
        return pts[0][1] if pts else 0.0
    area = 0.0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        area += (x1 - x0) * (y0 + y1) / 2.0
    return area
