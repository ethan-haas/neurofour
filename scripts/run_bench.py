"""Score every registered agent and (re)build bench_data/leaderboard.json.

    python scripts/run_bench.py            # writes leaderboard.json
    python scripts/run_bench.py --check    # recompute + assert committed file matches
                                           # (modulo latency) AND headline >= target

Exit 0 iff (in --check) the committed leaderboard matches and HEADLINE >= target.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agents import registry
from app.neurogolf import strength, cost, ladder, score
from app.neurogolf.positions import load_set
from app.neurogolf.config import seed as get_seed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "bench_data")
SEALED = os.path.join(DATA, "sealed.jsonl")
LEADERBOARD = os.path.join(DATA, "leaderboard.json")

TARGET_HEADLINE = 0.90

# Agents whose entire purpose is to BE the strength ceiling: by construction
# they must score optimality=1.0 / blunder_rate=0.0 against the sealed labels,
# always -- not "whatever the committed card happens to say". This is checked
# against the FRESH recompute, independent of what bench_data/leaderboard.json
# currently contains, so a wrong reference-agent card can never slip through
# merely by being internally self-consistent (recompute == committed) -- see
# LESSONS: a stale/corrupted app/data/solver_cache.json can silently make
# `perfect` not perfect while an unregenerated leaderboard.json still claims
# 1.0; comparing only "recompute vs committed" misses that if both sides were
# generated under the same broken cache, or if nobody re-ran --check after the
# cache drifted. This check is an absolute invariant, not a relative one.
REFERENCE_AGENTS = {"perfect"}


def _check_reference_invariants(records: list[dict]) -> list[str]:
    """Return a list of human-readable problems (empty = all reference agents
    are genuinely optimal on the fresh recompute)."""
    problems = []
    by_name = {r["name"]: r for r in records}
    for name in REFERENCE_AGENTS:
        r = by_name.get(name)
        if r is None:
            continue  # not registered in this build; nothing to check
        if r["optimality"] != 1.0 or r["blunder_rate"] != 0.0:
            problems.append(
                f"reference agent '{name}' is not exactly optimal on fresh recompute: "
                f"optimality={r['optimality']} blunder_rate={r['blunder_rate']} "
                f"(expected optimality=1.0 blunder_rate=0.0)"
            )
    return problems


def compute_leaderboard():
    if not os.path.exists(SEALED):
        print(f"ERROR: {SEALED} not found; run scripts/gen_positions.py first",
              file=sys.stderr)
        sys.exit(2)
    rows = load_set(SEALED)
    agents = registry.all_agents()

    lad = ladder.run(agents)

    records = []
    for ag in agents:
        man = ag.manifest()
        sc = strength.score(ag, rows)
        co = cost.measure(ag, rows)
        records.append({
            "name": man.name,
            "kind": man.kind,
            "optimality": sc.optimality,
            "blunder_rate": sc.blunder_rate,
            "soundness": sc.soundness,
            "per_outcome": sc.per_outcome,
            "size_bytes": co.size_bytes,
            "params": co.params,
            "flops_per_move": co.flops_per_move,
            "flops_plausible": co.flops_plausible,
            "latency_ms": co.latency_ms,
            "over_budget": co.over_budget,
            "elo": lad.elo.get(man.name, 0),
        })

    lb = score.build_leaderboard(records, seed=get_seed())
    lb["ladder"] = {
        "elo": lad.elo,
        "scores": {k: round(v, 3) for k, v in lad.scores.items()},
        "games": lad.games,
    }
    return lb


def _strip_latency(obj):
    """Deep-copy with all latency_ms fields removed (excluded from --check)."""
    obj = copy.deepcopy(obj)

    def scrub(o):
        if isinstance(o, dict):
            o.pop("latency_ms", None)
            for v in o.values():
                scrub(v)
        elif isinstance(o, list):
            for v in o:
                scrub(v)
    scrub(obj)
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--target", type=float, default=TARGET_HEADLINE)
    args = ap.parse_args()

    lb = compute_leaderboard()
    headline = lb["headline"]["value"]

    # Absolute invariant, checked on the FRESH recompute before anything else:
    # reference agents (e.g. `perfect`) must be exactly optimal. This holds in
    # BOTH modes -- it must be impossible to even *write* a leaderboard.json
    # that publishes a false "perfect" card, and `--check` must fail if the
    # live agent has drifted even when it happens to still match a stale
    # committed file.
    ref_problems = _check_reference_invariants(lb["agents"])
    if ref_problems:
        for p in ref_problems:
            print(f"CHECK FAIL: {p}", file=sys.stderr)
        sys.exit(1)

    if not args.check:
        os.makedirs(DATA, exist_ok=True)
        with open(LEADERBOARD, "w", encoding="utf-8", newline="\n") as f:
            json.dump(lb, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"wrote {LEADERBOARD}")
        print(f"HEADLINE optimality = {headline}  "
              f"({lb['headline']['agent']} @ {lb['headline']['tier']})")
        for a in lb["agents"]:
            print(f"  {a['name']:15s} opt={a['optimality']:.4f} "
                  f"size={a['size_bytes']:7d}B tier={a['tier']:5s} "
                  f"ngolf={a['neurogolf_score']:.3f} elo={a['elo']} "
                  f"pareto={a['pareto']} over_budget={a['over_budget']}")
        return

    # --check
    if not os.path.exists(LEADERBOARD):
        print("CHECK FAIL: committed leaderboard.json missing", file=sys.stderr)
        sys.exit(1)
    with open(LEADERBOARD, "r", encoding="utf-8") as f:
        committed = json.load(f)

    if _strip_latency(committed) != _strip_latency(lb):
        print("CHECK FAIL: recomputed leaderboard differs from committed (modulo latency)",
              file=sys.stderr)
        # help debugging: show headline diff
        print(f"  committed headline={committed.get('headline')}", file=sys.stderr)
        print(f"  recomputed headline={lb.get('headline')}", file=sys.stderr)
        sys.exit(1)

    if headline < args.target:
        print(f"CHECK FAIL: HEADLINE {headline} < target {args.target}", file=sys.stderr)
        sys.exit(1)

    print(f"CHECK OK: leaderboard reproducible; HEADLINE {headline} >= {args.target}")
    sys.exit(0)


if __name__ == "__main__":
    main()
