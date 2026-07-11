import math

import pytest

from app.engine.board import Board
from app.neurogolf import strength, score, positions
from app.neurogolf.config import tier_for, LATENCY_CAP_MS, FLOP_CAP
from app.agents.base import Agent, AgentManifest


class StubAgent(Agent):
    """Always plays a fixed column (falls back to first legal if illegal)."""
    def __init__(self, col, name="stub", size=0, flops=10, params=0, kind="heuristic"):
        self.col = col
        self.name = name
        self.kind = kind
        self._size = size
        self._flops = flops
        self._params = params

    def select_move(self, board):
        return self.col if board.can_play(self.col) else board.legal_moves()[0]

    def manifest(self):
        return AgentManifest(self.name, self.kind, self._params, self._size, self._flops)


def _pos(board_moves, optimal_cols, best_col, scored, value):
    b = Board.from_moves(board_moves)
    return {"board": board_moves, "to_move": b.player_to_move(),
            "value": value, "optimal_cols": optimal_cols, "best_col": best_col,
            "scored": scored}


def test_strength_optimality_and_blunder():
    # craft two positions with known labels
    # position A: empty-ish, best cols {3}, playing 3 is optimal, playing 0 is a blunder (value worse)
    pA = _pos([], [3], 3, {0: -1, 1: 0, 2: 0, 3: 5, 4: 0, 5: 0, 6: 0}, 1)
    # position B (after one move): best cols {2,4}; playing 2 optimal; value draw
    pB = _pos([3], [2, 4], 2, {0: -1, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: -1}, 0)
    positions_set = [pA, pB]

    # agent always plays col 3
    a3 = StubAgent(3)
    card = strength.score(a3, positions_set)
    # A: 3 in optimal -> hit; B: 3 not in {2,4} -> miss  => optimality 0.5
    assert card.optimality == 0.5
    # blunders: A move3 value sign(5)=1 == best 1 -> not blunder; B move3 sign(0)=0 == best 0 -> not blunder
    assert card.blunder_rate == 0.0

    # agent always plays col 0 -> A: sign(-1)=-1 < best 1 -> blunder; B: sign(-1)=-1 < 0 -> blunder
    a0 = StubAgent(0)
    card0 = strength.score(a0, positions_set)
    assert card0.optimality == 0.0
    assert card0.blunder_rate == 1.0
    assert card0.soundness == 0.0


def test_tier_boundaries():
    assert tier_for(0) == "nano"
    assert tier_for(4096) == "nano"
    assert tier_for(4097) == "micro"
    assert tier_for(32768) == "micro"
    assert tier_for(32769) == "mini"
    assert tier_for(262144) == "mini"
    assert tier_for(262145) == "small"
    assert tier_for(2_097_152) == "small"
    assert tier_for(2_097_153) == "open"


def test_neurogolf_score_formula():
    # strength=1, soundness=1, size=0 -> 100*(0.85+0.15)/(1+0)=100
    assert score.neurogolf_score(1.0, 1.0, 0) == 100.0
    # size grows penalty (function rounds to 3 decimals)
    s_small = score.neurogolf_score(1.0, 1.0, 1024)         # 1KB -> pen=log2(2)=1
    assert s_small == round(100.0 / (1 + 0.15 * 1.0), 3)
    # bigger artifact scores strictly less for equal strength
    assert score.neurogolf_score(1.0, 1.0, 32768) < score.neurogolf_score(1.0, 1.0, 4096)


def test_pareto_dominance():
    a = {"optimality": 0.9, "size_bytes": 1000, "flops_per_move": 100}
    b = {"optimality": 0.8, "size_bytes": 2000, "flops_per_move": 200}
    assert score._dominates(a, b)
    assert not score._dominates(b, a)
    # equal everything -> no strict domination
    c = dict(a)
    assert not score._dominates(a, c)


def test_over_budget_gate_excludes_from_headline():
    records = [
        {"name": "cheapnet", "kind": "nn", "optimality": 0.92, "blunder_rate": 0.05,
         "soundness": 0.95, "per_outcome": {}, "size_bytes": 6000, "params": 2500,
         "flops_per_move": 5000, "flops_plausible": True, "latency_ms": 1.0,
         "over_budget": False, "elo": 500},
        {"name": "perfect", "kind": "search", "optimality": 1.0, "blunder_rate": 0.0,
         "soundness": 1.0, "per_outcome": {}, "size_bytes": 0, "params": 0,
         "flops_per_move": 50_000_000, "flops_plausible": True, "latency_ms": 80.0,
         "over_budget": True, "elo": 1200},
    ]
    lb = score.build_leaderboard(records, seed=4)
    # perfect is over_budget -> excluded from micro headline; cheapnet owns it
    assert lb["headline"]["agent"] == "cheapnet"
    assert lb["headline"]["value"] == 0.92


def test_headline_agrees_with_tier_crown_on_exact_tie():
    """Regression for the auditor's ESCAPE 1: neurofour-net2 and
    neurofour-net4 tie exactly on optimality/soundness/size_bytes (and tie
    on BUCKETED latency, since 0.011ms and 0.012ms both round to the same
    1ms bucket per config.LATENCY_BUCKET_MS). The headline gate (sec.8) and
    the micro tier crown (sec.5) MUST answer the same "who wins this tie"
    question identically -- they are literally the same question -- so
    `headline.agent` must always equal `tiers.micro.name`. Reproduces the
    exact numbers from the live leaderboard where this diverged (net2 was
    reported as headline while net4 -- the agent with strictly lower flops
    AND lower latency -- correctly held the tier crown and pareto:true)."""
    records = [
        # registered/listed FIRST, exactly like the real registry -- the
        # bug picked whichever agent came first in iteration order, so the
        # regression must keep the inferior agent first to catch that.
        {"name": "neurofour-net2", "kind": "nn", "optimality": 0.956667,
         "blunder_rate": 0.013333, "soundness": 0.986667, "per_outcome": {},
         "size_bytes": 4837, "params": 1200, "flops_per_move": 3361428,
         "flops_plausible": True, "latency_ms": 0.012, "over_budget": False,
         "elo": 500},
        {"name": "neurofour-net4", "kind": "nn", "optimality": 0.956667,
         "blunder_rate": 0.013333, "soundness": 0.986667, "per_outcome": {},
         "size_bytes": 4837, "params": 1200, "flops_per_move": 1479044,
         "flops_plausible": True, "latency_ms": 0.011, "over_budget": False,
         "elo": 500},
    ]
    lb = score.build_leaderboard(records, seed=4)
    assert lb["headline"]["agent"] == lb["tiers"]["micro"]["name"]
    assert lb["headline"]["agent"] == "neurofour-net4"
    assert lb["headline"]["value"] == 0.956667  # value unchanged, only attribution


def test_headline_matches_tier_crown_generally():
    """Property: whatever the micro-qualifying pool looks like, the headline
    gate and the micro tier crown must always name the same agent -- they
    are both "who wins the sec.5 tiebreak among micro-qualifying agents"."""
    records = [
        {"name": "alpha", "kind": "nn", "optimality": 0.9, "blunder_rate": 0.1,
         "soundness": 0.9, "per_outcome": {}, "size_bytes": 2000, "params": 10,
         "flops_per_move": 1000, "flops_plausible": True, "latency_ms": 0.5,
         "over_budget": False, "elo": 400},
        {"name": "beta", "kind": "nn", "optimality": 0.9, "blunder_rate": 0.05,
         "soundness": 0.95, "per_outcome": {}, "size_bytes": 1500, "params": 10,
         "flops_per_move": 900, "flops_plausible": True, "latency_ms": 0.4,
         "over_budget": False, "elo": 420},
        {"name": "gamma", "kind": "nn", "optimality": 0.7, "blunder_rate": 0.2,
         "soundness": 0.8, "per_outcome": {}, "size_bytes": 500, "params": 5,
         "flops_per_move": 500, "flops_plausible": True, "latency_ms": 0.2,
         "over_budget": False, "elo": 300},
    ]
    lb = score.build_leaderboard(records, seed=4)
    assert lb["headline"]["agent"] == lb["tiers"]["micro"]["name"] == "beta"


def test_lower_flops_and_latency_wins_a_genuine_tie():
    """Between two agents identical on optimality/soundness/size_bytes, the
    one with strictly lower latency (a real, non-bucket-tied difference)
    wins BOTH the headline and the tier crown -- not whichever was
    registered/listed first."""
    worse_first = {"name": "zzz-worse", "kind": "nn", "optimality": 0.8,
                    "blunder_rate": 0.1, "soundness": 0.9, "per_outcome": {},
                    "size_bytes": 3000, "params": 50, "flops_per_move": 90_000,
                    "flops_plausible": True, "latency_ms": 5.0,
                    "over_budget": False, "elo": 450}
    better_second = {"name": "aaa-better", "kind": "nn", "optimality": 0.8,
                      "blunder_rate": 0.1, "soundness": 0.9, "per_outcome": {},
                      "size_bytes": 3000, "params": 50, "flops_per_move": 10_000,
                      "flops_plausible": True, "latency_ms": 1.0,
                      "over_budget": False, "elo": 450}
    lb = score.build_leaderboard([worse_first, better_second], seed=4)
    assert lb["headline"]["agent"] == "aaa-better"
    assert lb["tiers"]["micro"]["name"] == "aaa-better"


def test_dominated_agent_never_outranks_its_dominator():
    """DEFECT 2 (metric integrity): once optimality/soundness/bucketed-
    latency/size_bytes are ALL genuinely tied, the intra-tier tiebreak must
    NEVER fall through straight to agent `name` -- doing so lets a Pareto-
    DOMINATED agent (higher flops_per_move, sec.7's cost axis) outrank its
    own dominator purely because its name sorts alphabetically greater.
    Reproduces the claim from both directions with the exact same two
    agents (only the names swapped), so the outcome cannot be explained by
    which agent happens to be spelled later in the alphabet."""
    def mk(name, flops):
        return {"name": name, "kind": "nn", "optimality": 0.9, "blunder_rate": 0.05,
                "soundness": 0.95, "per_outcome": {}, "size_bytes": 5000, "params": 100,
                "flops_per_move": flops, "flops_plausible": True, "latency_ms": 1.0,
                "over_budget": False, "elo": 500}

    # Case 1: the dominator's name sorts alphabetically LAST. (Against the
    # PRE-FIX code this happens to pass by luck -- name fallback picks the
    # lexicographically greatest name, which here is also the dominator.)
    dominator_last = mk("zzz-dominator", 1000)
    dominated_first_alpha = mk("aaa-dominated", 5000)
    assert score._dominates(dominator_last, dominated_first_alpha)
    lb1 = score.build_leaderboard([dominated_first_alpha, dominator_last], seed=4)
    assert lb1["headline"]["agent"] == "zzz-dominator"
    assert lb1["tiers"]["micro"]["name"] == "zzz-dominator"

    # Case 2: the dominator's name sorts alphabetically FIRST -- this is the
    # actual bug reproduction. Against the PRE-FIX code (tiebreak falls
    # through to `name`, and `max()` picks the lexicographically GREATEST
    # name), this asserts the DOMINATED agent ("zzz-dominated", the one
    # with strictly higher flops_per_move) wins the headline and tier
    # crown -- i.e. this exact assertion FAILS against the code before the
    # flops_per_move tiebreak key was added, proving the claim that a
    # Pareto-dominated agent could outrank its own dominator by luck of the
    # alphabet alone.
    dominator_first = mk("aaa-dominator", 1000)
    dominated_last_alpha = mk("zzz-dominated", 5000)
    assert score._dominates(dominator_first, dominated_last_alpha)
    lb2 = score.build_leaderboard([dominated_last_alpha, dominator_first], seed=4)
    assert lb2["headline"]["agent"] == "aaa-dominator"
    assert lb2["tiers"]["micro"]["name"] == "aaa-dominator"


def test_headline_still_matches_tier_crown_with_flops_tiebreak():
    """headline.agent == tiers.micro.name must still hold once the
    flops_per_move tiebreak key is in play (not just on the old size/latency
    -only tie cases)."""
    records = [
        {"name": "zzz-costly", "kind": "nn", "optimality": 0.85, "blunder_rate": 0.1,
         "soundness": 0.9, "per_outcome": {}, "size_bytes": 4000, "params": 50,
         "flops_per_move": 20_000, "flops_plausible": True, "latency_ms": 1.0,
         "over_budget": False, "elo": 500},
        {"name": "aaa-cheap", "kind": "nn", "optimality": 0.85, "blunder_rate": 0.1,
         "soundness": 0.9, "per_outcome": {}, "size_bytes": 4000, "params": 50,
         "flops_per_move": 5_000, "flops_plausible": True, "latency_ms": 1.0,
         "over_budget": False, "elo": 500},
    ]
    lb = score.build_leaderboard(records, seed=4)
    assert lb["headline"]["agent"] == lb["tiers"]["micro"]["name"] == "aaa-cheap"


def test_genuine_latency_difference_decides_before_flops():
    """A real (non-bucket-tied) latency difference must still decide the
    tie BEFORE flops_per_move is even consulted -- even when the
    lower-flops agent has the WORSE latency (i.e. the two keys disagree),
    latency (earlier in the tuple, per sec.9) wins."""
    lower_latency_higher_flops = {
        "name": "slow-cheap", "kind": "nn", "optimality": 0.8, "blunder_rate": 0.1,
        "soundness": 0.9, "per_outcome": {}, "size_bytes": 3000, "params": 50,
        "flops_per_move": 1_000, "flops_plausible": True, "latency_ms": 50.0,
        "over_budget": False, "elo": 450,
    }
    higher_latency_lower_flops = {
        "name": "fast-costly", "kind": "nn", "optimality": 0.8, "blunder_rate": 0.1,
        "soundness": 0.9, "per_outcome": {}, "size_bytes": 3000, "params": 50,
        "flops_per_move": 50_000, "flops_plausible": True, "latency_ms": 1.0,
        "over_budget": False, "elo": 450,
    }
    lb = score.build_leaderboard(
        [lower_latency_higher_flops, higher_latency_lower_flops], seed=4)
    # fast-costly has the strictly lower (non-bucket-tied) latency, so it
    # must win despite having MORE flops than slow-cheap.
    assert lb["headline"]["agent"] == "fast-costly"
    assert lb["tiers"]["micro"]["name"] == "fast-costly"


def test_flops_tiebreak_registration_order_independent():
    """The flops_per_move tiebreak (and the whole ranking) must not depend
    on which order the records list is built in."""
    def mk(name, flops):
        return {"name": name, "kind": "nn", "optimality": 0.88, "blunder_rate": 0.05,
                "soundness": 0.92, "per_outcome": {}, "size_bytes": 4500, "params": 80,
                "flops_per_move": flops, "flops_plausible": True, "latency_ms": 2.0,
                "over_budget": False, "elo": 480}

    a = mk("mmm-agent", 2_000)   # lower flops -> should win
    b = mk("nnn-agent", 9_000)

    lb_ab = score.build_leaderboard([a, b], seed=4)
    lb_ba = score.build_leaderboard([b, a], seed=4)
    assert lb_ab["headline"]["agent"] == lb_ba["headline"]["agent"] == "mmm-agent"
    assert lb_ab["tiers"]["micro"]["name"] == lb_ba["tiers"]["micro"]["name"] == "mmm-agent"


def test_positions_generation_deterministic():
    a = positions.generate(seed=4, n_train=4, n_dev=2, n_sealed=6, min_ply=22, max_ply=30)
    b = positions.generate(seed=4, n_train=4, n_dev=2, n_sealed=6, min_ply=22, max_ply=30)
    ja = [r.to_json() for r in a["sealed"]]
    jb = [r.to_json() for r in b["sealed"]]
    assert ja == jb


def test_generated_labels_are_consistent():
    sets = positions.generate(seed=4, n_train=1, n_dev=1, n_sealed=10,
                              min_ply=22, max_ply=30)
    for r in sets["sealed"]:
        # value must equal the sign of the best scored move
        best = max(r.scored.values())
        sgn = (best > 0) - (best < 0)
        assert r.value == sgn
        # optimal_cols are exactly the argmax of scored
        assert set(r.optimal_cols) == {c for c, v in r.scored.items() if v == best}
        assert r.best_col in r.optimal_cols
