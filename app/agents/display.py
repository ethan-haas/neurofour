"""Canonical human-facing display name + subtitle for every agent id.

Single source of truth so both `/agents` (app/main.py::agents) and
`/leaderboard` (app/main.py::leaderboard, injected at serve time -- the
committed `bench_data/leaderboard.json` file itself is never rewritten) show
the same friendly names. Any agent id NOT present here falls back to using
its own id as the display name (see `display_info`) -- new/experimental
agents never crash this lookup, they're just unlabelled until added here.
"""
from __future__ import annotations

# id -> (display_name, subtitle)
DISPLAY_NAMES: dict[str, tuple[str, str]] = {
    "random": ("Random", "Picks any legal column"),
    "heuristic": ("Heuristic", "One-ply pattern score"),
    "minimax-2": ("Minimax-2", "Depth-2 search"),
    "minimax-4": ("Minimax-4", "Depth-4 search"),
    "perfect": ("Oracle", "Exact solver — perfect play"),
    "neurofour-net14": ("Zero", "0-byte champion — pure bitboard search"),
    "neurofour-net13": ("Tandem", "Learned leaf + two-currency search"),
    "neurofour-net": ("Policy", "Original policy network"),
    "neurofour-net0": ("Nano", "Tiny value network"),
    "neurofour-net0b": ("Nano+", "Nano value net + refutation search"),
    "neurofour-net0d": ("Nano-D", "Nano value net + deeper search"),
    "neurofour-net1": ("Micro", "Value network"),
    "neurofour-net2": ("Micro+", "Value net + depth-3 refutation"),
    "neurofour-net4": ("Micro-Beam", "Value net + beam search"),
    "neurofour-net5": ("Policy-7", "7-way policy head"),
    "neurofour-net15": ("Multitask", "Joint policy + value net"),
    "neurofour-net15s": ("Multitask+", "Joint net + depth-3 search"),
    "neurofour-net16": ("Compressed", "2.8 KB quantized + pruned leaf"),
    "neurofour-net16s": ("Compressed+", "Compressed leaf + depth-3 search"),
    "neurofour-net16b": ("Compressed-Beam", "Compressed leaf + beam search"),
}


def display_info(agent_id: str) -> tuple[str, str]:
    """Return `(display_name, subtitle)` for `agent_id`, falling back to the
    id itself (and an empty subtitle) for any id not in `DISPLAY_NAMES` --
    never raises."""
    return DISPLAY_NAMES.get(agent_id, (agent_id, ""))
