"""Reusable held-out-position collision guard for any trainer that derives
NEW training samples from CHILDREN of `train.jsonl`/`dev.jsonl` positions
(gen-net9 and onward).

Why this exists: `train.jsonl`/`dev.jsonl` rows are themselves disjoint from
`sealed.jsonl`/`dev_big.jsonl` (different generation seeds/ranges), but a
CHILD of a train/dev position (one ply deeper) can coincide with a position
that IS in a held-out set. If a trainer builds per-child samples (rather
than per-parent samples, as the original `train_net1.py` did) it must guard
against training directly on a held-out position -- otherwise any dev_big/
sealed optimality gain could be partially or wholly a memorization artifact
rather than genuine generalization.

Usage:
    from app.agents.contamination_guard import build_key_sets, board_key
    banned = build_key_sets({
        "sealed": "bench_data/sealed.jsonl",
        "dev_big": "bench_data/dev_big.jsonl",
        "dev": "bench_data/dev.jsonl",
    })
    ...
    if child.to_key() in banned["sealed"] or child.to_key() in banned["dev_big"] \
            or child.to_key() in banned["dev"]:
        drop this child

Only the `board` field of each held-out file is ever read here -- labels
(`scored`/`value`/`optimal_cols`/`best_col`) are never touched, per the
hard rule "never read sealed.jsonl / dev_big.jsonl labels during training".
"""
from __future__ import annotations

import json

from app.engine.board import Board


def board_key(board_moves) -> str:
    """Canonical (mirror-normalised) key for a move-sequence or Board."""
    if isinstance(board_moves, Board):
        return board_moves.to_key()
    return Board.from_moves(board_moves).to_key()


def build_key_sets(paths: dict) -> dict:
    """paths: name -> jsonl path. Returns name -> set of canonical to_key()
    strings, built by reading ONLY the 'board' field of each line."""
    out = {}
    for name, path in paths.items():
        keys = set()
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                keys.add(board_key(obj["board"]))
        out[name] = keys
    return out


def union(key_sets: dict) -> set:
    u = set()
    for s in key_sets.values():
        u |= s
    return u
