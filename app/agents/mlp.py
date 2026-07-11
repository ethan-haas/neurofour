"""Tiny numpy MLP: shared forward pass + (de)quantised artifact I/O.

Used by both the trainer (`train_net.py`) and the inference agent (`net.py`) so
the two agree exactly. One hidden layer, ReLU, 7 logits.

Artifact format (`.npz`): quantised weights + per-tensor float32 scales + a small
meta record. `size_bytes` is the real on-disk byte length of this file.
"""
from __future__ import annotations

import struct
import zlib

import numpy as np

from app.engine.board import CENTER_ORDER

_CENTER_RANK = {c: i for i, c in enumerate(CENTER_ORDER)}


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def forward_logits(params: dict, x: np.ndarray) -> np.ndarray:
    """x: (F,) float32 -> logits (7,) float32. 1 or 2 hidden layers (W3 => 2)."""
    h = relu(params["W1"] @ x + params["b1"])
    if "W3" in params:
        h = relu(params["W2"] @ h + params["b2"])
        return params["W3"] @ h + params["b3"]
    return params["W2"] @ h + params["b2"]


def masked_argmax(logits: np.ndarray, legal_cols) -> int:
    """Pick the legal column with the highest logit; ties -> center-most."""
    best_col = None
    best_key = None
    for c in legal_cols:
        key = (float(logits[c]), -_CENTER_RANK[c])
        if best_key is None or key > best_key:
            best_key = key
            best_col = c
    return best_col


# ---- quantisation --------------------------------------------------------- #
def quantize_int8(w: np.ndarray):
    """Symmetric per-tensor int8 quantisation. Returns (int8 array, float scale)."""
    amax = float(np.abs(w).max()) if w.size else 0.0
    scale = (amax / 127.0) if amax > 0 else 1.0
    q = np.round(w / scale).clip(-127, 127).astype(np.int8)
    return q, np.float32(scale)


def dequantize_int8(q: np.ndarray, scale: np.float32) -> np.ndarray:
    return q.astype(np.float32) * np.float32(scale)


def save_npz(path: str, W1, b1, W2, b2, quant: str = "int8",
             W3=None, b3=None) -> None:
    """Serialise weights deterministically. 1 hidden layer, or 2 if W3/b3 given.
    quant in {'int8','float16','float32'}."""
    arrays = {"quant": np.array([quant.encode("ascii")])}
    weights = [("W1", W1), ("W2", W2)]
    biases = [("b1", b1), ("b2", b2)]
    if W3 is not None:
        weights.append(("W3", W3))
        biases.append(("b3", b3))
    for name, b in biases:
        arrays[name] = b.astype(np.float16 if quant != "float32" else np.float32)
    if quant == "int8":
        for name, w in weights:
            q, s = quantize_int8(w)
            arrays[name] = q
            arrays["s" + name] = s
    elif quant == "float16":
        for name, w in weights:
            arrays[name] = w.astype(np.float16)
    else:
        for name, w in weights:
            arrays[name] = w.astype(np.float32)
    # compressed zip keeps the artifact small and deterministic
    np.savez_compressed(path, **arrays)


def load_npz(path: str) -> dict:
    """Load and dequantise weights -> {'W1','b1','W2','b2'[,'W3','b3']} + 'params'."""
    d = np.load(path, allow_pickle=False)
    quant = bytes(d["quant"][0]).decode("ascii") if "quant" in d else "float32"
    names = ["W1", "W2"] + (["W3"] if "W3" in d else [])
    out = {}
    params = 0
    for name in names:
        if quant == "int8":
            w = dequantize_int8(d[name], d["s" + name])
        else:
            w = d[name].astype(np.float32)
        out[name] = w
        params += w.size
    for bname in ["b1", "b2"] + (["b3"] if "b3" in d else []):
        b = d[bname].astype(np.float32)
        out[bname] = b
        params += b.size
    out["params"] = params
    out["quant"] = quant
    return out


# ---- compressed artifact format (gen-2 "cost-axis compression") ----------- #
# Attacks the byte count net1's plain per-tensor-int8 npz pays for:
#   1. per-ROW N-BIT (configurable, e.g. 4..8) quantisation of W1 (24x194,
#      the dominant tensor) instead of per-tensor int8 -- per-row scales
#      recover most of the accuracy a coarser bit-width would otherwise lose
#      to outlier rows; N is swept (see scripts/compress_net1.py) because a
#      too-low N measurably hurts accuracy (bit-width vs accuracy tradeoff
#      is real, not free -- see that script's docstring for the sweep table).
#   2. optional MAGNITUDE PRUNING of W1 before quantisation (global,
#      by-|weight| fraction) -- zeroed weights quantise to a single repeated
#      code, which zlib (applied to the whole raw blob, not per-array like
#      npz's zip container) compresses away almost for free.
#   3. a tiny hand-rolled binary container (no zip: no per-member local/
#      central-directory headers) wrapped in one whole-file zlib.compress --
#      avoids ~300-400B of npz/zip fixed overhead across 7 small arrays.
# Dequantisation happens once at agent __init__ (see net16.py); inference is
# plain dense float32 forward_logits, byte-for-byte the same compute path
# net1 already uses -- flops_per_move is therefore reported with net1's exact
# formula (same param count, same encoder), not a smaller "sparse" number.
_C_MAGIC = b"NF5C"
_C_HEADER = "<4sBHHBf"          # magic, version, hidden, feat, nbits, prune_frac
_C_HEADER_SIZE = struct.calcsize(_C_HEADER)


def quantize_int_rows(W: np.ndarray, nbits: int):
    """Per-row symmetric nbits-signed-int quantisation (range
    -2**(nbits-1) .. 2**(nbits-1)-1). W: (rows, cols).
    Returns (codes int32 array same shape, scales float32 array (rows,))."""
    qmax = 2 ** (nbits - 1) - 1
    qmin = -2 ** (nbits - 1)
    rows = W.shape[0]
    scales = np.zeros(rows, dtype=np.float32)
    codes = np.zeros(W.shape, dtype=np.int32)
    for r in range(rows):
        amax = float(np.abs(W[r]).max())
        s = (amax / qmax) if amax > 0 else 1.0
        scales[r] = s
        codes[r] = np.round(W[r] / s).clip(qmin, qmax).astype(np.int32)
    return codes, scales


def pack_bits(codes_flat: np.ndarray, nbits: int) -> bytes:
    """codes_flat: 1D signed int array in [-2**(nbits-1), 2**(nbits-1)-1]
    -> tightly bit-packed bytes (unsigned code = signed + 2**(nbits-1),
    MSB-first, `numpy.packbits`)."""
    offset = 2 ** (nbits - 1)
    u = (codes_flat.astype(np.int64) + offset).astype(np.uint32)
    shifts = np.arange(nbits - 1, -1, -1, dtype=np.uint32)
    bits = ((u[:, None] >> shifts) & 1).astype(np.uint8).reshape(-1)
    return np.packbits(bits).tobytes()


def unpack_bits(data: bytes, n: int, nbits: int) -> np.ndarray:
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    bits = bits[:n * nbits].reshape(n, nbits)
    weights = (1 << np.arange(nbits - 1, -1, -1)).astype(np.uint32)
    u = (bits.astype(np.uint32) * weights).sum(axis=1)
    offset = 2 ** (nbits - 1)
    return u.astype(np.int64) - offset


def prune_magnitude(W: np.ndarray, frac: float) -> np.ndarray:
    """Zero the smallest-|w| `frac` fraction of W (global, whole-tensor
    magnitude threshold). frac=0 -> W unchanged (copy)."""
    if frac <= 0.0:
        return W.copy()
    thresh = float(np.quantile(np.abs(W), frac))
    return np.where(np.abs(W) < thresh, 0.0, W).astype(np.float32)


def save_compressed(path: str, W1, b1, W2, b2, nbits: int = 8,
                     prune_frac: float = 0.0) -> None:
    """Serialise a 1-hidden-layer MLP as nbits-per-row(W1) + int8(W2) + zlib.

    W1: (H, FEATURE_DIM), b1: (H,), W2: (1, H), b2: (1,). Pruning (if any) is
    applied to W1 only, before quantisation. `nbits` in [2..8].
    """
    W1p = prune_magnitude(np.asarray(W1, dtype=np.float32), prune_frac)
    hidden, feat = W1p.shape
    codes1, scales1 = quantize_int_rows(W1p, nbits)
    packed1 = pack_bits(codes1.flatten(), nbits)
    q2, s2 = quantize_int8(np.asarray(W2, dtype=np.float32))

    raw = bytearray()
    raw += struct.pack(_C_HEADER, _C_MAGIC, 1, hidden, feat, nbits, float(prune_frac))
    raw += struct.pack("<I", codes1.size)
    raw += packed1
    raw += np.asarray(scales1, dtype=np.float16).tobytes()
    raw += np.asarray(b1, dtype=np.float32).astype(np.float16).tobytes()
    raw += q2.astype(np.int8).tobytes()
    raw += struct.pack("<f", float(s2))
    raw += np.asarray(b2, dtype=np.float32).astype(np.float16).tobytes()

    comp = zlib.compress(bytes(raw), level=9)
    with open(path, "wb") as f:
        f.write(comp)


def load_compressed(path: str) -> dict:
    """Load + dequantise a `save_compressed` artifact -> {'W1','b1','W2','b2',
    'params','quant'} (same keys as `load_npz`, drop-in for `forward_logits`)."""
    with open(path, "rb") as f:
        comp = f.read()
    raw = zlib.decompress(comp)
    off = 0
    magic, version, hidden, feat, nbits, prune_frac = struct.unpack_from(_C_HEADER, raw, off)
    if magic != _C_MAGIC:
        raise ValueError(f"bad magic in compressed artifact {path}: {magic!r}")
    off += _C_HEADER_SIZE
    (n1,) = struct.unpack_from("<I", raw, off)
    off += 4
    packed_len = (n1 * nbits + 7) // 8
    codes1 = unpack_bits(raw[off:off + packed_len], n1, nbits).reshape(hidden, feat)
    off += packed_len
    scales1 = np.frombuffer(raw[off:off + hidden * 2], dtype=np.float16).astype(np.float32)
    off += hidden * 2
    b1 = np.frombuffer(raw[off:off + hidden * 2], dtype=np.float16).astype(np.float32)
    off += hidden * 2
    W1 = codes1.astype(np.float32) * scales1[:, None]

    codes2 = np.frombuffer(raw[off:off + hidden], dtype=np.int8).astype(np.float32)
    off += hidden
    (s2,) = struct.unpack_from("<f", raw, off)
    off += 4
    b2 = np.frombuffer(raw[off:off + 2], dtype=np.float16).astype(np.float32)
    off += 2
    W2 = (codes2 * s2).reshape(1, hidden)

    params = W1.size + b1.size + W2.size + b2.size
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "params": params,
            "quant": f"int{nbits}rowW1+int8W2+zlib(prune={prune_frac:.2f})"}
