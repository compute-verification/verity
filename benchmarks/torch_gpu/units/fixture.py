"""The units' check data, from the twins: pools of random and special rows and every word the kernels must write for them,
and the exhaustive sweeps' chunk digests (sweep.py).

CHECK/rmsnorm.npz   per width N in RMS_N (16384 the frozen one): x (P, N), w (2, N) two gains, out (2, P, N) for each gain,
                    chunks (P, nc), groups (P, ng), rs (P,), kind (P,)
CHECK/quant.npz     per (N, RELU) in QUANT_CASES ((16384, false) and (65536, true) the frozen ones): x, codes, scale, tree
CHECK/softmax.npz   x (P, 32) and, at every length n = 1..32, p (P, 32, 32) [row, n-1, j], m (P, 32) and r (P, 32)
CHECK/argmax.npz    per V in ARGMAX_V (256 the frozen one): x (P, V), token, best, idx
CHECK/relu.npz      every bf16 word and its ReLU
CHECK/add.npz       a, b (P, 16384) and y, the residual add, kind (P,)
CHECK/sweep.json    every op's chunk digests (sweep.OPS)
CHECK/fixture.json  the seeds, kinds and counts, the cross-checks and each file's sha256
Each pool's rows come from KINDS of generators (normal values, wide exponents, subnormals, overflow, ties, signed zeros,
NaN and infinities, uniform random words, ...), each row from its own seed.

Two cross-checks, each of whose disagreements exits 1 before anything is written:
- with --replay PATH (the H200 demo's CPU replay, examples/h200/replay.py), every pool word is also computed by
  that file's references of circuits' templates (RmsSumSqPart_v1 ... ArgmaxCombine_v1, unit by unit as the replay reads
  them);
- with `verity.ml.circuits.ops.rows` installed, or its file given as --rows PATH, every argmax pool's tokens by
  `rows.argmax` (Argmax_v1{V}) and the add pool by `rows.add_row` (Bf16AddRow_v1{N}).

  uv run python benchmarks/torch_gpu/units/fixture.py CHECK [--replay PATH] [--rows PATH] [--no-sweep]   (env PROCS,
  default 8)
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import add as AD  # noqa: E402
import argmax as AM  # noqa: E402
import quant as Q  # noqa: E402
import relu as RL  # noqa: E402
import rmsnorm as RN  # noqa: E402
import softmax as SM  # noqa: E402
import sweep as SW  # noqa: E402
import words as W  # noqa: E402
from verity_config.settings import of  # noqa: E402

settings = of(__file__)
SEED = 20261026
SPECIALS = np.array([0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC0, 0xFFC0, 0x7F81, 0xFFFF, 0x0001, 0x8001, 0x007F, 0x807F,
                     0x0080, 0x7F7F, 0xFF7F, 0x3F80, 0xBF80], np.uint16)
FINITE_SPECIALS = np.array([0x0000, 0x8000, 0x7F80, 0xFF80, 0x0001, 0x8001, 0x007F, 0x807F, 0x0080, 0x7F7F, 0xFF7F, 0x3F80,
                            0xBF80], np.uint16)
RMS_N = (16384, 4096, 256, 64)
RMS_KINDS = ("normal", "scaled", "order", "tiny", "huge", "zero", "inf", "specials", "uniform")
QUANT_CASES = ((16384, False), (65536, True), (16384, True), (256, False), (256, True), (64, True))
QUANT_KINDS = ("normal", "scaled", "tiny", "zero", "negative", "ties", "inf", "nan", "uniform")
SM_KINDS = ("scores", "wide", "ftz", "equal", "zeros", "neg_inf", "big", "specials", "uniform")
ARGMAX_V = (256, 200, 100)
AM_KINDS = ("logits", "ties", "zeros", "nan0", "nans", "neg_inf", "inf", "chunk_tie", "uniform")
ADD_N = 16384
ADD_KINDS = ("normal", "cancel", "scaled", "tiny", "huge", "specials", "uniform")


def bf16(v) -> np.ndarray:
    """Finite float32 values rounded to bf16 words (nearest, ties to even; overflow to infinity)."""
    u = np.asarray(v, np.float32).view(np.uint32).astype(np.uint64)
    return (((u + 0x7FFF + ((u >> 16) & 1)) >> 16) & 0xFFFF).astype(np.uint16)


def sprinkle(rng, a: np.ndarray, p: float, pool: np.ndarray = SPECIALS) -> np.ndarray:
    hit = rng.random(a.shape) < p
    a = a.copy()
    a[hit] = rng.choice(pool, int(hit.sum()))
    return a


def words_with(rng, shape, exps) -> np.ndarray:
    """Words of random sign and mantissa whose exponent fields are drawn from ``exps``."""
    e = rng.choice(np.asarray(exps), size=shape)
    return ((rng.integers(0, 2, size=shape) << 15) | (e << 7) | rng.integers(0, 128, size=shape)).astype(np.uint16)


def uniform(rng, shape) -> np.ndarray:
    return rng.integers(0, 1 << 16, size=shape).astype(np.uint16)


# ---- pools ----------------------------------------------------------------------------------------------------------------

def rms_row(N: int, kind: str, rng) -> np.ndarray:
    normal = rng.standard_normal(N).astype(np.float32)
    if kind == "normal":
        return bf16(normal)
    if kind == "scaled":
        return bf16(normal * np.float32(2.0 ** int(rng.integers(-70, 67))))
    if kind == "order":
        big = rng.random(N) < 1 / 16
        return bf16(np.where(big, normal * np.float32(4096), normal))
    if kind == "tiny":
        return words_with(rng, N, np.arange(0, 21))
    if kind == "huge":
        return words_with(rng, N, np.arange(235, 255))
    if kind == "zero":
        return (rng.integers(0, 2, N) << 15).astype(np.uint16) if rng.random() < 0.75 else np.zeros(N, np.uint16)
    if kind == "inf":
        return sprinkle(rng, bf16(normal), 0.01, FINITE_SPECIALS)
    if kind == "specials":
        return sprinkle(rng, bf16(normal), 0.002)
    return uniform(rng, N)


def rms_pool(N: int) -> dict:
    per = 8 if N == 16384 else 4
    rows, kinds = [], []
    for i in range(per * len(RMS_KINDS)):
        kind = RMS_KINDS[i % len(RMS_KINDS)]
        rows.append(rms_row(N, kind, np.random.default_rng([SEED, 1, N, i])))
        kinds.append(kind)
    rng = np.random.default_rng([SEED, 2, N])
    w = np.stack([bf16(1 + 0.1 * rng.standard_normal(N).astype(np.float32)),
                  sprinkle(rng, bf16(rng.standard_normal(N).astype(np.float32) * 3), 0.02)])
    x = np.stack(rows)
    t = [RN.twin(x, w[k]) for k in range(2)]
    return {"x": x, "w": w, "out": np.stack([d["out"] for d in t]), "chunks": t[0]["chunks"], "groups": t[0]["groups"],
            "rs": t[0]["rs"], "kind": np.array(kinds)}


def quant_row(N: int, kind: str, rng) -> np.ndarray:
    normal = rng.standard_normal(N).astype(np.float32)
    if kind == "normal":
        return bf16(normal)
    if kind == "scaled":
        return bf16(normal * np.float32(2.0 ** int(rng.integers(-120, 120))))
    if kind == "tiny":
        return words_with(rng, N, np.arange(0, 3))
    if kind == "zero":
        return (rng.integers(0, 2, N) << 15).astype(np.uint16)
    if kind == "negative":
        return bf16(-np.abs(normal)) if rng.random() < 0.5 else bf16(normal - 2)
    if kind == "ties":
        # amax 127 2^j, so scale = 2^j exactly, and every other word (k + 1/2) 2^j: every code a tie
        j = int(rng.integers(-20, 21))
        k = rng.integers(-127, 127, N).astype(np.float32) + np.float32(0.5)
        x = bf16(k * np.float32(2.0 ** j))
        x[rng.integers(0, N)] = bf16(np.float32(127 * 2.0 ** j) * (1 if rng.random() < 0.5 else -1))
        return x
    if kind == "inf":
        return sprinkle(rng, bf16(normal), 0.01, FINITE_SPECIALS)
    if kind == "nan":
        x = bf16(normal * np.float32(rng.choice([1, 64, 300])))
        x = np.where(rng.random(N) < 0.1, bf16(rng.integers(-200, 200, N).astype(np.float32) + np.float32(0.5)), x)
        x[rng.integers(0, N)] = rng.choice(np.array([0x7FC0, 0xFFC1, 0x7F81, 0xFFFF], np.uint16))
        return x.astype(np.uint16)
    return uniform(rng, N)


def quant_pool(N: int, relu: bool) -> dict:
    per = 4 if N >= 16384 else 6
    rows, kinds = [], []
    for i in range(per * len(QUANT_KINDS)):
        kind = QUANT_KINDS[i % len(QUANT_KINDS)]
        rows.append(quant_row(N, kind, np.random.default_rng([SEED, 3, N, int(relu), i])))
        kinds.append(kind)
    x = np.stack(rows)
    return {"x": x, **Q.twin(x, relu), "kind": np.array(kinds)}


def sm_row(kind: str, rng) -> np.ndarray:
    L = SM.LMAX
    normal = rng.standard_normal(L).astype(np.float32)
    if kind == "scores":
        return bf16(normal * np.float32(rng.choice([0.5, 2, 8, 30])))
    if kind == "wide":
        return bf16(normal * np.float32(2.0) ** rng.integers(-8, 9, L).astype(np.float32))
    if kind == "ftz":
        # e's operand RN(x - m) log2 e across [-166, -87]: through ex2's subnormal results, which MUFU.EX2 flushes
        top = np.float32(rng.uniform(-4, 4))
        x = bf16(top - rng.uniform(60, 115, L).astype(np.float32))
        x[rng.integers(0, L)] = bf16(top)
        return x
    if kind == "equal":
        return np.full(L, bf16(normal[:1])[0], np.uint16)
    if kind == "zeros":
        z = (rng.integers(0, 2, L) << 15).astype(np.uint16)
        return np.where(rng.random(L) < 0.3, bf16(-np.abs(normal)), z).astype(np.uint16)
    if kind == "neg_inf":
        x = np.where(rng.random(L) < 0.7, np.uint16(0xFF80), bf16(normal))
        if rng.random() < 0.2:
            x[:] = 0xFF80
        return x.astype(np.uint16)
    if kind == "big":
        return words_with(rng, L, np.arange(248, 255))
    if kind == "specials":
        return sprinkle(rng, bf16(normal * 4), 0.08 if rng.random() < 0.5 else 0.02)
    return uniform(rng, L)


def sm_pool() -> dict:
    per = 128
    rows, kinds = [], []
    for i in range(per * len(SM_KINDS)):
        kind = SM_KINDS[i % len(SM_KINDS)]
        rows.append(sm_row(kind, np.random.default_rng([SEED, 4, i])))
        kinds.append(kind)
    x = np.stack(rows)
    P, L = x.shape
    t = SM.twin(np.repeat(x, L, axis=0), np.tile(np.arange(1, L + 1), P))
    return {"x": x, "p": t["p"].reshape(P, L, L), "m": t["m"].reshape(P, L), "r": t["r"].reshape(P, L),
            "kind": np.array(kinds)}


def am_row(V: int, kind: str, rng) -> np.ndarray:
    normal = rng.standard_normal(V).astype(np.float32)
    if kind == "logits":
        return bf16(normal * 3)
    if kind == "ties":
        return bf16(rng.integers(-2, 3, V).astype(np.float32))
    if kind == "zeros":
        z = (rng.integers(0, 2, V) << 15).astype(np.uint16)
        return np.where(rng.random(V) < 0.5, bf16(-np.abs(normal) - 1), z).astype(np.uint16)
    if kind == "nan0":
        x = bf16(normal)
        x[0] = rng.choice(np.array([0x7FC0, 0xFFC0, 0x7F81], np.uint16))
        return x
    if kind == "nans":
        return np.where(rng.random(V) < 0.2, rng.choice(np.array([0x7FC0, 0xFFC0, 0x7FFF, 0xFF81], np.uint16), V),
                        bf16(normal)).astype(np.uint16)
    if kind == "neg_inf":
        x = np.full(V, 0xFF80, np.uint16)
        r = rng.random()
        if r < 0.3:
            x[rng.integers(0, V)] = bf16(normal[:1])[0]
        elif r < 0.6:
            x[rng.random(V) < 0.1] = 0x7FC0
        return x
    if kind == "inf":
        x = bf16(normal)
        x[rng.integers(0, V, 3)] = 0x7F80
        return x
    if kind == "chunk_tie":
        x = bf16(normal)
        top = int(bf16(np.float32(np.abs(normal).max() + 1)))
        x[rng.integers(0, min(V, 128))] = top
        x[rng.integers(min(V, 128) - 1, V)] = top
        if rng.random() < 0.3:
            x[0] = top
        return x
    return uniform(rng, V)


def am_pool(V: int) -> dict:
    per = 256 if V == 256 else 32
    rows, kinds = [], []
    for i in range(per * len(AM_KINDS)):
        kind = AM_KINDS[i % len(AM_KINDS)]
        rows.append(am_row(V, kind, np.random.default_rng([SEED, 5, V, i])))
        kinds.append(kind)
    x = np.stack(rows)
    return {"x": x, **AM.twin(x), "kind": np.array(kinds)}


def relu_pool() -> dict:
    x = np.arange(1 << 16, dtype=np.uint32).astype(np.uint16)
    return {"x": x, "y": RL.twin(x)}


def add_rows(N: int, kind: str, rng) -> tuple[np.ndarray, np.ndarray]:
    a, b = rng.standard_normal(N).astype(np.float32), rng.standard_normal(N).astype(np.float32)
    if kind == "normal":
        return bf16(a), bf16(b * np.float32(rng.choice([1 / 64, 1, 64])))
    if kind == "cancel":
        # b = -a give or take a few ulps: exact zeros (+0 at RN) and differences far below either word
        x = bf16(a)
        y = (x ^ 0x8000).astype(np.int64) + np.where(rng.random(N) < 0.5, rng.integers(-2, 3, N), 0)
        return x, (y & 0xFFFF).astype(np.uint16)
    if kind == "scaled":
        return (bf16(a * np.float32(2.0) ** rng.integers(-126, 128, N).astype(np.float32)),
                bf16(b * np.float32(2.0) ** rng.integers(-126, 128, N).astype(np.float32)))
    if kind == "tiny":
        return words_with(rng, N, np.arange(0, 3)), words_with(rng, N, np.arange(0, 3))
    if kind == "huge":
        return words_with(rng, N, np.arange(250, 255)), words_with(rng, N, np.arange(250, 255))
    if kind == "specials":
        return sprinkle(rng, bf16(a), 0.05), sprinkle(rng, bf16(b), 0.05)
    return uniform(rng, N), uniform(rng, N)


def add_pool(N: int = ADD_N) -> dict:
    per = 8
    rows, kinds = [], []
    for i in range(per * len(ADD_KINDS)):
        kind = ADD_KINDS[i % len(ADD_KINDS)]
        rows.append(add_rows(N, kind, np.random.default_rng([SEED, 6, N, i])))
        kinds.append(kind)
    a, b = np.stack([r[0] for r in rows]), np.stack([r[1] for r in rows])
    return {"a": a, "b": b, "y": AD.twin(a, b), "kind": np.array(kinds)}


# ---- the replay cross-check -----------------------------------------------------------------------------------------------

def load_replay(path: str):
    spec = importlib.util.spec_from_file_location("units_replay", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["units_replay"] = mod
    spec.loader.exec_module(mod)
    return mod


def _cmp(ref, got) -> list[int]:
    ref, got = np.asarray(ref).reshape(-1).astype(np.int64), np.asarray(got).reshape(-1).astype(np.int64)
    if ref.shape != got.shape:
        return [int(got.size), int(got.size)]
    return [int(got.size), int((ref != got).sum())]


def _joined(R, half_of, *arrays) -> np.ndarray:
    """The 32-bit words whose two 16-bit halves the units ``half_of(0)`` and ``half_of(1)`` output."""
    lo, hi = (half_of(h).reference(*arrays)[:, 0] for h in (0, 1))
    return R.join16(lo, hi)


def replay_rms(R, d: dict) -> dict:
    x = d["x"]
    P, N = x.shape
    C, nc, G, ng = RN.forms(N)
    chunks = _joined(R, lambda h: R.rms_sum_sq_part(C, h), x.reshape(P * nc, C))
    groups = _joined(R, lambda h: R.f32_sum_part(G, h), d["chunks"].reshape(P * ng, G))
    rs = _joined(R, lambda h: R.rms_norm_recip(N, ng, W.EPS, h), d["groups"])
    out = [R.rms_norm_out().reference(x.reshape(-1), np.repeat(d["rs"], N), np.tile(d["w"][k], P)) for k in range(2)]
    return {"chunks": _cmp(chunks, d["chunks"]), "groups": _cmp(groups, d["groups"]), "rs": _cmp(rs, d["rs"]),
            "out": _cmp(np.stack(out), d["out"])}


def replay_quant(R, d: dict, relu: bool) -> dict:
    x = d["x"]
    P, N = x.shape
    tree, level, first = [], x, True
    for width in Q.levels(N):
        a = level.shape[1] // width
        level = R.row_abs_max_t(a, relu and first).reference(level.reshape(P * width, a)).reshape(P, width)
        tree.append(level)
        first = False
    amax = level[:, 0]
    scale = _joined(R, lambda h: R.quant_s8_scale(h), amax)
    codes = R.quant_s8_pair(relu).reference(x.reshape(P * N // 2, 2), np.repeat(amax, N // 2)).reshape(P, N)
    return {"tree": _cmp(np.concatenate(tree, 1), d["tree"]), "scale": _cmp(scale, d["scale"]), "codes": _cmp(codes, d["codes"])}


def replay_softmax(R, d: dict) -> dict:
    x = d["x"]
    P, L = x.shape
    res = {"m": [0, 0], "r": [0, 0], "p": [0, 0]}
    for n in range(1, L + 1):
        xs = x[:, :n]
        m = R.softmax_max_t(n).reference(xs)[:, 0]
        r = _joined(R, lambda h: R.softmax_recip_t(n, h), xs, d["m"][:, n - 1])
        p = R.softmax_out_t().reference(xs.reshape(-1), np.repeat(d["m"][:, n - 1], n), np.repeat(d["r"][:, n - 1], n))
        for k, (ref, got) in {"m": (m, d["m"][:, n - 1]), "r": (r, d["r"][:, n - 1]), "p": (p, d["p"][:, n - 1, :n])}.items():
            c = _cmp(ref, got)
            res[k] = [res[k][0] + c[0], res[k][1] + c[1]]
    return res


def replay_argmax(R, d: dict) -> dict:
    x = d["x"]
    P, V = x.shape
    C = min(AM.CHUNK, V)
    nch = -(-V // C)
    best = np.stack([R.argmax_chunk_best(x[:, c * C:(c + 1) * C].shape[1]).reference(x[:, c * C:(c + 1) * C])[:, 0]
                     for c in range(nch)], 1)
    idx = np.stack([R.argmax_chunk_idx(x[:, c * C:(c + 1) * C].shape[1]).reference(x[:, c * C:(c + 1) * C])[:, 0]
                    for c in range(nch)], 1)
    token = R.argmax_combine_t(V, C).reference(x[:, 0], d["best"], d["idx"])[:, 0]
    return {"best": _cmp(best, d["best"]), "idx": _cmp(idx, d["idx"]), "token": _cmp(token, d["token"])}


ROWS = "verity.ml.circuits.ops.rows"


def load_rows(path: str | None):
    """`ops.rows` and where it came from: the installed module, else the file at ``path`` loaded in its place in
    `verity.ml.circuits.ops` (its relative imports read main's fp32, prims and scalar), else (None, None)."""
    try:
        return importlib.import_module(ROWS), "installed"
    except ModuleNotFoundError as e:
        if e.name != ROWS:
            raise
    if not path:
        return None, None
    importlib.import_module("verity.ml.circuits.ops")
    spec = importlib.util.spec_from_file_location(ROWS, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[ROWS] = mod
    spec.loader.exec_module(mod)
    return mod, path


def rows_check(Rw, am: dict, ad: dict) -> dict:
    res = {f"argmax {V}": {"token": _cmp([Rw.argmax(row.tolist()) for row in d["x"]], d["token"])} for V, d in am.items()}
    res["add"] = {"y": _cmp(np.stack([Rw.add_row(a.tolist(), b.tolist()) for a, b in zip(ad["a"], ad["b"])]), ad["y"])}
    return res


# ---- main -----------------------------------------------------------------------------------------------------------------

def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _digest(task: tuple[int, int]) -> tuple[int, int, int]:
    op, c = task
    return op, c, SW.digest(op, c)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", nargs="?", default="/tmp/units-check")
    ap.add_argument("--replay", default=settings.UNITS_REPLAY)
    ap.add_argument("--rows", default=settings.UNITS_ROWS)
    ap.add_argument("--no-sweep", action="store_true")
    a = ap.parse_args(argv[1:])
    out, replay_path = a.out, a.replay
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    rms = {N: rms_pool(N) for N in RMS_N}
    qnt = {(N, relu): quant_pool(N, relu) for N, relu in QUANT_CASES}
    sm = sm_pool()
    am = {V: am_pool(V) for V in ARGMAX_V}
    rl = relu_pool()
    ad = add_pool()
    doc: dict = {"seed": SEED, "twins": "words.py's instructions, the modules' OPS", "pools": {
        "rmsnorm": {str(N): {"rows": len(d["x"]), "kinds": list(RMS_KINDS)} for N, d in rms.items()},
        "quant": {f"{N},{str(r).lower()}": {"rows": len(d["x"]), "kinds": list(QUANT_KINDS)} for (N, r), d in qnt.items()},
        "softmax": {"rows": len(sm["x"]), "L": SM.LMAX, "kinds": list(SM_KINDS)},
        "argmax": {str(V): {"rows": len(d["x"]), "kinds": list(AM_KINDS)} for V, d in am.items()},
        "relu": {"words": len(rl["x"])}, "add": {str(ADD_N): {"rows": len(ad["a"]), "kinds": list(ADD_KINDS)}}},
        "pool_seconds": round(time.time() - t0, 1)}
    bad = []
    Rw, rows_from = load_rows(a.rows)
    if Rw is not None:
        rc = {"module": rows_from, "sha256": sha256(Rw.__file__), **rows_check(Rw, am, ad)}
        doc["rows"] = rc
        bad += [f"rows {u} {k}" for u, part in rc.items() if isinstance(part, dict) for k, v in part.items() if v[1]]
        print(json.dumps(rc), file=sys.stderr)
    if replay_path:
        R = load_replay(replay_path)
        rep = {"file": replay_path, "sha256": sha256(replay_path)}
        rep |= {f"rmsnorm {N}": replay_rms(R, d) for N, d in rms.items()}
        rep |= {f"quant {N},{str(r).lower()}": replay_quant(R, d, r) for (N, r), d in qnt.items()}
        rep["softmax"] = replay_softmax(R, sm)
        rep |= {f"argmax {V}": replay_argmax(R, d) for V, d in am.items()}
        rep["relu"] = {"y": _cmp(R.relu16(rl["x"], True), rl["y"])}
        doc["replay"] = rep
        bad += [f"{u} {k}" for u, part in rep.items() if isinstance(part, dict) for k, v in part.items() if v[1]]
        print(json.dumps(rep), file=sys.stderr)
    if bad:
        print("REFUSED: the twins and circuits' references disagree: " + ", ".join(bad), file=sys.stderr)
        return 1
    np.savez(os.path.join(out, "rmsnorm.npz"), **{f"{N}.{k}": v for N, d in rms.items() for k, v in d.items()})
    np.savez(os.path.join(out, "quant.npz"), **{f"{N},{int(r)}.{k}": v for (N, r), d in qnt.items() for k, v in d.items()})
    np.savez(os.path.join(out, "softmax.npz"), **sm)
    np.savez(os.path.join(out, "argmax.npz"), **{f"{V}.{k}": v for V, d in am.items() for k, v in d.items()})
    np.savez(os.path.join(out, "relu.npz"), **rl)
    np.savez(os.path.join(out, "add.npz"), **ad)
    files = ["rmsnorm.npz", "quant.npz", "softmax.npz", "argmax.npz", "relu.npz", "add.npz"]
    if not a.no_sweep:
        t1 = time.time()
        tasks = [(op, c) for op in SW.OPS for c in range(SW.chunks(op))]
        with mp.get_context("spawn").Pool(settings.PROCS) as pool:
            res = pool.map(_digest, tasks, chunksize=1)
        sw = {str(op): {"name": SW.OPS[op][0], "n": SW.OPS[op][1], "arg": SW.OPS[op][2],
                        "chunks": [f"{d:016x}" for o, c, d in sorted(res) if o == op]} for op in SW.OPS}
        with open(os.path.join(out, "sweep.json"), "w") as f:
            json.dump(sw, f, indent=0)
        files.append("sweep.json")
        doc["sweep_seconds"] = round(time.time() - t1, 1)
    doc["files"] = {f: sha256(os.path.join(out, f)) for f in files}
    with open(os.path.join(out, "fixture.json"), "w") as f:
        json.dump(doc, f, indent=1)
    print(f"fixture: pools {doc['pool_seconds']} s, sweep {doc.get('sweep_seconds', 'skipped')} s, replay "
          f"{'agrees' if replay_path else 'not run'}, ops.rows {'agrees' if Rw is not None else 'not run'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
