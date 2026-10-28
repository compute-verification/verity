"""HIGHUTIL: an 8B model's linear layers at full batch on one GPU, each call credited through Dither's no-fold receipt
(v31.py's ``Fill``, under verity/core/protocols/pouw/dither/PROTOCOL.md v3.2's commit phase) on accel_v31's prover,
with live coins: the declared utilization the periods establish (the credited MACs a second over the tensor pipe's MAC
rate at the card's ceiling, their MFU) and the prover overhead γ against plain GEMMs on the same linears.

**The workload.** Llama-3.1-8B's linear layers (QKV 6144 x 4096, O 4096 x 4096, gate-and-up 28672 x 4096, down 4096 x
14336) at M = 16,384, s8 x s8 with int32 accumulation, called back to back, down as K-slices (`calls`, `lp_of`: Dither's
argument covers a segment of at most T_MAX L', and v31 takes L' <= 1,024). A step is one transformer block's calls,
and step g runs block g mod NB, so NB = 32 consecutive steps are the prefill's every linear. B is the published
checkpoint's weights (verity-int8/Llama-3.1-8B-INT8 at its pinned revision, q k v and gate up fused as served; `prep
--checkpoint`, the shards' SHA-256 in the registration). A is the s8 input each linear gets in a W8A8 forward of that
checkpoint over 16 x 1,024 tokens of wikitext-2 (`capture.py`), so each call computes what the model computes on that
batch, and the periods replay that one prefill. Both operands are committed before any coin: a frame-v3-sha512 tree a
(operand, layer, block), its rows hm96 rows (`OperandTree`), whose roots the registration carries. `prep --seeded`
seeds both instead (v31.py's ``llm`` rows for B), for tests and shapes the checkpoint doesn't have.

**A period** (`Period.proto`): the registration (the operand roots, the operands' source, the period's shape, W_reg) is
the root of one challenger session with families ``seeds`` and ``check``, a coin each a block. Then its sampling blocks run
in one sequence. Block b's operand-root receipt is H(registration, b, block b-2's value-root receipt; the registration's
digest for b < 2), and its seeds coin opens after it, so blocks 0 and 1's open with the session, before the wall starts. Call (s, layer)'s coin is ``derive(coin, "verity/pouw-accel/seeds/v1", ctx).stream(call)``, and its six
seeds (pad dA dB U V P) are dither.call_seeds of that coin under the block's window salt, its operand-root receipt. Each
call commits its words T_s = X W'^T mod 2^16 every L inner indices (no O, wraps or binding; `prep --served`: the served
form, T_1 .. T_{S-1} and O = (A + d_A)(B + d_B)^T mod 2^32, its seeds still the filler's streams) under a frame-v3-sha512 root,
its leaf pos salted prf_salt(Ks, off + pos) under the Prover's own Ks, off = c leaves for its c-th call (dither.salts);
block b's value-root receipt is H(registration, b, its
roots). Its check coin opens after that receipt, with block b+2's seeds coin in the same round trip, while block b+1 runs:
the host enqueues each block behind the one before and never waits for the device between them. The coin draws
Q W_b / W_reg proof blocks of b (`picks`): a call by its MACs, a cell uniformly, a segment by its length, so each MAC is
equally likely. Each pick's tree paths come off the host copy of its call's tree before block b+2 overwrites it.

**The verifier** (`verify_pick`), after the period: the prover's answer (`answer`: the opened rows recomputed from the
operands and coins, their salts and paths, and A's and B's rows with their operand openings) against v31.py's
``fill_check``, on the verifier's own seeds from the coins, the opened operand rows and the call's root. Credit is
all-or-nothing: the period's MACs if every pick passes, else none. ε_skip = ln(1/δ)/(ηQ) at δ = 1%.

**Deviations from v3.2**, each named in the report: the picks are a fixed Q W_b / W_reg a block, not ``poisson:Q``;
the coins are 64-byte session coins expanded here, not the challenger's streamed seeds; no certifier signs the receipts
(the challenger's release log records what each coin opened after); and the checks run in the clear, the stand-in for
the Definition and C-Flock --zk.

    period.py prep CFG (--checkpoint DIR | --seeded) [--row 8192] [--lp 512] [--lp-max 1024] [--L 16384] [--nb 32] [--served]   the weights and shapes
    capture.py CFG --checkpoint DIR                                           the activations (on the GPU)
    period.py diag CFG LIB OUT                                                one step's GPU openings against the answers'
    period.py run CFG LIB|ref OUT [--coins local|gate:USER@HOST ...] [--blocks --steps --q --periods --negative]
    period.py summary OUT [--card SPEC] [--to FILE]                           summary.json again from a run's stored periods
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
TREE = HERE.parents[2]
sys.path.insert(0, str(HERE))

import settings  # noqa: E402

settings.one_blas_thread()

import numpy as np  # noqa: E402

sys.path.insert(0, str(TREE / "benchmarks" / "one_stage"))
sys.path.insert(0, str(TREE / "benchmarks" / "pouw" / "accel_v31"))

import v31  # noqa: E402

from verity.ml.commitments.identity import identity_digest_sha512  # noqa: E402
from verity.ml.commitments.indexed import RangeIndexedDomain  # noqa: E402
from verity.ml.commitments.limits import VerificationLimits  # noqa: E402
from verity.ml.commitments.merkle import Commitment, CommitmentDomain, Opening, tree_levels, verify_opening  # noqa: E402
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2, row_leaf  # noqa: E402
from verity.ml.randomness import derive  # noqa: E402

#: the model's linears; a config's LAYERS are its calls (`calls`), the linears with down split along K
LINEARS = (("qkv", 6144, 4096), ("o", 4096, 4096), ("up", 28672, 4096), ("down", 4096, 14336))
LAYERS = LINEARS
M = 16384
NB = 32       # transformer blocks: step g runs block g mod NB
LP, L = 512, 16384   # noise tiles of L' (a call's least; `lp_of`) and the word interval L: S = ceil(K / L) = 1 at every call's K
#: Dither's argument covers segments of at most T_MAX L' inner indices (a proof block's cost goes as c_t / t; v3.2 proves it
#: for t <= 8), and v31's Fill takes L' <= LP_MAX, so a call takes a longer L' (`lp_of`), and a linear no L' covers runs as
#: K-slices (`calls`)
T_MAX = 8
LP_MAX = 1024
#: each call in its served form (`prep --served`, v31.py's Fill(o=True)): T_1 .. T_{S-1} and O = (A + d_A)(B + d_B)^T mod
#: 2^32 committed, no pad; else the filler's T_1 .. T_S
SERVED = False
TILE = v31.TILE
SCHEME = "verity/highutil/period/v1"
OPERANDS_TAG = "verity/highutil/operands/v1"
CHECKPOINT = {"repo": "verity-int8/Llama-3.1-8B-INT8", "revision": "723d342ac443b040931242c768df342b0f6f15a7"}
#: hu's layers from the checkpoint's modules, fused as served: (name, the modules whose rows it stacks)
FUSED = (("qkv", ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")), ("o", ("self_attn.o_proj",)),
         ("up", ("mlp.gate_proj", "mlp.up_proj")), ("down", ("mlp.down_proj",)))
ETA = 1 - 2.8e-5
DELTA = 0.01
#: the ANDs of a MAC under main's Boolean lowering of an s8 x s8 linear into int32: GemmAccumulatorS8_v2{K, DOT} on the
#: card's IMMA step (BlackwellS8ImmaDot32_v2; sm_90's step costs the same) is 8,316 ANDs at K = 64 and linear in K. It is
#: not creditSize's lowering pinned at registration: Dither's Fill has no Definition yet.
ANDS_PER_MAC = (2079, 16)


def lowering(card: dict | None) -> str:
    """The s8 lowering credited_ands counts, on the card's own step (sm_120's when no card is given)."""
    sm90 = bool(card) and str(card.get("arch", "")).startswith("sm_90")
    return (f"GemmAccumulatorS8_v2{{K, **{'GEMM_SM90_S8' if sm90 else 'GEMM_SM120_S8'}}} "
            "(verity.ml.circuits.ops.boolean.gemm)")


def load_card(spec: str) -> dict:
    """The card's numbers (verity/hardware's card file, one TOML a SKU): a path to the file, or a SKU that
    `verity.hardware.load` reads."""
    if Path(spec).is_file():
        import tomllib
        return tomllib.loads(Path(spec).read_text(encoding="utf-8"))
    try:
        from verity.hardware import load
    except ImportError:
        raise SystemExit(f"card {spec!r}: no such file, and verity.hardware isn't in this tree; pass --card with the "
                         "card file (verity/hardware/cards/<sku>.toml)") from None
    return load(spec)


def theta(card: dict, mhz: float | None = None) -> float:
    """θ, the card's dense s8 tensor peak in MACs a second: rate.tensor x sm.count x the SM clock (its ceiling, or
    `mhz`)."""
    return card["rate"]["tensor"]["value"] * card["sm"]["count"]["value"] * (mhz or card["clock"]["sm"]["value"]) * 1e6


def H(tag: str, *parts: bytes) -> bytes:
    h = hashlib.sha512(tag.encode() + b"\0")
    for p in parts:
        h.update(len(p).to_bytes(8, "big") + p)
    return h.digest()


def u64(x: int) -> bytes:
    return x.to_bytes(8, "big")


def shape(n: int, k: int, row: int, m: int | None = None) -> v31.Fill:
    lp = lp_of(k)
    if lp is None:
        raise SystemExit(f"K {k}: no L' from {LP} to {LP_MAX} divides it and L {L} with its segment at most {T_MAX} L' "
                         "(split it: `calls`)")
    return v31.Fill(M if m is None else m, n, k, lp, L, row, SERVED)


def lp_of(k: int) -> int | None:
    """A call's L' at its K: the least multiple of 128 from LP to LP_MAX that divides K and L with the segment min(L, K) at
    most T_MAX L' (None: none does)."""
    return next((lp for lp in range(LP, LP_MAX + 1, TILE) if k % lp == 0 and L % lp == 0 and min(L, k) <= T_MAX * lp), None)


def calls(linears) -> list[tuple[str, int, int, str, int]]:
    """Each linear's calls (name, n, k, linear, k0): the linear whole, or its fewest equal K-slices, each with an L'
    (`lp_of`). A slice's X W'^T is the linear's over columns k0..k0 + k, and the linear's output is its slices' sum."""
    out = []
    for name, n, k in linears:
        d = next((d for d in range(1, k // TILE + 1) if k % d == 0 and lp_of(k // d)), None)
        if d is None:
            raise SystemExit(f"{name}: no K-slice of {k} has an L' from {LP} to {LP_MAX} dividing it and L {L} with its "
                             f"segment at most {T_MAX} L'")
        out += [(name, n, k, name, 0)] if d == 1 else [(f"{name}.{s}", n, k // d, name, s * (k // d)) for s in range(d)]
    return out


def step_macs() -> int:
    return sum(M * n * k for _, n, k in LAYERS)


def eps_skip(q: int, delta: float = DELTA) -> float:
    return math.log(1 / delta) / (ETA * q)


# --------------------------------------------------------------------------------------------------------- operands
def seeded_weights(n: int, k: int, seed: int) -> np.ndarray:
    """v31.py's ``llm`` weights: Gaussian rows quantized to +-127 by their absolute maximum."""
    b = np.random.default_rng(seed).standard_normal(size=(n, k), dtype=np.float32)
    return np.clip(np.rint(127 * b / np.abs(b).max(axis=1, keepdims=True)), -127, 127).astype(np.int8)


def seeded_activations(m: int, k: int, seed: int) -> np.ndarray:
    """A sum of four bytes, scaled (sd ~9, |a| <= 32), one entry in 1024 a full-range outlier: an absmax-quantized
    activation's look, so A's rows rarely wrap at V +-80."""
    g = np.random.default_rng(seed)
    a = (g.integers(0, 256, size=(4, m, k), dtype=np.int16).sum(axis=0) - 510) >> 4
    out = g.random((m, k)) < 1 / 1024
    a[out] = g.integers(-127, 128, size=int(out.sum()))
    return a.astype(np.int8)


def operand_file(cfg: Path, kind: str, name: str) -> Path:
    return cfg / f"{kind}-{name}.bin"


def operand(cfg: Path, kind: str, li: int, t: int) -> np.ndarray:
    """Block t's A (M x K) or B (N x K) of layer li, from its file (NB of them, block-major)."""
    name, n, k = LAYERS[li]
    rows = M if kind == "A" else n
    return np.memmap(operand_file(cfg, kind, name), dtype=np.int8, mode="r", offset=t * rows * k, shape=(rows, k))


def operand_domain(kind: str, name: str, t: int, rows: int, k: int) -> CommitmentDomain:
    return CommitmentDomain(identity_digest_sha512(OPERANDS_TAG, {"operand": kind, "layer": name, "block": t, "rows": rows, "k": k}),
                            -1, RangeIndexedDomain(0, rows), hash="sha512")


@dataclass
class OperandTree:
    """A registered operand, block t's A or B of a layer: its rows as hm96 rows (salts prf_salt(kw, row)) in one
    frame-v3-sha512 tree."""
    kind: str
    name: str
    t: int
    rows: int
    k: int
    kw: bytes
    levels: tuple

    @property
    def root(self) -> bytes:
        return self.levels[-1][0]

    @staticmethod
    def build(kind: str, name: str, t: int, X: np.ndarray, kw: bytes) -> OperandTree:
        rows, k = X.shape
        dom = operand_domain(kind, name, t, rows, k)
        hs = [v31.leaf_h(X[p].tobytes(), v31.cref.prf_salt(kw, p)) for p in range(rows)]
        return OperandTree(kind, name, t, rows, k, kw, tree_levels(dom, [row_leaf(dom, p, h, V2) for p, h in enumerate(hs)]))

    def opening(self, X: np.ndarray, j: int) -> tuple[bytes, bytes, tuple[bytes, ...]]:
        """(row, salt, path) of row j."""
        return (X[j].tobytes(), v31.cref.prf_salt(self.kw, j),
                tuple(self.levels[d][(j >> d) ^ 1] for d in range(len(self.levels) - 1)))


def operand_reason(kind: str, name: str, t: int, rows: int, k: int, root: bytes, j: int, opening: tuple) -> str | None:
    row, salt, path = opening
    if len(row) != k or not verify_opening(operand_domain(kind, name, t, rows, k), Commitment(root, rows),
                                           Opening(j, v31.leaf_h(row, salt), tuple(path)), V2, VerificationLimits()):
        return f"{kind} row {j} doesn't open against the registered {kind} root of {name} block {t}"
    return None


def _tree(job: tuple) -> OperandTree:
    cfg, kind, li, t, kw, workload = job
    configure(workload)
    return OperandTree.build(kind, LAYERS[li][0], t, np.asarray(operand(Path(cfg), kind, li, t)), kw)


def operand_trees(cfg: Path, secret: bytes, pool) -> dict[tuple[str, int, int], OperandTree]:
    """Every operand's tree, keyed (kind, layer, block), each under its own salt key."""
    workload = {"layers": [list(x) for x in LAYERS], "m": M, "nb": NB, "served": SERVED}
    jobs = [(str(cfg), kind, li, t, H("verity/highutil/operand-salts/v1", secret, kind.encode(), u64(li), u64(t)), workload)
            for kind in ("A", "B") for li in range(len(LAYERS)) for t in range(NB)]
    return {(j[1], j[2], j[3]): tr for j, tr in zip(jobs, pool.map(_tree, jobs))}


def configure(prep: dict) -> None:
    """The workload a config was prepared for (its layers, M, NB, L' and L; the 8B shapes unless a test made it smaller)."""
    global LAYERS, M, NB, LP, L, LP_MAX, SERVED
    LAYERS, M, NB = tuple(tuple(x) for x in prep["layers"]), prep["m"], prep["nb"]
    LP, L, LP_MAX = prep.get("lp", LP), prep.get("L", L), prep.get("lp_max", 1024)
    SERVED = prep.get("served", False)


def checkpoint_weights(d: Path, name: str, modules: tuple[str, ...], t: int) -> np.ndarray:
    from safetensors import safe_open
    idx = json.loads((d / "model.safetensors.index.json").read_text())["weight_map"]
    parts = []
    for mod in modules:
        key = f"model.layers.{t}.{mod}.weight"
        with safe_open(str(d / idx[key]), framework="np") as f:
            parts.append(f.get_tensor(key))
    w = np.concatenate(parts)
    if w.dtype != np.int8:
        raise SystemExit(f"{d}: {name} block {t} is {w.dtype}, not int8")
    return w


def cmd_prep(a: argparse.Namespace) -> int:
    d = Path(a.cfg)
    d.mkdir(parents=True, exist_ok=True)
    if a.small:
        configure({"layers": [["a", 256, 1024], ["b", 128, 2048]], "m": 256, "nb": 2, "lp": 128, "L": a.L or 512,
                   "lp_max": a.lp_max, "served": a.served})
    else:
        configure({"layers": [list(x) for x in LINEARS], "m": M, "nb": a.nb, "lp": a.lp, "L": a.L or L, "lp_max": a.lp_max,
                   "served": a.served})
    linears = [x[0] for x in LAYERS]
    parts = calls(LAYERS)
    configure({"layers": [[c, n, k] for c, n, k, _, _ in parts], "m": M, "nb": NB, "lp": LP, "L": L, "lp_max": LP_MAX,
               "served": SERVED})
    if a.checkpoint:
        ck = Path(a.checkpoint)
        if ck.name != CHECKPOINT["revision"]:
            raise SystemExit(f"{ck}: not the pinned revision {CHECKPOINT['revision']}")
        if linears != [x[0] for x in FUSED]:
            raise SystemExit("--checkpoint: the 8B layers only")
        shards = sorted(set(json.loads((ck / "model.safetensors.index.json").read_text())["weight_map"].values()))
        digests = {}
        for s in shards:
            with open(ck / s, "rb") as f:
                digests[s] = hashlib.file_digest(f, "sha256").hexdigest()
        source = {"weights": {**CHECKPOINT, "shards_sha256": digests, "fused": [[n, list(ms)] for n, ms in FUSED]},
                  "activations": "capture.py (capture.json)"}
    else:
        source = {"weights": "seeded", "activations": "seeded", "seed": a.seed}
    lines = []
    for li, (name, n, k, linear, k0) in enumerate(parts):
        shape(n, k, a.row)
        with open(operand_file(d, "B", name), "wb") as f:
            for t in range(NB):
                if a.checkpoint:
                    w = checkpoint_weights(Path(a.checkpoint), linear, FUSED[linears.index(linear)][1], t)[:, k0:k0 + k]
                else:
                    w = seeded_weights(n, k, (a.seed * 16 + li) * 64 + t)
                if w.shape != (n, k):
                    raise SystemExit(f"{name} block {t}: {w.shape}, not {(n, k)}")
                np.ascontiguousarray(w).tofile(f)
        if not a.checkpoint:
            with open(operand_file(d, "A", name), "wb") as f:
                for t in range(NB):
                    seeded_activations(M, k, (a.seed * 16 + li) * 64 + 32 + t).tofile(f)
        lines.append(f"{name} {M} {n} {k} {NB} {linear} {k0}")
    (d / "shapes.txt").write_text("\n".join(lines) + "\n")
    source["calls"] = {"t_max": T_MAX, "slices": [[c, linear, k0, lp_of(k)] for c, _, k, linear, k0 in parts]}
    (d / "prep.json").write_text(json.dumps({"row": a.row, "layers": LAYERS, "m": M, "nb": NB, "lp": LP, "lp_max": LP_MAX, "L": L,
                                             **({"served": True} if SERVED else {}), "source": source}))
    return 0


# ---------------------------------------------------------------------------------------------------------- the coins
def call_seeds(coin: bytes, period: str, block: int, receipt: bytes, steps: int) -> list[bytes]:
    """The block's calls' seeds (384 bytes: dither's six streams, pad dA dB U V P), step-major, from its seeds coin: call
    (s, layer)'s coin is the block coin's stream at it, and its seeds are dither.call_seeds under the block's window salt
    (its operand-root receipt)."""
    key = derive(coin, "verity/pouw-accel/seeds/v1", {"period": period, "block": block, "receipt": receipt})
    return [v31.fill_seeds_bytes(v31.call_seeds(receipt, key.stream(("call", s, li), 64)))
            for s in range(steps) for li in range(len(LAYERS))]


@dataclass(frozen=True)
class Pick:
    block: int
    step: int     # in the block
    layer: int
    i: int
    j: int
    s: int


def picks(coin: bytes, period: str, block: int, receipt: bytes, steps: int, q: int, row: int) -> list[Pick]:
    """q proof blocks of a block from its check coin, each MAC equally likely: a step uniformly, a layer by its N K, a
    cell uniformly, then the inner index uniformly (so the segment by its length)."""
    key = derive(coin, "verity/pouw-accel/check/v1", {"period": period, "block": block, "receipt": receipt})
    w = [n * k for _, n, k in LAYERS]
    out = []
    for p in range(q):
        step = key.uniform(steps, ("step", p))
        x = key.uniform(sum(w), ("layer", p))
        li = next(i for i in range(len(w)) if x < sum(w[:i + 1]))
        _, n, k = LAYERS[li]
        sh = shape(n, k, row)
        i, j, kk = key.uniform(M, ("i", p)), key.uniform(n, ("j", p)), key.uniform(k, ("k", p))
        out.append(Pick(block, step, li, i, j, kk // sh.L + 1))
    return out


def positions(f: v31.Fill, p: Pick) -> list[int]:
    return v31.fill_cells_positions(f, p.i, p.j, p.s)


# ------------------------------------------------------------------------------------------------ prover and verifier
def _cheated(cheat: str) -> int | None:
    return int(cheat.split(":")[1]) if cheat.startswith("skip:") else None


def salt(key: tuple[bytes, int], pos: int) -> bytes:
    """Leaf pos's salt in a call whose key is (Ks, off): prf_salt(Ks, off + pos) (dither.salts(Ks, off); the c-th call of a
    Prover has off = c leaves)."""
    return v31.cref.prf_salt(key[0], key[1] + pos)


def leaf_row(f: v31.Fill, afile: str, bfile: str, t: int, seeds: dict[str, bytes], pos: int, skip: int | None = None) -> bytes:
    """Leaf `pos`'s row as the prover committed it, recomputed from block t's operands and the call's coins: T_s of its
    ROW / 2 cells (a run of a tile's rows, every column), mod 2^16, or a served call's O of its ROW / 4 cells, mod 2^32,
    little-endian, in the tile's row-major order."""
    tile, r = divmod(pos, f.lpt)
    o = r >= f.W * f.row_w
    s, piece = (None, r - f.W * f.row_w) if o else (r // f.row_w + 1, r % f.row_w)
    nr = f.row // (4 if o else 2) // TILE
    tm, tn = divmod(tile, f.n // TILE)
    i0, j0 = tm * TILE + piece * nr, tn * TILE
    A = np.fromfile(afile, dtype=np.int8, count=nr * f.k, offset=(t * f.m + i0) * f.k).reshape(nr, f.k)
    B = np.fromfile(bfile, dtype=np.int8, count=TILE * f.k, offset=(t * f.n + j0) * f.k).reshape(TILE, f.k)
    fm = v31.form(f, A, B, v31.Coins.rows(f, seeds, range(i0, i0 + nr), range(j0, j0 + TILE)))
    if o:
        return (v31.imatmul(fm.Ad, fm.Bd) & 0xFFFFFFFF).astype("<u4").tobytes()
    X = fm.X
    if skip is not None:
        X = X.copy()
        X[:, skip * f.lp:(skip + 1) * f.lp] = 0
    e = f.seg(s)[1]
    return (v31.imatmul(X[:, :e], fm.Wp[:, :e]) & 0xFFFF).astype("<u2").tobytes()


def answer(job: dict) -> dict:
    """The prover's answer to one pick: the opened rows (recomputed from the operands and coins, as committed), each with
    its salt and path, and A's and B's rows with their operand openings."""
    f = shape(job["n"], job["k"], job["row"], job["m"])
    seeds = v31.fill_seeds_from(job["seeds"])
    skip = _cheated(job["cheat"])
    rows = {pos: (leaf_row(f, job["afile"], job["bfile"], job["t"], seeds, pos, skip), salt(job["key"], pos),
                  tuple(path)) for pos, path in zip(job["positions"], job["paths"])}
    return {"rows": rows, "A_opening": job["A_opening"], "B_opening": job["B_opening"]}


def verify_pick(job: dict, ans: dict) -> str | None:
    """The verifier's checks of one pick (None: passed): A's row i and B's row j against the registered operand roots of
    the call's block, then v31.py's fill_check on the verifier's own seeds, those rows and the call's root."""
    f = shape(job["n"], job["k"], job["row"], job["m"])
    i, j, s, t = job["i"], job["j"], job["s"], job["t"]
    for kind, x, rows in (("A", i, f.m), ("B", j, f.n)):
        why = operand_reason(kind, job["name"], t, rows, f.k, job["operand_roots"][kind], x, ans[f"{kind}_opening"])
        if why:
            return why
    want = set(v31.fill_cells_positions(f, i, j, s))
    if set(ans["rows"]) != want:
        return f"the answer opens leaves {sorted(ans['rows'])}, not {sorted(want)}"
    return v31.fill_check(f, v31.fill_seeds_from(job["verifier_seeds"]), job["root"], i, j, s,
                          {"rows": ans["rows"], "A": np.frombuffer(ans["A_opening"][0], np.int8),
                           "B": np.frombuffer(ans["B_opening"][0], np.int8)})


def prove_and_verify(job: dict) -> tuple[int, str | None]:
    try:
        return job["index"], verify_pick(job, answer(job))
    except Exception as e:  # noqa: BLE001 (a failed answer, whatever raised, is a failed pick, never a retry)
        return job["index"], f"{type(e).__name__}: {e}"


def _split_cheat(cheat: str) -> tuple[str, int | None]:
    kind, _, at = cheat.partition("@")
    return kind, int(at) if at else None


# --------------------------------------------------------------------------------------------------------- the drivers
class GpuDriver:
    """accel_v31's no-fold prover (v31lib's Prover over libv31.so: one Prover a call of the step, every call on the
    binding's shared streams in issue order, each commit beside the next call's compute) and plain cuBLASLt
    (torch._int_mm) on the same linears, each whole (a K-sliced one as one GEMM). Both operands of every block sit on the
    GPU; step g's calls read block g mod NB (`Prover.point_weights`, no copy). One thread drives every Prover, as the
    binding requires: a step's calls are issued, then the step before's tree levels are queued for copy to page-locked
    host memory behind their commits (`Prover.levels_async`, before each layer's call two after is issued), so the host
    never waits on the device, in a block or between blocks, but in v31's own wait for call c - 2's commit (`issue`);
    the roots (each tree's last node) and the paths for the openings after the block's check coin are read from those
    copies once the block is done (`done`, on another thread), a copy a block, two in turn."""

    def __init__(self, lib: str, cfg: Path, row: int, rmax: int) -> None:
        import torch
        import v31lib
        self.torch, self.rmax = torch, rmax
        self.lib = v31lib.Lib(lib)
        self.params = Path(tempfile.mkdtemp(prefix="highutil-params-", dir="/dev/shm" if Path("/dev/shm").is_dir() else None))
        self.ps, self.A, self.B, self.out, self.geom = [], [], [], [], []
        for name, n, k in LAYERS:
            f = shape(n, k, row)
            p = v31lib.Prover(M, n, k, f.lp, L, row, served=SERVED, lib=self.lib, params_dir=str(self.params))
            self.ps.append(p)
            depth = p.info()["depth"]
            self.geom.append((1 << depth, depth, (2 * (1 << depth) - 1) * 64))
            for kind, rows, dst in (("A", M, self.A), ("B", n, self.B)):
                x = np.fromfile(operand_file(cfg, kind, name), dtype=np.int8).reshape(NB, rows, k)
                dst.append(torch.from_numpy(x).cuda())
            self.out.append(torch.empty((M, n), dtype=torch.int32, device="cuda"))
            assert f.S == p.info()["S"]
        # the plain arm runs each linear whole, as the model does: a K-sliced linear's operands joined along K
        self.plain = []
        for linear in dict.fromkeys(name.split(".")[0] for name, _, _ in LAYERS):
            lis = [li for li, (name, _, _) in enumerate(LAYERS) if name.split(".")[0] == linear]
            join = (lambda x: x[lis[0]]) if len(lis) == 1 else (lambda x: torch.cat([x[li] for li in lis], dim=2))
            self.plain.append((join(self.A), join(self.B), self.out[lis[0]] if len(lis) == 1 else
                               torch.empty((M, LAYERS[lis[0]][1]), dtype=torch.int32, device="cuda")))
        # cuBLASLt's baseline (v31's own lt_setup, autotuned once a linear on block 0's operands), on torch's stream
        self.stream = torch.cuda.current_stream().cuda_stream
        self.lts, self.lt_algos = [], {}
        for (a, w, o), linear in zip(self.plain, dict.fromkeys(name.split(".")[0] for name, _, _ in LAYERS)):
            h = self.lib.lt_create(M, w.shape[1], w.shape[2], a[0].data_ptr(), w[0].data_ptr(), o.data_ptr(), self.stream)
            self.lts.append(h)
            algo = np.zeros(4, np.int32)
            self.lib.lt_algo(h, algo.ctypes.data)
            self.lt_algos[linear] = dict(zip(("algo", "tile", "stages", "splitk"), algo.tolist()))
        # each layer's trees of a block, rmax steps, as native u64 words (8 a node), page-locked for levels_async: two
        # copies, so block b's are read for its paths while block b + 1 writes the other
        self.levels = [[torch.empty((rmax, 8 * (2 * w - 1)), dtype=torch.int64, pin_memory=True).numpy().view(np.uint64)
                        for w, _, _ in self.geom] for _ in range(2)]
        self.streams = [torch.cuda.ExternalStream(self.ps[0].stream(w)) for w in (0, 1)]
        torch.cuda.synchronize()
        self.inflight: dict[int, dict] = {}
        self.ready = ["ready", torch.cuda.get_device_name(), f"ltw {settings.ltw()}"]
        self._poll_clock()

    def _poll_clock(self) -> None:
        """The SM clock, sampled every 200 ms by one nvidia-smi beside the run (none inside a timed block); HU_CLOCK=0: none
        (run.sh's nsys runs, where nsys would trace it too)."""
        import threading
        uuid = settings.lease_uuid()
        self.clock, self.smi = 0, None
        if not settings.poll_clock() or not shutil.which("nvidia-smi"):
            return
        self.smi = subprocess.Popen(["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits", "-lms", "200",
                                     *(["-i", uuid] if uuid else [])], stdout=subprocess.PIPE, text=True)

        def read() -> None:
            for line in self.smi.stdout:
                if line.strip().isdigit():
                    self.clock = int(line)
        threading.Thread(target=read, daemon=True).start()

    def _mhz(self) -> int:
        return self.clock

    def block(self, b: int, g0: int, steps: int, arm: str, cheat: str, seeds: Path, roots: Path) -> dict:
        torch = self.torch
        if steps < 1 or steps > self.rmax:
            raise ValueError(f"a block is 1 .. {self.rmax} steps")
        if arm.startswith("plain"):
            base = arm.removeprefix("plain-")
            if base not in BASELINES:
                raise ValueError(f"a plain arm's baseline is one of {BASELINES}, not {base!r}")
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            t0 = time.monotonic()
            e0.record()
            for s in range(steps):
                t = (g0 + s) % NB
                if base == "lt":
                    for (a, w, o), h in zip(self.plain, self.lts):
                        self.lib.lt_run(h, a[t].data_ptr(), w[t].data_ptr(), o.data_ptr(), self.stream)
                else:
                    for a, w, o in self.plain:
                        torch._int_mm(a[t], w[t].t(), out=o)
            e1.record()
            e1.synchronize()
            return {"gpu_ms": e0.elapsed_time(e1), "host_ms": 1e3 * (time.monotonic() - t0), "flags": 0, "sm_mhz": self._mhz()}
        self.issue(b, g0, steps, cheat, seeds.read_bytes())
        r = self.done(b)
        roots.write_bytes(r.pop("roots"))
        return r

    def issue(self, b: int, g0: int, steps: int, cheat: str, rec: bytes) -> None:
        """Block b's calls, enqueued behind whatever the device is still running (block b - 1's last calls), with no
        wait on the host but v31's own for call c - 2's commit. Its trees' levels go to copy b mod 2 of the page-locked
        levels, which `done(b - 2)` and `paths(b - 2, ..)` must be finished with. Timing events on the compute stream
        before the first call and on the commit stream after the last copy bound the block on the device."""
        torch, NL = self.torch, len(LAYERS)
        if steps < 1 or steps > self.rmax:
            raise ValueError(f"a block is 1 .. {self.rmax} steps")
        kind, at = _split_cheat(cheat)
        bad = [kind if kind != "none" and at in (None, li) else "none" for li in range(NL)]
        lv, keys = self.levels[b & 1], {}
        if b == 0:
            self.inflight = {}

        def collect(s: int, li: int, c: int) -> None:
            p = self.ps[li]
            p.levels_async(c, lv[li][s].ctypes.data)
            keys[(s, li)] = (p.ks, p.calls[c]["off"])

        t0 = time.monotonic()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(self.streams[0])
        prev: list[tuple[int, int, int]] = []
        for s in range(steps):
            t = (g0 + s) % NB
            cur = []
            for li, p in enumerate(self.ps):
                p.point_weights(self.B[li][t].data_ptr())
                sd = v31.fill_seeds_from(rec[384 * (s * NL + li):384 * (s * NL + li + 1)], SERVED)
                cur.append((s, li, p.prove(sd, self.A[li][t].data_ptr(), bad[li])))
            for x in prev:
                collect(*x)
            prev = cur
        for x in prev:
            collect(*x)
        end.record(self.streams[1])
        self.inflight[b] = {"start": start, "end": end, "steps": steps, "keys": keys,
                            "host_ms": 1e3 * (time.monotonic() - t0), "sm_mhz": self._mhz()}

    def done(self, b: int) -> dict:
        """Block b once the device has finished it (any thread but the one driving the Provers may wait here): its
        roots, and its device time, from the later of its first call's start and block b - 1's end to its end, so
        that time the device spent with nothing of this period's to run is the host's (`idle_before_ms`)."""
        x = self.inflight[b]
        x["end"].synchronize()
        span = x["start"].elapsed_time(x["end"])
        prev = self.inflight.get(b - 1) if b else None
        since = prev["end"].elapsed_time(x["end"]) if prev else span
        self.inflight.pop(b - 2, None)
        lv = self.levels[b & 1]
        roots = b"".join(lv[li][s][-8:].byteswap().tobytes() for s in range(x["steps"]) for li in range(len(LAYERS)))
        return {"gpu_ms": min(span, since), "host_ms": x["host_ms"], "idle_before_ms": max(0.0, since - span), "flags": 0,
                "sm_mhz": x["sm_mhz"], "roots": roots}

    def key(self, b: int, step: int, layer: int) -> tuple[bytes, int]:
        return self.inflight[b]["keys"][(step, layer)]

    def paths(self, b: int, picks_file: Path, out: Path) -> None:
        recs = np.frombuffer(picks_file.read_bytes(), dtype="<u4").reshape(-1, 3)
        with open(out, "wb") as f:
            for s, li, pos in recs.tolist():
                f.write(self.path(li, s, pos, b))

    def path(self, li: int, s: int, pos: int, b: int = 0) -> bytes:
        """Leaf pos's siblings in block b's step s's tree of layer li, leaf level first (64 big-endian bytes a node)."""
        width, depth, _ = self.geom[li]
        lv = self.levels[b & 1][li][s]
        return b"".join(lv[8 * x:8 * x + 8].byteswap().tobytes() for x in
                        (2 * width - 2 * (width >> d) + ((pos >> d) ^ 1) for d in range(depth)))

    def open_now(self, li: int, c: int, positions: list[int]):
        """The GPU's own openings of call c (one of its layer's last two calls): rows, salts and paths."""
        return self.ps[li].open(c, positions)

    def close(self) -> None:
        if self.smi:
            self.smi.terminate()
        self.torch.cuda.synchronize()
        for h in self.lts:
            self.lib.lt_destroy(h)
        for p in self.ps:
            p.close()
        shutil.rmtree(self.params, ignore_errors=True)


class RefDriver:
    """The GPU driver's commands on the CPU through v31.py's no-fold reference (``period.py run CFG ref ...``, and the
    tests): the same words, roots and paths at small shapes, so the period's coins, picks, answers and verifier run
    without a GPU. Each layer has its own Ks, and call c of a layer salts from off = c leaves, as the Prover's."""

    def __init__(self, cfg: Path, row: int, secret: bytes | None = None) -> None:
        self.cfg, self.row = cfg, row
        self.ks = [os.urandom(64) if secret is None else H("verity/highutil/ref-ks/v1", secret, u64(li)) for li in range(len(LAYERS))]
        self.calls = [0] * len(LAYERS)
        self.levels: dict[tuple[int, int, int], list] = {}
        self.keys: dict[tuple[int, int, int], tuple[bytes, int]] = {}
        self.roots: dict[int, bytes] = {}
        self.ready = ["ready", "ref"]

    def block_of(self, step: int) -> int:
        return step % NB

    def block(self, b: int, g0: int, steps: int, arm: str, cheat: str, seeds: Path, roots: Path) -> dict:
        if arm.startswith("plain"):
            return {"gpu_ms": 0.0, "host_ms": 0.0, "flags": 0, "sm_mhz": 0}
        self.issue(b, g0, steps, cheat, seeds.read_bytes())
        r = self.done(b)
        roots.write_bytes(r.pop("roots"))
        return r

    def issue(self, b: int, g0: int, steps: int, cheat: str, rec: bytes) -> None:
        NL = len(LAYERS)
        kind, at = _split_cheat(cheat)
        if b == 0:
            self.levels, self.keys, self.roots = {}, {}, {}
        for x in [x for x in self.levels if x[0] <= b - 2]:
            del self.levels[x], self.keys[x]
        out = []
        for s in range(steps):
            for li, (_, n, k) in enumerate(LAYERS):
                f = shape(n, k, self.row)
                t = self.block_of(g0 + s)
                sd = v31.fill_seeds_from(rec[384 * (s * NL + li):384 * (s * NL + li + 1)])
                skip = _cheated(kind) if kind != "none" and at in (None, li) else None
                T, O = v31.fill_values(f, np.asarray(operand(self.cfg, "A", li, t)), np.asarray(operand(self.cfg, "B", li, t)),
                                       v31.Coins.expand(f, sd), skip)
                key = (self.ks[li], self.calls[li] * f.leaves)
                self.calls[li] += 1
                cm = v31.fill_commit(f, v31.fill_buffer(f, T, O), *key)
                self.levels[(b, s, li)], self.keys[(b, s, li)] = cm.levels, key
                out.append(cm.root)
        self.roots[b] = b"".join(out)

    def done(self, b: int) -> dict:
        return {"gpu_ms": 0.0, "host_ms": 0.0, "idle_before_ms": 0.0, "flags": 0, "sm_mhz": 0, "roots": self.roots.pop(b)}

    def key(self, b: int, step: int, layer: int) -> tuple[bytes, int]:
        return self.keys[(b, step, layer)]

    def paths(self, b: int, picks_file: Path, out: Path) -> None:
        recs = np.frombuffer(picks_file.read_bytes(), dtype="<u4").reshape(-1, 3)
        with open(out, "wb") as f:
            for s, li, pos in recs.tolist():
                lv = self.levels[(b, s, li)]
                f.write(b"".join(lv[d][(pos >> d) ^ 1] for d in range(len(lv) - 1)))

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------------------------------------- the period
class Coins:
    """One challenger session a period: a5_coins' Local (development, not of record) or GateClient (vy-challenger-1)."""

    def __init__(self, how: str, main: Path, peer: str | None, key_record: str | None, gate_key: str | None) -> None:
        import a5_coins as A5
        from verity.core.protocols.sampled_proofs import remote as R
        self.A5, self.R, self.how = A5, R, how
        self.main = R.main_lines(main)
        self.peer, self.key_record, self.gate_key = peer, key_record, gate_key

    def connect(self, address: str, secret: str):
        return self.A5.connect_when_listening(address, bytes.fromhex(secret))

    def session(self):
        if self.how == "local":
            return self.A5.Local(Path(tempfile.mkdtemp(prefix="highutil-challenger-")), self.main)
        return self.A5.GateClient(self.how.removeprefix("gate:"), self.gate_key, self.peer, self.key_record, self.main)

    def evidence(self, session, root: str, openings: list, dest: Path) -> dict:
        """The verifier's check of the session once it has ended: the auditor's ledger and the session's sealed files
        (`remote.trusted_session`), every coin used opened against its signed commitment after the receipt it was
        recorded after, and the open's pin on main."""
        from verity.core.protocols.sampled_proofs import challenger as Ch
        try:
            directory, ledger, pins_log, key = session.finish(dest)
            commitment, roots = self.R.trusted_session(directory, [ledger], {"root": root}, key, self.A5.lms_verify)
            bad = []
            for o in openings:
                why = Ch.live_reason(commitment, o, o.label, o.index, o.after)
                if why is None and roots.get(o.label, ())[o.index:o.index + 1] != (o.after,):
                    why = "the release log shows another receipt for it"
                if why:
                    bad.append(f"{o.label} {o.index}: {why}")
            try:
                Ch.pinned_session(self.R.ledger_lines(ledger), {"root": root}, key, self.A5.lines(pins_log), self.main,
                                  self.A5.lms_verify)
                pinned = None
            except Ch.CoinError as e:
                pinned = str(e)
            return {"trusted": True, "bad_openings": bad, "pinned_session": pinned, "released": {k: len(v) for k, v in roots.items()},
                    "session": session.session}
        except Exception as e:  # noqa: BLE001 (any failure of the challenger's session makes the period untrusted, recorded)
            return {"trusted": False, "error": f"{type(e).__name__}: {e}", "session": getattr(session, "session", None)}


@dataclass
class Plan:
    blocks: int
    steps: int    # a block's
    q: int        # proof blocks a period


class Period:
    def __init__(self, drv: GpuDriver | RefDriver, coins: Coins | None, trees: dict[tuple[str, int, int], OperandTree], cfg: Path, row: int,
                 source: dict, secret: bytes, work: Path, pool: ProcessPoolExecutor) -> None:
        self.drv, self.coins, self.trees, self.cfg, self.row = drv, coins, trees, cfg, row
        self.source, self.secret, self.work, self.pool = source, secret, work, pool
        self.g = 0   # the next global step

    def registration(self, pid: str, plan: Plan) -> tuple[bytes, dict]:
        reg = {"period": pid, "receipt": "dither no-fold (v31.py Fill)", "m": M, "nb": NB, "layers": [list(x) for x in LAYERS],
               "lp": {name: lp_of(k) for name, _, k in LAYERS}, "L": L, "row": self.row,
               **({"served": True} if SERVED else {}),
               "blocks": plan.blocks, "steps_per_block": plan.steps, "q": plan.q, "first_step": self.g,
               "W_reg": plan.blocks * plan.steps * step_macs(), "source": self.source,
               "operand_roots": {kind: [[self.trees[(kind, li, t)].root.hex() for t in range(NB)] for li in range(len(LAYERS))]
                                 for kind in ("A", "B")}}
        return H("verity/highutil/registration/v1", json.dumps(reg, sort_keys=True).encode()), reg

    def plain(self, pid: str, plan: Plan, base: str = "lt") -> dict:
        """A baseline: the same linears, each whole, back to back, no coins: cuBLASLt (``lt``) or torch._int_mm."""
        t0 = time.monotonic()
        blocks = []
        for b in range(plan.blocks):
            blocks.append(self.drv.block(b, self.g, plan.steps, f"plain-{base}", "none", self.work / "none", self.work / "none"))
            self.g += plan.steps
        wall = time.monotonic() - t0
        return {"period": pid, "arm": "plain", "baseline": base, "blocks": blocks, "steps": plan.blocks * plan.steps, "wall_s": wall,
                "macs": plan.blocks * plan.steps * step_macs()}

    def proto(self, pid: str, plan: Plan, cheat: str = "none") -> dict:
        NL = len(LAYERS)
        reg_digest, reg = self.registration(pid, plan)
        session = self.coins.session()
        t_s = time.monotonic()
        address, secret, _ = session.open(reg_digest.hex(), [["seeds", plan.blocks], ["check", plan.blocks]], 10, SCHEME)
        client = self.coins.connect(address, secret)
        t_open = time.monotonic() - t_s
        openings, blocks, all_picks, roots_of, seeds_of, keys_of = [], [], [], {}, {}, {}
        per_block_q = plan.q // plan.blocks
        val = {-2: reg_digest, -1: reg_digest}

        def receipt(b: int) -> bytes:
            return H("verity/highutil/operand-receipt/v2", reg_digest, u64(b), val[b - 2], u64(plan.steps * step_macs()))

        def seeded(b: int, coin: bytes) -> bytes:
            seeds_of[b] = call_seeds(coin, pid, b, receipt(b), plan.steps)
            return b"".join(seeds_of[b])

        def checked(b: int, coin: bytes) -> None:
            ps = picks(coin, pid, b, val[b], plan.steps, per_block_q, self.row)
            recs, meta = [], []
            for p in ps:
                _, n, k = LAYERS[p.layer]
                pos = positions(shape(n, k, self.row), p)
                recs += [(p.step, p.layer, x) for x in pos]
                meta.append((p, pos))
            pf, out = self.work / f"picks-{b}.bin", self.work / f"paths-{b}.bin"
            pf.write_bytes(np.array(recs, dtype="<u4").tobytes())
            self.drv.paths(b, pf, out)
            raw, off = out.read_bytes(), 0
            for p, pos in meta:
                _, n, k = LAYERS[p.layer]
                depth = (shape(n, k, self.row).leaves - 1).bit_length()
                keys_of[(p.block, p.step, p.layer)] = self.drv.key(p.block, p.step, p.layer)
                paths = []
                for _ in pos:
                    paths.append([raw[off + 64 * d:off + 64 * d + 64] for d in range(depth)])
                    off += 64 * depth
                all_picks.append((p, pos, paths))

        # Block b's seeds open after block b - 2's value receipt, so the first two open with the session, before the wall
        # starts, and block b + 2's open beside block b + 1 with block b's check coin: the thread driving the Provers
        # issues block after block, and this one waits for each block's end, reads its roots, opens its coins and
        # builds its paths (from its copy of the levels, which block b + 2 reuses once its seeds are out)
        seeds_q: queue.Queue = queue.Queue()
        issued: queue.Queue = queue.Queue()
        t_last, failed = [0.0], []
        t_a = time.monotonic()
        try:
            got = client.open_many([("seeds", b, receipt(b)) for b in range(min(2, plan.blocks))])
            openings += got
            for b, o in enumerate(got):
                seeds_q.put(seeded(b, o.coin))
            t_ahead = time.monotonic() - t_a

            def coin_side() -> None:
                try:
                    for b in range(plan.blocks):
                        if issued.get() is None:
                            return
                        r = self.drv.done(b)
                        raw = r.pop("roots")
                        roots_of[b] = [raw[64 * x:64 * x + 64] for x in range(plan.steps * NL)]
                        val[b] = H("verity/highutil/value-receipt/v1", reg_digest, u64(b), raw)
                        r["t_done_s"] = time.monotonic() - t0
                        blocks.append(r)
                        if b == plan.blocks - 1:
                            t_last[0] = r["t_done_s"]
                        ask = [("check", b, val[b])] + ([("seeds", b + 2, receipt(b + 2))] if b + 2 < plan.blocks else [])
                        got = client.open_many(ask) if len(ask) > 1 else [client.open(*ask[0])]
                        openings.extend(got)
                        checked(b, got[0].coin)
                        if len(got) > 1:
                            seeds_q.put(seeded(b + 2, got[1].coin))
                except BaseException as e:  # noqa: BLE001 (re-raised on the driving thread)
                    failed.append(e)
                    seeds_q.put(None)

            switch = sys.getswitchinterval()
            sys.setswitchinterval(1e-4)   # the driving thread back from a wait in well under a call's time
            side = threading.Thread(target=coin_side, daemon=True)
            t0 = time.monotonic()
            side.start()
            try:
                for b in range(plan.blocks):
                    rec = seeds_q.get()
                    if rec is None:
                        break
                    self.drv.issue(b, self.g, plan.steps, cheat, rec)
                    self.g += plan.steps
                    issued.put(b)
            finally:
                issued.put(None)
                side.join()
                sys.setswitchinterval(switch)
            if failed:
                raise failed[0]
        finally:
            client.close()
        t_last = t_last[0]
        ev = self.coins.evidence(session, reg_digest.hex(), openings, self.work / f"challenger-{pid}")
        # the verifier: its own seeds from the coins it was given, the operand roots it registered
        jobs = []
        cheat_layer = int(cheat.split("@")[1]) if "@" in cheat else None
        for idx, (p, pos, paths) in enumerate(all_picks):
            name, n, k = LAYERS[p.layer]
            g0 = reg["first_step"] + p.block * plan.steps
            call, t = (g0 + p.step) * NL + p.layer, (g0 + p.step) % NB
            seeds = seeds_of[p.block][p.step * NL + p.layer]
            ta, tb = self.trees[("A", p.layer, t)], self.trees[("B", p.layer, t)]
            jobs.append({"index": idx, "name": name, "m": M, "n": n, "k": k, "row": self.row, "i": p.i, "j": p.j, "s": p.s,
                         "t": t, "call": call, "seeds": seeds, "key": keys_of[(p.block, p.step, p.layer)],
                         "afile": str(operand_file(self.cfg, "A", name)), "bfile": str(operand_file(self.cfg, "B", name)),
                         "cheat": cheat.split("@")[0] if cheat_layer in (None, p.layer) else "none",
                         "positions": pos, "paths": paths,
                         "A_opening": ta.opening(operand(self.cfg, "A", p.layer, t), p.i),
                         "B_opening": tb.opening(operand(self.cfg, "B", p.layer, t), p.j),
                         "operand_roots": {"A": bytes.fromhex(reg["operand_roots"]["A"][p.layer][t]),
                                           "B": bytes.fromhex(reg["operand_roots"]["B"][p.layer][t])},
                         "verifier_seeds": seeds, "root": roots_of[p.block][p.step * NL + p.layer]})
        t_v = time.monotonic()
        fails = [(i, why) for i, why in self.pool.map(prove_and_verify, jobs, chunksize=4) if why]
        t_verify = time.monotonic() - t_v
        flags = 0
        for r in blocks:
            flags |= r["flags"]
        ok = not fails and not flags and ev.get("trusted") and not ev.get("bad_openings")
        macs = plan.blocks * plan.steps * step_macs()
        first = all_picks[fails[0][0]][0] if fails else None
        line = (f"{'ACCEPT' if ok else 'REFUSE'} period {pid}: {plan.blocks} blocks x {plan.steps} steps "
                f"({plan.blocks * plan.steps * NL} calls, {macs:.4g} MACs), {len(jobs)} proof blocks opened, "
                f"{len(jobs) - len(fails)} passed, {len(fails)} failed"
                + (f"; first: block {first.block} step {first.step} {LAYERS[first.layer][0]} ({first.i}, {first.j}, s {first.s}): "
                   f"{fails[0][1]}" if fails else "")
                + (f"; GPU flags {flags}" if flags else "")
                + ("" if ev.get("trusted") and not ev.get("bad_openings") else f"; coins: {ev.get('error') or ev.get('bad_openings')}"))
        return {"period": pid, "arm": "proto", "cheat": cheat, "registration": reg, "registration_digest": reg_digest.hex(),
                "blocks": blocks, "steps": plan.blocks * plan.steps, "wall_s": t_last, "session_open_s": t_open,
                "seeds_ahead_s": t_ahead,
                "verify_s": t_verify, "macs": macs, "credited_macs": macs if ok else 0, "accepted": bool(ok), "q": len(jobs),
                "eps_skip": eps_skip(len(jobs)), "delta": DELTA, "eta": ETA, "failures": [f"{all_picks[i][0]}: {w}" for i, w in fails[:8]],
                "failed": len(fails), "coins": ev, "verdict": line,
                "gaps_s": [blocks[b]["idle_before_ms"] / 1e3 for b in range(1, len(blocks))]}


def cmd_run(a: argparse.Namespace) -> int:
    cfg, out = Path(a.cfg), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prep = json.loads((cfg / "prep.json").read_text())
    configure(prep)
    row = prep["row"]
    plan = Plan(a.blocks, a.steps, a.q)
    card = load_card(a.card) if a.card else None
    secret = os.urandom(64)
    source = dict(prep["source"])
    if (cfg / "capture.json").exists():
        source["capture"] = json.loads((cfg / "capture.json").read_text())
    missing = [str(operand_file(cfg, "A", name)) for name, _, _ in LAYERS if not operand_file(cfg, "A", name).exists()]
    if missing:
        raise SystemExit(f"no activations: {missing} (capture.py writes them)")
    work = Path(tempfile.mkdtemp(prefix="highutil-", dir="/dev/shm" if Path("/dev/shm").is_dir() else None))
    coins = Coins(a.coins, Path(a.main), a.peer, a.challenger_key, a.gate_key)
    results, drv = [], None
    workers = a.workers or max(1, len(os.sched_getaffinity(0)) - 4)
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            t = time.monotonic()
            trees = operand_trees(cfg, secret, pool)
            print(f"{len(trees)} operand roots in {time.monotonic() - t:.1f} s: "
                  + " ".join(f"{k}{li}.{tb} {x.root.hex()[:16]}" for (k, li, tb), x in trees.items() if tb == 0), flush=True)
            drv = RefDriver(cfg, row) if a.lib == "ref" else GpuDriver(a.lib, cfg, row, plan.steps)
            print("driver " + " ".join(drv.ready), flush=True)
            per = Period(drv, coins, trees, cfg, row, source, secret, work, pool)
            # warm-up: one plain block, so the clocks and caches settle before anything is timed
            bases = baselines(a.plain)
            drv.block(0, 0, min(plan.steps, 50), f"plain-{bases[0]}", "none", work / "none", work / "none")
            arms = (["proto"] if a.arms in ("both", "proto") else []) + ([f"plain-{b}" for b in bases] if a.arms in ("both", "plain") else [])
            schedule = [(x, "none") for x in arms] * a.periods + ([("proto", a.negative)] if a.negative else [])
            for x, (arm, cheat) in enumerate(schedule):
                pid = f"{a.tag}-{x:02d}-{arm}" + ("" if cheat == "none" else "-" + cheat.replace(":", "").replace("@", "l"))
                r = per.proto(pid, plan, cheat) if arm == "proto" else per.plain(pid, plan, arm.removeprefix("plain-"))
                results.append(r)
                with open(out / "periods.jsonl", "a") as f:
                    f.write(json.dumps(r, default=str) + "\n")
                print(r.get("verdict") or f"plain period {pid}: {r['steps']} steps in {r['wall_s']:.3f} s", flush=True)
    finally:
        if drv:
            drv.close()
        for p in work.glob("challenger-*"):
            shutil.copytree(p, out / "challenger" / p.name.removeprefix("challenger-"), dirs_exist_ok=True)
        shutil.rmtree(work, ignore_errors=True)
    summary = summarize(results, plan, source, card, getattr(drv, "lt_algos", None))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    neg = [r for r in results if r["arm"] == "proto" and r["cheat"] != "none"]
    honest = [r for r in results if r["arm"] == "proto" and r["cheat"] == "none"]
    if a.arms == "plain":
        return 0
    return 0 if honest and all(r["accepted"] for r in honest) and all(not r["accepted"] for r in neg) else 1


def cmd_diag(a: argparse.Namespace) -> int:
    """One protocol step on the GPU (a block of the model other than the first) with seeds of its own (no coins), then
    leaves of each layer's call against the reference: the GPU's own openings (v31_open, inside the call's window) against
    the rows the prover's answers recompute from the operand files and against the paths from the host copy of the tree,
    every one opening the call's root."""
    cfg, out = Path(a.cfg), Path(a.out)
    prep = json.loads((cfg / "prep.json").read_text())
    configure(prep)
    row, NL = prep["row"], len(LAYERS)
    rng = np.random.default_rng(a.seed)
    g0 = NB - 1 if NB > 1 else 0
    t = g0 % NB
    work = Path(tempfile.mkdtemp(prefix="highutil-diag-", dir="/dev/shm" if Path("/dev/shm").is_dir() else None))
    drv = GpuDriver(a.lib, cfg, row, 1)
    rec = [v31.fill_seeds_bytes(v31.call_seeds(os.urandom(64), os.urandom(64))) for _ in range(NL)]
    (work / "seeds").write_bytes(b"".join(rec))
    drv.block(0, g0, 1, "proto", "none", work / "seeds", work / "roots")
    roots = (work / "roots").read_bytes()
    rep = []
    for li, (name, n, k) in enumerate(LAYERS):
        f = shape(n, k, row)
        seeds, key, root = v31.fill_seeds_from(rec[li]), drv.key(0, 0, li), roots[64 * li:64 * li + 64]
        pos = sorted({0, f.leaves - 1, *(int(x) for x in rng.integers(f.leaves, size=4))})
        gpu = drv.open_now(li, 0, pos)
        depth = drv.geom[li][1]
        dom, cmt = f.domain(), Commitment(root, f.leaves)
        bad = []
        for p, (grow, gsalt, gpath) in zip(pos, gpu):
            mine = leaf_row(f, str(operand_file(cfg, "A", name)), str(operand_file(cfg, "B", name)), t, seeds, p)
            path = drv.path(li, 0, p)
            if grow != mine:
                bad.append(f"leaf {p}: the GPU's row isn't the recomputed one")
            if gsalt != salt(key, p):
                bad.append(f"leaf {p}: salt")
            if b"".join(gpath) != path:
                bad.append(f"leaf {p}: the host copy's path isn't the GPU's")
            if not verify_opening(dom, cmt, Opening(p, v31.leaf_h(mine, gsalt), tuple(path[64 * d:64 * d + 64] for d in range(depth))),
                                  V2, VerificationLimits()):
                bad.append(f"leaf {p}: doesn't open the root")
        rep.append({"layer": name, "block": t, "S": f.S, "leaves": f.leaves, "checked": pos, "bad": bad})
        print(json.dumps(rep[-1]), flush=True)
    drv.close()
    shutil.rmtree(work, ignore_errors=True)
    ok = all(not r["bad"] for r in rep)
    (out / "diag.json").write_text(json.dumps({"exact": ok, "layers": rep}, indent=1))
    print(f"diag: {'exact' if ok else 'DIFFERS'}", flush=True)
    return 0 if ok else 1


#: the plain arm's baselines: the same linears, each whole, through
BASELINES = {"lt": "cuBLASLt's int32 GEMM, autotuned by v31.cu's lt_setup (the call a V31_LTW call makes for X W'^T)",
             "int_mm": "torch._int_mm (on the H200, CUTLASS's sm80 i16832 mma.sync kernel)"}


def baselines(spec: str) -> list[str]:
    bases = [b for b in spec.split(",") if b]
    if not bases or any(b not in BASELINES for b in bases):
        raise SystemExit(f"--plain: a comma-separated list of {', '.join(BASELINES)}, not {spec!r}")
    return bases


#: a plain period's baseline when its record names none (the runs before --plain timed torch._int_mm alone)
OLD_BASELINE = "int_mm"


def summarize(results: list[dict], plan: Plan, source: dict | None = None, card: dict | None = None,
              lt_algos: dict | None = None) -> dict:
    """The run's numbers, in the Glossary's words. The declared utilization is the declared work's MACs a second over θ,
    the tensor pipe's MAC rate at the card's SM-clock ceiling (the work's MFU), and the prover overhead (γ) is the
    protocol's time a step over the plain arm's; neither is a share of the bound. Both are against the first baseline
    the run timed (cuBLASLt by default), and ``baselines`` has them against each. The unaccounted share is ε_skip +
    ε_undecl; ε_undecl is only bounded here (1 - the declared utilization, the protocol's own work uncredited), since the
    protocol's own work counts as certified at the public best's cost, which breakdown.py's times let CERT-SHARE price."""
    honest = [r for r in results if r["arm"] == "proto" and r["cheat"] == "none"]
    plains = [r for r in results if r["arm"] == "plain"]
    base_of = lambda r: r.get("baseline", OLD_BASELINE)  # noqa: E731
    order = list(dict.fromkeys(base_of(r) for r in plains))
    plain = [r for r in plains if base_of(r) == (order[0] if order else None)]
    neg = [r for r in results if r["arm"] == "proto" and r["cheat"] != "none"]
    med = lambda v: float(np.median(v)) if v else None  # noqa: E731
    per_step = lambda r: r["wall_s"] / r["steps"]  # noqa: E731
    gpu_step = lambda r: sum(b["gpu_ms"] for b in r["blocks"]) / 1e3 / r["steps"]  # noqa: E731
    credited = sum(r["credited_macs"] for r in honest)
    wall = sum(r["wall_s"] for r in honest)
    mhz = [b["sm_mhz"] for r in honest for b in r["blocks"] if b["sm_mhz"]]
    out = {
        "workload": {"m": M, "nb": NB, "layers": [list(x) for x in LAYERS], "step_macs": step_macs(),
                     "lp": {name: lp_of(k) for name, _, k in LAYERS}, "L": L, "served": SERVED,
                     "source": source},
        "plan": plan.__dict__, "periods": len(honest),
        "accepted": sum(r["accepted"] for r in honest),
        "credited_macs": credited, "wall_s": wall,
        "credited_ands": {"value": credited * ANDS_PER_MAC[0] // ANDS_PER_MAC[1],
                          "lowering": lowering(card), "per_mac": ANDS_PER_MAC[0] / ANDS_PER_MAC[1],
                          "creditsize_pinned": False},
        "card": None, "peak_macs_s": {}, "declared_utilization": None, "declared_utilization_at_run_clock": None,
        "declared_utilization_without_protocol": None,
        "proto_s_per_step": med([per_step(r) for r in honest]), "plain_s_per_step": med([per_step(r) for r in plain]),
        "proto_gpu_s_per_step": med([gpu_step(r) for r in honest]), "plain_gpu_s_per_step": med([gpu_step(r) for r in plain]),
        "prover_overhead": {"gpu": None, "wall": None},
        "gaps_s": med([g for r in honest for g in r["gaps_s"]]),
        "session_open_s": med([r["session_open_s"] for r in honest]),
        "q": med([r["q"] for r in honest]), "eps_skip": med([r["eps_skip"] for r in honest]), "delta": DELTA,
        "unaccounted_share": None,
        "sm_mhz_median": med(mhz),
        "verdicts": [r["verdict"] for r in honest + neg],
        "negative_refused": [not r["accepted"] for r in neg],
    }
    ratio = lambda a, b: a / b if a and b else None  # noqa: E731
    if card:
        out["card"] = {"sku": card["sku"], "clock_mhz": card["clock"]["sm"]["value"], "theta_macs_s": theta(card),
                       "tensor": card["rate"]["tensor"]["value"], "sms": card["sm"]["count"]["value"],
                       "clock_source": card["clock"]["sm"]["source"], "tensor_source": card["rate"]["tensor"]["source"]}
        out["peak_macs_s"]["card"] = theta(card)
        if out["sm_mhz_median"]:
            out["peak_macs_s"]["run_clock"] = theta(card, out["sm_mhz_median"])
            out["declared_utilization_at_run_clock"] = ratio(credited, out["peak_macs_s"]["run_clock"] * wall)
        out["declared_utilization"] = du = ratio(credited, theta(card) * wall)
        out["declared_utilization_without_protocol"] = med([r["macs"] / r["wall_s"] / theta(card) for r in plain])
        if du is not None and out["eps_skip"] is not None:
            out["unaccounted_share"] = {
                "eps_skip": out["eps_skip"], "eps_undecl_at_most": 1 - du, "eps_at_most": out["eps_skip"] + 1 - du,
                "delta": DELTA,
                "eps_undecl": "1 - the certified share, which counts the protocol's own work at the public best's cost: "
                              "breakdown.json's times, priced at kernels' public-best rates (CERT-SHARE, CMP-190)"}
    out["prover_overhead"] = {"gpu": ratio(out["proto_gpu_s_per_step"], out["plain_gpu_s_per_step"]),
                              "wall": ratio(out["proto_s_per_step"], out["plain_s_per_step"])}
    out["baseline"] = order[0] if order else None
    out["baselines"] = {}
    for base in order:
        rs = [r for r in plains if base_of(r) == base]
        b_wall, b_gpu = med([per_step(r) for r in rs]), med([gpu_step(r) for r in rs])
        out["baselines"][base] = {
            "kernel": BASELINES[base], "plain_s_per_step": b_wall, "plain_gpu_s_per_step": b_gpu,
            "sm_mhz_median": med([b["sm_mhz"] for r in rs for b in r["blocks"] if b["sm_mhz"]]),
            "prover_overhead": {"gpu": ratio(out["proto_gpu_s_per_step"], b_gpu), "wall": ratio(out["proto_s_per_step"], b_wall)},
            "declared_utilization_without_protocol": med([r["macs"] / r["wall_s"] / theta(card) for r in rs]) if card else None,
            **({"cublaslt_algos": lt_algos} if base == "lt" and lt_algos else {})}
    return out


def cmd_summary(a: argparse.Namespace) -> int:
    """summary.json again from a run's stored periods.jsonl, prep.json and card.toml (the old summary's plan, source and
    cuBLASLt algorithms carried over), with no GPU: a change to the summary's words or numbers rebuilds a run of record."""
    out = Path(a.out)
    configure(json.loads((out / "prep.json").read_text()))
    old = json.loads((out / "summary.json").read_text())
    results = [json.loads(x) for x in (out / "periods.jsonl").read_text().splitlines() if x.strip()]
    card = load_card(a.card or str(out / "card.toml"))
    summary = summarize(results, Plan(**old["plan"]), old["workload"].get("source"), card,
                        old.get("baselines", {}).get("lt", {}).get("cublaslt_algos"))
    (Path(a.to) if a.to else out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("cfg")
    p.add_argument("--row", type=int, default=8192, help="leaf row bytes")
    p.add_argument("--lp", type=int, default=LP, help="the least noise tile length L'")
    p.add_argument("--lp-max", type=int, default=LP_MAX, help="the longest L' (v31's Fill takes up to 1024)")
    p.add_argument("--L", type=int, help=f"the word interval L (default {L}; --small's 512)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--checkpoint", help=f"{CHECKPOINT['repo']}'s snapshot directory at {CHECKPOINT['revision'][:10]} (B; "
                                        "capture.py writes A)")
    g.add_argument("--seeded", action="store_true", help="seeded A and B")
    g.add_argument("--small", action="store_true", help="two small layers at M 256, two blocks, seeded (tests)")
    p.add_argument("--served", action="store_true", help="each call in its served form: T_1 .. T_{S-1} and O committed "
                                                         "(v31.py's Fill(o=True)); else the filler's T_1 .. T_S")
    p.add_argument("--nb", type=int, default=32, help="transformer blocks")
    p.add_argument("--seed", type=int, default=1)
    p.set_defaults(fn=cmd_prep)
    p = sub.add_parser("run")
    p.add_argument("cfg")
    p.add_argument("lib", help="libv31.so (v31lib.build_lib), or `ref`: v31.py's reference on the CPU")
    p.add_argument("out")
    p.add_argument("--coins", default="local", help="local (development, not of record) or gate:USER@HOST")
    p.add_argument("--card", help="θ's card: verity/hardware's card file, or its SKU (verity.hardware.load)")
    p.add_argument("--main", required=True, help="the verifier's copy of main's first-parent line (git rev-list --first-parent)")
    p.add_argument("--peer")
    p.add_argument("--challenger-key")
    p.add_argument("--gate-key")
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--q", type=int, default=2048)
    p.add_argument("--periods", type=int, default=2, help="protocol periods, each followed by a plain one")
    p.add_argument("--negative", default="skip:2@2", help="the cheating period's v31 cheat (`@L`: layer L only); empty: none")
    p.add_argument("--arms", choices=("both", "proto", "plain"), default="both", help="one arm only (a profile of it)")
    p.add_argument("--plain", default="lt,int_mm", help="the plain arm's baselines, comma-separated, gamma's first: lt "
                   "(cuBLASLt, v31's own lt_setup) and int_mm (torch._int_mm)")
    p.add_argument("--tag", default="hu")
    p.add_argument("--workers", type=int, default=0)
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("diag")
    p.add_argument("cfg")
    p.add_argument("lib")
    p.add_argument("out")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_diag)
    p = sub.add_parser("summary")
    p.add_argument("out", help="a run's out directory (periods.jsonl, prep.json, card.toml and the old summary.json)")
    p.add_argument("--card", help="θ's card (default OUT/card.toml)")
    p.add_argument("--to", help="where to write it (default OUT/summary.json)")
    p.set_defaults(fn=cmd_summary)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
