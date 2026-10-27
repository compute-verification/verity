"""HIGHUTIL: an 8B model's linear layers at full batch on one GPU, each call credited through Dither's no-fold receipt
(verity/core/protocols/pouw/accel/PROTOCOL.md v3.2) on V31-GPU's prover, with live coins, and the certified share of the
GPU's tensor peak that the periods establish.

**The workload.** Llama-3.1-8B's linear layers (QKV 6144 x 4096, O 4096 x 4096, gate-and-up 28672 x 4096, down 4096 x
14336) at M = 16,384, s8 x s8 with int32 accumulation, called back to back. A step is one transformer block's four calls,
and step g runs block g mod NB, so NB = 32 consecutive steps are the prefill's every linear. B is the published
checkpoint's weights (verity-int8/Llama-3.1-8B-INT8 at its pinned revision, q k v and gate up fused as served; `prep
--checkpoint`, the shards' SHA-256 in the registration). A is the s8 input each linear gets in a W8A8 forward of that
checkpoint over 16 x 1,024 tokens of wikitext-2 (`capture.py`), so each call computes what the model computes on that
batch, and the periods replay that one prefill. Both operands are committed before any coin: a frame-v3-sha512 tree a
(operand, layer, block), its rows hm96 rows (`OperandTree`), whose roots the registration carries. `prep --seeded`
seeds both instead (v31.py's ``llm`` rows for B), for tests and shapes the checkpoint doesn't have.

**A period** (`Period.proto`): the registration (the operand roots, the operands' source, the period's shape, W_reg) is the root of one challenger session with families ``seeds`` and ``check``, a coin each a block. Then
its sampling blocks run in one sequence. Block b's operand-root receipt is H(registration, b, block b-1's value-root
receipt), and its seeds coin opens after it. Each call's five v31 seeds (dA, dB, U, V, P) are
``derive(coin, "verity/pouw-accel/seeds/v1", ctx).stream(call)``. The GPU runs the block's calls, each committed under a
frame-v3-sha512 root with a fresh salt key, and block b's value-root receipt is H(registration, b, its roots). Its check
coin opens after that receipt, with block b+1's seeds coin in the same round trip. The coin draws Q W_b / W_reg proof
blocks of b (`picks`): a call by its MACs, a cell uniformly, a segment by its length, so each MAC is equally likely.
Each pick's tree paths come off the host copy of its call's tree before the next block overwrites it.

**The verifier** (`verify_pick`), after the period: the prover's answer (`answer`: the opened rows recomputed from the
tile's operands and coins, their salts and paths, and A's and B's rows with their operand openings) against v31.py's
``check_block``, on the verifier's own seeds from the coins, the opened operand rows and the call's root.
Credit is all-or-nothing: the period's MACs if every pick passes, else none. ε_skip = ln(1/δ)/(ηQ) at δ = 1%.

**Deviations from v3.2**, each named in the report: the picks are a fixed Q W_b / W_reg a block, not ``poisson:Q``;
the coins are 64-byte session coins expanded here, not the challenger's streamed seeds; no certifier signs the receipts
(the challenger's release log records what each coin opened after); and the checks run in the clear, the stand-in for
the Definition and C-Flock --zk.

    period.py prep CFG (--checkpoint DIR | --seeded) [--row 32768] [--nb 32]   weights, params and shapes for hu
    capture.py CFG --checkpoint DIR                                           the activations (on the GPU)
    period.py diag CFG HU OUT                                                 one step's tiles against v31.py
    period.py run CFG HU OUT [--coins local|gate:USER@HOST ...] [--blocks --steps --q --periods --negative]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # one BLAS thread a pool worker
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
TREE = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(TREE / "benchmarks" / "one_stage"))

import v31  # noqa: E402

from verity.ml.commitments.identity import identity_digest_sha512  # noqa: E402
from verity.ml.commitments.indexed import RangeIndexedDomain  # noqa: E402
from verity.ml.commitments.limits import VerificationLimits  # noqa: E402
from verity.ml.commitments.merkle import Commitment, CommitmentDomain, Opening, tree_levels, verify_opening  # noqa: E402
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2, row_leaf  # noqa: E402
from verity.ml.randomness import derive  # noqa: E402

LAYERS = (("qkv", 6144, 4096), ("o", 4096, 4096), ("up", 28672, 4096), ("down", 4096, 14336))
M = 16384
NB = 32       # transformer blocks: step g runs block g mod NB
LP, T = 256, 3
TILE = v31.TILE
SCHEME = "verity/highutil/period/v1"
OPERANDS_TAG = "verity/highutil/operands/v1"
CHECKPOINT = {"repo": "verity-int8/Llama-3.1-8B-INT8", "revision": "723d342ac443b040931242c768df342b0f6f15a7"}
#: hu's layers from the checkpoint's modules, fused as served: (name, the modules whose rows it stacks)
FUSED = (("qkv", ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")), ("o", ("self_attn.o_proj",)),
         ("up", ("mlp.gate_proj", "mlp.up_proj")), ("down", ("mlp.down_proj",)))
ETA = 1 - 2.8e-5
DELTA = 0.01
#: θ's sm_120 INT8 dense tensor peak (benchmarks/pouw/machine_model `theta --device sm_120`): 1,024 MACs a clock an SM,
#: 188 SMs, at the SKU's maximum boost clock; the node's own maximum SM clock beside it
MACS_PER_CLOCK_SM, SMS, SKU_MHZ, NODE_MHZ = 1024, 188, 2617, 2430


def peak_macs(mhz: float) -> float:
    return MACS_PER_CLOCK_SM * SMS * mhz * 1e6


def H(tag: str, *parts: bytes) -> bytes:
    h = hashlib.sha512(tag.encode() + b"\0")
    for p in parts:
        h.update(len(p).to_bytes(8, "big") + p)
    return h.digest()


def u64(x: int) -> bytes:
    return x.to_bytes(8, "big")


def shape(n: int, k: int, row: int, m: int | None = None) -> v31.Shape:
    return v31.Shape(M if m is None else m, n, k, LP, T, row)


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
    workload = {"layers": [list(x) for x in LAYERS], "m": M, "nb": NB}
    jobs = [(str(cfg), kind, li, t, H("verity/highutil/operand-salts/v1", secret, kind.encode(), u64(li), u64(t)), workload)
            for kind in ("A", "B") for li in range(len(LAYERS)) for t in range(NB)]
    return {(j[1], j[2], j[3]): tr for j, tr in zip(jobs, pool.map(_tree, jobs))}


def configure(prep: dict) -> None:
    """The workload a config was prepared for (its layers, M and NB; the 8B shapes unless a test made it smaller)."""
    global LAYERS, M, NB
    LAYERS, M, NB = tuple(tuple(x) for x in prep["layers"]), prep["m"], prep["nb"]


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
        configure({"layers": [["a", 256, 768], ["b", 128, 1536]], "m": 256, "nb": 2})
    else:
        configure({"layers": [list(x) for x in LAYERS], "m": M, "nb": a.nb})
    if a.checkpoint:
        ck = Path(a.checkpoint)
        if ck.name != CHECKPOINT["revision"]:
            raise SystemExit(f"{ck}: not the pinned revision {CHECKPOINT['revision']}")
        if [x[0] for x in LAYERS] != [x[0] for x in FUSED]:
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
    for li, (name, n, k) in enumerate(LAYERS):
        sh = shape(n, k, a.row)
        with open(operand_file(d, "B", name), "wb") as f:
            for t in range(NB):
                w = checkpoint_weights(Path(a.checkpoint), name, FUSED[li][1], t) if a.checkpoint else \
                    seeded_weights(n, k, (a.seed * 16 + li) * 64 + t)
                if w.shape != (n, k):
                    raise SystemExit(f"{name} block {t}: {w.shape}, not {(n, k)}")
                w.tofile(f)
        if not a.checkpoint:
            with open(operand_file(d, "A", name), "wb") as f:
                for t in range(NB):
                    seeded_activations(M, k, (a.seed * 16 + li) * 64 + 32 + t).tofile(f)
        (d / f"params-{name}.bin").write_bytes(v31.params_file(sh, os.urandom(64)))
        lines.append(f"{name} {M} {n} {k} {NB}")
    (d / "shapes.txt").write_text("\n".join(lines) + "\n")
    (d / "prep.json").write_text(json.dumps({"row": a.row, "layers": LAYERS, "m": M, "nb": NB, "source": source}))
    return 0


# ---------------------------------------------------------------------------------------------------------- the coins
def call_seeds(coin: bytes, period: str, block: int, receipt: bytes, steps: int) -> list[bytes]:
    """The block's calls' v31 seeds (320 bytes: dA, dB, U, V, P), step-major, from its seeds coin."""
    key = derive(coin, "verity/pouw-accel/seeds/v1", {"period": period, "block": block, "receipt": receipt})
    return [key.stream(("call", s, li), 320) for s in range(steps) for li in range(len(LAYERS))]


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


def positions(sh: v31.Shape, p: Pick) -> list[int]:
    return ([sh.word_leaf(p.i, p.j, p.s)[0]] + ([sh.word_leaf(p.i, p.j, p.s - 1)[0]] if p.s > 1 else [])
            + ([sh.o_leaf(p.i, p.j)[0]] if p.s == sh.S else []))


def salt_key(secret: bytes, call: int) -> bytes:
    """The prover's salt key for global call `call`: fresh a call, so no two calls' leaves share a salt."""
    return H("verity/highutil/salt-key/v1", secret, u64(call))


# ------------------------------------------------------------------------------------------------ prover and verifier
class _RowView:
    """One row of a coin array, indexed as check_block indexes the call's full array (`c.dA[i:i + 1]`)."""

    def __init__(self, row: np.ndarray) -> None:
        self.row = row

    def __getitem__(self, _: slice) -> np.ndarray:
        return self.row


def _cheated(cheat: str) -> int | None:
    return int(cheat.split(":")[1]) if cheat.startswith("skip:") else None


def answer(job: dict) -> dict:
    """The prover's answer to one pick: the opened rows (recomputed from the tile's operands and coins, as committed), each
    with its salt, leaf digest and path, and A's and B's rows with their operand openings."""
    sh = shape(job["n"], job["k"], job["row"], job["m"])
    tl = v31.Shape(TILE, TILE, sh.k, sh.lp, sh.t, sh.row)
    seeds = v31.seeds_from(job["seeds"])
    i, j, t = job["i"], job["j"], job["t"]
    tm, tn = i // TILE, j // TILE
    ia, ja = range(tm * TILE, tm * TILE + TILE), range(tn * TILE, tn * TILE + TILE)
    A = np.fromfile(job["afile"], dtype=np.int8, count=TILE * sh.k, offset=(t * sh.m + tm * TILE) * sh.k).reshape(TILE, sh.k)
    B = np.fromfile(job["bfile"], dtype=np.int8, count=TILE * sh.k, offset=(t * sh.n + tn * TILE) * sh.k).reshape(TILE, sh.k)
    c = v31.Coins.rows(sh, seeds, ia, ja)
    f = v31.form(tl, A, B, c)
    skip = _cheated(job["cheat"])
    v = v31.values(tl, f, skip)
    O = v31.binding(tl, f, c, v.T_full) if skip is not None else v.O
    words, obuf = v31.words_buffer(tl, v.T), v31.o_buffer(O)
    tile = tm * (sh.n // TILE) + tn
    rows = {}
    for pos, path in zip(job["positions"], job["paths"]):
        row = v31.row_bytes(tl, words, obuf, pos - tile * sh.lpt)
        salt = v31.cref.prf_salt(job["ks"], pos)
        rows[pos] = (row, salt, v31.leaf_h(row, salt), tuple(path))
    return {"rows": rows, "A": A[i - tm * TILE], "B": B[j - tn * TILE], "A_opening": job["A_opening"],
            "B_opening": job["B_opening"]}


def verify_pick(job: dict, ans: dict) -> str | None:
    """The verifier's checks of one pick (None: passed): A's row i and B's row j against the registered operand roots of
    the call's block, then v31.py's check_block on the verifier's own seeds, those rows and the call's root."""
    sh = shape(job["n"], job["k"], job["row"], job["m"])
    i, j, s, t = job["i"], job["j"], job["s"], job["t"]
    for kind, x, rows in (("A", i, sh.m), ("B", j, sh.n)):
        op = ans[f"{kind}_opening"]
        why = operand_reason(kind, job["name"], t, rows, sh.k, job["operand_roots"][kind], x, op)
        if why or op[0] != ans[kind].tobytes():
            return why or f"the answer's {kind} row isn't its opened row"
    seeds = v31.seeds_from(job["verifier_seeds"])
    c = v31.Coins.rows(sh, seeds, range(i, i + 1), range(j, j + 1))
    one = v31.Coins(_RowView(c.dA), _RowView(c.dB), _RowView(c.U), _RowView(c.V), c.perm)
    want = set(positions(sh, Pick(0, 0, 0, i, j, s)))
    if set(ans["rows"]) != want:
        return f"the answer opens leaves {sorted(ans['rows'])}, not {sorted(want)}"
    return v31.check_block(sh, seeds, job["root"], i, j, s, {"rows": ans["rows"], "A": np.frombuffer(ans["A_opening"][0], np.int8),
                                                             "B": np.frombuffer(ans["B_opening"][0], np.int8)}, one)


def prove_and_verify(job: dict) -> tuple[int, str | None]:
    try:
        return job["index"], verify_pick(job, answer(job))
    except Exception as e:  # a failed answer is a failed pick, never a retry
        return job["index"], f"{type(e).__name__}: {e}"


# --------------------------------------------------------------------------------------------------------- the driver
class Driver:
    """hu.cu over its stdin and stdout."""

    def __init__(self, hu: str, cfg: Path, row: int, rmax: int, log: Path) -> None:
        self.p = subprocess.Popen([hu, str(cfg), str(row), str(LP), str(T), str(rmax)], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=open(log, "ab"), text=True, bufsize=1)
        self.ready = self._line("ready")

    def _line(self, head: str) -> list[str]:
        got = self.p.stdout.readline().split()
        if not got or got[0] != head:
            raise RuntimeError(f"hu answered {got} for {head} (rc {self.p.poll()}; its log has the rest)")
        return got

    def block(self, b: int, g0: int, steps: int, arm: str, cheat: str, seeds: Path, roots: Path) -> dict:
        self.p.stdin.write(f"block {b} {g0} {steps} {arm} {cheat} {seeds} {roots}\n")
        got = self._line("done")
        return {"gpu_ms": float(got[2]), "host_ms": float(got[3]), "flags": int(got[4]), "sm_mhz": int(got[5])}

    def paths(self, b: int, picks_file: Path, out: Path) -> None:
        self.p.stdin.write(f"paths {b} {picks_file} {out}\n")
        self._line("paths")

    def close(self) -> None:
        if self.p.poll() is None:
            self.p.stdin.write("quit\n")
            self.p.stdin.close()
            self.p.wait()


class RefDriver:
    """hu's commands on the CPU through v31.py's reference prover (``period.py run CFG ref ...``, and the tests): the same
    values, roots and paths at small shapes, so the period's coins, picks, answers and verifier run without a GPU."""

    def __init__(self, cfg: Path, row: int) -> None:
        self.cfg, self.row = cfg, row
        self.levels: dict[tuple[int, int], list] = {}
        self.ready = ["ready", "ref"]

    def block_of(self, step: int) -> int:
        return step % NB

    def block(self, b: int, g0: int, steps: int, arm: str, cheat: str, seeds: Path, roots: Path) -> dict:
        if arm == "plain":
            return {"gpu_ms": 0.0, "host_ms": 0.0, "flags": 0, "sm_mhz": 0}
        NL, rec = len(LAYERS), seeds.read_bytes()
        kind, _, at = cheat.partition("@")
        self.levels, out = {}, []
        for s in range(steps):
            for li, (_, n, k) in enumerate(LAYERS):
                r = rec[384 * (s * NL + li):384 * (s * NL + li + 1)]
                sh = shape(n, k, self.row)
                t = self.block_of(g0 + s)
                bad = kind if kind != "none" and (not at or int(at) == li) else None
                pv, _ = v31.prove(sh, np.asarray(operand(self.cfg, "A", li, t)), np.asarray(operand(self.cfg, "B", li, t)),
                                  v31.seeds_from(r[:320]), r[320:], 0, bad)
                self.levels[(s, li)] = pv.cm.levels
                out.append(pv.cm.root)
        roots.write_bytes(b"".join(out))
        return {"gpu_ms": 0.0, "host_ms": 0.0, "flags": 0, "sm_mhz": 0}

    def paths(self, b: int, picks_file: Path, out: Path) -> None:
        recs = np.frombuffer(picks_file.read_bytes(), dtype="<u4").reshape(-1, 3)
        with open(out, "wb") as f:
            for s, li, pos in recs.tolist():
                lv = self.levels[(s, li)]
                f.write(b"".join(lv[d][(pos >> d) ^ 1] for d in range(len(lv) - 1)))

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------------------------------------- the period
class Coins:
    """One challenger session a period: a5_coins' Local (development, not of record) or Gate (vy-challenger-1)."""

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
        return self.A5.Gate(self.how.removeprefix("gate:"), self.gate_key, self.peer, self.key_record, self.main)

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
        except Exception as e:
            return {"trusted": False, "error": f"{type(e).__name__}: {e}", "session": getattr(session, "session", None)}


@dataclass
class Plan:
    blocks: int
    steps: int    # a block's
    q: int        # proof blocks a period


class Period:
    def __init__(self, drv: Driver, coins: Coins | None, trees: dict[tuple[str, int, int], OperandTree], cfg: Path, row: int,
                 source: dict, secret: bytes, work: Path, pool: ProcessPoolExecutor) -> None:
        self.drv, self.coins, self.trees, self.cfg, self.row = drv, coins, trees, cfg, row
        self.source, self.secret, self.work, self.pool = source, secret, work, pool
        self.g = 0   # the next global step

    def registration(self, pid: str, plan: Plan) -> tuple[bytes, dict]:
        reg = {"period": pid, "m": M, "nb": NB, "layers": [list(x) for x in LAYERS], "lp": LP, "t": T, "row": self.row,
               "blocks": plan.blocks, "steps_per_block": plan.steps, "q": plan.q, "first_step": self.g,
               "W_reg": plan.blocks * plan.steps * step_macs(), "source": self.source,
               "operand_roots": {kind: [[self.trees[(kind, li, t)].root.hex() for t in range(NB)] for li in range(len(LAYERS))]
                                 for kind in ("A", "B")}}
        return H("verity/highutil/registration/v1", json.dumps(reg, sort_keys=True).encode()), reg

    def plain(self, pid: str, plan: Plan) -> dict:
        """The baseline: the same calls through cuBLASLt, back to back, no coins."""
        t0 = time.monotonic()
        blocks = []
        for b in range(plan.blocks):
            blocks.append(self.drv.block(b, self.g, plan.steps, "plain", "none", self.work / "none", self.work / "none"))
            self.g += plan.steps
        wall = time.monotonic() - t0
        return {"period": pid, "arm": "plain", "blocks": blocks, "steps": plan.blocks * plan.steps, "wall_s": wall,
                "macs": plan.blocks * plan.steps * step_macs()}

    def proto(self, pid: str, plan: Plan, cheat: str = "none") -> dict:
        NL = len(LAYERS)
        reg_digest, reg = self.registration(pid, plan)
        session = self.coins.session()
        t_s = time.monotonic()
        address, secret, _ = session.open(reg_digest.hex(), [["seeds", plan.blocks], ["check", plan.blocks]], 30, SCHEME)
        client = self.coins.connect(address, secret)
        t_open = time.monotonic() - t_s
        openings, blocks, all_picks, roots_of, seeds_of = [], [], [], {}, {}
        per_block_q = plan.q // plan.blocks
        prev = reg_digest
        val = {}
        t0 = time.monotonic()

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
                paths = []
                for _ in pos:
                    paths.append([raw[off + 64 * d:off + 64 * d + 64] for d in range(depth)])
                    off += 64 * depth
                all_picks.append((p, pos, paths))

        try:
            for b in range(plan.blocks):
                op = H("verity/highutil/operand-receipt/v1", reg_digest, u64(b), prev, u64(plan.steps * step_macs()))
                if b == 0:
                    got = [client.open("seeds", 0, op)]
                else:
                    got = client.open_many([("check", b - 1, val[b - 1]), ("seeds", b, op)])
                    checked(b - 1, got[0].coin)
                openings += got
                sc = got[-1]
                sd = call_seeds(sc.coin, pid, b, op, plan.steps)
                seeds_of[b] = sd
                g0 = self.g
                rec = b"".join(sd[s * NL + li] + salt_key(self.secret, (g0 + s) * NL + li) for s in range(plan.steps)
                               for li in range(NL))
                sf, rf = self.work / f"seeds-{b}.bin", self.work / f"roots-{b}.bin"
                sf.write_bytes(rec)
                r = self.drv.block(b, g0, plan.steps, "proto", cheat, sf, rf)
                self.g += plan.steps
                raw = rf.read_bytes()
                roots_of[b] = [raw[64 * x:64 * x + 64] for x in range(plan.steps * NL)]
                val[b] = prev = H("verity/highutil/value-receipt/v1", reg_digest, u64(b), raw)
                r["t_done_s"] = time.monotonic() - t0
                blocks.append(r)
            t_last = time.monotonic() - t0
            last = client.open("check", plan.blocks - 1, val[plan.blocks - 1])
            openings.append(last)
            checked(plan.blocks - 1, last.coin)
        finally:
            client.close()
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
                         "t": t, "call": call, "seeds": seeds, "ks": salt_key(self.secret, call),
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
                "verify_s": t_verify, "macs": macs, "credited_macs": macs if ok else 0, "accepted": bool(ok), "q": len(jobs),
                "eps_skip": eps_skip(len(jobs)), "delta": DELTA, "eta": ETA, "failures": [f"{all_picks[i][0]}: {w}" for i, w in fails[:8]],
                "failed": len(fails), "coins": ev, "verdict": line,
                "gaps_s": [blocks[b]["t_done_s"] - blocks[b - 1]["t_done_s"] - blocks[b]["host_ms"] / 1e3
                           for b in range(1, len(blocks))]}


def cmd_run(a: argparse.Namespace) -> int:
    cfg, out = Path(a.cfg), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prep = json.loads((cfg / "prep.json").read_text())
    configure(prep)
    row = prep["row"]
    plan = Plan(a.blocks, a.steps, a.q)
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
            drv = RefDriver(cfg, row) if a.hu == "ref" else Driver(a.hu, cfg, row, plan.steps, out / "hu.log")
            print("hu " + " ".join(drv.ready), flush=True)
            per = Period(drv, coins, trees, cfg, row, source, secret, work, pool)
            # warm-up: one plain block, so the clocks and caches settle before anything is timed
            drv.block(0, 0, min(plan.steps, 50), "plain", "none", work / "none", work / "none")
            arms = [x for x in ("proto", "plain") if a.arms in ("both", x)]
            schedule = [(x, "none") for x in arms] * a.periods + ([("proto", a.negative)] if a.negative else [])
            for x, (arm, cheat) in enumerate(schedule):
                pid = f"{a.tag}-{x:02d}-{arm}" + ("" if cheat == "none" else "-" + cheat.replace(":", "").replace("@", "l"))
                r = per.proto(pid, plan, cheat) if arm == "proto" else per.plain(pid, plan)
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
    summary = summarize(results, plan, source)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    neg = [r for r in results if r["arm"] == "proto" and r["cheat"] != "none"]
    honest = [r for r in results if r["arm"] == "proto" and r["cheat"] == "none"]
    if a.arms == "plain":
        return 0
    return 0 if honest and all(r["accepted"] for r in honest) and all(not r["accepted"] for r in neg) else 1


def cmd_diag(a: argparse.Namespace) -> int:
    """One protocol step on the GPU (a block of the model other than the first) with seeds and salt keys of its own (no
    coins), then two tiles of each layer's call against v31.py's reference: the A and B rows the GPU read against the
    files', its words and O against the tile's values, and its leaves against the reference's under the call's salt key."""
    cfg, out = Path(a.cfg), Path(a.out)
    prep = json.loads((cfg / "prep.json").read_text())
    configure(prep)
    row, NL = prep["row"], len(LAYERS)
    rng = np.random.default_rng(a.seed)
    g0 = NB - 1 if NB > 1 else 0
    work = Path(tempfile.mkdtemp(prefix="highutil-diag-", dir="/dev/shm" if Path("/dev/shm").is_dir() else None))
    drv = Driver(a.hu, cfg, row, 1, out / "hu-diag.log")
    rec = [os.urandom(384) for _ in range(NL)]
    (work / "seeds").write_bytes(b"".join(rec))
    drv.block(0, g0, 1, "proto", "none", work / "seeds", work / "roots")
    rep = []
    for li, (name, n, k) in enumerate(LAYERS):
        sh = shape(n, k, row)
        tl = v31.Shape(TILE, TILE, k, LP, T, row)
        for tile in (0, int(rng.integers(sh.tiles))):
            d = work / f"{name}-{tile}"
            drv.p.stdin.write(f"dump {li} {tile} {d}\n")
            got = drv._line("dumped")
            wraps = {"X_per_row": int(got[6]) / sh.m, "W_per_row": int(got[7]) / sh.n}
            tm, tn = divmod(tile, n // TILE)
            ia, ja = range(tm * TILE, tm * TILE + TILE), range(tn * TILE, tn * TILE + TILE)
            A = np.asarray(operand(cfg, "A", li, g0 % NB)[tm * TILE:(tm + 1) * TILE])
            B = np.asarray(operand(cfg, "B", li, g0 % NB)[tn * TILE:(tn + 1) * TILE])
            seeds, ks = v31.seeds_from(rec[li][:320]), rec[li][320:]
            c = v31.Coins.rows(sh, seeds, ia, ja)
            f = v31.form(tl, A, B, c)
            v = v31.values(tl, f)
            words, obuf = v31.words_buffer(tl, v.T), v31.o_buffer(v.O)
            dom = sh.domain()
            g = (d / "leaves.bin").read_bytes()
            bad = sum(row_leaf(dom, tile * sh.lpt + j, v31.leaf_h(v31.row_bytes(tl, words, obuf, j),
                                                                  v31.cref.prf_salt(ks, tile * sh.lpt + j)), V2)
                      != g[64 * j:64 * j + 64] for j in range(sh.lpt))
            rep.append({"layer": name, "block": g0 % NB, "tile": tile, "A_equal": (d / "A.bin").read_bytes() == A.tobytes(),
                        "B_equal": (d / "B.bin").read_bytes() == B.tobytes(), "words_equal": (d / "words.bin").read_bytes() == words, "O_equal": (d / "O.bin").read_bytes() == obuf,
                        "leaves": sh.lpt, "leaves_bad": bad, "wraps": wraps})
            print(json.dumps(rep[-1]), flush=True)
    drv.close()
    shutil.rmtree(work, ignore_errors=True)
    ok = all(r["A_equal"] and r["B_equal"] and r["words_equal"] and r["O_equal"] and not r["leaves_bad"] for r in rep)
    (out / "diag.json").write_text(json.dumps({"exact": ok, "tiles": rep}, indent=1))
    print(f"diag: {'exact' if ok else 'DIFFERS'}", flush=True)
    return 0 if ok else 1


def summarize(results: list[dict], plan: Plan, source: dict | None = None) -> dict:
    honest = [r for r in results if r["arm"] == "proto" and r["cheat"] == "none"]
    plain = [r for r in results if r["arm"] == "plain"]
    neg = [r for r in results if r["arm"] == "proto" and r["cheat"] != "none"]
    med = lambda v: float(np.median(v)) if v else None  # noqa: E731
    per_step = lambda r: r["wall_s"] / r["steps"]  # noqa: E731
    gpu_step = lambda r: sum(b["gpu_ms"] for b in r["blocks"]) / 1e3 / r["steps"]  # noqa: E731
    credited = sum(r["credited_macs"] for r in honest)
    wall = sum(r["wall_s"] for r in honest)
    mhz = [b["sm_mhz"] for r in results for b in r["blocks"] if b["sm_mhz"]]
    out = {
        "workload": {"m": M, "nb": NB, "layers": [list(x) for x in LAYERS], "step_macs": step_macs(), "lp": LP, "t": T,
                     "source": source},
        "plan": plan.__dict__, "periods": len(honest),
        "accepted": sum(r["accepted"] for r in honest),
        "credited_macs": credited, "wall_s": wall,
        "peak_macs_s": {"sku_2617": peak_macs(SKU_MHZ), "node_2430": peak_macs(NODE_MHZ)},
        "certified_share": {"sku_2617": credited / (peak_macs(SKU_MHZ) * wall) if wall else None,
                            "node_2430": credited / (peak_macs(NODE_MHZ) * wall) if wall else None},
        "plain_share_of_peak": {"sku_2617": med([r["macs"] / r["wall_s"] / peak_macs(SKU_MHZ) for r in plain])},
        "proto_s_per_step": med([per_step(r) for r in honest]), "plain_s_per_step": med([per_step(r) for r in plain]),
        "gamma_wall": None, "gamma_gpu": None,
        "gaps_s": med([g for r in honest for g in r["gaps_s"]]),
        "session_open_s": med([r["session_open_s"] for r in honest]),
        "q": med([r["q"] for r in honest]), "eps_skip": med([r["eps_skip"] for r in honest]), "delta": DELTA,
        "sm_mhz_median": med(mhz),
        "verdicts": [r["verdict"] for r in honest + neg],
        "negative_refused": [not r["accepted"] for r in neg],
    }
    if out["proto_s_per_step"] and out["plain_s_per_step"]:
        out["gamma_wall"] = out["proto_s_per_step"] / out["plain_s_per_step"]
        plain_gpu = med([gpu_step(r) for r in plain])
        out["gamma_gpu"] = med([gpu_step(r) for r in honest]) / plain_gpu if plain_gpu else None
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("cfg")
    p.add_argument("--row", type=int, default=32768)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--checkpoint", help=f"{CHECKPOINT['repo']}'s snapshot directory at {CHECKPOINT['revision'][:10]} (B; "
                                        "capture.py writes A)")
    g.add_argument("--seeded", action="store_true", help="seeded A and B")
    g.add_argument("--small", action="store_true", help="two small layers at M 256, two blocks, seeded (tests)")
    p.add_argument("--nb", type=int, default=32, help="transformer blocks")
    p.add_argument("--seed", type=int, default=1)
    p.set_defaults(fn=cmd_prep)
    p = sub.add_parser("run")
    p.add_argument("cfg")
    p.add_argument("hu")
    p.add_argument("out")
    p.add_argument("--coins", default="local", help="local (development, not of record) or gate:USER@HOST")
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
    p.add_argument("--tag", default="hu")
    p.add_argument("--workers", type=int, default=0)
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("diag")
    p.add_argument("cfg")
    p.add_argument("hu")
    p.add_argument("out")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_diag)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
