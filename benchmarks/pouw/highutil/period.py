"""HIGHUTIL: an 8B model's linear layers at full batch on one GPU, each call credited through Dither's no-fold receipt
(verity/core/protocols/pouw/accel/PROTOCOL.md v3.2) on V31-GPU's prover, with live coins, and the certified share of the
GPU's tensor peak that the periods establish.

**The workload.** Llama-3-8B's four linear layers (QKV 6144 x 4096, O 4096 x 4096, gate-and-up 28672 x 4096, down 4096 x
14336) at M = 16,384, s8 x s8 with int32 accumulation, called back to back (a step is one call of each). B is the layer's
weights, seeded (`weights`, v31.py's ``llm`` rows) and registered before any coin. A is a seeded activation, fresh each
call: call c's A is `act_rows` (hu.cu's act_value, generated on the GPU beside the other layers' GEMMs), keyed by a key
the registration commits to and the verifier is shown once the period's roots are in. Chaining the layers would
requantize each O into the next A, a full non-GEMM pass, so the activations are seeded instead.

**A period** (`Period.run`): the registration (the layers' weight roots, the activation key's commitment, the period's
shape, W_reg) is the root of one challenger session with families ``seeds`` and ``check``, a coin each a block. Then
its sampling blocks run in one sequence. Block b's operand-root receipt is H(registration, b, block b-1's value-root
receipt), and its seeds coin opens after it. Each call's five v31 seeds (dA, dB, U, V, P) are
``derive(coin, "verity/pouw-accel/seeds/v1", ctx).stream(call)``. The GPU runs the block's calls, each committed under a
frame-v3-sha512 root with a fresh salt key, and block b's value-root receipt is H(registration, b, its roots). Its check
coin opens after that receipt, with block b+1's seeds coin in the same round trip. The coin draws Q W_b / W_reg proof
blocks of b (`picks`): a call by its MACs, a cell uniformly, a segment by its length, so each MAC is equally likely.
Each pick's tree paths come off the host copy of its call's tree before the next block overwrites it.

**The verifier** (`verify_pick`), after the period: the prover's answer (`answer`: the opened rows recomputed from the
tile's operands and coins, their salts and paths, and B's row with its weight opening) against v31.py's
``check_block``, on the verifier's own seeds from the coins, its own A row from the revealed key, and the call's root.
Credit is all-or-nothing: the period's MACs if every pick passes, else none. ε_skip = ln(1/δ)/(ηQ) at δ = 1%.

**Deviations from v3.2**, each named in the report: the picks are a fixed Q W_b / W_reg a block, not ``poisson:Q``;
the coins are 64-byte session coins expanded here, not the challenger's streamed seeds; no certifier signs the receipts
(the challenger's release log records what each coin opened after); and the checks run in the clear, the stand-in for
the Definition and C-Flock --zk.

    period.py prep CFG [--row 32768] [--seed S]          weights, params and shapes for hu
    period.py run CFG HU OUT [--coins local|gate:USER@HOST ...] [--plan PLAN]
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

import numpy as np

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
LP, T = 256, 3
TILE = v31.TILE
SCHEME = "verity/highutil/period/v1"
WEIGHTS_TAG = "verity/highutil/weights/v1"
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


# ------------------------------------------------------------------------------------------------------ activations
def fmix64(z: np.ndarray) -> np.ndarray:
    z = z ^ (z >> np.uint64(30))
    z = z * np.uint64(0xBF58476D1CE4E5B9)
    z = z ^ (z >> np.uint64(27))
    z = z * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def act_rows(key: int, call: int, rows: range, k: int) -> np.ndarray:
    """Rows `rows` of call `call`'s A (K columns), hu.cu's act_value entry by entry."""
    with np.errstate(over="ignore"):
        e = np.arange(rows.start, rows.stop, dtype=np.uint64)[:, None] * np.uint64(k) + np.arange(k, dtype=np.uint64)[None, :]
        h = fmix64(np.uint64(key) ^ (((np.uint64(call) << np.uint64(32)) | e) * np.uint64(0x9E3779B97F4A7C15)))
    s = sum(((h >> np.uint64(8 * b)) & np.uint64(255)).astype(np.int64) for b in range(4)) - 510
    out = np.maximum(((h >> np.uint64(40)) & np.uint64(255)).astype(np.uint8).view(np.int8).astype(np.int64), -127)
    return np.where(((h >> np.uint64(32)) & np.uint64(1023)) == 0, out, s >> 4).astype(np.int8)


# ---------------------------------------------------------------------------------------------------------- weights
def weights(n: int, k: int, seed: int) -> np.ndarray:
    """v31.py's ``llm`` weights: Gaussian rows quantized to +-127 by their absolute maximum."""
    b = np.random.default_rng(seed).standard_normal(size=(n, k), dtype=np.float32)
    return np.clip(np.rint(127 * b / np.abs(b).max(axis=1, keepdims=True)), -127, 127).astype(np.int8)


def weight_domain(name: str, n: int, k: int) -> CommitmentDomain:
    return CommitmentDomain(identity_digest_sha512(WEIGHTS_TAG, {"layer": name, "n": n, "k": k}), -1, RangeIndexedDomain(0, n),
                            hash="sha512")


@dataclass
class WeightTree:
    """A layer's registered weights: B's rows as hm96 rows (salts prf_salt(kw, row)) in one frame-v3-sha512 tree."""
    name: str
    n: int
    k: int
    kw: bytes
    levels: tuple

    @property
    def root(self) -> bytes:
        return self.levels[-1][0]

    @staticmethod
    def build(name: str, B: np.ndarray, kw: bytes) -> WeightTree:
        n, k = B.shape
        dom = weight_domain(name, n, k)
        hs = [v31.leaf_h(B[p].tobytes(), v31.cref.prf_salt(kw, p)) for p in range(n)]
        return WeightTree(name, n, k, kw, tree_levels(dom, [row_leaf(dom, p, h, V2) for p, h in enumerate(hs)]))

    def opening(self, B: np.ndarray, j: int) -> tuple[bytes, bytes, tuple[bytes, ...]]:
        """(row, salt, path) of B's row j."""
        return (B[j].tobytes(), v31.cref.prf_salt(self.kw, j),
                tuple(self.levels[d][(j >> d) ^ 1] for d in range(len(self.levels) - 1)))


def weight_reason(name: str, n: int, k: int, root: bytes, j: int, row: bytes, salt: bytes, path: tuple) -> str | None:
    h = v31.leaf_h(row, salt)
    if len(row) != k or not verify_opening(weight_domain(name, n, k), Commitment(root, n), Opening(j, h, path), V2,
                                           VerificationLimits()):
        return f"B row {j} doesn't open against the registered weight root"
    return None


def configure(prep: dict) -> None:
    """The workload a config was prepared for (its layers and M; the 8B shapes unless a test made it smaller)."""
    global LAYERS, M
    LAYERS, M = tuple(tuple(x) for x in prep["layers"]), prep["m"]


def cmd_prep(a: argparse.Namespace) -> int:
    d = Path(a.cfg)
    d.mkdir(parents=True, exist_ok=True)
    if a.small:
        configure({"layers": [["a", 256, 768], ["b", 128, 1536]], "m": 256})
    lines = []
    for li, (name, n, k) in enumerate(LAYERS):
        sh = shape(n, k, a.row)
        weights(n, k, a.seed * 16 + li).tofile(d / f"B-{name}.bin")
        (d / f"params-{name}.bin").write_bytes(v31.params_file(sh, os.urandom(64)))
        lines.append(f"{name} {M} {n} {k}")
    (d / "shapes.txt").write_text("\n".join(lines) + "\n")
    (d / "prep.json").write_text(json.dumps({"seed": a.seed, "row": a.row, "layers": LAYERS, "m": M}))
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
    with its salt, leaf digest and path, and A's and B's rows with B's weight opening."""
    sh = shape(job["n"], job["k"], job["row"], job["m"])
    tl = v31.Shape(TILE, TILE, sh.k, sh.lp, sh.t, sh.row)
    seeds = v31.seeds_from(job["seeds"])
    i, j = job["i"], job["j"]
    tm, tn = i // TILE, j // TILE
    ia, ja = range(tm * TILE, tm * TILE + TILE), range(tn * TILE, tn * TILE + TILE)
    A = act_rows(job["actkey"], job["call"], ia, sh.k)
    B = np.fromfile(job["bfile"], dtype=np.int8, count=TILE * sh.k, offset=tn * TILE * sh.k).reshape(TILE, sh.k)
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
    return {"rows": rows, "A": A[i - tm * TILE], "B": B[j - tn * TILE], "B_opening": job["B_opening"]}


def verify_pick(job: dict, ans: dict) -> str | None:
    """The verifier's checks of one pick (None: passed): B's row against the registered weight root, then v31.py's
    check_block on the verifier's own seeds, its own A row and the call's root."""
    sh = shape(job["n"], job["k"], job["row"], job["m"])
    i, j, s = job["i"], job["j"], job["s"]
    brow, bsalt, bpath = ans["B_opening"]
    why = weight_reason(job["name"], sh.n, sh.k, job["weight_root"], j, brow, bsalt, bpath)
    if why or brow != ans["B"].tobytes():
        return why or "the answer's B row isn't its opened row"
    seeds = v31.seeds_from(job["verifier_seeds"])
    c = v31.Coins.rows(sh, seeds, range(i, i + 1), range(j, j + 1))
    one = v31.Coins(_RowView(c.dA), _RowView(c.dB), _RowView(c.U), _RowView(c.V), c.perm)
    A_row = act_rows(job["verifier_actkey"], job["call"], range(i, i + 1), sh.k)[0]
    want = set(positions(sh, Pick(0, 0, 0, i, j, s)))
    if set(ans["rows"]) != want:
        return f"the answer opens leaves {sorted(ans['rows'])}, not {sorted(want)}"
    return v31.check_block(sh, seeds, job["root"], i, j, s, {"rows": ans["rows"], "A": A_row, "B": np.frombuffer(brow, np.int8)},
                           one)


def prove_and_verify(job: dict) -> tuple[int, str | None]:
    try:
        return job["index"], verify_pick(job, answer(job))
    except Exception as e:  # a failed answer is a failed pick, never a retry
        return job["index"], f"{type(e).__name__}: {e}"


# --------------------------------------------------------------------------------------------------------- the driver
class Driver:
    """hu.cu over its stdin and stdout."""

    def __init__(self, hu: str, cfg: Path, row: int, rmax: int, actkey: int, log: Path) -> None:
        self.p = subprocess.Popen([hu, str(cfg), str(row), str(LP), str(T), str(rmax), f"{actkey:016x}"], stdin=subprocess.PIPE,
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

    def __init__(self, cfg: Path, row: int, actkey: int) -> None:
        self.cfg, self.row, self.actkey = cfg, row, actkey
        self.B = [np.fromfile(cfg / f"B-{name}.bin", dtype=np.int8).reshape(n, k) for name, n, k in LAYERS]
        self.levels: dict[tuple[int, int], list] = {}
        self.ready = ["ready", "ref"]

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
                A = act_rows(self.actkey, (g0 + s) * NL + li, range(M), k)
                bad = kind if kind != "none" and (not at or int(at) == li) else None
                pv, _ = v31.prove(sh, A, self.B[li], v31.seeds_from(r[:320]), r[320:], 0, bad)
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
    def __init__(self, drv: Driver, coins: Coins | None, trees: list[WeightTree], cfg: Path, row: int, actkey: int,
                 actsalt: bytes, secret: bytes, work: Path, pool: ProcessPoolExecutor) -> None:
        self.drv, self.coins, self.trees, self.cfg, self.row = drv, coins, trees, cfg, row
        self.actkey, self.actsalt, self.secret, self.work, self.pool = actkey, actsalt, secret, work, pool
        self.g = 0   # the next global step

    def registration(self, pid: str, plan: Plan) -> tuple[bytes, dict]:
        reg = {"period": pid, "m": M, "layers": [list(x) for x in LAYERS], "lp": LP, "t": T, "row": self.row,
               "blocks": plan.blocks, "steps_per_block": plan.steps, "q": plan.q, "first_step": self.g,
               "W_reg": plan.blocks * plan.steps * step_macs(),
               "weight_roots": [t.root.hex() for t in self.trees],
               "activations": H("verity/highutil/activations/v1", self.actsalt, u64(self.actkey)).hex()}
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
        # the verifier: its own seeds from the coins it was given, the revealed activation key against the registration
        vkey_ok = H("verity/highutil/activations/v1", self.actsalt, u64(self.actkey)).hex() == reg["activations"]
        jobs = []
        Bmm = [np.memmap(self.cfg / f"B-{name}.bin", dtype=np.int8, mode="r", shape=(n, k)) for name, n, k in LAYERS]
        cheat_layer = int(cheat.split("@")[1]) if "@" in cheat else None
        for idx, (p, pos, paths) in enumerate(all_picks):
            name, n, k = LAYERS[p.layer]
            g0 = reg["first_step"] + p.block * plan.steps
            call = (g0 + p.step) * NL + p.layer
            seeds = seeds_of[p.block][p.step * NL + p.layer]
            tree = self.trees[p.layer]
            jobs.append({"index": idx, "name": name, "m": M, "n": n, "k": k, "row": self.row, "i": p.i, "j": p.j, "s": p.s,
                         "call": call, "seeds": seeds, "ks": salt_key(self.secret, call), "actkey": self.actkey,
                         "bfile": str(self.cfg / f"B-{name}.bin"), "cheat": cheat.split("@")[0]
                         if cheat_layer in (None, p.layer) else "none",
                         "positions": pos, "paths": paths, "B_opening": tree.opening(Bmm[p.layer], p.j),
                         "weight_root": tree.root, "verifier_seeds": seeds, "verifier_actkey": self.actkey,
                         "root": roots_of[p.block][p.step * NL + p.layer]})
        t_v = time.monotonic()
        fails = [(i, why) for i, why in self.pool.map(prove_and_verify, jobs, chunksize=4) if why]
        t_verify = time.monotonic() - t_v
        flags = 0
        for r in blocks:
            flags |= r["flags"]
        ok = not fails and vkey_ok and not flags and ev.get("trusted") and not ev.get("bad_openings")
        macs = plan.blocks * plan.steps * step_macs()
        first = all_picks[fails[0][0]][0] if fails else None
        line = (f"{'ACCEPT' if ok else 'REFUSE'} period {pid}: {plan.blocks} blocks x {plan.steps} steps "
                f"({plan.blocks * plan.steps * NL} calls, {macs:.4g} MACs), {len(jobs)} proof blocks opened, "
                f"{len(jobs) - len(fails)} passed, {len(fails)} failed"
                + (f"; first: block {first.block} step {first.step} {LAYERS[first.layer][0]} ({first.i}, {first.j}, s {first.s}): "
                   f"{fails[0][1]}" if fails else "")
                + ("" if vkey_ok else "; the revealed activation key isn't the registered one")
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
    secret, actsalt = os.urandom(64), os.urandom(64)
    actkey = int.from_bytes(os.urandom(8), "big")
    t = time.monotonic()
    trees = []
    for name, n, k in LAYERS:
        trees.append(WeightTree.build(name, np.fromfile(cfg / f"B-{name}.bin", dtype=np.int8).reshape(n, k),
                                      hashlib.sha512(b"verity/highutil/weight-salts/v1\0" + secret + name.encode()).digest()))
    print(f"weight roots in {time.monotonic() - t:.1f} s: " + " ".join(f"{x.name} {x.root.hex()[:16]}" for x in trees), flush=True)
    work = Path(tempfile.mkdtemp(prefix="highutil-", dir="/dev/shm" if Path("/dev/shm").is_dir() else None))
    coins = Coins(a.coins, Path(a.main), a.peer, a.challenger_key, a.gate_key)
    drv = RefDriver(cfg, row, actkey) if a.hu == "ref" else Driver(a.hu, cfg, row, plan.steps, actkey, out / "hu.log")
    print("hu " + " ".join(drv.ready), flush=True)
    results = []
    workers = a.workers or max(1, len(os.sched_getaffinity(0)) - 4)
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            per = Period(drv, coins, trees, cfg, row, actkey, actsalt, secret, work, pool)
            # warm-up: one plain block, so the clocks and caches settle before anything is timed
            drv.block(0, 0, min(plan.steps, 50), "plain", "none", work / "none", work / "none")
            schedule = [("proto", "none"), ("plain", "none")] * a.periods + ([("proto", a.negative)] if a.negative else [])
            for x, (arm, cheat) in enumerate(schedule):
                pid = f"{a.tag}-{x:02d}-{arm}" + ("" if cheat == "none" else "-" + cheat.replace(":", "").replace("@", "l"))
                r = per.proto(pid, plan, cheat) if arm == "proto" else per.plain(pid, plan)
                results.append(r)
                with open(out / "periods.jsonl", "a") as f:
                    f.write(json.dumps(r, default=str) + "\n")
                print(r.get("verdict") or f"plain period {pid}: {r['steps']} steps in {r['wall_s']:.3f} s", flush=True)
    finally:
        drv.close()
        for p in work.glob("challenger-*"):
            shutil.copytree(p, out / "challenger" / p.name.removeprefix("challenger-"), dirs_exist_ok=True)
        shutil.rmtree(work, ignore_errors=True)
    summary = summarize(results, plan)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    neg = [r for r in results if r["arm"] == "proto" and r["cheat"] != "none"]
    honest = [r for r in results if r["arm"] == "proto" and r["cheat"] == "none"]
    return 0 if honest and all(r["accepted"] for r in honest) and all(not r["accepted"] for r in neg) else 1


def cmd_diag(a: argparse.Namespace) -> int:
    """One protocol step on the GPU with seeds and salt keys of its own (no coins), then two tiles of each layer's call
    against v31.py's reference: the GPU's A (regenerated) against `act_rows`, its words and O against the tile's values,
    and its leaves against the reference's leaves under the call's salt key."""
    cfg, out = Path(a.cfg), Path(a.out)
    prep = json.loads((cfg / "prep.json").read_text())
    configure(prep)
    row, NL = prep["row"], len(LAYERS)
    actkey = int.from_bytes(os.urandom(8), "big")
    work = Path(tempfile.mkdtemp(prefix="highutil-diag-", dir="/dev/shm" if Path("/dev/shm").is_dir() else None))
    drv = Driver(a.hu, cfg, row, 1, actkey, out / "hu-diag.log")
    rec = [os.urandom(384) for _ in range(NL)]
    (work / "seeds").write_bytes(b"".join(rec))
    drv.block(0, 0, 1, "proto", "none", work / "seeds", work / "roots")
    rng = np.random.default_rng(a.seed)
    rep = []
    for li, (name, n, k) in enumerate(LAYERS):
        sh = shape(n, k, row)
        tl = v31.Shape(TILE, TILE, k, LP, T, row)
        B = np.fromfile(cfg / f"B-{name}.bin", dtype=np.int8).reshape(n, k)
        for tile in (0, int(rng.integers(sh.tiles))):
            d = work / f"{name}-{tile}"
            drv.p.stdin.write(f"dump {li} {tile} {d}\n")
            drv._line("dumped")
            tm, tn = divmod(tile, n // TILE)
            ia, ja = range(tm * TILE, tm * TILE + TILE), range(tn * TILE, tn * TILE + TILE)
            A = act_rows(actkey, li, ia, k)
            seeds, ks = v31.seeds_from(rec[li][:320]), rec[li][320:]
            c = v31.Coins.rows(sh, seeds, ia, ja)
            f = v31.form(tl, A, B[tn * TILE:(tn + 1) * TILE], c)
            v = v31.values(tl, f)
            words, obuf = v31.words_buffer(tl, v.T), v31.o_buffer(v.O)
            dom = sh.domain()
            g = (d / "leaves.bin").read_bytes()
            bad = sum(row_leaf(dom, tile * sh.lpt + j, v31.leaf_h(v31.row_bytes(tl, words, obuf, j),
                                                                  v31.cref.prf_salt(ks, tile * sh.lpt + j)), V2)
                      != g[64 * j:64 * j + 64] for j in range(sh.lpt))
            rep.append({"layer": name, "tile": tile, "A_equal": (d / "A.bin").read_bytes() == A.tobytes(),
                        "words_equal": (d / "words.bin").read_bytes() == words, "O_equal": (d / "O.bin").read_bytes() == obuf,
                        "leaves": sh.lpt, "leaves_bad": bad})
            print(json.dumps(rep[-1]), flush=True)
    drv.close()
    shutil.rmtree(work, ignore_errors=True)
    ok = all(r["A_equal"] and r["words_equal"] and r["O_equal"] and not r["leaves_bad"] for r in rep)
    (out / "diag.json").write_text(json.dumps({"exact": ok, "tiles": rep}, indent=1))
    print(f"diag: {'exact' if ok else 'DIFFERS'}", flush=True)
    return 0 if ok else 1


def summarize(results: list[dict], plan: Plan) -> dict:
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
        "workload": {"m": M, "layers": [list(x) for x in LAYERS], "step_macs": step_macs(), "lp": LP, "t": T},
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
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--small", action="store_true", help="two small layers at M 256 (tests)")
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
