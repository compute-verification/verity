#!/usr/bin/env python3
"""Pearl-C4's per-tile check on the GPU, bit for bit: #580's `PearlC4.tile_cap` under `pearl_c_work.check_opened`'s tile
rule (F1′ with κ inside, F2 at c_L = 2.0, R1, the 1/400 reject), with its integer-heavy parts ported.

  forming      fp4_tc_torch.Engine.form: the noise atom per element and the block cast (exact integer align-add, IEEE
               single-rounded FP32 multiply and divide)
  salt_dead    fp4_tc_torch.Engine.salt_dead: the witness's three draws per element, on the same ops
  word_replay  fp4_replay.cu: each word's chain from +0, identity atoms and sub-grid blocks (CUDA; or its CPU build,
               the same functions under g++)
  the rest     the reference's own code, unchanged and on the CPU: stats (FP32 FMA), lines and bases, D-SB, D-NF,
               F1′ and R1, F2, the D-24 windows, the debit's assembly and the cap

The enforced outcome of a tile (`judge`): D-SB's reject while forming; the credited rows, those passing D-NF whose own
F1′ per word is within R1's bound (as an honest prover declares them, `r1_rows`, and `check_opened` accepts); the
debit's items over them (`Debit.to_json`), the cap, debit / cap, and the reject (debit > cap·(1 + R1_TOLERANCE)).

The reference is #580's tree, named by --rule-tree or PEARL_C4_RULE_TREE (its `protocols/pouw` and
`packages/verity/src` go first on sys.path, so import this module before anything imports verity).

    python fp4_tile_gpu.py gate --rule-tree TREE --out DIR --tiles-per-family 1000 --workers 32   (CPU: the port vs the reference)
    python fp4_tile_gpu.py gpu-check --rule-tree TREE --out DIR                                      (GPU: CUDA vs the CPU)
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE / "fp4_replay.cu"
F = P = W = None                                    # the reference's pearl_c4, pearl_kw and pearl_c_work (`use_tree`)


def use_tree(tree: str | Path | None = None) -> Path:
    """Put #580's tree first on sys.path and import its scheme modules."""
    global F, P, W
    root = Path(tree or os.environ.get("PEARL_C4_RULE_TREE", ""))
    if not (root / "protocols/pouw/verity_pouw/schemes/pearl_c4.py").is_file():
        raise SystemExit(f"{root}: not a tree with verity_pouw/schemes/pearl_c4.py (--rule-tree or PEARL_C4_RULE_TREE)")
    for p in (root / "packages/verity/src", root / "protocols/pouw"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from verity_pouw.schemes import pearl_c4, pearl_c_work, pearl_kw
    if not Path(pearl_c4.__file__).resolve().is_relative_to(root.resolve()):
        raise SystemExit(f"verity_pouw was imported from {pearl_c4.__file__}, not {root}")
    F, P, W = pearl_c4, pearl_kw, pearl_c_work
    return root


# ---- word_replay: the C / CUDA build -----------------------------------------------------------------------------

class Replay:
    """`fp4_replay.cu` built for `kind` ("cuda" or "cpu"), cached by the source's sha256."""

    def __init__(self, kind: str = "cuda", cache: str | Path | None = None):
        src = SRC.read_bytes()
        tag = hashlib.sha256(src + kind.encode()).hexdigest()[:16]
        cache = Path(cache or os.environ.get("FP4_REPLAY_CACHE") or Path.home() / ".cache/fp4_replay")
        lib = cache / f"fp4_replay-{kind}-{tag}.so"
        if not lib.exists():
            cache.mkdir(parents=True, exist_ok=True)
            tmp = lib.with_suffix(f".{os.getpid()}.tmp")
            if kind == "cuda":
                nvcc = shutil.which("nvcc") or str(Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "bin/nvcc")
                cmd = [nvcc, "-O3", "-arch=sm_120", "-shared", "-Xcompiler", "-fPIC", "-o", str(tmp), str(SRC)]
            else:
                cmd = ["g++", "-O2", "-fopenmp", "-x", "c++", "-shared", "-fPIC", "-o", str(tmp), str(SRC)]
            subprocess.run(cmd, check=True)
            tmp.replace(lib)
        self.kind, self.path = kind, lib
        self.lib = ctypes.CDLL(str(lib))
        u8, i32 = np.ctypeslib.ndpointer(np.uint8, flags="C"), np.ctypeslib.ndpointer(np.int32, flags="C")
        self.lib.fp4_replay.argtypes = [u8, u8, ctypes.c_int, u8, u8, ctypes.c_int, ctypes.c_int, i32, i32, i32]
        self.lib.fp4_replay.restype = ctypes.c_int

    def __call__(self, a_codes, a_scales, b_codes, b_scales) -> tuple[np.ndarray, np.ndarray]:
        """(identity atoms, sub-grid blocks), each [rows of A, rows of B], of every word; raises where the reference's
        chain does (a firing out of range, a NaN scale)."""
        a_codes, b_codes = (np.ascontiguousarray(v, np.uint8) for v in (a_codes, b_codes))
        a_scales, b_scales = (np.ascontiguousarray(v, np.uint8) for v in (a_scales, b_scales))
        (r, k), c = a_codes.shape, b_codes.shape[0]
        if k % F.ATOM or b_codes.shape[1] != k or a_scales.shape != (r, k // 16) or b_scales.shape != (c, k // 16):
            raise ValueError("operands of k = a multiple of 64, one scale per 16")
        ident, sub, err = (np.zeros((r, c), np.int32) for _ in range(3))
        rc = self.lib.fp4_replay(a_codes, a_scales, r, b_codes, b_scales, c, k, ident, sub, err)
        if rc:
            raise RuntimeError(f"fp4_replay ({self.kind}): CUDA error {rc}")
        if err.any():
            from verity.errors import InvalidArtifact
            raise InvalidArtifact(f"word_replay: {int((err != 0).sum())} words' chains leave the model (codes {sorted(set(err.flat) - {0})})")
        return ident, sub


# ---- forming ------------------------------------------------------------------------------------------------------

_ENGINES: dict = {}


def engine(device: str = "cpu"):
    if device not in _ENGINES:
        sys.path.insert(0, str(HERE))
        from fp4_tc_torch import Engine
        _ENGINES[device] = Engine(device)
    return _ENGINES[device]


def form(xs: list[list[int]], e_rows: list[list[int]], fb, device: str = "cpu", chunk: int = 64):
    """`pearl_c4.form` (NVFP4) with the atom and the cast on `Engine`: the reference's stats per row, then codes and
    scales for each chunk of rows."""
    import torch
    eng = engine(device)
    st = [F.stats(F.NVFP4, x) for x in xs]
    f_codes = torch.tensor(fb, dtype=torch.int64, device=eng.dev)
    codes, scales = [], []
    for r0 in range(0, len(xs), chunk):
        x = torch.tensor(xs[r0:r0 + chunk], dtype=torch.int64, device=eng.dev)
        alpha = torch.tensor([s[2] for s in st[r0:r0 + chunk]], dtype=torch.int64, device=eng.dev)
        ae = torch.tensor([s[4] for s in st[r0:r0 + chunk]], dtype=torch.int64, device=eng.dev)
        e = torch.tensor(e_rows[r0:r0 + chunk], dtype=torch.int64, device=eng.dev)
        c, s = eng.form(x, alpha, ae, e, f_codes)
        codes += c.cpu().tolist()
        scales += s.cpu().tolist()
    return F.Formed(codes, scales, [s[2] for s in st], [s[3] for s in st], [s[4] for s in st], [list(x) for x in xs])


def side(rows: list[bytes], idx, role_e: int, role_f: int, seed_e: bytes, seed_b: bytes, k: int, fb=None,
         device: str = "cpu"):
    """`PearlC4._side` with the port's forming: (Formed, its E lines); raises P.Reject where the reference does."""
    e = [F.line(seed_e, role_e, i) for i in idx]
    out = form([F.parse_row(r, k) for r in rows], e, F.basis(seed_b, role_f, k) if fb is None else fb, device)
    F.check_scale_bytes(F.NVFP4, out)
    return out, e


def sides(s, ctx, a_rows, b_rows, tile, port: bool, device: str = "cpu"):
    noise, k = s.noise(ctx), ctx.matmul.shape.k
    make = (lambda *a: side(*a, device=device)) if port else s._side
    a, e_a = make(a_rows, tile.rows, F.ROLE_EA, F.ROLE_FA, noise.seed_a, noise.seed_b, k)
    b, e_b = make(b_rows, tile.cols, F.ROLE_EB, F.ROLE_FB, noise.seed_b, noise.seed_b, k)
    return a, e_a, b, e_b


# ---- the tile rule ------------------------------------------------------------------------------------------------

def salt_dead_rows(fo, device: str = "cpu", chunk: int = 16) -> list:
    """Each row's salt-dead codes on `Engine` (None for a row the port leaves to the reference)."""
    import torch
    eng = engine(device)
    out = []
    for r0 in range(0, len(fo.x), chunk):
        dead, ok = eng.salt_dead(*(torch.tensor(v[r0:r0 + chunk], dtype=torch.int64, device=eng.dev)
                                   for v in (fo.x, fo.alpha, fo.ae)))
        out += [d if o else None for d, o in zip(dead.cpu().tolist(), ok.cpu().tolist())]
    return out


@contextmanager
def ported(a, b, words=None, dead=None):
    """`pearl_c4.word_replay` and `salt_dead` answered from the port's results over a's and b's rows: (identity,
    sub-grid) matrices, and each side's salt-dead codes per row (a row without them goes to the reference)."""
    saved = F.word_replay, F.salt_dead
    if words is not None:
        ident, sub = words
        ra, rb = {id(r): i for i, r in enumerate(a.codes)}, {id(r): j for j, r in enumerate(b.codes)}

        def replay(fmt, ac, bc, sa, sb):
            i, j = ra[id(ac)], rb[id(bc)]
            return None, int(ident[i, j]), int(sub[i, j])

        F.word_replay = replay
    if dead is not None:
        rows = {id(x): d for fo, ds in zip((a, b), dead) for x, d in zip(fo.x, ds) if d is not None}

        def salt(fmt, x, alpha, ae):
            d = rows.get(id(x)) if fmt is F.NVFP4 else None
            return saved[1](fmt, x, alpha, ae) if d is None else list(d)

        F.salt_dead = salt
    try:
        yield
    finally:
        F.word_replay, F.salt_dead = saved


def operand_digest(a, b) -> str:
    h = hashlib.sha256()
    for side_ in (a, b):
        for field in (side_.codes, side_.scales):
            h.update(bytes(v for row in field for v in row))
        h.update(json.dumps([side_.alpha, side_.beta, side_.ae]).encode())
    return h.hexdigest()


def judge(s, unit, k_kept: int, a, e_a, b, e_b, words=None, dead=None, memo=None) -> dict:
    """The enforced tile rule on formed operands (module docstring).  `words` and `dead` answer word_replay and
    salt_dead (the port's, `ported`), and `memo` is `row_terms`'s memo when rows were evaluated already."""
    record = W.UnitRecord(unit, unit.shape.m, k_kept)
    ops = SimpleNamespace(a=a, b=b, e_a=e_a, e_b=e_b)
    s._rows = (ops, s.f1, {} if memo is None else memo)
    with ported(a, b, words, dead):
        dnf_a = [p for p, x in enumerate(a.x) if F.row_passes(F.NVFP4, x)]
        dnf_b = [p for p, x in enumerate(b.x) if F.row_passes(F.NVFP4, x)]
        r1_a = [p for p in dnf_a if s.row_over_debit(record, ops, "A", p)]
        r1_b = [p for p in dnf_b if s.row_over_debit(record, ops, "B", p)]
        cred_a, cred_b = sorted(set(dnf_a) - set(r1_a)), sorted(set(dnf_b) - set(r1_b))
        got = {"rejected": "", "dnf_fail": [len(a.x) - len(dnf_a), len(b.x) - len(dnf_b)], "r1": [r1_a, r1_b],
               "credited": [len(cred_a), len(cred_b)]}
        if not (cred_a and cred_b):
            return got
        debit, cap = s.tile_cap(record, ops, cred_a, cred_b)
    return got | {"debit": s.last_debit.to_json(), "cap": str(cap), "ratio": str(debit / cap),
                  "reject": bool(debit > cap * (1 + Fraction(s.r1_tolerance)))}


def tile_outcome(s, ctx, a_rows, b_rows, tile, k_kept: int, port: bool, replay: Replay | None = None,
                 device: str = "cpu") -> dict:
    """One tile through the verifier: forming with D-SB, then `judge`; the reference's own code when not `port`."""
    try:
        a, e_a, b, e_b = sides(s, ctx, a_rows, b_rows, tile, port, device)
    except P.Reject as e:
        return {"rejected": str(e)}
    words = dead = None
    if port:
        words = replay(np.array(a.codes), np.array(a.scales), np.array(b.codes), np.array(b.scales))
        dead = salt_dead_rows(a, device), salt_dead_rows(b, device)
    return {"operands": operand_digest(a, b)} | judge(s, ctx.matmul, k_kept, a, e_a, b, e_b, words, dead)


# ---- the gate -----------------------------------------------------------------------------------------------------

def bf16_rows(x: np.ndarray) -> list[bytes]:
    """float rows as BF16 rows, nearest even (non-finite values stay non-finite, so parse_row rejects them)."""
    bits = np.ascontiguousarray(np.asarray(x, np.float32)).view(np.uint32).astype(np.uint64)
    bits = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    return [r.tobytes() for r in bits]


PLANTED = ("planted-screened", "planted-dead", "planted-reject")


def planted(fam: str, m: int, k: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Rows that reach the rule's other outcomes, after test_pouw_pearl_c4's constructions: blocks spiked to 9.6–14ρ at
    unsampled positions (screened; at 16 blocks the row's F1′ passes R1), rows salt-dead everywhere (1.0 at every 8th
    position, 256 once per block: every word's item 3 is k), and non-finite words (D-SB's reject) or a 2^16ρ spike (a_E
    subnormal: D-NF)."""
    out = []
    for scale in (1.0, 2.0 ** -6):
        x = rng.standard_normal((m, k))
        for r in range(m):
            u = rng.random()
            rho = float(np.sqrt(np.mean(x[r, ::8] ** 2)))
            if fam == "planted-screened" and u < 0.6:
                count = int(rng.choice([1, 2, 4, 8, 16, 32, k // 16]))
                for blk in rng.choice(k // 16, min(count, k // 16), replace=False):
                    x[r, 16 * blk + 1] = rho * rng.uniform(9.6, 14.0) * rng.choice([-1.0, 1.0])
            elif fam == "planted-dead" and u < 0.5:
                x[r] = 0.0
                x[r, ::8] = 1.0
                x[r, 1::16] = 256.0
            elif fam == "planted-reject" and u < 0.3:
                x[r, 16 * rng.integers(k // 16) + 1] = rho * 2.0 ** 16
        out.append(x * scale)
    if fam == "planted-reject" and rng.random() < 0.3:          # one non-finite word in the tile
        out[int(rng.integers(2))][rng.integers(m), rng.integers(k)] = rng.choice([np.inf, -np.inf, np.nan])
    return out[0], out[1]


def family_rows(fam: str, m: int, k: int, key: str) -> tuple[np.ndarray, np.ndarray]:
    """(A, B), m x k each: fp4_census's family, or a planted one."""
    sys.path.insert(0, str(HERE))
    import fp4_census as FC
    rng = np.random.default_rng(list(hashlib.sha256(key.encode()).digest()))
    a, b = planted(fam, m, k, rng) if fam in PLANTED else FC.family(fam, m, k, rng)
    return np.array(a, np.float64), np.array(b, np.float64)


def gate_case(fam: str, t: int, big: bool):
    """(scheme, context, A rows, B rows, tile, k_kept) of gate tile t of family `fam`: 16 x 16 tiles of the test
    instance at k in 128..512, or 64 x 64 production tiles at k = 1024; n, the tile column and a zero tail past
    k_kept drawn at random."""
    from verity_pouw.protocol import Context, Matmul, Shape
    rng = random.Random(f"fp4-tile-gpu/gate/{fam}/{t}/{big}")
    if big:
        s, m, k, n = F.PearlC4("nvfp4"), 64, 1024, rng.choice([64, 512, 3584, 18944])
    else:
        s, m, k, n = F.PearlC4("nvfp4", k_min=128, tile=(16, 16)), 16, rng.choice([128, 256, 384, 512]), 16 * rng.choice(
            [1, 2, 4, 32, 224, 1184])
    k_kept = k if rng.random() < 0.6 else k - rng.randrange(1, 128)
    a, b = family_rows(fam, m, k, f"{fam}/{t}/{big}")
    a[:, k_kept:] = 0.0
    b[:, k_kept:] = 0.0
    lab = lambda part: hashlib.sha256(f"fp4-tile-gpu/gate/{fam}/{t}/{big}/{part}".encode()).digest()
    ctx = Context(lab("salt"), Matmul(rng.randrange(1 << 20), Shape(m, k, n), rng.randrange(1 << 20)), lab("root_a"),
                  lab("root_b"))
    s.check(ctx.matmul)
    tiles = s.tiles(ctx.matmul.shape)
    return s, ctx, bf16_rows(a), bf16_rows(b), tiles[rng.randrange(len(tiles))], k_kept


def _outcome(fn, *args, **kw) -> dict:
    try:
        return fn(*args, **kw)
    except Exception as e:  # noqa: BLE001        both sides must fail alike: compared by type and message
        return {"raised": f"{type(e).__name__}: {e}"}


def gate_tiles(task: tuple) -> list[dict]:
    fam, big, t0, t1 = task
    rep = Replay("cpu")
    out = []
    for t in range(t0, t1):
        s, ctx, a_rows, b_rows, tile, k_kept = gate_case(fam, t, big)
        c0 = time.perf_counter()
        ref = _outcome(tile_outcome, s, ctx, a_rows, b_rows, tile, k_kept, port=False)
        c1 = time.perf_counter()
        got = _outcome(tile_outcome, s, ctx, a_rows, b_rows, tile, k_kept, port=True, replay=rep)
        c2 = time.perf_counter()
        diff = sorted(kk for kk in set(ref) | set(got) if ref.get(kk) != got.get(kk))
        if "debit" in diff:
            diff += [f"debit.{kk}" for kk in ref["debit"] if ref["debit"].get(kk) != got.get("debit", {}).get(kk)]
        out.append({"kind": "tile", "family": fam, "t": t, "big": big, "k": ctx.matmul.shape.k, "n": ctx.matmul.shape.n, "k_kept": k_kept,
                    "diff": diff, "ref": ref, "seconds": [round(c1 - c0, 2), round(c2 - c1, 2)]})
    return out


def gate_words(task: tuple) -> list[dict]:
    """Long chains, word by word: the reference's forming at k = 3584 or 18944 (the census's k), then its word_replay
    against the CPU build's (identity, sub-grid)."""
    fam, k, rows = task
    from verity_pouw.protocol import Context, Matmul, Shape
    s = F.PearlC4("nvfp4")
    a, b = family_rows(fam, rows, k, f"{fam}/words/{k}")
    lab = lambda part: hashlib.sha256(f"fp4-tile-gpu/gate-words/{fam}/{k}/{part}".encode()).digest()
    ctx = Context(lab("salt"), Matmul(0, Shape(64, k, 3584), 0), lab("root_a"), lab("root_b"))
    tile = SimpleNamespace(rows=range(rows), cols=range(rows))
    try:
        fa, _, fb, _ = sides(s, ctx, bf16_rows(a), bf16_rows(b), tile, port=False)
    except P.Reject as e:
        return [{"kind": "word", "family": fam, "k": k, "rejected": str(e), "diff": []}]
    ident, sub = Replay("cpu")(np.array(fa.codes), np.array(fa.scales), np.array(fb.codes), np.array(fb.scales))
    out = []
    for i in range(rows):
        for j in range(rows):
            _, idn, sg = F.word_replay(F.NVFP4, fa.codes[i], fb.codes[j], fa.scales[i], fb.scales[j])
            out.append({"kind": "word", "family": fam, "k": k, "word": [i, j], "ref": [idn, sg],
                        "diff": [] if (idn, sg) == (int(ident[i, j]), int(sub[i, j])) else ["identity/subgrid"]})
    return out


def gate_batch(task: tuple) -> list[dict]:
    """A census-shaped batch (64 A rows x 128 B rows at the census's k), formed by the port on the CPU, with its
    salt-dead codes and the CPU build's (identity, sub-grid), and the forming's inputs: what `gpu-check` holds CUDA to.
    Every row of a batch is salt-dead where the reference says so (checked here, row by row)."""
    fam, k, out = task
    from verity_pouw.protocol import Context, Matmul, Shape
    s = F.PearlC4("nvfp4")
    a, b = family_rows(fam, 64, k, f"{fam}/batch/{k}/a")[0], family_rows(fam, 128, k, f"{fam}/batch/{k}/b")[1]
    lab = lambda part: hashlib.sha256(f"fp4-tile-gpu/gate-batch/{fam}/{k}/{part}".encode()).digest()
    ctx = Context(lab("salt"), Matmul(0, Shape(64, k, 3584), 0), lab("root_a"), lab("root_b"))
    try:
        fa, e_a, fb, e_b = sides(s, ctx, bf16_rows(a), bf16_rows(b), SimpleNamespace(rows=range(64), cols=range(128)),
                                 port=True)
    except P.Reject as e:
        return [{"kind": "batch", "family": fam, "k": k, "rejected": str(e), "diff": []}]
    noise = s.noise(ctx)
    arrs = {"a_codes": np.array(fa.codes, np.uint8), "a_scales": np.array(fa.scales, np.uint8),
            "b_codes": np.array(fb.codes, np.uint8), "b_scales": np.array(fb.scales, np.uint8)}
    for tag, fo, e, role in (("a", fa, e_a, F.ROLE_FA), ("b", fb, e_b, F.ROLE_FB)):
        dead = salt_dead_rows(fo)
        ref = [F.salt_dead(F.NVFP4, x, al, ae) for x, al, ae in zip(fo.x, fo.alpha, fo.ae)]
        if dead != ref:
            return [{"kind": "batch", "family": fam, "k": k, "diff": [f"{tag}_dead"]}]
        arrs |= {f"{tag}_x": np.array(fo.x, np.uint32), f"{tag}_alpha": np.array(fo.alpha), f"{tag}_ae": np.array(fo.ae),
                 f"{tag}_e": np.array(e, np.uint8), f"{tag}_dead": np.array(dead, bool),
                 f"{tag}_f": np.array(F.basis(noise.seed_b, role, k), np.uint8)}
    c0 = time.perf_counter()
    ident, sub = Replay("cpu")(arrs["a_codes"], arrs["a_scales"], arrs["b_codes"], arrs["b_scales"])
    dt = time.perf_counter() - c0
    np.savez_compressed(Path(out) / f"batch-{fam}-{k}.npz", **arrs, ident=ident, sub=sub)
    return [{"kind": "batch", "family": fam, "k": k, "batch": [64, 128], "identity": int(ident.sum()), "subgrid": int(sub.sum()),
             "salt_dead": [int(arrs["a_dead"].sum()), int(arrs["b_dead"].sum())], "cpu_seconds": round(dt, 2), "diff": []}]


def gate_pinned(rep: Replay, device: str = "cpu") -> list[dict]:
    """The pinned NVFP4 vectors (tests/vectors/pearl_c4.json) through the port on `device`: operand hashes, α/β/a_E, the
    debit and the cap."""
    tree = Path(F.__file__).resolve().parents[2]
    sys.path.insert(0, str(tree / "tests"))
    import pearl_c4_vectors as V
    pinned = {c["case"]: c for c in json.loads((tree / "tests/vectors/pearl_c4.json").read_text())["cases"]}
    out = []
    for case, fmt, shape, index, weight, test in V.CASES:
        if fmt != "nvfp4":
            continue
        s, ctx = V.scheme(fmt, test), V.ctx_of(case, shape, index, weight)
        tile = s.tiles(ctx.matmul.shape)[0]
        a, e_a, b, e_b = sides(s, ctx, V.rows(case, "A", len(tile.rows), shape[1]), V.rows(case, "B", len(tile.cols),
                                                                                           shape[1]), tile, True, device)
        words = rep(np.array(a.codes), np.array(a.scales), np.array(b.codes), np.array(b.scales))
        with ported(a, b, words, (salt_dead_rows(a, device), salt_dead_rows(b, device))):
            _, cap = s.tile_cap(W.UnitRecord(ctx.matmul, shape[0], shape[1]), SimpleNamespace(a=a, b=b, e_a=e_a, e_b=e_b),
                                range(len(tile.rows)), range(len(tile.cols)))
        got = {"a_codes_sha256": V.sha(bytes(x for row in a.codes for x in row)),
               "b_codes_sha256": V.sha(bytes(x for row in b.codes for x in row)),
               "a_scales_sha256": V.sha(bytes(x for row in a.scales for x in row)),
               "b_scales_sha256": V.sha(bytes(x for row in b.scales for x in row)),
               "alpha_a": a.alpha[:4], "beta_a": a.beta[:4], "ae_a": a.ae[:4], "alpha_b": b.alpha[:4],
               "beta_b": b.beta[:4], "ae_b": b.ae[:4], "debit": s.last_debit.to_json(), "cap": str(cap)}
        out.append({"case": case, "replay": rep.kind, "device": device, "diff": sorted(kk for kk, v in got.items() if pinned[case][kk] != v)})
    return out


def _worker_init(tree: str) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    use_tree(tree)
    import torch
    torch.set_num_threads(1)


def cmd_gate(args) -> int:
    """The port against the reference: pinned vectors, --tiles-per-family 16 x 16 tiles and --big-per-family 64 x 64
    tiles per family, and long chains word by word; one JSON per task in OUT/gate (the checkpoints), exit 99 while tasks
    remain once --budget-min is spent, 0 with OUT/gate/summary.json."""
    sys.path.insert(0, str(HERE))
    import fp4_census as FC
    families = args.families.split(",") if args.families else [f for f in FC.FAMILIES if f != "coherent"] + list(PLANTED)
    out = Path(args.out) / "gate"
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.monotonic()
    if not (out / "pinned.json").exists():
        (out / "pinned.json").write_text(json.dumps(gate_pinned(Replay("cpu")), indent=1))
    tasks = []
    for fam in families:
        n = args.tiles_per_family
        tasks += [("tiles", (fam, False, t, min(t + args.chunk, n))) for t in range(0, n, args.chunk)]
        tasks += [("tiles", (fam, True, t, t + 1)) for t in range(args.big_per_family)]
        tasks += [("words", (fam, k, r)) for k, r in ((3584, 4), (18944, 2)) if args.words]
        tasks += [("batch", (fam, k, str(out.resolve()))) for k in (3584, 18944) if args.batches]

    def name(kind: str, a: tuple) -> str:
        return f"{kind}-" + "-".join(str(v) for v in a if not str(v).startswith("/")) + ".json"

    pending = [(kind, a) for kind, a in tasks if not (out / name(kind, a)).exists()]
    fns = {"tiles": gate_tiles, "words": gate_words, "batch": gate_batch}
    with ProcessPoolExecutor(args.workers, initializer=_worker_init, initargs=(args.rule_tree,)) as pool:
        flight = {}
        while pending or flight:
            while pending and len(flight) < 2 * args.workers and time.monotonic() - t_start < 60 * args.budget_min:
                kind, a = pending.pop(0)
                flight[pool.submit(fns[kind], a)] = (kind, a)
            if not flight:
                break
            for fut in wait(flight, return_when=FIRST_COMPLETED)[0]:
                kind, a = flight.pop(fut)
                tmp = out / (name(kind, a) + ".tmp")
                tmp.write_text(json.dumps(fut.result()))
                tmp.replace(out / name(kind, a))
    if pending:
        print(f"{len(pending)} gate tasks remain", flush=True)
        return 99
    summary = gate_summary(out, [name(kind, a) for kind, a in tasks], families)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({kk: summary[kk] for kk in ("pinned_mismatches", "tiles", "tile_mismatches", "words",
                                                  "word_mismatches", "batch_mismatches")}), flush=True)
    return 0


def gate_summary(out: Path, files: list[str], families: list[str]) -> dict:
    pinned = json.loads((out / "pinned.json").read_text())
    per: dict = {f: {"tiles": 0, "big": 0, "mismatches": 0, "fields": {}, "dsb_rejected": 0, "nothing_credited": 0,
                     "r1_rows": 0, "debited": 0, "over_cap": 0, "raised": 0, "words": 0, "word_mismatches": 0,
                     "batches": [], "batch_mismatches": 0}
                 for f in families}
    for fname in files:
        for r in json.loads((out / fname).read_text()):
            p = per[r["family"]]
            if r["kind"] == "batch":
                p["batches"].append({kk: r[kk] for kk in r if kk not in ("kind", "family")})
                p["batch_mismatches"] += bool(r["diff"])
                continue
            if r["kind"] == "word":
                p["words"] += "word" in r
                p["word_mismatches"] += bool(r["diff"])
                continue
            ref = r["ref"]
            p["tiles"] += 1
            p["big"] += r["big"]
            p["mismatches"] += bool(r["diff"])
            for kk in r["diff"]:
                p["fields"][kk] = p["fields"].get(kk, 0) + 1
            p["raised"] += "raised" in ref
            p["dsb_rejected"] += bool(ref.get("rejected"))
            p["nothing_credited"] += "credited" in ref and "debit" not in ref
            p["r1_rows"] += sum(len(v) for v in ref.get("r1", []))
            p["debited"] += "debit" in ref
            p["over_cap"] += bool(ref.get("reject"))
    return {"schema": "pouw/fp4-tile-gpu-gate/v0", "pinned": pinned,
            "pinned_mismatches": sum(bool(c["diff"]) for c in pinned),
            "tiles": sum(p["tiles"] for p in per.values()), "tile_mismatches": sum(p["mismatches"] for p in per.values()),
            "words": sum(p["words"] for p in per.values()),
            "word_mismatches": sum(p["word_mismatches"] for p in per.values()),
            "batch_mismatches": sum(p["batch_mismatches"] for p in per.values()), "families": per}


def gpu_name() -> dict:
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,clocks.sm", "--format=csv,noheader", "-i", vis or "0"],
                           capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        q = ""
    return {"cuda_visible_devices": vis, "gpu": q}


def cuda_forming(z) -> tuple[dict, float]:
    """A gate batch's forming and salt-dead codes again on CUDA (`Engine`), against the CPU's: mismatched elements per
    array."""
    import torch
    eng = engine("cuda")
    diff, c0 = {}, time.perf_counter()
    for tag in ("a", "b"):
        x, alpha, ae, e, f = (torch.tensor(z[f"{tag}_{name}"].astype(np.int64), device=eng.dev)
                              for name in ("x", "alpha", "ae", "e", "f"))
        codes, scales = eng.form(x, alpha, ae, e, f)
        dead, ok = eng.salt_dead(x, alpha, ae)
        got = {"codes": codes.cpu().numpy(), "scales": scales.cpu().numpy(), "dead": dead.cpu().numpy()}
        diff |= {f"{tag}_{name}": int((got[name] != z[f"{tag}_{name}"]).sum()) for name in got}
        diff[f"{tag}_unmodelled_rows"] = int((~ok).sum())
    torch.cuda.synchronize()
    return diff, time.perf_counter() - c0


def cmd_gpu_check(args) -> int:
    """CUDA against the CPU on the gate's census-shaped batches (OUT/gate/batch-*.npz: the forming and salt-dead codes
    on torch, the replay's CUDA build) and the pinned vectors through CUDA; then a census-sized replay (64 x 1024 words)
    per k, timed, for the GPU-hour estimate."""
    rep = Replay("cuda")
    out = Path(args.out) / "gate"
    res: dict = {"schema": "pouw/fp4-tile-gpu-check/v0", "replay": str(rep.path), **gpu_name(),
                 "pinned": gate_pinned(rep, "cuda"), "batches": [], "timing": []}
    big = {}
    for f in sorted(out.glob("batch-*.npz")):
        z = np.load(f)
        forming, dt_form = cuda_forming(z)
        c0 = time.perf_counter()
        ident, sub = rep(z["a_codes"], z["a_scales"], z["b_codes"], z["b_scales"])
        dt = time.perf_counter() - c0
        k = z["a_codes"].shape[1]
        res["batches"].append({"file": f.name, "k": k, "words": int(ident.size), "seconds": round(dt, 3),
                               "forming_seconds": round(dt_form, 3), "forming": forming,
                               "mismatches": int((ident != z["ident"]).sum() + (sub != z["sub"]).sum())
                               + sum(forming.values())})
        big.setdefault(k, z)
    for k, z in sorted(big.items()):
        b_codes, b_scales = np.tile(z["b_codes"], (8, 1)), np.tile(z["b_scales"], (8, 1))
        c0 = time.perf_counter()
        ident, sub = rep(z["a_codes"], z["a_scales"], b_codes, b_scales)
        dt = time.perf_counter() - c0
        same = bool((ident == np.tile(z["ident"], (1, 8))).all() and (sub == np.tile(z["sub"], (1, 8))).all())
        res["timing"].append({"k": k, "words": int(ident.size), "seconds": round(dt, 3), "matches_cpu": same})
    res["mismatches"] = (sum(bool(c["diff"]) for c in res["pinned"]) + sum(b["mismatches"] for b in res["batches"])
                         + sum(not t["matches_cpu"] for t in res["timing"]))
    (Path(args.out) / "gpu-check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({"mismatches": res["mismatches"], "batches": len(res["batches"]), "timing": res["timing"]}), flush=True)
    return 1 if res["mismatches"] or not res["batches"] else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    cmds = ap.add_subparsers(dest="cmd", required=True)
    g = cmds.add_parser("gate", help=cmd_gate.__doc__.split("\n\n")[0])
    c = cmds.add_parser("gpu-check", help=cmd_gpu_check.__doc__.split("\n\n")[0])
    for p in (g, c):
        p.add_argument("--out", required=True)
        p.add_argument("--rule-tree", default="")
    g.add_argument("--families", default="")
    g.add_argument("--tiles-per-family", type=int, default=1000)
    g.add_argument("--big-per-family", type=int, default=4)
    g.add_argument("--chunk", type=int, default=25)
    g.add_argument("--workers", type=int, default=8)
    g.add_argument("--budget-min", type=float, default=25.0)
    g.add_argument("--no-words", dest="words", action="store_false")
    g.add_argument("--no-batches", dest="batches", action="store_false")
    args = ap.parse_args(argv)
    args.rule_tree = str(use_tree(args.rule_tree))
    return cmd_gate(args) if args.cmd == "gate" else cmd_gpu_check(args)


if __name__ == "__main__":
    sys.exit(main())
