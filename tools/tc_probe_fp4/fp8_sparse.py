"""The 2:4-sparse E4M3 row of the sm_120 FP8 probe (``fp8.py --row sparse``): one
``mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32`` per tile (``mma_fp8.cu``'s
``sp_e4m3_tiles_kernel``, SASS ``QMMA.SP.16864.F32.E4M3.E4M3``), raw bits in and out, against two chained dense m16n8k32 atoms on
the same logical operands, the dense atom on the compressed operands, and GroupSum candidates over the 64 logical products.  A
capture: which candidates and which hardware twins reproduce every gated word is the finding; ``--expect-model NAME`` makes one
candidate the gate (a fresh-seed confirmation).

    python tools/tc_probe_fp4/fp8.py --row sparse --out $RESEARCH_RUN_DIR --lease-max-min 15 --clocks locked-2100 [--n-random N]

Operands (PTX ISA "Sparse matrix storage", 8-bit types): A is 16 x 64 logical, 2:4 structured along k.  Each group of four logical
columns keeps two, i0 < i1, named by the group's metadata nibble ``(i1 << 2) | i0`` (six valid values); the stored row is 32 codes,
group c's two at stored columns 2c (i0) and 2c + 1 (i1).  B is all of the 64 x 8 logical operand.  The stored A's fragment is the
dense m16n8k32 A's; B's four registers and the metadata register are packed on the host under each candidate layout (``LAYOUTS``),
and the layout check keeps the one that reproduces exact small integers (exactly one must).

Variants of every step family (``fp8.step_families`` for f8_e4m3), each also from C = +0 (``.zero_acc``) when C is not all +0:

* ``shared``: one metadata row per tile (every row keeps the same logical columns).  The family's A is the stored A and the
  family's B sits at the kept columns, so the kept products are the family's own; the other 32 B rows are finite filler from a
  second draw.  Dense twin: the dense atom on the family's operands (the compressed operands).
* ``per_row``: random metadata per row and group; B is the family's B in columns 0..31 and the previous tile's in 32..63.  Dense
  twin: the dense atom on each row's compressed operands (16 dense tiles per sparse tile, row i read from tile i).

Both run the chain twin as well: two dependent dense k32 atoms (``e4m3_chain``) on logical columns 0..31, then 32..63.

Candidates (``CANDIDATES``): GroupSum over the 64 logical products.  A zero product is not live, so ``(64,)`` is one align-add over
the 32 stored products, i.e. the registered dense step on the compressed operands; ``(32, 32)`` is two chained registered k32
steps; then finer groups, widths 25..28 and a higher floor.  The dense twins are checked against the registered model on the same
elements (``twin_checks``): a disagreement there is a named discrepancy against a registered model and fails the run.

Phases, so no GPU is held while the CPU verifies (server.md 08:30Z): the parent builds both libraries and runs both SASS gates, then
runs the GPU phase (``--phase gpu``: the device gates, every capture and twin, and ``w1.py --ops qmma_e4m3_f32,qmma_sp_e4m3`` for the
rates) under ``gpu-lease 1 --wait --max-min M`` when ``--lease-max-min M`` is given (in process otherwise), and verifies every word
against the candidates on ``--workers`` CPU processes after the lease is released.  The rates are Measured W1 prices of the sparse
atom per logical MAC and per nonzero product, in the same lease as the unit.
"""
from __future__ import annotations

import ctypes
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import fp8 as F
import probe as P4
import tc_probe as TP
from verity.ml.tc import models as TCM

PTX = "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32"
INSTRUCTION_ID = "sm120.mma.sp.m16n8k64.e4m3"
W1_OPS = "qmma_e4m3_f32,qmma_sp_e4m3"
GPU_JSON, CAPTURE_NPZ = "sparse_gpu.json", "sparse_capture.npz"
#: the metadata nibbles mma.sp::ordered_metadata accepts for 2:4: kept indices i0 < i1 as (i1 << 2) | i0
META_VALID = np.array([0b0100, 0b1000, 0b1001, 0b1100, 0b1101, 0b1110], np.uint8)

# --- layouts ------------------------------------------------------------------------------------------------------------------------

_G, _Q = np.arange(32)[:, None, None] >> 2, np.arange(32)[:, None, None] & 3
_R, _J = np.arange(4)[None, :, None], np.arange(4)[None, None, :]
#: stored A register r byte j of lane 4g + q: the dense m16n8k32 A fragment
A_ROW, A_K = np.broadcast_to(_G + 8 * (_R & 1), (32, 4, 4)), np.broadcast_to(4 * _Q + _J + 16 * (_R >> 1), (32, 4, 4))
B_COL = np.broadcast_to(_G, (32, 4, 4))
B_LAYOUTS = {"k=4q+j+16r": np.broadcast_to(4 * _Q + _J + 16 * _R, (32, 4, 4)),
             "k=16q+4r+j": np.broadcast_to(16 * _Q + 4 * _R + _J, (32, 4, 4))}
_g, _q, _c = np.arange(32)[:, None] >> 2, np.arange(32)[:, None] & 3, np.arange(8)[None, :]
E_LAYOUTS = {  # (row, group) of nibble c of lane 4g + q's metadata register
    "row g+8(q&1), groups 8(q>>1)+c": (np.broadcast_to(_g + 8 * (_q & 1), (32, 8)), np.broadcast_to(8 * (_q >> 1) + _c, (32, 8))),
    "row g+8(q>>1), groups 8(q&1)+c": (np.broadcast_to(_g + 8 * (_q >> 1), (32, 8)), np.broadcast_to(8 * (_q & 1) + _c, (32, 8))),
    "rows g|g+8 by c>>2, groups 4q+(c&3)": (np.broadcast_to(_g + 8 * (_c >> 2), (32, 8)), np.broadcast_to(4 * _q + (_c & 3), (32, 8))),
}
ORDERS = ("i0 first", "i1 first")  # which kept element stored column 2c holds
#: every layout the check tries; the first is the PTX ISA's (the analogue of the FP4 m16n8k128 layout GPU 4's check selected)
LAYOUTS = [(b, e, o) for b in B_LAYOUTS for e in E_LAYOUTS for o in ORDERS]
PRIMARY_LAYOUT = LAYOUTS[0]
_SH8, _SH4 = 8 * np.arange(4, dtype=np.uint32), 4 * np.arange(8, dtype=np.uint32)


def layout_name(layout) -> str:
    return " | ".join(layout)


def swap_pairs(As: np.ndarray) -> np.ndarray:
    """Stored columns 2c <-> 2c + 1."""
    return As.reshape(*As.shape[:-1], 16, 2)[..., ::-1].reshape(As.shape)


def columns(M) -> np.ndarray:
    """(..., 16) group nibbles -> (..., 32): the logical column of each stored column."""
    m = np.asarray(M, np.int64)
    return np.stack([m & 3, m >> 2], -1).reshape(*m.shape[:-1], 32) + 4 * (np.arange(32) // 2)


def logical_a(As, M) -> np.ndarray:
    """Stored A (n, 16, 32) and its metadata (n, 16, 16) -> the logical A (n, 16, 64), code 0 where nothing is kept."""
    Al = np.zeros(As.shape[:-1] + (64,), np.uint8)
    np.put_along_axis(Al, columns(M), As, axis=-1)
    return Al


def pack(As, M, Bl, layout) -> np.ndarray:
    """Stored A (n,16,32), metadata (n,16,16), logical B (n,8,64) -> every lane's registers R (n, 32, 9) under one layout."""
    b_name, e_name, order = layout
    A = As if order == "i0 first" else swap_pairs(As)
    R = np.zeros((As.shape[0], 32, 9), np.uint32)
    R[:, :, 0:4] = (A[:, A_ROW, A_K].astype(np.uint32) << _SH8).sum(-1)
    R[:, :, 4:8] = (Bl[:, B_COL, B_LAYOUTS[b_name]].astype(np.uint32) << _SH8).sum(-1)
    er, eg = E_LAYOUTS[e_name]
    R[:, :, 8] = (np.asarray(M)[:, er, eg].astype(np.uint32) << _SH4).sum(-1)
    return R


def unpack(R, layout=PRIMARY_LAYOUT) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """pack's inverse: R (n, 32, 9) -> stored A, metadata, logical B."""
    b_name, e_name, order = layout
    n = R.shape[0]
    As, M, Bl = np.zeros((n, 16, 32), np.uint8), np.zeros((n, 16, 16), np.uint8), np.zeros((n, 8, 64), np.uint8)
    As[:, A_ROW, A_K] = (R[:, :, 0:4, None] >> _SH8) & 0xFF
    Bl[:, B_COL, B_LAYOUTS[b_name]] = (R[:, :, 4:8, None] >> _SH8) & 0xFF
    er, eg = E_LAYOUTS[e_name]
    M[:, er, eg] = (R[:, :, 8, None] >> _SH4) & 0xF
    return (As if order == "i0 first" else swap_pairs(As)), M, Bl


# --- candidates ---------------------------------------------------------------------------------------------------------------------

S = F.spec
PRIMARY = "(64,) w26 f-133"
CANDIDATES = {
    PRIMARY: S((64,), 26, -133),  # one align-add over the 64-deep span: the registered dense step on the compressed operands
    "(32,32) w26 f-133": S((32, 32), 26, -133),  # two chained registered k32 steps
    "(16,16,16,16) w26 f-133": S((16,) * 4, 26, -133),
    "(8,)x8 w26 f-133": S((8,) * 8, 26, -133),
    "(64,) w25 f-133": S((64,), 25, -133), "(64,) w27 f-133": S((64,), 27, -133), "(64,) w28 f-133": S((64,), 28, -133),
    "(32,32) w25 f-133": S((32, 32), 25, -133), "(32,32) w27 f-133": S((32, 32), 27, -133),
    "(64,) w26 f-123": S((64,), 26, -123),
}
CHAINED, GATHERED = "(32,32) w26 f-133", PRIMARY  # the registered models of the chain twin and of the compressed-operand twin
TWINS = {"chain2": "two chained dense k32 atoms on the logical halves", "dense_compressed": "the dense atom on the compressed operands"}


def evaluate(sp: dict, As, M, Bl, C) -> np.ndarray:
    Al = logical_a(As, M)
    n = As.shape[0]
    out = np.zeros(n * 128, np.uint32)
    for s in range(0, n, 256):
        e = min(n, s + 256)
        a, b, acc = F.expand(Al[s:e], Bl[s:e], C[s:e])
        out[s * 128:e * 128] = F.groupsum(sp, acc, F.products("e4m3", a, "e4m3", b))
    return out.reshape(n, 16, 8)


# --- devices ------------------------------------------------------------------------------------------------------------------------


class Device(F.Device):
    def __init__(self, so: Path):
        super().__init__(so)
        self.lib.sp_e4m3_tiles.restype = ctypes.c_int

    def sp_tiles(self, R, C) -> np.ndarray:
        n = R.shape[0]
        R = np.ascontiguousarray(R, np.uint32).reshape(n, 32, 9)
        C = np.ascontiguousarray(C, np.uint32).reshape(n, 16, 8)
        D = np.zeros((n, 16, 8), np.uint32)
        for s in range(0, n, F.MAX_TILES):
            e = min(n, s + F.MAX_TILES)
            r, c, d = np.ascontiguousarray(R[s:e]), np.ascontiguousarray(C[s:e]), np.zeros((e - s, 16, 8), np.uint32)
            self._check(self.lib.sp_e4m3_tiles(F._ptr(r), F._ptr(c), F._ptr(d), e - s), "sp_e4m3_tiles")
            D[s:e] = d
        return D


class FakeDevice(F.FakeDevice):
    """A candidate standing in for the sparse instruction under the PTX layout (the CPU dry run), fp8's fakes for the dense twins."""

    def __init__(self, corrupt: int | None = None, model: str = PRIMARY, layout=PRIMARY_LAYOUT):
        super().__init__(corrupt)
        self.model, self.layout = CANDIDATES[model], layout

    def sp_tiles(self, R, C):
        As, M, Bl = unpack(R, self.layout)
        return self._maybe_corrupt(evaluate(self.model, As, M, Bl, C))


# --- variants, twins and the layout check -------------------------------------------------------------------------------------------


def variants(rng, n_random: int, families=None):
    """(name, stored A, metadata, logical B, C) per family and variant; every draw happens whatever ``families`` selects."""
    E = TP.E4M3
    for name, A, Bt, C, _, _ in F.step_families(rng, "f8_e4m3", n_random):
        n = A.shape[0]
        m_shared = np.repeat(META_VALID[rng.integers(0, 6, (n, 1, 16))], 16, axis=1)
        Bs = E.random_finite_bits(rng, (n, 8, 64))
        np.put_along_axis(Bs, np.broadcast_to(columns(m_shared[:, :1]), (n, 8, 32)), Bt, axis=-1)
        m_row = META_VALID[rng.integers(0, 6, (n, 16, 16))]
        Br = np.concatenate([Bt, np.roll(Bt, 1, axis=0)], axis=-1)
        if families and name not in families:
            continue
        for mode, M, Bl in (("shared", m_shared, Bs), ("per_row", m_row, Br)):
            yield f"{name}.{mode}", A, M, Bl, C
            if C.any():
                yield f"{name}.{mode}.zero_acc", A, M, Bl, np.zeros_like(C)


def twins(dev, As, M, Bl, C, shared: bool) -> tuple[np.ndarray, np.ndarray]:
    """The hardware twins: the chain of two dense k32 atoms (n, 2, 16, 8) and the dense atom on the compressed operands (n, 16, 8)."""
    n = As.shape[0]
    Al = logical_a(As, M)
    Dc = dev.e4m3_chain(np.stack([Al[..., :32], Al[..., 32:]], 1), np.stack([Bl[..., :32], Bl[..., 32:]], 1), C)
    cols = columns(M)
    if shared:
        Dg = dev.tiles("f8_e4m3", As, np.take_along_axis(Bl, np.broadcast_to(cols[:, :1], (n, 8, 32)), -1), C)
    else:
        Bg = np.take_along_axis(np.broadcast_to(Bl[:, None], (n, 16, 8, 64)), np.broadcast_to(cols[:, :, None], (n, 16, 8, 32)), -1)
        Dt = dev.tiles("f8_e4m3", np.repeat(As, 16, 0), Bg.reshape(n * 16, 8, 32), np.repeat(C, 16, 0)).reshape(n, 16, 16, 8)
        Dg = Dt[:, np.arange(16), np.arange(16)]
    return Dc, Dg


def layout_gate(dev, rng, n: int = 64) -> dict:
    """Exact small integers (random valid metadata per row and group, and B nonzero almost everywhere, so a layout that selects the
    wrong logical columns picks up the filler) under every candidate layout: exactly one may reproduce the exact matmul."""
    As, va = F._layout_values(rng, (n, 16, 32), "e4m3")
    M = META_VALID[rng.integers(0, 6, (n, 16, 16))]
    Bl, vb = F._layout_values(rng, (n, 8, 64), "e4m3")
    vl = np.zeros((n, 16, 64))
    np.put_along_axis(vl, columns(M), va, -1)
    want = np.einsum("tik,tjk->tij", vl, vb).astype(np.float32)
    zero = np.zeros((n, 16, 8), np.uint32)
    mism = {layout_name(lay): int(np.count_nonzero(dev.sp_tiles(pack(As, M, Bl, lay), zero).view(np.float32) != want)) for lay in LAYOUTS}
    matches = [k for k, v in mism.items() if v == 0]
    return {"ok": len(matches) == 1, "mismatches": mism, "layout": matches[0] if len(matches) == 1 else None,
            "primary_layout": layout_name(PRIMARY_LAYOUT)}


# --- the GPU phase ------------------------------------------------------------------------------------------------------------------


def _write_json(path: Path, doc: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=1, default=str))
    os.replace(tmp, path)


def gpu_phase(args, out: Path, so_fp8: Path | None, so_w1: Path | None) -> dict:
    """Device gates, every capture and twin (``sparse_capture.npz``, uncompressed: nothing slow while the GPU is held), then the W1
    rates; the report goes to ``sparse_gpu.json``."""
    t0 = time.time()
    res: dict = {"phase": "gpu", "failures": []}
    if args.fake_silicon:
        dev = FakeDevice(args.corrupt_word, args.expect_model or PRIMARY)
    else:
        dev = Device(so_fp8)
    res["identity"], id_fail = F.identity_gate(dev, args.expect_gpu_uuid)
    res["failures"] += id_fail
    if not res["failures"]:
        rng = np.random.default_rng(1)
        res["layout"] = {"f8_e4m3": F.layout_gate(dev, "step", "f8_e4m3", rng), "e4m3_chain": F.layout_gate(dev, "chain", None, rng),
                         "sparse": layout_gate(dev, rng)}
        bad = {k: v["mismatches"] for k, v in res["layout"].items() if not v["ok"]}
        if bad:
            res["failures"].append(f"layout check: {bad}")
    if res["failures"]:
        res["seconds"] = round(time.time() - t0, 3)
        _write_json(out / GPU_JSON, res)
        return res
    layout = tuple(res["layout"]["sparse"]["layout"].split(" | "))
    if isinstance(dev, FakeDevice):
        dev.armed = True
    arrays, res["variants"] = {}, {}
    for name, As, M, Bl, C in variants(np.random.default_rng(args.seed), args.n_random, args.families):
        t = time.time()
        D = dev.sp_tiles(pack(As, M, Bl, layout), C)
        Dc, Dg = twins(dev, As, M, Bl, C, ".shared" in name)
        arrays[name] = {"As": As, "M": M, "Bl": Bl, "C": C, "D": D, "Dc": Dc, "Dg": Dg}
        res["variants"][name] = {"tiles": int(As.shape[0]), "device_seconds": round(time.time() - t, 3)}
    np.savez(out / CAPTURE_NPZ, **{f"{v}/{k}": x for v, d in arrays.items() for k, x in d.items()})
    res["capture_seconds"] = round(time.time() - t0, 3)
    w1_argv = ["--out", str(out / "w1"), "--ops", W1_OPS, "--seed", str(args.seed), "--reps", str(args.reps), "--target-cycles",
               str(args.target_cycles), "--clocks", args.clocks]
    w1_argv += ["--prebuilt", str(so_w1)] if so_w1 else []
    w1_argv += ["--expect-gpu-uuid", args.expect_gpu_uuid] if args.expect_gpu_uuid else []
    if args.fake_silicon:  # no CUDA library is loaded: w1's fake runs in this process
        import w1 as W1

        code, _ = W1.run(W1.parser().parse_args([*w1_argv, "--fake-silicon"]))
        res["w1"] = {"argv": w1_argv, "returncode": code}
    else:  # its own process, so each CUDA library has a process of its own
        env = {k: v for k, v in os.environ.items() if k != "RESEARCH_RUN_ID"}  # w1's result.json would shadow this row's
        p = subprocess.run([sys.executable, str(F.HERE / "w1.py"), *w1_argv], env=env, capture_output=True, text=True)
        res["w1"] = {"argv": w1_argv, "returncode": p.returncode, "stdout_tail": p.stdout[-4000:], "stderr_tail": p.stderr[-4000:]}
    res["seconds"] = round(time.time() - t0, 3)
    _write_json(out / GPU_JSON, res)
    return res


def child_argv(args, out: Path) -> list[str]:
    argv = ["--row", "sparse", "--phase", "gpu", "--out", str(out), "--seed", str(args.seed), "--n-random", str(args.n_random),
            "--reps", str(args.reps), "--target-cycles", str(args.target_cycles), "--clocks", args.clocks]
    argv += ["--families", ",".join(sorted(args.families))] if args.families else []
    argv += ["--expect-gpu-uuid", args.expect_gpu_uuid] if args.expect_gpu_uuid else []
    argv += ["--expect-model", args.expect_model] if args.expect_model else []
    return argv


# --- verification (CPU, after the lease) --------------------------------------------------------------------------------------------

_CTX: dict = {}
CHUNK = 256


def _task(key):
    name, s, e, cand = key
    v = _CTX[name]
    return key, evaluate(CANDIDATES[cand], v["As"][s:e], v["M"][s:e], v["Bl"][s:e], v["C"][s:e])


def _hexrow(x) -> str:
    return "".join(f"{int(v):02x}" for v in np.asarray(x).reshape(-1))


def _nibbles(m) -> str:
    return "".join(f"{int(v):x}" for v in np.asarray(m).reshape(-1))


def _cross_check(v: dict, shared: bool, preds: dict, tiles: int = 32) -> None:
    """fp8.groupsum's multi-group candidates against the core batch kernel on the registered model, on the first tiles."""
    core = TCM.BLACKWELL_SM120_E4M3_M16N8K32
    As, M, Bl, C = (v[k][:tiles] for k in ("As", "M", "Bl", "C"))
    Al = logical_a(As, M)
    two = F.step_model(core, Al[..., 32:], Bl[..., 32:], F.step_model(core, Al[..., :32], Bl[..., :32], C))
    if not np.array_equal(two, preds[CHAINED][:tiles]):
        raise RuntimeError("fp8.groupsum (32, 32) disagrees with two chained kernels.group_sum_total_batch steps")
    if shared:
        Bg = np.take_along_axis(Bl, np.broadcast_to(columns(M[:, :1]), (As.shape[0], 8, 32)), -1)
        if not np.array_equal(F.step_model(core, As, Bg, C), preds[GATHERED][:tiles]):
            raise RuntimeError("fp8.groupsum (64,) disagrees with kernels.group_sum_total_batch on the compressed operands")


def verify(args, out: Path) -> dict:
    t0 = time.time()
    data = np.load(out / CAPTURE_NPZ)
    vs: dict[str, dict] = {}
    for key in data.files:
        name, k = key.split("/")
        vs.setdefault(name, {})[k] = data[key]
    _CTX.clear()
    _CTX.update(vs)
    tasks = [(name, s, min(v["As"].shape[0], s + CHUNK), c) for name, v in vs.items() for s in range(0, v["As"].shape[0], CHUNK)
             for c in CANDIDATES]
    preds = {name: {c: np.zeros((v["As"].shape[0], 16, 8), np.uint32) for c in CANDIDATES} for name, v in vs.items()}
    workers = max(1, min(args.workers or min(32, os.cpu_count() or 1), len(tasks)))
    if workers > 1:
        with mp.get_context("fork").Pool(workers) as pool:
            for (name, s, e, c), words in pool.imap_unordered(_task, tasks):
                preds[name][c][s:e] = words
    else:
        for key in tasks:
            (name, s, e, c), words = _task(key)
            preds[name][c][s:e] = words
    primary = args.expect_model or PRIMARY
    fams, first, samples = {}, [], {}
    rng = np.random.default_rng(args.seed + 1)
    for name, v in vs.items():
        shared = ".shared" in name
        _cross_check(v, shared, preds[name])
        As, M, Bl, C, D, Dc, Dg = (v[k] for k in ("As", "M", "Bl", "C", "D", "Dc", "Dg"))
        n = As.shape[0]
        fin = (((C & 0x7F800000) != 0x7F800000) & TP.E4M3.is_finite(As).all(-1)[:, :, None] & TP.E4M3.is_finite(Bl).all(-1)[:, None, :])
        gated = fin.reshape(-1) & ((preds[name][primary].reshape(-1) & 0x7F800000) != 0x7F800000)
        d = D.reshape(-1)
        rec = {"tiles": int(n), "elements": int(d.size), "gated_elements": int(gated.sum()), "mismatches": {}, "record_only_mismatches": {}}
        words = {c: p.reshape(-1) for c, p in preds[name].items()} | {"twin:chain2": Dc[:, 1].reshape(-1), "twin:dense_compressed": Dg.reshape(-1)}
        for c, p in words.items():
            bad = d != p
            rec["mismatches"][c] = int((bad & gated).sum())
            rec["record_only_mismatches"][c] = int((bad & ~gated).sum())
            if c in (primary, "twin:chain2", "twin:dense_compressed"):
                for r in np.flatnonzero(bad & gated)[:3]:
                    t, i, j = r // 128, (r // 8) % 16, r % 8
                    first.append({"variant": name, "against": c, "tile": int(t), "i": int(i), "j": int(j), "c": f"{int(C[t, i, j]):08x}",
                                  "a_stored": _hexrow(As[t, i]), "meta": _nibbles(M[t, i]), "b_logical": _hexrow(Bl[t, j]),
                                  "hardware": f"{int(d[r]):08x}", "chain2": f"{int(Dc[t, 1, i, j]):08x}", "chain2_step0": f"{int(Dc[t, 0, i, j]):08x}",
                                  "dense_compressed": f"{int(Dg[t, i, j]):08x}", "expected": f"{int(p[r]):08x}"})
        rec["twin_checks"] = {f"chain2 vs {CHAINED}": int(((Dc[:, 1].reshape(-1) != preds[name][CHAINED].reshape(-1)) & gated).sum()),
                              f"dense_compressed vs {GATHERED}": int(((Dg.reshape(-1) != preds[name][GATHERED].reshape(-1)) & gated).sum())}
        pick = rng.choice(d.size, size=min(32, d.size), replace=False)
        samples[name] = [{"c": f"{int(C.reshape(-1)[r]):08x}", "a_stored": _hexrow(As[r // 128, (r // 8) % 16]), "meta": _nibbles(M[r // 128, (r // 8) % 16]),
                          "b_logical": _hexrow(Bl[r // 128, r % 8]), "d": f"{int(d[r]):08x}"} for r in pick]
        fams[name] = rec
    keys = [*CANDIDATES, "twin:chain2", "twin:dense_compressed"]
    totals = {c: sum(f["mismatches"][c] for f in fams.values()) for c in keys}
    checks = {k: sum(f["twin_checks"][k] for f in fams.values()) for k in next(iter(fams.values()))["twin_checks"]} if fams else {}
    fits = [c for c in CANDIDATES if totals[c] == 0]
    np.savez_compressed(out / "sparse_tiles.npz", **{f"{v}.{k}": x for v, d in vs.items() for k, x in d.items()})
    (out / CAPTURE_NPZ).unlink()
    return {"families": fams, "candidates": {c: {**sp, "groups": list(sp["groups"])} for c, sp in CANDIDATES.items()}, "primary": primary,
            "twins": TWINS, "total_mismatches": totals, "agrees": {c: v == 0 for c, v in totals.items()}, "fits": fits,
            "twin_checks": checks, "first_mismatches": first[:40], "samples": samples,
            "elements": sum(f["elements"] for f in fams.values()), "gated_elements": sum(f["gated_elements"] for f in fams.values()),
            "workers": workers, "verify_seconds": round(time.time() - t0, 3)}


# --- main ---------------------------------------------------------------------------------------------------------------------------


def _w1_measurements(w1: dict, clocks: str) -> list[dict]:
    pr = w1.get("prices") or {}
    label = f"Measured, {clocks} clocks"
    rates, prices = pr.get("rates_per_sm_per_clock") or {}, pr.get("price_units") or {}
    meas = [{"name": f"rate.{k}", "value": v, "unit": ("logical MAC" if k == "qmma_sp_e4m3" else "MAC") + "/SM/clock", "label": label}
            for k, v in rates.items()]
    meas += [{"name": f"price.{k}", "value": v, "unit": "W1 units", "label": label} for k, v in prices.items() if k.startswith("qmma_")]
    if "sp_over_dense_instructions" in pr:
        meas.append({"name": "ratio.qmma_sp_e4m3_over_dense_instructions", "value": pr["sp_over_dense_instructions"], "unit": "1", "label": label})
    return meas


def run(args) -> tuple[int, dict]:
    out = Path(os.path.expandvars(args.out))
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if args.phase == "gpu":
        so_fp8, so_w1 = out / f"libmma_fp8_{F.ARCH}.so", out / "w1" / f"libw1_prices_{F.ARCH}.so"
        g = gpu_phase(args, out, so_fp8 if so_fp8.exists() else None, so_w1 if so_w1.exists() else None)
        return (0 if not g["failures"] else 1), {"validation": "gpu phase " + ("passed" if not g["failures"] else "failed"), **g}
    import w1 as W1

    results: dict = {"schema": "tc-probe-fp4/fp8/v1", "row": "sparse", "instruction": INSTRUCTION_ID, "ptx": PTX, "seed": args.seed,
                     "fake_silicon": bool(args.fake_silicon), "clocks": args.clocks}
    failures: list[str] = []
    so_fp8 = so_w1 = None
    if not args.fake_silicon or P4.nvcc_bin() and Path(P4.nvcc_bin()).exists():
        so_fp8, results["nvcc"] = P4.build(F.ARCH, out, F.SOURCE)
        results["sass"] = F.sass_gate(so_fp8, out)
        (out / "w1").mkdir(exist_ok=True)
        so_w1, _ = P4.build(F.ARCH, out / "w1", W1.SOURCE)
        w1s = W1.sass_gate(so_w1, None)
        results["sass_w1"] = {"ok": w1s["ok"], "kernels": {k: w1s["kernels"][k] for k in W1_OPS.split(",")}}
        if not results["sass"]["ok"]:
            failures.append("SASS gate: " + ", ".join(k for k, r in results["sass"]["kernels"].items() if not r["ok"]))
        if not w1s["ok"]:
            failures.append("W1 SASS gate: " + ", ".join(k for k, r in w1s["kernels"].items() if not r["ok"]))
    else:
        results["sass"] = {"ok": None, "detail": "skipped: fake silicon without nvcc"}
    if failures:
        results["seconds"] = round(time.time() - t0, 3)
        return F.finish(out, args, results, failures, None)
    if args.lease_max_min:
        cmd = ["gpu-lease", "1", "--wait", "--max-min", str(args.lease_max_min), "--", sys.executable, str(F.HERE / "fp8.py"), *child_argv(args, out)]
        t = time.time()
        p = subprocess.run(cmd)
        results["lease"] = {"command": cmd, "returncode": p.returncode, "seconds": round(time.time() - t, 3)}
    else:
        gpu_phase(args, out, so_fp8, so_w1)
    gpu_json = out / GPU_JSON
    g = json.loads(gpu_json.read_text()) if gpu_json.exists() else {"failures": ["the GPU phase wrote no " + GPU_JSON]}
    results["identity"], results["layout"] = g.get("identity"), g.get("layout")
    results["gpu_phase"] = {k: g.get(k) for k in ("variants", "capture_seconds", "seconds", "w1")}
    failures += g["failures"]
    if failures:
        results["seconds"] = round(time.time() - t0, 3)
        return F.finish(out, args, results, failures, results["identity"])
    res = verify(args, out)
    results["capture"] = {k: v for k, v in res.items() if k != "samples"}
    (out / "capture_sample.json").write_text(json.dumps(res["samples"], indent=1))
    if any(res["twin_checks"].values()):
        failures.append(f"a dense twin disagrees with the registered model on the same elements: {res['twin_checks']}")
    if args.expect_model and not res["agrees"][args.expect_model]:
        failures.append(f"--expect-model {args.expect_model}: {res['total_mismatches'][args.expect_model]} mismatches")
    w1_json = out / "w1" / "w1_results.json"
    w1 = json.loads(w1_json.read_text()) if w1_json.exists() else {"validation": "missing", "failures": ["w1.py wrote no w1_results.json"]}
    results["w1"] = {k: w1.get(k) for k in ("validation", "failures", "ops", "numeric_gate", "prices", "identity")}
    if w1.get("validation") != "passed":
        failures.append(f"W1 rates: {w1.get('failures')}")
    else:
        results["measurements"] = _w1_measurements(w1, args.clocks)
    results["seconds"] = round(time.time() - t0, 3)
    return F.finish(out, args, results, failures, results["identity"])
