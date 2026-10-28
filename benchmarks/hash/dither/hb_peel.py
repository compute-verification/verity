"""The `dither-1s-hbp` arm's gate: `dither_one.check` with z held to today's A B^T (the peel), for hb_one's
peeled unit (h1u.cu's kh_one_peel: z = O' - sum_d 128^d D_d + 256 (X WB^T + xa B^T) - (X c^T + b B^T) = A B^T, xa where
the int8 X = A + b wraps). `llm.dither_mod` for that arm, so sbench runs this module's `check` on the gate, control,
fault, fault_wb and fault_xa captures and `gate_verdict` over them; `install()` (before the engine is built) makes the
window raise F_FAULT_XA on the fault_xa step.

Unchanged from `dither_one.check`: every call's A, admit and z roots and the step's round against commit.py; every receipt
against h1r v1 (the receipts don't move with the peel); y (dumped) = dequant of the kernel's own z; wb's density by
h1o_ref; 0xA5 over every buffer read. Changed, all against A B^T by torch's int8 GEMM (int8_exact.int_mm) in place of
h1o_ref's v3 z:
  * gate: z on every slot row of every call (rows >= m and dead row tiles are A = 0, so z = 0); y equal to today's
    (int8_ops.scaled_mm's body, the exact arm's) on every call, required; three units on the CPU: the kernel's z = A B^T, the
    receipt = H1V3's Definition's (`dither_one.h1v3_unit`) and its own code's (`h1v3_itself`, H1V3's h1tile.reference),
    and v3's z peeled on the CPU from its own noise (`peel_cpu`) = A B^T, where the literal peel z - b B^T - A c^T - b c^T
    is reported (it misses the wrap term 256 xa (B + c)^T); and the kernel's z and receipt equal H1V3 with the peel's
    h1tile.reference(stream=True) itself (this source's, PouwH1Tile_v4 ungated, unless HB_H1V3_PEEL_TREE names another:
    z = A B^T is v3's definition), with no CPU peel; both H1V3 checks are required;
  * fault: z off A B^T by 2^FAULT_SHIFT at unit 0's cell (0, 0) of every call and nowhere else, that unit's receipt off
    h1r's and no other, y = dequant of that z and moved there;
  * fault_wb: every receipt unmoved, z off A B^T only in column 0 of unit 0, by `dither_one.wb_fault_expected` on
    CPU_CALLS and by multiples of 256 elsewhere, y = dequant of that z;
  * fault_xa (the flipped-xa fault): every receipt unmoved, z off A B^T only in row 0 of unit 0, by
    `xa_fault_expected` on CPU_CALLS and by multiples of 256 elsewhere, y = dequant of that z.
"""
from __future__ import annotations

import ctypes as C
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

import dither_fused as DF
import dither_one as DO
from dither_fused import CM, CR, H1R, Call

CAPTURES, LIB_ENV, M32 = DO.CAPTURES + ("fault_xa",), DO.LIB_ENV, DO.M32
CG = None
F_FAULT_XA = 32     # h1u.cu's: xa of unit 0's row 0 at k = 0 flipped (kh_one_peel only)


def install() -> None:
    """The window's step hook raises F_FAULT_XA on the fault_xa capture (dither_one's sets the other captures' flags)."""
    before = DO.DitherState.before

    def before_xa(self: DO.DitherState, m: int) -> None:
        before(self, m)
        if self.cap == "fault_xa":
            self.flags |= F_FAULT_XA
            self.S.copy_(torch.tensor([0, m, 0, self.flags], dtype=torch.int32))

    DO.DitherState.before = before_xa


def xa_fault_expected(a: np.ndarray, b: np.ndarray, salt: bytes, position: int, roots: tuple[bytes, bytes]) -> list[int]:
    """F_FAULT_XA's move of unit 0's z row 0 (rows of `a`, 32 x K; `b` the unit's 32 B rows), mod 2^32: xa at entry
    (0, 0) flipped changes it by 1 - 2 xa, so z[0, j] moves by 256 (1 - 2 xa) B[j, 0], xa = [A = 127 and b = 1]."""
    from verity.core.protocols.pouw.circuit import h1tile as HT
    from verity.core.protocols.pouw.circuit import seeds as SD
    rows, _ = HT.noise(SD.seed(salt, position, *roots), HT.TileShape(32, 32, a.shape[1]))
    d = -1 if int(a[0, 0]) == 127 and int(rows[0][0][0]) == 1 else 1
    return [(256 * d * int(b[j, 0])) & M32 for j in range(32)]


def h1v3_peeled(units: list[dict]) -> dict:
    """`dither_one.h1v3_itself` on the peeled H1V3 tree: its own h1tile.reference(stream=True), whose z is A B^T. The tree
    is this source's own (PouwH1Tile_v4, ungated) unless HB_H1V3_PEEL_TREE names another (hb_served.sh's H1V3_PEEL_SHA)."""
    tree = os.environ.get("HB_H1V3_PEEL_TREE") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    doc = {"tree": tree, "sha": os.environ.get("HB_H1V3_PEEL_SHA") or os.environ.get("RESEARCH_SOURCE_SHA") or "source"}
    with tempfile.TemporaryDirectory() as td:
        src, dst = os.path.join(td, "in.json"), os.path.join(td, "out.json")
        with open(src, "w") as f:
            json.dump(units, f)
        p = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(DO.__file__)), "h1v3_ref.py"), src, dst],
                           env={**os.environ, "PYTHONPATH": tree}, capture_output=True, text=True)
        if p.returncode:
            return {**doc, "error": p.stderr[-2000:]}
        with open(dst) as f:
            return {**doc, **json.load(f)}


def _s32(v: np.ndarray) -> np.ndarray:
    v = v & M32
    return np.where(v >> 31, v - (1 << 32), v)


def peel_cpu(z_v3: np.ndarray, a32: np.ndarray, b32: np.ndarray, salt: bytes, position: int,
             roots: tuple[bytes, bytes]) -> dict:
    """v3's z of one unit (s32 [32, 32]) peeled with its own noise (h1tile.noise): exact = z - X c^T - b B^T + 256 xa
    B^T, literal = z - b B^T - A c^T - b c^T (the peel as first written, which takes z for (A + b)(B + c)^T), xa = [A = 127 and
    b = 1]."""
    from verity.core.protocols.pouw.circuit import h1tile as HT
    from verity.core.protocols.pouw.circuit import seeds as SD
    rows, cols = HT.noise(SD.seed(salt, position, *roots), HT.TileShape(32, 32, a32.shape[1]))
    bits_a = np.array([rows[i][0] for i in range(32)], dtype=np.int64)
    bits_b = np.array([cols[j][0] for j in range(32)], dtype=np.int64)
    a, b, z = a32.astype(np.int64), b32.astype(np.int64), z_v3.astype(np.int64)
    x = DO._sx(a + bits_a)
    xa = ((a == 127) & (bits_a == 1)).astype(np.int64)
    exact = _s32(z - x @ bits_b.T - bits_a @ b.T + 256 * (xa @ b.T))
    literal = _s32(z - bits_a @ b.T - a @ bits_b.T - bits_a @ bits_b.T)
    return {"exact": exact, "literal": literal, "xa_entries": int(xa.sum())}


def check(win: DO.DitherState, lib_h1o: str, which: str = "gate", n_cpu_units: int = 3,
          cpu_calls: tuple = (0, 1, 2, 3, 126, 127)) -> dict:
    from verity_vllm.engine import int8_exact as E
    g = win.caps.get(which, {})
    if "receipts" not in g or "a" not in g:
        return {"ok": False, "complete": False, "why": f"{which} step incomplete: {sorted(g)}"}
    m, s, t, L = g["m"], g["s"], win.t, win.L
    doc: dict = {"which": which, "complete": True, "s": s, "m": m, "t": t, "poison": f"{DF.POISON:#04x}", "epsilon": "seeded",
                 "z_against": "A B^T (int8_exact.int_mm)"}
    key, W = win.key, win.window
    bits = np.arange(t) < m
    wrows = CR.word_rows(CR.words(bits))
    a_cpu, s_cpu, z_cpu = {}, {}, {}
    for w in range(L):
        n, k = win.nk[w]
        q, sc = g["a"][w].cpu().numpy(), g["sa"][w].cpu().numpy()
        fa = {"window": W, "step": s, "linear": w, "module": win.names[w], "slots": t, "k": k, "schema": CM.SCHEMA}
        a_cpu[w] = CM.s8_table(CM.domain(fa, t, "a-table"), key, f"a/{s}/{w}", q, sc).root
        fs = {"window": W, "step": s, "linear": w, "module": win.names[w], "slots": t, "schema": CM.SCHEMA}
        sd = CM.domain(fs, len(wrows), "admit")
        s_cpu[w] = CM.levels(sd, [CM.leaf(sd, key, f"s/{s}/{w}", i, x)[0] for i, x in enumerate(wrows)])[-1][0]
        fz = {"window": W, "step": s, "linear": w, "module": win.names[w], "slots": t, "n": n, "schema": CM.SCHEMA}
        zd = CM.domain(fz, t, "z-table")
        zrows = g["z"][w].cpu().numpy().astype("<i4").view(np.uint8)
        z_cpu[w] = CM.levels(zd, [CM.leaf(zd, key, f"z/{s}/{w}", i, bytes(zrows[i]))[0] for i in range(t)])[-1][0]
    doc["a_roots_bad"] = [w for w in range(L) if a_cpu[w] != g["a_roots"][w]]
    doc["a_bad_diag"] = DO.a_bad_diag(g, doc["a_roots_bad"])
    doc["admit_roots_bad"] = [w for w in range(L) if s_cpu[w] != g["admit_roots"][w]]
    doc["z_roots_bad"] = [w for w in range(L) if z_cpu[w] != g["z_roots"][w]]
    f_step = {"window": W, "step": s, "linears": L}
    a_step = CM.step_root(CM.domain(f_step, L, "a-step"), [a_cpu[w] for w in range(L)])
    admit_step = CM.step_root(CM.domain(f_step, L, "admit-step"), [s_cpu[w] for w in range(L)])
    z_step = CM.step_root(CM.domain(f_step, L, "z-step"), [z_cpu[w] for w in range(L)])
    rnd = CM.step_root(CM.domain({"window": W, "step": s, "leaves": "a-step root, admit-step root, z-step root"}, 3, "round"),
                       [a_step, admit_step, z_step])
    doc["round"] = {"a_step": a_step == g["leaves"][0], "admit_step": admit_step == g["leaves"][1],
                    "z_step": z_step == g["leaves"][2], "root": rnd == g["root"]}
    # receipts (h1r v1) on the saved slots; h1o_ref for wb's density; z against A B^T
    r = H1R(lib_h1o)
    o = C.CDLL(lib_h1o)
    o.h1o_ref_launch.argtypes = [C.c_void_p, C.c_int, C.c_void_p, C.c_void_p, C.c_int, C.c_void_p, C.c_void_p]
    shift = o.h1o_fault_shift()
    wbc = torch.zeros(L, 2, dtype=torch.int64, device=win.dev)
    live = torch.from_numpy((np.arange(t // 32) * 32 < m).astype(np.uint8)).to(win.dev)
    roots = [torch.from_numpy(np.concatenate([np.frombuffer(a_cpu[w], ">u8"), np.frombuffer(g["b_roots"][w], ">u8")])
                              .astype("<u8").view("<i8").copy()).to(win.dev) for w in range(L)]
    plan = r.plan([Call(g["a"][w], win.wt[w], s * win.u_step + win.offs[w], roots[w], live) for w in range(L)])
    r.launch(plan, DO.cap_salt(win, g)[1], version=1)
    cb = r.lib.h1r_call_bytes()
    zv3 = [torch.empty(t, n, dtype=torch.int32, device=win.dev) for n, _ in win.nk]
    for w in range(L):
        rc = o.h1o_ref_launch(plan.table.data_ptr() + cb * w, win.units[w], plan.seeds.data_ptr(), zv3[w].data_ptr(),
                              win.nk[w][0], wbc[w].data_ptr(), torch.cuda.current_stream().cuda_stream)
        if rc:
            raise RuntimeError(f"h1o_ref_launch({w}) returned {rc}")
    zab = [E.int_mm(g["a"][w], win.wt[w]) for w in range(L)]
    torch.cuda.synchronize()
    got = g["receipts"]

    def got_of(w: int) -> torch.Tensor:
        o_, units, nw = win.plan.offs[w]
        return got[o_:o_ + units * nw].view(units, nw)

    rc_bad, z_bad, y_bad, fault_bad, fault_moved, v3_cells = [], [], [], [], [], 0
    for w in range(L):
        rdiff = (plan.receipts(w) != got_of(w)).any(dim=1).nonzero().flatten().tolist()
        dz = (g["z"][w].to(torch.int64) - zab[w].to(torch.int64)) & M32
        cells = dz.nonzero().tolist()
        v3_cells += int((zv3[w][:m] != zab[w][:m]).sum())
        if g["y"] is not None:
            if not torch.equal(E.dequantize(g["z"][w][:m], g["sa"][w][:m], win.sw[w]).view(torch.int16),
                               g["y"][w].view(torch.int16)):
                y_bad.append(w)
        if which == "fault":
            moved = g["y"] is not None and float(g["sa"][w][0]) != 0.0 and not torch.equal(
                E.dequantize(zab[w][:1, :1], g["sa"][w][:1], win.sw[w][:1]).view(torch.int16), g["y"][w][:1, :1].view(torch.int16))
            if not (rdiff == [0] and cells == [[0, 0]] and int(dz[0, 0]) == 1 << shift and moved):
                fault_bad.append({"call": w, "receipt_units_off": rdiff[:4], "z_cells_off": cells[:4],
                                  "dz00": int(dz[0, 0]), "y_moved": bool(moved)})
        elif which == "fault_wb":
            col = dz[:32, 0].tolist()
            if w in cpu_calls:
                exp = DO.wb_fault_expected(g["a"][w][:32].cpu().numpy(), win.wt[w][:32].cpu().numpy(),
                                           g["salt"] if "salt" in g else win.salt_bytes, s * win.u_step + win.offs[w],
                                           (a_cpu[w], g["b_roots"][w]))
                exact = col == exp
            else:
                exact = all(v % 256 == 0 for v in col)
            if not (not rdiff and all(c == 0 and i < 32 for i, c in cells) and exact):
                fault_bad.append({"call": w, "receipt_units_off": rdiff[:4], "z_cells_off": cells[:4],
                                  "dz_col0": col[:4], "exact_checked": w in cpu_calls})
            fault_moved.append(len(cells))
        elif which == "fault_xa":
            row = dz[0, :32].tolist()
            if w in cpu_calls:
                exp = xa_fault_expected(g["a"][w][:32].cpu().numpy(), win.wt[w][:32].cpu().numpy(),
                                        g["salt"] if "salt" in g else win.salt_bytes, s * win.u_step + win.offs[w],
                                        (a_cpu[w], g["b_roots"][w]))
                exact = row == exp
            else:
                exact = all(v % 256 == 0 for v in row)
            if not (not rdiff and all(i == 0 and j < 32 for i, j in cells) and exact):
                fault_bad.append({"call": w, "receipt_units_off": rdiff[:4], "z_cells_off": cells[:4],
                                  "dz_row0": row[:4], "exact_checked": w in cpu_calls})
            fault_moved.append(len(cells))
        else:
            if rdiff:
                rc_bad.append(w)
            if cells:
                z_bad.append(w)
    doc.update(receipts_vs_h1r_bad=rc_bad, z_vs_ref_bad=z_bad, y_vs_dequant_z_bad=y_bad if g["y"] is not None else None)
    doc["v3_cells_ne_today"] = v3_cells
    doc["live_units"] = int(sum((min(m, t) + 31) // 32 * (n // 32) for n, _ in win.nk))
    rb = (min(m, t) + 31) // 32
    cnt = wbc.cpu().numpy()
    ent = np.array([rb * n * k for n, k in win.nk], dtype=np.int64)
    doc["wb"] = {"plus": int(cnt[:, 0].sum()), "minus": int(cnt[:, 1].sum()), "entries": int(ent.sum()),
                 "density": float(cnt.sum() / ent.sum()),
                 "by_site": {site: float(cnt[i::4].sum() / ent[i::4].sum()) for i, site in enumerate(DF.SITES)}}
    a127 = [int((g["a"][w][:m] == 127).sum()) for w in range(L)]
    doc["xa"] = {"entries": sum(a127), "by_call": a127,
                 "by_site": {site: sum(a127[i::4]) for i, site in enumerate(DF.SITES)},
                 "note": "A = 127 entries of each call's live rows (xa needs b = 1 too: about half of them; vs_cpu's "
                         "xa_entries counts both on the units checked on the CPU)"}
    if which in ("fault", "fault_wb", "fault_xa"):
        doc["fault_shift"] = shift
        doc["fault_bad"] = fault_bad
        if which in ("fault_wb", "fault_xa"):
            doc["z_cells_moved"] = {"min": min(fault_moved), "max": max(fault_moved)}
        doc["conforming"] = (not fault_bad and not y_bad and not doc["a_roots_bad"] and all(doc["round"].values())
                             and not doc["z_roots_bad"])
        doc["ok"] = doc["conforming"]
        return doc
    today_ok = True
    if g["y"] is not None and which == "gate":
        eq, rel = 0, []
        for w in range(L):
            # the op's own body, not the op: an -xl arm serves verity_int8::scaled_mm from exactlin
            yt = E.dequantize(E.int_mm(g["a"][w][:m], win.wt[w]), g["sa"][w][:m], DO.REG[w][1].weight_scale)
            yk = g["y"][w]
            same = torch.equal(yt.view(torch.int16), yk.view(torch.int16))
            eq += int(same)
            rel.append(float((yk.float() - yt.float()).norm() / yt.float().norm().clamp_min(1e-30)))
        doc["y_vs_today"] = {"calls_equal": eq, "calls": L, "rel_l2_median": float(np.median(rel)),
                             "rel_l2_max": float(np.max(rel))}
        today_ok = eq == L
    units_doc, h1v3_in, kernel_out = [], [], []
    hv: dict = {}
    if which == "gate":
        rng = np.random.default_rng(s)
        salt = g["salt"] if "salt" in g else win.salt_bytes
        for w in [0, 3, 2][:n_cpu_units]:
            n, k = win.nk[w]
            u = int(rng.integers(n // 32))
            a32 = g["a"][w][0:32].cpu().numpy()
            b32 = win.wt[w][32 * u:32 * u + 32].cpu().numpy()
            pos = s * win.u_step + win.offs[w] + u
            ref = DO.h1v3_unit(a32, b32, salt, pos, (a_cpu[w], g["b_roots"][w]))
            pc = peel_cpu(ref["z"], a32, b32, salt, pos, (a_cpu[w], g["b_roots"][w]))
            zk = g["z"][w][0:32, 32 * u:32 * u + 32].cpu().numpy().astype(np.int64)
            rk = (got_of(w)[u].cpu().numpy().astype(np.int64) & M32).tolist()
            units_doc.append({"call": w, "unit": u, "k": k, "z_equal_today": bool((zk == ref["today"]).all()),
                              "receipt_equal": rk == ref["receipt"],
                              "v3_cells_eq_x_bc": int((ref["z"] == ref["xw"]).sum()),
                              "v3_peeled_exact_eq_today": int((pc["exact"] == ref["today"]).sum()),
                              "v3_peeled_literal_eq_today": int((pc["literal"] == ref["today"]).sum()),
                              "xa_entries": pc["xa_entries"], "a_eq_127": int((a32 == 127).sum())})
            h1v3_in.append({"a": (a32.astype(np.int64) & 0xFF).tolist(), "b": (b32.astype(np.int64) & 0xFF).tolist(),
                            "salt": salt.hex(), "position": int(pos), "roots": [a_cpu[w].hex(), g["b_roots"][w].hex()]})
            kernel_out.append((zk, rk, a32, b32, salt, pos, (a_cpu[w], g["b_roots"][w])))
        hv = DO.h1v3_itself(h1v3_in)
        if "units" in hv:
            for x, (zk, rk, a32, b32, salt_, pos, rts), hu in zip(units_doc, kernel_out, hv["units"]):
                hz = np.array(hu["z"], dtype=np.int64)
                # a tree with the peel (its peel_seg) states z = A B^T itself; v3 before it, X (B + c)^T, is peeled here
                pz = hz if hv.get("peeled") else peel_cpu(hz, a32, b32, salt_, pos, rts)["exact"]
                x["h1v3_peeled_equal"] = bool((zk == pz).all())
                x["h1v3_receipt_equal"] = rk == hu["receipt"]
        doc["h1v3"] = {k_: v for k_, v in hv.items() if k_ != "units"}
        hp = h1v3_peeled(h1v3_in)
        if "units" in hp:
            for x, (zk, rk, *_), hu in zip(units_doc, kernel_out, hp["units"]):
                x["h1v3_peel_z_equal"] = bool((zk == np.array(hu["z"], dtype=np.int64)).all())
                x["h1v3_peel_receipt_equal"] = rk == hu["receipt"]
        doc["h1v3_peel"] = {k_: v for k_, v in hp.items() if k_ != "units"}
    doc["vs_cpu"] = units_doc
    h1v3_ok = which != "gate" or (
        "units" in hv and all(x["h1v3_peeled_equal"] and x["h1v3_receipt_equal"] for x in units_doc)
        and "units" in hp and all(x["h1v3_peel_z_equal"] and x["h1v3_peel_receipt_equal"] for x in units_doc))
    doc["ok"] = (not rc_bad and not z_bad and (g["y"] is None or not y_bad) and today_ok
                 and all(x["z_equal_today"] and x["receipt_equal"] and x["v3_cells_eq_x_bc"] == 1024
                         and x["v3_peeled_exact_eq_today"] == 1024 for x in units_doc)
                 and h1v3_ok and not doc["a_roots_bad"] and not doc["admit_roots_bad"] and not doc["z_roots_bad"]
                 and all(doc["round"].values()))
    return doc


def gate_verdict(gate: dict, control: dict, fault: dict, fault_wb: dict, fault_xa: dict) -> dict:
    """`dither_one.gate_verdict`, and the flipped xa conforms (only row 0 of unit 0's z and y moved, by 256 (1 - 2 xa) B,
    no receipt)."""
    v = DO.gate_verdict(gate, control, fault, fault_wb)
    v["fault_xa_conforming"] = bool(fault_xa.get("conforming"))
    v["ok"] = v["ok"] and v["fault_xa_conforming"]
    return v
