"""Iteration 5: the one-stream form of H1 + fx1 (h1u.cu's kh_one, CA's PouwH1Tile_v3 with the exact wrap correction) on one
H200 or B200. The call's served z and y come from the MMAs whose words its receipts fold, so the call costs this kernel
alone (seeds, noise, unit), against the plain int8 GEMM it replaces: gamma = os / plain. Each build's receipts-only
kh_unit_ws is timed beside it (arm NAME-ws, a second product: gamma 1 + ws / plain), and so is H1's noise floor.

    python h1zbench.py --lib h1u.so [--variant NAME=h1u_x.so ...] --ref-lib h1zref.so [--noise-lib h1noise.so]
                       --out OUT [--reps 5] [--ms ...] [--linears ...] [--fused]

Gates before any timing (a failed gate exits 4 and its build is not timed):
  * anchors (CPU): every build's kh_one, fused and two-launch, against h1v3_ca.reference(stream=True) (v3's words,
    copied verbatim) on every unit of a 64 x 64 x 512 call whose second row block is dead and of a 32 x 32 x 4608 call
    (m = 3), z and receipt, B over all of int8 (-128 included); the reference kernel (h1zref.cu) against the same z;
  * every timed shape, 0xA5 poison over receipts, z, zo, y and the workspace first:
    - the receipts on sampled units against receipts.py's call_receipts (receipts_ca.py), the seeds against seeds.py;
    - z against h1zref (X (B + c)^T by dp4a from full compressions) on every cell of the selected row blocks (all of
      them up to 8, else the first, the last and 6 random), no poison left in z, zo = z's rows < M, and y =
      int8_exact.dequantize(z) bit for bit;
    - the no-write control (F_NW_TILES | F_NW_UNITS) must be refused, on z and on the receipts;
    - F_FAULT (2^24 into unit 0's cell (0, 0) after segment 0) must move that cell's z by exactly 2^24, unit 0's
      receipt and no other, nothing else in z, and y = dequantize(z);
    - F_FAULT_WB (unit 0's column-0 wb+ at k = 0 flipped) must move exactly the cells (i, 0) of unit 0 with X[i][0] != 0,
      each by 256 dWB X[i][0], no receipt, nothing else in z, and y = dequantize(z).
The wrap bits' density (wb+ and wb- over the checked units' entries, from h1zref) is recorded per shape, never assumed.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import random
import statistics
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))
import h1v3_ca as V3
from h1bench import KEY_BLOCKS, auto_j, plain_arms
from h1bench import Lib as NoiseLib
from h1ubench import Call, Lib, floor_gate, gate_seeds, shape_ref, workspace
from hb_kbench import LINEARS, MS, POISON, Clock, capture, inputs, one_call_ms, w_seed, x_seed

from verity.core.protocols.pouw.circuit import h1tile as HT

F_DUMP, F_NW_TILES, F_NW_UNITS, F_FAULT, F_FAULT_WB = 1, 2, 4, 8, 16
M32 = (1 << 32) - 1
SEED_WORDS = 4


def dequantize(z: torch.Tensor, s_a: torch.Tensor, s_w: torch.Tensor) -> torch.Tensor:
    """int8_exact.dequantize: RN_bf16(RN32(RN32(z) s_a[i]) s_w[n])."""
    return (z.to(torch.float32) * s_a[:, None] * s_w[None, :]).to(torch.bfloat16)


class OneLib(Lib):
    """A build's kh_one: fused (ws False) or two-launch."""

    def __init__(self, path: str, ws: bool = True):
        super().__init__(path, ws=True)
        vp, u32 = ctypes.c_void_p, ctypes.c_uint
        self.h.kh_one.argtypes = [vp, vp, u32, u32, u32, vp, vp, vp, vp, u32, vp, vp, vp, vp, vp, vp, vp, u32, vp]
        self.two = ws

    @staticmethod
    def has_one(path: str) -> bool:
        return hasattr(ctypes.CDLL(path), "kh_one")

    def one(self, oc: "OneCall") -> None:
        c, n, k = oc.c, oc.c.b.shape[0], oc.c.a.shape[1]
        rows, st = c.a.shape[0], torch.cuda.current_stream().cuda_stream
        ws, j = None, 0
        if self.two:
            nb = int(self.h.kh_noise_blocks(k))
            j = auto_j(c.units, nb, torch.cuda.get_device_properties(0).multi_processor_count)
            ws = workspace(int(self.h.kh_ws_bytes(rows, n, k))).data_ptr()
        rc = self.h.kh_one(c.a.data_ptr(), c.b.data_ptr(), rows, n, k, c.seeds.data_ptr(), None, c.out.data_ptr(), ws, j,
                           oc.z.data_ptr(), oc.zo.data_ptr(), oc.y.data_ptr(), None, oc.sa.data_ptr(), oc.sw.data_ptr(),
                           oc.step.data_ptr(), oc.M, st)
        if rc:
            raise RuntimeError(f"kh_one rc {rc}")


class OneCall:
    """One call in the one-stream form: a Call (A padded to whole row blocks, B, seeds, receipts) and z [rows, N], zo and
    y [M, N], s_a [rows] (0 past M), s_w [N] and the step {0, M, 0, flags} on the device."""

    def __init__(self, lib: OneLib, c: Call, M: int, sa: torch.Tensor, sw: torch.Tensor):
        self.lib, self.c, self.M, self.sa, self.sw = lib, c, M, sa, sw
        rows, n = c.a.shape[0], c.b.shape[0]
        self.z = torch.empty((rows, n), dtype=torch.int32, device="cuda")
        self.zo = torch.empty((M, n), dtype=torch.int32, device="cuda")
        self.y = torch.empty((M, n), dtype=torch.bfloat16, device="cuda")
        self.step = torch.tensor([0, M, 0, 0], dtype=torch.int32, device="cuda")

    def run(self) -> None:
        self.c.lib.seeds(self.c.data, self.c.base, self.c.units, self.c.seeds)
        self.lib.one(self)

    def poison(self) -> None:
        self.c.poison()
        for t in (self.z, self.zo, self.y):
            t.view(torch.uint8).fill_(0xA5)
        if self.lib.two:
            k, n = self.c.a.shape[1], self.c.b.shape[0]
            workspace(int(self.lib.h.kh_ws_bytes(self.c.a.shape[0], n, k))).fill_(0xA5)

    def once(self, flags: int) -> dict:
        self.poison()
        self.step[3] = flags
        self.run()
        torch.cuda.synchronize()
        self.step[3] = 0
        return {"out": self.c.out[:self.c.units].clone(), "z": self.z.clone(), "zo": self.zo.clone(), "y": self.y.clone()}


class Ref:
    """h1zref.cu: z = X (B + c)^T at the selected row blocks, the wrap counts, unit 0's probe."""

    def __init__(self, path: str):
        h = self.h = ctypes.CDLL(path)
        vp, u32 = ctypes.c_void_p, ctypes.c_uint
        h.hz_ref.argtypes = [vp, vp, u32, u32, u32, vp, vp, vp, vp, vp, vp]

    def run(self, c: Call, rsel: list[int]) -> dict:
        rows, n, k = c.a.shape[0], c.b.shape[0], c.a.shape[1]
        sel = torch.zeros(rows // 32, dtype=torch.uint8, device="cuda")
        sel[rsel] = 1
        z = torch.full((rows, n), POISON, dtype=torch.int32, device="cuda")
        counts = torch.zeros(3, dtype=torch.int64, device="cuda")
        probe = torch.full((33,), POISON, dtype=torch.int32, device="cuda")
        rc = self.h.hz_ref(c.a.data_ptr(), c.b.data_ptr(), rows, n, k, c.seeds.data_ptr(), sel.data_ptr(), z.data_ptr(),
                           counts.data_ptr(), probe.data_ptr(), torch.cuda.current_stream().cuda_stream)
        if rc:
            raise RuntimeError(f"hz_ref rc {rc}")
        torch.cuda.synchronize()
        entries = len(rsel) * (n // 32) * 32 * k
        wbp, wbm, xw = (int(v) for v in counts.tolist())
        return {"z": z, "rows": torch.cat([torch.arange(32 * r, 32 * r + 32) for r in rsel]).cuda(), "probe": probe.cpu(),
                "wb_plus": wbp, "wb_minus": wbm, "x_wrap": xw, "entries": entries,
                "wb_density": round((wbp + wbm) / entries, 6), "x_wrap_density": round(xw / (len(rsel) * 32 * k * (n // 32)), 6)}


def zdiff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.to(torch.int64) - b.to(torch.int64)) & M32


def gate_one(oc: OneCall, picks: list[int], ref_t: torch.Tensor, ref: dict, shift: int) -> dict:
    c, M = oc.c, oc.M
    pk = torch.tensor(picks, device="cuda")
    rows = ref["rows"]
    sa = oc.sa[:M]

    def y_ok(g: dict) -> bool:
        return torch.equal(dequantize(g["z"][:M], sa, oc.sw).view(torch.int16), g["y"].view(torch.int16))

    g0 = oc.once(0)
    doc = {"units": c.units, "picks": picks,
           "receipt_bad_words": int((g0["out"][pk] != ref_t).sum()), "receipt_poison_left": int((g0["out"] == POISON).sum()),
           "seed_bad": gate_seeds(c, picks), "z_bad_cells": int((g0["z"][rows] != ref["z"][rows]).sum()),
           "z_checked_cells": int(rows.numel() * g0["z"].shape[1]),
           "z_poison_left": int((g0["z"] == POISON).sum()), "zo_equal": bool(torch.equal(g0["zo"], g0["z"][:M])),
           "y_equal_dequant": y_ok(g0), "y_poison_left": int((g0["y"].view(torch.int16) == -23131).all(dim=-1).sum())}
    doc["clean"] = (doc["receipt_bad_words"] == 0 and doc["receipt_poison_left"] == 0 and doc["seed_bad"] == 0
                    and doc["z_bad_cells"] == 0 and doc["z_poison_left"] == 0 and doc["zo_equal"] and doc["y_equal_dequant"])
    gc = oc.once(F_NW_TILES | F_NW_UNITS)
    doc["control_z_bad"] = int((gc["z"][rows] != ref["z"][rows]).sum())
    doc["control_receipt_bad"] = int((gc["out"][pk] != ref_t).sum())
    doc["control_refused"] = doc["control_z_bad"] > 0 and doc["control_receipt_bad"] > 0
    # F_FAULT: one perturbed word reaches its z, its y and its unit's receipt, and nothing else
    gf = oc.once(F_FAULT)
    cells = (gf["z"] != g0["z"]).nonzero().tolist()
    d00 = int(zdiff(gf["z"][:1, :1], ref["z"][:1, :1]))
    runits = (gf["out"] != g0["out"]).any(dim=1).nonzero().flatten().tolist()
    moved = bool(gf["y"][0, 0].view(torch.int16) != g0["y"][0, 0].view(torch.int16)) if float(sa[0]) else None
    doc["fault"] = {"z_cells": cells[:4], "dz00": d00, "receipt_units": runits[:4], "y_equal_dequant": y_ok(gf),
                    "y00_moved": moved}
    doc["fault"]["ok"] = (cells == [[0, 0]] and d00 == 1 << shift and runits == [0] and doc["fault"]["y_equal_dequant"]
                          and moved is not False)
    # F_FAULT_WB: one flipped wrap bit reaches exactly its column's z (and y) in unit 0, no receipt
    gw = oc.once(F_FAULT_WB)
    pr = ref["probe"].tolist()
    x0, wb0 = pr[:32], pr[32]
    dwb = 1 if wb0 in (0, -1) else -1
    want = {(i, 0): (256 * dwb * x0[i]) & M32 for i in range(32) if x0[i]}
    got_cells = (gw["z"] != g0["z"]).nonzero().tolist()
    got = {(i, j): int(zdiff(gw["z"][i:i + 1, j:j + 1], g0["z"][i:i + 1, j:j + 1])) for i, j in got_cells}
    runits_w = (gw["out"] != g0["out"]).any(dim=1).nonzero().flatten().tolist()
    doc["fault_wb"] = {"wb00": wb0, "dwb": dwb, "cells_want": len(want), "cells_got": len(got),
                       "receipt_units": runits_w[:4], "y_equal_dequant": y_ok(gw)}
    doc["fault_wb"]["ok"] = got == want and len(want) > 0 and not runits_w and doc["fault_wb"]["y_equal_dequant"]
    doc["ok"] = doc["clean"] and doc["control_refused"] and doc["fault"]["ok"] and doc["fault_wb"]["ok"]
    return doc


def anchors(libs: dict[str, OneLib], ref: Ref, rng: random.Random) -> list[dict]:
    """Every one-stream build, and the reference kernel, against v3's words on the CPU (h1v3_ca.reference, stream=True)
    on two small calls, a dead row block included."""
    res = []
    for m, n, k, dead in ((64, 64, 512, (1,)), (32, 32, 4608, ())):
        g = torch.Generator().manual_seed(rng.getrandbits(32))
        a = torch.randint(-128, 128, (m, k), dtype=torch.int8, generator=g)
        b = torch.randint(-128, 128, (n, k), dtype=torch.int8, generator=g)
        for ri in dead:
            a[32 * ri:32 * ri + 32] = 0
        salt, ra, rb = (rng.randbytes(64) for _ in range(3))
        base = rng.getrandbits(40)
        live = torch.tensor([0 if ri in dead else 1 for ri in range(m // 32)], dtype=torch.uint8, device="cuda")
        sa = torch.rand(m, device="cuda") * 0.02 + 1e-3
        sw = torch.rand(n, device="cuda") * 2e-3 + 1e-4
        t0 = time.time()
        want_z = np.zeros((m, n), dtype=np.int64)
        want_r = []
        for u, (ri, cj) in enumerate((ri, cj) for ri in range(m // 32) for cj in range(n // 32)):
            if ri in dead:
                want_r.append([0] * (2 * (k // 32 + int(HT.SG.n_digits(k // 32)))))
                continue
            ar = (a[32 * ri:32 * ri + 32].to(torch.int64) & 0xFF).tolist()
            br = (b[32 * cj:32 * cj + 32].to(torch.int64) & 0xFF).tolist()
            z, r = V3.reference(ar, br, salt, base + u, (ra, rb), stream=True)
            want_z[32 * ri:32 * ri + 32, 32 * cj:32 * cj + 32] = np.array(z, dtype=np.int64).reshape(32, 32)
            want_r.append(r)
        ref_s = round(time.time() - t0, 2)
        want_z = torch.from_numpy(((want_z + (1 << 31)) % (1 << 32) - (1 << 31)).astype(np.int32)).cuda()
        want_r = torch.from_numpy(np.array(want_r, dtype=np.uint32).view(np.int32)).cuda()
        live_rows = torch.cat([torch.arange(32 * ri, 32 * ri + 32) for ri in range(m // 32) if ri not in dead]).cuda()
        for name, lib in libs.items():
            c = Call(lib, a.cuda(), b.cuda(), salt, ra, rb, base)
            oc = OneCall(lib, c, m, sa, sw)
            oc.poison()
            c.lib.seeds(c.data, c.base, c.units, c.seeds)
            rc = lib.h.kh_one(c.a.data_ptr(), c.b.data_ptr(), m, n, k, c.seeds.data_ptr(), live.data_ptr(), c.out.data_ptr(),
                              workspace(int(lib.h.kh_ws_bytes(m, n, k))).data_ptr() if lib.two else None, 1 if lib.two else 0,
                              oc.z.data_ptr(), oc.zo.data_ptr(), oc.y.data_ptr(), None, sa.data_ptr(), sw.data_ptr(),
                              oc.step.data_ptr(), m, torch.cuda.current_stream().cuda_stream)
            torch.cuda.synchronize()
            r = {"lib": name, "m": m, "n": n, "k": k, "dead_row_blocks": list(dead), "rc": rc, "reference_s": ref_s,
                 "z_bad": int((oc.z != want_z).sum()), "receipt_bad": int((c.out[:c.units] != want_r).sum()),
                 "seed_bad": gate_seeds(c, list(range(c.units))),
                 "y_equal_dequant": bool(torch.equal(dequantize(oc.z, sa, sw).view(torch.int16), oc.y.view(torch.int16)))}
            r["ok"] = rc == 0 and r["z_bad"] == 0 and r["receipt_bad"] == 0 and r["seed_bad"] == 0 and r["y_equal_dequant"]
            res.append(r)
            if name == next(iter(libs)):
                rr = ref.run(c, [ri for ri in range(m // 32) if ri not in dead])
                res.append({"lib": "h1zref", "m": m, "n": n, "k": k, "z_bad": int((rr["z"][live_rows] != want_z[live_rows]).sum()),
                            "wb_density": rr["wb_density"], "x_wrap": rr["x_wrap"],
                            "ok": int((rr["z"][live_rows] != want_z[live_rows]).sum()) == 0})
    return res


def bench(libs: dict, wlibs: dict, nlibs: dict, ref: Ref, args, name: str, n: int, k: int, m: int, Xh: np.ndarray,
          clock: Clock, l2: int, sms: int, rng: random.Random, shift: int) -> dict:
    """libs: arm NAME-os -> one-stream build (the main one first); wlibs: arm NAME-ws -> its receipts-only form; nlibs: the
    noise floors."""
    lay = HT.Layout(32, 32, k)
    X = torch.from_numpy(Xh[:m]).to("cuda")
    a = X if m >= 32 else torch.cat([X, X.new_zeros(32 - m, k)])
    rows = a.shape[0]
    b = torch.from_numpy(inputs(w_seed(name), n, k)).to("cuda")
    gen = torch.Generator(device="cuda").manual_seed(rng.getrandbits(32))
    sa = torch.zeros(rows, device="cuda")
    sa[:m] = torch.rand(m, device="cuda", generator=gen) * 0.02 + 1e-3
    sw = torch.rand(n, device="cuda", generator=gen) * 2e-3 + 1e-4
    first = next(iter(libs.values()))
    call = Call(first, a.contiguous(), b, rng.randbytes(64), rng.randbytes(64), rng.randbytes(64), rng.getrandbits(40))
    S = call.units
    out: dict = {"linear": name, "m": m, "n": n, "k": k, "units": S, "padded_row_block": m < 32,
                 "blocks_per_unit": lay.blocks + KEY_BLOCKS + SEED_WORDS, "receipt_words": call.words, "arms": {},
                 "gates": {}}
    picks = sorted({0, S - 1, *rng.sample(range(S), min(S, args.picks))})
    out["picks"] = picks
    ref_t = shape_ref(call, picks)
    call.run()
    torch.cuda.synchronize()
    nrb = rows // 32
    rsel = list(range(nrb)) if nrb <= 8 else sorted({0, nrb - 1, *rng.sample(range(1, nrb - 1), 6)})
    rf = ref.run(call, rsel)
    out["ref"] = {x: rf[x] for x in ("wb_plus", "wb_minus", "x_wrap", "entries", "wb_density", "x_wrap_density")}
    out["ref"]["row_blocks"] = rsel
    ocs = {arm: OneCall(lib, call if lib is first else call.twin(lib), m, sa, sw) for arm, lib in libs.items()}
    for arm, oc in ocs.items():
        out["gates"][arm] = g = gate_one(oc, picks, ref_t, rf, shift)
        print(json.dumps({"linear": name, "m": m, "arm": arm, "gate": {x: g[x] for x in g if x != "picks"}}), flush=True)
    wcs = {arm: call.twin(lib) for arm, lib in wlibs.items()}
    for arm, c in wcs.items():
        from h1ubench import gate_shape
        out["gates"][arm] = g = gate_shape(c, picks, ref_t)
        print(json.dumps({"linear": name, "m": m, "arm": arm, "gate": {x: g[x] for x in g if x != "picks"}}), flush=True)
    failed = [arm for arm, g in out["gates"].items() if not g["ok"]]
    if failed:
        out["gate_failed"] = failed
    main = next(iter(libs))
    if main in failed:
        out["error"] = "gate failed: nothing timed"
        return out
    arms: dict = {arm: (lambda oc: lambda i: oc.run())(oc) for arm, oc in ocs.items() if arm not in failed}
    arms.update({arm: (lambda c: lambda i: c.run())(c) for arm, c in wcs.items() if arm not in failed})
    if nlibs:
        sj = torch.randint(0, 256, (S, 64), dtype=torch.uint8, device="cuda").view(torch.int64)
        J = auto_j(S, lay.blocks, sms)
        out["floor_j"] = J
        for arm, nl in nlibs.items():
            gs = [floor_gate(arm, nl, lay.blocks, J), floor_gate(arm, nl, KEY_BLOCKS, 1)]
            out["gates"][arm] = {"ok": all(g["ok"] for g in gs), "noise": gs}
            if not out["gates"][arm]["ok"]:
                out.setdefault("gate_failed", []).append(arm)
                continue
            snk = torch.empty((nl.items(S, lay.blocks, J) * 32,), dtype=torch.int32, device="cuda")
            ksnk = torch.empty((nl.items(S, KEY_BLOCKS, 1) * 32,), dtype=torch.int32, device="cuda")
            arms[arm] = (lambda nl, snk, ksnk: lambda i: (nl.mid(sj, S, lay.blocks, J, snk, True),
                                                          nl.mid(sj, S, KEY_BLOCKS, 1, ksnk, True)))(nl, snk, ksnk)
    parms, info = plain_arms(name, n, k, m, Xh, l2)
    out.update(info)
    arms.update(parms)
    graphs, calls = {}, {}
    for arm, fn in arms.items():
        t = one_call_ms(fn)
        G = max(1, min(400, round(args.item_ms / max(t, 1e-3))))
        graphs[arm], calls[arm] = capture(fn, G), G
    for gr in graphs.values():
        gr.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    per = {arm: [] for arm in graphs}
    mhz = {arm: [] for arm in graphs}
    order = list(graphs)
    for rep in range(args.reps):
        for arm in order[rep % len(order):] + order[:rep % len(order)]:
            e0.record()
            graphs[arm].replay()
            e1.record()
            e1.synchronize()
            per[arm].append(e0.elapsed_time(e1) * 1e3 / calls[arm])
            mhz[arm].append(clock.sm())
    for arm in graphs:
        out["arms"][arm] = {"calls_per_item": calls[arm], "us": [round(v, 3) for v in per[arm]],
                            "median_us": round(statistics.median(per[arm]), 3), "sm_mhz": mhz[arm]}
    med = {arm: v["median_us"] for arm, v in out["arms"].items()}
    plain = {p: med[p] for p in ("int_mm", "cutlass") if p in med}
    fp = min(plain, key=plain.get)
    out.update({"fastest_plain": fp, "plain_us": plain[fp]})
    out["gamma"] = {arm: round(med[arm] / plain[fp] if arm in ocs else 1 + med[arm] / plain[fp], 4)
                    for arm in med if arm not in plain}
    out["os_over_ws"] = {arm: round(med[arm] / med[arm[:-3] + "-ws"], 4) for arm in ocs
                         if arm in med and arm[:-3] + "-ws" in med}
    del graphs, ocs, wcs, call
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--variant", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--ref-lib", required=True)
    ap.add_argument("--noise-lib")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--item-ms", type=float, default=3.0)
    ap.add_argument("--picks", type=int, default=14)
    ap.add_argument("--fused", action="store_true", help="also time each build's fused one-stream form (arm NAME-osf)")
    ap.add_argument("--no-ws", action="store_true", help="skip the receipts-only arms NAME-ws")
    ap.add_argument("--ms", default=",".join(map(str, MS)))
    ap.add_argument("--linears", default=",".join(LINEARS))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"h1u": args.lib}
    for v in args.variant:
        vname, _, path = v.partition("=")
        paths[f"h1u-{vname}"] = path
    libs, wlibs = {}, {}
    for arm, p in paths.items():
        if not OneLib.has_one(p):
            print(f"h1zbench: {p} has no kh_one", file=sys.stderr)
            return 2
        libs[f"{arm}-os"] = OneLib(p)
        if args.fused:
            libs[f"{arm}-osf"] = OneLib(p, ws=False)
        if not args.no_ws:
            wlibs[f"{arm}-ws"] = Lib(p, ws=True)
    ref = Ref(args.ref_lib)
    nlibs = {"floor": NoiseLib(args.noise_lib)} if args.noise_lib else {}
    shift = int(next(iter(libs.values())).h.kh_fault_shift())
    clock = Clock()
    props = torch.cuda.get_device_properties(0)
    l2 = int(getattr(props, "L2_cache_size", 50 << 20))
    rng = random.Random(0x4831F5)
    doc = {"schema": "dither-hb/h1z/v1", "device": clock.name, "sms": props.multi_processor_count, "l2_bytes": l2,
           "torch": torch.__version__, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "reference": f"h1v3_ca.py ({V3.SOURCE}), receipts_ca.py, h1zref.cu", "noise": "per-unit",
           "libs": paths, "ref_lib": args.ref_lib, "noise_lib": args.noise_lib, "fault_shift": shift, "shapes": []}
    doc["anchors"] = anchors(libs, ref, rng)
    print(json.dumps({"anchors": doc["anchors"]}), flush=True)
    bad = doc["anchor_failed"] = sorted({x["lib"] for x in doc["anchors"] if not x["ok"]})
    (out / "h1z.json").write_text(json.dumps(doc, indent=1))
    main_arm = next(iter(libs))
    if main_arm in bad or "h1zref" in bad:
        print("h1zbench: an anchor failed (the main build or the reference); nothing timed", file=sys.stderr)
        return 4
    for arm in bad:
        print(f"h1zbench: {arm} failed an anchor; it is not timed", file=sys.stderr)
        del libs[arm]
    xs: dict = {}
    ms = [int(v) for v in args.ms.split(",")]
    for name in args.linears.split(","):
        n, k = LINEARS[name]
        if k not in xs:
            xs[k] = inputs(x_seed(k), max(ms), k)
        for m in ms:
            t0 = time.time()
            try:
                r = bench(libs, wlibs, nlibs, ref, args, name, n, k, m, xs[k], clock, l2, props.multi_processor_count, rng,
                          shift)
            except Exception as e:  # noqa: BLE001
                r = {"linear": name, "m": m, "error": "".join(traceback.format_exception(e))[-3000:]}
            r["wall_s"] = round(time.time() - t0, 2)
            doc["shapes"].append(r)
            print(json.dumps({x: r.get(x) for x in ("linear", "m", "units", "plain_us", "fastest_plain", "gamma",
                                                     "os_over_ws", "ref", "gate_failed", "error", "wall_s")}), flush=True)
            (out / "h1z.json").write_text(json.dumps(doc, indent=1))
    layer: dict = {}
    for r in doc["shapes"]:
        if "arms" not in r or "plain_us" not in r:
            continue
        d = layer.setdefault(r["m"], {"plain_us": 0.0, "linears": [], "arms": {}})
        d["plain_us"] += r["plain_us"]
        d["linears"].append(r["linear"])
        for arm, v in r["arms"].items():
            if arm in ("int_mm", "cutlass"):
                continue
            d["arms"].setdefault(arm, {"us": 0.0, "linears": []})
            d["arms"][arm]["us"] += v["median_us"]
            d["arms"][arm]["linears"].append(r["linear"])
    for d in layer.values():
        for arm, v in d["arms"].items():
            full = v["linears"] == d["linears"]
            one = arm.endswith("-os") or arm.endswith("-osf")
            v["gamma"] = round((v["us"] / d["plain_us"]) if one else 1 + v["us"] / d["plain_us"], 4) if full else None
    doc["per_layer"] = layer
    doc["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (out / "h1z.json").write_text(json.dumps(doc, indent=1))
    print(json.dumps({"per_layer": {m: {a: v["gamma"] for a, v in d["arms"].items()} for m, d in layer.items()}}))
    bad_shapes = [r for r in doc["shapes"] if "error" in r or "gate_failed" in r]
    return 4 if bad_shapes or bad else 0


if __name__ == "__main__":
    sys.exit(main())
