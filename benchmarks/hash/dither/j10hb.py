"""J10 banded noise on one H200 or B200 at kernel level (kernels-j10; measured only, nothing here is served or credited):
the credited one-stream call (kh_one_peel's two-launch form: h1s_noise into the workspace, then h1u, or h1k in a K-split
build; z = A B^T) on per-unit noise, against the same call on J10 banded noise at bands of g column tiles
(kh_one_peel_band: h1j_noise hashes each band's row blocks once, into its leader's workspace region, and every unit's
columns; the units read their rows from the leader's region), gated against CA's banded reference (#2866,
`banded.banded_reference` under the tree `j10b_ref.ca_tree` finds: send banded-1a4d1de024.tar.gz) and labelled
`j10b_ref.CA_LABEL`.

    arms  NAME-osp              per-unit noise (kh_one_peel), the baseline
          NAME-osp-noise        its noise launch alone (kh_one_peel_band, band 1, mode 2), seeds included
          NAME-osp-units        its unit launch alone (band 1, mode 1), seeds included
          NAME-osp-bG           banded at g = G column tiles a band (G in --gs, 64 and 16 by default)
          NAME-osp-bG-noise     its noise launch alone (mode 2): the noise share, one way
          NAME-osp-bG-units     its unit launch alone (mode 1)
    gate  before any timing, per shape and gated arm: h1zbench.gate_one (0xA5 over receipts, z, zo, y and the workspace;
          z = A B^T on the selected row blocks against float64; y = dequantize(z); the no-write control refused;
          F_FAULT, F_FAULT_WB and F_FAULT_XA conforming on unit 0, a band leader). The per-unit arm's receipts against
          receipts_ca on its picks. A banded arm's receipts and z: unit 0 and non-leaders (the first, the last, one
          drawn) against CA's banded_reference itself (without CA's tree the gate fails), every band leader's receipt
          equal to the per-unit arm's (a leader's noise is its own) and every non-leader's different; its workspace with no 0xA5 block left in a leader's rows or in any unit's columns, and drawn blocks
          against seeds.noise_block of their unit's seed_j.
    time  CUDA graphs replayed between events, the arms interleaved (h1zbench.bench's loop).

    python j10hb.py --lib NAME=h1u_ka.so [--lib ...] --ref-lib h1zref.so --out OUT [--ms 1,32,256,2048,8192] [--gs 64,16]
"""
from __future__ import annotations

import argparse
import copy
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
import h1zbench as Z
import j10b_ref as JB
from h1bench import auto_j, plain_arms
from h1ubench import Call, shape_ref, workspace
from hb_kbench import (
    LINEARS,
    MS,
    POISON,
    Clock,
    capture,
    inputs,
    one_call_ms,
    w_seed,
    x_seed,
)

from verity.core.protocols.pouw.circuit import seeds as SD

SCHEMA = "kernels-j10/hb/v1"


class BandLib(Z.OneLib):
    """A build's kh_one_peel_band: bands of `band` column tiles; mode 0 the call, 1 the units alone, 2 the noise alone."""

    def __init__(self, path: str, band: int, mode: int):
        super().__init__(path, ws=True, peel=True)
        vp, u32 = ctypes.c_void_p, ctypes.c_uint
        self.h.kh_one_peel_band.argtypes = [vp, vp, u32, u32, u32, vp, vp, vp, vp, u32, vp, vp, vp, vp, vp, vp, vp, u32,
                                            u32, ctypes.c_int, vp]
        self.band, self.mode = band, mode

    @staticmethod
    def has_band(path: str) -> bool:
        return hasattr(ctypes.CDLL(path), "kh_one_peel_band")

    def one(self, oc: Z.OneCall) -> None:
        c, n, k = oc.c, oc.c.b.shape[0], oc.c.a.shape[1]
        rows, st = c.a.shape[0], torch.cuda.current_stream().cuda_stream
        j = auto_j(c.units, int(self.h.kh_noise_blocks(k)), torch.cuda.get_device_properties(0).multi_processor_count)
        ws = workspace(int(self.h.kh_ws_bytes(rows, n, k))).data_ptr()
        rc = self.h.kh_one_peel_band(c.a.data_ptr(), c.b.data_ptr(), rows, n, k, c.seeds.data_ptr(), None, c.out.data_ptr(),
                                     ws, j, oc.z.data_ptr(), oc.zo.data_ptr(), oc.y.data_ptr(), None, oc.sa.data_ptr(),
                                     oc.sw.data_ptr(), oc.step.data_ptr(), oc.M, self.band, self.mode, st)
        if rc:
            raise RuntimeError(f"kh_one_peel_band(band {self.band}, mode {self.mode}) rc {rc}")


def leaders(S: int, ncb: int, g: int) -> list[int]:
    return [u for u in range(S) if (u % ncb) % g == 0]


def ws_check(lib: Z.OneLib, c: Call, g: int, rng: random.Random, draws: int = 24) -> dict:
    """The workspace of the last call: no 0xA5 block in a leader's rows or any unit's columns (non-leaders' rows are
    never written), and drawn blocks equal to seeds.noise_block of their unit's seed_j."""
    rows, n, k = c.a.shape[0], c.b.shape[0], c.a.shape[1]
    S, ncb, rb = c.units, n // 32, k // 512 + 1
    nb = 64 * rb
    ws = workspace(int(lib.h.kh_ws_bytes(rows, n, k))).view(torch.int32)[:16 * nb * S].view(S, nb, 16)
    lead = torch.tensor([(u % ncb) % g == 0 for u in range(S)], device="cuda")
    pr = pc = 0
    for s0 in range(0, S, 2048):
        p = (ws[s0:s0 + 2048] == Z.A5).all(dim=-1)
        pr += int(p[:, :32 * rb][lead[s0:s0 + 2048]].sum())
        pc += int(p[:, 32 * rb:].sum())
    sample = [(0, 0), (S - 1, nb - 1)]
    for _ in range(draws):
        u = rng.randrange(S)
        sample.append((u, rng.randrange(0 if (u % ncb) % g == 0 else 32 * rb, nb)))
    got = ws[torch.tensor([u for u, _ in sample], device="cuda"), torch.tensor([t for _, t in sample], device="cuda")]
    got = got.cpu().numpy().view(np.uint32)
    bad = sum(got[i].tolist() != SD.halves(SD.noise_block(SD.seed(c.salt, c.base + u, *c.roots), t))
              for i, (u, t) in enumerate(sample))
    return {"leader_row_poison_blocks": pr, "col_poison_blocks": pc, "blocks_checked": len(sample), "blocks_bad": bad,
            "ok": pr == 0 and pc == 0 and bad == 0}


def ca_ref(c: Call, picks: list[int], g: int, workers: int) -> tuple[dict, float]:
    """CA's banded_reference on the picked units of call c: JB.ca_reference's doc."""
    ncb = c.b.shape[0] // 32
    a8 = c.a.cpu().numpy().astype(np.int64) & 0xFF
    b8 = c.b.cpu().numpy().astype(np.int64) & 0xFF
    cells = []
    for u in picks:
        ri, cj = divmod(u, ncb)
        cells.append({"a": a8[32 * ri:32 * ri + 32].tolist(), "b": b8[32 * cj:32 * cj + 32].tolist(), "salt": c.salt.hex(),
                      "position": c.base + u, "roots": [c.roots[0].hex(), c.roots[1].hex()], "I": ri, "J": cj, "g": g,
                      "row_positions": [c.base + ri * ncb + j for j in range(ncb)]})
    t0 = time.time()
    return JB.ca_reference(cells, workers), round(time.time() - t0, 2)


def gate_band(oc: Z.OneCall, per_out: torch.Tensor, g: int, ref_ab: dict, shift: int, rng: random.Random,
              workers: int) -> dict:
    c = oc.c
    S, ncb = c.units, c.b.shape[0] // 32
    non = [u for u in range(S) if (u % ncb) % g]
    if not non:
        return {"ok": False, "error": f"no non-leader at g = {g} with {ncb} column tiles"}
    picks = sorted({0, non[0], non[-1], rng.choice(non)})
    ca, cpu_s = ca_ref(c, picks, g, workers)
    if "units" not in ca:
        return {"ok": False, "picks": picks, "ca_reference": {x: v for x, v in ca.items() if x != "units"},
                "error": "CA's banded reference did not run (send banded-1a4d1de024.tar.gz)"}
    ref_t = torch.from_numpy(np.array([x["receipt"] for x in ca["units"]], dtype=np.uint32).view(np.int32)).cuda()
    doc = Z.gate_one(oc, picks, ref_t, ref_ab, shift)
    doc["ca_reference"] = {"sha": ca["sha"], "banded": ca.get("banded"), "noise": ca.get("noise"), "cpu_s": cpu_s,
                           "band_starts": [x["band_start"] for x in ca["units"]]}
    g0 = oc.once(0)
    doc["ca_z_bad_cells"] = 0
    for u, x in zip(picks, ca["units"]):
        ri, cj = divmod(u, ncb)
        want = torch.tensor(x["z"], dtype=torch.int32, device="cuda")
        doc["ca_z_bad_cells"] += int((g0["z"][32 * ri:32 * ri + 32, 32 * cj:32 * cj + 32] != want).sum())
    ld = torch.tensor(leaders(S, ncb, g), device="cuda")
    nl = torch.tensor(non, device="cuda")
    doc["leaders"] = len(ld)
    doc["leader_receipt_bad_units"] = int((g0["out"][ld] != per_out[ld]).any(dim=1).sum())
    doc["nonleader_equal_per_unit"] = int((g0["out"][nl] == per_out[nl]).all(dim=1).sum())
    doc["ws"] = ws_check(oc.lib, c, g, rng)
    doc["ok"] = (doc["ok"] and doc["ca_z_bad_cells"] == 0 and doc["leader_receipt_bad_units"] == 0
                 and doc["nonleader_equal_per_unit"] == 0 and doc["ws"]["ok"])
    return doc


def runner(oc: Z.OneCall):
    return lambda i: oc.run()


def bench(builds: dict, ref: Z.Ref, args, name: str, n: int, k: int, m: int, Xh: np.ndarray, clock: Clock, l2: int,
          rng: random.Random, shift: int, gs: list[int]) -> dict:
    X = torch.from_numpy(Xh[:m]).to("cuda")
    a = X if m >= 32 else torch.cat([X, X.new_zeros(32 - m, k)])
    rows = a.shape[0]
    b = torch.from_numpy(inputs(w_seed(name), n, k)).to("cuda")
    gen = torch.Generator(device="cuda").manual_seed(rng.getrandbits(32))
    sa = torch.zeros(rows, device="cuda")
    sa[:m] = torch.rand(m, device="cuda", generator=gen) * 0.02 + 1e-3
    sw = torch.rand(n, device="cuda", generator=gen) * 2e-3 + 1e-4
    first = Z.OneLib(next(iter(builds.values())), peel=True)
    call = Call(first, a.contiguous(), b, rng.randbytes(64), rng.randbytes(64), rng.randbytes(64), rng.getrandbits(40))
    S = call.units
    out: dict = {"linear": name, "m": m, "n": n, "k": k, "units": S, "column_tiles": n // 32, "padded_row_block": m < 32,
                 "arms": {}, "gates": {}}
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
    zab = torch.full((rows, n), POISON, dtype=torch.int32, device="cuda")
    zab[rf["rows"]] = (a[rf["rows"]].double() @ b.double().T).to(torch.int32)
    rf_ab = {**rf, "z": zab}
    timed: dict = {}
    for bname, path in builds.items():
        lib = first if bname == next(iter(builds)) else Z.OneLib(path, peel=True)
        oc = Z.OneCall(lib, call.twin(lib), m, sa, sw)
        arm = f"{bname}-osp"
        out["gates"][arm] = g = Z.gate_one(oc, picks, ref_t, rf_ab, shift)
        print(json.dumps({"linear": name, "m": m, "arm": arm, "gate": {x: g[x] for x in g if x != "picks"}}), flush=True)
        if not g["ok"]:
            continue
        per_out = oc.once(0)["out"]
        timed[arm] = oc
        for sfx, band, mode in (("-noise", 1, 2), ("-units", 1, 1)):
            o2 = copy.copy(oc)
            o2.lib = BandLib(path, band, mode)
            timed[arm + sfx] = o2
        for gb in gs:
            barm = f"{arm}-b{gb}"
            bo = copy.copy(oc)
            bo.lib = BandLib(path, gb, 0)
            out["gates"][barm] = g = gate_band(bo, per_out, gb, rf_ab, shift, rng, args.workers)
            print(json.dumps({"linear": name, "m": m, "arm": barm, "gate": {x: g[x] for x in g if x != "picks"}}),
                  flush=True)
            if not g["ok"]:
                continue
            timed[barm] = bo
            for sfx, mode in (("-noise", 2), ("-units", 1)):
                o2 = copy.copy(oc)
                o2.lib = BandLib(path, gb, mode)
                timed[barm + sfx] = o2
    failed = [arm for arm, g in out["gates"].items() if not g["ok"]]
    if failed:
        out["gate_failed"] = failed
    if not timed:
        out["error"] = "every gate failed: nothing timed"
        return out
    arms: dict = {arm: runner(oc) for arm, oc in timed.items()}
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
    if plain:
        fp = min(plain, key=plain.get)
        out.update({"fastest_plain": fp, "plain_us": plain[fp]})
    out["over_per_unit"] = {}
    for arm in timed:
        base = arm.split("-b")[0] if "-b" in arm else arm.removesuffix("-noise").removesuffix("-units")
        if base in med and arm in med:
            out["over_per_unit"][arm] = round(med[arm] / med[base], 4)
    del graphs, timed, call
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--ref-lib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--item-ms", type=float, default=3.0)
    ap.add_argument("--picks", type=int, default=14)
    ap.add_argument("--gs", default="64,16")
    ap.add_argument("--ms", default=",".join(map(str, MS)))
    ap.add_argument("--linears", default=",".join(LINEARS))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    builds = {}
    for v in args.lib:
        bname, _, path = v.partition("=")
        if not BandLib.has_band(path):
            print(f"j10hb: {path} has no kh_one_peel_band: rebuild h1u.cu at this branch", file=sys.stderr)
            return 2
        builds[bname] = path
    gs = [int(v) for v in args.gs.split(",")]
    ref = Z.Ref(args.ref_lib)
    shift = int(ctypes.CDLL(next(iter(builds.values()))).kh_fault_shift())
    clock = Clock()
    props = torch.cuda.get_device_properties(0)
    l2 = int(getattr(props, "L2_cache_size", 50 << 20))
    rng = random.Random(0x4A3110)
    ca = JB.ca_tree()
    doc = {"schema": SCHEMA, "label": JB.CA_LABEL if ca else JB.LABEL, "ca_tree": ca, "noise": JB.NOISE,
           "device": clock.name,
           "sms": props.multi_processor_count, "l2_bytes": l2, "torch": torch.__version__,
           "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "builds": builds, "ref_lib": args.ref_lib,
           "gs": gs, "fault_shift": shift,
           "reference": "per-unit: receipts_ca.py, h1zref.cu (z = A B^T in float64); banded: CA's banded_reference "
                        f"(#2866 @ {JB.CA_SHA}) on 4 units a shape and arm, the leaders against the per-unit arm",
           "timing": "seeds (kh_seeds) included in every arm; -noise and -units arms are the two launches alone",
           "shapes": []}
    xs: dict = {}
    ms = [int(v) for v in args.ms.split(",")]
    for name in args.linears.split(","):
        n, k = LINEARS[name]
        if k not in xs:
            xs[k] = inputs(x_seed(k), max(ms), k)
        for m in ms:
            t0 = time.time()
            try:
                r = bench(builds, ref, args, name, n, k, m, xs[k], clock, l2, rng, shift, gs)
            except Exception as e:  # noqa: BLE001
                r = {"linear": name, "m": m, "error": "".join(traceback.format_exception(e))[-3000:]}
            r["wall_s"] = round(time.time() - t0, 2)
            doc["shapes"].append(r)
            print(json.dumps({x: r.get(x) for x in ("linear", "m", "units", "plain_us", "over_per_unit",
                                                     "gate_failed", "error", "wall_s")}), flush=True)
            (out / "j10hb.json").write_text(json.dumps(doc, indent=1))
    layer: dict = {}
    for r in doc["shapes"]:
        for arm, v in r.get("arms", {}).items():
            d = layer.setdefault(r["m"], {}).setdefault(arm, {"us": 0.0, "linears": []})
            d["us"] += v["median_us"]
            d["linears"].append(r["linear"])
    for m, d in layer.items():
        for arm, v in d.items():
            v["us"] = round(v["us"], 3)
            base = arm.split("-b")[0] if "-b" in arm else arm.removesuffix("-noise").removesuffix("-units")
            if base in d and d[base]["linears"] == v["linears"] and arm not in ("int_mm", "cutlass"):
                v["over_per_unit"] = round(v["us"] / d[base]["us"], 4)
    doc["per_layer"] = layer
    doc["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (out / "j10hb.json").write_text(json.dumps(doc, indent=1))
    print(json.dumps({"per_layer": {m: {a: v.get("over_per_unit") for a, v in d.items()} for m, d in layer.items()}}))
    bad = [r for r in doc["shapes"] if "error" in r or "gate_failed" in r]
    return 4 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
