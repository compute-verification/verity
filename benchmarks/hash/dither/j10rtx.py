"""J10 banded noise on the RTX PRO 6000 at kernel level: the credited one-stream arm `mix-cpeel` (block2 where a call's
column tiles outnumber the SMs, split elsewhere; peeled, z = A B^T) with per-unit noise, against the same units on banded
noise (h1o.cu's h1o_band_launch: a band-rows pass, then units that hash only their columns), labelled `j10b_ref.CA_LABEL`
with the banded reference's tree, else `j10b_ref.LABEL`. Measured only: nothing here is served or credited.

    arms  mix-cpeel        per-unit noise, h1o.cu's served kernels (the baseline)
          mix-cpeel-b1s    the banded kernels at g = 1: every unit its own leader, so per-unit noise with the rows hashed
                           by the pass (what the rows cost inside the unit: the noise share, one way)
          mix-cpeel-bG     g = G column tiles a band (G in --gs, 64 and 16 by default), block2's rows read from the
                           workspace in its segment loop (groups of 4 chunks, h1o.cu's forms 0 and 3)
          mix-cpeel-bGs    the same with block2's rows staged in its noise buffers (forms 4 and 5); split always stages
          mix-cpeel-bGp    mix-cpeel-bG with the units launched as the pass's programmatic dependents (h1o_band_launch's
                           mode 3: their seeds, midstates and column tiles overlap the pass, which decode's latency is)
    gate  per arm and window, as h1ogate.py: a dither_one.Window over NL layers with the served shapes, a step of m rows
          captured per Window.captures (0xA5 first, the band workspace too) and dither_one.check on each, the gate
          verdict over gate, control, fault, fault_wb and fault_xa. For a banded arm check runs with its receipts
          reference at band G (h1r.cu's h1r_launch_band: v1, full compressions, the rows from the leader's seed_j in the
          seeds buffer), its CPU units on j10b_ref's noise (h1v3_ca's word functions: dither_one.h1v3_unit with h1tile's
          rows taken from the leader), z held to A B^T (peeled), and credit's banded reference itself on the same units, which
          the gate requires (banded.banded_reference under the tree j10b_ref.ca_tree finds, j10ca_ref.py; send
          banded-1a4d1de024.tar.gz). Then the band workspace of the gate
          step against j10b_ref.row_block (the first and last live band of each call whole, up to 384 drawn bands 8
          blocks each) and no 0xA5 block left in a live band.
    time  per arm and m, CUDA graphs replayed between events with the arms interleaved: the window's calls (A table and
          units), the units launches alone, each site's alone; banded arms also the units without the pass (mode 1) and
          the pass alone (mode 2): the noise share, the other way.

    python j10rtx.py --out OUT [--ms 1,32,256,2048,8192] [--gs 64,16] [--reps 15]
        (DITHER_H1C, DITHER_H1O: the libraries; sbench.sh with SBENCH_SCRIPT=j10rtx.py builds them)
"""
from __future__ import annotations

import argparse
import ctypes as C
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))

import dither_fused as DF  # noqa: E402
import dither_one as DO  # noqa: E402
import h1ogate as HG  # noqa: E402
import j10b_ref as JB  # noqa: E402
from h1r import H1R  # noqa: E402

from verity.core.protocols.pouw.circuit import h1tile as HT  # noqa: E402
from verity.core.protocols.pouw.circuit import seeds as SD  # noqa: E402
from verity_config.settings import of  # noqa: E402

settings = of(__file__)
FORM_ID = {"block2": 0, "split": 1, "split2": 2, "block": 3}  # h1o_band_launch's forms: block rows read in the loop
FORM_STAGED = {"block2": 4, "split": 1, "split2": 2, "block": 5}  # every form's rows staged in its noise buffers
FORM_NAMES = ("block2", "split", "split2", "block", "block2-staged", "block-staged")
A5 = -1515870811
NLS = {64: 4, 256: 2, 2048: 1, 8192: 1}  # layers a window, by T


class BandWindow(DO.Window):
    """dither_one's window whose units_call is h1o_band_launch at g (mode 0 the pass and the units, 1 the units, 2 the
    pass), each call with its own band workspace (poisoned with the window's buffers at a captured step)."""

    def __init__(self, t: int, key: bytes, h1c: str, h1o: str, gate: bool, dev, g: int, staged: bool, pdl: bool = False):
        super().__init__(t, key, h1c, h1o, gate, dev)
        if not self.peel:
            raise RuntimeError("the banded kernels are peeled only (DITHER_PEEL=1)")
        o, vp, ci = self.o, C.c_void_p, C.c_int
        o.h1o_band_bytes.restype = C.c_ulonglong
        o.h1o_band_bytes.argtypes = [ci, ci, ci, ci]
        o.h1o_band_launch.argtypes = [ci, ci, vp, vp, ci, ci, ci, vp, vp, ci, vp, vp, vp, vp, vp, vp, ci, vp]
        if o.h1o_band_init():
            raise RuntimeError("h1o_band_init refused")
        self.mode0 = 3 if pdl else 0  # h1o_band_launch's mode 3: the units as the pass's programmatic dependents
        self.g, self.mode = g, self.mode0
        self.staged = staged
        self.form_id = [(FORM_STAGED if staged else FORM_ID)[f] for f in self.call_forms]
        self.bws = [torch.empty(int(o.h1o_band_bytes(self.units[w], n, k, g)), dtype=torch.uint8, device=dev)
                    for w, (n, k) in enumerate(self.nk)]
        self.bws_gate: list | None = None

    def units_call(self, w, xq, s_a, s_w, y, z, st) -> int:
        n, k = self.nk[w]
        return self.o.h1o_band_launch(self.form_id[w], self.mode, self.F.data_ptr() + self.fsize * w, self.S.data_ptr(),
                                      self.units[w], n, k, xq.data_ptr(), s_a.data_ptr(), int(xq.shape[0]), s_w.data_ptr(),
                                      y.data_ptr(), z.data_ptr(), self.salt.data_ptr(), self.step_base.data_ptr(),
                                      self.bws[w].data_ptr(), self.g, st)

    def before(self, m: int) -> None:
        super().before(m)
        if self.cap:
            for b in self.bws:
                b.fill_(DF.POISON)
            torch.cuda.synchronize()

    def after(self) -> None:
        if self.cap == "gate":
            torch.cuda.synchronize()
            self.bws_gate = [b.clone() for b in self.bws]
        super().after()


class BandH1R(H1R):
    """h1r.py's H1R whose version 1 is h1r_launch_band at G."""
    G = 1

    def __init__(self, path: str):
        super().__init__(path)
        self.lib.h1r_launch_band.argtypes = self.lib.h1r_launch.argtypes

    def launch(self, plan, salt, nowrite=False, version=2, step_base=None) -> None:
        if version != 1:
            raise RuntimeError("BandH1R: version 1 only")
        rc = self.lib.h1r_launch_band(plan.table.data_ptr(), len(plan.calls), plan.max_units, plan.ks, salt.data_ptr(),
                                      None if step_base is None else step_base.data_ptr(), plan.seeds.data_ptr(),
                                      plan.out.data_ptr(), int(nowrite), int(self.G), torch.cuda.current_stream().cuda_stream)
        if rc:
            raise RuntimeError(f"h1r_launch_band returned {rc}")


def band_check(win: BandWindow, lib: str, which: str) -> dict:
    """dither_one.check with the banded receipts reference (BandH1R at win.g) and CPU units on j10b_ref's noise."""
    g = win.caps.get(which, {})
    s = g.get("s", 0)
    offs = list(win.offs)
    orig_unit, orig_itself, orig_h1r = DO.h1v3_unit, DO.h1v3_itself, DO.H1R

    def lead_of(pos: int) -> int:
        rel = pos - s * win.u_step
        w = max(i for i in range(win.L) if offs[i] <= rel)
        return s * win.u_step + offs[w] + JB.leader(rel - offs[w], win.nk[w][0] // 32, win.g)

    def unit(a32, b32, salt, pos, roots):
        lead = lead_of(pos)
        seed_l = SD.seed(salt, lead, *roots)
        noise0 = HT.noise

        def noise(seed_j, lay):
            return JB.rows_of(seed_l, lay), JB.cols_of(seed_j, lay)
        HT.noise = noise
        try:
            out = orig_unit(a32, b32, salt, pos, roots)
        finally:
            HT.noise = noise0
        out["band_leader_position"] = lead
        return out

    def itself(units: list[dict]) -> dict:
        cells = []
        for x in units:
            rel = x["position"] - s * win.u_step
            w = max(i for i in range(win.L) if offs[i] <= rel)
            ncb, base = win.nk[w][0] // 32, s * win.u_step + offs[w]
            ri, cj = divmod(rel - offs[w], ncb)
            cells.append({**x, "I": ri, "J": cj, "g": win.g, "row_positions": [base + ri * ncb + j for j in range(ncb)]})
        return JB.ca_reference(cells)

    BandH1R.G = win.g
    DO.H1R, DO.h1v3_unit, DO.h1v3_itself = BandH1R, unit, itself
    try:
        doc = DO.check(win, lib, which)
    finally:
        DO.H1R, DO.h1v3_unit, DO.h1v3_itself = orig_h1r, orig_unit, orig_itself
    if which == "gate":
        cu = doc.get("vs_h1v3_cpu", [])
        doc["ca_reference_ok"] = bool(doc.get("h1v3", {}).get("tree") and not doc["h1v3"].get("error") and cu
                                      and all(x.get("h1v3_z_equal") and x.get("h1v3_receipt_equal") for x in cu))
        doc["ok"] = doc["ok"] and doc["ca_reference_ok"]
    doc["noise"] = JB.NOISE
    doc["band"] = win.g
    doc["cpu_unit_leaders"] = [x.get("unit") for x in doc.get("vs_h1v3_cpu", [])]
    return doc


def noise_check(win: BandWindow, rng: random.Random, max_bands: int = 384) -> dict:
    """The gate step's band workspace against j10b_ref.row_block: in every call, every block of the first and the last
    live band and 8 drawn (row, chunk) blocks of up to max_bands drawn live bands; no 0xA5 block in any live band."""
    cap = win.caps["gate"]
    s, m = cap["s"], cap["m"]
    salt = win.salt_bytes
    bad = checked = poison = bands_checked = 0
    first_bad = None
    for w, (n, k) in enumerate(win.nk):
        ncb, rbk = n // 32, k // 512 + 1
        nl = (ncb + win.g - 1) // win.g
        nrb = win.units[w] // ncb
        live = min(nrb, (m + 31) // 32)
        ws = win.bws_gate[w].view(torch.int32).view(nrb * nl, 32 * rbk, 16)
        poison += int((ws[:live * nl] == A5).all(dim=2).sum())
        roots = (cap["a_roots"][w], cap["b_roots"][w])
        base = s * win.u_step + win.offs[w]
        full = {(0, 0), (live - 1, nl - 1)}
        bands = [(ri, b) for ri in range(live) for b in range(nl)]
        pick = sorted(full | set(rng.sample(bands, min(len(bands), max_bands))))
        idx = torch.tensor([ri * nl + b for ri, b in pick], dtype=torch.long, device=ws.device)
        got_all = ws.index_select(0, idx).cpu().numpy().view(np.uint32)
        for (ri, b), got in zip(pick, got_all):
            bands_checked += 1
            seed_l = SD.seed(salt, base + ri * ncb + b * win.g, *roots)
            pairs = [(i, c) for i in range(32) for c in range(rbk)] if (ri, b) in full else \
                [(rng.randrange(32), rng.randrange(rbk)) for _ in range(8)]
            for i, c in pairs:
                checked += 1
                if got[i * rbk + c].tolist() != JB.row_block(seed_l, i, c, k):
                    bad += 1
                    first_bad = first_bad or {"call": w, "ri": ri, "band": b, "row": i, "chunk": c}
    return {"bands_checked": bands_checked, "blocks_checked": checked, "blocks_bad": bad, "poison_blocks": poison,
            "first_bad": first_bad, "ok": bad == 0 and poison == 0 and checked > 0}


def gate(arm: str, t: int, ms: list[int], dev, rng: random.Random) -> tuple:
    os.environ[DO.UNITS_ENV] = "mix"
    os.environ[DO.PEEL_ENV] = "1"
    h1c, h1o = settings.need("DITHER_H1C"), settings.need(DO.LIB_ENV)
    tail = None if arm == "mix-cpeel" else arm.rsplit("-b", 1)[1]
    g = None if tail is None else int(tail.rstrip("sp"))
    win = DO.Window(t, os.urandom(64), h1c, h1o, True, dev) if g is None else BandWindow(t, os.urandom(64), h1c, h1o,
                                                                                         True, dev, g, "s" in tail,
                                                                                         "p" in tail)
    res = {}
    for m in ms:
        outs = [(torch.empty(m, n, dtype=torch.bfloat16, device=dev), torch.empty(m, n, dtype=torch.int32, device=dev))
                for n, _ in win.nk]
        step(win, m, operands(win, m, 7), outs)
        win.arm(m)
        for i in range(len(win.captures)):
            step(win, m, operands(win, m, 100 + i), outs)
        chk = (lambda n: DO.check(win, h1o, n)) if g is None else (lambda n: band_check(win, h1o, n))
        caps = {n: chk(n) for n in win.captures}
        verdict = DO.gate_verdict(caps["gate"], *(caps[n] for n in win.captures[1:]))
        if g is not None:
            verdict["noise"] = noise_check(win, rng)
            verdict["ok"] = verdict["ok"] and verdict["noise"]["ok"]
        res[m] = {"verdict": verdict, "wb": caps["gate"].get("wb"), "xa": caps["gate"].get("xa"),
                  "vs_h1v3_cpu": caps["gate"].get("vs_h1v3_cpu"), "h1v3": caps["gate"].get("h1v3"),
                  "checks": {n: {x: v for x, v in c.items() if x not in ("vs_h1v3_cpu",)} for n, c in caps.items()}}
    return win, res


def operands(win, m: int, seed: int) -> list:
    gen = torch.Generator(device=win.dev).manual_seed(seed)
    return [(torch.randint(-128, 128, (m, k), dtype=torch.int8, device=win.dev, generator=gen),
             torch.rand(m, device=win.dev, generator=gen) * 0.05 + 1e-4) for _, k in win.nk]


def step(win, m: int, xs: list, outs: list) -> None:
    win.before(m)
    for w, (xq, sa) in enumerate(xs):
        win.launch(w, xq, sa, win.sw[w], *outs[w])
    win.after()


def graphs(win, m: int) -> dict:
    xs = operands(win, m, 9)
    outs = [(torch.empty(m, n, dtype=torch.bfloat16, device=win.dev), torch.empty(m, n, dtype=torch.int32, device=win.dev))
            for n, _ in win.nk]
    win.before(m)
    torch.cuda.synchronize()
    out = {}

    def cap(name, fn, mode=0):
        if isinstance(win, BandWindow):
            win.mode = win.mode0 if mode == 0 else mode
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            st = torch.cuda.current_stream().cuda_stream
            for w in range(win.L):
                rc = fn(w, st)
                if rc:
                    raise RuntimeError(f"{name}: call {w} returned {rc}")
        out[name] = gr

    unit = lambda w, st: win.units_call(w, xs[w][0], xs[w][1], win.sw[w], *outs[w], st)  # noqa: E731
    cap("calls", lambda w, st: win.a_call(w, xs[w][0], xs[w][1], st) or unit(w, st))
    cap("units", unit)
    for i, site in enumerate(DF.SITES):
        cap(f"units/{site}", lambda w, st, i=i: 0 if w % 4 != i else unit(w, st))
    if isinstance(win, BandWindow):
        cap("units_nopass", unit, 1)
        cap("pass", unit, 2)
        for i, site in enumerate(DF.SITES):
            cap(f"pass/{site}", lambda w, st, i=i: 0 if w % 4 != i else unit(w, st), 2)
        win.mode = win.mode0
    torch.cuda.synchronize()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch", default=None)
    ap.add_argument("--ms", default="1,32,256,2048,8192")
    ap.add_argument("--gs", default="64,16")
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--no-b1", action="store_true", help="skip the g = 1 banded arm")
    ap.add_argument("--variants", default="ldg,staged", help="the banded block forms: ldg (rows read in the segment "
                    "loop, groups of 4 chunks), staged (rows copied into the noise buffers), pdl (ldg with the units "
                    "launched as the pass's programmatic dependents); split always stages")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    rng = random.Random(0x4A10B)
    ms = [int(x) for x in a.ms.split(",")]
    vs = [{"ldg": "", "staged": "s", "pdl": "p"}[v] for v in a.variants.split(",")]
    arms = (["mix-cpeel"] + ([] if a.no_b1 else ["mix-cpeel-b1s"])
            + [f"mix-cpeel-b{g}{v}" for g in a.gs.split(",") for v in vs])
    by_t: dict[int, list[int]] = {}
    for m in ms:
        t = 64 if m <= 64 else m
        by_t.setdefault(t, []).append(m)
    ca = JB.ca_tree()
    doc = {"schema": "kernels-j10/rtx/v1", "gpu": torch.cuda.get_device_name(dev), "label": JB.CA_LABEL if ca else JB.LABEL,
           "noise_reference": (f"credit's banded_reference (at {JB.CA_SHA}) on the gate's CPU units, required; every "
                               "receipt against h1r_launch_band, the band workspace against j10b_ref (equal to the "
                               "reference on its pinned vectors)") if ca else "none: no banded reference tree, so no banded arm passes its gate",
           "arms": arms, "ms": ms,
           "shapes": HG.SHAPES, "sms": torch.cuda.get_device_properties(dev).multi_processor_count,
           "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "windows": {}}
    print(json.dumps({"j10rtx": {x: doc[x] for x in ("gpu", "label", "arms", "ms")}}), flush=True)
    ok_all = True
    for t, tms in sorted(by_t.items()):
        nl = NLS.get(t, 1)
        HG.model(nl, dev)
        wd = doc["windows"][str(t)] = {"t": t, "nl": nl, "ms": tms, "gates": {}, "timing": {}}
        wins = {}
        for arm in arms:
            t0 = time.time()
            try:
                win, res = gate(arm, t, tms, dev, rng)
            except Exception as e:  # noqa: BLE001
                import traceback
                wd["gates"][arm] = {"error": "".join(traceback.format_exception(e))[-3000:]}
                print(json.dumps({"t": t, "arm": arm, "error": wd["gates"][arm]["error"][-600:]}), flush=True)
                ok_all = False
                continue
            wd["gates"][arm] = res
            ok = all(r["verdict"]["ok"] for r in res.values())
            print(json.dumps({"t": t, "arm": arm, "gate_s": round(time.time() - t0, 1), "ok": ok,
                              "verdicts": {m: r["verdict"] for m, r in res.items()},
                              "cpu_units": {m: [(x["call"], x["unit"], x["z_equal"], x["receipt_equal"])
                                                for x in (r["vs_h1v3_cpu"] or [])] for m, r in res.items()}},
                             default=str), flush=True)
            if ok:
                wins[arm] = win
            else:
                ok_all = False
            (out / "j10rtx.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
        if "mix-cpeel" not in wins:
            print(json.dumps({"t": t, "error": "the per-unit arm failed its gate: nothing timed at this T"}), flush=True)
            continue
        if any(isinstance(w, BandWindow) for w in wins.values()):
            occ = next(w for w in wins.values() if isinstance(w, BandWindow)).o
            wd["band_occupancy_blocks_per_sm"] = {f: occ.h1o_band_occupancy(i) for i, f in enumerate(FORM_NAMES)}
        for m in tms:
            gs = {arm: graphs(win, m) for arm, win in wins.items()}
            for d in gs.values():
                for gr in d.values():
                    gr.replay()
            torch.cuda.synchronize()
            ms_ = {arm: {n: [] for n in d} for arm, d in gs.items()}
            order = list(gs)
            for rep in range(a.reps):
                for arm in order[rep % len(order):] + order[:rep % len(order)]:
                    for n, gr in gs[arm].items():
                        ms_[arm][n].append(HG.replay_ms(gr))
            nl_calls = 4 * nl
            per = {"calls": nl_calls, "units": nl_calls, "units_nopass": nl_calls, "pass": nl_calls,
                   **{f"units/{x}": nl for x in DF.SITES}, **{f"pass/{x}": nl for x in DF.SITES}}
            tm = {arm: {n: {"ms_median": round(statistics.median(v), 5), "us_per_call": round(1e3 * statistics.median(v)
                                                                                                / per[n], 3),
                            "ms_all": [round(x, 4) for x in v]} for n, v in d.items()} for arm, d in ms_.items()}
            base = tm["mix-cpeel"]
            ratio = {arm: {n: round(v["ms_median"] / base[n]["ms_median"], 4) for n, v in d.items() if n in base}
                     for arm, d in tm.items()}
            wd["timing"][str(m)] = {"by_arm": tm, "over_per_unit": ratio}
            print(json.dumps({"t": t, "m": m, "us_per_call": {arm: {n: v["us_per_call"] for n, v in d.items()}
                                                              for arm, d in tm.items()},
                              "over_per_unit": ratio}), flush=True)
            del gs
            (out / "j10rtx.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
        del wins
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    doc["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    doc["gate_ok"] = ok_all
    (out / "j10rtx.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    return 0 if ok_all else 2


if __name__ == "__main__":
    sys.exit(main())
