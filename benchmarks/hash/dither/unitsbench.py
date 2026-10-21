"""Dither-a's units launch on one GPU, as served (`-hbr -aomn`): each call's noise (`kh_one_noise`, h1s_noise into its
workspace), then its units (`kh_one_units` of a receipts-only build: h1k where the call's live units fit, h1u past
H1K_MAXU), at Llama-3.1-8B's four linears, a window of T rows with M live (decode B32 at T = 64: one live row block of
two). Each build in `--lib NAME=PATH` (the first is the base, today's) is gated, then timed against the others:

    python unitsbench.py --lib base=h1u_kr.so --lib NAME=h1u_x.so ... --out OUT [--t 64] [--m 32] [--reps 7]
                         [--linears qkv,o,gate_up,down] [--profile]

Gates before any timing, on every build and linear (a failed gate exits 4 and its build is not timed):
  * 0xA5 over the receipts and the workspace, then seeds, noise and units: the receipts of sampled live units (first,
    last and random) against served_h1/receipts.py's call_receipts, every dead unit's receipt zero, no poison left, the
    seeds against seeds.py;
  * a variant's receipts equal the base build's on every unit and word (the units compute what they computed);
  * the no-write control (the units launched over zero rows, under the same poison) refused;
  * the fault: one bit of unit 0's first noise block flipped in the workspace moves unit 0's receipt and no other.
Timing (CUDA graphs, builds interleaved rep by rep): each call's units alone (`units`) and its noise (`noise`, the base
build's), and the step's units stream: 32 layers x the four calls' units in order on two alternating weight copies
(`step`, 128 launches, what the served units stream runs apart from the z tables). --profile skips the timing and runs
each gated build's four calls once between torch.cuda.profiler.start and stop (for ncu --profile-from-start off),
and writes its gates to units-prof.json.

--lr --h1c LIBH1C (option B's lane records, each build with kh_one_units_lr): gated on every linear under 0xA5 over the
records: the XOR over lanes of every live unit's record equal to its receipt, the receipts equal to kh_one_units', the
sampled units' records against receipts.py's chain split by lane (`call_receipts(lanes=True)`, on the CPU), no live
record word left 0xA5 and every dead unit's record untouched, the no-write control (the lane step's F_NW_UNITS) leaving
every record unwritten; then the call's lane table (one hiding leaf a live unit, `dither_a.LR_KIND`) built by libh1c,
its root against commit.py's over the same records (--lr-cpu-root's linears), and one lane word flipped moving its leaf
and the root and no other leaf.  Timed beside the units: each call's `units_lr` (records written) and `leaves` (its
table built), and per step `step_lr` (the 128 launches writing 32 layers' records), `leaves_step` (one h1c_build of the
step's 128 tables), `step_lr_leaves` (the two in order on one stream) and `overlap` (step_lr with the previous step's
leaves on a second stream, records in two buffers), against `step`.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "benchmarks" / "pouw" / "served_h1"))
sys.path.insert(0, str(HERE.parents[2]))
import h1c as HC
import receipts as RC
from h1bench import auto_j
from h1ubench import call_data, digest
from hb_kbench import LINEARS, POISON, Clock, capture, inputs, one_call_ms, w_seed, x_seed

from verity.core.protocols.pouw.circuit import seeds as SD

ORDER = ("qkv", "o", "gate_up", "down")
# dither_a's lane tables (-lr): their kind, and F_FAULT's lane
LR_KIND, LR_LANE, F_NW_UNITS = "lane-receipts", 5, 4


class Lib:
    """One receipts-only build of h1u.cu (H1U_KSPLIT, H1K_RONLY): kh_seeds, kh_one_noise, kh_one_units."""

    def __init__(self, path: str):
        h = self.h = ctypes.CDLL(path)
        vp, u32 = ctypes.c_void_p, ctypes.c_uint
        one = [vp, vp, u32, u32, u32, vp, vp, vp, vp, u32, vp, vp, vp, vp, vp, vp, vp, u32, vp]
        h.kh_seeds.argtypes = [vp, ctypes.c_ulonglong, u32, vp, vp]
        h.kh_one_noise.argtypes = one
        h.kh_one_units.argtypes = one
        self.lr = hasattr(h, "kh_one_units_lr")
        if self.lr:
            h.kh_one_units_lr.argtypes = one[:-1] + [vp, vp, vp]
        h.kh_words.restype = u32
        h.kh_words.argtypes = [u32]
        h.kh_noise_blocks.restype = u32
        h.kh_noise_blocks.argtypes = [u32]
        h.kh_ws_bytes.restype = ctypes.c_ulonglong
        h.kh_ws_bytes.argtypes = [u32, u32, u32]


class Call:
    """One call's buffers on one build: the A slot [T, K] (rows >= M zero), B [N, K], live row blocks, seeds, the
    workspace, receipts, and the receipts-only units' z/y scratch and flagless step (dither_a's s_unit)."""

    def __init__(self, lib: Lib, a: torch.Tensor, b: torch.Tensor, m: int, salt: bytes, roots: tuple, base: int, sms: int):
        self.lib, self.a, self.b, self.m, self.salt, self.roots, self.base = lib, a, b, m, salt, roots, base
        t, k = a.shape
        n = b.shape[0]
        self.units = (t // 32) * (n // 32)
        self.live_rb = (m + 31) // 32
        self.live_units = self.live_rb * (n // 32)
        self.words = int(lib.h.kh_words(k))
        self.nb = int(lib.h.kh_noise_blocks(k))
        self.j = auto_j(self.live_rb * (n // 32), self.nb, sms)
        self.live = torch.tensor([1 if ri < self.live_rb else 0 for ri in range(t // 32)], dtype=torch.uint8,
                                 device="cuda")
        self.data = call_data(salt, *roots)
        self.seeds = torch.empty((self.units, 16), dtype=torch.int64, device="cuda")
        self.ws = torch.empty((int(lib.h.kh_ws_bytes(t, n, k)),), dtype=torch.uint8, device="cuda")
        self.out = torch.empty((self.units, self.words), dtype=torch.int32, device="cuda")
        self.zs = torch.empty((t, n), dtype=torch.int32, device="cuda")
        self.ys = torch.empty((t, n), dtype=torch.bfloat16, device="cuda")
        self.sa = torch.zeros((t,), dtype=torch.float32, device="cuda")
        self.sw = torch.zeros((n,), dtype=torch.float32, device="cuda")
        self.step = torch.zeros((4,), dtype=torch.int32, device="cuda")

    def twin(self, lib: Lib, sms: int) -> Call:
        return Call(lib, self.a, self.b, self.m, self.salt, self.roots, self.base, sms)

    def _args(self, rows: int, b: torch.Tensor | None = None):
        t, k = self.a.shape
        n = self.b.shape[0]
        return (self.a.data_ptr(), (self.b if b is None else b).data_ptr(), rows, n, k, self.seeds.data_ptr(),
                self.live.data_ptr(), self.out.data_ptr(), self.ws.data_ptr(), self.j, self.zs.data_ptr(), None,
                self.ys.data_ptr(), None, self.sa.data_ptr(), self.sw.data_ptr(), self.step.data_ptr(), min(self.m, t),
                torch.cuda.current_stream().cuda_stream)

    def seed(self) -> None:
        if self.lib.h.kh_seeds(self.data.data_ptr(), self.base, self.units, self.seeds.data_ptr(),
                               torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("kh_seeds")

    def noise(self) -> None:
        if rc := self.lib.h.kh_one_noise(*self._args(self.a.shape[0])):
            raise RuntimeError(f"kh_one_noise rc {rc}")

    def run_units(self, rows: int | None = None, b: torch.Tensor | None = None) -> None:
        if rc := self.lib.h.kh_one_units(*self._args(self.a.shape[0] if rows is None else rows, b)):
            raise RuntimeError(f"kh_one_units rc {rc}")

    def run_units_lr(self, lr: torch.Tensor, lstep: torch.Tensor, b: torch.Tensor | None = None) -> None:
        a = self._args(self.a.shape[0], b)
        if rc := self.lib.h.kh_one_units_lr(*a[:-1], lstep.data_ptr(), lr.data_ptr(), a[-1]):
            raise RuntimeError(f"kh_one_units_lr rc {rc}")

    def poison(self) -> None:
        self.out.fill_(POISON)
        self.ws.fill_(0xA5)

    def full(self) -> torch.Tensor:
        """Poisoned, then seeds, noise and units once: the receipts."""
        self.poison()
        self.seed()
        self.noise()
        self.run_units()
        torch.cuda.synchronize()
        return self.out.clone()


def gate(c: Call, picks: list[int], ref_t: torch.Tensor, base_out: torch.Tensor | None) -> dict:
    pk = torch.tensor(picks, device="cuda")
    out = c.full()
    bad = int((out[pk] != ref_t).sum())
    dead_bad = int((out[c.live_units:] != 0).sum())
    poison = int((out == POISON).sum())
    sd = c.seeds.cpu().numpy()
    seed_bad = sum(int(digest(sd[u, :8]) != SD.seed(c.salt, c.base + u, *c.roots))
                   + int(digest(sd[u, 8:]) != SD.receipt_seed(c.salt, c.base + u, *c.roots)) for u in picks)
    vs_base = None if base_out is None else int((out != base_out).sum())
    # the no-write control: the units over zero rows under the same poison
    c.out.fill_(POISON)
    c.run_units(rows=0)
    torch.cuda.synchronize()
    k_bad = int((c.out[pk] != ref_t).sum())
    # the fault: bit 0 of unit 0's first noise block in the workspace, the units again
    c.out.fill_(POISON)
    c.ws.view(torch.int32)[0] ^= 1
    c.run_units()
    torch.cuda.synchronize()
    moved = (c.out != out).any(dim=1).nonzero().flatten().tolist()
    c.ws.view(torch.int32)[0] ^= 1
    fault_ok = moved == [0]
    r = {"units": c.units, "live_units": c.live_units, "j": c.j, "bad_words": bad, "dead_bad_words": dead_bad,
         "poison_left": poison, "seed_bad": seed_bad, "vs_base_bad_words": vs_base, "control_bad_words": k_bad,
         "control_refused": k_bad > 0, "fault_moved_units": moved[:8], "fault_conforming": fault_ok}
    r["ok"] = (bad == 0 and dead_bad == 0 and poison == 0 and seed_bad == 0 and (vs_base in (None, 0)) and k_bad > 0
               and fault_ok)
    return r


def lane_tab(buf: torch.Tensor, units: int, words: int, linear: str):
    """The call's lane table: one hiding leaf a live unit, its row the unit's 32 lanes' words in lane order."""
    fields = {"window": "unitsbench", "step": 0, "linear": linear, "units": units, "words": words, "lanes": 32,
              "schema": HC.CM.SCHEMA}
    return HC.Tab(HC.Dom(LR_KIND, fields, units, {"step": (1, 0)}), units, data=buf.view(torch.uint8),
                  stride=128 * words, body=128 * words, length=128 * words, lab=("l/", f"/{linear}"))


def lr_gate(c: Call, name: str, picks: list[int], H, key: bytes, cpu_root: bool, base_out: torch.Tensor) -> dict:
    """--lr's gate on one call (the module docstring); leaves the call's records, flagless step and lane set on it."""
    W, lu, nu = c.words, c.live_units, c.units
    c.lr = torch.empty((nu, 32, W), dtype=torch.int32, device="cuda")
    c.lstep = torch.zeros(4, dtype=torch.int32, device="cuda")
    c.poison()
    c.lr.fill_(POISON)
    c.seed()
    c.noise()
    c.run_units_lr(c.lr, c.lstep)
    torch.cuda.synchronize()
    x = c.lr[:lu, 0].clone()
    for ln in range(1, 32):
        x ^= c.lr[:lu, ln]
    r = {"xor_vs_receipt_bad_words": int((x != c.out[:lu]).sum()),
         "receipts_vs_base_bad_words": int((c.out != base_out).sum()),
         "live_poison_left": int((c.lr[:lu] == POISON).sum()), "dead_touched": int((c.lr[lu:] != POISON).sum())}
    ref = RC.call_receipts(RC.Call(c.a, c.b), c.salt, c.base, *c.roots, only=picks, lanes=True)
    got = c.lr[torch.tensor(picks, device="cuda")].cpu().numpy().view(np.uint32)
    r["lanes_vs_cpu_bad_words"] = int((got != ref).sum())
    rec = c.lr.clone()
    ctl = c.lstep.clone()
    ctl[3] = F_NW_UNITS
    c.lr.fill_(POISON)
    c.run_units_lr(c.lr, ctl)
    torch.cuda.synchronize()
    r["control_written_words"] = int((c.lr != POISON).sum())
    c.lr.copy_(rec)
    c.lset = H.set([lane_tab(c.lr, lu, W, name)])
    s0 = torch.zeros(1, dtype=torch.int32, device="cuda")
    H.build(c.lset, s0)
    torch.cuda.synchronize()
    root = HC.unwords(c.lset.root(0))
    lv = c.lset.level[c.lset.level_off[0]:c.lset.level_off[0] + 8 * lu].view(lu, 8)
    leaves = lv.clone()
    r["root_vs_cpu"] = None
    if cpu_root:
        d = c.lset.tabs[0].dom.at(0)
        rows = rec[:lu].cpu().numpy().view(np.uint8).reshape(lu, 128 * W)
        r["root_vs_cpu"] = lane_root_cpu(d, key, f"l/0/{name}", rows) == root
    u_t = picks[len(picks) // 2]
    c.lr[u_t, LR_LANE, 0] ^= 1
    H.build(c.lset, s0)
    torch.cuda.synchronize()
    moved = (lv != leaves).any(dim=1).nonzero().flatten().tolist()
    r["tamper"] = {"unit": u_t, "lane": LR_LANE, "leaves_moved": moved[:8],
                   "root_moved": HC.unwords(c.lset.root(0)) != root}
    c.lr[u_t, LR_LANE, 0] ^= 1
    r["ok"] = (r["xor_vs_receipt_bad_words"] == 0 and r["receipts_vs_base_bad_words"] == 0
               and r["live_poison_left"] == 0 and r["dead_touched"] == 0 and r["lanes_vs_cpu_bad_words"] == 0
               and r["control_written_words"] == 0 and r["root_vs_cpu"] in (None, True)
               and moved == [u_t] and r["tamper"]["root_moved"])
    return r


def lane_root_cpu(d, key: bytes, label: str, rows: np.ndarray) -> bytes:
    CM = HC.CM
    return CM.levels(d, [CM.leaf(d, key, label, i, bytes(rows[i]))[0] for i in range(len(rows))])[-1][0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--out", required=True)
    ap.add_argument("--t", type=int, default=64)
    ap.add_argument("--m", type=int, default=32)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--item-ms", type=float, default=4.0)
    ap.add_argument("--picks", type=int, default=6)
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--linears", default=",".join(ORDER))
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--lr", action="store_true")
    ap.add_argument("--h1c", help="--lr's libh1c.so")
    ap.add_argument("--lr-cpu-root", default="o,down", help="--lr: the linears whose lane-table root commit.py rehashes")
    args = ap.parse_args()
    if args.lr and not args.h1c:
        ap.error("--lr needs --h1c LIBH1C")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    doc_path = out / ("units-prof.json" if args.profile else "units.json")
    paths = dict(v.partition("=")[::2] for v in args.lib)
    libs = {name: Lib(p) for name, p in paths.items()}
    names = list(libs)
    clock = Clock()
    props = torch.cuda.get_device_properties(0)
    sms = props.multi_processor_count
    rng = random.Random(0x55A17)
    doc = {"schema": "dither-a/units/v1", "device": clock.name, "sms": sms, "t": args.t, "m": args.m, "libs": paths,
           "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "reference": "served_h1/receipts.py",
           "linears": {}}
    T, M = args.t, args.m
    if args.lr:
        lr_key = rng.randbytes(32)
        H = HC.H1C(args.h1c, lr_key)
        cpu_root = set(args.lr_cpu_root.split(","))
        doc["lr"] = {"h1c": args.h1c, "kind": LR_KIND, "cpu_root": sorted(cpu_root)}
    linears = [x for x in ORDER if x in args.linears.split(",")]
    calls: dict = {}
    bcopy: dict = {}
    failed: set = set()
    for name in linears:
        n, k = LINEARS[name]
        x = torch.from_numpy(inputs(x_seed(k), M, k)).to("cuda")
        a = torch.cat([x, x.new_zeros(T - M, k)]).contiguous() if T > M else x.contiguous()
        b = torch.from_numpy(inputs(w_seed(name), n, k)).to("cuda")
        bcopy[name] = torch.from_numpy(inputs(w_seed(name) + 77, n, k)).to("cuda")
        first = Call(libs[names[0]], a, b, M, rng.randbytes(64), (rng.randbytes(64), rng.randbytes(64)),
                     rng.getrandbits(40), sms)
        cs = {nm: first if i == 0 else first.twin(libs[nm], sms) for i, nm in enumerate(names)}
        L = first.live_units
        picks = sorted({0, L - 1, *rng.sample(range(L), min(L, args.picks))})
        t0 = time.time()
        ref = RC.call_receipts(RC.Call(a, b), first.salt, first.base, *first.roots, only=picks)
        ref_t = torch.from_numpy(ref.view(np.int32)).to("cuda")
        d = doc["linears"][name] = {"n": n, "k": k, "units": first.units, "live_units": L, "words": first.words,
                                    "noise_blocks": first.nb, "picks": picks, "ref_s": round(time.time() - t0, 2),
                                    "gates": {}}
        base_out = None
        for nm, c in cs.items():
            g = gate(c, picks, ref_t, base_out)
            if base_out is None:
                base_out = c.full()
            d["gates"][nm] = g
            print(json.dumps({"linear": name, "lib": nm, "gate": g}), flush=True)
            if not g["ok"]:
                failed.add(nm)
        if args.lr:
            t0 = time.time()
            d["lr_gates"] = {}
            for nm, c in cs.items():
                if not c.lib.lr:
                    d["lr_gates"][nm] = {"ok": False, "why": "no kh_one_units_lr"}
                    failed.add(nm)
                    continue
                g = d["lr_gates"][nm] = lr_gate(c, name, picks, H, lr_key, name in cpu_root, base_out)
                print(json.dumps({"linear": name, "lib": nm, "lr_gate": g}), flush=True)
                if not g["ok"]:
                    failed.add(nm)
            W = first.words
            d["lr_bytes"] = {"record": 128 * W, "records": L * 128 * W, "segs": L * ((128 * W + 1023) // 1024),
                             "gate_s": round(time.time() - t0, 2)}
        calls[name] = cs
        for c in cs.values():
            c.full()
    doc["gate_failed"] = sorted(failed)
    doc_path.write_text(json.dumps(doc, indent=1))
    if names[0] in failed:
        print("unitsbench: the base build failed its gate; nothing timed", file=sys.stderr)
        return 4
    timed = [nm for nm in names if nm not in failed]
    if args.profile:
        for nm in timed:
            for name in linears:
                calls[name][nm].run_units()
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_push(nm)
            torch.cuda.profiler.start()
            for name in linears:
                torch.cuda.nvtx.range_push(f"{nm}/{name}")
                calls[name][nm].run_units()
                torch.cuda.nvtx.range_pop()
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()
            torch.cuda.nvtx.range_pop()
        doc["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        doc_path.write_text(json.dumps(doc, indent=1))
        return 4 if failed else 0

    arms: dict = {}
    for name in linears:
        for nm in timed:
            arms[(nm, f"units:{name}")] = (lambda c: lambda i: c.run_units(b=None))(calls[name][nm])
        arms[(names[0], f"noise:{name}")] = (lambda c: lambda i: c.noise())(calls[name][names[0]])
    if len(linears) == len(ORDER):
        for nm in timed:
            def step(i, nm=nm):
                for layer in range(args.layers):
                    for name in ORDER:
                        c = calls[name][nm]
                        c.run_units(b=c.b if layer % 2 == 0 else bcopy[name])
            arms[(nm, "step")] = step
    if args.lr:
        s0 = torch.zeros(1, dtype=torch.int32, device="cuda")
        for name in linears:
            for nm in timed:
                arms[(nm, f"units_lr:{name}")] = (lambda c: lambda i: c.run_units_lr(c.lr, c.lstep))(calls[name][nm])
            arms[(names[0], f"leaves:{name}")] = (lambda st: lambda i: H.build(st, s0))(calls[name][names[0]].lset)
        if len(linears) == len(ORDER):
            # the step's records in two buffers (a layer's own, live units only), each with its set of 128 lane tables
            c0 = {x: calls[x][names[0]] for x in ORDER}
            bufs = [[{x: torch.empty((c0[x].live_units, 32, c0[x].words), dtype=torch.int32, device="cuda")
                      for x in ORDER} for _ in range(args.layers)] for _ in range(2)]
            sets = [H.set([lane_tab(bufs[q][ly][x], c0[x].live_units, c0[x].words, f"{ly}.{x}")
                           for ly in range(args.layers) for x in ORDER]) for q in range(2)]
            side = torch.cuda.Stream()
            fork, join = torch.cuda.Event(), torch.cuda.Event()
            for nm in timed:
                def step_lr(i, nm=nm, q=0):
                    for layer in range(args.layers):
                        for x in ORDER:
                            c = calls[x][nm]
                            c.run_units_lr(bufs[q][layer][x], c.lstep, b=c.b if layer % 2 == 0 else bcopy[x])

                def both(i, nm=nm, step_lr=step_lr):
                    step_lr(i)
                    H.build(sets[0], s0)

                def overlap(i, nm=nm, step_lr=step_lr):
                    main = torch.cuda.current_stream()
                    fork.record(main)
                    side.wait_event(fork)
                    with torch.cuda.stream(side):
                        H.build(sets[(i + 1) % 2], s0)
                    step_lr(i, q=i % 2)
                    join.record(side)
                    main.wait_event(join)
                arms[(nm, "step_lr")], arms[(nm, "step_lr_leaves")], arms[(nm, "overlap")] = step_lr, both, overlap
            arms[(names[0], "leaves_step")] = lambda i: H.build(sets[0], s0)
            rec = sum(c0[x].live_units * 128 * c0[x].words for x in ORDER) * args.layers
            segs = sum(c0[x].live_units * ((128 * c0[x].words + 1023) // 1024) for x in ORDER) * args.layers
            doc["lr_step_bytes"] = {"records_written": rec, "records_read": rec, "segment_digests": 64 * segs,
                                    "segments": segs, "tables": len(ORDER) * args.layers}
    graphs, per_graph = {}, {}
    for key, fn in arms.items():
        t = one_call_ms(fn)
        G = max(1, min(400, round(args.item_ms / max(t, 1e-3))))
        graphs[key], per_graph[key] = capture(fn, G), G
    for g in graphs.values():
        g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    us = {key: [] for key in graphs}
    mhz = {key: [] for key in graphs}
    order = list(graphs)
    for rep in range(args.reps):
        for key in order[rep % len(order):] + order[:rep % len(order)]:
            e0.record()
            graphs[key].replay()
            e1.record()
            e1.synchronize()
            us[key].append(e0.elapsed_time(e1) * 1e3 / per_graph[key])
            mhz[key].append(clock.sm())
    res: dict = {}
    for (nm, arm), v in us.items():
        res.setdefault(nm, {})[arm] = {"median_us": round(statistics.median(v), 3), "us": [round(x, 3) for x in v],
                                       "per_graph": per_graph[(nm, arm)], "sm_mhz": mhz[(nm, arm)]}
    for nm, r in res.items():
        lay = [r[f"units:{x}"]["median_us"] for x in ORDER if f"units:{x}" in r]
        if len(lay) == len(ORDER):
            r["layer_units_us"] = round(sum(lay), 3)
            r["step_units_ms_from_calls"] = round(sum(lay) * args.layers / 1e3, 3)
        for arm in ("step", "step_lr", "leaves_step", "step_lr_leaves", "overlap"):
            if arm in r:
                r[f"{arm}_ms"] = round(r[arm]["median_us"] / 1e3, 3)
    b0 = res[names[0]]
    for nm in timed[1:]:
        r = res[nm]
        r["over_base"] = {arm: round(v["median_us"] / b0[arm]["median_us"], 4) for arm, v in r.items()
                          if isinstance(v, dict) and arm in b0 and "median_us" in v}
    doc["timing"] = res
    doc["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    doc_path.write_text(json.dumps(doc, indent=1))
    print(json.dumps({nm: {k: (v["median_us"] if isinstance(v, dict) and "median_us" in v else v)
                           for k, v in r.items() if k != "noise"} for nm, r in res.items()}, indent=1), flush=True)
    return 4 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
