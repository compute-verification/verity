"""`h1c.cu`'s served-call A tables (`h1c_a_init`, `h1c_a_step`, `h1c_a_call`) gated on Llama-3.1-8B's 128 linears against
`served_h1/commit.py` and against `h1c_build` on the same slots, then timed against the per-call `h1c_build` they replace.

    gate     T = 64 slots, step s, m live rows: each call's operands xq [M, K] (M > m, rows >= m random too, which the
             slot must not take) and sa [M]; the slots, scales and A levels poisoned with 0xA5 first; per call the slot
             rows (< m xq's, >= m zero), the scales (< m sa's, >= m 0) and the A root = `commit.s8_table`'s over the slots
             and = `h1c_build`'s; no poison left in any A level; at (s, m) = (7, 37), then captured as one CUDA graph (the
             step's part and the 128 calls) and replayed at (8, 64), (9, 1) and (10, 32) with step and m written in place;
             a control: one xq byte of one live row flipped moves that call's root and no other
    time     the graph's 128 calls (and the step's part alone) replayed between CUDA events at m = 32 and 64, against the
             128 per-call h1c_build launches of the earlier path (domains, segments, leaves, levels), captured the same way

    python h1agate.py --lib libh1c.so --out OUT
"""
from __future__ import annotations

import argparse
import ctypes as C
import json
import os
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import h1c as H1CM  # noqa: E402
from h1c import H1C, unwords  # noqa: E402
from h1cgate import POISON, Step, cpu_calls, linears  # noqa: E402

T, MX = 64, 72


class Calls:
    """The 128 calls' operands and the A tables' window and step buffers."""

    def __init__(self, lib: H1C, st: Step, dev):
        self.lib, self.st, self.L = lib, st, len(st.lin)
        self.tb = C.sizeof(H1CM._Tab)
        self.words = lib.lib.h1c_a_words(st.t)
        self.pre = torch.zeros(self.L * self.words, dtype=torch.int64, device=dev)
        self.S = torch.zeros(4, dtype=torch.int32, device=dev)
        self.step = torch.zeros(2, dtype=torch.int32, device=dev)
        self.xq = [torch.zeros(MX, k, dtype=torch.int8, device=dev) for _, _, k in st.lin]
        self.sa = [torch.zeros(MX, dtype=torch.float32, device=dev) for _ in st.lin]
        rc = lib.lib.h1c_a_init(C.addressof(st.calls.host), st.calls.dev.data_ptr(), self.L, self.pre.data_ptr(), self.words,
                                torch.cuda.current_stream().cuda_stream)
        if rc:
            raise RuntimeError(f"h1c_a_init returned {rc}")

    def fill(self, seed: int) -> None:
        g = torch.Generator(device="cpu").manual_seed(seed)
        for x, s in zip(self.xq, self.sa):
            x.copy_(torch.randint(-128, 128, x.shape, dtype=torch.int8, generator=g))
            s.copy_(torch.rand(s.shape, generator=g) * 0.05 + 1e-4)

    def set(self, s: int, m: int) -> None:
        self.step.fill_(s)
        self.S.copy_(torch.tensor([0, m, 0, 0], dtype=torch.int32))
        self.st.m = m

    def run_step(self) -> None:
        st = self.st
        rc = self.lib.lib.h1c_a_step(st.calls.dev.data_ptr(), self.L, self.pre.data_ptr(), self.words, self.step.data_ptr(),
                                     self.lib.btab.data_ptr(), self.S.data_ptr(), torch.cuda.current_stream().cuda_stream)
        if rc:
            raise RuntimeError(f"h1c_a_step returned {rc}")

    def run_calls(self) -> None:
        st, cs = self.st, torch.cuda.current_stream().cuda_stream
        for w, (_, _, k) in enumerate(st.lin):
            rc = self.lib.lib.h1c_a_call(C.addressof(st.calls.host) + self.tb * w, st.calls.dev.data_ptr() + self.tb * w,
                                         self.pre.data_ptr() + 8 * self.words * w, self.S.data_ptr(), self.xq[w].data_ptr(),
                                         self.sa[w].data_ptr(), st.slots[w].data_ptr(), st.scales[w].data_ptr(), k, cs)
            if rc:
                raise RuntimeError(f"h1c_a_call({w}) returned {rc}")

    def run_old(self) -> None:
        """The earlier path's per-call A table: h1c_build on that one table (domains, segments, leaves, levels)."""
        st, cs = self.st, torch.cuda.current_stream().cuda_stream
        for w in range(self.L):
            rc = self.lib.lib.h1c_build(C.addressof(st.calls.host) + self.tb * w, st.calls.dev.data_ptr() + self.tb * w, 1,
                                        self.step.data_ptr(), self.lib.btab.data_ptr(), 1, cs)
            if rc:
                raise RuntimeError(f"h1c_build({w}) returned {rc}")

    def poison(self) -> None:
        for x in self.st.slots:
            x.view(torch.uint8).fill_(0xA5)
        self.st.scales.view(torch.uint8).fill_(0xA5)
        self.st.calls.level.fill_(POISON)
        self.st.mid.level.fill_(POISON)

    def roots(self) -> list[bytes]:
        return [unwords(self.st.mid.leaf(0, w)) for w in range(self.L)]

    def a_levels_poisoned(self) -> list[int]:
        from verity.ml.commitments.merkle import merkle_depth
        st = self.st
        bad = []
        for w in range(self.L):
            n = 8 * (2 * (1 << merkle_depth(st.calls.tabs[w].count)) - 1)
            o = st.calls.level_off[w]
            if bool((st.calls.level[o:o + n] == POISON).any()):
                bad.append(w)
        return bad


def check(c: Calls, window: str, key: bytes, s: int, m: int) -> dict:
    st = c.st
    torch.cuda.synchronize()
    got = c.roots()
    slots_bad = [w for w in range(c.L)
                 if not torch.equal(st.slots[w][:m], c.xq[w][:m]) or bool(st.slots[w][m:].any())]
    sc = st.scales
    scales_bad = [w for w in range(c.L) if not torch.equal(sc[w][:m], c.sa[w][:m]) or bool(sc[w][m:].view(torch.int32).any())]
    poisoned = c.a_levels_poisoned()
    ref = cpu_calls(window, key, st, s)
    cpu_bad = [w for w in range(c.L) if got[w] != ref["a"][w]]
    st.mid.level.fill_(POISON)
    c.run_old()
    torch.cuda.synchronize()
    old = c.roots()
    old_bad = [w for w in range(c.L) if got[w] != old[w]]
    ok = not (slots_bad or scales_bad or poisoned or cpu_bad or old_bad)
    return {"s": s, "m": m, "calls": c.L, "slots_bad": slots_bad, "scales_bad": scales_bad, "a_levels_poisoned": poisoned,
            "roots_vs_commit_bad": cpu_bad, "roots_vs_h1c_build_bad": old_bad, "ok": ok}


def timed(g: torch.cuda.CUDAGraph, reps: int) -> list[float]:
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    g.replay()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        e0.record()
        g.replay()
        e1.record()
        e1.synchronize()
        out.append(e0.elapsed_time(e1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=9)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    key = os.urandom(64)
    window = os.environ.get("RESEARCH_RUN_ID") or "gate"
    lib = H1C(a.lib, key, dev)
    st = Step(lib, window, T, linears(), dev, receipts=False)
    c = Calls(lib, st, dev)
    c.fill(1)
    doc: dict = {"gpu": torch.cuda.get_device_name(), "t": T, "m_rows_of_xq": MX}

    c.set(7, 37)
    c.poison()
    c.run_step()
    c.run_calls()
    doc["eager"] = check(c, window, key, 7, 37)
    print(json.dumps({"eager": doc["eager"]}), flush=True)

    before = c.roots()
    c.xq[5][3, 100] ^= 1
    c.run_step()
    c.run_calls()
    torch.cuda.synchronize()
    after = c.roots()
    doc["control"] = {"tampered_call_moved": after[5] != before[5],
                      "others_kept": all(after[w] == before[w] for w in range(c.L) if w != 5)}
    c.xq[5][3, 100] ^= 1
    print(json.dumps({"control": doc["control"]}), flush=True)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        c.run_step()
        c.run_calls()
    replays = []
    for s, m in ((8, 64), (9, 1), (10, 32)):
        c.set(s, m)
        c.poison()
        g.replay()
        replays.append(check(c, window, key, s, m))
        print(json.dumps({"replay": replays[-1]}), flush=True)
    doc["graph_replay"] = replays
    ok = doc["eager"]["ok"] and all(doc["control"].values()) and all(r["ok"] for r in replays)
    doc["gate_ok"] = ok
    (out / "h1a-gate.json").write_text(json.dumps(doc, indent=1) + "\n")
    if not ok:
        print("h1agate: GATE FAILED", flush=True)
        return 2
    del g

    gs, gc, go = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(gs):
        c.run_step()
    with torch.cuda.graph(gc):
        c.run_calls()
    with torch.cuda.graph(go):
        c.run_old()
    timing = []
    for m in (32, 64):
        c.set(11, m)
        ms = {"step_part": timed(gs, a.reps), "calls": timed(gc, a.reps), "old_calls": timed(go, a.reps)}
        row = {"m": m, **{f"{k}_ms": round(statistics.median(v), 4) for k, v in ms.items()},
               "call_us": round(1e3 * statistics.median(ms["calls"]) / c.L, 2),
               "old_call_us": round(1e3 * statistics.median(ms["old_calls"]) / c.L, 2),
               **{f"{k}_ms_all": [round(x, 4) for x in v] for k, v in ms.items()}}
        timing.append(row)
        print(json.dumps({"time": {k: v for k, v in row.items() if not k.endswith("_all")}}), flush=True)
    doc["timing"] = timing
    (out / "h1a-gate.json").write_text(json.dumps(doc, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
