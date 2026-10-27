"""`h1c.cu`'s served-call A tables (`h1c_a_init`, `h1c_a_step`, `h1c_a_call`) gated on Llama-3.1-8B's 128 linears against
`served_h1/commit.py` and against `h1c_build` on the same slots, then timed against the per-call `h1c_build` they replace.

    gate     T = 64 slots (--t), step s, m live rows: each call's operands xq [M, K] (M = T + 8, rows >= m random too,
             which the slot must not take) and sa [M]; the slots, scales and A levels poisoned with 0xA5 first; per call
             the slot rows (< m xq's, >= m zero), the scales (< m sa's, >= m 0) and the A root = `commit.s8_table`'s over
             the slots and = `h1c_build`'s; no poison left in any A level; at (s, m) = (7, 37), then captured as one CUDA
             graph (the step's part and the 128 calls) and replayed at (8, T), (9, 1), (10, 32) and (12, T - 3) with step
             and m written in place; a control: one xq byte of one live row flipped moves that call's root and no other;
             every root also = one h1c_build of all 128 tables' (`batch`)
    time     the graph's 128 calls (and the step's part alone) replayed between CUDA events at m = 32, 64 and T, against
             the 128 per-call h1c_build launches of the earlier path (domains, segments, leaves, levels), captured the same
             way, and against `batch`: the 128 tables in one h1c_build, a per-step A commitment's build (a scheme change,
             measured only: its tables off the forward's critical path, built together once the step's A are known)

Each h1c_a_form value --forms names (0 h1c_a_call_k, 2 h1c_a_call2_k, its compressions paired, 3 h1c_a_call3_k, paired
with its messages in registers) is gated in turn, and only if every one passes is each timed, in the same process on the
same operands; forms 2's and 3's phases then come from their %globaltimer stamps (`stamps`) on a qkv, a gate_up and a
down call alone.

    python h1agate.py --lib libh1c.so --out OUT [--t 64] [--forms 0,2]
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
from verity_config.settings import of  # noqa: E402

settings = of(__file__)
T, MX = 64, 72


class Calls:
    """The 128 calls' operands and the A tables' window and step buffers."""

    def __init__(self, lib: H1C, st: Step, dev, mx: int = MX):
        self.lib, self.st, self.L = lib, st, len(st.lin)
        self.tb = C.sizeof(H1CM._Tab)
        self.words = lib.lib.h1c_a_words(st.t)
        self.pre = torch.zeros(self.L * self.words, dtype=torch.int64, device=dev)
        self.S = torch.zeros(4, dtype=torch.int32, device=dev)
        self.step = torch.zeros(2, dtype=torch.int32, device=dev)
        self.xq = [torch.zeros(mx, k, dtype=torch.int8, device=dev) for _, _, k in st.lin]
        self.sa = [torch.zeros(mx, dtype=torch.float32, device=dev) for _ in st.lin]
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

    def run_calls(self, ws=None) -> None:
        st, cs = self.st, torch.cuda.current_stream().cuda_stream
        for w, (_, _, k) in enumerate(st.lin):
            if ws is not None and w not in ws:
                continue
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

    def run_batch(self) -> None:
        """A per-step A commitment's build (a scheme change, measured only): the step's 128 tables in one h1c_build."""
        st = self.st
        rc = self.lib.lib.h1c_build(C.addressof(st.calls.host), st.calls.dev.data_ptr(), self.L, self.step.data_ptr(),
                                    self.lib.btab.data_ptr(), 1, torch.cuda.current_stream().cuda_stream)
        if rc:
            raise RuntimeError(f"h1c_build(all {self.L}) returned {rc}")

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
    st.mid.level.fill_(POISON)
    c.run_batch()
    torch.cuda.synchronize()
    batch = c.roots()
    batch_bad = [w for w in range(c.L) if got[w] != batch[w]]
    ok = not (slots_bad or scales_bad or poisoned or cpu_bad or old_bad or batch_bad)
    return {"s": s, "m": m, "calls": c.L, "slots_bad": slots_bad, "scales_bad": scales_bad, "a_levels_poisoned": poisoned,
            "roots_vs_commit_bad": cpu_bad, "roots_vs_h1c_build_bad": old_bad, "roots_vs_batch_bad": batch_bad, "ok": ok}


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


def gate(c: Calls, window: str, key: bytes) -> dict:
    t = c.st.t
    doc: dict = {}
    c.set(7, min(37, t))
    c.poison()
    c.run_step()
    c.run_calls()
    doc["eager"] = check(c, window, key, 7, min(37, t))
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
    for s, m in ((8, t), (9, 1), (10, 32), (12, t - 3)):
        c.set(s, m)
        c.poison()
        g.replay()
        replays.append(check(c, window, key, s, m))
        print(json.dumps({"replay": replays[-1]}), flush=True)
    doc["graph_replay"] = replays
    doc["gate_ok"] = doc["eager"]["ok"] and all(doc["control"].values()) and all(r["ok"] for r in replays)
    return doc


def timing(c: Calls, reps: int) -> list[dict]:
    gs, gc, go, gb = (torch.cuda.CUDAGraph() for _ in range(4))
    with torch.cuda.graph(gs):
        c.run_step()
    with torch.cuda.graph(gc):
        c.run_calls()
    with torch.cuda.graph(go):
        c.run_old()
    with torch.cuda.graph(gb):
        c.run_batch()
    rows = []
    for m in sorted({32, 64, c.st.t}):
        c.set(11, m)
        ms = {"step_part": timed(gs, reps), "calls": timed(gc, reps), "old_calls": timed(go, reps),
              "batch": timed(gb, reps)}
        row = {"m": m, **{f"{k}_ms": round(statistics.median(v), 4) for k, v in ms.items()},
               "call_us": round(1e3 * statistics.median(ms["calls"]) / c.L, 2),
               "old_call_us": round(1e3 * statistics.median(ms["old_calls"]) / c.L, 2),
               "batch_call_us": round(1e3 * statistics.median(ms["batch"]) / c.L, 2),
               **{f"{k}_ms_all": [round(x, 4) for x in v] for k, v in ms.items()}}
        rows.append(row)
        print(json.dumps({"time": {k: v for k, v in row.items() if not k.endswith("_all")}}), flush=True)
    return rows


def stamps(c: Calls, m: int, ws=(0, 2, 3), reps: int = 5) -> dict:
    """h1c_a_call2_k's or h1c_a_call3_k's phases (h1c_a_stamps) for calls ws alone at m live rows, in us from the first block's start: each
    block's segments, x, leaves and arrival, the last block's levels; the call's CUDA-event time beside them."""
    blocks = c.st.t // 4
    buf = torch.zeros(32 + 8 * blocks, dtype=torch.int64, device=c.pre.device)
    c.lib.lib.h1c_a_stamps.argtypes = [C.c_void_p]
    out = {}
    c.set(13, m)
    c.run_step()
    for w in ws:
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        rows = []
        for _ in range(reps):
            c.lib.lib.h1c_a_stamps(buf.data_ptr())
            buf.zero_()
            torch.cuda.synchronize()
            e0.record()
            c.run_calls([w])
            e1.record()
            e1.synchronize()
            b = buf.cpu().numpy().astype(np.int64)
            ph = b[32:].reshape(blocks, 8)[:, :5]
            lb = ph[:(m + 3) // 4]   # the blocks with a live row
            t0 = int(ph[:, 0].min())
            last = int(b[0])
            lv = [int(x) for x in b[1:1 + 16] if x]
            rows.append({"event_us": round(1e3 * e0.elapsed_time(e1), 2),
                         "start_spread_us": round((int(ph[:, 0].max()) - t0) / 1e3, 2),
                         "segments_us": round(float(np.median(lb[:, 1] - lb[:, 0])) / 1e3, 2),
                         "x_us": round(float(np.median(lb[:, 2] - lb[:, 1])) / 1e3, 2),
                         "leaf_us": round(float(np.median(lb[:, 3] - lb[:, 2])) / 1e3, 2),
                         "last_block": last, "last_arrival_us": round((int(ph[last, 4]) - t0) / 1e3, 2),
                         "levels_us": [round((y - x) / 1e3, 2) for x, y in zip([int(ph[last, 4])] + lv, lv)],
                         "end_us": round(((lv[-1] if lv else int(ph[last, 4])) - t0) / 1e3, 2)})
        c.lib.lib.h1c_a_stamps(None)
        out[f"call{w}_k{c.st.lin[w][2]}"] = rows[-1] | {"event_us_all": [r["event_us"] for r in rows]}
        print(json.dumps({"stamps": {"m": m, "w": w, **out[f"call{w}_k{c.st.lin[w][2]}"]}}), flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default=settings.DITHER_H1C)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch", default=None, help="unused (sbench.sh's SBENCH_SCRIPT passes it)")
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--t", default=str(T), help="the window's slots (each A table's rows), one or more: 64,256")
    ap.add_argument("--forms", default="0", help="h1c_a_form values, each gated, then each timed, in this process")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    key = os.urandom(64)
    window = settings.RESEARCH_RUN_ID or "gate"
    lib = H1C(a.lib, key, dev)
    forms = [int(f) for f in a.forms.split(",")]
    doc: dict = {"gpu": torch.cuda.get_device_name(), "by_t": {}}
    for t in (int(x) for x in a.t.split(",")):
        st = Step(lib, window, t, linears(), dev, receipts=False)
        c = Calls(lib, st, dev, t + 8)
        c.fill(t)
        d = doc["by_t"][t] = {"t": t, "m_rows_of_xq": t + 8, "forms": {}}
        for f in forms:
            lib.form(f)
            print(json.dumps({"t": t, "form": f}), flush=True)
            d["forms"][f] = gate(c, window, key)
        d["gate_ok"] = all(x["gate_ok"] for x in d["forms"].values())
        (out / "h1a-gate.json").write_text(json.dumps(doc, indent=1) + "\n")
        if not d["gate_ok"]:
            print("h1agate: GATE FAILED", flush=True)
            return 2
        for f in forms:
            lib.form(f)
            print(json.dumps({"t": t, "form": f}), flush=True)
            d["forms"][f]["timing"] = timing(c, a.reps)
            if f in (2, 3):
                d["forms"][f]["stamps"] = {m: stamps(c, m) for m in sorted({32, t})}
        (out / "h1a-gate.json").write_text(json.dumps(doc, indent=1) + "\n")
        del st, c
        torch.cuda.empty_cache()
    doc["gate_ok"] = True
    (out / "h1a-gate.json").write_text(json.dumps(doc, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
