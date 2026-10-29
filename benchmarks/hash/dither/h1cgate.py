"""`h1c.cu` gated on a full served step against `served_h1/commit.py` (and `serve.step_roots`), then timed.

A step of Llama-3.1-8B's 128 served linears at T slots, step s, m real rows (random s8 rows, random positive scales; every
other slot the canonical empty row), the window's random key:

    gate     outputs poisoned with 0xA5 first; every call's A table root (`commit.s8_table`, labels a/<s>/<w>), its admit
             tree (labels s/<s>/<w>, the T / 16 words of the slots < m), the step's a-step, admit-step and round roots
             (`serve.step_roots`), every call's receipt tree over random receipt words (labels r/<s>/<w>, the receipts
             domain with the call's base), and two B tables of the int8 pin (b/<w>, `serve.b_tables`' domain); no poison
             left in any level; a control: one A byte flipped moves that call's root and no other; the step's three sets
             captured as one CUDA graph, replayed at s, then with s + 1 written into the device step in place, against
             `commit.py` at s + 1; with a library that has h1c_tail, the step trees through `Tail` from poison, the
             same roots, and the time of each form from the leaves on
    time     the graph replayed between CUDA events: what precedes a step's coin (A and admit tables, the three step
             trees) and the receipt trees, at T = 64, 256 and 1024

    python h1cgate.py --lib libh1c.so --out OUT
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "pouw" / "served_h1"))
sys.path.insert(0, str(HERE))

import commit as CM  # noqa: E402
import credit as CR  # noqa: E402
import serve as SV  # noqa: E402

from h1c import H1C, Dom, Tab, Tail, unwords  # noqa: E402
from verity_config.settings import of  # noqa: E402

settings = of(__file__)
PIN = "/workspace/hf/hub/models--verity-int8--Llama-3.1-8B-INT8/snapshots/723d342ac443b040931242c768df342b0f6f15a7"
POISON = -0x5A5A5A5A5A5A5A5B  # 0xA5A5A5A5A5A5A5A5 as int64
SITES = [("self_attn.qkv_proj", 6144, 4096), ("self_attn.o_proj", 4096, 4096), ("mlp.gate_up_proj", 28672, 4096),
         ("mlp.down_proj", 4096, 14336)]


def linears() -> list[tuple[str, int, int]]:
    return [(f"model.layers.{i}.{m}", n, k) for i in range(32) for m, n, k in SITES]


def n_digits(ks: int) -> int:
    m, p = 1, 128
    while not (p >= 128 * ks and (127 * ks) // (p // 128) <= 127):
        m, p = m + 1, p * 128
    return m


class Step:
    """One step's tables at T slots: the pre-coin sets (calls' A and admit tables; a-step and admit-step; round) and the
    receipt set, on device buffers this owns."""

    def __init__(self, lib: H1C, window: str, t: int, lin, dev, receipts=True):
        self.t, self.lin = t, lin
        g = torch.Generator(device=dev).manual_seed(t)
        self.slots = [torch.zeros(t, k, dtype=torch.int8, device=dev) for _, _, k in lin]
        self.scales = torch.zeros(len(lin), t, dtype=torch.float32, device=dev)
        self.admit = torch.zeros(t // 16 * 2, dtype=torch.uint8, device=dev)
        self.offs, self.u_step = SV.static_layout([n for _, n, _ in lin], t)
        L = len(lin)
        f_step = {"window": window, "step": 0, "linears": L}
        self.round = lib.set([Tab(Dom("round", {"window": window, "step": 0, "leaves": "a-step root, admit-step root"}, 2,
                                      {"step": (1, 0)}), 2)])
        self.mid = lib.set([Tab(Dom("a-step", f_step, L, {"step": (1, 0)}), L, out=self.round.leaves_addr(0)),
                            Tab(Dom("admit-step", f_step, L, {"step": (1, 0)}), L, out=self.round.leaves_addr(0) + 64)])
        tabs = []
        for w, (name, n, k) in enumerate(lin):
            fa = {"window": window, "step": 0, "linear": w, "module": name, "slots": t, "k": k, "schema": CM.SCHEMA}
            tabs.append(Tab(Dom("a-table", fa, t, {"step": (1, 0)}), t, data=self.slots[w].view(torch.uint8), stride=k,
                            body=k, length=k + 4, tail=self.scales[w].view(torch.uint8), tail_stride=4,
                            lab=("a/", f"/{w}"), out=self.mid.leaves_addr(0) + 64 * w))
        for w, (name, n, k) in enumerate(lin):
            fs = {"window": window, "step": 0, "linear": w, "module": name, "slots": t, "schema": CM.SCHEMA}
            tabs.append(Tab(Dom("admit", fs, t // 16, {"step": (1, 0)}), t // 16, data=self.admit, stride=2, body=2,
                            length=2, lab=("s/", f"/{w}"), out=self.mid.leaves_addr(1) + 64 * w))
        self.calls = lib.set(tabs)
        self.rc, self.rset = [], None
        if receipts:
            rt = []
            for w, (name, n, k) in enumerate(lin):
                units, words = (t // 32) * (n // 32), 2 * (k // 32 + n_digits(k // 32))
                self.rc.append(torch.randint(-2 ** 31, 2 ** 31 - 1, (units, words), dtype=torch.int32, device=dev,
                                             generator=g))
                fr = {"window": window, "step": 0, "linear": w, "module": name, "base": 0, "units": units, "words": words,
                      "schema": CM.SCHEMA}
                rt.append(Tab(Dom("receipts", fr, units, {"step": (1, 0), "base": (self.u_step, self.offs[w])}), units,
                              data=self.rc[w].view(torch.uint8), stride=4 * words, body=4 * words, length=4 * words,
                              lab=("r/", f"/{w}")))
            self.rset = lib.set(rt)

    def fill(self, m: int, seed: int) -> None:
        g = torch.Generator(device="cpu").manual_seed(seed)
        for w, s in enumerate(self.slots):
            s.zero_()
            s[:m] = torch.randint(-127, 128, (m, s.shape[1]), dtype=torch.int8, generator=g).to(s.device)
        self.scales.zero_()
        self.scales[:, :m] = (torch.rand(len(self.slots), m, generator=g) * 0.05 + 1e-4).to(self.scales.device)
        bits = np.arange(self.t) < m
        rows = b"".join(CR.word_rows(CR.words(bits)))
        self.admit.copy_(torch.frombuffer(bytearray(rows), dtype=torch.uint8))
        self.m = m

    def poison(self) -> None:
        for st in (self.calls, self.mid, self.round) + ((self.rset,) if self.rset else ()):
            for b in (st.level, st.segs, st.dom, st.roots):
                b.fill_(POISON)

    def pre(self, lib: H1C, step) -> None:
        lib.build(self.calls, step)
        lib.build(self.mid, step)
        lib.build(self.round, step)

    def post(self, lib: H1C, step) -> None:
        lib.build(self.rset, step)


def cpu_calls(window: str, key: bytes, st: Step, s: int, ws=None) -> dict:
    """commit.py's roots of step s over st's buffers: per call A and admit roots, the step roots."""
    out = {"a": {}, "admit": {}}
    bits = np.arange(st.t) < st.m
    wrows = CR.word_rows(CR.words(bits))
    sc = st.scales.cpu().numpy()
    cs = []
    for w, (name, n, k) in enumerate(st.lin):
        q = st.slots[w].cpu().numpy()
        fa = {"window": window, "step": s, "linear": w, "module": name, "slots": st.t, "k": k, "schema": CM.SCHEMA}
        ra = CM.s8_table(CM.domain(fa, st.t, "a-table"), key, f"a/{s}/{w}", q, sc[w]).root
        fs = {"window": window, "step": s, "linear": w, "module": name, "slots": st.t, "schema": CM.SCHEMA}
        sd = CM.domain(fs, len(wrows), "admit")
        rs = CM.levels(sd, [CM.leaf(sd, key, f"s/{s}/{w}", i, x)[0] for i, x in enumerate(wrows)])[-1][0]
        out["a"][w], out["admit"][w] = ra, rs
        cs.append({"root_a": ra.hex(), "root_admit": rs.hex()})
    sr = SV.step_roots(window, s, cs)
    out.update(a_step=sr["a_step"], admit_step=sr["admit_step"], round=sr["root"])
    return out


def cpu_receipts(window: str, key: bytes, st: Step, s: int, w: int) -> bytes:
    name, n, k = st.lin[w]
    units, words = st.rc[w].shape
    fr = {"window": window, "step": s, "linear": w, "module": name, "base": s * st.u_step + st.offs[w], "units": units,
          "words": words, "schema": CM.SCHEMA}
    dom = CM.domain(fr, units)
    rows = CM.receipt_rows(st.rc[w].cpu().numpy().view(np.uint32))
    return CM.levels(dom, [CM.leaf(dom, key, f"r/{s}/{w}", u, bytes(r))[0] for u, r in enumerate(rows)])[-1][0]


def compare(st: Step, ref: dict) -> dict:
    L = len(st.lin)
    bad_a = [w for w in range(L) if unwords(st.mid.leaf(0, w)) != ref["a"][w]]
    bad_s = [w for w in range(L) if unwords(st.mid.leaf(1, w)) != ref["admit"][w]]
    return {"a_tables": L, "a_bad": bad_a, "admit_bad": bad_s,
            "a_step": unwords(st.round.leaf(0, 0)) == ref["a_step"], "admit_step": unwords(st.round.leaf(0, 1)) == ref["admit_step"],
            "round": unwords(st.round.root(0)) == ref["round"]}


def b_tables(lib: H1C, key: bytes, dev) -> dict:
    """Two linears' B tables (layer 0's o_proj and down_proj) against commit.s8_table under serve.b_tables' domain."""
    from safetensors import safe_open
    idx = json.loads((Path(PIN) / "model.safetensors.index.json").read_text())["weight_map"]
    model = {"repo": "verity-int8/Llama-3.1-8B-INT8"}
    res = []
    for w, mod in ((1, "model.layers.0.self_attn.o_proj"), (3, "model.layers.0.mlp.down_proj")):
        with safe_open(str(Path(PIN) / idx[mod + ".weight"]), framework="pt", device="cpu") as f:
            wt = f.get_tensor(mod + ".weight")
        with safe_open(str(Path(PIN) / idx[mod + ".weight_scale"]), framework="pt", device="cpu") as f:
            sc = f.get_tensor(mod + ".weight_scale").float().reshape(-1).contiguous()
        n, k = wt.shape
        fields = SV.b_fields(model, w, mod, n, k)
        t0 = time.perf_counter()
        want = CM.s8_table(CM.domain(fields, n, "b-table"), key, f"b/{w}", wt.numpy(), sc.numpy()).root
        cpu_s = time.perf_counter() - t0
        wd, sd = wt.contiguous().to(dev), sc.to(dev)
        st = lib.set([Tab(Dom("b-table", fields, n), n, data=wd.view(torch.uint8), stride=k, body=k, length=k + 4,
                          tail=sd.view(torch.uint8), tail_stride=4, lab=(f"b/{w}", ""), labs=False)])
        step = torch.zeros(1, dtype=torch.int32, device=dev)
        st.level.fill_(POISON)
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        lib.build(st, step)
        e1.record()
        e1.synchronize()
        res.append({"module": mod, "n": n, "k": k, "equal": unwords(st.root(0)) == want, "gpu_ms": round(e0.elapsed_time(e1), 3),
                    "cpu_s": round(cpu_s, 2)})
    return {"tables": res, "ok": all(r["equal"] for r in res)}


def no_poison(st: Step) -> bool:
    from verity.ml.commitments.merkle import merkle_depth
    for x in (st.calls, st.mid, st.round) + ((st.rset,) if st.rset else ()):
        for i, t in enumerate(x.tabs):
            n = 8 * (2 * (1 << merkle_depth(t.count)) - 1)
            if bool((x.level[x.level_off[i]:x.level_off[i] + n] == POISON).any()):
                return False
    return True


def tail_check(lib: H1C, st: Step, step, ref: dict, reps: int) -> dict:
    """The step's mid and round trees through `Tail` (h1c_tail), from poison, against `ref`; and the time from the
    leaves on of h1c_build's two builds and of `Tail.levels`."""
    tl = Tail(lib, st.mid, st.round, host_root=True)
    st.poison()
    lib.build(st.calls, step)
    tl.pre(step)
    tl.levels()
    st.post(lib, step)
    torch.cuda.synchronize()
    c = {**compare(st, ref), "no_poison_left": no_poison(st), "host_root": tl.root(0) == ref["round"]}
    e = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
    ms = {"build": [], "pre": [], "levels": []}
    for _ in range(reps):
        e[0].record()
        lib.build(st.mid, step)
        lib.build(st.round, step)
        e[1].record()
        tl.pre(step)
        e[2].record()
        tl.levels()
        e[3].record()
        e[3].synchronize()
        for k, (i, j) in zip(ms, ((0, 1), (1, 2), (2, 3))):
            ms[k].append(e[i].elapsed_time(e[j]))
    c["again"] = compare(st, ref)
    return {**c, **{f"{k}_ms": round(statistics.median(v), 4) for k, v in ms.items()}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=7)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    key = os.urandom(64)
    window = settings.RESEARCH_RUN_ID or "gate"
    t0 = time.perf_counter()
    lib = H1C(a.lib, key, dev)
    doc: dict = {"gpu": torch.cuda.get_device_name(), "init_s": round(time.perf_counter() - t0, 1)}
    lin = linears()

    st = Step(lib, window, 64, lin, dev)
    st.fill(37, 1)
    step = torch.tensor([7], dtype=torch.int32, device=dev)
    st.poison()
    st.pre(lib, step)
    st.post(lib, step)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    ref = cpu_calls(window, key, st, 7)
    doc["cpu_pre_s"] = round(time.perf_counter() - t0, 1)
    doc["pre"] = compare(st, ref)
    t0 = time.perf_counter()
    bad_r = [w for w in range(len(lin)) if unwords(st.rset.root(w)) != cpu_receipts(window, key, st, 7, w)]
    doc["receipts"] = {"tables": len(lin), "leaves": sum(int(r.shape[0]) for r in st.rc), "bad": bad_r,
                       "cpu_s": round(time.perf_counter() - t0, 1)}
    doc["no_poison_left"] = no_poison(st)
    print(json.dumps({k: doc[k] for k in ("pre", "receipts", "no_poison_left")}), flush=True)

    before = [unwords(st.mid.leaf(0, w)) for w in range(len(lin))]
    st.slots[5][3, 100] ^= 1
    st.pre(lib, step)
    torch.cuda.synchronize()
    after = [unwords(st.mid.leaf(0, w)) for w in range(len(lin))]
    doc["control"] = {"tampered_call_moved": after[5] != before[5],
                      "others_kept": all(after[w] == before[w] for w in range(len(lin)) if w != 5),
                      "round_moved": unwords(st.round.root(0)) != ref["round"]}
    st.slots[5][3, 100] ^= 1
    print(json.dumps({"control": doc["control"]}), flush=True)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        st.pre(lib, step)
        st.post(lib, step)
    replays = []
    for s in (7, 8):
        step.fill_(s)
        st.poison()
        g.replay()
        torch.cuda.synchronize()
        r = ref if s == 7 else cpu_calls(window, key, st, s)
        c = compare(st, r)
        rw = [0, 3, 127]
        c["receipts_bad"] = [w for w in rw if unwords(st.rset.root(w)) != cpu_receipts(window, key, st, s, w)]
        c["no_poison_left"] = no_poison(st)
        replays.append({"step": s, **c})
    doc["graph_replay"] = replays
    print(json.dumps({"graph_replay": replays}), flush=True)

    doc["b_tables"] = b_tables(lib, key, dev)
    print(json.dumps({"b_tables": doc["b_tables"]}), flush=True)

    def pre_ok(c: dict) -> bool:
        return not c["a_bad"] and not c["admit_bad"] and c["a_step"] and c["admit_step"] and c["round"]

    tail_ok = True
    if hasattr(lib.lib, "h1c_tail") and lib.lib.h1c_tail_words():
        step.fill_(7)
        doc["tail"] = tail_check(lib, st, step, ref, a.reps)
        print(json.dumps({"tail": doc["tail"]}), flush=True)
        tail_ok = (pre_ok(doc["tail"]) and pre_ok(doc["tail"]["again"]) and doc["tail"]["no_poison_left"]
                   and doc["tail"]["host_root"])
    ok = (pre_ok(doc["pre"]) and not doc["receipts"]["bad"] and doc["no_poison_left"] and all(doc["control"].values())
          and all(pre_ok(r) and not r["receipts_bad"] and r["no_poison_left"] for r in replays) and doc["b_tables"]["ok"]
          and tail_ok)
    doc["gate_ok"] = ok
    (out / "h1c-gate.json").write_text(json.dumps(doc, indent=1) + "\n")
    if not ok:
        print("h1cgate: GATE FAILED", flush=True)
        return 2
    del g, st
    torch.cuda.empty_cache()

    timing = []
    for t, m in ((64, 32), (256, 256), (1024, 1024)):
        stt = Step(lib, window, t, lin, dev, receipts=t <= 256)
        stt.fill(m, 2)
        step.fill_(3)
        gp, gr = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        with torch.cuda.graph(gp):
            stt.pre(lib, step)
        if stt.rset:
            with torch.cuda.graph(gr):
                stt.post(lib, step)
        for gg in (gp, gr) if stt.rset else (gp,):
            gg.replay()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        ms = {"pre": [], "receipt_trees": []}
        for _ in range(a.reps):
            for name, gg in (("pre", gp), ("receipt_trees", gr)) if stt.rset else (("pre", gp),):
                e0.record()
                gg.replay()
                e1.record()
                e1.synchronize()
                ms[name].append(e0.elapsed_time(e1))
        row = {"t": t, "rows": m, "a_rows": t * len(lin), "receipt_leaves": sum(int(r.shape[0]) for r in stt.rc),
               **{f"{k}_ms": round(statistics.median(v), 4) for k, v in ms.items() if v},
               **{f"{k}_ms_all": [round(x, 4) for x in v] for k, v in ms.items() if v}}
        timing.append(row)
        print(json.dumps({"time": row}), flush=True)
        del gp, gr, stt
        torch.cuda.empty_cache()
    doc["timing"] = timing
    (out / "h1c-gate.json").write_text(json.dumps(doc, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
