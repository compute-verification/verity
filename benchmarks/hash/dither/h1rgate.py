"""`h1r.cu` gated on a full served step, then timed (J12: the fused receipt kernel; item 6: at T = 64 and 256), every
kernel version (`--versions`) gated in the same run before any is timed.

One step of Llama-3.1-8B's 128 served linears (layer by layer qkv, o, gate_up, down, B the int8 pin's weights as vLLM merges
them) at T slots, each call's first M_c slot rows random s8 and admitted, the rest zero (`served_h1`'s slot table), its units
at SERVE-H1's static positions (step s, `serve.static_layout`), a random salt and random roots.  Calls alternate M_c between
two counts, so some calls have a dead row tile.

    gate     per version, h1r into an output poisoned with 0xA5, against `served_h1/receipts.gated_receipts` (served_h1's
             torch path, its noise through served_h1's `noise.cu`, computed once) on every unit of every call; `h1tile.reference`
             (`receipts.gated_reference`) on chosen units (K = 4096 and 14336, first and last, a dead one); a no-write control
             (nowrite: the poison must survive and be refused); the launch captured as a CUDA graph and replayed into a fresh
             poison, then under a second salt written into the same device buffer (as a served step does), each against
             receipts.py on sampled units
    time     each version's graph, replayed in turn REPS times between CUDA events: the step's receipts, per live unit, at
             T = 64 and T = 256

    python h1rgate.py --lib libh1r.so --cubin noise.cubin --out OUT [--versions 1,2]
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
sys.path.insert(0, str(ROOT))

import receipts as RC  # noqa: E402

from h1r import H1R, Call, words  # noqa: E402

PIN = "/workspace/hf/hub/models--verity-int8--Llama-3.1-8B-INT8/snapshots/723d342ac443b040931242c768df342b0f6f15a7"
LAYERS = 32
POISON = np.uint32(0xA5A5A5A5).view(np.int32)


def weights(dev) -> list[torch.Tensor]:
    """The 128 served linears' int8 weights [N, K] in SERVE-H1's order, merged as vLLM merges them."""
    from safetensors import safe_open
    idx = json.loads((Path(PIN) / "model.safetensors.index.json").read_text())["weight_map"]
    handles: dict = {}

    def get(name: str) -> torch.Tensor:
        f = idx[name]
        if f not in handles:
            handles[f] = safe_open(str(Path(PIN) / f), framework="pt", device="cpu")
        return handles[f].get_tensor(name)

    out = []
    for i in range(LAYERS):
        p = f"model.layers.{i}."
        for parts in (["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"], ["self_attn.o_proj"],
                      ["mlp.gate_proj", "mlp.up_proj"], ["mlp.down_proj"]):
            w = torch.cat([get(p + q + ".weight") for q in parts])
            if w.dtype != torch.int8:
                raise SystemExit(f"{p}{parts[0]}.weight is {w.dtype}, not int8")
            out.append(w.contiguous().to(dev))
    return out


def step(ws: list[torch.Tensor], t: int, rows: tuple[int, int], s: int, seed: int, dev):
    """The step's calls (h1r's and receipts.py's view of each) at T = t, call i admitting rows[i % 2] slots."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    rng = np.random.default_rng(seed)
    ns = [int(w.shape[0]) for w in ws]
    per = [(t // 32) * (n // 32) for n in ns]
    offs = [sum(per[:i]) for i in range(len(per))]
    u_step = sum(per)
    calls, meta = [], []
    for i, w in enumerate(ws):
        k = int(w.shape[1])
        mc = rows[i % 2]
        a = torch.zeros(t, k, dtype=torch.int8)
        a[:mc] = torch.randint(-127, 128, (mc, k), dtype=torch.int8, generator=g)
        a = a.to(dev)
        admit = np.arange(t) < mc
        live = torch.tensor([bool(admit[32 * r:32 * r + 32].any()) for r in range(t // 32)], dtype=torch.uint8, device=dev)
        root_a, root_b = rng.bytes(64), rng.bytes(64)
        roots = torch.from_numpy(np.concatenate([words(root_a), words(root_b)])).to(dev)
        base = s * u_step + offs[i]
        calls.append(Call(a, w, base, roots, live))
        meta.append({"admit": admit, "root_a": root_a, "root_b": root_b, "base": base, "k": k, "n": int(w.shape[0]),
                     "units": per[i], "live_units": int(live.sum().item()) * (int(w.shape[0]) // 32)})
    return calls, meta


def ours(plan, i: int) -> np.ndarray:
    return plan.receipts(i).cpu().numpy().view(np.uint32)


def gate(lib, v: int, calls, meta, salt: bytes, salt_dev, want: list, refs_want: list, replay_want: list) -> dict:
    """Version v against the step's answers (computed once, for every version)."""
    doc: dict = {"version": v}
    plan = lib.plan(calls)
    plan.out.fill_(int(POISON))
    lib.launch(plan, salt_dev, version=v)
    torch.cuda.synchronize()
    full = {"calls": 0, "units": 0, "bad_units": 0, "bad_calls": []}
    for i, w in enumerate(want):
        got = ours(plan, i)
        bad = int((got != w).any(1).sum())
        full["calls"] += 1
        full["units"] += int(w.shape[0])
        full["bad_units"] += bad
        if bad:
            full["bad_calls"].append({"call": i, "bad_units": bad, "first": int(np.nonzero((got != w).any(1))[0][0])})
    doc["vs_receipts_py"] = full
    doc["vs_h1tile_reference"] = [{**r, "equal": ours(plan, r["call"])[r["unit"]].tolist() == w}
                                  for r, w in refs_want]

    ctl = lib.plan(calls)
    ctl.out.fill_(int(POISON))
    lib.launch(ctl, salt_dev, nowrite=True, version=v)
    torch.cuda.synchronize()
    doc["control"] = {"poison_kept": bool((ctl.out == int(POISON)).all().item()),
                      "refused": bool(all((ours(ctl, i) != ours(plan, i)).any(1).all() for i in range(len(calls))))}
    del ctl

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        lib.launch(plan, salt_dev, version=v)
    replays = []
    for salt_i, picks in replay_want:
        salt_dev.copy_(torch.from_numpy(words(salt_i)))
        plan.out.fill_(int(POISON))
        g.replay()
        torch.cuda.synchronize()
        bad = sum(int(not np.array_equal(ours(plan, ci)[u], w)) for ci, u, w in picks)
        replays.append({"salt_changed": salt_i != salt, "bad_units": bad, "units": len(picks)})
    salt_dev.copy_(torch.from_numpy(words(salt)))
    doc["graph_replay"] = replays
    doc["ok"] = (full["bad_units"] == 0 and full["calls"] == len(calls) and all(r["equal"] for r in doc["vs_h1tile_reference"])
                 and doc["control"]["poison_kept"] and doc["control"]["refused"] and all(r["bad_units"] == 0 for r in replays))
    return doc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--cubin", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--versions", default="1,2")
    a = ap.parse_args()
    versions = [int(x) for x in a.versions.split(",")]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    lib = H1R(a.lib)
    cub = RC.Cubin(a.cubin)
    t0 = time.perf_counter()
    ws = weights(dev)
    doc: dict = {"weights_s": round(time.perf_counter() - t0, 1), "gpu": torch.cuda.get_device_name(), "versions": versions}
    print(json.dumps(doc), flush=True)

    # -- the step's answers at T = 64, once --------------------------------------------------------------------------
    calls, meta = step(ws, 64, (37, 20), 3, 1, dev)
    salt = os.urandom(64)
    salt_dev = torch.from_numpy(words(salt)).to(dev)
    t0 = time.perf_counter()
    want = []
    live_units = 0
    for i, m in enumerate(meta):
        w, lu = RC.gated_receipts(RC.Call(calls[i].a, calls[i].b), m["admit"], salt, m["base"], m["root_a"], m["root_b"], cub,
                                  chunk=max(16, (1 << 20) // m["k"]))
        want.append(w)
        live_units += len(lu)
    doc["receipts_py"] = {"calls": len(want), "live_units": live_units, "s": round(time.perf_counter() - t0, 1)}
    print(json.dumps({"receipts_py": doc["receipts_py"]}), flush=True)
    refs_want = []
    for ci, u in [(0, 0), (3, 0), (2, 895), (1, 128 + 5), (7, 127), (4, 192 + 5)]:
        m = meta[ci]
        t1 = time.perf_counter()
        w = RC.gated_reference(RC.Call(calls[ci].a, calls[ci].b), m["admit"], salt, m["base"], m["root_a"], m["root_b"], u)
        refs_want.append(({"call": ci, "k": m["k"], "unit": u, "live": bool(any(w)), "s": round(time.perf_counter() - t1, 1)},
                          w))
    salt2 = os.urandom(64)
    replay_want = []
    for salt_i in (salt, salt2):
        picks = []
        for ci, u in [(0, 0), (0, 191), (2, 400), (3, 64), (6, 300), (127, 127)]:
            m = meta[ci]
            if live_of(m, u):
                w = RC.call_receipts(RC.Call(calls[ci].a, calls[ci].b), salt_i, m["base"], m["root_a"], m["root_b"], cub,
                                     only=[u])
                picks.append((ci, u, w[0]))
        replay_want.append((salt_i, picks))

    doc["gates"] = []
    for v in versions:
        gd = gate(lib, v, calls, meta, salt, salt_dev, want, refs_want, replay_want)
        doc["gates"].append(gd)
        print(json.dumps({"gate": gd}), flush=True)
    ok = all(gd["ok"] for gd in doc["gates"])
    doc["gate_ok"] = ok
    (out / "gate.json").write_text(json.dumps(doc, indent=1) + "\n")
    if not ok:
        print("h1rgate: GATE FAILED", flush=True)
        return 2
    del want

    # -- time, versions interleaved rep by rep -----------------------------------------------------------------------
    timing = []
    for t, rows in ((64, (32, 32)), (64, (5, 5)), (256, (256, 256)), (256, (200, 200))):
        cs, ms = step(ws, t, rows, 3, 2, dev)
        plans, graphs = {}, {}
        for v in versions:
            plans[v] = lib.plan(cs)
            graphs[v] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graphs[v]):
                lib.launch(plans[v], salt_dev, version=v)
        if t == 256 and rows == (200, 200):
            spots = [(0, 0), (2, 896 * 6 + 3), (3, 128 * 5 + 7)]
            wants = {(ci, u): RC.call_receipts(RC.Call(cs[ci].a, cs[ci].b), salt_from(salt_dev), ms[ci]["base"],
                                               ms[ci]["root_a"], ms[ci]["root_b"], cub, only=[u])[0] for ci, u in spots}
            for v in versions:
                graphs[v].replay()
                torch.cuda.synchronize()
                bad = sum(int(not np.array_equal(ours(plans[v], ci)[u], w)) for (ci, u), w in wants.items())
                if bad:
                    print(json.dumps({"t256_bad_units": bad, "version": v}), flush=True)
                    return 2
        for v in versions:
            graphs[v].replay()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        ms_ = {v: [] for v in versions}
        for _ in range(a.reps):
            for v in versions:
                e0.record()
                graphs[v].replay()
                e1.record()
                e1.synchronize()
                ms_[v].append(e0.elapsed_time(e1))
        live_u = sum(m["live_units"] for m in ms)
        for v in versions:
            med = statistics.median(ms_[v])
            row = {"version": v, "t": t, "admitted_rows": rows[0], "units": sum(m["units"] for m in ms), "live_units": live_u,
                   "step_ms": round(med, 4), "step_ms_all": [round(x, 4) for x in ms_[v]],
                   "us_per_live_unit": round(1e3 * med / live_u, 5), "live_units_per_s": round(live_u / (med / 1e3))}
            timing.append(row)
            print(json.dumps({"time": row}), flush=True)
        del graphs, plans
    doc["timing"] = timing
    (out / "gate.json").write_text(json.dumps(doc, indent=1) + "\n")
    return 0


def live_of(m: dict, u: int) -> bool:
    rb = u // (m["n"] // 32)
    return bool(m["admit"][32 * rb:32 * rb + 32].any())


def salt_from(salt_dev: torch.Tensor) -> bytes:
    return salt_dev.cpu().numpy().view("<u8").astype(">u8").tobytes()


if __name__ == "__main__":
    sys.exit(main())
