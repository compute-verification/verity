"""CacheStep_v3's cost at the GLM-5.3 class: the RAM pool's operations per window on synthetic loads at the class's engine
(NB = 5,000, B = 64, R = 256, T = 8,192, P = 4,964 steps a window), each priced at its slot's ANDs (`slots.py`) plus
the in-circuit hash of its witness, plus the parts that are per call, per block and per window.

    python3 cost.py [--slots slots.json] [--json OUT] [--quick]

The pricing (ANDs; `price`):
- an operation runs in SHAPE[kind] consecutive slots of the window's access stream (time = the slot's index, wiring).
  Its accesses cost what its measured slot costs, plus MUX_OVERHEAD for the stream's per-(kind, position) multiplexing
  (an estimate: the stream is not built), plus 2 products into the running fingerprints, plus SHA-512 over its witness
  bits (the read values and times and its descriptor) at 58,120 ANDs per 1,024 bits: the witness is committed as hm96
  rows before the coins are drawn and its rows are hashed in-circuit.
- lookup and cache read their key from RK[(r, k)] (one more access, a 2-chunk record) instead of a key descriptor.
- each block of tokens gets its key once: a polynomial hash under a live coin β (9 products over its 64 tokens of 18
  bits, chained to its parent's key) and one RK write.
- each pool call is one factor on each side of the bus between ReplayStep and the access stream (a product and a mux),
  with its result (hit and evictable counts, or returns) as witness.
- each window: the initial and final records of the dense memories (B, K: NB each) and of the sparse ones' touched
  addresses (M, RB, RK: a sorted list with its comparators), the final times as witness, and the carried pool (B, K, M,
  RB, RK values) hashed in and out as the window's registered state.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

import pool_ram as PR
from pool_ram import RP

MUL = 2187
SHA_BIT = 58120 / 1024
MUX_OVERHEAD = 0.15
S = PR.GLM
P_STEPS = 4964
#: ODBRANCH's estimate for a memory-checked pool with a per-step unit budget, and CacheStep_v2's in CompileSlice_v3
ODBRANCH = 1.0e12
V2 = 9.2e14
#: ODBRANCH's share of the node at 1.0e12 ANDs a window: plain GPU proving (low, high) and --zk (median)
SHARE_PLAIN = (0.013, 0.046)
SHARE_ZK = 0.10

#: accesses' witness bits by memory: the value read and its time
VBITS = {"B": 4 * S.NW + S.REFW + S.RW, "K": S.HW, "M": 2 * S.NW, "RB": S.NW, "RK": S.HW}


def access_ands(mem: str) -> float:
    """One access's check, analytically: its two factors (γ-products for a second chunk, the unused-position mux), the
    two products into the running fingerprints, the time comparison and the unused-position zero assertions."""
    chunks = 2 if mem in ("K", "M", "RK") else 1
    return 2 * ((chunks - 1) * MUL + 128) + 2 * MUL + (S.TW + 1) + VBITS[mem] + S.TW


ANALYTIC_SLOTS = {k: {"ands": sum(access_ands(m) for m in mems), "witness_bits": wb}
                  for k, mems, wb in (
                      ("lookup", ("M", "B"), S.HW + 2 * S.TW + VBITS["M"] + VBITS["B"]),
                      ("take", ("B", "B", "B", "RB"), S.RW + S.KW + S.NW + 4 * S.TW + 3 * VBITS["B"]),
                      ("alloc", ("B", "B", "B", "K", "M", "B", "B", "RB"),
                       S.RW + S.KW + 8 * S.TW + 5 * VBITS["B"] + VBITS["K"] + VBITS["M"]),
                      ("cache", ("RB", "K", "M", "B", "B"), S.RW + S.KW + S.HW + 5 * S.TW + VBITS["RB"] + VBITS["M"] + 2 * VBITS["B"]),
                      ("free", ("RB", "B", "K", "B", "B"), S.RW + S.KW + S.NW + 5 * S.TW + VBITS["RB"] + 3 * VBITS["B"] + VBITS["K"]))}


def price(slots: dict[str, dict]) -> dict[str, float]:
    """ANDs per operation, per call, per keyed block, and the per-window parts' unit prices."""
    rk = access_ands("RK") + SHA_BIT * (S.HW + S.TW)
    out = {}
    for kind, row in slots.items():
        kind = kind.split(" ")[0]
        c = row["ands"] * (1 + MUX_OVERHEAD) + 2 * MUL + SHA_BIT * row["witness_bits"]
        if kind in ("lookup", "cache"):
            c += rk - SHA_BIT * S.HW + (SHA_BIT * (S.RW + S.KW) if kind == "lookup" else 0)
        out[kind] = c
    out["call"] = 2 * (MUL + 128) + SHA_BIT * 2 * (S.KW + 1)
    out["block key"] = 9 * MUL + access_ands("RK") + SHA_BIT * S.TW
    out["dense address"] = {"B": 2 * (MUL + 128) + SHA_BIT * (S.TW + 2 * VBITS["B"]),
                            "K": 2 * (2 * MUL + 128) + SHA_BIT * (S.TW + 2 * VBITS["K"])}
    out["sparse address"] = {m: 2 * ((2 if m != "RB" else 1) * MUL + 128) + aw + SHA_BIT * (aw + 2 * VBITS[m] + S.TW)
                             for m, aw in (("M", S.HW), ("RB", S.RW + S.KW), ("RK", S.RW + S.KW))}
    return out


# --- GLM-5.3-like loads ---------------------------------------------------------------------------------------------------


def glm_load(seed: int, *, kind: str, steps: int = 3000, NB: int = 5000, B: int = 64, R: int = 256, T: int = 8192,
             L: int = 32768) -> tuple[RP.Engine, list[RP.RequestIO], list[int]]:
    """Requests arriving over `steps` steps: a few system prompts, multi-turn conversations that resend their history,
    and fresh documents. `kind` "chat" (moderate arrivals, much reuse) or "pressure" (arrivals past what the pool holds,
    long prompts: preemption and a blocked waiting queue)."""
    rng = random.Random(seed)
    tok = lambda k: tuple(rng.randrange(1, 150000) for _ in range(k))  # noqa: E731
    systems = [tok(rng.randrange(800, 6000)) for _ in range(8)]
    rate = 0.35 if kind == "chat" else 0.9
    convs: list[tuple[int, ...]] = []
    reqs, admit = [], []
    step = 0.0
    while step < steps:
        step += rng.expovariate(rate)
        u = rng.random()
        if convs and u < 0.45:
            j = rng.randrange(len(convs))
            prompt = convs[j] + tok(rng.randrange(30, 600))
        elif u < 0.9:
            prompt = rng.choice(systems) + tok(rng.randrange(50, 3000 if kind == "chat" else 12000))
        else:
            prompt = tok(rng.randrange(2000, 20000))
        prompt = prompt[: L - 2100]
        out_len = rng.randrange(50, 2000)
        eos = rng.random() < 0.7
        toks = prompt + tok(out_len)
        reqs.append(RP.RequestIO(len(prompt), out_len + (rng.randrange(0, 200) if eos else 0), out_len, eos, toks))
        if rng.random() < 0.6:
            convs.append(toks)
            if len(convs) > 400:
                convs.pop(0)
        admit.append(int(step))
    eng = RP.Engine(max_num_seqs=R, max_num_batched_tokens=T, max_model_len=L, num_gpu_blocks=NB, block_size=B,
                    enable_prefix_caching=True, async_scheduling=True)
    return eng, reqs, admit


def measure(eng: RP.Engine, reqs: list[RP.RequestIO], admit: list[int]) -> dict[str, Any]:
    m = PR.run(eng, reqs, every=10**9, decide=lambda i, k, o: admit[i] <= k)
    pool = m.pool
    steps = m.counts["steps"]
    per_step = [sum(c.values()) for c in m.steps[1:]] or [0]
    acc_per_step = sorted(sum(PR.SHAPE[k] * n for k, n in c.items() if k in PR.SHAPE) for c in m.steps[1:])
    keyed = sum(min(len(r.tokens), r.prompt_len + r.output_len) // eng.block_size for r in reqs)
    touched = Counter(a[0] for a in pool.ram.first)
    return {"steps": steps, "tokens": m.counts["tokens"], "ops": {k: pool.ops[k] for k in (*PR.SHAPE, "evict")},
            "calls": m.counts["calls"], "lookups": m.counts["lookups"], "allocations": m.counts["allocations"],
            "refused after a lookup": m.counts["refused after a lookup"], "frees": m.counts["frees"],
            "certified refusals": m.counts["certified refusals"],
            "lookup units a certificate saves": m.counts["lookup units a certificate saves"],
            "repeated walks": m.counts["repeated walks"],
            "lookup units in repeated walks": m.counts["lookup units in repeated walks"],
            "blocks keyed": keyed, "touched": dict(touched),
            "accesses per step": {"mean": sum(acc_per_step) / max(len(acc_per_step), 1),
                                  "p99": acc_per_step[int(0.99 * (len(acc_per_step) - 1))] if acc_per_step else 0,
                                  "max": acc_per_step[-1] if acc_per_step else 0},
            "ops per step max": max(per_step), "requests": len(reqs)}


def window_cost(load: dict[str, Any], pr: dict[str, Any], NB: int = 5000, certify: bool = False) -> dict[str, float]:
    """ANDs for a window of P_STEPS steps at the load's per-step rates; with `certify`, a refused head-of-line
    candidate's walk replaced by the one-unit refusal certificate where it suffices (`pool_ram.Mirror.certify`)."""
    f = P_STEPS / load["steps"]
    ops = dict(load["ops"])
    if certify:
        ops["lookup"] -= load["lookup units a certificate saves"]
    parts = {f"ops: {k}": n * f * pr[k] for k, n in ops.items() if k in PR.SHAPE}
    parts["calls (bus)"] = load["calls"] * f * pr["call"]
    parts["block keys"] = load["blocks keyed"] * f * pr["block key"]
    parts["dense init/final (B, K)"] = NB * (pr["dense address"]["B"] + pr["dense address"]["K"])
    t = load["touched"]
    parts["sparse init/final (M, RB, RK)"] = f * (t.get("M", 0) * pr["sparse address"]["M"]
                                                  + t.get("RB", 0) * pr["sparse address"]["RB"]) \
        + load["blocks keyed"] * f * pr["sparse address"]["RK"]
    parts["total"] = sum(parts.values())
    return parts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--slots", type=Path)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--quick", action="store_true", help="one small load (a smoke test)")
    a = ap.parse_args()
    slots = ANALYTIC_SLOTS
    if a.slots and a.slots.exists():
        slots = {k: v for k, v in json.loads(a.slots.read_text())["slots"].items() if "no caching" not in k}
    pr = price(slots)
    print("ANDs per operation:", {k: round(v) for k, v in pr.items() if not isinstance(v, dict)})
    loads = [("chat", 1), ("pressure", 2)] if not a.quick else [("chat", 1)]
    report: dict[str, Any] = {"price": pr, "slots": slots, "loads": {}}
    for kind, seed in loads:
        kw = {"steps": 300, "NB": 800} if a.quick else {}
        eng, reqs, admit = glm_load(seed, kind=kind, **kw)
        load = measure(eng, reqs, admit)
        print(kind, json.dumps(load))
        report["loads"][kind] = {"load": load}
        for cert in (False, True):
            cost = window_cost(load, pr, NB=eng.num_gpu_blocks, certify=cert)
            share = cost["total"] / ODBRANCH
            report["loads"][kind]["certified" if cert else "walked"] = {
                "window ANDs": cost, "vs ODBRANCH": share, "vs CacheStep_v2": cost["total"] / V2,
                "share of the node, plain GPU": [share * x for x in SHARE_PLAIN], "share of the node, --zk": share * SHARE_ZK}
            print(kind, "certified" if cert else "walked", {k: f"{v:.3g}" for k, v in cost.items()},
                  f"= {share:.3g} × ODBRANCH's 1.0e12")
    if a.json:
        a.json.write_text(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
