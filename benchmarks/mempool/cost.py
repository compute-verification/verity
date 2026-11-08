"""CacheStep_v3's cost at the GLM-5.3 class: the RAM pool's operations in the busiest window (P = 4,964 consecutive
steps, `measure`) of synthetic loads at the class's engine (NB = 5,000, B = 64, R = 256, T = 8,192), each priced at its
slot's ANDs (`slots.py`) plus the in-circuit hash of its witness, plus the parts that are per call, per block and per
window.

    python3 cost.py [--slots slots.json] [--json OUT] [--quick]

The pricing (ANDs; `price`, `slot_price`):
- an operation runs in STREAM_SHAPE[kind] consecutive slots of the window's access stream (time = the slot's index,
  wiring). Which operation runs in which slot is private, so the stream is U slots that can each perform any kind's
  access at any position: the window pays U × a slot's price, U being what the window needs (a public budget at least
  that). The slot isn't built; `slot_price` bounds it from the measured operation slots. Every slot adds 2 products into
  the running fingerprints and SHA-512 over its witness (SLOT_WITNESS) at 58,120 ANDs per 1,024 bits: the witness is
  committed as hm96 rows before the coins are drawn and its rows are hashed in-circuit.
- `price` is the floor beside it: each operation at its own kind's slot (plus MUX_OVERHEAD), as if kinds were public.
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
#: lookup and cache also read their block's key from RK
STREAM_SHAPE = {k: n + (k in ("lookup", "cache")) for k, n in PR.SHAPE.items()}
#: a stream slot's witness: the widest read (a key, and its time) and an operation's descriptor (kind, request, block
#: index, block id), which every slot carries since its kind is private
SLOT_WITNESS = S.HW + S.TW + 3 + S.RW + S.KW + S.NW


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


def slot_price(slots: dict[str, dict]) -> tuple[float, float]:
    """A stream slot's ANDs, (low, high). Low: the costliest kind's logic per access (its slot's ANDs over its accesses)
    with MUX_OVERHEAD. High: every kind's side by side, and both γ-products of a 2-chunk record. Both add the two products
    into the running fingerprints and the hash of SLOT_WITNESS."""
    per = [slots[k]["ands"] / PR.SHAPE[k] for k in PR.SHAPE]
    common = 2 * MUL + SHA_BIT * SLOT_WITNESS
    return max(per) * (1 + MUX_OVERHEAD) + common, sum(per) + 2 * MUL + common


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


def accesses(c: Counter) -> int:
    return sum(PR.SHAPE[k] * n for k, n in c.items() if k in PR.SHAPE)


def measure(eng: RP.Engine, reqs: list[RP.RequestIO], admit: list[int]) -> dict[str, Any]:
    """The load's busiest window: the P_STEPS consecutive steps with the most accesses (the whole replay if shorter),
    found on a first replay; a second, identical one collects the sparse addresses that window touches."""
    how = {"every": 10**9, "keep": False, "decide": lambda i, k, o: admit[i] <= k}
    m = PR.run(eng, reqs, **how)
    per = [accesses(c) for c in m.steps]
    n = min(P_STEPS, len(per))
    run = sum(per[:n])
    best, s0 = run, 0
    for s in range(1, len(per) - n + 1):
        run += per[s + n - 1] - per[s - 1]
        if run > best:
            best, s0 = run, s
    window = range(s0, s0 + n)
    w = PR.run(eng, reqs, window=window, **how)
    assert [accesses(c) for c in w.steps] == per, "the replay is not deterministic"
    tot = sum((w.steps[s] for s in window), Counter())
    acc = sorted(per[1:])
    return {"NB": eng.num_gpu_blocks, "steps": m.counts["steps"], "requests": len(reqs), "tokens": m.counts["tokens"],
            "replay": {"ops": {k: m.pool.ops[k] for k in (*PR.SHAPE, "evict")}, **{k: m.counts[k] for k in (
                "calls", "lookups", "allocations", "refused allocations", "refused after a lookup", "frees",
                "certified refusals", "lookup units a certificate saves", "repeated walks",
                "lookup units in repeated walks")}},
            "window": {"steps": [s0, s0 + n], "accesses": best, "ops": {k: tot[k] for k in (*PR.SHAPE, "evict")},
                       "calls": tot["calls"], "keys": tot["keys"], "repeated units": tot["repeated units"],
                       "certificate saves": tot["certificate saves"],
                       "touched": {k: len(v) for k, v in w.touched.items()}},
            "accesses per step": {"mean": sum(acc) / max(len(acc), 1),
                                  "p99": acc[int(0.99 * (len(acc) - 1))] if acc else 0,
                                  "p99.9": acc[int(0.999 * (len(acc) - 1))] if acc else 0,
                                  "max": acc[-1] if acc else 0}}


def window_cost(load: dict[str, Any], pr: dict[str, Any], slot: tuple[float, float], NB: int = 5000,
                lookups: str = "walked") -> dict[str, float]:
    """ANDs for the load's busiest window: U, the stream at the slot's (low, high) price, the floor, and the per-call,
    per-key and per-window parts. `lookups`: "walked" (every lookup walks, as vLLM does), "certified" (a refused
    head-of-line candidate's walk replaced by the one-unit refusal certificate where it suffices,
    `pool_ram.Mirror.certify`), or "memo" (a walk whose (hit, evictable) equals the request's last one is reused; an
    upper bound on the head-of-line memo's saving, which doesn't count its marks and dirty bit)."""
    w = load["window"]
    ops = {k: w["ops"][k] for k in PR.SHAPE}
    ops["lookup"] -= {"walked": 0, "certified": w["certificate saves"], "memo": w["repeated units"]}[lookups]
    fixed = {"calls (bus)": w["calls"] * pr["call"], "block keys": w["keys"] * pr["block key"],
             "dense init/final (B, K)": NB * (pr["dense address"]["B"] + pr["dense address"]["K"]),
             "sparse init/final (M, RB, RK)": sum(w["touched"].get(k, 0) * pr["sparse address"][k] for k in ("M", "RB", "RK"))}
    u = sum(STREAM_SHAPE[k] * n for k, n in ops.items())
    out = {"U": u, "stream, low": u * slot[0], "stream, high": u * slot[1],
           "floor (each operation at its kind's slot)": sum(n * pr[k] for k, n in ops.items()), **fixed}
    f = sum(fixed.values())
    out |= {"total, low": out["stream, low"] + f, "total, high": out["stream, high"] + f,
            "total, floor": out["floor (each operation at its kind's slot)"] + f}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--slots", type=Path)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--quick", action="store_true", help="one small load (a smoke test)")
    ap.add_argument("--from", dest="src", type=Path, help="reprice a report's measured loads instead of replaying them")
    a = ap.parse_args()
    slots = ANALYTIC_SLOTS
    if a.slots and a.slots.exists():
        slots = {k: v for k, v in json.loads(a.slots.read_text())["slots"].items() if "no caching" not in k}
    pr, slot = price(slots), slot_price(slots)
    print("ANDs per operation:", {k: round(v) for k, v in pr.items() if not isinstance(v, dict)},
          "a stream slot:", [round(x) for x in slot])
    report: dict[str, Any] = {"price": pr, "slot": slot, "slots": slots, "loads": {}}
    if a.src:
        measured = {k: (v["load"], v["load"].get("NB", 5000)) for k, v in json.loads(a.src.read_text())["loads"].items()}
    else:
        measured = {}
        for kind, seed in [("chat", 1), ("pressure", 2)] if not a.quick else [("chat", 1)]:
            eng, reqs, admit = glm_load(seed, kind=kind, **({"steps": 300, "NB": 800} if a.quick else {}))
            measured[kind] = (measure(eng, reqs, admit), eng.num_gpu_blocks)
    for kind, (load, NB) in measured.items():
        print(kind, json.dumps(load))
        report["loads"][kind] = {"load": load}
        for how in ("walked", "certified", "memo"):
            cost = window_cost(load, pr, slot, NB=NB, lookups=how)
            lo, hi = cost["total, low"] / ODBRANCH, cost["total, high"] / ODBRANCH
            report["loads"][kind][how] = {
                "window ANDs": cost, "vs ODBRANCH": [lo, hi], "vs CacheStep_v2": [x * ODBRANCH / V2 for x in (lo, hi)],
                "share of the node, plain GPU": [lo * SHARE_PLAIN[0], hi * SHARE_PLAIN[1]],
                "share of the node, --zk": [lo * SHARE_ZK, hi * SHARE_ZK]}
            print(kind, how, {k: f"{v:.3g}" for k, v in cost.items()}, f"= {lo:.3g}–{hi:.3g} × ODBRANCH's 1.0e12")
    if a.json:
        a.json.write_text(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
