"""CacheStep_v3's cost at the GLM-5.3 class: the RAM pool's operations in the busiest window (P = 4,964 consecutive
steps, `measure`) of synthetic loads at the class's engine (NB = 5,000, B = 64, R = 256, T = 8,192), priced at the
measured parts (`measured`: `cache_step_v3`'s PoolSlot_v1, PoolEntry_v1 and PoolEnd at the class's widths, and
`block_key`'s BlockKey_v1 and KeyPrint_v1) plus the in-circuit hash of their witness.

    python3 cost.py [--json OUT] [--quick] [--from REPORT]

The pricing (ANDs; `window_cost`):
- the stream: a window's calls run in U slots, U the accesses its operations make (STREAM: each kind's accesses, at
  most; an access a call doesn't need takes no slot, so U is at most this), each slot a PoolSlot_v1 and SHA-512 over its
  witness (SLOT_WITNESS) at 58,120 ANDs per 1,024 bits: the witness is committed as hm96 rows before the coins are drawn
  and its rows are hashed in-circuit. The window takes the ladder's class of U (`size_class`), so it pays for that.
  With the head-of-line memo ("memo"), U is measured instead: the slots of CacheStep_v3's own stream over the replay
  (`stream_window`), its busiest P steps.
- the touched list: NE entries, each a PoolEntry_v1 and the hash of its witness (ENTRY_WITNESS: its address, its
  initial value, the carried pool's, and its final value and time). B and K are listed whole (NB each), M, RB and RK by
  the addresses the window touches; NE too is the ladder's class.
- the bus: the stream's side is in PoolSlot_v1; ReplayStep_v1's is one factor a call, with the call's results as its
  witness.
- each block of tokens gets its key once: BlockKey_v1 at 64 tokens of 18 bits (SHA-512 over its parent's key and its
  tokens, two compressions, and its print under the window's live coin).
- each window: the 512-bit keys it carries (the cached blocks', NB at most, and its requests' blocks', at most the RK
  addresses it touches) hashed in and out and printed under its coin (KeyPrint_v1).
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import pool_ram as PR
import stream as ST
from pool_ram import RP

from verity.core.protocols.certifier.head_check import ands
from verity.ml.circuits.defs import bind
from verity.ml.commitments.gates import block_key as BK
from verity.ml.protocols.certifier import cache_step_v3 as V3

MUL = 2187
SHA_BLOCK = 58120
SHA_BIT = SHA_BLOCK / 1024
S = PR.GLM
P_STEPS = 4964
#: ODBRANCH's estimate for a memory-checked pool with a per-step unit budget, and CacheStep_v2's in CompileSlice_v3
ODBRANCH = 1.0e12
V2 = 9.2e14
#: ODBRANCH's share of the node at 1.0e12 ANDs a window: plain GPU proving (low, high) and --zk (median)
SHARE_PLAIN = (0.013, 0.046)
SHARE_ZK = 0.10

#: the stream's accesses a pool operation makes, at most (`cache_step_v3`'s calls, per lookup unit or block): a lookup
#: RK, M and B; a take RK, M, B and its queue neighbours, RB; an alloc the queue's head, its next, K and RB, and an
#: eviction M and the evicted copy's two neighbours; a cache RB, RK, K, M, the last copy and B; a free RB, K, B and,
#: returning, the two it goes between
STREAM = {"lookup": 3, "take": 6, "alloc": 4, "evict": 3, "cache": 6, "free": 5}
#: a stream slot's witness: start, kind, the call's (r, a, b), the value read and its time
SLOT_WITNESS = 1 + 3 + S.RW + 2 * (S.KW + 1) + 128 + S.TW
#: a list entry's: (en, tag, address), its initial value, its final value and time
ENTRY_WITNESS = 1 + 3 + 128 + 2 * 128 + S.TW


def measured(NB: int = 5000) -> dict[str, int]:
    """ANDs of CacheStep_v3's parts at the class's widths, and of a block's key with its print and of a carried key's
    print."""
    p = V3.part_ands((NB, S.RW, S.KW, S.REFW, S.TW))
    return {"slot": p["PoolSlot"], "entry": p["PoolEntry"], "end": p["PoolEnd"],
            "block key": ands(bind(BK.BlockKey, B=64, TOK=18)), "key print": ands(bind(BK.KeyPrint))}


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


def busiest(per: list[int]) -> tuple[int, int, int]:
    """The P_STEPS consecutive steps (all, if fewer) with the largest sum: (sum, first step, steps)."""
    n = min(P_STEPS, len(per))
    run = sum(per[:n])
    best, s0 = run, 0
    for s in range(1, len(per) - n + 1):
        run += per[s + n - 1] - per[s - 1]
        if run > best:
            best, s0 = run, s
    return best, s0, n


def stream_window(eng: RP.Engine, reqs: list[RP.RequestIO], admit: list[int]) -> dict[str, Any]:
    """The load's stream with the head-of-line memo: `stream.py`'s `Stream` beside the replay's pool (each call's results
    held to the pool's, the pools compared at the end), in windows of 2^16 slots (each starts with no memo kept), and the
    P_STEPS consecutive steps with the most slots."""
    m = ST.run(eng, reqs, {"decide": lambda i, k, o: admit[i] <= k}, 1 << 16, check=False, every=0)
    best, s0, n = busiest([c["slots"] for c in m.steps])
    c = m.pool.counts
    return {"slots": best, "steps": [s0, s0 + n], "hits": c["hit"], "memo hits": c["memo hits"],
            "re-walks": c["re-walks"], "all slots": m.pool.total}


def measure(eng: RP.Engine, reqs: list[RP.RequestIO], admit: list[int]) -> dict[str, Any]:
    """The load's busiest window: the P_STEPS consecutive steps with the most accesses (the whole replay if shorter),
    found on a first replay; a second, identical one collects the sparse addresses that window touches."""
    how = {"every": 10**9, "keep": False, "decide": lambda i, k, o: admit[i] <= k}
    m = PR.run(eng, reqs, **how)
    per = [accesses(c) for c in m.steps]
    best, s0, n = busiest(per)
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


def window_cost(load: dict[str, Any], m: dict[str, int], NB: int = 5000, lookups: str = "walked") -> dict[str, float]:
    """ANDs for the load's busiest window: U and NE and their classes, the stream, the touched list, and the per-call,
    per-key and per-window parts. `lookups`: "walked" (every lookup walks, as vLLM does), "certified" (a refused
    head-of-line candidate's walk replaced by the one-unit refusal certificate where it suffices,
    `pool_ram.Mirror.certify`), or "memo": CacheStep_v3's own stream with its head-of-line memo, U its busiest window's
    slots (`stream_window`), or for a load measured without it, the bound "memo" had before the memo was built (a walk
    whose (hit, evictable) equals the request's last one is reused, its stamps and dirty bit not counted). The touched
    list, the calls and the keys are `measure`'s window's in every case."""
    w = load["window"]
    ops = {k: w["ops"][k] for k in (*PR.SHAPE, "evict")}
    ops["lookup"] -= {"walked": 0, "certified": w["certificate saves"], "memo": w["repeated units"]}[lookups]
    u = sum(STREAM[k] * n for k, n in ops.items())
    if lookups == "memo" and "stream" in load:
        u = load["stream"]["slots"]
    ne = 2 * NB + sum(w["touched"].get(k, 0) for k in ("M", "RB", "RK"))
    ns_c, ne_c = V3.size_class(u), V3.size_class(ne)
    out = {"U": u, "NS (U's class)": ns_c, "NE": ne, "NE's class": ne_c,
           "stream": ns_c * (m["slot"] + SHA_BIT * SLOT_WITNESS),
           "touched list": ne_c * (m["entry"] + SHA_BIT * ENTRY_WITNESS) + m["end"],
           "bus, ReplayStep_v1's side": w["calls"] * (MUL + 128 + SHA_BIT * 2 * (S.KW + 1)),
           "block keys": w["keys"] * m["block key"],
           "carried 512-bit keys": (NB + w["touched"].get("RK", 0)) * (SHA_BIT * 2 * 512 + m["key print"])}
    out["total"] = sum(v for k, v in out.items() if k not in ("U", "NS (U's class)", "NE", "NE's class"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", type=Path)
    ap.add_argument("--quick", action="store_true", help="one small load (a smoke test)")
    ap.add_argument("--from", dest="src", type=Path, help="reprice a report's measured loads instead of replaying them")
    a = ap.parse_args()
    m = measured()
    slot = m["slot"] + SHA_BIT * SLOT_WITNESS
    print("measured ANDs:", m, f"a stream slot with its witness hash: {slot:.0f}")
    report: dict[str, Any] = {"measured": m, "slot": slot, "loads": {}}
    if a.src:
        loads = {k: (v["load"], v["load"].get("NB", 5000)) for k, v in json.loads(a.src.read_text())["loads"].items()}
    else:
        loads = {}
        for kind, seed in [("chat", 1), ("pressure", 2)] if not a.quick else [("chat", 1)]:
            eng, reqs, admit = glm_load(seed, kind=kind, **({"steps": 300, "NB": 800} if a.quick else {}))
            loads[kind] = ({**measure(eng, reqs, admit), "stream": stream_window(eng, reqs, admit)}, eng.num_gpu_blocks)
    for kind, (load, NB) in loads.items():
        print(kind, json.dumps(load))
        report["loads"][kind] = {"load": load}
        for how in ("walked", "certified", "memo"):
            cost = window_cost(load, m if NB == 5000 else measured(NB), NB=NB, lookups=how)
            x = cost["total"] / ODBRANCH
            report["loads"][kind][how] = {
                "window ANDs": cost, "vs ODBRANCH": x, "vs CacheStep_v2": x * ODBRANCH / V2,
                "share of the node, plain GPU": [x * SHARE_PLAIN[0], x * SHARE_PLAIN[1]],
                "share of the node, --zk": x * SHARE_ZK}
            print(kind, how, {k: f"{v:.3g}" for k, v in cost.items()}, f"= {x:.3g} × ODBRANCH's 1.0e12")
    if a.json:
        a.json.write_text(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
