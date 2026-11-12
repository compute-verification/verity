"""CacheStep_v3's cost at the GLM-5.3 class: the RAM pool's operations in the busiest window (P = 4,964 consecutive
steps, `measure`) of synthetic loads at the class's engine (NB = 5,000, B = 64, R = 256, T = 8,192), priced at the
measured parts (`measured`: `cache_step_v3`'s PoolSlot_v1, PoolEntry_v1 and PoolEnd at the class's widths, and
`block_key`'s BlockKey_v1 and KeyPrint_v1) plus the in-circuit hash of their witness.

    python3 cost.py [--json OUT] [--quick] [--from REPORT] [--served NAME=RAW@CONFIG@STEPS]... [--only]

A served run is a load too (``--served``, `served_load`): the hook's record of it (RAW, the trace's ``raw/`` directory:
`requests.PID.jsonl.gz` and `steps.PID.jsonl.gz`) replayed under the scheduler's configuration (CONFIG, a JSON object as
a step trace's header records it, `replay.Engine.from_header`), its busiest STEPS consecutive steps the window. ``--only``
prices the served loads alone.

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

The proved side (`unit_cost`) is the two units every window proves, `compile_slice_v6`'s CompileSlice_v6 and
PoolWindow_v1, at the class of the load's busiest window (`compile_statics`): each part's ANDs at the class times its
calls (`compile_slice_v6.parts`, which its tests hold to the built units'), and SHA-512 over the pool window's checked
witness at the same rate. The class: the engine's R, W = 1 and A = R + 1, K (and Q, D and DA) the ladder's class of the
window's requests, P its steps, CU = L / B, and NS, NM, NRB and NRK the ladder's classes of its stream slots (as
`window_cost` counts them for each kind of lookup), its M addresses and the keys the pool carries in (NB − 1 at most),
its RB addresses, and its ragged row's blocks (each request's blocks to the last its keys are read at). The pool window
opens the pool the previous window left and commits the one it leaves, so its parts include the two pool rows' digests,
their keys' prints and a PoolOpen and a PoolClose step a block.

What a window's ANDs are compared with is the caller's to state, with its source: ``--against NAME=ANDS`` reports the
window's ANDs as a ratio of ANDS, and ``--node-share NAME=SHARE@ANDS`` scales a share of the node that a window of ANDS
ANDs takes under a proving setting NAME to each window. Both are recorded in the report as given; neither has a default.
"""
from __future__ import annotations

import argparse
import dataclasses
import gzip
import json
import random
import resource
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pool_ram as PR
import stream as ST
from pool_ram import RP

from verity.protocols.certifier.verifier.head_check import ands
from verity.circuits.defs import REGISTRY, bind
from verity.definitions.commitments.gates import block_key as BK
from verity.protocols.certifier.prover import cache_step_v3 as V3, compile_slice_v6 as V6

MUL = 2187
SHA_BLOCK = 58120
SHA_BIT = SHA_BLOCK / 1024
S = PR.GLM
P_STEPS = 4964
#: the compile side's request ids and token widths: a session's ids, and `glm_load`'s tokens (below 150,000)
IW = 20
CTW = 18

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


def served_load(raw: Path, config: dict[str, Any]) -> tuple[RP.Engine, list[RP.RequestIO], list[int], dict[str, int]]:
    """A served run from its hook's record under the scheduler's configuration `config`: its requests in the order the
    engine first scheduled them, each visible from that step (FCFS schedules new requests in arrival order), its prompt
    and outputs its tokens, and only steps that scheduled a token counted, as `replay.read_corpus_trace` takes them; and
    the served steps and tokens, which the replay of it should give again."""
    def rows(kind: str) -> list[dict[str, Any]]:
        [p] = raw.glob(f"{kind}.*.jsonl.gz")
        with gzip.open(p, "rt") as f:
            return [json.loads(line) for line in f]

    reqs = {r["id"]: r for r in rows("requests")}
    steps = [s for s in rows("steps") if s["total_num_scheduled_tokens"]]
    first: dict[str, int] = {}
    for k, s in enumerate(steps):
        for r in s["new"]:
            if not r.get("resumed"):
                first.setdefault(r["id"], k)
    eng = RP.Engine.from_header(config)
    out = []
    for rid in first:
        r = reqs[rid]
        p, o = r["prompt_ids"], r["output_ids"]
        if r["finish_reason"] not in ("stop", "length"):
            raise ValueError(f"request {rid}: finish {r['finish_reason']!r}; the replay models stop and length only")
        if max(p + o) >> CTW:
            raise ValueError(f"request {rid}: a token past CTW = {CTW} bits")
        max_tokens = r.get("max_tokens")
        out.append(RP.RequestIO(len(p), eng.max_model_len - len(p) if max_tokens is None else int(max_tokens), len(o),
                                r["finish_reason"] == "stop", tuple(p) + tuple(o)))
    served = {"steps": len(steps), "tokens": sum(s["total_num_scheduled_tokens"] for s in steps), "requests": len(out)}
    return eng, out, list(first.values()), served


def accesses(c: Counter) -> int:
    return sum(PR.SHAPE[k] * n for k, n in c.items() if k in PR.SHAPE)


def busiest(per: list[int], steps: int = P_STEPS) -> tuple[int, int, int]:
    """The `steps` consecutive steps (all, if fewer) with the largest sum: (sum, first step, steps)."""
    n = min(steps, len(per))
    run = sum(per[:n])
    best, s0 = run, 0
    for s in range(1, len(per) - n + 1):
        run += per[s + n - 1] - per[s - 1]
        if run > best:
            best, s0 = run, s
    return best, s0, n


def stream_window(eng: RP.Engine, reqs: list[RP.RequestIO], admit: list[int], steps: int = P_STEPS) -> dict[str, Any]:
    """The load's stream with the head-of-line memo: `stream.py`'s `Stream` beside the replay's pool (each call's results
    held to the pool's, the pools compared at the end), in windows of 2^16 slots (each starts with no memo kept), and the
    `steps` consecutive steps with the most slots."""
    m = ST.run(eng, reqs, {"decide": lambda i, k, o: admit[i] <= k}, 1 << 16, check=False, every=0)
    best, s0, n = busiest([c["slots"] for c in m.steps], steps)
    c = m.pool.counts
    return {"slots": best, "steps": [s0, s0 + n], "hits": c["hit"], "memo hits": c["memo hits"],
            "re-walks": c["re-walks"], "all slots": m.pool.total}


def measure(eng: RP.Engine, reqs: list[RP.RequestIO], admit: list[int], steps: int = P_STEPS) -> dict[str, Any]:
    """The load's busiest window: the `steps` consecutive steps with the most accesses (the whole replay if shorter),
    found on a first replay; a second, identical one collects the sparse addresses that window touches. Its requests are
    those that arrive before it ends and run at or past its start (step s of the count is the replay's step s − 1), and
    its ragged blocks each touched request's blocks up to the last whose key it reads."""
    how = {"every": 10**9, "keep": False, "decide": lambda i, k, o: admit[i] <= k}
    m = PR.run(eng, reqs, **how)
    per = [accesses(c) for c in m.steps]
    best, s0, n = busiest(per, steps)
    window = range(s0, s0 + n)
    w = PR.run(eng, reqs, window=window, **how)
    assert [accesses(c) for c in w.steps] == per, "the replay is not deterministic"
    tot = sum((w.steps[s] for s in window), Counter())
    acc = sorted(per[1:])
    last: dict[int, int] = {}
    for rid, k in w.touched["RK"]:
        last[rid] = max(last.get(rid, 0), k + 1)
    present = sum(1 for i, a in enumerate(admit) if a < s0 + n and w.last_step.get(i, len(per)) >= s0 - 1)
    return {"NB": eng.num_gpu_blocks, "steps": m.counts["steps"], "requests": len(reqs), "tokens": m.counts["tokens"],
            "replay": {"ops": {k: m.pool.ops[k] for k in (*PR.SHAPE, "evict")}, **{k: m.counts[k] for k in (
                "calls", "lookups", "allocations", "refused allocations", "refused after a lookup", "frees",
                "certified refusals", "lookup units a certificate saves", "repeated walks",
                "lookup units in repeated walks")}},
            "window": {"steps": [s0, s0 + n], "accesses": best, "ops": {k: tot[k] for k in (*PR.SHAPE, "evict")},
                       "calls": tot["calls"], "keys": tot["keys"], "repeated units": tot["repeated units"],
                       "certificate saves": tot["certificate saves"],
                       "touched": {k: len(v) for k, v in w.touched.items()},
                       "requests": present, "ragged blocks": sum(last.values())},
            "accesses per step": {"mean": sum(acc) / max(len(acc), 1),
                                  "p99": acc[int(0.99 * (len(acc) - 1))] if acc else 0,
                                  "p99.9": acc[int(0.999 * (len(acc) - 1))] if acc else 0,
                                  "max": acc[-1] if acc else 0}}


def stream_slots(load: dict[str, Any], lookups: str) -> int:
    """U, the stream slots of the load's busiest window under `lookups` (`window_cost`)."""
    w = load["window"]
    ops = {k: w["ops"][k] for k in (*PR.SHAPE, "evict")}
    ops["lookup"] -= {"walked": 0, "certified": w["certificate saves"], "memo": w["repeated units"]}[lookups]
    if lookups == "memo" and "stream" in load:
        return load["stream"]["slots"]
    return sum(STREAM[k] * n for k, n in ops.items())


def window_cost(load: dict[str, Any], m: dict[str, int], NB: int = 5000, lookups: str = "walked") -> dict[str, float]:
    """ANDs for the load's busiest window: U and NE and their classes, the stream, the touched list, and the per-call,
    per-key and per-window parts. `lookups`: "walked" (every lookup walks, as vLLM does), "certified" (a refused
    head-of-line candidate's walk replaced by the one-unit refusal certificate where it suffices,
    `pool_ram.Mirror.certify`), or "memo": CacheStep_v3's own stream with its head-of-line memo, U its busiest window's
    slots (`stream_window`), or for a load measured without it, the bound "memo" had before the memo was built (a walk
    whose (hit, evictable) equals the request's last one is reused, its stamps and dirty bit not counted). The touched
    list, the calls and the keys are `measure`'s window's in every case."""
    w = load["window"]
    u = stream_slots(load, lookups)
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


def compile_statics(load: dict[str, Any], engine: dict[str, Any], lookups: str) -> dict[str, Any]:
    """`compile_slice_v6`'s statics for the load's busiest window under `lookups` (the module's docstring)."""
    w, c = load["window"], V3.size_class
    R, steps = engine["max_num_seqs"], w["steps"][1] - w["steps"][0]
    K = c(w["requests"])
    return V6.statics(R, 1, R + 1, engine, IW, Q=K, D=K, P=steps, K=K, CTW=CTW,
                      CU=engine["max_model_len"] // engine["block_size"], DA=K, NS=c(stream_slots(load, lookups)),
                      NM=c(w["touched"]["M"] + engine["num_gpu_blocks"] - 1), NRB=c(w["touched"]["RB"]),
                      NRK=c(w["ragged blocks"]))


#: `compile_slice_v6.parts`' order for each unit, each part's term
UNIT_TERMS = {
    "compile": ("cut (CompileCut)", "load", "steps (CompileStep)", "edge (CompileEdge)", "arrivals' ids (CompileIds)",
                "calls row (CallsEq)", "verdict (CallsVerdict)"),
    "pool": ("bus records (CompileBus)", "arrivals' counts (PoolCounts)", "pool row digests in and out",
             "key prints in and out", "stream slots (PoolSlot)", "open steps (PoolOpen)", "touched list (PoolEntry)",
             "ragged keys (RaggedKey)", "pool end", "close steps (PoolClose)", "M run copies (PoolCopies)",
             "verdict (PoolVerdict)")}


def _drop_traces() -> None:
    """Drops every part's trace and every bound Definition's body: the step's at R = 256 and a pool row digest's at NB
    blocks each hold gigabytes."""
    for d in list(REGISTRY.defs.values()):
        if hasattr(getattr(d, "traced", None), "cache_clear"):
            d.traced.cache_clear()
        for sp in getattr(d, "_cache", {}).values():
            sp.set_body(None)


def unit_cost(st: dict[str, Any], memo: dict, rss: dict[str, float] | None = None) -> dict[str, dict[str, float]]:
    """ANDs of CompileSlice_v6 and PoolWindow_v1 at statics `st`: each part's ANDs times its calls, and for the pool window
    SHA-512 over its checked witness; `memo` holds each part's ANDs across calls (`head_check.ands`), so each part's
    trace is dropped once it is counted. `rss`, if given, gets the process's peak resident GiB after each part."""
    out: dict[str, dict[str, float]] = {}
    for unit, ps in V6.parts(**st).items():
        out[unit] = {}
        for term, (p, n) in zip(UNIT_TERMS[unit], ps, strict=True):
            out[unit][term] = n * ands(p, memo)
            _drop_traces()
            if rss is not None:
                rss[f"{unit}: {term}"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    bits = V6.data_bits(SimpleNamespace(**st))
    out["pool"]["checked witness hash"] = SHA_BIT * (bits["chk"] + bits["src"])
    for unit in ("compile", "pool"):
        out[unit]["total"] = sum(out[unit].values())
    return out


def _stated(items: list[str], share: bool) -> dict[str, Any]:
    """``NAME=ANDS`` (or, with `share`, ``NAME=SHARE@ANDS``) arguments as {NAME: ANDS} ({NAME: [SHARE, ANDS]})."""
    out: dict[str, Any] = {}
    for item in items:
        name, _, v = item.partition("=")
        if not name or not v or (share and "@" not in v):
            raise SystemExit(f"{item!r} is not NAME={'SHARE@ANDS' if share else 'ANDS'}")
        out[name] = [float(x) for x in v.split("@")] if share else float(v)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", type=Path)
    ap.add_argument("--quick", action="store_true", help="one small load at R = 16 (a smoke test)")
    ap.add_argument("--from", dest="src", type=Path, help="reprice a report's measured loads instead of replaying them")
    ap.add_argument("--no-compile", action="store_true", help="price the pool alone, not the proved units")
    ap.add_argument("--against", action="append", default=[], metavar="NAME=ANDS",
                    help="report each window's ANDs as a ratio of ANDS, a budget the caller states with its source")
    ap.add_argument("--node-share", action="append", default=[], metavar="NAME=SHARE@ANDS",
                    help="the share of the node a window of ANDS ANDs takes under proving setting NAME, scaled to each window")
    ap.add_argument("--served", action="append", default=[], metavar="NAME=RAW@CONFIG@STEPS",
                    help="a served run's hook record RAW replayed under the scheduler configuration CONFIG (a JSON file), "
                         "its busiest STEPS consecutive steps the window")
    ap.add_argument("--only", action="store_true", help="price the --served loads alone")
    ap.add_argument("--lookups", nargs="+", default=["walked", "certified", "memo"], choices=("walked", "certified", "memo"))
    a = ap.parse_args()
    against, shares = _stated(a.against, False), _stated(a.node_share, True)

    def compared(total: float) -> dict[str, Any]:
        return {"vs": {name: total / x for name, x in against.items()},
                "node share": {name: s * total / x for name, (s, x) in shares.items()}}

    m = measured()
    slot = m["slot"] + SHA_BIT * SLOT_WITNESS
    print("measured ANDs:", m, f"a stream slot with its witness hash: {slot:.0f}")
    report: dict[str, Any] = {"measured": m, "slot": slot, "parameters": {"against": against, "node share": shares},
                              "loads": {}}
    if a.src:
        loads = {k: (v["load"], v.get("engine")) for k, v in json.loads(a.src.read_text())["loads"].items()}
    else:
        loads = {}
        for kind, seed in [] if a.only else [("chat", 1), ("pressure", 2)] if not a.quick else [("chat", 1)]:
            eng, reqs, admit = glm_load(seed, kind=kind, **({"steps": 300, "NB": 800, "R": 16} if a.quick else {}))
            loads[kind] = ({**measure(eng, reqs, admit), "stream": stream_window(eng, reqs, admit)},
                           dataclasses.asdict(eng))
    for item in a.served:
        name, _, spec = item.partition("=")
        raw, config, steps = spec.split("@")
        eng, reqs, admit, served = served_load(Path(raw), json.loads(Path(config).read_text()))
        loads[name] = ({**measure(eng, reqs, admit, int(steps)), "stream": stream_window(eng, reqs, admit, int(steps)),
                        "served": served}, dataclasses.asdict(eng))
    memo: dict = {}
    for kind, (load, engine) in loads.items():
        NB = load.get("NB", 5000)
        print(kind, json.dumps(load))
        report["loads"][kind] = {"load": load, "engine": engine}
        mk = m if NB == 5000 else measured(NB)
        _drop_traces()
        for how in a.lookups:
            cost = window_cost(load, mk, NB=NB, lookups=how)
            report["loads"][kind][how] = {"window ANDs": cost, **compared(cost["total"])}
            print(kind, how, {k: f"{v:.3g}" for k, v in cost.items()}, report["loads"][kind][how]["vs"],
                  report["loads"][kind][how]["node share"])
            if a.no_compile or engine is None or "requests" not in load["window"]:
                continue
            st = compile_statics(load, engine, how)
            rss: dict[str, float] = {}
            uc = unit_cost(st, memo, rss)
            total = uc["compile"]["total"] + uc["pool"]["total"]
            report["loads"][kind][how]["proved units"] = {
                "statics": {k: st[k] for k in ("R", "W", "A", "K", "P", "CU", "NB", "NS", "NM", "NRB", "NRK")},
                "CompileSlice_v6": uc["compile"], "PoolWindow_v1": uc["pool"], "total": total, **compared(total),
                "peak GiB after each part": rss}
            print(kind, how, "peak GiB after each part", {k: round(v, 1) for k, v in rss.items()}, flush=True)
            for name, c in (("CompileSlice_v6", uc["compile"]), ("PoolWindow_v1", uc["pool"])):
                print(kind, how, name, {k: f"{v:.3g}" for k, v in c.items()}, flush=True)
            print(kind, how, "both", f"{total:.3g}", report["loads"][kind][how]["proved units"]["node share"], flush=True)
    if a.json:
        a.json.write_text(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
