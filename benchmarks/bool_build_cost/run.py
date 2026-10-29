"""The Boolean Build's cost against the word Build's on one workload's specializations: the descriptor of their union and the
`Q_word_v1{16,32}` proof-block rules the manifest computes for them (`verity.ml.definitions.word`'s prefetch, then its check, one rule per
distinct Definition), timed per Definition.

    uv run benchmarks/bool_build_cost/run.py [--workload qwen3-06b|small] [--ir word,boolean] [--part descriptor,rules]
                                             [--jobs N] [--t T,T,...] --out DIR

A workload is its Calls' word specializations (`WORKLOADS`); the Boolean ones are `verity.ml.definitions.boolean.boolean_version` of each, as the Boolean
Build's lift calls them.  `--t` keeps the attention specializations of those key lengths only.  Writes `DIR/<ir>.json` (descriptor
bytes, Program digest, the largest Definitions, rule seconds and the verdict per specialization, a digest of every rule) and
`DIR/rules-<ir>.jsonl` (one line per rule as it finishes, so a run cut short keeps what it measured).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

DOT = '{"fn":"HopperBF16WgmmaDot16_v1"}'
EPS = '{"f64":"0x1.0c6f7a0b5ed8dp-20"}'


def _attention(T: int, mf: int) -> str:
    return f'Attention_v5{{T={T},NH=16,KVH=8,D=128,BN=64,DOT={DOT},INV={{"fn":"Fa2InvSum_v1"}},MASKED_FROM={mf}}}'


def qwen3_06b() -> list[str]:
    """Qwen3-0.6B, B1, a 256-token prompt and 32 tokens out on FA2: prefill row r sees r + 1 keys,
    decode step s sees 256 + s; MASKED_FROM is FA2's tile rule (`attention_vocab.fa2_masked_from`, BN 64)."""
    from verity.ml.definitions.registry.attention_vocab import fa2_masked_from
    att = [_attention(r + 1, fa2_masked_from(256, 256, r, 64, True)) for r in range(256)]
    att += [_attention(T, fa2_masked_from(T, 1, 0, 64, False)) for T in range(257, 288)]
    other = [f"Gemm_v2{{K={k},N={n},DOT={DOT}}}" for k, n in ((1024, 4096), (2048, 1024), (1024, 6144), (3072, 1024), (1024, 151936))]
    other += [f"RMSNormFusedCuda_v2{{N=1024,EPS={EPS}}}", f"RMSNormTriton_v1{{N=128,EPS={EPS}}}", f"RMSNormTriton_v1{{N=1024,EPS={EPS}}}",
              "RoPE_v1{NHEADS=8,D=128}", "RoPE_v1{NHEADS=16,D=128}", "SiluMul_v2{I=3072}", "TokenSelect_v1{V=151936}"]
    return att + other


def small() -> list[str]:
    """Two padded attention rows (one tile, and a full tile plus one key), RoPE and SiLU-mul at Qwen3-0.6B's statics."""
    return [_attention(1, 0), _attention(65, 0), "RoPE_v1{NHEADS=8,D=128}", "SiluMul_v2{I=3072}"]


WORKLOADS = {"qwen3-06b": qwen3_06b, "small": small}


def definitions(specs: list[str], ir: str) -> list:
    from verity.ml.circuits.defs import CompositeDefinition, bind
    from verity.ml.definitions import boolean as BB
    from verity.ml.definitions.call_scope import specialization
    from verity_vllm.query.program_view import spec_base, spec_statics
    out = []
    for s in specs:
        fn = specialization(spec_base(s), spec_statics(s))
        if ir == "boolean":
            fn = BB.boolean_version(fn)
            fn = bind(fn) if isinstance(fn, CompositeDefinition) else fn
        out.append(fn)
    return out


def program(fns: list):
    """One Program calling each Definition once, on parameters of its own."""
    from verity.ml.circuits.defs import CompositeDefinition, bind
    from verity.ml.circuits.program import Program
    from verity.ml.circuits.refs import tuple_of
    from verity.ml.circuits.types import Tuple
    sigs = [fn.signature() for fn in fns]
    params = tuple((f"a{i}_{n}", t) for i, (ps, _r) in enumerate(sigs) for n, t in ps)

    def body(B, S, *args):
        outs, k = [], 0
        for fn, (ps, _r) in zip(fns, sigs):
            outs.append(B.call(fn, *args[k:k + len(ps)]))
            k += len(ps)
        return tuple_of(*outs)

    return Program(bind(CompositeDefinition("BoolBuildCost", 1, (), lambda S: (params, Tuple(*[r for _p, r in sigs])), body,
                                            register=False)))


def descriptor(fns: list) -> dict:
    from verity.ml.circuits.codec import canonical_json, descriptor_size, encode_program, program_digest
    t = time.time()
    d = encode_program(program(fns))
    secs = time.time() - t
    per = sorted(((len(canonical_json(v)), k) for k, v in d["definitions"].items()), reverse=True)
    fam: dict[str, int] = {}
    for n, k in per:
        fam[k.split("{")[0]] = fam.get(k.split("{")[0], 0) + n
    return {"bytes": descriptor_size(d), "digest": program_digest(d), "definitions": len(per), "encode_secs": round(secs, 2),
            "largest": [[n, k[:160]] for n, k in per[:12]], "by_family": dict(sorted(fam.items(), key=lambda kv: -kv[1])[:16])}


_PENDING: list = []


def _rule(i: int):
    from verity.ml.definitions import word as WQ
    fn, limits = _PENDING[i]
    t = time.time()
    try:
        r = WQ.unit_rule(fn, memo=WQ._RULES[tuple(sorted(limits.items()))], max_gates=limits)
    except Exception as e:  # noqa: BLE001
        r = repr(e)
    return r, time.time() - t, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


def verdict(r: dict) -> dict:
    return {"gates": r["gates"], "units": r["units_per_call"], "by_kind": r["by_kind"], "interior_words": r["committed_interior_words"],
            "violations": sorted({f'{v["class"]}:{",".join(v.get("codes", []))}' for v in r["violations"]}),
            "digest": hashlib.sha256(json.dumps(r, sort_keys=True, default=str).encode()).hexdigest()[:16]}


def rules(fns: list, jobs: int, log: Path) -> dict:
    """`word.prefetch`'s schedule (bodies cut as a whole first, then each separable body once its members are memoed), each rule
    timed in its worker, then `word.check_calls`'s lookups."""
    from verity.ml.definitions import word as WQ
    limits = WQ.gate_limits(None)
    memo = WQ._RULES.setdefault(tuple(sorted(limits.items())), {})
    depth: dict[str, int] = {}
    by_id: dict = {}
    for fn in fns:
        WQ._depth(fn, limits, depth, by_id)
    t0 = time.time()
    per: dict[str, dict] = {}
    with log.open("w") as f:
        for d in range(max(depth.values(), default=-1) + 1):
            todo = sorted((by_id[i] for i, k in depth.items() if k == d), key=lambda fn: -fn.gates)
            _PENDING[:] = [(fn, limits) for fn in todo]
            with ProcessPoolExecutor(max(1, min(jobs, len(todo))), mp_context=multiprocessing.get_context("fork")) as ex:
                for fn, (r, secs, rss) in zip(todo, ex.map(_rule, range(len(todo)))):
                    if isinstance(r, dict):
                        memo[(fn.id, 16, 32, 0)] = r
                    per[fn.id] = {"depth": d, "gates": fn.gates, "secs": round(secs, 3), "rss_mb": rss,
                                  **(verdict(r) if isinstance(r, dict) else {"error": r})}
                    f.write(json.dumps({"id": fn.id, **per[fn.id]}) + "\n")
                    f.flush()
            _PENDING.clear()
    prefetch = time.time() - t0
    t1 = time.time()
    top = {fn.id: WQ.unit_rule(fn, memo=memo, max_gates=limits) for fn in fns}
    check = time.time() - t1
    return {"jobs": jobs, "prefetch_secs": round(prefetch, 2), "check_secs": round(check, 2), "rules": len(per),
            "rules_digest": hashlib.sha256(json.dumps({k: v.get("digest") for k, v in sorted(per.items())}).encode()).hexdigest()[:16],
            "verdicts": {k: verdict(r) for k, r in top.items()},
            "slowest": sorted(([v["secs"], k[:120], v["gates"]] for k, v in per.items()), reverse=True)[:20]}


def main(argv: list[str] | None = None) -> int:
    a = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    a.add_argument("--workload", default="small", choices=sorted(WORKLOADS))
    a.add_argument("--ir", default="word,boolean")
    a.add_argument("--part", default="descriptor,rules")
    a.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    a.add_argument("--t", default="", help="keep the attention specializations of these key lengths only")
    a.add_argument("--out", required=True)
    o = a.parse_args(argv)
    out = Path(o.out)
    out.mkdir(parents=True, exist_ok=True)
    specs = WORKLOADS[o.workload]()
    if o.t:
        keep = {f"T={t}," for t in o.t.split(",")}
        specs = [s for s in specs if not s.startswith("Attention_") or any(k in s for k in keep)]
    for ir in o.ir.split(","):
        t = time.time()
        fns = definitions(specs, ir)
        res: dict = {"workload": o.workload, "ir": ir, "specs": len(specs), "build_secs": round(time.time() - t, 2)}
        if "descriptor" in o.part:
            res["descriptor"] = descriptor(fns)
            print(f"[{ir}] descriptor {res['descriptor']['bytes']:,} B digest {res['descriptor']['digest'][:16]} "
                  f"({res['descriptor']['encode_secs']} s)", flush=True)
        if "rules" in o.part:
            res["rules"] = rules(fns, o.jobs, out / f"rules-{ir}.jsonl")
            print(f"[{ir}] rules: prefetch {res['rules']['prefetch_secs']} s on {o.jobs} workers, check {res['rules']['check_secs']} s, "
                  f"{res['rules']['rules']} rules, digest {res['rules']['rules_digest']}", flush=True)
        res["max_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
        (out / f"{ir}.json").write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
