"""The sharded plan unit (`PlanShard_v1`, `verity.ml.circuits.requirement_units`) on a recorded remote session: each pass's
plan cut into the store's shards, one instance a shard a pass, every instance held to the unit's reference, each distinct one
to the circuit, and one staged by Flock's `UnitStager` for its fit in M0's block.

    uv run python benchmarks/one_stage/plan_shards.py --session remote_session.json --record verifier.json --out DIR \\
        [--S 32768] [--gaps 0] [--no-circuit] [--no-stage]

* **The plan.** The verifier's record (``verifier.json``, its session's record): each window's passes, their steps, plan
  lengths and stale bounds, and the store's blocks. v0's record keeps no plan, so a pass's plan is read from its public data
  only where that fixes it: nothing appended or released and a plan of every block below the stale bound, N of N, is every
  block in store order. Any other pass is refused here.
* **The blocks each step reads.** The session's registered layout (``remote_session.json``'s ``layout``, one region a
  registration, weights or filler; its registration digest the record's): every weight block, a gathered table (the
  embedding) counted whole. That is each instance's ``regmask`` at every step, a superset of the step's registered reads,
  which is what the one-sided clause needs; the circuit's other reads (its KV) are outside this store.
* **What it writes.** ``$OUT/plan_shards.json``: the binding, the shards (full, weight, mixed, filler), instances a pass and in
  all, each one's reference verdict, the distinct instances and the circuit's verdict on each, ANDs an instance, a pass and
  in all, the rows' bits, and the staged instance's ``k_log``, ranges and ok.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from bisect import bisect_left
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from verity.ml.circuits import requirement_units as RU
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate_batch


def passes(record: dict) -> tuple[int, list[dict]]:
    """The store's blocks and each pass's public data, refusing a pass whose plan the record doesn't fix."""
    pub = record["record"]["public"]
    n = pub["blocks"]
    if pub["appended"] or pub["released"]:
        raise SystemExit(f"the session appended {pub['appended']} and released {pub['released']} blocks: its plans are on the "
                         "wire, not fixed by the record")
    out = []
    for w in record["record"]["profile"]["windows"]:
        for i, (step, length, stale) in enumerate(zip(w["steps"], w["lengths"], w["stale"], strict=True)):
            if length != n or stale != n:
                raise SystemExit(f"window {w['window']} pass {i}: a plan of {length} below {stale} of {n} blocks isn't fixed by "
                                 "the record")
            out.append({"window": w["window"], "pass": i, "step": step, "length": length})
    return n, out


def weights(session: dict, n: int, registration: str) -> list[int]:
    """The registered layout's weight blocks, sorted, from regions that tile the store."""
    if session["memory_map"]["registration"] != registration or session["store"]["blocks"] != n:
        raise SystemExit("the session's registration or store isn't the record's")
    at, out = 0, []
    for r in session["layout"]:
        if r["first"] != at or r["kind"] not in ("weights", "filler"):
            raise SystemExit(f"region {r}: the layout's regions tile the store from 0, each weights or filler")
        if r["kind"] == "weights":
            out.extend(range(r["first"], r["first"] + r["blocks"]))
        at += r["blocks"]
    if at != n:
        raise SystemExit(f"the layout covers {at} of {n} blocks")
    return out


def _bits(words, shape) -> np.ndarray:
    w = np.asarray(words, dtype=np.uint64).reshape(shape[:-1])
    return ((w[..., None] >> np.arange(16, dtype=np.uint64)) & 1).astype(np.uint8)


def circuit_oks(defn, batch: list[dict]) -> list[int]:
    """The circuit's ok bit on each instance, bit-sliced, with bits 1 .. 15 checked zero."""
    params, _ret = defn.signature()
    arrays = [np.stack([_bits(a[name], (*np.shape(a[name]), 16) if np.ndim(a[name]) else (16,)) for a in batch])
              for name, _t in params]
    out = np.asarray(evaluate_batch(defn, *arrays)).reshape(len(batch), 16)
    if out[:, 1:].any():
        raise SystemExit("an ok word's bits 1 .. 15 aren't zero")
    return [int(x) for x in out[:, 0]]


def stage(defn, S, a: dict, scratch: Path) -> dict:
    """One instance staged: the statement's META (its block ``k_log``, at most 2^27, and a row compressing before its last
    block, both `compose`'s refusals) and its ok."""
    from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2
    from verity_flock import class_statement as CS

    t0 = time.time()
    rows = RU.plan_shard_rows(S)
    st = CS.UnitStager(defn, "unit/plan-shard", [(r, V2, b, ps) for r, b, ps in rows], [(V2, (0,))], circuit=scratch / "circuit.txt")
    composed = time.time() - t0
    words = RU.row_words(S, rows, a)
    got = st.stage({r: np.asarray([words[r]], dtype=np.uint16) for r, _b, _ps in rows}, scratch / "round",
                   hashlib.sha256(b"verity/one-stage/plan-shards/salts").digest())
    return {"k_log": st.meta["k_log"], "unit_log": st.meta["unit_log"], "vus_per_block": st.meta["vus_per_block"],
            "ranges": st.meta["ranges"], "rows": [[r, b] for r, b, _ps in rows], "ands": got["ands"],
            "ok": int(np.asarray(got["outs"]["o0"])[0, 0]), "pin": st.pin,
            "seconds": {"compose": round(composed, 1), "stage": round(time.time() - t0 - composed, 1)}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", type=Path, required=True, help="vllm's remote_session.json")
    ap.add_argument("--record", type=Path, required=True, help="the remote verifier's verifier.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--S", type=int, default=1 << 15, help="a shard's blocks")
    ap.add_argument("--gaps", type=int, default=0, help="the most blocks a shard's plan leaves out")
    ap.add_argument("--no-circuit", action="store_true", help="the references only")
    ap.add_argument("--no-stage", action="store_true", help="no UnitStager")
    a = ap.parse_args(argv)
    t0 = time.time()
    verifier = json.loads(a.record.read_text())
    if len(verifier["sessions"]) != 1:
        raise SystemExit(f"{len(verifier['sessions'])} sessions in the record, not one")
    rec = verifier["sessions"][0]
    n, ps = passes(rec)
    need = weights(json.loads(a.session.read_text()), n, rec["record"]["public"]["registration"])
    S = SimpleNamespace(S=a.S, GAPS=a.gaps)
    defn = bind(RU.PlanShard, S=a.S, GAPS=a.gaps)
    cut = RU.plan_shards(n, a.S)
    rows = RU.plan_shard_rows(S)
    distinct: dict[str, dict] = {}
    refused = []
    for p in ps:
        plan = range(p["length"])
        for k, (lo, size) in enumerate(cut):
            x = RU.shard_inputs(S, lo, size, plan, need)
            if not RU.plan_shard(S, **x):
                refused.append([p["window"], p["pass"], k])
            key = hashlib.sha256(json.dumps(RU.row_words(S, rows, x)).encode()).hexdigest()
            d = distinct.setdefault(key, {"args": x, "shards": set(), "passes": 0})
            d["shards"].add(k)
            d["passes"] += 1
    named = [bisect_left(need, lo + size) - bisect_left(need, lo) for lo, size in cut]
    shards = {"n": len(cut), "full": sum(size == a.S for _lo, size in cut), "last": cut[-1][1],
              "weight": sum(c == size for c, (_lo, size) in zip(named, cut)),
              "mixed": sum(0 < c < size for c, (_lo, size) in zip(named, cut)), "filler": sum(c == 0 for c in named)}
    ands = RU.PlanShard.traced(a.S, a.gaps).counts["and"]
    out = {"binding": defn.id, "store_blocks": n, "weight_blocks": len(need), "passes": ps, "steps": sorted({p["step"] for p in ps}),
           "shards": shards, "instances_a_pass": len(cut), "instances": len(cut) * len(ps), "reference_refused": refused,
           "rows_bits": [[r, b] for r, b, _ps in rows], "ands_an_instance": ands, "ands_a_pass": ands * len(cut),
           "ands": ands * len(cut) * len(ps), "seconds": {"references": round(time.time() - t0, 1)}}
    print(f"plan-shards: {n} blocks, {len(need)} weight, {len(ps)} passes, {len(cut)} shards ({shards}), "
          f"{len(refused)} refused by the reference, {len(distinct)} distinct instances, {ands} ANDs an instance", flush=True)
    view = [{"key": k, "shards": [min(d["shards"]), max(d["shards"])], "n_shards": len(d["shards"]), "instances": d["passes"],
             "count": d["args"]["count"], "size": d["args"]["size"],
             "named": sum(bin(w).count("1") for w in d["args"]["regmask"])} for k, d in distinct.items()]
    if not a.no_circuit:
        t1 = time.time()
        oks = circuit_oks(defn, [d["args"] for d in distinct.values()])
        for v, ok in zip(view, oks):
            v["circuit_ok"] = ok
        out["seconds"]["circuit"] = round(time.time() - t1, 1)
        print(f"plan-shards: the circuit's oks on the distinct instances {oks}, {out['seconds']['circuit']} s", flush=True)
    out["distinct"] = view
    if not a.no_stage:
        mixed = [d for d, v in zip(distinct.values(), view) if 0 < v["named"] < v["size"]]
        pick = (mixed or list(distinct.values()))[0]
        with tempfile.TemporaryDirectory() as d:
            out["staged"] = {"shards": sorted(pick["shards"])[:4], **stage(defn, S, pick["args"], Path(d))}
        print(f"plan-shards: staged {json.dumps(out['staged'])[:600]}", flush=True)
    out["seconds"]["total"] = round(time.time() - t0, 1)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "plan_shards.json").write_text(json.dumps(out, indent=1, sort_keys=True))
    bad = refused or any(v.get("circuit_ok") == 0 for v in view) or out.get("staged", {}).get("ok") == 0
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
