"""Where each served type's proving time goes, from prove_split_cpu.sh's output: hashing (the SHA-512 of the block's
committed inputs and outputs: its sha512x3 and hm96 slots and the rows' native chaining) and arithmetic (the units, the
stages and the lookups), beside the ZK extras and the session.

    python benchmarks/one_stage/prove_split.py OUT > split.md     (also writes OUT/split.json)

The witness's laps are measured by circuit (CONVERGE, WITNESSCPU); a phase over the whole witness (commit, zerocheck,
lincheck, opening: CPUPHASE) is measured whole and split by each role's share of the statement's cells (LAYOUT), which is
how those phases scale. Every second is a timed session's (the warm one dropped), the median over them.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

HASH, ARITH = ("sha512x3", "hm96"), ("unit", "stage", "lookup")
PIOP = ("t.commit", "t.hook_bind", "t.zerocheck", "t.lincheck", "t.open")


def lines(path: Path, tag: str) -> list[dict]:
    return [json.loads(x.split("\t", 1)[1]) for x in path.read_text().splitlines() if x.startswith(tag + "\t")]


def cells(layout: dict) -> dict:
    """Cells a statement by role (allocated and used), and the unused rest of 2^m."""
    lay, blocks = layout["layout"], layout["blocks"]
    alloc, used = {}, {}
    for r in lay["ranges"]:
        alloc[r["role"]] = alloc.get(r["role"], 0) + (blocks * r["count"] << r["slot_log"])
        used[r["role"]] = used.get(r["role"], 0) + blocks * r["count"] * min(r["useful"], 1 << r["slot_log"])
    alloc["unused"] = (1 << layout["m"]) - sum(alloc.values())
    return {"alloc": alloc, "used": used, "roles": {r["net"]: r["role"] for r in lay["ranges"]}}


def session_split(conv: dict, wit: dict, phases: dict, live: dict, cl: dict, setup: dict | None = None) -> dict:
    total = sum(cl["alloc"].values())
    share = {k: v / total for k, v in cl["alloc"].items()}
    w_hash = conv.get("row_inputs", 0.0)
    w_arith = conv.get("unit_inputs", 0.0) + conv.get("wires", 0.0) + wit.get("typed", 0.0)
    w_other = wit.get("mask", 0.0) + wit.get("z_lincheck", 0.0)
    for k, v in wit.items():
        if k.startswith("net"):
            role = cl["roles"].get(k.split(":", 1)[1], "unit")
            if role in HASH:
                w_hash += v
            else:
                w_arith += v
    piop = sum(phases.get(k, 0.0) for k in PIOP)
    p_hash = piop * sum(share.get(r, 0.0) for r in HASH)
    p_arith = piop * sum(share.get(r, 0.0) for r in ARITH)
    p_other = piop - p_hash - p_arith
    b = {}
    for x in live.get("buckets") or []:
        for k, v in (x or {}).items():
            if isinstance(v, (int, float)):
                b[k] = b.get(k, 0.0) + float(v)
    zk = b.get("t.prove", 0.0) - b.get("t.upstream", 0.0)
    upstream_rest = b.get("t.upstream", 0.0) - piop
    witness = live.get("witness_s") or 0.0
    w_rest = witness - w_hash - w_arith - w_other
    prove = (live.get("prove_total_s") or 0.0)
    return {"hash_s": w_hash + p_hash, "arith_s": w_arith + p_arith, "witness_hash_s": w_hash, "witness_arith_s": w_arith,
            "piop_hash_s": p_hash, "piop_arith_s": p_arith, "mask_unused_s": w_other + p_other, "zk_s": zk,
            "other_prover_s": upstream_rest + w_rest, "witness_s": witness, "prove_total_s": prove, "piop_s": piop,
            "phases": {k: phases.get(k, 0.0) for k in PIOP}, "e2e_s": live.get("e2e_s"), "wait_e2e_s": live.get("wait_e2e_s"),
            # before the witness: Register (the verifier's draw and its statement) and the prover's drawn statement
            "setup_s": (live.get("e2e_s") or 0.0) - witness - (live.get("tp_s") or 0.0),
            "register_s": (setup or {}).get("register_s"), "tables_s": (setup or {}).get("tables_s"),
            # after the witness, beside the prover: the verifier's rounds and the transport
            "rounds_s": (live.get("tp_s") or 0.0) - prove}


def one(point: Path, stage: dict) -> dict | None:
    err, out = point / "prove.err", point / "prove.out"
    if not err.exists() or not out.exists():
        return None
    lay = lines(err, "LAYOUT")
    if not lay:
        return None
    cl = cells(lay[0])
    lives = lines(out, "LIVE")
    conv, wit, setups = lines(err, "CONVERGE"), lines(err, "WITNESSCPU"), lines(err, "SESSIONPHASE")
    core = [x for x in lines(err, "CPUPHASE") if "t.commit" in x]
    opens = [x for x in lines(err, "CPUPHASE") if "t.open" in x]
    n = len(lives)
    if not n or not (len(conv) == len(wit) == len(core) == len(opens) == n):
        return {"error": f"{n} sessions, {len(conv)} CONVERGE, {len(wit)} WITNESSCPU, {len(core)}+{len(opens)} CPUPHASE"}
    timed = [i for i, x in enumerate(lives) if not x.get("warm") and x.get("accepted") is True] or list(range(n))
    rs = [session_split(conv[i], wit[i], {**core[i], **opens[i]}, lives[i], cl, setups[i] if len(setups) == n else None)
          for i in timed]
    med = {k: statistics.median(r[k] for r in rs) for k in rs[0] if all(isinstance(r[k], (int, float)) for r in rs)}
    med["phases"] = {k: statistics.median(r["phases"][k] for r in rs) for k in PIOP}
    alloc, used = cl["alloc"], cl["used"]
    used_hash = sum(used.get(r, 0) for r in HASH)
    used_all = sum(v for k, v in used.items() if k != "mask")
    return {"definition": stage.get("definition"), "n": stage.get("n"), "k_log": stage.get("k_log"),
            "vus_per_block": stage.get("vus_per_block"), "ands": stage.get("ands"), "blocks": lay[0]["blocks"],
            "blocks_real": lay[0]["blocks_real"], "m": lay[0]["m"], "cells": alloc, "used": used,
            "hash_share_alloc": sum(alloc.get(r, 0) for r in HASH) / (1 << lay[0]["m"]),
            "hash_share_used": used_hash / used_all if used_all else None,
            "sessions": len(rs), "accepted": all(lives[i].get("accepted") is True for i in timed), "median": med,
            "per_session": rs}


def main() -> int:
    o = Path(sys.argv[1])
    res = {}
    for st in sorted(o.glob("*-c*/stage/stage.json")):
        u = st.parent.parent.name.rsplit("-c", 1)[0]
        stage = json.loads(st.read_text())
        for point in sorted(o.glob(f"points/{u}-c*-k*/nofw-warm0")):
            r = one(point, stage)
            if r is not None:
                res[u] = r
    (o / "split.json").write_text(json.dumps(res, indent=1))
    print("| type | k | blocks | hash share of cells (alloc / used) | witness hash / arith s | PIOP hash / arith s | "
          "mask+unused s | ZK extras s | hash / arith / prove s | hash share of prove | setup / rounds / e2e s |")
    print("|---|---:|---:|---|---|---|---:|---:|---|---:|---|")
    for u, r in res.items():
        if "error" in r:
            print(f"| {u} | | | {r['error']} | | | | | | | |")
            continue
        m = r["median"]
        prove = m["witness_s"] + m["prove_total_s"]
        print(f"| {u} | {r['k_log']} | {r['blocks']} | {r['hash_share_alloc']:.1%} / {r['hash_share_used']:.1%} | "
              f"{m['witness_hash_s']:.2f} / {m['witness_arith_s']:.2f} | {m['piop_hash_s']:.2f} / {m['piop_arith_s']:.2f} | "
              f"{m['mask_unused_s']:.2f} | {m['zk_s']:.2f} | {m['hash_s']:.2f} / {m['arith_s']:.2f} / {prove:.2f} | "
              f"{m['hash_s'] / prove:.1%} | {m['setup_s']:.1f} / {m['rounds_s']:.1f} / {m['e2e_s']:.1f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
