#!/usr/bin/env python3
"""PAD-INT report-only check (GUIDANCE §50.3 on the padded records): the certificate's VU population (the capture's
committed leaves) against the padded program Serve_v3's value population.

Numbers only, nothing is fixed. The program's prefill is Serve_v2's (= Serve_v1's via CF1): the final RMSNorm is a program
value at the SAMPLED prefill row only (t = LP-1), while vLLM applies it to every prefill row; every decode step (live or
inactive) has its own Final call. So the committed leaves in EXCESS of the program are exactly the `norm.rmsnorm` leaves at
prefill rows t < LP-1 (both ports) -- the same rule CAPCORR's L2 locator found for the unpadded program (`NotInProgram`, 62 per
32-token request), applied here by leaf identity.  Everything else the capture commits (embed, layer ops, lm_head, sampler at
every position; kv writes) has a program value in Serve_v3 at every step by construction (inactive steps compute the same
kinds on the pad token).

    python -m scripts.padint_population_check --record RECORD [--lp 32] --out OUT.json
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
POC = HERE.parent
sys.path.insert(0, str(POC))

_T = re.compile(r"/t=(\d+)/ph=(\w+)/")


def _t_ph(cid: str) -> tuple[int, str]:
    m = _T.search(cid)
    return (int(m.group(1)), m.group(2)) if m else (-1, "?")


def _op(cid: str) -> str:
    m = re.search(r"/op=([^/]+)/", cid)
    return m.group(1) if m else "?"


def in_excess(cid: str, lp: int) -> bool:
    t, ph = _t_ph(cid)
    return _op(cid) == "norm.rmsnorm" and ph == "prefill" and t < lp - 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--record", required=True)
    ap.add_argument("--commit-dir", default=None, help="the capture's commit dir (default RECORD/adapter/selective/commit)")
    ap.add_argument("--lp", type=int, default=None, help="prompt length (default: program.json LP)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from verity_vllm.capture import CaptureReader

    d = Path(a.record).resolve()
    pj = json.load(open(d / "program" / "program.json"))
    lp = a.lp or int(pj["program"]["LP"])
    commit = Path(a.commit_dir) if a.commit_dir else None
    if commit is None:
        for cand in ("adapter/selective", "adapter/commit-v2"):
            if (d / cand).exists():
                commit = d / cand
                break
    reader = CaptureReader(commit)
    manifest = json.load(open(d / "manifest.json"))
    sv = json.load(open(d / "verify" / "selective_verify.json"))
    ch = json.load(open(d / "verify" / "challenge.json"))

    total = 0
    excess: list[str] = []
    per_request: dict[str, dict[str, int]] = {}
    for rid in reader.request_ids():
        n = e = 0
        for _, step in reader.ru_keys(rid):
            for ent in reader.ru(rid, step).index.entries:
                n += 1
                if in_excess(ent.canonical_id, lp):
                    e += 1
                    excess.append(ent.canonical_id)
        per_request[rid] = {"leaves": n, "excess": e}
        total += n
    exc_set = set(excess)
    cert = list(ch.get("certified_sample") or [])
    cov = list(ch.get("coverage_sample") or [])
    sampled = list(ch.get("sample") or (cert + cov))
    cert_ex = [c for c in cert if c in exc_set]
    cov_ex = [c for c in cov if c in exc_set]
    samp_ex = [c for c in sampled if c in exc_set]
    kinds = collections.Counter((_op(c), _t_ph(c)[1], "port=" + c.rsplit("port=", 1)[-1]) for c in excess)
    rows_t = sorted({_t_ph(c)[0] for c in excess})
    doc = {
        "schema": "padint/population-check/v1", "record": d.name, "program_digest": manifest["inputs"].get("program_digest"),
        "program_function": pj["program"]["function"], "LP": lp,
        "rule": "excess = committed leaves with no program value = op norm.rmsnorm at prefill rows t < LP-1 (all ports); Serve_v3's prefill "
                "applies the final RMSNorm at the sampled row only (as Serve_v1/Serve_v2), every decode step has its own Final; "
                "applied by leaf identity (the same rule CAPCORR's L2 locator raised as NotInProgram on the unpadded program)",
        "population": {"committed_leaves": total, "recorded_population_vus": (sv.get("population") or {}).get("vus"),
                       "leaf_predicate_leaves": (sv.get("obligations") or {}).get("leaf_predicate_leaves"),
                       "excess": len(excess), "program_values_sampled_from": total - len(excess)},
        "excess_breakdown": {"kinds": [{"op": k[0], "phase": k[1], "port": k[2], "count": v} for k, v in sorted(kinds.items())],
                             "rows_t": rows_t, "per_request": per_request},
        "sampled": {"vus_checked": len(sampled), "certified_sample": len(cert), "coverage_sample": len(cov),
                    "sampled_in_excess": len(samp_ex), "certified_in_excess": len(cert_ex), "coverage_in_excess": len(cov_ex),
                    "certified_in_excess_ids": cert_ex, "coverage_in_excess_ids": cov_ex},
    }
    Path(a.out).write_text(json.dumps(doc, indent=1, sort_keys=True))
    print(f"[population] {d.name} program {str(doc['program_digest'])[:16]}: committed leaves {total} (recorded population {doc['population']['recorded_population_vus']}); "
          f"excess {len(excess)} = {dict(collections.Counter(_op(c) for c in excess))} at prefill rows t={rows_t[0] if rows_t else None}..{rows_t[-1] if rows_t else None}; "
          f"sampled {len(sampled)} (certified {len(cert)}): in excess {len(samp_ex)} (certified {len(cert_ex)}, coverage {len(cov_ex)}) -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
