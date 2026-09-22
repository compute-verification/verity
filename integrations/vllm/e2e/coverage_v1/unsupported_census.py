"""Tabulate a record's UNSUPPORTED selected VUs (``obligations/unsupported_selected.json``) by kind and by the operator the verifier's
own ``ProgramLocator`` places them in (root call ``t=<t>/layers.<L>/<op>`` of the pinned program, built in-process) -- so the
attention-body share is by LOCATION, not by kind-name guess -- and bucket them: attention body (A5/6), CK-ELEM's four
elementwise kinds (now buildable), and the rest by operator, with the owner column left explicit.

    PYTHONPATH=. python -m e2e.coverage_v1.unsupported_census --record <rec> --out <dir>/unsupported_census
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

CK_ELEM = {"SiluMulBf16_v1|batch", "Bf16AddF2fp_v1|batch", "RopeOut_v1|batch", "RopeOutAdd_v1|batch"}
A56_INTERIOR = {"AmpereBF16TcDot16_v1|batch", "F32MulFtz_v1|primitive", "MufuEx2Ftz_v1|primitive", "DotBf16_v1{K=64}|call", "F2fpBf16_v1|primitive",
                "F32FmaSubFtz_v1|primitive", "F32Max_v1|primitive", "F32AddFtz_v1|primitive"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--record", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rec = Path(a.record)
    from e2e.verifier_v3 import locator as vloc

    t0 = time.perf_counter()
    inst = json.loads((rec / "protocol_instance.json").read_text())
    maps = json.loads((rec / "maps.json").read_text())
    program, C, problems = vloc.build_pinned_program(inst["program"]["identity"], expected_digest=maps["program"]["digest"], expected_gates=None)
    if program is None or problems:
        raise SystemExit(f"program not rebuilt: {problems}")
    loc = vloc.ProgramLocator(program, C)
    build_s = time.perf_counter() - t0
    u = json.loads((rec / "obligations" / "unsupported_selected.json").read_text())
    by_kind_op: dict[str, Counter] = defaultdict(Counter)
    by_op: Counter = Counter()
    per_t: Counter = Counter()
    for unit in u["units"]:
        lo = int(unit["gate_interval"][0])
        name, root = loc.root_of(lo)
        parts = (name or "").split("/")
        op = parts[2] if len(parts) == 3 else (name or "?")
        by_kind_op[unit["kind"]][op] += 1
        by_op[op] += 1
        if op == "attn":
            per_t[parts[0]] += 1
    rows = []
    buckets: Counter = Counter()
    for kind, ops in sorted(by_kind_op.items(), key=lambda kv: -sum(kv[1].values())):
        n = sum(ops.values())
        if kind in CK_ELEM:
            bucket, owner = "CK-ELEM four (buildable)", "CK-ELEM"
        elif ops.get("attn", 0) == n:
            bucket, owner = "attention body (interior, uncommitted -- InteriorRefuser / head VU PROPOSED)", "A5/6"
        elif ops.get("attn", 0):
            bucket, owner = "mixed (attention + other operators)", "A5/6 (attn part) / open"
        else:
            bucket, owner = "rest", "open (integrator to assign)"
        buckets[bucket] += n
        rows.append({"kind": kind, "count": n, "by_operator": dict(ops), "bucket": bucket, "owner": owner, "a56_interior_kind": kind in A56_INTERIOR})
    doc = {"schema": "veritor/a56/unsupported-census/v1", "record": str(rec), "program": inst["program"]["identity"].get("function"), "program_build_s": round(build_s, 2),
           "unsupported_selected": u["count"], "by_operator": dict(by_op.most_common()), "attention_by_t": dict(sorted(per_t.items())), "buckets": dict(buckets), "rows": rows,
           "label": "counts of SELECTED unsupported VUs in retained RUs (the record's own list), located by the verifier's ProgramLocator; not a population census"}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(doc, indent=1) + "\n")
    md = [f"# Unsupported selected VUs by kind and operator — `{rec.name}` ({u['count']} units; program `{doc['program']}`, built in {build_s:.1f} s)", "",
          "| kind | count | operator(s) | bucket | owner |", "|---|---|---|---|---|"]
    for r in rows:
        ops = ", ".join(f"{k} {v}" for k, v in sorted(r["by_operator"].items(), key=lambda kv: -kv[1]))
        md.append(f"| `{r['kind']}` | {r['count']} | {ops} | {r['bucket'].split(' (')[0]} | {r['owner']} |")
    md += ["", "| bucket | units | share |", "|---|---|---|"]
    for b, n in buckets.most_common():
        md.append(f"| {b} | {n} | {100 * n / u['count']:.1f} % |")
    md += ["", "By operator: " + ", ".join(f"`{k}` {v}" for k, v in by_op.most_common()), "", doc["label"]]
    out.with_suffix(".md").write_text("\n".join(md) + "\n")
    print("\n".join(md[-(len(buckets) + 6):]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
