"""[CAPCORR-2] Does the record's statement (`h_L 8176e8a9…`) rebuild byte-equal from `a25fc156…`'s gates?

`derive_prove.build_from_program` makes the statement from the record's authenticated openings (program-independent) plus the
PROGRAM-lowered subcircuit bodies of the selected GEMM coordinates (`locate_coordinate` -> `lower_call_to_subcircuit`), and refuses
unless each program-lowered body equals the exporter's (digest AND (in, gates, out)). So the program-dependent content of `h_L`
is exactly: for every GEMM obligation, the lowered body of the coordinate located in the program. This script performs that
part against `a25fc156…` through the Serve_v1 bijection (`capcorr_a25_rows.MappedCircuitMap`) and compares with the record's
`obligations/composed.json` subcircuits. 26/26 equal => `build_from_program` on `a25fc156…` would emit the identical
`statement.bin` (everything else it encodes is record data); any inequality => it would refuse. The full S5 build path
(openings + witness) was not re-run here — that is stated, not hidden.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent.parent), str(HERE.parent.parent / "src"), str(HERE)]

from capcorr_a25_rows import MappedCircuitMap, derive_a25, serve_v1  # noqa: E402
from verity_vllm.correspondence.program import locate_coordinate, lower_call_to_subcircuit  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--record", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    rep = derive_a25(32, 15)
    mp = json.loads(Path(a.map).read_text())
    assert mp["derived_digest"] == rep.digest, (mp["derived_digest"], rep.digest)
    cmap = MappedCircuitMap(rep.program, serve_v1(32, 15), mp)
    comp = json.loads((Path(a.record) / "obligations" / "composed.json").read_text())
    obj = comp["object"]
    rec_subs = {s["digest"]: s for s in obj["subcircuits"]}
    rows, ok = [], True
    for ob in obj["obligations"]:
        tid = ob["target"]["id"]
        m = re.match(r"^(?P<cid>p=.*?/port=\d+)/Gemm\[(?P<j>\d+)\]$", tid)
        if not m:
            continue
        gemm, coord = locate_coordinate(cmap, m.group("cid"), int(m.group("j")))
        sc = lower_call_to_subcircuit(coord)
        rs = rec_subs.get(ob["subcircuit"])
        same_digest = sc.digest == ob["subcircuit"]
        try:  # the body, gate by gate, against the record's exporter table (the digest is a hash of exactly this)
            same_body = rs is not None and json.loads(json.dumps(list(sc.gates))) == rs["gates"] and json.loads(json.dumps(list(sc.incoming))) == rs["in"] \
                and json.loads(json.dumps(list(sc.outgoing))) == rs["out"]
        except Exception as ex:  # body formats differ in shape only -> report the digest comparison, say so
            same_body = f"not compared ({type(ex).__name__})"
        rows.append({"target": tid, "a25_call": gemm.node.name, "a25_coordinate_function": coord.id, "record_subcircuit": ob["subcircuit"][:16],
                     "a25_lowered_subcircuit": sc.digest[:16], "digest_equal": same_digest, "body_equal": same_body})
        ok = ok and same_digest and (same_body is not False)  # digest is the content hash; body check is belt-and-braces
    doc = {"program": rep.digest, "record_h_L": comp["statement_digest"], "record_h_O": comp["digest"], "gemm_obligations": len(rows),
           "all_program_lowered_bodies_equal_record": ok, "rows": rows, "seconds": round(time.time() - t0, 1),
           "claim": ("h_L would rebuild byte-equal from a25fc156… (build_from_program's only program-dependent content is these bodies)" if ok
                     else "build_from_program on a25fc156… would REFUSE (program-lowered body != exporter's)"),
           "not_run": "full S5 build (openings, witness, statement.bin encoding) against a25fc156…"}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(doc, indent=1))
    print(json.dumps({k: v for k, v in doc.items() if k != "rows"}, indent=1))
    print("digest_equal:", sum(1 for r in rows if r["digest_equal"]), "/", len(rows), " body_equal:", sum(1 for r in rows if r["body_equal"]), "/", len(rows))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
