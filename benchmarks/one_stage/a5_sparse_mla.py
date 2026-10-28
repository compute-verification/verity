"""One-stage audit, run A5, on a sparse-MLA split stand-in: `integrations/vllm/tests/pipeline/test_serving_rows_splits.py`'s
window (one ``SparseMlaSplit_v1{T=40,NS=3,NH=2,DR=16,DV=32,BN=16}`` row in its split units: four 16-key parts, in two chunks, two
8-key parts and two three-split combines, each combine slot wired to the output rows of the part unit that wrote it), its run
``<ROOT>/rank0``, committed as serving commits it (`pipeline.serving_rows.read_all` on the fixture's served values, each part's
partial and lse by the twin, `ml.windows.rows.commit`) and audited by `a5.audit` as rank 0 of a one-rank step on a local
challenger's coins under ``stratified:1``, one unit of each template: the wiring check on the public bodies before the draw
(`commit.wiring.check`), each template's circuit (`compose_circuit.py --template`; the 16-key part's MUFU reads in lookup
slots, the 8-key part's and the combine's inline, `chunk_rows.reads`), M0's `--zk` session of each drawn unit behind the firewall, the Lean verifier of record, the stream's
outcome and the negatives (`a5.wiring_negatives` among them). It stands in for step 16's 78 rows (at T = 272, per head, four
64-key parts, one 16-key part and a five-split combine), whose units it does not prove. The run exits 0 only when the outcome
is accepted and every negative refused. ``--out`` holds the window serving wrote (``window/``, its private files too) and a5's
records (``rank0/``, ``challenger/``, ``summary.json``).

    PYTHONPATH=<tree>:<tree>/benchmarks/numerical/python:<tree>/integrations/vllm \\
      python benchmarks/one_stage/a5_sparse_mla.py --m0-pythonpath PATH --bin flock-circuit --lean flock-verify \\
        --firewall flock-firewall --main FILE --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "integrations" / "vllm"))

import a5  # noqa: E402
import a5_coins as AC  # noqa: E402
from tests.pipeline import test_serving_rows_splits as TS  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import draw as D, registration as R  # noqa: E402
from verity_vllm.commit import index as IX  # noqa: E402
from verity.ml.windows import rows as CS  # noqa: E402

#: the stand-in's session root, as a TP Commit's run root is its ranks' session root
ROOT = hashlib.sha512(b"verity/a5/sparse-mla-split-stand-in/v1").hexdigest()
RULE = "stratified:1"
OF = {"fixture": "integrations/vllm/tests/pipeline/test_serving_rows_splits.py", "row": TS.row()}


def commit(out: Path) -> CS.Binding:
    """The fixture's window of one row under ``<ROOT>/rank0`` committed into ``out`` as serving commits it, its salts fresh."""
    s = TS.Served()
    w = replace(TS.window(s, law=D.parse_law(RULE)), run=f"{ROOT}/rank0")
    got = TS.read(s, w.members)
    c = CS.commit(w, [g for g, _e in got])
    CS.write(c, out, {"of": OF, "committer": {"module": "a5_sparse_mla.py"}}, IX.index_doc(w.members, [e for _g, e in got], w.run),
             kind="stand-in", registrar=R)
    return w


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m0-pythonpath", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--lean", required=True)
    ap.add_argument("--firewall", required=True)
    ap.add_argument("--main", required=True, help="the verifier's own `git rev-list --first-parent origin/main`")
    ap.add_argument("--out", required=True)
    ap.add_argument("--delta", type=float, default=0.5)
    ap.add_argument("--port", type=int, default=7561)
    ap.add_argument("--session-timeout", type=float, default=3600)
    ap.add_argument("--compose-cache", help="a5.py's: this machine's composed circuits (`compose_circuit.py --cache`)")
    a = ap.parse_args()
    out = Path(a.out)
    mine = commit(out / "window")

    def source(w: dict) -> dict:
        ok = {"kind": w.get("kind") == "stand-in", "run": w.get("run") == mine.run, "of": w.get("of") == OF}
        return {"ok": all(ok.values()), "codes": [c for c, v in ok.items() if not v], "detail": {"kind": w.get("kind"), "run": w.get("run")}}

    a.rule, a.window_root = RULE, ROOT
    manager, a.coins, finisher = AC.start("local", ROOT, [0], out / "challenger", a.main)
    try:
        s, ok = a5.audit(a, {}, out / "rank0", mine, out / "window", source, a.port, {"rank": 0, "standin": OF}, rank=0)
    finally:
        a.coins.skip(0)
        finisher.join()
        published = a.coins.evidence()
        manager.shutdown()
    summary = {"root": ROOT, "session": (published.get("plan") or {}).get("session"),
               "window": {"population": mine.population, "members": [[m.key, m.n] for m in mine.members],
                          "wired": {m.key: [[p, o, list(ws)] for p, o, ws in m.wired] for m in mine.members if m.wired}},
               "challenger": {x: published.get(x) for x in ("trusted", "error")},
               "rank0": {x: s.get(x) for x in ("checks_before_draw", "refused", "audit", "draw", "draw_counts", "negatives", "coins",
                                                "circuits", "timing")},
               "accepted": ok and not published.get("error")}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    print("A5-SPARSE-MLA " + json.dumps({x: summary[x] for x in ("accepted", "window")} | {"draw": s.get("draw"),
                                         "negatives": s.get("negatives")}, sort_keys=True, default=str))
    return 0 if summary["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
