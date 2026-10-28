"""One-stage audit, run A5, on a row-table stand-in: `integrations/vllm/tests/pipeline/test_row_tables.py`'s window (40
`ScaledMmS8Coordinate_v1{K=64}` units in eight chunks, each of whose ``x`` rows link one of four row tables, one per engine step and linear
over two engine steps, under the stratified law), its run ``<ROOT>/rank0``, committed as serving commits it
(`pipeline.serving_rows._read_s8` on the fixture's served values, `ml.windows.rows.commit` under the step roots
serving records before each step's coin, `row_tables.step_roots`) and audited by `a5.audit` as rank 0 of a one-rank
step on the auditor's challenger's coins, a local session on ``ROOT``: the window's `service.Stream` (its ``row-table``
Value a tree over roots of row-seg leaves), `tables.check_row_tables` against those step roots before the draw, M0's
`--zk` session of each drawn chunk behind the firewall, the Lean verifier of record, the stream's outcome and the
negatives. No TP window of record has row tables yet (`tp.serving_rows.window_of_record` names no serve-time context),
so this window stands in for one. Each chunk's public file is a shared-row file, which Lean's verifier of record covers
only with its own registered reads: a5 gives each drawn chunk's verify the reads of its one table (`a5`'s module
docstring), and the run exits 0 only when the outcome is accepted and every negative refused. ``--out`` holds the window
serving wrote (``window/``, its private files too) and a5's records
(``rank0/``, ``challenger/``, ``summary.json``).

    PYTHONPATH=<tree>:<tree>/benchmarks/numerical/python:<tree>/integrations/vllm \\
      python benchmarks/one_stage/a5_row_tables.py --m0-pythonpath PATH --bin flock-circuit --lean flock-verify \\
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
from tests.pipeline import test_row_tables as TR  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import draw as D, registration as R  # noqa: E402
from verity_vllm.commit import index as IX, row_tables as RTB  # noqa: E402
from verity.ml.windows import rows as CS  # noqa: E402
from verity_vllm.pipeline import serving_rows as SRW  # noqa: E402

#: the stand-in's session root, as a TP Commit's run root is its ranks' session root
ROOT = hashlib.sha512(b"verity/a5/row-table-stand-in/v1").hexdigest()
RULE = "stratified:2"
OF = {"fixture": "integrations/vllm/tests/pipeline/test_row_tables.py", "context": TR.CTX.hex()}


def commit(out: Path) -> tuple[CS.Window, dict[int, bytes]]:
    """The fixture's window under ``<ROOT>/rank0`` committed into ``out`` as serving commits it, its members' salts fresh;
    -> (the verifier's window, the step roots serving recorded)."""
    w, s, ms = TR.window()
    w = replace(w, run=f"{ROOT}/rank0")
    if D.parse_law(RULE) != w.law:
        raise SystemExit(f"the fixture's law is {w.law}, not {RULE}")
    SRW._operand = s.operand  # the fixture's served operands, as `test_row_tables.committed` reads them
    got = SRW._read_s8(ms, s.rows, s, None, TR.ARRIVALS)
    live = {step: RTB.step_roots(TR.CTX, step, s.outputs(step), TR.ROW_KEY).root for step, _ in w.segments}
    c = CS.commit(w, [g[0] for g in got], row_salt_key=TR.ROW_KEY, segment_roots=live)
    prov = {"of": {**OF, "step_roots": {str(k): v.hex() for k, v in live.items()}}, "committer": {"module": "a5_row_tables.py"}}
    CS.write(c, out, prov, IX.index_doc(w.members, [g[1] for g in got], w.run), kind="stand-in", registrar=R)
    return w, live


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m0-pythonpath", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--lean", required=True)
    ap.add_argument("--firewall", required=True)
    ap.add_argument("--main", required=True, help="the verifier's own `git rev-list --first-parent origin/main`")
    ap.add_argument("--out", required=True)
    ap.add_argument("--delta", type=float, default=0.5)
    ap.add_argument("--port", type=int, default=7551)
    ap.add_argument("--session-timeout", type=float, default=3600)
    ap.add_argument("--compose-cache", help="a5.py's: this machine's composed circuits (`compose_circuit.py --cache`)")
    a = ap.parse_args()
    out = Path(a.out)
    mine, live = commit(out / "window")

    def source(w: dict) -> dict:
        ok = {"kind": w.get("kind") == "stand-in", "run": w.get("run") == mine.run,
              "of": w.get("of") == {**OF, "step_roots": {str(k): v.hex() for k, v in live.items()}}}
        return {"ok": all(ok.values()), "codes": [c for c, v in ok.items() if not v], "detail": {"kind": w.get("kind"), "run": w.get("run")}}

    a.rule, a.window_root = RULE, ROOT
    manager, a.coins, finisher = AC.start("local", ROOT, [0], out / "challenger", a.main)
    try:
        s, ok = a5.audit(a, {}, out / "rank0", mine, out / "window", source, a.port, {"rank": 0, "standin": OF}, rank=0,
                         step_roots=live)
    finally:
        a.coins.skip(0)
        finisher.join()
        published = a.coins.evidence()
        manager.shutdown()
    summary = {"root": ROOT, "session": (published.get("plan") or {}).get("session"), "window": {"population": mine.population, "members": len(mine.members),
                                                        "row_tables": [t.port for t in mine.row_tables],
                                                        "steps": {str(k): v.hex() for k, v in live.items()}},
               "challenger": {x: published.get(x) for x in ("trusted", "error")},
               "rank0": {x: s.get(x) for x in ("checks_before_draw", "refused", "audit", "draw", "draw_counts", "negatives", "coins")},
               "accepted": ok and not published.get("error")}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    print("A5-ROWTAB " + json.dumps({x: summary[x] for x in ("accepted", "window")} | {"draw": s.get("draw")}, sort_keys=True))
    return 0 if summary["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
