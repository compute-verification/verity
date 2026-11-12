"""One-stage audit, run A5, on a sparse-MLA split stand-in: `adapters/vllm/tests/pipeline/test_serving_rows_splits.py`'s
window (one ``SparseMlaSplit_v1{T=40,NS=2,NH=2,DR=16,DV=32,BN=16}`` row in its heads' split chains: per head, split 0's 16-key
first block and the full block after it, split 1's 8-key first block, each split's end in one column block and the two-split
combine, each read row wired to the output row of the unit that wrote it; each key block its Boolean body,
``SparseMlaHeadBlock_v2``, the fixture's ``BITS``), its run ``<ROOT>/rank0``, committed as serving
commits it (`pipeline.serving_rows.read_all` on the fixture's served values, each block's state and split end by the twin,
`ml.windows.rows.commit`) and audited by `a5.audit` as rank 0 of a one-rank step on a local challenger's coins under
``stratified-rate:1/64``, a 64th of each template's units and at least one: the wiring check on the public bodies before the
sample (`commit.wiring.check`), each template's circuit (`compose_circuit.py --template`; every one flat, `chunk_rows.reads`:
a key block's EX2 is its body's gates, and the column end's and the combine's MUFU reads are inline), M0's `--zk` session of each
sampled proof block (behind the ZK server under ``--firewall``), the Lean verifier, the stream's outcome and the negatives
(`a5.wiring_negatives` among them). It stands in for step 16's 78 rows (at T = 272, per head, four 64-key first blocks, one
16-key one, forty column ends and eight combines), whose units it does not prove. The run exits 0 only when the outcome
is accepted and every negative refused. ``--out`` holds the window serving wrote (``window/``, its private files too) and a5's
records (``rank0/``, ``challenger/``, ``summary.json``).

    PYTHONPATH=<tree>:<tree>/benchmarks/numerical/python:<tree>/adapters/vllm \\
      python benchmarks/one_stage/a5_sparse_mla.py --m0-pythonpath PATH --bin flock-circuit --lean flock-verify \\
        [--firewall flock-firewall] --main FILE --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "adapters" / "vllm"))

import a5  # noqa: E402
import a5_coins as AC  # noqa: E402
from tests.pipeline import test_serving_rows_splits as TS  # noqa: E402
from verity.protocols.spot.verifier.one_stage import registration as R  # noqa: E402
from verity_vllm.commit import index as IX  # noqa: E402
from verity.commitments.windows import rows as CS  # noqa: E402

#: the stand-in's session root, as a TP Commit's run root is its ranks' session root
ROOT = hashlib.sha512(b"verity/a5/sparse-mla-split-stand-in/v1").hexdigest()
RULE = "stratified-rate:1/64"
OF = {"fixture": "adapters/vllm/tests/pipeline/test_serving_rows_splits.py", "row": TS.row()}
#: what a run's summary keeps of the rank's own (`a5.audit`'s), and what its printed line shows of it
RANK = ("checks_before_draw", "refused", "audit", "draw", "draw_counts", "negatives", "coins", "circuits", "timing")
SHOWN = ("draw", "negatives")


def committed(w: CS.Binding, got: list, out: Path, of: dict, module: str) -> CS.Binding:
    """Window ``w`` on its members' values ``got`` (`serving_rows.read_all`'s) committed into ``out`` as serving commits it, its
    salts fresh: a stand-in of ``of``, committed by ``module``."""
    c = CS.commit(w, [g for g, _e in got])
    CS.write(c, out, {"of": of, "committer": {"module": module}}, IX.index_doc(w.members, [e for _g, e in got], w.run),
             kind="stand-in", registrar=R)
    return w


def commit(out: Path) -> CS.Binding:
    """The fixture's window of one row under ``<ROOT>/rank0`` committed into ``out`` (`committed`)."""
    s = TS.Served()
    w = a5.rated(replace(TS.window(s, TS.BITS, TS.BITS_PER_CHUNK), run=f"{ROOT}/rank0"), RULE)
    return committed(w, TS.read(s, w.members, TS.BITS), out, OF, "a5_sparse_mla.py")


def parser() -> argparse.ArgumentParser:
    """A stand-in's arguments: a5.py's own for one rank on a local challenger's coins (`audited`)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--m0-pythonpath", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--lean", required=True)
    ap.add_argument("--firewall", help="a5.py's --firewall: the core protocols' path, every session behind the ZK server; "
                    "without it each runs prove --zk itself, as SPOT[Flock]'s proof server does")
    ap.add_argument("--main", required=True, help="the verifier's own `git rev-list --first-parent origin/main`")
    ap.add_argument("--out", required=True)
    ap.add_argument("--delta", type=float, default=0.5)
    ap.add_argument("--port", type=int, default=7561)
    ap.add_argument("--session-timeout", type=float, default=3600)
    ap.add_argument("--compose-cache", help="a5.py's: this machine's composed circuits (`compose_circuit.py --cache`)")
    return ap


def audited(a: argparse.Namespace, mine: CS.Binding, of: dict, root: str = ROOT, rule: str = RULE) -> tuple[dict, dict]:
    """Window ``mine``, a stand-in of ``of`` committed in ``<a.out>/window`` (`committed`), audited by `a5.audit` as rank 0 of a
    one-rank step on a local challenger's coins, its session root ``root``, under ``rule``: the run's summary, with each
    template's units held and sampled, and the rank's own (`a5.audit`'s)."""
    out = Path(a.out)

    def source(w: dict) -> dict:
        ok = {"kind": w.get("kind") == "stand-in", "run": w.get("run") == mine.run, "of": w.get("of") == of}
        return {"ok": all(ok.values()), "codes": [c for c, v in ok.items() if not v], "detail": {"kind": w.get("kind"), "run": w.get("run")}}

    a.rule, a.window_root = rule, root
    manager, a.coins, finisher = AC.start("local", root, [0], out / "challenger", a.main)
    try:
        s, ok = a5.audit(a, {}, out / "rank0", mine, out / "window", source, a.port, {"rank": 0, "standin": of}, rank=0)
    finally:
        a.coins.skip(0)
        finisher.join()
        published = a.coins.evidence()
        manager.shutdown()
    held = Counter()
    for m in mine.members:
        held[m.template] += m.n
    sampled = s.get("draw_counts") or {}
    summary = {"root": root, "session": (published.get("plan") or {}).get("session"),
               "window": {"population": mine.population, "members": [[m.key, m.n] for m in mine.members],
                          "templates": {t: {"units": n, "sampled": sampled.get(t, 0)} for t, n in held.items()},
                          "wired": {m.key: [[p, o, list(ws)] for p, o, ws in m.wired] for m in mine.members if m.wired}},
               "challenger": {x: published.get(x) for x in ("trusted", "error")},
               "rank0": {x: s.get(x) for x in RANK},
               "accepted": ok and not published.get("error")}
    return summary, s


def main() -> int:
    a = parser().parse_args()
    out = Path(a.out)
    summary, s = audited(a, commit(out / "window"), OF)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    print("A5-SPARSE-MLA " + json.dumps({x: summary[x] for x in ("accepted", "window")} | {x: s.get(x) for x in SHOWN},
                                        sort_keys=True, default=str))
    return 0 if summary["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
