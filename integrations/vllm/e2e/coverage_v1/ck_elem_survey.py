"""CK-ELEM survey: run the elementwise builders over EVERY selected VU of the owned kinds in a record, and emit one
statement per buildable kind for the checker station (a CPU pod: ``veritor-zk-host execute/prove/verify-many``).

    .venv/bin/python zk/campaign/lane_run.py --gb 3 --class test --lane ck-elem -- \
        .venv/bin/python -m e2e.coverage_v1.ck_elem_survey --record ~/veritor-archive/REC-P1-B0-2026-09-10/REC-P1-B0 \
        --out data/contract/ck-elem/survey-b0 [--units-per-statement 1] [--max-per-kind N]

Outputs (``--out``):
* ``survey.json`` -- per kind: selected count, resolved count, refusal reasons (gate numbers / word indices collapsed),
  seconds; the record identity; LABEL MEASURED (the builder ran on the record's actual selection).
* ``statements/<kind>/{statement.bin, witness.bin, composed.json, expectations.json, local.json}`` -- the typed
  statement (format 3) over the first ``--units-per-statement`` resolved units of the kind, the packed witness, the
  composed record (``digest`` = h_O for ``--typed-object-digest``), the verifier's expectations for those units and the
  local verdicts (transparent backend; ``check_expected_positions``).  LABEL: local verdicts are the guest's Python
  mirror, not the approved checker -- the pod's ``prove``/``verify`` seconds are the checker measurement.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[3]
for _p in (str(_REPO), str(_REPO / "src"), str(_REPO / "vllm-poc")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from e2e.coverage_v1 import ck_elem as CE  # noqa: E402
from e2e.coverage_v1.elem_builders import BUILDERS  # noqa: E402

_COLLAPSE = re.compile(r"(@|\[|t=|/l=|layers\.|/c=)\d+")


def collapse(msg: str) -> str:
    """One reason per refusal family: gate numbers, word indices, t, layer collapsed to ``*``."""
    return _COLLAPSE.sub(lambda m: m.group(1) + "*", msg)


def open_record(rec: Path, timings: dict[str, float]):
    from e2e.verifier_v3.locator import ProgramLocator, build_pinned_program

    inst = json.loads((rec / "protocol_instance.json").read_text())
    ident = inst["program"]["identity"]
    program, C, problems = build_pinned_program(ident, expected_digest=inst["program"]["digest"], expected_gates=inst["program"]["gates"]["value"], timings=timings)
    if program is None or problems:
        raise SystemExit(f"pinned program: {problems}")
    t0 = time.perf_counter()
    loc = ProgramLocator(program, C)
    timings["locator_s"] = time.perf_counter() - t0
    handle = CE.RecordHandle.open(rec / "adapter" / "selective", program, C, loc)
    return inst, handle


def located_units(rec: Path, kinds: set[str], max_per_kind: int | None) -> dict[str, list[dict[str, Any]]]:
    sel = json.loads((rec / "chronology" / "selected.json").read_text())
    ru_req = {int(r["ru"]): r["request_id"] for r in json.loads((rec / "ru_map.json").read_text())["rus"]}
    per: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in sel["selected"]:
        k = f"{r['kind']['fn']}|{r['kind']['form']}"
        if k not in kinds:
            continue
        if max_per_kind is not None and len(per[k]) >= max_per_kind:
            continue
        ru = int(r["ru_index"])
        per[k].append({"vu_index": int(r["vu_index"]), "ru": ru, "request_id": ru_req[ru], "kind": k, "gate_interval": [int(x) for x in r["gate_interval"]]})
    return per


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--units-per-statement", type=int, default=1)
    ap.add_argument("--max-per-kind", type=int, default=None)
    a = ap.parse_args(argv)
    rec, out = Path(a.record).expanduser(), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    inst, handle = open_record(rec, timings)
    owned = set(CE.KINDS) | set(CE.NO_CHECKER)
    sel = json.loads((rec / "chronology" / "selected.json").read_text())
    consts = {f"{r['kind']['fn']}|{r['kind']['form']}" for r in sel["selected"] if CE._is_const(f"{r['kind']['fn']}|{r['kind']['form']}")}
    per = located_units(rec, owned | consts, a.max_per_kind)
    survey: dict[str, Any] = {
        "schema": "veritor/ck-elem/survey/v1",
        "label": "MEASURED: the builders ran on the record's actual selection (chronology/selected.json); resolved = one gate over committed words",
        "record": str(rec), "program": {"digest": inst["program"]["digest"], "identity": inst["program"]["identity"], "gates": inst["program"]["gates"]["value"]},
        "timings_s": {},
        "kinds": {},
    }
    resolved: dict[str, list[CE.ResolvedUnit]] = collections.defaultdict(list)
    for k in sorted(per):
        rows = per[k]
        t0 = time.perf_counter()
        reasons: collections.Counter[str] = collections.Counter()
        for u in rows:
            try:
                resolved[k].append(CE.resolve(u, handle))
            except CE.Refusal as e:
                reasons[collapse(str(e))] += 1
        dt = time.perf_counter() - t0
        survey["kinds"][k] = {
            "selected": len(rows), "resolved": len(resolved[k]),
            "builder": "BUILDERS" if k in BUILDERS else ("KINDS (gate in the approved guest; words not committed)" if k in CE.KINDS else "none"),
            "family": CE.KINDS[k].family if k in CE.KINDS else None,
            "refusals": dict(reasons.most_common()), "seconds": round(dt, 2),
        }
        print(f"[survey] {k:34s} selected={len(rows):4d} resolved={len(resolved[k]):4d} {dt:7.1f}s", flush=True)
    survey["timings_s"] = {k: round(v, 3) for k, v in timings.items()}
    # statements for the checker station
    from veritor.interface.expected import CommittedTree, check_expected_positions

    trees = {t: CommittedTree(r["root"], int(r["leaf_count"])) for t, r in handle.commitments["trees"].items()}

    def rank_of(tree: str, cid: str):
        if tree == CE.WEIGHTS_TREE:
            return handle.open_leaf(cid).index
        ids = handle.ids_of(tree)
        try:
            return ids.index(cid)
        except ValueError:
            return None

    emitted: dict[str, Any] = {}
    for k, units in sorted(resolved.items()):
        if k not in BUILDERS or not units:
            continue
        chosen = units[: a.units_per_statement]
        t0 = time.perf_counter()
        built = CE.build_many(chosen, handle, name=f"ck-elem-{k}")
        t_build = time.perf_counter() - t0
        t0 = time.perf_counter()
        verdict = CE.verify_transparent(built)
        t_local = time.perf_counter() - t0
        exps = built.expectations
        rep = check_expected_positions(built.obj, exps, trees=trees, rank_of=rank_of)
        d = out / "statements" / k.replace("|", "__")
        d.mkdir(parents=True, exist_ok=True)
        (d / "statement.bin").write_bytes(built.statement_bytes)
        (d / "witness.bin").write_bytes(built.witness_bytes)
        (d / "composed.json").write_text(json.dumps(built.record, indent=1, sort_keys=True))
        (d / "expectations.json").write_text(json.dumps([{"label": e.label, "subcircuit": e.subcircuit,
                                                           "incoming": [[v.tree, v.leaf, list(v.offset)] for v in e.incoming],
                                                           "outgoing": [[v.tree, v.leaf, list(v.offset)] for v in e.outgoing]} for e in exps], indent=1))
        local = {"kind": k, "family": built.family, "units": len(chosen), "obligations": len(built.obj.obligations),
                 "statement_bytes": len(built.statement_bytes), "witness_bytes": len(built.witness_bytes), "h_O": built.record["digest"],
                 "transparent_backend": verdict, "positions": rep.to_json(), "build_s": round(t_build, 3), "transparent_s": round(t_local, 3),
                 "label": "local verdicts = the guest's Python mirror at the pinned gate-set digest, NOT the approved checker",
                 "pod": {"execute": f"veritor-zk-host execute --statement statement.bin --witness witness.bin --typed-object-digest {built.record['digest']}",
                         "prove": f"veritor-zk-host prove --statement statement.bin --witness witness.bin --typed-object-digest {built.record['digest']} --out proof.bin --mode compressed",
                         "verify": "veritor-zk-host verify --proof proof.bin --statement statement.bin"}}
        (d / "local.json").write_text(json.dumps(local, indent=1))
        emitted[k] = local
        print(f"[emit] {k}: {len(chosen)} unit(s) transparent={verdict} positions={'ok' if rep.ok else rep.problems[:1]} -> {d}", flush=True)
    survey["statements"] = emitted
    (out / "survey.json").write_text(json.dumps(survey, indent=1))
    print(f"[survey] written {out / 'survey.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
