"""Step 5 AFTER the LIVE shipped driver (``e2e.verifier_v3.driver`` ends at ``DONE`` with ``chronology/selected.json``): the prover side builds the
statements of exactly the selected VUs (SHIPPED ``coverage_v1.ck_elem.build_many`` -> E7 ``elem_builders``), proves them on E10's pod (SHIPPED
``scripts/rec_p1_prove.py``: SP1 cuda prove + host verify), and the verifier side re-checks (``runner.step5``: ``V4.prover_summary``, ``V4.guest``,
``V4.proofs_binding``, ``V5.trees``, ``V5.expected``, ``V5.positions``, ``V5.coverage`` -- the same shipped functions the runner calls; nothing new).
Generic protocol demonstration (DemoCore_v1); NOT vLLM inference.

    python -m e2e.demo_core.step5_live --record <driver record dir> [--fixture <dir>] --pod-ssh <E10 wrapper> --pod-dir <remote dir>

Reads from the driver's record: ``protocol_instance.json`` (the pinned instance), ``protocol.json`` (J, F), ``chronology/selected.json``,
``adapter/selective/commitments.json``.  Writes ``obligations/``, ``proofs/``, ``verify/verifier.step5.log``, ``verify/coverage.json``,
``verify/host_verify.jsonl``, ``protocol.step5.json`` (verdict of step 5, coverage line, U).  The record's phase-0/2 trees are NEVER rewritten: the
prover side rebuilds them in ``adapter/selective/rebuilt`` (deterministic writer), compares every root with the committed one, then deletes the copy."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve()
POC = HERE.parents[2]
for p in (POC, POC.parent, POC.parent / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from e2e import partition_sample as ps  # noqa: E402

from . import producer as PR, record as RC  # noqa: E402
from .prover import forge, load_fixture  # noqa: E402
from .runner import LABEL, Log, _certificate_or_error, step5, utc  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="DemoCore_v1 step 5 after the live shipped driver")
    ap.add_argument("--record", required=True, help="the driver's --record dir (protocol.json, chronology/, adapter/selective/)")
    ap.add_argument("--fixture", default=None, help="fixture dir (default <record>/fixture)")
    ap.add_argument("--interior", default="e2", choices=list(RC.INTERIOR_MODES))
    ap.add_argument("--request", default="per-ru", choices=list(RC.REQUEST_MODES))
    ap.add_argument("--forge-gate", type=int, default=None, help="NEGATIVE TEST: the prover's record was built with this forged gate (prover --forge-gate); rebuild the same values")
    ap.add_argument("--statements-only", action="store_true", help="build and write obligations/ (the prover side of step 5) and stop: no proving, no verdict -- for a pod that is not up yet")
    ap.add_argument("--pod-ssh", default=None)
    ap.add_argument("--pod-dir", default=None)
    ap.add_argument("--pod-host", default="/workspace/bin/veritor-zk-host-cuda")
    ap.add_argument("--pod-repo", default="/workspace/veritor")
    ap.add_argument("--pod-python", default=None, help="the pod's runner venv python (default <pod-repo>/.venv/bin/python)")
    a = ap.parse_args(argv)
    rec = Path(a.record).resolve()
    sel = rec / "adapter" / "selective"
    fx = Path(a.fixture) if a.fixture else rec / "fixture"
    RC.set_request_mode(a.request)
    inst_path = rec / "protocol_instance.json"
    inst_doc = json.loads(inst_path.read_text())
    rids = list((inst_doc.get("epoch") or {}).get("request_ids") or [])
    if a.request == "per-ru" and len(rids) == 1:
        RC.set_ru_base(str(rids[0]))
    load_inst, _ = ps.loaders("contract")
    inst = load_inst(inst_path)
    proto = json.loads((rec / "protocol.json").read_text())
    J, F = [int(x) for x in proto["J"]], [int(x) for x in proto["F"]]
    retained = [r for r in J if r not in set(F)]
    selected = json.loads((rec / "chronology" / "selected.json").read_text())
    selected["retained_rus"] = retained
    c2 = json.loads((sel / "commitments.json").read_text())
    # [record integrity, 2026-09-12 02:25Z] step 5 MERGES into verify/ (host_verify.jsonl, coverage.json, verifier.step5.log) and never replaces it: the
    # driver's step 2-4 outputs must already be there, else the record is incomplete and the shipped reverify.py refuses (V8.ru_fields rus=0).  Fail closed
    # here rather than produce a step-5 verdict over a record that lost them (CORE-live-exhaustive-03: an `rm -rf <rec>/verify` between step-5 attempts).
    DRIVER_VERIFY_FILES = ("population_derived.json", "ru_fields.json", "seed_custody.json", "transcript.json")
    missing = [f for f in DRIVER_VERIFY_FILES if not (rec / "verify" / f).exists()]
    if missing:
        raise SystemExit(f"step5_live: {rec}/verify/ lacks the driver's step 2-4 outputs {missing} -- the record is incomplete (restore them from the driver run / git history; "
                         "step 5 only ADDS host_verify.jsonl, coverage.json, verifier.step5.log to verify/, never remove it)")
    log = Log(rec / "verify" / "verifier.step5.log")
    t0 = time.perf_counter()
    # the prover side re-derives its trees (deterministic) to build statements; roots must equal the committed ones
    part, vals, f = load_fixture(fx)
    if a.forge_gate is not None:
        vals = forge(part, vals, int(a.forge_gate))
    core = RC.CoreRecord(part, vals, sel / "rebuilt", interior=a.interior)
    core.commit_phase0()
    core.commit_interior(retained)
    same = all(core.commitments["trees"][t]["root"] == c2["trees"][t]["root"] for t in c2["trees"]) and set(core.commitments["trees"]) == set(c2["trees"])
    log.check("S5.prover_trees_rebuilt", same, "the prover's statement-building record reproduces the committed roots byte-for-byte (deterministic writer)", who="E8-CHECK",
              trees=len(c2["trees"]), request_mode=a.request, ru_id_base=RC.ru_base(), layout=c2.get("interior_layout"))
    shutil.rmtree(sel / "rebuilt", ignore_errors=True)
    if a.statements_only:
        from . import obligations as OB, prove_io as PIO

        Q = [int(r["vu_index"]) for r in selected["selected"]]
        chunks = OB.chunk_units([part.vus[i] for i in Q], core, max_units=64)
        builts = [OB.build_chunk(c, core, name=f"{rec.name}-chunk{k:02d}") for k, c in enumerate(chunks)]
        ob = PIO.write_obligations(rec, builts, core, selected_vus=Q, instance_digest=selected["instance_digest"], note=f"selected.json ({len(Q)} VUs) -> ck_elem.build_many; statements only (no pod yet)")
        log.check("S5.statements_built", ob["covered_vus"] == len(Q) and not ob["uncovered_selected"], "every selected VU is in exactly one typed statement built by coverage_v1.ck_elem.build_many", who="SHIPPED coverage_v1.ck_elem",
                  selected=len(Q), statements=len(builts), transparent=[c["transparent_backend"] for c in ob["chunks"]])
        print(f"[demo_core.step5_live] statements only: {len(builts)} statement(s) for {len(Q)} VUs written to {rec / 'obligations'}; proving pending (no pod)", flush=True)
        return 0
    pod = {"ssh": a.pod_ssh, "dir": a.pod_dir, "host": a.pod_host, "repo": a.pod_repo, "python": a.pod_python} if a.pod_ssh and a.pod_dir else None
    s5 = step5(log, rec, part, core, inst, selected, pod=pod, approved_path=POC / "e2e" / "approved_guests.json")
    law = inst.sampler.law
    U = _certificate_or_error(part, str(law.q), str(law.s), int(inst.epoch.B_ex), str(getattr(inst.epoch, "delta", "1/1000000000000")))
    verdict = "ACCEPTED" if not log.failed else "REJECTED"
    out = {"schema": "veritor/demo-core/protocol-step5/v1", "utc": utc(), "record": str(rec), "driver": "e2e.verifier_v3.driver (LIVE, shipped) -> this step 5",
           "verdict_step5": verdict, "checks_failed": log.failed, "checks": len([r for r in log.rows if "check" in r]), "J": J, "F": F, "retained": retained,
           "step5": s5, "U": U, "timing": {"step5_s": round(time.perf_counter() - t0, 2)}, "label": LABEL,
           "prove_seconds": [c.get("prove_s") for c in s5.get("per_chunk", [])], "host_verify_seconds": [r.get("verify_seconds") for r in s5.get("host_verify", [])]}
    (rec / "protocol.step5.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + "\n")
    print(f"[demo_core.step5_live] {verdict}: J={J} F={F} failed={log.failed} {out['timing']} U={U.get('U_bits')}/{U.get('W_bits')}", flush=True)
    return 0 if verdict == "ACCEPTED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
