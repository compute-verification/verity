"""PROPOSED (INTEGRATOR D20 fallback; NOT the shipped ``verifier_v3/reverify.py``): independent re-verification of a DemoCore_v1 archive written
by the LIVE shipped driver (``e2e.verifier_v3.driver``) + ``e2e.demo_core.step5_live``.  Every check below calls the SHIPPED verifier function
by name and unchanged (``e2e.verifier_v3.steps``, ``e2e.partition_sample``, ``e2e.population_v3``, ``e2e.coverage_v1``, ``e2e.verifier_v3.expected``,
``veritor.interface``, the approved ``veritor-zk-host`` ``verify``); the record is READ, never rebuilt from the prover's values (the statements
the verifier expects come from the pinned instance + the committed id lists; the proofs' public values are re-verified under the approved vk
by the verifier's own ``verify`` on E10's pod because no SP1 host binary runs on this laptop).  Generic protocol demonstration; NOT vLLM inference.

    python -m e2e.demo_core.reverify_registered --record <archive> --out <dir outside the record> [--population <file>] --pod-ssh … --pod-dir …

Checks executed (shipped function in brackets): V8.instance_load [protocol_instance.load_instance], V2.population_derived
[population_v3.population_derived_check via steps.population_derived], V8.step2 [steps.step2_check_registered WITH the builder module's
input_check -- the driver's NOT_APPLICABLE:step2.v_in_equals_x is CHECKED here], V2.chronology [partition_sample.verify], V8.step4
[steps.step4_check], V3.interior_layout [steps.required_source_live + steps.interior_layout_check -> E2 validate_step4], V4.guest
[steps.guest_acceptance], V4.proofs_binding [veritor-zk-host verify + veritor.interface.check_binding], V5.trees
[verifier_v3.expected.committed_trees], V5.expected [coverage_v1.expected_statements + RecordCircuit], V5.positions / V5.coverage
[coverage_v1.check_coverage].  ``reverify.py`` checks NOT executed are printed at the end by name (Serve_v1-only: V3.weights_root against a
registered weights root, V4.elf_archived, V5.inputs_derived_only, token / vocabulary / served-copy checks, the L1 compare)."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
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
from e2e.verifier_v3 import steps  # noqa: E402

from . import adapter as AD, circuit as C, obligations as OB, record as RC  # noqa: E402
from .runner import Log, _certificate_or_error, _ids, _index_ids, host_verify_on_pod, sha256_file, utc  # noqa: E402

NOT_EXECUTED = [
    "V3.weights_root (Serve_v1: prover weights root == registered weight commitment; DemoCore_v1 has no weights -- the committed weights tree is empty)",
    "V4.elf_archived (the archived approved ELF is compared on the pod's host identity instead: V4.guest pins elf_sha256 + vk_hash)",
    "V5.inputs_derived_only (Serve_v1 rope cos/sin table)", "V8.step2 served == committed / vocabulary typing (no serving layer)",
    "V6.* token-copy checks, prompt-token pins (Serve_v1)", "compare.json against L1's verifier.log (no L1 run of this program)",
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="PROPOSED re-verification of a DemoCore_v1 live-driver archive (shipped step functions)")
    ap.add_argument("--record", required=True)
    ap.add_argument("--out", required=True, help="output dir, never inside the record")
    ap.add_argument("--population", default=None, help="population.compact.json (default <record>/instance/population.compact.json)")
    ap.add_argument("--instances", type=int, default=1)
    ap.add_argument("--approved", default=str(POC / "e2e" / "approved_guests.json"))
    ap.add_argument("--pod-ssh", default=None)
    ap.add_argument("--pod-dir", default=None)
    ap.add_argument("--pod-host", default="/workspace/bin/veritor-zk-host-cuda")
    a = ap.parse_args(argv)
    rec, out = Path(a.record).resolve(), Path(a.out).resolve()
    if out == rec or rec in out.parents:
        raise SystemExit("--out must not be inside the record")
    out.mkdir(parents=True, exist_ok=True)
    sel, chron = rec / "adapter" / "selective", rec / "chronology"
    log = Log(out / "reverify.log")
    t0 = time.perf_counter()
    inst_path = rec / "protocol_instance.json"
    pop_path = Path(a.population) if a.population else rec / "instance" / "population.compact.json"
    load_inst, _ = ps.loaders("contract")
    inst = load_inst(inst_path)
    inst_doc = json.loads(inst_path.read_text())
    log.check("V8.instance_load", True, "protocol_instance.json (the record's copy) loads under the contract format", who="SHIPPED protocol_instance.load_instance", instance_id=inst_doc.get("instance_id"))
    idig = ps.instance_digest(inst)  # the SHIPPED instance digest (partition_sample.instance_digest), the one r_q / the exemption request name
    proto = json.loads((rec / "protocol.json").read_text())
    log.check("V8.instance_digest", proto.get("instance_digest") == idig, "the driver's protocol.json names this instance's digest (partition_sample.instance_digest of the record's copy)", who="SHIPPED partition_sample.instance_digest", digest=idig[:16])
    # V2.population_derived
    pd = steps.population_derived(inst_doc, pop_path, instances=a.instances)
    log.check("V2.population_derived", pd.get("status") == "matches", "the population file is the rule's derivation from the pinned instance", who="SHIPPED population_v3.population_derived_check",
              status=pd.get("status"), problems=(pd.get("problems") or pd.get("disagreements") or [])[:3])
    # RU map
    pkind = "registered"
    request_ids, ru_request_of, ru_rule = steps.ru_ids_of(inst, pkind)
    ru_of_request = {rid: r for r, rid in enumerate(request_ids)}
    RC.set_request_mode("per-ru")
    RC.set_ru_base(str(sorted(set(ru_request_of.values()))[0]))
    # step 2 (with the builder module's input_check: the check the live driver named NOT_APPLICABLE)
    c0 = json.loads((sel / "commitments_phase0.json").read_text())
    registered = json.loads((rec / "instance" / "weights_authorized.json").read_text()) if (rec / "instance" / "weights_authorized.json").exists() else {"expected_root": None}
    params = ((inst_doc.get("program") or {}).get("identity") or {}).get("params") or {}
    ic = AD.input_check(rec, params)
    p2, na = steps.step2_check_registered(inst, c0, registered_weights_root=registered.get("expected_root"), ru_ids=request_ids, input_check=ic)
    log.check("V8.step2", not p2, "step-2 checks of a registered program (boundary tree per RU, no RU tree before r_q, weights root) + v_in(C) = x through the builder module's input_check",
              who="SHIPPED steps.step2_check_registered + e2e.demo_core.adapter.input_check", problems=p2[:3], not_applicable=[x["check"] for x in na])
    # chronology
    cproblems = ps.verify(inst_path, pop_path, chron, instances=a.instances, instance_format="contract")
    log.check("V2.chronology", not cproblems, "seed opens its commitment; r_q (incl. redraws), exemption request digest, r_s and selected.json re-derive", who="SHIPPED partition_sample.verify", problems=list(cproblems)[:3])
    events = json.loads((chron / "chronology.json").read_text()) if (chron / "chronology.json").exists() else {}
    tr_doc = json.loads((rec / "verify" / "transcript.json").read_text()) if (rec / "verify" / "transcript.json").exists() else {}
    # step 4
    c2 = json.loads((sel / "commitments.json").read_text())
    exreq = json.loads((sel / "exemption_request.json").read_text())
    J = [int(x) for x in proto["J"]]
    rq_digest = next((e.get("digest") for e in (proto.get("events") or []) if e.get("name") == "r_q"), None) or exreq.get("r_q")
    p4, F, committed = steps.step4_check(inst, J=J, r_q=str(rq_digest), instance_digest=idig, c0=c0, c2=c2, exreq=exreq, ru_of_request=ru_of_request,
                                         policy=(proto.get("step4") or {}).get("exemption_policy", {}).get("configured", "gamble"))
    log.check("V8.step4", not p4 and sorted(F) == sorted(int(x) for x in proto["F"]), "F ⊆ J, |F| <= B_ex, every retained RU's interior committed, phase-0 trees unchanged; F equals the driver's",
              who="SHIPPED steps.step4_check", problems=p4[:3], F=F, committed=committed)
    retained = [r for r in J if r not in set(F)]
    retained_rids = [request_ids[r] for r in retained]
    # V3.interior_layout
    pop_doc = json.loads(pop_path.read_text())
    try:
        src, label, facts = steps.required_source_live(inst_doc, pop_doc, sel, c2, retained_rids)
        il = steps.interior_layout_check(inst_doc, sel=sel, c2=c2, retained_request_ids=retained_rids, interior_in_play=True, in_play_reason="the instance pins commitments.interior_layout",
                                         required_source=src, required_label=label)
        log.check("V3.interior_layout", bool(il["ok"]), "pin present and == E2's layout_digest(); every REQUIRED interior region of every retained RU supplied (E2 validate_step4 with the verifier-built required layout)",
                  who="SHIPPED steps.interior_layout_check (E3 PositionOracle -> E2 InteriorLayout.build)", status=il.get("status"), problems=(il.get("problems") or [])[:3], trees=len(il.get("trees") or []))
    except Exception as e:  # noqa: BLE001
        log.check("V3.interior_layout", False, "the verifier could not validate the interior layout (fail-closed)", who="SHIPPED steps.required_source_live", error=f"{type(e).__name__}: {e}")
    # step 5: proofs + coverage (read from disk)
    from e2e.coverage_v1.coverage import AcceptedProof, check_coverage, expected_statements, obligations_of_typed_object
    from e2e.verifier_v3.expected import RecordCircuit, committed_trees
    from veritor.interface import TypedObligationSet, check_binding

    selected = json.loads((chron / "selected.json").read_text())
    Q = [int(r["vu_index"]) for r in selected["selected"]]
    ob = json.loads((rec / "obligations" / "obligations.json").read_text()) if (rec / "obligations" / "obligations.json").exists() else {"chunks": []}
    proofs = json.loads((rec / "proofs" / "proofs.json").read_text()) if (rec / "proofs" / "proofs.json").exists() else {}
    approved = json.loads(Path(a.approved).read_text())
    info = {"elf_sha256": (proofs.get("guest") or {}).get("elf_sha256"), "vk_hash": (proofs.get("guest") or {}).get("vk_hash")}
    v4g = steps.guest_acceptance(approved, info, pinned_elfs=inst.epoch.approved_elf_sha256, pinned_family=inst.epoch.family) if proofs else ["no proofs.json"]
    log.check("V4.guest", not v4g, "the proofs' guest (elf_sha256, vk_hash) is approved AND pinned by the epoch", who="SHIPPED steps.guest_acceptance", elf=str(info["elf_sha256"])[:12], problems=v4g)
    hv: list[dict[str, Any]] = []
    if a.pod_ssh and a.pod_dir:
        line_ = [ln for ln in Path(a.pod_ssh).read_text().splitlines() if ln.strip().startswith("exec ssh")][0].split()
        rs, target = ["rsync", "-az", "-e", " ".join(line_[1:-2])], line_[-2]
        subprocess.run([a.pod_ssh, f"mkdir -p {a.pod_dir}"], check=True, capture_output=True)
        for sub in ("obligations", "proofs"):  # the verifier's copy of the archive's proofs + statements goes to the host
            subprocess.run(rs + [str(rec / sub) + "/", f"{target}:{a.pod_dir}/{sub}/"], check=True, capture_output=True)
        hv = host_verify_on_pod(rec, ssh_wrapper=a.pod_ssh, remote_dir=a.pod_dir, host=a.pod_host)
    (out / "host_verify.jsonl").write_text("".join(json.dumps(r) + "\n" for r in hv))
    accepted, per_chunk = [], []
    for ch in ob.get("chunks", []):
        name = ch["chunk"]
        cdir = rec / "obligations" / name
        composed = json.loads((cdir / "composed.json").read_text())
        obj = TypedObligationSet.from_json(composed["object"])
        pr = next((r for r in proofs.get("results", []) if r["chunk"] == name), None)
        v = next((r for r in hv if r["chunk"] == name), None)
        pv_hex = (v or {}).get("public_values") or ((pr or {}).get("verify") or {}).get("public_values")
        ok_host = bool(v and v.get("ok") and v.get("statement_match") and v.get("verdict") and not v.get("unsound") and str(v.get("vk_hash")).lower() == str(info["vk_hash"]).lower())
        bnd = check_binding(obj, bytes.fromhex(pv_hex), statement_format=3) if pv_hex else None
        st_ok = pr is not None and pr.get("statement_sha256") == sha256_file(cdir / "statement.bin") and pr.get("h_O") == composed["digest"]
        ok = ok_host and bnd is not None and bnd.ok and st_ok and str((pr or {}).get("status", "")).startswith("PROVEN")
        per_chunk.append({"chunk": name, "host_verify_ok": ok_host, "binding_ok": bool(bnd and bnd.ok), "statement_bound": st_ok, "prover_status": (pr or {}).get("status"), "accepted": ok,
                          "verify_s": (v or {}).get("verify_seconds"), "prove_s": ((pr or {}).get("prove") or {}).get("prove_seconds")})
        if ok:
            accepted.append(AcceptedProof(name, obligations_of_typed_object(obj), typed_object=obj))
    log.check("V4.proofs_binding", bool(per_chunk) and all(c["accepted"] for c in per_chunk) and bool(hv),
              "every proof object: SP1 proof verifies under the approved vk (verifier's own `verify` on the pod), verdict true, not unsound, public values bind the statement (check_binding), statement digest == the prover's",
              who="SHIPPED veritor-zk-host verify + veritor.interface.check_binding", chunks=per_chunk, host_verify_rows=len(hv))
    # expected statements (verifier side: circuit + committed id lists), coverage
    prog, d = C.build(int(params["N"]), int(params["R"]))
    part = C.partition(int(params["N"]), int(params["R"]), prog, d)
    from .prover import load_fixture  # positions only: the expectation reads gate positions, never the prover's values

    class _Cache:
        def tree(self, tid: str):
            return type("T", (), {"ids": [] if tid == "weights" else _index_ids(sel / tid)})()

    # the position oracle for the expectation: a CoreRecord over the fixture values is used ONLY for gate -> (tree, leaf, offset) positions
    part_f, vals_f, _ = load_fixture(rec / "fixture")
    core = RC.CoreRecord(part_f, vals_f, out / "positions-scratch", interior="e2" if any(t.startswith("interior/") for t in c2["trees"]) else "ru")
    core.commit_phase0(); core.commit_interior(retained)
    shutil.rmtree(out / "positions-scratch", ignore_errors=True)
    retained_trees = {core.interior_tree(r) for r in retained}
    tree_ids = ["weights"] + [t for t in c2["trees"] if t.startswith("boundary/")] + [t for t in c2["trees"] if t in retained_trees]
    ct = committed_trees(sel, c2, tree_ids, _Cache())
    log.check("V5.trees", not ct.problems, "every committed tree a statement may reference: recomputed ids_digest == committed, id count == leaf_count", who="SHIPPED verifier_v3.expected.committed_trees", trees=len(ct.trees), problems=ct.problems[:3])

    class _Part:
        def locate(self, vu_id: str):
            return OB.expectation_of(part.vus[int(vu_id)], core)

    circuit = RecordCircuit(inst.epoch.family, {v.vu_index: v.kind for v in part.vus})
    try:
        expected = expected_statements(circuit, _Part(), [str(i) for i in Q]); exp_err = None
    except Exception as e:  # noqa: BLE001
        expected, exp_err = None, f"{type(e).__name__}: {e}"
    log.check("V5.expected", expected is not None and len(expected) == len(Q), "the verifier derives the expected statement of every selected VU", who="SHIPPED coverage_v1.expected_statements + RecordCircuit", expected=len(expected or {}), error=exp_err)
    report = check_coverage(expected or {}, accepted, trees=ct.trees, rank_of=ct.rank_of()) if expected else None
    (out / "coverage.json").write_text(json.dumps(report.to_json() if report else {"error": exp_err}, indent=1) + "\n")
    log.check("V5.positions", report is not None and report.positions_checked and not report.position_problems, "every accepted proof binds exactly the positions the verifier derives", who="SHIPPED check_expected_positions via check_coverage", problems=(report.position_problems if report else None))
    log.check("V5.coverage", report is not None and report.ok, "every expected statement covered exactly once by an accepted proof; no duplicate, no unexpected obligation", who="SHIPPED coverage_v1.check_coverage",
              expected=len(expected or {}), covered=len(report.covered) if report else 0, uncovered=list(report.uncovered)[:8] if report else None, proofs=len(accepted))
    covered_ids = sorted(report.covered, key=int) if report else []
    uncovered_ids = sorted(report.uncovered, key=int) if report else [str(i) for i in Q]
    line = (f"sampled VUs = {len(Q)}; covered by a verified expected statement = {len(covered_ids)} (identities: {_ids(covered_ids)}); "
            f"proof objects = {len(accepted)}; uncovered = {len(uncovered_ids)} (identities: {_ids(uncovered_ids)})")
    log.note(line)
    law = inst.sampler.law
    U = _certificate_or_error(part, str(law.q), str(law.s), int(inst.epoch.B_ex), "1/1000000000000")
    verdict = "ACCEPTED" if not log.failed else "REJECTED"
    for x in NOT_EXECUTED:
        log.note("NOT EXECUTED (reverify.py check without a registered-program meaning here): " + x)
    doc = {"schema": "veritor/demo-core/reverify-registered/v1", "status": "PROPOSED (D20 fallback; not the shipped reverify.py)", "utc": utc(), "record": str(rec), "verdict": verdict,
           "checks_failed": log.failed, "checks": [r["check"] for r in log.rows if "check" in r], "not_executed": NOT_EXECUTED, "coverage_line": line, "U": U,
           "J": J, "F": F, "Q": len(Q), "proofs": len(accepted), "per_chunk": per_chunk, "seconds": round(time.perf_counter() - t0, 2)}
    (out / "reverify.json").write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")
    print(f"[demo_core.reverify_registered] {verdict} (PROPOSED re-verification): failed={log.failed} Q={len(Q)} proofs={len(accepted)} {doc['seconds']} s")
    return 0 if verdict == "ACCEPTED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
