#!/usr/bin/env python
"""[A9 contract 2026-09-10] Step 5 -- the verifier's outcome for the run and the certificate for exactly this configuration.

Three shapes of record, all reported honestly:

* **refused before r_s** (no ``protocol.json``; the live driver wrote ``protocol_events.json`` with a ``step2_REFUSED`` / ``step4_REFUSED`` event,
  e.g. ``|F| = |J| > B_ex`` under ``exempt-unsupported``): outcome ``REFUSED step N`` with the problems, J, F, the per-RU fields from
  ``adapter/selective/exemption_request.json``; no certificate (A3's ``certificate_inputs`` refuses a non-VERIFIED record; nothing is built);
* **drawn Q, then verified**: A3's ``e2e.verifier_v3.reverify`` composed by import (V1-V8: pins, chronology re-derivation, commitments, approved
  guest + host verify of every proof + binding, expected statements with A2's ``check_expected_positions``, V5.uncovered = a selected VU of a kind
  without a checker in a retained RU FAILS, R17/R18 fields, WR-6: population / program / output domain compared against the INSTANCE pins,
  V1.program_rebuilt in the verifier's own process, V5.located / V5.prover_mapping) with the v5 loaders installed; then A3's
  ``certificate_inputs`` -> A7's ``certificate_v2.build_certificate`` with the v5 population (``legacy-minus-inputs``) and the eight cost
  classes from ``costlog_v5.json``;
* **Q = ∅** (every retained RU exempted, or J = ∅): the same path; the certificate carries |Q| = 0 and U = W is its expected value (A7).

Outputs under ``REC/verify/``: ``reverify.log``, ``record_v3.json`` (A3), ``certificate_v2.json`` + ``certificate_v2.txt`` (A7) when built, and
``verify_v5.json`` (this module's summary: outcome, step, counts, unsupported selections, completion risk, substitutions made).

    python -m record_v5.verify_v5 --config run.json --record REC --population population.v5.json [--host SP1_HOST] [--costlog REC/costlog_v5.json]
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

from record_v5 import POC, deps, freeze
from record_v5 import config as cfgmod
from record_v5 import instance_v5 as i5
from record_v5 import population_v5 as p5

deps.install()

from e2e import certificate_v2 as cv2  # noqa: E402
from e2e import partition_sample as ps  # noqa: E402

_ORIG_LOAD_INSTANCE = ps.load_instance


def install_loaders() -> None:
    """Point A3's sampler / driver at the v5 population and instance loaders -- for this process, from ``main`` only (A7 finding
    22:05Z: a module import must not install anything process-wide; tests wrap what they need in ``loaders_installed()``)."""

    p5.install()
    ps.load_instance = i5.load_instance_v5


@contextlib.contextmanager
def loaders_installed():
    prev = ps.load_instance
    install_loaders()
    try:
        yield
    finally:
        p5.uninstall()
        ps.load_instance = prev


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# [Integrator 20:46Z §3] a verifier_v3 acceptance is not conformance evidence until A3/A8's R1/R9 integration regressions pass
PENDING_R1_R9 = "pending R1/R9 regressions: a verifier_v3 VERIFIED is not conformance evidence until A3/A8's R1/R9 integration regressions pass"


def refused_outcome(rec: Path) -> dict[str, Any] | None:
    ev_path = rec / "protocol_events.json"
    if (rec / "protocol.json").exists():
        return None
    events = json.loads(ev_path.read_text()) if ev_path.exists() else []  # the driver may die before its first event (e.g. no production record)
    ref = next((e for e in events if str(e.get("event", "")).endswith("_REFUSED")), None)
    ex_p = rec / "protocol_exit.json"
    ex = json.loads(ex_p.read_text()) if ex_p.exists() else {"kind": "UNAVAILABLE (no protocol_exit.json: the driver was not run through protocol_v5)", "reason": None, "rc": None}
    vx_p = rec / "verifier_exit.json"  # A3 ckpt 5 (W-A9-4): the driver's own machine-readable exit
    vx = json.loads(vx_p.read_text()) if vx_p.exists() else None
    exreq_p = rec / "adapter" / "selective" / "exemption_request.json"
    exreq = json.loads(exreq_p.read_text()) if exreq_p.exists() else None
    rq = next((e for e in events if e.get("event") == "r_q"), None)
    step = {"step2_REFUSED": 2, "step4_REFUSED": 4}.get(ref["event"] if ref else "", None)
    verifier_decision = ref is not None
    return {"status": "REFUSED" if verifier_decision else "NOT VERIFIED (no verifier decision)", "step": step, "exit_kind": ex["kind"], "exit_rc": ex["rc"], "exit_reason": ex["reason"],
            "verifier_exit": vx,
            "who": "verifier (step check refused the prover's record before the next draw)" if verifier_decision else
                   "not the verifier: the prover aborted / exited, the driver timed out or errored -- no acceptance and no rejection was decided",
            "event": ref, "problems": (ref or {}).get("problems"), "J": (rq or {}).get("J"), "F": (exreq or {}).get("F"),
            "budget_exceeded": (exreq or {}).get("budget_exceeded"), "B_ex": (exreq or {}).get("B_ex"), "f_policy": (exreq or {}).get("f_policy"),
            "per_ru": (exreq or {}).get("per_ru"), "Q": None,
            "meaning": {2: "the step-2 I/O checks failed: v_in(C) / v_out(C) / typing", 4: "rejected at step 4: F ⊄ J, |F| > B_ex (RU count), or a retained RU without committed interiors / disagreeing output copies",
                        None: "no step refused: see exit_kind (prover abort / prover exit / timeout / driver error); costs of stages never reached are UNAVAILABLE"}.get(step, "aborted"),
            "events": events}


INSTANCE_FORMAT = "record_v5"  # A3 ckpt 9: reverify's loader is selected by NAME on its own command line, not by this process's monkeypatch


def reverify_argv(rec: Path, pop_path: Path, host: str, rv_out: Path, extra: str = "") -> list[str]:
    """A9 W-A9-14 (attempt 6): ``verify_v5`` itself passes ``--instance-format record_v5`` to ``e2e.verifier_v3.reverify`` -- the loader
    selection is pinned in the invocation (and in reverify's ``record_v3.json#instance_format``), not left to ``--reverify-args``.  An
    explicit ``--instance-format`` in ``extra`` wins (so a caller can pin another registered format on purpose)."""
    extra_args = [x for x in extra.split() if x]
    argv = ["--record", str(rec), "--population", str(pop_path), "--expressibility", str(rec / "expressibility_v5.json"), "--host", host, "--out", str(rv_out)]
    if not any(x == "--instance-format" or x.startswith("--instance-format=") for x in extra_args):
        argv += ["--instance-format", INSTANCE_FORMAT]
    # [E10 D40 T-1, 2026-09-12] the verifier's OWN copy of the pinned checkpoint (`VERITOR_CHECKPOINT=<dir with model.safetensors>`, sha256-checked by
    # reverify against manifests/checkpoints.json -> `V6.checkpoint`): the weights word V6.gather / V6.inputs read.  Without it a Gather VU whose
    # weights leaf is not shipped is GATHER_UNRESOLVED by name (n2c 09:38Z).  An explicit --checkpoint in `extra` wins.
    ckpt = os.environ.get("VERITOR_CHECKPOINT", "")
    if ckpt and not any(x == "--checkpoint" or x.startswith("--checkpoint=") for x in extra_args):
        argv += ["--checkpoint", ckpt]
    # [E10 D40 T-1] VERITOR_REVERIFY_ELF=<path>: the archived APPROVED guest bytes for V4.elf_archived (reverify compares an explicit --elf as given, by
    # sha256 against the host's embedded guest; its default search over fixtures/results/GUEST-*/ finds only 9fd00e60 -- n2c's live verify 09:34Z FAILED
    # V4.elf_archived by name under guest 666e937f).  An explicit --elf in `extra` wins.
    elf = os.environ.get("VERITOR_REVERIFY_ELF", "")
    if elf and not any(x == "--elf" or x.startswith("--elf=") for x in extra_args):
        argv += ["--elf", elf]
    return argv + extra_args  # --out-bits-per-ru / --family are cross-checks only; the instance pins both (WR-6)


def certificate_or_refusal(cert: dict[str, Any], vdir: Path) -> dict[str, Any]:
    """A9 W-A9-15 (attempt 6): ``certificate_from_record_v3`` returns a REFUSED document (schema ``.../refused``, no ``bound``) for a
    record whose status is not VERIFIED; rendering it with ``render_text`` raised ``KeyError: 'bound'``.  A refused record gets the
    refusal document written and a NAMED status instead."""
    if cert.get("schema") == cv2.REFUSED_SCHEMA or cert.get("status") == "REFUSED" or "bound" not in cert:
        (vdir / "certificate_v2.refused.json").write_text(cv2.dumps(cert) + "\n")
        (vdir / "certificate_v2.refused.txt").write_text(cv2.render_refusal_text(cert) + "\n")
        return {"built": False, "status": "REFUSED_RUN_NO_CERTIFICATE", "reason": cert.get("reason") or "record status is not VERIFIED: no certificate is built for a refused run",
                "path": str(vdir / "certificate_v2.refused.json"), "same_population_and_budget_status": cert.get("same_population_and_budget_status"),
                "completion_risk": cert.get("completion_risk")}
    (vdir / "certificate_v2.json").write_text(cv2.dumps(cert) + "\n")
    (vdir / "certificate_v2.txt").write_text(cv2.render_text(cert) + "\n")
    return {"built": True, "status": "CERTIFICATE_BUILT", "path": str(vdir / "certificate_v2.json"), "U_bits": (cert.get("bound") or {}).get("U_bits") or (cert.get("result") or {}).get("U_bits"),
            "completion_risk": cert.get("completion_risk"),
            "note": "U = W is the expected value for a 4-request epoch at delta = 1e-12 (A7's structural finding: vacuous -- sampling rates too low for the domain) and is reported as such"}


# costlog_v5's short class keys -> certificate_v2.COST_CLASSES (the contract's names). [E1 2026-09-14 06:20Z GO] Without this map, a record that
# already carries a costlog_v5.json when verify_v5 runs (a resumed run after a pod change, or ANY re-verification of an archive, e.g. T1's)
# hits CertificateError E_COST_CLASS and the certificate is not built although the verdict is VERIFIED (256x64 resume pod fhcjdqf411heaz).
# The first verification never saw it because costlog_v5.json is written AFTER the certificate. "value_production" (E10's 9th class, the
# interpreter's interior values on other hardware) is not a contract class and is carried in the costlog only.
COSTLOG_TO_CONTRACT: dict[str, str] = {
    "native": "native_or_trusted_replay_baseline", "execution": "original_execution_and_collection",
    "commit_io": "initial_commitments_and_io_checks", "replay": "replay_including_attempted_work_for_later_exempted_RUs",
    "interior_commit": "interior_commitments", "prove": "proof_generation",
    "verifier_expected": "verifier_derivation_of_expected_statements", "verify_proofs": "proof_verification",
}


def costs_from_costlog(path: Path | None) -> dict[str, cv2.Cost]:
    if not path or not Path(path).exists():
        return {}
    doc = json.loads(Path(path).read_text())
    out: dict[str, cv2.Cost] = {}
    for cls, row in (doc.get("classes") or {}).items():
        contract_cls = COSTLOG_TO_CONTRACT.get(cls)
        if contract_cls is None or row.get("seconds") is None or row.get("label") == "UNAVAILABLE":
            continue
        out[contract_cls] = cv2.Cost(float(row["seconds"]), row.get("provenance", f"costlog_v5.json classes.{cls}"), row.get("label", "MEASURED"))
    return out


def main(argv: list[str] | None = None) -> int:
    install_loaders()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--record", required=True)
    ap.add_argument("--population", required=True)
    ap.add_argument("--host", default=None, help="SP1 host binary (A3 reverify V4); without it V4 is UNAVAILABLE and reverify is not run")
    ap.add_argument("--costlog", default=None, help="REC/costlog_v5.json (record_v5.costlog_v5) for the certificate's cost rows")
    ap.add_argument("--out-bits-per-ru", type=int, default=None, help="declared output domain per RU (default (STEPS+1) x 16)")
    ap.add_argument("--reverify-args", default="", help="extra args for e2e.verifier_v3.reverify (e.g. --info-from-ready)")
    a = ap.parse_args(argv)
    cfg = cfgmod.load(a.config)
    rec = Path(a.record)
    freeze.check(rec, "verify")
    vdir = rec / "verify"
    vdir.mkdir(exist_ok=True)
    pop_path = Path(a.population)
    summary: dict[str, Any] = {"schema": "veritor/a9/verify-v5/v1", "utc": utc(), "run_id": cfg.run_id, "f_policy": cfg.f_policy, "epoch": cfg.epoch.as_json(),
                               "randomness_mode": cfg.randomness_mode, "substitutions": [], "certificate": None,
                               "labels": [x for x in (cfg.record_label,) if x] + [PENDING_R1_R9]}
    t0 = time.perf_counter()

    refused = refused_outcome(rec)
    if refused is not None:
        summary.update(outcome=refused, status=refused["status"], certificate={"built": False, "reason": f"refused at step {refused['step']}: no certificate is built for a refused run (A3 certificate_inputs)"})
        (vdir / "verify_v5.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
        print(f"[verify_v5] {refused['status']} ({refused['exit_kind']}) step={refused['step']}: {refused['problems'] or refused['exit_reason']}; J={refused['J']} F={refused['F']} B_ex={refused['B_ex']} policy={refused['f_policy']}")
        return 0

    # --- A3's post-hoc verifier by import, with the population pin set to this run's population ---
    from e2e.verifier_v3 import reverify as rv

    # WR-6 (A3 ckpt 4): V1.partition compares against the instance's partition pin; nothing is substituted in the verifier (W-A9-1 closed)
    rv_out = vdir / "reverify_v3"
    rv_status: dict[str, Any] = {"ran": False}
    if a.host:
        rv_argv = reverify_argv(rec, pop_path, a.host, rv_out, a.reverify_args)
        t1 = time.perf_counter()
        try:
            rc = rv.main(rv_argv)
        except SystemExit as e:  # noqa: PERF203
            rc = int(e.code) if isinstance(e.code, int) else 1
        rv_status = {"ran": True, "rc": rc, "seconds": round(time.perf_counter() - t1, 2), "argv": rv_argv, "log": str(rv_out / "reverify.log")}
    else:
        rv_status = {"ran": False, "reason": "UNAVAILABLE: no SP1 host binary given (--host); V4 proof verification and the V-checks were not run"}
    summary["reverify_v3"] = rv_status
    record_v3 = json.loads((rv_out / "record_v3.json").read_text()) if (rv_out / "record_v3.json").exists() else None
    summary["record_v3"] = {k: record_v3.get(k) for k in ("status", "status_line", "epoch", "B_ex", "sampled", "checks_failed", "completion_risk")} if record_v3 else None
    # --- the runner's own view of unsupported selections (obligations_v5) ---
    us = rec / "obligations" / "unsupported_selected.json"
    summary["unsupported_selected"] = json.loads(us.read_text()) if us.exists() else None
    rf = rec / "verify" / "ru_fields.json"
    summary["ru_fields"] = json.loads(rf.read_text()) if rf.exists() else None
    ocp = rec / "verify" / "output_copy_checks.json"
    summary["output_copy_checks"] = {k: v for k, v in json.loads(ocp.read_text()).items() if k != "rows"} if ocp.exists() else "in A3's reverify (V8.output_copies) if present"

    # --- the certificate for exactly this configuration ---
    cert_info: dict[str, Any] = {"built": False}
    if record_v3 is not None:
        from e2e.verifier_v3 import certificate_inputs as ci

        try:
            # A7 (mainline 12c6456d): the runner's ONE call -- record_v3.json + exactly the population file the sampler ran on (pinned: sha256,
            # partition digest, N_R, stage-2 candidates, B_ex; E_POPULATION_MISMATCH otherwise); costs from costlog_v5; R18 companion inside
            cert = cv2.certificate_from_record_v3(rv_out / "record_v3.json", pop_path, costs=costs_from_costlog(Path(a.costlog) if a.costlog else rec / "costlog_v5.json"))
            cert["run"] = {"f_policy": cfg.f_policy, "randomness_mode": cfg.randomness_mode, "run_id": cfg.run_id, "workload": cfg.workload,
                           "population_construction": p5.CONSTRUCTION, "labels": summary["labels"]}
            cert_info = certificate_or_refusal(cert, vdir)
        except (ci.RecordRefused, cv2.CertificateError, cv2.PopulationError, KeyError, TypeError, ValueError) as e:
            cert_info = {"built": False, "status": "CERTIFICATE_BUILDER_ERROR", "reason": f"{type(e).__name__}: {e}"}
    else:
        cert_info = {"built": False, "reason": "no record_v3.json (A3's reverify did not run or did not finish)"}
    summary["certificate"] = cert_info
    summary["status"] = (record_v3 or {}).get("status") or ("UNAVAILABLE" if not rv_status.get("ran") else "NOT VERIFIED")
    summary["seconds"] = round(time.perf_counter() - t0, 2)
    (vdir / "verify_v5.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    # cost rows from reverify's own stage timings (timings.json#stages_s), so class 8 is the proof check alone and the verifier's
    # expected-statement derivation lands in class 7 (the TINY record of 21:34Z booked the whole 130 s reverify wall as class 8)
    tj = rv_out / "timings.json"
    stages = (json.loads(tj.read_text()).get("stages_s") or {}) if tj.exists() else {}
    with (rec / "costlog.jsonl").open("a") as f:
        if stages:
            v4 = sum(v for k, v in stages.items() if k.startswith("V4.") or k.startswith("D6.vk"))
            v5e = float(stages.get("V5.expected_statements", 0.0))
            rest = sum(v for k, v in stages.items() if k != "total_wall" and not (k.startswith("V4.") or k.startswith("D6.vk") or k == "V5.expected_statements"))
            f.write(json.dumps({"stage": "verifier: proof verification (reverify V4.*: vk derivation once + host verify + binding of every proof)", "class": "verify_proofs",
                                "seconds": round(v4, 3), "label": "MEASURED", "batch": "all chunks", "note": f"timings.json stages_s V4.* + D6.vk_once; vk derivation {stages.get('V4.vk_derivation_once')} s is paid once per verifier process"}) + "\n")
            f.write(json.dumps({"stage": "verifier: expected statements of the selected VUs (reverify V5.expected_statements)", "class": "verifier_expected",
                                "seconds": round(v5e, 3), "label": "MEASURED", "batch": "Q_proof", "note": "the verifier's own derivation (A2 expectations) inside reverify; obligations_v5's locate is booked separately"}) + "\n")
            f.write(json.dumps({"stage": "verifier: record checks (reverify V1-V3, V5 location/coverage, V6-V8, D6 request commitment)", "class": "verifier",
                                "seconds": round(rest, 3), "label": "MEASURED", "batch": "record", "note": "verifier work outside classes 7-8 (reported as other_verifier_s, not a contract class)"}) + "\n")
        else:
            f.write(json.dumps({"stage": "verifier: post-hoc re-verification (A3 reverify V1-V8 incl. host verify of every proof) + certificate_v2", "class": "verify_proofs",
                                "seconds": rv_status.get("seconds"), "label": "MEASURED" if rv_status.get("ran") else "UNAVAILABLE", "batch": "all chunks",
                                "note": "no timings.json from reverify: the whole reverify wall (class 8 + record checks) booked here"}) + "\n")
    print(f"[verify_v5] status {summary['status']}; reverify {rv_status.get('rc')}; certificate {cert_info}; -> {vdir / 'verify_v5.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
