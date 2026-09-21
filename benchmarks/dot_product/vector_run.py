#!/usr/bin/env python3
"""One recorded ``typed-obligation/v0`` vector through one backend, reported in the ``research`` result contract.

The vectors under ``fixtures/typed-obligation-v0`` are the exact dot-product instances the campaign starts from:
a ``GemmCoordinate<K>`` in ``veritor.tc-ampere-bf16`` over committed words, with the public values a correct
prover produced for them recorded in their ``binding`` column.  This workload re-establishes one of them through
a backend and reports what it observed:

    reference     the Python column: validate, lower, run the transparent reference checker on the witness
    sp1-execute   the SP1 guest in the executor (no proof): verdict, public values, exact cycles per phase
    sp1-prove     one SP1 proof (core or compressed), then the verifier's acceptance against the object

It writes ``$RESEARCH_RUN_DIR/result.json`` (or ``--out``) with the workload fingerprint -- what actually ran:
the vector's object and statement digests, the family, the backend and mode, the observed guest identity -- the
validation verdict, named measurements with units, and the artifacts worth preserving (the proof, the host's raw
documents).  ``research`` records and ships; this file decides what the run means.

Every result carries two labels so numbers are never compared across classes by accident:

    proof_class     ARITHMETIC_DIAGNOSTIC    a guest with authentication skipped (an ``unsound-*`` build): cost of
                                             the relation and statement machinery alone; never a result
                    NON_ZK_PROOF             SP1 core/compressed: a sound proof that is not zero-knowledge
                    (COMPLETE_ZK_BACKEND is reserved for backends that establish privacy; SP1 core is not one)
    authentication  included | excluded      whether the words were authenticated against the object's roots

``--diagnostic`` is required to run an unsound host: the guest identity then need not be the approved one, the
proof is verified under the guest's *own* key, and the result says so.  Without it, a diagnostic guest fails.

    uv run python benchmarks/dot_product/vector_run.py --vector accept--packed-word-row --backend sp1-prove --out /tmp/k16
    uv run research run --project verity --stage sp1.prove.k16 --require-result -- \\
        uv run python benchmarks/dot_product/vector_run.py --vector accept--packed-word-row --backend sp1-prove

The script runs from a plain checkout: when ``verity`` / ``verity_sp1`` are not installed it adds
``packages/verity/src`` and ``backends/sp1/python`` to ``sys.path``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "fixtures" / "typed-obligation-v0"
for candidate, module in ((REPO / "packages" / "verity" / "src", "verity"), (REPO / "backends" / "sp1" / "python", "verity_sp1")):
    try:
        __import__(module)
    except ImportError:
        sys.path.insert(0, str(candidate))

from verity.verification.binding import bound_public_values  # noqa: E402
from verity.verification.conformance import interpret  # noqa: E402

RESULT_SCHEMA = "research/result/v0.1"
BACKENDS = ("reference", "sp1-execute", "sp1-prove")


def source_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def measurement(name: str, value: float | int, unit: str) -> dict:
    return {"name": name, "value": value, "unit": unit}


def run_reference(rec: dict, out: Path) -> tuple[dict, list[dict], list[str], list[str]]:
    started = time.perf_counter()
    got = interpret(rec)
    wall = time.perf_counter() - started
    problems = []
    if not (got.canonical_ok and got.digest_ok):
        problems.append("canonical bytes or object digest differ from the record")
    if got.verdict != rec["verdict"] or (rec["verdict"] == "reject" and got.code != rec.get("code")):
        problems.append(f"verdict {got.verdict}/{got.code} differs from the record's {rec['verdict']}/{rec.get('code')}")
    if rec["verdict"] == "accept":
        if got.statement != rec["statement"] or got.statement_digest != rec["statement_digest"]:
            problems.append("the lowering's statement bytes differ from the record")
        if got.bound_public_values != rec["binding"]["object_digest"] + rec["binding"]["statement_digest"] + "01":
            problems.append("the bound public values differ from the record")
        if "witness" in rec and got.witness_verdict != rec["witness_verdict"]:
            problems.append(f"transparent checker said {got.witness_verdict}, the record {rec['witness_verdict']}")
        if "values" in rec and got.relations != rec["relations"]:
            problems.append(f"relation check said {got.relations}, the record {rec['relations']}")
    validation = {
        "status": "passed" if not problems else "failed",
        "checker": "verity.verification.conformance.interpret + TransparentBackend",
        "detail": "; ".join(problems) or f"every recorded field re-derived; transparent checker: {got.witness_verdict}",
        "evidence": [],
    }
    return validation, [measurement("reference_check_wall", wall, "s")], [], []


def run_sp1(rec: dict, out: Path, *, prove: bool, mode: str, want_info: bool, diagnostic: bool) -> tuple[dict, list[dict], list[str], list[str], dict]:
    from verity_sp1.host import APPROVED, Host
    from verity_sp1.typed import execute_object, prove_object, verify_object

    host = Host(timeout=None)
    from dataclasses import asdict

    identity = {"expected": asdict(APPROVED) | {"guest_features": list(APPROVED.guest_features)}}
    observed = None
    if want_info or diagnostic:
        started = time.perf_counter()
        observed = host.info()
        identity["observed"] = asdict(observed) | {"guest_features": list(observed.guest_features)}
        identity["info_wall_s"] = time.perf_counter() - started
        identity["approved"] = observed == APPROVED
    if diagnostic:
        # an unsound build is the point; verify under its own key and label the result, never accept it as sound
        if observed is None or not observed.guest_features:
            raise SystemExit("--diagnostic given but the host's guest is not a diagnostic (unsound-*) build")
        identity["diagnostic_features"] = list(observed.guest_features)
        trusted = observed
    else:
        trusted = APPROVED
    obj, witness = rec["object"], bytes.fromhex(rec["witness"])
    expected_pv = bound_public_values(obj).hex()
    documents: dict = {}
    measurements: list[dict] = []
    problems: list[str] = []
    artifacts: list[str] = []

    if not prove:
        started = time.perf_counter()
        report, binding = execute_object(host, obj, witness)
        wall = time.perf_counter() - started
        documents["execute"] = dict(report.document)
        measurements += [measurement("execute_wall", wall, "s"), measurement("total_cycles", report.total_cycles, "cycles")]
        measurements += [measurement(f"cycles.{k}", v, "cycles") for k, v in sorted(report.cycle_tracker.items())]
        if not report.verdict:
            problems.append("the guest rejected the witness")
        if report.unsound and not diagnostic:
            problems.append("the guest is a diagnostic build (run with --diagnostic if that is intended)")
        if report.public_values.hex() != expected_pv:
            problems.append("public values are not h_O || h_L || 01 of this object")
        if report.public_values.hex() != rec["binding"]["public_values"]:
            problems.append("public values differ from the ones recorded for this vector")
        if not (binding.ok and binding.verdict == 1):
            problems.append(f"binding check: {binding.code} {binding.detail}")
        checker = "verity_sp1 execute + verity.verification.binding.check_binding"
    else:
        proof = out / "proof.bin"
        started = time.perf_counter()
        report = prove_object(host, obj, witness, proof, mode=mode)
        prove_wall = time.perf_counter() - started
        started = time.perf_counter()
        acceptance = verify_object(host, obj, proof, approved=trusted)
        verify_wall = time.perf_counter() - started
        documents["prove"] = dict(report.document)
        documents["verify"] = dict(acceptance.verify.document)
        documents["acceptance"] = {
            "accepted": acceptance.accepted, "proof_verifies": acceptance.proof_verifies, "approved_key": acceptance.approved_key,
            "binding_ok": acceptance.binding.ok, "binding_code": None if acceptance.binding.code is None else str(acceptance.binding.code),
            "verdict": acceptance.binding.verdict, "sound": acceptance.sound, "reasons": acceptance.reasons(),
        }
        measurements += [
            measurement("prove_wall", prove_wall, "s"), measurement("setup_seconds", report.setup_seconds, "s"),
            measurement("prove_seconds", report.prove_seconds, "s"), measurement("host_verify_seconds", report.verify_seconds, "s"),
            measurement("verify_wall", verify_wall, "s"), measurement("proof_bytes", report.proof_bytes, "B"),
            measurement("shards", report.shards, "count"), measurement("total_cycles", report.execution.total_cycles, "cycles"),
        ]
        measurements += [measurement(f"cycles.{k}", v, "cycles") for k, v in sorted(report.execution.cycle_tracker.items())]
        if report.execution.public_values.hex() != rec["binding"]["public_values"]:
            problems.append("public values differ from the ones recorded for this vector")
        if diagnostic:
            # everything but soundness: the proof verifies under the diagnostic guest's key and binds the object
            if not (acceptance.proof_verifies and acceptance.approved_key and acceptance.binding.ok and acceptance.binding.verdict == 1):
                problems += [r for r in acceptance.reasons() if "diagnostic" not in r]
            checker = "DIAGNOSTIC: verity_sp1 prove + verify_object under the unsound guest's own key (binding, verdict); not a sound proof"
        else:
            if not acceptance.accepted:
                problems += acceptance.reasons()
            checker = "verity_sp1 prove + verify_object (approved key, binding, verdict)"
        artifacts.append("proof.bin")
    (out / "host.json").write_text(json.dumps(documents, indent=1, sort_keys=True) + "\n")
    artifacts.append("host.json")
    validation = {"status": "passed" if not problems else "failed", "checker": checker, "detail": "; ".join(problems) or "accepted", "evidence": ["host.json"]}
    return validation, measurements, artifacts, [], identity


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vector", required=True, help="vector name under fixtures/typed-obligation-v0 (without .json)")
    ap.add_argument("--backend", choices=BACKENDS, required=True)
    ap.add_argument("--mode", default="core", choices=("core", "compressed"), help="sp1-prove: proof mode")
    ap.add_argument("--no-info", action="store_true", help="sp1: do not observe the guest identity (saves ~10 s; the fingerprint then carries only the expected one)")
    ap.add_argument("--out", default=None, help="output directory (default $RESEARCH_RUN_DIR)")
    ap.add_argument("--diagnostic", action="store_true",
                    help="sp1: the host is an unsound-* build (authentication skipped); label ARITHMETIC_DIAGNOSTIC, verify under its own key")
    a = ap.parse_args(argv)
    if a.diagnostic and a.backend == "reference":
        ap.error("--diagnostic applies to the sp1 backends")

    out = Path(a.out or os.environ.get("RESEARCH_RUN_DIR") or ".").resolve()
    out.mkdir(parents=True, exist_ok=True)
    path = FIXTURES / f"{a.vector}.json"
    rec = json.loads(path.read_text())
    if a.backend != "reference" and "witness" not in rec:
        print(f"{a.vector} carries no witness; only the reference backend can run it", file=sys.stderr)
        return 2

    fingerprint = {
        "workload": "dot-product-vector",
        "vector": a.vector,
        "vector_sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest(),
        "object_digest": rec["digest"],
        "statement_digest": rec.get("statement_digest"),
        "statement_format": 2,
        "family": rec["object"].get("family"),
        "backend": a.backend,
        "source_sha": source_sha(),
        "python": sys.version.split()[0],
        # the two labels every campaign number carries (module docstring)
        "proof_class": ("ARITHMETIC_DIAGNOSTIC" if a.diagnostic else "NON_ZK_PROOF") if a.backend == "sp1-prove" else "NO_PROOF",
        "authentication": "excluded" if a.diagnostic else "included",
    }
    started = time.perf_counter()
    if a.backend == "reference":
        validation, measurements, artifacts, _ = run_reference(rec, out)
    else:
        validation, measurements, artifacts, _, identity = run_sp1(
            rec, out, prove=a.backend == "sp1-prove", mode=a.mode, want_info=not a.no_info, diagnostic=a.diagnostic
        )
        fingerprint["guest"] = identity
        if a.backend == "sp1-prove":
            fingerprint["mode"] = a.mode
            # which prover ran: SP1_PROVER=cuda (GPU) or the default CPU prover; and which installed host binary
            fingerprint["prover"] = os.environ.get("SP1_PROVER", "cpu")
            fingerprint["host_binary"] = os.environ.get("VERITY_SP1_HOST")
    measurements.append(measurement("workload_wall", time.perf_counter() - started, "s"))
    (out / "measurements.json").write_text(json.dumps(measurements, indent=1) + "\n")

    result = {
        "schema": RESULT_SCHEMA,
        "run_id": os.environ.get("RESEARCH_RUN_ID", "local"),
        "workload_fingerprint": fingerprint,
        "validation": validation,
        "measurements": measurements,
        "measurement_files": ["measurements.json"],
        "artifacts": artifacts,
    }
    tmp = out / "result.json.tmp"
    tmp.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, out / "result.json")
    print(json.dumps({"vector": a.vector, "backend": a.backend, "validation": validation["status"], "detail": validation["detail"],
                      "measurements": {m["name"]: m["value"] for m in measurements if not m["name"].startswith("cycles.")}}, indent=1))
    return 0 if validation["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
