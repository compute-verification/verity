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

The fingerprint carries the campaign contract's fields (``verity_numerical.bench.contract``): ``profile`` (the frozen
target's name when the vector is in its family, else null), ``domain``, ``K`` and ``B`` read off the object,
``zk_mode``, ``security``, ``hardware`` (nvidia-smi / cpuinfo, when present), ``software``, ``preprocessing``,
``retention``.  Where a value is not observed it is null, never a guess.  SP1's measurements are reported under the
host's own names (so the earlier baseline runs stay comparable) *and* under the contract's where they truly fit:
``setup_seconds`` -> ``fixed.setup_seconds``; ``prove_seconds`` -> the joint stage
``t.joint.witness+encoding_commitment+arithmetic+lookup`` (the SDK's one prover timer: trace generation,
commitments, constraint and lookup arguments, FRI -- the host cannot split it) and ``t.total``; the in-process
``verify_seconds`` -> ``verifier.seconds``; ``proof_bytes`` -> ``transcript.bytes``.  Writing the proof to disk is
outside the prover timer and untimed, so ``t.serialization`` is absent; peak memory is not measured here (the GPU
prover is a separate ``sp1-gpu-server`` process; ``research``'s ``resources.jsonl`` samples it).  A contract problem
is printed to stderr and in the summary; it does not change the exit status, which is the validation verdict.

    uv run python benchmarks/dot_product/vector_run.py --vector accept--packed-word-row --backend sp1-prove --out /tmp/k16
    uv run research run --project verity --stage sp1.prove.k16 --require-result -- \\
        uv run python benchmarks/dot_product/vector_run.py --vector accept--packed-word-row --backend sp1-prove

The script runs from a plain checkout: when ``verity`` / ``verity_sp1`` / ``verity_numerical`` are not installed it
adds ``packages/verity/src``, ``backends/sp1/python`` and ``backends/numerical/python`` to ``sys.path``.
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
for candidate, module in (
    (REPO / "packages" / "verity" / "src", "verity"),
    (REPO / "backends" / "sp1" / "python", "verity_sp1"),
    (REPO / "backends" / "numerical" / "python", "verity_numerical"),
):
    try:
        __import__(module)
    except ImportError:
        sys.path.insert(0, str(candidate))

from verity.verification.binding import bound_public_values  # noqa: E402
from verity.verification.conformance import interpret  # noqa: E402
from verity.verification.target import FIRST  # noqa: E402
from verity_numerical.bench import contract  # noqa: E402

RESULT_SCHEMA = contract.RESULT_SCHEMA
BACKENDS = ("reference", "sp1-execute", "sp1-prove")


def source_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def measurement(name: str, value: float | int, unit: str) -> dict:
    return contract.measurement(name, value, unit)


def vector_K(rec: dict) -> int | None:
    """The one ``K`` of the vector's ``GemmCoordinate`` subcircuits; ``None`` when there is none or they disagree."""

    ks = {s["definition"]["static"]["K"] for s in rec["object"].get("subcircuits", ()) if s.get("definition", {}).get("name") == "GemmCoordinate"}
    return ks.pop() if len(ks) == 1 else None


def campaign_fields(rec: dict, backend: str, diagnostic: bool) -> dict:
    """The contract's identity and batch fields for one vector through one backend -- what the judge compares before
    it looks at a number.  Read off the object, not asserted: another family is not the campaign target (``profile``
    null), a vector without one ``GemmCoordinate`` has no ``K``."""

    obj = rec["object"]
    fam = obj.get("family") or {}
    on_target = fam.get("id") == FIRST.family and fam.get("digest") == FIRST.family_digest
    gemm = {s["digest"] for s in obj.get("subcircuits", ()) if s.get("definition", {}).get("name") == "GemmCoordinate"}
    return {
        "profile": contract.PROFILE if on_target else None,
        # every registered typed-obligation/v0 family is the finite-domain relation (target.py: total semantics bind no family)
        "domain": contract.DOMAIN,
        "K": vector_K(rec),
        # one obligation is one GemmCoordinate verification unit; nothing is padded, so B_real is not reported
        "B": sum(1 for o in obj.get("obligations", ()) if o.get("subcircuit") in gemm),
        # the two labels every campaign number carries (module docstring)
        "proof_class": ("ARITHMETIC_DIAGNOSTIC" if diagnostic else "NON_ZK_PROOF") if backend == "sp1-prove" else "NO_PROOF",
        "authentication": "excluded" if diagnostic else "included",
    }


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


#: SP1's one prover timer covers trace generation, commitments, the constraint and lookup arguments and FRI; the host
#: cannot split it, so it is reported as this one joint stage, never as four buckets.
SP1_PROVE_STAGE = contract.joint("witness", "encoding_commitment", "arithmetic", "lookup")


def sp1_prove_measurements(report) -> list[dict]:
    """The contract's names for a ``ProofReport``, beside the host's own (which the earlier baseline runs carry alone).
    No zero-knowledge work exists in core/compressed (``t.zk_additional`` = 0).  Writing the proof to disk happens after
    the prover timer and is untimed: ``t.serialization`` is absent and ``t.total`` is the prover timer."""

    return [
        contract.measurement("fixed.setup_seconds", report.setup_seconds),
        contract.measurement(SP1_PROVE_STAGE, report.prove_seconds),
        contract.measurement("t.zk_additional", 0.0),
        contract.measurement(contract.T_TOTAL, report.prove_seconds),
        contract.measurement("verifier.seconds", report.verify_seconds),
        contract.measurement("transcript.bytes", report.proof_bytes),
    ]


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
        recorded = bytes.fromhex(rec["binding"]["public_values"])
        if diagnostic:
            if report.public_values[:-1] != recorded[:-1]:
                problems.append("h_O || h_L differ from the public values recorded for this vector")
            if report.public_values[-1:] != b"\xee":
                problems.append(f"diagnostic verdict byte is {report.public_values[-1:].hex()}, not ee (accepted, unsound)")
            if not binding.ok:
                problems.append(f"binding check: {binding.code} {binding.detail}")
            checker = "DIAGNOSTIC: verity_sp1 execute (unsound guest) + binding of h_O || h_L; verdict 0xEE"
        else:
            if report.public_values.hex() != expected_pv:
                problems.append("public values are not h_O || h_L || 01 of this object")
            if report.public_values != recorded:
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
        measurements += sp1_prove_measurements(report)
        recorded = bytes.fromhex(rec["binding"]["public_values"])
        public_values = report.execution.public_values
        if diagnostic:
            # a diagnostic guest commits h_O || h_L || 0xEE ("accepted, unsound"; VERDICT_ACCEPT in common/src/lib.rs):
            # the digests must match the record, the verdict byte is 0xEE, and the proof verifies under the guest's own key
            if public_values[:-1] != recorded[:-1]:
                problems.append("h_O || h_L differ from the public values recorded for this vector")
            if public_values[-1:] != b"\xee":
                problems.append(f"diagnostic verdict byte is {public_values[-1:].hex()}, not ee (accepted, unsound)")
            if not (acceptance.proof_verifies and acceptance.approved_key and acceptance.binding.ok):
                problems += [r for r in acceptance.reasons() if "diagnostic" not in r and "verdict byte" not in r]
            checker = "DIAGNOSTIC: verity_sp1 prove + verify_object under the unsound guest's own key (binding, verdict 0xEE); not a sound proof"
        else:
            if public_values != recorded:
                problems.append("public values differ from the ones recorded for this vector")
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

    sha = source_sha()
    python = sys.version.split()[0]
    fingerprint = {
        **campaign_fields(rec, a.backend, a.diagnostic),
        "workload": "dot-product-vector",
        "vector": a.vector,
        "vector_sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest(),
        "object_digest": rec["digest"],
        "statement_digest": rec.get("statement_digest"),
        "statement_format": 2,
        "family": rec["object"].get("family"),
        "backend": a.backend,
        "source_sha": sha,
        "python": python,
        # nothing here establishes privacy: SP1 core/compressed is a plain (non-ZK) proof, the others prove nothing
        "zk_mode": contract.ZK_NONE,
        # what the machine has (nvidia-smi, cpuinfo); whether the GPU was used is software.backend.prover
        "hardware": contract.probe_hardware(),
        # the guest reads the whole witness (the openings) from stdin into memory; the reference checker holds the record
        "retention": "retained",
    }
    started = time.perf_counter()
    if a.backend == "reference":
        validation, measurements, artifacts, _ = run_reference(rec, out)
        fingerprint["security"] = {"target": None, "note": "no proof: the transparent checker's own verdict, nothing established to a verifier"}
        fingerprint["software"] = {"backend": {"name": "verity.verification.transparent", "version": None, "commit": sha}, "toolchain": {"python": python}}
        fingerprint["preprocessing"] = "none"
    else:
        validation, measurements, artifacts, _, identity = run_sp1(
            rec, out, prove=a.backend == "sp1-prove", mode=a.mode, want_info=not a.no_info, diagnostic=a.diagnostic
        )
        fingerprint["guest"] = identity
        guest = identity.get("observed") or identity["expected"]
        # the rustc that built the installed host binary is not reported by the host's `info`: null, not the channel
        fingerprint["software"] = {
            "backend": {"name": "sp1", "version": guest["sp1_version"], "commit": None, "guest_elf_sha256": guest["elf_sha256"],
                        "vk_hash": guest["vk_hash"], "guest_features": guest["guest_features"], "host_binary": os.environ.get("VERITY_SP1_HOST")},
            "toolchain": {"python": python, "sp1_sdk": guest["sp1_version"], "rust": None},
        }
        if a.backend == "sp1-prove":
            fingerprint["mode"] = a.mode
            # which prover ran: SP1_PROVER=cuda (GPU) or the default CPU prover; and which installed host binary
            fingerprint["prover"] = os.environ.get("SP1_PROVER", "cpu")
            fingerprint["host_binary"] = os.environ.get("VERITY_SP1_HOST")
            fingerprint["software"]["backend"]["prover"] = fingerprint["prover"]
            # SP1 does not expose its soundness parameters (field, FRI rate, query count) through the host: null, not a guess
            fingerprint["security"] = {"target": None, "parameters": {"mode": a.mode},
                                       "note": "SP1 core/compressed: proof-system parameters not exposed here; target soundness not established"}
            fingerprint["preprocessing"] = ("none offline; SP1 setup (proving and verifying keys from the guest ELF) runs in the host every time "
                                            "and is reported as fixed.setup_seconds, outside t.total")
        else:
            fingerprint["security"] = {"target": None, "note": "no proof: the executor's verdict, nothing established to a verifier"}
            fingerprint["preprocessing"] = "none"
    # VU throughput: one obligation is one verification unit; the rate only exists where something was proved
    B = fingerprint["B"]
    measurements.append(contract.measurement("vu.count", B))
    total = next((m["value"] for m in measurements if m["name"] == contract.T_TOTAL), None)
    if total and B:
        measurements += [contract.measurement("vu.seconds_per_vu", total / B), contract.measurement("vu.per_second", B / total)]
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
    # the campaign contract: reported, not enforced -- a vector outside the target legitimately fails it
    problems = contract.validate(result)
    if problems:
        print(f"contract: {len(problems)} problem(s): " + "; ".join(problems), file=sys.stderr)
    print(json.dumps({"vector": a.vector, "backend": a.backend, "validation": validation["status"], "detail": validation["detail"],
                      "contract": problems or "conforms",
                      "measurements": {m["name"]: m["value"] for m in measurements if not m["name"].startswith("cycles.")}}, indent=1))
    return 0 if validation["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
