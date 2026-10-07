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
A third label, ``statement_machinery`` (``included | excluded``), says whether the guest parsed the statement and
built the opening table: ``excluded`` only for ``--relation-only`` (with ``--diagnostic``), which needs the
``unsound-relation-only`` host.  That guest runs nothing but the ``GemmCoordinate<K>`` gate chain on the compact
input ``K || x || W`` built here from the vector's object and witness (``verity_sp1.relation_only``: the witness is
authenticated in Python, the words read off it, the Python reference agrees on the output); it is judged by its own
key and by its public values being exactly ``sha256(compact input) || sha256(committed output word) || 0xEE``.
It is the arithmetic baseline of the zkVM path -- what the relation alone costs to prove -- and never a result.

The fingerprint carries the campaign contract's fields (``verity_numerical.bench.contract``): ``profile`` (the frozen
target's name when the vector is in its family, else null), ``domain``, ``K`` and ``B`` read off the object,
``zk_mode``, ``security``, ``hardware`` (nvidia-smi / cpuinfo, when present), ``software``, ``preprocessing``,
``retention``, ``instances`` (the ``bench-instances/v1`` ids the vector is, by content: the K=1536 vector is
``vu-k1536`` id 0; every other vector null).  Where a value is not observed it is null, never a guess.  SP1's measurements are reported under the
host's own names (so the earlier baseline runs stay comparable) *and* under the contract's where they truly fit:
``setup_seconds`` -> ``fixed.setup_seconds``; ``prove_seconds`` -> the joint stage
``t.joint.witness+encoding_commitment+arithmetic+lookup`` (the SDK's one prover timer: trace generation,
commitments, constraint and lookup arguments, FRI -- the host cannot split it) and ``t.total``; the in-process
``verify_seconds`` -> ``verifier.seconds``; ``proof_bytes`` -> ``transcript.bytes``.  Writing the proof to disk is
outside the prover timer and untimed, so ``t.serialization`` is absent; peak memory is not measured here (the GPU
prover is a separate ``sp1-gpu-server`` process; ``research``'s ``resources.jsonl`` samples it).  A contract problem
is printed to stderr and in the summary; it does not change the exit status, which is the validation verdict.

    uv run python archive/sp1/benchmarks/dot_product/vector_run.py --vector accept--packed-word-row --backend sp1-prove --out /tmp/k16
    uv run research run --project verity --stage sp1.prove.k16 --require-result -- \\
        uv run python archive/sp1/benchmarks/dot_product/vector_run.py --vector accept--packed-word-row --backend sp1-prove

The script runs from a plain checkout: when ``verity`` / ``verity_sp1`` / ``verity_numerical`` are not installed it
adds ``.``, ``archive/sp1/python`` and ``benchmarks/numerical/python`` to ``sys.path``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
FIXTURES = REPO / "fixtures" / "typed-obligation-v0"
for candidate, module in (
    (REPO, "verity"),
    (REPO / "archive" / "sp1" / "python", "verity_sp1"),
    (REPO / "benchmarks" / "numerical" / "python", "verity_numerical"),
):
    try:
        __import__(module)
    except ImportError:
        sys.path.insert(0, str(candidate))

from verity_sp1.proofs.binding import bound_public_values  # noqa: E402
from verity_sp1.proofs.conformance import interpret  # noqa: E402
from verity.ml.circuits.targets.target import FIRST  # noqa: E402
from verity_numerical.bench import contract  # noqa: E402

RESULT_SCHEMA = contract.RESULT_SCHEMA
BACKENDS = ("reference", "sp1-execute", "sp1-prove", "sp1-bare")
INSTANCES_ROOT = REPO / "fixtures" / "bench-instances" / "v1"


def instances_of(rec: dict) -> dict | None:
    """The ``bench-instances/v1`` ids a vector proves, found by content: the K=1536 vector is instance 0 of ``vu-k1536``
    (seed ``qkv_proj_decode_0``, j=1014) iff its packed witness holds that seed's ``x.u16`` and ``w_row.u16`` bytes.  Null
    for every other vector (not measured on the shared set), and when the dataset is not checked out."""

    manifest = INSTANCES_ROOT / "manifest.json"
    seed = INSTANCES_ROOT / "seeds" / "qkv_proj_decode_0"
    if not (manifest.is_file() and (seed / "x.u16").is_file() and (seed / "w_row.u16").is_file() and isinstance(rec.get("witness"), str)):
        return None
    try:
        packed = bytes.fromhex(rec["witness"])
    except ValueError:
        return None
    if (seed / "x.u16").read_bytes() not in packed or (seed / "w_row.u16").read_bytes() not in packed:
        return None
    index = json.loads((INSTANCES_ROOT / "vu-k1536.index.json").read_text())
    row0 = index["rows"][0]
    if not ("seed" in row0[1] and index["seeds"][row0[2]] == "qkv_proj_decode_0"):
        return None
    return contract.instances_ref("vu-k1536", 0, 1, __import__("hashlib").sha256(manifest.read_bytes()).hexdigest())


def source_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass
    # a `research run --source` checkout has no .git: the run's job.json names the commit it shipped
    try:
        job = json.loads((Path(os.environ["RESEARCH_RUN_DIR"]) / "job.json").read_text())
        return job["computation"]["tree_sha"] or None
    except (KeyError, OSError, ValueError, TypeError):
        return None


def measurement(name: str, value: float | int, unit: str) -> dict:
    return contract.measurement(name, value, unit)


def vector_K(rec: dict) -> int | None:
    """The one ``K`` of the vector's ``GemmCoordinate`` subcircuits; ``None`` when there is none or they disagree."""

    ks = {s["definition"]["static"]["K"] for s in rec["object"].get("subcircuits", ()) if s.get("definition", {}).get("name") == "GemmCoordinate"}
    return ks.pop() if len(ks) == 1 else None


def campaign_fields(rec: dict, backend: str, diagnostic: bool, relation_only: bool = False) -> dict:
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
        # one obligation is one GemmCoordinate proof unit; nothing is padded, so B_real is not reported
        "B": sum(1 for o in obj.get("obligations", ()) if o.get("subcircuit") in gemm),
        # the two labels every campaign number carries (module docstring)
        "proof_class": ("ARITHMETIC_DIAGNOSTIC" if diagnostic else "NON_ZK_PROOF") if backend == "sp1-prove" else "NO_PROOF",
        "authentication": "excluded" if diagnostic else "included",
        # whether the guest parsed the statement and built the opening table (the skip-authentication diagnostic
        # still does; only the relation-only guest does not)
        "statement_machinery": "excluded" if relation_only else "included",
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
        "checker": "verity_sp1.proofs.conformance.interpret + TransparentBackend",
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


def run_sp1_relation_only(rec: dict, out: Path, host, observed, *, prove: bool, mode: str) -> tuple[dict, list[dict], list[str], dict]:
    """The relation-only DIAGNOSTIC: the compact input built from the vector, the bare gate chain executed or
    proved, and the judgment -- the proof verifies under the guest's own key, the public values are exactly
    ``sha256(compact input) || sha256(committed output word) || 0xEE``.  Returns ``(validation, measurements,
    artifacts, relation_only fingerprint)``."""

    from verity_sp1.relation_only import compact_input

    compact = compact_input(rec["object"], bytes.fromhex(rec["witness"]))
    expected = compact.public_values
    described = {
        "K": compact.K, "compact_input_bytes": len(compact.data),
        "compact_input_sha256": __import__("hashlib").sha256(compact.data).hexdigest(),
        "committed_output_word": f"{compact.output_word:04x}", "expected_public_values": expected.hex(),
        "what": "K || x || W of the one GemmCoordinate obligation, words authenticated in Python; the guest ran the gate chain only",
    }
    documents: dict = {"compact_input": described}
    measurements: list[dict] = []
    problems: list[str] = []
    artifacts: list[str] = []

    def judge_public_values(public_values: bytes, what: str) -> None:
        if public_values != expected:
            problems.append(f"{what} public values {public_values.hex()} are not sha256(compact input) || sha256(output word) || ee")

    if not prove:
        started = time.perf_counter()
        report = host.execute(b"", compact.data)  # the statement is ignored by this host; the "witness" is the compact input
        wall = time.perf_counter() - started
        documents["execute"] = dict(report.document)
        measurements += [measurement("execute_wall", wall, "s"), measurement("total_cycles", report.total_cycles, "cycles")]
        measurements += [measurement(f"cycles.{k}", v, "cycles") for k, v in sorted(report.cycle_tracker.items())]
        judge_public_values(report.public_values, "execute")
        if not (report.verdict and report.unsound):
            problems.append("the guest's verdict byte is not the diagnostic accept (0xEE)")
        checker = "RELATION-ONLY DIAGNOSTIC: verity_sp1 execute (unsound-relation-only guest) on the compact input; public values sha256(input) || sha256(output) || 0xEE"
    else:
        proof = out / "proof.bin"
        started = time.perf_counter()
        report = host.prove(b"", compact.data, proof, mode=mode)
        prove_wall = time.perf_counter() - started
        started = time.perf_counter()
        verified = host.verify(proof)
        verify_wall = time.perf_counter() - started
        documents["prove"] = dict(report.document)
        documents["verify"] = dict(verified.document)
        judge_public_values(report.execution.public_values, "prove")
        if not verified.ok:
            problems.append("the proof does not verify under the guest's verifying key")
        elif verified.vk_hash != observed.vk_hash:
            problems.append(f"the proof's verifying key {verified.vk_hash} is not the guest's own {observed.vk_hash}")
        else:
            judge_public_values(verified.public_values, "verified proof")
            if not (verified.verdict and verified.unsound):
                problems.append("the verified proof's verdict byte is not the diagnostic accept (0xEE)")
        documents["acceptance"] = {
            "accepted_as_sound": False, "proof_verifies": verified.ok, "own_key": verified.ok and verified.vk_hash == observed.vk_hash,
            "public_values_expected": verified.public_values == expected, "unsound": verified.unsound, "reasons": list(problems),
        }
        measurements += [
            measurement("prove_wall", prove_wall, "s"), measurement("setup_seconds", report.setup_seconds, "s"),
            measurement("prove_seconds", report.prove_seconds, "s"), measurement("host_verify_seconds", report.verify_seconds, "s"),
            measurement("verify_wall", verify_wall, "s"), measurement("proof_bytes", report.proof_bytes, "B"),
            measurement("shards", report.shards, "count"), measurement("total_cycles", report.execution.total_cycles, "cycles"),
        ]
        measurements += [measurement(f"cycles.{k}", v, "cycles") for k, v in sorted(report.execution.cycle_tracker.items())]
        measurements += sp1_prove_measurements(report)
        checker = "RELATION-ONLY DIAGNOSTIC: verity_sp1 prove + verify under the unsound-relation-only guest's own key; public values sha256(input) || sha256(output) || 0xEE; not a proof of the statement"
        artifacts.append("proof.bin")
    (out / "host.json").write_text(json.dumps(documents, indent=1, sort_keys=True) + "\n")
    artifacts.append("host.json")
    validation = {"status": "passed" if not problems else "failed", "checker": checker, "detail": "; ".join(problems) or "accepted (diagnostic)", "evidence": ["host.json"]}
    return validation, measurements, artifacts, described


def run_sp1(rec: dict, out: Path, *, prove: bool, mode: str, want_info: bool, diagnostic: bool, relation_only: bool = False) -> tuple[dict, list[dict], list[str], list[str], dict]:
    from verity_sp1.host import APPROVED, Host
    from verity_sp1.relation_only import FEATURE as RELATION_ONLY_FEATURE
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
        if relation_only != (RELATION_ONLY_FEATURE in observed.guest_features):
            raise SystemExit(f"--relation-only needs (and only fits) a host whose guest has the {RELATION_ONLY_FEATURE} feature; this one has {list(observed.guest_features)}")
    else:
        trusted = APPROVED
    if relation_only:
        validation, measurements, artifacts, described = run_sp1_relation_only(rec, out, host, observed, prove=prove, mode=mode)
        identity["relation_only"] = described
        return validation, measurements, artifacts, [], identity
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
            checker = "verity_sp1 execute + verity_sp1.proofs.binding.check_binding"
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


# --- sp1-bare: the Table 2 bare statement over a frozen batch ---------------------------------------------------------
#
# ``--backend sp1-bare`` runs a ``relation-bare`` host (``archive/sp1/host --features relation-bare``, or any SP1 variant
# with the same ``bare-*`` CLI: ``--host``, ``--backend-name``) on a range of a frozen VU set -- the BF16 fixture
# (``--instances DIR``, the A100 row) or a ``verity_sp1.format_instances`` file (``--batch FILE``, the other rows) -- and
# writes one ``bench-result/v1``: public = format + input-set identity + y, private = x and W (authentication
# excluded).  Steps: the arrays against their digests; negatives through the executor (the 52 committed ``vu-k1536-neg``
# instances, or ``--flip-y`` on a batch); ``bare-prove`` (an executor pass for exact cycles, setup, a warm-up proof,
# ``--reps`` proofs, each dumped under ``proofs/`` and verified in process); ``verify --proof --statement`` per proof in
# a fresh CPU process (the producer's check; independent verification is not the producer's).  ``--layout indexed``
# hands the guest each distinct row once plus a row pair per VU instead of every VU's rows (the statement is the same).
#
# Phases: SP1 6's prover pipelines execution, CPU event generation, GPU trace generation, commitment and proving across
# shards and exposes no split, so the prove call is ``SP1_PROVE_STAGE`` and ``t.serialization`` is the proof's bincode
# write; ``t.total`` = the median rep's sum.  The GPU server's DEBUG log gives the executor time, the wait before the first
# shard reaches the GPU and the GPU span (``sp1.*``, diagnostics, not buckets).  Soundness is SP1's own accounting:
# ``SP1_TARGET_BITS_OF_SECURITY = 100`` per STARK proof (sp1-primitives 6.6.0 ``fri_params.rs``), union-bounded over the
# core proof's shard proofs -- below the tables' 2^-128, recorded as it is.

SP1_BARE_HOST = "/workspace/bin/veritor-zk-host-cuda-relation-bare"
SP1_BARE_BACKEND_NAME = "sp1 relation-bare/v2 (stock SP1 6.4.0, core STARK, CUDA prover)"
SP1_VERSION = "6.4.0"
SP1_SECURITY_BITS = 100
BARE_TIER = "vu-k1536"
BARE_NEG_TIER = "vu-k1536-neg"


def _sha256_file(p: Path) -> str:
    h = __import__("hashlib").sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_lines(text: str) -> list[dict]:
    """The JSON objects among a process's output lines (the SP1 GPU server's log lines share the streams)."""

    out = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def _run_logged(cmd: list[str], env: dict, log: Path, timeout: float) -> tuple[int, str, float]:
    t0 = time.perf_counter()
    with open(log, "ab") as err:
        err.write(("\n$ " + " ".join(cmd) + "\n").encode())
        err.flush()
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=err, env=env, timeout=timeout)
    return p.returncode, p.stdout.decode(errors="replace"), time.perf_counter() - t0


def sp1_timeline(log_text: str) -> list[dict]:
    """Per prove call (proof_id), from the GPU server's DEBUG log: the executor's own elapsed time, the time from the
    call's first log line to the first shard's GPU work ("Proving shard") and from there to the last shard task."""

    import re
    from datetime import datetime, timezone

    ts_re = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d+)Z\s+\w+\s+(.*)$")
    proofs: dict[str, dict] = {}
    for raw in log_text.splitlines():
        line = re.sub(r"\x1b\[[0-9;]*m", "", raw)
        m = ts_re.match(line)
        pid = re.search(r"proof_id=(proof_[a-z0-9]+)", line) if m else None
        if not pid:
            continue
        t = datetime.strptime(m.group(1)[:26], "%Y-%m-%dT%H:%M:%S.%f").replace(tzinfo=timezone.utc).timestamp()
        d = proofs.setdefault(pid.group(1), {"first": t, "gpu": [], "done": [], "executor_seconds": None})
        d["last"], msg = t, m.group(2)
        if "minimal Executor finished" in msg:
            e = re.search(r"elapsed: ([0-9.]+)s", msg)
            d["executor_seconds"] = float(e.group(1)) if e else None
        elif "Proving shard" in msg:
            d["gpu"].append(t)
        elif "ProveShard" in msg and "task succeeded" in msg:
            d["done"].append(t)
    return [{"proof_id": k, "executor_seconds": d["executor_seconds"], "shard_tasks": len(d["done"]),
             "until_first_gpu_shard_seconds": min(d["gpu"]) - d["first"],
             "gpu_phase_seconds": (max(d["done"]) if d["done"] else d["last"]) - min(d["gpu"])}
            for k, d in proofs.items() if d["gpu"]]


def _check_bf16_fixtures(root: Path) -> tuple[str, dict]:
    """sha256 of ``manifest.json`` (must be the frozen BF16 set's, and the checkout's own) and of every array read."""

    from verity_numerical.bench import tables

    man_path = root / "manifest.json"
    manifest_sha = _sha256_file(man_path)
    if manifest_sha != tables.BF16_INSTANCES_MANIFEST_SHA256:
        raise SystemExit(f"sp1-bare: {man_path} is {manifest_sha}, not the frozen {tables.BF16_INSTANCES_MANIFEST_SHA256}")
    own = INSTANCES_ROOT / "manifest.json"
    if own.is_file() and _sha256_file(own) != manifest_sha:
        raise SystemExit(f"sp1-bare: {man_path} differs from the checkout's {own}")
    man = json.loads(man_path.read_text())
    checked = {}
    for tier in (BARE_TIER, BARE_NEG_TIER):
        files = man["tiers"][tier]["files"]
        for n in ("x", "w", "y"):
            fname = f"{tier}.{n}.u16"
            got = _sha256_file(root / fname)
            if got != files[fname]["sha256"]:
                raise SystemExit(f"sp1-bare: {fname} sha256 {got} != manifest {files[fname]['sha256']}")
            checked[fname] = got
    return manifest_sha, checked


def _batch_header(path: Path) -> dict:
    data = path.open("rb").read(1 << 20)
    if data[:8] != b"VYSP1FI\x01":
        raise SystemExit(f"sp1-bare: {path} is not an sp1-format-instances/v1 file")
    hl = int.from_bytes(data[8:12], "little")
    return json.loads(data[12:12 + hl])


def run_sp1_bare(a) -> int:
    from verity_numerical.bench import tables

    started = time.perf_counter()
    out = Path(a.out or os.environ.get("RESEARCH_RUN_DIR") or ".").resolve()
    out.mkdir(parents=True, exist_ok=True)
    proofs_dir, log, prove_log = out / "proofs", out / "host.log", out / "prover.log"
    host = Path(a.host)
    variant = json.loads(Path(a.variant).read_text()) if a.variant else {}
    prover_env = dict(kv.split("=", 1) for kv in a.prover_env)
    cpu_env = {**os.environ, "SP1_PROVER": "cpu", "RUST_LOG": "error"}
    gpu_env = {**os.environ, **prover_env, "SP1_PROVER": "cuda", "RUST_LOG": "debug"}

    # the set: the frozen BF16 fixture or a format-instances file
    if a.batch:
        batch = Path(a.batch).resolve()
        header = _batch_header(batch)
        target_name, ref, n = header["target"], header["instances"], header["n"]
        source = ["--batch", str(batch)]
        fixture_digests = {"batch_file_sha256": _sha256_file(batch), **{k: header[k] for k in ("x_sha256", "w_sha256", "y_sha256")}}
        fmt = header["format"]
    else:
        root = Path(a.instances).resolve()
        manifest_sha, fixture_digests = _check_bf16_fixtures(root)
        target_name, n, fmt = FIRST.name, 4096, "bf16-ampere"
        ref = contract.instances_ref(BARE_TIER, 0, n, manifest_sha)
        source = ["--instances", str(root), "--manifest-sha256", manifest_sha]
    lo, hi = a.lo, (a.hi if a.hi is not None else n)
    B = hi - lo
    base = ref["range"][0]
    instances = contract.instances_ref(ref["tier"], base + lo, base + hi, ref["manifest_sha256"], dataset=ref["dataset"])
    layout = ["--layout", a.layout]
    rng = ["--lo", str(lo), "--hi", str(hi), "--vus-per-read", str(a.vus_per_read), *layout]
    print(f"sp1-bare: {fmt} {instances} on {host}", flush=True)

    rc, text, _ = _run_logged([str(host), "info"], cpu_env, log, 900)
    info = (_json_lines(text) or [{}])[-1]
    if rc != 0 or "relation-bare" not in info.get("guest_features", []):
        raise SystemExit(f"sp1-bare: {host} is not a relation-bare build: rc {rc} {info}")

    negatives = []
    if not a.skip_negatives:
        if a.batch:
            for j in sorted({0, B // 2, B - 1}):
                rc, text, wall = _run_logged([str(host), "bare-execute", *source, "--lo", str(lo + j), "--hi", str(lo + j + 1), "--vus-per-read", "1",
                                              *layout, "--flip-y", "0"], cpu_env, log, 900)
                doc = (_json_lines(text) or [{}])[-1]
                negatives.append({"kind": "flip-y", "vu": lo + j, "rc": rc, "verdict": doc.get("verdict"), "matches_native": doc.get("matches_expected")})
            summary = {"negatives": len(negatives), "claimed_rejected": sum(1 for x in negatives if x["rc"] == 0 and x["verdict"] is False),
                       "in_domain": 0, "correct_accepted": 0}
        else:
            rc, text, _ = _run_logged([str(host), "bare-negatives", "--instances", source[1], "--manifest-sha256", source[3], *layout], cpu_env, log, 1800)
            lines = _json_lines(text)
            negatives = [d for d in lines if d.get("event") == "negative"]
            summary = next((d for d in lines if d.get("event") == "summary"), {"negatives": 0, "claimed_rejected": 0, "in_domain": 0, "correct_accepted": 0})
        summary["ok"] = summary["negatives"] > 0 and summary["claimed_rejected"] == summary["negatives"] and summary["correct_accepted"] == summary["in_domain"]
        (out / "negatives.json").write_text(json.dumps({"summary": summary, "items": negatives}, indent=1))
        print(f"sp1-bare: negatives {summary}", flush=True)
        if not summary["ok"]:
            raise SystemExit("sp1-bare: a negative was not rejected -- no result")
    else:
        summary = None

    cmd = [str(host), "bare-prove", *source, *rng, "--out-dir", str(proofs_dir), "--reps", str(a.reps), "--warmup-vus", str(a.warmup_vus), "--mode", a.mode]
    peak = {"mib": None}

    def sample(stop):
        while not stop.wait(0.5):
            try:
                q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
                v = int(q.stdout.split()[0])
                peak["mib"] = v if peak["mib"] is None else max(peak["mib"], v)
            except (OSError, subprocess.TimeoutExpired, ValueError, IndexError):
                pass

    import threading
    stop = threading.Event()
    sampler = threading.Thread(target=sample, args=(stop,), daemon=True)
    sampler.start()
    t0 = time.perf_counter()
    with open(prove_log, "wb") as plog:
        plog.write(("$ " + " ".join(cmd) + "\n").encode())
        plog.flush()
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=plog, env=gpu_env, timeout=a.timeout)
    prove_wall = time.perf_counter() - t0
    stop.set()
    sampler.join(timeout=10)
    text = p.stdout.decode(errors="replace")
    with open(prove_log, "ab") as plog:
        plog.write(b"\n--- stdout ---\n" + text.encode())
    lines = _json_lines(text)
    execute = next((d["report"] for d in lines if d.get("event") == "execute"), None)
    setup = next((d for d in lines if d.get("event") == "setup"), None)
    warmup = next((d for d in lines if d.get("event") == "warmup"), None)
    reps = [d for d in lines if d.get("event") == "rep"]
    (out / "prove.json").write_text(json.dumps({"cmd": cmd, "rc": p.returncode, "wall": prove_wall, "execute": execute, "setup": setup,
                                                "warmup": warmup, "reps": reps}, indent=1))
    if p.returncode != 0 or len(reps) != a.reps or not all(r["accepted"] for r in reps):
        raise SystemExit(f"sp1-bare: bare-prove failed (rc {p.returncode}, {len(reps)}/{a.reps} reps): see {prove_log}")
    timeline = sp1_timeline(prove_log.read_text(errors="replace"))[-a.reps:]

    verdicts = []
    for r in reps:
        rc, text, wall = _run_logged([str(host), "verify", "--proof", r["proof_path"], "--statement", str(proofs_dir / "statement.bin")], cpu_env, log, 1800)
        doc = (_json_lines(text) or [{}])[-1]
        ok = rc == 0 and bool(doc.get("ok")) and doc.get("statement_match") is True and bool(doc.get("verdict"))
        verdicts.append({"rep": r["rep"], "proof": os.path.relpath(r["proof_path"], out), "accepted": ok, "rc": rc,
                         "verify_seconds": doc.get("verify_seconds"), "process_wall": wall, "vk_hash": doc.get("vk_hash")})
        print(f"sp1-bare: verify rep{r['rep']}: {'accept' if ok else 'REJECT'} in {doc.get('verify_seconds')} s (process {wall:.1f} s)", flush=True)
    verifier_name = variant.get("verifier_name") or f"{host} verify"
    (out / "verify.json").write_text(json.dumps({"verifier": f"{verifier_name}, SP1_PROVER=cpu (the producer's binary: not independent)",
                                                 "items": verdicts}, indent=1))

    totals = [r["prove_seconds"] + r["save_seconds"] for r in reps]
    med = sorted(range(len(reps)), key=totals.__getitem__)[len(reps) // 2]
    rep, t_total = reps[med], totals[med]
    shards = int(rep["shards"])
    sha = source_sha() or os.environ.get("RESEARCH_SOURCE_COMMIT")
    bits = int(variant.get("security_bits", SP1_SECURITY_BITS))
    # A variant whose SP1_TARGET_BITS_OF_SECURITY no longer bounds the whole shard proof (it sets only the FRI query
    # count) gives the per-shard figure of the full accounting instead, with its terms.
    per_shard = float(variant.get("security_per_shard_log2", -bits))
    security = {"target": -bits, "achieved_log2": round(per_shard + __import__("math").log2(shards), 3),
                "source": variant.get("security_source") or
                          "SP1's own accounting: sp1-primitives 6.6.0 src/fri_params.rs SP1_TARGET_BITS_OF_SECURITY = 100 per STARK proof "
                          "(CORE_LOG_BLOWUP 2, unique_decoding_queries = ceil((100 - 16) / -log2(5/8)) = 124 at SP1_PROOF_OF_WORK_BITS 16; "
                          "KoalaBear^4), compile-time constants of the pinned SP1",
                "union_bound": f"{shards} shard proofs in the core proof: {per_shard:g} + log2({shards})"}
    if variant.get("security_parameters"):
        security["parameters"] = variant["security_parameters"]
    software = {"backend": {"name": variant.get("backend_name") or a.backend_name, "version": SP1_VERSION, "commit": sha},
                "toolchain": {"sp1-sdk": SP1_VERSION, "sp1-zkvm": SP1_VERSION, "sp1-primitives": "6.6.0", "sp1-gpu-server": "6.6.0 (release binary)",
                              "guest toolchain": "succinct sp1up v6.4.0", "python": sys.version.split()[0], **variant.get("toolchain", {})},
                "guest": {"elf_sha256": setup["elf_sha256"], "vk_hash": setup["vk_hash"], "features": setup["guest_features"]},
                "host_binary_sha256": _sha256_file(host), "prover": "cuda", "prover_options": prover_env or "SP1 defaults",
                "verifier": verifier_name}
    for key in ("fork", "label"):
        if variant.get(key):
            software[key] = variant[key]
    fp = contract.fingerprint(
        profile=target_name, domain=contract.DOMAIN, K=1536, B=B, proof_class="NON_ZK_PROOF", authentication="excluded", zk_mode=contract.ZK_NONE,
        security=security, hardware=contract.probe_hardware(), software=software,
        preprocessing={"setup": "SP1 setup of the relation-bare ELF (proving and verifying keys), once per ELF, outside t.total",
                       "setup_seconds": setup["setup_seconds"], "per_instance": "none"},
        retention="retained", instances=instances,
        workload="sp1-relation-bare", statement="relation-bare/v2", format=fmt, statement_sha256=setup["statement_sha256"],
        identity=setup["identity"], mode=a.mode, reps=len(reps), input_layout=a.layout,
        **({"vus_per_read": a.vus_per_read} if a.layout == "chunks" else {"table_rows": setup.get("rows")}), input_bytes=setup["input_bytes"],
        warmup={"vus": warmup["vus"], "prove_seconds": warmup["prove_seconds"]} if warmup else None, fixtures=fixture_digests,
        phases="the prove call is one joint stage (SP1 6 pipelines execution, event generation, trace generation, commitment and proving "
               "across shards); t.serialization is the proof's bincode write",
        note=a.note or None,
    )
    verify_s = sorted(v["verify_seconds"] for v in verdicts)[len(verdicts) // 2]
    meas = [
        contract.measurement(contract.T_TOTAL, t_total), contract.measurement(SP1_PROVE_STAGE, rep["prove_seconds"]),
        contract.measurement("t.serialization", rep["save_seconds"]), contract.measurement("t.zk_additional", 0.0),
        contract.measurement("vu.count", B), contract.measurement("vu.seconds_per_vu", t_total / B), contract.measurement("vu.per_second", B / t_total),
        contract.measurement("transcript.bytes", rep["proof_bytes"]), contract.measurement("verifier.seconds", verify_s),
        contract.measurement("fixed.setup_seconds", setup["setup_seconds"]),
        *contract.rate_measurements(t_total, B, 1536, target_name),
        measurement("prove_seconds_reps", sum(r["prove_seconds"] for r in reps) / len(reps), "s"),
        measurement("prove_seconds_min", min(r["prove_seconds"] for r in reps), "s"),
        measurement("prove_seconds_max", max(r["prove_seconds"] for r in reps), "s"),
        measurement("shards", shards, "count"), measurement("proof_bytes", rep["proof_bytes"], "B"),
        measurement("verify_process_wall", sorted(v["process_wall"] for v in verdicts)[len(verdicts) // 2], "s"),
        measurement("host_verify_seconds", sorted(r["verify_seconds"] for r in reps)[len(reps) // 2], "s"),
        measurement("soundness_bits", bits - __import__("math").log2(shards), "bits"),
        measurement("proofs_verified_producer", sum(v["accepted"] for v in verdicts), "count"),
    ]
    if execute:
        meas += [measurement("total_cycles", execute["total_cycles"], "cycles"), measurement("cycles_per_vu", execute["total_cycles"] / B, "cycles"),
                 measurement("guest_input_bytes", execute["input_bytes"], "B"), measurement("execute_wall", execute["execute_seconds"], "s")]
        if execute.get("gas") is not None:
            meas.append(measurement("gas", execute["gas"], "count"))
        meas += [measurement(f"cycles.{k}", v, "cycles") for k, v in sorted(execute["cycle_tracker"].items())]
    if peak["mib"] is not None:
        meas.append(contract.measurement("mem.peak_device_bytes", peak["mib"] << 20))
    if len(timeline) == len(reps):
        t = timeline[med]
        meas += [measurement(f"sp1.{k}", t[k], "s") for k in ("until_first_gpu_shard_seconds", "gpu_phase_seconds") if t[k] is not None]
        if t["executor_seconds"] is not None:
            meas.append(measurement("sp1.executor_seconds", t["executor_seconds"], "s"))
    if summary:
        meas += [measurement("negatives_total", summary["negatives"], "count"), measurement("negatives_rejected", summary["claimed_rejected"], "count"),
                 measurement("negatives_correct_accepted", summary["correct_accepted"], "count")]
    meas.append(measurement("workload_wall", time.perf_counter() - started, "s"))
    (out / "measurements.json").write_text(json.dumps(meas, indent=1) + "\n")
    ok = all(v["accepted"] for v in verdicts) and len(verdicts) == a.reps and (summary is None or summary["ok"])
    detail = (f"{len(reps)} {a.mode} proofs of B={B} ({fmt}): prove {[round(r['prove_seconds'], 3) for r in reps]} s, serialize "
              f"{[round(r['save_seconds'], 4) for r in reps]} s, {shards} shards, {rep['proof_bytes']} B; the producer's verify in a fresh CPU "
              f"process accepted {sum(v['accepted'] for v in verdicts)}/{len(verdicts)} (ok, statement_match, verdict 1)")
    if summary:
        detail += f"; negatives: {summary['claimed_rejected']}/{summary['negatives']} wrong words rejected, {summary['correct_accepted']}/{summary['in_domain']} correct words accepted"
    result = {
        "schema": RESULT_SCHEMA, "run_id": os.environ.get("RESEARCH_RUN_ID", "local"), "kind": "bench-result/v1", "workload_fingerprint": fp,
        "validation": {"status": "passed" if ok else "failed", "checker": "executor negatives + veritor-zk-host verify on every dumped proof (producer)",
                       "detail": detail, "evidence": {"reps": reps, "verify": verdicts, "timeline": timeline, "negatives": summary}},
        "measurements": meas, "measurement_files": ["measurements.json", "prove.json", "verify.json"] + (["negatives.json"] if summary else []),
        "artifacts": [os.path.relpath(proofs_dir / "statement.bin", out)] + [v["proof"] for v in verdicts],
    }
    problems = contract.validate(result)
    from verity.ml.circuits.targets.target import TARGETS
    sp1 = next(c for c in tables.CANDIDATES if c.id == "SP1")
    tgt = TARGETS.get(target_name)
    reasons = tables.reject_reasons(result, [], None, tgt, sp1) if tgt else [f"unknown target {target_name!r}"]
    result["validation"]["evidence"]["table2_reasons_at_production"] = reasons
    tmp = out / "result.json.tmp"
    tmp.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, out / "result.json")
    if problems:
        print(f"contract: {len(problems)} problem(s): " + "; ".join(problems), file=sys.stderr)
    print(json.dumps({"backend": "sp1-bare", "format": fmt, "B": B, "t_total": t_total, "prove": rep["prove_seconds"], "serialize": rep["save_seconds"],
                      "shards": shards, "proof_bytes": rep["proof_bytes"], "cycles": execute["total_cycles"] if execute else None,
                      "verify_s": verify_s, "validation": result["validation"]["status"], "contract": problems or "conforms",
                      "table2_reasons": reasons}, indent=1))
    return 0 if ok and not problems else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vector", help="vector name under fixtures/typed-obligation-v0 (without .json); not for sp1-bare")
    ap.add_argument("--backend", choices=BACKENDS, required=True)
    ap.add_argument("--mode", default="core", choices=("core", "compressed"), help="sp1-prove / sp1-bare: proof mode")
    bare = ap.add_argument_group("sp1-bare (the Table 2 bare statement over a frozen batch)")
    bare.add_argument("--host", default=os.environ.get("VERITY_SP1_HOST", SP1_BARE_HOST), help="a relation-bare host binary")
    bare.add_argument("--backend-name", default=SP1_BARE_BACKEND_NAME, help="software.backend.name (Candidate SP1 recognises 'sp1...')")
    bare.add_argument("--variant", default=None,
                      help="JSON file naming a modified SP1 variant: {backend_name, label, toolchain, fork, verifier_name} (all optional)")
    bare.add_argument("--prover-env", action="append", default=[], metavar="KEY=VALUE",
                      help="an SP1 prover option for the GPU server's environment (e.g. SP1_WORKER_NUM_CORE_WORKERS=4); recorded")
    src = bare.add_mutually_exclusive_group()
    src.add_argument("--instances", help="directory of the frozen BF16 set (fixtures/bench-instances/v1 with the built arrays)")
    src.add_argument("--batch", help="a verity_sp1.format_instances file (sp1-format-instances/v1)")
    bare.add_argument("--lo", type=int, default=0)
    bare.add_argument("--hi", type=int, default=None)
    bare.add_argument("--reps", type=int, default=3)
    bare.add_argument("--warmup-vus", type=int, default=64)
    bare.add_argument("--vus-per-read", type=int, default=64)
    bare.add_argument("--layout", default="chunks", choices=("chunks", "indexed"),
                      help="guest input layout: every VU's rows in order, or a table of distinct rows and a row pair per VU (same statement)")
    bare.add_argument("--skip-negatives", action="store_true")
    bare.add_argument("--timeout", type=float, default=7200.0)
    bare.add_argument("--note", default="", help="what this run varies (recorded in the fingerprint)")
    ap.add_argument("--no-info", action="store_true", help="sp1: do not observe the guest identity (saves ~10 s; the fingerprint then carries only the expected one)")
    ap.add_argument("--out", default=None, help="output directory (default $RESEARCH_RUN_DIR)")
    ap.add_argument("--diagnostic", action="store_true",
                    help="sp1: the host is an unsound-* build (authentication skipped); label ARITHMETIC_DIAGNOSTIC, verify under its own key")
    ap.add_argument("--relation-only", action="store_true",
                    help="sp1, with --diagnostic: the host is the unsound-relation-only build; run the bare gate chain on the compact "
                         "input K || x || W built from the vector (statement_machinery excluded); the arithmetic baseline")
    a = ap.parse_args(argv)
    if a.backend == "sp1-bare":
        if not (a.instances or a.batch):
            ap.error("sp1-bare needs --instances DIR or --batch FILE")
        return run_sp1_bare(a)
    if not a.vector:
        ap.error("--vector is required")
    if a.diagnostic and a.backend == "reference":
        ap.error("--diagnostic applies to the sp1 backends")
    if a.relation_only and not a.diagnostic:
        ap.error("--relation-only is a diagnostic: give --diagnostic too")

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
        **campaign_fields(rec, a.backend, a.diagnostic, a.relation_only),
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
        # which bench-instances/v1 ids this vector is (by content): the K=1536 vector is vu-k1536 id 0; others null
        "instances": instances_of(rec),
    }
    started = time.perf_counter()
    if a.backend == "reference":
        validation, measurements, artifacts, _ = run_reference(rec, out)
        fingerprint["security"] = {"target": None, "note": "no proof: the transparent checker's own verdict, nothing established to a verifier"}
        fingerprint["software"] = {"backend": {"name": "verity_sp1.proofs.transparent", "version": None, "commit": sha}, "toolchain": {"python": python}}
        fingerprint["preprocessing"] = "none"
    else:
        validation, measurements, artifacts, _, identity = run_sp1(
            rec, out, prove=a.backend == "sp1-prove", mode=a.mode, want_info=not a.no_info, diagnostic=a.diagnostic,
            relation_only=a.relation_only,
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
    # proof-unit throughput: one obligation is one proof unit; the rate only exists where something was proved
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
