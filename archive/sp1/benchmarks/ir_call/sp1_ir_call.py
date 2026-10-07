#!/usr/bin/env python3
"""IR subcircuit instances from a vLLM export through the approved SP1 guest: the IR's own gate list, re-run and proved.

A ``vllm-vu-set/v1`` set (``integrations/vllm/verity_vllm/pipeline/vu_export.py``) holds instances of one subcircuit
of a real serving run -- operand and output words the row's replay opened and evaluated bit-exactly.  This workload
proves, for a range of those instances, that the IR evaluator's gate list for the subcircuit maps the committed
inputs to the committed outputs.  Two stages, because the IR definition and its MUFU table live in the frontend
(``verity_vllm``) and the pod needs only the backend:

    prepare   (any machine; needs verity_vllm)  read the set, check its files against its manifest, bind the
              definition the manifest names, run the IR oracle on every instance and require the recorded outputs,
              lower the instances (``verity_sp1.ir_call.lower_call``), check the Python reference (packed openings +
              ``check_relations``), build the negatives, write one ``bundle.tar``.
    run       (the SP1 pod; verity + verity_sp1)  execute every bundle (honest and negatives) in the SP1 executor and
              require exactly ``h_O || h_L || 01`` for the honest one and verdict 00 for each negative; with
              ``--prove``, one proof per mode, verified under the APPROVED key with its public values bound to the
              object (``verity_sp1.typed.verify_object``).  Writes ``$RESEARCH_RUN_DIR/result.json``.

    uv run python archive/sp1/benchmarks/ir_call/sp1_ir_call.py prepare --set DIR --lo 0 --hi 1 --out /tmp/b1
    uv run research run --on vy-sp1-evaluator --project verity --source . --cwd source --stage sp1.ir_call \\
        --send /tmp/b1/bundle.tar --env VERITY_SP1_HOST=/workspace/bin/veritor-zk-host-cuda --env SP1_PROVER=cuda \\
        --env PYTHONPATH=.:archive/sp1/python:benchmarks/numerical/python --require-result -- \\
        /root/.local/bin/uv run --no-project --with numpy --python 3.12 python archive/sp1/benchmarks/ir_call/sp1_ir_call.py run \\
        --bundle '$RESEARCH_RUN_DIR/inputs/bundle.tar' --prove core

Labels: ``proof_class = NON_ZK_PROOF`` (SP1 core/compressed), ``authentication = included`` (every boundary word is
opened against the roots in the statement).  The inputs/outputs roots are over the exported words, not the serving
run's own commitment tree: the export records the replay's opening, not its paths.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import sys
import tarfile
import time
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for candidate, module in (
    (REPO, "verity"),
    (REPO / "archive" / "sp1" / "python", "verity_sp1"),
    (REPO / "benchmarks" / "numerical" / "python", "verity_numerical"),
):
    try:
        __import__(module)
    except ImportError:
        sys.path.insert(0, str(candidate))

from verity_sp1.proofs.binding import bound_public_values
from verity_sp1.proofs.typed_obligation import TypedObligationSet, object_digest

SP1_SECURITY_BITS = 100
REDUCE = "veritor.reduce-mufu@1"
#: Sets whose port names are not the definition's parameters (``in0, in1, ...`` / ``out*``): (arguments, returns).
PORTS = {"Gemm_v1": (["x", "w"], ["y"])}
NEGATIVES = ("wrong-first-output-word", "wrong-last-output-word", "tampered-output-leaf")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --- prepare -------------------------------------------------------------------------------------------------------


def _bind(relation: dict):
    """The specialised IR definition a set's manifest names, its tables and the guest family (frontend imports stay here)."""

    from verity.ml.circuits.defs import bind
    from verity_sp1.proofs.reduce_mufu import div_full_rcp_index, mufu_sqrt_index, rsqrt_approx_index
    from verity_sp1.ir_call import Table
    from verity_vllm.program.kernels import rms_relation
    from verity_vllm.program.registry import b1

    statics = relation["statics"]
    definition = relation["definition"]
    t = rms_relation.tables("cuda")
    if definition in ("RMSNormFusedCuda_v2", "RMSNormTriton_v1"):
        eps = float.fromhex(json.loads(statics["EPS"])["f64"])
        if definition == "RMSNormFusedCuda_v2":
            return (bind(b1.RMSNormFusedCuda, N=int(statics["N"]), EPS=eps), {"rsq": Table("rsq", t.rsq, rsqrt_approx_index, t.source["rsq"]["sha256"])},
                    {"rsq": t.source["rsq"]}, REDUCE)
        tables = {"sqrt": Table("sqrt", t.sqrt, mufu_sqrt_index, t.source["sqrt"]["sha256"]),
                  "rcp": Table("rcp", t.rcp, div_full_rcp_index, t.source["rcp"]["sha256"])}
        return bind(b1.RMSNormTriton, N=int(statics["N"]), EPS=eps), tables, {k: t.source[k] for k in tables}, REDUCE
    if definition == "RoPE_v1":
        return bind(b1.RoPEHead, D=int(statics["D"])), {}, {}, REDUCE
    if definition == "SiluMul_v1":
        return bind(b1.SiluMul, I=int(statics["I"])), {}, {}, "veritor.elem-bf16@1"
    if definition == "Gemm_v1":
        return bind(b1.GemmCoordinate, K=int(statics["K"])), {}, {}, "veritor.tc-ampere-bf16@2"
    raise SystemExit(f"no binding for {relation['definition']} yet")


def _instances(spec, set_dir: Path, manifest: dict, lo: int, hi: int):
    """Instances ``[lo, hi)``: argument ports in the definition's parameter order (by name, else ``in0, in1, ...``),
    return ports ``out*`` in order."""

    import numpy as np
    from verity_sp1.ir_call import Instance

    ports = manifest["ports"]
    names = [name for name, _ in spec.signature()[0]]
    if manifest["relation"]["definition"] in PORTS:
        ins, outs = PORTS[manifest["relation"]["definition"]]
    else:
        ins = names if all(n in ports for n in names) else sorted((p for p in ports if not p.startswith("out")), key=lambda p: int(p[2:]))
        outs = sorted(p for p in ports if p.startswith("out"))
    arrays = {p: np.fromfile(set_dir / ports[p]["file"], dtype="<u2").reshape(manifest["n"], ports[p]["words"]) for p in ins + outs}
    return [Instance(i, [int(v) for p in ins for v in arrays[p][i]], [int(v) for p in outs for v in arrays[p][i]]) for i in range(lo, hi)]


def _python_verdict(lowered) -> bool:
    from verity.errors import ProtocolError
    from verity_vllm.check.codes import Reject
    from verity_sp1.proofs.lowering import check_relations
    from verity_sp1.proofs.packed import check_packed_openings, decode_packed_witness
    from verity_sp1.proofs.wire import decode_any_statement

    try:
        words = check_packed_openings(decode_any_statement(lowered.statement).expand(), decode_packed_witness(lowered.witness))
    except (Reject, ProtocolError):
        return False
    return check_relations(lowered.obj, words)


def _bundle_entry(lowered) -> dict[str, bytes]:
    obj = lowered.obj.to_json()
    return {"object.json": json.dumps(obj, sort_keys=True).encode(), "statement.bin": lowered.statement, "witness.bin": lowered.witness}


def prepare(a) -> int:
    from dataclasses import replace

    from verity_sp1.proofs.packed import PackedWitness, decode_packed_witness, encode_packed_witness
    from verity_sp1.ir_call import lower_call

    set_dir = Path(a.set)
    manifest = json.loads((set_dir / "manifest.json").read_text())
    for name, f in manifest["files"].items():
        if _sha256((set_dir / name).read_bytes()) != f["sha256"]:
            raise SystemExit(f"{name} does not match the set's manifest")
    hi = a.hi if a.hi is not None else manifest["n"]
    spec, tables, table_sources, family = _bind(manifest["relation"])
    instances = _instances(spec, set_dir, manifest, a.lo, hi)
    prov = manifest["provenance"]
    session, compiled = prov["run_root"], prov["program_digests"]["r0"]
    started = time.perf_counter()
    honest = lower_call(spec, instances, tables=tables, session=session, compiled=compiled, family=family)  # raises unless the IR oracle reproduces every instance
    lower_s = time.perf_counter() - started
    if not _python_verdict(honest):
        raise SystemExit("the Python reference rejects the honest bundle")
    first = instances[0]
    last = len(first.returns) - 1

    def flipped(j: int):
        return replace(first, returns=[w ^ 1 if k == j else w for k, w in enumerate(first.returns)])

    negatives = {
        "wrong-first-output-word": lower_call(spec, [flipped(0)], tables=tables, session=session, compiled=compiled, family=family, oracle=False),
        "wrong-last-output-word": lower_call(spec, [flipped(last)], tables=tables, session=session, compiled=compiled, family=family, oracle=False),
    }
    one = lower_call(spec, instances[:1], tables=tables, session=session, compiled=compiled, family=family)
    # one byte of the first output leaf's value, flipped: the leaf no longer authenticates
    target = f"u{first.unit:06d}/out/0"
    trees = decode_packed_witness(one.witness).trees
    tampered = tuple(tuple(replace(leaf, value=bytes([leaf.value[0] ^ 1]) + leaf.value[1:]) if leaf.canonical_id == target else leaf
                           for leaf in tree) for tree in trees)
    assert tampered != trees
    negatives["tampered-output-leaf"] = replace(one, witness=encode_packed_witness(PackedWitness(tampered)))
    python_negatives = {name: _python_verdict(low) for name, low in negatives.items()}
    if any(python_negatives.values()):
        raise SystemExit(f"the Python reference accepts a negative: {python_negatives}")

    meta = {
        "schema": "sp1-ir-call-bundle/v1",
        "set": {"art": a.set_art, "name": manifest.get("set"), "content_digest": manifest["content_digest"], "relation": manifest["relation"],
                "provenance": prov, "lo": a.lo, "hi": hi},
        "family": family,
        "definition": {"id": spec.id, "gates": honest.gates, "incoming": len(honest.obj.subcircuits[0].incoming),
                       "outgoing": len(honest.obj.subcircuits[0].outgoing)},
        "tables": table_sources,
        "roots": honest.roots,
        "object_digest": object_digest(honest.obj),
        "statement_sha256": _sha256(honest.statement),
        "public_values": bound_public_values(honest.obj, statement_format=3).hex(),
        "ir_oracle": f"verity.ml.circuits.evaluation.reference evaluate_call of {spec.id} reproduces the recorded outputs of instances [{a.lo}, {hi})",
        "python_reference": {"honest": True, **{k: v for k, v in python_negatives.items()}},
        "negatives": {name: {"object_digest": object_digest(low.obj), "statement_sha256": _sha256(low.statement)} for name, low in negatives.items()},
        "source_sha": _git_sha(),
    }
    out = Path(a.out or os.environ["RESEARCH_RUN_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        entries = {f"honest/{k}": v for k, v in _bundle_entry(honest).items()}
        for name, low in negatives.items():
            entries |= {f"{name}/{k}": v for k, v in _bundle_entry(low).items()}
        entries["meta.json"] = json.dumps(meta, indent=1, sort_keys=True).encode()
        for name, data in sorted(entries.items()):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    (out / "bundle.tar").write_bytes(buf.getvalue())
    (out / "meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    from verity_numerical.bench import contract

    m = contract.measurement
    result = {
        "schema": contract.RESULT_SCHEMA, "run_id": os.environ.get("RESEARCH_RUN_ID", "local"),
        "workload_fingerprint": {"workload": "sp1-ir-call.prepare", "subcircuit": manifest["relation"]["definition"], "B": hi - a.lo,
                                 "instances": meta["set"], "object_digest": meta["object_digest"], "statement_sha256": meta["statement_sha256"],
                                 "bundle_sha256": _sha256(buf.getvalue()), "source_sha": meta["source_sha"], "proof_class": "NO_PROOF"},
        "validation": {"status": "passed", "checker": "IR oracle == recorded outputs; Python reference accepts honest, rejects every negative",
                       "detail": "accepted", "evidence": ["meta.json"]},
        "measurements": [m("lower_seconds", lower_s, "s"), m("gates", honest.gates, "count"), m("statement_bytes", len(honest.statement), "B"),
                         m("witness_bytes", len(honest.witness), "B")],
        "artifacts": ["bundle.tar", "meta.json"],
    }
    (out / "result.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"bundle": str(out / "bundle.tar"), "bytes": len(buf.getvalue()), "sha256": _sha256(buf.getvalue()),
                      "instances": hi - a.lo, "gates": honest.gates, "statement_bytes": len(honest.statement),
                      "witness_bytes": len(honest.witness), "lower_seconds": round(lower_s, 3)}, indent=1))
    return 0


def _git_sha() -> str | None:
    import subprocess

    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return os.environ.get("RESEARCH_SOURCE_COMMIT")


# --- run -----------------------------------------------------------------------------------------------------------


def _read_bundle(path: Path) -> tuple[dict, dict[str, dict]]:
    with tarfile.open(path) as tar:
        files = {m.name: tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}
    meta = json.loads(files["meta.json"])
    bundles = {}
    for name in ("honest", *NEGATIVES):
        if f"{name}/object.json" in files:
            bundles[name] = {"obj": TypedObligationSet.from_json(json.loads(files[f"{name}/object.json"])),
                             "statement": files[f"{name}/statement.bin"], "witness": files[f"{name}/witness.bin"]}
    return meta, bundles


def run(a) -> int:
    from verity_numerical.bench import contract
    from verity_sp1.host import APPROVED, Host
    from verity_sp1.typed import prove_object, verify_object

    out = Path(a.out or os.environ.get("RESEARCH_RUN_DIR") or ".").resolve()
    out.mkdir(parents=True, exist_ok=True)
    bundle_path = Path(os.path.expandvars(a.bundle))
    meta, bundles = _read_bundle(bundle_path)
    host = Host(timeout=None)
    m = contract.measurement
    measurements: list[dict] = []
    problems: list[str] = []
    documents: dict = {}
    started = time.perf_counter()
    observed = host.info()
    identity = {"expected": asdict(APPROVED) | {"guest_features": list(APPROVED.guest_features)},
                "observed": asdict(observed) | {"guest_features": list(observed.guest_features)}, "approved": observed == APPROVED}
    if not identity["approved"]:
        problems.append("the host's guest is not the approved identity")

    honest = bundles["honest"]
    expected_pv = bytes.fromhex(meta["public_values"])
    if object_digest(honest["obj"]) != meta["object_digest"] or _sha256(honest["statement"]) != meta["statement_sha256"]:
        problems.append("the bundle's honest object or statement is not the one prepare recorded")
    executions = {}
    for name, b in bundles.items():
        t = time.perf_counter()
        report = host.execute(b["statement"], b["witness"], typed_object_digest=object_digest(b["obj"]))
        wall = time.perf_counter() - t
        executions[name] = {"verdict": report.verdict, "public_values": report.public_values.hex(), "total_cycles": report.total_cycles,
                            "wall_s": wall, "cycle_tracker": dict(report.cycle_tracker)}
        if name == "honest":
            if report.public_values != expected_pv:
                problems.append("the executor's public values are not h_O || h_L || 01 of the honest object")
            measurements += [m("execute_wall", wall, "s"), m("total_cycles", report.total_cycles, "cycles")]
            measurements += [m(f"cycles.{k}", v, "cycles") for k, v in sorted(report.cycle_tracker.items())]
        elif report.verdict or report.public_values[-1:] != b"\x00":
            problems.append(f"negative {name} was not rejected by the guest")
    documents["execute"] = executions

    B = meta["set"]["hi"] - meta["set"]["lo"]
    security = {"target": None, "note": "executor only: nothing established to a verifier"}
    proofs = {}
    for mode in a.prove:
        proof = out / f"proof-{mode}.bin"
        t = time.perf_counter()
        report = prove_object(host, honest["obj"], honest["witness"], proof, statement_format=3, mode=mode)
        prove_wall = time.perf_counter() - t
        t = time.perf_counter()
        acceptance = verify_object(host, honest["obj"], proof, statement_format=3)
        verify_wall = time.perf_counter() - t
        proofs[mode] = {"prove": dict(report.document), "verify": dict(acceptance.verify.document),
                        "acceptance": {"accepted": acceptance.accepted, "reasons": acceptance.reasons(),
                                       "proof_verifies": acceptance.proof_verifies, "approved_key": acceptance.approved_key,
                                       "binding_ok": acceptance.binding.ok, "verdict": acceptance.binding.verdict, "sound": acceptance.sound}}
        if not acceptance.accepted:
            problems += [f"{mode}: {r}" for r in acceptance.reasons()]
        if report.execution.public_values != expected_pv:
            problems.append(f"{mode}: the proof's public values are not h_O || h_L || 01")
        shards = int(report.shards)
        p = f"{mode}."
        measurements += [
            m(p + "prove_wall", prove_wall, "s"), m(p + "setup_seconds", report.setup_seconds, "s"), m(p + "prove_seconds", report.prove_seconds, "s"),
            m(p + "host_verify_seconds", report.verify_seconds, "s"), m(p + "verify_wall", verify_wall, "s"),
            m(p + "proof_bytes", report.proof_bytes, "B"), m(p + "shards", shards, "count"),
            m(p + "soundness_bits", SP1_SECURITY_BITS - math.log2(max(shards, 1)), "bits"),
            m(p + "seconds_per_instance", report.prove_seconds / B, "s"),
        ]
        if mode == a.prove[0]:
            measurements += [m("fixed.setup_seconds", report.setup_seconds), m("t.joint.witness+encoding_commitment+arithmetic+lookup", report.prove_seconds),
                             m("t.zk_additional", 0.0), m(contract.T_TOTAL, report.prove_seconds), m("verifier.seconds", report.verify_seconds),
                             m("transcript.bytes", report.proof_bytes)]
            security = {"target": -SP1_SECURITY_BITS, "achieved_log2": round(-SP1_SECURITY_BITS + math.log2(max(shards, 1)), 3),
                        "source": "SP1's own accounting: SP1_TARGET_BITS_OF_SECURITY = 100 per STARK proof (sp1-primitives 6.6.0 fri_params.rs), "
                                  "compile-time constants of the pinned SP1",
                        "union_bound": f"{shards} shard proofs in the {mode} proof: -100 + log2({shards})" if mode == "core" else
                                       f"{mode} proof: one recursion-compressed STARK over {shards} core shards"}
    documents["prove"] = proofs
    (out / "host.json").write_text(json.dumps(documents, indent=1, sort_keys=True) + "\n")
    measurements += [m("vu.count", B), m("workload_wall", time.perf_counter() - started, "s")]
    total = next((x["value"] for x in measurements if x["name"] == contract.T_TOTAL), None)
    if total:
        measurements += [m("vu.seconds_per_vu", total / B), m("vu.per_second", B / total)]

    fingerprint = {
        "workload": "sp1-ir-call",
        "subcircuit": meta["set"]["relation"]["definition"],
        "statics": meta["set"]["relation"]["statics"],
        "instances": {"set_art": meta["set"]["art"], "set": meta["set"]["name"], "content_digest": meta["set"]["content_digest"],
                      "lo": meta["set"]["lo"], "hi": meta["set"]["hi"], "row": meta["set"]["provenance"].get("row"),
                      "run": meta["set"]["provenance"].get("run"), "run_root": meta["set"]["provenance"].get("run_root")},
        "B": B,
        "definition": meta["definition"],
        "family": meta.get("family", REDUCE),
        "statement_format": 3,
        "object_digest": meta["object_digest"],
        "statement_sha256": meta["statement_sha256"],
        "roots": meta["roots"],
        "tables": meta["tables"],
        "bundle_sha256": _sha256(bundle_path.read_bytes()),
        "prepare_source_sha": meta["source_sha"],
        "source_sha": _git_sha(),
        "proof_class": "NON_ZK_PROOF" if a.prove else "NO_PROOF",
        "authentication": "included",
        "zk_mode": contract.ZK_NONE,
        "modes": list(a.prove),
        "prover": os.environ.get("SP1_PROVER", "cpu"),
        "guest": identity,
        "security": security,
        "hardware": contract.probe_hardware(),
        "software": {"backend": {"name": "sp1", "version": observed.sp1_version, "guest_elf_sha256": observed.elf_sha256, "vk_hash": observed.vk_hash,
                                 "host_binary": os.environ.get("VERITY_SP1_HOST"), "prover": os.environ.get("SP1_PROVER", "cpu")},
                     "toolchain": {"python": sys.version.split()[0], "sp1_sdk": observed.sp1_version}},
        "retention": "retained",
        "preprocessing": "none offline; SP1 setup (keys from the guest ELF) runs in the host every time, reported as fixed.setup_seconds",
    }
    validation = {"status": "passed" if not problems else "failed",
                  "checker": "sp1 executor on honest + negatives; verify_object (approved key, h_O || h_L || 01 binding) per proof",
                  "detail": "; ".join(problems) or "accepted", "evidence": ["host.json"]}
    (out / "measurements.json").write_text(json.dumps(measurements, indent=1) + "\n")
    result = {"schema": contract.RESULT_SCHEMA, "run_id": os.environ.get("RESEARCH_RUN_ID", "local"), "workload_fingerprint": fingerprint,
              "validation": validation, "measurements": measurements, "measurement_files": ["measurements.json"],
              "artifacts": ["host.json", *(f"proof-{mode}.bin" for mode in a.prove)]}
    (out / "result.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"validation": validation, "measurements": {x["name"]: x["value"] for x in measurements if not x["name"].startswith("cycles.")}}, indent=1))
    return 0 if not problems else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--set", required=True, help="a vllm-vu-set/v1 tree (research data fetch <art>)")
    p.add_argument("--set-art", default=None, help="the set's art id, recorded")
    p.add_argument("--lo", type=int, default=0)
    p.add_argument("--hi", type=int, default=None)
    p.add_argument("--out", default=None, help="default $RESEARCH_RUN_DIR")
    r = sub.add_parser("run")
    r.add_argument("--bundle", required=True, help="bundle.tar from prepare; $VARS expand (research ships it to $RESEARCH_RUN_DIR/inputs/)")
    r.add_argument("--prove", nargs="*", default=[], choices=("core", "compressed"))
    r.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    return prepare(a) if a.cmd == "prepare" else run(a)


if __name__ == "__main__":
    sys.exit(main())
