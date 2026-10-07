#!/usr/bin/env python3
"""A sampled cover of a vLLM row's proof units on SP1: many instances of one subcircuit, in chunks.

``sp1_ir_call.py`` proves one bundle whose typed object travels with it.  A cover proves a whole range of a
``vllm-vu-set/v1`` set, so the objects (one per chunk, JSON listing every word) stay in the recorded ``prepare`` run
(``objects/*.json.gz``) and the pod gets only what the guest reads -- each chunk's statement and witness -- with the
public values the object binds (``h_O || h_L || 01``, :func:`verity_sp1.proofs.binding.bound_public_values`):

    prepare   (needs verity_vllm)  per chunk of ``--chunk`` instances: the IR oracle reproduces the recorded outputs,
              lower (``verity_sp1.ir_call``), the Python reference accepts; three negatives on the first instance
              (wrong first / last output word committed consistently, a tampered output leaf), each rejected by the
              Python reference.  Writes ``bundle.tgz``, ``meta.json``, ``objects/``, ``result.json``.
    run       (the SP1 pod)  execute every negative and require verdict 00 with its own ``h_O || h_L``; prove every
              chunk (``--mode``, default compressed) and accept it only when ``verify`` passes under the APPROVED
              verifying key, the guest is sound, the statement digest matches and the public values are exactly the
              chunk's ``h_O || h_L || 01`` -- the facts ``verify_object`` checks, with the binding precomputed from the
              object.  Writes ``result.json``, ``proofs/chunkNNN-<mode>.bin``.

A non-producer verifier needs the prepare run (objects, bundle) and the proofs: it recomputes each object's
lowering, ``h_O`` and ``h_L`` itself and checks the proof under the pinned key.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import os
import sys
import tarfile
import time
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sp1_ir_call as spike
from verity_sp1.proofs.binding import bound_public_values
from verity_sp1.proofs.typed_obligation import object_digest

SP1_SECURITY_BITS = 100
#: Core-shard size of the pinned SP1 on these guests (cycles / shards over the recorded core runs: 364.5M / 77).
CYCLES_PER_SHARD = 4.73e6


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _gz(data: bytes) -> bytes:
    return gzip.compress(data, mtime=0)


def prepare(a) -> int:
    from verity_sp1.proofs.packed import PackedWitness, decode_packed_witness, encode_packed_witness
    from verity_numerical.bench import contract
    from verity_sp1.ir_call import lower_call

    set_dir = Path(a.set)
    manifest = json.loads((set_dir / "manifest.json").read_text())
    for name, f in manifest["files"].items():
        if _sha256((set_dir / name).read_bytes()) != f["sha256"]:
            raise SystemExit(f"{name} does not match the set's manifest")
    hi = a.hi if a.hi is not None else manifest["n"]
    spec, tables, table_sources, family = spike._bind(manifest["relation"])
    prov = manifest["provenance"]
    kw = {"tables": tables, "session": prov["run_root"], "compiled": prov["program_digests"]["r0"], "family": family}
    out = Path(a.out or os.environ["RESEARCH_RUN_DIR"])
    (out / "objects").mkdir(parents=True, exist_ok=True)
    entries: dict[str, bytes] = {}
    chunks, gates = [], None
    started = time.perf_counter()
    for c, lo in enumerate(range(a.lo, hi, a.chunk)):
        instances = spike._instances(spec, set_dir, manifest, lo, min(lo + a.chunk, hi))
        low = lower_call(spec, instances, **kw)  # raises unless the IR oracle reproduces every instance
        if not spike._python_verdict(low):
            raise SystemExit(f"the Python reference rejects chunk {c}")
        gates = low.gates
        name = f"chunk{c:03d}"
        entries[f"{name}/statement.bin"], entries[f"{name}/witness.bin"] = low.statement, low.witness
        obj_bytes = json.dumps(low.obj.to_json(), sort_keys=True).encode()
        (out / "objects" / f"{name}.json.gz").write_bytes(_gz(obj_bytes))
        chunks.append({"chunk": c, "lo": instances[0].unit, "hi": instances[-1].unit + 1, "object_digest": object_digest(low.obj),
                       "object_json_sha256": _sha256(obj_bytes), "statement_sha256": _sha256(low.statement),
                       "public_values": bound_public_values(low.obj, statement_format=3).hex(), "roots": low.roots})
        print(f"chunk {c}: [{chunks[-1]['lo']}, {chunks[-1]['hi']}) {len(low.statement)} + {len(low.witness)} B", flush=True)
    lower_s = time.perf_counter() - started

    first = spike._instances(spec, set_dir, manifest, a.lo, a.lo + 1)[0]
    flip = lambda j: replace(first, returns=[w ^ 1 if k == j else w for k, w in enumerate(first.returns)])
    negatives = {"wrong-first-output-word": lower_call(spec, [flip(0)], **kw, oracle=False),
                 "wrong-last-output-word": lower_call(spec, [flip(len(first.returns) - 1)], **kw, oracle=False)}
    one = lower_call(spec, [first], **kw)
    target = f"u{first.unit:06d}/out/0"
    trees = decode_packed_witness(one.witness).trees
    tampered = tuple(tuple(replace(leaf, value=bytes([leaf.value[0] ^ 1]) + leaf.value[1:]) if leaf.canonical_id == target else leaf
                           for leaf in tree) for tree in trees)
    negatives["tampered-output-leaf"] = replace(one, witness=encode_packed_witness(PackedWitness(tampered)))
    neg_meta = {}
    for name, low in negatives.items():
        if spike._python_verdict(low):
            raise SystemExit(f"the Python reference accepts the negative {name}")
        entries[f"neg/{name}/statement.bin"], entries[f"neg/{name}/witness.bin"] = low.statement, low.witness
        obj_bytes = json.dumps(low.obj.to_json(), sort_keys=True).encode()
        (out / "objects" / f"neg-{name}.json.gz").write_bytes(_gz(obj_bytes))
        neg_meta[name] = {"object_digest": object_digest(low.obj), "statement_sha256": _sha256(low.statement),
                          "public_values": bound_public_values(low.obj, 0, statement_format=3).hex()}

    meta = {"schema": "sp1-ir-call-cover/v1",
            "set": {"art": a.set_art, "name": manifest.get("set"), "content_digest": manifest["content_digest"],
                    "relation": manifest["relation"], "provenance": prov, "lo": a.lo, "hi": hi},
            "family": family, "definition": {"id": spec.id, "gates": gates}, "tables": table_sources,
            "chunks": chunks, "negatives": neg_meta, "source_sha": spike._git_sha(),
            "ir_oracle": f"evaluate_call of {spec.id} reproduces the recorded outputs of every instance in [{a.lo}, {hi})",
            "python_reference": "accepts every chunk, rejects every negative"}
    entries["meta.json"] = json.dumps(meta, indent=1, sort_keys=True).encode()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, data in sorted(entries.items()):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    bundle = _gz(buf.getvalue())
    (out / "bundle.tgz").write_bytes(bundle)
    (out / "meta.json").write_text(entries["meta.json"].decode() + "\n")
    m = contract.measurement
    result = {"schema": contract.RESULT_SCHEMA, "run_id": os.environ.get("RESEARCH_RUN_ID", "local"),
              "workload_fingerprint": {"workload": "sp1-ir-call-cover.prepare", "subcircuit": manifest["relation"]["definition"],
                                       "B": hi - a.lo, "chunk": a.chunk, "instances": meta["set"], "bundle_sha256": _sha256(bundle),
                                       "source_sha": meta["source_sha"], "proof_class": "NO_PROOF"},
              "validation": {"status": "passed", "checker": meta["python_reference"] + "; " + meta["ir_oracle"], "detail": "accepted",
                             "evidence": ["meta.json"]},
              "measurements": [m("lower_seconds", lower_s, "s"), m("chunks", len(chunks), "count"), m("bundle_bytes", len(bundle), "B")],
              "artifacts": ["bundle.tgz", "meta.json", *sorted(f"objects/{p.name}" for p in (out / "objects").iterdir())]}
    (out / "result.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"bundle_bytes": len(bundle), "chunks": len(chunks), "instances": hi - a.lo, "lower_seconds": round(lower_s, 1)}))
    return 0


def run(a) -> int:
    from verity_numerical.bench import contract
    from verity_sp1.host import APPROVED, Host

    out = Path(a.out or os.environ.get("RESEARCH_RUN_DIR") or ".").resolve()
    (out / "proofs").mkdir(parents=True, exist_ok=True)
    bundle_path = Path(os.path.expandvars(a.bundle))
    with tarfile.open(bundle_path) as tar:
        files = {x.name: tar.extractfile(x).read() for x in tar.getmembers() if x.isfile()}
    meta = json.loads(files["meta.json"])
    host = Host(timeout=None)
    m = contract.measurement
    problems: list[str] = []
    documents: dict = {"negatives": {}, "chunks": {}}
    started = time.perf_counter()
    observed = host.info()
    if observed != APPROVED:
        problems.append("the host's guest is not the approved identity")

    for name, expect in meta["negatives"].items():
        st, wi = files[f"neg/{name}/statement.bin"], files[f"neg/{name}/witness.bin"]
        report = host.execute(st, wi, typed_object_digest=expect["object_digest"])
        documents["negatives"][name] = {"verdict": report.verdict, "public_values": report.public_values.hex(), "total_cycles": report.total_cycles}
        if report.verdict or report.public_values.hex() != expect["public_values"]:
            problems.append(f"negative {name} was not rejected with its own h_O || h_L || 00")

    totals = {"prove_seconds": 0.0, "setup_seconds": 0.0, "verify_seconds": 0.0, "proof_bytes": 0, "cycles": 0, "stark_proofs_est": 0}
    per_chunk = []
    for ch in meta["chunks"]:
        name = f"chunk{ch['chunk']:03d}"
        st, wi = files[f"{name}/statement.bin"], files[f"{name}/witness.bin"]
        if _sha256(st) != ch["statement_sha256"]:
            problems.append(f"{name}: statement is not the one prepare recorded")
        proof = out / "proofs" / f"{name}-{a.mode}.bin"
        t = time.perf_counter()
        report = host.prove(st, wi, proof, typed_object_digest=ch["object_digest"], mode=a.mode)
        prove_wall = time.perf_counter() - t
        v = host.verify(proof, st)
        accepted = (v.ok and v.vk_hash == APPROVED.vk_hash and not v.unsound and v.statement_match is True
                    and v.public_values.hex() == ch["public_values"])
        if not accepted:
            problems.append(f"{name}: not accepted (ok={v.ok} vk={v.vk_hash} unsound={v.unsound} statement_match={v.statement_match} "
                            f"public values match={v.public_values.hex() == ch['public_values']})")
        cycles = report.execution.total_cycles
        core_shards = int(report.shards) if a.mode == "core" else math.ceil(cycles / CYCLES_PER_SHARD)
        starks = core_shards if a.mode == "core" else 2 * core_shards  # compressed: every core shard plus the recursion tree above it
        n = ch["hi"] - ch["lo"]
        row = {"chunk": ch["chunk"], "lo": ch["lo"], "hi": ch["hi"], "instances": n, "cycles": cycles, "prove_seconds": report.prove_seconds,
               "setup_seconds": report.setup_seconds, "prove_wall": prove_wall, "verify_seconds": report.verify_seconds,
               "proof_bytes": report.proof_bytes, "shards_reported": int(report.shards), "stark_proofs_est": starks, "accepted": accepted,
               "public_values": v.public_values.hex()}
        per_chunk.append(row)
        documents["chunks"][name] = {"prove": dict(report.document), "verify": dict(v.document)}
        for k in ("prove_seconds", "setup_seconds", "verify_seconds", "proof_bytes"):
            totals[k] += row[k]
        totals["cycles"] += cycles
        totals["stark_proofs_est"] += starks
        print(f"{name}: {n} inst {cycles / 1e6:.1f}M cyc prove {report.prove_seconds:.2f}s accepted={accepted}", flush=True)

    (out / "host.json").write_text(json.dumps(documents, indent=1, sort_keys=True) + "\n")
    B = sum(r["instances"] for r in per_chunk)
    achieved = round(-SP1_SECURITY_BITS + math.log2(max(totals["stark_proofs_est"], 1)), 3)
    measurements = [m(contract.T_TOTAL, totals["prove_seconds"]), m("t.joint.witness+encoding_commitment+arithmetic+lookup", totals["prove_seconds"]),
                    m("t.zk_additional", 0.0), m("fixed.setup_seconds", totals["setup_seconds"]), m("verifier.seconds", totals["verify_seconds"]),
                    m("transcript.bytes", totals["proof_bytes"]), m("total_cycles", totals["cycles"], "cycles"), m("chunks", len(per_chunk), "count"),
                    m("vu.count", B), m("vu.seconds_per_vu", totals["prove_seconds"] / B), m("vu.per_second", B / totals["prove_seconds"]),
                    m("soundness_bits", -achieved, "bits"), m("workload_wall", time.perf_counter() - started, "s")]
    (out / "chunks.json").write_text(json.dumps(per_chunk, indent=1) + "\n")
    (out / "measurements.json").write_text(json.dumps(measurements, indent=1) + "\n")
    fingerprint = {
        "workload": "sp1-ir-call-cover", "subcircuit": meta["set"]["relation"]["definition"], "statics": meta["set"]["relation"]["statics"],
        "instances": {"set_art": meta["set"]["art"], "set": meta["set"]["name"], "content_digest": meta["set"]["content_digest"],
                      "lo": meta["set"]["lo"], "hi": meta["set"]["hi"], "row": meta["set"]["provenance"].get("row"),
                      "run": meta["set"]["provenance"].get("run"), "run_root": meta["set"]["provenance"].get("run_root")},
        "B": B, "definition": meta["definition"], "family": meta["family"], "tables": meta["tables"], "statement_format": 3,
        "bundle_sha256": _sha256(bundle_path.read_bytes()), "prepare_source_sha": meta["source_sha"], "source_sha": spike._git_sha(),
        "proof_class": "NON_ZK_PROOF", "authentication": "included", "zk_mode": contract.ZK_NONE, "mode": a.mode,
        "prover": os.environ.get("SP1_PROVER", "cpu"), "guest": {"observed": asdict(observed) | {"guest_features": list(observed.guest_features)},
                                                                  "approved": observed == APPROVED},
        "security": {"target": -SP1_SECURITY_BITS, "achieved_log2": achieved,
                     "source": "SP1's own accounting: SP1_TARGET_BITS_OF_SECURITY = 100 per STARK proof (sp1-primitives 6.6.0 fri_params.rs)",
                     "union_bound": (f"over every STARK proof behind the {len(per_chunk)} {a.mode} proofs: {totals['stark_proofs_est']} "
                                     + ("core shards" if a.mode == "core" else f"(core shards estimated as cycles / {CYCLES_PER_SHARD:g}, x2 for the recursion tree)"))},
        "hardware": contract.probe_hardware(),
        "software": {"backend": {"name": "sp1", "version": observed.sp1_version, "guest_elf_sha256": observed.elf_sha256, "vk_hash": observed.vk_hash,
                                 "host_binary": os.environ.get("VERITY_SP1_HOST"), "prover": os.environ.get("SP1_PROVER", "cpu")},
                     "toolchain": {"python": sys.version.split()[0], "sp1_sdk": observed.sp1_version}},
        "retention": "retained", "preprocessing": "none offline; SP1 key setup per proof, reported as fixed.setup_seconds",
    }
    validation = {"status": "passed" if not problems else "failed",
                  "checker": "negatives rejected in the executor; each chunk's proof verifies under the APPROVED key with its h_O || h_L || 01",
                  "detail": "; ".join(problems) or "accepted", "evidence": ["host.json", "chunks.json"]}
    result = {"schema": contract.RESULT_SCHEMA, "run_id": os.environ.get("RESEARCH_RUN_ID", "local"), "workload_fingerprint": fingerprint,
              "validation": validation, "measurements": measurements, "measurement_files": ["measurements.json", "chunks.json"],
              "artifacts": ["host.json", "chunks.json", *(f"proofs/chunk{r['chunk']:03d}-{a.mode}.bin" for r in per_chunk)]}
    (out / "result.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"validation": validation["status"], "detail": validation["detail"][:500], "B": B, "t_total": totals["prove_seconds"]}))
    return 0 if not problems else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--set", required=True)
    p.add_argument("--set-art", default=None)
    p.add_argument("--lo", type=int, default=0)
    p.add_argument("--hi", type=int, default=None)
    p.add_argument("--chunk", type=int, required=True)
    p.add_argument("--out", default=None, help="default $RESEARCH_RUN_DIR")
    r = sub.add_parser("run")
    r.add_argument("--bundle", required=True, help="bundle.tgz from prepare; $VARS expand")
    r.add_argument("--mode", default="compressed", choices=("core", "compressed"))
    r.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    return prepare(a) if a.cmd == "prepare" else run(a)


if __name__ == "__main__":
    sys.exit(main())
