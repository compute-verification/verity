"""The minimal independently re-verifiable bundle (contract 16.8): pack, unpack, verify.

Specification: ``docs/vllm-poc/bundle-spec.md``.  In one sentence: a tarball holding the capture
run directory with its commitments (the *bundle* of ``verity_vllm.verify``), the production
record it must agree with, the verifier output that was produced from it, and a top-level
``BUNDLE.json`` naming every member with its sha256 -- so that a fresh process on another
machine can (1) check that it received exactly what was produced, (2) re-run the fresh
verifier on it with the recorded seed and reproduce the recorded challenge, or (3) run it with
a new seed.  Weights are referenced by checkpoint sha (``manifest.model.weights_sha256`` and
``weights/index.json`` with a per-tensor sha256 and leaf hash) and not copied unless asked.

Layout inside the tarball (one top-level directory named after the experiment)::

    <name>/BUNDLE.json                  this file: schema, ids, digests, sha256 of every member, commands
    <name>/bundle/manifest.json         the profile (profile_id = sha256[:12] of these bytes)
    <name>/bundle/description.json      the circuit checks are executed against
    <name>/bundle/required.json         frozen required-interface set
    <name>/bundle/commitments.json      every root (ru/<request>/<step>, boundary/<request>, weights)
    <name>/bundle/ru/**                 boundary.json, index.json, values.bin per replay unit
    <name>/bundle/weights/index.json    whole-tensor leaves: canonical id, dtype, shape, sha256, leaf
    <name>/bundle/weights/values.bin    only with weights="copy"
    <name>/production/production.jsonl  the production record (bound by sha256 in BUNDLE.json)
    <name>/verify/**                    challenge.json, report.json | verify.json, sampling_plan.json,
                                        table*.jsonl, negatives/*.json (whatever the verifier wrote)

Python::

    from verity_vllm import bundle
    path = bundle.pack(run_dir, "B0.tar.gz", production_record=..., verify_dir=..., name="B0-...")
    unpacked = bundle.unpack(path, dest)                 # verifies every sha256; raises BundleError
    result = bundle.verify_tarball(path, seed=7, out_dir=..., torch_mode="never", plugins=[...])

Command line::

    python -m verity_vllm.bundle pack RUN_DIR --out B0.tar.gz --production production.jsonl \\
        [--verify-dir RUN_DIR/verify] [--name ID] [--weights omit|copy] [--extra PATH ...]
    python -m verity_vllm.bundle inspect B0.tar.gz
    python -m verity_vllm.bundle unpack B0.tar.gz DEST
    python -m verity_vllm.bundle verify B0.tar.gz --seed 7 [--out DIR] [--sample K] [--torch never]
        [--plugin module:function ...] [--negatives none] [--weights-values weights/values.bin]
    python -m verity_vllm.bundle rederive B1.tar.gz [--out DIR]     # staged challenge (verity_vllm.challenge)

``rederive`` is the third-party check of a bundle whose verifier used the staged, verifier-owned
challenge (``verify/challenge_transcript.json``, GUIDANCE C1): CPU only, it re-derives ``T_0..T_3``,
``r_q``/``r_s`` from the revealed seed, the selected replay units, the VU population, the certified
and the coverage sample, and the statement fields the bundle can answer for, and writes
``rederive.json``.  Re-running the relation checks and the C3 bindings on that transcript is
``python -m verity_vllm_adapter.verify <bundle> --transcript <transcript>`` (GPU + checkpoint).

``verify`` unpacks into ``--out`` (or a temporary directory), runs ``verity_vllm.verify.verify`` on
``<name>/bundle`` with ``--production-record <name>/production/production.jsonl``, and writes
``reverify.json`` next to the verifier output: the recorded verdict and challenge digest, the
fresh ones, and whether the challenge was reproduced (same seed derivation, same roots -> same
sample).  A recorded challenge from the adapter's verifier (raw seed, no root mixing) is reported
as ``not comparable`` rather than as a mismatch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import hashing
from .capture import CaptureReader

BUNDLE_SCHEMA = "verity-vllm/bundle/v1"
BUNDLE_FILE = "BUNDLE.json"
BUNDLE_SUBDIR = "bundle"
PRODUCTION_SUBDIR = "production"
VERIFY_SUBDIR = "verify"
PRODUCTION_FILE = "production.jsonl"
CORE_FILES = ("manifest.json", "description.json", "required.json", "commitments.json")


class BundleError(Exception):
    pass


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _walk(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file())


@dataclass
class Member:
    arcname: str  # path inside the tarball, below <name>/
    source: Path
    sha256: str
    bytes: int

    def to_json(self) -> dict[str, Any]:
        return {"sha256": self.sha256, "bytes": self.bytes}


@dataclass
class PackPlan:
    name: str
    run_dir: Path
    members: list[Member] = field(default_factory=list)
    omitted: list[str] = field(default_factory=list)

    def add(self, arcname: str, source: Path) -> None:
        self.members.append(Member(arcname, source, _sha256_file(source), source.stat().st_size))

    @property
    def total_bytes(self) -> int:
        return sum(m.bytes for m in self.members)


def plan(
    run_dir: str | os.PathLike[str],
    *,
    name: str,
    production_record: str | os.PathLike[str] | None,
    verify_dir: str | os.PathLike[str] | None,
    weights: str = "omit",
    extra: Iterable[str | os.PathLike[str]] = (),
) -> PackPlan:
    """Decide what goes into the tarball (no I/O beyond hashing)."""
    run_dir = Path(run_dir)
    if weights not in ("omit", "copy"):
        raise BundleError("weights must be 'omit' or 'copy'")
    for name_ in ("manifest.json", "description.json", "commitments.json"):
        if not (run_dir / name_).exists():
            raise BundleError(f"{run_dir}: {name_} is missing; not a bundle")
    if not (run_dir / "ru").is_dir():
        raise BundleError(f"{run_dir}: ru/ is missing; the committed values are the bundle")
    p = PackPlan(name, run_dir)
    for f in CORE_FILES:
        if (run_dir / f).exists():
            p.add(f"{BUNDLE_SUBDIR}/{f}", run_dir / f)
        elif f != "required.json":
            raise BundleError(f"{run_dir}: {f} is missing")
    for path in _walk(run_dir / "ru"):
        p.add(f"{BUNDLE_SUBDIR}/{path.relative_to(run_dir).as_posix()}", path)
    wdir = run_dir / "weights"
    if (wdir / "index.json").exists():
        p.add(f"{BUNDLE_SUBDIR}/weights/index.json", wdir / "index.json")
        for v2_file in ("leaves.bin", "conversion.json"):  # capture format v2 (row-block leaves): the leaf hashes travel with the index
            if (wdir / v2_file).exists():
                p.add(f"{BUNDLE_SUBDIR}/weights/{v2_file}", wdir / v2_file)
        if (wdir / "values.bin").exists():
            if weights == "copy":
                p.add(f"{BUNDLE_SUBDIR}/weights/values.bin", wdir / "values.bin")
            else:
                p.omitted.append(f"{BUNDLE_SUBDIR}/weights/values.bin (referenced by checkpoint sha)")
    for aux in ("capture.json", "replay.json", "capture_summary.json", "kv_audit.json",
                # a selective run (verity_vllm.selective, RC): what was committed before r_q and the prover's own record
                "commitments_phase0.json", "phase0.json", "selective_prover.json", "description_observed.json"):
        if (run_dir / aux).exists():
            p.add(f"{BUNDLE_SUBDIR}/{aux}", run_dir / aux)
    if production_record is not None:
        p.add(f"{PRODUCTION_SUBDIR}/{PRODUCTION_FILE}", Path(production_record))
    if verify_dir is not None:
        vdir = Path(verify_dir)
        if not vdir.is_dir():
            raise BundleError(f"{vdir}: verify dir is missing")
        for path in _walk(vdir):
            if path.suffix in (".json", ".jsonl", ".txt", ".tail", ".log"):
                p.add(f"{VERIFY_SUBDIR}/{path.relative_to(vdir).as_posix()}", path)
    for e in extra:
        e = Path(e)
        if e.is_dir():
            for path in _walk(e):
                p.add(f"extra/{e.name}/{path.relative_to(e).as_posix()}", path)
        else:
            p.add(f"extra/{e.name}", e)
    return p


def _bundle_manifest(p: PackPlan, *, created: str) -> dict[str, Any]:
    reader = CaptureReader(p.run_dir)
    manifest = reader.manifest
    commitments = json.loads((p.run_dir / "commitments.json").read_text())
    description = json.loads((p.run_dir / "description.json").read_text())
    required_path = p.run_dir / "required.json"
    required = json.loads(required_path.read_text()) if required_path.exists() else None
    by_arc = {m.arcname: m for m in p.members}
    prod = by_arc.get(f"{PRODUCTION_SUBDIR}/{PRODUCTION_FILE}")
    verify_files = sorted(a for a in by_arc if a.startswith(VERIFY_SUBDIR + "/"))
    recorded: dict[str, Any] = {}
    staged: dict[str, Any] | None = None
    for candidate in ("verify/challenge.json",):
        if candidate in by_arc:
            ch = json.loads(by_arc[candidate].source.read_text())
            inner = ch.get("challenge", ch)
            if isinstance(inner, dict) and inner.get("mode") in ("security", "diagnostic"):
                # the adapter verifier's staged challenge (verity_vllm.challenge): challenge.json is the
                # relations.Challenge fields plus a `challenge` block describing the source
                staged = inner
                recorded["challenge_digest"] = ch.get("digest")
                recorded["sample_size"] = len(ch.get("sample", []))
                recorded["challenge_mode"] = inner.get("mode")
                recorded["challenge_source"] = inner.get("source")
                recorded["policy"] = inner.get("policy")
                recorded["phase_digests"] = inner.get("phase_digests")
                recorded["transcript"] = f"{VERIFY_SUBDIR}/challenge_transcript.json"
                recorded["seed"] = ch.get("seed")  # diagnostic mode only; None in security mode
                recorded["seed_derivation"] = (
                    f"staged ({inner.get('mode')}): r_q = HMAC(seed, T_0), r_s = HMAC(seed, T_2); "
                    + ("verifier secret seed, revealed in the transcript" if inner.get("mode") == "security"
                       else "public --seed (diagnostic, not a security instantiation)")
                )
                continue
            recorded["challenge_digest"] = inner.get("digest")
            recorded["seed"] = ch.get("seed")
            recorded["effective_seed"] = ch.get("effective_seed")
            recorded["seed_derivation"] = (
                ch.get("derivation")
                or ("raw --seed (adapter verifier; not mixed with the roots)" if "effective_seed" not in ch else None)
            )
            recorded["sample_size"] = len(inner.get("sample", []))
            recorded["challenge_mode"] = "diagnostic"
    selective = "verify/selective_verify.json" in by_arc
    for candidate in ("verify/report.json", "verify/verify.json", "verify/selective_verify.json"):
        if candidate in by_arc:
            rep = json.loads(by_arc[candidate].source.read_text())
            v = rep.get("verdict")
            if isinstance(v, dict):
                recorded["verdict"] = v.get("result")
            elif isinstance(rep.get("table1"), dict):
                recorded["verdict"] = rep["table1"].get("result")
            elif isinstance(v, str) and candidate.endswith("selective_verify.json"):
                # RC's selective verifier (verity_vllm_adapter.selective_verifier): one verdict string over J
                recorded["verdict"] = v
                recorded["protocol"] = "selective"
                recorded["selected_requests"] = list(rep.get("selected") or [])
                recorded["requests_total"] = rep.get("requests_total")
                recorded["verifier_output"] = candidate
            break
    for candidate in ("verify/table1.jsonl",):
        if candidate in by_arc and "verdict" not in recorded:
            line = by_arc[candidate].source.read_text().splitlines()[0]
            recorded["verdict"] = json.loads(line).get("result")
    trees = commitments.get("trees", {})
    if staged is not None and selective:
        reverify_cmd = {
            "command": (
                f"python -m verity_vllm.bundle rederive {p.name}.tar.gz --out <DIR>   # CPU only: re-derive the selective "
                "transcript (T_0..T_3, r_q -> J, r_s, both samples), the selective statement, first-stage completeness over J "
                "and the phase-0 commitments from the bundled records"
            ),
            "equivalent": (
                "the protocol itself is interactive (prover and verifier are two processes: verity_vllm_adapter.selective_prover / "
                "selective_verifier); a fresh r_q needs the prover, i.e. the checkpoint and the GPU. The relation checks over the "
                f"recorded J re-run on CPU from the bundle: python -m verity_vllm_adapter.coordinate_check <DIR>/{p.name}/{BUNDLE_SUBDIR} --out <DIR>/coord "
                "(weights/values.bin needed beside the run)"
            ),
        }
    elif staged is not None:
        reverify_cmd = {
            "command": (
                f"python -m verity_vllm.bundle rederive {p.name}.tar.gz --out <DIR>   # CPU only: re-derive the "
                "recorded transcript (T_0..T_3, r_q, r_s, both samples) from the bundled commitments"
            ),
            "equivalent": (
                f"python -m verity_vllm.bundle unpack {p.name}.tar.gz <DIR> && "
                f"python -m verity_vllm_adapter.verify <DIR>/{p.name}/{BUNDLE_SUBDIR} "
                f"--transcript <DIR>/{p.name}/{VERIFY_SUBDIR}/challenge_transcript.json "
                f"--production <DIR>/{p.name}/{PRODUCTION_SUBDIR} --checkpoints manifests/checkpoints.json "
                "[--weights-from-model <MODEL_DIR>] --out <DIR>/verify   # GPU + pinned checkpoint: re-derives the "
                "transcript, re-binds and re-checks (MODEL_DIR when the checkpoint is not at the manifest's local_path)"
            ),
            "fresh_challenge": (
                f"python -m verity_vllm_adapter.verify <DIR>/{p.name}/{BUNDLE_SUBDIR} --challenge security "
                f"--production <DIR>/{p.name}/{PRODUCTION_SUBDIR} [--weights-from-model <MODEL_DIR>] --out <DIR>/verify-fresh"
            ),
        }
    else:
        reverify_cmd = {
            "command": (
                f"python -m verity_vllm.bundle verify {p.name}.tar.gz --seed {recorded.get('seed', '<N>')} "
                "--out <DIR> [--torch require --plugin verity_vllm_adapter.relations_gpu:install]"
            ),
            "equivalent": (
                f"python -m verity_vllm.bundle unpack {p.name}.tar.gz <DIR> && "
                f"python -m verity_vllm.verify <DIR>/{p.name}/{BUNDLE_SUBDIR} --seed <N> "
                f"--production-record <DIR>/{p.name}/{PRODUCTION_SUBDIR}/{PRODUCTION_FILE}"
            ),
        }
    return {
        "schema": BUNDLE_SCHEMA,
        "name": p.name,
        "created_utc": created,
        "profile_id": reader.profile_id,
        "case": manifest.get("case"),
        "model": manifest.get("model"),
        "digests": {
            "manifest_sha256": by_arc[f"{BUNDLE_SUBDIR}/manifest.json"].sha256,
            "description_digest": hashing.json_digest(description).hex(),
            "required_digest": hashing.json_digest(required).hex() if required is not None else None,
            "commitments_digest": hashing.json_digest(commitments).hex(),
            "production_record_sha256": prod.sha256 if prod else None,
        },
        "roots": {
            "trees": len(trees),
            "ru": sum(1 for t in trees.values() if t.get("kind") == "ru"),
            "boundary": sum(1 for t in trees.values() if t.get("kind") == "boundary"),
            "weights": "weights" in trees,
            "leaves": sum(int(t.get("leaf_count", 0)) for t in trees.values()),
        },
        "weights": {
            "included": f"{BUNDLE_SUBDIR}/weights/values.bin" in by_arc,
            "referenced_by": (manifest.get("model") or {}).get("weights_sha256"),
            "index": f"{BUNDLE_SUBDIR}/weights/index.json" in by_arc,
        },
        "recorded_verification": recorded or None,
        "verify_files": verify_files,
        "omitted": list(p.omitted),
        "reverify": reverify_cmd,
        "members": {m.arcname: m.to_json() for m in p.members},
        "total_bytes": p.total_bytes,
    }


def pack(
    run_dir: str | os.PathLike[str],
    out: str | os.PathLike[str],
    *,
    production_record: str | os.PathLike[str] | None = None,
    verify_dir: str | os.PathLike[str] | None = None,
    name: str | None = None,
    weights: str = "omit",
    extra: Iterable[str | os.PathLike[str]] = (),
    compress: bool | None = None,
) -> Path:
    """Write the tarball; returns its path.  ``compress`` defaults to the suffix (``.gz``)."""
    out = Path(out)
    run_dir = Path(run_dir)
    if name is None:
        name = out.name[: -len(".tar.gz")] if out.name.endswith(".tar.gz") else out.stem
    p = plan(run_dir, name=name, production_record=production_record, verify_dir=verify_dir,
             weights=weights, extra=extra)
    doc = _bundle_manifest(p, created=time.strftime("%Y%m%dT%H%MZ", time.gmtime()))
    if compress is None:
        compress = out.name.endswith(".gz")
    out.parent.mkdir(parents=True, exist_ok=True)
    mode = "w:gz" if compress else "w"
    with tarfile.open(out, mode) as tar:
        data = (json.dumps(doc, indent=1, sort_keys=True) + "\n").encode("utf-8")
        info = tarfile.TarInfo(f"{name}/{BUNDLE_FILE}")
        info.size = len(data)
        info.mtime = int(time.time())
        import io

        tar.addfile(info, io.BytesIO(data))
        for m in p.members:
            tar.add(m.source, arcname=f"{name}/{m.arcname}", recursive=False)
    return out


@dataclass
class Unpacked:
    root: Path  # <dest>/<name>
    manifest: dict[str, Any]

    @property
    def bundle_dir(self) -> Path:
        return self.root / BUNDLE_SUBDIR

    @property
    def production_record(self) -> Path | None:
        p = self.root / PRODUCTION_SUBDIR / PRODUCTION_FILE
        return p if p.exists() else None

    @property
    def verify_dir(self) -> Path | None:
        p = self.root / VERIFY_SUBDIR
        return p if p.is_dir() else None


def read_manifest(tar_path: str | os.PathLike[str]) -> dict[str, Any]:
    with tarfile.open(tar_path, "r:*") as tar:
        for member in tar:
            parts = member.name.split("/")
            if len(parts) == 2 and parts[1] == BUNDLE_FILE:
                f = tar.extractfile(member)
                assert f is not None
                return json.loads(f.read().decode("utf-8"))
    raise BundleError(f"{tar_path}: no <name>/{BUNDLE_FILE}")


def _safe_members(tar: tarfile.TarFile, name: str) -> list[tarfile.TarInfo]:
    out = []
    for m in tar.getmembers():
        if not m.name.startswith(name + "/") or ".." in m.name.split("/") or m.name.startswith("/"):
            raise BundleError(f"unsafe or foreign member {m.name!r}")
        if not (m.isfile() or m.isdir()):
            raise BundleError(f"member {m.name!r} is not a regular file")
        out.append(m)
    return out


def unpack(tar_path: str | os.PathLike[str], dest: str | os.PathLike[str]) -> Unpacked:
    """Extract and verify every member's sha256 against BUNDLE.json; raise on any mismatch."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    doc = read_manifest(tar_path)
    name = doc["name"]
    with tarfile.open(tar_path, "r:*") as tar:
        members = _safe_members(tar, name)
        tar.extractall(dest, members=members, filter="data") if hasattr(tarfile, "data_filter") else tar.extractall(dest, members=members)
    root = dest / name
    problems = []
    for arc, rec in doc["members"].items():
        path = root / arc
        if not path.exists():
            problems.append(f"{arc}: missing")
            continue
        got = _sha256_file(path)
        if got != rec["sha256"]:
            problems.append(f"{arc}: sha256 {got[:16]} != {rec['sha256'][:16]}")
    present = {p.relative_to(root).as_posix() for p in _walk(root)} - {BUNDLE_FILE}
    for extra in sorted(present - set(doc["members"])):
        problems.append(f"{extra}: not listed in {BUNDLE_FILE}")
    if problems:
        raise BundleError(f"{tar_path}: {len(problems)} member problem(s): " + "; ".join(problems[:5]))
    # the production record's hash is the binding the row's verifier never checked (review F2)
    prod = root / PRODUCTION_SUBDIR / PRODUCTION_FILE
    want = (doc.get("digests") or {}).get("production_record_sha256")
    if prod.exists() and want and _sha256_file(prod) != want:
        raise BundleError("production record hash differs from BUNDLE.json")
    return Unpacked(root, doc)


def verify_tarball(
    tar_path: str | os.PathLike[str],
    *,
    seed: int | None = None,
    out_dir: str | os.PathLike[str] | None = None,
    weights_values: str | os.PathLike[str] | None = None,
    **verify_kwargs: Any,
) -> dict[str, Any]:
    """Unpack, run the fresh verifier, compare with the recorded verification.

    ``seed`` defaults to the recorded one (so the challenge can be reproduced); ``weights_values``
    is an optional ``weights/values.bin`` to drop in when the tarball referenced the weights by
    sha (the reader checks every tensor against the index sha256 and leaf hash).  Extra keyword
    arguments go to :func:`verity_vllm.verify.verify`.  Returns the ``reverify.json`` document."""
    from . import verify as _verify

    tmp = None
    if out_dir is None:
        tmp = tempfile.mkdtemp(prefix="verity-bundle-")
        out_dir = tmp
    out_dir = Path(out_dir)
    unpacked = unpack(tar_path, out_dir / "unpacked")
    recorded = unpacked.manifest.get("recorded_verification") or {}
    if seed is None:
        seed = recorded.get("seed")
        if seed is None:
            raise BundleError("no recorded seed; pass seed=")
    if weights_values is not None:
        wdir = unpacked.bundle_dir / "weights"
        wdir.mkdir(exist_ok=True)
        shutil.copyfile(weights_values, wdir / "values.bin")
    result = _verify.verify(
        unpacked.bundle_dir,
        seed=int(seed),
        out_dir=out_dir / "verify",
        production_record=unpacked.production_record,
        **verify_kwargs,
    )
    fresh_digest = result.challenge.digest().hex()
    comparable = recorded.get("effective_seed") is not None and int(seed) == recorded.get("seed")
    doc = {
        "tarball": str(tar_path),
        "name": unpacked.manifest["name"],
        "profile_id": result.table1["profile_id"],
        "seed": int(seed),
        "recorded": recorded or None,
        "fresh": {
            "verdict": result.verdict,
            "first_blocker": result.table1.get("first_blocker"),
            "challenge_digest": fresh_digest,
            "sample_size": len(result.sample),
            "structural_problems": list(result.structural_problems),
        },
        "challenge_reproduced": (
            (fresh_digest == recorded.get("challenge_digest")) if comparable else None
        ),
        "challenge_comparison": (
            "same seed and derivation: digests compared"
            if comparable
            else "not comparable: different seed, or the recorded challenge used the raw seed (adapter verifier)"
        ),
        "verdict_agrees": (result.verdict == recorded.get("verdict")) if recorded.get("verdict") else None,
    }
    (out_dir / "reverify.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    if tmp is not None:
        doc["out_dir"] = tmp
    return doc


def rederive_transcript(
    tar_path: str | os.PathLike[str],
    *,
    out_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Third-party re-derivation of a staged challenge transcript (``verity_vllm.challenge``), CPU only.

    Unpacks (every member sha256 checked), then re-derives from the bundled ``commitments.json`` and
    ``ru/`` indices: the statement fields a bundle can answer for (profile, description/required
    digests, boundary and weights roots, declared replay units, production record sha256), ``T_0..T_3``,
    ``r_q``/``r_s`` from the revealed seed and its commitment, the selected replay units, the ranked VU
    population, the certified sample ids and the coverage sample.  The C3 binding *outcomes* recorded
    in the statement need the pinned checkpoint and are reported as ``recorded, not re-derived``;
    ``python -m verity_vllm_adapter.verify --transcript`` re-derives those too.  Writes ``rederive.json``."""
    from . import challenge as vchallenge
    from .description import Description

    tmp = None
    if out_dir is None:
        tmp = tempfile.mkdtemp(prefix="verity-bundle-")
        out_dir = tmp
    out_dir = Path(out_dir)
    unpacked = unpack(tar_path, out_dir / "unpacked")
    recorded = unpacked.manifest.get("recorded_verification") or {}
    vdir = unpacked.root / VERIFY_SUBDIR
    tpath = vdir / "challenge_transcript.json"
    if not tpath.exists():
        raise BundleError("the bundle records no challenge_transcript.json (not a staged challenge); use `verify`")
    transcript = json.loads(tpath.read_text())
    challenge_doc = json.loads((vdir / "challenge.json").read_text())
    reader = CaptureReader(unpacked.bundle_dir)
    description = Description(reader.description)
    commitments = json.loads((unpacked.bundle_dir / "commitments.json").read_text())
    required_doc = reader.required if reader.has("required.json") else None
    prod_sha = _sha256_file(unpacked.production_record) if unpacked.production_record and Path(unpacked.production_record).exists() else None
    statement = transcript.get("statement") or {}
    request_ids = reader.request_ids()
    selected = list((transcript.get("replay_challenge") or {}).get("selected", []))
    selective_run = (vdir / "selective_verify.json").exists() or "selective" in statement
    first_stage: dict[str, Any] | None = None
    if selective_run:
        # RC's selective preparation (verity_vllm.selective): T_0 declares no interior, r_q selects J, only J is committed
        from . import selective as vselective

        c0_path = unpacked.bundle_dir / "commitments_phase0.json"
        commitments0 = json.loads(c0_path.read_text()) if c0_path.exists() else None
        statement_problems = vselective.statement_checks(statement, reader=reader, description=description, required_doc=required_doc,
                                                         commitments=commitments, production_sha256=prod_sha, selected=selected,
                                                         commitments_phase0=commitments0)
        first_stage = vselective.first_stage_problems_for(reader, commitments, selected)
        first_stage["phase0_records_checked"] = commitments0 is not None
    else:
        statement_problems = vchallenge.statement_checks(statement, reader=reader, description=description, required_doc=required_doc,
                                                         commitments=commitments, production_sha256=prod_sha)
    interiors, units = vchallenge.interiors_of(commitments, reader, selected)
    adjacent = list(challenge_doc.get("adjacent_vus", []))
    coverage = vchallenge.make_coverage(description, reader, adjacent, int(challenge_doc.get("coverage_size", 1000)), selected)
    if selective_run:
        r = vselective.rederive(transcript, statement=statement, interiors=interiors, request_ids=request_ids, units=units,
                                resolve_ids=vchallenge.make_resolver(reader), coverage=coverage)
    else:
        r = vchallenge.reverify(transcript, statement=statement, interiors=interiors, request_ids=request_ids, units=units,
                                resolve_ids=vchallenge.make_resolver(reader), coverage=coverage)
    # the recorded challenge (relations.Challenge) must be the one the transcript's two samples make
    sample = list(dict.fromkeys(list((transcript.get("sample_challenge") or {}).get("coverage_sample", []))
                                + list((transcript.get("sample_challenge") or {}).get("certified_sample", []))))
    from . import relations

    fresh = relations.make_challenge(commitments, description, sample, required_doc)
    digest_ok = fresh.digest().hex() == challenge_doc.get("digest")
    mode = (transcript.get("source") or {}).get("mode")
    doc = {
        "tarball": str(tar_path),
        "name": unpacked.manifest["name"],
        "profile_id": reader.profile_id,
        "challenge_mode": mode,
        "recorded": recorded or None,
        "protocol": "selective (T_0 declares no interior; r_q selects J; only J committed)" if selective_run else "staged (every replay unit committed before T_0)",
        "statement": {
            "rederived_from_bundle": not statement_problems,
            "problems": statement_problems,
            "not_rederived_here": ["bindings.model (needs the pinned checkpoint)", "bindings.architecture (needs config.json + semantic profile)"]
            + ([] if selective_run else ["first_stage (re-run the verifier)"]),
        },
        "first_stage": first_stage,
        "transcript": r,
        "challenge_digest": {"recorded": challenge_doc.get("digest"), "rederived": fresh.digest().hex(), "agrees": digest_ok},
        "population": {"requests": len(request_ids), "selected_requests": len(selected), "vus": units.count,
                       "certified": len((transcript.get("sample_challenge") or {}).get("certified_sample", [])),
                       "coverage": len((transcript.get("sample_challenge") or {}).get("coverage_sample", []))},
        "ok": bool(r["ok"] and not statement_problems and digest_ok and (first_stage is None or first_stage["ok"])),
        "verdict_label": vchallenge.verdict_label(str(recorded.get("verdict")), mode),
    }
    (out_dir / "rederive.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    if tmp is not None:
        doc["out_dir"] = tmp
    return doc


def inspect_text(doc: Mapping[str, Any]) -> str:
    lines = [
        f"{doc['name']}  profile {doc['profile_id']}  case {doc.get('case')}  created {doc.get('created_utc')}",
        f"model: {json.dumps(doc.get('model'))}",
        f"roots: {doc['roots']}",
        f"weights: {doc['weights']}",
        f"digests: {json.dumps(doc['digests'], indent=None)}",
        f"recorded verification: {json.dumps(doc.get('recorded_verification'))}",
        f"members: {len(doc['members'])}  total {doc['total_bytes'] / 1e6:.1f} MB  omitted: {doc.get('omitted')}",
        f"reverify: {doc['reverify']['command']}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m verity_vllm.bundle", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack")
    p.add_argument("run_dir")
    p.add_argument("--out", required=True)
    p.add_argument("--production", default=None, help="production.jsonl")
    p.add_argument("--verify-dir", default=None, help="the verifier's --out directory to include")
    p.add_argument("--name", default=None)
    p.add_argument("--weights", choices=("omit", "copy"), default="omit")
    p.add_argument("--extra", action="append", default=[])
    i = sub.add_parser("inspect")
    i.add_argument("tarball")
    u = sub.add_parser("unpack")
    u.add_argument("tarball")
    u.add_argument("dest")
    v = sub.add_parser("verify")
    v.add_argument("tarball")
    v.add_argument("--seed", type=int, default=None)
    v.add_argument("--out", default=None)
    v.add_argument("--sample", type=int, default=None)
    v.add_argument("--torch", default="auto", choices=("auto", "never", "require"))
    v.add_argument("--plugin", action="append", default=None)
    v.add_argument("--negatives", default=None, help="comma-separated ids, or 'none'")
    v.add_argument("--weights-values", default=None)
    v.add_argument("--no-audit", action="store_true")
    r = sub.add_parser("rederive", help="re-derive a staged challenge transcript from the bundle (CPU only)")
    r.add_argument("tarball")
    r.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "rederive":
            doc = rederive_transcript(args.tarball, out_dir=args.out)
            print(json.dumps(doc, indent=1, sort_keys=True))
            return 0 if doc["ok"] else 1
        if args.cmd == "pack":
            out = pack(args.run_dir, args.out, production_record=args.production, verify_dir=args.verify_dir,
                       name=args.name, weights=args.weights, extra=args.extra)
            print(inspect_text(read_manifest(out)))
            print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
            return 0
        if args.cmd == "inspect":
            print(inspect_text(read_manifest(args.tarball)))
            return 0
        if args.cmd == "unpack":
            up = unpack(args.tarball, args.dest)
            print(f"unpacked and verified {len(up.manifest['members'])} members into {up.root}")
            return 0
        if args.cmd == "verify":
            from .negatives import NEGATIVE_IDS
            from .verify import DEFAULT_PLUGINS, DEFAULT_SAMPLE

            negatives = (
                list(NEGATIVE_IDS) if args.negatives is None
                else None if args.negatives == "none"
                else [x.strip() for x in args.negatives.split(",") if x.strip()]
            )
            doc = verify_tarball(
                args.tarball, seed=args.seed, out_dir=args.out, weights_values=args.weights_values,
                sample_size=args.sample or DEFAULT_SAMPLE, torch_mode=args.torch,
                plugins=args.plugin or list(DEFAULT_PLUGINS), negatives=negatives, audit=not args.no_audit,
            )
            print(json.dumps(doc, indent=1, sort_keys=True))
            return 0 if doc["fresh"]["verdict"] in ("PASS", "PARTIAL") else 1
    except BundleError as error:
        print(f"bundle error: {error}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    sys.exit(main())
