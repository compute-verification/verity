"""DERIVATION TOOL (WB2, 2026-09-07; audit-security-baseline finding 1) -- **not a capture**.

Rewrites the boundary trees of an existing security-mode archive so that they bind the chosen logprobs
(capture-format rev. 2026-09-07c: one ``float32`` ``output_logprob`` leaf per produced token), using the
logprobs the archive's own ``production/production.jsonl`` already records.  Nothing is measured or
re-executed: the values were reported by production's API and shipped in the archive; only the boundary
leaves and the ``boundary/<request>`` records of ``commitments.json`` were missing.

What the derived bundle is and is not:

* every replay-unit tree, the weights tree, ``description.json``, ``required.json``, ``production.jsonl``,
  the manifest and the kernel sources are **byte-identical** to the source archive (the tool asserts it);
* the boundary roots differ (they now cover the logprob leaves), hence ``T_0`` differs and the archive's
  recorded ``challenge_transcript.json`` does **not** re-derive against the derived bundle -- it belongs to
  the source archive.  A derived bundle has **no security-mode challenge of its own**: the interior
  commitments were fixed by WB's original capture, the derivation happened later on a CPU box, and a
  verifier's fresh seed over the derived ``T_0`` would still only exercise the relations on a GPU.
  **A fresh GPU capture with the new format is required for a row to be re-labelled**; this tool exists
  so the bound format can be exercised and its negatives run from the archive alone before that capture
  exists.  ``DERIVED.json`` in the output states all of this machine-readably.

Usage::

    python -m verity_vllm_adapter.derive_logprobs <archive dir or .tar.zst> --out <dir>

The output directory holds the derived ``commit/`` (a copy of the source commit dir with the two changed
documents), the source's ``production/``, and ``DERIVED.json``.  Verify it CPU-only with
``verity_vllm_adapter.verify_cpu``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any

from verity_vllm import commit as vcommit
from verity_vllm import rowcheck
from verity_vllm.capture import BOUNDARY_FILE, RU_DIR, Boundary, CaptureError, CaptureReader, boundary_tree_id
from verity_vllm.commit import MerkleTree

TOOL = "verity_vllm_adapter.derive_logprobs"
DERIVED_FILE = "DERIVED.json"


class DeriveError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True,
                              cwd=Path(__file__).resolve().parent).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def extract_archive(archive: Path, dest: Path) -> Path:
    """Extract ``<name>.tar.zst`` (or ``.tar.gz`` / ``.tar``) under ``dest``; returns the archive root dir.
    ``zstd`` is decoded with the ``zstandard`` module when installed, else the ``zstd`` CLI."""
    dest.mkdir(parents=True, exist_ok=True)
    src = archive
    if archive.suffix == ".zst":
        tar_path = dest / archive.with_suffix("").name
        try:
            import zstandard  # type: ignore

            with open(archive, "rb") as f, open(tar_path, "wb") as g:
                zstandard.ZstdDecompressor().copy_stream(f, g)
        except ImportError:
            if shutil.which("zstd") is None:
                raise DeriveError("neither the zstandard module nor the zstd CLI is available to decode the archive") from None
            with open(tar_path, "wb") as g:
                subprocess.run(["zstd", "-dc", str(archive)], stdout=g, check=True)
        src = tar_path
    with tarfile.open(src, "r:*") as tar:
        names = [m.name for m in tar.getmembers()]
        tops = {n.split("/", 1)[0] for n in names}
        if len(tops) != 1:
            raise DeriveError(f"archive has {len(tops)} top-level entries, expected one")
        if hasattr(tarfile, "data_filter"):
            tar.extractall(dest, filter="data")
        else:
            tar.extractall(dest)
    if src != archive:
        src.unlink()
    return dest / next(iter(tops))


def locate(root: Path) -> tuple[Path, Path]:
    """``(commit dir, production.jsonl)`` of an extracted archive or a bare run dir."""
    for commit_dir in (root / "commit", root / "bundle", root):
        if (commit_dir / "commitments.json").exists() and (commit_dir / "capture.json").exists():
            break
    else:
        raise DeriveError(f"{root}: no commit dir (commitments.json + capture.json) found")
    for prod in (root / "production" / "production.jsonl", root / "production.jsonl"):
        if prod.exists():
            return commit_dir, prod
    raise DeriveError(f"{root}: no production/production.jsonl")


def derive(commit_dir: Path, production_path: Path, out_dir: Path, *, source_label: str,
           source_archive_sha256: str | None = None) -> dict[str, Any]:
    """Copy ``commit_dir`` to ``out_dir/commit``, rewrite every ``boundary.json`` with the production
    record's ``chosen_logprobs`` and the ``boundary/*`` records of ``commitments.json``; assert every
    other tree record is unchanged; write ``DERIVED.json``.  Returns the ``DERIVED.json`` document."""
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "commit"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(commit_dir, dst)
    prod_dst = out_dir / "production"
    prod_dst.mkdir(exist_ok=True)
    shutil.copy2(production_path, prod_dst / "production.jsonl")

    reader = CaptureReader(dst)
    commitments = vcommit.load_commitments(reader)
    production = {str(r["request_id"]): r for r in (json.loads(line) for line in production_path.read_text().splitlines() if line.strip())}
    before = {tid: dict(rec) for tid, rec in commitments["trees"].items()}

    # the source boundaries must recompute from production.jsonl as they stand (else the archive is not
    # what it claims and the derivation would launder a mismatch)
    boundaries_before, status_before = rowcheck.production_boundaries_with_status(production, commitments)
    for rid, b in boundaries_before.items():
        tid = boundary_tree_id(rid)
        want = before[tid]["root"]
        got = MerkleTree.from_values(b.leaves(reader.profile_id), tid).root.hex()
        if got != want:
            raise DeriveError(f"{rid}: source boundary root {got[:16]}… != committed {want[:16]}…; refusing to derive from an inconsistent archive")

    requests: dict[str, Any] = {}
    changed_files: list[str] = []
    for rid in reader.request_ids():
        rec = production.get(rid)
        if rec is None:
            raise DeriveError(f"{rid}: committed boundary has no production record")
        lps = rec.get("chosen_logprobs")
        if not isinstance(lps, list):
            raise DeriveError(f"{rid}: production record carries no chosen_logprobs; nothing to derive")
        old = reader.boundary(rid)
        if old.binds_logprobs:
            raise DeriveError(f"{rid}: the source boundary already binds logprobs (status {status_before.get(rid)})")
        new = Boundary(
            request_id=old.request_id, prompt_token_ids=list(old.prompt_token_ids), output_token_ids=list(old.output_token_ids),
            sampling_params=dict(old.sampling_params), seed=old.seed, finish_reason=old.finish_reason,
            output_records=list(old.output_records), steps=list(old.steps),
            chosen_logprobs=[None if x is None else float(x) for x in lps],
        )
        problems = new.validate()
        if problems:
            raise DeriveError(f"{rid}: derived boundary does not validate: " + "; ".join(problems))
        path = dst / RU_DIR / rid / BOUNDARY_FILE
        path.write_text(json.dumps(new.to_json(), indent=1, sort_keys=True) + "\n")
        changed_files.append(str(path.relative_to(out_dir)))
        tree = MerkleTree.from_values(new.leaves(reader.profile_id), boundary_tree_id(rid))
        requests[rid] = {
            "root_before": before[boundary_tree_id(rid)]["root"], "root_after": tree.root.hex(),
            "leaf_count_before": before[boundary_tree_id(rid)]["leaf_count"], "leaf_count_after": tree.leaf_count,
            "logprob_leaves": sum(1 for x in lps if x is not None), "produced_tokens": len(old.output_token_ids),
        }

    # rebuild commitments.json from the derived commit dir; only the boundary records may differ
    fresh = CaptureReader(dst)
    after_doc = vcommit.build_commitments(fresh, recompute_leaves=False, write=False)
    after = after_doc["trees"]
    if set(after) != set(before):
        raise DeriveError(f"tree set changed: {sorted(set(after) ^ set(before))}")
    for tid in before:
        if tid.startswith("boundary/"):
            if after[tid]["root"] == before[tid]["root"]:
                raise DeriveError(f"{tid}: root unchanged after adding logprob leaves")
        elif after[tid] != before[tid]:
            raise DeriveError(f"{tid}: record changed by the derivation ({before[tid]} -> {after[tid]})")
    doc = dict(commitments)
    doc["trees"] = after
    (dst / "commitments.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    changed_files.append("commit/commitments.json")

    # the derived boundaries must now recompute from production.jsonl with status "bound"
    _, status_after = rowcheck.production_boundaries_with_status(production, doc)
    if any(s != "bound" for s in status_after.values()):
        raise DeriveError(f"derived bundle does not read back as bound: {status_after}")

    derived = {
        "schema": "verity-vllm/derived/v1",
        "tool": TOOL,
        "tool_revision": _git_rev(),
        "utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "kind": "DERIVED BUNDLE -- not a capture; the boundary trees were rewritten from the source archive's own production.jsonl",
        "derived_from": {"label": source_label, "archive_sha256": source_archive_sha256,
                         "production_record_sha256": _sha256(production_path),
                         "commitments_sha256_before": _sha256(commit_dir / "commitments.json")},
        "change": {
            "what": "chosen_logprobs of production.jsonl bound as one float32 output_logprob leaf per produced token "
                    "(IEEE-754 binary32 word, little-endian, shape []); boundary.json gains `chosen_logprobs`; "
                    "commitments.json boundary/* records (root, leaf_count, ids_digest) recomputed",
            "files": changed_files,
            "requests": requests,
            "logprob_status_before": status_before, "logprob_status_after": status_after,
        },
        "unchanged": sorted(t for t in before if not t.startswith("boundary/")),
        "consequences": [
            "T_0 (statement digest) differs from the source archive's: the recorded challenge_transcript.json belongs to the source archive and does not re-derive here",
            "no security-mode challenge exists for this bundle; a fresh GPU capture in the new format is required for the row to be re-labelled",
            "CPU-only checks that apply from the archive alone: description validation, first-stage completeness, replay-unit tree recomputation, output binding (C3c) with logprobs bound, N17/N19 negatives (verity_vllm_adapter.verify_cpu)",
        ],
        "seconds": round(time.perf_counter() - t0, 3),
    }
    (out_dir / DERIVED_FILE).write_text(json.dumps(derived, indent=1, sort_keys=True) + "\n")
    return derived


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("source", help="extracted archive dir, bare commit dir with production/, or a .tar.zst / .tar.gz archive")
    ap.add_argument("--out", required=True, help="output dir (gets commit/, production/, DERIVED.json)")
    ap.add_argument("--production", default=None, help="production.jsonl when the source is a bare commit dir")
    args = ap.parse_args(argv)
    src = Path(args.source)
    tmp: tempfile.TemporaryDirectory[str] | None = None
    sha = None
    try:
        if src.is_file():
            sha = _sha256(src)
            tmp = tempfile.TemporaryDirectory(prefix="derive-logprobs-")
            root = extract_archive(src, Path(tmp.name))
            commit_dir, prod = locate(root)
        elif args.production:
            commit_dir, prod = src, Path(args.production)
        else:
            commit_dir, prod = locate(src)
        derived = derive(commit_dir, prod, Path(args.out), source_label=src.name, source_archive_sha256=sha)
    except (DeriveError, CaptureError, vcommit.CommitError) as e:
        print(f"derive_logprobs: {e}", file=sys.stderr)
        return 1
    finally:
        if tmp is not None:
            tmp.cleanup()
    for rid, r in derived["change"]["requests"].items():
        print(f"{rid}: root {r['root_before'][:12]}… -> {r['root_after'][:12]}…, leaves {r['leaf_count_before']} -> {r['leaf_count_after']} "
              f"(+{r['logprob_leaves']} float32 output_logprob)")
    print(f"DERIVED (not a capture): {Path(args.out) / DERIVED_FILE}; unchanged trees: {len(derived['unchanged'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
