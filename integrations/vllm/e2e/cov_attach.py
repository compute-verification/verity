"""Attach the PROOF-COV coverage proofs (embed-gather + ordinary-inactive) to ONE record.

Takes the builder outputs (``zk/bench/embed_relations.py`` / ``pad_relations.py`` batch
directories), the pod proving records (``pb_prove.py`` JSONL), the kept proof files and their
CPU-mode host verifications, and writes ``<record>/proofs/proof-cov/``:

* per attached batch: ``statement.bin`` + ``plan.bin`` + ``census.json`` + the kept proof
  files (``proof-<mode>.bin``) + their CPU-mode ``verify-<mode>.json``;
* ``records.jsonl`` — every pod proving record (all batches, including the ones whose proof
  files were measured-then-deleted; their shas stay pinned in the rows);
* ``cov_summary.json`` — the coverage summary the whole-program certificate consumes: gate-set
  identities, per-batch pins (statement sha256 + digest, witness sha256, proof shas, CUDA
  prove + CPU verify outcomes), the prefill token-binding evidence, the labeled exclusions,
  and the host identity (vk / ELF / host binary sha) of the family proofs.

Every number here is read from the run's own artifacts; the certificate re-verifies by
recomputing the file hashes.  Exit 0 iff every attached batch has a True verdict on both the
CUDA prove and the CPU-mode verify, under one uniform vk.

Usage (pod)::

    python -m e2e.cov_attach --record <record> --batches <dir> [--batches <dir>...]
        --records <records.jsonl> --proof-dir <dir> --verify-dir <dir> --host <host binary>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

from e2e.run_composed import sha256_file, write_json


def _load(p: Path):
    return json.loads(p.read_text())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--record", required=True)
    ap.add_argument("--batches", action="append", required=True, help="builder output dir (has manifest.json)")
    ap.add_argument("--records", required=True, help="pb_prove records.jsonl")
    ap.add_argument("--proof-dir", required=True, help="kept proof files (pb_prove --keep-proofs)")
    ap.add_argument("--verify-dir", required=True, help="CPU-mode host verify JSONs, verify-<sha12>-<mode>.json")
    ap.add_argument("--host", required=True, help="the host binary that proved/verified the family batches")
    args = ap.parse_args()
    record = Path(args.record).resolve()
    out = record / "proofs" / "proof-cov"
    out.mkdir(parents=True, exist_ok=True)
    proof_dir, verify_dir = Path(args.proof_dir), Path(args.verify_dir)
    problems: list[str] = []

    # the pod proving records, keyed by (batch_dir, mode)
    rows = [json.loads(l) for l in Path(args.records).read_text().splitlines() if l.strip()]
    proved: dict[tuple[str, str], dict] = {}
    vks, elves = set(), set()
    for r in rows:
        if r.get("mode") == "execute":
            continue
        if not r.get("ok") or not r.get("verdict") or r.get("unsound"):
            problems.append(f"proving record not ok: {r.get('batch_dir')} {r.get('mode')}")
            continue
        proved[(r["batch_dir"], r["mode"])] = r
        vks.add(r.get("vk_hash"))
        elves.add(r.get("elf_sha256"))
    if len(vks) != 1 or len(elves) != 1:
        problems.append(f"family proofs are not under one program identity: vks={sorted(vks)}")
    host_sha = sha256_file(Path(args.host))

    # attach policy: the LARGEST batch of every relation carries its files (statement + plan +
    # census + the COMPRESSED proof, ~1.3 MB constant); every other (batch, mode), core modes
    # included, stays records-only with the proof sha pinned by the pod proving record.
    manifests = []
    batches_out = []
    largest: dict[str, int] = {}
    for bdir in args.batches:
        for entry in _load(Path(bdir) / "manifest.json")["batches"]:
            rel = entry.get("relation") or ""
            largest[rel] = max(largest.get(rel, 0), entry.get("size") or 0)
    for bdir in args.batches:
        manifest = _load(Path(bdir) / "manifest.json")
        manifests.append({"dir": bdir, **{k: manifest[k] for k in manifest if k != "batches"}})
        for entry in manifest["batches"]:
            src = Path(entry["dir"])
            key12 = hashlib.sha256(str(src).encode()).hexdigest()[:12]
            # a unique, path-safe batch name (write_batch dirs are "b<size>" under nested
            # relation directories, so src.name alone collides across relations)
            name = f"{(entry.get('relation') or src.parent.name).replace('/', '-')}-b{entry.get('size')}"
            attach_this = (entry.get("size") or 0) == largest.get(entry.get("relation") or "", -1)
            modes = {}
            for mode in ("core", "compressed"):
                rec = proved.get((str(src), mode))
                if rec is None:
                    continue
                kept = sorted(proof_dir.glob(f"{key12}-{mode}-*.bin"))
                vfile = verify_dir / f"verify-{key12}-{mode}.json"
                verify = _load(vfile) if vfile.exists() else None
                mode_doc = {
                    "proved_cuda": {"verdict": rec["verdict"], "prove_s": rec.get("prove_s"),
                                    "total_cycles": rec.get("total_cycles"), "proof_bytes": rec.get("proof_bytes"),
                                    "proof_sha256": rec.get("proof_sha256")},
                    "verified_cpu": None,
                    "attached": None,
                }
                if kept and attach_this and mode == "compressed":
                    dst_dir = out / name
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    dst = dst_dir / f"proof-{mode}.bin"
                    shutil.copyfile(kept[0], dst)
                    got = sha256_file(dst)
                    if rec.get("proof_sha256") and got != rec["proof_sha256"]:
                        problems.append(f"{name} {mode}: kept proof sha {got[:12]} != proving record {rec['proof_sha256'][:12]}")
                    mode_doc["attached"] = {"file": str(dst.relative_to(record)), "sha256": got, "bytes": dst.stat().st_size}
                if verify is not None:
                    ok = bool(verify.get("ok")) and bool(verify.get("statement_match")) and verify.get("vk_hash") in vks
                    mode_doc["verified_cpu"] = {"ok": ok, "host_ok": verify.get("ok"),
                                                "statement_match": verify.get("statement_match"),
                                                "vk_hash": verify.get("vk_hash")}
                    if not ok:
                        problems.append(f"{name} {mode}: CPU-mode verify not ok")
                    if mode_doc["attached"]:
                        shutil.copyfile(vfile, out / name / f"verify-{mode}.json")
                modes[mode] = mode_doc
            if not modes:
                problems.append(f"{name}: no proving record for any mode")
                continue
            attached_any = any(m.get("attached") for m in modes.values())
            if attach_this and not attached_any:
                problems.append(f"{name}: the relation's largest batch has no attachable compressed proof")
            if attached_any:
                dst_dir = out / name
                for fname in ("statement.bin", "plan.bin", "census.json"):
                    shutil.copyfile(src / fname, dst_dir / fname)
            wit = src / "witness-fmt2.bin"
            batches_out.append({
                "name": name,
                "relation": entry.get("relation"),
                "relation_id": entry.get("relation_id"),
                "family": entry.get("family"),
                "gate_set": manifest.get("gate_set"),
                "size": entry.get("size"),
                "obligations": entry.get("obligations"),
                "positions": entry.get("positions"),
                "inactive_units": entry.get("inactive_units"),
                "prefill_units": entry.get("prefill_units"),
                "decode_units": entry.get("decode_units"),
                "statement_digest": entry.get("statement_digest"),
                "statement_sha256": sha256_file(src / "statement.bin"),
                "witness_sha256": sha256_file(wit) if wit.exists() else None,
                "attached": attached_any,
                "attached_dir": (str((out / name).relative_to(record)) if attached_any else None),
                "modes": modes,
                "provenance": entry.get("provenance"),
            })

    shutil.copyfile(args.records, out / "records.jsonl")
    embed_manifest = next((m for m in manifests if m.get("workload") == "proof-cov-embed"), None)
    kinds = sorted({b["relation_id"] for b in batches_out if b["relation_id"]})
    summary = {
        "schema": "veritor/cov-summary/v1",
        "record": record.name,
        "families": {m.get("workload"): {"gate_set": m.get("gate_set"), "gate_set_digest": m.get("gate_set_digest")}
                     for m in manifests},
        "kinds_proved": kinds,
        "batches": batches_out,
        "prefill_binding": (embed_manifest or {}).get("prefill_binding"),
        "prefill_exclusions": (embed_manifest or {}).get("prefill_exclusions") or [],
        "weights_sha256": (embed_manifest or {}).get("weights_sha256"),
        "table_root": (embed_manifest or {}).get("table_root"),
        "program_identity": {"vk_hash": next(iter(vks), None), "elf_sha256": next(iter(elves), None),
                             "host_sha256": host_sha, "host_archived_as": "bin/veritor-zk-host-cov"},
        "records_jsonl": "proofs/proof-cov/records.jsonl",
        "problems": problems,
        "ok": not problems,
    }
    write_json(out / "cov_summary.json", summary)
    attached = sum(1 for b in batches_out if b["attached"])
    print(f"[cov_attach] {'OK' if not problems else 'PROBLEMS'}: {len(batches_out)} batches "
          f"({attached} with attached proofs), kinds: {', '.join(kinds)}; -> {out / 'cov_summary.json'}")
    for p in problems:
        print(f"[cov_attach]   problem: {p}")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
