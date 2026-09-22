#!/usr/bin/env python
"""[E10 EOD 2026-09-11] Slim archive of a record_v5 record -- compressed proofs ALWAYS included, refused otherwise.

    python -m record_v5.archive_v5 pack <RECORD> <OUT.tgz> [--exclude-glob G ...] [--allow-missing-proofs]
    python -m record_v5.archive_v5 check <OUT.tgz>            # re-hash every member against ARCHIVE.json

What goes in (relative to the record directory), by default everything EXCEPT
  adapter/*/ptx/**                 kernel PTX dumps of the serving passes (reproducible from the pinned engine; never read by the verifier)
  **/__pycache__/**, *.pyc
and what MUST be in (else exit 3 with the missing list, unless --allow-missing-proofs for a record that never reached `prove`):
  proofs/<chunk>-compressed.bin    for every proofs/<chunk>.json whose status starts with PROVEN
  proofs/proofs.json, proofs/guest.json
  obligations/<chunk>/statement.bin + composed.json   for every chunk (the verifier's inputs; witness.bin is kept too when present)
  freeze.json, run_config.json, protocol_instance.json (or instance/), population.v5.json, expressibility_v5.json
`ARCHIVE.json` (inside the tarball, at the top) lists every member with sha256 + bytes, the record's frozen revision (freeze.json), and the
verdict files' digests, so `check` re-verifies the archive's integrity without unpacking it anywhere permanent.  The archive is not the
record: it is the record's content at pack time (the tarball's own sha256 is printed and should be written into the report).
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import io
import json
import os
import sys
import tarfile
import time
from pathlib import Path

EXCLUDE_DEFAULT = ["adapter/*/ptx/*", "adapter/*/ptx", "*/__pycache__/*", "*.pyc"]
REQUIRED_TOP = ["freeze.json", "run_config.json", "population.v5.json", "expressibility_v5.json"]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def members(rec: Path, excludes: list[str]) -> list[Path]:
    out = []
    for dp, dns, fns in os.walk(rec):
        for fn in sorted(fns):
            p = Path(dp) / fn
            rel = str(p.relative_to(rec))
            if any(fnmatch.fnmatch(rel, g) for g in excludes):
                continue
            out.append(p)
    return sorted(out)


def required_missing(rec: Path, present: set[str], allow_missing_proofs: bool, *, top: bool = True) -> list[str]:
    missing = [t for t in REQUIRED_TOP if t not in present] if top else []
    if top and "protocol_instance.json" not in present and "instance/protocol_instance.json" not in present:
        missing.append("protocol_instance.json | instance/protocol_instance.json")
    pdir = rec / "proofs"
    if pdir.is_dir():
        for cj in sorted(pdir.glob("chunk*.json")):
            st = str((json.loads(cj.read_text()) or {}).get("status", ""))
            if st.startswith("PROVEN"):
                b = f"proofs/{cj.stem}-compressed.bin"
                if b not in present:
                    missing.append(b)
        for f in ("proofs/proofs.json", "proofs/guest.json"):
            if f not in present:
                missing.append(f)
    elif not allow_missing_proofs:
        missing.append("proofs/ (no proofs directory: pass --allow-missing-proofs only for a record refused before `prove`)")
    odir = rec / "obligations"
    if odir.is_dir():
        for c in sorted(p for p in odir.iterdir() if p.is_dir() and p.name.startswith("chunk")):
            for f in ("statement.bin", "composed.json"):
                if f"obligations/{c.name}/{f}" not in present:
                    missing.append(f"obligations/{c.name}/{f}")
    return missing


def pack(a: argparse.Namespace) -> int:
    rec = Path(a.record).resolve()
    out = Path(a.out).resolve()
    excludes = EXCLUDE_DEFAULT + list(a.exclude_glob or [])
    files = members(rec, excludes)
    present = {str(p.relative_to(rec)) for p in files}
    missing = required_missing(rec, present, a.allow_missing_proofs, top=not a.no_record_v5_top)
    if missing:
        print("[archive_v5] REFUSED: required members missing from the record:\n  " + "\n  ".join(missing), file=sys.stderr)
        return 3
    manifest = {"schema": "veritor/e10/archive-v5/v1", "record": str(rec), "run_id": rec.name, "packed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "excluded_globs": excludes, "members": [], "bytes_total": 0}
    fz = rec / "freeze.json"
    if fz.exists():
        f = json.loads(fz.read_text())
        manifest["frozen_revision"] = {k: f.get(k) for k in ("commit", "branch", "dirty_patch_sha256", "dirty_patch_bytes", "revision_source")}
    for p in files:
        rel = str(p.relative_to(rec))
        n = p.stat().st_size
        manifest["members"].append({"path": rel, "bytes": n, "sha256": sha256_file(p)})
        manifest["bytes_total"] += n
    verdicts = {}
    for v in ("verify/verify_v5.json", "verify/reverify_v3/record_v3.json", "protocol_exit.json", "verifier_exit.json", "proofs/proofs.json"):
        if (rec / v).exists():
            verdicts[v] = sha256_file(rec / v)
    manifest["verdict_files_sha256"] = verdicts
    proofs = [m for m in manifest["members"] if m["path"].startswith("proofs/") and m["path"].endswith("-compressed.bin")]
    manifest["compressed_proofs"] = {"count": len(proofs), "bytes": sum(m["bytes"] for m in proofs), "sha256": {m["path"]: m["sha256"] for m in proofs}}
    out.parent.mkdir(parents=True, exist_ok=True)
    mb = json.dumps(manifest, indent=1).encode()
    with tarfile.open(out, "w:gz", compresslevel=6) as tf:
        ti = tarfile.TarInfo(name=f"{rec.name}/ARCHIVE.json")
        ti.size = len(mb)
        ti.mtime = int(time.time())
        tf.addfile(ti, io.BytesIO(mb))
        for p in files:
            tf.add(p, arcname=f"{rec.name}/{p.relative_to(rec)}", recursive=False)
    size = out.stat().st_size
    print(f"[archive_v5] {out} {size / 1e6:.1f} MB ({manifest['bytes_total'] / 1e6:.1f} MB unpacked, {len(files)} files, {len(proofs)} compressed proofs); sha256 {sha256_file(out)}")
    return 0


def check(a: argparse.Namespace) -> int:
    p = Path(a.archive)
    bad, seen = [], 0
    with tarfile.open(p, "r:gz") as tf:
        names = tf.getnames()
        top = names[0].split("/")[0]
        man = json.loads(tf.extractfile(f"{top}/ARCHIVE.json").read())
        want = {m["path"]: m for m in man["members"]}
        for name in names:
            rel = name[len(top) + 1:]
            if rel == "ARCHIVE.json" or not rel:
                continue
            m = tf.getmember(name)
            if not m.isfile():
                continue
            h = hashlib.sha256()
            f = tf.extractfile(m)
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
            w = want.get(rel)
            seen += 1
            if w is None or w["sha256"] != h.hexdigest() or w["bytes"] != m.size:
                bad.append(rel)
        absent = sorted(set(want) - {n[len(top) + 1:] for n in names})
    print(f"[archive_v5] {p.name}: {seen} members hashed, {len(bad)} mismatched, {len(absent)} listed-but-absent; compressed proofs {man['compressed_proofs']['count']}; "
          f"frozen revision {(man.get('frozen_revision') or {}).get('commit', '?')[:12]}; archive sha256 {sha256_file(p)}")
    if bad or absent:
        print("  mismatched: " + ", ".join(bad[:10]) + ("\n  absent: " + ", ".join(absent[:10]) if absent else ""))
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack")
    p.add_argument("record")
    p.add_argument("out")
    p.add_argument("--exclude-glob", action="append", default=None)
    p.add_argument("--allow-missing-proofs", action="store_true")
    p.add_argument("--no-record-v5-top", action="store_true", help="a non-record_v5 record (E8's CORE-*): only the proofs/obligations rules apply")
    c = sub.add_parser("check")
    c.add_argument("archive")
    a = ap.parse_args(argv)
    return pack(a) if a.cmd == "pack" else check(a)


if __name__ == "__main__":
    sys.exit(main())
