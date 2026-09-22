"""[R8 C-codec] Corpus staging on the pod: verify every *.small.tgz / *.compact.tgz against MANIFEST.json (sha256), extract the small
archives to <records>/<run_id>/, and drop the compact archives' program.json/instances.jsonl next to them when present.

  python -m verity_capture.experimental.aot_codec.corpus --archives /vol/corpus/archive-20260917 --records /workspace/cp/corpus/records [--no-compact]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
import time
from pathlib import Path


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archives", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--no-compact", action="store_true")
    a = ap.parse_args(argv)
    arch, recs = Path(a.archives), Path(a.records)
    recs.mkdir(parents=True, exist_ok=True)
    man = json.loads((arch / "MANIFEST.json").read_text())
    entries = man.get("files") if isinstance(man, dict) else man
    if isinstance(entries, dict):
        entries = [dict(name=k, **v) for k, v in entries.items()]
    want = {e["name"]: e for e in entries}
    report = []
    for p in sorted(arch.glob("*.tgz")):
        t0 = time.time()
        e = want.get(p.name)
        sha = sha256_file(p)
        ok = e is not None and e.get("sha256") == sha and int(e.get("bytes", p.stat().st_size)) == p.stat().st_size
        row = {"name": p.name, "bytes": p.stat().st_size, "sha256": sha, "manifest_ok": ok, "kind": "compact" if ".compact." in p.name else "small"}
        if not ok:
            row["manifest_entry"] = e
            report.append(row)
            print(f"[corpus] MISMATCH {p.name}", file=sys.stderr)
            continue
        run_id = p.name[: -len(".small.tgz")] if p.name.endswith(".small.tgz") else p.name[: -len(".compact.tgz")]  # keeps e.g. `.run1`
        if row["kind"] == "small":
            dest = recs / run_id
            if not (dest / "spec.json").exists():
                with tarfile.open(p, "r:gz") as tf:
                    top = tf.getnames()[0].split("/")[0]
                    tf.extractall(recs)
                if top != run_id and (recs / top).exists():
                    (recs / top).rename(dest)
            row["extracted_to"] = str(dest)
        elif not a.no_compact:
            dest = recs / run_id / "compact"
            dest.mkdir(parents=True, exist_ok=True)
            with tarfile.open(p, "r:gz") as tf:
                for m in tf.getmembers():
                    if m.name.split("/")[-1] in ("program.json", "instances.jsonl", "wiring.json", "pruned.json"):
                        m.name = m.name.split("/")[-1]
                        tf.extract(m, dest)
            row["extracted_to"] = str(dest)
        row["seconds"] = round(time.time() - t0, 1)
        report.append(row)
        print(f"[corpus] ok {p.name} {row['bytes'] / 1e6:.1f} MB -> {row.get('extracted_to')} ({row['seconds']}s)", file=sys.stderr)
    (recs / "STAGING.json").write_text(json.dumps({"archives": str(arch), "manifest_sha256": sha256_file(arch / "MANIFEST.json"),
                                                   "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "files": report}, indent=1))
    bad = [r for r in report if not r["manifest_ok"]]
    print(f"[corpus] {len(report)} archives, {len(bad)} manifest mismatches", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
