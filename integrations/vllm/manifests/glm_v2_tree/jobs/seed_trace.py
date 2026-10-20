"""Seed a traced Build's row dir with the derives an earlier trace of the same row finished, so a restarted trace derives only
the rest (`row_tp.build_todo` keeps a derive dir whose result.json says ok).  A derive dir is seeded, hard-linked, only when
  * its `.rc` says 0 (the Build writes it after the derive process exits),
  * result.json says ok, and every JSON file in it parses,
  * every `.gz` file decompresses to its end (gzip's CRC and length), and instances.json.gz opens with the Program's digest,
  * descriptor.json.gz is the bytes result.json's `descriptor_sha256` names, and the Program digest recomputed from it
    (SHA-512 of its canonical JSON less annotations, as `pipeline.build._write_program` computes it) is result.json's.
A dir already in the destination is left alone.  Anything refused is derived again by the Build.
    python seed_trace.py SRC_ROW_DIR DST_ROW_DIR OUT_JSON [WORKERS (8)]
exit 0 with OUT_JSON {source, seeded, refused}; 3 when SRC holds no derive dir."""
from __future__ import annotations

import glob
import gzip
import hashlib
import json
import os
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor

from verity.ml.circuits.codec import canonical_json

NAME = re.compile(r"build_step|build_request_LP\d+_T\d+")
CHUNK = 1 << 24


def _gz_bytes(path: str, keep: bool) -> tuple[bytes, bytes, bytes]:
    """(the whole text if `keep`, else b"", its first 64 KiB, its last 2 bytes), reading the stream to its end."""
    parts, head, tail = [], b"", b""
    with gzip.open(path, "rb") as f:
        while b := f.read(CHUNK):
            if keep:
                parts.append(b)
            if len(head) < 1 << 16:
                head += b[: (1 << 16) - len(head)]
            tail = (tail + b)[-2:]
    return b"".join(parts), head, tail


def check(d: str) -> str | None:
    """None when derive dir `d` may be seeded, else why not."""
    try:
        if open(f"{d}.rc").read().strip() != "0":
            return f".rc {open(f'{d}.rc').read().strip()!r}"
        res = json.load(open(f"{d}/result.json"))
        if res.get("ok") is not True:
            return "result.json not ok"
        prog = res.get("program") or {}
        if not prog.get("digest") or not prog.get("descriptor_sha256"):
            return "result.json names no program digest"
        names = set(os.listdir(d))
        for n in ("descriptor.json.gz", "instances.json.gz", "artifact.json", "construction_manifest.json"):
            if n not in names:
                return f"no {n}"
        for n in sorted(names):
            p = os.path.join(d, n)
            if n.endswith(".json"):
                json.load(open(p))
            elif n.endswith(".gz") and n != "descriptor.json.gz":
                _, head, tail = _gz_bytes(p, keep=False)
                if n == "instances.json.gz" and (f'"digest": "{prog["digest"]}"'.encode() not in head or tail != b"]}"):
                    return "instances.json.gz is not this Program's whole instance file"
        raw, _, _ = _gz_bytes(os.path.join(d, "descriptor.json.gz"), keep=True)
        if hashlib.sha256(raw).hexdigest() != prog["descriptor_sha256"]:
            return "descriptor.json.gz is not result.json's descriptor_sha256"
        desc = json.loads(raw)
        if hashlib.sha512(canonical_json({k: v for k, v in desc.items() if k != "annotations"})).hexdigest() != prog["digest"]:
            return "the descriptor's Program digest is not result.json's"
    except (OSError, ValueError, EOFError, gzip.BadGzipFile) as e:
        return f"{type(e).__name__}: {e}"[:300]
    return None


def main(src: str, dst: str, out: str, workers: int = 8) -> int:
    dirs = sorted(d for d in glob.glob(f"{src}/build/rank*/build_*") if os.path.isdir(d) and NAME.fullmatch(os.path.basename(d)))
    if not dirs:
        print(f"{src}: no derive dir under build/rank*/")
        return 3
    with ProcessPoolExecutor(max_workers=workers) as pool:
        why = dict(zip(dirs, pool.map(check, dirs)))
    seeded, refused = [], []
    for d in dirs:
        rel = os.path.relpath(d, src)
        if why[d] is not None:
            refused.append({"dir": rel, "why": why[d]})
            continue
        to = os.path.join(dst, rel)
        if os.path.exists(to):
            refused.append({"dir": rel, "why": "already in the destination"})
            continue
        os.makedirs(os.path.dirname(to), exist_ok=True)
        shutil.copytree(d, to, copy_function=os.link)
        for ext in (".rc", ".log"):
            if os.path.exists(d + ext):
                shutil.copy2(d + ext, to + ext)
        prog = json.load(open(f"{d}/result.json"))["program"]
        seeded.append({"dir": rel, "digest": prog["digest"], "descriptor_sha256": prog["descriptor_sha256"]})
    json.dump({"source": os.path.realpath(src), "seeded": seeded, "refused": refused}, open(out, "w"), indent=1)
    print(f"seeded {len(seeded)} derive dirs from {src}; refused {len(refused)}"
          + "".join(f"\n    {r['dir']}: {r['why']}" for r in refused))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3], *map(int, sys.argv[4:5])))
