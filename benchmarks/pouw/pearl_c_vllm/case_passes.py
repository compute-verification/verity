"""A Pearl+ run's passes, rebuilt from the parts of its case: `served_passes.py case`'s inverse.

    case_passes.py CASE PASSES

`served_passes.py case` leaves the served node with the parts of the run's case (``CASE/parts.json``: the epoch's coin
commitment, each retained pass's manifest, and each pass's files the opened tiles read, by the SHA-512 of the blob
holding them) and those blobs (``CASE/blobs/<sha512>``). This writes them back as the directory the verifiers read, so
`served_passes.py verify` and `pouw-pearlplus` run off the served node on the verifier's recorded coin:
``PASSES/coins.json`` and, for the i-th retained pass, ``PASSES/pass-0000i/manifest.json`` with each file the parts name
at its path, a link to its blob. A blob is linked only once its bytes hash to its name, so a file a verifier reads there
is the one the case commits to; a file the parts don't name isn't there, and a verifier that asks for one fails on it.

The blobs are opened tile values, witness: PASSES is created 0700 and must not exist, and nothing here copies a blob.
The manifests are written from the parts' parsed form (one-space indent), so a pass's name there (its path and its
manifest's SHA-256) is not the served one's; the run's commitment, which the coin record pins, reads their fields alone.
Exit 0 with one line counting what was written, 2 with ``case_passes REFUSED: why`` and nothing usable left behind.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

SCHEMA = "verity-pouw/pearl-plus-case-parts/v1"
PARTS, BLOBS = "parts.json", "blobs"


class Refused(ValueError):
    pass


def _sha512(path: Path) -> str:
    h = hashlib.sha512()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def rebuild(case: Path, passes: Path) -> dict[str, int]:
    """Write the case's passes under ``passes`` (module docstring) and return what was written: the passes, the files
    linked, and the distinct blobs and their bytes. `Refused` names the first thing wrong."""
    try:
        parts = json.loads((case / PARTS).read_text())
    except (OSError, ValueError) as e:
        raise Refused(f"{case / PARTS} can't be read ({type(e).__name__}: {e})") from e
    if not isinstance(parts, dict) or parts.get("schema") != SCHEMA:
        raise Refused(f"{case / PARTS} is not Pearl+'s case's parts ({SCHEMA})")
    manifests, files = parts.get("manifests"), (parts.get("evidence") or {}).get("files")
    coins = (parts.get("params") or {}).get("coins")
    if not (isinstance(manifests, list) and manifests and isinstance(files, list) and len(files) == len(manifests)
            and all(isinstance(m, dict) for m in manifests) and all(isinstance(f, dict) for f in files)):
        raise Refused("the parts don't hold one manifest and one set of files per pass")
    if coins is None:
        raise Refused("the parts hold no coin commitment of the epoch (params.coins)")
    held: dict[str, int] = {}
    for h in sorted({h for f in files for h in f.values()}):
        blob = case / BLOBS / str(h)
        if not blob.is_file():
            raise Refused(f"the parts name blob {str(h)[:16]}..., which is no file in {case / BLOBS}")
        if _sha512(blob) != h:
            raise Refused(f"blob {str(h)[:16]}...'s bytes don't hash to its name")
        held[h] = blob.stat().st_size
    blobs = (case / BLOBS).resolve()
    passes.mkdir(mode=0o700)
    (passes / "coins.json").write_text(json.dumps(coins, indent=1) + "\n")
    for i, (man, mine) in enumerate(zip(manifests, files, strict=True)):
        d = passes / f"pass-{i:05d}"
        d.mkdir(mode=0o700)
        (d / "manifest.json").write_text(json.dumps(man, indent=1) + "\n")
        for rel, h in sorted(mine.items()):
            at = d / rel
            if Path(rel).is_absolute() or d.resolve() not in at.resolve().parents:
                raise Refused(f"pass {i} names a file outside it: {rel}")
            at.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.symlink(blobs / h, at)
    return {"passes": len(manifests), "files": sum(map(len, files)), "blobs": len(held), "bytes": sum(held.values())}


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print(__doc__.split("\n\n")[1].strip(), file=sys.stderr)
        return 64
    case, passes = map(Path, args)
    try:
        if passes.exists():
            raise Refused(f"{passes} exists: the passes are rebuilt into a directory of their own")
        got = rebuild(case, passes)
    except Refused as e:
        print(f"case_passes REFUSED: {e}", flush=True)
        return 2
    print(f"case_passes: {got['passes']} passes, {got['files']} files the opened tiles read, {got['blobs']} blobs of "
          f"{got['bytes']:,} bytes -> {passes}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
