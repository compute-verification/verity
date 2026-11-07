"""circuits_out.py VW OUT [--max-bytes N] [--keep DIR]: V3c's public C-Flock circuit files and their pins into OUT, after
`served_verdicts.sh v3c` has published OUT (a tree without the period's --circuits, whose publish would rewrite their
wire names and whose kind scan declares no circuit file).

Each template's file is the audit's own, `VW/audit/window/m<k>/circuit.txt` at the template's first member k (members by
base), held to its pin, `VW/audit/window/summary.json` `circuits.<template>.pin`, the SHA-512 every session of that
template was held to: a symlink or a file of another digest is refused, never copied. A file of at most --max-bytes
(1 GiB) goes to OUT/v3c-circuits/<sha512>.txt; a larger one to --keep DIR/<sha512>.txt on this node, a path naming no
window (default the serve dir's v3c-circuits/, beside VW). Each is a hard link where it can be, so a window's ~550 GB
of files take no second copy: a file in OUT is copied only where OUT is on another filesystem, and a larger one is
never copied, so where DIR is on another filesystem the row names the audit's own file. OUT/v3c-circuits/pins.json lists
every template in order with its pin, bytes and where its file is. Exit 1 on a refusal, with pins.json saying why by
paths under VW alone, since VW's name holds the window's number."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import sys
from pathlib import Path


def sha512(p: Path) -> str:
    h = hashlib.sha512()
    with open(p, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def place(f: Path, dst: Path, copy: bool) -> Path:
    """`f` at `dst` as a hard link, else (another filesystem) a copy where `copy`, else `f` where it lies."""
    dst.unlink(missing_ok=True)
    try:
        os.link(f, dst)
    except OSError:
        if not copy:
            return f
        shutil.copyfile(f, dst)
    return dst


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("vw", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--max-bytes", type=int, default=1 << 30)
    ap.add_argument("--keep", type=Path)
    a = ap.parse_args(argv)
    a.keep = a.keep or a.vw.resolve().parent / "v3c-circuits"
    win, dst = a.vw / "audit" / "window", a.out / "v3c-circuits"
    dst.mkdir(parents=True, exist_ok=True)
    rows, why = [], None
    try:
        s = json.loads((win / "summary.json").read_text())
        first: dict[str, int] = {}
        for m in sorted(s.get("members") or (), key=lambda m: m["base"]):
            first.setdefault(m["template"], m["k"])
        if not first:
            raise ValueError("the audit's summary lists no members")
        for t, k in first.items():
            f, pin = win / f"m{k}" / "circuit.txt", ((s.get("circuits") or {}).get(t) or {}).get("pin")
            if pin is None or f.is_symlink() or not f.is_file():
                raise ValueError(f"template {t}: no pin, or m{k}/circuit.txt isn't a file")
            if (got := sha512(f)) != pin:
                raise ValueError(f"template {t}: m{k}/circuit.txt hashes to {got[:16]}…, not its pin {pin[:16]}…")
            row = {"template": t, "sha512": pin, "bytes": f.stat().st_size}
            if row["bytes"] <= a.max_bytes:
                place(f, dst / f"{pin}.txt", copy=True)
                row["file"] = f"v3c-circuits/{pin}.txt"
            else:
                a.keep.mkdir(parents=True, exist_ok=True)
                kept = place(f, a.keep / f"{pin}.txt", copy=False)
                row["kept"] = (f"{socket.gethostname()}:{kept}" if kept != f
                               else f"{socket.gethostname()}: the audit's own, VW/audit/window/m{k}/circuit.txt")
            rows.append(row)
    except OSError as e:
        p = Path(e.filename).resolve() if e.filename else None
        rel = "?" if p is None else p.relative_to(a.vw.resolve()) if p.is_relative_to(a.vw.resolve()) else p.name
        why = f"{type(e).__name__} on {rel}: {e.strerror}"
    except (KeyError, TypeError, ValueError) as e:
        why = f"{type(e).__name__}: {e}"
    doc = {"circuits": rows, "max_bytes": a.max_bytes,
           "pins": "the audit's summary.json circuits.<template>.pin: each file's SHA-512, every session's"}
    if why:
        doc["error"] = why
    (dst / "pins.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    inn = [r for r in rows if "file" in r]
    print(f"V3C-CIRCUITS: {len(rows)} templates, {len(inn)} in OUT ({sum(r['bytes'] for r in inn)} bytes), "
          f"{len(rows) - len(inn)} kept on this node" + (f"; REFUSED: {why}" if why else ""), flush=True)
    return 1 if why else 0


if __name__ == "__main__":
    sys.exit(main())
