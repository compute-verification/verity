#!/usr/bin/python3 -I
"""vy-verifier-figures TREE: each peer's figures file of /etc/vy/verifier.json, made as root from TREE's data sheets, which
vy-verifier-pin runs before `current` moves to TREE. A peer names its node kind (`node`, a node file of
calibration/hardware/nodes) and `figures`, /etc/vy/figures/NODE.json; the file is what TREE's `benchmarks/poub_session/sheets.py
NODE` writes, and nothing is written unless, for every peer, the digest TREE's `sheets.py --sha512 NODE` gives is the peer's
`figures_sha512`."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

CONFIG = Path("/etc/vy/verifier.json")
DIR = Path("/etc/vy/figures")
SHEETS = "benchmarks/poub_session/sheets.py"
TEMPLATE = "tools/research/src/research/pods/nebius/verifier/verifier.json"
KIND = re.compile(r"[a-z0-9][a-z0-9.-]{0,63}")


class Refused(Exception):
    pass


def figures(tree: Path, node: str, python: str) -> tuple[str, str]:
    """TREE's figures of a node of kind `node`, as `sheets.py` writes them, and their digest (`sheets.py --sha512`)."""
    out = []
    for args in ([node], ["--sha512", node]):
        r = subprocess.run([python, "-I", str(tree / SHEETS), *args], capture_output=True, text=True)
        if r.returncode:
            raise Refused(f"{SHEETS} {' '.join(args)} in {tree} failed ({r.returncode}): {r.stderr.strip()}")
        out.append(r.stdout)
    return out[0], out[1].strip()


def plan(tree: Path, settings: dict, *, out: Path = DIR, python: str = "/usr/bin/python3") -> dict[Path, str]:
    """Each figures file to write -> its text; Refused, naming the peer, for a peer that names no node kind, a `figures`
    elsewhere than `out`/NODE.json, or a `figures_sha512` that isn't the digest of what TREE's sheets give."""
    made: dict[str, tuple[str, str]] = {}
    files = {}
    for peer, entry in sorted(settings.get("peers", {}).items()):
        node = entry.get("node")
        if not isinstance(node, str) or not KIND.fullmatch(node):
            raise Refused(f"the peer {peer!r} names no node kind (`node`, a node file's name under calibration/hardware/nodes) "
                          f"in its entry: its figures are made from that node's data sheets")
        if entry.get("figures") != str(out / f"{node}.json"):
            raise Refused(f"the peer {peer!r} has `figures` {entry.get('figures')!r}, not {str(out / f'{node}.json')!r}, "
                          "where its node kind's figures are written")
        if node not in made:
            made[node] = figures(tree, node, python)
        text, got = made[node]
        if entry.get("figures_sha512") != got:
            raise Refused(f"the peer {peer!r} has figures_sha512 {entry.get('figures_sha512')}, but the figures that "
                          f"`{SHEETS} {node}` writes in {tree} have the digest {got}: the template ({TEMPLATE}) or the data sheets "
                          "(calibration/hardware) changed since the other was written. Put the new digest in the template "
                          "and install it, or pin the tree the template's digest was made at")
        files[out / f"{node}.json"] = text
    return files


def write(files: dict[Path, str]) -> None:
    for path, text in files.items():
        path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_text(text)
        tmp.chmod(0o644)
        if os.geteuid() == 0:
            os.chown(tmp, 0, 0)
        tmp.replace(path)


def main(argv: list[str] | None = None, *, config: Path = CONFIG, out: Path = DIR, python: str = "/usr/bin/python3") -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1 or not Path(args[0]).is_dir():
        print("usage: vy-verifier-figures TREE (a pinned tree, /opt/vy-verifier/COMMIT)", file=sys.stderr)
        return 2
    try:
        files = plan(Path(args[0]), json.loads(config.read_text()), out=out, python=python)
    except (Refused, OSError, ValueError) as e:
        print(f"vy-verifier-figures: {e}", file=sys.stderr)
        return 3
    write(files)
    for path in files:
        print(f"figures: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
