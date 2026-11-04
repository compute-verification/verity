"""covert_witness's node list (`nodes.txt`), which launch.py and promote.py read: one node a line, `#` comments,
`NODE pool=NAME` for a node that takes runs only through a slot pool and `cpu=NAME` its CPU pool; `vy-dc-h200*` is
refused whatever the list says."""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import Path

REFUSED_NODES = ("vy-dc-h200*",)


class Refused(RuntimeError):
    pass


@dataclass(frozen=True)
class Node:
    name: str
    pool: str = ""
    cpu: str = ""


def nodes(path: Path) -> list[Node]:
    out = []
    for line in Path(path).read_text().splitlines():
        words = line.split("#", 1)[0].split()
        if not words or any(fnmatch.fnmatchcase(words[0], pat) for pat in REFUSED_NODES):
            continue
        opts = dict(w.split("=", 1) for w in words[1:] if "=" in w)
        out.append(Node(words[0], opts.get("pool", ""), opts.get("cpu", "")))
    if not out:
        raise Refused(f"{path}: no usable node")
    return out
