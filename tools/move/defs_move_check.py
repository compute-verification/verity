"""`move_check` for the Definitions' move to verity.ml.definitions: BASE..HEAD with `--map` extras from defs_map.toml.

The move crosses out of the integration, so a path the codemod writes from the integration's root
(`verity_vllm/program/registry/b1.py`) lands under `verity/` with no directory the two share, and it splits `registry/` and
`kernels/`, which no rename pairs whole. Each `[paths]` entry, from the root and from the integration's root, and each
`[aliases]` entry is an extra. `[modules]` is not: move_check pairs an import line by its dotted names itself, and a line
an extra has already rewritten no longer pairs.

    python3 tools/move/defs_move_check.py BASE HEAD [--json]
"""
from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = "integrations/vllm/"


def extras(map_file: Path = HERE / "defs_map.toml") -> dict[str, str]:
    m = tomllib.loads(map_file.read_text())
    out: dict[str, str] = {}
    for old, new in m["paths"].items():
        o, n = old.strip("/"), new.strip("/")
        out[o] = n
        if o.startswith(PKG):
            out[o[len(PKG):]] = n.removeprefix(PKG)
    out.update(m.get("aliases", {}))
    return out


if __name__ == "__main__":
    check = HERE.parent / "verity" / "check" / "move_check.py"
    os.execv(sys.executable, [sys.executable, str(check), *sys.argv[1:], *(f"--map={o}={n}" for o, n in extras().items())])
