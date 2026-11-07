"""The serve dir's re-proof inputs: every file under D that one of onsite_record.py's `REPROOF` serve-dir paths names, with
<K> the audited window, <class> V3c's class tag and <tie> TIE (-tie where H1_TIE ties the window, else empty), one path
relative to D a line, sorted. Read from TREE's onsite_record.py
by its syntax alone (stdlib only, so it runs before the node's venv exists), with the tree's own `named`.

    python3 reproof_files.py TREE D K VTAG [TIE]
"""
from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path


def table(src: str) -> tuple[list[str], object]:
    mod = ast.parse(src)
    paths, fn = None, None
    for node in mod.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "REPROOF" for t in node.targets):
            paths = []
            for v in node.value.values:
                if isinstance(v, ast.Tuple) and isinstance(v.elts[0], ast.Constant) and v.elts[0].value == "serve dir":
                    paths.append(v.elts[1].value)
        if isinstance(node, ast.FunctionDef) and node.name == "named":
            fn = node
    if not paths or fn is None:
        raise SystemExit("REFUSED: the tree's onsite_record.py has no REPROOF serve-dir paths or no named()")
    ns: dict = {"re": re}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "onsite_record.named", "exec"), ns)
    return paths, ns["named"]


def main(tree: str, d: str, k: str, vtag: str, tie: str = "") -> int:
    paths, named = table((Path(tree) / "benchmarks/one_stage/onsite_record.py").read_text())
    rxs = [rx for p in paths for rx in named(p.replace("<K>", k).replace("<class>", vtag).replace("<tie>", tie))]
    root = Path(d)
    files = sorted(str(Path(dp, f).relative_to(root)) for dp, _, fs in os.walk(root) for f in fs
                   if not Path(dp, f).is_symlink())
    print("".join(f + "\n" for f in files if any(rx.fullmatch(f) for rx in rxs)), end="")
    return 0


if __name__ == "__main__":
    if len(sys.argv) not in (5, 6):
        raise SystemExit(__doc__)
    raise SystemExit(main(*sys.argv[1:]))
