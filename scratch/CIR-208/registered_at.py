#!/usr/bin/env python3
"""The registries' ids at a worktree of another commit, with the checkout's environment: `registered_at.py WORKTREE`.

Prints {id: [the file that defines it, whether it is a Boolean version]} as the test's own loader gives it, with the
import roots of the checkout's environment pointed at WORKTREE (a root that isn't there in WORKTREE is tried at the place
it had before the protocols went into groups).
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path("/agent/repos/verity")
SP = REPO / ".venv/lib/python3.12/site-packages"
PY = REPO / ".venv/bin/python"

LOAD = r"""
import json, sys
sys.path.insert(0, "tools/verity/circuit_check/tests")
import test_superseded_versions as t
import verity, circuit_check, verity_vllm
here = str(t.REPO)
assert all(m.__file__.startswith(here) for m in (verity, circuit_check, verity_vllm)), (verity.__file__, circuit_check.__file__)
print(json.dumps({i: [f, b] for i, (f, b) in sorted(t.registered().items())}))
"""


def roots() -> list[str]:
    out: list[str] = []
    for f in sorted(SP.glob("_editable_impl_*.pth")):
        line = f.read_text().strip().splitlines()[0]
        if line.startswith("/"):
            out.append(line)
        else:
            for m in re.findall(r"/agent/repos/verity[^'\"]*", (SP / f"{line.split()[1]}.py").read_text()):
                p = Path(m)
                out.append(str(p.parent.parent if p.name == "__init__.py" else p.parent))
    return list(dict.fromkeys(out))


def at(wt: Path) -> list[str]:
    out = []
    for r in roots():
        rel = Path(r).relative_to(REPO)
        old = Path(re.sub(r"^verity/protocols/(proofs|accounting|policy)/", "verity/protocols/", str(rel)))
        out.append(str(wt / rel if (wt / rel).is_dir() or not (wt / old).is_dir() else wt / old))
    return out


def main(wt: Path) -> int:
    env = {**os.environ, "PYTHONPATH": ":".join(at(wt))}
    env.pop("VIRTUAL_ENV", None)
    p = subprocess.run([str(PY), "-c", LOAD], cwd=wt, env=env, capture_output=True, text=True)
    if p.returncode:
        print(p.stderr[-3000:], file=sys.stderr)
        return 1
    print(p.stdout.strip().splitlines()[-1])
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]).resolve()))
