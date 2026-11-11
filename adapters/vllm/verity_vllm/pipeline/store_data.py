"""The `research data` CLI as the pipeline and the circuit database (`circuit_db`) run it: this interpreter's `research`, in this
process's environment."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    p = subprocess.run([sys.executable, "-m", "research", "data", *args], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"research data {' '.join(args[:2])}: rc={p.returncode}: {p.stderr.strip()[-1500:]}")
    return p


def data(*args: str) -> str:
    """`research data ARGS`'s stdout (this interpreter's research, in this process's environment); a refusal raises with what the CLI
    said."""
    return _run(*args).stdout


def selected_from(*args: str) -> str:
    """What `research data select ARGS` read, as its summary line names it: "the index" (Neon's with $STORE_PRIVATE_READER_URL, or this
    machine's `research data index` build) or "the catalog" (this store's own, where there is no index)."""
    p = _run("select", *args)
    if m := re.search(r"artifact\(s\) from (the index|the catalog)\b", p.stderr):
        return m[1]
    raise RuntimeError(f"research data select {' '.join(args)}: its summary names neither the index nor the catalog: "
                       f"{p.stderr.strip()[-500:]!r}")


def fetch(art: str, to: Path | None = None, paths: list[str] = ()) -> Path:
    """Materialise `art`'s payload (only `paths` of a tree, when given) and return its path.  A store with no remote (a `research
    run` job's) that lacks it names the bundle that serves a job instead."""
    try:
        out = data("fetch", art, *(["--to", str(to)] if to else []), *(f"--path={p}" for p in paths))
    except RuntimeError as e:
        if "no remote configured" not in str(e):
            raise
        raise RuntimeError(f"{e}: a research run job has no store; the launcher writes the row's bundle (`verity-vllm records fetch "
                           f"ROW --to FILE`), sends it (`research run --send FILE`) and the job reads it (`records programs ROW --bundle "
                           f"$RESEARCH_RUN_DIR/inputs/FILE`, `programs(row, bundle=...)`)") from None
    return Path(out.strip().splitlines()[-1])


def listing(art: str) -> list[str]:
    """The paths of a tree artifact's files."""
    return [f["path"] for f in json.loads(data("fetch", art, "--list", "--json"))]
