"""The evidence store through this interpreter's research CLI (`python -m research data ...`), so `RESEARCH_STORE` /
`RESEARCH_STORE_CONFIG` select it: what the circuit database (`verity_vllm.records`) and a row's AOT Build (`pipeline.row_tp`) read."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def data(*args: str) -> str:
    """`research data ARGS`'s stdout (this interpreter's research, in this process's environment); a refusal raises with what the CLI
    said."""
    p = subprocess.run([sys.executable, "-m", "research", "data", *args], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"research data {' '.join(args[:2])}: rc={p.returncode}: {p.stderr.strip()[-1500:]}")
    return p.stdout


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
