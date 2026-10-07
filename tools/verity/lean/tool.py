"""Store declarations for the Lean audit's scheduled runs (research.store README §6.1; `tools/verity/lean/README.md`).

    research run --on MACHINE --project verity --source . --cwd source --tool lean_audit -- python tools/verity/lean/check.py --all --build --fresh
    research run --on MACHINE --project verity --source . --cwd source --tool lean_upstream -- \
        python tools/verity/lean/upstream.py Security/Proofs --fetch --rev origin/main

The steward launches the first nightly on `main` and the second weekly (`[[run]]` entries of its steward.toml,
`research.notes.run_lines`).  A nonzero exit is a failed attempt: the audit failed, or upstream proves a theorem about a
watched statement that nobody has acknowledged, so a bump would gain something.  In a recorded run `audit.py` writes its
reports to `$RESEARCH_RUN_DIR/lean-audit`, and `upstream.py` prints its report.  The key params are what is audited and
how; `--rev` names a branch, so the upstream commit a run scanned is the first line it prints, not part of its derivation.
Nothing here imports anything but `research.store.tool`, and nothing else in this directory imports this file.
"""
from __future__ import annotations

from research.store.tool import Tool

LEAN_AUDIT = Tool(name="lean_audit", version="1", command=["python", "tools/verity/lean/check.py"],
                  key_params=["all", "fresh", "no_replay", "_positional"])
LEAN_UPSTREAM = Tool(name="lean_upstream", version="1", command=["python", "tools/verity/lean/upstream.py"],
                     key_params=["_positional", "rev", "lean"])
