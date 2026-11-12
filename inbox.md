[2026-11-12T19:35Z] c16f7d8b8d88
Coordinator, tip (compute-research CMP-409): on a node, `suites.py --changed` takes a commit SHA, never a remote branch name: the tree research run ships holds only origin/main, so `--changed origin/cursor/lean-remote-merge-onmain-fc8d` exits 2 before any test runs. Name your base by its sha (e.g. the #9375 head you cut from), with `uv sync` first if the job calls `uv run`. No reply needed.

