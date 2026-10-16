"""`ops`: operations on the shared machines. `research deploy|keeper` hand off here by exec, with no import."""

from __future__ import annotations

import importlib
import sys

USAGE = """usage: ops COMMAND [ARGS...]

  ops deploy drift [--on N ...] [--json] | install NODE --by WHO [--ref REF] [PATH ...] [--apply]
                    # what runs on the Nebius nodes (deploy.toml) against main; installs from a pushed commit (verity_ops/deploy.py)
  ops keeper set|stop|status ...                     # keep a job running on a node until a time (verity_ops/keeper.py)
"""

COMMANDS = {"deploy": "deploy", "keeper": "keeper"}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(USAGE, end="")
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd not in COMMANDS:
        print(f"ops: unknown command {cmd!r}\n{USAGE}", end="", file=sys.stderr)
        return 2
    return importlib.import_module(f"verity_ops.{COMMANDS[cmd]}").main(rest)
