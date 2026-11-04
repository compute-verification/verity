"""`verity graph --json` of the checkout ROOT under an audit hook: the graph's bytes on stdout, and in OUT a JSON list of every
path the listing opened to read or listed outside ROOT and the interpreter's own installation. A write-only open reads nothing
(the suite guard's tracer, inherited by this process, appends its log that way).

    python audit_listing.py ROOT OUT
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

root, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
inside = [root, *{Path(p).resolve() for p in (sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix)}]
outside: set[str] = set()
busy: list[int] = []


def hook(event: str, args: tuple) -> None:
    if busy or event not in ("open", "os.listdir", "os.scandir") or not args:
        return
    if not isinstance(args[0], str | bytes | os.PathLike):
        return
    if event == "open" and len(args) > 2 and isinstance(args[2], int) and args[2] & os.O_ACCMODE == os.O_WRONLY:
        return
    busy.append(1)
    try:
        p = Path(os.fsdecode(args[0])).resolve()
        if not any(p == d or p.is_relative_to(d) for d in inside):
            outside.add(str(p))
    finally:
        busy.pop()


sys.addaudithook(hook)
from verity_graph import cli  # noqa: E402

rc = cli.main(["--json", "--root", str(root)])
busy.append(1)
out.write_text(json.dumps(sorted(outside)))
sys.exit(rc)
