"""Old flag spellings kept as hidden aliases (one name per concept: tests/test_cli_conventions.py).

`renamed(argv, {"--node": "--on"}, prog)` reads `--node X` and `--node=X` as `--on`, and says once on stderr which flag to use,
so a unit file or script that still passes the old name keeps working while the parser defines only the new one."""
from __future__ import annotations

import sys


def renamed(argv: list[str] | None, names: dict[str, str], prog: str) -> list[str]:
    out, said = [], set()
    for a in sys.argv[1:] if argv is None else argv:
        old, eq, value = a.partition("=")
        if old in names:
            if old not in said:
                print(f"{prog}: {old} is now {names[old]}", file=sys.stderr)
                said.add(old)
            a = names[old] + eq + value
        out.append(a)
    return out
