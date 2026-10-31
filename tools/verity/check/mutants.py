"""`mutants`: delete each check a verifier source runs, one at a time, and name each deletion its tests don't notice.

    uv run python tools/verity/check/mutants.py FILE --build CMD --test CMD [--only TEXT] [--line N ...] [--json OUT]

A check here is a statement of a `do` block that is a bare call of a function named `check…` (`checkIdentifier st own`,
`Registered.checkFull registered draw`): it returns nothing, and its only effect is to refuse. A line holding `:=`, `↔`,
`∧` or `→` is part of a declaration (a theorem's statement, a field's type), not a call, and a block comment holds no
call; a call is deleted with the lines that continue it (closing its brackets, or indented deeper). Each mutant replaces
one such line of FILE with `pure ()` at its indentation, runs `--build` (the verifier's incremental build, e.g. `lake
build flock-verify` from its package), then `--test`; a mutant whose test still exits 0 survived, which says no test
refuses an input that only that check refuses. `--only` keeps the checks whose line holds TEXT, `--line` those that start on line N. FILE is restored after
each mutant, and `--build` runs once more at the end, so the build matches the source again; a stop (SIGINT) restores it
too.

The test runs once on the unmutated source first, and a failure there exits 2: every mutant would be killed by it. A
mutant whose build fails is `unbuilt`, neither killed nor survived. Exit 0 when every mutant was killed, 1 when one survived
or wasn't built, 2 on a usage error or a failing baseline.

A merge that drops both of `FlockVerify.lean`'s `Registered.checkFull` calls while `check` stays green is the kind of
deletion these mutants are.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

#: a `do` statement that only calls a check: an optional namespace path, then `check` or `checkX…` (not `checked`), and its
#: arguments
CHECK = re.compile(r"^(?P<indent>[ \t]+)(?:[A-Z][\w']*\.)*check(?:[A-Z][\w']*)?(?:\s.*)?$")
#: what a `do` statement that only calls a check never holds, and an indented line of a theorem's statement or of a
#: structure's field may (`checkInRange k n m = .ok () ↔ … := by`, `checkLayout : Circuit → V Unit`)
DECLARATION = re.compile(r":=|↔|∧|→")
STRING = re.compile(r'"(?:\\.|[^"\\])*"')


@dataclass
class Mutant:
    line: int           # 1-based
    text: str
    outcome: str = ""   # killed, survived, unbuilt
    last: int = 0       # the call's last line, when its arguments continue past `line`


def _depth(code: str) -> int:
    code = STRING.sub('""', code)
    return sum(code.count(o) - code.count(c) for o, c in ("()", "[]", "{}"))


def mutants(text: str, only: str | None = None) -> list[Mutant]:
    """Each bare check call of ``text`` outside a block comment, as a mutant (holding ``only`` when given). A call takes
    the lines that continue it: those that close its open brackets, and those indented deeper than it."""
    out, comment = [], 0
    lines = text.splitlines()
    for i, line in enumerate(lines, 1):
        opened = comment
        comment += line.count("/-") - line.count("-/")
        if opened or comment:
            continue
        code = line.split("--", 1)[0].rstrip()
        if CHECK.match(code) and not DECLARATION.search(code) and (only is None or only in line):
            m, depth, indent = Mutant(i, line.strip()), _depth(code), len(line) - len(line.lstrip())
            while (m.last or i) < len(lines):
                nxt = lines[m.last or i]
                if depth <= 0 and not (nxt.strip() and len(nxt) - len(nxt.lstrip()) > indent):
                    break
                m.last = (m.last or i) + 1
                depth += _depth(nxt.split("--", 1)[0])
            out.append(m)
    return out


def mutate(text: str, line: int, last: int = 0) -> str:
    """``text`` with line ``line`` (1-based) replaced by `pure ()` at its indentation, and lines up to ``last`` emptied."""
    lines = text.splitlines(keepends=True)
    old = lines[line - 1]
    indent = old[:len(old) - len(old.lstrip())]
    lines[line - 1] = f"{indent}pure (){old[len(old.rstrip(chr(10))):]}"
    for j in range(line, last):
        lines[j] = lines[j][len(lines[j].rstrip(chr(10))):]
    return "".join(lines)


def _sh(cmd: str, log) -> int:
    return subprocess.run(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT).returncode


def sweep(path: Path, build: str, test: str, only: str | None, log, lines: set[int] | None = None) -> list[Mutant]:
    """Run every mutant of ``path`` (`mutants`) through ``build`` and ``test``; each with its outcome."""
    original = path.read_text()
    found = [m for m in mutants(original, only) if lines is None or m.line in lines]
    try:
        for m in found:
            path.write_text(mutate(original, m.line, m.last))
            if _sh(build, log) != 0:
                m.outcome = "unbuilt"
            else:
                m.outcome = "killed" if _sh(test, log) != 0 else "survived"
            print(f"{m.outcome:8} {path}:{m.line}  {m.text}", flush=True)
    finally:
        path.write_text(original)
        _sh(build, log)
    return found


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mutants", description=__doc__.split("\n\n")[0])
    ap.add_argument("file", type=Path)
    ap.add_argument("--build", required=True, help="the verifier's incremental build, a shell command")
    ap.add_argument("--test", required=True, help="the tests, a shell command; nonzero kills a mutant")
    ap.add_argument("--only", help="only the checks whose line holds this text")
    ap.add_argument("--line", type=int, action="append", help="only the check starting on this line (repeatable)")
    ap.add_argument("--json", type=Path, help="write the mutants and their outcomes here")
    ap.add_argument("--log", type=Path, help="the build's and tests' output (default: discarded)")
    a = ap.parse_args(argv)
    if not a.file.is_file():
        print(f"mutants: no file {a.file}", file=sys.stderr)
        return 2
    lines = set(a.line) if a.line else None
    if not [m for m in mutants(a.file.read_text(), a.only) if lines is None or m.line in lines]:
        print(f"mutants: no check call in {a.file}" + (f" holding {a.only!r}" if a.only else ""), file=sys.stderr)
        return 2
    with (a.log.open("w") if a.log else open("/dev/null", "w")) as log:
        if _sh(a.build, log) != 0 or _sh(a.test, log) != 0:
            print("mutants: the unmutated source fails its build or tests, so it would kill every mutant", file=sys.stderr)
            return 2
        found = sweep(a.file, a.build, a.test, a.only, log, lines)
    if a.json:
        a.json.write_text(json.dumps({"file": str(a.file), "build": a.build, "test": a.test,
                                      "mutants": [asdict(m) for m in found]}, indent=1) + "\n")
    left = [m for m in found if m.outcome != "killed"]
    print(f"{len(found)} mutants: {len(found) - len(left)} killed, "
          f"{sum(m.outcome == 'survived' for m in found)} survived, {sum(m.outcome == 'unbuilt' for m in found)} unbuilt")
    return 1 if left else 0


if __name__ == "__main__":
    sys.exit(main())
