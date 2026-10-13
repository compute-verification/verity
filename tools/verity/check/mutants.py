"""`mutants`: delete each check a verifier source runs, one at a time, and name each deletion its tests don't notice.

    uv run python tools/verity/check/mutants.py FILE --build CMD --test CMD [--only TEXT] [--json OUT]

A check here is a statement of a `do` block that is a bare call of a function named `check…` (`checkIdentifier st own`,
`Registered.checkFull registered draw`): it returns nothing, and its only effect is to refuse. Each mutant replaces one such
line of FILE with `pure ()` at its indentation, runs `--build` (the verifier's incremental build, e.g. `lake build
flock-verify` from its package), then `--test`; a mutant whose test still exits 0 survived, which says no test refuses an
input that only that check refuses. `--only` keeps the checks whose line holds TEXT. FILE is restored after each mutant, and
`--build` runs once more at the end, so the build matches the source again; a stop (SIGINT) restores it too.

The test runs once on the unmutated source first, and a failure there exits 2: every mutant would be killed by it. A
mutant whose build fails is `unbuilt`, neither killed nor survived. Exit 0 when every mutant was killed, 1 when one survived
or wasn't built, 2 on a usage error or a failing baseline.

Started from rec-lean's merge that dropped both of `FlockVerify.lean`'s `Registered.checkFull` calls while `check` stayed
green (F7, 10 Oct): a deletion like that is one of these mutants.
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


@dataclass
class Mutant:
    line: int           # 1-based
    text: str
    outcome: str = ""   # killed, survived, unbuilt


def mutants(text: str, only: str | None = None) -> list[Mutant]:
    """Each line of ``text`` that is a bare check call, as a mutant (holding ``only`` when given)."""
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        code = line.split("--", 1)[0].rstrip()
        if CHECK.match(code) and (only is None or only in line):
            out.append(Mutant(i, line.strip()))
    return out


def mutate(text: str, line: int) -> str:
    """``text`` with line ``line`` (1-based) replaced by `pure ()` at its indentation."""
    lines = text.splitlines(keepends=True)
    old = lines[line - 1]
    indent = old[:len(old) - len(old.lstrip())]
    lines[line - 1] = f"{indent}pure (){old[len(old.rstrip(chr(10))):]}"
    return "".join(lines)


def _sh(cmd: str, log) -> int:
    return subprocess.run(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT).returncode


def sweep(path: Path, build: str, test: str, only: str | None, log) -> list[Mutant]:
    """Run every mutant of ``path`` (`mutants`) through ``build`` and ``test``; each with its outcome."""
    original = path.read_text()
    found = mutants(original, only)
    try:
        for m in found:
            path.write_text(mutate(original, m.line))
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
    ap.add_argument("--json", type=Path, help="write the mutants and their outcomes here")
    ap.add_argument("--log", type=Path, help="the build's and tests' output (default: discarded)")
    a = ap.parse_args(argv)
    if not a.file.is_file():
        print(f"mutants: no file {a.file}", file=sys.stderr)
        return 2
    if not mutants(a.file.read_text(), a.only):
        print(f"mutants: no check call in {a.file}" + (f" holding {a.only!r}" if a.only else ""), file=sys.stderr)
        return 2
    with (a.log.open("w") if a.log else open("/dev/null", "w")) as log:
        if _sh(a.build, log) != 0 or _sh(a.test, log) != 0:
            print("mutants: the unmutated source fails its build or tests, so it would kill every mutant", file=sys.stderr)
            return 2
        found = sweep(a.file, a.build, a.test, a.only, log)
    if a.json:
        a.json.write_text(json.dumps({"file": str(a.file), "build": a.build, "test": a.test,
                                      "mutants": [asdict(m) for m in found]}, indent=1) + "\n")
    left = [m for m in found if m.outcome != "killed"]
    print(f"{len(found)} mutants: {len(found) - len(left)} killed, "
          f"{sum(m.outcome == 'survived' for m in found)} survived, {sum(m.outcome == 'unbuilt' for m in found)} unbuilt")
    return 1 if left else 0


if __name__ == "__main__":
    sys.exit(main())
