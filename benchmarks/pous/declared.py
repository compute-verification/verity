"""The Lean executables storage's, the bandwidth audit's and the wipe's pieces build and answer with: their schemes' declared
ones (`Launch.executables`), read here and nowhere else, so no variable, flag or file of a node's names another.

    declared.py SCHEME...            # each declared executable of SCHEME's piece, a line `PACKAGE TARGET`
    declared.py --built SCHEME TARGET ROOT   # where `lake build TARGET` writes it in its package under ROOT

A node writes its executables' SHA-512s into its own record (`period.decided_by`): they are held only where an
auditor-side composer compares them with these targets' builds.
"""
from __future__ import annotations

import sys
from pathlib import Path

from verity.protocols.compose import schemes
from verity.protocols.resource_accounting.scheme import Executable

ROOT = Path(__file__).resolve().parents[2]
#: the developer-side party whose piece each of the three schemes declares
PARTY = "prover"


def executables(scheme: str) -> dict[str, Executable]:
    """``scheme``'s declared executables by target."""
    return {e.target: e for e in schemes.get(scheme).pieces[PARTY].executables}


def built(scheme: str, target: str, root: Path | str = ROOT) -> Path:
    """Where ``lake build`` writes ``scheme``'s declared executable ``target`` in its package under ``root``; LookupError
    for a target ``scheme`` doesn't declare."""
    found = executables(scheme)
    if target not in found:
        raise LookupError(f"{target} is no executable {scheme} declares ({', '.join(sorted(found))})")
    return found[target].built(root)


def main(argv: list[str]) -> int:
    try:
        if argv[:1] == ["--built"] and len(argv) == 4:
            print(built(*argv[1:]))
        elif argv and not argv[0].startswith("-"):
            for name in argv:
                for e in executables(name).values():
                    print(e.package, e.target)
        else:
            print(__doc__.split("\n\n")[1], file=sys.stderr)
            return 2
    except (KeyError, LookupError) as e:
        print(f"declared.py: {e}", file=sys.stderr)
        return 2
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
