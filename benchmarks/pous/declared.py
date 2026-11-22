"""The Lean executables storage's, the bandwidth audit's and the wipe's pieces build and answer with: their schemes' declared
ones (`Launch.executables`), read here and nowhere else, so no variable, flag or file of a node's names another.

    declared.py SCHEME...            # each declared executable of SCHEME's prover piece, a line `PACKAGE TARGET`
    declared.py --built SCHEME TARGET ROOT   # where `lake build TARGET` writes it in its package under ROOT
"""
from __future__ import annotations

import sys
from pathlib import Path

from verity.protocols.run import schemes


def built(scheme: str, target: str, root: Path | str) -> Path:
    """Where ``lake build`` writes ``scheme``'s declared executable ``target`` under ``root``; LookupError for a target
    ``scheme`` doesn't declare."""
    found = {e.target: e for e in schemes.get(scheme).pieces["prover"].executables}
    if target not in found:
        raise LookupError(f"{target} is no executable {scheme} declares ({', '.join(sorted(found))})")
    return found[target].built(root)


def main(argv: list[str]) -> int:
    try:
        if argv[:1] == ["--built"] and len(argv) == 4:
            print(built(*argv[1:]))
        elif argv and not argv[0].startswith("-"):
            print("\n".join(f"{e.package} {e.target}" for n in argv for e in schemes.get(n).pieces["prover"].executables))
        else:
            print(__doc__.split("\n\n")[1], file=sys.stderr)
            return 2
    except (KeyError, LookupError) as e:
        print(f"declared.py: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
