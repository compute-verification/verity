"""The one rule for which files a launcher may copy under its run's output directory: a served run's witness
(`served_verdicts.witness`, its three tables) and what a launcher's own work directory adds to them are never copied.

    find . -type f -print0 | python3 public_files.py filter | xargs -0 -r -I{} cp --parents {} "$OUT/dir/"

`filter` reads NUL-separated paths on stdin and writes the public ones, NUL-separated, on stdout, in order; `why PATH` says
why a path is not public.
"""
from __future__ import annotations

import fnmatch
import sys
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))
from served_verdicts import witness  # noqa: E402

#: a directory of this name, anywhere in a path, holds a prover's or a device's private state
PRIVATE_DIRS = {"private": "a private directory (prover or device state)"}
#: a file or directory name (`fnmatch`) the three witness tables do not name
PRIVATE_NAMES = {"*-private.bin": "a private file (prover only)", "index-salt.bin": "the index's salt",
                 "salts.bin": "a commitment's salts", "saved-*.tar": "a staged units' value archive", "keys.bin": "a key file"}


def private_why(path: str) -> str | None:
    """Why ``path`` (relative to a work directory) is never copied under a run's output directory, None when it may be."""
    parts = Path(path).parts
    why = next((PRIVATE_DIRS[x] for x in parts if x in PRIVATE_DIRS), None)
    why = why or next((w for x in parts for pat, w in PRIVATE_NAMES.items() if fnmatch.fnmatchcase(x, pat)), None)
    if why:
        return why
    held = witness([path])
    return held[0][1] if held else None


def public(paths: Iterable[str]) -> list[str]:
    return [p for p in paths if p and private_why(p) is None]


def main(argv: list[str]) -> int:
    if argv[:1] == ["filter"] and len(argv) == 1:
        data = sys.stdin.buffer.read().decode("utf-8", "surrogateescape").split("\0")
        keep = public(data)
        sys.stdout.buffer.write("".join(p + "\0" for p in keep).encode("utf-8", "surrogateescape"))
        return 0
    if argv[:1] == ["why"] and len(argv) == 2:
        print(private_why(argv[1]) or "public")
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
