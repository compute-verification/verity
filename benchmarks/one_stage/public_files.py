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
#: the files a served run may leave under its output directory (`fnmatch`, relative to $OUT), each with what of it is
#: public; every other file stays in the serve dir's private directory. Read by nothing yet.
PUBLIC = {
    "verdicts.txt": "its say lines, each a tool's verdict or its failure's class and exit, no tool-log line",
    "machine.txt": "the tree's commit alone",
    "weights_registration.txt": "the deployment record's digest line alone",
    "registration.json": "the window's registration record",
    "compiler.txt": "the registered compiler's digest",
    "serve/phase.json": "the execution phase's public layout",
    "serve/h1/window-program.json.gz": "H1's window Program",
    "serve/h1/window-program.meta.json": "its digest and size",
    "window-circuit-*.json": "a class's window-circuit count and record digest",
    "block-counts-*.json": "the inference-only block counts",
    "inference-only-policy.json": "the inference-only policy's verdict",
    "inference-only-limits.json": "its limits",
    "record/inference-only.json": "the inference-only verdict of record",
    "served-compare/*.json": "the served comparison's verdict and circuit facts",
    "v3c-audit/registration.json": "V3c's registration record",
    "v3c-audit/summary.json": "V3c's audit summary",
    "v3c-audit/window/audit.json": "its window's audit",
    "pieces/*/answer.json": "a piece's answer",
    "pieces/*/development.json": "a development piece's marker",
    "pieces/*/piece.json": "a piece's statement",
    "pieces/*/served.json": "a piece's served counts by kind",
    "pieces/*/runs.json": "a piece's run ids",
    "pieces/*/coin.json": "a piece's coins",
    "pieces/*/ledger.jsonl": "a piece's ledger of sessions",
    "pieces/*/refused.txt": "a piece's refusal",
    "pieces/*/session/*/out/key.json": "a session's public key",
    "record/*": "the final record, without its node",
    "composite/**": "the composite",
    "compose/**": "the compose gate's answer",
    "sampled-proofs/**": "the sampled proofs' records, sessions and ran_with.txt",
}


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
