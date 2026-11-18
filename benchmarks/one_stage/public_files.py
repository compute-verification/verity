"""The one list of the files a served run's launcher lets go, each with what of it is public: under ``OUT/`` the files that
may leave the node (`served_verdicts.publish` holds every other file in the serve dir's private directory), under ``D/``
the serve dir's files a compose hand-off may carry to COMPOSE_NODE (every other file refuses the hand-off).

    find . -type f -print0 | python3 public_files.py filter | xargs -0 -r -I{} cp --parents {} "$OUT/dir/"
    python3 public_files.py handoff < inputs.txt

`filter` reads NUL-separated paths on stdin and writes the public ones, NUL-separated, on stdout, in order; `why PATH` says
why a path is not public. `handoff` reads a hand-off's paths (relative to the serve dir), one a line, prints each one the
list has no ``D/`` entry for, and exits 1 when it printed any.
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
#: the serve's own records, which leave only as vLLM's mirror writes them: its table (`serve_mirror.PUBLIC`) names their
#: public fields, and this list names the files alone
MIRROR = "through the mirror (serve_mirror.PUBLIC names its fields)"
#: Every file a served run lets go, by its root and path (`listed`: ``*`` is one path part, ``**`` any number of them),
#: each with what of it is public. ``OUT/``: what may leave the node; ``D/``: what a compose hand-off may carry.
PUBLIC = {
    "OUT/verdicts.txt": "its say lines, each a tool's verdict or its failure's class and exit, no tool-log line",
    "OUT/machine.txt": "the tree's commit alone",
    "OUT/main.txt": "main's first-parent line, the commits a challenger session pins on",
    "OUT/weights_registration.txt": "the deployment record's digest line alone",
    "OUT/registration.json": "the window's registration record",
    "OUT/compiler.txt": "the registered compiler's digest",
    "OUT/serve/serve.json": MIRROR,
    "OUT/serve/windows.jsonl": MIRROR,
    "OUT/serve/seals.jsonl": MIRROR,
    "OUT/serve/phase.json": "the execution phase's public layout",
    "OUT/serve/period.json": "the period's public layout",
    "OUT/serve/circuit-files.json": "H1's circuit files' digests",
    "OUT/serve/overhead.json": "the serve's overhead aggregate (served_overhead.py metrics)",
    "OUT/serve/h1/window-program.json.gz": "H1's window Program",
    "OUT/serve/h1/window-program.meta.json": "its digest and size",
    "OUT/window-circuit-*.json": "a class's window-circuit count and record digest",
    "OUT/block-counts-*.json": "the inference-only block counts",
    "OUT/inference-only-policy.json": "the inference-only policy's verdict",
    "OUT/inference-only-limits.json": "its limits",
    "OUT/served-compare/*.json": "the served comparison's verdict and circuit facts",
    "OUT/v3c-audit/registration.json": "V3c's registration record",
    "OUT/v3c-audit/summary.json": "V3c's audit summary",
    "OUT/v3c-audit/period.json": "the audit as the network attester's period reads it",
    "OUT/v3c-audit/main.txt": "main's first-parent line its challenger session pins on",
    "OUT/v3c-audit/window/audit.json": "its window's audit",
    "OUT/pieces/*/answer.json": "a piece's answer",
    "OUT/pieces/*/case.json": "a piece's case",
    "OUT/pieces/*/development.json": "a development piece's marker",
    "OUT/pieces/*/piece.json": "a piece's statement",
    "OUT/pieces/*/served.json": "a piece's served counts by kind",
    "OUT/pieces/*/runs.json": "a piece's run ids",
    "OUT/pieces/*/coin.json": "a piece's coins",
    "OUT/pieces/*/ledger.jsonl": "a piece's ledger of sessions",
    "OUT/pieces/*/refused.txt": "a piece's refusal",
    "OUT/pieces/*/pearlplus-*.json": "Pearl+'s answer and its Lean and Python checks, in their public form",
    "OUT/pieces/*/verify.json": "a piece's verification of its answer",
    "OUT/pieces/*/session/*/out/key.json": "a session's public key",
    "OUT/pieces/certifier/chain/report/report.json": "the network attester's chain report",
    "OUT/record/*": "the final record, without its node",
    "OUT/composite/**": "the composite",
    "OUT/compose/**": "the compose gate's answer",
    "OUT/sampled-proofs/*.json": "the sampled proofs' records",
    "OUT/sampled-proofs/ran_with.txt": "what the sampled proofs ran with",
    "OUT/sampled-proofs/served-records/registration.json": "the registration's public form",
    "OUT/sampled-proofs/**/session/*/out/key.json": "a session's public key",
    "D/h1/log.jsonl": "H1's log of records",
    "D/h1/**/*.json": "an H1 record its log names",
    "D/deployment-registration.json": "the deployment's registration record",
    "D/served-compare/served_compare.json": "the served comparison's verdict",
    "D/private/front-window": "the attester front's window, its release order",
    "D/private/pouw/period/case.json": "PoUW's period case",
    "D/private/pouw/period/answer.json": "its answer",
    "D/pearlplus/*/registration.json": "Pearl+'s window registration",
    "D/pearlplus/*/tie/window/registration.json": "its tie's registration",
    "D/pearlplus/*/tie/summary.json": "its tie's summary",
    "D/pearlplus/*/tie/circuit-record.json": "its tie's circuit record",
    "D/*/summary.json": "an audit's summary",
    "D/*/runs.json": "an audit's run ids",
    "D/*/window/audit.json": "an audit's record",
    "D/*/window/registration.json": "an audit's registration",
    "D/*/window/case.json": "an audit's case",
    "D/*/window/case-v2.json": "its v2 case",
    "D/*/window/case-error.txt": "why its case wasn't written",
    "D/*/window/case-v2-error.txt": "why its v2 case wasn't written",
    "D/*/audit/summary.json": "V3c's audit summary",
    "D/*/audit/runs.json": "its run ids",
    "D/*/audit/window/*.json": "V3c's audit record, registration and cases",
    "D/*/audit/window/*-error.txt": "why a case wasn't written",
    "D/*/window/*/circuit*": "a case's blob: a member's circuit",
    "D/*/window/*/pub-*.bin": "a case's blob: a member's public file",
    "D/*/window/*/tables/*": "a case's blob: a lookup table",
    "D/*/window/*/session-*/sessions/*/session.json": "a case's blob: a session's record",
    "D/*/window/*/session-*/sessions/*/*.bin": "a case's blob: a session's proof",
    "D/private/compose/pieces/**": "the hand-off's copy of the pieces' public files",
    "D/private/compose/compose.env": "the hand-off's settings",
    "D/private/compose/inference-only.json": "the inference-only verdict",
    "D/private/compose/inputs.sha256": "the hand-off's sums",
}


def _match(pat: tuple[str, ...], parts: tuple[str, ...]) -> bool:
    if not pat:
        return not parts
    if pat[0] == "**":
        return any(_match(pat[1:], parts[i:]) for i in range(len(parts) + 1))
    return bool(parts) and fnmatch.fnmatchcase(parts[0], pat[0]) and _match(pat[1:], parts[1:])


def listed(path: str) -> str | None:
    """What of ``path`` (``OUT/…`` or ``D/…``) is public, by its `PUBLIC` entry; None where the list has none."""
    parts = Path(path).parts
    return next((w for p, w in PUBLIC.items() if _match(Path(p).parts, parts)), None)


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


def unlisted(paths: Iterable[str]) -> list[str]:
    """Each of ``paths`` (relative to the serve dir) that no ``D/`` entry lets a hand-off carry."""
    return [p for p in paths if p.strip() and listed("D/" + p.strip()) is None]


def main(argv: list[str]) -> int:
    if argv[:1] == ["filter"] and len(argv) == 1:
        data = sys.stdin.buffer.read().decode("utf-8", "surrogateescape").split("\0")
        keep = public(data)
        sys.stdout.buffer.write("".join(p + "\0" for p in keep).encode("utf-8", "surrogateescape"))
        return 0
    if argv[:1] == ["why"] and len(argv) == 2:
        print(private_why(argv[1]) or "public")
        return 0
    if argv[:1] == ["handoff"] and len(argv) == 1:
        found = unlisted(sys.stdin.read().splitlines())
        for p in found:
            print(f"{p}: not on the hand-off's list (public_files.PUBLIC)")
        return 1 if found else 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
