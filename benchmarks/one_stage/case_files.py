"""v2's executable on a case as a driver writes one (``{"blobs", "commitment", "evidence", "params"}``,
`a5.case_members`'s or `a5.case_v2`'s), its blobs passed under short names: each distinct file once, linked into a
directory of its own as ``0``, ``1``, ..., and the executable run from that directory with the parameters, record and
evidence there as ``p``, ``c`` and ``e``. A window's per-member case names a public file for each of its members, and
Linux takes at most 6 MiB of a command's arguments and environment, their pointers included, however high the stack's
limit (and a quarter of the limit below 24 MiB, `STACK`, to which it is raised for the executable): a served window's
blobs under their own paths pass that, and under short names they don't. v2 finds a blob by its digest
(`Core.Blobs.get`), so which of several names of one file it is given changes nothing it decides.

    case_files.py CASE --exe EXE [--command check] [--dir DIR]

prints one JSON line (`run`): the executable's exit, its last line, its seconds and peak RSS, the blobs the case lists
and the distinct files and bytes it was given.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

#: the stack limit the executable runs under, where its caller's is lower: Linux's cap on arguments is then its own
STACK = 24 << 20
#: Linux's cap on a command's arguments and environment, each string's bytes, its NUL and its pointer
ARGS_CAP = 6 << 20


def linked(blobs: Sequence[str | Path], d: Path) -> list[str]:
    """Each distinct file of `blobs` (one file under several paths or links is one) as a link in `d` named by its
    position among them, in the order first listed -> the names. Raises FileNotFoundError naming a blob that is not
    there: a case whose files are missing has no check."""
    seen: set[tuple[int, int]] = set()
    names: list[str] = []
    for b in map(Path, blobs):
        try:
            st = b.stat()
        except FileNotFoundError:
            raise FileNotFoundError(f"the case names {b}, which is not there") from None
        if (st.st_dev, st.st_ino) in seen:
            continue
        seen.add((st.st_dev, st.st_ino))
        names.append(str(len(names)))
        os.symlink(b.resolve(), d / names[-1])
    return names


def argv(case: Mapping[str, Any], exe: Path, command: str, d: Path) -> list[str]:
    """`command` of `exe` on `case`, run from `d` (module docstring): writes ``p``, ``c``, ``e`` and the blobs' links."""
    d.mkdir(parents=True, exist_ok=True)
    params = json.dumps(case["params"], sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    for name, b in (("p", params.encode()), ("c", bytes.fromhex(case["commitment"])),
                    ("e", bytes.fromhex(case["evidence"]))):
        (d / name).write_bytes(b)
    return [str(exe), command, "p", "c", "e", *linked(case["blobs"], d)]


def arg_bytes(args: Sequence[str], env: Mapping[str, str]) -> int:
    """What `args` and `env` take of `ARGS_CAP`."""
    return sum(len(a.encode()) + 1 + 8 for a in args) + sum(len(f"{k}={v}".encode()) + 1 + 8 for k, v in env.items())


def _stack() -> None:
    soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
    if soft != resource.RLIM_INFINITY and soft < STACK:
        resource.setrlimit(resource.RLIMIT_STACK, (STACK if hard == resource.RLIM_INFINITY else min(STACK, hard), hard))


def run(case: Mapping[str, Any], exe: Path, command: str, d: Path) -> dict[str, Any]:
    """`argv` run from `d`, its output in ``d/stdout`` and ``d/stderr`` -> its exit (``rc``), its last line, its
    ``seconds``, its peak RSS (``peak_rss_kb``, its own, from wait4), and the blobs listed, linked and their bytes.
    Raises ValueError where even the short names pass `ARGS_CAP`."""
    args = argv(case, exe, command, d)
    took = arg_bytes(args, os.environ)
    if took > ARGS_CAP:
        raise ValueError(f"{len(args)} arguments and the environment take {took} bytes, past Linux's {ARGS_CAP}")
    t = time.monotonic()
    with open(d / "stdout", "w") as out, open(d / "stderr", "w") as err:
        p = subprocess.Popen(args, cwd=d, stdout=out, stderr=err, preexec_fn=_stack)
        _, status, use = os.wait4(p.pid, 0)
        p.returncode = os.waitstatus_to_exitcode(status)
    seconds = round(time.monotonic() - t, 1)
    lines = (d / "stdout").read_text().strip().splitlines()
    names = args[5:]
    return {"rc": p.returncode, "line": lines[-1] if lines else None, "seconds": seconds, "peak_rss_kb": use.ru_maxrss,
            "listed": len(case["blobs"]), "linked": len(names), "bytes": sum((d / n).stat().st_size for n in names),
            "arg_bytes": took}


def main(argv_: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("case", type=Path)
    ap.add_argument("--exe", type=Path, required=True)
    ap.add_argument("--command", default="check")
    ap.add_argument("--dir", type=Path)
    a = ap.parse_args(argv_)
    d = a.dir or Path(tempfile.mkdtemp(prefix="case-files-"))
    print(json.dumps(run(json.loads(a.case.read_text()), a.exe, a.command, d), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
