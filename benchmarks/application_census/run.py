#!/usr/bin/env python3
"""Where each application stops: `verity run NAME` for every file in `applications/`, on the machines the file names
(no `--on`), each in its own state directory under a scratch directory outside the tree, with a time limit of MINUTES
minutes (default 12; the first run builds the Lean verifier, about 5 minutes). Prints one table: application, the
furthest step the runner's own lines reached, the file:line its output names, the last refusal's or error's text, the
kind (accept, refused NAME, error, time limit) and the seconds. The exit code is 0 only if every application ended in
the verifier's accept (`verity run`'s exit code 0). With `$OUT` set, each application's stdout and stderr and the
table are kept there.

    research run --on auto --cwd clone -- uv run verity bench application_census [--minutes MINUTES]
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HEAD = ("application", "furthest step", "stops at", "text", "kind", "seconds")
LAUNCH = "import sys; from verity_cli import main; sys.exit(main())"
REFUSED = re.compile(r"\brefused ([\w-]+)")
PLACE = re.compile(r'File "([^"]+)", line (\d+)|([\w./-]+\.(?:py|sh|toml|lean)):(\d+)')


def step(out: str) -> str:
    last = ""
    for line in out.splitlines():
        what, _, body = line.partition(" ")
        if what.isupper() and what.isalpha():
            party = re.search(r'"party": "([^"]+)"', body)
            last = f"{what} {party.group(1)}" if party else what
    return last or "before RUN"


def place(text: str) -> str:
    found = PLACE.findall(text)
    if not found:
        return ""
    m = found[-1]
    return f"{m[0] or m[2]}:{m[1] or m[3]}"


def say(out: str, err: str) -> str:
    lines = [x for x in err.splitlines() if x.strip()]
    said = [x for x in lines if REFUSED.search(x)] or [x for x in lines if x.startswith("verity run:")] or lines[-1:]
    return (said or [x for x in out.splitlines() if x.strip()][-1:] or [""])[-1].strip()


def one(name: str, scratch: Path, seconds: float, keep: Path | None) -> list[str]:
    start = time.monotonic()
    p = subprocess.Popen([sys.executable, "-c", LAUNCH, "run", name, "--state", str(scratch / name / "state")],
                         cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    limit = False
    try:
        out, err = p.communicate(timeout=seconds)
    except subprocess.TimeoutExpired:
        limit = True
        os.killpg(p.pid, signal.SIGKILL)
        out, err = p.communicate()
    took = round(time.monotonic() - start)
    if keep:
        (keep / f"{name}.out").write_text(out)
        (keep / f"{name}.err").write_text(err)
    text = say(out, err)
    if limit:
        kind = "time limit"
    elif p.returncode == 0:
        kind = "accept"
    elif m := REFUSED.search(text):
        kind = f"refused {m.group(1)}"
    else:
        kind = "error"
    return [name, step(out), place(err + out), text, kind, str(took)]


def table(rows: list[list[str]]) -> str:
    return "\n".join(" | ".join(r) for r in [list(HEAD), *rows])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="verity bench application_census", description=__doc__.split("\n\n")[0])
    ap.add_argument("--minutes", type=float, default=12, help="each application's time limit (default 12)")
    a = ap.parse_args(argv)
    keep = Path(os.environ["OUT"]) if os.environ.get("OUT") else None
    scratch = Path(tempfile.mkdtemp(prefix="application-census-"))
    try:
        rows = [one(f.stem, scratch, a.minutes * 60, keep) for f in sorted((ROOT / "applications").glob("*.toml"))]
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    print(table(rows))
    if keep:
        (keep / "table.txt").write_text(table(rows) + "\n")
    return 0 if rows and all(r[4] == "accept" for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
