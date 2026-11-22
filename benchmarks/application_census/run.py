#!/usr/bin/env python3
"""Where each application stops: `verity run NAME` for every file in `applications/`, on the machines the file names
(no `--on`), each in its own state directory under a scratch directory outside the tree, with a time limit of MINUTES
minutes (default 12; the first run builds the Lean verifier, about 5 minutes). Prints one table: application, the
furthest step the runner's own lines reached, the file:line its error output names, the first refusal's or error's text, the
kind (accept, refused NAME, error, time limit) and the seconds. The exit code is 0 only if every application ended in
the verifier's accept (`verity run`'s exit code 0). With `--out DIR`, each application's stdout and stderr and the
table are kept there. NAME ... names the applications to run (default every file); what follows a `--` is handed unchanged
to each `verity run NAME` (`--bindings FILE`, `--on MACHINE`). Each child inherits this process's environment, from which
the runner's preflight reads D (the serve directory), PRED, WINDOW_CLASS and VENV. A run on a card needs a larger MINUTES
(the record's step alone took 31 minutes on an H100).

    research run --on auto --cwd clone -- bash -lc 'uv run verity bench application_census [--minutes MINUTES] [--out "$OUT"] [NAME ...] [-- VERITY_RUN_ARGUMENTS]'
"""
from __future__ import annotations

import argparse
import json
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
        if what in ("STARTED", "PARTY", "VERDICT"):
            got = json.loads(body)
            note = [f"{k} {got[k]}" for k in ("exit", "refused", "verdict") if got.get(k) is not None]
            last = " ".join([what, got.get("party", ""), *note]).replace("  ", " ")
        elif what == "RUN" and not last:
            last = "RUN"
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
    return (said or [x for x in out.splitlines() if x.strip()][-1:] or [""])[0].strip()


def one(name: str, scratch: Path, seconds: float, keep: Path | None, extra: list[str]) -> list[str]:
    start = time.monotonic()
    p = subprocess.Popen([sys.executable, "-c", LAUNCH, "run", name, "--state", str(scratch / name / "state"), *extra],
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
        said = {json.loads(x[8:])["verdict"] for x in out.splitlines() if x.startswith("VERDICT ")}
        kind = "accept" if said == {"accept"} else "dry run" if any(x.startswith("DRY ") for x in out.splitlines()) \
            else "exit 0, no verdict line"
    elif m := REFUSED.search(text):
        kind = f"refused {m.group(1)}"
    else:
        kind = "error"
    return [name, step(out), place(err), text, kind, str(took)]


def road(name: str, state: Path, keep: Path | None) -> str:
    found = ""
    for log in sorted((state / "logs").glob("*.log")):
        lines = log.read_text(errors="replace").splitlines()
        at = [i for i, x in enumerate(lines) if x.startswith("ROAD END")]
        if at:
            block = [lines[at[-1]]]
            for x in lines[at[-1] + 1:]:
                if not x[:1].isspace():
                    break
                block.append(x)
            found = "\n".join([f"{name} ({log.name}):", *block])
    if found and keep:
        (keep / f"{name}.road").write_text(found + "\n")
    return found


def table(rows: list[list[str]]) -> str:
    return "\n".join(" | ".join(r) for r in [list(HEAD), *rows])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="verity bench application_census", description=__doc__.split("\n\n")[0])
    ap.add_argument("--minutes", type=float, default=12, help="each application's time limit (default 12)")
    ap.add_argument("names", nargs="*", metavar="NAME", help="applications to run (default every file)")
    ap.add_argument("--out", type=Path, help="keep each application's stdout and stderr and the table here")
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    a = ap.parse_args(argv[:argv.index("--")] if "--" in argv else argv)
    every = sorted(f.stem for f in (ROOT / "applications").glob("*.toml"))
    if unknown := [n for n in a.names if n not in every]:
        ap.error(f"{', '.join(unknown)}: NAME is one of {', '.join(every)}")
    keep = a.out
    scratch = Path(tempfile.mkdtemp(prefix="application-census-"))
    try:
        rows, roads = [], []
        for n in a.names or every:
            rows.append(one(n, scratch, a.minutes * 60, keep, extra))
            roads.append(road(n, scratch / n / "state", keep))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    print(table(rows))
    for r in filter(None, roads):
        print("\n" + r)
    if keep:
        (keep / "table.txt").write_text(table(rows) + "\n")
    return 0 if rows and all(r[4] == "accept" for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
