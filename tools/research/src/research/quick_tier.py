"""The quick tier at one checkout, as facts and a verdict: each Lake package `--lean` names built first (with the PATH's `lake`, else
elan's; with neither it refuses), then CMD (`suites.py --quick`) with `--changed BASE`.  `tier` gathers the facts; `judge` reads them
as the refusal `research queue ready` prints, or None for a pass, the same way wherever the tier ran.

    python3 quick_tier.py --head SHA --base SHA [--lean DIR ...] -- CMD...

runs it in the working directory inside a `research run`, in the environment check.py gives its steps there (`run_env`: the store
required, through the run's custody key), and writes the facts to `$RESEARCH_RUN_DIR/result.json` (kind `quick-tier/v1`, under
`tier`), with the suites' report and logs under `$RESEARCH_RUN_DIR/out/`.  `research queue ready --on MACHINE` sends this file with
the run, so the machine runs the caller's version whatever the PR's tree holds.  Stdlib only and no import from `research`: it runs
from the run's inputs under the machine's `python3`.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

KIND = "quick-tier/v1"
SUITES = "tools/check/suites.py"
#: variables that would point the tier's Python at the caller's tree or environment instead of the checkout's own
CALLER_ENV = ("PYTHONPATH", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT")
ROW_KEYS = ("name", "suite", "status", "failed_tests", "log")
#: what check.py sets for its steps inside a recorded run (its CUSTODY_ENV and STORE_REQUIRED_ENV), never inherited
CUSTODY_ENV, STORE_REQUIRED_ENV = "CHECK_STORE_CUSTODY", "VERITY_STORE_REQUIRED"


def _git(tree: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=tree, capture_output=True, text=True)


def run_env(environ, run_dir: str, run_id: str) -> dict:
    """What the tier's builds and suites see inside a recorded run, as check.py's `env()` gives its steps: nothing of the recording
    run's own (`RESEARCH_*`: the tests start runs of their own, which would inherit them); the evidence store required (a
    store-backed test that can't reach it fails instead of skipping) and reached through the run's custody dir, `<root>/requests/<run
    id>/custody/` beside `<root>/runs/<run id>/`, when it holds the key (on a machine the workload's environment holds no store
    credential); and a git identity for the tests that commit in throwaway repositories."""
    out = {k: v for k, v in environ.items() if not k.startswith("RESEARCH_") and k not in (CUSTODY_ENV, STORE_REQUIRED_ENV)}
    out[STORE_REQUIRED_ENV] = "1"
    custody = Path(run_dir).resolve().parent.parent / "requests" / run_id / "custody"
    if (custody / "store.toml").is_file() and (custody / "cred.json").is_file():
        out[CUSTODY_ENV] = str(custody)
    for k in ("GIT_AUTHOR_NAME", "GIT_COMMITTER_NAME"):
        out.setdefault(k, "check")
    for k in ("GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL"):
        out.setdefault(k, "check@localhost")
    return out


def tier(tree: Path, out: Path, head: str, base: str, command: list[str], lean=(), env: dict | None = None) -> dict:
    """The facts of one run of the tier in ``tree``: the commit checked out there, what differs from it before anything runs (each
    kept `.venv` and `.lake` aside), each Lean build's exit, and CMD's exit and report (`out/suites.json`, logs in `out/logs/`).  It
    stops at the first fact that refuses.  `lake` is the PATH's, else elan's in `~/.elan/bin` (put first on the PATH, as check.py
    does); a `--lean` package with neither is an error, since without the build its tests skip."""
    env = {k: v for k, v in (os.environ if env is None else env).items() if k not in CALLER_ENV}
    logs = out / "logs"
    doc = {"kind": KIND, "head": None, "base": base, "dirty": [], "lean": [], "changed": True, "rc": None, "rows": None,
           "logs": str(logs), "error": None}
    r = _git(tree, "rev-parse", "HEAD")
    if r.returncode:
        doc["error"] = f"{tree} isn't a git checkout ({(r.stderr or r.stdout).strip()[-200:]})"
        return doc
    doc["head"] = r.stdout.strip()
    if doc["head"] != head:
        return doc
    kept = [":(exclude).venv", *(f":(exclude){d}/.lake" for d in lean)]
    doc["dirty"] = _git(tree, "status", "--porcelain", "--", ".", *kept).stdout.splitlines()
    if doc["dirty"]:
        return doc
    lake = shutil.which("lake", path=env.get("PATH"))
    elan = Path(env.get("HOME") or Path.home()) / ".elan" / "bin"
    if lake is None and (lake := shutil.which("lake", path=str(elan))):
        env["PATH"] = os.pathsep.join(p for p in (str(elan), env.get("PATH")) if p)
    for d in lean:
        if (tree / d / "lakefile.toml").is_file():
            if lake is None:
                doc["error"] = f"no `lake` on the PATH or in {elan} to build {d} with, and without that build its tests skip"
                return doc
            print(f"quick tier: `lake build` in {d}", flush=True)
            doc["lean"].append({"dir": d, "rc": subprocess.run([lake, "build"], cwd=tree / d, env=env).returncode})
            if doc["lean"][-1]["rc"]:
                return doc
    runner = tree / SUITES
    if runner.is_file() and '"--changed"' not in runner.read_text():
        print(f"quick tier: {head[:12]}'s suites.py predates --changed, so every suite runs (merge main into the branch to run only "
              f"the ones it changes)", flush=True)
        doc["changed"] = False
    report = out / "suites.json"
    out.mkdir(parents=True, exist_ok=True)
    report.unlink(missing_ok=True)
    try:
        doc["rc"] = subprocess.run([*command, *(["--changed", base] if doc["changed"] else []), "--json", str(report), "--logs", str(logs)],
                                   cwd=tree, env=env).returncode
    except FileNotFoundError as e:
        doc["error"] = f"can't run it here ({e})"
        return doc
    try:
        doc["rows"] = [{k: r.get(k) for k in ROW_KEYS} for r in json.loads(report.read_text())]
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    return doc


def judge(doc, head: str, output: str = "its output is above") -> str | None:
    """None when ``doc``, `tier`'s facts, show the tier passed at ``head``; else what failed, for the refusal.  ``output``: where the
    builds' and the suites' output is."""
    if not isinstance(doc, dict) or doc.get("kind") != KIND:
        return f"it reported no {KIND} facts"
    if doc.get("error"):
        return str(doc["error"])
    if doc.get("head") != head:
        return f"it ran at {str(doc.get('head'))[:12]}, not at {head[:12]}"
    if dirty := doc.get("dirty"):
        return (f"the checkout differed from {head[:12]} before anything ran: {', '.join(map(str, dirty[:5]))}"
                + (f" and {len(dirty) - 5} more" if len(dirty) > 5 else ""))
    for b in doc.get("lean") or []:
        b = b if isinstance(b, dict) else {"dir": b}
        if b.get("rc") != 0:
            return f"`lake build` in {b.get('dir')} exited {b.get('rc')} ({output})"
    rc = doc.get("rc")
    if type(rc) is not int:
        return "it reported no exit for `suites.py`"
    if rc == 0:
        return None
    rows = doc.get("rows")
    if not isinstance(rows, list):
        return f"`suites.py` exited {rc} before it reported ({output})"
    failed = [r for r in rows if isinstance(r, dict) and r.get("status") == "failed"]
    if not failed:
        return f"`suites.py` exited {rc}: the tests changed files in the checkout ({output})"
    return "; ".join(f"{r.get('name')}: {', '.join(r.get('failed_tests') or []) or 'see ' + str(r.get('log') or doc.get('logs'))}"
                     for r in failed)


def main(argv: list[str]) -> int:
    usage = "usage: quick_tier.py --head SHA --base SHA [--lean DIR ...] -- CMD...  (inside a `research run`)"
    if "--" not in argv:
        print(usage, file=sys.stderr)
        return 2
    opts, command = argv[:argv.index("--")], argv[argv.index("--") + 1:]
    pairs = list(zip(opts[::2], opts[1::2]))
    run_dir, rid = os.environ.get("RESEARCH_RUN_DIR"), os.environ.get("RESEARCH_RUN_ID")
    if len(opts) % 2 or {k for k, _ in pairs} - {"--head", "--base", "--lean"} or not command or not run_dir or not rid:
        print(usage, file=sys.stderr)
        return 2
    head, base = dict(pairs).get("--head"), dict(pairs).get("--base")
    if not head or not base:
        print(usage, file=sys.stderr)
        return 2
    lean = [v for k, v in pairs if k == "--lean"]
    d = Path(run_dir)
    doc = tier(Path.cwd(), d / "out", head, base, command, lean, run_env(os.environ, run_dir, rid))
    why = judge(doc, head, "its output is in this run's log")
    result = {"schema": "research/result/v0.1", "kind": KIND, "run_id": rid,
              "workload_fingerprint": {"head": doc["head"], "base": base, "command": command, "lean": lean},
              "validation": {"status": "failed" if why else "passed", "checker": "quick_tier.py", "detail": why or "passed"},
              "tier": doc, "artifacts": [p for p in ("out/suites.json", "out/logs") if (d / p).exists()]}
    tmp = d / "result.json.tmp"
    tmp.write_text(json.dumps(result, indent=1) + "\n")
    os.replace(tmp, d / "result.json")
    print(f"quick tier: {'passed' if why is None else 'failed: ' + why}", flush=True)
    return 1 if why else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
