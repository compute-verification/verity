"""The verify step (panel rule (b), server.md 08:35Z): after the lease, in the same research run, run each arm's reference
verifier on the transcripts its timed code emitted, and write beside each panel row of the run's result JSON the verifier's
commit, its accept line verbatim, and the transcript as ``<run id>:<path> sha256:<hex>``; then print each accepted row's
`panel.py append` arguments.

The arm names its verifier in its build record (arm interface v0.4, `arm.check_verifier`): the commands, the transcript file
per shape and, if it isn't the run's source commit, the verifier's commit. A row's transcript is accepted only by a line of
its commands' output that starts with ``ACCEPT`` and names ``sha256:<hex>`` of the transcript file as this step hashes it
after the commands ran, with no ``REJECT`` line naming it. A per-shape command that exits nonzero rejects its shape; a
per-run command's exit code is recorded, and a nonzero one fails this step without voiding a transcript the verifier's own
line accepted. Every command's output is kept under ``verify/`` in the run's directory.

  python verify.py RESULT_JSON [--run-id ID] [--commit SHA] [--out FILE]

The run's directory is RESULT_JSON's (bench.py gives it to arms as ``ctx["out_dir"]``); --run-id defaults to
RESEARCH_RUN_ID and --commit to RESEARCH_SOURCE_SHA. RESULT_JSON is rewritten in place unless --out names another file.
Exit 0 only if every panel row has an accepted transcript and every command exited 0; 1 otherwise (each row says why);
2 on a usage error.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

import arm as A

VERSION = "verify/v0"
EXIT_OK, EXIT_REJECTED, EXIT_USAGE = 0, 1, 2
ACCEPT = re.compile(r"^\s*ACCEPT\b")
REJECT = re.compile(r"^\s*REJECT\b")
SHA256 = re.compile(r"\bsha256:([0-9a-f]{64})\b")
COMMIT = re.compile(r"[0-9a-f]{7,40}")


class UsageError(ValueError):
    pass


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def panel_rows(result: dict) -> list[tuple[str, str, dict]]:
    """(arm, shape label, panel block) of every arm row the run reported a panel block for."""
    return [(name, s["label"], r["panel"]) for s in result.get("shapes", [])
            for name, r in (s.get("arms") or {}).items() if "panel" in r]


class Runner:
    """Runs verifier commands from the current directory, each once (the same argv twice is run once), keeping their
    output under ``run_dir/verify/``."""

    def __init__(self, run_dir: Path, log):
        self.dir, self.log, self.done = run_dir / "verify", log, {}

    def run(self, argv: list[str]) -> dict:
        key = tuple(argv)
        if key in self.done:
            return self.done[key]
        self.dir.mkdir(parents=True, exist_ok=True)
        n = len(self.done)
        self.log(f"verify: {shlex.join(argv)}")
        t0 = time.monotonic()
        try:
            p = subprocess.run(argv, capture_output=True, text=True)
            code, out, err = p.returncode, p.stdout, p.stderr
        except OSError as ex:
            code, out, err = None, "", f"{type(ex).__name__}: {ex}\n"
        rec = {"n": n, "argv": argv, "exit": code, "seconds": round(time.monotonic() - t0, 3),
               "stdout": f"verify/{n}.out", "stderr": f"verify/{n}.err", "lines": out.splitlines()}
        (self.dir / f"{n}.out").write_text(out)
        (self.dir / f"{n}.err").write_text(err)
        self.log(f"verify: exit {code}")
        self.done[key] = rec
        return rec


def judge(path: Path, recs: list[dict], per: str) -> dict:
    """One transcript's verdict from its commands' records."""
    if not path.is_file():
        return {"accepted": False, "why": f"no transcript at {path}"}
    sha = sha256_file(path)
    lines = [ln for r in recs for ln in r["lines"]]
    named = [ln for ln in lines if sha in SHA256.findall(ln)]
    accepts = [ln for ln in named if ACCEPT.match(ln)]
    rejects = [ln for ln in named if REJECT.match(ln)]
    out = {"sha256": sha, "exits": [r["exit"] for r in recs]}
    if rejects:
        return out | {"accepted": False, "why": "the verifier rejected it", "reject_line": rejects[0]}
    if per == "shape" and any(r["exit"] != 0 for r in recs):
        return out | {"accepted": False, "why": f"a verifier command exited {[r['exit'] for r in recs]}"}
    if not accepts:
        return out | {"accepted": False, "why": f"no line starting ACCEPT names sha256:{sha}"}
    return out | {"accepted": True, "accept_line": accepts[0]}


def fill(template: str, values: dict) -> str:
    return template.format(**values)


def panel_args(run_id: str, stream: str, p: dict) -> tuple[list[str] | None, str | None]:
    """`panel.py append`'s arguments for an accepted row (the row's owner adds --line, --version, --change and
    --description), or why the row can't be appended as measured."""
    if "sm_clock_arm" not in p or "sm_clock_base" not in p:
        return None, p.get("sm_clock_missing", "no per-rep SM clocks for both sides")
    args = ["--kind", "measured", "--run", run_id, "--source", run_id, "--precision", stream, "--phase", p["phase"],
            "--shape", p["shape"], "--slowdown", str(p["slowdown"]), "--hash-free", str(p["hash_free"])]
    if p.get("decode_method"):
        args += ["--decode-method", p["decode_method"]]
    args += ["--verifier-commit", p["verifier_commit"], "--verifier-accept", p["verifier_accept"],
             "--transcript", p["transcript"], "--sm-clock-arm", p["sm_clock_arm"], "--sm-clock-base", p["sm_clock_base"]]
    return args, None


def verify(result: dict, run_dir: Path, run_id: str, commit: str | None, log) -> dict:
    """Runs every arm's verifier and writes each panel row's verdict into it; returns the step's record."""
    runner = Runner(run_dir, log)
    rows = []
    for name, label, p in panel_rows(result):
        arm = (result.get("arms") or {}).get(name) or {}
        v = arm.get("verifier")
        for k in ("verifier_commit", "verifier_accept", "transcript", "panel_args", "verification"):
            p.pop(k, None)
        verdict: dict = {"arm": name, "shape": label}
        if v is None:
            verdict |= {"accepted": False, "why": "the arm names no verifier (arm interface v0.4)"}
        else:
            v = A.check_verifier(name, {"verifier": v})
            c = v["commit"] or commit
            if not c:
                raise UsageError("no verifier commit: pass --commit or set RESEARCH_SOURCE_SHA")
            rel = fill(v["transcript"], {"arm": name, "shape": label})
            path = run_dir / rel
            values = {"python": sys.executable, "run_dir": str(run_dir), "run_id": run_id, "commit": c, "arm": name}
            if v["per"] == "shape":
                values |= {"shape": label, "transcript": str(path), "dir": str(path.parent)}
            recs = []
            for cmd in v["commands"]:
                recs.append(runner.run([fill(x, values) for x in cmd]))
                if v["per"] == "shape" and recs[-1]["exit"] != 0:
                    break
            verdict |= judge(path, recs, v["per"]) | {"commit": c, "path": rel, "commands": [r["n"] for r in recs]}
            if verdict["accepted"]:
                p |= {"verifier_commit": c, "verifier_accept": verdict["accept_line"],
                      "transcript": f"{run_id}:{rel} sha256:{verdict['sha256']}"}
                args, why = panel_args(run_id, arm.get("stream", ""), p)
                if args:
                    p["panel_args"] = args
                else:
                    verdict["not_appendable"] = why
        p["verification"] = {k: verdict[k] for k in ("accepted", "why", "reject_line", "not_appendable") if k in verdict}
        rows.append(verdict)
    commands = [{k: v for k, v in r.items() if k != "lines"} for r in runner.done.values()]
    ok = bool(rows) and all(r["accepted"] for r in rows) and all(c["exit"] == 0 for c in commands)
    return {"interface": VERSION, "run_id": run_id, "commit": commit, "rows": rows, "commands": commands, "ok": ok}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("result", type=Path, help="the run's result JSON (bench.py --out)")
    p.add_argument("--run-id", default=os.environ.get("RESEARCH_RUN_ID"))
    p.add_argument("--commit", default=os.environ.get("RESEARCH_SOURCE_SHA"))
    p.add_argument("--out", type=Path, help="write the result here instead of rewriting RESULT_JSON")
    a = p.parse_args(argv)

    def log(s: str) -> None:
        print(s, file=sys.stderr, flush=True)

    try:
        if not a.run_id:
            raise UsageError("no run id: pass --run-id or set RESEARCH_RUN_ID (the transcript must name its run)")
        if a.commit is not None and not COMMIT.fullmatch(a.commit):
            raise UsageError(f"--commit must be a git commit (7-40 hex characters), not {a.commit!r}")
        result = json.loads(a.result.read_text())
        started = now()
        rec = verify(result, a.result.resolve().parent, a.run_id, a.commit, log)
    except (UsageError, A.ArmError, OSError, ValueError, KeyError, TypeError) as ex:
        log(f"verify: {type(ex).__name__}: {ex}")
        return EXIT_USAGE
    result["verification"] = rec | {"started": started, "finished": now()}
    out = a.out or a.result
    tmp = out.with_name(out.name + ".verify-tmp")
    tmp.write_text(json.dumps(result, indent=1, default=str))
    os.replace(tmp, out)
    for r in rec["rows"]:
        row = next(pb for n, lb, pb in panel_rows(result) if n == r["arm"] and lb == r["shape"])
        if r["accepted"] and "panel_args" in row:
            print(f"{r['arm']} {r['shape']}: python panel.py append --line LINE --version VERSION --change CHANGE "
                  f"--description DESCRIPTION {shlex.join(row['panel_args'])}")
        else:
            print(f"{r['arm']} {r['shape']}: not measured: {r.get('why') or r.get('not_appendable')}")
    bad = [c for c in rec["commands"] if c["exit"] != 0]
    for c in bad:
        print(f"verifier command exited {c['exit']}: {shlex.join(c['argv'])} (output: {c['stdout']}, {c['stderr']})")
    if not rec["rows"]:
        print("no panel rows in the result: nothing to verify")
    log(f"verify: {'ok' if rec['ok'] else 'NOT ok'} -> {out}")
    return EXIT_OK if rec["ok"] else EXIT_REJECTED


if __name__ == "__main__":
    sys.exit(main())
