#!/usr/bin/env python3
"""VLLM-289: the restack rehearsal's driver for many branches in one `research run --on auto --cwd source` job. Scratch: it
lands nowhere.

    drive.py --staged STAGED --move M [--move M ...] --main MAIN [--workers N] [--pins auto|on|off] [--real] [--smoke] PR=HEAD...

It makes the job's own clone of the source root with the repository's merge drivers registered (`research.clone.make`, what
`--cwd clone` calls: a source root has none, and a lock, a pyproject's test lists or a lint allowlist then merges as text),
and runs everything there, so the shared source tree gains no worktree and no object.

In the clone, in order:

1. the staged tree's own pins (`pins.py --stale` at STAGED's root). Where one is stale every branch's pins step would run
   its command and still end `pins-stale`, so with `--pins auto` the branches then run with `--no-pins`;
2. `restack.check_generated` for each move, once: its result stays in `restack._GENERATED`, which the forked workers inherit,
   so the job pays for the check once and not once a branch;
3. for each PR=HEAD, in N forked workers, `tools/move/restack.py HEAD --move-commit M... --onto STAGED --main MAIN --dry-run
   --json` (the checkout's copy, called as `restack.main`), its JSON written to $OUT/results/pr-PR.json as soon as it ends,
   with the minutes it took, the moves the command would cross with no `--move-commit` (`restack.moves_on`), each conflicted
   file's merge attribute, and a bundle of the commits it made ($OUT/bundles/pr-PR.bundle: the result, or the last step
   before a stop), so a stop can be read and mended on a VM without the script running again.

`--real` only changes the commits' identity: the caller's GIT_AUTHOR_*/GIT_COMMITTER_* instead of the rehearsal's fixed
one. Nothing here pushes.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

SOURCE = os.getcwd()
OUT = os.environ.get("OUT") or os.path.join(os.environ.get("RESEARCH_RUN_DIR", SOURCE), "out")
IDENT = {"GIT_AUTHOR_NAME": "vllm restack rehearsal", "GIT_AUTHOR_EMAIL": "vllm-restack@verity.invalid",
         "GIT_COMMITTER_NAME": "vllm restack rehearsal", "GIT_COMMITTER_EMAIL": "vllm-restack@verity.invalid",
         "GIT_AUTHOR_DATE": "2026-11-10T16:00:00+00:00", "GIT_COMMITTER_DATE": "2026-11-10T16:00:00+00:00"}
REF = "refs/restack289"
T0 = time.time()
#: set once in `main`, before the workers fork, and read by them: the checkout's `restack` module, the arguments, and whether
#: the branches run with `--no-pins`
RS = None
A: argparse.Namespace | None = None
NO_PINS = False


def minutes(t0: float) -> float:
    return round((time.time() - t0) / 60, 2)


def say(*a) -> None:
    print(time.strftime("%H:%M:%SZ", time.gmtime()), *a, flush=True)


def save(rel: str, obj) -> None:
    path = os.path.join(OUT, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")
    os.replace(path + ".tmp", path)


def tool(*argv: str) -> str:
    try:
        p = subprocess.run(list(argv), capture_output=True, text=True)
    except OSError as e:
        return f"missing: {e}"
    text = (p.stdout or p.stderr).strip()
    return text.splitlines()[0] if text else f"exit {p.returncode}"


def staged_pins(staged: str) -> dict:
    """Whether STAGED's own pins are fresh: its `pins.py --stale` at its root (no command runs, only hashes)."""
    if RS.git("cat-file", "-e", f"{staged}:{RS.PINS}", ok=(0, 128)).returncode:
        return {"status": "no-list", "fresh": True}
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="restack289-pins-", ignore_cleanup_errors=True) as tmp:
        wt = os.path.join(tmp, os.path.basename(tmp))
        RS.add_worktree(wt, staged, None)
        try:
            p = subprocess.run([sys.executable, RS.PINS, "--stale"], cwd=wt, capture_output=True, text=True)
        finally:
            RS.drop_worktree(wt, None)
    return {"status": "fresh" if p.returncode == 0 else "stale", "fresh": p.returncode == 0, "rc": p.returncode,
            "output": (p.stdout + p.stderr).strip().splitlines()[-40:], "minutes": minutes(t0)}


def bundle(pr: str, r: dict, staged: str) -> dict | None:
    """The commits the restack made for ``pr`` (its result, or its last step before a stop), as a bundle on top of STAGED and
    the branch's head."""
    tip = r.get("result") or next((s["commit"] for s in reversed(r.get("steps", [])) if s.get("commit")), None)
    if not tip or tip == r.get("head"):
        return None
    ref = f"{REF}/pr-{pr}"
    path = os.path.join(OUT, "bundles", f"pr-{pr}.bundle")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    RS.git("update-ref", ref, tip)
    p = RS.git("bundle", "create", "-q", path, ref, f"^{staged}", f"^{r['head']}", ok=tuple(range(256)))
    if p.returncode:
        return {"ref": ref, "commit": tip, "error": p.stderr.decode(errors="replace").strip()[-400:]}
    return {"ref": ref, "commit": tip, "file": f"bundles/pr-{pr}.bundle", "bytes": os.path.getsize(path)}


def one(job: tuple) -> dict:
    """One branch through `restack.main`, as the command line runs it; never raises."""
    pr, head = job
    a, no_pins = A, NO_PINS
    t0 = time.time()
    argv = [head, *(x for m in a.move for x in ("--move-commit", m)), "--onto", a.staged, "--main", a.main, "--dry-run",
            "--json", *(["--no-pins"] if no_pins else [])]
    rec: dict = {"pr": pr, "head": head, "argv": argv}
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            rc = RS.main(argv)
        r = json.loads(buf.getvalue())[0]
        rec.update(r, rc=rc)
    except SystemExit as e:
        rec.update(status="refused", error=str(e))
    except Exception as e:  # noqa: BLE001
        rec.update(status="error", error=repr(e), traceback=traceback.format_exc()[-3000:], stdout=buf.getvalue()[-2000:])
    try:
        rec["auto_moves"] = [{"commit": c, "script": s, "args": list(args)} for c, s, args in RS.moves_on(a.staged, head, None)]
        if rec.get("files"):
            attrs = RS.out("check-attr", "merge", "--", *rec["files"]).splitlines()
            rec["merge_attr"] = {line.split(": merge: ")[0]: line.split(": merge: ")[1] for line in attrs if ": merge: " in line}
        rec["bundle"] = bundle(pr, rec, a.staged)
    except Exception as e:  # noqa: BLE001
        rec["extras_error"] = repr(e)
    rec["minutes"] = minutes(t0)
    save(f"results/pr-{pr}.json", rec)
    at = f" at {rec['step']} ({len(rec.get('files', []))})" if rec.get("step") else ""
    say(f"#{pr} {head[:12]}: {rec.get('status')}{at} {(rec.get('result') or '')[:12]} [{rec['minutes']} min; plain "
        f"{(rec.get('plain') or {}).get('status')}]")
    return rec


def main() -> int:
    global RS, A, NO_PINS
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("branches", nargs="+", metavar="PR=HEAD")
    ap.add_argument("--staged", required=True)
    ap.add_argument("--move", action="append", required=True)
    ap.add_argument("--main", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--pins", choices=("auto", "on", "off"), default="auto")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="stop before the first script runs: the clone, the drivers, the pins")
    a = ap.parse_args()
    if not a.real:
        os.environ.update(IDENT)
    os.makedirs(OUT, exist_ok=True)

    sys.path.insert(0, os.path.join(SOURCE, "tools", "research", "src"))
    from research import clone as research_clone

    scratch = tempfile.mkdtemp(prefix="restack289-")
    tree = os.path.join(scratch, "tree")
    try:
        from pathlib import Path
        shipped = research_clone.make(Path(SOURCE), Path(tree))
        os.chdir(tree)
        sys.path.insert(0, os.path.join(tree, "tools", "move"))
        import restack
        RS, A = restack, a

        a.staged, a.main = RS.rev(a.staged, None), RS.rev(a.main, None)
        a.move = [RS.rev(m, None) for m in a.move]
        env = {"python": sys.version.split()[0], "executable": sys.executable, "git": tool("git", "--version"),
               "uv": tool("uv", "--version"), "cpus": os.cpu_count(), "shipped": shipped, "source": SOURCE,
               "host": os.uname().nodename, "restack_blob": RS.out("rev-parse", "HEAD:tools/move/restack.py"),
               "drivers": RS.out("config", "--local", "--get-regexp", r"^merge\..*\.name$").splitlines(),
               "staged": a.staged, "moves": a.move, "main": a.main, "workers": a.workers}
        say("environment", json.dumps(env))
        summary: dict = {"environment": env, "results": []}
        save("summary.json", summary)

        summary["staged_pins"] = sp = staged_pins(a.staged)
        NO_PINS = no_pins = a.pins == "off" or (a.pins == "auto" and not sp["fresh"])
        summary["no_pins"] = no_pins
        say(f"the staged tree's pins: {sp['status']}; branches run {'with --no-pins' if no_pins else 'with the pins step'}")
        for line in sp.get("output", [])[-12:] if not sp["fresh"] else []:
            say("   ", line)
        save("summary.json", summary)

        if a.smoke:
            say("smoke: stopping before the checks;", len(a.branches), "branches named")
            return 0
        summary["checks"] = []
        for m in a.move:
            t0 = time.time()
            RS.SCRIPT, RS.SCRIPT_ARGS = RS.move_line(m, None) or (RS.SCRIPT, ())
            RS.SCRIPT_ARGS = tuple(RS.SCRIPT_ARGS)
            base = RS.move_base(m, None)
            try:
                RS.check_generated(m, base, sys.executable, None)
                rec = {"move": m, "base": base, "script": RS.SCRIPT, "args": list(RS.SCRIPT_ARGS), "status": "generated"}
            except SystemExit as e:
                rec = {"move": m, "base": base, "script": RS.SCRIPT, "args": list(RS.SCRIPT_ARGS), "status": "refused",
                       "error": str(e)}
            rec["minutes"] = minutes(t0)
            summary["checks"].append(rec)
            say(f"check_generated {m[:12]} on {base[:12]} ({' '.join([RS.SCRIPT, *RS.SCRIPT_ARGS])}): {rec['status']} "
                f"[{rec['minutes']} min]" + (f"\n    {rec['error']}" if rec.get("error") else ""))
            save("summary.json", summary)
            if rec["status"] != "generated":
                return 2

        jobs = [tuple(b.split("=", 1)) for b in a.branches]
        with multiprocessing.get_context("fork").Pool(a.workers) as pool:
            for rec in pool.imap_unordered(one, jobs, chunksize=1):
                summary["results"].append({k: rec.get(k) for k in ("pr", "head", "status", "step", "files", "result", "minutes")})
                save("summary.json", summary)
        summary["minutes"] = minutes(T0)
        save("summary.json", summary)
        say(f"done: {len(summary['results'])} branches in {summary['minutes']} min")
        return 0
    finally:
        os.chdir(SOURCE)
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
