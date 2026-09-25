"""A row's stages as research Attempts: `verity-vllm row stage|chain|shed`.

* ``stage <build|match|commit> ROW ROLE REPO REV [flags]``: the workload of ONE ``research run --tool vllm.<stage>``
  Attempt (``research_tools``).  A Match / Commit whose row dir holds a Build with another Program digest than the cited
  Build artifact (the run's ``job.json`` ``store.inputs``) stops by name (PRECHECK_FAIL_BUILD_INPUT_*, exit 3).  On exit,
  whatever the rc: the stage's typed outputs (``research-outputs write``) and the run id on the pod guard's durability
  list, so the pod is not terminated before the outputs are preserved.
* ``chain ROW ROLE REPO REV [--stages ...] [--campaign C] [--project P] [flags]``: one nested ``research run`` per stage
  in order, each citing the earlier stages' artifacts; stops at the first stage that exits non-zero or publishes no
  account artifact.  Prints ``<stage> <run-id> <artifact-id> rc=<rc>``; ``row.json`` beside the row keeps them.
* ``shed [--force]``: the teardown -- the pod terminates (the pod guard's gate) only when every registered Attempt is
  preserved.

The research CLI is ``--research-bin``, else ``research`` on PATH, else ``python3 -m research`` with the tree's copy of
``tools/research/src`` when the interpreter has none.  Relative ``RESEARCH_STORE_CONFIG`` / ``RESEARCH_STORE`` /
``RESEARCH_RUNS`` are made absolute before any nested run (they resolve against the reader's cwd).
"""
from __future__ import annotations

import json
import os
import secrets
import shutil
import subprocess
import sys
import time
from typing import Any, Callable

from verity_vllm.config import ROOT
from verity_vllm.pipeline.row_driver import CLI, utc

REPO_ROOT = os.path.realpath(os.path.join(ROOT, "../.."))
STORE_VARS = ("RESEARCH_STORE_CONFIG", "RESEARCH_STORE", "RESEARCH_RUNS")
ACCOUNT = {"build": "build", "match": "match", "commit": "verdict"}


def abs_store_env(environ: dict[str, str]) -> dict[str, str]:
    env = dict(environ)
    for v in STORE_VARS:
        val = env.get(v) or ""
        if val and not val.startswith(("/", "~")):
            env[v] = os.path.abspath(val)
    return env


def research_argv(research_bin: str | None, research_py: str, env: dict[str, str]) -> tuple[list[str], dict[str, str]]:
    """The research CLI's argv prefix and the environment it runs with."""
    if research_bin:
        return [research_bin], env
    exe = shutil.which("research", path=env.get("PATH"))
    if exe:
        return [exe], env
    src = os.path.join(REPO_ROOT, "tools/research/src")
    if subprocess.run([research_py, "-c", "import research"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env).returncode != 0 \
            and os.path.isdir(src):
        env = {**env, "PYTHONPATH": src + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")}
    return [research_py, "-m", "research"], env


def job_input(run_dir: str | None, name: str) -> str:
    """The artifact id ``research run --input NAME=art:...`` recorded in the run's job.json, "" when none."""
    if not run_dir:
        return ""
    try:
        d = json.load(open(os.path.join(run_dir, "job.json")))
        return ((d.get("store") or {}).get("inputs") or {}).get(name) or ""
    except (OSError, ValueError, AttributeError):
        return ""


def _show(research: tuple[list[str], dict[str, str]], ident: str) -> Any:
    argv, env = research
    r = subprocess.run([*argv, "data", "show", ident, "--json"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env)
    try:
        return json.loads(r.stdout)
    except ValueError:
        return None


def artifact_meta(research: tuple[list[str], dict[str, str]], art: str, key: str) -> str:
    """meta[key] of an artifact's manifest (the local store, or the remote), "" when unreadable."""
    d = _show(research, art)
    if not isinstance(d, dict):
        return ""
    v = ((d.get("manifest") or d).get("meta") or {}).get(key)
    return "" if v is None else str(v)


def attempt_output(research: tuple[list[str], dict[str, str]], run_id: str, name: str) -> str:
    """outputs[name] of a published attempt, "" when none."""
    d = _show(research, run_id)
    if isinstance(d, list):
        d = d[0] if d else {}
    return str((d.get("outputs") or {}).get(name) or "") if isinstance(d, dict) else ""


def _tee(path: str, line: str) -> None:
    print(line, flush=True)
    with open(path, "a") as f:
        f.write(line + "\n")


def stage(o: Any, run: Callable[[Any, str], int]) -> int:
    """One stage of the row as the workload of one research Attempt; ``run(o, stage)`` runs the row's stage."""
    env = o.env
    d = f"{env.sweep_dir}/{o.row}"
    os.makedirs(d, exist_ok=True)
    build_in, match_in = job_input(env.research_run_dir, "build"), job_input(env.research_run_dir, "match")
    child = abs_store_env(dict(env.environ))
    try:
        if o.stage != "build" and build_in:
            r = subprocess.run([env.py, *CLI, "research-outputs", "digest", d, o.row], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                               text=True, cwd=ROOT, env=child)
            here = r.stdout.strip() if r.returncode == 0 else ""
            if not here:
                _tee(f"{d}/stages.txt", f"precheck FAIL {utc()} PRECHECK_FAIL_BUILD_INPUT_ABSENT (cited {build_in} but {d} has no Build; run the build stage on this pod first)")
                return 3
            cited = artifact_meta(research_argv(o.research_bin, o.research_py, child), build_in, "program_digest")
            if not cited:
                _tee(f"{d}/stages.txt", f"precheck FAIL {utc()} PRECHECK_FAIL_BUILD_INPUT_UNREADABLE (cited {build_in}: manifest not readable here, or no program_digest in meta)")
                return 3
            if cited != here:
                _tee(f"{d}/stages.txt", f"precheck FAIL {utc()} PRECHECK_FAIL_BUILD_INPUT_MISMATCH (cited {build_in} program_digest={cited[:16]} != this dir's Build {here[:16]})")
                return 3
            _tee(f"{d}/row.log", f"{utc()} [row {o.row}] v2 {o.stage}: consumes Build {build_in} (program_digest {here[:16]} verified)"
                                 f"{', Match ' + match_in if match_in else ''}")
        return run(o, o.stage)
    finally:
        if env.research_run_dir:
            inputs = [*(["--input", f"build={build_in}"] if build_in else []), *(["--input", f"match={match_in}"] if match_in else [])]
            with open(f"{d}/row.log", "a") as f:
                subprocess.run([env.py, *CLI, "research-outputs", "write", d, o.row, o.stage, *inputs], stdout=f, stderr=subprocess.STDOUT,
                               cwd=ROOT, env=child)
            if env.research_run_id:
                preserve(env.guard_preserve_file, env.research_run_id)


def preserve(path: str, run_id: str) -> None:
    """Register ``run_id`` on the pod guard's durability list (once)."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        have = open(path).read().split("\n") if os.path.exists(path) else []
        if run_id not in have:
            with open(path, "a") as f:
                f.write(run_id + "\n")
    except OSError as e:
        print(f"guard preserve list {path} not updated: {e}", file=sys.stderr)


def chain(o: Any, sweep_dir: str) -> int:
    """One nested ``research run --tool vllm.<stage>`` per stage, each citing the earlier stages' artifacts (``sweep_dir``:
    the stages' own, from their flags)."""
    env = o.env
    d = f"{sweep_dir}/{o.row}"
    os.makedirs(d, exist_ok=True)
    child = abs_store_env(dict(env.environ))
    child["PYTHONPATH"] = str(ROOT) + (":" + child["PYTHONPATH"] if child.get("PYTHONPATH") else "")
    research = research_argv(o.research_bin, o.research_py, child)
    project = o.project or "verity"
    build_art = match_art = ""
    for st in o.stages.split(","):
        run_id = f"r{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}-{secrets.token_hex(2)}"
        args = ["run", "--id", run_id, "--tool", f"vllm.{st}", "--project", project, "--cwd", REPO_ROOT]
        if o.campaign:
            args += ["--campaign", o.campaign]
        if st != "build" and build_art:
            args += ["--input", f"build={build_art}"]
        if st == "commit" and match_art:
            args += ["--input", f"match={match_art}"]
        _tee(f"{d}/row.log", f"{utc()} [row {o.row}] v2 row: research {' '.join(args)} -- verity-vllm row stage {st} ...")
        workload = [sys.executable, *CLI, "row", "stage", st, o.row, o.role, o.repo, o.rev, *o.flags]
        rc = subprocess.run([*research[0], *args, "--", *workload], env=research[1]).returncode
        art = attempt_output(research, run_id, ACCOUNT[st])
        print(f"{st} {run_id} {art or '-'} rc={rc}", flush=True)
        p = f"{d}/row.json"
        try:
            doc = json.load(open(p)) if os.path.exists(p) else {"schema": "verity-sweep/row-v2/v1", "stages": {}}
            doc["stages"][st] = {"run_id": run_id, "artifact": art or None, "rc": rc}
            json.dump(doc, open(p, "w"), indent=1)
        except (OSError, ValueError, KeyError, TypeError) as e:
            print(f"row.json not updated: {e}", file=sys.stderr)
        if rc != 0 or not art:
            _tee(f"{d}/row.log", f"{utc()} [row {o.row}] v2 row: stopping at {st} (rc={rc}, artifact={art or 'none'})")
            return 20 if rc == 0 else rc
        if st == "build":
            build_art = art
        elif st == "match":
            match_art = art
    return 0


#: sourced into bash: the pod guard library (its research lookup replaced by this command's), then the durability gate
_SHED = r'''
guard=$1; force=$2; shift 2; research_argv=("$@")
source "$guard"
guard_research() { "${research_argv[@]}" "$@"; }
[ "$force" = 1 ] && export GUARD_FORCE_TERMINATE=1
if ! guard_preserved_ok; then
  echo "verity-vllm row shed: outputs NOT preserved (see the guard log, guard.preserved.log); attempting research data push" >&2
  guard_preserve_repair || { [ "$force" = 1 ] || { echo "verity-vllm row shed: REFUSING to terminate (--force overrides)" >&2; exit 1; }; }
fi
guard_terminate_self "verity-vllm row shed: stage outputs preserved"
'''


def shed(o: Any) -> int:
    """Terminate the pod only when every Attempt it registered is preserved (``--force`` overrides the gate)."""
    env = o.env
    guard = next((g for g in (env.pod_guard, os.path.join(REPO_ROOT, "tools/research/src/research/pods/sh/pod_guard.sh")) if os.path.isfile(g)), None)
    if guard is None:
        print("verity-vllm row shed: no pod_guard.sh found; not terminating", file=sys.stderr)
        return 2
    child = abs_store_env(dict(env.environ))
    argv, child = research_argv(o.research_bin, o.research_py, child)
    return subprocess.run(["bash", "-c", _SHED, "shed", guard, "1" if o.force else "0", *argv], env=child).returncode
