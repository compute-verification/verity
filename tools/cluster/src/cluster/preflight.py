"""`cluster preflight`: whether a job could start on a node now, checked from the submitter's machine in one SSH round trip
before it queues. One line per check; all must pass. Read-only: it reads files, `df`, the agent's state and Kueue, and asks
whether a Secret exists, never for its data.

- **ssh**: the node answers the caller's key (BatchMode, `CONNECT_S` to connect, `TIMEOUT_S` in all).
- **state**: node 2's agent state is fresh (`status.node2`'s rule: a timed window holds it back by design); node 1's Kueue
  answers, naming any stopped ClusterQueue; with the disk's use.
- **input**, one per ref: `art:<sha256>`, its manifest `/workspace/jobs/store/manifests/<sha256>.json`, which hashes to its
  id; `model:ORG/NAME@REV`, `/workspace/models/ORG--NAME@REV/.entry.json`; `hf:ORG/NAME@REV`, its snapshot
  (`<hub>/models--ORG--NAME/snapshots/<REV>`, or the commit `refs/<REV>` names) with a `config.json`. Node 2's jobs read
  snapshots materialized (`/workspace/jobs/hf/hub`), so it holds no symlink; node 1's pods read the ordinary cache
  (`/workspace/hf/hub`), whose snapshot files are symlinks into its blobs, so none may dangle.
- **custody** (`--custody`): the run can be published: R2 credentials in the caller's environment (`R2_ENV`, by name only),
  else node 1's Secret `research-r2`, which node 2 reaches as `vy-n1`.

`checks` is the pure half: the script's sectioned output in, checks out.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from . import nebius1, nebius2, status

CONNECT_S, TIMEOUT_S = 5, 30
MANIFESTS = "/workspace/jobs/store/manifests"
MODELS = "/workspace/models"
HUB = {nebius1.NODE: "/workspace/hf/hub", nebius2.NODE: "/workspace/jobs/hf/hub"}
SECRET = ("sudo", "-n", "k3s", "kubectl", "get", "secret", "research-r2", "-o", "name")
R2_ENV = ("R2_ENDPOINT", "R2_BUCKET", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
_PART = r"[A-Za-z0-9][A-Za-z0-9._-]*"
REF = re.compile(rf"art:[0-9a-f]{{64}}|(model|hf):{_PART}/{_PART}@{_PART}")
MARK = "@@preflight"

# Runs on the node, read-only: each ref's check, as [ref, ok, detail].
INPUTS = r'''
import hashlib, json, os, sys
a = json.loads(sys.argv[1])
out = []
for ref in a["refs"]:
    kind, _, rest = ref.partition(":")
    try:
        if kind == "art":
            p = os.path.join(a["manifests"], rest + ".json")
            if not os.path.isfile(p):
                out.append([ref, False, "no manifest " + p])
                continue
            with open(p, "rb") as f:
                h = hashlib.sha256(f.read()).hexdigest()
            out.append([ref, h == rest, p + (" hashes to its id" if h == rest else " hashes to " + h + ", not its id")])
        elif kind == "model":
            repo, _, rev = rest.partition("@")
            p = os.path.join(a["models"], repo.replace("/", "--") + "@" + rev, ".entry.json")
            out.append([ref, os.path.isfile(p), p + (" is there" if os.path.isfile(p) else " is missing")])
        else:
            repo, _, rev = rest.partition("@")
            base = os.path.join(a["hub"], "models--" + repo.replace("/", "--"))
            if not os.path.isdir(os.path.join(base, "snapshots", rev)) and os.path.isfile(os.path.join(base, "refs", rev)):
                with open(os.path.join(base, "refs", rev)) as f:
                    rev = f.read().strip()
            snap = os.path.join(base, "snapshots", rev)
            if not os.path.isfile(os.path.join(snap, "config.json")):
                out.append([ref, False, snap + (" has no config.json" if os.path.isdir(snap) else " is missing")])
                continue
            links = dangling = 0
            for root, dirs, files in os.walk(snap):
                for n in dirs + files:
                    q = os.path.join(root, n)
                    if os.path.islink(q):
                        links += 1
                        dangling += not os.path.exists(q)
            if a["materialized"] and links:
                out.append([ref, False, "%s holds %d symlinks; this node's jobs read materialized snapshots" % (snap, links)])
            elif dangling:
                out.append([ref, False, "%s has %d dangling symlinks" % (snap, dangling)])
            else:
                out.append([ref, True, snap + " has config.json, " + ("no symlinks" if not links else "%d symlinks into its blobs" % links)])
    except OSError as e:
        out.append([ref, False, "%s: %s" % (type(e).__name__, e)])
print(json.dumps(out))
'''


class Refused(ValueError):
    pass


def refs(values: list[str]) -> list[str]:
    """The refs, checked for form before any reaches the node (no path in them but the ones the rules build)."""
    bad = [v for v in values if not REF.fullmatch(v)]
    if bad:
        raise Refused(f"not an art:<sha256>, model:ORG/NAME@REV or hf:ORG/NAME@REV ref: {', '.join(bad)}")
    return list(dict.fromkeys(values))


def script(node: str, inputs: list[str], custody: bool) -> str:
    """The one command the node runs: each section's output after its marker, and its exit status."""
    sections = [("state", status.COMMANDS["state"]) if node == nebius2.NODE
                else ("kueue", shlex.join([*nebius1.KUBECTL, "get", "clusterqueues", "-o", "json"])), ("df", status.COMMANDS["df"])]
    if inputs:
        args = {"refs": inputs, "manifests": MANIFESTS, "models": MODELS, "hub": HUB[node], "materialized": node == nebius2.NODE}
        sections.append(("inputs", f"python3 -c {shlex.quote(INPUTS)} {shlex.quote(json.dumps(args))}"))
    if custody:
        secret = shlex.join(SECRET)
        sections.append(("custody", secret if node == nebius1.NODE else
                         f"ssh -o BatchMode=yes -o ConnectTimeout={CONNECT_S} vy-n1 {shlex.quote(secret)}"))
    return "; ".join(f"echo {MARK}:{k}; {{ {cmd}; }} 2>&1; echo {MARK}-rc:$?" for k, cmd in sections)


def sections(out: str) -> dict[str, tuple[int, str]]:
    found: dict[str, tuple[int, str]] = {}
    for part in out.split(f"{MARK}:")[1:]:
        name, _, rest = part.partition("\n")
        body, _, rc = rest.rpartition(f"{MARK}-rc:")
        found[name.strip()] = (int(rc.strip() or 1), body)
    return found


def _tail(s: str) -> str:
    return " ".join(s.split())[-200:] or "no output"


def checks(node: str, dest: str, out: str | None, error: str | None, now: int, inputs: list[str], custody: bool,
           env: Mapping[str, str]) -> list[dict[str, Any]]:
    """The checks from the node's sectioned output (`script`), or every check failed by why SSH did; `env` is the caller's."""
    rows: list[dict[str, Any]] = []

    def add(check: str, ok: bool, detail: str) -> None:
        rows.append({"check": check, "ok": ok, "detail": detail})

    caller_r2 = all(env.get(k) for k in R2_ENV)
    if out is None:
        add("ssh", False, f"{dest}: {error}")
        add("state", False, "not read: no SSH")
        for ref in inputs:
            add("input", False, f"{ref}: not read: no SSH")
        if custody:
            add("custody", caller_r2, f"R2 credentials in the caller's environment ({', '.join(R2_ENV)})" if caller_r2
                else "not read: no SSH, and no R2 credentials in the caller's environment")
        return rows
    sec = sections(out)
    add("ssh", True, f"{dest} answered the caller's key")
    rc_df, df = sec.get("df", (1, "no output"))
    if rc_df:
        add("state", False, f"df /workspace failed: {_tail(df)}")
    elif node == nebius2.NODE:
        rc, body = sec.get("state", (1, "no output"))
        if rc:
            add("state", False, f"no agent state: {_tail(body)}")
        else:
            rep = status.node2({"node": node, "t": now, "df": df, **status.state_snap(body)}, now)
            lg = rep["ledger"]
            note = " (a timed window holds the node: the agent writes no state until it ends)" if lg["window"] else ""
            add("state", not lg["stale"], f"the agent's ledger state is seq {lg['seq']}, {status.span(lg['age_s'])} old{note}"
                + (": is the agent running?" if lg["stale"] else "") + f"; /workspace {rep['disk']['use_pct']}% used")
    else:
        rc, body = sec.get("kueue", (1, "no output"))
        if rc:
            add("state", False, f"Kueue doesn't answer: {_tail(body)}")
        else:
            stopped = [q["metadata"]["name"] for q in json.loads(body)["items"]
                       if (q.get("spec") or {}).get("stopPolicy") not in (None, "None")]
            add("state", True, "Kueue answers" + (f"; stopped ClusterQueues: {', '.join(stopped)}" if stopped else "")
                + f"; /workspace {status.disk_pct(df)}% used")
    if inputs:
        rc, body = sec.get("inputs", (1, "no output"))
        try:
            got = {ref: (ok, detail) for ref, ok, detail in json.loads(body)} if not rc else {}
        except ValueError:
            got = {}
        for ref in inputs:
            ok, detail = got.get(ref, (False, f"not checked: {_tail(body)}"))
            add("input", ok, f"{ref}: {detail}")
    if custody:
        rc, body = sec.get("custody", (1, "no output"))
        has = not rc and body.strip() == "secret/research-r2"
        where = "node 1" if node == nebius1.NODE else "node 1, which node 2 reaches as vy-n1,"
        if caller_r2:
            add("custody", True, f"R2 credentials in the caller's environment ({', '.join(R2_ENV)})")
        else:
            add("custody", has, f"{where} holds Secret research-r2" if has else
                f"no R2 credentials in the caller's environment, and {where} doesn't show Secret research-r2: {_tail(body)}")
    return rows


def ssh(dest: str, key: Path | None = status.SSH_KEY) -> Callable[[str], str]:
    """`status.ssh` with a preflight's short timeouts."""
    def run(cmd: str) -> str:
        argv = ["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={CONNECT_S}",
                *(["-i", str(key)] if key and key.exists() else []), dest, cmd]
        r = subprocess.run(argv, capture_output=True, text=True, timeout=TIMEOUT_S)
        if r.returncode == 255:
            raise status.ReadError(r.stderr.strip()[-300:] or "ssh exited 255")
        return r.stdout
    return run


def render(node: str, rows: list[dict[str, Any]]) -> str:
    failed = sum(1 for r in rows if not r["ok"])
    lines = [f"{'ok  ' if r['ok'] else 'FAIL'}  {r['check']:<8} {r['detail']}" for r in rows]
    lines.append(f"preflight {node}: " + (f"{failed} of {len(rows)} checks failed" if failed else f"all {len(rows)} checks passed"))
    return "\n".join(lines) + "\n"


clock = time.time


def main(node: str, inputs: list[str], custody: bool, as_json: bool, *, dest: str | None = None,
         runner: Callable[[str], Callable[[str], str]] | None = None, env: Mapping[str, str] | None = None) -> int:
    """Check `node` for a job with `inputs`; print one line per check (or JSON); 0 only when every check passes."""
    now = int(clock())
    try:
        inputs = refs(inputs)
    except Refused as e:
        print(f"preflight {node}: {e}")
        return 2
    dest = dest or status.HOSTS[node]
    run = (runner or ssh)(dest)
    try:
        out, error = run(script(node, inputs, custody)), None
    except (status.ReadError, subprocess.TimeoutExpired, OSError) as e:
        out, error = None, f"{type(e).__name__}: {e}"
    rows = checks(node, dest, out, error, now, inputs, custody, os.environ if env is None else env)
    ok = all(r["ok"] for r in rows)
    if as_json:
        print(json.dumps({"schema": "cluster/preflight/v1", "node": node, "t": now, "ok": ok, "checks": rows}, indent=1, sort_keys=True))
    else:
        print(render(node, rows), end="")
    return 0 if ok else 1


__all__ = ["CONNECT_S", "TIMEOUT_S", "MANIFESTS", "MODELS", "HUB", "SECRET", "R2_ENV", "REF", "INPUTS", "Refused", "refs", "script",
           "sections", "checks", "ssh", "render", "main"]
