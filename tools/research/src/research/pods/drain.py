"""Pod teardown gated on durability (F-r19-int-23: the deadman's `pod-max-hours` killed `tp2` with unarchived evidence).

    research pods unpreserved <machine|pod-id> [--mode head|readback|recorded] [--json]
        every Attempt the store recorded on that machine that is NOT `preserved` (the same predicate as `research data preserved`:
        `store.preserved.check`), with the reasons.  Exit 0 = nothing unpreserved, 1 = something is, 2 = machine unknown / no store.
    research pods drain <machine|pod-id> [--mode ...] [--force --reason TEXT] [--json]
        the same report, then terminates the pod ONLY when nothing on it is unpreserved.  `--force` terminates anyway; the reason is
        required and is logged (`~/.research/pods/drain.log` and the report) -- an operator's decision, on the record.  Exit 0 =
        terminated, 1 = refused (unpreserved Attempts listed, pod left running), 2 = usage / machine unknown / API error.
    research pods terminate <pod-id> --require-preserved [--force --reason TEXT]
        `drain` by pod id (part.main routes it here).
    --custody-r2 (either verb): from any machine -- the catalog is refreshed from the remote first, and `pod_custody` must pass too:
        the pod guard's fresh state says every run dir on the pod has custody (.fetched or .custody; store.custody).

Machine -> Attempt link.  `execution.machine` / the indexed condition `exec.machine` = the `machines.toml` name a `research run --on`
ran on (recorded from job.json `run.remote.machine`, `env.KEYS`); older manifests carry it at `execution.remote.machine`.  Runs
started ON the pod (no `--on`) have neither, so the pod id (`cost.pod_id`) and, when the pod answers ssh, its container hostname
(`hw.hostname` / `execution.host`) are matched too; each Attempt's `matched_by` says which.  An Attempt that is in the store but
whose files are only on the pod is exactly what this gate protects; an Attempt that never reached the store is invisible to it
(the report says so: `note`), which is why `research data push` on the pod comes first.

The predicate is imported, never re-implemented: `preserved.check(st, run_ids, mode)`; mode `recorded` works offline from the
catalog, `head` (default) asks the remote."""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

from .. import remote
from ..store import preserved as P
from ..store.local import LocalStore



def drain_log_path() -> Path:
    return Path(os.environ.get("RESEARCH_DRAIN_LOG", Path.home() / ".research" / "pods" / "drain.log"))


POD_ID_RE = re.compile(r"[a-z0-9]{12,20}")      # runpod ids; a typo'd machine name must not become a terminate call


def resolve_target(name: str) -> dict[str, Any]:
    """`name` is a machines.toml entry, the pod id of one, or a raw pod id (`POD_ID_RE`).  {"machine": name|None, "pod_id", "entry"};
    `pod_id` None when it is none of these."""
    ms = remote.load_machines()
    if name in ms:
        return {"machine": name, "pod_id": ms[name].get("pod_id"), "entry": ms[name]}
    by_pod = [k for k, v in ms.items() if v.get("pod_id") == name]
    if by_pod:
        return {"machine": by_pod[0], "pod_id": name, "entry": ms[by_pod[0]]}
    return {"machine": None, "pod_id": name if POD_ID_RE.fullmatch(name) else None, "entry": None}


def probe_hostname(pod_id: str | None, timeout: int = 30) -> str | None:
    """The container hostname of a RUNNING pod (best effort, read-only; None when unreachable)."""
    if not pod_id:
        return None
    try:
        from . import runpod

        pod = runpod.get(pod_id)
        tgt = runpod.ssh_target(pod) if pod and (pod.get("desiredStatus") or "").upper() == "RUNNING" else None
        if tgt is None:
            return None
        r = remote.ssh_run(remote.ssh_argv(*tgt), ["hostname"], timeout=timeout)
        return r.stdout.decode(errors="replace").strip() or None if r.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def attempts_on(st: LocalStore, machine: str | None, pod_id: str | None = None, hostname: str | None = None) -> dict[str, list[str]]:
    """run id -> the facts that tie it to the machine (`exec.machine`, `execution.machine`, `execution.remote.machine`, `cost.pod_id`,
    `hw.hostname`), from the store's catalog."""
    out: dict[str, list[str]] = {}

    def hit(aid: str, why: str) -> None:
        out.setdefault(aid, [])
        if why not in out[aid]:
            out[aid].append(why)

    if machine:
        for (aid,) in st.index.sql("SELECT DISTINCT attempt FROM attempt_conditions WHERE key='exec.machine' AND value=?", (machine,)):
            hit(aid, "exec.machine")
    for aid, mj in st.index.sql("SELECT id, manifest_json FROM attempts"):
        try:
            doc = json.loads(mj)
        except ValueError:
            continue
        ex, conds, cost = doc.get("execution") or {}, doc.get("conditions") or {}, doc.get("cost") or {}
        if machine:
            if ex.get("machine") == machine:
                hit(aid, "execution.machine")
            if (ex.get("remote") or {}).get("machine") == machine:
                hit(aid, "execution.remote.machine")
        if pod_id and (cost.get("pod_id") == pod_id or (ex.get("remote") or {}).get("pod_id") == pod_id):
            hit(aid, "cost.pod_id")
        if hostname and (ex.get("host") == hostname or conds.get("hw.hostname") == hostname):
            hit(aid, "hw.hostname")
    return dict(sorted(out.items()))


def report(st: LocalStore, target: str, mode: str = "head", hostname: str | None = None, probe: bool = True) -> dict[str, Any]:
    """The `unpreserved` report for a machine (module docstring)."""
    t = resolve_target(target)
    if hostname is None and probe:
        hostname = probe_hostname(t["pod_id"])
    linked = attempts_on(st, t["machine"], t["pod_id"], hostname)
    rep: dict[str, Any] = {"schema": "research/pods-unpreserved/v0.1", "target": target, "machine": t["machine"], "pod_id": t["pod_id"], "hostname": hostname,
                           "mode": mode, "attempts": {k: {"matched_by": v} for k, v in linked.items()}, "n_attempts": len(linked),
                           "unpreserved": [], "preserved": [], "check": None,
                           "note": "Attempts never published to the store (no `research data push` on the pod) are not visible here"}
    if not linked:
        return rep
    chk = P.check(st, list(linked), mode=mode)
    rep["check"] = {"remote": chk.get("remote"), "reason": chk.get("reason"), "preserved": chk["preserved"]}
    for tr in chk["targets"]:
        aid = tr["target"]
        rep["attempts"][aid].update(preserved=tr["preserved"], missing_names=tr.get("missing_names") or [], error=tr.get("error"),
                                    n_artifacts=tr.get("n_artifacts"))
        (rep["preserved"] if tr["preserved"] else rep["unpreserved"]).append(aid)
    rep["unpreserved_detail"] = [x for x in chk["not_preserved"]]
    return rep


def format_report(rep: dict[str, Any]) -> list[str]:
    who = f"{rep['machine'] or rep['target']} (pod {rep['pod_id'] or '?'}" + (f", host {rep['hostname']}" if rep.get("hostname") else "") + ")"
    lines = [f"{who}: {rep['n_attempts']} attempt(s) recorded on it; {len(rep['unpreserved'])} NOT preserved, {len(rep['preserved'])} preserved  [mode {rep['mode']}]"]
    if rep.get("check") and rep["check"].get("reason"):
        lines.append(f"  remote: {rep['check']['reason']}")
    for aid in rep["unpreserved"]:
        if aid not in rep["attempts"]:                 # the --custody-r2 leg (pod_custody)
            lines.append(f"  NOT preserved  {aid}")
            continue
        a = rep["attempts"][aid]
        why = a.get("error") or (("missing: " + ", ".join(a["missing_names"])) if a.get("missing_names") else "attempt manifest not on the remote")
        lines.append(f"  NOT preserved  {aid}  ({', '.join(a['matched_by'])})  {why}")
    for aid in rep["preserved"]:
        lines.append(f"  preserved      {aid}  ({', '.join(rep['attempts'][aid]['matched_by'])})")
    if not rep["n_attempts"]:
        lines.append(f"  {rep['note']}")
    return lines


GUARD_STATE_MAX_AGE_S = 600       # a guard ticks every minute; older state is a dead guard's


def guard_state(pod_id: str | None, root: str, timeout: int = 30) -> tuple[dict[str, Any] | None, str]:
    """The pod guard's <root>/guard/state.json (read-only, over ssh), or (None, why not)."""
    if not pod_id:
        return None, "no pod id"
    try:
        from . import runpod

        pod = runpod.get(pod_id)
        tgt = runpod.ssh_target(pod) if pod and (pod.get("desiredStatus") or "").upper() == "RUNNING" else None
        if tgt is None:
            return None, f"pod {pod_id} is not RUNNING with ssh"
        r = remote.ssh_run(remote.ssh_argv(*tgt), ["cat", f"{root.rstrip('/')}/guard/state.json"], timeout=timeout)
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {str(e)[:120]}"
    if r.returncode != 0:
        return None, "no guard state on the pod (no guard = N in its machines entry?)"
    try:
        doc = json.loads(r.stdout)
    except ValueError:
        return None, "unreadable guard state"
    return (doc, "") if isinstance(doc, dict) else (None, "unreadable guard state")


def pod_custody(st: LocalStore, target: str, *, refresh: bool = True, probe: bool = True, mode: str = "recorded", now: float | None = None,
                state: Callable[[str | None, str], tuple[dict[str, Any] | None, str]] = guard_state) -> tuple[bool, str, dict[str, Any]]:
    """Custody of everything on a pod, from any machine (--custody-r2): the catalog refreshed from the store's remote (so attempts published
    on the pod or by another launcher are seen), every attempt tied to the pod PRESERVED (`report`), and the pod guard's own verdict that
    every run dir under <root>/runs has custody (a valid .fetched or .custody; pods/sh/pod_guard.sh).  Without a fresh guard state the
    pod's unpublished run dirs are unknown, and custody does not pass.  -> (ok, why, detail)."""
    t = resolve_target(target)
    detail: dict[str, Any] = {"target": target, "pod_id": t["pod_id"], "refresh": None, "attempts": None, "guard": None}
    if refresh:
        try:
            r = st.refresh()
            detail["refresh"] = {k: r.get(k) for k in ("manifests", "attempts", "labels", "errors")}
        except Exception as e:  # noqa: BLE001
            return False, f"catalog refresh from the remote failed ({type(e).__name__}: {str(e)[:100]})", detail
    rep = report(st, target, mode=mode, probe=probe)
    detail["attempts"] = {"n": rep["n_attempts"], "unpreserved": rep["unpreserved"]}
    if rep["unpreserved"]:
        return False, f"{len(rep['unpreserved'])} attempt(s) on it not preserved (first {rep['unpreserved'][0]}) -> research data preserved RUN", detail
    root = (t.get("entry") or {}).get("root") or "/workspace/research"
    gs, why = state(t["pod_id"], root)
    if gs is None:
        detail["guard"] = {"error": why}
        return False, f"its run dirs' custody is unknown: {why}", detail
    age = (time.time() if now is None else now) - float(gs.get("updated_epoch") or 0)
    detail["guard"] = {"verdict": gs.get("verdict"), "age_s": round(age), "unfetched": gs.get("unfetched") or [], "runs": gs.get("runs")}
    if age > GUARD_STATE_MAX_AGE_S:
        return False, f"the pod guard's state is {age / 60:.0f} min old (guard dead?): its run dirs' custody is unknown", detail
    if gs.get("unfetched"):
        u = gs["unfetched"][0]
        return False, f"{len(gs['unfetched'])} run dir(s) on it without custody (first {u.get('run')}: {u.get('why')})", detail
    return True, f"{rep['n_attempts']} attempt(s) preserved, {gs.get('runs', 0)} run dir(s) with custody (guard)", detail


def log_drain(entry: dict[str, Any], path: Path | None = None) -> None:
    path = path or drain_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")


def drain(st: LocalStore, target: str, *, mode: str = "head", force: bool = False, reason: str | None = None, hostname: str | None = None,
          probe: bool = True, terminate=None, log_path: Path | None = None, custody_r2: bool = False,
          pod_custody_fn: Callable[..., tuple[bool, str, dict[str, Any]]] | None = None) -> tuple[int, dict[str, Any]]:
    """Report, then terminate unless something is unpreserved (module docstring).  `terminate(pod_id)` defaults to runpod's; the
    decision and its reason are appended to `log_path`.  `custody_r2`: the catalog is refreshed from the remote first and `pod_custody`
    must pass too (every run dir on the pod has custody, by the pod guard's verdict).  Returns (exit code, report)."""
    log_path = log_path or drain_log_path()
    if custody_r2:
        ok, why, detail = (pod_custody_fn or pod_custody)(st, target, refresh=True, probe=probe)
    rep = report(st, target, mode=mode, hostname=hostname, probe=probe)
    if not rep["pod_id"]:
        rep["decision"] = {"action": "refused", "why": f"{target!r} is neither a machines.toml entry nor a pod id"}
        return 2, rep
    if custody_r2:
        rep["custody_r2"] = {"ok": ok, "why": why, **detail}
        if not ok and why not in rep["unpreserved"]:
            rep["unpreserved"].append(f"custody: {why}")
    blocked = bool(rep["unpreserved"])
    if blocked and not force:
        rep["decision"] = {"action": "refused", "why": f"{len(rep['unpreserved'])} attempt(s) on the machine are not preserved; `research data push` them "
                                                       f"(or --force --reason TEXT to terminate anyway, logged)", "forced": False}
        log_drain({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "target": target, "machine": rep["machine"], "pod_id": rep["pod_id"],
                   "action": "refused", "unpreserved": rep["unpreserved"], "mode": mode}, log_path)
        return 1, rep
    if blocked and force and not reason:
        rep["decision"] = {"action": "refused", "why": "--force needs --reason TEXT (the override is logged with it)", "forced": False}
        return 2, rep
    if terminate is None:
        from . import runpod

        terminate = runpod.terminate
    try:
        terminate(rep["pod_id"])
    except Exception as e:  # noqa: BLE001
        rep["decision"] = {"action": "error", "why": f"terminate {rep['pod_id']}: {type(e).__name__}: {e}", "forced": bool(blocked and force)}
        log_drain({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "target": target, "machine": rep["machine"], "pod_id": rep["pod_id"],
                   "action": "error", "error": rep["decision"]["why"], "unpreserved": rep["unpreserved"], "forced": blocked and force, "reason": reason}, log_path)
        return 2, rep
    rep["decision"] = {"action": "terminated", "forced": bool(blocked and force), "reason": reason if blocked else None,
                       "why": (f"FORCED over {len(rep['unpreserved'])} unpreserved attempt(s): {reason}" if blocked else
                               ("nothing recorded on the machine" if not rep["n_attempts"] else f"all {rep['n_attempts']} attempt(s) preserved"))}
    log_drain({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "target": target, "machine": rep["machine"], "pod_id": rep["pod_id"],
               "action": "terminated", "forced": blocked and force, "reason": reason if blocked else None, "unpreserved": rep["unpreserved"],
               "n_attempts": rep["n_attempts"], "mode": mode}, log_path)
    return 0, rep


def _parser(prog: str, drain_flags: bool):
    import argparse

    ap = argparse.ArgumentParser(prog=prog)
    ap.add_argument("target", metavar="MACHINE|POD_ID")
    ap.add_argument("--mode", choices=list(P.MODES), default="head", help="the `research data preserved` mode (head: ask the remote; recorded: offline)")
    ap.add_argument("--store", default=None, help="the research store root (default: the configured one)")
    ap.add_argument("--hostname", default=None, help="the pod's container hostname (default: probed over ssh when the pod is RUNNING)")
    ap.add_argument("--no-probe", action="store_true", help="do not ssh the pod for its hostname")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--custody-r2", action="store_true", help="from any machine: refresh the catalog from the remote first, and require every run dir "
                                                               "on the pod to have custody by the pod guard's verdict (.fetched or .custody)")
    if drain_flags:
        ap.add_argument("--force", action="store_true", help="terminate although attempts are unpreserved (needs --reason; logged)")
        ap.add_argument("--reason", default=None, help="why the override is acceptable (goes to the drain log and the report)")
    return ap


def main(argv: list[str]) -> int:
    """`research pods unpreserved ...` | `research pods drain ...` | `research pods terminate <pod-id> --require-preserved ...`."""
    verb, rest = argv[0], argv[1:]
    if verb == "terminate":
        rest = [x for x in rest if x != "--require-preserved"]
        verb = "drain"
    ap = _parser(f"research pods {verb}", drain_flags=(verb == "drain"))
    ns = ap.parse_args(rest)
    try:
        st = LocalStore(ns.store)
    except Exception as e:  # noqa: BLE001
        print(f"research pods {verb}: no store: {e}", file=sys.stderr)
        return 2
    if verb == "unpreserved":
        custody = pod_custody(st, ns.target, refresh=True, probe=not ns.no_probe) if ns.custody_r2 else None
        rep = report(st, ns.target, mode=ns.mode, hostname=ns.hostname, probe=not ns.no_probe)
        if custody is not None:
            rep["custody_r2"] = {"ok": custody[0], "why": custody[1], **custody[2]}
            if not custody[0]:
                rep["unpreserved"].append(f"custody: {custody[1]}")
        if not rep["pod_id"]:
            print(f"research pods unpreserved: {ns.target!r} is neither a machines.toml entry nor a pod id", file=sys.stderr)
            return 2
        print(json.dumps(rep, indent=2, sort_keys=True) if ns.json else "\n".join(format_report(rep)))
        return 1 if rep["unpreserved"] else 0
    rc, rep = drain(st, ns.target, mode=ns.mode, force=ns.force, reason=ns.reason, hostname=ns.hostname, probe=not ns.no_probe,
                    custody_r2=ns.custody_r2)
    if ns.json:
        print(json.dumps(rep, indent=2, sort_keys=True))
    else:
        for line in format_report(rep):
            print(line)
        d = rep.get("decision") or {}
        print(f"{d.get('action', '?').upper()}: {d.get('why')}")
    return rc
