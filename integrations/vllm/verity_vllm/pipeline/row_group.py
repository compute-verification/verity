"""A config run's Commit stage as one grouped Commit of several rows of one engine (`commit_group`), and the replays of the rows it
committed: `row_stages.SingleRow` with ``--commit-group MEMBERS`` (``COMMIT_GROUP``) on the lead row, every row's Build done.

MEMBERS (``verity-vllm/commit-group-members/v1``): ``{"rows": [{"row", "role", "repo", "rev", "sweep_dir", "gpu_util"?}]}``, the lead
among them, in the order their Commits run.  Each other row (a member) is the row the lead would be with that row's id, model and
sweep dir: the lead's stages, configuration and environment, its own row dir and timeline.

Commit (`commit`): each member's precheck and Commit preparation as its own Commit stage runs them (`SingleRow.prepare`,
`commit_prepare`, and `research.build_input` when its Build task left its run id), then ONE `verity-vllm commit-group` process under
the lead's GPU lease (`Row.run_gpu`) runs every row's Commit with that row's own command line and environment, into its own row dir
and ``commit.log``.  Each row's verdict is its own (`row_config.commit_verdict`); a member whose precheck or preparation stops, whose
environment differs from the lead's in more than its row dir and timeline, or whose Commit the group refuses has a Commit FAIL that
names why.  Beside every row, ``gpu_lease.json`` is the group's one lease (with ``group``) and ``commit_group.json`` its part in the
group.  The stage exits as the lead's verdict does.

Replay (`replay`): every member whose Commit in this group ended PENDING replays from its own bundle (`row_config.replay`), one
after another, then the lead.
"""
from __future__ import annotations

import dataclasses
import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

from verity_vllm import protocols as PO
from verity_vllm.config import row_world
from verity_vllm.pipeline import commit_group as CG
from verity_vllm.pipeline import gpu_lease as GL
from verity_vllm.pipeline import research
from verity_vllm.pipeline import row_config as RC
from verity_vllm.pipeline.row_driver import Stop, utc

MEMBERS_SCHEMA = "verity-vllm/commit-group-members/v1"
ROW_SCHEMA = "verity-vllm/commit-group-row/v1"
PER_ROW = frozenset({"VERITY_TIMELINE", "SWEEP_DIR"})      # the environment keys a member's Commit has of its own
FOLLOW_S = 30.0


@dataclass(frozen=True)
class Member:
    row: str
    role: str
    repo: str
    rev: str
    sweep_dir: str
    gpu_util: str | None = None


def load(path: str) -> list[Member]:
    """MEMBERS -> its rows; ValueError names what is wrong with it."""
    with open(path) as f:
        doc = json.load(f)
    if not isinstance(doc, dict) or doc.get("schema") != MEMBERS_SCHEMA or not isinstance(doc.get("rows"), list) or not doc["rows"]:
        raise ValueError(f"not a {MEMBERS_SCHEMA} document with rows")
    try:
        out = [Member(row=r["row"], role=r["role"], repo=r["repo"], rev=r["rev"], sweep_dir=r["sweep_dir"], gpu_util=r.get("gpu_util"))
               for r in doc["rows"]]
    except (KeyError, TypeError) as e:
        raise ValueError(f"a row needs row, role, repo, rev and sweep_dir ({e!r})") from None
    dirs = [os.path.join(os.path.realpath(m.sweep_dir), m.row) for m in out]
    if len(set(dirs)) != len(dirs):
        raise ValueError("two rows share a row dir")
    return out


def is_lead(lead: Any, m: Member) -> bool:
    return m.row == lead.row and os.path.realpath(m.sweep_dir) == os.path.realpath(lead.env.sweep_dir)


def member(lead: Any, m: Member) -> Any:
    """The row ``m`` as the lead's stage would run it: the lead's stages, configuration (but its group) and environment, in ``m``'s
    sweep dir with its own timeline, and no research run of its own."""
    environ = {k: v for k, v in lead.env.environ.items() if k != "VERITY_TIMELINE"}
    env = dataclasses.replace(lead.env, sweep_dir=m.sweep_dir, timeline=None, research_run_dir=None, research_run_id=None,
                              environ={**environ, "SWEEP_DIR": m.sweep_dir})
    return type(lead)(row=m.row, role=m.role, repo=m.repo, rev=m.rev, stages=lead.stages, env=env, gpu_util=m.gpu_util or lead.gpu_util,
                      build=lead.cb, match=lead.cm, commit=dataclasses.replace(lead.ck_, group=""), protocols=None)


def refusal(lead: Any) -> str | None:
    """Why the lead's Commit can't be a grouped Commit, None when it can."""
    ck = lead.ck_
    return ("a grouped Commit is a config run's (--config-run 1)" if not lead.config_run else
            "a grouped Commit defers every row's replay (--replay-deferred 1 or slim)" if ck.replay_deferred not in ("1", "slim") else
            "a grouped Commit is not the hot worker's (--hot-engine 1)" if ck.hot_engine == "1" else
            "a grouped Commit takes no --serving-rows: a row's own deferred Commit commits its window at the C2 site "
            "(`c2_replay.serving_rows`), and `verity-vllm serving-rows BUNDLE` writes a stand-in from each row's bundle" if ck.serving_rows else
            f"a grouped Commit installs no protocol options ({list(lead.protocols)})" if lead.protocols else None)


def build_input(r: Any) -> str | None:
    """`research.stage`'s Build-input check for a member, against the Build its Build task published (``two-task-build-run``): the
    precheck FAIL detail, or None (a member whose Build task left no run id, or no artifact, cites none, as its own Commit task would)."""
    try:
        with open(f"{r.d}/two-task-build-run") as f:
            run_id = f.read().strip()
    except OSError:
        return None
    child = research.abs_store_env(dict(r.env.environ))
    cli = research.research_argv(r.env.environ.get("RESEARCH_BIN"), r.env.environ.get("RESEARCH_PY") or "python3", child)
    build_in = research.attempt_output(cli, run_id, "build") if run_id else ""
    if not build_in:
        return None
    why, here = research.build_input(r.py, r.d, r.row, build_in, child, lambda: cli)
    if why is None:
        r.log(f"grouped Commit: consumes Build {build_in} (program_digest {here[:16]} verified)")
    return why


def prepared(lead: Any, r: Any) -> tuple[int, list[str]] | str:
    """A member's Commit stage up to its command line (its timeout and command), or why it is refused."""
    os.makedirs(r.d, exist_ok=True)
    why = build_input(r)
    if why:
        with open(f"{r.d}/stages.txt", "a") as f:
            f.write(f"precheck FAIL {utc()} {why}\n")
        return why
    stage = "precheck"
    try:
        r.prepare()
        if row_world(r.row) > 1:
            return "a tensor-parallel row (a grouped Commit is a single-GPU row's)"
        with open(r.path(r.wl)) as f:
            if PO.at_row(None, json.load(f), world=1) is not None:
                return "the row runs protocol options beside sampled proofs, which a grouped Commit does not install"
        stage = "Commit preparation"
        to, cmd, _ = r.commit_prepare()
    except Stop as s:
        return f"its {stage} stopped (exit {s.rc}; {r.d}/stages.txt)"
    except PO.Refused as e:
        return f"its protocol set is refused: {e}"
    differs = sorted(k for k in set(lead.child_env) | set(r.child_env) if k not in PER_ROW and lead.child_env.get(k) != r.child_env.get(k))
    if differs:
        return f"its Commit's environment differs from the lead's in {differs}"
    return to, cmd


def mirror(target: str, sources: list[str]) -> float | None:
    """``target``'s modification time moved to the newest of ``sources``' when that is later -> the new time, else None."""
    try:
        newest = max((os.stat(s).st_mtime for s in sources if os.path.exists(s)), default=None)
        if newest is None or newest <= os.stat(target).st_mtime:
            return None
        os.utime(target, (newest, newest))
    except OSError:
        return None
    return newest


def follow(target: str, sources: list[str], stop: threading.Event, every_s: float = FOLLOW_S) -> None:
    """Until ``stop``: `mirror` every ``every_s``, so the lead's ``commit.log``, which the GPU task's stall watchdog watches, moves while
    any row of the group writes its log."""
    while not stop.wait(every_s):
        mirror(target, sources)


def _load(path: str) -> dict[str, Any]:
    try:
        with open(path) as f:
            doc = json.load(f)
    except (OSError, ValueError):
        return {}
    return doc if isinstance(doc, dict) else {}


def _write(path: str, doc: dict[str, Any]) -> None:
    with open(path + ".tmp", "w") as f:
        json.dump(doc, f, indent=1, default=str)
    os.replace(path + ".tmp", path)


def _refused(r: Any, rc: int, why: str) -> None:
    r.verdict("commit", "FAIL", f"rc={rc} REFUSED by the grouped Commit: {why}")
    r.tl("stage.commit", "end", ("ok", "false"))
    r.tl("row", "end", ("ok", "false"), ("stopped_at", "commit"))
    RC.discard_bundles(r)


def run_group(lead: Any, ready: list[tuple[Any, int, list[str]]], cap: int) -> tuple[int, int, str]:
    """The one `commit-group` process over the ``ready`` rows (row, timeout, command), under the lead's lease -> (its exit, its wall,
    the result path)."""
    gdir = f"{lead.d}/commit_group"
    os.makedirs(gdir, exist_ok=True)
    spec_path, result = f"{gdir}/spec.json", f"{gdir}/result.json"
    if os.path.exists(result):
        os.remove(result)
    rows = [{"row": r.row, "dir": r.d, "argv": cmd[4:], "env": {k: r.child_env[k] for k in PER_ROW if k in r.child_env},
             "log": f"{r.d}/commit.log"} for r, _, cmd in ready]
    _write(spec_path, {"schema": CG.SCHEMA, "lead": lead.row, "members": lead.ck_.group, "result": result, "rows": rows})
    gcmd = lead.cli("commit-group", spec_path)
    for i, (r, _, cmd) in enumerate(ready):
        with open(f"{r.d}/commit.log", "w") as f:
            f.write(f"$ {' '.join(cmd)}\n# row {i + 1} of {len(ready)} of the grouped Commit {' '.join(gcmd)} (lead {lead.row})\n")
        if r is not lead:
            r.tl("commit.commit_delta", "start", ("group", lead.row))
    to = min(cap, sum(t for _, t, _ in ready))
    lead.log(f"grouped Commit: {len(ready)} rows ({', '.join(r.row for r, _, _ in ready)}) in one process under one GPU lease, timeout {to}s")
    stop = threading.Event()
    watch = threading.Thread(target=follow, args=(f"{lead.d}/commit.log", [f"{lead.d}/commit_group.log", *(row["log"] for row in rows)], stop),
                             daemon=True)
    watch.start()
    lead.gpu_exclusive("commit-group")
    t0 = int(time.time())
    try:
        rc = lead.run_gpu("commit.commit_group", to, f"{lead.d}/commit_group.log", gcmd)
    finally:
        stop.set()
        watch.join()
    return rc, int(time.time()) - t0, result


def commit(lead: Any, cap: int) -> bool:
    """The lead's grouped Commit stage (the module docstring) -> whether the lead's Commit ended PENDING."""
    why = refusal(lead)
    members: list[Member] = []
    if why is None:
        try:
            members = load(lead.ck_.group)
        except (OSError, ValueError) as e:
            why = f"--commit-group {lead.ck_.group}: {e}"
    if why is None and not any(is_lead(lead, m) for m in members):
        why = f"--commit-group {lead.ck_.group} does not name this row ({lead.row} in {lead.env.sweep_dir})"
    if why is not None:
        lead.tl("stage.commit", "start")
        lead.commit_stop(CG.RC_REFUSED, f"rc={CG.RC_REFUSED} REFUSED: {why}")
    to, cmd, _ = lead.commit_prepare()
    ready: list[tuple[Any, int, list[str]]] = []
    refused: list[tuple[Any, str]] = []
    for m in members:
        if is_lead(lead, m):
            ready.append((lead, to, cmd))
            continue
        r = member(lead, m)
        p = prepared(lead, r)
        if isinstance(p, str):
            refused.append((r, p))
        elif p[1][:4] != cmd[:4]:
            refused.append((r, f"its Commit runs another interpreter or command ({p[1][:4]})"))
        else:
            ready.append((r, *p))
    lead.install_signal_forwarding()
    for r, w in refused:
        lead.log(f"grouped Commit: {r.row} refused: {w}")
    try:
        rc, wall, result = run_group(lead, ready, cap)
    except Stop as s:
        for r, _, _ in ready:
            if r is not lead:
                _refused(r, s.rc, f"the grouped Commit was stopped (exit {s.rc})")
                RC.config_record(r)
        raise
    return verdicts(lead, ready, refused, rc, wall, result)


def verdicts(lead: Any, ready: list[tuple[Any, int, list[str]]], refused: list[tuple[Any, str]], rc: int, wall: int, result: str) -> bool:
    """Each row's Commit verdict from the group's result, the members' first; the lead's last -> whether it ended PENDING."""
    doc = _load(result)
    lease = _load(f"{lead.d}/{GL.RECORD}")
    group = {"lead": lead.row, "rows": [r.row for r, _, _ in ready], "record": f"{lead.d}/{GL.RECORD}", "wall_s": wall, "rc": rc}
    if lease:
        _write(f"{lead.d}/{GL.RECORD}", {**lease, "group": group})
    before = {id(r): why for r, why in refused}
    lead_res: dict[str, Any] = {}
    lead_rc = rc
    for r in [*(r for r, _, _ in ready), *(r for r, _ in refused)]:
        res = (doc.get("rows") or {}).get(r.row)
        why = before.get(id(r))
        if why is None and res is None:
            why = f"the grouped Commit ended (exit {rc}) before this row's Commit ran"
        elif why is None and res.get("refused"):
            why = res["refused"]
        rck = (res or {}).get("rc", rc or 1) if why is None else ((res or {}).get("rc") or CG.RC_REFUSED)
        if r is not lead and lease:
            _write(f"{r.d}/{GL.RECORD}", {**lease, "group": group})
        if r is lead:
            if why is not None:
                _write(f"{r.d}/commit_group.json", {"schema": ROW_SCHEMA, **group, "this": res, "refused": why, "pending": False})
                lead.commit_stop(CG.RC_REFUSED, f"rc={rck} REFUSED by the grouped Commit: {why}")
            lead_res, lead_rc = res, rck
            continue
        pending = False
        if why is not None:
            _refused(r, rck, why)
        else:
            try:
                pending = RC.commit_verdict(r, rck, int(res.get("wall_s") or 0))
            except Stop:
                pending = False
        _write(f"{r.d}/commit_group.json", {"schema": ROW_SCHEMA, **group, "this": res, "refused": why, "pending": pending})
        if pending:
            r.log(f"config run PENDING: the C2 replay runs on CPU from the sealed bundle (the replay stage of the group's lead {lead.row})")
        elif r.trap:
            RC.config_record(r)
    pending = False
    try:
        pending = RC.commit_verdict(lead, lead_rc, int(lead_res.get("wall_s") or wall))
    finally:
        _write(f"{lead.d}/commit_group.json", {"schema": ROW_SCHEMA, **group, "this": lead_res, "refused": None, "pending": pending})
    return pending


def replay(lead: Any) -> None:
    """The lead's replay stage: each member whose Commit in this group ended PENDING (its ``commit_group.json``), then the lead."""
    try:
        members = load(lead.ck_.group)
    except (OSError, ValueError) as e:
        lead.log(f"grouped replay: --commit-group {lead.ck_.group} unreadable ({e}): only this row replays")
        members = []
    for m in members:
        if is_lead(lead, m):
            continue
        r = member(lead, m)
        rec = _load(f"{r.d}/commit_group.json")
        if rec.get("lead") != lead.row or not rec.get("pending"):
            lead.log(f"grouped replay: {m.row} not replayed (its Commit in this group did not end PENDING; {r.d}/commit_group.json)")
            continue
        try:
            r.prepare()
            RC.replay(r)
        except Stop as s:
            lead.log(f"grouped replay: {m.row} stopped (exit {s.rc}; {r.d}/stages.txt)")
        if r.trap:
            RC.config_record(r)
        lead.log(f"grouped replay: {m.row} done ({r.d}/config_record.json)")
    lead.install_signal_forwarding()
    RC.replay(lead)
