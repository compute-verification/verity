"""failure.json classifier v1 -- a reproducible interpretation of preserved evidence (brief §6).  Observations first; rc=137 alone is
UNKNOWN_SIGNAL(9) with the incident list, never OOM; incidents are recorded separately from the job's final outcome.

    tele classify --attempt-dir D [--rc N] [--signal S] [--log PATH ...] [--check-record PATH] [--stages-txt PATH]

Rules (first match by priority; every rule's evidence refs recorded; ambiguity lowers certainty, never raises it):
  1 structured checker record with verdict != PASS                -> WORKLOAD_CHECK_FAILED   (BEFORE the rc==0 rule: a checker that writes
                                                                     FAIL and exits 0 is a failed workload -- rev d1)
  2 rc==0 and checker record absent or PASS                       -> SUCCESS (+ incidents; an open root span is a limitation, not a failure)
  3 CUDA allocation failure string in the job's own log           -> CUDA_ALLOC_FAIL
  4 ENOSPC/EDQUOT in the log, or a declared mount sampled at 0    -> STORAGE_EXHAUSTED
  5 cancel_intent matching the job's pid/pgid before the exit, delivered or exit signal == requested
                                                                  -> STAGE_TIMEOUT (actor=timeout) | CANCELLED_DEADMAN | CANCELLED_GUARDIAN | CANCELLED_MANUAL
  6 exit by SIGKILL + cgroup oom_kill delta >= 1 in the job's interval:
        job_cgroup scope                                          -> CGROUP_OOM_ATTRIBUTED (established)
        shared scope, PRIMARY evidence (rev S-r17-51 c2): a job-tree process vanished in the oom_kill window AND no outsider vanished
                                                                  -> CGROUP_OOM_ATTRIBUTED, certainty inferred, never established
        corroboration, recorded in oom.attribution and NEVER a veto: the vanished process had the highest sampled /proc/<pid>/oom_score
        (the kernel's own victim rule) / the largest anon+shmem at the preceding sample; disagreement is a limitation line
        otherwise                                                 -> OOM_UNRESOLVED_ORIGIN
     The run's own queue scope (`cluster-<run id>.scope`, which `research run --queue` starts) counts as the job's own cgroup here and in 7b.
  6b exit by SIGTERM + an oom_kill increment in the run's own queue scope
                                                                  -> CGROUP_OOM_ATTRIBUTED (0.1.4), established: systemd stopped the scope after the kernel
                                                                     OOM-killed a process in it.  Every OOM class in the job's own cgroup names its cap
                                                                     (`MemoryMax=NG`, job.json's environment.cgroup.memory_max_bytes) in `why`.
  COEXISTENCE (rev S-r17-51 c1, brief s6 'retain both'): whenever the job exited by SIGKILL with an oom_kill delta in its interval and rules 3-5
  chose another class (CUDA string, low mount, marker), the class is that first match but certainty = ambiguous, both evidence sets are
  recorded (`competing` = [chosen, other]; the OOM side is also an incident).  Exempt: a DELIVERED and identity-VERIFIED marker stays
  established (the kill is observed, not inferred).  The sampled-mount branch of rule 4 applies only to the attempt's own mount or a
  declared-output mount; other low mounts are incidents.
  7 exit by signal                                                -> UNKNOWN_SIGNAL (timeout inferred from the runner's timeout_s + SIGTERM/SIGKILL -> STAGE_TIMEOUT,
                                                                     certainty inferred; evaluated AFTER rule 6; with no resource samples at all: ambiguous)
  7b rc != 0 (no signal), an oom_kill increment in the job's interval AND a `Killed` line in the job's log
                                                                  -> CGROUP_OOM (0.1.3, r20 issue 3): the shell/wrapper reported a SIGKILLed descendant
                                                                     (`bash: line N: PID Killed cmd`, `Out of memory: Killed process`) and the cgroup counted a
                                                                     kill in the same interval.  certainty established in job_cgroup scope, inferred in the
                                                                     shared pod scope (the victim is named by the log line and the disappearance window, not by
                                                                     the kernel).  cgroup v1 counts through memory.oom_control's oom_kill (sampler maps it to
                                                                     memory_events.oom_kill); memory.failcnt growth is recorded as corroboration only.
                                                                     A v1 cgroup with NO oom_kill counter (kernel < 4.13, sampled None) substitutes memory.failcnt
                                                                     growth or under_oom=1 for the increment: inferred in job_cgroup, ambiguous shared (failcnt
                                                                     also counts reclaim at the limit).  A counter that exists and did not move vetoes the rule.
  8 rc != 0                                                       -> UNKNOWN_EXIT (a traceback is evidence, not a class; a child OOM-killed while the parent exits rc!=0
                                                                     WITHOUT a Killed line stays UNKNOWN_EXIT with the incident carrying the vanished child's pid,
                                                                     delta and last oom_score -- rev d4; the coexisting incident is named in limitations)
Every cgroup_oom incident carries, per window: who vanished (tree / outsiders), the highest-oom_score process at the preceding sample, and
attribution_sample_age_s = window end - preceding process sample (rev d8).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from . import CLASSIFIER_VERSION, FAILURE_SCHEMA
from .events import ReadReport, iter_records, read_all, spans
from .job import read_status, write_json_atomic

CUDA_RE = re.compile(r"CUDA out of memory|cudaErrorMemoryAllocation|CUBLAS_STATUS_ALLOC_FAILED|CUDA error: out of memory|torch\.OutOfMemoryError", re.I)
ENOSPC_RE = re.compile(r"No space left on device|Disk quota exceeded|ENOSPC|EDQUOT")
TRACEBACK_RE = re.compile(r"^Traceback \(most recent call last\):", re.M)
MEMERR_RE = re.compile(r"MemoryError|std::bad_alloc|Cannot allocate memory")
# the shell's job-control report of a SIGKILLed child (`bash: line 1: 4242 Killed    python ...`, `sh: 1: python: Killed`, a bare `Killed` line),
# the kernel's own line when dmesg reaches the log, or Python's subprocess report of the kill.  Prose ("Killed by ...") does not match.  Rule 7b
# needs a cgroup kill count (or, v1 without one, failcnt/under_oom) beside it; the string alone is evidence only.
KILLED_RE = re.compile(r"\b\d+\s+Killed(?:\s|$)|^\s*Killed\s*$|:\s+Killed\s*$|Out of memory: Killed process|Memory cgroup out of memory|oom-kill:"
                       r"|died with <Signals\.SIGKILL|returned non-zero exit status -9\b", re.M)
ACTOR_CLASS = {"timeout": "STAGE_TIMEOUT", "deadman": "CANCELLED_DEADMAN", "guardian": "CANCELLED_GUARDIAN", "manual": "CANCELLED_MANUAL"}
SIGNAMES = {9: "SIGKILL", 15: "SIGTERM", 2: "SIGINT", 1: "SIGHUP", 6: "SIGABRT", 11: "SIGSEGV", 7: "SIGBUS"}


def _sha(path: Path, cap: int = 64 << 20) -> tuple[str | None, int | None]:
    try:
        n = path.stat().st_size
    except OSError:
        return None, None
    if n > cap:
        return None, n
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest(), n


def _tail(path: Path, n: int = 256 << 10) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - n))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def _events_of(kind: str, recs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in recs if r.get("kind") == kind]


def _oom_count(cg: dict[str, Any]) -> dict[str, int]:
    out = {}
    for key in ("memory_events", "memory_events_local", "v1_memory_events"):
        ev = cg.get(key)
        if isinstance(ev, dict):
            for k in ("oom", "oom_kill", "oom_group_kill", "max"):
                if k in ev and ev[k] is not None:
                    out[f"{key}.{k}"] = int(ev[k])
    return out


def classify(attempt_dir: str | Path, rc: int | None = None, sig: int | None = None, logs: list[str] | None = None,
             check_record: str | None = None, stages_txt: str | None = None) -> dict[str, Any]:
    d = Path(attempt_dir)
    status = read_status(d)
    try:
        jobdoc = json.loads((d / "job.json").read_text()) if (d / "job.json").exists() else {}
    except ValueError:
        jobdoc = {}
    declared_outputs = [str(x) for x in ((jobdoc.get("execution") or {}).get("declared_outputs") or [])]
    evidence: list[dict[str, Any]] = []
    limitations: list[str] = []
    incidents: list[dict[str, Any]] = []
    inputs: dict[str, Any] = {}

    def ref(kind: str, **kv: Any) -> None:
        evidence.append({"kind": kind, **kv})

    # ---- inputs ------------------------------------------------------------------------------------------------------------------------
    ev_rep = read_all(d, "events")
    res_rep = read_all(d, "resources")
    for p in ev_rep.files + res_rep.files:
        sha, n = _sha(Path(p))
        inputs[os.path.basename(p)] = {"sha256": sha, "bytes": n}
    telemetry_partial = bool(ev_rep.partial or res_rep.partial)
    for part in ev_rep.partial + res_rep.partial:
        limitations.append(f"partial telemetry: {part.get('file')} {part.get('reason')}")
    recs = ev_rep.records
    samples = res_rep.records
    if not samples:
        limitations.append("no resource samples: cgroup/process evidence unavailable")
    sampler_ends = _events_of("sampler_end", recs)
    if _events_of("sampler_start", recs) and not sampler_ends:
        telemetry_partial = True
        limitations.append("sampler started but never wrote sampler_end: sampling interrupted (killed?) -- coverage of the final interval unknown")
    elif sampler_ends and sampler_ends[-1].get("reason") not in ("workload_exit", None) and not str(sampler_ends[-1].get("reason", "")).startswith("signal"):
        telemetry_partial = True
        limitations.append(f"sampler ended abnormally: {sampler_ends[-1].get('reason')}")
    gaps = _events_of("sampler_gap", recs)
    if gaps:
        limitations.append(f"{len(gaps)} sampling gap(s) > 2x cadence; transients in them were not observed")

    # ---- termination -------------------------------------------------------------------------------------------------------------------
    term_rc = rc if rc is not None else status.get("rc")
    term_sig = sig if sig is not None else status.get("signal")
    if term_sig is None and isinstance(term_rc, int) and term_rc > 128 and status.get("rc_is_shell_status", True) and term_rc - 128 in SIGNAMES:
        term_sig = term_rc - 128
        term_note = f"signal inferred from shell status {term_rc} = 128 + {term_sig}"
    else:
        term_note = None
    observed_by = "argument" if rc is not None or sig is not None else ("status.json" if status.get("rc") is not None or status.get("signal") is not None else "none")
    termination = {"rc": term_rc, "signal": term_sig, "signal_name": SIGNAMES.get(term_sig) if term_sig else None, "observed_by": observed_by, "note": term_note}
    if term_rc is None and term_sig is None:
        limitations.append("no termination status recorded (status.json has neither rc nor signal)")

    # ---- location ----------------------------------------------------------------------------------------------------------------------
    sp = spans(recs)
    open_spans = [s for s in sp["spans"] if s["outcome"] == "interrupted"]
    last_span = open_spans[-1] if open_spans else (sp["spans"][-1] if sp["spans"] else None)
    location: dict[str, Any] = {"stage": last_span["stage"] if last_span else None, "span_id": last_span["id"] if last_span else None,
                                "span_outcome": last_span["outcome"] if last_span else None, "pid": status.get("pid"), "pgid": status.get("pgid")}
    st_path = Path(stages_txt) if stages_txt else d / "stages.txt"
    if st_path.exists():
        lines = [l for l in _tail(st_path, 64 << 10).splitlines() if l.strip()]
        if lines:
            location["stages_txt_last"] = lines[-1][:200]
            ref("stages.txt", path=str(st_path), last_line=lines[-1][:200])
    job_start = status.get("started_mono") or next((t["t_mono"] for t in status.get("transitions", []) if t.get("state") == "running"), None)
    job_end = status.get("ended_mono") or next((t["t_mono"] for t in reversed(status.get("transitions", [])) if t.get("state") in ("done", "failed", "cancelled")), None)
    host = str(status.get("updated_stream", "")).split(":")[0]

    # ---- incidents: cgroup OOM deltas over the job interval -------------------------------------------------------------------------------
    cg_samples = [s for s in samples if s.get("kind") == "cgroup" and isinstance(s.get("cgroup"), dict)]
    cg_scope = cg_samples[0]["scope"] if cg_samples else None
    run_id = (jobdoc.get("ownership") or {}).get("job") or d.name
    cg_path = str((cg_samples[0]["cgroup"].get("path") if cg_samples else None) or ((jobdoc.get("environment") or {}).get("cgroup") or {}).get("path") or "")
    queue_scope = cg_path.rstrip("/").rsplit("/", 1)[-1] == f"cluster-{run_id}.scope"
    own_scope = cg_scope == "job_cgroup" or queue_scope
    cap_bytes = ((jobdoc.get("environment") or {}).get("cgroup") or {}).get("memory_max_bytes")
    cap = (f"MemoryMax={cap_bytes >> 30}G" if cap_bytes % (1 << 30) == 0 else f"MemoryMax={cap_bytes}") if isinstance(cap_bytes, int) else None
    if queue_scope:
        ref("cgroup.queue_scope", path=cg_path, memory_max=cap,
            note="the run's own allocation (`research run --queue`'s systemd scope): it holds the runner, its sampler and the workload, nothing else")
    oom_delta: dict[str, int] = {}
    delta_windows: list[dict[str, Any]] = []
    failcnt_delta: int | None = None
    under_oom_seen = kill_counter_seen = False
    if cg_samples:
        same_host = all(str(s.get("stream", "")).split(":")[0] == host for s in cg_samples) if host else True
        if not same_host:
            limitations.append("resource samples come from another host than the runner status: interval alignment by UTC only")
        def in_interval(s: dict[str, Any]) -> bool:
            if job_start is None or not same_host:
                return True
            t = float(s["t_mono"])
            return (t >= job_start - 5.0) and (job_end is None or t <= job_end + 30.0)
        window = [s for s in cg_samples if in_interval(s)] or cg_samples
        prev = None
        prev_counts = _oom_count(window[0]["cgroup"])
        first_counts = dict(prev_counts)
        for s in window[1:]:
            cur = _oom_count(s["cgroup"])
            for k, v in cur.items():
                if k in prev_counts and v > prev_counts[k]:
                    delta_windows.append({"counter": k, "from": prev_counts[k], "to": v, "window_start_mono": prev["t_mono"] if prev else window[0]["t_mono"],
                                          "window_end_mono": s["t_mono"], "window_end_utc": s.get("t_utc")})
            prev_counts, prev = cur, s
        for k, v in prev_counts.items():
            if k in first_counts and v != first_counts[k]:
                oom_delta[k] = v - first_counts[k]
        # cgroup v1 corroboration: memory.failcnt counts limit hits (reclaim or kill), so its growth is recorded beside the kill counters, never as one
        fails = [int(s["cgroup"]["memory_failcnt"]) for s in window if isinstance(s["cgroup"].get("memory_failcnt"), int)]
        failcnt_delta = (fails[-1] - fails[0]) if len(fails) >= 2 else None
        under_oom_seen = any((s["cgroup"].get("memory_events") or {}).get("under_oom") == 1 for s in window)
        kill_counter_seen = any(k.endswith("oom_kill") for s in window for k in _oom_count(s["cgroup"]))
        if oom_delta:
            inc = {"kind": "cgroup_oom", "scope": cg_scope, "counters_delta": oom_delta, "windows": delta_windows,
                   "note": "an oom_kill increment in this scope means a process of the scope was OOM-killed; it does not by itself identify the victim"}
            if failcnt_delta:
                inc["v1_failcnt_delta"] = failcnt_delta
            incidents.append(inc)
            ref("cgroup.memory.events", scope=cg_scope, delta=oom_delta, file="resources.jsonl")
        peak = max((int(s["cgroup"].get("memory_peak_bytes") or 0) for s in window), default=0) or None
        cur_max = max((int(s["cgroup"].get("memory_current_bytes") or 0) for s in window), default=0) or None
        ref("cgroup.sampled", scope=cg_scope, samples=len(window), max_sampled_memory_current_bytes=cur_max, kernel_memory_peak_bytes=peak,
            peak_between_samples_possible=(peak is not None and cur_max is not None and peak > cur_max),
            peak_note="memory.peak is the cgroup's high-water mark and may predate the job (shared/long-lived cgroup); not differenced")

    # ---- process disappearance windows (for attribution) -----------------------------------------------------------------------------------
    proc_samples = [s for s in samples if s.get("kind") == "process"]

    def vanish_in(w: dict[str, Any]) -> dict[str, Any]:
        """who vanished in (window_start, window_end]: tree pids (gone) and outsiders; largest anon+shmem at the last sample before the window end."""
        a, b = float(w["window_start_mono"]), float(w["window_end_mono"])
        # one sampler round writes cgroup THEN process a few ms later: the process record of round j (which carries `gone`) sits just
        # after the cgroup record whose counter moved, so the window is extended by half a round on both ends
        half = 0.5 * max(b - a, 1e-3)
        tree_gone: list[int] = []
        out_gone: list[dict[str, Any]] = []
        before: dict[str, Any] | None = None
        for s in proc_samples:
            t = float(s["t_mono"])
            if a + half < t <= b + half:
                tree_gone += list(s.get("gone") or [])
                out_gone += list(s.get("outsiders_gone") or [])
            if t <= a + half:
                before = s
        largest = None          # largest anon+shmem (kept as evidence)
        highest = None          # highest sampled /proc/<pid>/oom_score (the kernel's victim rule)
        gone_detail: list[dict[str, Any]] = []
        scores_sampled = False
        if before is not None:
            rows = [(int((p.get("rss_anon_bytes") or 0) + (p.get("rss_shmem_bytes") or 0)), p.get("oom_score"), p.get("oom_score_adj"), "tree", p["pid"], p.get("comm"))
                    for p in before.get("processes") or []]
            rows += [(int(o.get("anon_shmem_bytes") or 0), o.get("oom_score"), o.get("oom_score_adj"), "outsider", o["pid"], o.get("comm"))
                     for o in before.get("outsiders") or []]
            if rows:
                m = max(rows, key=lambda r: r[0])
                largest = {"anon_shmem_bytes": m[0], "oom_score": m[1], "membership": m[3], "pid": m[4], "comm": m[5]}
                scored = [r for r in rows if r[1] is not None]
                scores_sampled = bool(scored)
                if scored:
                    h = max(scored, key=lambda r: (r[1], r[0]))
                    ties = [r for r in scored if r[1] == h[1]]
                    highest = {"oom_score": h[1], "oom_score_adj": h[2], "anon_shmem_bytes": h[0], "membership": h[3], "pid": h[4], "comm": h[5],
                               "tied": len(ties) > 1, "tied_memberships": sorted({r[3] for r in ties})}
            by_pid = {r[4]: r for r in rows}
            for pid in tree_gone:
                r = by_pid.get(pid)
                gone_detail.append({"pid": pid, "membership": "tree", "comm": r[5] if r else None, "last_oom_score": r[1] if r else None,
                                    "last_anon_shmem_bytes": r[0] if r else None})
            for o in out_gone:
                r = by_pid.get(o.get("pid")) if isinstance(o, dict) else None
                gone_detail.append({"pid": o.get("pid") if isinstance(o, dict) else o, "membership": "outsider", "comm": r[5] if r else (o.get("comm") if isinstance(o, dict) else None),
                                    "last_oom_score": r[1] if r else None, "last_anon_shmem_bytes": r[0] if r else None})
        age = (b - float(before["t_mono"])) if before is not None else None
        return {"tree_gone": tree_gone, "outsiders_gone": out_gone, "gone_detail": gone_detail, "largest_before": largest, "highest_oom_score_before": highest,
                "oom_score_sampled": scores_sampled, "have_before_sample": before is not None, "attribution_sample_age_s": round(age, 3) if age is not None else None}

    if oom_delta:
        for inc in incidents:
            if inc.get("kind") == "cgroup_oom":
                inc["windows"] = [{**w, **vanish_in(w)} for w in inc["windows"]]

    # ---- logs ----------------------------------------------------------------------------------------------------------------------------
    log_paths = [Path(p) for p in (logs or [])]
    for cand in ("stdout.log", "stderr.log", "job.log"):
        if (d / cand).exists() and (d / cand) not in log_paths:
            log_paths.append(d / cand)
    cuda_hit = enospc_hit = traceback_hit = memerr_hit = killed_hit = None
    for lp in log_paths:
        text = _tail(lp)
        if not text:
            continue
        inputs[lp.name] = {"sha256": _sha(lp)[0], "bytes": _sha(lp)[1], "tail_bytes_scanned": min(len(text), 256 << 10)}
        for name, rx in (("cuda", CUDA_RE), ("enospc", ENOSPC_RE), ("traceback", TRACEBACK_RE), ("memerr", MEMERR_RE), ("killed", KILLED_RE)):
            m = rx.search(text)
            if m:
                line = text[text.rfind("\n", 0, m.start()) + 1: text.find("\n", m.end()) if text.find("\n", m.end()) != -1 else None][:200]
                hit = {"file": str(lp), "match": line}
                if name == "cuda" and cuda_hit is None:
                    cuda_hit = hit
                elif name == "enospc" and enospc_hit is None:
                    enospc_hit = hit
                elif name == "traceback" and traceback_hit is None:
                    traceback_hit = hit
                elif name == "memerr" and memerr_hit is None:
                    memerr_hit = hit
                elif name == "killed" and killed_hit is None:
                    killed_hit = hit
    if traceback_hit:
        ref("log.traceback", **traceback_hit)
    if memerr_hit:
        ref("log.allocation_error", **memerr_hit, note="user-space allocation failure (not an OOM kill)")
    if killed_hit:
        ref("log.killed", **killed_hit, note="a SIGKILLed process reported in the job's log; a class only beside a cgroup kill count (rule 7b)")

    # ---- storage --------------------------------------------------------------------------------------------------------------------------
    storage_zero = []          # low mounts that hold the attempt dir or a declared output (rule 4 material)
    storage_zero_other = []    # low mounts elsewhere (incident only)
    relevant_paths = [str(d.resolve())] + declared_outputs
    def holds_relevant(mount_path: str | None) -> bool:
        if not mount_path:
            return False
        mp = mount_path.rstrip("/") or "/"
        return any(rp == mp or rp.startswith(mp + "/") or mp == "/" for rp in relevant_paths)
    for s in samples:
        if s.get("kind") == "storage":
            for m in s.get("mounts") or []:
                fb = m.get("free_bytes")
                if fb is not None and fb < (16 << 20):
                    row = {"path": m.get("path"), "free_bytes": fb, "t_utc": s.get("t_utc")}
                    (storage_zero if holds_relevant(m.get("path")) else storage_zero_other).append(row)
    if storage_zero or storage_zero_other:
        incidents.append({"kind": "storage_low", "samples": (storage_zero + storage_zero_other)[:5],
                          "holds_attempt_or_declared_output": bool(storage_zero)})

    # ---- cancel markers ---------------------------------------------------------------------------------------------------------------------
    intents = _events_of("cancel_intent", recs)
    results = _events_of("cancel_result", recs)
    my_pid, my_pgid = status.get("pid"), status.get("pgid")
    matched_intent = None
    for it in intents:
        tgt = it.get("target") or {}
        hit = (tgt.get("pid") is not None and tgt.get("pid") == my_pid) or (tgt.get("pgid") is not None and tgt.get("pgid") == my_pgid)
        if not hit and my_pid is None and my_pgid is None:
            hit = True   # runner did not record its pid: any marker in the attempt dir is about this attempt
            limitations.append("status.json lacks pid/pgid: cancel marker matched to the attempt by location only")
        if not hit:
            continue
        before_end = job_end is None or it.get("t_mono") is None or float(it["t_mono"]) <= job_end + 1.0
        res = next((r for r in results if r.get("actor") == it.get("actor") and (r.get("target") or {}).get("pid") == tgt.get("pid")
                    and (r.get("target") or {}).get("pgid") == tgt.get("pgid") and float(r.get("t_mono", 0)) >= float(it.get("t_mono", 0))), None)
        delivered = bool(res and res.get("delivered"))
        sig_match = term_sig is not None and it.get("signal") == term_sig
        if before_end and (delivered or sig_match):
            matched_intent = {"intent": it, "result": res, "delivered": delivered, "signal_matches_exit": sig_match}
            break
        if before_end:
            incidents.append({"kind": "cancel_intent_unconfirmed", "actor": it.get("actor"), "signal": it.get("signal"), "reason": it.get("reason"),
                              "note": "intent recorded but neither delivery nor a matching exit signal was observed"})

    # ---- checker record -------------------------------------------------------------------------------------------------------------------
    check = None
    if check_record and Path(check_record).exists():
        try:
            doc = json.loads(Path(check_record).read_text())
            check = {"path": check_record, "verdict": doc.get("verdict"), "comparator": doc.get("comparator") or doc.get("schema")}
            inputs[Path(check_record).name] = {"sha256": _sha(Path(check_record))[0], "bytes": _sha(Path(check_record))[1]}
        except ValueError:
            limitations.append(f"check record {check_record} is not JSON")

    # ---- OOM attribution (rule 6 body; also consulted by rule 5 when a marker and an increment compete) --------------------------------------
    def oom_attribution() -> dict[str, Any]:
        """-> {class, why, certainty, limitations, detail}; class is CGROUP_OOM_ATTRIBUTED or OOM_UNRESOLVED_ORIGIN."""
        if own_scope:
            return {"class": "CGROUP_OOM_ATTRIBUTED", "why": "SIGKILL exit and an oom_kill increment in the job's own cgroup during the job",
                    "certainty": "established", "limitations": [], "detail": {"scope": cg_scope}}
        detail: dict[str, Any] = {}
        for w in delta_windows:
            if not w["counter"].endswith("oom_kill") and not w["counter"].endswith("oom_group_kill"):
                continue
            v = vanish_in(w)
            detail = {"window": w, **v}
            tree_vanished = bool(v["tree_gone"])
            no_outsider_vanished = not v["outsiders_gone"]
            hi, lg = v["highest_oom_score_before"], v["largest_before"]
            gone_set = set(v["tree_gone"])
            score_agrees = (bool(hi) and hi["membership"] == "tree" and hi["pid"] in gone_set and not hi["tied"]) if v["oom_score_sampled"] else None
            rss_agrees = bool(lg) and lg["membership"] == "tree" and lg["pid"] in gone_set
            detail["corroboration"] = {"highest_oom_score_is_the_vanished_tree_process": score_agrees, "largest_anon_shmem_is_the_vanished_tree_process": rss_agrees,
                                       "note": "corroboration only (rev c2): recorded, never a veto on the primary disappearance evidence"}
            if tree_vanished and no_outsider_vanished:
                lim = ["shared cgroup: attribution by disappearance window (a tree process vanished, no outsider vanished), not by kernel victim identity (dmesg unavailable)",
                       "processes that appeared and vanished between two samples are unobservable"]
                if score_agrees is False:
                    lim.append("corroboration disagrees: the highest sampled oom_score at the preceding sample was not the vanished tree process "
                               f"({hi['membership']}/{hi['pid']} {hi['comm']} score {hi['oom_score']})" if hi else "corroboration: oom_score ranking unavailable")
                if not rss_agrees and lg:
                    lim.append(f"corroboration: the largest anon+shmem at the preceding sample was {lg['membership']}/{lg['pid']} {lg['comm']} ({lg['anon_shmem_bytes']} B), not the vanished process")
                if v["attribution_sample_age_s"] is not None:
                    lim.append(f"attribution_sample_age_s={v['attribution_sample_age_s']}: the last process sample predates the kill by this much (cadence + gaps)")
                return {"class": "CGROUP_OOM_ATTRIBUTED", "certainty": "inferred", "limitations": lim, "detail": detail,
                        "why": ("SIGKILL exit; oom_kill increment in the SHARED pod cgroup; a process of this job's tree vanished in the same sample window "
                                "and no other process of the cgroup vanished" + (" (oom_score and RSS ranking corroborate)" if score_agrees and rss_agrees else ""))}
        return {"class": "OOM_UNRESOLVED_ORIGIN", "certainty": "ambiguous", "detail": detail,
                "why": "SIGKILL exit and an oom_kill increment in the shared pod cgroup, but the victim could not be attributed to this job's tree",
                "limitations": ["shared cgroup OOM: victim identity unresolved (an outsider vanished in the same window, or no tree process did)"]}

    def oom_window_overlaps(t: float | None, slack: float = 1.0) -> bool:
        if t is None:
            return False
        return any(float(w["window_start_mono"]) - slack <= t <= float(w["window_end_mono"]) + slack for w in delta_windows if "oom_kill" in w["counter"])

    # ---- rules -------------------------------------------------------------------------------------------------------------------------------
    cls, certainty, why = None, "established", ""
    competing: list[dict[str, Any]] = []
    open_root = any(s["outcome"] == "interrupted" and s["parent"] not in {x["id"] for x in sp["spans"]} for s in sp["spans"])
    if check and check["verdict"] not in (None, "PASS"):
        cls, why = "WORKLOAD_CHECK_FAILED", f"checker record verdict {check['verdict']!r}"
        ref("check_record", **check)
    elif term_rc == 0 and term_sig in (None, 0):
        cls, why = "SUCCESS", "exit status 0"
        if open_root:
            limitations.append("exit 0 but a root span was never closed: the runner did not emit span_end (evidence, not a failure)")
    elif cuda_hit:
        cls, why = "CUDA_ALLOC_FAIL", "CUDA allocation failure in the job's log"
        ref("log.cuda", **cuda_hit)
    elif enospc_hit or (storage_zero and term_rc not in (0, None)):
        cls, why = "STORAGE_EXHAUSTED", "ENOSPC/EDQUOT in the log" if enospc_hit else "a declared mount sampled below 16 MiB free while the job failed"
        if enospc_hit:
            ref("log.enospc", **enospc_hit)
        if storage_zero:
            ref("storage.sampled", samples=storage_zero[:5])
    elif matched_intent:
        actor = matched_intent["intent"].get("actor")
        intent_cls = ACTOR_CLASS.get(actor, "CANCELLED_MANUAL")
        intent_why = f"cancel_intent by {actor} ({matched_intent['intent'].get('reason')}) " + ("delivered" if matched_intent["delivered"] else "and the exit signal matches the request")
        intent_cert = "established" if matched_intent["delivered"] else "inferred"
        ref("cancel_intent", actor=actor, reason=matched_intent["intent"].get("reason"), signal=matched_intent["intent"].get("signal"),
            requested_utc=matched_intent["intent"].get("requested_utc"), target=matched_intent["intent"].get("target"), verified=matched_intent["intent"].get("verified"))
        if matched_intent["result"]:
            ref("cancel_result", delivered=matched_intent["result"].get("delivered"), errno=matched_intent["result"].get("errno"),
                target_gone_within_s=matched_intent["result"].get("target_gone_within_s"))
        if not matched_intent["delivered"]:
            limitations.append("delivery result not recorded; the marker is correlated with the exit signal, which is not proof it caused the exit")
        cls, why, certainty = intent_cls, intent_why, intent_cert
        verified = bool(matched_intent["intent"].get("verified"))
        marker_established = matched_intent["delivered"] and verified
        if term_sig == 9 and oom_delta and not marker_established:
            # rev c1: a SIGKILL marker that is not both delivered and identity-verified, beside an oom_kill delta: retain both, certainty ambiguous
            oa = oom_attribution()
            ref("oom.attribution", **oa["detail"])
            certainty = "ambiguous"
            competing = [{"class": intent_cls, "why": intent_why, "certainty": intent_cert, "evidence": "cancel_intent/cancel_result"},
                         {"class": oa["class"], "why": oa["why"], "certainty": oa["certainty"], "evidence": "cgroup.memory.events oom_kill delta + process disappearance"}]
            limitations.append(f"coexistence: a {actor} SIGKILL marker (delivered={matched_intent['delivered']}, verified={verified}) and an oom_kill increment both fall in the "
                               "job's interval; class = first match, both evidence sets retained under `competing`, certainty ambiguous")
        elif term_sig == 9 and oom_delta:
            limitations.append("an oom_kill incident coexists in the scope; the marker is delivered and identity-verified, so the kill is observed (incident retained, not competing)")
    elif term_sig == 9 and oom_delta:
        oa = oom_attribution()
        cls, why, certainty = oa["class"], oa["why"], oa["certainty"]
        limitations += oa["limitations"]
        ref("oom.attribution", **oa["detail"])
    elif term_sig == 15 and queue_scope and any(k.endswith("oom_kill") or k.endswith("oom_group_kill") for k in oom_delta):
        # rule 6b (0.1.4): systemd stops a scope (OOMPolicy=stop) with SIGTERM once the kernel OOM-kills a process in it
        cls, certainty = "CGROUP_OOM_ATTRIBUTED", "established"
        why = (f"exit by SIGTERM after an oom_kill increment in the run's own queue scope {cg_path.rsplit('/', 1)[-1]}: the kernel OOM-killed a process "
               "at the scope's memory cap and systemd then stopped the scope")
        limitations.append("the recorded signal is systemd's stop of the scope, not the OOM-killed process's SIGKILL")
    elif term_sig is not None:
        timeout_s = status.get("timeout_s") or (status.get("execution") or {}).get("timeout_s")
        ran = (job_end - job_start) if (job_start is not None and job_end is not None) else None
        if timeout_s and ran is not None and ran >= float(timeout_s) - 1.0 and term_sig in (9, 15) and status.get("timed_out"):
            cls, why, certainty = "STAGE_TIMEOUT", f"runner recorded timed_out after {ran:.1f}s >= timeout_s {timeout_s} with exit by {SIGNAMES.get(term_sig)}; no marker found", "inferred"
        else:
            cls, why = "UNKNOWN_SIGNAL", f"exit by signal {term_sig} ({SIGNAMES.get(term_sig, '?')}) with no attributable cause"
            certainty = "ambiguous" if incidents else "established"
            if oom_delta:
                limitations.append("an oom_kill incident coexists in the shared scope but is not attributed to this job (listed under incidents)")
    elif term_rc is not None and killed_hit and (oom_delta or (not kill_counter_seen and (bool(failcnt_delta) or under_oom_seen))):
        # rule 7b (0.1.3): the workload's wrapper exited rc != 0 and reported a SIGKILLed descendant while the cgroup counted a kill
        # (or, cgroup v1 with no oom_kill counter at all, hit its limit: failcnt grew / under_oom was raised)
        shared = not own_scope
        gone = [{"window": {k: w[k] for k in ("counter", "from", "to", "window_start_mono", "window_end_mono")}, **vanish_in(w)}
                for w in delta_windows if w["counter"].endswith("oom_kill") or w["counter"].endswith("oom_group_kill")]
        tree_gone = sorted({p for g in gone for p in g["tree_gone"]})
        outsiders_gone = sorted({(o.get("pid") if isinstance(o, dict) else o) for g in gone for o in g["outsiders_gone"]}, key=str)
        cls = "CGROUP_OOM"
        where = "shared pod" if shared else "job"
        if oom_delta:
            certainty = "inferred" if shared else "established"
            why = (f"exit status {term_rc} (no signal) with a Killed line in the job's log and an oom_kill increment in the {where} cgroup "
                   f"during the job: a descendant of the job was OOM-killed" + (f" (tree pid(s) {tree_gone} vanished in the kill window)" if tree_gone else ""))
        else:
            certainty = "ambiguous" if shared else "inferred"
            hit = " and ".join(x for x in (f"memory.failcnt grew by {failcnt_delta}" if failcnt_delta else "", "under_oom was raised" if under_oom_seen else "") if x)
            why = (f"exit status {term_rc} (no signal) with a Killed line in the job's log while the {where} cgroup (v1, no oom_kill counter) hit "
                   f"its limit ({hit}): a descendant of the job was OOM-killed")
            limitations.append("cgroup v1 without an oom_kill field in memory.oom_control (kernel < 4.13): no kill was counted; memory.failcnt / under_oom "
                               "show the limit was hit, which reclaim alone can also cause")
        ref("oom.attribution", scope=cg_scope, windows=gone, tree_gone=tree_gone, outsiders_gone=outsiders_gone, v1_failcnt_delta=failcnt_delta,
            v1_under_oom_seen=under_oom_seen, kill_counter=kill_counter_seen,
            note="rule 7b: the victim is named by the wrapper's Killed line and the disappearance window, not by the kernel")
        limitations.append("the recorded exit status is the wrapper's (bash/timeout/subprocess), not the OOM-killed descendant's signal")
        if shared:
            limitations.append("shared cgroup: the cgroup's counters cover every process of the pod; the Killed line ties the kill to this job's tree (dmesg unavailable)")
            if outsiders_gone:
                certainty = "ambiguous"
                limitations.append(f"an outsider ({outsiders_gone}) also vanished in the kill window: the counted kill may be theirs")
    elif term_rc is not None:
        cls, why = "UNKNOWN_EXIT", f"exit status {term_rc} with no structured checker record"
        if oom_delta:
            limitations.append("an oom_kill incident coexists in the scope without a Killed line in the job's log: not attributed (listed under incidents; rev d4)")
    else:
        cls, why, certainty = "UNKNOWN_EXIT", "no termination status recorded", "ambiguous"

    # ---- coexistence post-pass (rev c1): CUDA string / low mount chosen while the job died by SIGKILL with an oom_kill delta ------------------------
    if cls in ("CUDA_ALLOC_FAIL", "STORAGE_EXHAUSTED") and term_sig == 9 and oom_delta:
        oa = oom_attribution()
        ref("oom.attribution", **oa["detail"])
        competing = [{"class": cls, "why": why, "certainty": certainty, "evidence": "log string" if cls == "CUDA_ALLOC_FAIL" else "log ENOSPC / sampled mount"},
                     {"class": oa["class"], "why": oa["why"], "certainty": oa["certainty"], "evidence": "cgroup.memory.events oom_kill delta + process disappearance"}]
        certainty = "ambiguous"
        limitations.append(f"coexistence: {cls} evidence and an oom_kill increment with a SIGKILL exit are both present; class = first match, both retained under `competing`, "
                           "certainty ambiguous (a stale log string or a low mount does not exclude a kernel kill)")
    if cls == "UNKNOWN_SIGNAL" and not samples:
        certainty = "ambiguous"
        limitations.append("no resource samples: a cgroup OOM in this interval could not have been observed")
    if cap and cls in ("CGROUP_OOM_ATTRIBUTED", "CGROUP_OOM") and own_scope:
        why += f"; the cap was {cap}" + (" (`research run --queue --mem-gb G` sets it)" if queue_scope else "")

    out = {"schema": FAILURE_SCHEMA, "classifier_version": CLASSIFIER_VERSION, "attempt_dir": str(d), "inputs": inputs,
           "termination": termination, "class": cls, "certainty": certainty, "why": why, "competing": competing, "location": location,
           "incidents": incidents, "evidence": evidence, "limitations": limitations, "telemetry_partial": telemetry_partial,
           "spans_summary": {"total_union_s": sp["total_union_s"], "overlap_s": sp["overlap_s"], "open": len(open_spans), "by_stage": sp["by_stage"]},
           "note": "an interpretation of preserved evidence, not the project's definition of blocked; incidents are separate from the outcome"}
    return out


def write(attempt_dir: str | Path, doc: dict[str, Any]) -> Path:
    d = Path(attempt_dir)
    p = d / "failure.json"
    write_json_atomic(p, doc)
    versioned = d / f"failure.{CLASSIFIER_VERSION}.json"
    if not versioned.exists():
        write_json_atomic(versioned, doc)
    return p


def add_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--attempt-dir", required=True)
    ap.add_argument("--rc", type=int, default=None)
    ap.add_argument("--signal", type=int, default=None)
    ap.add_argument("--log", action="append", default=[])
    ap.add_argument("--check-record", default=None, help="structured checker document (e.g. global_match_fast.json) whose verdict is the workload result")
    ap.add_argument("--stages-txt", default=None)
    ap.add_argument("--print", action="store_true")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tele classify", description=__doc__.splitlines()[0])
    add_args(ap)
    a = ap.parse_args(argv)
    doc = classify(a.attempt_dir, a.rc, a.signal, a.log, a.check_record, a.stages_txt)
    p = write(a.attempt_dir, doc)
    if a.print:
        print(json.dumps(doc, indent=1))
    else:
        print(f"{p}: {doc['class']} ({doc['certainty']}) -- {doc['why']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
