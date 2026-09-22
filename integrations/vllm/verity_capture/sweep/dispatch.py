"""sweep/dispatch.py — R13 report owner: dispatch established row procedures to the completion pool and write records per stage.

    python -m verity_capture.sweep.dispatch --once            # one poll: launch queued rows on idle pods, ingest stage verdicts
    python -m verity_capture.sweep.dispatch --daemon [--interval 60]

Pool:   /tmp/r13-pool.txt      lines `<pod> <ssh-wrapper> [<max-parallel=1>]`   (default cp-sw1, cp-sw2)
Queue:  /tmp/r13-queue.txt     lines `<row-id> [stages=build,match,commit] [K=V ...]` (env for row_pod.sh; `#` comments); consumed top-down
State:  /tmp/r13-dispatch-state.json ; status summary: /tmp/r13-dispatch-status.txt (read by the report loop)
Gate:   the pod's /workspace/cp/ready/ready.json must name the release tag being cited (`--tag`, default: newest RELEASED line in
        out/gen/r13/RELEASES.md; with `--candidate <tag>` a not-yet-announced tag whose commit matches ready.json is allowed and the
        attempt is marked `release_tag_pending`).  One GPU job per pod.  Records: a new `attempts[]` entry per launch, stage dicts
        written the moment the pod's stages.txt shows the verdict; failure records F-report-NN on a FAIL.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SWEEP = REPO / "out" / "gen" / "sweep"
ROWS, FAILS, EVID = SWEEP / "rows", SWEEP / "failures", SWEEP / "evidence" / "report"
POOL_FILE, QUEUE_FILE = "/tmp/r13-pool.txt", "/tmp/r13-queue.txt"
STATE_FILE, STATUS_FILE = "/tmp/r13-dispatch-state.json", "/tmp/r13-dispatch-status.txt"
REMOTE_SWEEP = "/workspace/cp/sweep"
DEFAULT_POOL = {"cp-sw1": str(REPO / "out/capture/pods/cp-sw1.ssh"), "cp-sw2": str(REPO / "out/capture/pods/cp-sw2.ssh")}
LANE = "report"
FOLLOWUPS: list[dict] = []  # commit-only follow-ups queued by ingest (same pod: the Build/Match artifacts live there)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(msg: str) -> None:
    print(f"{now()} {msg}", flush=True)


def ssh(wrapper: str, cmd: str, timeout: int = 120) -> tuple[int, str]:
    try:
        r = subprocess.run(["bash", wrapper, cmd], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return 124, "ssh timeout"
    out = "\n".join(l for l in (r.stdout + r.stderr).splitlines() if "Permanently added" not in l)
    return r.returncode, out


def load_json(p, default):
    try:
        with open(p, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return default


def save_json(p, obj) -> None:
    tmp = f"{p}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, sort_keys=False, default=str)
    os.replace(tmp, p)


def pool() -> dict[str, str]:
    if not os.path.exists(POOL_FILE):
        return dict(DEFAULT_POOL)
    out = {}
    for line in open(POOL_FILE, encoding="utf-8"):
        parts = line.split()
        if parts and not parts[0].startswith("#"):
            out[parts[0]] = parts[1]
    return out


def queue() -> list[dict]:
    if not os.path.exists(QUEUE_FILE):
        return []
    items = []
    for line in open(QUEUE_FILE, encoding="utf-8"):
        parts = line.split()
        if not parts or parts[0].startswith("#"):
            continue
        it = {"row": parts[0], "stages": "build,match,commit", "env": {}, "pod": None, "run": ""}
        for p in parts[1:]:
            if p.startswith("#"):
                break
            if p.startswith("stages="):
                it["stages"] = p[7:]
            elif p.startswith("pod="):
                it["pod"] = p[4:]
            elif p.startswith("run="):
                it["run"] = p[4:]  # distinguishes a deliberate re-run of the same row+stages (e.g. on a newer release tag)
            elif p.startswith("min="):
                it["min"] = p[4:]  # earliest release tag the row may run on (its blocker set is covered from this tag onwards)
            elif p.startswith("role="):
                it["role"] = p[5:]  # "reproduction": second-pod rerun for the per-pod spread; never the row's headline attempt
            elif "=" in p:
                k, v = p.split("=", 1)
                it["env"][k] = v
        it["key"] = it["row"] + "|" + it["stages"] + ("|" + it["run"] if it["run"] else "")
        items.append(it)
    return items


def releases() -> list[tuple[str, str]]:
    """[(tag, fixes)] newest first from out/gen/r13/RELEASES.md."""
    p = REPO / "out/gen/r13/RELEASES.md"
    out = []
    if p.exists():
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.match(r"^\W*RELEASED\s+(r13-rel\d+)\s+(\S*)", line.strip())
            if m:
                out.append((m.group(1), m.group(2)))
    # every earlier release stays announced (its `### r13-relN` section remains); rel keeps only the newest on the top line
    try:
        from verity_capture.sweep import deps as _deps
        seen = {t for t, _ in out}
        for r in _deps.parse_releases(str(REPO / "out/gen/sweep")):
            if r["tag"] not in seen:
                out.append((r["tag"], ",".join(r.get("fixes") or [])))
                seen.add(r["tag"])
    except Exception:
        pass
    return out


def tag_commit_of(tag: str) -> str | None:
    """Commit of an announced release tag: git tag, else a hex sha on the RELEASED line, else the release lane's RELEASE.json note."""
    r = subprocess.run(["git", "-C", str(REPO), "rev-parse", "-q", "--verify", f"{tag}^{{commit}}"], capture_output=True, text=True)
    if r.returncode == 0 and r.stdout.strip():
        return r.stdout.strip()
    p = REPO / "out/gen/r13/RELEASES.md"
    if p.exists():
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            if f"RELEASED {tag}" in line:
                m = re.search(r"\b([0-9a-f]{9,40})\b", line.split(tag, 1)[1])
                if m:
                    return m.group(1)
    return None


def model_args(row_id: str) -> tuple[str, str, str]:
    """(ROLE, hf repo, revision) from the row's workload manifest + checkpoints.json."""
    wl = load_json(SWEEP / "workloads" / f"{row_id}.json", None)
    if not wl:
        raise RuntimeError(f"no workload manifest for {row_id}")
    repo = wl["model"]
    ck = load_json(REPO / "manifests/checkpoints.json", {"checkpoints": []})
    role_pref = {"HuggingFaceTB/SmolLM2-135M": "B0", "unsloth/Llama-3.2-1B": "LLAMA32_1B", "Qwen/Qwen2.5-1.5B": "B1",
                 "microsoft/Phi-3-mini-4k-instruct": "PHI3_MINI", "unsloth/gemma-2-2b": "GEMMA2_2B", "unsloth/mistral-7b-v0.3": "MISTRAL7B",
                 "allenai/OLMoE-1B-7B-0924": "OLMOE", "Qwen/Qwen3-4B-Instruct-2507": "QWEN3_4B", "Qwen/Qwen3-4B-Instruct-2507-FP8": "QWEN3_4B_FP8",
                 "Qwen/Qwen3-30B-A3B": "QWEN3_30B_A3B"}
    rev = None
    for c in ck["checkpoints"]:
        if c.get("repo") == repo and str(c.get("revision", "")).startswith(str(wl.get("tokenizer_revision", ""))[:10]):
            rev = c["revision"]
            break
    rev = rev or wl.get("tokenizer_revision")
    return role_pref.get(repo, wl.get("case") or "?"), repo, rev


def pod_ready(wrapper: str) -> dict:
    rc, out = ssh(wrapper, "cat /workspace/cp/ready/ready.json 2>/dev/null; echo; cat /workspace/cp/RELEASE.json 2>/dev/null", 60)
    ready, release = {}, {}
    if rc == 0:
        try:
            i = out.index("}") + 1
            ready = json.loads(out[:i])
            release = json.loads(out[i:].strip() or "{}")
        except Exception:
            pass
    return {"ready": ready, "release": release}


def pod_busy(wrapper: str) -> tuple[bool, str]:
    # REL-07: the release owner touches /workspace/cp/RELEASING while re-shipping a pod; never launch on it then
    rc, out = ssh(wrapper, "pgrep -fc 'sweep/row_pod[.]sh' 2>/dev/null; nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l; "
                           "[ -e /workspace/cp/RELEASING ] && echo RELEASING || echo free", 60)
    if rc not in (0, 1):
        return True, f"ssh rc={rc}"
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    nums = [int(l) for l in lines if l.isdigit()]
    rowpods = nums[0] if len(nums) >= 2 else 0
    procs = nums[-1] if nums else 0
    releasing = any(l == "RELEASING" for l in lines)
    return rowpods > 0 or procs > 0 or releasing, f"row_pod={rowpods} gpu_procs={procs}" + (" RELEASING (REL-07)" if releasing else "")


# ---------------------------------------------------------------------------------------------------------------- records


def read_row(row_id: str) -> dict:
    p = ROWS / f"{row_id}.json"
    row = load_json(p, None)
    if row is None:
        # new configuration: init via record.py (fills config/workload_actual from the manifest)
        subprocess.run([sys.executable, "-m", "verity_capture.sweep.record", "row", "--init", "--row-id", row_id, "--lane", LANE, "--pod", "pool",
                        "--tier", "2", "--workload", str(SWEEP / "workloads" / f"{row_id}.json"), "--set", "status=QUEUED"], cwd=REPO, check=False)
        row = load_json(p, {"schema": "verity-sweep/row/v1", "row_id": row_id, "lane": LANE, "status": "QUEUED"})
    return row


def write_row(row: dict) -> None:
    row["updated_at"] = now()
    save_json(ROWS / f"{row['row_id']}.json", row)


def new_attempt(row: dict, pod: str, tag: str, tag_pending: bool, revision: str, stages: str, env: dict, cmd: str) -> int:
    row.setdefault("attempts", [])
    _stage_dicts = {st: row[st] for st in ("build", "match", "commit", "bench") if isinstance(row.get(st), dict) and row[st].get("status") not in (None, "-", "")}
    _informative = any(isinstance(a, dict) and any(isinstance(a.get(st), dict) for st in ("build", "match", "commit", "bench")) for a in row["attempts"])
    if _stage_dicts and not _informative and row.get("lane") not in (None, LANE):
        # the owning lane recorded its verdicts at the top level only: snapshot them as an attempt so a pool rerun never erases them
        snap = {"campaign": row.get("campaign", "r13"), "lane": row.get("lane"), "pod": row.get("pod"), "release_tag": row.get("release_tag"),
                "revision": row.get("revision"), "at": row.get("updated_at") or now(), "status": row.get("status"),
                "notes": "snapshot of the lane's top-level stage verdicts taken by the dispatcher before a pool attempt", **_stage_dicts}
        row["attempts"].append(snap)
    att = {"campaign": "r13", "lane": LANE, "pod": pod, "release_tag": tag, "revision": revision, "at": now(), "stages_requested": stages,
           "env": env, "cmd": cmd, "status": "RUNNING", "evidence_root": f"pod:{pod}:{REMOTE_SWEEP}/{row['row_id']}/"}
    if tag_pending:
        att["release_tag_pending"] = True
    req = [x.strip() for x in stages.split(",") if x.strip()]
    if req and "build" not in req:
        # a stage-subset run continues the previous attempt on the same pod directory (row_pod.sh reuses its Build/Match
        # artefacts and stages.txt): carry the earlier stage verdicts into this attempt, labelled with their own tag/revision,
        # so the row shows Build/Match/Commit together rather than "not reached in latest attempt"
        for prev in reversed(row["attempts"]):
            if prev is att or prev.get("status") == "RUNNING":
                continue
            carried = False
            for stg in ("build", "match", "commit"):
                if stg in req:
                    continue
                d = prev.get(stg)
                if isinstance(d, dict) and d.get("status") in ("PASS", "DONE"):
                    dd = dict(d)
                    dd.setdefault("release_tag", prev.get("release_tag"))
                    dd.setdefault("revision", prev.get("revision"))
                    dd["carried_from_attempt"] = f"#{row['attempts'].index(prev) + 1} ({prev.get('release_tag') or prev.get('revision')}; same pod dir / program digest)"
                    att[stg] = dd
                    carried = True
            if carried:
                break
    row["attempts"].append(att)
    row["campaign"] = "r13"
    row["last_attempt_by"] = LANE
    row["rerun_in_progress"] = {"pod": pod, "release_tag": tag, "revision": revision, "at": att["at"]}
    return len(row["attempts"]) - 1


FREEZE_UTC = "2026-09-19T12:57:00Z"


def next_failure_id() -> str:
    n = 0
    for p in FAILS.glob("F-report-*.json"):
        m = re.search(r"F-report-(\d+)", p.name)
        if m:
            n = max(n, int(m.group(1)))
    return f"F-report-{n + 1:02d}"


def first_error(log_txt: str) -> str:
    lines = [l.strip() for l in log_txt.splitlines() if l.strip()]
    pat = re.compile(r"(Error|error:|refus|OOM|out of memory|FAIL|Traceback|TIMEOUT|unsupported|not compiled|missing)", re.I)
    for l in reversed(lines):
        if pat.search(l) and "Warning" not in l:
            return l[:400]
    return (lines[-1] if lines else "")[:400]


def write_failure(row: dict, att: dict, stage: str, cls: str, cmd: str, err: str, artifact: str) -> str:
    fid = next_failure_id()
    rec = {"schema": "verity-sweep/failure/v1", "failure_id": fid, "row_id": row["row_id"], "campaign": "r13", "lane": LANE, "pod": att["pod"],
           "stage": stage, "class": cls, "cmd": cmd, "first_error": err, "artifact": artifact, "release_tag": att.get("release_tag"),
           "revision": att.get("revision"), "recorded_at": now(),
           "notes": "pool rerun by the report owner via row_pod.sh; classify/link `same_as` and `cluster` when reviewing"}
    save_json(FAILS / f"{fid}.json", rec)
    return fid


def classify(stage: str, err: str) -> str:
    e = err.lower()
    if "out of memory" in e or "oom" in e or "arena" in e and "refus" in e:
        return "collector footprint / memory" if stage == "commit" else "memory"
    if "unsupported" in e or "no rule" in e or "refus" in e:
        return "unsupported op / applicability refusal" if stage == "build" else "unresolved kernel / no rule"
    if "timeout" in e:
        return "timeout"
    if "not compiled" in e or "hdim" in e or "head_dim" in e:
        return "attention tap specialization missing"
    return "unclassified (see log)"


ENV_FIX_IDS = {"X-00", "X-03"}


def fill_fix_citations(row: dict, att: dict) -> None:
    """R-19 for pool attempts: `code_changes_required` = Verity-mechanism fixes in the cited release that resolve one of this row's
    blockers (from DEPS.json) plus the deployment-facts fixes every fresh-pod row depends on (X-01 SM count, X-02 arena bound);
    `infra_fixes` = environment/harness/release fixes in the release (X-00, X-03, REL-*).  Only fixes whose commit is in the tag."""
    deps = load_json(SWEEP / "DEPS.json", {}) or {}
    drow = (deps.get("rows") or {}).get(row["row_id"], {})
    blockers = set(drow.get("blockers") or []) | set(drow.get("seed") or [])
    fclust = deps.get("fix_clusters") or {}
    tag = att["release_tag"]
    try:
        from verity_capture.sweep import deps as D, validate as R
        recs = R.load(str(SWEEP))
        in_tag = D.fixes_in_tag(tag, recs, str(REPO))
    except Exception:  # noqa: BLE001
        in_tag = set()
    if not in_tag:
        return
    infra = sorted(x for x in in_tag if x.startswith("REL-") or x in ENV_FIX_IDS)
    ccr = {x for x, cls in fclust.items() if x in in_tag and x not in infra and set(cls) & blockers}
    ccr |= {x for x in ("X-01", "X-02") if x in in_tag}
    att["code_changes_required"] = sorted(ccr)
    att["infra_fixes"] = infra
    att["fix_citation_note"] = f"filled by the dispatcher from DEPS.json blockers {sorted(blockers)} and the fixes contained in {tag}"
    row["code_changes_required"] = att["code_changes_required"]
    row["infra_fixes"] = infra


# ---------------------------------------------------------------------------------------------------------------- ingest


def ingest(pod: str, wrapper: str, st: dict) -> bool:
    """Pull stages.txt + summaries for the running job on `pod`; write stage dicts as they appear.  Returns True when finished."""
    row_id = st["row"]
    d = f"{REMOTE_SWEEP}/{row_id}"
    rc, out = ssh(wrapper, f"cd {d} 2>/dev/null || exit 3; echo '=== stages'; cat stages.txt 2>/dev/null; "
                           f"for f in build_summary.json match_summary.json commit_summary.json commit/summary.json commit/verdict.json; do echo; echo \"=== $f\"; cat $f 2>/dev/null; done; "
                           f"echo; echo '=== commit/runs.jsonl'; grep '\"arm\": \"instrumented\"' commit/runs.jsonl 2>/dev/null | tail -3; "
                           f"echo; echo '=== commit_cmd'; head -1 commit.log 2>/dev/null; echo; echo '=== match_cmd'; head -1 match.log 2>/dev/null; "
                           f"echo; echo '=== build_cmd'; head -1 build_request.log 2>/dev/null; echo; echo '=== rowlog'; tail -5 row.log 2>/dev/null", 90)
    if rc != 0:
        log(f"[{pod}] ingest ssh rc={rc}: {out[:200]}")
        return False
    parts = re.split(r"^=== (.+)$", out, flags=re.M)
    sec = {parts[i].strip(): parts[i + 1] for i in range(1, len(parts) - 1, 2)}
    stages = [l.split() for l in sec.get("stages", "").splitlines() if l.strip()]
    stages = [l for l in stages if len(l) >= 3]
    row = read_row(row_id)
    atts = row.setdefault("attempts", [])
    att = atts[st["attempt"]] if isinstance(st.get("attempt"), int) and 0 <= st["attempt"] < len(atts) else None
    if not (isinstance(att, dict) and att.get("lane") == LANE and att.get("pod") == pod and att.get("at") == st.get("started")):
        # another lane re-wrote the record (or re-ordered attempts) while our job ran: find our attempt by (lane, pod, start)
        # or re-create it from the dispatcher state so the verdicts are never lost
        found = [i for i, a in enumerate(atts) if isinstance(a, dict) and a.get("lane") == LANE and a.get("pod") == pod and a.get("at") == st.get("started")]
        if found:
            st["attempt"] = found[0]
        else:
            atts.append({"campaign": "r13", "lane": LANE, "pod": pod, "release_tag": st.get("tag"), "revision": st.get("revision"),
                         "at": st.get("started") or now(), "stages_requested": st.get("stages", "build,match,commit"), "status": "RUNNING",
                         "evidence_root": f"pod:{pod}:{REMOTE_SWEEP}/{row_id}/",
                         "note": "attempt entry re-created by the dispatcher: the record was re-written by another lane while this job ran"})
            st["attempt"] = len(atts) - 1
        att = atts[st["attempt"]]
    changed = False
    done = False
    for parts_ in stages:
        stage, verdict, ts = parts_[0], parts_[1], parts_[2]
        note = " ".join(parts_[3:])
        if stage == "DONE":
            done = True
            continue
        if stage not in ("build", "match", "commit"):
            # REL-09 (rel8+): stages.txt opens with a `release <tag> <commit> ...` line; keep it as attempt evidence
            if stage == "release" and not att.get("release_line"):
                att["release_line"] = " ".join(parts_[1:])[:200]
                changed = True
            continue
        if stage in att and isinstance(att[stage], dict) and att[stage].get("status") not in (None, "RUNNING"):
            continue
        summ_name = {"build": "build_summary.json", "match": "match_summary.json", "commit": "commit_summary.json"}[stage]
        summ = {}
        try:
            summ = json.loads(sec.get(summ_name, "") or "{}")
        except Exception:
            pass
        if ts and str(ts) >= FREEZE_UTC:
            att["post_freeze"] = True  # COORD 12:00Z: runs in flight may finish and be recorded, tagged `post-freeze completion`
        ev_dir = EVID / row_id / f"a{st['attempt']}_{att['at'][11:16].replace(':', '')}"
        ev_dir.mkdir(parents=True, exist_ok=True)
        if summ:
            save_json(ev_dir / summ_name, summ)
        rel_ev = str(ev_dir.relative_to(REPO))
        art = f"pod:{pod}:{d}/{ {'build': 'build_request', 'match': 'match', 'commit': 'commit'}[stage]} (+ {rel_ev}/{summ_name})"
        sd = {"status": "PASS" if verdict == "PASS" else "FAIL", "at": ts, "verdict_line": note, "artifact": art, "revision": att["revision"]}
        if stage == "build":
            req = (summ.get("request") or {})
            sd.update({"program_digest": req.get("program_digest") or (summ.get("step") or {}).get("program_digest"),
                       "artifact_identity": req.get("identity"), "target_profile": req.get("target") or (summ.get("step") or {}).get("target"),
                       "applicability": req.get("applicability"),
                       "cmd": (sec.get("build_cmd", "").strip().lstrip("$ ") or f"derive_step via row_pod.sh: {att['cmd']}") + " (+ --wrapper step; REL-01 T=max_output-1)"})
            if not sd.get("target_profile"):
                sd["target"] = "device TargetProfile (X-01; see build_summary.json target)"
        elif stage == "match":
            if verdict != "PASS" and summ.get("compare_verdict") == "PASS" and summ.get("fold_ok") and summ.get("control_vs_instrumented_tokens_equal") is False:
                # row_pod.sh (r13-rel0) compares the whole tokens.json (incl. `timing`, `per_step_sampled`) — a false negative when the
                # request token ids are equal.  Verify the request-level ids ourselves and record the outcome either way.
                chk = ("import json; a=json.load(open('%s/match/control/tokens.json')); b=json.load(open('%s/match/capture/tokens.json')); "
                       "ra=a.get('requests', a); rb=b.get('requests', b); "
                       "ok = len(ra)==len(rb) and all(x.get('prompt_token_ids')==y.get('prompt_token_ids') and x.get('generated_token_ids', x.get('output_token_ids'))==y.get('generated_token_ids', y.get('output_token_ids')) for x,y in zip(ra,rb)); "
                       "print('REQ_TOKENS_EQUAL', ok, len(ra))") % (d, d)
                rc3, o3 = ssh(wrapper, f"python3 -c \"{chk}\"", 60)
                if "REQ_TOKENS_EQUAL True" in o3:
                    verdict = "PASS"
                    sd["status"] = "PASS"
                    sd["control_vs_instrumented_tokens_equal"] = True
                    sd["harness_note"] = ("row_pod.sh tok_eq compared whole tokens.json (timing/per_step_sampled) -> FAIL; request-level "
                                          "prompt/output token ids verified equal by the dispatcher (" + o3.strip().splitlines()[-1] + "); reported to rel")
                    log(f"[{pod}] {row_id} match: harness false-negative overridden after request-token check")
                    if "commit" in att.get("stages_requested", "").split(","):
                        FOLLOWUPS.append({"row": row_id, "stages": "commit", "env": dict(att.get("env") or {}), "pod": pod})
                else:
                    sd["harness_note"] = f"request-token check: {o3.strip()[-200:]}"
            sd.update({"comparator": "cb_a.compare (canonical instance sequence + operand wiring) over run_config capture+control+fold",
                       "verdict": summ.get("compare_verdict"), "control_vs_instrumented_tokens_equal": sd.get("control_vs_instrumented_tokens_equal", summ.get("control_vs_instrumented_tokens_equal")),
                       "steps": summ.get("steps"), "totals": summ.get("totals"), "fold_ok": summ.get("fold_ok"),
                       "cmd": (sec.get("match_cmd", "").strip().lstrip("$ ") or f"run_config via row_pod.sh: {att['cmd']}") + " ; then cb_a.compare --a build_request --b match"})
        elif stage == "commit":
            cs = {}
            try:
                cs = json.loads(sec.get("commit/summary.json", "") or "{}")
            except Exception:
                pass
            if cs:
                cs_small = {k: v for k, v in cs.items() if k not in ("table_md", "steady_hashing_by_step_s", "versions")}
                save_json(ev_dir / "commit_delta_summary.json", cs_small)
            vd = {}
            try:
                vd = json.loads(sec.get("commit/verdict.json", "") or "{}")
            except Exception:
                pass
            runs = []
            for ln in sec.get("commit/runs.jsonl", "").splitlines():
                try:
                    runs.append(json.loads(ln))
                except Exception:
                    pass
            if vd:
                save_json(ev_dir / "commit_verdict.json", vd)
            if runs:
                save_json(ev_dir / "commit_runs_instrumented.json", [{k: v for k, v in r.items() if k != "step_wall_s"} for r in runs])
            decl = cs.get("declaration") or {}
            classes = vd.get("required_classes") or [c.get("class") for c in decl.get("classes", []) if isinstance(c, dict)]
            tens = {c.get("class"): c.get("tensors") for c in decl.get("classes", []) if isinstance(c, dict)}
            rc_ = [r.get("commit") or {} for r in runs]
            op_n = sum((r.get("openings") or {}).get("n", 0) for r in rc_) or None
            op_ok = sum((r.get("openings") or {}).get("verified", 0) for r in rc_) or None
            roots = vd.get("run_roots") or [r.get("run_root") for r in rc_ if r.get("run_root")]
            exp_t = [r.get("expected_tensors") for r in rc_ if r.get("expected_tensors") is not None] or cs.get("expected_tensors")
            com_t = [r.get("committed_tensors") for r in rc_ if r.get("committed_tensors") is not None] or cs.get("committed_tensors")
            err_n = sum(int(r.get("errors_n") or 0) for r in rc_)
            sd.update({"committer": f"{cs.get('committer') or summ.get('committer') or 'native_collect_v2b'} + FA2 matReq hidden_m1 (fail-closed, --arena-mb 0)",
                       "layout": "native-host-sha256, pos_leaf chunk=256, per-step bind_root, run root",
                       "required_classes": classes, "bound_classes": [c for c in classes if (tens.get(c) or 0) > 0],
                       "class_tensors": tens, "hidden_tensors": tens.get("fa2_hidden_m1_stream"),
                       "openings_checked": op_n, "openings_ok": op_ok,
                       "tokens_equal": all(t is not False for t in summ.get("tokens_equal", [])) if summ.get("tokens_equal") else cs.get("tokens_equal"),
                       "backlog_max": cs.get("backlog_max"), "expected_tensors": exp_t, "committed_tensors": com_t,
                       "errors": (cs.get("commit_fail_reasons") or []) if err_n == 0 else [f"errors_n={err_n}"], "errors_n": err_n,
                       "fail_closed": True, "commit_pass": vd.get("commit_pass", cs.get("commit_pass")),
                       "runs_instrumented": vd.get("runs_instrumented", len(rc_)), "roots_equal_across_runs": (len(set(roots)) == 1) if roots else None,
                       "run_root": roots[0] if roots else None, "run_roots": roots,
                       "required_classes_source": vd.get("required_classes_source"),
                       "hidden_so": (re.search(r"--hidden-so\s+(\S+)", sec.get("commit_cmd", "")) or [None, None])[1],
                       "cmd": (sec.get("commit_cmd", "").strip().lstrip("$ ") or f"commit_delta via row_pod.sh: {att['cmd']}")})
            if verdict == "PASS" and cs:
                ctl, trt = summ.get("control_wall") or [], summ.get("treatment_wall") or []
                rel = cs.get("delta_wall_mean")
                if rel is None and ctl and trt:
                    rel = (sum(trt) / len(trt)) / (sum(ctl) / len(ctl)) - 1
                bench = {"status": "DONE", "rounds": cs.get("pairs") or len(ctl), "warmup": 2, "control_s": ctl, "treatment_s": trt,
                         "control_mean_s": (sum(ctl) / len(ctl)) if ctl else None, "treatment_mean_s": (sum(trt) / len(trt)) if trt else None,
                         "overhead_rel": rel, "overhead_abs_ms_per_step": (sum(cs["abs_wall_ms"]) / len(cs["abs_wall_ms"]) / max(1, int(row.get("config", {}).get("max_output", 1)))) if cs.get("abs_wall_ms") else None,
                         "delta_wall": cs.get("delta_wall"), "delta_wall_ci95": cs.get("delta_wall_ci95"), "profiled": False,
                         "artifact": art, "cmd": sd["cmd"], "interleaved": True}
                att["bench"] = bench
        att[stage] = sd
        att["status"] = {"build": "BUILD", "match": "MATCH", "commit": "COMMIT"}[stage] + ("_PASS" if verdict == "PASS" else "_FAIL")
        if stage == "commit" and verdict == "PASS" and att.get("bench"):
            att["status"] = "BENCHMARKED"
        if verdict != "PASS":
            rc2, logtail = ssh(wrapper, f"tail -60 {d}/{stage}.log 2>/dev/null; tail -20 {d}/{stage}_request.log 2>/dev/null", 60)
            err = first_error(logtail)
            (ev_dir / f"{stage}.log.tail").write_text(logtail, encoding="utf-8")
            cls, cluster, extra = classify(stage, err), None, {}
            if stage == "match" and isinstance(summ, dict) and summ:
                # structured verdict from match_summary.json beats the log tail
                batch = int((row.get("config") or {}).get("batch") or 1)
                tot, stp = summ.get("totals") or {}, summ.get("steps") or [None, None]
                if summ.get("compare_rc") == 124:
                    cls, cluster = "infrastructure/timeout (comparator)", "INFRA-TIMEOUT"
                    err = f"cb_a.compare rc=124: row_pod.sh hard-codes run_to 600 for the comparator (B={batch}); run_config rc={summ.get('run_config_rc')}, fold_ok={summ.get('fold_ok')}, tokens_equal={summ.get('control_vs_instrumented_tokens_equal')}"
                elif summ.get("compare_verdict") == "FAIL" and summ.get("canonical_equal") is False and batch >= 2:
                    cls, cluster = "configuration/applicability (batch): derived Program is single-request", "M-BATCH-DECOMP"
                    err = (f"cb_a.compare: canonical_equal=False; derived (single-request wrapper) steps={stp[0]} vs oracle steps={stp[1]}; "
                           f"operands_equal={tot.get('operands_equal')} operands_different={tot.get('operands_different')}; fold_ok={summ.get('fold_ok')}; B={batch} mixed batch (F-report-02)")
                    extra = {"same_as": "F-report-02"}
                elif summ.get("compare_verdict") == "PASS" and not summ.get("control_vs_instrumented_tokens_equal"):
                    cls, cluster = "infrastructure/harness (token compare)", "INFRA-HARNESS"
                    err = "cb_a.compare PASS but row_pod.sh tokens_equal=False (whole-document compare)"
                elif not summ.get("fold_ok"):
                    cls = "fold failed (see match.log)"
            fid = write_failure(row, att, stage, cls, sd.get("cmd", ""), err, art)
            if cluster or extra:
                fp = FAILS / f"{fid}.json"
                fr = load_json(fp, {})
                if cluster:
                    fr["cluster"] = cluster
                fr.update(extra)
                save_json(fp, fr)
            sd["failure_id"] = fid
            att["failure_id"] = fid
            log(f"[{pod}] {row_id} {stage} FAIL -> {fid}: {err[:120]}")
        else:
            log(f"[{pod}] {row_id} {stage} PASS {note[:100]}")
        changed = True
    if done:
        att["finished_at"] = now()
        if att["status"] == "RUNNING":
            att["status"] = "BUILD_FAIL" if "build" in att.get("stages_requested", "") else "NOT_RUN"
        changed = True
    if changed:
        # top-level mirrors the latest attempt
        for k in ("build", "match", "commit", "bench"):
            if isinstance(att.get(k), dict):
                row[k] = att[k]
        row["status"] = att["status"]
        row["release_tag"] = att.get("release_tag")
        row["revision"] = att.get("revision")
        if done:
            row.pop("rerun_in_progress", None)
        # a stage the new attempt did not (yet) reach must not keep the previous attempt's result
        reached = [k for k in ("build", "match", "commit", "bench") if isinstance(att.get(k), dict)]
        requested = att.get("stages_requested", "build,match,commit").split(",") + ["bench"]
        for k in ("build", "match", "commit", "bench"):
            if k not in reached and k in requested and k in row:
                row[k] = {"status": "-", "note": f"not reached in attempt {st['attempt']}"}
            if k not in requested and k in row and isinstance(row[k], dict) and row[k].get("revision") == att.get("revision"):
                att[k] = row[k]  # same revision: the follow-up attempt carries the earlier stage result
        if row.get("code_changes_required") is None:
            row["code_changes_required"] = []
        if done and att.get("release_tag"):
            fill_fix_citations(row, att)
        write_row(row)
    return done


# ---------------------------------------------------------------------------------------------------------------- launch


def launch(pod: str, wrapper: str, item: dict, tag: str, tag_pending: bool, revision: str) -> dict | None:
    row_id = item["row"]
    try:
        role, repo, rev = model_args(row_id)
    except Exception as e:  # noqa: BLE001
        log(f"[{pod}] cannot launch {row_id}: {e}")
        return None
    env = " ".join(f"{k}={v}" for k, v in item["env"].items())
    d = f"{REMOTE_SWEEP}/{row_id}"
    sname = f"r13row_{abs(hash(row_id)) % 100000}"
    # the release tree on the pod only carries the manifests that existed at tag time; ship this row's manifest (deterministic
    # from the row id, BRIEF §12) if the pod lacks it or has a different one
    try:
        local = (SWEEP / "workloads" / f"{row_id}.json").read_text(encoding="utf-8")
        remote_path = f"/workspace/cp/veritor/workloads/{row_id}.json"
        import hashlib
        h = hashlib.sha256(local.encode()).hexdigest()
        rc0, out0 = ssh(wrapper, f"sha256sum {remote_path} 2>/dev/null | cut -d' ' -f1", 30)
        if out0.strip() != h:
            import base64
            b64 = base64.b64encode(local.encode()).decode()
            tmp = f"{remote_path}.b64.part"
            ssh(wrapper, f"mkdir -p $(dirname {remote_path}) && : > {tmp}", 30)
            for i in range(0, len(b64), 60000):  # keep each remote command line small
                ssh(wrapper, f"echo -n {b64[i:i + 60000]} >> {tmp}", 60)
            rc1, out1 = ssh(wrapper, f"base64 -d {tmp} > {remote_path} && rm -f {tmp} && sha256sum {remote_path} | cut -d' ' -f1", 60)
            if out1.strip() != h:
                log(f"[{pod}] cannot launch {row_id}: manifest ship failed ({out1.strip()[:80]})")
                return None
            log(f"[{pod}] shipped workload manifest for {row_id} ({h[:12]})")
    except Exception as e:  # noqa: BLE001
        log(f"[{pod}] cannot launch {row_id}: manifest check failed: {e}")
        return None
    inner = (f"cd /workspace/cp/veritor && mkdir -p {d} && [ -f {d}/stages.txt ] && mv {d}/stages.txt {d}/stages.$(date -u +%H%M%S).prev; "
             f"{env} bash verity_capture/sweep/row_pod.sh {row_id} {role} {repo} {rev} {item['stages']} > {d}/dispatch.log 2>&1; echo DONE $? $(date -u +%FT%TZ) >> {d}/stages.txt")
    cmd = f"mkdir -p {d}; nohup setsid bash -c '{inner}' > {d}/dispatch.nohup 2>&1 < /dev/null & echo $! > {d}/dispatch.pid; sleep 1; cat {d}/dispatch.pid"
    rc, out = ssh(wrapper, cmd, 60)
    if rc != 0:
        log(f"[{pod}] launch failed rc={rc}: {out[:200]}")
        return None
    row = read_row(row_id)
    human_cmd = f"{env} bash verity_capture/sweep/row_pod.sh {row_id} {role} {repo} {rev} {item['stages']}".strip()
    idx = new_attempt(row, pod, tag, tag_pending, revision, item["stages"], item["env"], human_cmd)
    if item.get("role"):
        row["attempts"][idx]["role"] = item["role"]
    write_row(row)
    log(f"[{pod}] launched {row_id} stages={item['stages']} tag={tag}{' (pending)' if tag_pending else ''} rev={revision[:9]} attempt#{idx}")
    return {"row": row_id, "attempt": idx, "started": row["attempts"][idx]["at"], "screen": sname, "stages": item["stages"], "tag": tag, "revision": revision}


def once(a) -> None:
    state = load_json(STATE_FILE, {"pods": {}, "done": [], "launched": []})
    rels = releases()
    announced = {t for t, _ in rels}
    cite = a.tag or (rels[0][0] if rels else None)
    lines = []
    for pod, wrapper in pool().items():
        ps = state["pods"].get(pod)
        if ps:
            finished = ingest(pod, wrapper, ps)
            busy, why = pod_busy(wrapper)
            if finished or not busy:
                if not finished:
                    ingest(pod, wrapper, ps)
                    row = read_row(ps["row"])
                    att = row["attempts"][ps["attempt"]]
                    if att.get("status") == "RUNNING":
                        att["status"] = "ABORTED"
                        att["finished_at"] = now()
                        row["status"] = row.get("status") if row.get("status") != "RUNNING" else "NOT_RUN"
                        write_row(row)
                        log(f"[{pod}] {ps['row']} job vanished without DONE — marked ABORTED")
                state["done"].append({**ps, "finished": now()})
                state["pods"].pop(pod, None)
                ps = None
            else:
                lines.append(f"{pod}: RUNNING {ps['row']} ({ps['stages']}) since {ps['started'][11:16]}Z")
        if ps is None and not a.no_launch and now() >= FREEZE_UTC:
            lines.append(f"{pod}: IDLE — matrix frozen at {FREEZE_UTC[11:16]}Z (COORD 12:00Z: no new runs start)")
            continue
        if ps is None and not a.no_launch:
            info = pod_ready(wrapper)
            rtag = info["ready"].get("tag")
            rcommit = info["release"].get("commit") or info["ready"].get("commit") or ""
            tag, pending = None, False
            if rtag and rtag in announced:
                # any announced tag is a legitimate citation (the record cites the tag actually run); the pod tree must match it
                tc = tag_commit_of(rtag)
                if tc and rcommit and not tc.startswith(rcommit[:9]) and not rcommit.startswith(tc[:9]):
                    lines.append(f"{pod}: IDLE — pod tree {rcommit[:9]} != announced {rtag} commit {tc[:9]} (needs re-ship)")
                    continue
                tag = rtag
            elif a.candidate and rtag == a.candidate:
                tag, pending = a.candidate, a.candidate not in announced
            elif a.auto_candidate and rtag and re.fullmatch(r"r13-rel\d+", str(rtag)) and rtag not in announced \
                    and int(rtag[7:]) == max([int(x[7:]) for x in announced if re.fullmatch(r"r13-rel\d+", x)] or [-1]) + 1:
                # the release owner ships the next candidate to the pool before announcing it (rel0/rel1/rel3 all did):
                # run on it as release_tag_pending; the tag/commit is reconciled against RELEASES.md once announced
                tag, pending = rtag, True
            if not tag:
                lines.append(f"{pod}: IDLE — gate closed (pod ready tag={rtag!r}, announced={sorted(announced)}, cite={cite!r}, candidate={a.candidate!r})")
                continue
            busy, why = pod_busy(wrapper)
            if busy:
                lines.append(f"{pod}: busy outside the dispatcher ({why}) — not launching")
                continue
            started = {x.get("key") or (x["row"] + "|" + x.get("stages", "build,match,commit")) for x in state["launched"]}
            running_rows = {p["row"] for p in state["pods"].values()}
            cand = state.get("followups", []) + queue()
            def _tagn(t: str) -> int:
                m = re.fullmatch(r"r13-rel(\d+)", str(t or ""))
                return int(m.group(1)) if m else -1
            item = next((it for it in cand if (it.get("key") or (it["row"] + "|" + it["stages"])) not in started and it["row"] not in running_rows
                         and (it["pod"] in (None, pod)) and _tagn(tag) >= _tagn(it.get("min") or "r13-rel0")), None)
            if not item:
                lines.append(f"{pod}: IDLE — queue empty (tag {tag})")
                continue
            ps = launch(pod, wrapper, item, tag, pending, rcommit or info["ready"].get("commit", "?"))
            if ps:
                state["pods"][pod] = ps
                state["launched"].append({"row": item["row"], "stages": item["stages"], "key": item.get("key") or (item["row"] + "|" + item["stages"]), "pod": pod, "at": ps["started"], "tag": tag})
                lines.append(f"{pod}: LAUNCHED {item['row']} ({item['stages']}) on {tag}")
    if FOLLOWUPS:
        state.setdefault("followups", []).extend(FOLLOWUPS)
        FOLLOWUPS.clear()
    # a candidate tag that has since been announced: clear the pending flag on the records that cited it
    for x in state["launched"]:
        if x.get("tag") in announced and not x.get("confirmed"):
            tag_commit = tag_commit_of(x["tag"])
            row = read_row(x["row"])
            for att in row.get("attempts", []):
                if att.get("release_tag") == x["tag"] and att.get("release_tag_pending"):
                    att.pop("release_tag_pending", None)
                    row.pop("release_tag_pending", None)
                    if tag_commit and not (tag_commit.startswith(str(att.get("revision", ""))[:9]) or str(att.get("revision", "")).startswith(tag_commit[:9])):
                        # the announced tag points at a different commit than the candidate this attempt ran: lane-only revision
                        att["release_candidate_mismatch"] = f"ran {att.get('revision')} but {x['tag']} = {tag_commit[:9]}"
                        att["release_tag"] = None
                        if row.get("release_tag") == x["tag"]:
                            row["release_tag"] = None
                        log(f"{x['row']}: {att['release_candidate_mismatch']} -> lane-only")
                    write_row(row)
            x["confirmed"] = True
    save_json(STATE_FILE, state)
    n_done = len(state["done"])
    n_run = len(state["pods"])
    pending_q = [it for it in queue() if (it.get("key") or (it["row"] + "|" + it["stages"])) not in {x.get("key") or (x["row"] + "|" + x.get("stages", "build,match,commit")) for x in state["launched"]}]
    summary = [f"pool {now()[11:16]}Z: {n_run} running / {len(pool())} pods; {n_done} jobs finished; {len(pending_q)} queued; cite={cite or 'none announced'}"] + lines
    Path(STATUS_FILE).write_text("\n".join(summary) + "\n", encoding="utf-8")
    for l in summary:
        log(l)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--daemon", action="store_true")
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--tag", default=None, help="release tag to cite (default: newest RELEASED line)")
    ap.add_argument("--auto-candidate", action="store_true", help="treat the pod's ready tag r13-rel<N+1> (N = newest announced) as a pending candidate")
    ap.add_argument("--candidate", default=None, help="allow this not-yet-announced tag if the pod's ready.json names it (attempt marked release_tag_pending)")
    ap.add_argument("--no-launch", action="store_true", help="ingest only")
    ap.add_argument("--reingest", default=None, help="row id: rebuild the stage dicts of its latest dispatcher attempt from the pod evidence")
    a = ap.parse_args(argv)
    EVID.mkdir(parents=True, exist_ok=True)
    if a.reingest:
        row = read_row(a.reingest)
        idx = len(row["attempts"]) - 1
        att = row["attempts"][idx]
        for k in ("build", "match", "commit", "bench"):
            if k in att and k in att.get("stages_requested", "build,match,commit").split(",") + ["bench"]:
                att.pop(k, None)
        att["status"] = "RUNNING"
        for fid in list(FAILS.glob("F-report-*.json")):
            f = load_json(fid, {})
            if f.get("row_id") == a.reingest and f.get("revision") == att.get("revision"):
                fid.unlink()
        write_row(row)
        ingest(att["pod"], pool()[att["pod"]], {"row": a.reingest, "attempt": idx})
        return 0
    if a.daemon:
        while True:
            try:
                once(a)
            except Exception as e:  # noqa: BLE001
                log(f"dispatch error: {e!r}")
            time.sleep(a.interval)
    else:
        once(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
