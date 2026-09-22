"""sweep/deps.py — R13 dependency map + ready queue (report owner; BRIEF §6, PLAN §3).

    python -m verity_capture.sweep.deps --root out/gen/sweep [--now HH:MMZ]

Writes `DEPS.json` (row id -> complete blocker set, cluster ids; seed = PLAN §3 patterns, refined from the row's unresolved
failure records) and `READY.md` (rows whose blocker set is fully covered by fixes in a RELEASED tag, ordered per the matrix
completion order: B1 diagnostic -> medium mixed batch -> largest batch -> other envelopes -> stochastic).  Pure; no torch.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

from verity_capture.sweep import validate as R

# fix id -> clusters it resolves (seed; refined from fix records: `clusters`/`resolves` fields or cluster ids in the text)
FIX_CLUSTERS_SEED: dict[str, list[str]] = {
    "X-00": ["C-HIDDENGPU"], "X-01": ["M-SMCOUNT", "B-SMCOUNT"], "X-02": ["C-ARENA"], "X-03": [], "REL-01": [], "REL-02": [], "REL-03": [],
    # COORD 09:55Z: AMP-01 (r13-rel2) resolves the Ampere GEMM-tile pair; HOP-01 (HANDOFF-READY 09:5xZ) resolves both Hopper clusters
    "AMP-01": ["B-AMPERE-GEMM-TILE", "M-AMPERE-GEMM-TILE"], "HOP-01": ["B-HOPPER-ATTN", "C-HOPPER-FA3"], "X-09": ["M-BATCH-DECOMP"],
}
# fixes whose text names a cluster they do NOT clear on their own (dense STATUS 07:xxZ: X-05 = clean refusal `sampler`, X-06 =
# GumbelTokenSelect Definition only; F-dense-03 program-shape (seed/pos/temp as Program inputs) is still open -> B-SAMPLING stays a blocker)
FIX_CLUSTERS_OVERRIDE: dict[str, list[str]] = {"X-05": [], "X-06": []}
# clusters that are pure harness/deployment and are cleared by ANY r13 release (the release tree contains X-00..X-05)
DEPLOYMENT_CLEARED_BY_ANY_RELEASE = {"C-HIDDENGPU", "M-SMCOUNT", "B-SMCOUNT", "C-ARENA"}

CLUSTER_ID_RE = re.compile(r"\b([BMC]-[A-Z0-9]+(?:-[A-Z0-9]+)*)\b")


def seed_blockers(p: dict) -> set[str]:
    """PLAN §3 row-family -> blocker set, from the parsed row id."""
    m, hw, tp, b, i, o = p["model"], p["hw"], p["tp"], p["batch"], p["max_input"], p["max_output"]
    s: set[str] = set()
    if m.startswith("qwen25"):
        s |= {"B-QKVBIAS", "C-FA2-HDIM"}
    elif m.startswith("phi3"):
        s |= {"C-FA2-HDIM"}
    elif m.startswith("mistral"):
        s |= {"M-NOPROFILE", "C-FA2-HDIM"}
    elif m.startswith("gemma2"):
        s |= {"B-GEMMA-SCALE", "B-GEMMA-NORM", "C-FA2-HDIM"}
    elif m.startswith("olmoe"):
        s |= {"B-QNORM-STRIDE", "B-MOE", "C-FA2-HDIM"}
    elif m.startswith("qwen3-30b"):
        s |= {"B-MOE", "M-TP-ENGINE", "C-FA2-HDIM"}
    elif m.startswith("qwen3"):
        s |= {"C-FA2-HDIM"}
    if hw == "a100":
        s |= {"B-AMPERE-GEMM-TILE", "M-AMPERE-GEMM-TILE"}
    if hw == "h100":
        s |= {"B-HOPPER-ATTN", "C-HOPPER-FA3"}
    if tp > 1:
        s |= {"M-TP-ENGINE"}
    if p["dtype"] == "fp8":
        s |= {"B-FP8"}
    if "stoch" in p["sampling"]:
        s |= {"B-SAMPLING"}
    if p["execution"] == "bi-compiled":
        s |= {"M-COMPILED-FUSION"}
    # F-report-02 (06:28Z): the derived Program is the single-request wrapper; a mixed batch of B>=2 requests
    # executes a different instance sequence -> Match cannot pass for any B>=2 row until a batch/schedule wrapper exists
    if b >= 2:
        s |= {"M-BATCH-DECOMP"}
    # collector footprint: B>=2 at >=1024 tokens for every model; Llama even at B=1 (F-bc-04) and any long-context row
    if i >= 1024 and (b >= 2 or m.startswith("llama") or i >= 4096) and not m.startswith("smollm2"):
        s |= {"C-COLLECTOR-FOOTPRINT"}
    if i >= 1024 and m.startswith("smollm2") and b >= 32:
        s |= {"C-COLLECTOR-FOOTPRINT"}
    # FA2 hidden stream O(rows x N) (F-fa2-02, F-report-07/08): hd-128 / 32-head families at 1024 tokens do not arm on 48 GB;
    # Phi-3 (hd 96, 32 heads) at B=8/1024 = 132 GiB per prefill step (fa2). Not on 80 GB A100 for B=1 (coordinator 08:57Z).
    if i >= 1024 and (m.startswith("mistral") or (m.startswith("phi3") and b >= 8)) and "a100" not in hw and "h100" not in hw:
        s |= {"C-FA2-STREAM"}
    if i >= 1024 and m.startswith("mistral") and b >= 8:
        s |= {"C-FA2-STREAM"}
    return s


def fix_clusters(recs: R.Records) -> dict[str, list[str]]:
    out = {k: list(v) for k, v in FIX_CLUSTERS_SEED.items()}
    for xid, x in recs.fixes.items():
        cl: list[str] = []
        for k in ("clusters", "resolves", "cluster"):
            v = x.get(k)
            if isinstance(v, str):
                cl.append(v)
            elif isinstance(v, list):
                cl += [str(c) for c in v]
        if not cl:
            txt = " ".join(str(x.get(k) or "") for k in ("mechanism", "why_it_failed", "generic_change", "notes"))
            cl = [c for c in CLUSTER_ID_RE.findall(txt) if c in R.CLUSTER_MECHANISM]
        if xid in FIX_CLUSTERS_OVERRIDE:
            cl = list(FIX_CLUSTERS_OVERRIDE[xid])
            out[xid] = cl
            continue
        if cl:
            out[xid] = sorted(set(out.get(xid, [])) | set(cl))
    return out


def parse_releases(root: str) -> list[dict]:
    p = os.path.join(os.path.dirname(root.rstrip("/")), "r13", "RELEASES.md")
    out: list[dict] = []
    if not os.path.exists(p):
        return out
    seen: set[str] = set()
    with open(p, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"^\W*RELEASED\s+(r13-rel\d+)\s+(\S*)\s*(.*)$", line.strip())
        if m and m.group(1) not in seen:
            seen.add(m.group(1))
            out.append({"tag": m.group(1), "fixes": [x for x in re.split(r"[,+/\s]+", m.group(2)) if x], "hint": m.group(3).strip()})
            continue
        # older releases live in the body as '### r13-relN — tag <sha> (<time>) — CANARY-GREEN ...' sections (candidates are '### r13-relN — candidate')
        m = re.match(r"^###\s+(r13-rel\d+)\s+—\s+tag\s+([0-9a-f]{7,40})\b(.*)$", line.strip())
        if m and m.group(1) not in seen and (not out or int(m.group(1)[7:]) <= int(out[0]["tag"][7:])):
            # a section for a tag newer than the RELEASED line (canary still running) is not an announced release
            seen.add(m.group(1))
            body = " ".join(lines[i + 1:i + 4])
            fixes = sorted(set(re.findall(r"\*\*([A-Z][A-Z0-9]*-\d+)\*\*", body)))
            out.append({"tag": m.group(1), "fixes": fixes, "hint": m.group(3).strip(" —"), "commit": m.group(2)})
    out.sort(key=lambda r: -int(r["tag"][7:]))  # newest first
    return out


def _first_sha(text: str) -> str | None:
    m = re.search(r"\b([0-9a-f]{9,40})\b", str(text or ""))
    return m.group(1) if m else None


def fixes_in_tag(tag: str, recs: "R.Records", repo: str) -> set[str]:
    """Fix records whose commit is an ancestor of the release tag (the RELEASED line may abbreviate, e.g. rel1 lists the fa2 v8 tap
    as '(+fa2 v8 tap hd64/96/128)' without the FA2-01 id).  Records whose commit is not resolvable contribute nothing."""
    out: set[str] = set()
    try:
        tip = subprocess.run(["git", "-C", repo, "rev-parse", "-q", "--verify", f"{tag}^{{commit}}"], capture_output=True, text=True).stdout.strip()
    except Exception:
        tip = ""
    if not tip:
        return out
    for xid, x in recs.fixes.items():
        sha = _first_sha(x.get("commit") or x.get("code_commit") or "")
        if not sha:
            continue
        r = subprocess.run(["git", "-C", repo, "merge-base", "--is-ancestor", sha, tip], capture_output=True)
        if r.returncode == 0:
            out.add(xid)
    return out


def lane_fixes(recs: R.Records) -> dict[str, str]:
    """fix id -> commit for fixes that exist as records but are in no release yet (lane-only)."""
    return {xid: str(x.get("commit") or "") for xid, x in recs.fixes.items()}


def completion_order_key(p: dict) -> tuple:
    stoch = "stoch" in p["sampling"] or p["execution"] != "bi-eager"
    b, i = p["batch"], p["max_input"]
    if stoch:
        k = 4
    elif i >= 4096:
        k = 3
    elif b == 1:
        k = 0
    elif b <= 16:
        k = 1
    else:
        k = 2
    return (k, R.MODEL_ORDER.get(p["model"], 99), R.HW_ORDER.index(p["hw"]) if p["hw"] in R.HW_ORDER else 9, p["tp"], b, i)


def build(recs: R.Records, now: str) -> tuple[dict, str]:
    clusters = R.cluster_failures(recs)
    by_f = {f: cl["id"] for cl in clusters for f in cl["failure_ids"]}
    fclust = fix_clusters(recs)
    rels = parse_releases(recs.root)
    released_fixes: dict[str, str] = {}
    repo = os.path.dirname(os.path.dirname(os.path.dirname(recs.root.rstrip("/"))))  # out/gen/sweep -> repo
    for r in reversed(rels):  # oldest tag first so a fix is attributed to the first release that carried it
        for x in list(r["fixes"]) + sorted(fixes_in_tag(r["tag"], recs, repo)):
            released_fixes.setdefault(x, r["tag"])
    resolved_by: dict[str, str] = {}
    if rels:
        for c in DEPLOYMENT_CLEARED_BY_ANY_RELEASE:
            resolved_by[c] = rels[-1]["tag"] if rels else ""
    for x, tag in released_fixes.items():
        for c in fclust.get(x, []):
            resolved_by.setdefault(c, tag)
    lane_only_resolved: dict[str, str] = {}
    for x, commit in lane_fixes(recs).items():
        if x not in released_fixes:
            for c in fclust.get(x, []):
                if c not in resolved_by:
                    lane_only_resolved[c] = f"{x}@{commit[:9]}"

    rows_out: dict[str, dict] = {}
    for rid, row in recs.rows.items():
        p = R.parse_row_id(rid)
        if not p:
            continue
        blockers = seed_blockers(p)
        observed: set[str] = set()
        for fid, f in recs.failures.items():
            if f.get("row_id") == rid and fid in by_f:
                cid = by_f[fid]
                if not cid.startswith("?"):
                    observed.add(cid)
        blockers |= observed
        # deployment clusters are cleared by the release tree; keep them visible but they never block once a release exists
        remaining = {c for c in blockers if c not in resolved_by}
        remaining_after_lane = {c for c in remaining if c not in lane_only_resolved}
        # how far a row can run now: B-* clusters block Build, M-* Match, C-* Commit (prefix of the cluster id)
        pref = {c[0] for c in remaining}
        through = "commit" if not pref else ("match" if pref == {"C"} else ("build" if pref <= {"C", "M"} else "none"))
        rows_out[rid] = {
            "runnable_through": through,
            "blockers": sorted(blockers), "observed_in_failures": sorted(observed), "seed": sorted(seed_blockers(p)),
            "resolved": {c: resolved_by[c] for c in sorted(blockers) if c in resolved_by},
            "resolved_lane_only": {c: lane_only_resolved[c] for c in sorted(remaining) if c in lane_only_resolved},
            "remaining": sorted(remaining), "remaining_if_lane_fixes_released": sorted(remaining_after_lane),
            "status": row.get("status"), "release_tag": R.release_label(row), "attempts": row.get("_n_attempts", 0),
            "ready": not remaining, "order_key": completion_order_key(p),
        }
    deps = {"schema": "verity-sweep/deps/v1", "generated": now, "releases": rels, "fix_clusters": fclust,
            "resolved_clusters": resolved_by, "lane_only_resolved_clusters": lane_only_resolved, "rows": rows_out}
    return deps, ready_md(recs, deps, now)


def ready_md(recs: R.Records, deps: dict, now: str) -> str:
    from verity_capture.sweep import render as X  # local import: render imports validate only
    rows = deps["rows"]
    rels = deps["releases"]
    lines = [f"# READY queue — {now} — releases: {', '.join(r['tag'] for r in rels) or 'none yet (nothing dispatchable to the pool)'}", ""]
    ready = [(rid, d) for rid, d in rows.items() if d["ready"]]
    ready_todo = [(rid, d) for rid, d in ready if X.partition(recs, recs.rows[rid]) != "complete"]
    ready_done = [(rid, d) for rid, d in ready if X.partition(recs, recs.rows[rid]) == "complete"]
    ready_todo.sort(key=lambda kv: kv[1]["order_key"])
    lines.append(f"## Ready to run ({len(ready_todo)}) — every blocker covered by a released fix; ordered B1 → medium batch → large batch → long envelope → stochastic")
    lines.append("")
    if not ready_todo:
        lines.append("_(none)_")
    lines.append("| # | row | status | latest attempt | blockers (all resolved) | resolved by |\n|---:|---|---|---|---|---|")
    for n, (rid, d) in enumerate(ready_todo, 1):
        lines.append(f"| {n} | `{rid}` | {d['status']} | {d['release_tag'] or '—'} ×{d['attempts']} | {', '.join(d['blockers']) or '—'} | "
                     f"{', '.join(f'{c}:{t}' for c, t in d['resolved'].items()) or 'no mechanism blocker'} |")
    if ready_done:
        lines.append(f"\nAlready fully completed on a release: {', '.join(f'`{r}`' for r, _ in ready_done)}")
    # partial-progress jobs: Build/Match can run on a release even though a later stage is still blocked
    stage_rank = {"none": 0, "build": 1, "match": 2, "commit": 3}
    partial = []
    for rid, d in rows.items():
        if d["ready"] or d["runnable_through"] == "none" or not rels:
            continue
        row = recs.rows[rid]
        # skip if the runnable stages are already PASS on a release-tag attempt
        done_on_release = d["release_tag"].startswith("r13-rel") and all(
            R.stage_status(row, st) == "PASS" for st in ("build", "match")[: stage_rank[d["runnable_through"]]])
        if not done_on_release:
            partial.append((rid, d))
    partial.sort(key=lambda kv: kv[1]["order_key"])
    lines.append(f"\n## Partial-progress jobs ({len(partial)}) — Build/Match runnable on a release while a later stage is still blocked (fill idle pool time; record the stage result, stop at the blocked stage)")
    lines.append("")
    for rid, d in partial:
        lines.append(f"- `{rid}` — run through **{d['runnable_through']}**; still blocked at: {', '.join(d['remaining'])}")
    if not partial:
        lines.append("_(none)_")
    lane = [(rid, d) for rid, d in rows.items() if not d["ready"] and not d["remaining_if_lane_fixes_released"]]
    lane.sort(key=lambda kv: kv[1]["order_key"])
    lines.append(f"\n## Ready once lane fixes are released ({len(lane)}) — blocker set covered only by lane-only fix records (not dispatched to the pool)")
    lines.append("")
    for rid, d in lane:
        lines.append(f"- `{rid}` — waiting for release of: {', '.join(f'{c} ({t})' for c, t in d['resolved_lane_only'].items())}")
    if not lane:
        lines.append("_(none)_")
    waiting = [(rid, d) for rid, d in rows.items() if d["remaining_if_lane_fixes_released"]]
    by: dict[str, list[str]] = {}
    for rid, d in waiting:
        for c in d["remaining_if_lane_fixes_released"]:
            by.setdefault(c, []).append(rid)
    lines.append(f"\n## Waiting on a mechanism ({len(waiting)} rows) — rows per unresolved cluster (a row appears under each of its blockers)")
    lines.append("")
    lines.append("| cluster | rows waiting | owner lane | rows |\n|---|---:|---|---|")
    owners = {"M-ARRIVAL-ORDER": "R14 ruling (canonical within-step order) — COORD 13:40Z", "C-FOLD-COST": "rel (harness time limits)", "M-BATCH-DECOMP": "p0b's dA agent (reassigned, COORD 08:35Z)", "C-COLLECTOR-FOOTPRINT": "p0", "C-FA2-STREAM": "unowned (COORD 09:55Z: remaining O(rows x N) object)", "B-QKVBIAS": "dense", "B-GEMMA-NORM": "dense (F-dense-05; not scheduled tonight)", "B-SAMPLING": "dense", "M-NOPROFILE": "dense", "B-GEMMA-SCALE": "dense",
              "C-FA2-HDIM": "fa2", "B-MOE": "moe", "B-QNORM-STRIDE": "moe", "M-TP-ENGINE": "tp", "B-AMPERE-GEMM-TILE": "ampere",
              "M-AMPERE-GEMM-TILE": "ampere", "B-HOPPER-ATTN": "hopper", "C-HOPPER-FA3": "hopper", "B-FP8": "hopper (exploratory)",
              "M-COMPILED-FUSION": "— (classified, not pursued)"}
    for c, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"| **{c}** | {len(rs)} | {owners.get(c, '?')} | " + ", ".join(f"`{r}`" for r in sorted(rs)) + " |")
    lines.append(f"\n_Multi-blocker rows are the norm: {sum(1 for d in rows.values() if len(d['blockers']) > 1)} of {len(rows)} rows have ≥2 blockers. "
                 f"Deployment clusters (C-HIDDENGPU, M-SMCOUNT, C-ARENA) are cleared by any r13 release tree (X-00..X-02 merged)._")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="out/gen/sweep")
    ap.add_argument("--now", default=None)
    a = ap.parse_args(argv)
    now = a.now or datetime.now(timezone.utc).strftime("%H:%MZ")
    recs = R.load(a.root)
    deps, md = build(recs, now)
    with open(os.path.join(a.root, "DEPS.json"), "w", encoding="utf-8") as fh:
        json.dump(deps, fh, indent=1, sort_keys=True)
    with open(os.path.join(a.root, "READY.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    rows = deps["rows"]
    n_ready = sum(1 for d in rows.values() if d["ready"])
    top = {}
    for d in rows.values():
        for c in d["remaining"]:
            top[c] = top.get(c, 0) + 1
    top_s = ", ".join(f"{c}:{n}" for c, n in sorted(top.items(), key=lambda kv: -kv[1])[:4])
    print(f"[deps {now}] rows={len(rows)} ready={n_ready} releases={[r['tag'] for r in deps['releases']]} top blockers: {top_s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
