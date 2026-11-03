"""PORTCHK: the per-PR table from /tmp/portchk/out (steps 1-2) and /tmp/portchk/results (step 3).

    python3 /tmp/portchk/report.py > table.md      (also writes /tmp/portchk/rows.json)
"""
from __future__ import annotations

import collections
import glob
import json
import re
import subprocess
from pathlib import Path

REPO = Path("/agent/repos/verity")
OUT = Path("/tmp/portchk/out")
RES = Path("/tmp/portchk/results")
#: later runs win for a PR: E (moved or new heads) over A2/B2/C/D, and those over the cancelled A
RUN_ORDER = ["r20261103-222317-f7eb", "r20261103-222541-89f3", "r20261103-222532-6526", "r20261103-223644-ff12",
             "r20261103-223617-52f6", "r20261103-224317-c190"]
RUN_NAME = {"r20261103-222317-f7eb": "A0", "r20261103-222541-89f3": "D", "r20261103-222532-6526": "C",
            "r20261103-223644-ff12": "B", "r20261103-223617-52f6": "A", "r20261103-224317-c190": "E"}
EXTRA_RUNS: dict[str, str] = {}

ISSUE_GOAL = {
    "PRF-164": "proofs/sampled-audits-sound", "PRF-195": "accounting/sampled-draw-from-log",
    "PRF-274": "proofs/sampled-audits-sound", "PRF-294": "accounting/pouw-lead-on-main",
    "PRF-295": "proofs/sampled-audits-sound", "PRF-302": "proofs/sampled-audits-sound",
    "PRF-310": "proofs/zero-knowledge", "PRF-314": "proofs/recursion-knowledge-sound",
    "PRF-317": "proofs/recursion-knowledge-sound", "PRF-318": "proofs/recursion-knowledge-sound",
    "PRF-342": "proofs/recursion-knowledge-sound", "PRF-320": "proofs/zkp-system-done",
    "PRF-331": "proofs/sampled-audits-sound", "PRF-335": "proofs/zk-leaks-closed",
    "PRF-347": "proofs/sampled-audits-sound", "NET-175": "accounting/live-any-worker",
    "PRF-210": "proofs/recursion-knowledge-sound", "PRF-260": "cleanup/no-lane-text",
    "PRF-281": "proofs/zkp-system-done", "PRF-282": "proofs/zkp-system-done", "PRF-298": "proofs/sampled-audits-sound",
    "PRF-193": "proofs/soundness-nothing-open",
}
#: PRs whose body names no goal and no labelled issue: the best match among the six (~), or none of them (—)
BEST = {
    5280: "— (docs)", 5370: "~sampled-audits-sound", 5489: "~zero-knowledge", 5534: "— (cleanup/no-lane-text)",
    5617: "~sampled-audits-sound", 5664: "~sampled-audits-sound", 5674: "~sampled-audits-sound",
    5747: "~sampled-audits-sound", 5766: "~sampled-audits-sound", 5774: "~sampled-audits-sound",
    5796: "~sampled-audits-sound", 5864: "~sampled-audits-sound", 5953: "~sampled-audits-sound",
    5969: "~sampled-audits-sound", 5978: "~sampled-audits-sound", 6075: "— (cleanup)", 6140: "— (cleanup)",
    6280: "~sampled-audits-sound", 6298: "— (vocabulary)", 6401: "~sampled-audits-sound", 6403: "— (vocabulary)",
    6418: "— (vocabulary)", 6422: "~sampled-audits-sound", 6426: "— (tree cut)", 6460: "~sampled-audits-sound",
    6471: "— (vocabulary)", 6476: "— (vocabulary)", 6477: "— (vocabulary)", 6478: "— (vocabulary)",
    6479: "— (vocabulary)", 6484: "— (vocabulary)", 6486: "— (vocabulary)", 6493: "— (vocabulary)",
    6495: "— (vocabulary)", 6497: "— (vocabulary)", 6503: "— (tree cut)", 6509: "— (vocabulary)",
    6511: "— (vocabulary)", 6513: "— (vocabulary)", 6529: "~sampled-audits-sound", 6532: "~sampled-audits-sound",
    6533: "~sampled-audits-sound", 6591: "~zero-knowledge (PRF-281: zkp-system-done)",
    6676: "— (cleanup: PROTOCOL.md)", 6681: "~recursion-knowledge-sound",
    6714: "~recursion-knowledge-sound (PRF-282: zkp-system-done)", 6716: "~sampled-audits-sound",
    6727: "~recursion-knowledge-sound", 6768: "~sampled-audits-sound", 6770: "~sampled-audits-sound",
    6789: "~sampled-audits-sound", 6844: "~zero-knowledge", 6854: "~zero-knowledge-any-worker",
    6935: "— (cleanup/no-lane-text)", 6998: "~sampled-audits-sound", 7093: "~sampled-audits-sound",
    7144: "~sampled-audits-sound (#7159, stacked on it)", 6072: "— (cleanup/no-lane-text)",
    5483: "— (cleanup/no-lane-text)", 5512: "— (cleanup/no-lane-text)", 6404: "proofs/sampled-audits-sound",
    6378: "proofs/sampled-audits-sound", 6655: "proofs/sampled-audits-sound", 6803: "proofs/sampled-audits-sound",
    6822: "proofs/sampled-audits-sound", 6836: "proofs/zero-knowledge", 6837: "proofs/zero-knowledge",
    6841: "proofs/zero-knowledge", 6842: "proofs/zero-knowledge", 6852: "proofs/zk-leaks-closed",
    5302: "proofs/recursion-knowledge-sound", 7134: "accounting/sampled-draw-from-log",
}
SHORT = {"Security/Proofs/lean-audit.json": "lock(Proofs)", "Security/lean-audit.json": "lock(Security)",
         "Security/Properties/lean-audit.json": "lock(Properties)",
         "verity/core/protocols/pous/lean/TRUSTED.sha256": "pin(TRUSTED)",
         "verity/core/protocols/sampled_proofs/tests/fixtures/exfiltration_lean.json": "pin(exfiltration)",
         "verity/ml/flock/tests/fixtures/randomness_lean.json": "pin(randomness)",
         "verity/core/protocols/onsite/CoreProduct.lean": "CoreProduct.lean(del)",
         "tools/move/lean.py": "tools/move(del)", "tools/move/tests/test_lean.py": None}


def git(*a) -> str:
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout


def goal(n: int, body: str) -> str:
    m = re.search(r"^\s*\**Goal\**:?\s*`?([a-z][\w/-]+)", body, re.M)
    if m:
        return m.group(1)
    m = re.search(r"Part of ((?:PRF|TOP|NET|VLLM)-\d+)", body[:1200])
    if m and m.group(1) in ISSUE_GOAL:
        return ISSUE_GOAL[m.group(1)]
    if n in BEST:
        return BEST[n]
    for ref in re.findall(r"\b(?:PRF|TOP|NET|VLLM)-\d+", body[:1500]):
        if ref in ISSUE_GOAL:
            return ISSUE_GOAL[ref]
    return "?"


def conflicts(r: dict) -> str:
    c, m = r["conflicts"], r.get("main_conflicts", {})
    if not c:
        return "0"
    parts = []
    for p, xy in sorted(c.items()):
        if p in SHORT:
            if SHORT[p]:
                parts.append(SHORT[p])
            continue
        tag = p.rsplit("/", 1)[-1]
        if xy in ("UD", "DU", "DD"):
            tag += "(del)"
        if p in m:
            tag += "*"
        parts.append(tag)
    if sum(1 for p in c if r["conflict_kinds"][p] == "lean-deleted" and p not in SHORT) > 6:
        dels = [p for p in c if r["conflict_kinds"][p] == "lean-deleted" and p not in SHORT]
        parts = [x for x in parts if not x.endswith("(del)") or x in SHORT.values()] + [f"{len(dels)} .lean(del)"]
    return f"{len(c)}: " + ", ".join(dict.fromkeys(parts))


def results() -> dict[int, tuple[str, dict]]:
    best: dict[int, tuple[str, dict]] = {}
    for run in RUN_ORDER + list(EXTRA_RUNS):
        f = RES / run / "results.jsonl"
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            rec = json.loads(line)
            if run == "r20261103-222317-f7eb":
                rec["partial"] = True
            best[rec["pr"]] = (run, rec)
    return best


def classify(err: str) -> str:
    if re.search(r"Could not find native implementation|cannot evaluate .*meta|must be marked .meta.|not a meta", err):
        return "meta"
    if re.search(r"Tactic `rfl` failed|not definitionally equal|decide. failed|failed to reduce|`decide` failed|"
                 r"simp made no progress|unsolved goals|The rfl tactic|Tactic `decide`", err):
        return "unfold"
    if re.search(r"is private|private declaration", err):
        return "private"
    if re.search(r"unknown (identifier|constant)", err):
        return "unknown"
    return "other"


def first_errors(run: str, n: int) -> list[dict]:
    """Each distinct `error: FILE:L:C: msg` in the PR's log, with its class."""
    f = RES / run / "err" / f"{n}.err"
    if not f.exists():
        return []
    text = f.read_text()
    out, seen = [], set()
    blocks = re.split(r"^--$", text, flags=re.M)
    for b in blocks:
        m = re.search(r"error: ([^\s:]+\.lean):(\d+):(\d+): (.*)", b)
        if not m:
            continue
        key = (m.group(1), m.group(2))
        if key in seen:
            continue
        seen.add(key)
        out.append({"file": m.group(1), "line": int(m.group(2)), "msg": m.group(4)[:160], "cls": classify(b)})
    return out


def own_or_inherited(r: dict, pkg_file: str, line: int) -> str:
    """'own' when the failing line's text is the PR's own (not in origin/next's copy of the file), else 'next'."""
    cands = [f for f in r["own_lean"] if f.endswith("/" + pkg_file) or f == pkg_file]
    allfiles = cands or [f for f in git("ls-tree", "-r", "--name-only", r["tree_commit"]).splitlines()
                         if f.endswith("/" + pkg_file.lstrip("./"))]
    if not allfiles:
        return "?"
    f = allfiles[0]
    text = git("show", f"{r['tree_commit']}:{f}").splitlines()
    if not (0 < line <= len(text)):
        return "?"
    want = text[line - 1].strip()
    nxt = git("show", f"origin/next:{f}")
    if want and want in {x.strip() for x in nxt.splitlines()}:
        return "next"
    return "own" if f in r["own_lean"] else "dep"


def main() -> None:
    prs = {p["number"]: p for p in json.load(open("/tmp/portchk_prs.json"))}
    res = results()
    rows = []
    for n in sorted(prs, reverse=True):
        r = json.loads((OUT / f"{n}.json").read_text())
        row = {"pr": n, "goal": goal(n, prs[n]["body"] or ""), "conflicts": conflicts(r),
               "n_conflicts": len(r["conflicts"]), "refused": "none" if not r["unported_after"] and not r["codemod_err"]
               else "; ".join(r["unported_after"])[:200], "own_lean": len(r["own_lean"])}
        if not r["own_lean"]:
            row["try"] = "skipped (no Lean)"
        elif n in res:
            run, rec = res[n]
            doc = rec.get("doc") or {}
            row["run"] = RUN_NAME.get(run, run)
            if rec.get("error"):
                row["try"] = "error: " + rec["error"][:120]
            else:
                built, failed, skipped = doc.get("built", {}), doc.get("failed", {}), doc.get("skipped", {})
                bad = [p for p, rc in built.items() if rc]
                errs = first_errors(run, n)
                for e in errs:
                    e["whose"] = own_or_inherited(r, e["file"], e["line"])
                row["errors"] = errs
                row["skipped"] = skipped
                partial = rec.get("partial") and any(p.startswith("Security") for p in skipped)
                row["try"] = ("pass" if not bad else f"fail ({len(bad)} pkg)") + (" (Security not built)" if partial else "")
                row["bad_pkgs"] = bad
        else:
            row["try"] = "pending"
        rows.append(row)
    Path("/tmp/portchk/rows.json").write_text(json.dumps(rows, indent=1))
    print("| PR | Goal | Conflicts | Codemod refused | Try | Hand fixes |")
    print("|---|---|---|---|---|---|")
    for row in rows:
        fixes = row.get("fixes", "")
        print(f"| #{row['pr']} | {row['goal']} | {row['conflicts']} | {row['refused']} | {row['try']} | {fixes} |")


if __name__ == "__main__":
    main()
