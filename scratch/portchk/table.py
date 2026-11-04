"""PORTCHK's table for TOP-188, from steps12.json (steps 1-2), prs.json (gh, now), next_queue.json and results/ (step 3).

    python3 /tmp/portchk/table.py > /tmp/portchk/table.md       (rows also in /tmp/portchk/rows.json)
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/tmp/portchk")
from report import ISSUE_GOAL, BEST, goal  # noqa: E402

D = Path("/tmp/portchk")
REPO = Path("/agent/repos/verity")
#: later runs win: A0 (cancelled), D, C, B, A, E, then the rerun
RUNS = [("r20261103-222317-f7eb", "A0"), ("r20261103-222541-89f3", "D"), ("r20261103-222532-6526", "C"),
        ("r20261103-223644-ff12", "B"), ("r20261103-223617-52f6", "A"), ("r20261103-224800-304d", "E"),
        ("r20261104-070723-ab3a", "R")]

#: failing file (suffix) -> (origin PR, fix, lines, how the fix was checked)
CATALOG = {
    "Flock/Claims/Complete/Json/RoundTrip.lean": (5105, "`import all` Init.Data.Array.QSort.Basic, Init.Data.Repr, "
                                                  "Init.Data.Int.Repr, Lean.Data.Json.Basic", 4, "core repro"),
    "Flock/Recursive/FirewallJson.lean": (7026, "`import all Init.Data.Repr` (Nat.digitChar)", 1, "core repro"),
    "Flock/Recursive/FirewallWire.lean": (7026, "`import all Init.Data.String.Defs` (fromUTF8, toUTF8)", 1, "core repro"),
    "Flock/Recursive/FirewallAny.lean": (7064, "`import all Lean.Data.Json.Basic` + Init.Data.Ord.String (roundSpec at "
                                         "Json.null)", 2, "estimate"),
    "VStarRecursive/Protocol.lean": (6677, "`import all Lean.Data.Json.Basic` + Init.Data.Ord.String (getD of mkObj)",
                                     2, "core repro"),
    "Properties/Composition.lean": (6677, "`import all Init.Data.String.Defs` (String.toUTF8 in msg_inj)", 1,
                                    "core repro"),
    "Properties/Nci/Protocol.lean": (6677, "`import all` of the Json modules Baseline.evaluate reduces through", 2,
                                     "estimate"),
    "SampledProofs/Json.lean": (7092, "`import all Lean.Data.Json.Basic` + Init.Data.Ord.String (jsonStrIs of mkObj)",
                                2, "core repro"),
    "Flock/SessionLoop/FinPiopRet.lean": (6842, "`import all Init.Data.Range.Basic` (decide +kernel over Zk.schedule)",
                                          1, "flock repro"),
    "Flock/SessionLoop/LiveInner.lean": (6912, "`import all Init.Data.String.Defs` (toByteArray = toUTF8 by rfl)", 1,
                                         "core repro"),
    "Properties/Verity/Sampled/Deployed.lean": (7137, "`import all Proofs.Flock.Soundness.Discharge.ZkLink.DrawOkJson` "
                                                "(open private of its lemmas)", 1, "estimate"),
    "Flock/Test/RegBudget.lean": (7150, "not the port: fails on main's toolchain too (structure-instance field column); "
                                  "fixed at #7150's head 13076c1ddc", 0, "plain v4.34 repro"),
    "Soundness/Discharge/ZkReg/SoundHJSt.lean": (5825, "`import all Init.Data.Array.Basic` (Array.idxOf)", 1,
                                                 "core repro"),
    "Flock/Recursive/KSLinked.lean": (7127, "not the port: #7127 moves `nS x ≤ 16` into the PartIn premise, and main's "
                                      "KSLinked (added after #7127 forked) reads the old shape; update its two uses", 6,
                                      "read"),
    "Claims/Recursion/IrDep.lean": (5302, "not the port: OpenLower.statements_dense changed shape on main since #5302 "
                                    "forked (29 Oct); rebind its witness before `rw [hh] at hA`", 2, "read"),
    "Flock/Soundness/ExecStrataOf.lean": (7098, "keep the port's qsort_perm beside #7098's chain_covers (my first "
                                          "resolution took #7098's qsort_perm)", 3, "E run passes"),
}
#: conflicts that are derived records or pins: regenerated, no hand edit
DERIVED = ("lean-audit.json", "TRUSTED.sha256", "exfiltration_lean.json", "randomness_lean.json")
GONE_ON_MAIN = ("verity/core/protocols/onsite/CoreProduct.lean", "tools/move/lean.py", "tools/move/tests/test_lean.py")


def git(*a) -> str:
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout


def results() -> dict[int, tuple[str, dict]]:
    best = {}
    for run, name in RUNS:
        f = D / "results" / run / "results.jsonl"
        if f.exists():
            for line in f.read_text().splitlines():
                rec = json.loads(line)
                if name == "A0":
                    continue
                best[rec["pr"]] = (name, rec)
    return best


def catalog_key(path: str) -> str | None:
    for k in CATALOG:
        if path.endswith(k):
            return k
    return None


def where(n: int, prs: dict, queue: dict) -> str:
    p = prs.get(str(n))
    if p and p["state"] == "MERGED":
        return "main"
    if str(n) in queue:
        return "next"
    return "open" if not p or p["state"] == "OPEN" else p["state"].lower()


def main() -> None:
    s = json.loads((D / "steps12.json").read_text())
    prs = json.loads((D / "prs.json").read_text())
    queue = json.loads((D / "next_queue.json").read_text())
    res = results()
    main_files = set(git("ls-tree", "-r", "--name-only", "origin/main").splitlines())
    rows = []
    for n in sorted(s, key=int, reverse=True):
        r, p, num = s[n], prs[n], int(n)
        w = where(num, prs, queue)
        row = {"pr": num, "title": p["title"], "goal": goal(num, p["body"] or ""), "now": w,
               "side": "below" if w in ("main", "next") else "above",
               "moved": p["state"] == "OPEN" and not p["headRefOid"].startswith(r["head"][:10])}
        # conflicts
        hand, mech, dels, stale = [], [], [], []
        for path, xy in r["conflicts"].items():
            kind = r["conflict_kinds"].get(path, "")
            base = path.rsplit("/", 1)[-1]
            if path.endswith(DERIVED):
                mech.append(base)
            elif path in GONE_ON_MAIN:
                stale.append(base + "(del)")
            elif kind == "lean-deleted":
                (stale if path not in main_files else dels).append(base)
            else:
                hand.append(base + ("*" if path in r.get("main_conflicts", {}) else ""))
        row.update(n_conf=len(r["conflicts"]), hand=hand, mech=mech, dels=dels, stale=stale)
        # try
        fails = []
        if not r["own_lean"]:
            row["try"] = "no Lean"
        elif num in res:
            run, rec = res[num]
            row["run"] = run
            doc = rec.get("doc") or {}
            if rec.get("error"):
                row["try"] = "empty patch (in the port already)" if "No valid patches" in rec["error"] else "error"
            else:
                failed = doc.get("failed", {})
                row["try"] = "pass" if not failed and rec.get("rc") == 0 else "fail"
                for path in failed:
                    k = catalog_key(path)
                    if k is None:
                        fails.append({"file": path, "origin": "?", "fix": "?", "lines": None})
                        continue
                    origin, fix, lines, how = CATALOG[k]
                    fails.append({"file": k, "origin": origin, "fix": fix, "lines": lines, "how": how,
                                  "own": origin == num or (origin is None)})
        else:
            row["try"] = "pending"
        row["fails"] = fails
        rows.append(row)
    (D / "rows.json").write_text(json.dumps(rows, indent=1))
    render(rows)


#: where each failure's origin PR is now
_PRS = json.loads((D / "prs.json").read_text())
_Q = json.loads((D / "next_queue.json").read_text())
ORIGIN_NOW = {o: where(o, _PRS, _Q) for o in (5105, 7026, 7064, 6677, 6842, 6912, 7092, 7137, 7150, 7098, 5825, 7127,
                                                5302)}


def short(key: str) -> str:
    return ".".join(key.removesuffix(".lean").split("/")[-2:])


def fix_cell(row: dict) -> tuple[str, int, str]:
    """The PR's own hand fixes, their size in lines, and the failures it inherits from PRs below the port."""
    own, inh, size = [], [], 0
    for f in row["fails"]:
        base = short(f["file"])
        if f["origin"] == "?":
            own.append(f"{base}: unattributed")
        elif f["origin"] in (7127, 5302) and f["origin"] == row["pr"]:
            own.append(f"{base}: {f['fix']}")
            size += f["lines"]
        elif f["origin"] in (7127, 5302):
            inh.append(f"{base} (#{f['origin']}'s own, not the port)")
        elif f["origin"] == 7150:
            own.append(f"{base}: #7150's own parse error, not the port (fails on main's toolchain too); fixed at "
                       "#7150's head 13076c1ddc" + ("" if row["pr"] == 7150 else ", stacked here"))
        elif f["origin"] == row["pr"]:
            own.append(f"{base}: {f['fix']}")
            size += f["lines"]
        else:
            inh.append(f"{base} (#{f['origin']}, {ORIGIN_NOW.get(f['origin'], '?')})")
    if "ExecStrataOf.lean" in row["hand"]:
        if row["pr"] != 7098:
            inh.append("Soundness.ExecStrataOf (#7098, next)")
        elif not any(f["file"].endswith("ExecStrataOf.lean") for f in row["fails"]):
            own.append(f"Soundness.ExecStrataOf: {CATALOG['Flock/Soundness/ExecStrataOf.lean'][1]}")
            size += 3
    hand = [h for h in row["hand"] if h != "ExecStrataOf.lean"]
    if hand:
        own.append("resolve " + ", ".join(dict.fromkeys(hand)))
    return "; ".join(own) or "none", size, ", ".join(dict.fromkeys(inh)) or "-"


def render(rows: list[dict]) -> None:
    print("| PR | Goal | Now | Port | Conflicts | Try | The PR's own hand fix | Size | Inherited, fixed below the port |")
    print("|---|---|---|---|---|---|---|---|---|")
    for row in rows:
        conf = "0" if not row["n_conf"] else (
            f"{row['n_conf']}: " + "; ".join(x for x in [", ".join(dict.fromkeys(row["hand"])),
                                                         f"{len(row['mech'])} records/pins" if row["mech"] else "",
                                                         f"{len(row['dels'])} deleted" if row["dels"] else "",
                                                         f"{len(row['stale'])} gone on main" if row["stale"] else ""] if x))
        fix, size, inh = fix_cell(row)
        t = row["try"] + (f" ({row['run']})" if row.get("run") else "") + (" [head moved]" if row["moved"] else "")
        g = row["goal"].replace("proofs/", "")
        print(f"| #{row['pr']} | {g} | {row['now']} | {row['side']} | {conf} | {t} | {fix} | {size or '-'} | {inh} |")


if __name__ == "__main__":
    main()
