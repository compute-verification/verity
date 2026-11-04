"""PORTCHK: who wrote each failing line: the PR's own commits, main, or next (a merge of next into the PR).

    python3 /tmp/portchk/attribute.py RUN_LETTER:PR ...     (A, B, C or D: the shipped patch; E: the PR's head, rebuilt)

For each failed file in the PR's result, the merged-and-converted tree is the port plus the shipped patch; the failing
line's text is looked up in the PR's head (pre-codemod), blamed there, and the blamed commit classified."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path("/agent/repos/verity")
PORT = "27ce82e6010eefb5064f2a497b8a9c4500a9a4c2"
WT = Path("/tmp/portchk/wt-recon")
RUNS = {"A": "r20261103-223617-52f6", "B": "r20261103-223644-ff12", "C": "r20261103-222532-6526",
        "D": "r20261103-222541-89f3", "E": "r20261103-224800-304d"}
STEPS = json.loads(Path("/tmp/portchk/steps12.json").read_text())
DECL = re.compile(r"^\s*(?:@\[[^\]]*\]\s*)?(?:private\s+|protected\s+|noncomputable\s+|meta\s+)*"
                  r"(theorem|lemma|def|abbrev|instance|example|structure|inductive)\s+(\S+)?")


def git(*a, cwd=REPO, check=True) -> str:
    p = subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True)
    if check and p.returncode:
        raise RuntimeError(f"git {' '.join(a)}: {p.stderr[-500:]}")
    return p.stdout


def tree_for(letter: str, n: int) -> None:
    if not WT.exists():
        git("worktree", "add", "-q", "--detach", str(WT), PORT)
    git("checkout", "-q", "-f", "--detach", PORT, cwd=WT)
    git("clean", "-q", "-fd", cwd=WT)
    patch = Path(f"/tmp/portchk/ship/{letter}/{n}.patch")
    if patch.stat().st_size:
        git("apply", "--index", "--binary", str(patch), cwd=WT)


def full_path(pkg_rel: str, steps: dict) -> str | None:
    """`../Proofs/X.lean` or `Flock/Y.lean`, as Lake prints them, to a repository path."""
    rel = pkg_rel.lstrip("./").removeprefix("../")
    hits = [f for f in git("ls-files", cwd=WT).splitlines() if f.endswith("/" + rel)]
    own = [f for f in hits if f in steps.get("own_lean", [])]
    return (own or hits or [None])[0]


def decl_at(lines: list[str], i: int) -> str:
    for j in range(i, -1, -1):
        m = DECL.match(lines[j])
        if m:
            return f"{m.group(1)} {m.group(2) or ''}".strip()
    return "?"


def classify(head: str, path: str, text: str) -> tuple[str, str]:
    try:
        body = git("show", f"{head}:{path}").splitlines()
    except RuntimeError:
        return "?", "file not at head"
    idx = [i for i, l in enumerate(body) if l.strip() == text.strip()]
    if not idx:
        return "?", "line not at head (codemod or resolution)"
    blame = git("blame", "-l", "-L", f"{idx[0] + 1},{idx[0] + 1}", head, "--", path).split()[0].lstrip("^")
    own = set(git("log", "--first-parent", "--no-merges", "--format=%H", f"origin/main..{head}").split())
    if blame in own:
        return "own", blame[:10]
    if subprocess.run(["git", "merge-base", "--is-ancestor", blame, "origin/main"], cwd=REPO).returncode == 0:
        port_has = subprocess.run(["git", "merge-base", "--is-ancestor", blame, PORT], cwd=REPO).returncode == 0
        return ("main (in the port's base)" if port_has else "main (after the port's base)"), blame[:10]
    return "next", blame[:10]


def main(argv: list[str]) -> None:
    out = {}
    for arg in argv:
        letter, n = arg.split(":")
        n = int(n)
        steps = STEPS[str(n)]
        head = steps["head"]
        res = [json.loads(l) for l in open(f"/tmp/portchk/results/{RUNS[letter]}/results.jsonl")]
        rec = next(r for r in res if r["pr"] == n)
        failed = (rec.get("doc") or {}).get("failed", {})
        tree_for(letter, n)
        rows = []
        for _, msg in failed.items():
            m = re.search(r"error: (\S+\.lean):(\d+):(\d+): (.*)", msg)
            if not m:
                continue
            path = full_path(m.group(1), steps)
            if not path:
                rows.append({"file": m.group(1), "why": "not found"})
                continue
            lines = (WT / path).read_text().splitlines()
            ln = int(m.group(2))
            text = lines[ln - 1] if 0 < ln <= len(lines) else ""
            who, c = classify(head, path, text)
            rows.append({"file": path, "line": ln, "decl": decl_at(lines, ln - 1), "text": text.strip()[:120],
                         "who": who, "commit": c, "msg": m.group(4)[:100]})
        out[n] = rows
        for r in rows:
            print(n, json.dumps(r))
    Path("/tmp/portchk/attrib.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
