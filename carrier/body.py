"""One line at the top of a pull request's body, the rest kept as it is.

    python3 body.py N LINE_FILE [--goal GOAL] [--apply]

Reads #N's body as GitHub has it now, puts LINE_FILE's text as a paragraph of its own at the top (after the agent's
`<!-- CURSOR_AGENT_PR_BODY_BEGIN -->` marker where the body opens with one), adds `Goal GOAL` after the `Lane` line where
`--goal` is given and the body names no goal, and hands the result to `research queue edit N --body-file F` (`--dry-run`
unless `--apply`). After `--apply` it reads the body back and prints the diff against what it read first: only added
lines, or it says so loudly. Bodies are kept under /tmp/carrier/bodies/.
"""
import difflib
import pathlib
import re
import subprocess
import sys

REPO = "/agent/repos/verity"
OUT = pathlib.Path("/tmp/carrier/bodies")
MARK = "<!-- CURSOR_AGENT_PR_BODY_BEGIN -->"
GOAL_RE = re.compile(r"^[ \t>*_]*Goal:?[ \t]+`?([A-Za-z0-9](?:[\w./-]*[A-Za-z0-9])?)`?[ \t.]*$", re.M)
LANE_RE = re.compile(r"^[ \t>*_]*Lane[:\s]+`?([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)`?.*$", re.M)


def body(n: str) -> str:
    out = subprocess.run(["gh", "pr", "view", n, "--json", "body", "-q", ".body"], cwd=REPO, capture_output=True,
                         text=True, check=True).stdout
    return out[:-1] if out.endswith("\n") else out


def main() -> int:
    args = sys.argv[1:]
    apply = "--apply" in args
    goal = args[args.index("--goal") + 1] if "--goal" in args else None
    n, line_file = args[0], args[1]
    line = pathlib.Path(line_file).read_text().strip()
    assert "\n" not in line, "the top line is one line"
    old = body(n)
    (OUT / f"{n}.before-carry.md").write_text(old)
    if line in old:
        print(f"#{n}: the body already holds the line; nothing to do")
        return 0
    new = old
    if goal and not GOAL_RE.search(new):
        m = LANE_RE.search(new)
        assert m, "no Lane line to put the goal after"
        new = new[:m.end()] + f"\n\nGoal {goal}" + new[m.end():]
    if new.startswith(MARK):
        rest = new[len(MARK):].lstrip("\r\n")
        new = f"{MARK}\n{line}\n\n{rest}"
    else:
        new = f"{line}\n\n{new}"
    f = OUT / f"{n}.carry.md"
    f.write_text(new)
    cmd = ["research", "queue", "edit", n, "--body-file", str(f)] + ([] if apply else ["--dry-run"])
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=170)
    print(f"#{n}: {' '.join(cmd[1:])} -> exit {r.returncode}")
    print("\n".join((r.stdout + r.stderr).strip().splitlines()[-6:])[:1200])
    if r.returncode or not apply:
        return r.returncode
    now = body(n)
    (OUT / f"{n}.after-carry.md").write_text(now)
    d = [x for x in difflib.unified_diff(old.replace("\r\n", "\n").splitlines(), now.replace("\r\n", "\n").splitlines(),
                                         lineterm="", n=0) if not x.startswith(("---", "+++", "@@"))]
    removed = [x for x in d if x.startswith("-")]
    print(f"#{n}: read back: {len([x for x in d if x.startswith('+')])} line(s) added, {len(removed)} removed")
    for x in d:
        print("   ", x[:240])
    if removed:
        print(f"#{n}: LINES WERE REMOVED: look at it")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
