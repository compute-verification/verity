"""Exact replacements in a pull request's body, the rest kept as it is.

    python3 bodyedit.py N EDITS.py [--apply]

EDITS.py defines `EDITS`, a list of (old, new) strings. The body is read from GitHub now; each `old` must occur exactly once
in it, or nothing is done. The `Lane`, `Refs`, `Goal` and `Issue` lines must be the same lines before and after. The result
goes to `research queue edit N --body-file F` (`--dry-run` unless `--apply`); after `--apply` the body is read back and held
to the text that was sent. The body as it was is kept under /tmp/carrier/bodies/N.before-edit.md.
"""
import pathlib
import re
import runpy
import subprocess
import sys

REPO = "/agent/repos/verity"
OUT = pathlib.Path("/tmp/carrier/bodies")
KEPT = re.compile(r"^[ \t>*_]*(Lane|Refs|Goal|Issue)\b.*$", re.M)


def body(n: str) -> str:
    out = subprocess.run(["gh", "pr", "view", n, "--json", "body", "-q", ".body"], cwd=REPO, capture_output=True,
                         text=True, check=True).stdout
    return (out[:-1] if out.endswith("\n") else out).replace("\r\n", "\n")


def main() -> int:
    n, edits_file = sys.argv[1], sys.argv[2]
    apply = "--apply" in sys.argv[3:]
    edits = runpy.run_path(edits_file)["EDITS"]
    old = body(n)
    (OUT / f"{n}.before-edit.md").write_text(old)
    new = old
    for i, (a, b) in enumerate(edits, 1):
        if new.count(a) != 1:
            print(f"#{n}: edit {i}: its old text occurs {new.count(a)} time(s), not once; nothing done: {a[:90]!r}")
            return 2
        new = new.replace(a, b)
    if KEPT.findall(old) != KEPT.findall(new) or [m.group(0) for m in KEPT.finditer(old)] != [
            m.group(0) for m in KEPT.finditer(new)]:
        print(f"#{n}: a Lane, Refs, Goal or Issue line would change; nothing done")
        return 2
    f = OUT / f"{n}.edit.md"
    f.write_text(new)
    cmd = ["research", "queue", "edit", n, "--body-file", str(f)] + ([] if apply else ["--dry-run"])
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=170)
    print(f"#{n}: {len(edits)} edit(s); {' '.join(cmd[1:])} -> exit {r.returncode}")
    if r.returncode:
        print("\n".join((r.stdout + r.stderr).strip().splitlines()[-6:])[:1500])
        return r.returncode
    if not apply:
        return 0
    now = body(n)
    (OUT / f"{n}.after-edit.md").write_text(now)
    print(f"#{n}: read back: {'the body is the text sent' if now.rstrip() == new.rstrip() else 'THE BODY DIFFERS FROM THE TEXT SENT'}")
    return 0 if now.rstrip() == new.rstrip() else 1


if __name__ == "__main__":
    sys.exit(main())
