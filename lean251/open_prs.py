#!/usr/bin/env python3
"""Every open PR onto main, merged with a staged move's base: which pins, which files a pin records and which Lean locks
it still changes, so which heads `restack.py`'s pins step or lock step would change.  Git and `gh` (read-only) alone.

    open_prs.py BASE MOVE STAGED OUT.json [NAME=COMMIT ...]

BASE is the move's base, MOVE the move's commit, STAGED the head `restack.py` takes as `--onto`.  A NAME=COMMIT is
measured the same way beside the open PRs (a rehearsal branch).
"""
from __future__ import annotations

import collections
import importlib.util
import json
import os
import subprocess
import sys
import tempfile


def git(*a: str, ok: bool = False, timeout: float | None = None) -> subprocess.CompletedProcess:
    p = subprocess.run(["git", *a], capture_output=True, text=True, timeout=timeout)
    if p.returncode and not ok:
        raise SystemExit(f"git {' '.join(a)}: {p.stderr}")
    return p


def pins_of(commit: str) -> dict[str, tuple[list[str], bool]]:
    """`commit`'s pins: the pin's path -> (the files it records, whether its command builds Lean)."""
    src = git("show", f"{commit}:tools/verity/check/pins.py").stdout
    d = tempfile.mkdtemp()
    os.makedirs(d + "/a/b/c")
    with open(d + "/a/b/c/pins_at.py", "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("pins_at", d + "/a/b/c/pins_at.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["pins_at"] = m
    spec.loader.exec_module(m)
    out = {}
    for pin in m.PINS:
        text = git("show", f"{commit}:{pin.path}").stdout
        paths = ([ln.split("  ", 1)[1] for ln in text.splitlines() if ln.strip()] if pin.key is None
                 else list(json.loads(text)[pin.key]))
        out[pin.path] = (sorted({os.path.normpath(os.path.join(pin.under, p)) if pin.under else p for p in paths}),
                         bool(pin.builds))
    return out


def moved_path(base: str, move: str, path: str) -> str:
    """Where the move puts `path`."""
    for ln in git("diff", "-M", "--name-status", base, move).stdout.splitlines():
        f = ln.split("\t")
        if f[0].startswith("R") and f[1] == path:
            return f[2]
    return path


def main(argv: list[str]) -> int:
    base, move, staged, out = argv[:4]
    extra = [a.split("=", 1) for a in argv[4:]]
    pins = pins_of(base)
    lean_pin = {p for p, (_, b) in pins.items() if b}
    sum_pin = set(pins) - lean_pin
    lean_rec = {f for fs, b in pins.values() if b for f in fs}
    sum_rec = {f for fs, b in pins.values() if not b for f in fs}
    locks = {p for p in git("ls-tree", "-r", "--name-only", base).stdout.split("\n") if p.endswith("lean-audit.json")}
    new = {p: moved_path(base, move, p) for p in pins}
    after = set(git("diff", "--name-only", "--no-renames", move, staged).stdout.split("\n")) - {""}
    summary: dict = {
        "base": git("rev-parse", base).stdout.strip(), "move": git("rev-parse", move).stdout.strip(),
        "staged": git("rev-parse", staged).stdout.strip(),
        "pins": {p: {"lean_build": b, "records": len(fs), "moved_to": new[p],
                     "moved_by_the_script_unchanged": git("rev-parse", f"{base}:{p}").stdout == git("rev-parse", f"{move}:{new[p]}").stdout,
                     "rewritten_after_the_move": new[p] in after} for p, (fs, b) in pins.items()},
        "files_behind_a_lean_backed_pin": len(lean_rec), "files_behind_the_sha256sum_pin": len(sum_rec),
        "locks": len(locks),
        "locks_the_move_changes": sum(1 for ln in git("diff", "-M", "--name-status", base, move).stdout.splitlines()
                                      if ln.split("\t")[-1].endswith("lean-audit.json")),
        "locks_changed_after_the_move": sorted(p for p in after if p.endswith("lean-audit.json")),
    }
    prs = json.loads(subprocess.run(
        ["gh", "pr", "list", "--state", "open", "--limit", "1000", "--json",
         "number,headRefOid,baseRefName,isDraft,title"], capture_output=True, text=True, check=True).stdout)
    summary["open_prs"] = len(prs)
    prs = [p for p in prs if p["baseRefName"] == "main"]
    miss = [p for p in prs if git("cat-file", "-e", p["headRefOid"] + "^{commit}", ok=True).returncode]
    for i in range(0, len(miss), 40):
        git("fetch", "-q", "origin", *[f"refs/pull/{p['number']}/head" for p in miss[i:i + 40]], ok=True)
    rows = [{"name": f"#{p['number']}", "n": p["number"], "head": p["headRefOid"], "draft": p["isDraft"],
             "title": p["title"]} for p in sorted(prs, key=lambda p: p["number"])]
    rows += [{"name": n, "n": None, "head": git("rev-parse", c).stdout.strip(), "draft": False, "title": n}
             for n, c in extra]
    tmp = tempfile.mkdtemp()
    res = []
    for r in rows:
        if git("cat-file", "-e", r["head"] + "^{commit}", ok=True).returncode:
            res.append({**r, "error": "head not fetched"})
            continue
        try:
            mt = git("merge-tree", "--write-tree", "--name-only", r["head"], base, ok=True, timeout=240)
        except subprocess.TimeoutExpired:
            res.append({**r, "error": "merge-tree with the base took over 240 s"})
            continue
        lines = mt.stdout.split("\n")
        tree = lines[0]
        conf = [ln for ln in lines[1:lines.index("")] if ln] if mt.returncode == 1 else []
        ch = set(git("diff", "--name-only", "--no-renames", base, tree).stdout.split("\n")) - {""}
        row = {**r, "changed": len(ch), "base_conflicts": len(conf),
               "conflicted_pins_or_locks": sorted(set(conf) & (set(pins) | locks)),
               "lean_pin": sorted(ch & lean_pin), "sum_pin": sorted(ch & sum_pin),
               "lean_rec": sorted(ch & lean_rec), "sum_rec": sorted(ch & sum_rec), "locks": sorted(ch & locks),
               "pin_merge": {}}
        for p in sorted(ch & set(pins)):
            for name, ref in (("o", f"{tree}:{p}"), ("b", f"{move}:{new[p]}"), ("t", f"{staged}:{new[p]}")):
                with open(f"{tmp}/{name}", "w") as f:
                    f.write(git("show", ref, ok=True).stdout)
            mf = subprocess.run(["git", "merge-file", "-p", f"{tmp}/o", f"{tmp}/b", f"{tmp}/t"], capture_output=True)
            row["pin_merge"][p] = "conflict" if mf.returncode > 0 else "merges"
        res.append(row)
    with open(out, "w") as f:
        json.dump({"summary": summary, "rows": res}, f, indent=1)
    print(json.dumps(summary, indent=1))
    got = [x for x in res if x["n"] is not None and "error" not in x]
    clean = [x for x in got if not x["base_conflicts"]]

    def lean(x: dict) -> bool:
        return bool(x["lean_pin"] or x["lean_rec"])

    def summed(x: dict) -> bool:
        return bool(x["sum_pin"] or x["sum_rec"]) and not lean(x)

    def lock(x: dict) -> bool:
        return bool(x["locks"])

    print(f"open PRs onto main: {len(got)} measured of {len(rows) - len(extra)}; drafts: {sum(x['draft'] for x in got)}")
    print(f"merge the base with no conflict: {len(clean)}; in conflict (restack.py stops at `base`): {len(got) - len(clean)}")
    for name, f in (("change a Lean-backed pin or a file one records", lean),
                    ("change the sha256sum pin or a file it records, nothing Lean-backed", summed),
                    ("change a lean-audit.json", lock)):
        hit = [x for x in got if f(x)]
        print(f"{name}: {len(hit)}; of those that merge the base: {sum(1 for x in clean if f(x))}",
              [x["name"] + ("" if not x["base_conflicts"] else "*") for x in hit] if len(hit) <= 12 else "")
    print("a recorded file changed and its pin not:",
          [x["name"] for x in got if (x["sum_rec"] and not x["sum_pin"]) or (x["lean_rec"] and not x["lean_pin"])])
    print("pin merges with the staged head's:", dict(collections.Counter(
        (p.split("/")[-1], v) for x in got for p, v in x["pin_merge"].items())))
    for label, xs in (("all", got), ("merge the base", clean)):
        print(f"locks changed, {label}:", dict(collections.Counter(p for x in xs for p in x["locks"]).most_common()))
    for x in res:
        if x["n"] is None or "error" in x:
            print(x["name"], x["head"][:12], x.get("error") or
                  {k: (len(x[k]) if isinstance(x[k], list) else x[k]) for k in
                   ("base_conflicts", "lean_pin", "lean_rec", "sum_pin", "sum_rec", "locks")})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
