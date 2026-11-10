"""regen.py: rewrite the paths a word map names onto a tree that layout moves have changed, and check each rewrite.

    python3 regen.py --ref REF --moves MOVES [--moves MOVES ...] [--after AFTER.json] --out DIR MAP.toml...

REF is the commit the maps were written on (`git ls-tree` lists its files). Each MOVES is one move, applied in the order
given: a `layout.py` map (its `[paths]` are read) or a JSON list of [old, new] rows, a directory's ending in "/", the
longest old prefix winning (the root's rename is [["Security/", "security/"]]). AFTER.json, if given, maps a path an earlier map of the set writes to the path it has on
REF (the rehearsal's renames, new -> old), for a glob that names such a path.

For each path entry of each map (a scoped table's `files`, `[keep_in]`, `[keep_paths]`, and the `[rewrite]` globs that
name a moved place) it finds the files the entry matches on REF, moves them, and writes the entry with its literal
directory moved the same way. The rewrite stands only when the new entry matches, among REF's moved files, exactly
those: otherwise the entry is reported as `by hand`, with the files it would gain or lose, and left as it was.
Each map is written to DIR with only those strings changed; the maps given are not touched. It also lists the comment
lines and word rows that still spell a moved directory, which are hand edits.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import re
import subprocess
import tomllib
from pathlib import Path

WILD = re.compile(r"[*?\[]")


def mover(rows: list[list[str]]):
    dirs = sorted(((o, n) for o, n in rows if o.endswith("/")), key=lambda r: -len(r[0]))
    files = {o: n for o, n in rows if not o.endswith("/")}

    def move(p: str) -> str:
        if p in files:
            return files[p]
        for o, n in dirs:
            if p.startswith(o):
                return n + p[len(o):]
        return p
    return move


def entries(data: dict) -> list[tuple[str, str]]:
    out = [(f"words_in[{i}].files", g) for i, t in enumerate(data.get("words_in", [])) for g in t["files"]]
    out += [(sec, g) for sec in ("keep_in", "keep_paths") for g in data.get(sec, {})]
    rw = data.get("rewrite", {})
    out += [("rewrite." + k, g) for k in ("exclude", "include") for g in rw.get(k, [])]
    return out


def literal_prefixes(glob: str) -> list[str]:
    """The glob's literal directory prefixes, longest first, then the glob itself when it has no wildcard."""
    m = WILD.search(glob)
    lit = glob if m is None else glob[:m.start()]
    cuts = [i + 1 for i, ch in enumerate(lit) if ch == "/"]
    out = [lit[:c] for c in reversed(cuts)]
    return ([glob] if m is None else []) + out


def rewrite(glob: str, old: list[str], move) -> str | None:
    """`glob` with the longest literal prefix that every matched file leaves the same way replaced by where it goes."""
    for pre in literal_prefixes(glob):
        new = {move(p)[: len(move(p)) - (len(p) - len(pre))] for p in old if p.startswith(pre)}
        if len(new) == 1 and all(p.startswith(pre) and move(p).endswith(p[len(pre):]) for p in old):
            return new.pop() + glob[len(pre):]
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--moves", required=True, action="append")
    ap.add_argument("--after")
    ap.add_argument("--out", required=True)
    ap.add_argument("maps", nargs="+")
    a = ap.parse_args()
    steps, rows = [], []
    for m in a.moves:
        r = [list(kv) for kv in tomllib.loads(Path(m).read_text()).get("paths", {}).items()] if m.endswith(".toml") else json.loads(Path(m).read_text())
        steps.append(mover(r))
        rows += r

    def move(p: str) -> str:
        for step in steps:
            p = step(p)
        return p
    after = json.loads(Path(a.after).read_text()) if a.after else {}
    tracked = [p for p in subprocess.run(["git", "-C", a.repo, "ls-tree", "-r", "--name-only", a.ref], capture_output=True,
                                         text=True, check=True).stdout.split("\n") if p]
    moved = {p: move(p) for p in tracked}
    moved_files = sorted(set(moved.values()))
    old_dirs = sorted({o for o, _ in rows if o.endswith("/")}, key=len, reverse=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report = []
    for path in map(Path, a.maps):
        text = path.read_text()
        data = tomllib.loads(text)
        seen = set()
        for sec, glob in entries(data):
            if (sec.split("[")[0], glob) in seen:
                continue
            seen.add((sec.split("[")[0], glob))
            old = [p for p in tracked if fnmatch.fnmatchcase(p, glob)]
            via = ""
            if not old and after:
                old = sorted({after[p] for p in after if fnmatch.fnmatchcase(p, glob)})
                via = " (a path an earlier map writes)"
            image = {moved[p] for p in old}
            if not old:
                if not sec.startswith("rewrite."):
                    report.append((path.name, sec, glob, "", "matches nothing on REF", 0))
                continue
            same = {p for p in moved_files if fnmatch.fnmatchcase(p, glob)}
            if same == image and not via:
                if not sec.startswith("rewrite."):
                    report.append((path.name, sec, glob, glob, "stands", len(old)))
                continue
            new = None if via else rewrite(glob, old, move)
            got = {p for p in moved_files if fnmatch.fnmatchcase(p, new)} if new else set()
            if new and got == image:
                text, n = re.subn(r'(?<=")' + re.escape(glob) + r'(?=")', new.replace("\\", r"\\"), text)
                report.append((path.name, sec, glob, new, f"rewritten ({n} place{'s' if n != 1 else ''})", len(old)))
            else:
                why = "no one directory moves all its files" if not new else (
                    f"the rewrite {new!r} would gain {len(got - image)} and lose {len(image - got)} files"
                    + (": " + ", ".join(sorted(got - image)[:3]) if got - image else ""))
                dests = sorted({str(Path(p).parent) for p in image})[:4]
                report.append((path.name, sec, glob, "", f"by hand{via}: {why}; its files go to {dests}", len(old)))
        words = dict(data.get("words", {}))
        for t in data.get("words_in", []):
            words |= t["words"]
        for k, v in list(words.items()) + list(data.get("paths", {}).items()) + list(data.get("modules", {}).items()):
            if any(d.rstrip("/") in k or d.rstrip("/") in v for d in old_dirs) or "/" in k.strip("`"):
                report.append((path.name, "row", f"{k} -> {v}", "", "a row that spells a moved directory: by hand", 0))
        for s in data.get("keep", {}).get("patterns", []) + data.get("keep", {}).get("strings", []):
            if any(d.rstrip("/") in s for d in old_dirs):
                report.append((path.name, "keep", s, "", "a kept span that spells a moved directory: by hand", 0))
        comments = [i for i, line in enumerate(text.splitlines(), 1)
                    if line.lstrip().startswith("#") and any(d.rstrip("/") in line for d in old_dirs)]
        if comments:
            report.append((path.name, "comments", f"lines {comments}", "", "comment lines that spell a moved directory: by hand", 0))
        (out / path.name).write_text(text)
    width = max(len(r[0]) for r in report)
    for name, sec, glob, new, what, n in report:
        print(f"{name:<{width}}  {sec:<20}  {glob}" + (f"  ->  {new}" if new and new != glob else "") + f"  [{n} files] {what}")
    (out / "regen-report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    tally = {}
    for r in report:
        key = r[4].split(" (")[0].split(":")[0]
        tally[key] = tally.get(key, 0) + 1
    print("\n" + "; ".join(f"{k}: {v}" for k, v in sorted(tally.items())))


main()
