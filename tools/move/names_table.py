"""names_table.py W WORKTREE BASE: the table of a names_rehearse.sh run, one line a map, from its working directory W, the
commits BASE..HEAD of WORKTREE and the maps in W/moved. Rows are the map's own (words, or paths); sites are what its tool
counted rewriting; files are the files its commit changed, a moved file counted once."""
import re
import subprocess
import sys
import tomllib
from pathlib import Path

W, wt, base = Path(sys.argv[1]), sys.argv[2], sys.argv[3]


def git(*a: str) -> str:
    return subprocess.run(["git", "-C", wt, *a], check=True, capture_output=True, text=True).stdout


def rows(name: str) -> tuple[str, int]:
    d = tomllib.loads((W / "moved" / name).read_text())
    if d.get("paths"):
        return "path", len(d["paths"])
    n = len(d.get("words", {})) + sum(len(t.get("words", {})) for t in d.get("words_in", []))
    return ("prose" if d.get("scope") == "prose" else "word"), n


def sites(text: str) -> int:
    body = text.split("left standing")[0]
    n = sum(int(m[1]) for m in re.finditer(r"^\s+(\d+)\s+'.*' -> '.*'$", body, re.M))
    for k in ("paths", "imports", "package-relative paths", "relative paths", "anchors"):
        if m := re.search(rf"^  {k}: (\d+)$", body, re.M):
            n += int(m[1])
    return n


stage = (W / "stage.log").read_text()
sections = {m[1]: m[2] for m in re.finditer(r"^tools/move/(\w+\.toml): `python3 .*?$(.*?)(?=^tools/move/\w+\.toml: `python3 |^staged \d+ moves|\Z)",
                                            stage, re.M | re.S)}
print("| # | map | kind | rows | sites | files | commit | verdict |")
print("|---|---|---|---|---|---|---|---|")
i = 0
for line in git("log", "--reverse", "--format=%H\t%s\t%b%x00", f"{base}..HEAD").split("\0"):
    line = line.strip("\n")
    if not line:
        continue
    sha, subject, body = line.split("\t", 2)
    m = re.search(r"tools/move/(\w+_map\.toml)", subject + "\n" + body)
    if not m or subject.startswith("tools/move:"):
        continue
    name = m[1]
    i += 1
    kind, r = rows(name)
    files = sum(1 for s in git("diff", "--name-status", "-M", f"{sha}^", sha).split("\n") if s and s[0] != "D")
    if subject.startswith("Move by"):
        text = sections[name]
        c = re.search(r"move-check: certified \S+ is a pure move \((.*?)\)", text)
        again = "run again, it changes nothing" if "run again on it, it changes nothing" in text else "NOT RERUN"
        verdict = f"certified ({c[1]}); {again}" if c else "NOT CERTIFIED"
        kind = f"certified {kind} move"
    else:
        text = (W / f"{name}.out").read_text()
        verdict = "run again, it changes nothing" if (W / f"{name}.again.out").exists() else "not rerun"
        kind = f"ordinary {kind} map"
    print(f"| {i} | `{name.removesuffix('_map.toml')}` | {kind} | {r} | {sites(text)} | {files} | `{sha[:12]}` | {verdict} |")
