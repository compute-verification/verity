"""The Python string texts a map's applied run changed, for its `[keep] strings`: run in the tree the map was applied in
(staged), against HEAD. A string or f-string piece that is one identifier follows the names it spells (`__all__`, a
`monkeypatch` attribute); any other text keeps its bytes. Docstrings are prose and are left to the map.

usage: python3 strings_hold.py TREE MAP  (rewrites MAP's `strings = [...]` line under [keep])
"""
import io
import json
import re
import subprocess
import sys
import tokenize
import tomllib

tree, map_path = sys.argv[1], sys.argv[2]


def pieces(src: str) -> list[tuple[str, bool]]:
    lines = src.splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))
    out = []
    for t in tokenize.generate_tokens(io.StringIO(src).readline):
        if t.type not in (tokenize.STRING, tokenize.FSTRING_MIDDLE):
            continue
        a = starts[t.start[0] - 1] + t.start[1]
        b = starts[t.end[0] - 1] + t.end[1]
        text = src[a:b]
        doc = t.type == tokenize.STRING and text.lstrip("rbfuRBFU")[:3] in ('"""', "'''")
        out.append((text, doc))
    return out


files = subprocess.run(["git", "-C", tree, "diff", "--cached", "--name-only", "--", "*.py"], capture_output=True,
                       text=True, check=True).stdout.split()
hold = []
for f in files:
    old = subprocess.run(["git", "-C", tree, "show", "HEAD:" + f], capture_output=True, text=True, check=True).stdout
    new = open(f"{tree}/{f}").read()
    po, pn = pieces(old), pieces(new)
    if len(po) != len(pn):
        sys.exit(f"{f}: the run changed how many strings it has")
    for (a, doc), (b, _) in zip(po, pn):
        if a == b or doc:
            continue
        inner = a.strip("rbfuRBFU\"'")
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", inner):
            continue
        for part in [p for p in a.splitlines() if re.search(r"(?i)draw|drew", p)]:
            hold.append(part.strip())
hold = sorted(set(h for h in hold if h))
text = open(map_path).read()
m = re.search(r"^strings = (\[.*\])$", text, re.M)
have = tomllib.loads(text).get("keep", {}).get("strings", [])
line = "strings = " + json.dumps(sorted(set(have) | set(hold)), ensure_ascii=False)
text = text[:m.start()] + line + text[m.end():] if m else text.replace("[keep]\n", "[keep]\n" + line + "\n", 1)
open(map_path, "w").write(text)
print(f"{len(hold)} string texts held, {len(set(have) | set(hold))} in all")
