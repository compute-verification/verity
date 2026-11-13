"""The lines of circuit-check's source that name a module chain 2's served map moves to verity_vllm.definitions:
python3 served.py REV...  (run in the repository).  Each line is classed: import (a Python import statement), string (inside
a string literal on a code line), doc (a comment or docstring line) or data (a JSON file)."""
import re
import subprocess
import sys

SERVED = ("registry", "numerics", "boolean", "encoding", "evidence", "replay", "tests")
NAME = re.compile(r"verity\.definitions\.(" + "|".join(SERVED) + r")\b|from verity\.definitions import (" + "|".join(SERVED)
                  + r")\b|verity/definitions/(" + "|".join(SERVED) + r")\b")
SRC = "tools/verity/circuit_check/src/circuit_check"


def kind(path, line):
    s = line.strip()
    if path.endswith(".json"):
        return "data"
    if s.startswith("#"):
        return "doc"
    if re.match(r"(from|import)\s", s):
        return "import"
    if '"' in s or "'" in s:
        return "string" if not s.startswith(('"""', "'''")) else "doc"
    return "doc"


for rev in sys.argv[1:]:
    out = subprocess.run(["git", "grep", "-nE", NAME.pattern, rev, "--", SRC], capture_output=True, text=True).stdout
    rows = []
    for ln in out.splitlines():
        _, path, no, text = ln.split(":", 3)
        rows.append((kind(path, text), path.removeprefix(SRC + "/"), int(no), text.strip()[:150]))
    counts = {}
    for k, *_ in rows:
        counts[k] = counts.get(k, 0) + 1
    print(f"== {rev[:12]}: {len(rows)} lines {counts}")
    for k, p, n, t in rows:
        if k != "data":
            print(f"  {k:6} {p}:{n}  {t}")
