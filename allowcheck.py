"""allowcheck.py: tests/lint/allowlists/repo_retired_terms.json parses and lists no path twice."""
import collections
import json
import sys

p = "tests/lint/allowlists/repo_retired_terms.json"
try:
    d = json.load(open(p))
except Exception as e:
    print(f"ALLOWLIST UNPARSEABLE: {e}")
    sys.exit(1)
files = []


def walk(o):
    if isinstance(o, dict):
        if "file" in o:
            files.append(o["file"])
        for v in o.values():
            walk(v)
    elif isinstance(o, list):
        for v in o:
            walk(v)


walk(d)
dup = [f for f, n in collections.Counter(files).items() if n > 1]
print(f"allowlist: parses, {len(files)} entries, {len(dup)} paths twice" + (f": {dup[:5]}" if dup else ""))
sys.exit(1 if dup else 0)
