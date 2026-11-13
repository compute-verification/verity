"""New-side changed lines per file of `git diff -U0 BASE HEAD`, as ranges, and deleted-only hunks as 'deleted at N':
python3 lines.py BASE HEAD"""
import re
import subprocess
import sys

base, head = sys.argv[1:3]
out = subprocess.run(["git", "diff", "-U0", base, head], capture_output=True, text=True, check=True).stdout
files: dict[str, list[str]] = {}
cur = None
for ln in out.splitlines():
    if ln.startswith("+++ "):
        cur = ln[6:] if ln != "+++ /dev/null" else None
        if cur:
            files.setdefault(cur, [])
    elif ln.startswith("@@") and cur:
        m = re.match(r"@@ -\d+(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", ln)
        start, n = int(m.group(2)), int(m.group(3) or 1)
        if n == 0:
            files[cur].append(f"one deleted at {start}" if (m.group(1) or "1") == "1" else f"{m.group(1)} deleted at {start}")
        else:
            files[cur].append(str(start) if n == 1 else f"{start}-{start + n - 1}")
for f, r in files.items():
    print(f"{f}: {', '.join(r)}")
