"""In-repo names a file reads that its module doesn't define: `from M import n` and `X.n` for X an in-tree module, n
holding a family word or its new spelling. One line per miss; run in a tree and compare runs."""
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/tmp/kit")
from pyrefs import Refs  # noqa: E402

FAM = re.compile(r"(?i)draw|drew|sampl|flip|planpoint|design_of")
refs = Refs(Path("."), subprocess.run(["git", "ls-files", "*.py"], capture_output=True, text=True).stdout.split())
for f in refs.files:
    for kind, alias, targets, name, line in refs.uses(f):
        if kind != "mod" and FAM.search(name) and not any(name in refs.defines(m)[0] or refs.defines(m)[1] for m in targets):
            print(f"{f}:{line}: {kind} {alias}.{name}: not defined in {targets[0]}")
