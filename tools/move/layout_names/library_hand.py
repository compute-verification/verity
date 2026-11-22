#!/usr/bin/env python3
"""The references to `input_datasets/` that `tools/move/layout.py --map library_datasets_map.toml` doesn't read, from the
root of a moved checkout: the layout of record's entry (with `library/`'s own), the import listing's place, AGENTS.md's
map, the member's wheel package (which layout.py rewrites as a root path), the root directories `test_boundaries.py`
forbids, and the whole directory named as data (a PYTHONPATH entry, a suite's input, a path list). Prints each file it
changed; refuses a required edit whose old text is not there once."""
import re
import subprocess
from pathlib import Path

NEW = "library/datasets"
#: (file, old, new): each must match exactly once
REQUIRED = [
    ("AGENTS.md", "- `input_datasets/`:", f"- `{NEW}/`:"),
    (f"{NEW}/pyproject.toml", f'packages = ["{NEW}"]', 'packages = ["input_datasets"]'),
    ("tests/import_order.toml", '    ["input_datasets"],\n', f'    ["{NEW}"],\n'),
    ("tests/tree.toml", '[[tree]]\npath = "input_datasets/"\n',
     '[[tree]]\npath = "library/"\nmeans = "what applications and benchmarks share and Verity never looks at"\n\n'
     f'[[tree]]\npath = "{NEW}/"\n'),
    ("tests/test_boundaries.py", '"applications", "input_datasets")', '"applications", "input_datasets", "library")'),
]
#: (file glob, regex, new): the directory as data, wherever a tip has it
OPTIONAL = [
    ("**/*.sh", r"(?<=\$TREE/)input_datasets(?=[:\"'\s])", NEW),
    ("tests/adapters/vllm/tests/pyproject.toml", r'(?<=^inputs = \["adapters/vllm", "verity", ")input_datasets(?=")', NEW),
    ("tests/tools/research/test_pythonpath.py", r'(?<=")input_datasets(?=")', NEW),
    ("tests/adapters/vllm/tests/tools/test_h100_smoke.py", r'(?<=PATH_PARTS = \()(.*?)"input_datasets"', None),
]

changed = set()
for f, old, new in REQUIRED:
    p = Path(f)
    text = p.read_text()
    if text.count(old) != 1:
        raise SystemExit(f"library_hand.py: {f} holds {text.count(old)} of {old!r}, not one: edit it by hand and report")
    p.write_text(text.replace(old, new))
    changed.add(f)
for glob, rx, new in OPTIONAL:
    tracked = subprocess.run(["git", "ls-files", "--", glob], capture_output=True, text=True, check=True).stdout.split()
    for p in map(Path, tracked):
        if not p.is_file() or p.is_symlink():
            continue
        text = p.read_text()
        if new is None:
            out = re.sub(rx, lambda m: f'{m[1]}"{NEW}"', text, flags=re.M | re.S)
        else:
            out = re.sub(rx, new, text, flags=re.M)
        if out != text:
            p.write_text(out)
            changed.add(str(p))
for f in sorted(changed):
    print(f)
