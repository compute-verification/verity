"""The three fixes that make `layout.py --map tools/move/one14_map.toml`'s output a range move_check certifies; run at the
moved tree's root after layout.py, before committing (`regen.sh` does both).

1. A split certifier import keeps the original line's `# noqa: E402` on both halves (move_check matches each binding's
   comment).
2. A relative docstring path in a moved file stays as written (layout.py rewrites it to climb out of its directory, which
   move_check refuses); part 2 names those files from the root.
3. The four empty `__init__.py`s layout.py creates are left out (move_check refuses an added file); part 2 adds them.
"""
import re
import subprocess
from pathlib import Path

SPLIT = ["benchmarks/network_e2e/encodings/compile_slice_pool.py", "benchmarks/network_traces/egress_zk.py",
         "benchmarks/network_traces/scan_zk.py", "benchmarks/prove_path/served_statics_zk.py",
         "benchmarks/prove_path/served_tie_zk.py"]
RELATIVE = ["verity/ml/protocols/certifier/served_statics.circuit-check.py",
            "verity/ml/protocols/certifier/served_tie.circuit-check.py",
            "verity/ml/protocols/certifier/window_charge.py", "verity/ml/protocols/pous/answer_circuit.py",
            "verity/ml/protocols/pous/decode_circuit.py"]
INITS = ["verity/ml/protocols/__init__.py", "verity/ml/protocols/pouw/__init__.py",
         "verity/ml/protocols/pous/__init__.py", "verity/ml/protocols/certifier/__init__.py"]
IMPORT = re.compile(r"^from verity\.(ml|core)\.protocols\.certifier import ")
NOQA = "  # noqa: E402"

for f in SPLIT:
    p = Path(f)
    lines = p.read_text().split("\n")
    n = 0
    for i in range(len(lines) - 1):
        a, b = lines[i], lines[i + 1]
        if IMPORT.match(a) and IMPORT.match(b) and a.endswith(NOQA) != b.endswith(NOQA):
            lines[i if not a.endswith(NOQA) else i + 1] += NOQA
            n += 1
    if n != 1:
        raise SystemExit(f"fix_move: {f}: expected one split certifier import, found {n}")
    p.write_text("\n".join(lines))

for f in RELATIVE:
    p = Path(f)
    t = p.read_text()
    u = re.sub(r"`\.\./\.\./\.\./core/protocols/(certifier|pous)/", "`", t)
    if u == t:
        raise SystemExit(f"fix_move: {f}: no relative path climbing out of verity/ml/protocols")
    p.write_text(u)

for f in INITS:
    if Path(f).read_text() != "":
        raise SystemExit(f"fix_move: {f} is not the empty file layout.py creates")
    subprocess.run(["git", "rm", "-q", "--cached", f], check=True)
    Path(f).unlink()

subprocess.run(["git", "add", "-u", *SPLIT, *RELATIVE], check=True)
print(f"fix_move: {len(SPLIT)} split imports keep their noqa, {len(RELATIVE)} relative paths kept, {len(INITS)} __init__.py left out")
