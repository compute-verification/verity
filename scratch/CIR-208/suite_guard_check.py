"""suite_guard_check.py WORK: hold every file one test-file run opened (the graph runner's guard.json for the test's process,
guard.reads for the Pythons it starts) to what its suite's guard allows (suites.py's inputs_of, dropped and covered), with the
suite's pyproject as the tree has it."""
import json, subprocess, sys
from pathlib import Path
sys.path.insert(0, "tools/verity/check")
import suites as S
work = Path(sys.argv[1])
suite = next(s for s in S.discover() if s.path == "tools/verity/circuit_check")
assert suite.inputs_only   # so its dependencies' directories are left out, and the packages map is not read
allowed = S.inputs_of(suite, {})
cut = S.dropped(suite, allowed)
tracked = set(subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True).stdout.split("\n"))
src = lambda p: p.replace("/__pycache__/", "/").rsplit(".cpython-", 1)[0] + ".py" if p.endswith(".pyc") else p
rep = json.loads((work / "guard.json").read_text())
own = {src(p) for p in rep["outside"]} | {src(t.split("\t", 1)[1]) for t in rep["traced"]}
child = {src(line.rstrip("\n").split("\t")[2]) for line in (work / "guard.reads").read_text().splitlines() if line[0] in "FO"}
print(f"suite {suite.path}: inputs_only, {len(allowed)} allowed paths, nested suites cut: {cut}")
for name, paths in (("the test's process", own), ("the interpreter it starts", child)):
    files = sorted(p for p in paths if p in tracked)
    out = [p for p in files if not S.covered(p, allowed, cut)]
    print(f"{name}: {len(files)} tracked files opened, {len(out)} outside the suite's inputs")
    for p in out[:40]:
        print("   ", p)
