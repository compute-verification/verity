"""Where restack.py's merges of each lacked move's base stand for HEAD as it is (before any script): prints the moves the
head lacks and, for each, whether HEAD merges with that move's base without conflict (the first that does not is where the
tool would stop at step `base`, unless an earlier move's script changes the answer)."""
import subprocess, sys
from pathlib import Path
sys.path.insert(0, "tools/research/src")
from research import queue as q
main, head = sys.argv[1], sys.argv[2]
repo = Path(".")
moves = q.moves_between(repo, main, head)
print(f"moves main has and the head lacks: {len(moves)}")
first = True
for commit, parents in moves:
    body = subprocess.run(["git", "log", "-1", "--format=%B", commit], capture_output=True, text=True).stdout
    line = q.MOVE_LINE_RE.search(body)[0]
    base = parents[0]
    r = subprocess.run(["git", "merge-tree", "--write-tree", "--name-only", "--no-messages", head, base], capture_output=True, text=True)
    files = r.stdout.strip().splitlines()[1:]
    print(f"  {commit[:12]} {line}: base {base[:12]}: {'no conflict' if r.returncode == 0 else f'CONFLICT in {len(files)} file(s)'}")
    if r.returncode and first:
        first = False
        for f in files:
            print("       ", f)
