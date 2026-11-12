"""compose.py: CIR-259's Part 1 comment: comment-head.md, comment.py's table, comment-tail.md with its NOTROWS filled."""
import subprocess
import sys
from pathlib import Path

here = Path(__file__).parent
run = lambda *a: subprocess.run([sys.executable, str(here / "comment.py"), *a], check=True, capture_output=True, text=True).stdout
tail = (here / "comment-tail.md").read_text().replace("NOTROWS", run("--not").strip())
out = (here / "comment-head.md").read_text() + run() + tail
for w in ("TOKENSEL", "HEAD64B", "NOTROWS", "MISSING", "None"):
    assert w not in out, w
sys.stdout.write(out)
