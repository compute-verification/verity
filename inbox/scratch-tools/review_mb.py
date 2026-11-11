"""`research queue review` against merge-base(checked, next): while `checked`'s tip is a commit `next` lacks (a lock
regeneration after landing), `checked` isn't below `next` and the stock command lists "0 of 0". Run from /tmp/rchk with
PYTHONPATH=/tmp/rchk/tools/research/src."""
import subprocess
import sys

from research import queue as Q
from research.cli import main

mb = subprocess.run(["git", "-C", "/workspace", "merge-base", "origin/checked", "origin/next"],
                    capture_output=True, text=True, check=True).stdout.strip()
Q.Queue.chain = lambda self: Q.chain_of(self.workdir, mb, f"refs/remotes/origin/{self.branch}")
sys.argv = ["research", "queue", "review", *sys.argv[1:]]
main()
