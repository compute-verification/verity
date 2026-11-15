"""`tools/check/check.py`, run with this one's arguments: watchers installed from before the move run the check from this path
(`--record`, `--preflight`). It goes once no installed watcher runs this path."""
import os
import sys

REAL = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "check", "check.py"))
os.execv(sys.executable, [sys.executable, REAL, *sys.argv[1:]])
