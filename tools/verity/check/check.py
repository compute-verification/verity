"""`tools/check/check.py`, run with this one's arguments: main's train.sh runs a candidate's check from this path and greps it
for what the check takes (`--bookkeeping-warn`, `DESELECT_FILE`). It goes once the architecture branch lands."""
import os
import sys

REAL = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "check", "check.py"))
os.execv(sys.executable, [sys.executable, REAL, *sys.argv[1:]])
