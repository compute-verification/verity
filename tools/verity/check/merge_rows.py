"""`tools/check/merge_rows.py`, run with this one's arguments: clones made before the move name this path as git's merge driver
(`research.clone`), and a driver that fails leaves ours unmerged with no conflict markers. It goes once every clone registers
`tools/check/`'s driver."""
import os
import sys

REAL = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "check", "merge_rows.py"))
os.execv(sys.executable, [sys.executable, REAL, *sys.argv[1:]])
