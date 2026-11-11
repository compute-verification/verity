#!/usr/bin/env python3
"""checkpoint.py: stdin is one checkpoint (a `- HH:MMZ: ...` item); it goes into the note before `## How each crossing was run`."""
import os, sys, textwrap
p = os.path.expanduser("~/.research/notes/lanes/circuits/20261111T0117Z-report-cir223-cross2.md")
s = open(p).read()
mark = "\n## How each crossing was run\n"
assert s.count(mark) == 1
text = " ".join(sys.stdin.read().split())
assert text.startswith("- ")
item = "\n".join(textwrap.wrap(text, 118, subsequent_indent="  ", break_long_words=False, break_on_hyphens=False))
s = s.replace(mark, "\n" + item + "\n" + mark)
open(p, "w").write(s)
print(item)
