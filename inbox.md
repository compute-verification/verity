[2026-11-12T23:45Z] 69ffc1c9fe7c
From the lead, Thu 12 Nov 3:45 PM PST. Tonight's move is on (TOP-398, top's comment of 3:27 PM PST; Daniel, 3:25 PM PST: as much of the path-only work as possible by tonight or tomorrow morning). If top's chain is on main by Fri 8 AM PST, the Site's file is verity/device.py from Friday morning; if it misses 8 AM it waits for Fri 7 PM PST. So order (ii) of my last note is the expected one. Keep building on today's main as briefed; your branch crosses the move with tools/move/restack.py.

Three facts for the fold:
1. top said yes to both rule changes: the loaders join the unit core.site, and the Site's file is in the public set.
2. NUMPY_FREE (tests/test_boundaries.py:214) lists "hardware". top's condition: the folded file stays free of numpy and stays on that list at its new path. Say in your note how the folded file meets it today, and what the list needs once the file is verity/device.py.
3. proofs said yes to the loaders being added to the Site's file. Their Site v2 (branch cursor/proofs-site-v2-1dc6 at a88978bf950f, no pull request yet) changes verity/hardware/site/__init__.py itself, site/tests/test_site.py and nodes/__init__.py. Whichever lands second merges. Run git merge-tree of your head against that head and list the conflicts in your note, file by file. Resolve nothing.

