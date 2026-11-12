[2026-11-12T23:29Z] d4d83a5cb894
From the lead, Thu 12 Nov 3:29 PM PST, an addition to your brief (nothing in the design changes).

The Site's file may move to verity/device.py before your fold lands: top is weighing a path-only move of it tonight or Friday night (TOP-398, top's comment of 3:19 PM PST; nothing decided yet), instead of Sunday. So write the restage steps in your note for both orders:
(i) the fold lands first, on verity/hardware/site/__init__.py, as briefed;
(ii) the move lands first: your branch is carried across it with tools/move/restack.py (read tools/move/README.md for how a branch crosses a landed move), commit 1 is made again on verity/device.py, and the tests, rules and docs name the new path.
Keep finding the sheets from the verity package's directory, as briefed: it is what makes (ii) need no hand edit of that line. Do not wait for the decision; build on today's main as briefed.

