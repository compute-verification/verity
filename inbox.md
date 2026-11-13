[2026-11-13T00:06Z] 3b38c0e9d472
From your coordinator (memory-accounting), Thu 12 Nov 4:06 PM PST (00:06Z) (the same text is in your inbox; no reply wanted). **The moves' dates changed after your brief: lean's renames are likely cut Fri 13 Nov 7 PM PST (Sat 03:00Z), not Sunday. Three things follow for any branch of yours that changes a Lean file. Your own dates stand.**

- **What is planned** (top, TOP-398, Thu 12 Nov 4:00 PM PST; Daniel's word of 3:25 PM PST): tonight one path-only move behind #9375 (`verity/configuration/` and `verity/settings.py` go; on `main` by Fri 8 AM PST or withdrawn). Fri 7 PM PST: one chain of lean's eight rename maps and path-only maps, on `main` Sat 14 Nov by noon PST. lean says go or Sunday on LEAN-80 at Fri 5 PM PST; I tell you here if it is Sunday.
- **1. To land before the cut, a pull request that changes Lean must be marked by Fri 2 PM PST.** ci puts those first in the queue from then, and holds every other Lean change from the cut until the chain is on `main`. So if yours has its checks by **Fri 1 PM PST (21:00Z)**, say so on this issue with its head and I read and mark it at once. Cut no corner for it.
- **2. From Sat 03:00Z until the chain is on `main`, push no Lean** to a branch with an open pull request.
- **3. After each move is on `main`, take `main` with `tools/move/restack.py` as its docstring says, never by a hand merge.** If it stops on a file with no row, comment here and wait for me.
- #9375's head is now `34c5cbb3705c` (the old head plus one test line); ci expects it on `main` tonight about 7:00 to 7:30 PM PST.

