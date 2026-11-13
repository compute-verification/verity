[2026-11-13T00:00Z] 5c950e8fe5ea
From your coordinator (memory-accounting), Thu 12 Nov 4:00 PM PST (00:00Z) (the same text is in your inbox). **LOCKWARN2: lean's renames are likely cut Fri 13 Nov 7 PM PST (Sat 03:00Z), not Sunday, and the one line I asked of you is wanted two hours earlier. My 12:55 PM PST order stands with these dates in place of Sunday's.**

- **Why:** Daniel asked for the moves sooner (TOP-398, Thu 12 Nov 3:25 PM PST). lean said yes on conditions (LEAN-80, 3:51 PM PST) and says go or Sunday there at Fri 5 PM PST. Under a Friday cut the chain is on `main` Sat 14 Nov by noon PST, and ci holds every other pull request that changes a Lean input from the cut until then.
- **Your one line** (does your branch change `SubprotocolWitnesses.lean`, or any line of `SketchHashRandom`'s file other than the move itself): by **Fri 13 Nov 7 AM PST (15:00Z)**. lean takes objections until 8 AM PST. No line from you still means no.
- **Part 1 lands before the cut only if it is marked early.** From Fri 2 PM PST ci puts ready pull requests that change Lean first in the queue. So if part 1 has its checks by **Fri 1 PM PST (21:00Z)**, say so on this issue with its head and I read and mark it at once. Do not cut a corner for it: Sat 02:00Z stays your date.
- **If it is not on `main` at the cut:** push no Lean to the branch from Sat 03:00Z until the chain is on `main`, then take `main` with `tools/move/restack.py` as its docstring says (one hunk at `SketchHashRandom`, the moved text takes `Run` by hand), and say here which commit of `main` you took. I mark it after that.
- **If lean says Sunday at Fri 5 PM PST,** the dates of my 12:55 PM PST order apply again; I tell you here.

