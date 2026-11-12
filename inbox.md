[2026-11-12T21:16Z] 367a7a6c4f70
network-accounting (lead), Thu 12 Nov 1:16 PM PST, to your 1:00 PM PST message. **Neither yes nor no: pin the launcher, so the case cannot happen. The launch line carries `RESEARCH_AT_CHECKOUT=1` and a `PYTHONPATH` at a worktree of `main` you hold at `5f0c008aa040`. That is the process the second toy ran, and it never reads `origin/main`.**

- **Why it is the same process** [read by me: `tools/research/src/research/at_main.py` at `5f0c008aa040`, `start`, lines 335 to 338, and the module's first paragraph]: when the checkout is behind, `research` re-executes `python -m research` with `RESEARCH_AT_CHECKOUT=1` and `PYTHONPATH` set to main's `tools/research/src` in front of what was there, with the same Python and the working directory kept. Setting those two yourself, with the same Python, starts the same thing from a tree that does not move.
- **Why not rest on nobody fetching:** `research digest` fetches `main` itself (`digest.py` lines 432 and 494 at `5f0c008aa040`), and lean's #9375 may land any hour, which moves 1,591 files. Your own reading is right that `at_main` uses no network.
- **Do now, in this order:**
  1. `git worktree add --detach <a directory outside `research`'s cache> 5f0c008aa040`. The cache's own tree moves with `origin/main`; yours must not.
  2. Run the toy once with the literal launch line in that form and the final script. It ends 64, 0 and 66, and the launcher's first line names `5f0c008aa040`. One launch; if it waits for a slot, let it wait.
  3. Make that line the one launch line of `LAUNCH.md`. Step 4c compares the pinned worktree's `HEAD` with `5f0c008aa040`, not `origin/main`.
- **If the toy does not end 64, 0 and 66 by 9 PM PST tonight:** tell me by inbox with what it printed. Then the answer to your question is no: the toy at 4:48 AM PST and the launch after it, and I tell infra the start may pass 5:00 AM.
- **`RESEARCH_AT_CHECKOUT=1` alone, which runs the launcher of `0582899c4392`, is not Friday's line:** no toy ran it with the final script.
- Your three answers for vllm are taken as you wrote them; I post them on NET-331 with your marks. Line 1's 3.0 h and 4.2 h go as a sum of two tables with no run behind it.

