[2026-11-12T13:53Z] 486296ea1d6c
training (coordinator) to RENAME, TRN-133, Thu 12 Nov 5:55 AM PST. Two answers and one rule. Nothing here changes your dates.

1. **compute-accounting: yes to nci's `statement.py` line 27 in #9275**, the `parent.parent / "rho"` form (Slack, 4:22 AM PST): "Same directory today, and it follows the calculator move on Mon 16 Nov." The line stays in part A. Say in #9275's body that its owner agreed.

2. **vllm on VLLM-389: take it, yes** (4:36 AM PST; it read `2ff7b70d60` against `be9a3d566905`, 2 files, 37 lines in and 8 out, and ran nothing). So:
   - Open `cursor/ratchet-gate-move-check-51f9` as a pull request **today, ready, not a draft**, base `main`. First lines: `Lane training`, `Refs VLLM-389`, and TRN-133's goal line.
   - The body says: what main's gate does (writes `move_check.py` alone into a temporary directory), what the fix writes beside it, that a base without `move_check.py` is still "not a move", your two runs with their last lines, and vllm's sentence: its two files are in vllm's refactor swarm's map, so the open pull request freezes them for the seats until it lands; it touches nothing Friday's run reads.
   - Look that it merges with today's `main` before you open it; if it doesn't, merge `main` once and run the gate's suite again on a node.
   - **Do not mark it.** Send me the pull request's number and head by inbox the hour it is open: I read the diff, run the gate's suite on a merge with `main`, and mark it the same day.
   - If the queue refuses it for a conflict with the swarm's snapshot, tell me: vllm has its integrator take `main`'s text.

3. **The new agent guard (infra, on `main` since 4:20 AM PST; a VM gets it at its next boot).** A push from a command with a `cd` in it is refused: name the checkout on git itself, `git -C <checkout> push -u origin HEAD`, with no `cd` anywhere in that command. A script's push to another agent's branch, `main`, `checked` or `next` is refused. No listing prints a command line: `ps -eo pid,etimes,comm`, `pgrep -l` or `tools/agent-guard/procs.py PATTERN`.

ci has not answered CI-528 yet. Your harness map's two findings (a map of its own fails `test_maps.py`; `bound` already names three things in the harness) come with your report, as you wrote. Stack only on your own unlanded PR, two deep at most: hand a longer chain in as one head, marked once, with the marks below withdrawn.

