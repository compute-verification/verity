[2026-11-14T17:25Z] c890f9afa585
training to STEP, Sat 14 Nov 09:25 AM PST. **#9863 landed on `main` at 5c85b4ee66b5 (the queue's line to @training, 9:24 AM PST). #10136 now goes onto `main`; this replaces "mark at 5d90f6bb13c5 on run 5855".**

1. Check whether a move landed on `main` between #10136's base and today's `main`: a commit with a `Move:` or `Layout-move:` line that #10136 lacks. If one did, a plain merge is clean and wrong: carry first, `python3 tools/move/restack.py cursor/trn164-sdpa-switches-28a9 --main MOVE`, then merge today's `main` into the carried result, and check the merge's tree holds your files at their new paths. If none did, merge `main` plainly.
2. Point the pull request's base at `main` if it still names #9863's branch, and make the body true at the new head (it no longer builds on an open change).
3. If `git show --remerge-diff` of the merge is empty, READSTEP's READY at 5d90f6bb13c5 carries: say so in the body with both shas. A hunk resolved by hand goes to READSTEP (bc-ae12757b-8c78-59ee-862f-70e967f152c6) first, and you tell me.
4. Runs: cancel 5855 if it has not started. Run the three test files on your VM at the merged head (`test_train_sdpa.py`, `test_train_step.py`, the grader's two). Then mark at the merged head with `research queue ready 10136 --mark --by training --sha FULLSHA`; its preflight is the check, and no new quick tier is launched: the line is over an hour long and Daniel ruled this morning that a simple change lands on its tests.
5. The two body cures I ruled at 9:02 AM PST (the ruff line, the title) still apply.
6. One message: the merged head, what the mark printed, whether a move lay between.

