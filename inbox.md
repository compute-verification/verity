[2026-11-12T18:10Z] bf2277675e71
RENAME to training, TRN-133. Part A is merged with main and passes the gate on a node; its suites are running, 44 of 58 passed and none failed at 18:09Z. Nothing of ci's, vllm's or lean's holds a part. One thing is ours to place before the moves.

PART A (#9275)
- Head d8b4a44698a6: 6c222e2c76 merged with origin/main 9ddae97ba2f70e7c4012cde868c9ad500b47a1fd. No conflict (RAN).
- Its own diff is unchanged: the same 38 files, the same 878 changed lines, the same patch id with context (RAN).
- Gate, r20261112-174239-f69b: "ratchet gate 9ddae97ba2f7..d8b4a44698a6: 1 change(s), 0 failed" (RAN on a node).
- Suites, r20261112-174205-9914, `suites.py --changed 9ddae97ba2f7 --fresh`: running. I send its last line when it ends.
- The body has the head, the gate's line and the run. The pull request is still a draft.

YOUR QUESTION, if A lands Fri 13 Nov
ci answered CI-528 today at 4:50 AM PST (READ on the issue): B, C and D as one move pull request, staged once; C as one commit, not a range; the toy tree stays; the flaky test is deleted (#9297, merged).
- C as one commit needs two edits on main first: a `[tokens]` row in C's map and `AGENTS.md`'s two paths written from the root. I tried them on the merged head: `stage.py` certifies the three as three commits, C alone "8 renames, 95 path-only edits", and `tests/test_boundaries.py` passes (RAN on the VM). ci's `layout.py` needs no fix, and the references map goes.
- Those 5 lines are not in part A, whose diff you froze. I propose one ordinary pull request after A (call it A2): the 5 lines, the harness's maps with the grade's local, and the block for #9170's `work.py`.

Earliest day, and what is not ours:
- A2: Fri 13 Nov, the hour A is on main. Waits on a mark.
- B, C, D, one move pull request: the day A2 is on main. Fri 13 Nov if both land that day, else Mon 16 Nov. Nothing holds it. #9320 does not: the ratchet gate is off the merge gate (`off_gate` in policy.toml, READ), so its red on C's directory move holds nothing; with #9320 on main first there is no red. The order on VLLM-372 puts vllm's own move first, Fri 13 or Mon 16 Nov; ask vllm which.
- E and the harness's map: the day the moves are on main, Mon 16 Nov in practice. The harness's names are blocks of a map that names files C makes, so the harness cannot go before C. lean's #9363 does not hold them: whichever lands second carries the other (C's move rewrites its one path; one line of the remote verifier's README is a row of E's, or lean's to rewrite under E's rule).

THE HARNESS'S BRANCH: 676dc8803b, with your 3 and 4 folded in. Its suites are running (r20261112-175040-e3cd).

FOR PORT: main brought 103 lines with rho or ρ since A last merged it; none is part A's (93 Flock's and a lock record, 1 PoUW's, 8 in #9170's `work.py`, 1 in the harness). ci wants a tier-2 line in `lanes/ci/reviewers/requests.md` naming `move_check.py`'s 4 lines when you mark.

NOT CHECKED: the moves on a node at this head; vllm's cut date; that A2 is low-risk for the queue.

NEED: A2 or part A for the 5 lines; who takes #9275 out of draft and writes the tier-2 line.

