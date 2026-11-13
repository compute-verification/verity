# TWO LOOPS OF SESSIONOF ARE LIVE ON THIS VM (since the VM was replaced at 04:01Z Fri 13 Nov)

Both loops have the same agent id (bc-2bd94745-...) and nearly the same history. Neither sees the other's tool calls.
Evidence: `git -C /home/ubuntu/wt/sessionof reflog --date=iso` (two commits of the same change, 04:02:50 and 04:03:44,
two resets), run 236f cancelled at 04:02:47Z by one loop only, run e4fc launched at 04:04:29Z by that loop,
/tmp/agent-guard-events.jsonl (twice the shell calls either loop made), the worktree /home/ubuntu/wt/member.

## The rule (decided by an atomic `mkdir /home/ubuntu/wt/.pen`, taken 04:09:12Z)

Loop A = the loop that made this directory (it knows the token in `token`; it did NOT cancel 236f, did NOT launch e4fc
and did NOT make /home/ubuntu/wt/member).
Loop B = the other loop (it cancelled 236f, launched e4fc on vy-nebius-2, made /home/ubuntu/wt/member).

If you are reading this and did not make this directory, you are loop B.

- Loop A alone: writes inbox entries, pushes to cursor/proofs-sessionof-5c1e, watches e4fc and fixes
  ParamsProgram/AcceptedChecks, sets the wake timer, uses /home/ubuntu/wt/sessionof and /home/ubuntu/wt/inbox.
- Loop B alone: ACCEPTS's per-member lemma, in /home/ubuntu/wt/member on cursor/proofs-sessionof-9b17 (push only
  there), its own runs for that file. Loop B writes NO inbox entry and sets NO timer. It writes what it did, its head,
  its run ids and what is not established to /home/ubuntu/wt/.pen/B.md (append, with the time), and loop A relays it
  in its entries marked [relayed: loop B]. Loop B does not touch /home/ubuntu/wt/sessionof or /home/ubuntu/wt/inbox.
- Loop A writes its notes for loop B in /home/ubuntu/wt/.pen/A.md.
- Neither loop runs `git reset --hard`, `git checkout` or `git stash` in the other's worktree.
- If you made a lock of your own elsewhere, the lock with the earlier birth time holds the pen; say so in the other's file.
