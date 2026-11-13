[2026-11-13T22:15Z] dbf8ff25be19
GRADES, Fri 13 Nov 2:15 PM PST, to your 1:41 PM message: final.

**Draft #9833 is open against `main`. Its head is `fc4a5534aad80d44f23a551d07747ed80b91f32d`, `isDraft` is true, and its suites passed fresh at that head in `r20261113-214339-f634`.**

- RAN `research queue open ... --base main --draft` from a `main` checkout at `6c52a8dd9ca`, with `--sha 698da49f86e41995418b192374eaffd57d3acac1`. It printed two lines, both expected: the PR's url (https://github.com/compute-verification/verity/pull/9833), and that the PR is linked to TRN-157 through the Linear app. There was nothing else.
- RAN `gh pr view 9833 --json isDraft,baseRefName,headRefOid,title` after the edit: `isDraft` true, `baseRefName` `main`, `headRefOid` `fc4a5534aad80d44f23a551d07747ed80b91f32d`. The title is "training-slowdown: the command grades a config once every job of it and of its honest config has its record". The body has one `Lane training`, one `Goal training/attack-search-one-command` and one `Refs TRN-157`.
- I woke on a new machine (`uptime` 0 min). The body was gone from `/home/ubuntu/measure/`, but the artifacts copy held it. I rewrote it to your decision:
  - a paragraph of its own on what the command has been run on and what it has not; no registered config has been run through it on any machine, and its first whole run is the parked GPU run;
  - the measured figures as RAN with their run ids, and the scaled hours as INFERRED;
  - the `training-attack` finding (`phase = "gpu1"`, 60 minutes, at most one checkpoint a run on CPU).
  - No 838-step job was started.
- `main` had moved, from `fa3e619ee146` to `6c52a8dd9cae` (505 commits). READ: the path moves those commits mention (`5eb1231b2558`, `c6e3167fa310`, `0932f6acd0a8`) are all ancestors of my earlier merge base. The tree diff is 4 added and 65 modified files, with no rename and nothing under `benchmarks/training_slowdown`. So I merged it by hand: `fc4a5534aad`, authored and committed by the checkout's own identity, pushed.
- RAN `r20261113-214339-f634` (vy-nebius-cpu-14, fresh), `suites.py --changed 6c52a8dd9cae --fresh`. Last line: `4 suites: 4 passed (0 cached), 0 failed; 30.8 min of test time`. Then `ruff check benchmarks/training_slowdown`: `All checks passed!`.
- RAN on the VM, `ratchet_gate.py origin/main HEAD`: `ratchet gate 6c52a8dd9cae..fc4a5534aad8: 1 change(s), 0 failed`.
- RAN `research queue edit 9833 --body-file F`: the body is replaced, with the new run and head.

Nothing is in flight. If `main` moves before Sat 6 PM and you want it crossed, I need a new turn for it. I end here.

