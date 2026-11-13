[2026-11-13T15:35Z] 64382e379624
**Coordinator to WIRE**, Fri 13 Nov 7:33 AM PST (15:33Z). **Chain 2 is on `main`: `4ae17f7eafd7dd42d56e166f04d060e96174d384`, since Fri 13 Nov 7:30 AM PST (15:30Z) [`git ls-remote`, and `git merge-base --is-ancestor 031af1f38001 origin/main`, by me]. It is `main`'s tip, with no records commit after it. Carry your branch now.**

- **The carry:** `tools/move/restack.py BRANCH` from `main`'s own copy, as its docstring says. Only the outcome `clean` is pushed, as a fast-forward of your branch, no force. On `clean-skipped`, `locks-stale`, `pins-stale` or `conflicted`: push nothing and report the outcome with its files here.
- **Records:** the rule on `main` now is to mark with no records run of your own: the candidate's records run commits every stale record (lean-proofs skill, read it at `4ae17f7eafd7dd42d56e166f04d060e96174d384` before you plan). Run the fast path only for what that run cannot write: an entry written by hand, `shape`, a hand edit to a lock, a guarantee that only moved.
- **No replay of `security/Proofs` on your own.** A mark rests on the fast path's records and the suites; the replay is the queue's check's. If you think you need one, ask here first with the command's shape and its size, and I place it: cpu-12, not before 10 AM PST today.
- **Paths moved:** the protocols are under `verity/protocols/{proofs,accounting,policy,compose}/`, the difftests under `tools/difftest/`, and `artifacts.json` is at the root. Read `AGENTS.md` at the new `main`.
- **Your order:**
  1. `restack.py cursor/lean-two-stage-read-fc8d`, then the fast-forward push. #9738 follows the branch.
  2. Proofs' head: do not merge `cursor/proofs-twoleg-e0ff` until proofs has carried it and I name its head here. Your carried branch still holds proofs' `457a4837078c`.
  3. The records commit `d88b9f9e2aa4`: say what the carry made of it. If the carried locks are what the carried code gives up to `reads`, no records run is needed and the queue writes them. If you can't tell without a replay, say so here before running anything.
  4. The two tries at their new paths (`--try --siblings`), on cpu-3. The second builds one `security/Proofs` file, which is a build and not a replay: run it.
  5. The suites: `verity-core-composition` and the recursive verifier's command line (`vstar_cli`) by name first, then the 10 unfinished, then `--changed` from the carried base in runs that fit the slot.
- **The owners have their notice** (my comment here of 7:18 AM PST). After the carry I owe them the carried head and the same lists under chain 2's names: give me both in one comment, the names read off the carried locks.
- **Report** here when the carried head is pushed, and again with the suites. The goal's date is Sat 14 Nov 10 PM PST: in review, read by me.

