[2026-11-13T08:20Z] a2ebec928f42
COUNTS to training, Fri 13 Nov 12:20 AM PST. **#9519 is ready for your read at `d198a01da9ef`: the records, both kernel replays and the 56 suites pass there, and the body is rewritten. Draft, not marked.**

Head `d198a01da9ef` = the branch merged with main `fa0877afdc12` (merged before your no-merge rule; I merge nothing more and carry it with `restack.py` when you say). Against main: 6 files, +133 −147; the lock +59 −6, all mine.

Each run, RAN by me at that head, with its last line:

- **Records** r20261113-045355-e95c, rc 0: `lean-records: OUTCOME {"outcome": "pass", "package": "security/Proofs", …}`. No record written: the committed lock is what the build makes.
- **Kernel replay, `security/Proofs`** r20261113-060040-608a, rc 0: "AUDIT: PASS; reports in …/lean-audit". Its audit: "PASS  204287 declarations in 4051 modules; axioms propext, Classical.choice, Quot.sound"; replay: "203587 constants of 3944 modules accepted by the kernel".
- **Kernel replay, `security`** r20261113-060040-e242, rc 0: "AUDIT: PASS; reports in …/lean-audit". Its audit: "PASS  27777 declarations in 730 modules; axioms propext, Classical.choice, Quot.sound; 1140 guarantees"; replay: "27413 constants of 697 modules accepted by the kernel".
- **Suites** r20261113-060041-1267 (`suites.py --changed fa0877afdc12`), rc 0: "56 suites: 56 passed (0 cached), 0 failed; 368.0 min of test time". The calculator's: 612 passed.
- **Ratchet gate** on my VM: "ratchet gate fa0877afdc12..d198a01da9ef: 1 change(s), 0 failed".

Your question, answered in the body in one line [READ]: `--update` makes the Flock move by itself; main's lock holds it since its own last records run, so the diff has no such hunk.

Not done: the try was not run again after the merge (main changed none of the three files; the replay builds them). compute-accounting has said nothing.

One thing for lean or infra, not for this PR: the Lean README's `--mem-gb 32` is far under what these runs reach. The replays' scopes peaked at 242.6 and 410.7 GiB.

