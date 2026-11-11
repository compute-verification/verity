[2026-11-11T14:21Z] c255e03beb03
From compute-research's lead, Wed 11 Nov 6:22 AM PST: the follow-up your CMP-406 brief promised. **lean answered on CMP-392 at 6:19 AM PST: yes. Part 2 now includes lean's witness and the definition's move.** This replaces the brief's default ("`Accounting.h200S8Price` stays, lean's witness is not touched"). Everything else in the brief stands. Keep going with what you are doing and fold this into part 2.

**What part 2 now does, beyond taking `Bridge.WorkOfH1CheckedCreditPeriodH200` off the Properties list:**

1. **Restate lean's witness `CommOfPeriodLeakAccepted_nontrivial` at the constant price `fun _ => 1`.** In `security/Proofs/Accounting/JointBridge/CommOfPeriodLeakAccepted.lean` the price `Accounting.h200S8Price 0 0` appears twice: in the statement (line 156 on `main`) and in the proof's `hpr` (line 161). Both become the constant. `Price` is `OpClass → ℕ`, so the constant is one at every op class. The proof closes on `fun h => h` and reads no price, so it should stand as written.
2. **Move `Accounting.h200S8Price` out of `security/Definitions/Accounting/JointBridge/H200.lean` to beside the proofs**, with our H200 statement and what `HonestH200.lean` needs. Whether `w1Price` in the same file goes with it depends on what else under the definitions tree or a listed statement reads it: find out and say which. Fix the docstring in `security/Definitions/Accounting/Assumptions/CreditedWords.lean` (line 58 on `main`) that names `Accounting.h200S8Price`, so that no text under the definitions tree names a definition that is no longer there.

**lean's three conditions. Each is a done-condition of yours:**

- **Both places change and nothing else in that file.** The witness's docstring names no price and stays as it is. The diff of `CommOfPeriodLeakAccepted.lean` is those two expressions and nothing more.
- **The record stays `@lean`'s**, and the `witnesses` entry of `Accounting.SecurityProofs.Bridge.CommOfPeriodLeakAccepted` in `security/lean-audit.json` keeps naming `CommOfPeriodLeakAccepted_nontrivial`. Check both in the lock `--update` writes: owner unchanged, the entry unchanged, only the witness's signature (and what the record derives from it) different. If `--update` changes the owner or drops the entry, stop and tell me.
- **The landing message shows the changed statement.** Run `research.spec_alert` on `main`'s lock and the branch's, as you did for #8972, and put its line for this witness, with the review's before and after of the statement, in the description under what reaches Daniel. lean reads the head before I mark it, so the description's text for this witness must be exact.

**Lock changes this adds to what you post on CMP-392:** in `security`'s lock the witness's record changes its statement and `reads` loses `Accounting.h200S8Price` (the group of `Definitions.Accounting.JointBridge.H200` changes or goes); in the Properties lock the same group loses what only our H200 statement read. lean read on `main` at `1a59728741f9` that in `security`'s lock this witness is the only listed statement whose record names `h200S8Price`. Check that it still holds on your branch, and for the Properties lock too, before you move the definition. If another listed statement reads it, do not move it: tell me which.

**A separate commit** for lean's witness and the definition's move, apart from the commit that takes our statement off the list. Its title says it is lean's witness restated with lean's word (CMP-392, Wed 11 Nov 6:19 AM PST).

**In your report, add:** the diff of `CommOfPeriodLeakAccepted.lean` in full (it is two lines), the witness's record before and after, and where `h200S8Price` and `w1Price` now live.

`main` has moved since the brief (lean read it at `1a59728741f9`). Name the `main` commit your branch holds in your report.

