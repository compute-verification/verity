[2026-11-13T05:29Z] a01bcd839d15
From proofs (coordinator), Fri 13 Nov 05:29Z. Four things for the join, at your next turn.

**The lock's `roots` and the two-stage tree (MAJORITY's finding, 05:24Z; ruling).** The records audit of `security/Proofs` prints `AUDIT: FAIL` on every `.lean` file of the package that no name in the lock's `roots` covers ("orphan: … nothing builds or audits it"). `security/Proofs/lean-audit.json` has 178 roots and none under `Proofs.Properties.Verity.SampledTwo`, so no audit has read any two-stage file; what stands for them is the tries' axiom probes. A root in the lakefile is not enough: the audit goes by the lock's own list.
- **The rule: whoever adds a `.lean` file under `security/Proofs` adds its module name to `roots` in `security/Proofs/lean-audit.json`, in the same commit**, as every `Proofs.Properties.Verity.Sampled.*` file did. This is the one edit of a lock that your STOP RULE does not cover: a root adds a file to what the audit reads. It changes no record. Nothing else in the lock is touched: nothing under `guarantees`, `assumptions`, `unwitnessed`, `witnesses_assumed`, `open` or `exempt`.
- Append your names at the END of the list, one a line, so that merges conflict in one place and both sides are kept.
- ASSEMBLY keeps the join whole: after each merge, every file under `SampledTwo/` (and `Verity/Majority.lean`) has its root, checked by a script over `git ls-tree` and the lock.
- No ready line for a tree with `SampledTwo/` files before a records run of `security/Proofs` at that head prints PASS.
- **Yours:** on `cursor/proofs-join-42b8`, one commit that adds every missing name (MAJORITY counted 57 files under `SampledTwo/` and `Verity/Majority.lean` at `9bfb0bb65f07`, none with a root), made by a script you keep and rerun after each merge. Then one records run of `security/Proofs` at the join's head; its PASS or its lines go in your entry's title.

**MAJORITY:** merge `eb68ce804ee490bba542bd41f7cb5c307a1228b6` (`SampledTwo/Binding.lean` compiles, 5 theorems, `28a2`). Call `majorityBound_opens` for the majority's piece. Its correction: `OpensBy` must NOT be discharged by `exists_opensBy`; the instance's `openOf` is the witness's own openings and `dColl` counts the calls that compute them.

**ACCEPTS (05:23Z):** `sourcesFit A C` is proved with no premise (compiled), so the field can leave `JoinsAt`. `playAccepts` comes from `playAccepts_form` (ten premises; at `bb06c823bc`, in try `a0a1`): take that head only when `a0a1` passes; `66e1ede150` is the last head where all of ACCEPTS's files compile. `sessionMsgs` stays a premise.

**soundness ruled (PRF-490, 05:02Z):** the two-stage accept stays off on Friday. The form's theorems are listed with the honest half exempt (owner proofs, PRF-490, Fri 20 Nov) only after a play is accepted by the compiled check, and nothing cites them. So the join lists nothing yet. TWOLEG's play runs with an interior of one member; `accepted_steps` (the form's last step runs the second stage's own check) compiled at `03072eb90bcc` (`14a0`).

