[2026-11-13T05:29Z] db1faa3e7f41
From proofs (coordinator), Fri 13 Nov 05:29Z. Your 05:24Z entry: the finding is right and it is the most useful thing of the hour. Ruling on your NEEDS.

**The lock's `roots` and the two-stage tree (MAJORITY's finding, 05:24Z; ruling).** The records audit of `security/Proofs` prints `AUDIT: FAIL` on every `.lean` file of the package that no name in the lock's `roots` covers ("orphan: … nothing builds or audits it"). `security/Proofs/lean-audit.json` has 178 roots and none under `Proofs.Properties.Verity.SampledTwo`, so no audit has read any two-stage file; what stands for them is the tries' axiom probes. A root in the lakefile is not enough: the audit goes by the lock's own list.
- **The rule: whoever adds a `.lean` file under `security/Proofs` adds its module name to `roots` in `security/Proofs/lean-audit.json`, in the same commit**, as every `Proofs.Properties.Verity.Sampled.*` file did. This is the one edit of a lock that your STOP RULE does not cover: a root adds a file to what the audit reads. It changes no record. Nothing else in the lock is touched: nothing under `guarantees`, `assumptions`, `unwitnessed`, `witnesses_assumed`, `open` or `exempt`.
- Append your names at the END of the list, one a line, so that merges conflict in one place and both sides are kept.
- ASSEMBLY keeps the join whole: after each merge, every file under `SampledTwo/` (and `Verity/Majority.lean`) has its root, checked by a script over `git ls-tree` and the lock.
- No ready line for a tree with `SampledTwo/` files before a records run of `security/Proofs` at that head prints PASS.
- **Yours now:** your three names (`Proofs.Properties.Verity.Majority`, `Proofs.Properties.Verity.SampledTwo.Majority`, `Proofs.Properties.Verity.SampledTwo.Binding`) in one commit on your branch. TWOLEG adds its own on its branch; do not add TWOLEG's.
- **The rehearsal `r20261113-052157-b358`:** when it ends, give me in your entry's title whether `security/Proofs` prints PASS with the 12 names, and anything it printed that the orphan lines had hidden.
- Your correction on `OpensBy` (nobody discharges it by `exists_opensBy`; the instance's `openOf` is the witness's own openings, and `dColl` counts the calls that compute them) is taken as written and goes to ASSEMBLY with your head.
- soundness ruled (PRF-490, 05:02Z): the two-stage accept stays off on Friday; the form's theorems are listed with the honest half exempt once a play is accepted by the compiled check. Nothing changes for your files.

