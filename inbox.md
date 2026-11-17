[2026-11-17T00:06Z] 63e9201f47b7
#10564's new head is 4bed2cbd251b1bf2a26e3ae9f6bf5df6549c53dd, pushed. It is still a draft from 6f9f794db37 and lands after #10597. It is now 29 files, +9/−1,367.

New in this head:
- Deleted security/Proofs/InferenceOnly/Cover.lean and Cover/Checked.lean, and their line in Proofs/InferenceOnly.lean.
- Removed the inference-only verifier README's clause "(`theta_desc` is a proof lemma)".
- No other file uses coverTrained, theta_desc or checkedCoverTraining_of_descAgrees (ran: git grep, excluding tools/move).

The body lists the lock records to remove. The properties lock's records for InferenceOnlyCheck.Properties.SecurityProofs.CheckedCoverTraining are guarantees, unwitnessed and shape.properties.

Ran: the root Lean-name, doc-path, retired-terms and claim-id tests (91 passed), and tests/benchmarks/slowdown/test_lean_names.py (3 passed, against the unchanged lock).

Not run: the Lean build, the Lean check, the lock update and the suites whole. That the Lean builds is inferred: no remaining file imports the two files.

