[2026-11-11T15:09Z] c90ecd05c391
WIDTH, TRN-122, 7:09 AM PST. THE FIRST PIECE IS READY TO READ: decode_keys, root_first and decode_ret pass on a node.

HEAD AND RUN. Branch cursor/training-root-first-59ab, head 552776000e, pushed. RAN: run r20261111-145932-bc57 on vy-nebius-cpu-1, source 552776000e (read from research inspect --json), verity lean try on security/Proofs/Properties/Nci/DecodeKeys.lean: lean exit 0 in 263 s, "none of its 24 theorems uses an axiom beyond propext, Classical.choice and Quot.sound". No warning from the file. No sorry, axiom or native_decide in it (read).

WHAT IT STATES, in namespace NciCheck.Properties.SecurityProofs.DecodeKeys, each from decode bytes = .ok P:
- decode_keys: FORM's statement, character for character, both conjuncts.
- root_first: forall i < P.root, (P.defs[i]!).key is not (P.defs[P.root]!).key. That is RootFirst P with its definition unfolded; RootFirst is on FORM's branch, not on main, so FORM closes its hypothesis with: root_first bytes P h. It uses decode_keys' second conjunct and lean's decode_root.
- decode_ret: COUNTS' statement, as written in its 6:50 AM message. COUNTS: it compiles; import Proofs.Properties.Nci.DecodeKeys.

HOW. One file of ours, 422 lines; nothing else changes. mvcgen. composite_step and staticFns_step are for any K on the decoder's state; all conditions closed (item 3: none was left to send). The invariant Kept B is the two conjuncts of decode_keys, plus "a busy name has no index entry", plus COUNTS' fact, plus "the names of B are busy". decode is re-walked there with lean's own tactic script for decode_full, because I change nothing that exists in lean's files; one shared walk would need decode_full's proof split, which is lean's to decide.

NOT CHECKED. --siblings (launched now), the lakefile root for the new module (check.py fails an unimported file as an orphan), the retired-terms allowlist entry (3 sites, all the import path and the namespace), check.py, the lock. I do those when you say how the file lands: its own pull request, or with FORM's. No pull request opened.

ITEM 2 (the public lemma in LeafWidthsResolve.lean for any callee): I wait for FORM's exact statement and do it on cursor/training-record-width-59ab.

