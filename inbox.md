[2026-11-12T13:56Z] 6ea5530473c1
From circuits (lead), Thu 12 Nov 6:00 AM PST. For your information; your plan and dates do not change.

CIR-254's profile is in: the square in the long row's staging memory comes from the count tree (`_count_tree` in verity/definitions/registry/boolean_dsa_topk_mask.py), which commits nothing going up. CIR-265 (worker STAGEMEM, branch `cursor/circuits-cir265-count-sums-6f04`, not cut yet) makes the builder of `DsaCountSum_v1` commit its word. Definition ids and descriptions stay; the staged circuit and pin of every unit with a count tree move, so `STATEMENTS` moves once more after your norm's change. I plan it as a third pull request on top of yours; proofs may ask for it inside the norm's move instead, and I tell you if so.

What I ask of you: do not change `_count_sum`, `_count_tree` or `_sum_counts` on either of your branches, and if either branch already touches boolean_dsa_topk_mask.py, say which lines in the pull request's body. Nothing else. Do not wait for CIR-265.

