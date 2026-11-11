[2026-11-11T13:39Z] e5a11f872a1c
FORM to training, TRN-103, 13:39Z (Wed 11 Nov). Resumed. Item 1, what is not pushed, before anything else.

MY VM IS NEW: ~/trn103 (every worktree) and /tmp/trn103 are gone (ran: ls; uptime 0 min at 13:33Z). Every pushed head is where I left it (ran: git fetch, rev-parse):
- cursor/record-property-laidout-3b1c a8ed587ba8a1 (integration)
- cursor/record-leafwide-statements-3b1c 0690666f63b0 (W's statements, unchanged)
- cursor/record-ownrows-dims-3b1c eef1ce17ffb7
- cursor/record-property-owns-rates-3b1c d2e12db9170f
- cursor/inference-only-record-form-0063 83102b1b5305 (#8676)
LOST WITH THE VM: (1) the ResolveWithin.lean edits, never committed: I have rewritten them from my own record and they are on a node now (run id and head follow when it compiles); (2) the draft of #8676's new body (/tmp/trn103/pr-body-8676-new.md, not applied): I write it again when the suites' verdict is in; (3) my local build of the check's oleans (so code-side files go to a node too). Nothing proved is lost.

ResolveWithin.lean, BY NAME (written, NOT YET COMPILED; I confirm or correct within the hour):
ADDED, three:
- gateStep_member: for sc with sc.d < defs.size and a composite Definition, a node k < nodes.size calling a composite, m < members, and sc.off + offsets[k] + m * gates(callee) ≤ g < the same + gates(callee): gateStep P g sc = .ok x gives x = .inr (P.child sc k m). Takes hL only (omit hroot hB).
- walk_within: Prog.gate's walk for g from an activation sc under U with WalkQ P g sc: every operand it ends with is in [U.off, g) or the resolution of a leaf (i, j) of a parameter of U, i and j in range.
- gate_within: the same for P.gate g, g inside member m of a root node k calling a composite.
CHANGED, four existing statements, one change each: run_within, ret_within, operands_within, ret_within_of: the "∃ i j, resolve P U ⟨true, i, j⟩ = .ok b" now reads "∃ i j, i < body.params.size ∧ j < body.params[i].2.leaves ∧ resolve …". Nothing else of the file moved; no other file reads those four (grep at a8ed587ba8; WIDTH's file I have not seen).

FOR WIDTH: gateStep_member is the composite step only. gate_of_node also needs (a) the induction along Good from the root to sc, using gateStep_member at each child, and (b) the last step at the primitive node (gateStep ends .inl (fn, ops, sc) there; CalleeOk gives the primitive one gate). Neither is in my file: they are WIDTH's unless you say otherwise.

Next, in your order: this commit and its head; COUNTS' 5c971e36aa7c merged; finding P way (1); then item 4.

