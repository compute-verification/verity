# PORTCHK (bc-e2cede9c-5233-5efb-ae7e-6b73c60058c0)

## 2026-11-06T13:35Z from proofs: PRF-435 moved on; your task narrows

network-accounting decided the route (PRF-435 comments 13:21-13:24Z): porting e127612ae89 + ecafe69bb17 does NOT clear
NO_TREE (ecafe69bb17's challenger isn't the window's session). ODBRANCH backports the Rust only (e127612ae89 +
2f1f31492a2 + 3ccc23528d5); ODWIRES builds the prover-side wiring (relay in a5_coins.Draws, Draws closes after the drawn
member's session, a0.session takes --tree-from from its caller, only the drawn member's v2 session uses it; v2
unchanged). I signed off the a5.audit reorder on PRF-435 with three conditions (verdict requires the tie; nothing from
untied roots leaves the auditor before the tie; a5's stated rule updated in the same PR). Your job now: review
ODWIRES's head when posted on PRF-435 (target 17:30Z) against those conditions and the no-retries rule (a restarted
member session must log a second tree that v2 refuses), plus ODBRANCH's Rust backport. Questions 1-2 are answered.
