to: @proofs
title: The verifier's texts, three goals and the post say one thing about what the Lean verifier does with a two-stage run (Thu 12 Nov; the wording follows the page to Daniel)

**Ask.** When the page to Daniel on what Thursday's result is called has its outcome, make the places below say that one thing. This asks for the wording to follow the page. It asks for no decision and names none as made.

**Why.**
- The Lean verifier refuses every two-stage run last. `def FORM : Bool := false` (`verity/core/service/spot/Spot/SecondStage.lean:115` on `main`). Lines 91–92: "the run is refused last (`two-stage-form`) unless `FORM` allows it; while it doesn't, no run passes" [read].
- The Python mirror has `FORM = False` (`verity/core/protocols/spot/second_stage.py:71`). The README: "its `FORM` still refuses two-stage runs" (`verity/core/protocols/spot/README.md:139`) [read].
- The flip waits for goal `proofs/spot-two-stage-sound`, Mon 30 Nov. The post is Mon 16 Nov [read: the Goals page, Tue 10 Nov about 02:00Z].
- Two goals say "two-stage is proved for the Lean verifier with `Evaluates` and `Reveals` open": `proofs/spot-two-stage-sound` and `proofs/spot-claims-checked`. The lock on `main` gives `SpotKSTwo` three open premises (`CompiledTwo.Scope`, `SpotFlock.Embeds`, `SpotFlock.FrameIs`) and does not list it under `deployed` [read].
- `proofs/spot-two-stage-e2e` says "the Lean verifier of record checks both rounds", and not the verdict the run gets [read].
- The page is not up. The coordinator's comment on PRF-477 (Tue 02:02Z) says it posts by Tue 07:00Z [relayed].

**Done when.**
- The page's outcome is quoted on this issue.
- `FORM`, both `FORM_WHY` texts and the README's line agree with it. If the outcome keeps `FORM` false, they need no change.
- The three goals state the verdict a two-stage run gets, as `spot-128k-attention-proved` does, and name the lock's open premises, as `spot-post-ready` does.
- The post's sentence on the verifier is the outcome's.
- Thursday's two-stage run's record shows that verdict.

**Pointers.**
- Note `proofs/20261109T2010Z-report-claims-ledger`, row H12 and section 6.
- Top's freeze (TOP-387) covers `verity/core/protocols/spot`: ask top before a text edit lands there.
