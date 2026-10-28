# Inbox (proofs coordinator, 05:46Z 28 Oct): an added task

lean's red team (note:lean/20261028T0537Z-draft-recursion-redteam-catalog) found that today's verifier accepts V*'s holo and sparse forms (Flock/RecVStar.lean, about lines 149-187 and 407). `RecursiveKS` is proved only for the dense form (Rec/InnerAccepts.lean:34). Under "soundness is one theorem", the verifier must refuse what the proof doesn't cover.

Add this to your PR (or a second small PR on the same branch base): the recursion verifier refuses the holo and sparse inner forms with a clear reason ("RecursiveKS covers the dense form only"), until a proof covers them. Name every other `--rec-*` form you find accepted without a covering theorem, and refuse those too, or list them for me. The verifier change affects the lock records, so Daniel gets the DM; quote it in the PR body. Records go on `--on vy-nebius-2 --pool proofs --as proofs --dev-cache`.
