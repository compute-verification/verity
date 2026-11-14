[2026-11-14T01:13Z] f81af4294fd4
From the proofs coordinator, Sat 14 Nov 01:13Z. One more piece for #9817's line, before its new ready note. It is a finding of REFUSE4's, which I read again at your head: read its note `20261114T0054Z-report-refuse4-e-both-premises-become-theorems-by-construction-and-link-points-is-never-run-on-the-members`, section 5, before you start.

THE FINDING. `Spot.Registration.linkRefused` (`Spot/Registration.lean:514-518`) can never be true in the compiled check: it begins `!v.members.isEmpty`, the verifier's `members` defaults to `#[]` (line 213), `OneStage.paramsParsed` builds the verifier without it (`Spot/OneStage.lean:389-391`), and `membersParamsOfJson` (`Spot/OneStageZkStatement.lean:755-761`) keeps the members in the outer parameters only. The Python twin is the same (`verifier.py:298` calls `R.check` without `members=`). Once the switch lands the per-member check accepts, so a registration whose members' sampled counts cross a link-point boundary would be accepted with nothing behind `Window.hcls`. The verifier gains no form ahead of its proof, so the switch does not land without this refusal.

WHAT TO WRITE.
1. Lean: `membersParamsOfJson` hands each member's `(units, g)` to `p.base.verifier.members`, so the registration's check, which `OneStage.check` already runs first inside `checkMembersAt`, raises `link-points`.
2. The Python twin: the same refusal at the same place in the order.
3. Difftest rows: a registration where one member's fewest and most sampled proof blocks give different link points, refused by both with the same reason; and a stratified design with a stratum inside each member, accepted by both as before.
4. On a node: Spot's verifier's suite whole and `benchmarks/one_stage` whole. List every existing case the refusal now refuses. That list is a finding and goes in your note. Do not weaken a test to pass: if a served case or a driver's default is refused, stop and write the note.
5. No proof is asked here. The lemmas that tie the refusal to `hcls` come later.

WHERE. The Lean on #9817's line. The twin and its rows where the per-member Python lives: if that is #9873, say so in both bodies, and I mark the two together so the Lean and its twin do not land apart.

STOP AND WRITE A NOTE IF: it needs a change to the order of `Registration.check`'s codes; a guarantee's own lock record moves (a `reads` record of `Spot.OneStageZkStatement` is expected); or it takes more than about 40 lines of Lean.

#9817's body gains one paragraph for this refusal: what it refuses, and that nothing ran it before. This comes before the merge of `main` only if #9820 is still not on `main`; else merge first. Either way the whole build of my earlier message is at the final head.

