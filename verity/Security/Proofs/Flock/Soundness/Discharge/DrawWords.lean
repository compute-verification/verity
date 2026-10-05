import Proofs.Flock.Soundness.Discharge.DrawWords.Statements
import Proofs.Flock.Soundness.Discharge.DrawWords.Words
import Proofs.Flock.Soundness.Discharge.DrawWords.Prob
import Proofs.Flock.Soundness.Discharge.DrawWords.LiveCard

/-!
# The link points at the live law's fallback draw

`DrawWords/Statements.lean`: under A3, the live stratified draw reads past its source's bytes with probability at most
`(∑ s, k s) · 2^-64`, and the headline at the executable with the decoder needed only at the draws of `∑ s, k s` units,
where setupH sets up at the one value `mPtsOf c.g (∑ s, k s)`. `DrawWords/Words.lean`: the deterministic lemma
(`words_run`). `DrawWords/Prob.lean`: the word and fallback bounds (`word_uniform`, `word_reject`, `fallback_rare`,
`draw_card_rare`). `DrawWords/LiveCard.lean`: R11 at the `good` draws (`execRefines_where`, `decodesWhere_of_on`),
`m_pts` at the draw's size (`mPts_at_card`) and the headline (`headline_card`).
-/
