import FlockSoundness.Discharge.DrawWords.Statements
import FlockSoundness.Discharge.DrawWords.Words

/-!
# The link points at the live law's fallback draw

`DrawWords/Statements.lean`: under A3, the live stratified draw reads past its source's bytes with probability at most
`(∑ s, k s) · 2^-64`, and the headline at the executable with the decoder needed only at the draws of `∑ s, k s` units,
where setupH sets up at the one value `mPtsOf c.g (∑ s, k s)`. `DrawWords/Words.lean`: the deterministic lemma
(`words_run`).
-/
