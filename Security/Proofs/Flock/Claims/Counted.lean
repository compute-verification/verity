import Definitions.Claims

/-!
# A counted program against a size of the proof's choosing

`DecoderComplexity decode counted size p`: `counted` computes what `decode` does, and its count is at most `p(size)`.
The proofs bound the deployed decoder's pieces with it, each at a size of its own (the root finder's at its input's
length, a level's decoder at its opened data's), on the way to `Claims.Flock.Efficient`, which reads the deployed
decoder's own count (`Claims.Flock.Deploy.dec`) at the opened word's encoded length (`Claims.Flock.wordLen`). It is no
claim: with `counted` and `size` its own arguments, a count of `0` meets it.
-/

namespace Claims

/-- **A counted program at a size**: `counted` computes what `decode` does, and the steps it reports are at most
`p(size)`. -/
def DecoderComplexity {I T : Type} (decode : I → T) (counted : I → T × ℕ) (size : I → ℕ) (p : Polynomial ℕ) : Prop :=
  ∀ i, (counted i).1 = decode i ∧ (counted i).2 ≤ p.eval (size i)

end Claims
