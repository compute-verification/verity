module

public import Definitions.Pous.Model.M1p
import Mathlib.Basic.ENNReal.Real

/-! The deployed uniform sampler `Key.uniform(N, j)` on one word: exact rejection sampling on the keyed SHA-512 stream. -/

@[expose] public section

namespace ResourceAccounting

namespace Actual

/-- **The deployed sampler on one word** (`uniform_from_words`): the word's value mod `N` when it is below
`2^w − (2^w mod N)`, the largest multiple of `N` at most `2^w`, and none when the sampler rejects it. -/
def wordIndex (N : ℕ) {w : ℕ} (u : Pous.Bits w) : Option (Fin N) :=
  if h : Pous.bitsVal u < 2 ^ w - 2 ^ w % N then
    some ⟨Pous.bitsVal u % N, Nat.mod_lt _ (Nat.pos_of_ne_zero fun hN => by simp [hN] at h)⟩
  else none

end Actual

end ResourceAccounting
