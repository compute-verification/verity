import Definitions.Pous.Model.M1p

/-!
# The deployed uniform sampler on one word (trusted)

`Key.uniform(N, j)` (`verity/ml/randomness`) is exact rejection sampling on the keyed SHA-512 stream at `("u", j)`: it
reads `w`-bit big-endian words, `w = 8 ⌈(bitlen N + 64) / 8⌉`, and returns the first word below `2^w − (2^w mod N)`, the
largest multiple of `N` that fits, reduced mod `N`. The remote audit's opening draw (`Accounting.Remote.Draw`) takes
each step of its partial Fisher–Yates by it.

* `wordIndex`: the sampler on one word `u`: `u mod N` when `u < 2^w − (2^w mod N)`, none when it rejects `u`. The word's
  value is `Pous.bitsVal u` (bit `k` with weight `2^k`), the big-endian integer of the stream's bytes.
-/

namespace Accounting

namespace Actual

/-- **The deployed sampler on one word** (`uniform_from_words`): the word's value mod `N` when it is below
`2^w − (2^w mod N)`, the largest multiple of `N` at most `2^w`, and none when the sampler rejects it. -/
def wordIndex (N : ℕ) {w : ℕ} (u : Pous.Bits w) : Option (Fin N) :=
  if h : Pous.bitsVal u < 2 ^ w - 2 ^ w % N then
    some ⟨Pous.bitsVal u % N, Nat.mod_lt _ (Nat.pos_of_ne_zero fun hN => by simp [hN] at h)⟩
  else none

end Actual

end Accounting
