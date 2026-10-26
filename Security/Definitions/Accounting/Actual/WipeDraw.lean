import Definitions.Accounting.Actual.WipePrf

/-!
# The deployed index draw, and one audit as a distinguisher of the challenger's coins (trusted)

The streamed-label wipe's challenger draws its `k` indices once the fill is complete:
`draw_indices(coin_secret, server, session, N, k)` (`verity/core/protocols/pous/streamed_wipe.py`), index `j` being
`derive(coin_secret, COIN_DOMAIN, {server, session, n: N}).uniform(N, j)` (`verity/ml/randomness`), with
`coin_secret` 32 bytes fresh from the OS on the challenger's machine. `Key.uniform(N, j)` is exact rejection sampling on
the keyed SHA-512 stream at `("u", j)`: it reads `w`-bit big-endian words, `w = 8 ⌈(bitlen N + 64) / 8⌉`, and returns the
first word below `2^w − (2^w mod N)`, the largest multiple of `N` that fits, reduced mod `N`. At the deployed
`N = 40,625,000,000` a word is 104 bits, inside the stream's first block, and it is rejected with probability
`(2^104 mod N) / 2^104 ≈ 2^-70.4`.

* `wordIndex`: the sampler on one word `u`: `u mod N` when `u < 2^w − (2^w mod N)`, none when it rejects `u`. The word's
  value is `Pous.bitsVal u` (bit `k` with weight `2^k`), the big-endian integer of the stream's bytes.
* `drawIndex rest u`: the deployed index of a read whose first word is `u`: `wordIndex`'s when it accepts `u`, else
  `rest`, what the deployed sampler draws from the read's later words. An accepted first word fixes the index, so with
  `rest` free every index the deployed sampler draws is one of these.
* `drawDistinguisher`: the distinguisher one audit makes of the reads' first words, in
  `Pous.StreamedWipe.prfAdvantage`'s form. With coins `(ω, s)` (the other coins and the label secret) and the words `u`,
  when every word gives an index it runs the system once on the labels `prf s` and those indices and outputs
  `heldEvent`; when some word is rejected it outputs true. It reads each word once, as the challenger does, and never
  the coin secret.
-/

namespace Accounting

namespace Actual

/-- **The deployed sampler on one word** (`uniform_from_words`): the word's value mod `N` when it is below
`2^w − (2^w mod N)`, the largest multiple of `N` at most `2^w`, and none when the sampler rejects it. -/
def wordIndex (N : ℕ) {w : ℕ} (u : Pous.Bits w) : Option (Fin N) :=
  if h : Pous.bitsVal u < 2 ^ w - 2 ^ w % N then
    some ⟨Pous.bitsVal u % N, Nat.mod_lt _ (Nat.pos_of_ne_zero fun hN => by simp [hN] at h)⟩
  else none

/-- **The deployed index of a read** whose first word is `u`: the sampler's index on `u` when it accepts `u`, else
`rest`, what the deployed sampler draws from the read's later words. -/
def drawIndex {N w : ℕ} (rest : Fin N) (u : Pous.Bits w) : Fin N :=
  (wordIndex N u).getD rest

/-- **One audit as a distinguisher of the coin words**: with coins `x.1 = (ω, s)` and the reads' first words `x.2`,
it runs the system once on the labels `prf s` and the indices the words give, and outputs `heldEvent`; when some word
gives no index it outputs true. -/
def drawDistinguisher {Ω L : Type*} {n M N b κ w : ℕ} (prf : Pous.Bits κ → Fin M → Pous.Bits b)
    (S : (Ω × (Fin M → Pous.Bits b)) × (Fin n → Fin N) → Actual.Run L) (X V : Finset L)
    (Ch : Fin n → Set (ℕ × L × ℕ × ℕ × ℕ) → Prop)
    (arrive : (Ω × (Fin M → Pous.Bits b)) × (Fin n → Fin N) → Fin n → ℕ) (T : Finset ℕ) (m K : ℕ)
    (x : (Ω × Pous.Bits κ) × (Fin n → Pous.Bits w)) : Prop :=
  ∀ c : Fin n → Fin N, (∀ j, wordIndex N (x.2 j) = some (c j)) →
    heldEvent S X V Ch arrive T m K ((x.1.1, prf x.1.2), c)

end Actual

end Accounting
