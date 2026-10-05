import FlockSoundness.Protocol.NovelBasis
import FlockSoundness.Protocol.Accounting.Padded

/-!
# The padded level-0 code: `RS[2^d, k]` on the first `k` novel-basis monomials

M1's level-0 lane of `2^cols` message coefficients and `t` padding coefficients is a codeword of
`RS[2^d, k]`, `k = 2^cols + t`: the evaluation of `Σ_{c < k} a_c·X̂_c` at `d` bits (`encAt`). This
file gives that code what `ListSize`, `LigeritoLevels` and `LigeritoDecode` give a level's code:

* its codewords are exactly the polynomials of degree below `k` (`NovelBasis`), so `encAt` is in
  ArkLib's `ReedSolomon.code` and is injective, and a `K` codeword that agrees with an `F` row on
  `T` is an `F` encoding there (`exists_encAt_of_agree`);
* a `K` combination of the `k` monomials that vanishes on `k` positions is zero (`encKAt_eq_zero`);
* at most `⌊100·2^d/k⌋` row families encode within `1 − √(k/2^d) − η` of a table
  (`card_closeRows_le`, ArkLib's Johnson bound);
* `prCoin_mca_le` bounds its MCA events by `padMcaA / 2^256` (`prCoin_pad_mca_le`).
-/

namespace FlockSoundness

open Game Model Accounting Polynomial
open CodingTheory CoreDefinitions
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

/-- A lane of `k` coefficients on the first `k` monomials of the novel basis at `d` bits. -/
def encAt (A : Arith F K) (d k : ℕ) (a : Fin k → F) (q : ℕ) : F :=
  ∑ c, a c * A.Xhat c.val d (A.omega q)

variable (A : Arith F K) (hA : A.Correct)

/-! ## The list size -/

/-- `γ`-close rows: the table's rows agree with the rows' encodings on a common `1 − γ` fraction
of the positions. -/
def CloseRows (d k : ℕ) (γ : ℝ) {s : ℕ} (T : Fin s → Fin (2 ^ d) → F) (R : Fin s → Fin k → F) :
    Prop :=
  ((Finset.univ.filter fun q : Fin (2 ^ d) => ∀ ℓ, T ℓ q = encAt A d k (R ℓ) q.val).card : ℝ) ≥
    (1 - γ) * 2 ^ d

/-! ## The MCA bound on the padded code -/

end FlockSoundness
