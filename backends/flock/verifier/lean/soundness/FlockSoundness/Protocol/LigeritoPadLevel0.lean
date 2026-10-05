import FlockSoundness.Protocol.PadCode
import FlockSoundness.Protocol.LigeritoPhase
import FlockSoundness.Protocol.Model.LigeritoPad

/-!
# The padded level 0: lane rounds, extra lanes, and decoding

The pieces of `ligeritoPad`'s level 0 that differ from `ligerito`'s:

* **The lane rounds** send no message after the last coin (`foldRoundsNL`). For a post-condition
  that does not read that message, they are the lane rounds (`value_foldRoundsNL_le`).
* **The extra lanes' coins** (`drawKs`). Each coin `ρ_i` is an affine line `w + ρ_i·X_i` for mutual
  correlated agreement (`ExtrasGood`, `value_drawKs_extras_le`), and a coin on the restarted
  claim, an affine function of the coins with a nonzero constant for every list element
  (`value_drawKs_affine_le`).
* **Rep `r`'s list** (`padList`): the row families, witness lanes then the rep's extra lanes, within
  the padded radius of the rep's rows. A family's next-level message is its witness lanes' messages
  folded by the lane challenges plus its extra lanes' weighted by `ρ` (`padFold`).
* **Decoding** (`decode_pad`): if no fold round and no extra coin was an MCA event, and the padded
  column agrees with the encoding of `G` on `(1 − γ')·2^d` positions, then `G` is `padFold` of a
  family in the list. The padding correction `ybar` is then the family's folded padding.
-/

namespace FlockSoundness

open Game Model Accounting
open CodingTheory CoreDefinitions
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

/-! ## The lane rounds without their last message -/

/-! ## The extra lanes' coins -/

/-! ## Rep `r`'s list -/

/-- Rep `r`'s lanes: the witness lanes, then its extra lanes. -/
def padLane (k0 e r : ℕ) : Fin (2 ^ k0 + e) → ℕ :=
  Fin.addCases (fun ℓ : Fin (2 ^ k0) => ℓ.val) (extraLane k0 e r)

/-- Rep `r`'s rows of the committed table. -/
def padRows (k0 e r d : ℕ) (C : Oracle F) : Fin (2 ^ k0 + e) → Fin (2 ^ d) → F :=
  fun ℓ q => C (padLane k0 e r ℓ) q.val

open Classical in
/-- **Rep `r`'s list**: the row families within the padded radius of the rep's rows. -/
noncomputable def padList (A : Arith F K) (l : Level) (t k0 e r : ℕ) (C : Oracle F) :
    Finset (Fin (2 ^ k0 + e) → Fin (2 ^ l.logCols + t) → F) :=
  Finset.univ.filter fun R => CloseRows A l.logLen (2 ^ l.logCols + t) (l.padRadius t)
    (padRows k0 e r l.logLen C) R

/-- A padded lane's message: its first `2^cols` coefficients less `c₀` times its padding. -/
def padMsgAt {cols t : ℕ} (c₀ : F) (a : Fin (2 ^ cols + t) → F) (x : Fin (2 ^ cols)) : F :=
  a (Fin.castAdd t x) - if h : x.val < t then c₀ * a ⟨2 ^ cols + x.val, by omega⟩ else 0

/-- A row family's witness lanes' messages. -/
def padWit {k0 e cols t : ℕ} (c₀ : F) (R : Fin (2 ^ k0 + e) → Fin (2 ^ cols + t) → F) :
    Message F k0 cols :=
  fun ℓ => padMsgAt c₀ (R (Fin.castAdd e ℓ))

/-! ## Decoding the padded column -/

omit [Fintype F] [DecidableEq F] [Fintype K] [DecidableEq K] in
theorem Xhat_lift (A : Arith F K) (c cols : ℕ) (hc : c < 2 ^ cols) :
    ∀ j ω, A.Xhat c (cols + j) ω = A.Xhat c cols ω
  | 0, _ => rfl
  | j + 1, ω => by
      rw [← add_assoc, Xhat_succ_low A (cols + j) c
        (lt_of_lt_of_le hc (Nat.pow_le_pow_right (by norm_num) (by omega))), Xhat_lift A c cols hc j ω]

end FlockSoundness
