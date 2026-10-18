import Proofs.Flock.Claims.KS.RepDist
import Proofs.Flock.Claims.KS.Rounds

/-!
# A rep's level caps at the plain fork round's count

With four lanes at every level after level 0 (`fast100_lanes`), rep `r`'s level-`ℓ + 1` cap is `capRound k0 (ℓ + 1)`
coins into Ligerito, after the zerocheck's, the lincheck's and the opening's coins: the count `forkRound` reads after
the link message and rep `r`'s `r · R` earlier rounds.

* `ligSlotDist_capRound`: Ligerito's part of the count is `capRound`'s;
* `dist_lockRep_capRound`: so every path of rep `r` that meets the cap meets it after that many coins, at every level
  (past the last cap there is no such node).
-/

namespace FlockDeploy.KS

open FlockSoundness FlockSoundness.Game

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]
variable {D : Type}

/-- **Ligerito's level-`ℓ + 1` cap is `capRound k0 (ℓ + 1)` coins in**, with four lanes after level 0. -/
theorem ligSlotDist_capRound (k0 : ℕ) (levs : List Accounting.Level) (hl : ∀ lv ∈ levs, lv.lanes = 4) (ℓ : ℕ)
    (hℓ : ℓ < levs.length) : ligSlotDist levs ℓ + k0 + 1 + 1 = capRound k0 (ℓ + 1) := by
  cases ℓ with
  | zero =>
    have h0 : ligSlotDist levs 0 = 0 := rfl
    have h1 : capRound k0 (0 + 1) = k0 + 2 := if_pos (by omega)
    omega
  | succ p =>
    have h0 : ligSlotDist levs (p + 1) = 7 + recSlotDist levs p := rfl
    have h1 : capRound k0 (p + 1 + 1) = k0 + 13 + 11 * (p + 1 + 1 - 2) := if_neg (by omega)
    have h2 := recSlotDist_eq levs p hl hℓ
    omega

/-- **A rep's level-`ℓ + 1` cap, at the plain fork round's count**: `capRound k0 (ℓ + 1)` coins into Ligerito after
the zerocheck's, the lincheck's and the opening's, at every level of a schedule with four lanes after level 0 (past
the last cap there is no such node). -/
theorem dist_lockRep_capRound (A : Model.Arith F K) (r : ℕ) (hr : r < 2) (S : Model.Statement) (m k0 : ℕ)
    (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ) (hk0 : l₀.lanes = k0) (hm : 13 ≤ S.m)
    (C₀ : Model.Oracle F) (extra : List (Model.Claim F S.m)) (hl : ∀ lv ∈ l₁ :: rest, lv.lanes = 4) (ℓ : ℕ) :
    Lock.Dist (slotOf r (ℓ + 1))
      (capRound k0 (ℓ + 1) + (extra.length + 1 + 1 + 1) + (S.kLog - 6 + 3) + (S.m - 6 + 3))
      (lockRep (D := D) A r hr S m k0 l₀ l₁ rest ext hk0 hm C₀ extra) := by
  by_cases hlt : ℓ < (l₁ :: rest).length
  · have h := dist_lockRep (D := D) A r hr S m k0 l₀ l₁ rest ext hk0 hm C₀ extra ℓ
    unfold repSlotDist at h
    rw [← ligSlotDist_capRound k0 (l₁ :: rest) hl ℓ hlt]
    exact h
  · exact .noSlot (noSlot_lockRep_ge A r hr S m k0 l₀ l₁ rest ext hk0 hm C₀ extra ℓ
      (by simp only [List.length_cons] at hlt ⊢; omega))

end FlockDeploy.KS
