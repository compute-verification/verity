import Proofs.Flock.Soundness.Refine.Run
import Mathlib.SetTheory.Cardinal.Finite
import Mathlib.SetTheory.Cardinal.NatCard

/-!
# A run has one result

`Plays ι g ms cs a` determines `a` from `g`, `ms` and `cs` when the coins' field `F` is finite and nontrivial: a coin
node of type `Fin n → F` (`n ≠ 1`) and one of type `F` have different cardinalities, so a coin array is read by the
one step its node's type allows.

* `fin_fun_inj`: `Fin n → F` determines `n`.
* `plays_det`: two runs of one game on the same messages and coins end in the same result.
-/

namespace FlockSoundness.Discharge.ZkExec

open Game

variable {W F : Type} [Inhabited W] {ι : W → F}

theorem fin_fun_card [Finite F] (n : ℕ) : Nat.card (Fin n → F) = Nat.card F ^ n := by
  rw [Nat.card_fun, Nat.card_fin]

/-- **`Fin n → F` determines `n`** for a finite nontrivial `F`. -/
theorem fin_fun_inj [Finite F] [Nontrivial F] {n n' : ℕ} (h : (Fin n → F) = (Fin n' → F)) : n = n' := by
  have hc := congrArg Nat.card h
  rw [fin_fun_card, fin_fun_card] at hc
  exact Nat.pow_right_injective Finite.one_lt_card hc

/-- `F` is `Fin n → F` only at `n = 1`, for a finite nontrivial `F`. -/
theorem fin_fun_ne [Finite F] [Nontrivial F] {n : ℕ} (h : F = (Fin n → F)) : n = 1 := by
  have hc := congrArg Nat.card h
  rw [fin_fun_card] at hc
  exact Nat.pow_right_injective Finite.one_lt_card (hc.symm.trans (pow_one _).symm)

/-- **A run has one result**: two runs of `g` on the messages `ms` and the coins `cs` end in the same result, for a
finite nontrivial coin field. -/
theorem plays_det [Finite F] [Nontrivial F] {α : Type} {g : Game α} {ms : List Msg} {cs : List (Array W)} {a b : α}
    (h₁ : Plays ι g ms cs a) (h₂ : Plays ι g ms cs b) : a = b := by
  induction h₁ generalizing b with
  | ret a => cases h₂; rfl
  | send m _ ih =>
      cases h₂ with
      | send _ h' => exact ih h'
  | @coinVec n iF iN k c hc hn ms cs a _ ih =>
      generalize hg : @Game.coin α (Fin n → F) iF iN k = g at h₂
      cases h₂ with
      | send => cases hg
      | @coinVec n' iF' iN' k' _ _ _ _ _ _ h' =>
          injection hg with hC hiF hk
          obtain rfl := fin_fun_inj hC
          obtain rfl := eq_of_heq hiF
          obtain rfl := eq_of_heq hk
          exact ih h'
      | coinOne _ _ _ =>
          injection hg with hC
          exact absurd (fin_fun_ne hC.symm) hn
  | @coinOne iF iN k c hc ms cs a _ ih =>
      generalize hg : @Game.coin α F iF iN k = g at h₂
      cases h₂ with
      | send => cases hg
      | coinVec _ _ hn' _ =>
          injection hg with hC
          exact absurd (fin_fun_ne hC) hn'
      | @coinOne iF' iN' k' _ _ _ _ _ h' =>
          injection hg with _ hiF hk
          subst hiF hk
          exact ih h'

end FlockSoundness.Discharge.ZkExec
