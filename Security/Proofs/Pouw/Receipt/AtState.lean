import Proofs.Pouw.Receipt.Duo
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Logic.Equiv.Fin.Basic
import Mathlib.Data.BitVec
import Mathlib.Tactic.Ring

/-!
# duo runs at one start state: three words are never bound

At a fixed start state a run of `L` words maps `2^(nL)` word sequences to `2^(2n)` states, so for `L > 2` two
different runs reach one state, whatever the keys and the state (`duoRun_atState_not_injective`). A whole-state
checkpoint after a block of three or more free words binds them only with a probability over the words, never for
every choice of them.
-/

namespace Pouw.ReceiptProofs

open Pouw.Receipt

theorem duoRun_atState_not_injective {n L : ℕ} (k a : Fin L → BitVec n) (r : Fin L → ℕ)
    (s : BitVec n × BitVec n) (hn : 0 < n) (hL : 2 < L) :
    ¬ Function.Injective (fun w : Fin L → BitVec n => duoRun k a r w s) := by
  intro hinj
  let e₁ : (Fin L → BitVec n) ≃ Fin ((2 ^ n) ^ L) :=
    (Equiv.arrowCongr (Equiv.refl (Fin L)) (BitVec.equivFin (m := n)).toEquiv).trans finFunctionFinEquiv
  let e₂ : (BitVec n × BitVec n) ≃ Fin (2 ^ n * 2 ^ n) :=
    (Equiv.prodCongr (BitVec.equivFin (m := n)).toEquiv (BitVec.equivFin (m := n)).toEquiv).trans finProdFinEquiv
  have hf : Function.Injective (e₂ ∘ (fun w : Fin L → BitVec n => duoRun k a r w s) ∘ e₁.symm) :=
    e₂.injective.comp (hinj.comp e₁.symm.injective)
  have h := Fintype.card_le_of_injective _ hf
  simp only [Fintype.card_fin] at h
  have h2 : 2 ≤ 2 ^ n :=
    calc 2 = 2 ^ 1 := (Nat.pow_one 2).symm
      _ ≤ 2 ^ n := Nat.pow_le_pow_right (by decide) hn
  have h4 : 2 ^ n * 2 ^ n * 2 ≤ (2 ^ n) ^ L :=
    calc 2 ^ n * 2 ^ n * 2 ≤ 2 ^ n * 2 ^ n * 2 ^ n := Nat.mul_le_mul_left _ h2
      _ = (2 ^ n) ^ 3 := by ring
      _ ≤ (2 ^ n) ^ L := Nat.pow_le_pow_right (Nat.two_pow_pos n) hL
  have hB : 0 < 2 ^ n * 2 ^ n := Nat.mul_pos (Nat.two_pow_pos n) (Nat.two_pow_pos n)
  omega

end Pouw.ReceiptProofs
