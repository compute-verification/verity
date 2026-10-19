import Proofs.Pouw.Receipt.Run
import Mathlib.SetTheory.Cardinal.Finite

/-!
# duo runs: four words bound anywhere in a run, checkpoints every four words, and the kernel of long runs

A run is a bijection of the state (`duoRun_state_injective`, and so `duoRun_state_surjective` on the finite states).
Two runs that share the words before and after four middle words, with final states equal at every start state, then
have the four-word maps equal at every state the shared prefix reaches, which is every state, and `duoQuad_injective`
makes the middle words equal (`duoWindow_injective`). Checkpoints every four words are equal from the first differing
block's start on, whose state the earlier blocks map onto every state (`duoCheckpoints_injective`).

No run of more than `2·4ⁿ` words is injective into the final-state maps: `2^(nL)` runs against `(4ⁿ)^(4ⁿ)` maps
(`duoRun_kernel`).
-/

namespace Pouw.ReceiptProofs

open Pouw.Receipt

private theorem bitVec_finite (n : ℕ) : Finite (BitVec n) := Finite.of_equiv _ BitVec.equivFin.symm.toEquiv

theorem duoRun_state_surjective {n L : ℕ} (k a : Fin L → BitVec n) (r : Fin L → ℕ) (w : Fin L → BitVec n) :
    Function.Surjective (duoRun k a r w) :=
  haveI := bitVec_finite n
  Finite.injective_iff_surjective.mp (duoRun_state_injective k a r w)

theorem duoQuadWith_state_injective {n : ℕ} (q : DuoQuadKeys n) (d : BitVec n × BitVec n × BitVec n × BitVec n) :
    Function.Injective (duoQuadWith q d) := fun _ _ h =>
  duoWord_state_injective _ _ _ _ (duoWord_state_injective _ _ _ _
    (duoWord_state_injective _ _ _ _ (duoWord_state_injective _ _ _ _ h)))

theorem duoQuadWith_state_surjective {n : ℕ} (q : DuoQuadKeys n) (d : BitVec n × BitVec n × BitVec n × BitVec n) :
    Function.Surjective (duoQuadWith q d) :=
  haveI := bitVec_finite n
  Finite.injective_iff_surjective.mp (duoQuadWith_state_injective q d)

theorem duoQuadWith_injective {n : ℕ} (q : DuoQuadKeysOk n) : Function.Injective (duoQuadWith q.1) := by
  obtain ⟨⟨k₁, a₁, r₁, k₂, a₂, r₂, k₃, a₃, r₃, k₄, a₄, r₄⟩, ha₁, hr₁, hr₁n, ha₂, ha₂', hr₂, ha₃, hr₃⟩ := q
  exact duoQuad_injective k₁ ⟨a₁, ha₁⟩ r₁ hr₁ hr₁n k₂ ⟨a₂, ha₂⟩ ha₂' ⟨r₂, hr₂⟩ k₃ ⟨a₃, ha₃⟩ ⟨r₃, hr₃⟩ k₄ a₄ r₄

theorem duoWindow_injective {n p t : ℕ} (k a : Fin p → BitVec n) (r : Fin p → ℕ) (w : Fin p → BitVec n)
    (q : DuoQuadKeysOk n) (k' a' : Fin t → BitVec n) (r' : Fin t → ℕ) (w' : Fin t → BitVec n) :
    Function.Injective (fun d s => duoRun k' a' r' w' (duoQuadWith q.1 d (duoRun k a r w s))) := by
  intro d d' h
  apply duoQuadWith_injective q
  funext x
  obtain ⟨s, rfl⟩ := duoRun_state_surjective k a r w x
  exact duoRun_state_injective k' a' r' w' (congrFun h s)

theorem duoCheckpoints_injective {n : ℕ} :
    ∀ {m : ℕ} (q : Fin m → DuoQuadKeysOk n), Function.Injective (duoCheckpoints (fun j => (q j).1))
  | 0, _ => fun _ _ _ => funext fun j => j.elim0
  | _ + 1, q => fun w w' h => by
    have h0 : w 0 = w' 0 := duoQuadWith_injective (q 0) (funext fun s => by
      simpa [duoCheckpoints] using congrFun (congrFun h s) 0)
    have ht : Fin.tail w = Fin.tail w' := duoCheckpoints_injective (Fin.tail q) (funext fun x => by
      obtain ⟨s, rfl⟩ := duoQuadWith_state_surjective (q 0).1 (w 0) x
      funext j
      have hj := congrFun (congrFun h s) j.succ
      simp only [duoCheckpoints, Fin.cons_succ] at hj
      rw [← h0] at hj
      exact hj)
    rw [← Fin.cons_self_tail w, ← Fin.cons_self_tail w', h0, ht]

theorem duoRun_kernel {n L : ℕ} (hn : 0 < n) (hL : 2 * 4 ^ n < L) (k a : Fin L → BitVec n) (r : Fin L → ℕ) :
    ∃ w w' : Fin L → BitVec n, w ≠ w' ∧ duoRun k a r w = duoRun k a r w' := by
  have := bitVec_finite n
  by_contra hne
  have hc := Nat.card_le_card_of_injective _ fun w w' h => by_contra fun hw => hne ⟨w, w', hw, h⟩
  have hb : Nat.card (BitVec n) = 2 ^ n := by
    rw [Nat.card_congr BitVec.equivFin.toEquiv, Nat.card_eq_fintype_card, Fintype.card_fin]
  rw [Nat.card_fun, Nat.card_fun, Nat.card_prod, hb, Nat.card_eq_fintype_card, Fintype.card_fin, ← pow_mul,
    ← pow_add, ← pow_mul] at hc
  have h4 : 2 ^ (n + n) = 4 ^ n := by rw [← two_mul, pow_mul]; norm_num
  rw [h4] at hc
  have hlt : (n + n) * 4 ^ n < n * L := by
    rw [← two_mul, mul_comm 2 n, mul_assoc]
    exact Nat.mul_lt_mul_of_pos_left hL hn
  exact absurd hc (not_le.mpr (Nat.pow_lt_pow_right (by norm_num) hlt))

end Pouw.ReceiptProofs
