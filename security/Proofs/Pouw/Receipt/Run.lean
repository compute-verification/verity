module

public import Definitions.Pouw.Receipt.Duo
import Mathlib.Data.Finset.Attr
import Mathlib.Tactic.NormNum.Ineq
import Mathlib.Tactic.NormNum.Pow
import Mathlib.Tactic.SetLike
import Proofs.Pouw.Receipt.Duo

/-! duo's step is a bijection of the state, and a run binds each word at every state. -/

@[expose] public section

namespace Pouw.ReceiptProofs

open Pouw.Receipt

theorem duoWord_state_bijective {n : ℕ} (k a : BitVec n) (r : ℕ) (d : BitVec n) :
    Function.Bijective (fun s => duoWord k a r s d) :=
  ⟨duoWord_state_injective k a r d, duoWord_state_surjective k a r d⟩

theorem duoRun_state_injective {n : ℕ} :
    ∀ {L : ℕ} (k a : Fin L → BitVec n) (r : Fin L → ℕ) (w : Fin L → BitVec n), Function.Injective (duoRun k a r w)
  | 0, _, _, _, _ => fun _ _ h => h
  | _ + 1, k, a, r, w => fun s s' h => by
    simp only [duoRun] at h
    exact duoWord_state_injective (k 0) (a 0) (r 0) (w 0) (duoRun_state_injective _ _ _ _ h)

theorem duoRun_update_injective {n : ℕ} :
    ∀ {L : ℕ} (k : Fin L → BitVec n) (a : Fin L → OddKey n) (r : Fin L → ℕ) (s : BitVec n × BitVec n)
      (w : Fin L → BitVec n) (i : Fin L),
      Function.Injective (fun x => duoRun k (fun j => (a j).1) r (Function.update w i x) s)
  | 0, _, _, _, _, _, i => i.elim0
  | _ + 1, k, a, r, s, w, i => by
    intro x y h
    simp only [duoRun] at h
    induction i using Fin.cases with
    | zero =>
      rw [Fin.tail_update_zero, Fin.tail_update_zero, Function.update_self, Function.update_self] at h
      exact duoWord_injective (k 0) (a 0) (r 0) s (duoRun_state_injective _ _ _ _ h)
    | succ j =>
      rw [Fin.tail_update_succ, Fin.tail_update_succ, Function.update_of_ne (Fin.succ_ne_zero j).symm,
        Function.update_of_ne (Fin.succ_ne_zero j).symm] at h
      exact duoRun_update_injective (Fin.tail k) (Fin.tail a) (Fin.tail r) _ (Fin.tail w) j h

end Pouw.ReceiptProofs
