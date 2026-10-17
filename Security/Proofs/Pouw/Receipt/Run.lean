import Models.Pouw.Guarantees.ReceiptRun
import Proofs.Accounting.FanIn
import Proofs.Pouw.Receipt.Duo

/-!
# duo and fx1 bind each word of a run given the others

duo's step has `z′ − y′ = rotl(u, r)` (`duoWord_sub`), and rotation is injective, so two steps with equal results have
equal `u = y ⊕ d ⊕ k` (`duoWord_u_eq`). At one state that gives the word, for any multiplier
(`duoWord_word_injective`); at one word the step is a bijection of the state (`duoWord_state_injective`,
`duoWord_state_surjective`). A fold of steps injective in word and state changes when any one word does
(`Accounting.FanInProofs.foldl_ne`). fx1's step is duo's after `d ↦ d·m₁`, which is injective for odd `m₁`
(`mul_odd_cancel`).
-/

namespace Pouw.ReceiptProofs

open Pouw.Receipt

theorem duoWord_sub {n : ℕ} (k a : BitVec n) (r : ℕ) (s : BitVec n × BitVec n) (d : BitVec n) :
    (duoWord k a r s d).2 - (duoWord k a r s d).1 = (s.1 ^^^ d ^^^ k).rotateLeft r :=
  BitVec.add_sub_cancel _ _

theorem duoWord_u_eq {n : ℕ} {k a : BitVec n} {r : ℕ} {s t : BitVec n × BitVec n} {d e : BitVec n}
    (h : duoWord k a r s d = duoWord k a r t e) : s.1 ^^^ d ^^^ k = t.1 ^^^ e ^^^ k :=
  rotateLeft_inj (r := r) (by rw [← duoWord_sub k a r s d, ← duoWord_sub k a r t e, h])

theorem duoWord_word_injective {n : ℕ} (k a : BitVec n) (r : ℕ) (s : BitVec n × BitVec n) :
    Function.Injective (duoWord k a r s) := fun _ _ h =>
  (BitVec.xor_right_inj s.1).1 ((BitVec.xor_left_inj k).1 (duoWord_u_eq h))

theorem duoRun_update_injective {n N : ℕ} (k a : Fin N → BitVec n) (r : Fin N → ℕ) (s : BitVec n × BitVec n)
    (d : Fin N → BitVec n) (i : Fin N) : Function.Injective fun x => duoRun k a r s (Function.update d i x) := by
  intro x y h
  by_contra hxy
  exact Accounting.FanInProofs.foldl_ne (fun j t u => duoWord (k j) (a j) (r j) t u)
    (fun j t => duoWord_word_injective (k j) (a j) (r j) t) (fun j u => duoWord_state_injective (k j) (a j) (r j) u)
    s i (Function.update d i x) (Function.update d i y) (fun j hj => by simp [hj]) (by simpa using hxy) h

theorem fx1Run_update_injective {n N : ℕ} (m₁ : Fin N → OddKey n) (k a : Fin N → BitVec n) (r : Fin N → ℕ)
    (s : BitVec n × BitVec n) (d : Fin N → BitVec n) (i : Fin N) :
    Function.Injective fun x => fx1Run (fun j => (m₁ j).1) k a r s (Function.update d i x) := by
  intro x y h
  by_contra hxy
  exact Accounting.FanInProofs.foldl_ne (fun j t u => duoWord (k j) (a j) (r j) t (u * (m₁ j).1))
    (fun j t _ _ huv => mul_odd_cancel (m₁ j).2 (duoWord_word_injective (k j) (a j) (r j) t huv))
    (fun j u => duoWord_state_injective (k j) (a j) (r j) (u * (m₁ j).1))
    s i (Function.update d i x) (Function.update d i y) (fun j hj => by simp [hj]) (by simpa using hxy) h

end Pouw.ReceiptProofs

namespace Pouw.SecurityProofs.Receipt

theorem DuoStepBijective : Pouw.Guarantees.Receipt.DuoStepBijective := fun _ k a r d =>
  ⟨Pouw.ReceiptProofs.duoWord_state_injective k a r d, Pouw.ReceiptProofs.duoWord_state_surjective k a r d⟩

theorem DuoRunWordInjective : Pouw.Guarantees.Receipt.DuoRunWordInjective :=
  fun _ _ k a r s d i => Pouw.ReceiptProofs.duoRun_update_injective k a r s d i

theorem Fx1RunWordInjective : Pouw.Guarantees.Receipt.Fx1RunWordInjective :=
  fun _ _ m₁ k a r s d i => Pouw.ReceiptProofs.fx1Run_update_injective m₁ k a r s d i

end Pouw.SecurityProofs.Receipt
