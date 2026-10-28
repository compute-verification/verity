import Mathlib
import Proofs.Pouw.RelativeSecurity

/-!
# Attention's noise at every operand value

On the design `dP`/`dX` (`Definitions.Pouw.RelativeSecurity`): the peeled side `P̃ = P + U`, `U` of period `L` per
row from the planes `β_p(i, j)` at weights `w`, and the served side `X = V + b` with a fresh coin per entry. Adopted
P·V (H1's `peel`: `P̃ = P_prob − 64 + U(·, κ mod 32)`, `X = V + b`) is this design with `L = 32`.

* `pairDiff_dPrefix_operands`: a committed prefix word's second differences don't read the operands, so every floor
  the chain lemma (`chain_products_ge`, `chainBound_gProd`) gives holds at every `P` and `V` alike: the zero instance,
  a one-hot or uniform `P`, `V = 0`. A cheater gains nothing on the words by choosing operands, so the zero-operand case
  (`K = V = 0` for one KV head) leaves the words' floor whole.
* `dPrefix_split`: the word at `(P, V)` is the noise's own product's word (the word at `P = 0`) plus the clean
  product `Σ P X` (`dClean`), which is bit-affine (`crossRankLe_dClean`: cross rank 0). The peel, the noise's own
  product, is therefore the committed word less a bit-affine word, and the peeled served output `Σ P X` forces no
  product at any `P`: what the honest prover spends on it beyond the words has no floor.
* `pairDiff_dPrefix_noNoise`, `not_chainBound_noNoise`: with no noise on the peeled side (`w = 0`, one-sided noise)
  every word is bit-affine and no chain bound holds, for any class of products and any factor: B3's collapse is
  exactly the one-sided case.
-/

namespace Pouw.RelativeSecurity

open Finset

section AttnNoise

variable {R : Type} [CommRing R] {n L np B n' : ℕ}

theorem dPrefix_eq_sum (pc : Fin n → Fin B → Fin L → R) (vq : Fin B → Fin L → Fin n' → R) (w : Fin np → R)
    (i : Fin n) (l : Fin n') (t : ℕ) :
    dPrefix pc vq w i l t = fun z => ∑ t' : Fin B, (if t'.val ≤ t then 1 else 0) * dIncr pc vq w i l t' z := by
  funext z
  simp only [dPrefix, ite_mul, one_mul, zero_mul]

/-- **The words' second differences don't read the operands**: a committed prefix word at `(P, V)` has, on every
pair of distinct bits, the second difference it has at any other `(P′, V′)`, the zero instance among them. -/
theorem pairDiff_dPrefix_operands (pc pc' : Fin n → Fin B → Fin L → R) (vq vq' : Fin B → Fin L → Fin n' → R)
    (w : Fin np → R) (i : Fin n) (l : Fin n') (t : ℕ) {u v : DBits n L np B n'} (huv : u ≠ v) :
    pairDiff (dPrefix pc vq w i l t) u v = pairDiff (dPrefix pc' vq' w i l t) u v := by
  rw [dPrefix_eq_sum, dPrefix_eq_sum, pairDiff_sum, pairDiff_sum]
  refine Finset.sum_congr rfl fun t' _ => ?_
  rw [pairDiff_mul_left, pairDiff_mul_left, pairDiff_dIncr _ _ _ _ _ _ huv, pairDiff_dIncr _ _ _ _ _ _ huv]
  rfl

/-- The clean product's prefix `Σ_{t′ ≤ t} Σ_j P_(i, t′L+j) X_(t′L+j, l)`: bit-affine, the coins entering through `X`
alone. At `t = B − 1` it is the peeled served output of cell `(i, l)`. -/
def dClean (pc : Fin n → Fin B → Fin L → R) (vq : Fin B → Fin L → Fin n' → R) (i : Fin n) (l : Fin n') (t : ℕ) :
    BitAff (DBits n L np B n') R :=
  ⟨∑ t' : Fin B, if t'.val ≤ t then ∑ j, pc i t' j * vq t' j l else 0,
    Sum.elim (fun _ => 0) fun b => if b.2.2 = l then (if b.1.val ≤ t then pc i b.1 b.2.1 else 0) else 0⟩

theorem dClean_eval (pc : Fin n → Fin B → Fin L → R) (vq : Fin B → Fin L → Fin n' → R) (i : Fin n) (l : Fin n')
    (t : ℕ) (z : DBits n L np B n' → R) :
    (dClean (np := np) pc vq i l t).eval z =
      ∑ t' : Fin B, if t'.val ≤ t then ∑ j, pc i t' j * (dX vq t' j l).eval z else 0 := by
  simp only [dX_eval]
  rw [BitAff.eval]
  simp only [dClean, Fintype.sum_sum_type, Sum.elim_inl, Sum.elim_inr, zero_mul, Finset.sum_const_zero, zero_add,
    Fintype.sum_prod_type, ite_mul, Finset.sum_ite_eq', Finset.mem_univ, ite_true]
  rw [← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun t' _ => ?_
  by_cases h : t'.val ≤ t <;> simp [h, mul_add, Finset.sum_add_distrib]

theorem dIncr_split (pc : Fin n → Fin B → Fin L → R) (vq : Fin B → Fin L → Fin n' → R) (w : Fin np → R)
    (i : Fin n) (l : Fin n') (t : Fin B) (z : DBits n L np B n' → R) :
    dIncr pc vq w i l t z = dIncr 0 vq w i l t z + ∑ j, pc i t j * (dX vq t j l).eval z := by
  unfold dIncr
  rw [← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [dP_eval pc, dP_eval 0]
  simp only [Pi.zero_apply]
  ring

/-- **A noised word is the noise's own product plus the clean product**: the prefix word at `(P, V)` is the word at
`P = 0` (the noise's product `U·X`, which a peel subtracts) plus `dClean`, a bit-affine word. -/
theorem dPrefix_split (pc : Fin n → Fin B → Fin L → R) (vq : Fin B → Fin L → Fin n' → R) (w : Fin np → R)
    (i : Fin n) (l : Fin n') (t : ℕ) (z : DBits n L np B n' → R) :
    dPrefix pc vq w i l t z = dPrefix 0 vq w i l t z + (dClean pc vq i l t).eval z := by
  rw [dClean_eval]
  simp only [dPrefix, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun t' _ => ?_
  split_ifs
  · exact dIncr_split pc vq w i l t' z
  · simp

/-- **The peeled served output forces no product**: its cross block is 0 at every `P` and `V`. -/
theorem crossRankLe_dClean (pc : Fin n → Fin B → Fin L → R) (vq : Fin B → Fin L → Fin n' → R) (i : Fin n)
    (l : Fin n') (t : ℕ) : CrossRankLe (fun z => (dClean (np := np) pc vq i l t).eval z) 0 :=
  ⟨Fin.elim0, Fin.elim0, fun _ _ => by simpa using pairDiff_aff _ Sum.inl_ne_inr⟩

/-- With no noise on the peeled side every committed word is bit-affine. -/
theorem pairDiff_dPrefix_noNoise (pc : Fin n → Fin B → Fin L → R) (vq : Fin B → Fin L → Fin n' → R)
    (i : Fin n) (l : Fin n') (t : ℕ) {u v : DBits n L np B n'} (huv : u ≠ v) :
    pairDiff (dPrefix pc vq 0 i l t) u v = 0 := by
  rw [dPrefix_eq_sum, pairDiff_sum]
  refine Finset.sum_eq_zero fun t' _ => ?_
  rw [pairDiff_mul_left, pairDiff_dIncr _ _ _ _ _ _ huv]
  cases u <;> cases v <;> simp [dP]

/-- **One-sided noise has no floor**: with `w = 0` no chain bound holds, for any class of products and any factor;
at the zero instance every word is 0. -/
theorem not_chainBound_noNoise (hL : 0 < L) (hn : 0 < n) (hn' : 0 < n') (hB : 0 < B)
    (Ok : ((DBits n L np B n' → R) → R) → Prop) (c : ℕ) : ¬ ChainBound R n L np B n' 0 Ok c := by
  intro h
  have := h Empty Unit ∅ (fun k => k.elim) (by simp) 0 0 ⟨0, hn⟩ (fun _ => (⟨0, hn'⟩, ⟨0, hB⟩))
    (fun _ _ _ => rfl) (fun _ k => k.elim) (fun _ => ⟨0, fun _ => 0⟩) (fun _ U _ => by
      simp [dPrefix, dIncr, BitAff.eval, dP])
  simp at this
  omega

end AttnNoise

end Pouw.RelativeSecurity
