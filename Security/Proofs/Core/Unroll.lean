import Definitions.Core.Unroll
import Definitions.Core.Formalism
import Proofs.Core.Circuit

/-!
# A fixed machine run for `T` steps: its size, its gates and isomorphisms

* `Circuit.unroll_boolean`: `H` run `T` steps is Boolean when `H` is;
* `Circuit.size_unroll`: its size is `T` times `H`'s, `|H.unroll S T| = T · |H|`: its computed gates are the copies of
  `H`'s, one for each step (`computed_unroll`);
* `Circuit.Iso.pull_eval`: an isomorphism carries evaluation, `φ.pull (D.eval a) = C.eval (φ.pull a)`;
* `Circuit.Iso.size_computed`: isomorphic circuits have the same size.
-/

namespace Verity

open Finset

namespace Circuit

namespace StateWires

variable {H : Circuit} (S : H.StateWires) (T : ℕ)

theorem div_len {t j : ℕ} (hj : j < S.len) : (S.len * t + j) / S.len = t := by
  rw [Nat.add_comm, Nat.add_mul_div_left _ _ (by omega : 0 < S.len), Nat.div_eq_of_lt hj, Nat.zero_add]

theorem src_of_lt {i : Fin (S.k + S.len * T)} (h : (i : ℕ) < S.k) :
    S.src T i = ⟨i, lt_of_lt_of_le h S.k_le⟩ := by
  unfold src
  rw [dite_eq_left h]

theorem le_of_isInput_false {b : Fin H.N} (hb : H.isInput b = false) : S.k ≤ (b : ℕ) := by
  by_contra h
  rw [S.input_lt b (Nat.lt_of_not_le h)] at hb
  simp at hb

theorem le_of_computed {i : Fin (S.k + S.len * T)} (h : H.isInput (S.src T i) = false) : S.k ≤ (i : ℕ) := by
  by_contra hk
  rw [S.src_of_lt T (Nat.lt_of_not_le hk), S.input_lt _ (Nat.lt_of_not_le hk)] at h
  simp at h

theorem copy_src (i : Fin (S.k + S.len * T)) (hk : S.k ≤ (i : ℕ)) :
    S.copy T (S.step_lt T hk i.isLt) (S.src T i) = i := by
  have hdm := Nat.div_add_mod ((i : ℕ) - S.k) S.len
  have hs := S.src_val_of_le T hk
  have hle : S.k ≤ (S.src T i : ℕ) := by rw [hs]; omega
  apply Fin.ext
  unfold copy
  rw [dite_eq_left hle]
  dsimp only
  rw [hs]
  unfold step
  omega

theorem copy_inj {t t' : ℕ} (ht : t < T) (ht' : t' < T) {b b' : Fin H.N} (hb : S.k ≤ (b : ℕ))
    (hb' : S.k ≤ (b' : ℕ)) (h : S.copy T ht b = S.copy T ht' b') : t = t' ∧ b = b' := by
  have hN := S.k_add_len
  have hj : (b : ℕ) - S.k < S.len := by have := b.isLt; omega
  have hj' : (b' : ℕ) - S.k < S.len := by have := b'.isLt; omega
  have hv := congrArg Fin.val h
  unfold copy at hv
  rw [dite_eq_left hb, dite_eq_left hb'] at hv
  dsimp only at hv
  have hv' : S.len * t + ((b : ℕ) - S.k) = S.len * t' + ((b' : ℕ) - S.k) := by omega
  have h1 := congrArg (· % S.len) hv'
  have h2 := congrArg (· / S.len) hv'
  simp only [S.mod_len hj, S.mod_len hj'] at h1
  simp only [S.div_len hj, S.div_len hj'] at h2
  exact ⟨h2, Fin.ext (by omega)⟩

end StateWires

theorem mem_computed {C : Circuit} {g : Fin C.N} : g ∈ C.computed ↔ C.isInput g = false := by
  simp [computed]

variable (H : Circuit) (S : H.StateWires) (T : ℕ)

theorem reads_unroll {i : Fin (S.k + S.len * T)} (hk : S.k ≤ (i : ℕ)) :
    (H.unroll S T).reads i = (H.reads (S.src T i)).image (S.copy T (S.step_lt T hk i.isLt)) := by
  simp only [unroll, dite_eq_left hk]

/-- **`H` run `T` steps is Boolean when `H` is.** -/
theorem unroll_boolean (hH : H.Boolean) : (H.unroll S T).Boolean where
  one_bit i := hH.one_bit (S.src T i)
  fanin_pos i hi := by
    have hk := S.le_of_computed T hi
    rw [H.reads_unroll S T hk]
    exact (hH.fanin_pos _ hi).image _

/-- Each step's copy of a computed gate of `H` is a computed gate of `H` run `T` steps. -/
theorem copy_mem_computed (t : Fin T) {b : Fin H.N} (hb : b ∈ H.computed) :
    (S.copy T t.isLt b : Fin (H.unroll S T).N) ∈ (H.unroll S T).computed := by
  rw [mem_computed] at hb
  refine mem_computed.2 ?_
  show H.isInput (S.src T (S.copy T t.isLt b)) = false
  rw [S.src_copy T t.isLt b, dite_eq_left (S.le_of_isInput_false hb)]
  exact hb

/-- Each computed gate of `H` run `T` steps is the copy, at its step, of the computed gate of `H` it holds. -/
theorem eq_copy_of_mem_computed {i : Fin (H.unroll S T).N} (hi : i ∈ (H.unroll S T).computed) :
    ∃ (t : Fin T) (b : Fin H.N), b ∈ H.computed ∧ (S.copy T t.isLt b : Fin (H.unroll S T).N) = i := by
  simp only [computed, mem_filter, mem_univ, true_and] at hi ⊢
  have hk := S.le_of_computed T hi
  exact ⟨⟨S.step i, S.step_lt T hk i.isLt⟩, S.src T i, hi, S.copy_src T i hk⟩

/-- **`|H.unroll S T| = T · |H|`**: `H` run `T` steps has `T` times `H`'s operation gates, one copy of them for each
step. -/
theorem size_unroll : (H.unroll S T).size (H.unroll S T).computed = T * H.size H.computed := by
  unfold size
  have hc : ((univ : Finset (Fin T)) ×ˢ H.computed).card = T * H.computed.card := by
    rw [Finset.card_product, Finset.card_univ, Fintype.card_fin]
  rw [← hc]
  symm
  refine Finset.card_bij (t := (H.unroll S T).computed)
    (fun p _ => (S.copy T p.1.isLt p.2 : Fin (H.unroll S T).N)) ?_ ?_ ?_
  · rintro ⟨t, b⟩ hp
    exact H.copy_mem_computed S T t (mem_product.1 hp).2
  · rintro ⟨t, b⟩ hp ⟨t', b'⟩ hp' h
    have hb := (mem_product.1 hp).2
    have hb' := (mem_product.1 hp').2
    simp only [computed, mem_filter, mem_univ, true_and] at hb hb'
    obtain ⟨h1, h2⟩ := S.copy_inj T t.isLt t'.isLt (S.le_of_isInput_false hb) (S.le_of_isInput_false hb') h
    exact Prod.ext (Fin.ext h1) h2
  · intro i hi
    obtain ⟨t, b, hb, rfl⟩ := H.eq_copy_of_mem_computed S T hi
    exact ⟨(t, b), mem_product.2 ⟨mem_univ t, hb⟩, rfl⟩

namespace Iso

variable {C D : Circuit} (φ : Iso C D)

/-- **An isomorphism carries evaluation**: `D`'s evaluation, read at the images of `C`'s gates, is `C`'s evaluation on
the inputs `D`'s gives it. -/
theorem pull_eval (a : D.Transcript) : φ.pull (D.eval a) = C.eval (φ.pull a) := by
  funext g
  induction g using WellFoundedLT.induction with
  | _ g ih => ?_
  cases hi : C.isInput g
  · have hD : D.isInput (φ.e g) = false := by rw [φ.isInput_e, hi]
    show (D.eval a (φ.e g)).cast (φ.w_e g) = C.eval (φ.pull a) g
    rw [D.eval_op a hD, C.eval_op _ hi, φ.op_e g hi]
    refine C.op_congr g _ _ fun x hx => ?_
    exact ih x (C.reads_lt g x hx)
  · have hD : D.isInput (φ.e g) = true := by rw [φ.isInput_e, hi]
    show (D.eval a (φ.e g)).cast (φ.w_e g) = C.eval (φ.pull a) g
    rw [D.eval_input a hD, C.eval_input _ hi]
    rfl

theorem computed_map : C.computed.map φ.e.toEmbedding = D.computed := by
  ext g
  simp only [computed, mem_map, mem_filter, mem_univ, true_and, Equiv.toEmbedding_apply]
  constructor
  · rintro ⟨a, ha, rfl⟩
    rw [φ.isInput_e]
    exact ha
  · intro hg
    refine ⟨φ.e.symm g, ?_, φ.e.apply_symm_apply g⟩
    rw [← φ.isInput_e, φ.e.apply_symm_apply]
    exact hg

include φ in
/-- **Isomorphic circuits have the same size.** -/
theorem size_computed : D.size D.computed = C.size C.computed := by
  unfold size
  rw [← φ.computed_map, Finset.card_map]

/-- Every gate of `D` has the width of the gate of `C` it is the image of. -/
theorem w_symm (g : Fin D.N) : D.w g = C.w (φ.e.symm g) := by
  rw [← φ.w_e, φ.e.apply_symm_apply]

/-- A gate with a wire into `S` in `C` has one into `S`'s image in `D`. -/
theorem map_mem_into {S : Finset (Fin C.N)} {b : Fin C.N} (hb : b ∈ C.into S) :
    φ.e b ∈ D.into (S.map φ.e.toEmbedding) := by
  simp only [into, mem_filter, mem_univ, true_and] at hb ⊢
  obtain ⟨hbS, g, hg, hbg⟩ := hb
  refine ⟨fun h => hbS ((mem_map' φ.e.toEmbedding).1 h), φ.e g, (mem_map' φ.e.toEmbedding).2 hg, ?_⟩
  rw [φ.reads_e]
  exact (mem_map' φ.e.toEmbedding).2 hbg

end Iso

theorem width_eq_card {C : Circuit} (hw : ∀ g, C.w g = 1) (S : Finset (Fin C.N)) : C.width S = S.card := by
  simp [width, hw]

end Circuit

end Verity
