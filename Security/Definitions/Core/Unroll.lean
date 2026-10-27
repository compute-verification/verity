import Definitions.Core.Circuit

/-!
# A fixed machine run for `T` steps

A **fixed machine** is one circuit `H`, a single step, run again and again: each step reads the state the step before
wrote and writes the next. `H`'s state wires (`StateWires`) are its first `k` gates, the state it reads, which are input
gates, and for each state wire `i`, the gate `out i` holding it after the step. Everything a step does (control,
decoding, generation, selection, retries) is inside `H`; each step's other input gates are fresh input gates of the run.

`H.unroll S T` is `H` run `T` steps along `S`: the `k` initial state gates, input gates, then `T` copies of `H`'s other
`len = N - k` gates, copy `t` holding `H`'s gate `k + j` at gate `k + len · t + j`. A copy reads its own gates, and the
state from the previous copy's `out` gates or, at step `0`, from the initial state (`StateWires.copy`), and applies `H`'s
operations. A gate of the unrolling holds `H`'s gate `src i`. Its size is `T` times `H`'s
(`Verity.Circuit.size_unroll`), and it is Boolean when `H` is (`Verity.Circuit.unroll_boolean`).

`Iso C D` (`C ≅ D`) is a bijection of the gates that keeps each gate's width, whether it is an input gate, the wires into
it and its operation: `D` is `C` with its gates renumbered. Output gates are not kept.
-/

namespace Verity

open Finset

namespace Circuit

/-- **An isomorphism** `C ≅ D`: a bijection `e` of the gates that keeps each gate's width, whether it is an input gate,
the wires into it and, for a computed gate, its operation. -/
structure Iso (C D : Circuit) where
  e : Fin C.N ≃ Fin D.N
  w_e : ∀ g, D.w (e g) = C.w g
  isInput_e : ∀ g, D.isInput (e g) = C.isInput g
  reads_e : ∀ g, D.reads (e g) = (C.reads g).map e.toEmbedding
  op_e : ∀ g, C.isInput g = false → ∀ v : D.Transcript,
    (D.op (e g) v).cast (w_e g) = C.op g fun a => (v (e a)).cast (w_e a)

/-- The transcript of `C` that a transcript `X` of `D` gives through `φ`: each gate takes `X`'s value at its image. -/
def Iso.pull {C D : Circuit} (φ : Iso C D) (X : D.Transcript) : C.Transcript := fun g => (X (φ.e g)).cast (φ.w_e g)

/-- Every circuit is isomorphic to itself. -/
def Iso.refl (C : Circuit) : Iso C C where
  e := Equiv.refl _
  w_e _ := rfl
  isInput_e _ := rfl
  reads_e _ := Finset.map_refl.symm
  op_e _ _ _ := rfl

/-- **The state wires** of a step `H`: its first `k` gates are the state it reads, input gates, and `out i` is the gate
past them holding state wire `i` after the step, of the same width. -/
structure StateWires (H : Circuit) where
  k : ℕ
  k_le : k ≤ H.N
  input_lt : ∀ g : Fin H.N, (g : ℕ) < k → H.isInput g = true
  out : Fin k → Fin H.N
  le_out : ∀ i, k ≤ (out i : ℕ)
  w_out : ∀ i, H.w (out i) = H.w ⟨i, lt_of_lt_of_le i.isLt k_le⟩

namespace StateWires

variable {H : Circuit} (S : H.StateWires) (T : ℕ)

/-- The gates of a step past the state it reads, `N - k`. -/
def len : ℕ := H.N - S.k

theorem k_add_len : S.k + S.len = H.N := by
  have := S.k_le
  unfold len
  omega

/-- The step of a gate `i ≥ k` of `H` run `T` steps: `⌊(i - k) / len⌋`. -/
def step (i : ℕ) : ℕ := (i - S.k) / S.len

theorem len_pos {i : ℕ} (hk : S.k ≤ i) (hi : i < S.k + S.len * T) : 0 < S.len := by
  rcases Nat.eq_zero_or_pos S.len with h | h
  · rw [h, Nat.zero_mul] at hi
    omega
  · exact h

theorem step_lt {i : ℕ} (hk : S.k ≤ i) (hi : i < S.k + S.len * T) : S.step i < T := by
  unfold step
  rw [Nat.div_lt_iff_lt_mul (S.len_pos T hk hi), Nat.mul_comm]
  omega

theorem mod_len {t j : ℕ} (hj : j < S.len) : (S.len * t + j) % S.len = j := by
  rw [Nat.add_comm, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hj]

/-- The gate of `H` that gate `i` of `H` run `T` steps holds: itself in the initial state, and gate
`k + (i - k) mod len` past it. -/
def src (i : Fin (S.k + S.len * T)) : Fin H.N :=
  if h : (i : ℕ) < S.k then ⟨i, lt_of_lt_of_le h S.k_le⟩
  else ⟨S.k + ((i : ℕ) - S.k) % S.len, by
    have := Nat.mod_lt ((i : ℕ) - S.k) (S.len_pos T (Nat.le_of_not_lt h) i.isLt)
    have := S.k_add_len
    omega⟩

/-- The gate of `H` run `T` steps that holds `H`'s gate `b` at step `t < T`: past the state, `b`'s place in copy `t`; a
state gate at step `0`, the initial state; and a state gate at a later step, the gate of the copy before that wrote it. -/
def copy {t : ℕ} (ht : t < T) (b : Fin H.N) : Fin (S.k + S.len * T) :=
  if hb : S.k ≤ (b : ℕ) then ⟨S.k + (S.len * t + ((b : ℕ) - S.k)), by
    have h1 := Nat.mul_le_mul_left S.len (show t + 1 ≤ T from ht)
    rw [Nat.mul_add, Nat.mul_one] at h1
    have := S.k_add_len
    have := b.isLt
    omega⟩
  else if h0 : t = 0 then ⟨b, by omega⟩
  else ⟨S.k + (S.len * (t - 1) + ((S.out ⟨b, Nat.lt_of_not_le hb⟩ : ℕ) - S.k)), by
    have h1 := Nat.mul_le_mul_left S.len (show t - 1 + 1 ≤ T by omega)
    rw [Nat.mul_add, Nat.mul_one] at h1
    have := S.k_add_len
    have := (S.out ⟨b, Nat.lt_of_not_le hb⟩).isLt
    have := S.le_out ⟨b, Nat.lt_of_not_le hb⟩
    omega⟩

theorem src_val_of_le {i : Fin (S.k + S.len * T)} (h : S.k ≤ (i : ℕ)) :
    (S.src T i : ℕ) = S.k + ((i : ℕ) - S.k) % S.len := by
  unfold src
  rw [dite_eq_right (Nat.not_lt.2 h)]

theorem src_copy {t : ℕ} (ht : t < T) (b : Fin H.N) :
    S.src T (S.copy T ht b) = if hb : S.k ≤ (b : ℕ) then b else if t = 0 then b
      else S.out ⟨b, Nat.lt_of_not_le hb⟩ := by
  have hN := S.k_add_len
  apply Fin.ext
  by_cases hb : S.k ≤ (b : ℕ)
  · have hj : (b : ℕ) - S.k < S.len := by have := b.isLt; omega
    rw [dite_eq_left hb, S.src_val_of_le T (by unfold copy; rw [dite_eq_left hb]; dsimp only; omega)]
    unfold copy
    rw [dite_eq_left hb]
    dsimp only
    rw [Nat.add_sub_cancel_left, S.mod_len hj]
    omega
  · by_cases h0 : t = 0
    · rw [dite_eq_right hb, ite_eq_left h0]
      unfold src copy
      rw [dite_eq_right hb, dite_eq_left h0]
      dsimp only
      rw [dite_eq_left (Nat.lt_of_not_le hb)]
    · have ho := S.le_out ⟨b, Nat.lt_of_not_le hb⟩
      have hj : (S.out ⟨b, Nat.lt_of_not_le hb⟩ : ℕ) - S.k < S.len := by
        have := (S.out ⟨b, Nat.lt_of_not_le hb⟩).isLt
        omega
      rw [dite_eq_right hb, ite_eq_right h0, S.src_val_of_le T (by unfold copy; rw [dite_eq_right hb, dite_eq_right h0]; dsimp only; omega)]
      unfold copy
      rw [dite_eq_right hb, dite_eq_right h0]
      dsimp only
      rw [Nat.add_sub_cancel_left, S.mod_len hj]
      omega

theorem w_copy {t : ℕ} (ht : t < T) (b : Fin H.N) : H.w (S.src T (S.copy T ht b)) = H.w b := by
  rw [S.src_copy T ht b]
  by_cases hb : S.k ≤ (b : ℕ)
  · rw [dite_eq_left hb]
  · rw [dite_eq_right hb]
    by_cases h0 : t = 0
    · rw [ite_eq_left h0]
    · rw [ite_eq_right h0, S.w_out]

theorem copy_lt (i : Fin (S.k + S.len * T)) (hk : S.k ≤ (i : ℕ)) {b : Fin H.N} (hb : b < S.src T i) :
    S.copy T (S.step_lt T hk i.isLt) b < i := by
  have hN := S.k_add_len
  have hdm := Nat.div_add_mod ((i : ℕ) - S.k) S.len
  have hst : S.step i = ((i : ℕ) - S.k) / S.len := rfl
  rw [← hst] at hdm
  rw [Fin.lt_def, S.src_val_of_le T hk] at hb
  rw [Fin.lt_def]
  unfold copy
  split_ifs with h1 h0
  · dsimp only
    omega
  · dsimp only
    omega
  · dsimp only
    have := (S.out ⟨b, Nat.lt_of_not_le h1⟩).isLt
    have := S.le_out ⟨b, Nat.lt_of_not_le h1⟩
    have h2 := Nat.mul_le_mul_left S.len (show S.step i - 1 + 1 ≤ S.step i by omega)
    rw [Nat.mul_add, Nat.mul_one] at h2
    omega

end StateWires

/-- **`H` run `T` steps** along its state wires `S`: gate `i` holds `H`'s gate `src i`, with its width and whether it is
an input gate. Past the initial state, gate `i` of step `t` reads the gates holding, at step `t`, the gates `src i`
reads in `H` (`StateWires.copy`), and applies `src i`'s operation to them. Its output gates are those holding an output
gate of `H`. -/
def unroll (H : Circuit) (S : H.StateWires) (T : ℕ) : Circuit where
  N := S.k + S.len * T
  w i := H.w (S.src T i)
  isInput i := H.isInput (S.src T i)
  reads i :=
    if h : S.k ≤ (i : ℕ) then (H.reads (S.src T i)).image (S.copy T (S.step_lt T h i.isLt)) else ∅
  reads_lt i a ha := by
    by_cases h : S.k ≤ (i : ℕ)
    · simp only [dite_eq_left h] at ha
      obtain ⟨b, hb, rfl⟩ := Finset.mem_image.1 ha
      exact S.copy_lt T i h (H.reads_lt _ b hb)
    · simp [dite_eq_right h] at ha
  input_reads i hi := by
    by_cases h : S.k ≤ (i : ℕ)
    · simp only [dite_eq_left h]
      rw [H.input_reads _ hi, Finset.image_empty]
    · simp only [dite_eq_right h]
  op i v :=
    if h : S.k ≤ (i : ℕ) then
      H.op (S.src T i) fun b => (v (S.copy T (S.step_lt T h i.isLt) b)).cast (S.w_copy T _ b)
    else 0
  op_congr i v v' hv := by
    by_cases h : S.k ≤ (i : ℕ)
    · simp only [dite_eq_left h] at hv ⊢
      exact H.op_congr _ _ _ fun b hb => by rw [hv _ (Finset.mem_image_of_mem _ hb)]
    · simp only [dite_eq_right h]
  outputs := univ.filter fun i => S.src T i ∈ H.outputs

end Circuit

end Verity
