import Definitions.Pouw.Assumptions.Dimension.H100Sem

/-!
# What each `sm_90` instruction class computes: proofs

* **The documented `k`s** (`TcKind.minK_mem`, `TcKind.minK_le`): each type's least `k` is one of its `k`s.
* **`MH100w` credits a word 32, whatever `L`** (`anyFunctionCeiling`): one `other` instruction a target writes every
  finite set of targets at `32` each. **The tensor-core premise alone buys nothing** (`tcOnlyCeiling`): that program
  has no tensor-core instruction, so it meets `Sm90TcOnly`.
* **The rank fact** (`innerProductRank`): an `L`-term inner product of fresh variables needs `L` rank-one terms.
* **The counting lemma's arithmetic** (`lcommitFloorArith`): `BamFloor`'s minimisation over the number of words a
  tensor-core register writes.
-/

namespace Pouw.Dimension.H100

open Finset

variable {G : Type}

theorem TcKind.minK_mem (κ : TcKind) : κ.minK ∈ κ.ks := by cases κ <;> decide

theorem TcKind.minK_le (κ : TcKind) : ∀ k ∈ κ.ks, κ.minK ≤ k := by cases κ <;> decide

theorem regs100_map_other (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (l L : List (G → ℚ)) :
    regs100 fpSem (l.map Op100.other) L = L ++ l := by
  induction l generalizing L with
  | nil => simp [regs100]
  | cons f l ih =>
    have h := ih (L ++ [f])
    simp only [regs100, List.map_cons, List.foldl_cons, Op100.outs] at h ⊢
    rw [h, List.append_assoc, List.singleton_append]

theorem cost100_map_other (pr : Prices100) (l : List (G → ℚ)) :
    cost100 pr (l.map Op100.other) = pr.other * l.length := by
  induction l with
  | nil => simp [cost100]
  | cons f l ih =>
    simp only [cost100, List.map_map, List.map_cons, List.sum_cons, List.length_cons] at ih ⊢
    rw [ih]; simp [Op100.price]; ring

/-- **`MH100w` credits a word 32, whatever `L`**: for every finite set of targets, writing each with one `other`
instruction is a well-formed program at `h100Floor` that outputs them all at exactly `32·|T|`. -/
theorem anyFunctionCeiling (F : Set (G → ℚ)) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (T : Finset (G → ℚ))
    (L : List (G → ℚ)) :
    ∃ P : List (Op100 G), (∀ o ∈ P, o.WFw h100Floor F) ∧ (∀ t ∈ T, t ∈ regs100 fpSem P L) ∧
      cost100 h100Floor P = 32 * T.card := by
  refine ⟨T.toList.map Op100.other, ?_, ?_, ?_⟩
  · intro o ho
    obtain ⟨f, -, rfl⟩ := List.mem_map.1 ho
    trivial
  · intro t ht
    rw [regs100_map_other]
    exact List.mem_append_right _ (Finset.mem_toList.2 ht)
  · rw [cost100_map_other, Finset.length_toList]
    rfl

/-- **The tensor-core premise alone buys nothing**: the same program meets it (it has no tensor-core instruction). -/
theorem tcOnlyCeiling (I : Sm90Isa) (F : Set (G → ℚ)) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (T : Finset (G → ℚ))
    (L : List (G → ℚ)) :
    ∃ P : List (Op100 G), (∀ o ∈ P, o.WFw h100Floor F) ∧ Sm90TcOnly I fpSem F L P ∧
      (∀ t ∈ T, t ∈ regs100 fpSem P L) ∧ cost100 h100Floor P = 32 * T.card := by
  refine ⟨T.toList.map Op100.other, ?_, ?_, ?_, ?_⟩
  · intro o ho
    obtain ⟨f, -, rfl⟩ := List.mem_map.1 ho
    trivial
  · intro i hi
    simp [Op100.TcOnly]
  · intro t ht
    rw [regs100_map_other]
    exact List.mem_append_right _ (Finset.mem_toList.2 ht)
  · rw [cost100_map_other, Finset.length_toList]
    rfl

/-- **The rank fact**: if `r` rank-one terms `α p ⊗ β p` sum to the identity on `L` coordinates, then `L ≤ r`. So an
`L`-term inner product of fresh variables needs `L` products, at most 32 to an accumulator register. -/
theorem innerProductRank (L r : ℕ) (α β : Fin r → Fin L → ℚ)
    (h : ∀ k l, ∑ p, α p k * β p l = if k = l then 1 else 0) : L ≤ r := by
  let A : Matrix (Fin L) (Fin r) ℚ := Matrix.of fun k p => α p k
  let B : Matrix (Fin r) (Fin L) ℚ := Matrix.of fun p l => β p l
  have hAB : A * B = 1 := by
    ext k l
    simp only [A, B, Matrix.mul_apply, Matrix.of_apply, Matrix.one_apply]
    exact h k l
  have h1 : (1 : Matrix (Fin L) (Fin L) ℚ).rank = L := by simp
  have h2 : (A * B).rank ≤ A.rank := Matrix.rank_mul_le_left A B
  have h3 : A.rank ≤ r := by simpa using Matrix.rank_le_width A
  rw [hAB, h1] at h2
  omega

/-- **The counting lemma's arithmetic**: with `p ≤ N` words of the chain kind at `L/32` registers each and every other
word paying a merge, `32·max(p·L/32, N) + 32·(N − p) ≥ N·(64 − 1024/L)`. -/
theorem lcommitFloorArith (L N p : ℚ) (hL : 32 ≤ L) (hp0 : 0 ≤ p) (hpN : p ≤ N) :
    N * (64 - 1024 / L) ≤ 32 * max (p * L / 32) N + 32 * (N - p) := by
  have hL0 : 0 < L := by linarith
  rcases le_total (p * L / 32) N with h | h
  · rw [max_eq_right h]
    have : p ≤ 32 * N / L := by
      rw [le_div_iff₀ hL0]; linarith
    have e : N * (64 - 1024 / L) = 64 * N - 32 * (32 * N / L) := by field_simp; ring
    rw [e]; linarith
  · rw [max_eq_left h]
    have hN : 0 ≤ N := le_trans hp0 hpN
    have e : N * (64 - 1024 / L) = 64 * N - 1024 * N / L := by field_simp
    have h' : 32 * N ≤ p * L := by linarith
    have hq : 1024 * N / L ≤ 32 * p := by
      rw [div_le_iff₀ hL0]; nlinarith
    have : 32 * N + p * L - 32 * p ≥ 64 * N - 1024 * N / L := by
      have : p * L - 32 * p ≥ 32 * N - 1024 * N / L := by
        have h4 : (p - 32 * N / L) * (L - 32) ≥ 0 := by
          apply mul_nonneg
          · rw [sub_nonneg, div_le_iff₀ hL0]; linarith
          · linarith
        have e2 : (p - 32 * N / L) * (L - 32) = p * L - 32 * p - 32 * N + 1024 * N / L := by
          field_simp; ring
        linarith
      linarith
    linarith

end Pouw.Dimension.H100
