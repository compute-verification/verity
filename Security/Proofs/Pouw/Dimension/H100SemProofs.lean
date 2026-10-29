import Definitions.Pouw.Assumptions.Dimension.H100Sem

/-!
# What each `sm_90` instruction class computes: proofs

* **The documented `k`s** (`TcKind.minK_mem`, `TcKind.minK_le`): each type's least `k` is one of its `k`s.
* **`MH100w` credits a word 32, whatever `L`** (`anyFunctionCeiling`): one `other` instruction a target writes every
  finite set of targets at `32` each. **The tensor-core premise alone buys nothing** (`tcOnlyCeiling`): that program
  has no tensor-core instruction, so it meets `Sm90TcOnly`.
* **The rank fact** (`innerProductRank`): an `L`-term inner product of fresh variables needs `L` rank-one terms.
* **The counting lemma's arithmetic** (`bamFloorArith`): `BamFloor`'s minimisation over the number of words a
  tensor-core register writes.
* **Range-forced purity** (`Lane.pure_of_fitsOn`, `pureLanes_of_wfOn`, `pureLanes_s8_rfp`): on noise spanning more than
  half a type's range, its lanes are pure; s8 lanes on noise in `[−64, 64]` are.
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

/-- **The counting lemma's arithmetic**: with `p` of the `N` words written by a tensor-core register, `N` words and at least
`max(p, N − p)` other forms (`max(p/2, N − p)` when two words may share a pre-image) cost at least `48` a word
(`128/3`). -/
theorem bamFloorArith (N p : ℚ) :
    N * 48 ≤ 32 * (N + max p (N - p)) ∧ N * (128 / 3) ≤ 32 * (N + max (p / 2) (N - p)) := by
  have h1 := le_max_left p (N - p)
  have h2 := le_max_right p (N - p)
  have h3 := le_max_left (p / 2) (N - p)
  have h4 := le_max_right (p / 2) (N - p)
  constructor <;> linarith

/-- **Range-forced purity**: a lane that fits a type on noise in `[nlo, nhi]` carries at most one noise variable once
the type's range is narrower than `2·(nhi − nlo)`. Two variables with nonzero integer coefficients move the lane by at
least `2·(nhi − nlo)` between two corners of the noise box. -/
theorem Lane.pure_of_fitsOn {α : Type} [Fintype α] {nlo nhi : ℤ} {τ : BamType} {u : Lane α}
    (hle : nlo ≤ nhi) (hw : τ.hi - τ.lo < 2 * (nhi - nlo)) (hu : u.FitsOn nlo nhi τ) : u.Pure := by
  classical
  intro i j hi hj
  by_contra hij
  let lo1 : α → ℤ := fun k => if 0 < u (some k) then nlo else nhi
  let hi1 : α → ℤ := fun k => if 0 < u (some k) then nhi else nlo
  let e₁ : α → ℤ := fun k => if k = i ∨ k = j then lo1 k else nlo
  let e₂ : α → ℤ := fun k => if k = i ∨ k = j then hi1 k else nlo
  have hr₁ : ∀ k, nlo ≤ e₁ k ∧ e₁ k ≤ nhi := by
    intro k; simp only [e₁, lo1]; split_ifs <;> omega
  have hr₂ : ∀ k, nlo ≤ e₂ k ∧ e₂ k ≤ nhi := by
    intro k; simp only [e₂, hi1]; split_ifs <;> omega
  have gain : ∀ k, (k = i ∨ k = j) → nhi - nlo ≤ u (some k) * (e₂ k - e₁ k) := by
    intro k hk
    have hne : u (some k) ≠ 0 := by rcases hk with rfl | rfl <;> assumption
    simp only [e₁, e₂, lo1, hi1, hk, ↓reduceIte]
    split_ifs with hpos
    · nlinarith
    · have : u (some k) ≤ -1 := by omega
      nlinarith
  have hdiff : u.eval e₂ - u.eval e₁ = u (some i) * (e₂ i - e₁ i) + u (some j) * (e₂ j - e₁ j) := by
    have : u.eval e₂ - u.eval e₁ = ∑ k, u (some k) * (e₂ k - e₁ k) := by
      simp only [Lane.eval, mul_sub, sum_sub_distrib]; ring
    rw [this, Fintype.sum_eq_add i j hij]
    intro k ⟨hki, hkj⟩
    simp [e₁, e₂, hki, hkj]
  have h₁ := hu e₁ hr₁
  have h₂ := hu e₂ hr₂
  have gi := gain i (Or.inl rfl)
  have gj := gain j (Or.inr rfl)
  omega

/-- **Range-forced purity of a program**: on noise in `[nlo, nhi]`, a program whose every type is narrower than
`2·(nhi − nlo)` has pure lanes, so `CoverCount`'s covering count applies to it. -/
theorem pureLanes_of_wfOn {α β : Type} [Fintype α] [Fintype β] {nlo nhi : ℤ} {Ts : List BamType}
    (hle : nlo ≤ nhi) (hTs : ∀ τ ∈ Ts, τ.hi - τ.lo < 2 * (nhi - nlo)) (P : List (BamOp α β))
    (hP : ∀ o ∈ P, o.WFOn nlo nhi Ts) : PureLanes P := by
  intro o ho
  have h := hP o ho
  cases o with
  | tc τ src u v =>
    obtain ⟨hτ, hu, hv⟩ := h
    exact ⟨fun t => Lane.pure_of_fitsOn hle (hTs τ hτ) (hu t),
      fun t => Lane.pure_of_fitsOn hle (hTs τ hτ) (hv t)⟩
  | merge c => trivial
  | copy i => trivial
  | const q => trivial

/-- **s8 lanes on noise in `[−64, 64]` are pure**: the noise spans 129 values, so int7 data plus noise is an s8 value
and no s8 lane sums two noise variables. -/
theorem pureLanes_s8_rfp {α β : Type} [Fintype α] [Fintype β] (P : List (BamOp α β))
    (hP : ∀ o ∈ P, o.WFOn (-64) 64 [⟨32, -128, 127, 32⟩]) : PureLanes P :=
  pureLanes_of_wfOn (by norm_num) (by decide) P hP

end Pouw.Dimension.H100
