import Mathlib

/-!
# Steiner rigidity in the rank metric (PoUW work floor)

The setting of `note:20261015T0416Z-draft-steiner-rigidity-laminar`: matrices over a field `F`, the rank metric
`d(A, B) = rank (A - B)`, and a Steiner tree as a finite tree whose vertices carry matrices, its length the sum over
its edges of the rank of the difference of the endpoints' labels.

* `BlockRigidity` is Theorem 1 (block rigidity) and `ConjectureA` is its matmul case, both stated as `Prop`s.
* Proved here: the rank metric's facts; for any connected labelled graph, the column space of the difference of two
  labels lies in the sum of the edges' column spaces; and from it the merge-proof case (Corollary 6 of the note):
  targets whose column spaces (or row spaces) are independent cost at least the sum of their ranks. One target, and
  two targets in distinct cells of a block structure, are corollaries.
-/

namespace Pouw.SteinerRigidity

open Matrix Module

section RankMetric

variable {F : Type*} [Field F] {m n : Type*} [Fintype n]

/-- The column space of `A`, the range of `v ↦ A *ᵥ v`. Its dimension is `A.rank`. -/
noncomputable def colSpace (A : Matrix m n F) : Submodule F (m → F) :=
  LinearMap.range A.mulVecLin

/-- The rank metric `d(A, B) = rank (A - B)`. -/
noncomputable def rankDist (A B : Matrix m n F) : ℕ :=
  (A - B).rank

theorem rank_eq_finrank_colSpace (A : Matrix m n F) : A.rank = finrank F (colSpace A) :=
  rfl

theorem colSpace_zero : colSpace (0 : Matrix m n F) = ⊥ := by
  rw [colSpace, mulVecLin_zero, LinearMap.range_zero]

theorem colSpace_neg (A : Matrix m n F) : colSpace (-A) = colSpace A := by
  have h : (-A).mulVecLin = -A.mulVecLin := by
    ext v i
    simp
  rw [colSpace, h, LinearMap.range_neg]
  rfl

theorem colSpace_add_le (A B : Matrix m n F) : colSpace (A + B) ≤ colSpace A ⊔ colSpace B := by
  rw [colSpace, Matrix.mulVecLin_add]
  exact LinearMap.range_add_le _ _

theorem rankDist_self (A : Matrix m n F) : rankDist A A = 0 := by
  rw [rankDist, sub_self, rank_zero]

theorem rank_neg (A : Matrix m n F) : (-A).rank = A.rank := by
  rw [rank_eq_finrank_colSpace, rank_eq_finrank_colSpace, colSpace_neg]

theorem rankDist_comm (A B : Matrix m n F) : rankDist A B = rankDist B A := by
  rw [rankDist, rankDist, ← neg_sub, rank_neg]

theorem rankDist_zero_left (A : Matrix m n F) : rankDist 0 A = A.rank := by
  rw [rankDist, zero_sub, rank_neg]

variable [Fintype m]

/-- The row space of `A`, the column space of `Aᵀ`. -/
noncomputable def rowSpace (A : Matrix m n F) : Submodule F (n → F) :=
  colSpace Aᵀ

/-- Subadditivity of rank. -/
theorem rank_add_le (A B : Matrix m n F) : (A + B).rank ≤ A.rank + B.rank := by
  rw [rank_eq_finrank_colSpace, rank_eq_finrank_colSpace, rank_eq_finrank_colSpace]
  exact (Submodule.finrank_mono (colSpace_add_le A B)).trans
    (Submodule.finrank_add_le_finrank_add_finrank _ _)

theorem rankDist_eq_zero_iff (A B : Matrix m n F) : rankDist A B = 0 ↔ A = B := by
  classical
  rw [rankDist, rank_eq_finrank_colSpace, Submodule.finrank_eq_zero, colSpace,
    LinearMap.range_eq_bot]
  constructor
  · intro h
    rw [← sub_eq_zero]
    ext i j
    have := congrFun (LinearMap.congr_fun h (Pi.single j 1)) i
    simpa [Matrix.mulVecLin_apply, Matrix.mulVec_single_one] using this
  · intro h
    rw [h, sub_self, mulVecLin_zero]

/-- The triangle inequality of the rank metric. -/
theorem rankDist_triangle (A B C : Matrix m n F) : rankDist A C ≤ rankDist A B + rankDist B C := by
  have h : A - C = (A - B) + (B - C) := by abel
  rw [rankDist, h]
  exact rank_add_le _ _

theorem rankDist_transpose (A B : Matrix m n F) : rankDist Aᵀ Bᵀ = rankDist A B := by
  rw [rankDist, rankDist, ← Matrix.transpose_sub, Matrix.rank_transpose]

end RankMetric

section Steiner

variable {F : Type*} [Field F] {m n : Type*} [Fintype m] [Fintype n]

/-- A Steiner tree in the rank metric: a finite tree whose vertices carry matrices. -/
structure SteinerTree (F : Type*) [Field F] (m n : Type*) where
  /-- The vertices. -/
  V : Type
  [instFintype : Fintype V]
  [instDecEq : DecidableEq V]
  /-- The tree. -/
  G : SimpleGraph V
  [instDecAdj : DecidableRel G.Adj]
  isTree : G.IsTree
  /-- Each vertex's matrix. -/
  label : V → Matrix m n F

attribute [instance] SteinerTree.instFintype SteinerTree.instDecEq SteinerTree.instDecAdj

namespace SteinerTree

/-- The length of an edge, the rank distance between its endpoints' labels. -/
noncomputable def edgeLength {V : Type} (label : V → Matrix m n F) : Sym2 V → ℕ :=
  Sym2.lift ⟨fun u v => rankDist (label u) (label v), fun _ _ => rankDist_comm _ _⟩

/-- The length of a Steiner tree: the sum over its edges of `rank (label u - label v)`. -/
noncomputable def length (T : SteinerTree F m n) : ℕ :=
  ∑ e ∈ T.G.edgeFinset, edgeLength T.label e

/-- The tree has a vertex labelled `A`. -/
def Reaches (T : SteinerTree F m n) (A : Matrix m n F) : Prop :=
  ∃ v, T.label v = A

/-- The same tree with every label transposed. -/
def transpose (T : SteinerTree F m n) : SteinerTree F n m where
  V := T.V
  G := T.G
  isTree := T.isTree
  label v := (T.label v)ᵀ

theorem length_transpose (T : SteinerTree F m n) : T.transpose.length = T.length := by
  refine Finset.sum_congr rfl fun e _ => ?_
  induction e using Sym2.ind with
  | _ u v => exact rankDist_transpose _ _

omit [Fintype m] [Fintype n] in
theorem reaches_transpose {T : SteinerTree F m n} {A : Matrix m n F} (h : T.Reaches A) :
    T.transpose.Reaches Aᵀ := by
  obtain ⟨v, hv⟩ := h
  exact ⟨v, by simp [transpose, hv]⟩

end SteinerTree

end Steiner

section Subspaces

variable {K W : Type*} [Field K] [AddCommGroup W] [Module K W] [FiniteDimensional K W]

/-- The dimension of a finite sum of subspaces is at most the sum of their dimensions. -/
theorem finrank_biSup_le_sum {ι : Type*} (p : ι → Submodule K W) (s : Finset ι) :
    finrank K ↥(⨆ i ∈ s, p i) ≤ ∑ i ∈ s, finrank K ↥(p i) := by
  classical
  induction s using Finset.induction_on with
  | empty => simp
  | insert a s ha ih =>
    rw [Finset.sum_insert ha, Finset.iSup_insert]
    exact (Submodule.finrank_add_le_finrank_add_finrank _ _).trans (Nat.add_le_add_left ih _)

/-- The dimension of a finite sum of independent subspaces is the sum of their dimensions. -/
theorem finrank_biSup_eq_sum {ι : Type*} (p : ι → Submodule K W) (hp : iSupIndep p)
    (s : Finset ι) : finrank K ↥(⨆ i ∈ s, p i) = ∑ i ∈ s, finrank K ↥(p i) := by
  classical
  induction s using Finset.induction_on with
  | empty => simp
  | insert a s ha ih =>
    rw [Finset.sum_insert ha, Finset.iSup_insert, ← ih]
    have hd : Disjoint (p a) (⨆ i ∈ s, p i) := hp.disjoint_biSup (y := (s : Set ι)) ha
    rw [← Submodule.finrank_sup_add_finrank_inf_eq, hd.eq_bot, finrank_bot, add_zero]

end Subspaces

section Walks

variable {F : Type*} [Field F] {m n : Type*} [Fintype n]

/-- The column space of an edge's difference (it does not depend on the orientation). -/
noncomputable def edgeColSpace {V : Type} (label : V → Matrix m n F) : Sym2 V → Submodule F (m → F) :=
  Sym2.lift ⟨fun u v => colSpace (label u - label v), fun u v => by
    dsimp only
    rw [← neg_sub, colSpace_neg]⟩

theorem finrank_edgeColSpace {V : Type} (label : V → Matrix m n F) (e : Sym2 V) :
    finrank F ↥(edgeColSpace label e) = SteinerTree.edgeLength label e := by
  induction e using Sym2.ind with
  | _ u v => rfl

/-- Along any walk, the column space of the difference of the endpoints' labels lies in the sum of the column
spaces of the graph's edges. -/
theorem colSpace_sub_le_of_walk {V : Type} [Fintype V] [DecidableEq V] {G : SimpleGraph V}
    [DecidableRel G.Adj] (label : V → Matrix m n F) {u v : V} (w : G.Walk u v) :
    colSpace (label v - label u) ≤ ⨆ e ∈ G.edgeFinset, edgeColSpace label e := by
  induction w with
  | nil => rw [sub_self, colSpace_zero]; exact bot_le
  | @cons a b c h p ih =>
    have hsplit : label c - label a = (label c - label b) + (label b - label a) := by abel
    rw [hsplit]
    refine (colSpace_add_le _ _).trans (sup_le ih ?_)
    have he : s(a, b) ∈ G.edgeFinset := by simpa using h
    refine le_trans ?_ (le_biSup (fun e => edgeColSpace label e) he)
    change colSpace (label b - label a) ≤ colSpace (label a - label b)
    rw [← neg_sub, colSpace_neg]

variable [Fintype m]

/-- The sum of the edges' column spaces has dimension at most the tree's length. -/
theorem finrank_edgeColSpaces_le_length (T : SteinerTree F m n) :
    finrank F ↥(⨆ e ∈ T.G.edgeFinset, edgeColSpace T.label e) ≤ T.length := by
  refine (finrank_biSup_le_sum _ _).trans_eq ?_
  exact Finset.sum_congr rfl fun e _ => finrank_edgeColSpace T.label e

omit [Fintype m] in
/-- If the tree reaches `0` and `A`, the column space of `A` lies in the sum of the edges' column spaces. -/
theorem colSpace_le_edgeColSpaces (T : SteinerTree F m n) {A : Matrix m n F} (h0 : T.Reaches 0)
    (hA : T.Reaches A) : colSpace A ≤ ⨆ e ∈ T.G.edgeFinset, edgeColSpace T.label e := by
  obtain ⟨v₀, hv₀⟩ := h0
  obtain ⟨v, hv⟩ := hA
  obtain ⟨w⟩ := T.isTree.connected.preconnected v₀ v
  have := colSpace_sub_le_of_walk T.label w
  rwa [hv, hv₀, sub_zero] at this

end Walks

section MergeProof

variable {F : Type*} [Field F] {m n : Type*} [Fintype m] [Fintype n]

/-- **One target.** A Steiner tree spanning `0` and `A` has length at least `rank A`. -/
theorem rank_le_length (T : SteinerTree F m n) {A : Matrix m n F} (h0 : T.Reaches 0)
    (hA : T.Reaches A) : A.rank ≤ T.length := by
  rw [rank_eq_finrank_colSpace]
  exact (Submodule.finrank_mono (colSpace_le_edgeColSpaces T h0 hA)).trans
    (finrank_edgeColSpaces_le_length T)

/-- **Corollary 6, column form.** Targets whose column spaces are independent: every Steiner tree spanning `0` and
the targets has length at least the sum of their ranks. -/
theorem sum_rank_le_length_of_colSpace_indep {ι : Type*} (T : SteinerTree F m n)
    (M : ι → Matrix m n F) (s : Finset ι) (hind : iSupIndep fun t => colSpace (M t))
    (h0 : T.Reaches 0) (hM : ∀ t ∈ s, T.Reaches (M t)) : ∑ t ∈ s, (M t).rank ≤ T.length := by
  have hle : (⨆ t ∈ s, colSpace (M t)) ≤ ⨆ e ∈ T.G.edgeFinset, edgeColSpace T.label e :=
    iSup₂_le fun t ht => colSpace_le_edgeColSpaces T h0 (hM t ht)
  calc ∑ t ∈ s, (M t).rank = finrank F ↥(⨆ t ∈ s, colSpace (M t)) :=
        (finrank_biSup_eq_sum (fun t => colSpace (M t)) hind s).symm
    _ ≤ finrank F ↥(⨆ e ∈ T.G.edgeFinset, edgeColSpace T.label e) := Submodule.finrank_mono hle
    _ ≤ T.length := finrank_edgeColSpaces_le_length T

/-- **Corollary 6, row form.** Targets whose row spaces are independent. -/
theorem sum_rank_le_length_of_rowSpace_indep {ι : Type*} (T : SteinerTree F m n)
    (M : ι → Matrix m n F) (s : Finset ι) (hind : iSupIndep fun t => rowSpace (M t))
    (h0 : T.Reaches 0) (hM : ∀ t ∈ s, T.Reaches (M t)) : ∑ t ∈ s, (M t).rank ≤ T.length := by
  have h := sum_rank_le_length_of_colSpace_indep T.transpose (fun t => (M t)ᵀ) s (by exact hind)
    (by simpa using SteinerTree.reaches_transpose h0) fun t ht => SteinerTree.reaches_transpose (hM t ht)
  simpa [Matrix.rank_transpose, SteinerTree.length_transpose] using h

end MergeProof

section Block

variable {F : Type*} [Field F] {m n : Type*} [Fintype m] [Fintype n]

/-- **Theorem 1 (block rigidity).** Split the row coordinates as a direct sum `R 0 ⊕ … ⊕ R (p-1)` (independent
subspaces of the column vectors) and the column coordinates as `C 0 ⊕ … ⊕ C (q-1)`. Put one target in each cell
`t = (r, c)` of a set `P`, with column space in `R r` and row space in `C c`. Then every Steiner tree spanning `0`
and the targets has length at least `∑ t ∈ P, rank (M t)`. -/
def BlockRigidity (F : Type*) [Field F] (m n : Type*) [Fintype m] [Fintype n] : Prop :=
  ∀ (p q : ℕ) (R : Fin p → Submodule F (m → F)) (C : Fin q → Submodule F (n → F)),
    iSupIndep R → iSupIndep C →
    ∀ (P : Finset (Fin p × Fin q)) (M : Fin p × Fin q → Matrix m n F),
      (∀ t ∈ P, colSpace (M t) ≤ R t.1 ∧ rowSpace (M t) ≤ C t.2) →
      ∀ T : SteinerTree F m n, T.Reaches 0 → (∀ t ∈ P, T.Reaches (M t)) →
        ∑ t ∈ P, (M t).rank ≤ T.length

/-- Theorem 1 when the cells of `P` lie in distinct rows of the grid. -/
theorem blockRigidity_of_injOn_fst {p q : ℕ} {R : Fin p → Submodule F (m → F)} (hR : iSupIndep R)
    {P : Finset (Fin p × Fin q)} {M : Fin p × Fin q → Matrix m n F}
    (hcol : ∀ t ∈ P, colSpace (M t) ≤ R t.1) (hP : Set.InjOn Prod.fst (P : Set (Fin p × Fin q)))
    (T : SteinerTree F m n) (h0 : T.Reaches 0) (hM : ∀ t ∈ P, T.Reaches (M t)) :
    ∑ t ∈ P, (M t).rank ≤ T.length := by
  have hind : iSupIndep fun t : P => colSpace (M t) :=
    (hR.comp (f := fun t : P => t.1.1) fun a b h =>
      Subtype.ext (hP (Finset.mem_coe.mpr a.2) (Finset.mem_coe.mpr b.2) h)).mono
      fun t => hcol t t.2
  have h := sum_rank_le_length_of_colSpace_indep T (fun t : P => M t) Finset.univ hind h0
    fun t _ => hM t t.2
  rwa [Finset.sum_coe_sort P (fun t => (M t).rank)] at h

/-- Theorem 1 when the cells of `P` lie in distinct columns of the grid. -/
theorem blockRigidity_of_injOn_snd {p q : ℕ} {C : Fin q → Submodule F (n → F)} (hC : iSupIndep C)
    {P : Finset (Fin p × Fin q)} {M : Fin p × Fin q → Matrix m n F}
    (hrow : ∀ t ∈ P, rowSpace (M t) ≤ C t.2) (hP : Set.InjOn Prod.snd (P : Set (Fin p × Fin q)))
    (T : SteinerTree F m n) (h0 : T.Reaches 0) (hM : ∀ t ∈ P, T.Reaches (M t)) :
    ∑ t ∈ P, (M t).rank ≤ T.length := by
  have hind : iSupIndep fun t : P => rowSpace (M t) :=
    (hC.comp (f := fun t : P => t.1.2) fun a b h =>
      Subtype.ext (hP (Finset.mem_coe.mpr a.2) (Finset.mem_coe.mpr b.2) h)).mono
      fun t => hrow t t.2
  have h := sum_rank_le_length_of_rowSpace_indep T (fun t : P => M t) Finset.univ hind h0
    fun t _ => hM t t.2
  rwa [Finset.sum_coe_sort P (fun t => (M t).rank)] at h

/-- **Two targets in distinct cells.** In the block setting, a Steiner tree spanning `0` and the targets of two
distinct cells has length at least the sum of their ranks. -/
theorem rank_add_rank_le_length {p q : ℕ} {R : Fin p → Submodule F (m → F)}
    {C : Fin q → Submodule F (n → F)} (hR : iSupIndep R) (hC : iSupIndep C)
    {t₁ t₂ : Fin p × Fin q} (hne : t₁ ≠ t₂) {M : Fin p × Fin q → Matrix m n F}
    (hM : ∀ t ∈ ({t₁, t₂} : Finset (Fin p × Fin q)), colSpace (M t) ≤ R t.1 ∧ rowSpace (M t) ≤ C t.2)
    (T : SteinerTree F m n) (h0 : T.Reaches 0) (h₁ : T.Reaches (M t₁)) (h₂ : T.Reaches (M t₂)) :
    (M t₁).rank + (M t₂).rank ≤ T.length := by
  rw [← Finset.sum_pair (f := fun t => (M t).rank) hne]
  have hreach : ∀ t ∈ ({t₁, t₂} : Finset (Fin p × Fin q)), T.Reaches (M t) := by
    intro t ht
    rcases Finset.mem_insert.mp ht with rfl | ht
    · exact h₁
    · rw [Finset.mem_singleton.mp ht]; exact h₂
  have hinj : ∀ f : Fin p × Fin q → Fin p ⊕ Fin q, f t₁ ≠ f t₂ →
      Set.InjOn f (({t₁, t₂} : Finset (Fin p × Fin q)) : Set (Fin p × Fin q)) := by
    intro f hf x hx y hy hxy
    simp only [Finset.coe_insert, Finset.coe_singleton, Set.mem_insert_iff,
      Set.mem_singleton_iff] at hx hy
    rcases hx with rfl | rfl <;> rcases hy with rfl | rfl
    · rfl
    · exact absurd hxy hf
    · exact absurd hxy.symm hf
    · rfl
  by_cases hr : t₁.1 = t₂.1
  · have hc : t₁.2 ≠ t₂.2 := fun hc => hne (Prod.ext hr hc)
    have h := hinj (fun t => Sum.inr t.2) (fun h => hc (Sum.inr_injective h))
    exact blockRigidity_of_injOn_snd hC (fun t ht => (hM t ht).2)
      (fun x hx y hy hxy => h hx hy (congrArg Sum.inr hxy)) T h0 hreach
  · have h := hinj (fun t => Sum.inl t.1) (fun h => hr (Sum.inl_injective h))
    exact blockRigidity_of_injOn_fst hR (fun t ht => (hM t ht).1)
      (fun x hx y hy hxy => h hx hy (congrArg Sum.inl hxy)) T h0 hreach

end Block

section ConjectureA

variable {F : Type*} [Field F]

/-- The coordinate subspace of the vectors supported on the coordinates `i` with `blk i = k`. -/
def blockSubspace (F : Type*) [Field F] {ι κ : Type*} (blk : ι → κ) (k : κ) : Submodule F (ι → F) where
  carrier := {v | ∀ i, blk i ≠ k → v i = 0}
  add_mem' := by
    intro a b ha hb i hi
    simp [ha i hi, hb i hi]
  zero_mem' := fun _ _ => rfl
  smul_mem' := by
    intro c v hv i hi
    simp [hv i hi]

/-- The coordinate blocks of a partition of the coordinates are independent. -/
theorem iSupIndep_blockSubspace {ι κ : Type*} (blk : ι → κ) : iSupIndep (blockSubspace F blk) := by
  refine iSupIndep_def.mpr fun k => ?_
  let S : Submodule F (ι → F) :=
    { carrier := {v | ∀ i, blk i = k → v i = 0}
      add_mem' := by
        intro a b ha hb i hi
        simp [ha i hi, hb i hi]
      zero_mem' := fun _ _ => rfl
      smul_mem' := by
        intro c v hv i hi
        simp [hv i hi] }
  have hS : (⨆ (j) (_ : j ≠ k), blockSubspace F blk j) ≤ S :=
    iSup₂_le fun j hj v hv i hi => hv i (by rw [hi]; exact hj.symm)
  refine Submodule.disjoint_def.mpr fun v hv₁ hv₂ => funext fun i => ?_
  by_cases h : blk i = k
  · exact hS hv₂ i h
  · exact hv₁ i h

/-- The prefix target `T_{(r,c),s}` of `N × N × N` matrix multiplication with chunk `L`: the bilinear form
`∑_{k < s L} x_{r k} y_{k c}`, as a matrix whose rows are the `x_{i k}`, indexed `(i, k)`, and whose columns are the
`y_{k j}`, indexed `(j, k)`. In block form it is `E_{r c} ⊗ diag(1, …, 1, 0, …, 0)` with `s L` ones; at `s L = N` it
is the full output `E_{r c} ⊗ I_N`. -/
def prefixTarget (F : Type*) [Field F] (N L : ℕ) (r c : Fin N) (s : ℕ) :
    Matrix (Fin N × Fin N) (Fin N × Fin N) F :=
  fun i j => if i.1 = r ∧ j.1 = c ∧ i.2 = j.2 ∧ (i.2 : ℕ) < s * L then 1 else 0

/-- **Conjecture A (Steiner rigidity).** For `N × N × N` matrix multiplication and every chunk `L ∣ N`, the minimum
Steiner tree in the rank metric spanning `0` and every prefix target `T_{π,s}` (`π ∈ [N]²`, `1 ≤ s ≤ N / L`) has
length exactly `N³`: every such tree has length at least `N³`, and one has length `N³`. -/
def ConjectureA (F : Type*) [Field F] : Prop :=
  ∀ N L : ℕ, 0 < L → L ∣ N →
    (∀ T : SteinerTree F (Fin N × Fin N) (Fin N × Fin N), T.Reaches 0 →
      (∀ r c : Fin N, ∀ s ∈ Finset.Icc 1 (N / L), T.Reaches (prefixTarget F N L r c s)) →
        N ^ 3 ≤ T.length) ∧
    (∃ T : SteinerTree F (Fin N × Fin N) (Fin N × Fin N), T.Reaches 0 ∧
      (∀ r c : Fin N, ∀ s ∈ Finset.Icc 1 (N / L), T.Reaches (prefixTarget F N L r c s)) ∧
        T.length = N ^ 3)

variable {N L : ℕ}

theorem colSpace_prefixTarget_le (r c : Fin N) (s : ℕ) :
    colSpace (prefixTarget F N L r c s) ≤ blockSubspace F (fun i : Fin N × Fin N => i.1) r := by
  rintro _ ⟨y, rfl⟩ i hi
  simp [Matrix.mulVec, dotProduct, prefixTarget, hi]

theorem rowSpace_prefixTarget_le (r c : Fin N) (s : ℕ) :
    rowSpace (prefixTarget F N L r c s) ≤ blockSubspace F (fun j : Fin N × Fin N => j.1) c := by
  rintro _ ⟨x, rfl⟩ j hj
  simp only at hj
  simp [Matrix.vecMul, dotProduct, prefixTarget, hj]

/-- A prefix target that covers the whole inner dimension has rank at least `N`. -/
theorem le_rank_prefixTarget (r c : Fin N) {s : ℕ} (hs : N ≤ s * L) :
    N ≤ (prefixTarget F N L r c s).rank := by
  classical
  let A : Matrix (Fin N) (Fin N × Fin N) F := Matrix.of fun k i => if i = (r, k) then 1 else 0
  let B : Matrix (Fin N × Fin N) (Fin N) F := Matrix.of fun j k => if j = (c, k) then 1 else 0
  have hAMB : A * prefixTarget F N L r c s * B = 1 := by
    ext k k'
    have hk : (k : ℕ) < s * L := lt_of_lt_of_le k.2 hs
    simp_rw [Matrix.mul_apply]
    simp only [A, B, Matrix.of_apply, mul_ite, mul_one, mul_zero, Finset.sum_ite_eq',
      Finset.mem_univ, ite_true, ite_mul, one_mul, zero_mul, Matrix.one_apply]
    by_cases hkk : k = k'
    · subst hkk
      simp [prefixTarget, hk]
    · simp [prefixTarget, hkk]
  calc N = (1 : Matrix (Fin N) (Fin N) F).rank := by simp
    _ = (A * prefixTarget F N L r c s * B).rank := by rw [hAMB]
    _ ≤ (A * prefixTarget F N L r c s).rank := rank_mul_le_left _ _
    _ ≤ (prefixTarget F N L r c s).rank := rank_mul_le_right _ _

/-- **Conjecture A's lower bound from Theorem 1.** Theorem 1 on the `N × N` grid of row blocks `x_{r ·}` and column
blocks `y_{· c}`, applied to the full outputs (the prefix targets with `s = N / L`), gives `N³`. -/
theorem conjectureA_lower_of_blockRigidity (h1 : BlockRigidity F (Fin N × Fin N) (Fin N × Fin N))
    (hL : 0 < L) (hLN : L ∣ N) (T : SteinerTree F (Fin N × Fin N) (Fin N × Fin N)) (h0 : T.Reaches 0)
    (hT : ∀ r c : Fin N, ∀ s ∈ Finset.Icc 1 (N / L), T.Reaches (prefixTarget F N L r c s)) :
    N ^ 3 ≤ T.length := by
  rcases Nat.eq_zero_or_pos N with hN | hN
  · simp [hN]
  have hsL : N / L * L = N := Nat.div_mul_cancel hLN
  have hs : N / L ∈ Finset.Icc 1 (N / L) :=
    Finset.mem_Icc.mpr ⟨Nat.div_pos (Nat.le_of_dvd hN hLN) hL, le_rfl⟩
  have key := h1 N N (blockSubspace F fun i : Fin N × Fin N => i.1)
    (blockSubspace F fun j : Fin N × Fin N => j.1) (iSupIndep_blockSubspace _)
    (iSupIndep_blockSubspace _) Finset.univ (fun t => prefixTarget F N L t.1 t.2 (N / L))
    (fun t _ => ⟨colSpace_prefixTarget_le t.1 t.2 _, rowSpace_prefixTarget_le t.1 t.2 _⟩) T h0
    (fun t _ => hT t.1 t.2 _ hs)
  calc N ^ 3 = ∑ _t : Fin N × Fin N, N := by
        simp [Finset.card_univ, Fintype.card_prod, Fintype.card_fin]; ring
    _ ≤ ∑ t : Fin N × Fin N, (prefixTarget F N L t.1 t.2 (N / L)).rank :=
        Finset.sum_le_sum fun t _ => le_rank_prefixTarget t.1 t.2 hsL.ge
    _ ≤ T.length := key

end ConjectureA

end Pouw.SteinerRigidity
