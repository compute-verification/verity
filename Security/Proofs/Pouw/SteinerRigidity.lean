import Mathlib

/-!
# Steiner rigidity in the rank metric (PoUW work floor)

The setting of `note:20261015T0416Z-draft-steiner-rigidity-laminar`: matrices over a field `F`, the rank metric
`d(A, B) = rank (A - B)`, and a Steiner tree as a finite tree whose vertices carry matrices, its length the sum over
its edges of the rank of the difference of the endpoints' labels.

* `BlockRigidity` is Theorem 1 (block rigidity) and `ConjectureA` is its matmul case, both stated as `Prop`s, and
  both are proved: `blockRigidity` and `conjectureA`.
* The proof follows the note. Lemma 1 (`treeLaminarDecomposition`): rooting the tree at a vertex labelled `0`, the
  targets below each edge form a laminar family, and the steps of the edges sum to each target. Lemma 2: the
  flattening is no longer than the decomposition (`rank_flattening_le`) and, for labels independent within each row
  and each column of cells, at least the sum of the targets' ranks (`flatteningLowerBound`). Lemma 3 (`labelLemma`):
  such labels exist for every laminar family, by induction up a binary refinement of it (`Hier`), typing each node
  by the rule of Claims 3–5 and taking the labels from annihilators (Claims 1, 2). That gives Theorem 1 in coordinate
  blocks (`blockRigidityCoord`), and an invertible change of coordinates gives it for independent subspaces.
* Conjecture A: Theorem 1 on the full outputs gives `N³ ≤ length` (`conjectureA_lower`), and a spanning tree of the
  spider (`0`, then each output cell's chain of prefix targets) has length `N³` (`conjectureA_upper`).
* Also here: the rank metric's facts, and the merge-proof case (Corollary 6 of the note) for any connected labelled
  graph: targets whose column spaces (or row spaces) are independent cost at least the sum of their ranks.
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

section Flattening

variable {F : Type*} [Field F] {m n : Type*} [Fintype m] [Fintype n]

/-- Rank is subadditive over finite sums. -/
theorem rank_sum_le {ι : Type*} (s : Finset ι) (A : ι → Matrix m n F) :
    (∑ i ∈ s, A i).rank ≤ ∑ i ∈ s, (A i).rank := by
  classical
  induction s using Finset.induction_on with
  | empty => simp
  | insert a s ha ih =>
    rw [Finset.sum_insert ha, Finset.sum_insert ha]
    exact (rank_add_le _ _).trans (Nat.add_le_add_left ih _)

/-- `Λ ⊗ W` for a label matrix `Λ`, entrywise: `((i, x), (j, y)) ↦ Λ i j * W x y`. -/
def kron {α β : Type*} (Λ : Matrix α β F) (W : Matrix m n F) : Matrix (α × m) (β × n) F :=
  fun i j => Λ i.1 j.1 * W i.2 j.2

omit [Fintype m] [Fintype n] in
theorem kron_sum_right {α β ι : Type*} (Λ : Matrix α β F) (s : Finset ι) (W : ι → Matrix m n F) :
    kron Λ (∑ k ∈ s, W k) = ∑ k ∈ s, kron Λ (W k) := by
  ext i j
  simp [kron, Matrix.sum_apply, Finset.mul_sum]

omit [Fintype m] [Fintype n] in
theorem kron_sum_left {α β ι : Type*} (s : Finset ι) (Λ : ι → Matrix α β F) (W : Matrix m n F) :
    kron (∑ k ∈ s, Λ k) W = ∑ k ∈ s, kron (Λ k) W := by
  ext i j
  simp [kron, Matrix.sum_apply, Finset.sum_mul]

/-- `rank ((u vᵀ) ⊗ W) ≤ rank W`. -/
theorem rank_kron_vecMulVec_le {α β : Type*} [Fintype α] [Fintype β] [DecidableEq m] [DecidableEq n]
    (u : α → F) (v : β → F) (W : Matrix m n F) : (kron (vecMulVec u v) W).rank ≤ W.rank := by
  let U : Matrix (α × m) m F := Matrix.of fun i x => if i.2 = x then u i.1 else 0
  let V : Matrix n (β × n) F := Matrix.of fun y j => if y = j.2 then v j.1 else 0
  have h : kron (vecMulVec u v) W = U * W * V := by
    ext i j
    simp_rw [Matrix.mul_apply]
    simp only [U, V, Matrix.of_apply, ite_mul, zero_mul, Finset.sum_ite_eq, Finset.mem_univ,
      ite_true, mul_ite, mul_zero, Finset.sum_ite_eq', kron, vecMulVec_apply]
    ring
  rw [h]
  exact (rank_mul_le_left _ _).trans (rank_mul_le_right _ _)

/-- The flattening `Φ = ∑ t ∈ P, (a t (b t)ᵀ) ⊗ M t` of Lemma 2. -/
def flattening {κ α β : Type*} (P : Finset κ) (a : κ → α → F) (b : κ → β → F)
    (M : κ → Matrix m n F) : Matrix (α × m) (β × n) F :=
  ∑ t ∈ P, kron (vecMulVec (a t) (b t)) (M t)

/-- A family of finite sets is laminar: any two of its sets are nested or disjoint. -/
def Laminar {κ : Type*} (𝓕 : Finset (Finset κ)) : Prop :=
  ∀ S ∈ 𝓕, ∀ S' ∈ 𝓕, S ⊆ S' ∨ S' ⊆ S ∨ Disjoint S S'

omit [Fintype m] [Fintype n] in
/-- The flattening of a decomposition `M t = ∑ S ∈ 𝓕, S ∋ t, W S` regroups by the sets of `𝓕`. -/
theorem flattening_eq_sum {κ α β : Type*} [DecidableEq κ] (P : Finset κ) (a : κ → α → F)
    (b : κ → β → F) (M : κ → Matrix m n F) (𝓕 : Finset (Finset κ)) (W : Finset κ → Matrix m n F)
    (hsub : ∀ S ∈ 𝓕, S ⊆ P) (hdec : ∀ t ∈ P, ∑ S ∈ 𝓕.filter (t ∈ ·), W S = M t) :
    flattening P a b M = ∑ S ∈ 𝓕, kron (∑ t ∈ S, vecMulVec (a t) (b t)) (W S) := by
  calc flattening P a b M
      = ∑ t ∈ P, ∑ S ∈ 𝓕, if t ∈ S then kron (vecMulVec (a t) (b t)) (W S) else 0 := by
        refine Finset.sum_congr rfl fun t ht => ?_
        rw [← hdec t ht, kron_sum_right, Finset.sum_filter]
    _ = ∑ S ∈ 𝓕, ∑ t ∈ P, if t ∈ S then kron (vecMulVec (a t) (b t)) (W S) else 0 :=
        Finset.sum_comm
    _ = ∑ S ∈ 𝓕, kron (∑ t ∈ S, vecMulVec (a t) (b t)) (W S) := by
        refine Finset.sum_congr rfl fun S hS => ?_
        rw [← Finset.sum_filter, kron_sum_left, Finset.filter_mem_eq_inter,
          Finset.inter_eq_right.mpr (hsub S hS)]

/-- **Lemma 2(a).** If every `Λ(S) = ∑ t ∈ S, a t (b t)ᵀ` (`S ∈ 𝓕`) is a rank-one product, the flattening of a
decomposition over `𝓕` has rank at most `∑ S ∈ 𝓕, rank (W S)`. -/
theorem rank_flattening_le {κ α β : Type*} [DecidableEq κ] [Fintype α] [Fintype β] [DecidableEq m]
    [DecidableEq n] (P : Finset κ) (a : κ → α → F) (b : κ → β → F) (M : κ → Matrix m n F)
    (𝓕 : Finset (Finset κ)) (W : Finset κ → Matrix m n F) (hsub : ∀ S ∈ 𝓕, S ⊆ P)
    (hdec : ∀ t ∈ P, ∑ S ∈ 𝓕.filter (t ∈ ·), W S = M t)
    (hL : ∀ S ∈ 𝓕, ∃ u v, ∑ t ∈ S, vecMulVec (a t) (b t) = vecMulVec u v) :
    (flattening P a b M).rank ≤ ∑ S ∈ 𝓕, (W S).rank := by
  rw [flattening_eq_sum P a b M 𝓕 W hsub hdec]
  refine (rank_sum_le _ _).trans (Finset.sum_le_sum fun S hS => ?_)
  obtain ⟨u, v, huv⟩ := hL S hS
  rw [huv]
  exact rank_kron_vecMulVec_le u v (W S)

/-- The coordinate block setting: target `M t` has its nonzero entries in rows of block `t.1` and columns of block
`t.2`. -/
def InCells {p q : ℕ} (rowBlk : m → Fin p) (colBlk : n → Fin q) (P : Finset (Fin p × Fin q))
    (M : Fin p × Fin q → Matrix m n F) : Prop :=
  ∀ t ∈ P, ∀ x y, M t x y ≠ 0 → rowBlk x = t.1 ∧ colBlk y = t.2

/-- **Lemma 2(b).** In the coordinate block setting, if the labels `a t` of each grid row are linearly independent
and the labels `b t` of each grid column are, the flattening has rank at least `∑ t ∈ P, rank (M t)`. -/
def FlatteningLowerBound (F : Type*) [Field F] (m n : Type*) [Fintype m] [Fintype n] : Prop :=
  ∀ (p q α β : ℕ) (rowBlk : m → Fin p) (colBlk : n → Fin q) (P : Finset (Fin p × Fin q))
    (M : Fin p × Fin q → Matrix m n F) (a : Fin p × Fin q → Fin α → F) (b : Fin p × Fin q → Fin β → F),
    InCells rowBlk colBlk P M →
    (∀ r, LinearIndepOn F a ↑(P.filter (·.1 = r))) → (∀ c, LinearIndepOn F b ↑(P.filter (·.2 = c))) →
      ∑ t ∈ P, (M t).rank ≤ (flattening P a b M).rank

/-- `z ↦ u ⊗ z`, entrywise `(i, x) ↦ u i * z x`. -/
def tensorLeft {α : Type*} (u : α → F) : (m → F) →ₗ[F] (α × m → F) where
  toFun z i := u i.1 * z i.2
  map_add' z z' := by
    ext i
    simp [mul_add]
  map_smul' c z := by
    ext i
    simp only [Pi.smul_apply, smul_eq_mul, RingHom.id_apply]
    ring

/-- Independent vectors have a dual vector: one that pairs to `1` with a chosen vector and `0` with the rest. -/
theorem exists_dual_of_linearIndepOn {κ β : Type*} [Fintype β] [DecidableEq κ] {s : Finset κ}
    {b : κ → β → F} (h : LinearIndepOn F b ↑s) {t₀ : κ} (ht₀ : t₀ ∈ s) :
    ∃ g : β → F, ∀ t ∈ s, b t ⬝ᵥ g = if t = t₀ then 1 else 0 := by
  classical
  let B : Matrix s β F := Matrix.of fun t => b t
  have hrank : B.rank = Fintype.card s := by
    rw [Matrix.rank_eq_finrank_span_row]
    exact finrank_span_eq_card (b := B.row) h
  have htop : LinearMap.range B.mulVecLin = ⊤ := by
    refine Submodule.eq_top_of_finrank_eq ?_
    rw [Module.finrank_fintype_fun_eq_card]
    exact hrank
  obtain ⟨g, hg⟩ := LinearMap.range_eq_top.mp htop (Pi.single ⟨t₀, ht₀⟩ 1)
  refine ⟨g, fun t ht => ?_⟩
  have := congrFun hg ⟨t, ht⟩
  simp only [Matrix.mulVecLin_apply, Matrix.mulVec, B, Matrix.of_apply] at this
  rw [this, Pi.single_apply]
  simp [Subtype.ext_iff]

omit [Fintype m] in
theorem mem_colSpace_eq_zero {A : Matrix m n F} {v : m → F} (hv : v ∈ colSpace A) {x : m}
    (hx : ∀ y, A x y = 0) : v x = 0 := by
  obtain ⟨y, rfl⟩ := hv
  simp [Matrix.mulVec, dotProduct, hx]

omit [Fintype m] in
theorem kron_mulVec_tensor {α β : Type*} [Fintype β] (u : α → F) (v : β → F) (W : Matrix m n F)
    (g : β → F) (y : n → F) :
    kron (vecMulVec u v) W *ᵥ (fun j => g j.1 * y j.2) = fun i => u i.1 * (v ⬝ᵥ g) * (W *ᵥ y) i.2 := by
  ext i
  simp only [Matrix.mulVec, dotProduct, kron, vecMulVec_apply, Fintype.sum_prod_type]
  conv_rhs => rw [mul_assoc, Finset.sum_mul_sum, Finset.mul_sum]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun y' _ => ?_
  ring

/-- **Lemma 2(b), proved.** -/
theorem flatteningLowerBound [DecidableEq m] [DecidableEq n] : FlatteningLowerBound F m n := by
  classical
  intro p q α β rowBlk colBlk P M a b hM hIa hIb
  let Φ := flattening P a b M
  let Ψ : (∀ t : P, colSpace (M t)) →ₗ[F] (Fin α × m → F) :=
    ∑ t : P, tensorLeft (a t) ∘ₗ (colSpace (M t)).subtype ∘ₗ LinearMap.proj t
  have hΨ : ∀ z, Ψ z = ∑ t : P, tensorLeft (a t) (z t : m → F) := by
    intro z
    simp [Ψ]
  -- Ψ is injective: the `a t` of each grid row are independent and `M t` lives in its row block.
  have hinj : Function.Injective Ψ := by
    rw [← LinearMap.ker_eq_bot, LinearMap.ker_eq_bot']
    intro z hz
    funext t₀
    apply Subtype.ext
    funext x
    let f : Fin p × Fin q → F := fun t => if h : t ∈ P then (z ⟨t, h⟩ : m → F) x else 0
    have hf0 : ∀ t, t.1 ≠ rowBlk x → f t = 0 := by
      intro t ht
      by_cases h : t ∈ P
      · simp only [f, h, ↓reduceDIte]
        refine mem_colSpace_eq_zero (z ⟨t, h⟩).2 fun y => ?_
        by_contra hne
        exact ht (hM t h x y hne).1.symm
      · simp [f, h]
    have hsum : ∑ t ∈ P.filter (·.1 = rowBlk x), f t • a t = 0 := by
      funext i
      have h1 := congrFun (hΨ z ▸ hz) (i, x)
      simp only [Finset.sum_apply, tensorLeft, LinearMap.coe_mk, AddHom.coe_mk, Pi.zero_apply] at h1
      rw [Finset.sum_apply, Pi.zero_apply, Finset.sum_filter]
      rw [← Finset.sum_coe_sort P]
      refine Eq.trans (Finset.sum_congr rfl fun t _ => ?_) h1
      by_cases ht : (t : Fin p × Fin q).1 = rowBlk x
      · simp [f, ht, mul_comm]
      · simp only [ht, ↓reduceIte]
        have := mem_colSpace_eq_zero (z t).2 (x := x) fun y => by
          by_contra hne
          exact ht (hM t t.2 x y hne).1.symm
        simp [this]
    have hf := (linearIndepOn_finset_iff.mp (hIa (rowBlk x))) f hsum
    have : f t₀ = 0 := by
      by_cases ht : (t₀ : Fin p × Fin q).1 = rowBlk x
      · exact hf t₀ (Finset.mem_filter.mpr ⟨t₀.2, ht⟩)
      · exact hf0 t₀ ht
    simpa [f, t₀.2] using this
  -- The range of Ψ lies in the column space of Φ: the `b t` of each grid column have dual vectors.
  have hrange : LinearMap.range Ψ ≤ colSpace Φ := by
    rintro _ ⟨z, rfl⟩
    rw [hΨ]
    refine Submodule.sum_mem _ fun t₀ _ => ?_
    obtain ⟨y, hy⟩ := (z t₀).2
    let c := (t₀ : Fin p × Fin q).2
    obtain ⟨g, hg⟩ := exists_dual_of_linearIndepOn (hIb c)
      (Finset.mem_filter.mpr ⟨t₀.2, rfl⟩)
    let yc : n → F := fun y' => if colBlk y' = c then y y' else 0
    refine ⟨fun j => g j.1 * yc j.2, ?_⟩
    simp only [Matrix.mulVecLin_apply, Φ, flattening, Matrix.sum_mulVec, kron_mulVec_tensor]
    rw [Finset.sum_eq_single_of_mem (t₀ : Fin p × Fin q) t₀.2]
    · have hyc : M t₀ *ᵥ yc = M t₀ *ᵥ y := by
        funext x
        refine Finset.sum_congr rfl fun y' _ => ?_
        by_cases h : colBlk y' = c
        · simp [yc, h]
        · have : M t₀ x y' = 0 := by
            by_contra hne
            exact h (hM t₀ t₀.2 x y' hne).2
          simp [this]
      have h1 : b t₀ ⬝ᵥ g = 1 := by
        simpa using hg t₀ (Finset.mem_filter.mpr ⟨t₀.2, rfl⟩)
      funext i
      simp only [tensorLeft, LinearMap.coe_mk, AddHom.coe_mk, h1, mul_one, hyc]
      rw [← hy]
      rfl
    · intro t ht hne
      funext i
      by_cases htc : t.2 = c
      · have h0 : b t ⬝ᵥ g = 0 := by
          simpa [hne] using hg t (Finset.mem_filter.mpr ⟨ht, htc⟩)
        simp [h0]
      · have h0 : M t *ᵥ yc = 0 := by
          funext x
          refine Finset.sum_eq_zero fun y' _ => ?_
          by_cases h : colBlk y' = c
          · have : M t x y' = 0 := by
              by_contra hne'
              exact htc ((hM t ht x y' hne').2.symm.trans h)
            simp [this]
          · simp [yc, h]
        simp [h0]
  calc ∑ t ∈ P, (M t).rank = ∑ t : P, finrank F ↥(colSpace (M t)) := by
        rw [← Finset.sum_coe_sort P]
        rfl
    _ = finrank F (∀ t : P, colSpace (M t)) := by rw [Module.finrank_pi_fintype]
    _ = finrank F ↥(LinearMap.range Ψ) := (LinearMap.finrank_range_of_inj hinj).symm
    _ ≤ finrank F ↥(colSpace Φ) := Submodule.finrank_mono hrange
    _ = Φ.rank := rfl

/-- **Lemma 3 (labels for a laminar family).** For one target per cell of a set `P` and a laminar family `𝓕` of
subsets of `P`, there are labels with (L) every `Λ(S)`, `S ∈ 𝓕`, a rank-one product, and (I) the `a t` of each grid
row linearly independent and the `b t` of each grid column linearly independent. -/
def LabelLemma (F : Type*) [Field F] : Prop :=
  ∀ (p q : ℕ) (P : Finset (Fin p × Fin q)) (𝓕 : Finset (Finset (Fin p × Fin q))),
    Laminar 𝓕 → (∀ S ∈ 𝓕, S ⊆ P) →
      ∃ (α β : ℕ) (a : Fin p × Fin q → Fin α → F) (b : Fin p × Fin q → Fin β → F),
        (∀ S ∈ 𝓕, ∃ u v, ∑ t ∈ S, vecMulVec (a t) (b t) = vecMulVec u v) ∧
        (∀ r, LinearIndepOn F a ↑(P.filter (·.1 = r))) ∧ (∀ c, LinearIndepOn F b ↑(P.filter (·.2 = c)))

/-- **Theorem 1 for laminar decompositions** (Lemma 1(d)'s side): in the coordinate block setting, any
decomposition `M t = ∑ S ∈ 𝓕, S ∋ t, W S` over a laminar family of subsets of `P` has
`∑ S ∈ 𝓕, rank (W S) ≥ ∑ t ∈ P, rank (M t)`. -/
def LaminarBound (F : Type*) [Field F] (m n : Type*) [Fintype m] [Fintype n] : Prop :=
  ∀ (p q : ℕ) (rowBlk : m → Fin p) (colBlk : n → Fin q) (P : Finset (Fin p × Fin q))
    (M : Fin p × Fin q → Matrix m n F), InCells rowBlk colBlk P M →
    ∀ (𝓕 : Finset (Finset (Fin p × Fin q))) (W : Finset (Fin p × Fin q) → Matrix m n F),
      Laminar 𝓕 → (∀ S ∈ 𝓕, S ⊆ P) → (∀ t ∈ P, ∑ S ∈ 𝓕.filter (t ∈ ·), W S = M t) →
        ∑ t ∈ P, (M t).rank ≤ ∑ S ∈ 𝓕, (W S).rank

/-- Lemma 3 gives Theorem 1 for laminar decompositions (with Lemma 2, proved). -/
theorem laminarBound_of_labelLemma [DecidableEq m] [DecidableEq n] (h3 : LabelLemma F) :
    LaminarBound F m n := by
  intro p q rowBlk colBlk P M hM 𝓕 W hlam hsub hdec
  obtain ⟨α, β, a, b, hL, hIa, hIb⟩ := h3 p q P 𝓕 hlam hsub
  exact (flatteningLowerBound p q α β rowBlk colBlk P M a b hM hIa hIb).trans
    (rank_flattening_le P a b M 𝓕 W hsub hdec hL)

/-- **Lemma 1(a, b)**: a Steiner tree spanning `0` and the targets gives a laminar decomposition no longer than the
tree: a laminar family `𝓕` of subsets of `P` and matrices `W S` with `M t = ∑ S ∈ 𝓕, S ∋ t, W S` and
`∑ S ∈ 𝓕, rank (W S) ≤ length`. (`S` is the set of targets below an edge, `W S` the sum of the steps of the edges
with that set.) -/
def TreeLaminarDecomposition (F : Type*) [Field F] (m n : Type*) [Fintype m] [Fintype n] : Prop :=
  ∀ (κ : Type) [DecidableEq κ] (P : Finset κ) (M : κ → Matrix m n F) (T : SteinerTree F m n),
    T.Reaches 0 → (∀ t ∈ P, T.Reaches (M t)) →
      ∃ (𝓕 : Finset (Finset κ)) (W : Finset κ → Matrix m n F), Laminar 𝓕 ∧ (∀ S ∈ 𝓕, S ⊆ P) ∧
        (∀ t ∈ P, ∑ S ∈ 𝓕.filter (t ∈ ·), W S = M t) ∧ ∑ S ∈ 𝓕, (W S).rank ≤ T.length

/-- **Theorem 1 in coordinates**: the row and column coordinates split into blocks, one target per cell. -/
def BlockRigidityCoord (F : Type*) [Field F] (m n : Type*) [Fintype m] [Fintype n] : Prop :=
  ∀ (p q : ℕ) (rowBlk : m → Fin p) (colBlk : n → Fin q) (P : Finset (Fin p × Fin q))
    (M : Fin p × Fin q → Matrix m n F), InCells rowBlk colBlk P M →
    ∀ T : SteinerTree F m n, T.Reaches 0 → (∀ t ∈ P, T.Reaches (M t)) →
      ∑ t ∈ P, (M t).rank ≤ T.length

/-- Lemma 1 and Theorem 1 for laminar decompositions give Theorem 1 in coordinates. -/
theorem blockRigidityCoord_of_laminarBound (h1 : TreeLaminarDecomposition F m n)
    (hlam : LaminarBound F m n) : BlockRigidityCoord F m n := by
  intro p q rowBlk colBlk P M hM T h0 hT
  obtain ⟨𝓕, W, hl, hsub, hdec, hlen⟩ := h1 (Fin p × Fin q) P M T h0 hT
  exact (hlam p q rowBlk colBlk P M hM 𝓕 W hl hsub hdec).trans hlen

/-- **Conjecture A's lower bound from Theorem 1 in coordinates.** -/
theorem conjectureA_lower_of_blockRigidityCoord {N L : ℕ}
    (h1 : BlockRigidityCoord F (Fin N × Fin N) (Fin N × Fin N)) (hL : 0 < L) (hLN : L ∣ N)
    (T : SteinerTree F (Fin N × Fin N) (Fin N × Fin N)) (h0 : T.Reaches 0)
    (hT : ∀ r c : Fin N, ∀ s ∈ Finset.Icc 1 (N / L), T.Reaches (prefixTarget F N L r c s)) :
    N ^ 3 ≤ T.length := by
  rcases Nat.eq_zero_or_pos N with hN | hN
  · simp [hN]
  have hsL : N / L * L = N := Nat.div_mul_cancel hLN
  have hs : N / L ∈ Finset.Icc 1 (N / L) :=
    Finset.mem_Icc.mpr ⟨Nat.div_pos (Nat.le_of_dvd hN hLN) hL, le_rfl⟩
  have hcells : InCells (fun i : Fin N × Fin N => i.1) (fun j : Fin N × Fin N => j.1) Finset.univ
      (fun t => prefixTarget F N L t.1 t.2 (N / L)) := by
    intro t _ x y hxy
    simp only [prefixTarget, ne_eq, ite_eq_right_iff, one_ne_zero, imp_false, not_not] at hxy
    exact ⟨hxy.1, hxy.2.1⟩
  have key := h1 N N (fun i => i.1) (fun j => j.1) Finset.univ
    (fun t => prefixTarget F N L t.1 t.2 (N / L)) hcells T h0 (fun t _ => hT t.1 t.2 _ hs)
  calc N ^ 3 = ∑ _t : Fin N × Fin N, N := by
        simp [Finset.card_univ, Fintype.card_prod, Fintype.card_fin]; ring
    _ ≤ ∑ t : Fin N × Fin N, (prefixTarget F N L t.1 t.2 (N / L)).rank :=
        Finset.sum_le_sum fun t _ => le_rank_prefixTarget t.1 t.2 hsL.ge
    _ ≤ T.length := key

/-- **Conjecture A's lower bound from Lemmas 1 and 3.** -/
theorem conjectureA_lower_of_lemmas {N L : ℕ}
    (h1 : TreeLaminarDecomposition F (Fin N × Fin N) (Fin N × Fin N)) (h3 : LabelLemma F)
    (hL : 0 < L) (hLN : L ∣ N) (T : SteinerTree F (Fin N × Fin N) (Fin N × Fin N)) (h0 : T.Reaches 0)
    (hT : ∀ r c : Fin N, ∀ s ∈ Finset.Icc 1 (N / L), T.Reaches (prefixTarget F N L r c s)) :
    N ^ 3 ≤ T.length :=
  conjectureA_lower_of_blockRigidityCoord
    (blockRigidityCoord_of_laminarBound h1 (laminarBound_of_labelLemma h3)) hL hLN T h0 hT

end Flattening

/-!
## Lemma 3's induction, on binary hierarchies

A full binary hierarchy on the cells, each internal node typed x (`true`) or y (`false`). `xi h` is the
functional `ξ_S` (the indicator of `D(S)`), `Vx h` is `V_x(S)`; the column side's `η_S` and `V_y(S)` are `xi` and
`Vx` of the hierarchy with every type flipped. A side is valid when `Vx` meets no row block (`V_x ∩ F^row r = 0`);
its handles are `ρ(S) = {r | ξ_S ∈ V_x(S) + F^row r}`. Claims 3, 4 and 5 of the note and the typing rule are proved
below: every hierarchy with disjoint siblings has a typing valid on both sides.
-/

section Hierarchy

/-- A binary hierarchy on cells; an internal node's flag is its type, `true` for x and `false` for y. -/
inductive Hier (κ : Type*) where
  | leaf (t : κ)
  | node (x : Bool) (l r : Hier κ)

namespace Hier

variable {κ : Type*}

/-- The cells below a node. -/
def cells [DecidableEq κ] : Hier κ → Finset κ
  | leaf t => {t}
  | node _ l r => l.cells ∪ r.cells

/-- Siblings are disjoint at every node. -/
def WF [DecidableEq κ] : Hier κ → Prop
  | leaf _ => True
  | node _ l r => l.WF ∧ r.WF ∧ Disjoint l.cells r.cells

/-- Swap every node's type. -/
def flip : Hier κ → Hier κ
  | leaf t => leaf t
  | node x l r => node (!x) l.flip r.flip

theorem cells_flip [DecidableEq κ] : ∀ h : Hier κ, h.flip.cells = h.cells
  | leaf _ => rfl
  | node _ l r => by simp only [flip, cells, cells_flip l, cells_flip r]

theorem wf_flip [DecidableEq κ] : ∀ h : Hier κ, h.flip.WF ↔ h.WF
  | leaf _ => Iff.rfl
  | node _ l r => by simp only [flip, WF, wf_flip l, wf_flip r, cells_flip]

theorem flip_flip : ∀ h : Hier κ, h.flip.flip = h
  | leaf _ => rfl
  | node x l r => by simp only [flip, Bool.not_not, flip_flip l, flip_flip r]

variable (F : Type*) [Field F] [DecidableEq κ]

/-- `ξ_S`: at an x-node the first child's, at a y-node the sum of the children's. -/
def xi : Hier κ → κ → F
  | leaf t => Pi.single t 1
  | node true l _ => l.xi
  | node false l r => l.xi + r.xi

/-- `V_x(S)`: the span of `ξ_{U'} - ξ_{U''}` over the x-nodes `U` below. -/
def Vx : Hier κ → Submodule F (κ → F)
  | leaf _ => ⊥
  | node true l r => l.Vx ⊔ r.Vx ⊔ Submodule.span F {l.xi F - r.xi F}
  | node false l r => l.Vx ⊔ r.Vx

/-- The vectors supported on `S`. -/
def supp (S : Finset κ) : Submodule F (κ → F) where
  carrier := {v | ∀ t ∉ S, v t = 0}
  add_mem' := by
    intro a b ha hb t ht
    simp [ha t ht, hb t ht]
  zero_mem' := fun _ _ => rfl
  smul_mem' := by
    intro c v hv t ht
    simp [hv t ht]

/-- Restriction to `S`: zero outside `S`. -/
def restrict (S : Finset κ) : (κ → F) →ₗ[F] (κ → F) where
  toFun v t := if t ∈ S then v t else 0
  map_add' v w := by
    ext t
    by_cases h : t ∈ S <;> simp [h]
  map_smul' c v := by
    ext t
    by_cases h : t ∈ S <;> simp [h]

/-- The handles `{r | ξ_S ∈ V_x(S) + F^row r}`, for the block map `blk`. -/
def handle {ι : Type*} (blk : κ → ι) (h : Hier κ) : Set ι :=
  {r | h.xi F ∈ h.Vx F ⊔ blockSubspace F blk r}

/-- One side of validity: `V_x(S)` meets no block of `blk`. -/
def SideValid {ι : Type*} (blk : κ → ι) (h : Hier κ) : Prop :=
  ∀ r, Disjoint (h.Vx F) (blockSubspace F blk r)

variable {F}

omit [DecidableEq κ] in
theorem supp_mono {S S' : Finset κ} (h : S ⊆ S') : supp F S ≤ supp F S' :=
  fun _ hv t ht => hv t fun h' => ht (h h')

theorem xi_mem_supp : ∀ h : Hier κ, h.xi F ∈ supp F h.cells
  | leaf t => by
    intro u hu
    simp only [cells, Finset.mem_singleton] at hu
    simp [xi, hu]
  | node true l r => supp_mono Finset.subset_union_left (xi_mem_supp l)
  | node false l r => Submodule.add_mem _ (supp_mono Finset.subset_union_left (xi_mem_supp l))
      (supp_mono Finset.subset_union_right (xi_mem_supp r))

theorem Vx_le_supp : ∀ h : Hier κ, h.Vx F ≤ supp F h.cells
  | leaf _ => bot_le
  | node true l r => sup_le (sup_le ((Vx_le_supp l).trans (supp_mono Finset.subset_union_left))
      ((Vx_le_supp r).trans (supp_mono Finset.subset_union_right)))
      ((Submodule.span_le).mpr (Set.singleton_subset_iff.mpr (Submodule.sub_mem _
        (supp_mono Finset.subset_union_left (xi_mem_supp l))
        (supp_mono Finset.subset_union_right (xi_mem_supp r)))))
  | node false l r => sup_le ((Vx_le_supp l).trans (supp_mono Finset.subset_union_left))
      ((Vx_le_supp r).trans (supp_mono Finset.subset_union_right))

theorem restrict_of_mem_supp {S : Finset κ} {v : κ → F} (hv : v ∈ supp F S) : restrict F S v = v := by
  ext t
  change (if t ∈ S then v t else 0) = v t
  split_ifs with h
  · rfl
  · exact (hv t h).symm

theorem restrict_of_mem_supp_disjoint {S S' : Finset κ} {v : κ → F} (hv : v ∈ supp F S')
    (hd : Disjoint S S') : restrict F S v = 0 := by
  ext t
  change (if t ∈ S then v t else 0) = 0
  split_ifs with h
  · exact hv t (Finset.disjoint_left.mp hd h)
  · rfl

theorem restrict_mem_blockSubspace {ι : Type*} {blk : κ → ι} {r : ι} {S : Finset κ} {v : κ → F}
    (hv : v ∈ blockSubspace F blk r) : restrict F S v ∈ blockSubspace F blk r := by
  intro t ht
  change (if t ∈ S then v t else 0) = 0
  split_ifs
  · exact hv t ht
  · rfl

variable {ι : Type*} {blk : κ → ι}

/-- **Claim 5, the other type's side**: at a node whose `V_x` is the sum of the children's. -/
theorem sideValid_node_false {l r : Hier κ} (hd : Disjoint l.cells r.cells)
    (hl : SideValid F blk l) (hr : SideValid F blk r) : SideValid F blk (node false l r) := by
  intro r₀
  rw [Submodule.disjoint_def]
  intro w hw hw₀
  obtain ⟨v', hv', v'', hv'', rfl⟩ := Submodule.mem_sup.mp hw
  have e₁ : restrict F l.cells (v' + v'') = v' := by
    rw [map_add, restrict_of_mem_supp (Vx_le_supp l hv'),
      restrict_of_mem_supp_disjoint (Vx_le_supp r hv'') hd, add_zero]
  have e₂ : restrict F r.cells (v' + v'') = v'' := by
    rw [map_add, restrict_of_mem_supp (Vx_le_supp r hv''),
      restrict_of_mem_supp_disjoint (Vx_le_supp l hv') hd.symm, zero_add]
  have h₁ : v' = 0 := Submodule.disjoint_def.mp (hl r₀) v' hv'
    (e₁ ▸ restrict_mem_blockSubspace hw₀)
  have h₂ : v'' = 0 := Submodule.disjoint_def.mp (hr r₀) v'' hv''
    (e₂ ▸ restrict_mem_blockSubspace hw₀)
  rw [h₁, h₂, add_zero]

/-- A node's `V_x` at an x-node, as a sum. -/
theorem mem_Vx_node_true {l r : Hier κ} {w : κ → F} (hw : w ∈ (node true l r).Vx F) :
    ∃ v' ∈ l.Vx F, ∃ v'' ∈ r.Vx F, ∃ μ : F, w = v' + v'' + μ • (l.xi F - r.xi F) := by
  obtain ⟨u, hu, s, hs, rfl⟩ := Submodule.mem_sup.mp hw
  obtain ⟨v', hv', v'', hv'', rfl⟩ := Submodule.mem_sup.mp hu
  obtain ⟨μ, rfl⟩ := Submodule.mem_span_singleton.mp hs
  exact ⟨v', hv', v'', hv'', μ, rfl⟩

/-- **Claim 5, the node's own side**: at an x-node whose children have no handle in common. -/
theorem sideValid_node_true {l r : Hier κ} (hd : Disjoint l.cells r.cells)
    (hl : SideValid F blk l) (hr : SideValid F blk r)
    (hρ : ∀ r₀, r₀ ∈ handle F blk l → r₀ ∈ handle F blk r → False) :
    SideValid F blk (node true l r) := by
  intro r₀
  rw [Submodule.disjoint_def]
  intro w hw hw₀
  obtain ⟨v', hv', v'', hv'', μ, rfl⟩ := mem_Vx_node_true hw
  have hl' := xi_mem_supp (F := F) l
  have hr' := xi_mem_supp (F := F) r
  have e₁ : restrict F l.cells (v' + v'' + μ • (l.xi F - r.xi F)) = v' + μ • l.xi F := by
    rw [map_add, map_add, map_smul, map_sub, restrict_of_mem_supp (Vx_le_supp l hv'),
      restrict_of_mem_supp_disjoint (Vx_le_supp r hv'') hd, restrict_of_mem_supp hl',
      restrict_of_mem_supp_disjoint hr' hd, add_zero, sub_zero]
  have e₂ : restrict F r.cells (v' + v'' + μ • (l.xi F - r.xi F)) = v'' - μ • r.xi F := by
    rw [map_add, map_add, map_smul, map_sub, restrict_of_mem_supp (Vx_le_supp r hv''),
      restrict_of_mem_supp_disjoint (Vx_le_supp l hv') hd.symm, restrict_of_mem_supp hr',
      restrict_of_mem_supp_disjoint hl' hd.symm, zero_add, zero_sub, smul_neg, sub_eq_add_neg]
  have m₁ := e₁ ▸ restrict_mem_blockSubspace (S := l.cells) hw₀
  have m₂ := e₂ ▸ restrict_mem_blockSubspace (S := r.cells) hw₀
  by_cases hμ : μ = 0
  · subst hμ
    simp only [zero_smul, add_zero, sub_zero] at m₁ m₂ ⊢
    rw [Submodule.disjoint_def.mp (hl r₀) v' hv' m₁, Submodule.disjoint_def.mp (hr r₀) v'' hv'' m₂,
      add_zero]
  · exfalso
    refine hρ r₀ ?_ ?_
    · have : l.xi F = -(μ⁻¹ • v') + μ⁻¹ • (v' + μ • l.xi F) := by
        rw [smul_add, smul_smul, inv_mul_cancel₀ hμ, one_smul]
        abel
      change l.xi F ∈ l.Vx F ⊔ blockSubspace F blk r₀
      rw [this]
      exact Submodule.add_mem_sup (Submodule.neg_mem _ (Submodule.smul_mem _ _ hv'))
        (Submodule.smul_mem _ _ m₁)
    · have : r.xi F = μ⁻¹ • v'' + -(μ⁻¹ • (v'' - μ • r.xi F)) := by
        rw [smul_sub, smul_smul, inv_mul_cancel₀ hμ, one_smul]
        abel
      change r.xi F ∈ r.Vx F ⊔ blockSubspace F blk r₀
      rw [this]
      exact Submodule.add_mem_sup (Submodule.smul_mem _ _ hv'')
        (Submodule.neg_mem _ (Submodule.smul_mem _ _ m₂))

/-- **Claim 3, at a leaf**: `ρ(t) ⊆ {r_t}`. -/
theorem handle_leaf {t : κ} {r₀ : ι} (h : r₀ ∈ handle F blk (leaf t)) : blk t = r₀ := by
  change Pi.single t 1 ∈ (⊥ : Submodule F (κ → F)) ⊔ blockSubspace F blk r₀ at h
  rw [bot_sup_eq] at h
  by_contra hne
  have := h t hne
  simp at this

/-- **Claim 3, the node's own side**: `ρ(S) ⊆ ρ(S') ∪ ρ(S'')` at an x-node. -/
theorem handle_node_true {l r : Hier κ} (hd : Disjoint l.cells r.cells) {r₀ : ι}
    (h : r₀ ∈ handle F blk (node true l r)) : r₀ ∈ handle F blk l ∨ r₀ ∈ handle F blk r := by
  change l.xi F ∈ (node true l r).Vx F ⊔ blockSubspace F blk r₀ at h
  obtain ⟨w, hw, f, hf, hsum⟩ := Submodule.mem_sup.mp h
  obtain ⟨v', hv', v'', hv'', μ, rfl⟩ := mem_Vx_node_true hw
  have hl' := xi_mem_supp (F := F) l
  have hr' := xi_mem_supp (F := F) r
  have e₁ := congrArg (restrict F l.cells) hsum
  have e₂ := congrArg (restrict F r.cells) hsum
  rw [map_add, map_add, map_add, map_smul, map_sub, restrict_of_mem_supp (Vx_le_supp l hv'),
    restrict_of_mem_supp_disjoint (Vx_le_supp r hv'') hd, restrict_of_mem_supp hl',
    restrict_of_mem_supp_disjoint hr' hd, add_zero, sub_zero] at e₁
  rw [map_add, map_add, map_add, map_smul, map_sub, restrict_of_mem_supp (Vx_le_supp r hv''),
    restrict_of_mem_supp_disjoint (Vx_le_supp l hv') hd.symm, restrict_of_mem_supp hr',
    restrict_of_mem_supp_disjoint hl' hd.symm, zero_add, zero_sub] at e₂
  have mf₁ : restrict F l.cells f ∈ blockSubspace F blk r₀ := restrict_mem_blockSubspace hf
  have mf₂ : restrict F r.cells f ∈ blockSubspace F blk r₀ := restrict_mem_blockSubspace hf
  by_cases hμ : μ = 1
  · subst hμ
    right
    change r.xi F ∈ r.Vx F ⊔ blockSubspace F blk r₀
    have : r.xi F = v'' + restrict F r.cells f := by
      rw [one_smul] at e₂
      calc r.xi F = r.xi F + (v'' + -r.xi F + restrict F r.cells f) := by rw [e₂, add_zero]
        _ = v'' + restrict F r.cells f := by abel
    rw [this]
    exact Submodule.add_mem_sup hv'' mf₂
  · left
    change l.xi F ∈ l.Vx F ⊔ blockSubspace F blk r₀
    have hμ' : (1 - μ) ≠ 0 := sub_ne_zero.mpr (Ne.symm hμ)
    have : l.xi F = (1 - μ)⁻¹ • v' + (1 - μ)⁻¹ • restrict F l.cells f := by
      rw [← smul_add]
      apply smul_right_injective (κ → F) hμ'
      simp only
      rw [smul_smul, mul_inv_cancel₀ hμ', one_smul, sub_smul, one_smul]
      calc l.xi F - μ • l.xi F = (v' + μ • l.xi F + restrict F l.cells f) - μ • l.xi F := by
            rw [e₁]
        _ = v' + restrict F l.cells f := by abel
    rw [this]
    exact Submodule.add_mem_sup (Submodule.smul_mem _ _ hv') (Submodule.smul_mem _ _ mf₁)

/-- **Claim 3, the other type's side**: `γ(S) ⊆ γ(S') ∩ γ(S'')` at a node whose `V` is the children's sum. -/
theorem handle_node_false {l r : Hier κ} (hd : Disjoint l.cells r.cells) {r₀ : ι}
    (h : r₀ ∈ handle F blk (node false l r)) : r₀ ∈ handle F blk l ∧ r₀ ∈ handle F blk r := by
  change l.xi F + r.xi F ∈ (l.Vx F ⊔ r.Vx F) ⊔ blockSubspace F blk r₀ at h
  obtain ⟨w, hw, f, hf, hsum⟩ := Submodule.mem_sup.mp h
  obtain ⟨v', hv', v'', hv'', rfl⟩ := Submodule.mem_sup.mp hw
  have hl' := xi_mem_supp (F := F) l
  have hr' := xi_mem_supp (F := F) r
  have e₁ := congrArg (restrict F l.cells) hsum
  have e₂ := congrArg (restrict F r.cells) hsum
  rw [map_add, map_add, map_add, restrict_of_mem_supp (Vx_le_supp l hv'),
    restrict_of_mem_supp_disjoint (Vx_le_supp r hv'') hd, restrict_of_mem_supp hl',
    restrict_of_mem_supp_disjoint hr' hd, add_zero, add_zero] at e₁
  rw [map_add, map_add, map_add, restrict_of_mem_supp (Vx_le_supp r hv''),
    restrict_of_mem_supp_disjoint (Vx_le_supp l hv') hd.symm, restrict_of_mem_supp hr',
    restrict_of_mem_supp_disjoint hl' hd.symm, zero_add, zero_add] at e₂
  refine ⟨?_, ?_⟩
  · change l.xi F ∈ l.Vx F ⊔ blockSubspace F blk r₀
    rw [← e₁]
    exact Submodule.add_mem_sup hv' (restrict_mem_blockSubspace hf)
  · change r.xi F ∈ r.Vx F ⊔ blockSubspace F blk r₀
    rw [← e₂]
    exact Submodule.add_mem_sup hv'' (restrict_mem_blockSubspace hf)

/-- The hierarchy's shape: every type set to `false`. -/
def untype : Hier κ → Hier κ
  | leaf t => leaf t
  | node _ l r => node false l.untype r.untype

end Hier

section Typing

variable {F : Type*} [Field F] {p q : ℕ}

open Hier

/-- **Claim 4 (a handle pair pins a cell).** If `r ∈ ρ(S)` and `c ∈ γ(S)`, the cell `(r, c)` is below `S`. -/
theorem Hier.mem_cells_of_handles :
    ∀ h : Hier (Fin p × Fin q), h.WF → ∀ {r : Fin p} {c : Fin q},
      r ∈ handle F Prod.fst h → c ∈ handle F Prod.snd h.flip → (r, c) ∈ h.cells
  | leaf t, _, r, c, hr, hc => by
    have h₁ : t.1 = r := handle_leaf hr
    have h₂ : t.2 = c := handle_leaf (t := t) hc
    subst h₁ h₂
    simp [cells]
  | node true l r', hwf, r, c, hr, hc => by
    obtain ⟨hl, hr', hd⟩ := hwf
    have hc₀ : c ∈ handle F Prod.snd (node false l.flip r'.flip) := hc
    have hc' := handle_node_false (by rwa [cells_flip, cells_flip]) hc₀
    rcases handle_node_true hd hr with h | h
    · exact Finset.mem_union_left _ (mem_cells_of_handles l hl h hc'.1)
    · exact Finset.mem_union_right _ (mem_cells_of_handles r' hr' h hc'.2)
  | node false l r', hwf, r, c, hr, hc => by
    obtain ⟨hl, hr', hd⟩ := hwf
    have hc₀ : c ∈ handle F Prod.snd (node true l.flip r'.flip) := hc
    have hr₂ := handle_node_false hd hr
    rcases handle_node_true (by rwa [cells_flip, cells_flip]) hc₀ with h | h
    · exact Finset.mem_union_left _ (mem_cells_of_handles l hl hr₂.1 h)
    · exact Finset.mem_union_right _ (mem_cells_of_handles r' hr' hr₂.2 h)

open Classical in
/-- **The typing rule**: bottom-up, a node is an x-node when its children's row handles are disjoint, and a y-node
otherwise. -/
noncomputable def Hier.retype (F : Type*) [Field F] : Hier (Fin p × Fin q) → Hier (Fin p × Fin q)
  | leaf t => leaf t
  | node _ l r =>
    node (if Disjoint (handle F Prod.fst (retype F l)) (handle F Prod.fst (retype F r)) then true
      else false) (retype F l) (retype F r)

/-- **Lemma 3's induction.** The typing rule never gets stuck: on a hierarchy with disjoint siblings it gives a
typing of the same shape that is valid on both sides (`V_x ∩ F^row r = 0` and `V_y ∩ F^col c = 0` for all `r, c`). -/
theorem Hier.retype_spec : ∀ h : Hier (Fin p × Fin q), h.WF →
    (retype F h).untype = h.untype ∧ (retype F h).cells = h.cells ∧ (retype F h).WF ∧
      SideValid F Prod.fst (retype F h) ∧ SideValid F Prod.snd (retype F h).flip
  | leaf t, _ => ⟨rfl, rfl, trivial, fun _ => disjoint_bot_left, fun _ => disjoint_bot_left⟩
  | node b l r, hwf => by
    obtain ⟨hl, hr, hd⟩ := hwf
    obtain ⟨ul, cl, wl, vl, vl'⟩ := retype_spec l hl
    obtain ⟨ur, cr, wr, vr, vr'⟩ := retype_spec r hr
    have hd' : Disjoint (retype F l).cells (retype F r).cells := by rwa [cl, cr]
    have hdf : Disjoint (retype F l).flip.cells (retype F r).flip.cells := by
      rwa [cells_flip, cells_flip]
    by_cases hdis : Disjoint (handle F Prod.fst (retype F l)) (handle F Prod.fst (retype F r))
    · have e : retype F (node b l r) = node true (retype F l) (retype F r) := by
        simp [retype, hdis]
      rw [e]
      refine ⟨by simp [untype, ul, ur], by simp [cells, cl, cr], ⟨wl, wr, hd'⟩,
        sideValid_node_true hd' vl vr fun r₀ h₁ h₂ => Set.disjoint_left.mp hdis h₁ h₂, ?_⟩
      exact sideValid_node_false (l := (retype F l).flip) (r := (retype F r).flip) hdf vl' vr'
    · have e : retype F (node b l r) = node false (retype F l) (retype F r) := by
        simp [retype, hdis]
      rw [e]
      obtain ⟨r₀, h₁, h₂⟩ := Set.not_disjoint_iff.mp hdis
      refine ⟨by simp [untype, ul, ur], by simp [cells, cl, cr], ⟨wl, wr, hd'⟩,
        sideValid_node_false hd' vl vr, ?_⟩
      refine sideValid_node_true (l := (retype F l).flip) (r := (retype F r).flip) hdf vl' vr'
        fun c hc₁ hc₂ => ?_
      exact Finset.disjoint_left.mp hd' (mem_cells_of_handles (retype F l) wl h₁ hc₁)
        (mem_cells_of_handles (retype F r) wr h₂ hc₂)

end Typing

section Labels

variable {F : Type*} [Field F] {κ : Type*} [Fintype κ] [DecidableEq κ]

open Hier

/-- A functional `φ` applied to vector labels: `∑ t, φ t • a t` (the note's `ξ_S(a)` for `φ = ξ_S`). -/
def ev {α : Type*} (a : κ → α → F) (φ : κ → F) : α → F :=
  ∑ t, φ t • a t

omit [DecidableEq κ] in
theorem ev_add {α : Type*} (a : κ → α → F) (φ ψ : κ → F) : ev a (φ + ψ) = ev a φ + ev a ψ := by
  simp [ev, add_smul, Finset.sum_add_distrib]

omit [DecidableEq κ] in
theorem ev_sub {α : Type*} (a : κ → α → F) (φ ψ : κ → F) : ev a (φ - ψ) = ev a φ - ev a ψ := by
  simp [ev, sub_smul, Finset.sum_sub_distrib]

theorem ev_single {α : Type*} (a : κ → α → F) (t : κ) : ev a (Pi.single t 1) = a t := by
  simp [ev, Pi.single_apply]

/-- Labels whose coordinates span the annihilator of `V`: `ev a φ = 0` exactly when `φ ∈ V`. -/
theorem exists_labels (V : Submodule F (κ → F)) :
    ∃ (α : ℕ) (a : κ → Fin α → F), ∀ φ, ev a φ = 0 ↔ φ ∈ V := by
  let W := V.dualAnnihilator
  have : Module.Free F W := Module.Free.of_divisionRing F W
  let B := Module.finBasis F W
  refine ⟨Module.finrank F W, fun t i => (B i : Module.Dual F (κ → F)) (Pi.single t 1), fun φ => ?_⟩
  have hev : ev (fun t i => (B i : Module.Dual F (κ → F)) (Pi.single t 1)) φ =
      fun i => (B i : Module.Dual F (κ → F)) φ := by
    funext i
    conv_rhs => rw [← Finset.univ_sum_single φ]
    simp only [ev, Finset.sum_apply, Pi.smul_apply, smul_eq_mul, map_sum]
    refine Finset.sum_congr rfl fun t _ => ?_
    rw [← smul_eq_mul, ← map_smul]
    congr 1
    funext u
    by_cases hu : u = t
    · subst hu
      simp
    · simp [hu]
  rw [hev]
  constructor
  · intro h0
    rw [← Subspace.dualAnnihilator_dualCoannihilator_eq (W := V), Submodule.mem_dualCoannihilator]
    intro ψ hψ
    let f : W →ₗ[F] F := (Module.Dual.eval F (κ → F) φ).comp W.subtype
    have hf : f = 0 := B.ext fun i => by
      simpa [f] using congrFun h0 i
    simpa [f] using LinearMap.congr_fun hf ⟨ψ, hψ⟩
  · intro hφ
    funext i
    exact (Submodule.mem_dualAnnihilator _).mp (B i).2 φ hφ

omit [Fintype κ] in
theorem Vx_children_le {b : Bool} {l r : Hier κ} :
    l.Vx F ≤ (node b l r).Vx F ∧ r.Vx F ≤ (node b l r).Vx F := by
  cases b
  · exact ⟨le_sup_left, le_sup_right⟩
  · exact ⟨le_sup_left.trans le_sup_left, le_sup_right.trans le_sup_left⟩

/-- **Claim 1 (rank one).** If the labels annihilate `V ⊇ V_x(S)` and `V' ⊇ V_y(S)`, then
`Λ(S) = ξ_S(a) η_S(b)ᵀ`. -/
theorem claim1 {α β : Type*} (a : κ → α → F) (b : κ → β → F) {V V' : Submodule F (κ → F)}
    (ha : ∀ φ ∈ V, ev a φ = 0) (hb : ∀ φ ∈ V', ev b φ = 0) :
    ∀ h : Hier κ, h.WF → h.Vx F ≤ V → h.flip.Vx F ≤ V' →
      ∑ t ∈ h.cells, vecMulVec (a t) (b t) = vecMulVec (ev a (h.xi F)) (ev b (h.flip.xi F))
  | leaf t, _, _, _ => by simp [cells, xi, Hier.flip, ev_single]
  | node true l r, hwf, hV, hV' => by
    obtain ⟨hl, hr, hd⟩ := hwf
    have hV₀ : l.Vx F ⊔ r.Vx F ⊔ Submodule.span F {l.xi F - r.xi F} ≤ V := hV
    have hV₀' : l.flip.Vx F ⊔ r.flip.Vx F ≤ V' := hV'
    have hlr : ev a (l.xi F) = ev a (r.xi F) := sub_eq_zero.mp <| by
      rw [← ev_sub]
      exact ha _ (hV₀ (le_sup_right (a := l.Vx F ⊔ r.Vx F) (Submodule.mem_span_singleton_self _)))
    rw [cells, Finset.sum_union hd, claim1 a b ha hb l hl (le_sup_left.trans (le_sup_left.trans hV₀))
      (le_sup_left.trans hV₀'), claim1 a b ha hb r hr (le_sup_right.trans (le_sup_left.trans hV₀))
      (le_sup_right.trans hV₀'), ← hlr]
    change _ = vecMulVec (ev a (l.xi F)) (ev b (l.flip.xi F + r.flip.xi F))
    rw [ev_add, vecMulVec_add]
  | node false l r, hwf, hV, hV' => by
    obtain ⟨hl, hr, hd⟩ := hwf
    have hV₀ : l.Vx F ⊔ r.Vx F ≤ V := hV
    have hV₀' : l.flip.Vx F ⊔ r.flip.Vx F ⊔ Submodule.span F {l.flip.xi F - r.flip.xi F} ≤ V' := hV'
    have hlr : ev b (l.flip.xi F) = ev b (r.flip.xi F) := sub_eq_zero.mp <| by
      rw [← ev_sub]
      exact hb _ (hV₀' (le_sup_right (a := l.flip.Vx F ⊔ r.flip.Vx F)
        (Submodule.mem_span_singleton_self _)))
    rw [cells, Finset.sum_union hd, claim1 a b ha hb l hl (le_sup_left.trans hV₀)
      (le_sup_left.trans (le_sup_left.trans hV₀')), claim1 a b ha hb r hr (le_sup_right.trans hV₀)
      (le_sup_right.trans (le_sup_left.trans hV₀')), ← hlr]
    change _ = vecMulVec (ev a (l.xi F + r.xi F)) (ev b (l.flip.xi F))
    rw [ev_add, add_vecMulVec]

/-- The cell sets of a hierarchy's nodes. -/
def Hier.sets : Hier κ → Finset (Finset κ)
  | leaf t => {{t}}
  | node _ l r => insert (l.cells ∪ r.cells) (l.sets ∪ r.sets)

omit [Fintype κ] in
theorem Hier.cells_untype : ∀ h : Hier κ, h.untype.cells = h.cells
  | leaf _ => rfl
  | node _ l r => by simp only [untype, cells, cells_untype l, cells_untype r]

omit [Fintype κ] in
theorem Hier.sets_untype : ∀ h : Hier κ, h.untype.sets = h.sets
  | leaf _ => rfl
  | node _ l r => by
    simp only [untype, sets, sets_untype l, sets_untype r, cells_untype l, cells_untype r]

/-- **Claim 1 at every node**: (L) for every node set of the hierarchy. -/
theorem claim1_sets {α β : Type*} (a : κ → α → F) (b : κ → β → F) {V V' : Submodule F (κ → F)}
    (ha : ∀ φ ∈ V, ev a φ = 0) (hb : ∀ φ ∈ V', ev b φ = 0) :
    ∀ h : Hier κ, h.WF → h.Vx F ≤ V → h.flip.Vx F ≤ V' →
      ∀ S ∈ h.sets, ∃ u v, ∑ t ∈ S, vecMulVec (a t) (b t) = vecMulVec u v
  | leaf t, hwf, hV, hV', S, hS => by
    simp only [sets, Finset.mem_singleton] at hS
    subst hS
    exact ⟨_, _, claim1 a b ha hb (leaf t) hwf hV hV'⟩
  | node bb l r, hwf, hV, hV', S, hS => by
    rcases Finset.mem_insert.mp hS with rfl | hS
    · exact ⟨_, _, claim1 a b ha hb (node bb l r) hwf hV hV'⟩
    · obtain ⟨hl, hr, _⟩ := hwf
      have hVl := Vx_children_le (F := F) (b := bb) (l := l) (r := r)
      have hVf := Vx_children_le (F := F) (b := !bb) (l := l.flip) (r := r.flip)
      rcases Finset.mem_union.mp hS with hS | hS
      · exact claim1_sets a b ha hb l hl (hVl.1.trans hV) (hVf.1.trans hV') S hS
      · exact claim1_sets a b ha hb r hr (hVl.2.trans hV) (hVf.2.trans hV') S hS

/-- **Claim 2 (independence).** If `ev a φ = 0` only for `φ ∈ V`, and `V` meets no block of `blk`, the labels of each
block are linearly independent. -/
theorem claim2 {α ι : Type*} [DecidableEq ι] (a : κ → α → F) (V : Submodule F (κ → F))
    (ha : ∀ φ, ev a φ = 0 → φ ∈ V) (blk : κ → ι) (r : ι) (hV : Disjoint V (blockSubspace F blk r))
    (s : Finset κ) : LinearIndepOn F a ↑(s.filter (blk · = r)) := by
  rw [linearIndepOn_finset_iff]
  intro g hg t ht
  let φ : κ → F := fun u => if u ∈ s.filter (blk · = r) then g u else 0
  have hφ : ev a φ = 0 := by
    rw [← hg]
    simp only [ev, φ, ite_smul, zero_smul]
    rw [Finset.sum_ite_mem, Finset.univ_inter]
  have hmem : φ ∈ blockSubspace F blk r := by
    intro u hu
    simp only [φ, Finset.mem_filter]
    rw [ite_eq_right_iff]
    exact fun h => absurd h.2 hu
  have h0 := congrFun (Submodule.disjoint_def.mp hV φ (ha φ hφ) hmem) t
  simp only [φ, ht, ↓reduceIte, Pi.zero_apply] at h0
  exact h0

/-- **Lemma 3 for a binary hierarchy**: labels with (L) at every node and (I) in every row and column. -/
theorem labels_of_hier {p q : ℕ} (h : Hier (Fin p × Fin q)) (hwf : h.WF) :
    ∃ (α β : ℕ) (a : Fin p × Fin q → Fin α → F) (b : Fin p × Fin q → Fin β → F),
      (∀ S ∈ h.sets, ∃ u v, ∑ t ∈ S, vecMulVec (a t) (b t) = vecMulVec u v) ∧
      (∀ r, LinearIndepOn F a ↑(h.cells.filter (·.1 = r))) ∧
      (∀ c, LinearIndepOn F b ↑(h.cells.filter (·.2 = c))) := by
  obtain ⟨hu, hc, hw, hvr, hvc⟩ := retype_spec (F := F) h hwf
  obtain ⟨α, a, ha⟩ := exists_labels ((retype F h).Vx F)
  obtain ⟨β, b, hb⟩ := exists_labels ((retype F h).flip.Vx F)
  refine ⟨α, β, a, b, fun S hS => ?_, fun r => ?_, fun c => ?_⟩
  · rw [← sets_untype, ← hu, sets_untype] at hS
    exact claim1_sets a b (fun φ hφ => (ha φ).mpr hφ) (fun φ hφ => (hb φ).mpr hφ) (retype F h) hw
      le_rfl le_rfl S hS
  · rw [← hc]
    exact claim2 a _ (fun φ hφ => (ha φ).mp hφ) Prod.fst r (hvr r) _
  · rw [← hc]
    exact claim2 b _ (fun φ hφ => (hb φ).mp hφ) Prod.snd c (hvc c) _

/-- **Binary refinement** (the note's reduction to full binary hierarchies): the nonempty sets of a laminar family
of subsets of a nonempty `P` are node sets of a hierarchy on `P` with disjoint siblings. -/
def BinaryRefinement (κ : Type*) [DecidableEq κ] : Prop :=
  ∀ (P : Finset κ) (𝓕 : Finset (Finset κ)), P.Nonempty → Laminar 𝓕 → (∀ S ∈ 𝓕, S ⊆ P) →
    ∃ h : Hier κ, h.WF ∧ h.cells = P ∧ ∀ S ∈ 𝓕, S.Nonempty → S ∈ h.sets

/-- **Lemma 3 from binary refinement.** -/
theorem labelLemma_of_binaryRefinement (href : ∀ p q : ℕ, BinaryRefinement (Fin p × Fin q)) :
    LabelLemma F := by
  intro p q P 𝓕 hlam hsub
  rcases P.eq_empty_or_nonempty with hP | hP
  · subst hP
    refine ⟨0, 0, fun _ => 0, fun _ => 0, fun S hS => ⟨0, 0, ?_⟩, fun r => ?_, fun c => ?_⟩
    · rw [Finset.subset_empty.mp (hsub S hS)]
      ext i
      exact i.elim0
    · simp
    · simp
  · obtain ⟨h, hwf, hcells, hsets⟩ := href p q P 𝓕 hP hlam hsub
    obtain ⟨α, β, a, b, hL, hIa, hIb⟩ := labels_of_hier (F := F) h hwf
    refine ⟨α, β, a, b, fun S hS => ?_, hcells ▸ hIa, hcells ▸ hIb⟩
    rcases S.eq_empty_or_nonempty with hS0 | hS0
    · refine ⟨0, 0, ?_⟩
      rw [hS0, Finset.sum_empty]
      ext i j
      simp [vecMulVec_apply]
    · exact hL S (hsets S hS hS0)

omit [Fintype κ] [DecidableEq κ] in
theorem laminar_filter {𝓕 : Finset (Finset κ)} (h : Laminar 𝓕) (p : Finset κ → Prop) [DecidablePred p] :
    Laminar (𝓕.filter p) := fun S hS S' hS' =>
  h S (Finset.mem_filter.mp hS).1 S' (Finset.mem_filter.mp hS').1

omit [Fintype κ] in
/-- A laminar family on `P` with `|P| ≥ 2` leaves a proper nonempty `A ⊆ P` that every other set of the family
lies in or misses: a largest proper nonempty set of the family, or a singleton if there is none. -/
theorem exists_split {P : Finset κ} {𝓕 : Finset (Finset κ)} (hP : P.Nonempty) (h1 : P.card ≠ 1)
    (hlam : Laminar 𝓕) (hsub : ∀ S ∈ 𝓕, S ⊆ P) :
    ∃ A ⊆ P, A.Nonempty ∧ A ≠ P ∧ ∀ S ∈ 𝓕, S ≠ P → S ⊆ A ∨ Disjoint S A := by
  classical
  let 𝓕' := 𝓕.filter fun S => S.Nonempty ∧ S ≠ P
  rcases 𝓕'.eq_empty_or_nonempty with h0 | hne
  · obtain ⟨t, ht⟩ := hP
    refine ⟨{t}, Finset.singleton_subset_iff.mpr ht, Finset.singleton_nonempty t, fun h => h1 ?_,
      fun S hS hSP => Or.inl ?_⟩
    · rw [← h, Finset.card_singleton]
    · rcases S.eq_empty_or_nonempty with hS0 | hS0
      · rw [hS0]
        exact Finset.empty_subset _
      · exact absurd (Finset.mem_filter.mpr ⟨hS, hS0, hSP⟩) (h0 ▸ Finset.notMem_empty S)
  · obtain ⟨M, hM, hmax⟩ := Finset.exists_max_image 𝓕' Finset.card hne
    obtain ⟨hM𝓕, hMne, hMP⟩ := Finset.mem_filter.mp hM
    refine ⟨M, hsub M hM𝓕, hMne, hMP, fun S hS hSP => ?_⟩
    rcases hlam M hM𝓕 S hS with h | h | h
    · rcases S.eq_empty_or_nonempty with hS0 | hS0
      · rw [hS0]
        exact Or.inl (Finset.empty_subset _)
      · have hcard := hmax S (Finset.mem_filter.mpr ⟨hS, hS0, hSP⟩)
        exact Or.inl (Finset.eq_of_subset_of_card_le h hcard).symm.subset
    · exact Or.inl h
    · exact Or.inr h.symm

omit [Fintype κ] in
/-- **Binary refinement, proved.** -/
theorem binaryRefinement : BinaryRefinement κ := by
  intro P
  induction P using Finset.strongInduction with
  | H P ih =>
    intro 𝓕 hP hlam hsub
    by_cases h1 : P.card = 1
    · obtain ⟨t, rfl⟩ := Finset.card_eq_one.mp h1
      refine ⟨leaf t, trivial, rfl, fun S hS hne => ?_⟩
      rcases Finset.subset_singleton_iff.mp (hsub S hS) with h | h
      · exact absurd h hne.ne_empty
      · rw [h]
        exact Finset.mem_singleton_self _
    · obtain ⟨A, hAP, hAne, hAneP, hsplit⟩ := exists_split hP h1 hlam hsub
      have hBne : (P \ A).Nonempty := by
        rw [Finset.sdiff_nonempty]
        exact fun h => hAneP (Finset.Subset.antisymm hAP h)
      obtain ⟨hA, wA, cA, sA⟩ := ih A (Finset.ssubset_iff_subset_ne.mpr ⟨hAP, hAneP⟩)
        (𝓕.filter (· ⊆ A)) hAne (laminar_filter hlam _) fun S hS => (Finset.mem_filter.mp hS).2
      obtain ⟨hB, wB, cB, sB⟩ := ih (P \ A) (Finset.sdiff_ssubset hAP hAne)
        (𝓕.filter (· ⊆ P \ A)) hBne (laminar_filter hlam _) fun S hS => (Finset.mem_filter.mp hS).2
      have hcells : (node false hA hB).cells = P := by
        simp only [cells, cA, cB, Finset.union_sdiff_of_subset hAP]
      refine ⟨node false hA hB, ⟨wA, wB, by rw [cA, cB]; exact Finset.disjoint_sdiff⟩, hcells,
        fun S hS hne => ?_⟩
      by_cases hSP : S = P
      · rw [hSP, ← hcells]
        exact Finset.mem_insert_self _ _
      · refine Finset.mem_insert_of_mem (Finset.mem_union.mpr ?_)
        rcases hsplit S hS hSP with h | h
        · exact Or.inl (sA S (Finset.mem_filter.mpr ⟨hS, h⟩) hne)
        · exact Or.inr (sB S (Finset.mem_filter.mpr ⟨hS, Finset.subset_sdiff.mpr ⟨hsub S hS, h⟩⟩) hne)

/-- **Lemma 3 (labels for a laminar family), proved.** -/
theorem labelLemma : LabelLemma F :=
  labelLemma_of_binaryRefinement fun _ _ => binaryRefinement

end Labels

section Assembly

variable {F : Type*} [Field F] {m n : Type*} [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n]

/-- **Theorem 1 for laminar decompositions, proved.** -/
theorem laminarBound : LaminarBound F m n :=
  laminarBound_of_labelLemma labelLemma

/-- **Theorem 1 in coordinates from Lemma 1.** -/
theorem blockRigidityCoord_of_treeLaminarDecomposition (h1 : TreeLaminarDecomposition F m n) :
    BlockRigidityCoord F m n :=
  blockRigidityCoord_of_laminarBound h1 laminarBound

/-- **Conjecture A's lower bound from Lemma 1.** -/
theorem conjectureA_lower_of_treeLaminarDecomposition {N L : ℕ}
    (h1 : TreeLaminarDecomposition F (Fin N × Fin N) (Fin N × Fin N)) (hL : 0 < L) (hLN : L ∣ N)
    (T : SteinerTree F (Fin N × Fin N) (Fin N × Fin N)) (h0 : T.Reaches 0)
    (hT : ∀ r c : Fin N, ∀ s ∈ Finset.Icc 1 (N / L), T.Reaches (prefixTarget F N L r c s)) :
    N ^ 3 ≤ T.length :=
  conjectureA_lower_of_blockRigidityCoord (blockRigidityCoord_of_treeLaminarDecomposition h1)
    hL hLN T h0 hT

end Assembly

end Hierarchy

section TreeDecomposition

variable {F : Type*} [Field F] {m n : Type*}

namespace SteinerTree

/-- The tree's path from `u` to `x`. -/
noncomputable def path (T : SteinerTree F m n) (u x : T.V) : T.G.Walk u x :=
  (T.isTree.existsUnique_path u x).exists.choose

theorem path_isPath (T : SteinerTree F m n) (u x : T.V) : (T.path u x).IsPath :=
  (T.isTree.existsUnique_path u x).exists.choose_spec

theorem eq_path (T : SteinerTree F m n) {u x : T.V} {p : T.G.Walk u x} (hp : p.IsPath) :
    p = T.path u x :=
  (T.isTree.existsUnique_path u x).unique hp (T.path_isPath u x)

/-- Every dart of a path from `u` is the last dart of the tree's path from `u` to its head; that path is a prefix,
and the head is one step further from `u` than the tail. -/
theorem path_darts (T : SteinerTree F m n) {u x : T.V} (p : T.G.Walk u x) (hp : p.IsPath) :
    ∀ d ∈ p.darts, d ∈ (T.path u d.snd).darts ∧ (T.path u d.snd).darts <+: p.darts ∧
      (T.path u d.snd).length = (T.path u d.fst).length + 1 := by
  induction p using SimpleGraph.Walk.concatRec with
  | Hnil => simp
  | @Hconcat u v w p h ih =>
    have hp' : p.IsPath := by
      rw [SimpleGraph.Walk.concat_eq_append] at hp
      exact hp.of_append_left
    intro d hd
    rw [SimpleGraph.Walk.darts_concat, List.concat_eq_append, List.mem_append, List.mem_singleton] at hd
    rcases hd with hd | rfl
    · obtain ⟨h1, h2, h3⟩ := ih hp' d hd
      refine ⟨h1, h2.trans ?_, h3⟩
      rw [SimpleGraph.Walk.darts_concat, List.concat_eq_append]
      exact List.prefix_append _ _
    · change _ ∈ (T.path u w).darts ∧ (T.path u w).darts <+: _ ∧
        (T.path u w).length = (T.path u v).length + 1
      rw [← T.eq_path hp, ← T.eq_path hp']
      exact ⟨by simp, List.prefix_refl _, by simp⟩

end SteinerTree

theorem sum_darts_sub {V : Type} {G : SimpleGraph V} (label : V → Matrix m n F) {u x : V}
    (p : G.Walk u x) : (p.darts.map fun d => label d.snd - label d.fst).sum = label x - label u := by
  induction p with
  | nil => simp
  | cons h p ih =>
    simp only [SimpleGraph.Walk.darts_cons, List.map_cons, List.sum_cons, ih]
    abel

variable [Fintype m] [Fintype n]

/-- **Lemma 1, proved.** Root the tree at a vertex labelled `0`; each dart gets the set of targets whose root path
uses it, and each set of the family gets the sum of the steps of the darts with that set. -/
theorem treeLaminarDecomposition : TreeLaminarDecomposition F m n := by
  classical
  intro κ _ P M T h0 hP
  obtain ⟨v₀, hv₀⟩ := h0
  have : Nonempty T.V := ⟨v₀⟩
  have hP' : ∀ t ∈ P, ∃ v, T.label v = M t := hP
  choose! vt hvt using hP'
  let pt : κ → List T.G.Dart := fun t => (T.path v₀ (vt t)).darts
  let S : T.G.Dart → Finset κ := fun d => P.filter fun t => d ∈ pt t
  let U : Finset T.G.Dart := Finset.univ.filter fun d => (S d).Nonempty
  let D : T.G.Dart → Matrix m n F := fun d => T.label d.snd - T.label d.fst
  have hpd : ∀ t, ∀ d ∈ pt t, d ∈ (T.path v₀ d.snd).darts ∧ (T.path v₀ d.snd).darts <+: pt t ∧
      (T.path v₀ d.snd).length = (T.path v₀ d.fst).length + 1 := fun t =>
    T.path_darts _ (T.path_isPath _ _)
  refine ⟨U.image S, fun A => ∑ d ∈ U.filter (S · = A), D d, ?_, ?_, ?_, ?_⟩
  · intro A hA A' hA'
    obtain ⟨d, -, rfl⟩ := Finset.mem_image.1 hA
    obtain ⟨d', -, rfl⟩ := Finset.mem_image.1 hA'
    by_cases hdis : Disjoint (S d) (S d')
    · exact Or.inr (Or.inr hdis)
    obtain ⟨t, htd, htd'⟩ := Finset.not_disjoint_iff.1 hdis
    simp only [S, Finset.mem_filter] at htd htd'
    obtain ⟨h1, h2, -⟩ := hpd t d htd.2
    obtain ⟨h1', h2', -⟩ := hpd t d' htd'.2
    have key : ∀ {d d' : T.G.Dart}, d ∈ (T.path v₀ d.snd).darts →
        (T.path v₀ d.snd).darts <+: (T.path v₀ d'.snd).darts → S d' ⊆ S d := by
      intro d d' hd hpre t' ht'
      simp only [S, Finset.mem_filter] at ht' ⊢
      exact ⟨ht'.1, (hpre.trans (hpd t' d' ht'.2).2.1).subset hd⟩
    rcases le_total (T.path v₀ d.snd).darts.length (T.path v₀ d'.snd).darts.length with hle | hle
    · exact Or.inr (Or.inl (key h1 (List.prefix_of_prefix_length_le h2 h2' hle)))
    · exact Or.inl (key h1' (List.prefix_of_prefix_length_le h2' h2 hle))
  · intro A hA
    obtain ⟨d, -, rfl⟩ := Finset.mem_image.1 hA
    exact Finset.filter_subset _ _
  · intro t ht
    have hfib : ∑ A ∈ (U.image S).filter (t ∈ ·), ∑ d ∈ U.filter (S · = A), D d =
        ∑ d ∈ U.filter (t ∈ S ·), D d := by
      rw [← Finset.sum_fiberwise_of_maps_to (s := U.filter (t ∈ S ·)) (t := (U.image S).filter (t ∈ ·))
        (g := S)]
      · refine Finset.sum_congr rfl fun A hA => Finset.sum_congr ?_ fun _ _ => rfl
        ext d
        simp only [Finset.mem_filter] at hA ⊢
        constructor
        · rintro ⟨hd, rfl⟩
          exact ⟨⟨hd, hA.2⟩, rfl⟩
        · rintro ⟨⟨hd, -⟩, rfl⟩
          exact ⟨hd, rfl⟩
      · intro d hd
        simp only [Finset.mem_filter, Finset.mem_image] at hd ⊢
        exact ⟨⟨d, hd.1, rfl⟩, hd.2⟩
    rw [hfib]
    have hset : U.filter (t ∈ S ·) = (pt t).toFinset := by
      ext d
      simp only [U, S, Finset.mem_filter, Finset.mem_univ, true_and, List.mem_toFinset]
      constructor
      · rintro ⟨-, -, h⟩
        exact h
      · intro h
        exact ⟨⟨t, Finset.mem_filter.2 ⟨ht, h⟩⟩, ht, h⟩
    rw [hset, List.sum_toFinset _
      (SimpleGraph.Walk.darts_nodup_of_support_nodup (T.path_isPath _ _).support_nodup)]
    rw [sum_darts_sub, hvt t ht, hv₀, sub_zero]
  · have hlen : ∀ d ∈ U, (T.path v₀ d.snd).length = (T.path v₀ d.fst).length + 1 := by
      intro d hd
      simp only [U, Finset.mem_filter, Finset.mem_univ, true_and] at hd
      obtain ⟨t, ht⟩ := hd
      simp only [S, Finset.mem_filter] at ht
      exact (hpd t d ht.2).2.2
    calc ∑ A ∈ U.image S, (∑ d ∈ U.filter (S · = A), D d).rank
        ≤ ∑ A ∈ U.image S, ∑ d ∈ U.filter (S · = A), (D d).rank :=
          Finset.sum_le_sum fun A _ => rank_sum_le _ _
      _ = ∑ d ∈ U, (D d).rank :=
          Finset.sum_fiberwise_of_maps_to (fun d hd => Finset.mem_image_of_mem S hd) _
      _ = ∑ e ∈ U.image SimpleGraph.Dart.edge, SteinerTree.edgeLength T.label e := by
          rw [Finset.sum_image]
          · refine Finset.sum_congr rfl fun d _ => ?_
            obtain ⟨⟨a, b⟩, hab⟩ := d
            change rankDist (T.label b) (T.label a) = rankDist (T.label a) (T.label b)
            exact rankDist_comm _ _
          · intro d hd d' hd' he
            rcases (SimpleGraph.dart_edge_eq_iff d d').1 he with h | h
            · exact h
            · exfalso
              subst h
              have h1 := hlen _ hd
              have h2 := hlen _ hd'
              change (T.path v₀ d'.fst).length = (T.path v₀ d'.snd).length + 1 at h1
              omega
      _ ≤ T.length := by
          refine Finset.sum_le_sum_of_subset fun e he => ?_
          obtain ⟨d, -, rfl⟩ := Finset.mem_image.1 he
          exact SimpleGraph.mem_edgeFinset.2 d.edge_mem

variable [DecidableEq m] [DecidableEq n]

/-- **Theorem 1 in coordinates, proved**: rows and columns split into coordinate blocks, targets in distinct cells;
every Steiner tree spanning `0` and the targets has length at least the sum of their ranks. -/
theorem blockRigidityCoord : BlockRigidityCoord F m n :=
  blockRigidityCoord_of_treeLaminarDecomposition treeLaminarDecomposition

/-- **Conjecture A's lower bound, proved**: a Steiner tree spanning `0` and every prefix target
`E_rc ⊗ I_{sL}` has length at least `N³`. -/
theorem conjectureA_lower {N L : ℕ} (hL : 0 < L) (hLN : L ∣ N)
    (T : SteinerTree F (Fin N × Fin N) (Fin N × Fin N)) (h0 : T.Reaches 0)
    (hT : ∀ r c : Fin N, ∀ s ∈ Finset.Icc 1 (N / L), T.Reaches (prefixTarget F N L r c s)) :
    N ^ 3 ≤ T.length :=
  conjectureA_lower_of_treeLaminarDecomposition treeLaminarDecomposition hL hLN T h0 hT

end TreeDecomposition

section ChangeOfBasis

variable {F : Type*} [Field F]

theorem iSupIndep_option_elim {M : Type*} [AddCommGroup M] [Module F M] {ι : Type*}
    {R : ι → Submodule F M} (hR : iSupIndep R) {W : Submodule F M} (hW : Disjoint W (⨆ i, R i)) :
    iSupIndep fun o : Option ι => o.elim W R := by
  rw [iSupIndep_def]
  rintro (_ | i)
  · refine hW.mono_right (iSup₂_le fun j hj => ?_)
    cases j with
    | none => exact absurd rfl hj
    | some j => exact le_iSup R j
  · have hle : (⨆ (j : Option ι) (_ : j ≠ some i), j.elim W R) ≤ W ⊔ ⨆ (k) (_ : k ≠ i), R k := by
      refine iSup₂_le fun j hj => ?_
      cases j with
      | none => exact le_sup_left
      | some k => exact le_sup_of_le_right (le_iSup₂_of_le k (fun h => hj (congrArg some h)) le_rfl)
    have hX : (⨆ (k) (_ : k ≠ i), R k) ≤ ⨆ k, R k := iSup₂_le fun k _ => le_iSup R k
    refine Disjoint.mono_right hle ?_
    rw [Submodule.disjoint_def]
    intro v hv hvs
    obtain ⟨w, hw, x, hx, rfl⟩ := Submodule.mem_sup.1 hvs
    have hwR : w ∈ ⨆ k, R k := by
      have hw' : w = (w + x) - x := by abel
      rw [hw']
      exact Submodule.sub_mem _ (le_iSup R i hv) (hX hx)
    have hw0 : w = 0 := Submodule.disjoint_def.1 hW w hw hwR
    subst hw0
    rw [zero_add] at hv ⊢
    exact Submodule.disjoint_def.1 (iSupIndep_def.1 hR i) x hv hx

variable {m : Type*} [Fintype m] [DecidableEq m]

/-- Independent subspaces of `m → F` lie in coordinate blocks after an invertible change of coordinates. -/
theorem exists_blockCoords {p : ℕ} (hp : 0 < p) (R : Fin p → Submodule F (m → F)) (hR : iSupIndep R) :
    ∃ (A : Matrix m m F) (blk : m → Fin p), IsUnit A.det ∧
      ∀ r, ∀ v ∈ R r, A *ᵥ v ∈ blockSubspace F blk r := by
  obtain ⟨W, hW⟩ := Submodule.exists_isCompl (⨆ r, R r)
  let R' : Option (Fin p) → Submodule F (m → F) := fun o => o.elim W R
  have hind : iSupIndep R' := iSupIndep_option_elim hR hW.symm.disjoint
  have htop : iSup R' = ⊤ := by
    change (⨆ o : Option (Fin p), o.elim W R) = ⊤
    rw [iSup_option_elim, sup_comm]
    exact hW.sup_eq_top
  have hint := DirectSum.isInternal_submodule_of_iSupIndep_of_iSup_eq_top hind htop
  let b := hint.collectedBasis fun o => Module.finBasis F (R' o)
  have hcard : Fintype.card (Σ o, Fin (finrank F (R' o))) = Fintype.card m := by
    rw [← Module.finrank_eq_card_basis b, Module.finrank_fintype_fun_eq_card]
  let e := Fintype.equivOfCardEq hcard
  let b' := b.reindex e
  refine ⟨LinearMap.toMatrix' (b'.equivFun : (m → F) →ₗ[F] m → F),
    fun x => (e.symm x).1.elim ⟨0, hp⟩ id, ?_, ?_⟩
  · rw [LinearMap.det_toMatrix']
    exact b'.equivFun.isUnit_det'
  · intro r v hv x hx
    rw [LinearMap.toMatrix'_mulVec]
    simp only [LinearEquiv.coe_coe, Basis.equivFun_apply, b', Basis.repr_reindex_apply]
    change (e.symm x).1.elim ⟨0, hp⟩ id ≠ r at hx
    generalize e.symm x = y at hx ⊢
    obtain ⟨o, a⟩ := y
    refine hint.collectedBasis_repr_of_mem_ne _ (i := some r) ?_ hv
    rintro rfl
    exact hx rfl

omit [DecidableEq m] in
theorem entry_eq_zero_of_colSpace_le {n' k : Type*} [Fintype n'] [DecidableEq n'] {p : ℕ}
    {A : Matrix m m F} {blk : m → Fin p} {r : Fin p} {R : Submodule F (m → F)}
    (hA : ∀ v ∈ R, A *ᵥ v ∈ blockSubspace F blk r) {X : Matrix m n' F} (hX : colSpace X ≤ R)
    (Y : Matrix n' k F) {x : m} (hx : blk x ≠ r) (y : k) : (A * X * Y) x y = 0 := by
  have hcol : ∀ j, (A * X) x j = 0 := by
    intro j
    have h1 : X *ᵥ Pi.single j 1 ∈ R := hX (LinearMap.mem_range_self X.mulVecLin _)
    have h2 := hA _ h1 x hx
    rw [Matrix.mulVec_mulVec, Matrix.mulVec_single_one] at h2
    exact h2
  simp [Matrix.mul_apply, hcol]

variable {n : Type*} [Fintype n] [DecidableEq n]

/-- The same tree with every label `X` replaced by `A * X * B`. -/
def SteinerTree.conj (T : SteinerTree F m n) (A : Matrix m m F) (B : Matrix n n F) :
    SteinerTree F m n where
  V := T.V
  G := T.G
  isTree := T.isTree
  label v := A * T.label v * B

theorem SteinerTree.length_conj (T : SteinerTree F m n) {A : Matrix m m F} {B : Matrix n n F}
    (hA : IsUnit A.det) (hB : IsUnit B.det) : (T.conj A B).length = T.length := by
  refine Finset.sum_congr rfl fun e _ => ?_
  induction e using Sym2.ind with
  | _ u v =>
    change rankDist (A * T.label u * B) (A * T.label v * B) = rankDist (T.label u) (T.label v)
    rw [rankDist, rankDist, ← Matrix.sub_mul, ← Matrix.mul_sub,
      Matrix.rank_mul_eq_left_of_isUnit_det _ _ hB, Matrix.rank_mul_eq_right_of_isUnit_det _ _ hA]

/-- **Theorem 1, proved** (block rigidity as stated: independent row subspaces `R_r`, independent column subspaces
`C_c`, targets in distinct cells): every Steiner tree spanning `0` and the targets has length at least the sum of
their ranks. -/
theorem blockRigidity : BlockRigidity F m n := by
  intro p q R C hR hC P M hM T h0 hT
  rcases Nat.eq_zero_or_pos p with rfl | hp
  · rw [Finset.eq_empty_of_isEmpty P, Finset.sum_empty]
    exact Nat.zero_le _
  rcases Nat.eq_zero_or_pos q with rfl | hq
  · rw [Finset.eq_empty_of_isEmpty P, Finset.sum_empty]
    exact Nat.zero_le _
  obtain ⟨A, rowBlk, hA, hAR⟩ := exists_blockCoords hp R hR
  obtain ⟨B, colBlk, hB, hBC⟩ := exists_blockCoords hq C hC
  have hBt : IsUnit Bᵀ.det := by rwa [Matrix.det_transpose]
  have hrank : ∀ X : Matrix m n F, (A * X * Bᵀ).rank = X.rank := fun X => by
    rw [Matrix.rank_mul_eq_left_of_isUnit_det _ _ hBt, Matrix.rank_mul_eq_right_of_isUnit_det _ _ hA]
  have hcells : InCells rowBlk colBlk P fun t => A * M t * Bᵀ := by
    intro t ht x y hxy
    have hxy' : (A * M t * Bᵀ) x y ≠ 0 := hxy
    constructor
    · by_contra hx
      exact hxy' (entry_eq_zero_of_colSpace_le (hAR t.1) (hM t ht).1 Bᵀ hx y)
    · by_contra hy
      apply hxy'
      have h := entry_eq_zero_of_colSpace_le (hBC t.2) (hM t ht).2 Aᵀ hy x
      rw [← Matrix.transpose_apply (A * M t * Bᵀ) y x, Matrix.transpose_mul, Matrix.transpose_mul,
        Matrix.transpose_transpose, ← Matrix.mul_assoc]
      exact h
  have h := blockRigidityCoord p q rowBlk colBlk P (fun t => A * M t * Bᵀ) hcells (T.conj A Bᵀ)
    (by obtain ⟨v, hv⟩ := h0; exact ⟨v, by simp [SteinerTree.conj, hv]⟩)
    (fun t ht => by obtain ⟨v, hv⟩ := hT t ht; exact ⟨v, by simp [SteinerTree.conj, hv]⟩)
  rw [SteinerTree.length_conj T hA hBt] at h
  simpa only [hrank] using h

end ChangeOfBasis

section Spider

variable {F : Type*} [Field F] {N L : ℕ}

theorem prefixTarget_zero (r c : Fin N) : prefixTarget F N L r c 0 = 0 := by
  ext i j
  simp [prefixTarget]

/-- Consecutive prefix targets are at rank distance at most `L`. -/
theorem rank_prefixTarget_succ_sub_le (r c : Fin N) (k : ℕ) :
    (prefixTarget F N L r c (k + 1) - prefixTarget F N L r c k).rank ≤ L := by
  classical
  let Q : Fin N × Fin N → Prop := fun j => j.1 = c ∧ k * L ≤ (j.2 : ℕ) ∧ (j.2 : ℕ) < (k + 1) * L
  let w : Fin N × Fin N → F := fun j => if Q j then 1 else 0
  have hD : prefixTarget F N L r c (k + 1) - prefixTarget F N L r c k =
      (prefixTarget F N L r c (k + 1) - prefixTarget F N L r c k) * Matrix.diagonal w := by
    ext i j
    rw [Matrix.mul_diagonal]
    by_cases hQ : Q j
    · simp [w, hQ]
    · have hw : w j = 0 := by simp [w, hQ]
      rw [hw, mul_zero]
      simp only [Matrix.sub_apply]
      have hiff : (i.1 = r ∧ j.1 = c ∧ i.2 = j.2 ∧ (i.2 : ℕ) < (k + 1) * L) ↔
          (i.1 = r ∧ j.1 = c ∧ i.2 = j.2 ∧ (i.2 : ℕ) < k * L) := by
        constructor
        · rintro ⟨h1, h2, h3, h4⟩
          refine ⟨h1, h2, h3, ?_⟩
          have h5 : (i.2 : ℕ) = j.2 := congrArg Fin.val h3
          by_contra h6
          exact hQ ⟨h2, by omega, by omega⟩
        · rintro ⟨h1, h2, h3, h4⟩
          exact ⟨h1, h2, h3, lt_of_lt_of_le h4 (Nat.mul_le_mul_right _ (Nat.le_succ k))⟩
      exact (sub_eq_zero (G := F)).2 (if_congr hiff (rfl : (1 : F) = 1) (rfl : (0 : F) = 0))
  have hQw : ∀ j, w j ≠ 0 → Q j := by
    intro j hj
    by_contra h
    exact hj (by simp [w, h])
  calc (prefixTarget F N L r c (k + 1) - prefixTarget F N L r c k).rank
      = ((prefixTarget F N L r c (k + 1) - prefixTarget F N L r c k) * Matrix.diagonal w).rank := by
        rw [← hD]
    _ ≤ (Matrix.diagonal w).rank := Matrix.rank_mul_le_right _ _
    _ = Fintype.card {j // w j ≠ 0} := Matrix.rank_diagonal w
    _ ≤ (Finset.Ico (k * L) ((k + 1) * L)).card := by
        rw [Fintype.card_subtype]
        apply Finset.card_le_card_of_injOn (fun j : Fin N × Fin N => (j.2 : ℕ))
        · intro j hj
          have hQ := hQw j (by simpa using hj)
          simp only [Finset.coe_Ico, Set.mem_Ico]
          exact ⟨hQ.2.1, hQ.2.2⟩
        · intro j hj j' hj' h
          have hQ := hQw j (by simpa using hj)
          have hQ' := hQw j' (by simpa using hj')
          exact Prod.ext (hQ.1.trans hQ'.1.symm) (Fin.ext h)
    _ = L := by rw [Nat.card_Ico, Nat.succ_mul, Nat.add_sub_cancel_left]

/-- The spider's parent map: the root, and for each output cell `(r, c)` a chain of `K` vertices. -/
def spiderParent {K : ℕ} : Option (Fin N × Fin N × Fin K) → Option (Fin N × Fin N × Fin K)
  | none => none
  | some (r, c, k) => if h : (k : ℕ) = 0 then none else some (r, c, ⟨k - 1, by omega⟩)

/-- The spider: each vertex joined to its parent. -/
def spiderGraph (K : ℕ) : SimpleGraph (Option (Fin N × Fin N × Fin K)) :=
  SimpleGraph.fromRel fun u v => u = spiderParent v

/-- The spider's labels: `0` at the root, the prefix target `T_{(r,c),k+1}` at `(r, c, k)`. -/
def spiderLabel (F : Type*) [Field F] (N L K : ℕ) :
    Option (Fin N × Fin N × Fin K) → Matrix (Fin N × Fin N) (Fin N × Fin N) F
  | none => 0
  | some (r, c, k) => prefixTarget F N L r c (k + 1)

theorem rankDist_spiderParent_le {K : ℕ} (v : Option (Fin N × Fin N × Fin K)) :
    rankDist (spiderLabel F N L K (spiderParent v)) (spiderLabel F N L K v) ≤ L := by
  rcases v with _ | ⟨r, c, k⟩
  · simp [spiderParent, spiderLabel, rankDist_self]
  · have hp : spiderLabel F N L K (spiderParent (some (r, c, k))) = prefixTarget F N L r c k := by
      by_cases hk : (k : ℕ) = 0
      · simp [spiderParent, spiderLabel, hk, prefixTarget_zero]
      · simp only [spiderParent, hk, ↓reduceDIte, spiderLabel]
        rw [Nat.sub_add_cancel (Nat.one_le_iff_ne_zero.2 hk)]
    rw [hp, rankDist_comm]
    exact rank_prefixTarget_succ_sub_le r c k

theorem spiderGraph_connected (K : ℕ) : (spiderGraph (N := N) K).Connected := by
  have hroot : ∀ v, (spiderGraph (N := N) K).Reachable none v := by
    rintro (_ | ⟨r, c, ⟨k, hk⟩⟩)
    · rfl
    · induction k with
      | zero => exact SimpleGraph.Adj.reachable (by simp [spiderGraph, spiderParent])
      | succ k ih =>
        refine (ih (by omega)).trans (SimpleGraph.Adj.reachable ?_)
        simp [spiderGraph, spiderParent]
  exact ⟨fun u v => (hroot u).symm.trans (hroot v)⟩

theorem spider_edgeLength_le {K : ℕ} {H : SimpleGraph (Option (Fin N × Fin N × Fin K))}
    (hH : H ≤ spiderGraph K) {e : Sym2 (Option (Fin N × Fin N × Fin K))} (he : e ∈ H.edgeSet) :
    SteinerTree.edgeLength (spiderLabel F N L K) e ≤ L := by
  induction e using Sym2.ind with
  | _ a b =>
    have hab := hH ((SimpleGraph.mem_edgeSet _).1 he)
    rw [spiderGraph, SimpleGraph.fromRel_adj] at hab
    change rankDist (spiderLabel F N L K a) (spiderLabel F N L K b) ≤ L
    rcases hab.2 with h | h
    · subst h
      exact rankDist_spiderParent_le b
    · subst h
      rw [rankDist_comm]
      exact rankDist_spiderParent_le a

/-- **Conjecture A's upper bound, proved**: a spanning tree of the spider (`0`, then for each output cell the chain
of its prefix targets) has length `N³`. -/
theorem conjectureA_upper (hL : 0 < L) (hLN : L ∣ N) :
    ∃ T : SteinerTree F (Fin N × Fin N) (Fin N × Fin N), T.Reaches 0 ∧
      (∀ r c : Fin N, ∀ s ∈ Finset.Icc 1 (N / L), T.Reaches (prefixTarget F N L r c s)) ∧
        T.length = N ^ 3 := by
  classical
  obtain ⟨H, hHle, hH⟩ := (spiderGraph_connected (N := N) (N / L)).exists_isTree_le
  let T : SteinerTree F (Fin N × Fin N) (Fin N × Fin N) :=
    { V := Option (Fin N × Fin N × Fin (N / L)), G := H, isTree := hH, label := spiderLabel F N L (N / L) }
  have h0 : T.Reaches 0 := ⟨none, rfl⟩
  have hT : ∀ r c : Fin N, ∀ s ∈ Finset.Icc 1 (N / L), T.Reaches (prefixTarget F N L r c s) := by
    intro r c s hs
    rw [Finset.mem_Icc] at hs
    refine ⟨some (r, c, ⟨s - 1, by omega⟩), ?_⟩
    change prefixTarget F N L r c (s - 1 + 1) = _
    rw [Nat.sub_add_cancel hs.1]
  refine ⟨T, h0, hT, le_antisymm ?_ (conjectureA_lower hL hLN T h0 hT)⟩
  have hcard := hH.card_edgeFinset
  rw [Fintype.card_option, Fintype.card_prod, Fintype.card_prod, Fintype.card_fin,
    Fintype.card_fin] at hcard
  have hE : H.edgeFinset.card = N * (N * (N / L)) := Nat.add_right_cancel hcard
  calc T.length ≤ ∑ _e ∈ H.edgeFinset, L :=
        Finset.sum_le_sum fun e he => spider_edgeLength_le hHle (SimpleGraph.mem_edgeFinset.1 he)
    _ = N ^ 3 := by
        rw [Finset.sum_const, smul_eq_mul, hE, mul_assoc, mul_assoc, Nat.div_mul_cancel hLN]
        ring

/-- **Conjecture A, proved**: for `N × N × N` matrix multiplication and every chunk `L ∣ N`, every Steiner tree in
the rank metric spanning `0` and every prefix target has length at least `N³`, and one has length `N³`. -/
theorem conjectureA : ConjectureA F := fun _ _ hL hLN =>
  ⟨fun T h0 hT => conjectureA_lower hL hLN T h0 hT, conjectureA_upper hL hLN⟩

end Spider

end Pouw.SteinerRigidity
