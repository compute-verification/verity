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

end Pouw.SteinerRigidity
