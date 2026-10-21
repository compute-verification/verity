import Models.Nci.Guarantees.CoverWidth
import Proofs.Nci.Cover.Width

/-!
# The cover theorem: the width bound from the wiring

`WidthOfCut`: a block handed the set `X` of gates of `A j`, fewer than `f`, is handed no gate of `A j \ X`, so it computes
every gate that reaches the product's activation inside `A j \ X` (`BackwardClosure`, from the activation gate, which
the block holds since a gate it computes reads it). The cut width gives upstream products over `D` parameters whose
output gates do, and each of their gates reaches its output inside the product (`reachesIn_output`), within `A j \ X`,
so the block computes them whole and `UpdateHardness` charges `b D > k` bits. `CutOfDisjointUpstreams` is pigeonhole: fewer than `f` gates miss one of
`f` disjoint parts (`exists_disjoint_of_card_lt`). `cutWidth_le_reads` is the cap: the cut width is at most the number of
activation gates the product reads, since a set holding all of them leaves nothing connected.
-/

namespace Nci.Cover.SecurityProofs

theorem reachesIn_mono {C : Circuit} {R R' : Finset (Fin C.N)} (h : R ⊆ R') {x y : Fin C.N}
    (hxy : C.ReachesIn R x y) : C.ReachesIn R' x y := by
  obtain ⟨π, hπ, hπR⟩ := hxy
  exact ⟨π, hπ, fun v hv => h (hπR v hv)⟩

/-- Fewer than `f` elements miss one of `f` pairwise disjoint sets. -/
theorem exists_disjoint_of_card_lt {α : Type*} {f : ℕ} {B : Fin f → Finset α}
    (hB : ∀ i i', i ≠ i' → Disjoint (B i) (B i')) {X : Finset α} (hX : X.card < f) : ∃ i, Disjoint (B i) X := by
  by_contra h
  have hmeet : ∀ i, ∃ x ∈ B i, x ∈ X := fun i => Finset.not_disjoint_iff.1 fun hd => h ⟨i, hd⟩
  choose x hxB hxX using hmeet
  have hinj : Set.InjOn x (Finset.univ : Finset (Fin f)) := by
    intro i _ i' _ hii'
    by_contra hne
    exact Finset.disjoint_left.1 (hB i i' hne) (hxB i) (hii' ▸ hxB i')
  have hle := Finset.card_le_card_of_injOn x (fun i _ => hxX i) hinj
  rw [Finset.card_univ, Fintype.card_fin] at hle
  omega

theorem WidthOfCut : Guarantees.WidthOfCut := by
  intro C J L _ M A s k D f b hb hH hW hk I hI p hp hpI
  by_contra hlt
  push Not at hlt
  obtain ⟨U, hUP, hDU, hreach⟩ := hW p hp (C.inputs I ∩ A p.1) hlt
  have hR : C.inputs I ∩ (A p.1 \ (C.inputs I ∩ A p.1)) = ∅ := by
    ext y
    simp only [Finset.mem_inter, Finset.mem_sdiff, Finset.notMem_empty, iff_false]
    tauto
  have hUc : ∀ q ∈ U, M.m q ⊆ C.computed I := by
    intro q hq x hx
    obtain ⟨hqA, g, hg, a, ha, hoa⟩ := hreach q hq
    have hoI := BackwardClosure C I _ a (M.o q) (pred_subset_of_mem_computed (hpI hg) ha) hR hoa
    exact BackwardClosure C I _ (M.o q) x (computed_subset C I hoI) hR
      (reachesIn_mono hqA (reachesIn_output C (M.m q) (M.o q) (M.m_outputs q (hUP hq)) x hx))
  have hpay := hH I hI.1.le U hUP hUc
  have hcap : ((C.inputs I \ C.θ).card : ℝ) ≤ k := by exact_mod_cast hI.2
  have hD : (D : ℝ) ≤ (U.image Prod.snd).card := by exact_mod_cast hDU
  have := mul_le_mul_of_nonneg_left hD hb.le
  linarith

theorem CutOfDisjointUpstreams : Guarantees.CutOfDisjointUpstreams := by
  intro C J L _ M A D f hT p hp X hX
  obtain ⟨B, hBdisj, hB⟩ := hT p hp
  obtain ⟨i, hi⟩ := exists_disjoint_of_card_lt hBdisj hX
  obtain ⟨hBA, U, hUP, hDU, g, hg, a, ha, hreach⟩ := hB i
  have hBX : B i ⊆ A p.1 \ X := fun v hv => Finset.mem_sdiff.2 ⟨hBA hv, Finset.disjoint_left.1 hi hv⟩
  exact ⟨U, hUP, hDU, fun q hq => ⟨(hreach q hq).1.trans hBX, g, hg, a, ha, reachesIn_mono hBX (hreach q hq).2⟩⟩

/-- **The cap**: at a positive depth, the cut width is at most the number of gates of `A j` that a counted product of
use `j` reads. -/
theorem cutWidth_le_reads {C : Circuit} {J L : Type} [DecidableEq L] (M : MulBlocks C J L)
    (A : J → Finset (Fin C.N)) {D f : ℕ} (hW : M.CutWidth A D f) (hD : 0 < D) {p : J × L} (hp : p ∈ M.P) :
    f ≤ ((M.m p).biUnion C.pred ∩ A p.1).card := by
  by_contra hlt
  push Not at hlt
  obtain ⟨U, hUP, hDU, hreach⟩ := hW p hp _ hlt
  obtain ⟨l, hl⟩ : (U.image Prod.snd).Nonempty := Finset.card_pos.1 (lt_of_lt_of_le hD hDU)
  obtain ⟨q, hq, -⟩ := Finset.mem_image.1 hl
  obtain ⟨-, g, hg, a, ha, π, ⟨-, hlast, -⟩, hπ⟩ := hreach q hq
  obtain ⟨haA, haX⟩ := Finset.mem_sdiff.1 (hπ a (List.mem_of_getLast? hlast))
  exact haX (Finset.mem_inter.2 ⟨Finset.mem_biUnion.2 ⟨g, hg, ha⟩, haA⟩)

theorem CutWidthIntensity : Guarantees.CutWidthIntensity := by
  intro C J L _ _ M A s k D D₀ f b hb hCH hW hU hk I hI
  refine finished_le_of_charge M A (c := f) hb (Nat.cast_nonneg f) hCH hU hI fun j hj => ?_
  obtain ⟨p, hp, rfl⟩ := Finset.mem_image.1 hj
  obtain ⟨hpP, hpI⟩ := Finset.mem_filter.1 hp
  exact_mod_cast WidthOfCut C M A s k D f b hb (updateHardness_of_conditional M A hCH) hW hk I hI p hpP hpI

theorem CutWidthCoverBound : Guarantees.CutWidthCoverBound := by
  intro C J L _ _ M A s k D D₀ f b 𝓘 hb hCH hW hU hk h𝓘 hcov
  exact cover_sum M hcov (by positivity) fun I hI =>
    CutWidthIntensity C M A s k D D₀ f b hb hCH hW hU hk I (h𝓘 I hI)

end Nci.Cover.SecurityProofs
