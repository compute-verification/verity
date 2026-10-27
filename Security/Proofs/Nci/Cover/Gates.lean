import Proofs.Nci.Cover.Paths
import Definitions.Nci.Assumptions.CoverAdditive

/-!
# The cover theorem's vocabulary on a declaration's gates

The covering bound (`Nci.Covering`) is stated over a declaration's gates; the cover theorem's tiers over a bare
wiring (`Nci.Cover.Circuit`), which also reads a declaration bit by bit (`Circuit.ofDecl`). On the gates they agree.
`ofGates d` is the gate view: the gates of `d`, their wires and its registered parameters.

* A subcircuit's inputs hold every gate that the gate view hands its gates (`inputs_subset_ins`), so it computes at most
  what the gate view says its gates compute (`inner_subset_computed`); any set of gates, with the inputs the gate view
  hands it, is a subcircuit computing exactly that (`toSubcircuit`).
* So the gate view's inference blocks are `Nci.Covering`'s (`inferenceBlock_iff`), an `(s, k)`-covering by `n` blocks
  gives a cover of the gate view by at most `n` inference blocks (`cover_of_covering`), and a cover gives a covering by
  as many blocks (`covering_of_cover`): the cover theorem's bounds count the blocks of `Nci.Covering`'s coverings.
* `Nci.Covering`'s multiplication blocks are products of the gate view, a token's use of a weight (`mulBlocksOf`), the
  reach condition giving each its one output gate (`outputs_eq_of_reach`). On them `UpdateHardness` is `ParameterHardAt`
  (`updateHardness_iff`), and `ConditionalParameterHardAt` implies `ConditionalUpdateHardness` with the token
  computations as the regions (`conditionalUpdateHardness_of_conditional`).
-/

namespace Nci.Cover.SecurityProofs

open Finset

/-- **The gate view** of a declaration: its gates, their wires and its registered parameters. -/
abbrev ofGates (d : Verity.Declaration) : Circuit where
  N := d.D.N
  pred := d.D.reads
  pred_lt := d.D.reads_lt
  θ := d.θ
  θ_input g hg := d.D.input_reads g (d.θ_input g hg)

namespace Gates

variable {d : Verity.Declaration}

theorem mem_sources {g : Fin d.D.N} : g ∈ Nci.Covering.sources d.D ↔ d.D.reads g = ∅ := by
  simp [Nci.Covering.sources]

/-- A subcircuit's inputs hold every gate that the gate view hands its gates. -/
theorem inputs_subset_ins (S : Nci.Covering.Subcircuit d.D) : (ofGates d).inputs S.gates ⊆ S.ins := by
  intro g hg
  obtain ⟨hgS, hg'⟩ := mem_inputs.1 hg
  by_contra hgi
  rcases hg' with h | h
  · exact hgi (S.sources_ins g hgS (mem_sources.2 h))
  · exact h (S.closed g hgS hgi)

/-- A subcircuit computes at most what the gate view says its gates compute. -/
theorem inner_subset_computed (S : Nci.Covering.Subcircuit d.D) : S.inner ⊆ (ofGates d).computed S.gates :=
  fun _ hg => mem_computed.2 ⟨(mem_sdiff.1 hg).1, fun h => (mem_sdiff.1 hg).2 (inputs_subset_ins S h)⟩

/-- **A set of gates as a subcircuit**, with the inputs the gate view hands it. -/
def toSubcircuit (d : Verity.Declaration) (I : Finset (Fin d.D.N)) : Nci.Covering.Subcircuit d.D where
  gates := I
  ins := (ofGates d).inputs I
  ins_sub := filter_subset _ _
  closed g hg hgi := by
    by_contra h
    exact hgi (mem_inputs.2 ⟨hg, Or.inr h⟩)
  sources_ins g hg hs := mem_inputs.2 ⟨hg, Or.inl (mem_sources.1 hs)⟩

theorem toSubcircuit_inner (I : Finset (Fin d.D.N)) : (toSubcircuit d I).inner = (ofGates d).computed I := rfl

/-- On a declaration's gates the inference blocks are `Nci.Covering`'s. -/
theorem inferenceBlock_iff (s k : ℕ) (I : Finset (Fin d.D.N)) :
    (ofGates d).InferenceBlock s k I ↔ Nci.Covering.InferenceBlock d s k (toSubcircuit d I) := Iff.rfl

/-- **`Nci.Covering`'s coverings are covers of the gate view**: an `(s, k)`-covering of `d` by `n` blocks gives a
cover of the gate view by at most `n` inference blocks, the blocks' gates. -/
theorem cover_of_covering {s k n : ℕ} {I : Fin n → Nci.Covering.Subcircuit d.D} (h : Nci.Covering.Covers d s k I) :
    ∃ 𝓘 : Finset (Finset (Fin d.D.N)), 𝓘.card ≤ n ∧ (∀ J ∈ 𝓘, (ofGates d).InferenceBlock s k J) ∧
      (ofGates d).Covers 𝓘 := by
  classical
  refine ⟨univ.image fun u => (I u).gates, card_image_le.trans (by simp), ?_, ?_⟩
  · intro J hJ
    obtain ⟨u, -, rfl⟩ := mem_image.1 hJ
    obtain ⟨hs, hk⟩ := h.1 u
    exact ⟨hs, (card_le_card (sdiff_subset_sdiff (inputs_subset_ins (I u)) subset_rfl)).trans hk⟩
  · intro g hg
    obtain ⟨u, hu⟩ := h.2 g fun hs => hg.ne_empty (mem_sources.1 hs)
    exact ⟨(I u).gates, mem_image.2 ⟨u, mem_univ _, rfl⟩, inner_subset_computed (I u) hu⟩

/-- **Covers of the gate view are `Nci.Covering`'s coverings**: a cover of the gate view by inference blocks, each with
the inputs the gate view hands it, is an `(s, k)`-covering of `d` by as many blocks. -/
theorem covering_of_cover {s k : ℕ} {𝓘 : Finset (Finset (Fin d.D.N))}
    (hI : ∀ J ∈ 𝓘, (ofGates d).InferenceBlock s k J) (hcov : (ofGates d).Covers 𝓘) :
    Nci.Covering.Covers d s k fun u : Fin 𝓘.toList.length => toSubcircuit d (𝓘.toList.get u) := by
  refine ⟨fun u => (inferenceBlock_iff s k _).1 (hI _ (mem_toList.1 (List.mem_iff_get.2 ⟨u, rfl⟩))), fun g hg => ?_⟩
  obtain ⟨J, hJ, hgJ⟩ := hcov g (nonempty_iff_ne_empty.2 fun h => hg (mem_sources.2 h))
  obtain ⟨u, hu⟩ := List.mem_iff_get.1 (mem_toList.2 hJ)
  refine ⟨u, ?_⟩
  show g ∈ (toSubcircuit d (𝓘.toList.get u)).inner
  rw [hu]
  exact hgJ

/-- A path inside `R` starts in `R`. -/
theorem feedsWithin_mem_left {R : Finset (Fin d.D.N)} {g t : Fin d.D.N} (h : Nci.Covering.FeedsWithin d.D R g t) :
    g ∈ R := by
  cases h with
  | refl ht => exact ht
  | step hg _ _ => exact hg

/-- Wires run forward, so a path's first gate is at most its last. -/
theorem feedsWithin_le {R : Finset (Fin d.D.N)} {g t : Fin d.D.N} (h : Nci.Covering.FeedsWithin d.D R g t) : g ≤ t := by
  induction h with
  | refl _ => exact le_rfl
  | step _ hgh _ ih => exact (d.D.reads_lt _ _ hgh).le.trans ih

/-- A set of gates each of which reaches `o ∈ S` inside `S` has `o` as its one output gate in the gate view. -/
theorem outputs_eq_of_reach {S : Finset (Fin d.D.N)} {o : Fin d.D.N} (ho : o ∈ S)
    (hreach : ∀ g ∈ S, Nci.Covering.FeedsWithin d.D S g o) : (ofGates d).outputs S = {o} := by
  ext g
  rw [mem_outputs, mem_singleton]
  constructor
  · rintro ⟨hg, hno⟩
    cases hreach g hg with
    | refl _ => rfl
    | step _ hgh hh => exact absurd hgh (hno _ (feedsWithin_mem_left hh))
  · rintro rfl
    exact ⟨ho, fun h hh hoh => absurd ((d.D.reads_lt h g hoh).trans_le (feedsWithin_le (hreach h hh))) (lt_irrefl _)⟩

theorem mem_swap {B : Nci.Covering.MulBlocks d} {p : Fin B.nt × Fin B.nw} :
    p ∈ B.P.image Prod.swap ↔ p.swap ∈ B.P := by
  constructor
  · intro hp
    obtain ⟨q, hq, rfl⟩ := mem_image.1 hp
    simpa using hq
  · intro hp
    exact mem_image.2 ⟨p.swap, hp, Prod.swap_swap p⟩

/-- **`Nci.Covering`'s multiplication blocks as products of the gate view**: the multiplication `m_{i,j}` is the
product of the use `j`, its token, and the updated parameter `i`, its weight, all of them counted. Its output gate is
the one gate of `m_{i,j}` that no gate of it reads (`outputs_eq_of_reach`). -/
def mulBlocksOf (B : Nci.Covering.MulBlocks d) : MulBlocks (ofGates d) (Fin B.nt) (Fin B.nw) where
  Pall := B.P.image Prod.swap
  P := B.P.image Prod.swap
  P_subset := subset_rfl
  m p := B.m p.swap
  m_nonempty p hp := ⟨B.o p.swap, B.o_mem _ (mem_swap.1 hp)⟩
  m_computed p hp g hg :=
    nonempty_iff_ne_empty.2 fun h => disjoint_left.1 (B.m_computed _ (mem_swap.1 hp)) hg (mem_sources.2 h)
  m_disjoint p hp q hq hpq :=
    B.m_disjoint _ (mem_swap.1 hp) _ (mem_swap.1 hq) fun h => hpq (by simpa using congrArg Prod.swap h)
  o p := B.o p.swap
  m_outputs p hp := outputs_eq_of_reach (B.o_mem _ (mem_swap.1 hp)) (B.reach _ (mem_swap.1 hp))

theorem card_P_mulBlocksOf (B : Nci.Covering.MulBlocks d) : (mulBlocksOf B).P.card = B.P.card :=
  card_image_of_injective _ Prod.swap_injective

open Classical in
theorem weightsCompleted_eq (B : Nci.Covering.MulBlocks d) (S : Nci.Covering.Subcircuit d.D) :
    ((B.P.filter fun p => B.Completes S p).image Prod.swap).image Prod.snd = B.weightsCompleted S := by
  ext i
  constructor
  · intro hi
    obtain ⟨q, hq, rfl⟩ := mem_image.1 hi
    obtain ⟨p, hp, rfl⟩ := mem_image.1 hq
    obtain ⟨hpP, hpC⟩ := mem_filter.1 hp
    exact mem_image.2 ⟨p, mem_filter.2 ⟨hpP, hpC⟩, rfl⟩
  · intro hi
    obtain ⟨p, hp, rfl⟩ := mem_image.1 hi
    obtain ⟨hpP, hpC⟩ := mem_filter.1 hp
    exact mem_image.2 ⟨p.swap, mem_image.2 ⟨p, mem_filter.2 ⟨hpP, hpC⟩, rfl⟩, rfl⟩

/-- The weights of products of the gate view that a set computes whole are among the weights its subcircuit
completes. -/
theorem image_snd_subset (B : Nci.Covering.MulBlocks d) {S : Finset (Fin d.D.N)}
    {B' : Finset (Fin B.nt × Fin B.nw)} (hB' : B' ⊆ (mulBlocksOf B).Pall)
    (hB'S : ∀ p ∈ B', (mulBlocksOf B).m p ⊆ (ofGates d).computed S) :
    B'.image Prod.snd ⊆ B.weightsCompleted (toSubcircuit d S) := by
  classical
  intro i hi
  obtain ⟨p, hp, rfl⟩ := mem_image.1 hi
  exact mem_image.2 ⟨p.swap, mem_filter.2 ⟨mem_swap.1 (hB' hp), hB'S p hp⟩, rfl⟩

/-- **On a declaration's gates, `UpdateHardness` is the covering bound's parameter-hardness** at the same cost. -/
theorem updateHardness_iff (B : Nci.Covering.MulBlocks d) (s : ℕ) (c : ℝ) :
    Nci.Assumptions.ParameterHardAt d B s c ↔ Assumptions.UpdateHardness (mulBlocksOf B) s c := by
  classical
  constructor
  · intro h S hS B' hB' hB'S
    rcases le_or_gt 0 c with hc | hc
    · exact (mul_le_mul_of_nonneg_left (Nat.cast_le.2 (card_le_card (image_snd_subset B hB' hB'S))) hc).trans
        (h (toSubcircuit d S) hS)
    · exact (mul_nonpos_of_nonpos_of_nonneg hc.le (Nat.cast_nonneg _)).trans (Nat.cast_nonneg _)
  · intro h S hS
    have h1 := h S.gates hS ((B.P.filter fun p => B.Completes S p).image Prod.swap)
      (fun p hp => by
        obtain ⟨q, hq, rfl⟩ := mem_image.1 hp
        exact mem_image.2 ⟨q, (mem_filter.1 hq).1, rfl⟩)
      (fun p hp => by
        obtain ⟨q, hq, rfl⟩ := mem_image.1 hp
        exact (mem_filter.1 hq).2.trans (inner_subset_computed S))
    rw [weightsCompleted_eq] at h1
    exact h1.trans (Nat.cast_le.2 (card_le_card (sdiff_subset_sdiff (inputs_subset_ins S) subset_rfl)))

/-- **On a declaration's gates, conditional parameter-hardness gives `ConditionalUpdateHardness`**, with the token
computations as the regions: its inputs outside every token computation are outside those of the counted uses. -/
theorem conditionalUpdateHardness_of_conditional (B : Nci.Covering.MulBlocks d) (s : ℕ) (c : ℝ)
    (h : Assumptions.ConditionalParameterHardAt d B s c) :
    Assumptions.ConditionalUpdateHardness (mulBlocksOf B) B.T s c := by
  classical
  intro S hS B' hB' hB'S
  have hT : (toSubcircuit d S).ins \ (d.θ ∪ univ.biUnion B.T) ⊆
      (ofGates d).inputs S \ ((ofGates d).θ ∪ (mulBlocksOf B).uses.biUnion B.T) :=
    sdiff_subset_sdiff subset_rfl (union_subset_union subset_rfl (biUnion_subset_biUnion_of_subset_left _ (subset_univ _)))
  rcases le_or_gt 0 c with hc | hc
  · exact ((mul_le_mul_of_nonneg_left (Nat.cast_le.2 (card_le_card (image_snd_subset B hB' hB'S))) hc).trans
      (h (toSubcircuit d S) hS)).trans (Nat.cast_le.2 (card_le_card hT))
  · exact (mul_nonpos_of_nonpos_of_nonneg hc.le (Nat.cast_nonneg _)).trans (Nat.cast_nonneg _)

end Gates

end Nci.Cover.SecurityProofs
