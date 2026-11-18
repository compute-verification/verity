module

public import Definitions.Core.Levels
public import Definitions.Core.Partition

/-!
# Levels at a circuit

Partitions of a circuit `C`, each refining the one above it, each with a sampling rule over its units inside each unit of
the partition above (`Partition.Chain`), are levels over `C`'s computed gates (`Levels.ofPartitions`). Input gates are in
no unit, so the levels partition the computed gates, and a level's blocks are its partition's nonempty units
(`Partition.mem_blocks`).
-/

@[expose] public section

namespace Verity

open Finset

/-- A circuit's computed gates, as a type. -/
abbrev Circuit.Computed (C : Circuit) : Type := {g : Fin C.N // C.isInput g = false}

namespace Partition

variable {C : Circuit} {n : ℕ}

/-- The partition with one unit, holding every computed gate. -/
def whole (C : Circuit) : Partition C 1 where
  unit g := if C.isInput g = true then none else some 0
  input_iff g := by split_ifs with h <;> simp [h]

/-- Unit `u`'s gates, as computed gates. -/
def block (P : Partition C n) (u : Fin n) : Finset C.Computed :=
  univ.filter fun g => P.unit g.1 = some u

/-- The nonempty units' blocks, a finpartition of the computed gates. -/
def blocks (P : Partition C n) : Finpartition (univ : Finset C.Computed) :=
  Finpartition.ofErase (univ.image P.block)
    (Finset.supIndep_iff_pairwiseDisjoint.2 fun b hb b' hb' hne => by
      obtain ⟨u, -, rfl⟩ := Finset.mem_image.1 hb
      obtain ⟨v, -, rfl⟩ := Finset.mem_image.1 hb'
      refine Finset.disjoint_left.2 fun g hg hg' => hne ?_
      rw [Option.some_inj.1 ((Finset.mem_filter.1 hg).2.symm.trans (Finset.mem_filter.1 hg').2)])
    (Finset.eq_univ_of_forall fun g => by
      obtain ⟨u, hu⟩ := Option.ne_none_iff_exists'.1 fun h =>
        Bool.false_ne_true (g.2.symm.trans ((P.input_iff g.1).1 h))
      exact Finset.mem_sup.2 ⟨P.block u, Finset.mem_image_of_mem _ (mem_univ u),
        Finset.mem_filter.2 ⟨mem_univ _, hu⟩⟩)

theorem mem_blocks {P : Partition C n} {b : Finset C.Computed} :
    b ∈ P.blocks.parts ↔ b.Nonempty ∧ ∃ u, P.block u = b := by
  simp [blocks, Finpartition.ofErase_parts, Finset.nonempty_iff_ne_empty]

theorem Refines.block_subset {nf nc : ℕ} {Pf : Partition C nf} {Pc : Partition C nc} (R : Pf.Refines Pc)
    (u : Fin nf) : Pf.block u ⊆ Pc.block (R.parent u) := fun g hg => by
  simp only [block, mem_filter, mem_univ, true_and] at hg ⊢
  rw [R.sub, hg]
  rfl

open Classical in
/-- At the block `X` of `Pc`, the rule of the unit whose block `X` is draws units of `Pf`; the sample is the blocks of
those inside it. -/
noncomputable def Refines.design {nf nc : ℕ} {Pf : Partition C nf} {Pc : Partition C nc} (R : Pf.Refines Pc)
    (rule : Fin nc → SamplingDesign (Fin nf)) (X : Finset C.Computed) : SamplingDesign (Finset C.Computed) :=
  if h : ∃ p, Pc.block p = X then
    { Ω := (rule h.choose).Ω
      draw := fun ω => ((((rule h.choose).draw ω).filter fun u => R.parent u = h.choose).image Pf.block).erase ∅ }
  else SamplingDesign.const ∅

theorem Refines.design_inside {nf nc : ℕ} {Pf : Partition C nf} {Pc : Partition C nc} (R : Pf.Refines Pc)
    (rule : Fin nc → SamplingDesign (Fin nf)) (X : Finset C.Computed) :
    ∀ ω, ∀ b ∈ (R.design rule X).draw ω, b ∈ Pf.blocks.parts ∧ b ⊆ X := by
  unfold Refines.design
  by_cases h : ∃ p, Pc.block p = X
  · rw [dite_cond_eq_true (eq_true h)]
    intro ω b hb
    obtain ⟨hb0, hb⟩ := Finset.mem_erase.1 hb
    obtain ⟨u, hu, rfl⟩ := Finset.mem_image.1 hb
    refine ⟨mem_blocks.2 ⟨Finset.nonempty_iff_ne_empty.2 hb0, u, rfl⟩, ?_⟩
    rw [← h.choose_spec, ← (Finset.mem_filter.1 hu).2]
    exact R.block_subset u
  · rw [dite_cond_eq_false (eq_false h)]
    intro ω b hb
    simp [SamplingDesign.const] at hb

/-- Partitions nested below `Pc`, coarsest first, each with a sampling rule over its units inside each unit of the one
above. -/
inductive Chain (C : Circuit) : {nc : ℕ} → Partition C nc → Type 1
  | nil {nc : ℕ} {Pc : Partition C nc} : Chain C Pc
  | cons {nc nf : ℕ} {Pc : Partition C nc} (Pf : Partition C nf) (R : Pf.Refines Pc)
      (rule : Fin nc → SamplingDesign (Fin nf)) (rest : Chain C Pf) : Chain C Pc

/-- The chain's levels. -/
noncomputable def Chain.levels : {nc : ℕ} → {Pc : Partition C nc} → Chain C Pc → List (Level C.Computed)
  | _, _, .nil => []
  | _, _, .cons Pf R rule rest => ⟨Pf.blocks, R.design rule⟩ :: rest.levels

theorem Chain.nestedBelow {nc : ℕ} {Pc : Partition C nc} (ch : Chain C Pc) (P : Finset (Finset C.Computed))
    (hP : ∀ u, (Pc.block u).Nonempty → ∃ X ∈ P, Pc.block u ⊆ X) : Level.NestedBelow P ch.levels := by
  induction ch generalizing P with
  | nil => simp [Chain.levels, Level.NestedBelow]
  | cons Pf R rule rest ih =>
    simp only [Chain.levels, Level.NestedBelow]
    refine ⟨fun b hb => ?_, ih _ fun u hu => ⟨_, mem_blocks.2 ⟨hu, u, rfl⟩, subset_rfl⟩⟩
    obtain ⟨hne, u, rfl⟩ := mem_blocks.1 hb
    obtain ⟨X, hX, hsub⟩ := hP (R.parent u) (hne.mono (R.block_subset u))
    exact ⟨X, hX, (R.block_subset u).trans hsub⟩

theorem Chain.insideBelow {nc : ℕ} {Pc : Partition C nc} (ch : Chain C Pc) (P : Finset (Finset C.Computed)) :
    Level.InsideBelow P ch.levels := by
  induction ch generalizing P with
  | nil => simp [Chain.levels, Level.InsideBelow]
  | cons Pf R rule rest ih =>
    simp only [Chain.levels, Level.InsideBelow]
    exact ⟨fun X _ ω b hb => R.design_inside rule X ω b hb, ih _⟩

end Partition

/-- **The levels of a circuit's partitions**, the coarsest refining the whole circuit. -/
noncomputable def Levels.ofPartitions {C : Circuit} (ch : Partition.Chain C (Partition.whole C)) :
    Levels C.Computed where
  levels := ch.levels
  nested := ch.nestedBelow _ fun _ _ => ⟨univ, mem_singleton_self _, subset_univ _⟩
  inside := ch.insideBelow _

end Verity
