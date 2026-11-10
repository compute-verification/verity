module

public import Proofs.Nci.Untied
public import Proofs.Accounting.Actual.Windows

/-!
# Instances against the untied-words statements

A scratch file of instances, never merged: what `Proofs.Nci.Untied`'s statements say on small circuits, and what they
leave out.

* **Two readings.** `RegisteredCircuit.untie d E` changes what a set of gates reads only at `θ ∩ E`
  (`unregistered_untie_of_disjoint`), so at a word of computed gates it is `d` (`unregistered_untie_of_computed`,
  `novelReads_untie_of_computed`). On `TA` the word `{y}` is computed inside the one block: untied at it the block reads
  0 gates outside `θ` (`toyA_untie`, and `toyA_untie_file` through `readsAtMost_untie` at `w = 0`), and with it as an
  input the block reads 1 (`toyA_asInputs`, `toyA_asInputs_not`). No block reads the word on `TA`
  (`toyA_wordsRead`); the block reads it on the circuit with it as an input (`toyA_wordsRead_asInputs`).
* **Not vacuous.** With the word a registered input, the hypotheses hold and the bounds are met with equality
  (`toyB_*`). Without `OneReader` the total fails (`toyC_*`).
* **With the words as inputs**, in bits (`novelReads_asInputs_le`) and all blocks together
  (`sum_card_unregistered_asInputs_le`), which the file states only for `untie`.
* **The output values' term over a set `G` of blocks** is at most the output wires in the blocks of `G`
  (`sum_card_fedBy_on_le`), which are at most `O` (`sum_card_inter_on_le_O`).
* **Each block at its own cap**, with the words as inputs: `k` and `b` for each word it reads, with no threshold
  (`card_unregistered_asInputs_block_le`), and all blocks together without `OneReader`
  (`sum_card_unregistered_asInputs_pairs_le`). One block's count at its own cap, with the outputs it holds a wire of
  in place of `O` (`counts_block`).
* **A block newly reads only the words it held.** With computed gates `T` as inputs, a gate of `T` in another block
  was read already, so a block reads at most what it read and the gates of `T` it held
  (`unregistered_asInputs_subset_held`, `card_unregistered_asInputs_held_le`), and all blocks together at most
  `m k + |T|` with no condition on who reads a word (`sum_card_unregistered_asInputs_held_le`). On `TD` a computed
  word has two readers and the reads are the same tied and with it as an input (`toyD_*`).
* **Two instances of the count over a set of blocks.** A MAC with a gate in each of two blocks is an entry of the
  block that reads across (`toyS_*`). Of the two output wires of `Nci.Cut.Toy`, one is in block 0 (`toyO_*`).
* **(H) carries over.** `κ` from the unregistered inputs only grows when gates leave `θ` or become inputs
  (`kappaWords_le_untie`, `kappaWords_le_asInputs`), so (H) at `d` gives (H) at `d.untie E` and at `d.asInputs T`
  (`hardness_untie`, `hardness_asInputs`), `Π` at the untied circuit is at most `Π` at `d` (`piMax_untie_le`), and
  `HE` at the untied circuit takes (H) at `d` (`he_untie`).
* **`Circuit.asInputs C T` and `Accounting.Placed.cut A Tᶜ`** have the same gates, widths, inputs, reads and
  operations (`cut_compl_isInput`, `cut_compl_reads`, `cut_compl_op`, `cut_compl_into`), and differ in their outputs.
-/

@[expose] public section

namespace Nci.Untied.Attack

open Verity Finset Nci.Cut

/-! ### Untying gates outside `θ` changes nothing -/

/-- Untied at gates none of which is registered, a set of gates reads outside `θ` what it read. -/
theorem unregistered_untie_of_disjoint (d : RegisteredCircuit) (E R : Finset (Fin d.D.N)) (hE : Disjoint d.θ E) :
    (d.untie E).unregistered R = d.unregistered R := by
  rw [d.unregistered_untie E R]
  have h : d.θ ∩ E = ∅ := disjoint_iff_inter_eq_empty.1 hE
  rw [h, inter_empty, union_empty]

/-- Untied at computed gates, a set of gates reads outside `θ` what it read. -/
theorem unregistered_untie_of_computed (d : RegisteredCircuit) (E R : Finset (Fin d.D.N))
    (hE : ∀ g ∈ E, d.D.isInput g = false) : (d.untie E).unregistered R = d.unregistered R :=
  unregistered_untie_of_disjoint d E R (disjoint_left.2 fun g hθ hg => by
    have h := hE g hg
    rw [d.θ_input g hθ] at h
    cases h)

/-- In bits: untied at computed gates, a set of gates reads the fresh input bits it read. -/
theorem novelReads_untie_of_computed (d : RegisteredCircuit) (E R : Finset (Fin d.D.N))
    (hE : ∀ g ∈ E, d.D.isInput g = false) : novelReads (d.untie E) R = novelReads d R :=
  congrArg d.D.width (unregistered_untie_of_computed d E R hE)

/-! ### Toy A: a word computed inside the block that reads it -/

/-- The gates each gate of the toy reads. -/
def readsA : Fin 4 → Finset (Fin 4) := ![∅, ∅, {0, 1}, {2}]

/-- Four one-bit gates: inputs `w` (gate 0) and `x` (gate 1), `y = w x` (gate 2), and `z` (gate 3), which reads `y`. -/
abbrev TA : Circuit where
  N := 4
  w := fun _ => 1
  isInput := fun g => decide (g.val < 2)
  reads := readsA
  reads_lt := by decide
  input_reads := by decide
  op := fun _ _ => 0
  op_congr := fun _ _ _ _ => rfl
  outputs := {3}

/-- The toy with `w` and `x` registered. -/
abbrev dA : RegisteredCircuit := .ofCircuit TA {0, 1} (by decide)

/-- One block holding `y` and `z`. -/
def PA : Partition TA 1 where
  unit := fun g => if g.val < 2 then none else some 0
  input_iff := by decide

/-- A block for `y` and a block for `z`. -/
def PA2 : Partition TA 2 where
  unit := fun g => if g.val < 2 then none else if g.val = 2 then some 0 else some 1
  input_iff := by decide

/-- The word `{y}`. -/
abbrev wordY : Finset (Finset (Fin TA.N)) := {{2}}

/-- The word `{x}`. -/
abbrev wordX : Finset (Finset (Fin TA.N)) := {{1}}

/-- The block reads nothing outside `θ`. -/
theorem toyA_tied : ReadsAtMost dA PA 0 := by
  unfold ReadsAtMost
  decide

/-- On the toy, no block reads the word `{y}`: it lies inside the one block. -/
theorem toyA_wordsRead : (wordsRead TA wordY (PA.gates 0)).card = 0 := by decide

/-- Untied at `{y}`, the block reads what it read: nothing. -/
theorem toyA_untie : ReadsAtMost (dA.untie ({2} : Finset (Fin TA.N))) PA 0 := by
  unfold ReadsAtMost
  decide

/-- The same from `readsAtMost_untie` at `w = 0`. -/
theorem toyA_untie_file : ReadsAtMost (dA.untie (wordY.biUnion id)) PA (0 + 1 * 0) :=
  readsAtMost_untie (k := 0) (b := 1) (w := 0) dA PA wordY (by decide) toyA_tied (by decide)

/-- With `y` as an input outside `θ`, the block reads it: one gate. -/
theorem toyA_asInputs :
    ((dA.asInputs ({2} : Finset (Fin TA.N))).unregistered ((PA.asInputs {2}).gates 0)).card = 1 := by decide

/-- So with `y` as an input the block does not read at most 0 gates outside `θ`. -/
theorem toyA_asInputs_not : ¬ ReadsAtMost (dA.asInputs ({2} : Finset (Fin TA.N))) (PA.asInputs {2}) 0 := by
  unfold ReadsAtMost
  decide

/-- On the circuit with `y` as an input, the block reads the word `{y}`. -/
theorem toyA_wordsRead_asInputs : (wordsRead (TA.asInputs {2}) wordY (PA.gates 0 \ {2})).card = 1 := by decide

/-! ### Toy B: a word that is a registered input -/

/-- The block reads the word `{x}`. -/
theorem toyB_wordsRead : (wordsRead TA wordX (PA.gates 0)).card = 1 := by decide

/-- Untied at `{x}`, the block reads one gate outside `θ`: `readsAtMost_untie` at `k = 0`, `b = 1`, `w = 1` with
equality. -/
theorem toyB_untie : ((dA.untie ({1} : Finset (Fin TA.N))).unregistered (PA.gates 0)).card = 1 := by decide

/-- With a block for each computed gate, the word `{x}` has one reader. -/
theorem toyB_oneReader : OneReader PA2 wordX := by
  unfold OneReader
  decide

/-- Each of the two blocks reads at most one gate outside `θ`. -/
theorem toyB_tied2 : ReadsAtMost dA PA2 1 := by
  unfold ReadsAtMost
  decide

/-- One block reads more than 0 words: `card_readersOver_le` at `w = 0` with equality. -/
theorem toyB_readersOver : (readersOver PA2 wordX 0).card = 1 := by decide

/-- All blocks together, at one block: `sum_card_unregistered_untie_le` with equality, `1 = 1 · 0 + 1 · 1`. -/
theorem toyB_sum : ∑ u, ((dA.untie ({1} : Finset (Fin TA.N))).unregistered (PA.gates u)).card = 1 := by decide

/-! ### Toy C: a word with two readers -/

/-- The gates each gate of the toy reads. -/
def readsC : Fin 3 → Finset (Fin 3) := ![∅, {0}, {0}]

/-- Three one-bit gates: an input `x` (gate 0) and two computed gates that each read it. -/
abbrev TC : Circuit where
  N := 3
  w := fun _ => 1
  isInput := fun g => decide (g.val < 1)
  reads := readsC
  reads_lt := by decide
  input_reads := by decide
  op := fun _ _ => 0
  op_congr := fun _ _ _ _ => rfl
  outputs := {1, 2}

/-- The toy with `x` registered. -/
abbrev dC : RegisteredCircuit := .ofCircuit TC {0} (by decide)

/-- A block for each computed gate. -/
def PC : Partition TC 2 where
  unit := fun g => if g.val < 1 then none else if g.val = 1 then some 0 else some 1
  input_iff := by decide

/-- The word `{x}`. -/
abbrev wordC : Finset (Finset (Fin TC.N)) := {{0}}

theorem toyC_tied : ReadsAtMost dC PC 0 := by
  unfold ReadsAtMost
  decide

/-- Both blocks read the word. -/
theorem toyC_not_oneReader : ¬ OneReader PC wordC := by
  unfold OneReader
  decide

/-- Untied at the word, the blocks read 2 gates outside `θ` between them, more than `m k + b |𝒲| = 2 · 0 + 1 · 1`. -/
theorem toyC_sum : ∑ u, ((dC.untie ({0} : Finset (Fin TC.N))).unregistered (PC.gates u)).card = 2 := by decide

/-! ### In bits, with the words as inputs -/

/-- With the words `𝒲`, each of at most `b` bits, as inputs outside `θ`, what is left of `R` reads at most the fresh
input bits `R` read and `b` bits for each word it now reads. -/
theorem novelReads_asInputs_le (d : RegisteredCircuit) {b : ℕ} (𝒲 : Finset (Finset (Fin d.D.N)))
    (h𝒲 : ∀ W ∈ 𝒲, d.D.width W ≤ b) (R : Finset (Fin d.D.N)) :
    novelReads (d.asInputs (𝒲.biUnion id)) (R \ 𝒲.biUnion id) ≤
      novelReads d R + b * (wordsRead (d.D.asInputs (𝒲.biUnion id)) 𝒲 (R \ 𝒲.biUnion id)).card := by
  show ∑ g ∈ (d.asInputs (𝒲.biUnion id)).unregistered (R \ 𝒲.biUnion id), d.D.w g ≤
    ∑ g ∈ d.unregistered R, d.D.w g + b * (wordsRead (d.D.asInputs (𝒲.biUnion id)) 𝒲 (R \ 𝒲.biUnion id)).card
  calc ∑ g ∈ (d.asInputs (𝒲.biUnion id)).unregistered (R \ 𝒲.biUnion id), d.D.w g
      ≤ ∑ g ∈ d.unregistered R ∪ (d.D.asInputs (𝒲.biUnion id)).into (R \ 𝒲.biUnion id) ∩ 𝒲.biUnion id, d.D.w g :=
        sum_le_sum_of_subset (d.unregistered_asInputs_subset (𝒲.biUnion id) R)
    _ ≤ ∑ g ∈ d.unregistered R, d.D.w g +
          ∑ g ∈ (d.D.asInputs (𝒲.biUnion id)).into (R \ 𝒲.biUnion id) ∩ 𝒲.biUnion id, d.D.w g :=
        Nat.le.intro sum_union_inter
    _ ≤ ∑ g ∈ d.unregistered R, d.D.w g +
          b * (wordsRead (d.D.asInputs (𝒲.biUnion id)) 𝒲 (R \ 𝒲.biUnion id)).card :=
        Nat.add_le_add_left
          (width_into_inter_biUnion_le (d.D.asInputs (𝒲.biUnion id)) 𝒲 h𝒲 (R \ 𝒲.biUnion id)) _

/-- **All blocks together, with the words as inputs.** When each word is read by at most one block on the circuit with
the words as inputs, the blocks read there at most `m k + b |𝒲|` gates outside `θ` between them. -/
theorem sum_card_unregistered_asInputs_le {m : ℕ} (d : RegisteredCircuit) (P : Partition d.D m) {k b : ℕ}
    (𝒲 : Finset (Finset (Fin d.D.N))) (h𝒲 : ∀ W ∈ 𝒲, W.card ≤ b) (hk : ReadsAtMost d P k)
    (hone : OneReader (P.asInputs (𝒲.biUnion id)) 𝒲) :
    ∑ u, ((d.asInputs (𝒲.biUnion id)).unregistered ((P.asInputs (𝒲.biUnion id)).gates u)).card ≤
      m * k + b * 𝒲.card := by
  have hsum : ∑ u : Fin m, (wordsRead (d.D.asInputs (𝒲.biUnion id)) 𝒲 (P.gates u \ 𝒲.biUnion id)).card ≤
      𝒲.card := by
    have h := sum_card_wordsRead_le (P.asInputs (𝒲.biUnion id)) 𝒲 hone univ
    simp only [Partition.asInputs_gates] at h
    exact h
  have hblock : ∀ u, ((d.asInputs (𝒲.biUnion id)).unregistered ((P.asInputs (𝒲.biUnion id)).gates u)).card ≤
      k + b * (wordsRead (d.D.asInputs (𝒲.biUnion id)) 𝒲 (P.gates u \ 𝒲.biUnion id)).card := by
    intro u
    have hg : (P.asInputs (𝒲.biUnion id)).gates u = P.gates u \ 𝒲.biUnion id := P.asInputs_gates (𝒲.biUnion id) u
    calc ((d.asInputs (𝒲.biUnion id)).unregistered ((P.asInputs (𝒲.biUnion id)).gates u)).card
        = ((d.asInputs (𝒲.biUnion id)).unregistered (P.gates u \ 𝒲.biUnion id)).card :=
          congrArg (fun S => ((d.asInputs (𝒲.biUnion id)).unregistered S).card) hg
      _ ≤ (d.unregistered (P.gates u) ∪
            (d.D.asInputs (𝒲.biUnion id)).into (P.gates u \ 𝒲.biUnion id) ∩ 𝒲.biUnion id).card :=
          card_le_card (d.unregistered_asInputs_subset (𝒲.biUnion id) (P.gates u))
      _ ≤ (d.unregistered (P.gates u)).card +
            ((d.D.asInputs (𝒲.biUnion id)).into (P.gates u \ 𝒲.biUnion id) ∩ 𝒲.biUnion id).card :=
          card_union_le _ _
      _ ≤ k + b * (wordsRead (d.D.asInputs (𝒲.biUnion id)) 𝒲 (P.gates u \ 𝒲.biUnion id)).card :=
          Nat.add_le_add (hk u)
            (card_into_inter_biUnion_le (d.D.asInputs (𝒲.biUnion id)) 𝒲 h𝒲 (P.gates u \ 𝒲.biUnion id))
  calc ∑ u, ((d.asInputs (𝒲.biUnion id)).unregistered ((P.asInputs (𝒲.biUnion id)).gates u)).card
      ≤ ∑ u : Fin m, (k + b * (wordsRead (d.D.asInputs (𝒲.biUnion id)) 𝒲 (P.gates u \ 𝒲.biUnion id)).card) :=
        sum_le_sum fun u _ => hblock u
    _ = m * k + b * ∑ u : Fin m, (wordsRead (d.D.asInputs (𝒲.biUnion id)) 𝒲 (P.gates u \ 𝒲.biUnion id)).card := by
        rw [sum_add_distrib, sum_const, card_univ, Fintype.card_fin, smul_eq_mul, mul_sum]
    _ ≤ m * k + b * 𝒲.card := Nat.add_le_add_left (Nat.mul_le_mul_left b hsum) _

/-- **Each block at its own cap, with the words as inputs.** A block that read at most `k` gates outside `θ` reads, with
the words as inputs, at most `k` and `b` for each word it then reads: a cap of its own, with no threshold. -/
theorem card_unregistered_asInputs_block_le {m : ℕ} (d : RegisteredCircuit) (P : Partition d.D m) {k b : ℕ}
    (𝒲 : Finset (Finset (Fin d.D.N))) (h𝒲 : ∀ W ∈ 𝒲, W.card ≤ b) (hk : ReadsAtMost d P k) (u : Fin m) :
    ((d.asInputs (𝒲.biUnion id)).unregistered ((P.asInputs (𝒲.biUnion id)).gates u)).card ≤
      k + b * (wordsRead (d.D.asInputs (𝒲.biUnion id)) 𝒲 (P.gates u \ 𝒲.biUnion id)).card := by
  have hg : (P.asInputs (𝒲.biUnion id)).gates u = P.gates u \ 𝒲.biUnion id := P.asInputs_gates (𝒲.biUnion id) u
  calc ((d.asInputs (𝒲.biUnion id)).unregistered ((P.asInputs (𝒲.biUnion id)).gates u)).card
      = ((d.asInputs (𝒲.biUnion id)).unregistered (P.gates u \ 𝒲.biUnion id)).card :=
        congrArg (fun S => ((d.asInputs (𝒲.biUnion id)).unregistered S).card) hg
    _ ≤ (d.unregistered (P.gates u) ∪
          (d.D.asInputs (𝒲.biUnion id)).into (P.gates u \ 𝒲.biUnion id) ∩ 𝒲.biUnion id).card :=
        card_le_card (d.unregistered_asInputs_subset (𝒲.biUnion id) (P.gates u))
    _ ≤ (d.unregistered (P.gates u)).card +
          ((d.D.asInputs (𝒲.biUnion id)).into (P.gates u \ 𝒲.biUnion id) ∩ 𝒲.biUnion id).card :=
        card_union_le _ _
    _ ≤ k + b * (wordsRead (d.D.asInputs (𝒲.biUnion id)) 𝒲 (P.gates u \ 𝒲.biUnion id)).card :=
        Nat.add_le_add (hk u)
          (card_into_inter_biUnion_le (d.D.asInputs (𝒲.biUnion id)) 𝒲 h𝒲 (P.gates u \ 𝒲.biUnion id))

/-- **All blocks together, a word read by any number of blocks.** With the words as inputs the blocks read at most
`m k` gates outside `θ` and `b` for each pair of a block and a word it reads. -/
theorem sum_card_unregistered_asInputs_pairs_le {m : ℕ} (d : RegisteredCircuit) (P : Partition d.D m) {k b : ℕ}
    (𝒲 : Finset (Finset (Fin d.D.N))) (h𝒲 : ∀ W ∈ 𝒲, W.card ≤ b) (hk : ReadsAtMost d P k) :
    ∑ u, ((d.asInputs (𝒲.biUnion id)).unregistered ((P.asInputs (𝒲.biUnion id)).gates u)).card ≤
      m * k + b * ∑ u : Fin m, (wordsRead (d.D.asInputs (𝒲.biUnion id)) 𝒲 (P.gates u \ 𝒲.biUnion id)).card :=
  calc ∑ u, ((d.asInputs (𝒲.biUnion id)).unregistered ((P.asInputs (𝒲.biUnion id)).gates u)).card
      ≤ ∑ u : Fin m, (k + b * (wordsRead (d.D.asInputs (𝒲.biUnion id)) 𝒲 (P.gates u \ 𝒲.biUnion id)).card) :=
        sum_le_sum fun u _ => card_unregistered_asInputs_block_le d P 𝒲 h𝒲 hk u
    _ = m * k + b * ∑ u : Fin m, (wordsRead (d.D.asInputs (𝒲.biUnion id)) 𝒲 (P.gates u \ 𝒲.biUnion id)).card := by
        rw [sum_add_distrib, sum_const, card_univ, Fintype.card_fin, smul_eq_mul, mul_sum]

/-! ### With computed words as inputs, a block newly reads only the words it held -/

/-- With computed gates `T` as inputs, what is left of `R` reads outside `θ` at most what `R` read outside `θ` and the
gates of `T` that `R` held: a gate of `T` outside `R` that it reads was a computed gate outside it, read already. -/
theorem unregistered_asInputs_subset_held (d : RegisteredCircuit) (T R : Finset (Fin d.D.N))
    (hT : ∀ g ∈ T, d.D.isInput g = false) :
    (d.asInputs T).unregistered (R \ T) ⊆ d.unregistered R ∪ R ∩ T := by
  intro a ha
  have ha' : a ∈ (d.D.asInputs T).into (R \ T) \ (d.θ \ T) := ha
  obtain ⟨hin, hout⟩ := mem_sdiff.1 ha'
  obtain ⟨haS, g, hg, hag⟩ := (mem_filter.1 hin).2
  have hag' : a ∈ d.D.reads g := by
    have h : a ∈ (if g ∈ T then ∅ else d.D.reads g) := hag
    rwa [ite_eq_right (mem_sdiff.1 hg).2] at h
  by_cases haR : a ∈ R
  · exact mem_union_right _ (mem_inter.2 ⟨haR, Classical.byContradiction fun hT' => haS (mem_sdiff.2 ⟨haR, hT'⟩)⟩)
  · refine mem_union_left _ ?_
    show a ∈ d.D.into R \ d.θ
    refine mem_sdiff.2 ⟨mem_filter.2 ⟨mem_univ a, haR, g, (mem_sdiff.1 hg).1, hag'⟩, fun hθ => ?_⟩
    by_cases haT : a ∈ T
    · have h := hT a haT
      rw [d.θ_input a hθ] at h
      cases h
    · exact hout (mem_sdiff.2 ⟨hθ, haT⟩)

/-- **Each block, at the gates of the words it held.** With computed gates `T` as inputs, a block reads outside `θ` at
most what it read and the gates of `T` it held. -/
theorem card_unregistered_asInputs_held_le {m : ℕ} (d : RegisteredCircuit) (P : Partition d.D m)
    (T : Finset (Fin d.D.N)) (hT : ∀ g ∈ T, d.D.isInput g = false) (u : Fin m) :
    ((d.asInputs T).unregistered ((P.asInputs T).gates u)).card ≤
      (d.unregistered (P.gates u)).card + (T ∩ P.gates u).card := by
  have hg : (P.asInputs T).gates u = P.gates u \ T := P.asInputs_gates T u
  calc ((d.asInputs T).unregistered ((P.asInputs T).gates u)).card
      = ((d.asInputs T).unregistered (P.gates u \ T)).card :=
        congrArg (fun S => ((d.asInputs T).unregistered S).card) hg
    _ ≤ (d.unregistered (P.gates u) ∪ P.gates u ∩ T).card :=
        card_le_card (unregistered_asInputs_subset_held d T (P.gates u) hT)
    _ ≤ (d.unregistered (P.gates u)).card + (P.gates u ∩ T).card := card_union_le _ _
    _ = (d.unregistered (P.gates u)).card + (T ∩ P.gates u).card := by rw [inter_comm]

/-- **All blocks together, whoever reads a word.** With computed gates `T` as inputs, the blocks read at most
`m k + |T|` gates outside `θ` between them: each gate of `T` is newly read by the one block that held it. -/
theorem sum_card_unregistered_asInputs_held_le {m : ℕ} (d : RegisteredCircuit) (P : Partition d.D m) {k : ℕ}
    (T : Finset (Fin d.D.N)) (hT : ∀ g ∈ T, d.D.isInput g = false) (hk : ReadsAtMost d P k) :
    ∑ u, ((d.asInputs T).unregistered ((P.asInputs T).gates u)).card ≤ m * k + T.card :=
  calc ∑ u, ((d.asInputs T).unregistered ((P.asInputs T).gates u)).card
      ≤ ∑ u : Fin m, (k + (T ∩ P.gates u).card) := sum_le_sum fun u _ =>
        (card_unregistered_asInputs_held_le d P T hT u).trans (Nat.add_le_add_right (hk u) _)
    _ = m * k + ∑ u : Fin m, (T ∩ P.gates u).card := by
        rw [sum_add_distrib, sum_const, card_univ, Fintype.card_fin, smul_eq_mul]
    _ ≤ m * k + T.card := Nat.add_le_add_left (Nci.Cut.SecurityProofs.sum_card_inter_gates_le P T) _

/-- Toy D: a computed word with two readers. A registered input, the word's gate reading it, and two gates reading
the word. -/
def readsD : Fin 4 → Finset (Fin 4) := ![∅, {0}, {1}, {1}]

/-- The toy circuit, one bit a gate. -/
abbrev TD : Circuit where
  N := 4
  w := fun _ => 1
  isInput := fun g => decide (g.val < 1)
  reads := readsD
  reads_lt := by decide
  input_reads := by decide
  op := fun _ _ => 0
  op_congr := fun _ _ _ _ => rfl
  outputs := {2, 3}

/-- Toy D, its input registered. -/
abbrev dD : RegisteredCircuit := .ofCircuit TD {0} (by decide)

/-- A block for each computed gate: the word's, and its two readers'. -/
def PD : Partition TD 3 where
  unit := fun g => if g.val < 1 then none else if g.val = 1 then some 0 else if g.val = 2 then some 1 else some 2
  input_iff := by decide

/-- The word: the one computed gate both readers read. -/
abbrev wordD : Finset (Finset (Fin TD.N)) := {{1}}

/-- Tied, the blocks read two gates outside `θ` between them: each reader reads the word, a computed gate of another
block. -/
theorem toyD_tied : ∑ u, (dD.unregistered (PD.gates u)).card = 2 := by decide

/-- With the word as an input they read the same two. -/
theorem toyD_asInputs :
    ∑ u, ((dD.asInputs ({1} : Finset (Fin TD.N))).unregistered ((PD.asInputs {1}).gates u)).card = 2 := by decide

/-- The word has two readers on the circuit with it as an input. -/
theorem toyD_not_oneReader : ¬ OneReader (PD.asInputs ({1} : Finset (Fin TD.N))) wordD := by
  unfold OneReader
  decide

/-! ### (H) carries over to the untied circuit and to the circuit with gates as inputs -/

/-- `κ` grows with the set the paths start from. -/
theorem kappa_mono_left {C : Circuit} {X X' Y : Finset (Fin C.N)} (h : X ⊆ X') : kappa C X Y ≤ kappa C X' Y := by
  obtain ⟨Z, hZ, hsep⟩ := Nci.Cut.SecurityProofs.exists_separator X' Y
  rw [← hZ]
  have hsep' : Separates C Z X Y := fun x hx y hy => hsep x (h hx) y hy
  exact Nci.Cut.SecurityProofs.kappa_le hsep'

/-- An unregistered input stays one when gates leave `θ`. -/
theorem unregisteredInputs_subset_untie (d : RegisteredCircuit) (E : Finset (Fin d.D.N)) :
    unregisteredInputs d ⊆ unregisteredInputs (d.untie E) := by
  intro g hg
  obtain ⟨hi, hθ⟩ := (mem_filter.1 hg).2
  exact mem_filter.2 ⟨mem_univ g, hi, fun h => hθ (mem_sdiff.1 (show g ∈ d.θ \ E from h)).1⟩

/-- `κ(S)` only grows when gates leave `θ`. -/
theorem kappaWords_le_untie (d : RegisteredCircuit) (E : Finset (Fin d.D.N)) (S : Finset (Finset (Fin d.D.N))) :
    kappaWords d S ≤ kappaWords (d.untie E) S := by
  show kappa d.D (unregisteredInputs d) (S.biUnion id) ≤ kappa d.D (unregisteredInputs (d.untie E)) (S.biUnion id)
  exact kappa_mono_left (unregisteredInputs_subset_untie d E)

/-- **(H) at `d` gives (H) at `d` untied at any `E`**, at the same arrays, costs and cap. -/
theorem hardness_untie {ρ V : Type} [Fintype ρ] [DecidableEq V] (d : RegisteredCircuit) (E : Finset (Fin d.D.N))
    (A : ρ → Finset (Finset (Fin d.D.N))) (val : Finset (Fin d.D.N) → V) (β : ρ → ℝ) (K : ℕ)
    (h : Nci.Cut.Assumptions.Hardness d A val β K) : Nci.Cut.Assumptions.Hardness (d.untie E) A val β K := by
  intro S hS
  have hκ : kappaWords d S ≤ kappaWords (d.untie E) S := kappaWords_le_untie d E S
  exact (h S (hκ.trans hS)).trans (Nat.cast_le.2 hκ)

/-- `Π` at the untied circuit is at most `Π` at `d`. -/
theorem piMax_untie_le {ι : Type} [Fintype ι] (d : RegisteredCircuit) (E : Finset (Fin d.D.N)) (W : Work d.D ι)
    (K : ℕ) : piMax (d.untie E) W K ≤ piMax d W K := by
  unfold piMax
  exact Finset.sup_mono fun S hS => mem_filter.2 ⟨mem_univ S, (kappaWords_le_untie d E S).trans (mem_filter.1 hS).2⟩

/-- **`HE` at the untied circuit, with (H) taken at `d`**, at the cap `k + b w`. -/
theorem he_untie {ι V : Type} [Fintype ι] [DecidableEq ι] [DecidableEq V] (d : RegisteredCircuit) (W : Work d.D ι)
    (val : Finset (Fin d.D.N) → V) (k m c b w : ℕ) (P : Partition d.D m) (𝒲 : Finset (Finset (Fin d.D.N)))
    (βw βx bE : ℝ) (hw : 0 < βw) (hx : 0 < βx) (hb : 0 < bE) (hval : ValuesInjective W val)
    (hshare : NoSharedPair W val) (hH : Nci.Cut.Assumptions.Hardness d ![W.allW, W.allX] val ![βw, βx] (k + b * w))
    (hE : Nci.Cut.Assumptions.Exports W bE) (h𝒲 : ∀ X ∈ 𝒲, X.card ≤ b) (hk : ReadsAtMost d P k)
    (hw' : ∀ u, (wordsRead d.D 𝒲 (P.gates u)).card ≤ w) :
    (W.N : ℝ) ≤ m * (((k + b * w : ℕ) : ℝ) + alphaHE (k + b * w) (βw * βx) bE) +
      ((k + b * w : ℕ) : ℝ) / 2 * √(m * W.O / (βw * βx)) :=
  (Nci.Cut.SecurityProofs.HE ι V (d.untie (𝒲.biUnion id)) W val (k + b * w) m c P βw βx bE hw hx hb hval hshare
    (hardness_untie d (𝒲.biUnion id) _ val _ _ hH) hE
    (readsAtMost_untie d P 𝒲 h𝒲 hk hw')).1

/-- A path of `C` is a path of `C` with `T` as inputs, or its end is reached from a gate of `T` there. -/
theorem feedsWithin_asInputs {C : Circuit} (T R : Finset (Fin C.N)) {x y : Fin C.N}
    (hp : Nci.Covering.FeedsWithin C R x y) :
    Nci.Covering.FeedsWithin (C.asInputs T) R x y ∨ ∃ g ∈ T, Nci.Covering.FeedsWithin (C.asInputs T) R g y := by
  induction hp with
  | refl ht => exact Or.inl (.refl ht)
  | @step g h t hg hgh _ ih =>
    rcases ih with ih | ⟨g', hg', ih⟩
    · by_cases hT : h ∈ T
      · exact Or.inr ⟨h, hT, ih⟩
      · refine Or.inl (.step hg ?_ ih)
        show g ∈ (if h ∈ T then ∅ else C.reads h)
        rw [ite_eq_right hT]
        exact hgh
    · exact Or.inr ⟨g', hg', ih⟩

/-- An unregistered input stays one when gates become inputs. -/
theorem unregisteredInputs_subset_asInputs (d : RegisteredCircuit) (T : Finset (Fin d.D.N)) :
    unregisteredInputs d ⊆ unregisteredInputs (d.asInputs T) := by
  intro g hg
  obtain ⟨hi, hθ⟩ := (mem_filter.1 hg).2
  refine mem_filter.2 ⟨mem_univ g, ?_, fun h => hθ (mem_sdiff.1 (show g ∈ d.θ \ T from h)).1⟩
  show (d.D.isInput g || decide (g ∈ T)) = true
  rw [hi, Bool.true_or]

/-- A gate made an input is an unregistered input. -/
theorem mem_unregisteredInputs_asInputs (d : RegisteredCircuit) (T : Finset (Fin d.D.N)) {g : Fin d.D.N}
    (hg : g ∈ T) : g ∈ unregisteredInputs (d.asInputs T) := by
  refine mem_filter.2 ⟨mem_univ g, ?_, fun h => (mem_sdiff.1 (show g ∈ d.θ \ T from h)).2 hg⟩
  show (d.D.isInput g || decide (g ∈ T)) = true
  rw [decide_eq_true hg, Bool.or_true]

/-- `κ(S)` only grows when gates become inputs outside `θ`. -/
theorem kappaWords_le_asInputs (d : RegisteredCircuit) (T : Finset (Fin d.D.N))
    (S : Finset (Finset (Fin d.D.N))) : kappaWords d S ≤ kappaWords (d.asInputs T) S := by
  obtain ⟨Z, hZ, hsep⟩ := Nci.Cut.SecurityProofs.exists_separator (C := (d.asInputs T).D)
    (unregisteredInputs (d.asInputs T)) (S.biUnion id)
  have hZ' : kappaWords (d.asInputs T) S = Z.card := hZ.symm
  rw [hZ']
  have hsep' : Separates d.D Z (unregisteredInputs d) (S.biUnion id) := by
    intro x hx y hy hp
    rcases feedsWithin_asInputs T (univ \ Z) hp with h | ⟨g, hg, h⟩
    · exact hsep x (unregisteredInputs_subset_asInputs d T hx) y hy h
    · exact hsep g (mem_unregisteredInputs_asInputs d T hg) y hy h
  exact Nci.Cut.SecurityProofs.kappa_le hsep'

/-- **(H) at `d` gives (H) at `d` with any gates `T` as inputs**, at the same arrays, costs and cap. -/
theorem hardness_asInputs {ρ V : Type} [Fintype ρ] [DecidableEq V] (d : RegisteredCircuit) (T : Finset (Fin d.D.N))
    (A : ρ → Finset (Finset (Fin d.D.N))) (val : Finset (Fin d.D.N) → V) (β : ρ → ℝ) (K : ℕ)
    (h : Nci.Cut.Assumptions.Hardness d A val β K) : Nci.Cut.Assumptions.Hardness (d.asInputs T) A val β K := by
  intro S hS
  have hκ : kappaWords d S ≤ kappaWords (d.asInputs T) S := kappaWords_le_asInputs d T S
  exact (h S (hκ.trans hS)).trans (Nat.cast_le.2 hκ)

/-- (H) at arrays gives (H) at any arrays inside them, when no cost is negative. -/
theorem hardness_mono_arrays {ρ V : Type} [Fintype ρ] [DecidableEq V] (d : RegisteredCircuit)
    (A A' : ρ → Finset (Finset (Fin d.D.N))) (val : Finset (Fin d.D.N) → V) (β : ρ → ℝ) (K : ℕ)
    (hβ : ∀ r, 0 ≤ β r) (hA : ∀ r, A' r ⊆ A r) (h : Nci.Cut.Assumptions.Hardness d A val β K) :
    Nci.Cut.Assumptions.Hardness d A' val β K := by
  intro S hS
  refine le_trans (sum_le_sum fun r _ => ?_) (h S hS)
  refine mul_le_mul_of_nonneg_left ?_ (hβ r)
  exact Nat.cast_le.2 (card_le_card (image_subset_image (inter_subset_inter (Finset.Subset.refl S) (hA r))))

/-! ### The output values' term over a set of blocks -/

section Outputs

open Nci.Cut.SecurityProofs (fedBy card_fedBy_le)

variable {C : Circuit} {m : ℕ}

/-- The gates of `Y` in each block of `G`, summed, are at most the gates of `Y` in the blocks of `G`. -/
theorem sum_card_inter_gates_on_le (P : Partition C m) (G : Finset (Fin m)) (Y : Finset (Fin C.N)) :
    ∑ u ∈ G, (Y ∩ P.gates u).card ≤ (Y ∩ G.biUnion P.gates).card := by
  rw [← card_biUnion]
  · exact card_le_card (biUnion_subset.2 fun u hu g hg =>
      mem_inter.2 ⟨(mem_inter.1 hg).1, mem_biUnion.2 ⟨u, hu, (mem_inter.1 hg).2⟩⟩)
  · intro u _ v _ huv
    refine disjoint_left.2 fun g hu hv => huv ?_
    exact Option.some_injective _
      ((P.mem_gates.1 (mem_inter.1 hu).2).symm.trans (P.mem_gates.1 (mem_inter.1 hv).2))

/-- The pairs of a block of `G` and an output with a wire in it are at most the output wires in the blocks of `G`. -/
theorem sum_card_hit_on_le {ι : Type} [Fintype ι] (P : Partition C m) (W : Work C ι) (G : Finset (Fin m)) :
    ∑ u ∈ G, (univ.filter fun o : W.Out => ¬ Disjoint (W.y o) (P.gates u)).card ≤
      ∑ o : W.Out, (W.y o ∩ G.biUnion P.gates).card := by
  calc ∑ u ∈ G, (univ.filter fun o : W.Out => ¬ Disjoint (W.y o) (P.gates u)).card
      = ∑ u ∈ G, ∑ o : W.Out, if ¬ Disjoint (W.y o) (P.gates u) then 1 else 0 :=
        sum_congr rfl fun u _ => card_filter _ _
    _ ≤ ∑ u ∈ G, ∑ o : W.Out, (W.y o ∩ P.gates u).card := by
        refine sum_le_sum fun u _ => sum_le_sum fun o _ => ?_
        by_cases h : Disjoint (W.y o) (P.gates u)
        · rw [ite_eq_right (not_not.2 h)]
          exact Nat.zero_le _
        · rw [ite_eq_left h]
          exact card_pos.2 (not_disjoint_iff_nonempty_inter.1 h)
    _ = ∑ o : W.Out, ∑ u ∈ G, (W.y o ∩ P.gates u).card := sum_comm
    _ ≤ ∑ o : W.Out, (W.y o ∩ G.biUnion P.gates).card :=
        sum_le_sum fun o _ => sum_card_inter_gates_on_le P G (W.y o)

/-- **The output values' term, at the blocks of `G` alone.** The pairs of a block of `G` and an output it feeds are at
most the output wires in the blocks of `G`, and the pairs of a block of `G` and an output in its `E(block)`. -/
theorem sum_card_fedBy_on_le {ι : Type} [Fintype ι] [DecidableEq ι] (P : Partition C m) (W : Work C ι)
    (G : Finset (Fin m)) :
    ∑ u ∈ G, (fedBy P W u).card ≤
      ∑ o : W.Out, (W.y o ∩ G.biUnion P.gates).card + ∑ u ∈ G, (W.clear (P.gates u)).card :=
  calc ∑ u ∈ G, (fedBy P W u).card
      ≤ ∑ u ∈ G, ((univ.filter fun o : W.Out => ¬ Disjoint (W.y o) (P.gates u)).card +
          (W.clear (P.gates u)).card) := sum_le_sum fun u _ => card_fedBy_le P W u
    _ = ∑ u ∈ G, (univ.filter fun o : W.Out => ¬ Disjoint (W.y o) (P.gates u)).card +
          ∑ u ∈ G, (W.clear (P.gates u)).card := sum_add_distrib
    _ ≤ ∑ o : W.Out, (W.y o ∩ G.biUnion P.gates).card + ∑ u ∈ G, (W.clear (P.gates u)).card :=
        Nat.add_le_add_right (sum_card_hit_on_le P W G) _

/-- The output wires in the blocks of `G` are at most all the output wires, `O`. -/
theorem sum_card_inter_on_le_O {ι : Type} [Fintype ι] (P : Partition C m) (W : Work C ι) (G : Finset (Fin m)) :
    ∑ o : W.Out, (W.y o ∩ G.biUnion P.gates).card ≤ W.O :=
  calc ∑ o : W.Out, (W.y o ∩ G.biUnion P.gates).card
      ≤ ∑ o : W.Out, (W.y o).card := sum_le_sum fun o _ => card_le_card inter_subset_left
    _ = W.O := rfl

end Outputs

/-! ### One block's count, and two instances of the count over a set of blocks -/

section Block

open Nci.Cut.SecurityProofs (inBlock readWords fedBy leaving card_inBlock_sq_le card_fedBy_le kappaWords_readWords_le
  bY_mul_card_clear_le)

/-- **One block, at its own cap and with its own share of the output values.** The MACs block `u` holds, squared, are
at most `Π` at the block's cap times the outputs it holds a wire of and the outputs in its `E(block)`, and those cost
it `b_y` leaving gates each. -/
theorem counts_block {d : RegisteredCircuit} {ι : Type} [Fintype ι] [DecidableEq ι] {m κ : ℕ} (P : Partition d.D m)
    (W : Work d.D ι) (u : Fin m) (hκ : (d.unregistered (P.gates u)).card ≤ κ) :
    (∑ a, (inBlock P W u a).card) ^ 2 ≤ piMax d W κ *
        ((univ.filter fun o : W.Out => ¬ Disjoint (W.y o) (P.gates u)).card + (W.clear (P.gates u)).card) ∧
      W.bY * (W.clear (P.gates u)).card ≤ (leaving (P.gates u)).card := by
  have hpi : W.pi (readWords P W u) ≤ piMax d W κ := by
    unfold piMax
    exact Finset.le_sup (f := W.pi) (mem_filter.2 ⟨mem_univ _, (kappaWords_readWords_le P W u).trans hκ⟩)
  exact ⟨(card_inBlock_sq_le P W u).trans (Nat.mul_le_mul hpi (card_fedBy_le P W u)),
    bY_mul_card_clear_le W (P.gates_subset_computed u)⟩

/-- On `TA`, one matmul whose one MAC is the two computed gates, with the last gate its output's wire. -/
def MS : Matmul TA where
  ni := 1
  nj := 1
  nl := 1
  w := fun _ => {0}
  x := fun _ => {1}
  y := fun _ => {3}
  w_inj := fun a b _ => Subsingleton.elim a b
  x_inj := fun a b _ => Subsingleton.elim a b
  mac := fun _ => {2, 3}
  mac_computed := by decide
  mac_connected := fun _ => ⟨⟨2, by decide⟩, fun T hT hne hT' => ⟨2, by decide, 3, by decide, by decide, by
    constructor
    · intro h2 h3
      refine hT' (Subset.antisymm hT fun g hg => ?_)
      rcases mem_insert.1 hg with rfl | hg
      · exact h2
      · rw [mem_singleton.1 hg]
        exact h3
    · intro h3
      obtain ⟨g, hg⟩ := hne
      rcases mem_insert.1 (hT hg) with rfl | hg'
      · exact hg
      · rw [mem_singleton.1 hg'] at hg
        exact absurd hg h3⟩⟩
  reads_w := by decide
  reads_x := by decide
  feeds_y := fun _ => ⟨3, by decide, 3, mem_singleton_self _, .refl _⟩

/-- The charged work on `TA`: `MS` alone. -/
def WS : Work TA Unit where
  M := fun _ => MS
  mac_disjoint := by decide

/-- One MAC is charged on `TA`. -/
theorem toyS_N : WS.N = 1 := by decide

/-- `PA2` splits it: one gate in block 0, one in block 1. -/
theorem toyS_split : (univ.filter fun s : WS.Mac => ∀ u, ¬ WS.mac s ⊆ PA2.gates u).card = 1 := by decide

/-- Neither block holds a MAC whole. -/
theorem toyS_whole : ∀ u, ∑ a, (inBlock PA2 WS u a).card = 0 := by decide

/-- Block 0 reads nothing outside `θ`. -/
theorem toyS_reads_0 : (dA.unregistered (PA2.gates 0)).card = 0 := by decide

/-- Block 1 reads one gate outside `θ`, the MAC's gate in block 0. -/
theorem toyS_reads_1 : (dA.unregistered (PA2.gates 1)).card = 1 := by decide

/-- The two blocks read one gate outside `θ` between them: the MAC is counted against the entries of both blocks, and
against those of block 0 alone it would not be counted. -/
theorem toyS_entries : ∑ u, (dA.unregistered (PA2.gates u)).card = 1 := by decide

/-- On `Nci.Cut.Toy` the work has two output wires. -/
theorem toyO_whole : Toy.W.O = 2 := by decide

/-- One of them is in block 0 of `Toy.P2`. -/
theorem toyO_share :
    ∑ o : Toy.W.Out, (Toy.W.y o ∩ ({0} : Finset (Fin 2)).biUnion Toy.P2.gates).card = 1 := by decide

/-- Block 0 of `Toy.P2` holds a wire of one output. -/
theorem toyO_hit : (univ.filter fun o : Toy.W.Out => ¬ Disjoint (Toy.W.y o) (Toy.P2.gates 0)).card = 1 := by decide

/-- Block 0 of `Toy.P2` feeds one output. -/
theorem toyO_fed : (fedBy Toy.P2 Toy.W 0).card = 1 := by decide

end Block

/-! ### `Circuit.asInputs` and `Accounting.Placed.cut` -/

section Cut

variable {L : Type*} (A : Accounting.Placed L) (T : Finset (Fin A.N))

theorem cut_compl_isInput (g : Fin A.N) : (A.cut Tᶜ).isInput g = (A.toCircuit.asInputs T).isInput g := by
  by_cases h : g ∈ T
  · simp [Accounting.Placed.cut, h]
  · simp [Accounting.Placed.cut, h]

theorem cut_compl_reads (g : Fin A.N) : (A.cut Tᶜ).reads g = (A.toCircuit.asInputs T).reads g := by
  by_cases h : g ∈ T
  · simp [Accounting.Placed.cut, h]
  · simp [Accounting.Placed.cut, h]

theorem cut_compl_op (g : Fin A.N) (v : (a : Fin A.N) → BitVec (A.w a)) :
    (A.cut Tᶜ).op g v = (A.toCircuit.asInputs T).op g v := by
  by_cases h : g ∈ T
  · simp [Accounting.Placed.cut, h]
  · simp [Accounting.Placed.cut, h]

/-- What a set of gates reads from outside itself is the same in the two. -/
theorem cut_compl_into (R : Finset (Fin A.N)) : (A.cut Tᶜ).into R = (A.toCircuit.asInputs T).into R := by
  refine Finset.ext fun a => (Accounting.SecurityProofs.Actual.mem_into_iff (A.cut Tᶜ)).trans
    (Iff.trans ?_ (Accounting.SecurityProofs.Actual.mem_into_iff (A.toCircuit.asInputs T)).symm)
  refine and_congr_right fun _ => ⟨fun ⟨g, hg, h⟩ => ⟨g, hg, ?_⟩, fun ⟨g, hg, h⟩ => ⟨g, hg, ?_⟩⟩
  · rwa [cut_compl_reads A T g] at h
  · rwa [cut_compl_reads A T g]

/-- The cut's outputs are the gates it keeps. -/
theorem cut_compl_outputs : (A.cut Tᶜ).outputs = Tᶜ := rfl

/-- With `T` as inputs, the outputs are the circuit's. -/
theorem asInputs_outputs : (A.toCircuit.asInputs T).outputs = A.outputs := rfl

end Cut

end Nci.Untied.Attack
