module

public import Proofs.Nci.Cut
public import Mathlib.Data.Finset.Sort
public import Mathlib.Logic.Function.Basic

/-!
# Untied words: what a block reads once gates leave `θ`

**Draft, never elaborated** (LEAN-238, stopped Tue 10 Nov 08:19Z on the coordinator's order). No `lake env lean` and no
run has read this file, `Proofs.Nci` does not import it, and every proof below is unchecked. The statements are the
ones posted on LEAN-238.

PoUW proves a linear layer's output words, so the inference-only policy may count them as not fresh: the value tie.
Its draw leaves a set `E` of words that may be wrong, which nobody knows, and each is fresh input of the block that
reads it. The lemmas here are stated per block and with no threshold.

Two readings of "untied at `E`", since `θ` holds input gates only (`RegisteredCircuit.θ_input`):

* `RegisteredCircuit.untie`: `E`'s gates leave `θ`. The circuit, its blocks and any charged work on it stay as they
  are, so `Nci.Cut.SecurityProofs.HE` applies to `d.untie E` with the same partition and the same `Work`.
* `RegisteredCircuit.asInputs`: `E`'s gates become input gates outside `θ` and read nothing. This is the reading for a
  tied word that is computed gates (a linear layer put back as gates), and it is also the circuit without a set of
  blocks, whose gates the kept blocks then read as input.

In either reading a set `R` of gates reads, outside `θ`, what it read before and the gates of `E` it reads
(`unregistered_untie_subset`, `unregistered_asInputs_subset`). When `E` is a union of words of at most `b` gates, that
is at most `b` for each word `R` reads (`card_unregistered_untie_words_le`), so blocks that read at most `k` gates
outside `θ` and at most `w` words of `E` read at most `k + b w` (`readsAtMost_untie`, `readsAtMost_asInputs`; `b = 32`
for a `y` word, `readsAtMost_untie_32`).

When each word is read by at most one block (`OneReader`), at most `|E| / (w + 1)` blocks read more than `w` words
(`card_over_le`), every other block reads at most `k + b w` (`card_unregistered_untie_of_not_over`), and all blocks
together read at most `m k + b |E|` (`sum_card_unregistered_untie_le`).

`HE` quantifies over a `Partition` of every computed gate of the circuit, each block within the cap, so it takes no
subset of a partition's blocks. The form it can take is the circuit with the other blocks' gates as inputs and the
partition of the kept blocks, renumbered (`Partition.keep`, `readsAtMost_keep`, `readsAtMost_untie_without`). Its
other premises are then about that circuit, not the record's: the charged work is a `Work` of it, so a matmul with one
MAC in a dropped block is lost whole, and (H) and (E) are taken at it, for a set of dropped blocks nobody knows.
-/

@[expose] public section

namespace Verity

open Finset

/-- `d` **untied at `E`**: `d`'s circuit, inference blocks, proof blocks and positions, with `E`'s gates taken out of
`θ`. -/
@[reducible] def RegisteredCircuit.untie (d : RegisteredCircuit) (E : Finset (Fin d.D.N)) : RegisteredCircuit :=
  .ofOutput ⟨d.D, d.θ \ E, fun g hg => d.θ_input g (mem_sdiff.1 hg).1, d.n, d.𝓘, d.m, d.P, d.refines, d.pos⟩

/-- `C` with the gates `T` as input gates that read nothing. Every other gate, every width and every other operation
is `C`'s. -/
@[reducible] def Circuit.asInputs (C : Circuit) (T : Finset (Fin C.N)) : Circuit where
  N := C.N
  w := C.w
  isInput g := C.isInput g || decide (g ∈ T)
  reads g := if g ∈ T then ∅ else C.reads g
  reads_lt g a ha := by
    have ha' : a ∈ (if g ∈ T then ∅ else C.reads g) := ha
    by_cases hT : g ∈ T
    · rw [if_pos hT] at ha'
      exact absurd ha' (notMem_empty a)
    · rw [if_neg hT] at ha'
      exact C.reads_lt g a ha'
  input_reads g h := by
    have h' : (C.isInput g || decide (g ∈ T)) = true := h
    show (if g ∈ T then ∅ else C.reads g) = ∅
    by_cases hT : g ∈ T
    · exact if_pos hT
    · rw [decide_eq_false hT, Bool.or_false] at h'
      rw [if_neg hT]
      exact C.input_reads g h'
  op g v := if g ∈ T then (0 : BitVec (C.w g)) else C.op g v
  op_congr g v v' h := by
    have h' : ∀ a ∈ (if g ∈ T then ∅ else C.reads g), v a = v' a := h
    show (if g ∈ T then (0 : BitVec (C.w g)) else C.op g v) = (if g ∈ T then (0 : BitVec (C.w g)) else C.op g v')
    by_cases hT : g ∈ T
    · rw [if_pos hT, if_pos hT]
    · rw [if_neg hT, if_neg hT]
      exact C.op_congr g v v' fun a ha => h' a (by rw [if_neg hT]; exact ha)
  outputs := C.outputs

/-- `d` with the gates `T` as **inputs outside `θ`**: its circuit with `T`'s gates as input gates
(`Circuit.asInputs`), none of them registered, each computed gate its own block. -/
@[reducible] def RegisteredCircuit.asInputs (d : RegisteredCircuit) (T : Finset (Fin d.D.N)) : RegisteredCircuit :=
  .ofCircuit (d.D.asInputs T) (d.θ \ T) fun g hg => by
    show (d.D.isInput g || decide (g ∈ T)) = true
    rw [d.θ_input g (mem_sdiff.1 hg).1, Bool.true_or]

/-- `P` with the gates `T` as inputs: each gate of `T` leaves its unit. -/
def Partition.asInputs {C : Circuit} {n : ℕ} (P : Partition C n) (T : Finset (Fin C.N)) :
    Partition (C.asInputs T) n where
  unit g := if g ∈ T then none else P.unit g
  input_iff g := by
    show (if g ∈ T then none else P.unit g) = none ↔ (C.isInput g || decide (g ∈ T)) = true
    by_cases hT : g ∈ T
    · rw [if_pos hT, decide_eq_true hT, Bool.or_true]
      exact ⟨fun _ => rfl, fun _ => rfl⟩
    · rw [if_neg hT, decide_eq_false hT, Bool.or_false]
      exact P.input_iff g

theorem Partition.mem_asInputs_gates {C : Circuit} {n : ℕ} (P : Partition C n) (T : Finset (Fin C.N)) {u : Fin n}
    {g : Fin C.N} : g ∈ (P.asInputs T).gates u ↔ g ∈ P.gates u ∧ g ∉ T := by
  rw [Partition.mem_gates, Partition.mem_gates]
  show (if g ∈ T then none else P.unit g) = some u ↔ P.unit g = some u ∧ g ∉ T
  by_cases hT : g ∈ T
  · rw [if_pos hT]
    exact ⟨fun h => by cases h, fun h => absurd hT h.2⟩
  · rw [if_neg hT]
    exact ⟨fun h => ⟨h, hT⟩, fun h => h.1⟩

/-- A unit of `P` with `T` as inputs is its unit in `P` without `T`'s gates. -/
theorem Partition.asInputs_gates {C : Circuit} {n : ℕ} (P : Partition C n) (T : Finset (Fin C.N)) (u : Fin n) :
    (P.asInputs T).gates u = P.gates u \ T := by
  ext g
  exact (P.mem_asInputs_gates T).trans mem_sdiff.symm

/-- The gates of the blocks of `P` that `ι` does not keep: those outside its range. -/
def Partition.dropped {C : Circuit} {n n' : ℕ} (P : Partition C n) (ι : Fin n' → Fin n) : Finset (Fin C.N) :=
  univ.filter fun g => ∃ u, P.unit g = some u ∧ ∀ v, ι v ≠ u

theorem partialInv_eq_none {α β : Type} {f : α → β} (hf : Function.Injective f) (b : β) :
    Function.partialInv f b = none ↔ ∀ a, f a ≠ b := by
  constructor
  · intro h a ha
    have h' := (hf.isPartialInv a b).2 ha
    rw [h] at h'
    cases h'
  · intro h
    cases hp : Function.partialInv f b with
    | none => rfl
    | some a => exact absurd ((hf.isPartialInv a b).1 hp) (h a)

/-- **The blocks `ι` keeps**, renumbered: block `v` is `P`'s block `ι v`, on the circuit with the other blocks' gates
as inputs. -/
noncomputable def Partition.keep {C : Circuit} {n n' : ℕ} (P : Partition C n) (ι : Fin n' → Fin n)
    (hι : Function.Injective ι) : Partition (C.asInputs (P.dropped ι)) n' where
  unit g := (P.unit g).bind (Function.partialInv ι)
  input_iff g := by
    show (P.unit g).bind (Function.partialInv ι) = none ↔ (C.isInput g || decide (g ∈ P.dropped ι)) = true
    cases hu : P.unit g with
    | none =>
      rw [(P.input_iff g).1 hu, Bool.true_or]
      exact ⟨fun _ => rfl, fun _ => rfl⟩
    | some u =>
      have hi : C.isInput g = false := Nci.Cut.SecurityProofs.gate_computed P (P.mem_gates.2 hu)
      have hd : g ∈ P.dropped ι ↔ ∀ v, ι v ≠ u := by
        constructor
        · intro hg
          obtain ⟨u', hu', hv⟩ := (mem_filter.1 hg).2
          rw [hu] at hu'
          rw [Option.some.inj hu']
          exact hv
        · intro hv
          exact mem_filter.2 ⟨mem_univ g, u, hu, hv⟩
      rw [hi, Bool.false_or, decide_eq_true_eq, hd]
      exact partialInv_eq_none hι u

/-- A kept block's gates are its gates in `P`. -/
theorem Partition.keep_gates {C : Circuit} {n n' : ℕ} (P : Partition C n) (ι : Fin n' → Fin n)
    (hι : Function.Injective ι) (v : Fin n') : (P.keep ι hι).gates v = P.gates (ι v) := by
  ext g
  rw [Partition.mem_gates, Partition.mem_gates]
  show (P.unit g).bind (Function.partialInv ι) = some v ↔ P.unit g = some (ι v)
  cases hu : P.unit g with
  | none =>
    show none = some v ↔ none = some (ι v)
    exact ⟨fun h => by cases h, fun h => by cases h⟩
  | some u =>
    show Function.partialInv ι u = some v ↔ some u = some (ι v)
    rw [hι.isPartialInv v u]
    exact ⟨fun h => congrArg some h.symm, fun h => (Option.some.inj h).symm⟩

theorem Partition.isInput_of_mem_dropped {C : Circuit} {n n' : ℕ} (P : Partition C n) (ι : Fin n' → Fin n)
    {g : Fin C.N} (hg : g ∈ P.dropped ι) : C.isInput g = false := by
  obtain ⟨u, hu, -⟩ := (mem_filter.1 hg).2
  exact Nci.Cut.SecurityProofs.gate_computed P (P.mem_gates.2 hu)

theorem Partition.notMem_dropped {C : Circuit} {n n' : ℕ} (P : Partition C n) (ι : Fin n' → Fin n) {v : Fin n'}
    {g : Fin C.N} (hg : g ∈ P.gates (ι v)) : g ∉ P.dropped ι := by
  intro hd
  obtain ⟨u, hu, hv⟩ := (mem_filter.1 hd).2
  rw [P.mem_gates.1 hg] at hu
  exact hv v (Option.some.inj hu)

/-- **Gates, as an equality.** Untied at `E`, `R` reads outside `θ` what it read outside `θ` and the gates of `E` it
reads that were registered. -/
theorem RegisteredCircuit.unregistered_untie (d : RegisteredCircuit) (E R : Finset (Fin d.D.N)) :
    (d.untie E).unregistered R = d.unregistered R ∪ d.D.into R ∩ (d.θ ∩ E) := by
  show d.D.into R \ (d.θ \ E) = d.D.into R \ d.θ ∪ d.D.into R ∩ (d.θ ∩ E)
  ext a
  constructor
  · intro h
    obtain ⟨hin, hout⟩ := mem_sdiff.1 h
    by_cases hθ : a ∈ d.θ
    · exact mem_union_right _ (mem_inter.2 ⟨hin, mem_inter.2
        ⟨hθ, Classical.byContradiction fun hE => hout (mem_sdiff.2 ⟨hθ, hE⟩)⟩⟩)
    · exact mem_union_left _ (mem_sdiff.2 ⟨hin, hθ⟩)
  · intro h
    rcases mem_union.1 h with h | h
    · exact mem_sdiff.2 ⟨(mem_sdiff.1 h).1, fun h' => (mem_sdiff.1 h).2 (mem_sdiff.1 h').1⟩
    · exact mem_sdiff.2 ⟨(mem_inter.1 h).1, fun h' => (mem_sdiff.1 h').2 (mem_inter.1 (mem_inter.1 h).2).2⟩

/-- **Gates.** Untied at `E`, `R` reads outside `θ` at most what it read outside `θ` and the gates of `E` it reads. -/
theorem RegisteredCircuit.unregistered_untie_subset (d : RegisteredCircuit) (E R : Finset (Fin d.D.N)) :
    (d.untie E).unregistered R ⊆ d.unregistered R ∪ d.D.into R ∩ E := by
  intro a ha
  rcases mem_union.1 ((Finset.ext_iff.1 (d.unregistered_untie E R) a).1 ha) with h | h
  · exact mem_union_left _ h
  · exact mem_union_right _ (mem_inter.2 ⟨(mem_inter.1 h).1, (mem_inter.1 (mem_inter.1 h).2).2⟩)

theorem RegisteredCircuit.card_unregistered_untie_le (d : RegisteredCircuit) (E R : Finset (Fin d.D.N)) :
    ((d.untie E).unregistered R).card ≤ (d.unregistered R).card + (d.D.into R ∩ E).card :=
  (card_le_card (d.unregistered_untie_subset E R)).trans (card_union_le _ _)

/-- Untying at `U` and then at `E` is untying at `U ∪ E`. -/
theorem RegisteredCircuit.unregistered_untie_untie (d : RegisteredCircuit) (U E R : Finset (Fin d.D.N)) :
    ((d.untie U).untie E).unregistered R = (d.untie (U ∪ E)).unregistered R := by
  show d.D.into R \ ((d.θ \ U) \ E) = d.D.into R \ (d.θ \ (U ∪ E))
  ext a
  constructor
  · intro h
    refine mem_sdiff.2 ⟨(mem_sdiff.1 h).1, fun h' => (mem_sdiff.1 h).2 ?_⟩
    exact mem_sdiff.2 ⟨mem_sdiff.2 ⟨(mem_sdiff.1 h').1, fun hU => (mem_sdiff.1 h').2 (mem_union_left _ hU)⟩,
      fun hE => (mem_sdiff.1 h').2 (mem_union_right _ hE)⟩
  · intro h
    refine mem_sdiff.2 ⟨(mem_sdiff.1 h).1, fun h' => (mem_sdiff.1 h).2 ?_⟩
    exact mem_sdiff.2 ⟨(mem_sdiff.1 (mem_sdiff.1 h').1).1, fun hUE =>
      (mem_union.1 hUE).elim (mem_sdiff.1 (mem_sdiff.1 h').1).2 (mem_sdiff.1 h').2⟩

/-- **Gates, with `T` as inputs.** What is left of `R` reads outside `θ` at most what `R` read outside `θ` and the
gates of `T` it reads. -/
theorem RegisteredCircuit.unregistered_asInputs_subset (d : RegisteredCircuit) (T R : Finset (Fin d.D.N)) :
    (d.asInputs T).unregistered (R \ T) ⊆ d.unregistered R ∪ (d.D.asInputs T).into (R \ T) ∩ T := by
  intro a ha
  have ha' : a ∈ (d.D.asInputs T).into (R \ T) \ (d.θ \ T) := ha
  obtain ⟨hin, hout⟩ := mem_sdiff.1 ha'
  by_cases hT : a ∈ T
  · exact mem_union_right _ (mem_inter.2 ⟨hin, hT⟩)
  · obtain ⟨haS, g, hg, hag⟩ := (mem_filter.1 hin).2
    have hag' : a ∈ d.D.reads g := by
      have h : a ∈ (if g ∈ T then ∅ else d.D.reads g) := hag
      rwa [if_neg (mem_sdiff.1 hg).2] at h
    refine mem_union_left _ ?_
    show a ∈ d.D.into R \ d.θ
    exact mem_sdiff.2 ⟨mem_filter.2 ⟨mem_univ a, fun haR => haS (mem_sdiff.2 ⟨haR, hT⟩), g, (mem_sdiff.1 hg).1, hag'⟩,
      fun hθ => hout (mem_sdiff.2 ⟨hθ, hT⟩)⟩

/-- With computed gates `T` as inputs, a set `R` of gates off `T` reads no unregistered gate it didn't read: a gate of
`T` it reads was a computed gate outside it. -/
theorem RegisteredCircuit.unregistered_asInputs_of_disjoint (d : RegisteredCircuit) {T R : Finset (Fin d.D.N)}
    (hT : ∀ g ∈ T, d.D.isInput g = false) (hR : ∀ g ∈ R, g ∉ T) :
    (d.asInputs T).unregistered R ⊆ d.unregistered R := by
  intro a ha
  have ha' : a ∈ (d.D.asInputs T).into R \ (d.θ \ T) := ha
  obtain ⟨hin, hout⟩ := mem_sdiff.1 ha'
  obtain ⟨haR, g, hg, hag⟩ := (mem_filter.1 hin).2
  have hag' : a ∈ d.D.reads g := by
    have h : a ∈ (if g ∈ T then ∅ else d.D.reads g) := hag
    rwa [if_neg (hR g hg)] at h
  show a ∈ d.D.into R \ d.θ
  refine mem_sdiff.2 ⟨mem_filter.2 ⟨mem_univ a, haR, g, hg, hag'⟩, fun hθ => ?_⟩
  by_cases haT : a ∈ T
  · have h1 := hT a haT
    rw [d.θ_input a hθ] at h1
    cases h1
  · exact hout (mem_sdiff.2 ⟨hθ, haT⟩)

end Verity

namespace Nci.Untied

open Verity Finset Nci.Cut

/-- The words of `𝒲` that `R` reads: those with a gate in `in(R)`. -/
def wordsRead (C : Circuit) (𝒲 : Finset (Finset (Fin C.N))) (R : Finset (Fin C.N)) : Finset (Finset (Fin C.N)) :=
  𝒲.filter fun W => ¬ Disjoint W (C.into R)

/-- **Words.** The gates of words of at most `b` gates that `R` reads are at most `b` for each word it reads. -/
theorem card_into_inter_biUnion_le (C : Circuit) {b : ℕ} (𝒲 : Finset (Finset (Fin C.N)))
    (h𝒲 : ∀ W ∈ 𝒲, W.card ≤ b) (R : Finset (Fin C.N)) :
    (C.into R ∩ 𝒲.biUnion id).card ≤ b * (wordsRead C 𝒲 R).card :=
  calc (C.into R ∩ 𝒲.biUnion id).card ≤ ((wordsRead C 𝒲 R).biUnion id).card := by
        refine card_le_card fun a ha => ?_
        obtain ⟨hin, hE⟩ := mem_inter.1 ha
        obtain ⟨W, hW, haW⟩ := mem_biUnion.1 hE
        exact mem_biUnion.2 ⟨W, mem_filter.2 ⟨hW, not_disjoint_iff.2 ⟨a, haW, hin⟩⟩, haW⟩
    _ ≤ ∑ W ∈ wordsRead C 𝒲 R, W.card := card_biUnion_le
    _ ≤ ∑ _W ∈ wordsRead C 𝒲 R, b := sum_le_sum fun W hW => h𝒲 W (mem_filter.1 hW).1
    _ = b * (wordsRead C 𝒲 R).card := by rw [sum_const, smul_eq_mul, mul_comm]

/-- **Bits.** Untied at `E`, `R` reads at most the fresh input bits it read and the width of the gates of `E` it
reads. -/
theorem novelReads_untie_le (d : RegisteredCircuit) (E R : Finset (Fin d.D.N)) :
    novelReads (d.untie E) R ≤ novelReads d R + d.D.width (d.D.into R ∩ E) := by
  show ∑ g ∈ (d.untie E).unregistered R, d.D.w g ≤
    ∑ g ∈ d.unregistered R, d.D.w g + ∑ g ∈ d.D.into R ∩ E, d.D.w g
  calc ∑ g ∈ (d.untie E).unregistered R, d.D.w g
      ≤ ∑ g ∈ d.unregistered R ∪ d.D.into R ∩ E, d.D.w g := sum_le_sum_of_subset (d.unregistered_untie_subset E R)
    _ ≤ ∑ g ∈ d.unregistered R, d.D.w g + ∑ g ∈ d.D.into R ∩ E, d.D.w g := Nat.le.intro sum_union_inter

/-- **The per-block lemma, in words.** Untied at the words `𝒲`, each of at most `b` gates, `R` reads outside `θ` at
most what it read outside `θ`, and `b` gates for each word of `𝒲` it reads. -/
theorem card_unregistered_untie_words_le (d : RegisteredCircuit) {b : ℕ} (𝒲 : Finset (Finset (Fin d.D.N)))
    (h𝒲 : ∀ W ∈ 𝒲, W.card ≤ b) (R : Finset (Fin d.D.N)) :
    ((d.untie (𝒲.biUnion id)).unregistered R).card ≤ (d.unregistered R).card + b * (wordsRead d.D 𝒲 R).card :=
  (d.card_unregistered_untie_le (𝒲.biUnion id) R).trans
    (Nat.add_le_add_left (card_into_inter_biUnion_le d.D 𝒲 h𝒲 R) _)

/-- **The per-block lemma, from `U` to `U ∪ E`.** Untied at `U` and the words `𝒲`, `R` reads outside `θ` at most what
it read untied at `U`, and `b` gates for each word of `𝒲` it reads. -/
theorem card_unregistered_untie_union_le (d : RegisteredCircuit) {b : ℕ} (U : Finset (Fin d.D.N))
    (𝒲 : Finset (Finset (Fin d.D.N))) (h𝒲 : ∀ W ∈ 𝒲, W.card ≤ b) (R : Finset (Fin d.D.N)) :
    ((d.untie (U ∪ 𝒲.biUnion id)).unregistered R).card ≤
      ((d.untie U).unregistered R).card + b * (wordsRead d.D 𝒲 R).card :=
  calc ((d.untie (U ∪ 𝒲.biUnion id)).unregistered R).card
      = (((d.untie U).untie (𝒲.biUnion id)).unregistered R).card :=
        congrArg Finset.card (d.unregistered_untie_untie U (𝒲.biUnion id) R).symm
    _ ≤ ((d.untie U).unregistered R).card + b * (wordsRead d.D 𝒲 R).card :=
        card_unregistered_untie_words_le (d.untie U) 𝒲 h𝒲 R

/-- **`ReadsAtMost`, untied.** If every block reads at most `k` gates outside `θ` and at most `w` words of `𝒲`, each
of at most `b` gates, then untied at `𝒲` every block reads at most `k + b w` gates outside `θ`. -/
theorem readsAtMost_untie {m : ℕ} (d : RegisteredCircuit) (P : Partition d.D m) {k b w : ℕ}
    (𝒲 : Finset (Finset (Fin d.D.N))) (h𝒲 : ∀ W ∈ 𝒲, W.card ≤ b) (hk : ReadsAtMost d P k)
    (hw : ∀ u, (wordsRead d.D 𝒲 (P.gates u)).card ≤ w) :
    ReadsAtMost (d.untie (𝒲.biUnion id)) P (k + b * w) := fun u =>
  (card_unregistered_untie_words_le d 𝒲 h𝒲 (P.gates u)).trans
    (Nat.add_le_add (hk u) (Nat.mul_le_mul_left b (hw u)))

/-- `readsAtMost_untie` at words of 32 gates: `k + 32 w`. -/
theorem readsAtMost_untie_32 {m : ℕ} (d : RegisteredCircuit) (P : Partition d.D m) {k w : ℕ}
    (𝒲 : Finset (Finset (Fin d.D.N))) (h𝒲 : ∀ W ∈ 𝒲, W.card = 32) (hk : ReadsAtMost d P k)
    (hw : ∀ u, (wordsRead d.D 𝒲 (P.gates u)).card ≤ w) :
    ReadsAtMost (d.untie (𝒲.biUnion id)) P (k + 32 * w) :=
  readsAtMost_untie d P 𝒲 (fun W hW => (h𝒲 W hW).le) hk hw

/-- Each word of `𝒲` is read by at most one block of `P`. -/
def OneReader {C : Circuit} {m : ℕ} (P : Partition C m) (𝒲 : Finset (Finset (Fin C.N))) : Prop :=
  ∀ W ∈ 𝒲, ∀ u v, ¬ Disjoint W (C.into (P.gates u)) → ¬ Disjoint W (C.into (P.gates v)) → u = v

/-- The blocks of `P` that read more than `w` words of `𝒲`: the blocks charged whole at the threshold `w`. -/
def over {C : Circuit} {m : ℕ} (P : Partition C m) (𝒲 : Finset (Finset (Fin C.N))) (w : ℕ) : Finset (Fin m) :=
  univ.filter fun u => w < (wordsRead C 𝒲 (P.gates u)).card

/-- When each word is read by at most one block, any blocks read at most `|𝒲|` words between them. -/
theorem sum_card_wordsRead_le {C : Circuit} {m : ℕ} (P : Partition C m) (𝒲 : Finset (Finset (Fin C.N)))
    (hone : OneReader P 𝒲) (S : Finset (Fin m)) : ∑ u ∈ S, (wordsRead C 𝒲 (P.gates u)).card ≤ 𝒲.card := by
  rw [← card_biUnion]
  · exact card_le_card (biUnion_subset.2 fun u _ => filter_subset _ _)
  · intro u _ v _ huv
    refine disjoint_left.2 fun W hu hv => huv ?_
    exact hone W (mem_filter.1 hu).1 u v (mem_filter.1 hu).2 (mem_filter.1 hv).2

/-- **The charged blocks are few.** When each word is read by at most one block, at most `|𝒲| / (w + 1)` blocks read
more than `w` words. -/
theorem card_over_le {C : Circuit} {m : ℕ} (P : Partition C m) (𝒲 : Finset (Finset (Fin C.N)))
    (hone : OneReader P 𝒲) (w : ℕ) : (w + 1) * (over P 𝒲 w).card ≤ 𝒲.card :=
  calc (w + 1) * (over P 𝒲 w).card = ∑ _u ∈ over P 𝒲 w, (w + 1) := by rw [sum_const, smul_eq_mul, mul_comm]
    _ ≤ ∑ u ∈ over P 𝒲 w, (wordsRead C 𝒲 (P.gates u)).card :=
        sum_le_sum fun u hu => Nat.succ_le_of_lt (mem_filter.1 hu).2
    _ ≤ 𝒲.card := sum_card_wordsRead_le P 𝒲 hone _

/-- **The `w = 0` row.** When each word is read by at most one block, at most `|𝒲|` blocks read a word. -/
theorem card_readers_le {C : Circuit} {m : ℕ} (P : Partition C m) (𝒲 : Finset (Finset (Fin C.N)))
    (hone : OneReader P 𝒲) : (over P 𝒲 0).card ≤ 𝒲.card := by
  have h := card_over_le P 𝒲 hone 0
  rwa [Nat.zero_add, Nat.one_mul] at h

/-- **The other blocks.** A block that reads at most `w` words of `𝒲` reads, untied at `𝒲`, at most `k + b w` gates
outside `θ`; at `w = 0` it keeps `k`. -/
theorem card_unregistered_untie_of_not_over {m : ℕ} (d : RegisteredCircuit) (P : Partition d.D m) {k b w : ℕ}
    (𝒲 : Finset (Finset (Fin d.D.N))) (h𝒲 : ∀ W ∈ 𝒲, W.card ≤ b) (hk : ReadsAtMost d P k) {u : Fin m}
    (hu : u ∉ over P 𝒲 w) : ((d.untie (𝒲.biUnion id)).unregistered (P.gates u)).card ≤ k + b * w :=
  (card_unregistered_untie_words_le d 𝒲 h𝒲 (P.gates u)).trans
    (Nat.add_le_add (hk u)
      (Nat.mul_le_mul_left b (Nat.le_of_not_lt fun h => hu (mem_filter.2 ⟨mem_univ u, h⟩))))

/-- **All blocks together.** When each word is read by at most one block, untied at `𝒲` the blocks read at most
`m k + b |𝒲|` gates outside `θ` between them, whatever the threshold. -/
theorem sum_card_unregistered_untie_le {m : ℕ} (d : RegisteredCircuit) (P : Partition d.D m) {k b : ℕ}
    (𝒲 : Finset (Finset (Fin d.D.N))) (h𝒲 : ∀ W ∈ 𝒲, W.card ≤ b) (hk : ReadsAtMost d P k)
    (hone : OneReader P 𝒲) :
    ∑ u, ((d.untie (𝒲.biUnion id)).unregistered (P.gates u)).card ≤ m * k + b * 𝒲.card :=
  calc ∑ u, ((d.untie (𝒲.biUnion id)).unregistered (P.gates u)).card
      ≤ ∑ u : Fin m, (k + b * (wordsRead d.D 𝒲 (P.gates u)).card) :=
        sum_le_sum fun u _ =>
          (card_unregistered_untie_words_le d 𝒲 h𝒲 (P.gates u)).trans (Nat.add_le_add_right (hk u) _)
    _ = m * k + b * ∑ u : Fin m, (wordsRead d.D 𝒲 (P.gates u)).card := by
        rw [sum_add_distrib, sum_const, card_univ, Fintype.card_fin, smul_eq_mul, mul_sum]
    _ ≤ m * k + b * 𝒲.card :=
        Nat.add_le_add_left (Nat.mul_le_mul_left b (sum_card_wordsRead_le P 𝒲 hone univ)) _

/-- **`ReadsAtMost`, with the words as inputs.** The reading for words that are computed gates: with `𝒲`'s gates as
inputs outside `θ`, a block that read at most `k` gates outside `θ` and now reads at most `w` words of `𝒲` reads at
most `k + b w`. -/
theorem readsAtMost_asInputs {m : ℕ} (d : RegisteredCircuit) (P : Partition d.D m) {k b w : ℕ}
    (𝒲 : Finset (Finset (Fin d.D.N))) (h𝒲 : ∀ W ∈ 𝒲, W.card ≤ b) (hk : ReadsAtMost d P k)
    (hw : ∀ u, (wordsRead (d.D.asInputs (𝒲.biUnion id)) 𝒲 (P.gates u \ 𝒲.biUnion id)).card ≤ w) :
    ReadsAtMost (d.asInputs (𝒲.biUnion id)) (P.asInputs (𝒲.biUnion id)) (k + b * w) := by
  intro u
  have hg : (P.asInputs (𝒲.biUnion id)).gates u = P.gates u \ 𝒲.biUnion id := P.asInputs_gates (𝒲.biUnion id) u
  calc ((d.asInputs (𝒲.biUnion id)).unregistered ((P.asInputs (𝒲.biUnion id)).gates u)).card
      = ((d.asInputs (𝒲.biUnion id)).unregistered (P.gates u \ 𝒲.biUnion id)).card :=
        congrArg (fun S => ((d.asInputs (𝒲.biUnion id)).unregistered S).card) hg
    _ ≤ (d.unregistered (P.gates u) ∪
          (d.D.asInputs (𝒲.biUnion id)).into (P.gates u \ 𝒲.biUnion id) ∩ 𝒲.biUnion id).card :=
        card_le_card (d.unregistered_asInputs_subset (𝒲.biUnion id) (P.gates u))
    _ ≤ (d.unregistered (P.gates u)).card +
          ((d.D.asInputs (𝒲.biUnion id)).into (P.gates u \ 𝒲.biUnion id) ∩ 𝒲.biUnion id).card := card_union_le _ _
    _ ≤ k + b * w :=
        Nat.add_le_add (hk u)
          ((card_into_inter_biUnion_le (d.D.asInputs (𝒲.biUnion id)) 𝒲 h𝒲 (P.gates u \ 𝒲.biUnion id)).trans
            (Nat.mul_le_mul_left b (hw u)))

/-- **The sub-partition, in the form `HE` takes.** On the circuit with the other blocks' gates as inputs, the kept
blocks, renumbered, read outside `θ` what they read in `d`: at most `k` when each did. -/
theorem readsAtMost_keep {m m' : ℕ} (d : RegisteredCircuit) (P : Partition d.D m) (ι : Fin m' → Fin m)
    (hι : Function.Injective ι) {k : ℕ} (hk : ∀ v, (d.unregistered (P.gates (ι v))).card ≤ k) :
    ReadsAtMost (d.asInputs (P.dropped ι)) (P.keep ι hι) k := by
  intro v
  have hg : (P.keep ι hι).gates v = P.gates (ι v) := P.keep_gates ι hι v
  calc ((d.asInputs (P.dropped ι)).unregistered ((P.keep ι hι).gates v)).card
      = ((d.asInputs (P.dropped ι)).unregistered (P.gates (ι v))).card :=
        congrArg (fun S => ((d.asInputs (P.dropped ι)).unregistered S).card) hg
    _ ≤ (d.unregistered (P.gates (ι v))).card :=
        card_le_card (d.unregistered_asInputs_of_disjoint (fun g hg => P.isInput_of_mem_dropped ι hg)
          fun g hg => P.notMem_dropped ι hg)
    _ ≤ k := hk v

/-- The blocks outside `S`, renumbered: `m - |S|` of them. -/
theorem exists_kept {m : ℕ} (S : Finset (Fin m)) :
    ∃ ι : Fin (m - S.card) → Fin m, Function.Injective ι ∧ ∀ v, ι v ∉ S := by
  have h : Sᶜ.card = m - S.card := by rw [card_compl, Fintype.card_fin]
  exact ⟨Sᶜ.orderEmbOfFin h, (Sᶜ.orderEmbOfFin h).injective, fun v => mem_compl.1 (Sᶜ.orderEmbOfFin_mem h v)⟩

/-- **Two tiers, in the form `HE` takes.** Untied at the words `𝒲`, each of at most `b` gates, and without the blocks
that read more than `w` of them, the remaining `m - |over|` blocks read at most `k + b w` gates outside `θ`. -/
theorem readsAtMost_untie_without {m : ℕ} (d : RegisteredCircuit) (P : Partition d.D m) {k b : ℕ}
    (𝒲 : Finset (Finset (Fin d.D.N))) (h𝒲 : ∀ W ∈ 𝒲, W.card ≤ b) (hk : ReadsAtMost d P k) (w : ℕ) :
    ∃ (ι : Fin (m - (over P 𝒲 w).card) → Fin m) (hι : Function.Injective ι), (∀ v, ι v ∉ over P 𝒲 w) ∧
      ReadsAtMost ((d.untie (𝒲.biUnion id)).asInputs (P.dropped ι)) (P.keep ι hι) (k + b * w) := by
  obtain ⟨ι, hι, hS⟩ := exists_kept (over P 𝒲 w)
  exact ⟨ι, hι, hS, readsAtMost_keep (d.untie (𝒲.biUnion id)) P ι hι fun v =>
    card_unregistered_untie_of_not_over d P 𝒲 h𝒲 hk (hS v)⟩

end Nci.Untied
