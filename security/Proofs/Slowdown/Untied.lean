module

public import Definitions.ResourceAccounting.Actual
public import Definitions.InferenceOnly
public import Proofs.Slowdown.Cut
public import Mathlib.Data.Finset.Sort
public import Mathlib.Logic.Function.Basic

/-! Untied words: what a block reads once a word is no longer tied to registered data, per block and for every `E`. -/

@[expose] public section

open InferenceOnly

namespace Verity

open Finset

@[reducible] def RegisteredCircuit.untie (d : RegisteredCircuit) (E : Finset (Fin d.D.N)) : RegisteredCircuit :=
  .ofOutput ⟨d.D, d.θ \ E, fun g hg => d.θ_input g (mem_sdiff.1 hg).1, d.n, d.𝓘, d.m, d.P, d.refines, d.pos⟩

@[reducible] def Circuit.asInputs (C : Circuit) (T : Finset (Fin C.N)) : Circuit where
  N := C.N
  w := C.w
  isInput g := C.isInput g || decide (g ∈ T)
  reads g := if g ∈ T then ∅ else C.reads g
  reads_lt g a ha := by
    have ha' : a ∈ (if g ∈ T then ∅ else C.reads g) := ha
    by_cases hT : g ∈ T
    · rw [ite_eq_left hT] at ha'
      exact absurd ha' (notMem_empty a)
    · rw [ite_eq_right hT] at ha'
      exact C.reads_lt g a ha'
  input_reads g h := by
    have h' : (C.isInput g || decide (g ∈ T)) = true := h
    show (if g ∈ T then ∅ else C.reads g) = ∅
    by_cases hT : g ∈ T
    · exact ite_eq_left hT
    · rw [decide_eq_false hT, Bool.or_false] at h'
      rw [ite_eq_right hT]
      exact C.input_reads g h'
  op g v := if g ∈ T then (0 : BitVec (C.w g)) else C.op g v
  op_congr g v v' h := by
    have h' : ∀ a ∈ (if g ∈ T then ∅ else C.reads g), v a = v' a := h
    show (if g ∈ T then (0 : BitVec (C.w g)) else C.op g v) = (if g ∈ T then (0 : BitVec (C.w g)) else C.op g v')
    by_cases hT : g ∈ T
    · rw [ite_eq_left hT, ite_eq_left hT]
    · rw [ite_eq_right hT, ite_eq_right hT]
      exact C.op_congr g v v' fun a ha => h' a (by rw [ite_eq_right hT]; exact ha)
  outputs := C.outputs

theorem Circuit.mk_congr {N : ℕ} {w : Fin N → ℕ} {isInput isInput' : Fin N → Bool}
    {reads reads' : Fin N → Finset (Fin N)} {op op' : (g : Fin N) → ((a : Fin N) → BitVec (w a)) → BitVec (w g)}
    {outputs outputs' : Finset (Fin N)} {h₁ : ∀ g, ∀ a ∈ reads g, a < g} {h₁' : ∀ g, ∀ a ∈ reads' g, a < g}
    {h₂ : ∀ g, isInput g = true → reads g = ∅} {h₂' : ∀ g, isInput' g = true → reads' g = ∅}
    {h₃ : ∀ g (v v' : (a : Fin N) → BitVec (w a)), (∀ a ∈ reads g, v a = v' a) → op g v = op g v'}
    {h₃' : ∀ g (v v' : (a : Fin N) → BitVec (w a)), (∀ a ∈ reads' g, v a = v' a) → op' g v = op' g v'}
    (hi : isInput = isInput') (hr : reads = reads') (hop : op = op') (ho : outputs = outputs') :
    Circuit.mk N w isInput reads h₁ h₂ op h₃ outputs = Circuit.mk N w isInput' reads' h₁' h₂' op' h₃' outputs' := by
  subst hi hr hop ho
  rfl

theorem _root_.ResourceAccounting.Placed.cut_eq_asInputs {L : Type*} (A : ResourceAccounting.Placed L) (S : Finset (Fin A.N)) :
    A.cut S = { A.toCircuit.asInputs Sᶜ with outputs := S } := by
  unfold ResourceAccounting.Placed.cut
  refine Circuit.mk_congr (funext fun g => ?_) (funext fun g => ?_) (funext fun g => funext fun v => ?_) rfl
  · show _ = (A.isInput g || decide (g ∈ Sᶜ))
    by_cases h : g ∈ S <;> simp [h]
  · show _ = if g ∈ Sᶜ then ∅ else A.reads g
    by_cases h : g ∈ S <;> simp [h]
  · show _ = if g ∈ Sᶜ then (0 : BitVec (A.w g)) else A.op g v
    by_cases h : g ∈ S <;> simp [h]

@[reducible] def RegisteredCircuit.asInputs (d : RegisteredCircuit) (T : Finset (Fin d.D.N)) : RegisteredCircuit :=
  .ofCircuit (d.D.asInputs T) (d.θ \ T) fun g hg => by
    show (d.D.isInput g || decide (g ∈ T)) = true
    rw [d.θ_input g (mem_sdiff.1 hg).1, Bool.true_or]

def Partition.asInputs {C : Circuit} {n : ℕ} (P : Partition C n) (T : Finset (Fin C.N)) :
    Partition (C.asInputs T) n where
  unit g := if g ∈ T then none else P.unit g
  input_iff g := by
    show (if g ∈ T then none else P.unit g) = none ↔ (C.isInput g || decide (g ∈ T)) = true
    by_cases hT : g ∈ T
    · rw [ite_eq_left hT, decide_eq_true hT, Bool.or_true]
      exact ⟨fun _ => rfl, fun _ => rfl⟩
    · rw [ite_eq_right hT, decide_eq_false hT, Bool.or_false]
      exact P.input_iff g

theorem Partition.mem_asInputs_gates {C : Circuit} {n : ℕ} (P : Partition C n) (T : Finset (Fin C.N)) {u : Fin n}
    {g : Fin C.N} : g ∈ (P.asInputs T).gates u ↔ g ∈ P.gates u ∧ g ∉ T := by
  rw [Partition.mem_gates, Partition.mem_gates]
  show (if g ∈ T then none else P.unit g) = some u ↔ P.unit g = some u ∧ g ∉ T
  by_cases hT : g ∈ T
  · rw [ite_eq_left hT]
    exact ⟨fun h => (by cases h), fun h => absurd hT h.2⟩
  · rw [ite_eq_right hT]
    exact ⟨fun h => ⟨h, hT⟩, fun h => h.1⟩

theorem Partition.asInputs_gates {C : Circuit} {n : ℕ} (P : Partition C n) (T : Finset (Fin C.N)) (u : Fin n) :
    (P.asInputs T).gates u = P.gates u \ T := by
  ext g
  exact (P.mem_asInputs_gates T).trans mem_sdiff.symm

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
      have hi : C.isInput g = false := Slowdown.Cut.SecurityProofs.gate_computed P (P.mem_gates.2 hu)
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

theorem Partition.keep_gates {C : Circuit} {n n' : ℕ} (P : Partition C n) (ι : Fin n' → Fin n)
    (hι : Function.Injective ι) (v : Fin n') : (P.keep ι hι).gates v = P.gates (ι v) := by
  ext g
  rw [Partition.mem_gates, Partition.mem_gates]
  show (P.unit g).bind (Function.partialInv ι) = some v ↔ P.unit g = some (ι v)
  cases hu : P.unit g with
  | none =>
    show none = some v ↔ none = some (ι v)
    exact ⟨fun h => (by cases h), fun h => (by cases h)⟩
  | some u =>
    show Function.partialInv ι u = some v ↔ some u = some (ι v)
    rw [hι.isPartialInv v u]
    exact ⟨fun h => congrArg some h.symm, fun h => (Option.some.inj h).symm⟩

theorem Partition.isInput_of_mem_dropped {C : Circuit} {n n' : ℕ} (P : Partition C n) (ι : Fin n' → Fin n)
    {g : Fin C.N} (hg : g ∈ P.dropped ι) : C.isInput g = false := by
  obtain ⟨u, hu, -⟩ := (mem_filter.1 hg).2
  exact Slowdown.Cut.SecurityProofs.gate_computed P (P.mem_gates.2 hu)

theorem Partition.notMem_dropped {C : Circuit} {n n' : ℕ} (P : Partition C n) (ι : Fin n' → Fin n) {v : Fin n'}
    {g : Fin C.N} (hg : g ∈ P.gates (ι v)) : g ∉ P.dropped ι := by
  intro hd
  obtain ⟨u, hu, hv⟩ := (mem_filter.1 hd).2
  rw [P.mem_gates.1 hg] at hu
  exact hv v (Option.some.inj hu)

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

theorem RegisteredCircuit.unregistered_untie_subset (d : RegisteredCircuit) (E R : Finset (Fin d.D.N)) :
    (d.untie E).unregistered R ⊆ d.unregistered R ∪ d.D.into R ∩ E := by
  intro a ha
  rcases mem_union.1 ((Finset.ext_iff.1 (d.unregistered_untie E R) a).1 ha) with h | h
  · exact mem_union_left _ h
  · exact mem_union_right _ (mem_inter.2 ⟨(mem_inter.1 h).1, (mem_inter.1 (mem_inter.1 h).2).2⟩)

theorem RegisteredCircuit.card_unregistered_untie_le (d : RegisteredCircuit) (E R : Finset (Fin d.D.N)) :
    ((d.untie E).unregistered R).card ≤ (d.unregistered R).card + (d.D.into R ∩ E).card :=
  (card_le_card (d.unregistered_untie_subset E R)).trans (card_union_le _ _)

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
      rwa [ite_eq_right (mem_sdiff.1 hg).2] at h
    refine mem_union_left _ ?_
    show a ∈ d.D.into R \ d.θ
    exact mem_sdiff.2 ⟨mem_filter.2 ⟨mem_univ a, fun haR => haS (mem_sdiff.2 ⟨haR, hT⟩), g, (mem_sdiff.1 hg).1, hag'⟩,
      fun hθ => hout (mem_sdiff.2 ⟨hθ, hT⟩)⟩

theorem RegisteredCircuit.unregistered_asInputs_of_disjoint (d : RegisteredCircuit) {T R : Finset (Fin d.D.N)}
    (hT : ∀ g ∈ T, d.D.isInput g = false) (hR : ∀ g ∈ R, g ∉ T) :
    (d.asInputs T).unregistered R ⊆ d.unregistered R := by
  intro a ha
  have ha' : a ∈ (d.D.asInputs T).into R \ (d.θ \ T) := ha
  obtain ⟨hin, hout⟩ := mem_sdiff.1 ha'
  obtain ⟨haR, g, hg, hag⟩ := (mem_filter.1 hin).2
  have hag' : a ∈ d.D.reads g := by
    have h : a ∈ (if g ∈ T then ∅ else d.D.reads g) := hag
    rwa [ite_eq_right (hR g hg)] at h
  show a ∈ d.D.into R \ d.θ
  refine mem_sdiff.2 ⟨mem_filter.2 ⟨mem_univ a, haR, g, hg, hag'⟩, fun hθ => ?_⟩
  by_cases haT : a ∈ T
  · have h1 := hT a haT
    rw [d.θ_input a hθ] at h1
    cases h1
  · exact hout (mem_sdiff.2 ⟨hθ, haT⟩)

end Verity

namespace Slowdown.Cut.SecurityProofs.Untied

open Verity Finset

def wordsRead (C : Circuit) (𝒲 : Finset (Finset (Fin C.N))) (R : Finset (Fin C.N)) : Finset (Finset (Fin C.N)) :=
  𝒲.filter fun W => ¬ Disjoint W (C.into R)

theorem into_inter_biUnion_subset (C : Circuit) (𝒲 : Finset (Finset (Fin C.N))) (R : Finset (Fin C.N)) :
    C.into R ∩ 𝒲.biUnion id ⊆ (wordsRead C 𝒲 R).biUnion id := by
  intro a ha
  obtain ⟨hin, hE⟩ := mem_inter.1 ha
  obtain ⟨W, hW, haW⟩ := mem_biUnion.1 hE
  exact mem_biUnion.2 ⟨W, mem_filter.2 ⟨hW, not_disjoint_iff.2 ⟨a, haW, hin⟩⟩, haW⟩

theorem card_into_inter_biUnion_le (C : Circuit) {b : ℕ} (𝒲 : Finset (Finset (Fin C.N)))
    (h𝒲 : ∀ W ∈ 𝒲, W.card ≤ b) (R : Finset (Fin C.N)) :
    (C.into R ∩ 𝒲.biUnion id).card ≤ b * (wordsRead C 𝒲 R).card :=
  calc (C.into R ∩ 𝒲.biUnion id).card ≤ ((wordsRead C 𝒲 R).biUnion id).card :=
        card_le_card (into_inter_biUnion_subset C 𝒲 R)
    _ ≤ ∑ W ∈ wordsRead C 𝒲 R, W.card := card_biUnion_le
    _ ≤ ∑ _W ∈ wordsRead C 𝒲 R, b := sum_le_sum fun W hW => h𝒲 W (mem_filter.1 hW).1
    _ = b * (wordsRead C 𝒲 R).card := by rw [sum_const, smul_eq_mul, mul_comm]

theorem width_biUnion_le (C : Circuit) (s : Finset (Finset (Fin C.N))) :
    C.width (s.biUnion id) ≤ ∑ W ∈ s, C.width W := by
  induction s using Finset.induction_on with
  | empty => simp [Circuit.width]
  | insert W s hW ih =>
      rw [biUnion_insert, sum_insert hW]
      show ∑ g ∈ W ∪ s.biUnion id, C.w g ≤ ∑ g ∈ W, C.w g + ∑ W' ∈ s, C.width W'
      exact (Nat.le.intro sum_union_inter).trans (Nat.add_le_add_left ih _)

theorem width_into_inter_biUnion_le (C : Circuit) {b : ℕ} (𝒲 : Finset (Finset (Fin C.N)))
    (h𝒲 : ∀ W ∈ 𝒲, C.width W ≤ b) (R : Finset (Fin C.N)) :
    C.width (C.into R ∩ 𝒲.biUnion id) ≤ b * (wordsRead C 𝒲 R).card :=
  calc C.width (C.into R ∩ 𝒲.biUnion id) ≤ C.width ((wordsRead C 𝒲 R).biUnion id) :=
        sum_le_sum_of_subset (into_inter_biUnion_subset C 𝒲 R)
    _ ≤ ∑ W ∈ wordsRead C 𝒲 R, C.width W := width_biUnion_le C _
    _ ≤ ∑ _W ∈ wordsRead C 𝒲 R, b := sum_le_sum fun W hW => h𝒲 W (mem_filter.1 hW).1
    _ = b * (wordsRead C 𝒲 R).card := by rw [sum_const, smul_eq_mul, mul_comm]

theorem freshInput_untie_le (d : RegisteredCircuit) (E R : Finset (Fin d.D.N)) :
    freshInput (d.untie E) R ≤ freshInput d R + d.D.width (d.D.into R ∩ E) := by
  show ∑ g ∈ (d.untie E).unregistered R, d.D.w g ≤
    ∑ g ∈ d.unregistered R, d.D.w g + ∑ g ∈ d.D.into R ∩ E, d.D.w g
  calc ∑ g ∈ (d.untie E).unregistered R, d.D.w g
      ≤ ∑ g ∈ d.unregistered R ∪ d.D.into R ∩ E, d.D.w g := sum_le_sum_of_subset (d.unregistered_untie_subset E R)
    _ ≤ ∑ g ∈ d.unregistered R, d.D.w g + ∑ g ∈ d.D.into R ∩ E, d.D.w g := Nat.le.intro sum_union_inter

theorem freshInput_untie_union_le (d : RegisteredCircuit) {b : ℕ} (U : Finset (Fin d.D.N))
    (𝒲 : Finset (Finset (Fin d.D.N))) (h𝒲 : ∀ W ∈ 𝒲, d.D.width W ≤ b) (R : Finset (Fin d.D.N)) :
    freshInput (d.untie (U ∪ 𝒲.biUnion id)) R ≤ freshInput (d.untie U) R + b * (wordsRead d.D 𝒲 R).card :=
  calc freshInput (d.untie (U ∪ 𝒲.biUnion id)) R
      = freshInput ((d.untie U).untie (𝒲.biUnion id)) R :=
        congrArg d.D.width (d.unregistered_untie_untie U (𝒲.biUnion id) R).symm
    _ ≤ freshInput (d.untie U) R + d.D.width (d.D.into R ∩ 𝒲.biUnion id) :=
        freshInput_untie_le (d.untie U) (𝒲.biUnion id) R
    _ ≤ freshInput (d.untie U) R + b * (wordsRead d.D 𝒲 R).card :=
        Nat.add_le_add_left (width_into_inter_biUnion_le d.D 𝒲 h𝒲 R) _

theorem card_unregistered_untie_words_le (d : RegisteredCircuit) {b : ℕ} (𝒲 : Finset (Finset (Fin d.D.N)))
    (h𝒲 : ∀ W ∈ 𝒲, W.card ≤ b) (R : Finset (Fin d.D.N)) :
    ((d.untie (𝒲.biUnion id)).unregistered R).card ≤ (d.unregistered R).card + b * (wordsRead d.D 𝒲 R).card :=
  (d.card_unregistered_untie_le (𝒲.biUnion id) R).trans
    (Nat.add_le_add_left (card_into_inter_biUnion_le d.D 𝒲 h𝒲 R) _)

theorem card_unregistered_untie_union_le (d : RegisteredCircuit) {b : ℕ} (U : Finset (Fin d.D.N))
    (𝒲 : Finset (Finset (Fin d.D.N))) (h𝒲 : ∀ W ∈ 𝒲, W.card ≤ b) (R : Finset (Fin d.D.N)) :
    ((d.untie (U ∪ 𝒲.biUnion id)).unregistered R).card ≤
      ((d.untie U).unregistered R).card + b * (wordsRead d.D 𝒲 R).card :=
  calc ((d.untie (U ∪ 𝒲.biUnion id)).unregistered R).card
      = (((d.untie U).untie (𝒲.biUnion id)).unregistered R).card :=
        congrArg Finset.card (d.unregistered_untie_untie U (𝒲.biUnion id) R).symm
    _ ≤ ((d.untie U).unregistered R).card + b * (wordsRead d.D 𝒲 R).card :=
        card_unregistered_untie_words_le (d.untie U) 𝒲 h𝒲 R

theorem readsAtMost_untie {m : ℕ} (d : RegisteredCircuit) (P : Partition d.D m) {k b w : ℕ}
    (𝒲 : Finset (Finset (Fin d.D.N))) (h𝒲 : ∀ W ∈ 𝒲, W.card ≤ b) (hk : ReadsAtMost d P k)
    (hw : ∀ u, (wordsRead d.D 𝒲 (P.gates u)).card ≤ w) :
    ReadsAtMost (d.untie (𝒲.biUnion id)) P (k + b * w) := fun u =>
  (card_unregistered_untie_words_le d 𝒲 h𝒲 (P.gates u)).trans
    (Nat.add_le_add (hk u) (Nat.mul_le_mul_left b (hw u)))

theorem readsAtMost_untie_32 {m : ℕ} (d : RegisteredCircuit) (P : Partition d.D m) {k w : ℕ}
    (𝒲 : Finset (Finset (Fin d.D.N))) (h𝒲 : ∀ W ∈ 𝒲, W.card = 32) (hk : ReadsAtMost d P k)
    (hw : ∀ u, (wordsRead d.D 𝒲 (P.gates u)).card ≤ w) :
    ReadsAtMost (d.untie (𝒲.biUnion id)) P (k + 32 * w) :=
  readsAtMost_untie d P 𝒲 (fun W hW => (h𝒲 W hW).le) hk hw

theorem readsAtMost_untie_union {m : ℕ} (d : RegisteredCircuit) (P : Partition d.D m) {k b w : ℕ}
    (U : Finset (Fin d.D.N)) (𝒲 : Finset (Finset (Fin d.D.N))) (h𝒲 : ∀ W ∈ 𝒲, W.card ≤ b)
    (hk : ReadsAtMost (d.untie U) P k) (hw : ∀ u, (wordsRead d.D 𝒲 (P.gates u)).card ≤ w) :
    ReadsAtMost (d.untie (U ∪ 𝒲.biUnion id)) P (k + b * w) := fun u =>
  (card_unregistered_untie_union_le d U 𝒲 h𝒲 (P.gates u)).trans
    (Nat.add_le_add (hk u) (Nat.mul_le_mul_left b (hw u)))

theorem isCK_untie {m : ℕ} (d : RegisteredCircuit) (P : Partition d.D m) {c k b w : ℕ}
    (𝒲 : Finset (Finset (Fin d.D.N))) (h𝒲 : ∀ W ∈ 𝒲, W.card ≤ b) (hP : P.IsCK d c k)
    (hw : ∀ u, (wordsRead d.D 𝒲 (P.gates u)).card ≤ w) :
    Partition.IsCK (d.untie (𝒲.biUnion id)) P c (k + b * w) :=
  ⟨hP.boolean, hP.card, readsAtMost_untie d P 𝒲 h𝒲 hP.reads hw⟩

def OneReader {C : Circuit} {m : ℕ} (P : Partition C m) (𝒲 : Finset (Finset (Fin C.N))) : Prop :=
  ∀ W ∈ 𝒲, ∀ u v, ¬ Disjoint W (C.into (P.gates u)) → ¬ Disjoint W (C.into (P.gates v)) → u = v

def readersOver {C : Circuit} {m : ℕ} (P : Partition C m) (𝒲 : Finset (Finset (Fin C.N))) (w : ℕ) : Finset (Fin m) :=
  univ.filter fun u => w < (wordsRead C 𝒲 (P.gates u)).card

theorem sum_card_wordsRead_le {C : Circuit} {m : ℕ} (P : Partition C m) (𝒲 : Finset (Finset (Fin C.N)))
    (hone : OneReader P 𝒲) (S : Finset (Fin m)) : ∑ u ∈ S, (wordsRead C 𝒲 (P.gates u)).card ≤ 𝒲.card := by
  rw [← card_biUnion]
  · exact card_le_card (biUnion_subset.2 fun u _ => filter_subset _ _)
  · intro u _ v _ huv
    refine disjoint_left.2 fun W hu hv => huv ?_
    exact hone W (mem_filter.1 hu).1 u v (mem_filter.1 hu).2 (mem_filter.1 hv).2

theorem card_readersOver_le {C : Circuit} {m : ℕ} (P : Partition C m) (𝒲 : Finset (Finset (Fin C.N)))
    (hone : OneReader P 𝒲) (w : ℕ) : (w + 1) * (readersOver P 𝒲 w).card ≤ 𝒲.card :=
  calc (w + 1) * (readersOver P 𝒲 w).card = ∑ _u ∈ readersOver P 𝒲 w, (w + 1) := by
        rw [sum_const, smul_eq_mul, mul_comm]
    _ ≤ ∑ u ∈ readersOver P 𝒲 w, (wordsRead C 𝒲 (P.gates u)).card :=
        sum_le_sum fun u hu => Nat.succ_le_of_lt (mem_filter.1 hu).2
    _ ≤ 𝒲.card := sum_card_wordsRead_le P 𝒲 hone _

theorem card_readers_le {C : Circuit} {m : ℕ} (P : Partition C m) (𝒲 : Finset (Finset (Fin C.N)))
    (hone : OneReader P 𝒲) : (readersOver P 𝒲 0).card ≤ 𝒲.card := by
  have h := card_readersOver_le P 𝒲 hone 0
  rwa [Nat.zero_add, Nat.one_mul] at h

theorem card_unregistered_untie_of_notMem_readersOver {m : ℕ} (d : RegisteredCircuit) (P : Partition d.D m) {k b w : ℕ}
    (𝒲 : Finset (Finset (Fin d.D.N))) (h𝒲 : ∀ W ∈ 𝒲, W.card ≤ b) (hk : ReadsAtMost d P k) {u : Fin m}
    (hu : u ∉ readersOver P 𝒲 w) : ((d.untie (𝒲.biUnion id)).unregistered (P.gates u)).card ≤ k + b * w :=
  (card_unregistered_untie_words_le d 𝒲 h𝒲 (P.gates u)).trans
    (Nat.add_le_add (hk u)
      (Nat.mul_le_mul_left b (Nat.le_of_not_lt fun h => hu (mem_filter.2 ⟨mem_univ u, h⟩))))

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

theorem kappa_mono_left {C : Circuit} {X X' Y : Finset (Fin C.N)} (h : X ⊆ X') : kappa C X Y ≤ kappa C X' Y := by
  obtain ⟨Z, hZ, hsep⟩ := Slowdown.Cut.SecurityProofs.exists_separator X' Y
  rw [← hZ]
  have hsep' : Separates C Z X Y := fun x hx y hy => hsep x (h hx) y hy
  exact Slowdown.Cut.SecurityProofs.kappa_le hsep'

theorem unregisteredInputs_subset_untie (d : RegisteredCircuit) (E : Finset (Fin d.D.N)) :
    unregisteredInputs d ⊆ unregisteredInputs (d.untie E) := by
  intro g hg
  obtain ⟨hi, hθ⟩ := (mem_filter.1 hg).2
  exact mem_filter.2 ⟨mem_univ g, hi, fun h => hθ (mem_sdiff.1 (show g ∈ d.θ \ E from h)).1⟩

theorem kappaWords_le_untie (d : RegisteredCircuit) (E : Finset (Fin d.D.N)) (S : Finset (Finset (Fin d.D.N))) :
    kappaWords d S ≤ kappaWords (d.untie E) S := by
  show kappa d.D (unregisteredInputs d) (S.biUnion id) ≤ kappa d.D (unregisteredInputs (d.untie E)) (S.biUnion id)
  exact kappa_mono_left (unregisteredInputs_subset_untie d E)

theorem hardness_untie {ρ V : Type} [Fintype ρ] [DecidableEq V] (d : RegisteredCircuit) (E : Finset (Fin d.D.N))
    (A : ρ → Finset (Finset (Fin d.D.N))) (val : Finset (Fin d.D.N) → V) (β : ρ → ℝ) (K : ℕ)
    (h : Assumptions.Hardness d A val β K) : Assumptions.Hardness (d.untie E) A val β K := by
  intro S hS
  have hκ : kappaWords d S ≤ kappaWords (d.untie E) S := kappaWords_le_untie d E S
  exact (h S (hκ.trans hS)).trans (Nat.cast_le.2 hκ)

theorem piMax_untie_le {ι : Type} [Fintype ι] (d : RegisteredCircuit) (E : Finset (Fin d.D.N)) (W : Work d.D ι)
    (K : ℕ) : piMax (d.untie E) W K ≤ piMax d W K := by
  unfold piMax
  exact Finset.sup_mono fun S hS => mem_filter.2 ⟨mem_univ S, (kappaWords_le_untie d E S).trans (mem_filter.1 hS).2⟩

theorem feedsWithin_asInputs {C : Circuit} (T R : Finset (Fin C.N)) {x y : Fin C.N}
    (hp : Slowdown.Covering.FeedsWithin C R x y) :
    Slowdown.Covering.FeedsWithin (C.asInputs T) R x y ∨ ∃ g ∈ T, Slowdown.Covering.FeedsWithin (C.asInputs T) R g y := by
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

theorem unregisteredInputs_subset_asInputs (d : RegisteredCircuit) (T : Finset (Fin d.D.N)) :
    unregisteredInputs d ⊆ unregisteredInputs (d.asInputs T) := by
  intro g hg
  obtain ⟨hi, hθ⟩ := (mem_filter.1 hg).2
  refine mem_filter.2 ⟨mem_univ g, ?_, fun h => hθ (mem_sdiff.1 (show g ∈ d.θ \ T from h)).1⟩
  show (d.D.isInput g || decide (g ∈ T)) = true
  rw [hi, Bool.true_or]

theorem mem_unregisteredInputs_asInputs (d : RegisteredCircuit) (T : Finset (Fin d.D.N)) {g : Fin d.D.N}
    (hg : g ∈ T) : g ∈ unregisteredInputs (d.asInputs T) := by
  refine mem_filter.2 ⟨mem_univ g, ?_, fun h => (mem_sdiff.1 (show g ∈ d.θ \ T from h)).2 hg⟩
  show (d.D.isInput g || decide (g ∈ T)) = true
  rw [decide_eq_true hg, Bool.or_true]

theorem kappaWords_le_asInputs (d : RegisteredCircuit) (T : Finset (Fin d.D.N))
    (S : Finset (Finset (Fin d.D.N))) : kappaWords d S ≤ kappaWords (d.asInputs T) S := by
  obtain ⟨Z, hZ, hsep⟩ := Slowdown.Cut.SecurityProofs.exists_separator (C := (d.asInputs T).D)
    (unregisteredInputs (d.asInputs T)) (S.biUnion id)
  have hZ' : kappaWords (d.asInputs T) S = Z.card := hZ.symm
  rw [hZ']
  have hsep' : Separates d.D Z (unregisteredInputs d) (S.biUnion id) := by
    intro x hx y hy hp
    rcases feedsWithin_asInputs T (univ \ Z) hp with h | ⟨g, hg, h⟩
    · exact hsep x (unregisteredInputs_subset_asInputs d T hx) y hy h
    · exact hsep g (mem_unregisteredInputs_asInputs d T hg) y hy h
  exact Slowdown.Cut.SecurityProofs.kappa_le hsep'

theorem hardness_asInputs {ρ V : Type} [Fintype ρ] [DecidableEq V] (d : RegisteredCircuit) (T : Finset (Fin d.D.N))
    (A : ρ → Finset (Finset (Fin d.D.N))) (val : Finset (Fin d.D.N) → V) (β : ρ → ℝ) (K : ℕ)
    (h : Assumptions.Hardness d A val β K) : Assumptions.Hardness (d.asInputs T) A val β K := by
  intro S hS
  have hκ : kappaWords d S ≤ kappaWords (d.asInputs T) S := kappaWords_le_asInputs d T S
  exact (h S (hκ.trans hS)).trans (Nat.cast_le.2 hκ)

/-- It is not the one stated: a word of `θ` has `κ = 0` at `d`, so `Π` at `d` counts a work that multiplies by an
untied word as if that word cost nothing. -/
theorem counts_untie {ι : Type} [Fintype ι] [DecidableEq ι] {m : ℕ} (d : RegisteredCircuit) (P : Partition d.D m)
    (W : Work d.D ι) {k b : ℕ} (𝒲 : Finset (Finset (Fin d.D.N))) (h𝒲 : ∀ S ∈ 𝒲, S.card ≤ b)
    (hk : ReadsAtMost d P k) (hone : OneReader P 𝒲) (G : Finset (Fin m)) :
    ∃ V F E : ℕ,
      W.N ≤ m * k + b * 𝒲.card + V + ∑ u ∈ Gᶜ, ∑ a, (inBlock P W u a).card ∧
      V ^ 2 ≤ (∑ u ∈ G, piMax (d.untie (𝒲.biUnion id)) W (k + b * (wordsRead d.D 𝒲 (P.gates u)).card)) * F ∧
      F ≤ outOn P W G + E ∧ W.bY * E ≤ (m * k + b * 𝒲.card : ℕ) ∧ (E = 0 ∨ 0 < W.bY) :=
  counts_on (d := d.untie (𝒲.biUnion id)) W P G (fun u => k + b * (wordsRead d.D 𝒲 (P.gates u)).card)
    (fun u _ => (card_unregistered_untie_words_le d 𝒲 h𝒲 (P.gates u)).trans (Nat.add_le_add_right (hk u) _))
    (sum_card_unregistered_untie_le d P 𝒲 h𝒲 hk hone)

/-- It is not what is asked: a word of `θ` has `κ = 0` at `d`, so (H) at `d` fails at every budget and positive cost
for a work that multiplies by an untied word (`untie_hH_false`). -/
theorem bound_untie_he {ι V : Type} [Fintype ι] [DecidableEq ι] [DecidableEq V] {m : ℕ} (d : RegisteredCircuit)
    (P : Partition d.D m) (W : Work d.D ι) (val : Finset (Fin d.D.N) → V) {k b : ℕ} {βw βx e : ℝ}
    (𝒲 : Finset (Finset (Fin d.D.N))) (h𝒲 : ∀ S ∈ 𝒲, S.card ≤ b) (hk : ReadsAtMost d P k) (hone : OneReader P 𝒲)
    (G : Finset (Fin m)) (hw : 0 < βw) (hx : 0 < βx) (he : 0 < e) (hval : ValuesInjective W val)
    (hshare : NoSharedPair W val)
    (hH : ∀ u ∈ G, Assumptions.Hardness (d.untie (𝒲.biUnion id)) ![W.allW, W.allX] val ![βw, βx]
      (k + b * (wordsRead d.D 𝒲 (P.gates u)).card))
    (hE : Assumptions.Exports W e) :
    (W.N : ℝ) ≤ (∑ u ∈ Gᶜ, ∑ a, (inBlock P W u a).card : ℕ) + (m * k + b * 𝒲.card : ℕ) +
      1 / 2 * √((∑ u ∈ G, (k + b * (wordsRead d.D 𝒲 (P.gates u)).card) ^ 2 : ℕ) * outOn P W G / (βw * βx)) +
      1 / 2 * √((∑ u ∈ G, (k + b * (wordsRead d.D 𝒲 (P.gates u)).card) ^ 2 : ℕ) * (m * k + b * 𝒲.card : ℕ) /
        (βw * βx * e)) :=
  bound_on_budgets_he (d := d.untie (𝒲.biUnion id)) (W := W) P G
    (fun u => k + b * (wordsRead d.D 𝒲 (P.gates u)).card) hw hx he hval hshare hH hE
    (fun u _ => (card_unregistered_untie_words_le d 𝒲 h𝒲 (P.gates u)).trans (Nat.add_le_add_right (hk u) _))
    (sum_card_unregistered_untie_le d P 𝒲 h𝒲 hk hone)

theorem freshInput_asInputs_le (d : RegisteredCircuit) (T R : Finset (Fin d.D.N)) :
    freshInput (d.asInputs T) (R \ T) ≤ freshInput d R + d.D.width ((d.D.asInputs T).into (R \ T) ∩ T) := by
  show ∑ g ∈ (d.asInputs T).unregistered (R \ T), d.D.w g ≤
    ∑ g ∈ d.unregistered R, d.D.w g + ∑ g ∈ (d.D.asInputs T).into (R \ T) ∩ T, d.D.w g
  calc ∑ g ∈ (d.asInputs T).unregistered (R \ T), d.D.w g
      ≤ ∑ g ∈ d.unregistered R ∪ (d.D.asInputs T).into (R \ T) ∩ T, d.D.w g :=
        sum_le_sum_of_subset (d.unregistered_asInputs_subset T R)
    _ ≤ ∑ g ∈ d.unregistered R, d.D.w g + ∑ g ∈ (d.D.asInputs T).into (R \ T) ∩ T, d.D.w g :=
        Nat.le.intro sum_union_inter

theorem freshInput_asInputs_words_le (d : RegisteredCircuit) {b : ℕ} (𝒲 : Finset (Finset (Fin d.D.N)))
    (h𝒲 : ∀ W ∈ 𝒲, d.D.width W ≤ b) (R : Finset (Fin d.D.N)) :
    freshInput (d.asInputs (𝒲.biUnion id)) (R \ 𝒲.biUnion id) ≤
      freshInput d R + b * (wordsRead (d.D.asInputs (𝒲.biUnion id)) 𝒲 (R \ 𝒲.biUnion id)).card := by
  have h : d.D.width ((d.D.asInputs (𝒲.biUnion id)).into (R \ 𝒲.biUnion id) ∩ 𝒲.biUnion id) ≤
      b * (wordsRead (d.D.asInputs (𝒲.biUnion id)) 𝒲 (R \ 𝒲.biUnion id)).card :=
    width_into_inter_biUnion_le (d.D.asInputs (𝒲.biUnion id)) 𝒲 h𝒲 (R \ 𝒲.biUnion id)
  exact (freshInput_asInputs_le d (𝒲.biUnion id) R).trans (Nat.add_le_add_left h _)

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

theorem freshInput_asInputs_held_le (d : RegisteredCircuit) (T R : Finset (Fin d.D.N))
    (hT : ∀ g ∈ T, d.D.isInput g = false) :
    freshInput (d.asInputs T) (R \ T) ≤ freshInput d R + d.D.width (R ∩ T) := by
  show ∑ g ∈ (d.asInputs T).unregistered (R \ T), d.D.w g ≤
    ∑ g ∈ d.unregistered R, d.D.w g + ∑ g ∈ R ∩ T, d.D.w g
  calc ∑ g ∈ (d.asInputs T).unregistered (R \ T), d.D.w g
      ≤ ∑ g ∈ d.unregistered R ∪ R ∩ T, d.D.w g :=
        sum_le_sum_of_subset (unregistered_asInputs_subset_held d T R hT)
    _ ≤ ∑ g ∈ d.unregistered R, d.D.w g + ∑ g ∈ R ∩ T, d.D.w g := Nat.le.intro sum_union_inter

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

theorem readsAtMost_asInputs_held {m : ℕ} (d : RegisteredCircuit) (P : Partition d.D m) {k h : ℕ}
    (T : Finset (Fin d.D.N)) (hT : ∀ g ∈ T, d.D.isInput g = false) (hk : ReadsAtMost d P k)
    (hh : ∀ u, (T ∩ P.gates u).card ≤ h) : ReadsAtMost (d.asInputs T) (P.asInputs T) (k + h) := fun u =>
  (card_unregistered_asInputs_held_le d P T hT u).trans (Nat.add_le_add (hk u) (hh u))

theorem sum_card_unregistered_asInputs_held_le {m : ℕ} (d : RegisteredCircuit) (P : Partition d.D m) {k : ℕ}
    (T : Finset (Fin d.D.N)) (hT : ∀ g ∈ T, d.D.isInput g = false) (hk : ReadsAtMost d P k) :
    ∑ u, ((d.asInputs T).unregistered ((P.asInputs T).gates u)).card ≤ m * k + T.card :=
  calc ∑ u, ((d.asInputs T).unregistered ((P.asInputs T).gates u)).card
      ≤ ∑ u : Fin m, (k + (T ∩ P.gates u).card) := sum_le_sum fun u _ =>
        (card_unregistered_asInputs_held_le d P T hT u).trans (Nat.add_le_add_right (hk u) _)
    _ = m * k + ∑ u : Fin m, (T ∩ P.gates u).card := by
        rw [sum_add_distrib, sum_const, card_univ, Fintype.card_fin, smul_eq_mul]
    _ ≤ m * k + T.card := Nat.add_le_add_left (Slowdown.Cut.SecurityProofs.sum_card_inter_gates_le P T) _

theorem eq_of_feeds_of_reads_eq_empty {C : Circuit} {g t : Fin C.N} (hf : C.Feeds g t) (ht : C.reads t = ∅) :
    g = t := by
  induction hf with
  | refl _ => rfl
  | step hr _ ih =>
    rw [ih ht, ht] at hr
    exact absurd hr (notMem_empty _)

theorem y_not_subset_asInputs {C : Circuit} {ι : Type} {T : Finset (Fin C.N)} (W : Work (C.asInputs T) ι)
    (s : W.Mac) : ¬ W.y (W.out s) ⊆ T := by
  intro hsub
  obtain ⟨h, hh, a, ha, hf⟩ := (W.M s.1).feeds_y s.2
  have haT : a ∈ T := hsub ha
  have hreads : (C.asInputs T).reads a = ∅ := by
    show (if a ∈ T then ∅ else C.reads a) = ∅
    exact ite_eq_left haT
  have hcomp : (C.isInput h || decide (h ∈ T)) = false := (W.M s.1).mac_computed s.2 h hh
  rw [eq_of_feeds_of_reads_eq_empty hf hreads, decide_eq_true haT, Bool.or_true] at hcomp
  cases hcomp

theorem counts_asInputs {ι : Type} [Fintype ι] [DecidableEq ι] {m : ℕ} (d : RegisteredCircuit) (P : Partition d.D m)
    {k : ℕ} (T : Finset (Fin d.D.N)) (hT : ∀ g ∈ T, d.D.isInput g = false) (W : Work (d.D.asInputs T) ι)
    (hk : ReadsAtMost d P k) (G : Finset (Fin m)) :
    ∃ V F E : ℕ,
      W.N ≤ m * k + T.card + V + ∑ u ∈ Gᶜ, ∑ a, (inBlock (P.asInputs T) W u a).card ∧
      V ^ 2 ≤ (∑ u ∈ G, piMax (d.asInputs T) W (k + (T ∩ P.gates u).card)) * F ∧
      F ≤ outOn (P.asInputs T) W G + E ∧ W.bY * E ≤ (m * k + T.card : ℕ) ∧ (E = 0 ∨ 0 < W.bY) :=
  counts_on (d := d.asInputs T) W (P.asInputs T) G (fun u => k + (T ∩ P.gates u).card)
    (fun u _ => (card_unregistered_asInputs_held_le d P T hT u).trans (Nat.add_le_add_right (hk u) _))
    (sum_card_unregistered_asInputs_held_le d P T hT hk)

/-- It is not what is asked: a word `d` computes from `θ` alone has `κ = 0` at `d`, so (H) at `d` fails at every
budget and positive cost for a work that multiplies by such a word (`asInputs_hH_false`). -/
theorem bound_asInputs_he {ι V : Type} [Fintype ι] [DecidableEq ι] [DecidableEq V] {m : ℕ} (d : RegisteredCircuit)
    (P : Partition d.D m) {k : ℕ} {βw βx e : ℝ} (T : Finset (Fin d.D.N)) (hT : ∀ g ∈ T, d.D.isInput g = false)
    (W : Work (d.D.asInputs T) ι) (val : Finset (Fin d.D.N) → V) (hk : ReadsAtMost d P k) (G : Finset (Fin m))
    (hw : 0 < βw) (hx : 0 < βx) (he : 0 < e) (hval : ValuesInjective W val) (hshare : NoSharedPair W val)
    (hH : ∀ u ∈ G, Assumptions.Hardness (d.asInputs T) ![W.allW, W.allX] val ![βw, βx] (k + (T ∩ P.gates u).card))
    (hE : Assumptions.Exports W e) :
    (W.N : ℝ) ≤ (∑ u ∈ Gᶜ, ∑ a, (inBlock (P.asInputs T) W u a).card : ℕ) + (m * k + T.card : ℕ) +
      1 / 2 * √((∑ u ∈ G, (k + (T ∩ P.gates u).card) ^ 2 : ℕ) * outOn (P.asInputs T) W G / (βw * βx)) +
      1 / 2 * √((∑ u ∈ G, (k + (T ∩ P.gates u).card) ^ 2 : ℕ) * (m * k + T.card : ℕ) / (βw * βx * e)) :=
  bound_on_budgets_he (d := d.asInputs T) (W := W) (P.asInputs T) G (fun u => k + (T ∩ P.gates u).card) hw hx he hval
    hshare hH hE
    (fun u _ => (card_unregistered_asInputs_held_le d P T hT u).trans (Nat.add_le_add_right (hk u) _))
    (sum_card_unregistered_asInputs_held_le d P T hT hk)

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

theorem exists_kept {m : ℕ} (S : Finset (Fin m)) :
    ∃ ι : Fin (m - S.card) → Fin m, Function.Injective ι ∧ ∀ v, ι v ∉ S := by
  have h : Sᶜ.card = m - S.card := by rw [card_compl, Fintype.card_fin]
  exact ⟨Sᶜ.orderEmbOfFin h, (Sᶜ.orderEmbOfFin h).injective, fun v => mem_compl.1 (Sᶜ.orderEmbOfFin_mem h v)⟩

theorem readsAtMost_untie_without {m : ℕ} (d : RegisteredCircuit) (P : Partition d.D m) {k b : ℕ}
    (𝒲 : Finset (Finset (Fin d.D.N))) (h𝒲 : ∀ W ∈ 𝒲, W.card ≤ b) (hk : ReadsAtMost d P k) (w : ℕ) :
    ∃ (ι : Fin (m - (readersOver P 𝒲 w).card) → Fin m) (hι : Function.Injective ι), (∀ v, ι v ∉ readersOver P 𝒲 w) ∧
      ReadsAtMost ((d.untie (𝒲.biUnion id)).asInputs (P.dropped ι)) (P.keep ι hι) (k + b * w) := by
  obtain ⟨ι, hι, hS⟩ := exists_kept (readersOver P 𝒲 w)
  exact ⟨ι, hι, hS, readsAtMost_keep (d.untie (𝒲.biUnion id)) P ι hι fun v =>
    card_unregistered_untie_of_notMem_readersOver d P 𝒲 h𝒲 hk (hS v)⟩

end Slowdown.Cut.SecurityProofs.Untied
