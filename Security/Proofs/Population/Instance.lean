import Proofs.Population.Determinacy
import Proofs.Core.Circuit
import Definitions.Population.Partition

/-!
# Sampled proofs' partitions as populations

* `singleAssignment`: a partition's population is always single assignment: every computed gate is in exactly one unit,
  and input gates in none.
* `checked_of_ordered`: when no wire runs from a later unit into an earlier one (`Ordered`), it is acyclic too, so
  `Checked`, and every theorem of `Determinacy`, `Necessity`, `Egress` and `Work` applies to it.
* `ordered_of_inLayoutOrder`: units that are intervals of gates numbered in layout order, as `Q_call` v1 cuts them, are
  ordered.
* `wrong_eq`, **the bridge**: with every value carried in `W` bits, a unit is wrong in the population exactly when it is
  wrong in the partition (`correct_iff`), so the population's hypothesis `|wrong| ≤ m` is sampled proofs' guarantee.
* `not_acyclic_of_sandwich`, **what fails**: a unit holding a gate before and a gate after a gate of another unit, on
  one path of wires, makes a cycle. `Q_word` v1 cuts such units: it packs a Call's returned gates into output units in
  return order, so a chain `g₁ → g₂ → g₃` returned as `[g₁, g₃, g₂]` at 8 bits a gate and `X = 16` puts `g₁` and `g₃` in
  one unit and `g₂` in another, and `check_cut` accepts it. Integrity survives there only through the circuit's gate
  order (`Partition.compose_cone`, which holds for every partition); an opaque unit with the same dataflow would not
  have it (`Necessity.cycle_two_fixed_points`).
-/

namespace Verity.Partition

open Finset

variable {C : Circuit} {n : ℕ} (P : Partition C n)

theorem mem_input {s : P.Slot} : s ∈ P.toDataflow.input ↔ C.isInput s.1 = true := by
  simp [toDataflow]

theorem mem_writes {u : Fin n} {s : P.Slot} : s ∈ P.toDataflow.writes u ↔ P.unit s.1 = some u := by
  simp [toDataflow]

theorem mem_reads {u : Fin n} {s : P.Slot} :
    s ∈ P.toDataflow.reads u ↔ P.unit s.1 ≠ some u ∧ ∃ h, P.unit h = some u ∧ s.1 ∈ C.reads h := by
  simp [toDataflow, Circuit.into, mem_gates]

/-- **A partition's population is single assignment.** -/
theorem singleAssignment : P.toDataflow.SingleAssignment := by
  intro s
  refine ⟨fun hs u hu => ?_, fun hs => ?_⟩
  · rw [mem_input] at hs
    rw [mem_writes, (P.input_iff s.1).2 hs] at hu
    cases hu
  · rw [mem_input] at hs
    obtain ⟨u, hu⟩ := Option.ne_none_iff_exists'.1 fun h => hs ((P.input_iff s.1).1 h)
    exact ⟨u, (P.mem_writes).2 hu, fun v hv => Option.some_injective _ (((P.mem_writes).1 hv).symm.trans hu)⟩

theorem dep_lt (ho : P.Ordered) {u v : Fin n} (h : P.toDataflow.Dep u v) : u < v := by
  obtain ⟨s, hsu, hsv⟩ := h
  rw [mem_writes] at hsu
  obtain ⟨hne, g, hg, hsg⟩ := (P.mem_reads).1 hsv
  refine lt_of_le_of_ne (ho s.1 g u v hsg hsu hg) ?_
  rintro rfl
  exact hne hsu

/-- **An ordered partition's population is acyclic.** -/
theorem acyclic_of_ordered (ho : P.Ordered) : P.toDataflow.Acyclic := by
  have hlt : ∀ a b, Relation.TransGen P.toDataflow.Dep a b → a < b := fun a b h => by
    induction h with
    | single h => exact P.dep_lt ho h
    | tail _ h ih => exact ih.trans (P.dep_lt ho h)
  exact fun u hu => lt_irrefl u (hlt u u hu)

/-- **An ordered partition's population is checked.** -/
theorem checked_of_ordered (ho : P.Ordered) : P.toDataflow.Checked :=
  ⟨P.singleAssignment, P.acyclic_of_ordered ho⟩

/-- **Units in layout order are ordered**: a gate reads only earlier gates. -/
theorem ordered_of_inLayoutOrder (hl : P.InLayoutOrder) : P.Ordered :=
  fun g h u v hg hu hv => hl g h u v (C.reads_lt h g hg).le hu hv

/-- A unit's local evaluation of its own gates reads, of the transcript, only in(R). -/
theorem localEval_congr_into {X Y : C.Transcript} {u : Fin n} (hXY : ∀ g ∈ C.into (P.gates u), X g = Y g) :
    ∀ m (g : Fin C.N), g.val = m → P.unit g = some u → P.localEval X u g = P.localEval Y u g := by
  intro m
  induction m using Nat.strong_induction_on with
  | _ m ih =>
  intro g hgm hu
  rw [P.localEval_of_mem X hu, P.localEval_of_mem Y hu]
  refine C.op_congr g _ _ fun x hx => ?_
  by_cases hxu : P.unit x = some u
  · exact ih x.val (hgm ▸ C.reads_lt g x hx) x rfl hxu
  · rw [P.localEval_of_not_mem X hxu, P.localEval_of_not_mem Y hxu]
    refine hXY x ?_
    simp only [Circuit.into, mem_filter, mem_univ, true_and, mem_gates]
    exact ⟨hxu, g, hu, hx⟩

/-- The population's unit function, on a transcript's committed values, is the partition's local evaluation. -/
theorem fns_encode {W : ℕ} (hW : ∀ g, C.w g ≤ W) (X : C.Transcript) (u : Fin n) (s : P.toDataflow.writes u) :
    P.fns W u (fun r => P.encode W X r) s = (P.localEval X u s.1.1).setWidth W := by
  unfold fns
  congr 1
  refine P.localEval_congr_into (fun g hg => ?_) _ _ rfl ((P.mem_writes).1 s.2)
  have hg' := hg
  simp only [Circuit.into, mem_filter, mem_univ, true_and, mem_gates] at hg'
  obtain ⟨hgu, h, hh, hgh⟩ := hg'
  have hc : ∃ hc : g ∈ P.committed, (⟨g, hc⟩ : P.Slot) ∈ P.toDataflow.reads u :=
    ⟨P.mem_committed_of_read hgh (by rw [hh]; exact fun e => hgu e.symm), mem_filter.2 ⟨mem_univ _, hg⟩⟩
  rw [dite_eq_left hc]
  simp [encode, hW g]

/-- **A unit is correct in the population exactly when it is correct in the partition.** -/
theorem correct_iff {W : ℕ} (hW : ∀ g, C.w g ≤ W) (X : C.Transcript) (u : Fin n) :
    P.toDataflow.Correct (P.fns W) (P.encode W X) u ↔ P.Correct X u := by
  have hinj : ∀ g (x y : BitVec (C.w g)), x.setWidth W = y.setWidth W → x = y := fun g x y h => by
    simpa [hW g] using congrArg (BitVec.setWidth (C.w g)) h
  constructor
  · intro h g hg hu
    have := h ⟨⟨g, hg⟩, (P.mem_writes).2 hu⟩
    rw [P.fns_encode hW X u] at this
    exact hinj g _ _ this
  · intro h s
    rw [P.fns_encode hW X u]
    exact congrArg (BitVec.setWidth W) (h s.1.1 s.1.2 ((P.mem_writes).1 s.2))

/-- **The bridge**: the population's wrong units are the partition's. -/
theorem wrong_eq {W : ℕ} (hW : ∀ g, C.w g ≤ W) (X : C.Transcript) :
    P.toDataflow.wrong (P.fns W) (P.encode W X) = P.wrong X := by
  ext u
  simp only [Population.Dataflow.wrong, mem_filter, mem_univ, true_and, P.mem_wrong, P.correct_iff hW X u]

/-- **A sandwich is a cycle**: if a unit `u` holds gates `g` and `k`, another unit `v` holds `h`, and `g` is read by
`h`, which `k` reads, the population is not acyclic. -/
theorem not_acyclic_of_sandwich {g h k : Fin C.N} {u v : Fin n} (hgh : g ∈ C.reads h) (hhk : h ∈ C.reads k)
    (hg : P.unit g = some u) (hk : P.unit k = some u) (hh : P.unit h = some v) (huv : u ≠ v) :
    ¬ P.toDataflow.Acyclic := by
  have hne : (some u : Option (Fin n)) ≠ some v := fun e => huv (Option.some_injective _ e)
  have hgc : g ∈ P.committed := P.mem_committed_of_read hgh (by rw [hh, hg]; exact hne.symm)
  have hhc : h ∈ P.committed := P.mem_committed_of_read hhk (by rw [hk, hh]; exact hne)
  have d₁ : P.toDataflow.Dep u v :=
    ⟨⟨g, hgc⟩, (P.mem_writes).2 hg, (P.mem_reads).2 ⟨by rw [hg]; exact hne, h, hh, hgh⟩⟩
  have d₂ : P.toDataflow.Dep v u :=
    ⟨⟨h, hhc⟩, (P.mem_writes).2 hh, (P.mem_reads).2 ⟨by rw [hh]; exact hne.symm, k, hk, hhk⟩⟩
  exact fun hac => hac u (.tail (.single d₁) d₂)

end Verity.Partition

namespace Verity.Partition.Sandwich

open Finset

/-- A chain of 8-bit gates: gate `0` is the input and gate `i + 1` copies gate `i`; gates `1, 2, 3` are returned. -/
def chain : Circuit where
  N := 4
  w := fun _ => 8
  isInput := fun g => g == 0
  reads := fun g => if g = 0 then ∅ else {g - 1}
  reads_lt := by decide
  input_reads := by decide
  op := fun g v => if g = 0 then 0 else v (g - 1)
  op_congr := by
    intro g v v' hv
    by_cases h : g = 0
    · simp [h]
    · simp only [h, ite_false]
      exact hv _ (by simp [h])
  outputs := {1, 2, 3}

/-- `Q_word` v1's cut of the chain returned as `[g₁, g₃, g₂]` at `X = 16`: `g₁` and `g₃` in unit `0`, `g₂` in unit
`1`. -/
def qword : Partition chain 2 where
  unit := fun (g : Fin 4) => if g = 0 then none else if g = 2 then some 1 else some 0
  input_iff := by decide

/-- **`Q_word` v1's cut has a cycle.** -/
theorem qword_not_acyclic : ¬ qword.toDataflow.Acyclic :=
  qword.not_acyclic_of_sandwich (g := (1 : Fin 4)) (h := (2 : Fin 4)) (k := (3 : Fin 4)) (u := 0) (v := 1)
    (show (1 : Fin 4) ∈ (if (2 : Fin 4) = 0 then ∅ else {2 - 1} : Finset (Fin 4)) by decide)
    (show (2 : Fin 4) ∈ (if (3 : Fin 4) = 0 then ∅ else {3 - 1} : Finset (Fin 4)) by decide) rfl rfl rfl
    (by decide)

end Verity.Partition.Sandwich
