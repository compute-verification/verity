module

public import Definitions.Core.Guarantees.Circuit
import Mathlib.Algebra.BigOperators.Group.Finset.Basic

/-!
# Lemmas about circuits and partitions

Evaluation unfolds one gate at a time (`run_eq`), and the partition lemmas Flock's audit uses, for gates of any
width:

* `compose`: if every unit is correct and the committed inputs are the anchors' values, every committed wire carries its
  value in the circuit's evaluation;
* `compose_cone`: the same for the wires `T` a consumer reads, given only that the units of `T`'s cone are correct;
* `correct_congr`: a unit's correctness depends only on its committed inputs and outputs (`Partition.io`);
* `Refines.exists_incorrect_fine`: when a fine partition refines a coarse one, a incorrect coarse unit has a incorrect fine unit
  inside it, whatever the values on the coarse unit's interior;
* `Refines.glue`: fine traces that each agree with one coarse trace on its opened wires splice into one that does
  and has, inside each coarse unit, that unit's trace's incorrect fine units;
* `Containment.implementation`: a circuit that contains `C` implements it.
-/

@[expose] public section

namespace Verity

open Finset

namespace Circuit

variable (C : Circuit)

theorem run_eq (fix : (g : Fin C.N) → Option (BitVec (C.w g))) (g : Fin C.N) :
    C.run fix g = match fix g with
      | some b => b
      | none => C.op g (C.run fix) := by
  rw [run]
  cases fix g with
  | some b => rfl
  | none => exact C.op_congr g _ _ fun x hx => by simp [C.reads_lt g x hx]

theorem run_fixed {fix : (g : Fin C.N) → Option (BitVec (C.w g))} {g : Fin C.N} {b : BitVec (C.w g)}
    (h : fix g = some b) : C.run fix g = b := by
  rw [run_eq, h]

theorem run_free {fix : (g : Fin C.N) → Option (BitVec (C.w g))} {g : Fin C.N} (h : fix g = none) :
    C.run fix g = C.op g (C.run fix) := by
  rw [run_eq, h]

theorem eval_input (a : C.Trace) {g : Fin C.N} (h : C.isInput g = true) : C.eval a g = a g :=
  C.run_fixed (by simp [h])

theorem eval_op (a : C.Trace) {g : Fin C.N} (h : C.isInput g = false) :
    C.eval a g = C.op g (C.eval a) :=
  C.run_free (by simp [h])

/-- The honest trace agrees with its inputs. -/
theorem inputsAgree_eval (a : C.Trace) : C.InputsAgree (C.eval a) a :=
  fun _ h => C.eval_input a h

/-- The evaluation reads only the inputs. -/
theorem eval_congr {a b : C.Trace} (h : C.InputsAgree a b) : C.eval a = C.eval b := by
  funext g
  induction g using WellFoundedLT.induction with
  | _ g ih => ?_
  cases hi : C.isInput g
  · rw [C.eval_op a hi, C.eval_op b hi]
    exact C.op_congr g _ _ fun x hx => ih x (C.reads_lt g x hx)
  · rw [C.eval_input a hi, C.eval_input b hi, h g hi]

end Circuit

namespace Partition

variable {C : Circuit} {n : ℕ} (P : Partition C n)

theorem mem_gates {u : Fin n} {g : Fin C.N} : g ∈ P.gates u ↔ P.unit g = some u := by
  simp [gates]

/-- The units' gates are the computed gates, each in exactly one unit. -/
theorem sum_card_gates : ∑ u, (P.gates u).card = C.computed.card := by
  classical
  rw [← Finset.card_biUnion]
  · congr 1
    ext g
    simp only [Finset.mem_biUnion, Finset.mem_univ, true_and, mem_gates, Circuit.computed, Finset.mem_filter]
    rw [← Bool.not_eq_true, ← P.input_iff]
    exact Option.ne_none_iff_exists'.symm
  · intro u _ v _ huv
    refine Finset.disjoint_left.2 fun g hu hv => huv ?_
    rw [mem_gates] at hu hv
    exact Option.some_injective _ (hu.symm.trans hv)

/-- A unit's gates are computed gates. -/
theorem gates_subset_computed (u : Fin n) : P.gates u ⊆ C.computed := fun g hg => by
  simp only [Circuit.computed, Finset.mem_filter, Finset.mem_univ, true_and]
  cases h : C.isInput g
  · rfl
  · rw [← P.input_iff] at h
    rw [P.mem_gates, h] at hg
    cases hg

theorem mem_committed_of_read {g h : Fin C.N} (hg : g ∈ C.reads h) (hne : P.unit h ≠ P.unit g) :
    g ∈ P.committed := by
  simp only [committed, Finset.mem_filter, Finset.mem_univ, true_and]
  exact Or.inr (Or.inr ⟨h, hg, hne⟩)

theorem mem_committed_of_output {g : Fin C.N} (hg : g ∈ C.outputs) : g ∈ P.committed := by
  simp only [committed, Finset.mem_filter, Finset.mem_univ, true_and]
  exact Or.inr (Or.inl hg)

theorem mem_committed_of_input {g : Fin C.N} (hg : C.isInput g = true) : g ∈ P.committed := by
  simp only [committed, Finset.mem_filter, Finset.mem_univ, true_and]
  exact Or.inl hg

/-- Unit `u`'s committed gates are out(R) for its gates `R`. -/
theorem committed_inter_gates (u : Fin n) : P.committed ∩ P.gates u = C.exits (P.gates u) := by
  ext g
  simp only [Finset.mem_inter, committed, Circuit.exits, Finset.mem_filter, Finset.mem_univ, true_and, mem_gates]
  constructor
  · rintro ⟨hc | hc | ⟨h, hgh, hne⟩, hu⟩
    · exact absurd ((P.input_iff g).2 hc) (by simp [hu])
    · exact ⟨hu, Or.inl hc⟩
    · exact ⟨hu, Or.inr ⟨h, fun hh => hne (by rw [hu, hh]), hgh⟩⟩
  · rintro ⟨hu, hc | ⟨h, hh, hgh⟩⟩
    · exact ⟨Or.inr (Or.inl hc), hu⟩
    · exact ⟨Or.inr (Or.inr ⟨h, hgh, fun he => hh (by rw [he, hu])⟩), hu⟩

theorem localEval_of_mem (X : C.Trace) {u : Fin n} {g : Fin C.N} (h : P.unit g = some u) :
    P.localEval X u g = C.op g (P.localEval X u) :=
  C.run_free (by simp [h])

theorem localEval_of_not_mem (X : C.Trace) {u : Fin n} {g : Fin C.N} (h : P.unit g ≠ some u) :
    P.localEval X u g = X g :=
  C.run_fixed (by simp [h])

/-- **A unit's local evaluation is its restricted circuit's run**: `localEval X u` runs `C_S`, for `S` the unit's
gates, with every gate outside the unit held at `X`. -/
theorem localEval_eq_restrict_run (X : C.Trace) (u : Fin n) :
    P.localEval X u = (C.restrict (P.gates u)).run fun g => if P.unit g = some u then none else some (X g) := by
  funext g
  induction g using WellFoundedLT.induction with
  | _ g ih =>
  by_cases hu : P.unit g = some u
  · have hin : C.isInput g = false := by
      cases hi : C.isInput g
      · rfl
      · exact absurd ((P.input_iff g).2 hi) (by simp [hu])
    have hS : g ∈ P.gates u ∧ C.isInput g = false := ⟨by simp [gates, hu], hin⟩
    rw [P.localEval_of_mem X hu]
    refine Eq.trans ?_ ((C.restrict (P.gates u)).run_free
      (fix := fun g => if P.unit g = some u then none else some (X g)) (g := g) (by simp [hu])).symm
    show _ = (if g ∈ P.gates u ∧ C.isInput g = false then C.op g _ else 0)
    rw [if_pos hS]
    exact C.op_congr g _ _ fun a ha => ih a (C.reads_lt g a ha)
  · rw [P.localEval_of_not_mem X hu]
    exact ((C.restrict (P.gates u)).run_fixed
      (fix := fun g => if P.unit g = some u then none else some (X g)) (g := g) (b := X g) (by simp [hu])).symm

theorem mem_incorrect {X : C.Trace} {u : Fin n} : u ∈ P.incorrect X ↔ ¬ P.Correct X u := by
  classical
  simp [incorrect]

theorem mem_boundary {g : Fin C.N} :
    g ∈ P.boundary ↔ g ∈ P.committed ∧ (C.isInput g = false ∨ g ∈ C.outputs ∨ ∃ h, g ∈ C.reads h) := by
  classical
  simp only [boundary, io, Finset.mem_union, Finset.mem_biUnion, Finset.mem_univ, true_and, Finset.mem_filter]
  constructor
  · rintro (⟨u, hc, hu | ⟨h, -, hgh⟩⟩ | ho)
    · refine ⟨hc, Or.inl ?_⟩
      cases hi : C.isInput g
      · rfl
      · exact absurd ((P.input_iff g).2 hi) (by simp [hu])
    · exact ⟨hc, Or.inr (Or.inr ⟨h, hgh⟩)⟩
    · exact ⟨P.mem_committed_of_output ho, Or.inr (Or.inl ho)⟩
  · rintro ⟨hc, hi | ho | ⟨h, hgh⟩⟩
    · obtain ⟨u, hu⟩ := Option.ne_none_iff_exists'.1 fun hn => by simp [(P.input_iff g).1 hn] at hi
      exact Or.inl ⟨u, hc, Or.inl hu⟩
    · exact Or.inr ho
    · have hh : C.isInput h = false := by
        cases hi : C.isInput h
        · rfl
        · simp [C.input_reads h hi] at hgh
      obtain ⟨u, hu⟩ := Option.ne_none_iff_exists'.1 fun hn => by simp [(P.input_iff h).1 hn] at hh
      exact Or.inl ⟨u, hc, Or.inr ⟨h, hu, hgh⟩⟩

theorem boundary_subset_committed : P.boundary ⊆ P.committed := fun _ hg => (P.mem_boundary.1 hg).1

theorem before_irrefl (g : Fin C.N) : ¬ P.Before g g := by
  rintro (h | ⟨-, h⟩) <;> exact lt_irrefl _ h

theorem before_trans {g h k : Fin C.N} (hgh : P.Before g h) (hhk : P.Before h k) : P.Before g k := by
  unfold Before at *
  rcases hgh with h1 | ⟨h1, h2⟩ <;> rcases hhk with h3 | ⟨h3, h4⟩
  · exact Or.inl (by omega)
  · exact Or.inl (by omega)
  · exact Or.inl (by omega)
  · exact Or.inr ⟨by omega, lt_trans h2 h4⟩

theorem before_total {g h : Fin C.N} (hne : g ≠ h) : P.Before g h ∨ P.Before h g := by
  unfold Before
  rcases lt_trichotomy (P.writerRank g) (P.writerRank h) with h1 | h1 | h1
  · exact Or.inl (Or.inl h1)
  · rcases lt_or_gt_of_ne hne with h2 | h2
    · exact Or.inl (Or.inr ⟨h1, h2⟩)
    · exact Or.inr (Or.inr ⟨h1.symm, h2⟩)
  · exact Or.inr (Or.inl h1)

/-- **A position is defined exactly on the boundary.** -/
theorem pos_some {L : Layout C} {g : Fin C.N} : (P.pos L g).isSome ↔ g ∈ P.boundary := by
  unfold pos
  split_ifs with h
  · simp only [h, iff_true]
    split <;> rfl
  · simp [h]

theorem rank_lt_of_before {T : Finset (Fin C.N)} {g h : Fin C.N} (hg : g ∈ T) (hb : P.Before g h) :
    (T.filter fun k => P.Before k g).card < (T.filter fun k => P.Before k h).card := by
  apply Finset.card_lt_card
  refine Finset.ssubset_iff_of_subset (fun k hk => ?_) |>.2 ⟨g, ?_, ?_⟩
  · rw [Finset.mem_filter] at hk ⊢
    exact ⟨hk.1, P.before_trans hk.2 hb⟩
  · exact Finset.mem_filter.2 ⟨hg, hb⟩
  · exact fun hk => P.before_irrefl g (Finset.mem_filter.1 hk).2

/-- **No two boundary gates share a position**, for every circuit, partition and layout. -/
theorem pos_inj {L : Layout C} {g h : Fin C.N} (hg : (P.pos L g).isSome) (he : P.pos L g = P.pos L h) : g = h := by
  have hgb := P.pos_some.1 hg
  have hhb : h ∈ P.boundary := P.pos_some.1 (he ▸ hg)
  unfold pos at he
  rw [if_pos hgb, if_pos hhb] at he
  rcases hrg : L.reg g with _ | ⟨i, r⟩ <;> rcases hrh : L.reg h with _ | ⟨i', r'⟩ <;>
    simp only [hrg, hrh, Option.some.injEq, Prod.mk.injEq, Root.phase.injEq, Root.reg.injEq, reduceCtorEq,
      false_and] at he
  · obtain ⟨hj, hc⟩ := he
    by_contra hne
    have hgw : g ∈ P.window L (L.phase g) := Finset.mem_filter.2 ⟨hgb, hrg, rfl⟩
    have hhw : h ∈ P.window L (L.phase g) := Finset.mem_filter.2 ⟨hhb, hrh, hj.symm⟩
    rw [← hj] at hc
    rcases P.before_total hne with hb | hb
    · exact absurd hc (Nat.ne_of_lt (P.rank_lt_of_before hgw hb))
    · exact absurd hc.symm (Nat.ne_of_lt (P.rank_lt_of_before hhw hb))
  · exact L.reg_inj g h (by simp [hrg]) (by rw [hrg, hrh, he.1, he.2])

theorem io_subset_boundary (u : Fin n) : P.io u ⊆ P.boundary := by
  classical
  intro g hg
  simp only [boundary, Finset.mem_union, Finset.mem_biUnion, Finset.mem_univ, true_and]
  exact Or.inl ⟨u, hg⟩

/-- **A reader's input is its writer's output, at one position**: a gate of unit `w` that unit `r` reads as an input is
also among `w`'s committed outputs, and both open it at the one position `pos g`, which every gate a unit opens has. -/
theorem mem_io_writer {L : Layout C} {g : Fin C.N} {r w : Fin n} (hr : g ∈ P.io r) (hw : P.unit g = some w) :
    g ∈ P.io w ∧ ∃ p, P.pos L g = some p := by
  classical
  refine ⟨?_, ?_⟩
  · simp only [io, Finset.mem_filter] at hr ⊢
    exact ⟨hr.1, Or.inl hw⟩
  · have hb := P.io_subset_boundary r hr
    exact Option.isSome_iff_exists.1 (P.pos_some.2 hb)

/-- A proof block's in(S) and out(S) are its committed inputs and outputs, `io u`, so each has a position. -/
theorem into_union_exits_eq_io (u : Fin n) : C.into (P.gates u) ∪ C.exits (P.gates u) = P.io u := by
  classical
  ext g
  constructor
  · intro hg
    rcases Finset.mem_union.1 hg with h | h
    · simp only [Circuit.into, Finset.mem_filter, Finset.mem_univ, true_and] at h
      obtain ⟨hgS, k, hk, hgk⟩ := h
      have hku := P.mem_gates.1 hk
      simp only [io, Finset.mem_filter]
      refine ⟨P.mem_committed_of_read hgk fun he => hgS (P.mem_gates.2 (he ▸ hku)), Or.inr ⟨k, hku, hgk⟩⟩
    · rw [← P.committed_inter_gates u, Finset.mem_inter] at h
      simp only [io, Finset.mem_filter]
      exact ⟨h.1, Or.inl (P.mem_gates.1 h.2)⟩
  · intro hg
    simp only [io, Finset.mem_filter] at hg
    obtain ⟨hc, hu | ⟨k, hku, hgk⟩⟩ := hg
    · rw [Finset.mem_union, ← P.committed_inter_gates u, Finset.mem_inter]
      exact Or.inr ⟨hc, P.mem_gates.2 hu⟩
    · by_cases hgS : g ∈ P.gates u
      · rw [Finset.mem_union, ← P.committed_inter_gates u, Finset.mem_inter]
        exact Or.inr ⟨hc, hgS⟩
      · refine Finset.mem_union.2 (Or.inl ?_)
        simp only [Circuit.into, Finset.mem_filter, Finset.mem_univ, true_and]
        exact ⟨hgS, k, P.mem_gates.2 hku, hgk⟩

/-- **`C_S`'s run is the block's correctness**: out(S) holds what `C_S` computes exactly when the block is correct. -/
theorem restrict_run_iff_correct (X : C.Trace) (u : Fin n) :
    (∀ g ∈ C.exits (P.gates u),
      X g = (C.restrict (P.gates u)).run (fun a => if P.unit a = some u then none else some (X a)) g) ↔
      P.Correct X u := by
  rw [← P.localEval_eq_restrict_run X u, ← P.committed_inter_gates u]
  simp only [Correct, Finset.mem_inter, mem_gates]
  constructor
  · intro h g hc hu
    exact h g ⟨hc, hu⟩
  · rintro h g ⟨hc, hu⟩
    exact h g hc hu

/-- **How the boundary and the committed wires differ**: exactly by the input gates that no gate reads and that are
not outputs. -/
theorem mem_committed_sdiff_boundary {g : Fin C.N} :
    g ∈ P.committed \ P.boundary ↔ C.isInput g = true ∧ g ∉ C.outputs ∧ ∀ h, g ∉ C.reads h := by
  rw [Finset.mem_sdiff, P.mem_boundary]
  constructor
  · rintro ⟨hc, hb⟩
    refine ⟨?_, fun ho => hb ⟨hc, Or.inr (Or.inl ho)⟩, fun h hgh => hb ⟨hc, Or.inr (Or.inr ⟨h, hgh⟩)⟩⟩
    cases hi : C.isInput g
    · exact absurd ⟨hc, Or.inl hi⟩ hb
    · rfl
  · rintro ⟨hi, ho, hr⟩
    refine ⟨by simp [committed, hi], ?_⟩
    rintro ⟨-, hi' | ho' | ⟨h, hgh⟩⟩
    · simp [hi] at hi'
    · exact ho ho'
    · exact hr h hgh

theorem mem_cone {T : Finset (Fin C.N)} {u : Fin n} {t g : Fin C.N} (ht : t ∈ T) (hg : C.Feeds g t)
    (hu : P.unit g = some u) : u ∈ P.cone T := by
  classical
  simp only [cone, Finset.mem_filter, Finset.mem_univ, true_and]
  exact ⟨t, ht, g, hg, hu⟩

/-- The induction behind `compose_cone`: on the gates feeding `T`, committed wires carry their evaluation, and so does
each unit's local evaluation of its own gates. -/
theorem cone_induction (X a : C.Trace) (T : Finset (Fin C.N)) (hin : C.InputsAgree X a)
    (hc : ∀ u ∈ P.cone T, P.Correct X u) :
    ∀ m (g : Fin C.N), g.val = m → (∃ t ∈ T, C.Feeds g t) →
      (∀ u, P.unit g = some u → P.localEval X u g = C.eval a g) ∧
      (g ∈ P.committed → X g = C.eval a g) := by
  intro m
  induction m using Nat.strong_induction_on with
  | _ m ih =>
  intro g hgm ⟨t, ht, hgt⟩
  have hB : ∀ u, P.unit g = some u → P.localEval X u g = C.eval a g := by
    intro u hu
    have hni : C.isInput g = false := by
      cases h : C.isInput g
      · rfl
      · exact absurd ((P.input_iff g).2 h) (by simp [hu])
    rw [P.localEval_of_mem X hu, C.eval_op a hni]
    refine C.op_congr g _ _ fun x hx => ?_
    have hxg : x.val < m := hgm ▸ C.reads_lt g x hx
    have hxt : ∃ t ∈ T, C.Feeds x t := ⟨t, ht, .step hx hgt⟩
    obtain ⟨ihB, ihA⟩ := ih x.val hxg x rfl hxt
    by_cases hxu : P.unit x = some u
    · exact ihB u hxu
    · rw [P.localEval_of_not_mem X hxu]
      exact ihA (P.mem_committed_of_read hx (by rw [hu]; exact Ne.symm hxu))
  refine ⟨hB, fun hg => ?_⟩
  cases hi : C.isInput g
  · obtain ⟨u, hu⟩ := Option.ne_none_iff_exists'.1 fun h => by
      simp [(P.input_iff g).1 h] at hi
    rw [hc u (P.mem_cone ht hgt hu) g hg hu]
    exact hB u hu
  · rw [hin g hi, C.eval_input a hi]

/-- **A consumer's value, through its cone.** If the committed inputs are the anchors' values and no unit of `T`'s cone
is incorrect, every committed wire of `T` carries its evaluation. -/
theorem compose_cone (X a : C.Trace) (T : Finset (Fin C.N)) (hin : C.InputsAgree X a)
    (h : Disjoint (P.incorrect X) (P.cone T)) : ∀ t ∈ T, t ∈ P.committed → X t = C.eval a t := by
  intro t ht htc
  have hc : ∀ u ∈ P.cone T, P.Correct X u := fun u hu => by
    by_contra hw
    exact Finset.disjoint_left.1 h (P.mem_incorrect.2 hw) hu
  exact (P.cone_induction X a T hin hc t.val t rfl ⟨t, ht, .refl t⟩).2 htc

/-- **Composition.** If every unit is correct and the committed inputs are the anchors' values, every committed wire
carries its value in the circuit's evaluation. -/
theorem compose (X a : C.Trace) (hin : C.InputsAgree X a) (h : P.incorrect X = ∅) :
    ∀ g ∈ P.committed, X g = C.eval a g :=
  fun g hg => P.compose_cone X a P.committed hin (by simp [h]) g hg hg

theorem localEval_congr {X Y : C.Trace} {u : Fin n} (hXY : ∀ g ∈ P.io u, X g = Y g) :
    ∀ m (g : Fin C.N), g.val = m → P.unit g = some u → P.localEval X u g = P.localEval Y u g := by
  classical
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
    simp only [io, Finset.mem_filter]
    exact ⟨P.mem_committed_of_read hx (by rw [hu]; exact Ne.symm hxu), Or.inr ⟨g, hu, hx⟩⟩

/-- A unit's correctness depends only on its committed inputs and outputs. -/
theorem correct_congr {X Y : C.Trace} {u : Fin n} (hXY : ∀ g ∈ P.io u, X g = Y g)
    (hY : P.Correct Y u) : P.Correct X u := by
  classical
  intro g hg hu
  have hio : g ∈ P.io u := by simp only [io, Finset.mem_filter]; exact ⟨hg, Or.inl hu⟩
  rw [hXY g hio, P.localEval_congr hXY g.val g rfl hu]
  exact hY g hg hu

/-! ## Refinement -/

/-- Each unit's gates from its own trace `Y u`, every other gate from `X`. -/
def splice (X : C.Trace) (Y : Fin n → C.Trace) : C.Trace := fun g =>
  match P.unit g with
  | some u => Y u g
  | none => X g

theorem splice_of_unit {X : C.Trace} {Y : Fin n → C.Trace} {g : Fin C.N} {u : Fin n} (h : P.unit g = some u) :
    P.splice X Y g = Y u g := by
  simp only [splice, h]

theorem splice_of_none {X : C.Trace} {Y : Fin n → C.Trace} {g : Fin C.N} (h : P.unit g = none) :
    P.splice X Y g = X g := by
  simp only [splice, h]

/-- Traces that each agree with `X` on the gates `S` splice to one that does too. -/
theorem splice_agree {X : C.Trace} {Y : Fin n → C.Trace} {S : Finset (Fin C.N)} (hY : ∀ u, ∀ g ∈ S, Y u g = X g) :
    ∀ g ∈ S, P.splice X Y g = X g := by
  intro g hg
  cases hu : P.unit g with
  | none => exact P.splice_of_none hu
  | some u => rw [P.splice_of_unit hu]; exact hY u g hg

open Classical in
/-- The units with a committed input or output among the gates `S` outside the opened gates `O`. -/
noncomputable def unopened (S O : Finset (Fin C.N)) : Finset (Fin n) :=
  univ.filter fun u => ∃ g ∈ P.io u, g ∈ S ∧ g ∉ O

theorem unopened_self (S : Finset (Fin C.N)) : P.unopened S S = ∅ := by
  classical
  ext u
  simp [unopened]

theorem gates_nonempty_of_io {u : Fin n} {g : Fin C.N} (hg : g ∈ P.io u) : (P.gates u).Nonempty := by
  obtain ⟨-, h | ⟨h, hh, -⟩⟩ := Finset.mem_filter.1 hg
  · exact ⟨g, Finset.mem_filter.2 ⟨Finset.mem_univ _, h⟩⟩
  · exact ⟨h, Finset.mem_filter.2 ⟨Finset.mem_univ _, hh⟩⟩

theorem gates_nonempty_of_unopened {S O : Finset (Fin C.N)} {u : Fin n} (hu : u ∈ P.unopened S O) :
    (P.gates u).Nonempty := by
  obtain ⟨g, hg, -⟩ := (Finset.mem_filter.1 hu).2
  exact P.gates_nonempty_of_io hg

theorem gates_nonempty_of_incorrect {X : C.Trace} {u : Fin n} (hu : u ∈ P.incorrect X) : (P.gates u).Nonempty := by
  classical
  simp only [incorrect, Correct, Finset.mem_filter, Finset.mem_univ, true_and, not_forall] at hu
  obtain ⟨g, -, h, -⟩ := hu
  exact ⟨g, Finset.mem_filter.2 ⟨Finset.mem_univ _, h⟩⟩

namespace Refines

variable {nf nc : ℕ} {Pf : Partition C nf} {Pc : Partition C nc} (r : Refines Pf Pc)
include r

/-- A coarse committed wire is a fine committed wire. -/
theorem committed_subset : Pc.committed ⊆ Pf.committed := by
  intro g hg
  simp only [committed, Finset.mem_filter, Finset.mem_univ, true_and] at hg ⊢
  rcases hg with hg | hg | ⟨h, hgh, hne⟩
  · exact Or.inl hg
  · exact Or.inr (Or.inl hg)
  · refine Or.inr (Or.inr ⟨h, hgh, fun heq => hne ?_⟩)
    rw [r.sub h, r.sub g, heq]

/-- The induction behind `exists_incorrect_fine`: when every fine unit inside `u` is correct in `Y`, and `Y` agrees with
`X` on the coarse committed wires, `u`'s local evaluation in `X` is each fine unit's local evaluation in `Y`. -/
theorem localEval_eq {X Y : C.Trace} {u : Fin nc}
    (hY : ∀ g ∈ Pc.committed, Y g = X g) (hf : ∀ v, r.parent v = u → Pf.Correct Y v) :
    ∀ m (g : Fin C.N) (v : Fin nf), g.val = m → Pf.unit g = some v → r.parent v = u →
      Pc.localEval X u g = Pf.localEval Y v g := by
  intro m
  induction m using Nat.strong_induction_on with
  | _ m ih =>
  intro g v hgm hv hvu
  have hu : Pc.unit g = some u := by rw [r.sub g, hv]; simp [hvu]
  rw [Pc.localEval_of_mem X hu, Pf.localEval_of_mem Y hv]
  refine C.op_congr g _ _ fun x hx => ?_
  have hxg : x.val < m := hgm ▸ C.reads_lt g x hx
  by_cases hxu : Pc.unit x = some u
  · obtain ⟨v', hv', hv'u⟩ : ∃ v', Pf.unit x = some v' ∧ r.parent v' = u := by
      rw [r.sub x] at hxu
      simpa using hxu
    rw [ih x.val hxg x v' rfl hv' hv'u]
    by_cases hvv : v' = v
    · subst hvv; rfl
    · have hxv : Pf.unit x ≠ some v := by rw [hv']; simpa using hvv
      rw [Pf.localEval_of_not_mem Y hxv]
      have hxc : x ∈ Pf.committed :=
        Pf.mem_committed_of_read hx (by rw [hv, hv']; simpa using Ne.symm hvv)
      exact (hf v' hv'u x hxc hv').symm
  · have hxv : Pf.unit x ≠ some v := fun h => hxu (by rw [r.sub x, h]; simp [hvu])
    rw [Pc.localEval_of_not_mem X hxu, Pf.localEval_of_not_mem Y hxv]
    exact (hY x (Pc.mem_committed_of_read hx (by rw [hu]; exact Ne.symm hxu))).symm

/-- **A incorrect coarse unit has a incorrect fine unit inside it**, whatever `Y` holds on the coarse unit's interior, as long
as `Y` agrees with `X` on the coarse committed wires. -/
theorem exists_incorrect_fine {X Y : C.Trace} (hY : ∀ g ∈ Pc.committed, Y g = X g)
    {u : Fin nc} (hu : u ∈ Pc.incorrect X) : ∃ v, r.parent v = u ∧ v ∈ Pf.incorrect Y := by
  by_contra hno
  push Not at hno
  have hf : ∀ v, r.parent v = u → Pf.Correct Y v := fun v hv => by
    have := hno v hv
    rwa [Pf.mem_incorrect, not_not] at this
  refine Pc.mem_incorrect.1 hu fun g hg hgu => ?_
  obtain ⟨v, hv, hvu⟩ : ∃ v, Pf.unit g = some v ∧ r.parent v = u := by
    rw [r.sub g] at hgu
    simpa using hgu
  rw [← hY g hg, hf v hvu g (r.committed_subset hg) hv, r.localEval_eq hY hf g.val g v rfl hv hvu]

/-- A fine trace that extends a coarse one has at least as many incorrect fine units as the coarse one has incorrect
coarse units. -/
theorem card_incorrect_le {X Y : C.Trace} (hY : ∀ g ∈ Pc.committed, Y g = X g) :
    (Pc.incorrect X).card ≤ (Pf.incorrect Y).card := by
  classical
  calc (Pc.incorrect X).card ≤ ((Pf.incorrect Y).image r.parent).card := by
        refine Finset.card_le_card fun u hu => ?_
        obtain ⟨v, hv, hvw⟩ := r.exists_incorrect_fine hY hu
        exact Finset.mem_image.2 ⟨v, hvw, hv⟩
    _ ≤ (Pf.incorrect Y).card := Finset.card_image_le

/-- A fine unit's committed input or output outside its coarse unit is a coarse committed wire. -/
theorem mem_committed_of_io {v : Fin nf} {g : Fin C.N} (hg : g ∈ Pf.io v) (hk : Pc.unit g ≠ some (r.parent v)) :
    g ∈ Pc.committed := by
  simp only [io, Finset.mem_filter] at hg
  rcases hg.2 with hv | ⟨h, hh, hgh⟩
  · exact absurd (by rw [r.sub g, hv]; rfl : Pc.unit g = some (r.parent v)) hk
  · exact Pc.mem_committed_of_read hgh (by rw [r.sub h, hh]; simpa using Ne.symm hk)

/-- Inside coarse unit `k`, a fine unit whose coarse committed inputs and outputs are all opened holds in the splice
what it holds in `Y k`. -/
theorem splice_io {X : C.Trace} {Y : Fin nc → C.Trace} {O : Finset (Fin C.N)} (hY : ∀ k, ∀ g ∈ O, Y k g = X g)
    {v : Fin nf} (hv : ∀ g ∈ Pf.io v, g ∈ Pc.committed → g ∈ O) :
    ∀ g ∈ Pf.io v, Pc.splice X Y g = Y (r.parent v) g := by
  intro g hg
  by_cases hk : Pc.unit g = some (r.parent v)
  · exact Pc.splice_of_unit hk
  · have hO := hv g hg (r.mem_committed_of_io hg hk)
    rw [Pc.splice_agree hY g hO, hY _ g hO]

/-- **Gluing**: fine traces `Y k` that each agree with `X` on the opened gates `O` splice to one fine trace that
agrees with `X` there and has, inside every coarse unit `k`, exactly `Y k`'s incorrect fine units, beside those with a
coarse committed wire left unopened. The best interior for one coarse unit does not depend on the others'. -/
theorem glue {X : C.Trace} {Y : Fin nc → C.Trace} {O : Finset (Fin C.N)} (hY : ∀ k, ∀ g ∈ O, Y k g = X g) :
    (∀ g ∈ O, Pc.splice X Y g = X g) ∧
      ∀ k, (Pf.unopened Pc.committed O ∪ Pf.incorrect (Pc.splice X Y)).filter (r.parent · = k) =
        (Pf.unopened Pc.committed O ∪ Pf.incorrect (Y k)).filter (r.parent · = k) := by
  classical
  refine ⟨Pc.splice_agree hY, fun k => ?_⟩
  ext v
  simp only [Finset.mem_filter, Finset.mem_union]
  refine and_congr_left fun hk => ?_
  by_cases hu : v ∈ Pf.unopened Pc.committed O
  · simp only [hu, true_or]
  simp only [hu, false_or]
  have hv : ∀ g ∈ Pf.io v, g ∈ Pc.committed → g ∈ O := fun g hg hc => by
    by_contra hO
    exact hu (by simp only [unopened, Finset.mem_filter, Finset.mem_univ, true_and]; exact ⟨g, hg, hc, hO⟩)
  have h := r.splice_io hY hv
  rw [Pf.mem_incorrect, Pf.mem_incorrect, ← hk]
  exact not_congr ⟨fun hc => Pf.correct_congr (fun g hg => (h g hg).symm) hc, fun hc => Pf.correct_congr h hc⟩

end Refines

end Partition

namespace Containment

variable {C D : Circuit} (M : Containment C D)

/-- The inputs `D` gives `C` through `M`. -/
def pull (a : D.Trace) : C.Trace := fun g => (a (M.emb g)).cast (M.emb_w g)

/-- `D`'s evaluation, read at the images of `C`'s gates, is `C`'s evaluation on the inputs `D` gives it. -/
theorem eval_emb (a : D.Trace) (g : Fin C.N) : (D.eval a (M.emb g)).cast (M.emb_w g) = C.eval (M.pull a) g := by
  induction g using WellFoundedLT.induction with
  | _ g ih => ?_
  cases hi : C.isInput g
  · have hD : D.isInput (M.emb g) = false := by rw [M.isInput_emb, hi]
    rw [D.eval_op a hD, C.eval_op _ hi, M.op_emb g hi]
    refine C.op_congr g _ _ fun x hx => ?_
    exact ih x (C.reads_lt g x hx)
  · have hD : D.isInput (M.emb g) = true := by rw [M.isInput_emb, hi]
    rw [D.eval_input a hD, C.eval_input _ hi]
    rfl

/-- Every circuit contains itself. -/
def refl (C : Circuit) : Containment C C where
  emb := id
  emb_inj := Function.injective_id
  emb_w _ := rfl
  isInput_emb _ := rfl
  reads_emb _ _ := (Finset.map_refl).symm
  op_emb _ _ _ := rfl
  outputs_emb _ h := h

/-- **A circuit that contains `C` implements it.** -/
def implementation : Implementation C D where
  inp := M.emb
  inp_input g h := by rw [M.isInput_emb, h]
  inp_inj g h _ _ he := M.emb_inj he
  inp_w g _ := M.emb_w g
  out := M.emb
  out_mem := M.outputs_emb
  out_w o _ := M.emb_w o
  correct a o _ := by
    rw [M.eval_emb a o]
    exact congrFun (C.eval_congr fun g hg => by simp [pull, hg]) o

end Containment

theorem Contains.implements {C D : Circuit} (h : Contains C D) : Implements C D :=
  h.elim fun M => ⟨M.implementation⟩

namespace Compiler.Output

variable (o : Compiler.Output) (b : Compiler.Phase o)

/-- `o`'s gate `g` as a gate of `o ‖ b`. -/
def old (g : Fin o.D.N) : Fin (o.append b).D.N := Fin.castAdd b.E.n g

/-- `b`'s gate `i` as a gate of `o ‖ b`. -/
def new (i : Fin b.E.n) : Fin (o.append b).D.N := Fin.natAdd o.D.N i

/-- `old` as an embedding. -/
def oldEmb : Fin o.D.N ↪ Fin (o.append b).D.N := ⟨o.old b, Fin.castAdd_injective _ _⟩

theorem oldEmb_apply (g : Fin o.D.N) : o.oldEmb b g = o.old b g := rfl

theorem append_cases (x : Fin (o.append b).D.N) : (∃ g, x = o.old b g) ∨ ∃ i, x = o.new b i :=
  Fin.addCases (motive := fun x : Fin (o.D.N + b.E.n) => (∃ g, x = o.old b g) ∨ ∃ i, x = o.new b i)
    (fun g => Or.inl ⟨g, rfl⟩) (fun i => Or.inr ⟨i, rfl⟩) x

theorem old_ne_new (g : Fin o.D.N) (i : Fin b.E.n) : o.old b g ≠ o.new b i := fun h => by
  have h1 : (g : ℕ) = o.D.N + i := congrArg Fin.val h
  omega

theorem mem_map_old (S : Finset (Fin o.D.N)) (g : Fin o.D.N) : o.old b g ∈ S.map (o.oldEmb b) ↔ g ∈ S :=
  Finset.mem_map' (o.oldEmb b)

theorem new_notMem_map (S : Finset (Fin o.D.N)) (i : Fin b.E.n) : o.new b i ∉ S.map (o.oldEmb b) := fun h => by
  obtain ⟨g, -, hg⟩ := Finset.mem_map.1 h
  exact o.old_ne_new b g i hg

theorem append_unit_old (g : Fin o.D.N) : (o.append b).P.unit (o.old b g) = (o.P.unit g).map (Fin.castAdd b.m) :=
  Fin.append_left (fun g => (o.P.unit g).map (Fin.castAdd b.m)) (fun i => (b.unit i).map (Fin.natAdd o.m)) g

theorem append_isInput_old (g : Fin o.D.N) : (o.append b).D.isInput (o.old b g) = o.D.isInput g :=
  Fin.append_left o.D.isInput b.E.isInput g

theorem append_reads_old (h : Fin o.D.N) : (o.append b).D.reads (o.old b h) = (o.D.reads h).map (o.oldEmb b) :=
  Fin.append_left (fun g => (o.D.reads g).map (Fin.castAddEmb b.E.n)) b.E.reads h

theorem append_mem_reads_old (a h : Fin o.D.N) : o.old b a ∈ (o.append b).D.reads (o.old b h) ↔ a ∈ o.D.reads h := by
  rw [append_reads_old, mem_map_old]

theorem append_outputs : (o.append b).D.outputs = o.D.outputs.map (o.oldEmb b) ∪ b.E.outputs := rfl

theorem append_phase_old (g : Fin o.D.N) : (o.append b).L.phase (o.old b g) = o.L.phase g :=
  Fin.append_left o.L.phase (fun _ => o.phases + 1) g

theorem append_phase_new (i : Fin b.E.n) : (o.append b).L.phase (o.new b i) = o.phases + 1 :=
  Fin.append_right o.L.phase (fun _ => o.phases + 1) i

theorem append_reg_old (g : Fin o.D.N) : (o.append b).L.reg (o.old b g) = o.L.reg g :=
  Fin.append_left o.L.reg (fun _ => none) g

theorem append_writerRank_old (g : Fin o.D.N) : (o.append b).P.writerRank (o.old b g) = o.P.writerRank g := by
  unfold Partition.writerRank
  rw [append_unit_old]
  cases o.P.unit g <;> rfl

theorem append_before_old (g h : Fin o.D.N) : (o.append b).P.Before (o.old b g) (o.old b h) ↔ o.P.Before g h := by
  unfold Partition.Before
  rw [append_writerRank_old, append_writerRank_old]
  exact Iff.rfl

theorem append_reads_new (i : Fin b.E.n) : (o.append b).D.reads (o.new b i) = b.E.reads i :=
  Fin.append_right (fun g => (o.D.reads g).map (Fin.castAddEmb b.E.n)) b.E.reads i

variable {o b}

theorem append_mem_outputs_old (g : Fin o.D.N) : o.old b g ∈ (o.append b).D.outputs ↔ g ∈ o.D.outputs := by
  rw [append_outputs]
  exact ⟨fun h => (Finset.mem_union.1 h).elim (mem_map_old o b _ g).1 (b.outputs_old g),
    fun h => Finset.mem_union.2 (Or.inl ((mem_map_old o b _ g).2 h))⟩

theorem append_unit_ne_old {g h : Fin o.D.N} :
    (o.append b).P.unit (o.old b h) ≠ (o.append b).P.unit (o.old b g) ↔ o.P.unit h ≠ o.P.unit g := by
  rw [append_unit_old, append_unit_old]
  exact not_congr (Option.map_injective (Fin.castAdd_injective _ _)).eq_iff

theorem append_mem_boundary_old {g : Fin o.D.N} (hg : g ∈ o.P.boundary) : o.old b g ∈ (o.append b).P.boundary := by
  obtain ⟨hc, hr⟩ := o.P.mem_boundary.1 hg
  have hc' : o.old b g ∈ (o.append b).P.committed := by
    rcases (Finset.mem_filter.1 hc).2 with hi | ho | ⟨h, hgh, hne⟩
    · exact (o.append b).P.mem_committed_of_input (by rw [append_isInput_old]; exact hi)
    · exact (o.append b).P.mem_committed_of_output ((append_mem_outputs_old g).2 ho)
    · exact (o.append b).P.mem_committed_of_read ((append_mem_reads_old o b g h).2 hgh) (append_unit_ne_old.2 hne)
  refine (o.append b).P.mem_boundary.2 ⟨hc', ?_⟩
  rw [append_isInput_old]
  rcases hr with hf | ho | ⟨h, hgh⟩
  · exact Or.inl hf
  · exact Or.inr (Or.inl ((append_mem_outputs_old g).2 ho))
  · exact Or.inr (Or.inr ⟨o.old b h, (append_mem_reads_old o b g h).2 hgh⟩)

theorem mem_boundary_of_append {g : Fin o.D.N} (hg : o.old b g ∈ (o.append b).P.boundary) : g ∈ o.P.boundary := by
  obtain ⟨hc, hr⟩ := (o.append b).P.mem_boundary.1 hg
  have hc := (Finset.mem_filter.1 hc).2
  rw [append_isInput_old] at hc hr
  have key : ∀ h, o.old b g ∈ (o.append b).D.reads h →
      (∃ h', g ∈ o.D.reads h' ∧ h = o.old b h') ∨ g ∈ o.P.boundary := by
    intro h hgh
    rcases o.append_cases b h with ⟨h, rfl⟩ | ⟨i, rfl⟩
    · exact Or.inl ⟨h, (append_mem_reads_old o b g h).1 hgh, rfl⟩
    · rw [append_reads_new] at hgh
      exact Or.inr (b.reads_boundary i g hgh)
  have hcg : g ∈ o.P.committed := by
    rcases hc with hi | ho | ⟨h, hgh, hne⟩
    · exact o.P.mem_committed_of_input hi
    · exact o.P.mem_committed_of_output ((append_mem_outputs_old g).1 ho)
    · rcases key h hgh with ⟨h', hgh', rfl⟩ | hb
      · exact o.P.mem_committed_of_read hgh' (append_unit_ne_old.1 hne)
      · exact o.P.boundary_subset_committed hb
  rcases hr with hf | ho | ⟨h, hgh⟩
  · exact o.P.mem_boundary.2 ⟨hcg, Or.inl hf⟩
  · exact o.P.mem_boundary.2 ⟨hcg, Or.inr (Or.inl ((append_mem_outputs_old g).1 ho))⟩
  · rcases key h hgh with ⟨h', hgh', -⟩ | hb
    · exact o.P.mem_boundary.2 ⟨hcg, Or.inr (Or.inr ⟨h', hgh'⟩)⟩
    · exact hb

/-- **Appending a phase keeps the earlier phases' windows.** -/
theorem append_window_old {j : ℕ} (hj : j ≤ o.phases) :
    (o.append b).P.window (o.append b).L j = (o.P.window o.L j).map (o.oldEmb b) := by
  ext x
  rcases o.append_cases b x with ⟨y, rfl⟩ | ⟨i, rfl⟩
  · rw [mem_map_old]
    unfold Partition.window
    rw [Finset.mem_filter, Finset.mem_filter, append_reg_old, append_phase_old]
    exact and_congr ⟨mem_boundary_of_append, append_mem_boundary_old⟩ Iff.rfl
  · refine iff_of_false (fun hx => ?_) (new_notMem_map o b _ i)
    have hx := (Finset.mem_filter.1 hx).2.2
    rw [append_phase_new] at hx
    omega

/-- **Appending a phase keeps every earlier boundary gate's position.** -/
theorem append_pos_old {g : Fin o.D.N} (hg : g ∈ o.P.boundary) :
    (o.append b).P.pos (o.append b).L (o.old b g) = o.P.pos o.L g := by
  unfold Partition.pos
  rw [if_pos (append_mem_boundary_old hg), if_pos hg, append_reg_old, append_phase_old]
  rcases o.L.reg g with _ | ⟨i, r⟩
  · rw [append_window_old (o.phase_le g), Finset.filter_map, Finset.card_map]
    simp only [Function.comp_def, oldEmb_apply, append_before_old]
  · rfl

end Compiler.Output

end Verity

namespace Verity.SecurityProofs

theorem Compose : Guarantees.Compose :=
  fun _ _ P X a hin h => P.compose X a hin h

theorem ComposeCone : Guarantees.ComposeCone :=
  fun _ _ P X a T hin h => P.compose_cone X a T hin h

theorem CorrectOfOpening : Guarantees.CorrectOfOpening :=
  fun _ _ P _ _ _ hXY => ⟨P.correct_congr fun g hg => (hXY g hg).symm, P.correct_congr hXY⟩

theorem RefineIncorrect : Guarantees.RefineIncorrect :=
  fun _ _ _ _ _ r _ _ hY => r.card_incorrect_le hY

theorem ContainsImplements : Guarantees.ContainsImplements :=
  fun _ _ M hM => ⟨M.implementation, hM⟩

end Verity.SecurityProofs
