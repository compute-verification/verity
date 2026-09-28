import Models.Core.Guarantees.Circuit

/-!
# Lemmas about circuits and partitions

Evaluation unfolds one gate at a time (`run_eq`), and the partition lemmas C-Flock's audit uses, for gates of any
width:

* `compose`: if every unit is correct and the committed inputs are the anchors' values, every committed wire carries its
  value in the circuit's evaluation;
* `compose_cone`: the same for the wires `T` a consumer reads, given only that the units of `T`'s cone are correct;
* `correct_congr`: a unit's correctness depends only on its committed inputs and outputs (`Partition.io`);
* `Refines.exists_wrong_fine`: when a fine partition refines a coarse one, a wrong coarse unit has a wrong fine unit
  inside it, whatever the values on the coarse unit's interior;
* `Containment.implementation`: a circuit that contains `C` implements it.
-/

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

theorem eval_input (a : C.Transcript) {g : Fin C.N} (h : C.isInput g = true) : C.eval a g = a g :=
  C.run_fixed (by simp [h])

theorem eval_op (a : C.Transcript) {g : Fin C.N} (h : C.isInput g = false) :
    C.eval a g = C.op g (C.eval a) :=
  C.run_free (by simp [h])

/-- The honest transcript agrees with its inputs. -/
theorem inputsAgree_eval (a : C.Transcript) : C.InputsAgree (C.eval a) a :=
  fun _ h => C.eval_input a h

/-- The evaluation reads only the inputs. -/
theorem eval_congr {a b : C.Transcript} (h : C.InputsAgree a b) : C.eval a = C.eval b := by
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

theorem localEval_of_mem (X : C.Transcript) {u : Fin n} {g : Fin C.N} (h : P.unit g = some u) :
    P.localEval X u g = C.op g (P.localEval X u) :=
  C.run_free (by simp [h])

theorem localEval_of_not_mem (X : C.Transcript) {u : Fin n} {g : Fin C.N} (h : P.unit g ≠ some u) :
    P.localEval X u g = X g :=
  C.run_fixed (by simp [h])

theorem mem_wrong {X : C.Transcript} {u : Fin n} : u ∈ P.wrong X ↔ ¬ P.Correct X u := by
  classical
  simp [wrong]

theorem mem_cone {T : Finset (Fin C.N)} {u : Fin n} {t g : Fin C.N} (ht : t ∈ T) (hg : C.Feeds g t)
    (hu : P.unit g = some u) : u ∈ P.cone T := by
  classical
  simp only [cone, Finset.mem_filter, Finset.mem_univ, true_and]
  exact ⟨t, ht, g, hg, hu⟩

/-- The induction behind `compose_cone`: on the gates feeding `T`, committed wires carry their evaluation, and so does
each unit's local evaluation of its own gates. -/
theorem cone_induction (X a : C.Transcript) (T : Finset (Fin C.N)) (hin : C.InputsAgree X a)
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
is wrong, every committed wire of `T` carries its evaluation. -/
theorem compose_cone (X a : C.Transcript) (T : Finset (Fin C.N)) (hin : C.InputsAgree X a)
    (h : Disjoint (P.wrong X) (P.cone T)) : ∀ t ∈ T, t ∈ P.committed → X t = C.eval a t := by
  intro t ht htc
  have hc : ∀ u ∈ P.cone T, P.Correct X u := fun u hu => by
    by_contra hw
    exact Finset.disjoint_left.1 h (P.mem_wrong.2 hw) hu
  exact (P.cone_induction X a T hin hc t.val t rfl ⟨t, ht, .refl t⟩).2 htc

/-- **Composition.** If every unit is correct and the committed inputs are the anchors' values, every committed wire
carries its value in the circuit's evaluation. -/
theorem compose (X a : C.Transcript) (hin : C.InputsAgree X a) (h : P.wrong X = ∅) :
    ∀ g ∈ P.committed, X g = C.eval a g :=
  fun g hg => P.compose_cone X a P.committed hin (by simp [h]) g hg hg

theorem localEval_congr {X Y : C.Transcript} {u : Fin n} (hXY : ∀ g ∈ P.io u, X g = Y g) :
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
theorem correct_congr {X Y : C.Transcript} {u : Fin n} (hXY : ∀ g ∈ P.io u, X g = Y g)
    (hY : P.Correct Y u) : P.Correct X u := by
  classical
  intro g hg hu
  have hio : g ∈ P.io u := by simp only [io, Finset.mem_filter]; exact ⟨hg, Or.inl hu⟩
  rw [hXY g hio, P.localEval_congr hXY g.val g rfl hu]
  exact hY g hg hu

/-! ## Refinement -/

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

/-- The induction behind `exists_wrong_fine`: when every fine unit inside `u` is correct in `Y`, and `Y` agrees with
`X` on the coarse committed wires, `u`'s local evaluation in `X` is each fine unit's local evaluation in `Y`. -/
theorem localEval_eq {X Y : C.Transcript} {u : Fin nc}
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

/-- **A wrong coarse unit has a wrong fine unit inside it**, whatever `Y` holds on the coarse unit's interior, as long
as `Y` agrees with `X` on the coarse committed wires. -/
theorem exists_wrong_fine {X Y : C.Transcript} (hY : ∀ g ∈ Pc.committed, Y g = X g)
    {u : Fin nc} (hu : u ∈ Pc.wrong X) : ∃ v, r.parent v = u ∧ v ∈ Pf.wrong Y := by
  by_contra hno
  push Not at hno
  have hf : ∀ v, r.parent v = u → Pf.Correct Y v := fun v hv => by
    have := hno v hv
    rwa [Pf.mem_wrong, not_not] at this
  refine Pc.mem_wrong.1 hu fun g hg hgu => ?_
  obtain ⟨v, hv, hvu⟩ : ∃ v, Pf.unit g = some v ∧ r.parent v = u := by
    rw [r.sub g] at hgu
    simpa using hgu
  rw [← hY g hg, hf v hvu g (r.committed_subset hg) hv, r.localEval_eq hY hf g.val g v rfl hv hvu]

/-- A fine transcript that extends a coarse one has at least as many wrong fine units as the coarse one has wrong
coarse units. -/
theorem card_wrong_le {X Y : C.Transcript} (hY : ∀ g ∈ Pc.committed, Y g = X g) :
    (Pc.wrong X).card ≤ (Pf.wrong Y).card := by
  classical
  calc (Pc.wrong X).card ≤ ((Pf.wrong Y).image r.parent).card := by
        refine Finset.card_le_card fun u hu => ?_
        obtain ⟨v, hv, hvw⟩ := r.exists_wrong_fine hY hu
        exact Finset.mem_image.2 ⟨v, hvw, hv⟩
    _ ≤ (Pf.wrong Y).card := Finset.card_image_le

end Refines

end Partition

namespace Containment

variable {C D : Circuit} (M : Containment C D)

/-- The inputs `D` gives `C` through `M`. -/
def pull (a : D.Transcript) : C.Transcript := fun g => (a (M.emb g)).cast (M.emb_w g)

/-- `D`'s evaluation, read at the images of `C`'s gates, is `C`'s evaluation on the inputs `D` gives it. -/
theorem eval_emb (a : D.Transcript) (g : Fin C.N) : (D.eval a (M.emb g)).cast (M.emb_w g) = C.eval (M.pull a) g := by
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

end Verity

namespace Verity.SecurityProofs

theorem Compose : Guarantees.Compose :=
  fun _ _ P X a hin h => P.compose X a hin h

theorem ComposeCone : Guarantees.ComposeCone :=
  fun _ _ P X a T hin h => P.compose_cone X a T hin h

theorem CorrectOfOpening : Guarantees.CorrectOfOpening :=
  fun _ _ P _ _ _ hXY => ⟨P.correct_congr fun g hg => (hXY g hg).symm, P.correct_congr hXY⟩

theorem RefineWrong : Guarantees.RefineWrong :=
  fun _ _ _ _ _ r _ _ hY => r.card_wrong_le hY

theorem ContainsImplements : Guarantees.ContainsImplements :=
  fun _ _ ⟨M, hM⟩ => ⟨M.implementation, hM⟩

end Verity.SecurityProofs
