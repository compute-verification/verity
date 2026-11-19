module

public import Proofs.Properties.InferenceOnly.RecordOwn

/-!
# The runs `own-rows` reads of a record that declares nodes are apart

`OwnRunsOf` reads the runs of the parts that are no edge and over no declared node. This module gives `runs_apart`
for them: two parts at two positions of a root, each a run of some leaf that is no edge and over no declared node,
are runs that share none.
-/

@[expose] public section

namespace InferenceOnlyVerifier.Record.SecurityProofs

open Flock.Program

theorem flatMap_congr_mem {α β : Type} {l : List α} {f g : α → List β} (h : ∀ a ∈ l, f a = g a) :
    l.flatMap f = l.flatMap g := by
  induction l with
  | nil => rfl
  | cons x xs ih =>
    simp only [List.flatMap_cons, h x List.mem_cons_self, ih fun a ha => h a (List.mem_cons_of_mem _ ha)]

/-- The parts `ownRows` reads as runs of a record declaring `D`: no edge, over no declared node. -/
def ReadAsRun (P : Prog) (D : List Nat) (q : Part) : Bool := !edgeOf P q && !overDeclared D q

/-- The relation the parts `ownRows` reads as runs are pairwise in: where both are read as runs, their runs of some
leaf are apart. -/
def RunsApart (P : Prog) (D : List Nat) (q q' : Part) : Prop :=
  ReadAsRun P D q = true → ReadAsRun P D q' = true →
    ∀ r, runOf P q = some r → ∀ r', runOf P q' = some r' → 0 < r.2.2 → 0 < r'.2.2 → Apart r r'

theorem runsApart_symm (P : Prog) (D : List Nat) (q q' : Part) (h : RunsApart P D q q') : RunsApart P D q' q :=
  fun hq' hq r hr r' hr' hp hp' => (h hq hq' r' hr' r hr hp' hp).symm

/-- **The parts `own-rows` reads as runs are pairwise apart**, in the order they stand in. -/
theorem parts_apart_of {P : Prog} {D : List Nat} (h : OwnRunsOf P D) :
    ((unitsOf P).flatMap partsIn).Pairwise (RunsApart P D) := by
  unfold OwnRunsOf at h
  rw [Bool.and_eq_true, Bool.and_eq_true] at h
  have hsorted := apart_of_sorted _ (List.pairwise_mergeSort (le := runLe) runLe_trans runLe_total _) h.2
  have hperm := (List.Perm.pairwise_iff (R := Apart) (fun hab => Apart.symm hab)
    (List.mergeSort_perm ((((unitsOf P).flatMap (ownPartsIn P D)).filterMap (runOf P)).filter fun r => r.2.2 > 0)
      runLe)).1 hsorted
  rw [List.pairwise_filter, List.pairwise_filterMap] at hperm
  have hfl : (unitsOf P).flatMap (ownPartsIn P D) =
      ((unitsOf P).flatMap partsIn).filter (ReadAsRun P D) := by
    rw [List.filter_flatMap]
    refine flatMap_congr_mem fun n _ => ?_
    unfold ownPartsIn partsIn
    rw [List.filter_flatMap]
    rfl
  rw [hfl, List.pairwise_filter] at hperm
  exact hperm.imp fun {q q'} hqq hq hq' r hr r' hr' hp hp' => hqq hq hq' r hr r' hr' (by simpa using hp)
    (by simpa using hp')

/-- **Two parts at two positions, each a run of some leaf that `own-rows` reads as a run, are runs that share none.**
Positions are a root node that is no input node, one of its arguments and one of that argument's parts. -/
theorem runs_apart_of {P : Prog} {D : List Nat} (h : OwnRunsOf P D) {k i q k' i' q' : Nat}
    (hk : k < (P.defs[P.root]!).body.nodes.size)
    (hu : Desc.isInputNode P ((P.defs[P.root]!).body.nodes[k]!) = false)
    (hi : i < ((P.defs[P.root]!).body.nodes[k]!).args.size)
    (hq : q < ((((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.parts).size)
    (hk' : k' < (P.defs[P.root]!).body.nodes.size)
    (hu' : Desc.isInputNode P ((P.defs[P.root]!).body.nodes[k']!) = false)
    (hi' : i' < ((P.defs[P.root]!).body.nodes[k']!).args.size)
    (hq' : q' < ((((P.defs[P.root]!).body.nodes[k']!).args[i']!).seq.parts).size)
    {r r' : Nat × Nat × Nat}
    (hG : ReadAsRun P D ((((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.parts[q]!) = true)
    (hG' : ReadAsRun P D ((((P.defs[P.root]!).body.nodes[k']!).args[i']!).seq.parts[q']!) = true)
    (hr : runOf P ((((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.parts[q]!) = some r)
    (hr' : runOf P ((((P.defs[P.root]!).body.nodes[k']!).args[i']!).seq.parts[q']!) = some r')
    (hp : 0 < r.2.2) (hp' : 0 < r'.2.2) (hne : ¬ (k = k' ∧ i = i' ∧ q = q')) : Apart r r' := by
  have hall := parts_apart_of h
  have hsym := runsApart_symm P D
  unfold unitsOf at hall
  rw [List.pairwise_flatMap] at hall
  obtain ⟨hin, hout⟩ := hall
  rw [List.pairwise_filter] at hout
  have hkl : k < (P.defs[P.root]!).body.nodes.toList.length := by simpa using hk
  have hkl' : k' < (P.defs[P.root]!).body.nodes.toList.length := by simpa using hk'
  have hnk : (P.defs[P.root]!).body.nodes.toList[k] = (P.defs[P.root]!).body.nodes[k]! := by
    rw [getElem!_pos _ k hk, Array.getElem_toList]
  have hnk' : (P.defs[P.root]!).body.nodes.toList[k'] = (P.defs[P.root]!).body.nodes[k']! := by
    rw [getElem!_pos _ k' hk', Array.getElem_toList]
  have hmem : ∀ (n : Node) (i q : Nat) (hi : i < n.args.size) (hq : q < ((n.args[i]!).seq.parts).size),
      (n.args[i]!).seq.parts[q]! ∈ partsIn n := by
    intro n i q hi hq
    refine List.mem_flatMap.2 ⟨n.args[i]!, ?_, ?_⟩
    · rw [getElem!_pos _ i hi]
      exact Array.mem_toList_iff.2 (Array.getElem_mem hi)
    · rw [getElem!_pos _ q hq]
      exact Array.mem_toList_iff.2 (Array.getElem_mem hq)
  by_cases hkk : k = k'
  · subst hkk
    have hunit : (P.defs[P.root]!).body.nodes[k]! ∈
        (P.defs[P.root]!).body.nodes.toList.filter fun n => !Desc.isInputNode P n := by
      refine List.mem_filter.2 ⟨?_, by simp [hu]⟩
      rw [getElem!_pos _ k hk]
      exact Array.mem_toList_iff.2 (Array.getElem_mem hk)
    have hnode := hin _ hunit
    unfold partsIn at hnode
    by_cases hii : i = i'
    · subst hii
      have hqq : q ≠ q' := fun hc => hne ⟨rfl, rfl, hc⟩
      have harg := flatMap_in hnode (a := ((P.defs[P.root]!).body.nodes[k]!).args[i]!) (by
        rw [getElem!_pos _ i hi]
        exact Array.mem_toList_iff.2 (Array.getElem_mem hi))
      have hql : q < ((((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.parts).toList.length := by simpa using hq
      have hql' : q' < ((((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.parts).toList.length := by simpa using hq'
      have := pairwise_at hsym harg hql hql' hqq
      rw [Array.getElem_toList, Array.getElem_toList, ← getElem!_pos _ q hq, ← getElem!_pos _ q' hq'] at this
      exact this hG hG' r hr r' hr' hp hp'
    · have hil : i < ((P.defs[P.root]!).body.nodes[k]!).args.toList.length := by simpa using hi
      have hil' : i' < ((P.defs[P.root]!).body.nodes[k]!).args.toList.length := by simpa using hi'
      refine flatMap_at hsym hnode hil hil' hii (x := (((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.parts[q]!)
        (y := (((P.defs[P.root]!).body.nodes[k]!).args[i']!).seq.parts[q']!) ?_ ?_ hG hG' r hr r' hr' hp hp'
      · rw [Array.getElem_toList, ← getElem!_pos _ i hi, getElem!_pos _ q hq]
        exact Array.mem_toList_iff.2 (Array.getElem_mem hq)
      · rw [Array.getElem_toList, ← getElem!_pos _ i' hi', getElem!_pos _ q' hq']
        exact Array.mem_toList_iff.2 (Array.getElem_mem hq')
  · have hout' := pairwise_at (A := fun x y => (!Desc.isInputNode P x) = true → (!Desc.isInputNode P y) = true →
        ∀ a ∈ partsIn x, ∀ b ∈ partsIn y, RunsApart P D a b)
      (fun x y hxy hy hx a ha b hb => hsym _ _ (hxy hx hy b hb a ha)) hout hkl hkl' hkk
    rw [hnk, hnk'] at hout'
    exact hout' (by simp [hu]) (by simp [hu']) _ (hmem _ i q hi hq) _ (hmem _ i' q' hi' hq') hG hG' r hr r' hr' hp hp'

end InferenceOnlyVerifier.Record.SecurityProofs
