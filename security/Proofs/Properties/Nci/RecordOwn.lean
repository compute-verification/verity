module

public import NciCheck.Record

/-!
# What `ownRows` reads off the root

`Record.ownRows` reads the root's body once. This module states what it found, by position: every part of every
argument of a root node that is no input node is a run of leaves of one input node (`part_run`), a batch node of
those has `dims` on every argument (`batch_dims`), and two such parts at two positions, each of some leaf, are runs
that share none (`runs_apart`). The test sorts the runs and compares neighbours; sorted, neighbours apart are all
apart (`apart_of_sorted`), and the order they stood in does not matter (`List.Perm.pairwise_iff`).
-/

@[expose] public section

namespace NciCheck.Record.SecurityProofs

open Flock.Program

/-- The run of leaves a part is, as `ownRows` reads it: its input node, its first leaf and its count. -/
def runOf (P : Prog) : Part → Option (Nat × Nat × Nat)
  | .affine false i base count =>
    if (((P.defs[P.root]!).body.nodes[i]?.map fun n =>
        Desc.isInputNode P n && (n.form == .prim || n.form == .batch)).getD false) then
      some (i, base, count)
    else none
  | _ => none

/-- Two forms `==` holds of are one. -/
theorem form_eq {a b : Form} (h : (a == b) = true) : a = b := by
  cases a <;> cases b <;> first | rfl | cases (show false = true from h)

/-- **A part that is a run** is an affine part over an input node of the root, a primitive or a batch. -/
theorem runOf_some {P : Prog} {q : Part} {r : Nat × Nat × Nat} (h : runOf P q = some r) :
    q = .affine false r.1 r.2.1 r.2.2 ∧ r.1 < (P.defs[P.root]!).body.nodes.size ∧
      Desc.isInputNode P ((P.defs[P.root]!).body.nodes[r.1]!) = true ∧
      (((P.defs[P.root]!).body.nodes[r.1]!).form = .prim ∨ ((P.defs[P.root]!).body.nodes[r.1]!).form = .batch) := by
  cases q with
  | affine p i base count =>
    cases p with
    | true => cases h
    | false =>
      simp only [runOf] at h
      split at h
      · rename_i hc
        cases h
        cases hn : (P.defs[P.root]!).body.nodes[i]? with
        | none => simp [hn] at hc
        | some n =>
          have hi : i < (P.defs[P.root]!).body.nodes.size := by
            rcases Nat.lt_or_ge i (P.defs[P.root]!).body.nodes.size with hlt | hge
            · exact hlt
            · rw [Array.getElem?_eq_none hge] at hn
              cases hn
          have hget : (P.defs[P.root]!).body.nodes[i]! = n := by
            rw [getElem!_pos _ i hi]
            rw [Array.getElem?_eq_getElem hi] at hn
            exact Option.some.inj hn
          simp only [hn, Option.map_some, Option.getD_some, Bool.and_eq_true, Bool.or_eq_true] at hc
          exact ⟨rfl, hi, by rw [hget]; exact hc.1, by rw [hget]; exact hc.2.imp form_eq form_eq⟩
      · cases h
  | strided p i base inner os is count => cases h
  | explicit items => cases h

/-- Where `removeAxis` strips an axis of a type, `axisDims` has that axis' length. -/
theorem axisDims_of_removeAxis : ∀ (ax : Nat) (t : Ty) {n : Nat} {mt : Ty}, t.removeAxis ax = some (n, mt) →
    ∃ L, t.axisDims ax = some (n, L)
  | 0, .array k e, n, mt, h => by
    simp only [Ty.removeAxis, Option.some.injEq, Prod.mk.injEq] at h
    exact ⟨e.leaves, by simp only [Ty.axisDims, h.1]⟩
  | ax + 1, .array k e, n, mt, h => by
    simp only [Ty.removeAxis, Option.map_eq_some_iff] at h
    obtain ⟨⟨m, inner⟩, hm, heq⟩ := h
    simp only [Prod.mk.injEq] at heq
    obtain ⟨L, hL⟩ := axisDims_of_removeAxis ax e hm
    exact ⟨L, by simp only [Ty.axisDims, hL, heq.1]⟩
  | 0, .value _, _, _, h => by simp [Ty.removeAxis] at h
  | 0, .tuple _, _, _, h => by simp [Ty.removeAxis] at h
  | 0, .record _, _, _, h => by simp [Ty.removeAxis] at h
  | _ + 1, .value _, _, _, h => by simp [Ty.removeAxis] at h
  | _ + 1, .tuple _, _, _, h => by simp [Ty.removeAxis] at h
  | _ + 1, .record _, _, _, h => by simp [Ty.removeAxis] at h

/-- The root's nodes that are no input nodes. -/
def unitsOf (P : Prog) : List Node :=
  (P.defs[P.root]!).body.nodes.toList.filter fun n => !Desc.isInputNode P n

/-- A node's argument parts, argument by argument. -/
def partsIn (n : Node) : List Part := n.args.toList.flatMap fun a => a.seq.parts.toList

/-- The order `ownRows` sorts runs in: by node, then by first leaf. -/
def runLe (a b : Nat × Nat × Nat) : Bool := a.1 < b.1 || (a.1 == b.1 && a.2.1 ≤ b.2.1)

/-- Two runs share no leaf: they are of two nodes, or one ends where the other starts or before. -/
def Apart (a b : Nat × Nat × Nat) : Prop := a.1 ≠ b.1 ∨ a.2.1 + a.2.2 ≤ b.2.1 ∨ b.2.1 + b.2.2 ≤ a.2.1

theorem Apart.symm {a b : Nat × Nat × Nat} (h : Apart a b) : Apart b a := by
  unfold Apart at h ⊢
  omega

theorem ownRows_eq (P : Prog) : ownRows P =
    ((unitsOf P).all (fun n => n.form != .batch || n.args.all (·.dims.isSome)) &&
      (((unitsOf P).flatMap partsIn).filterMap (runOf P)).length == ((unitsOf P).flatMap partsIn).length &&
      ((((((unitsOf P).flatMap partsIn).filterMap (runOf P)).filter fun r => r.2.2 > 0).mergeSort runLe).zip
        (((((unitsOf P).flatMap partsIn).filterMap (runOf P)).filter fun r => r.2.2 > 0).mergeSort runLe).tail).all
        fun (a, b) => a.1 != b.1 || a.2.1 + a.2.2 ≤ b.2.1) := rfl

theorem runLe_trans (a b c : Nat × Nat × Nat) (hab : runLe a b = true) (hbc : runLe b c = true) :
    runLe a c = true := by
  simp only [runLe, Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hab hbc ⊢
  omega

theorem runLe_total (a b : Nat × Nat × Nat) : (runLe a b || runLe b a) = true := by
  simp only [runLe, Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq]
  omega

/-- Sorted by node and first leaf, runs whose neighbours are apart are all apart. -/
theorem apart_of_sorted : ∀ (l : List (Nat × Nat × Nat)), l.Pairwise (fun a b => runLe a b = true) →
    ((l.zip l.tail).all fun (a, b) => a.1 != b.1 || a.2.1 + a.2.2 ≤ b.2.1) = true → l.Pairwise Apart
  | [], _, _ => List.Pairwise.nil
  | [_], _, _ => List.pairwise_singleton _ _
  | x :: y :: rest, hs, ha => by
    rw [List.pairwise_cons] at hs
    obtain ⟨hx, hs'⟩ := hs
    simp only [List.tail_cons, List.zip_cons_cons, List.all_cons, Bool.and_eq_true] at ha
    obtain ⟨hxy, ha'⟩ := ha
    have ih := apart_of_sorted (y :: rest) hs' (by simpa using ha')
    refine List.pairwise_cons.2 ⟨fun z hz => ?_, ih⟩
    have hxy' : x.1 ≠ y.1 ∨ x.2.1 + x.2.2 ≤ y.2.1 := by simpa using hxy
    have hlxy : x.1 < y.1 ∨ (x.1 = y.1 ∧ x.2.1 ≤ y.2.1) := by simpa [runLe] using hx y List.mem_cons_self
    rcases List.mem_cons.1 hz with rfl | hz'
    · unfold Apart
      omega
    · have hlyz : y.1 < z.1 ∨ (y.1 = z.1 ∧ y.2.1 ≤ z.2.1) := by
        simpa [runLe] using (List.pairwise_cons.1 hs').1 z hz'
      unfold Apart
      omega

/-- Where a `filterMap` keeps the list's length, it kept every element. -/
theorem filterMap_all {α β : Type} (f : α → Option β) : ∀ (l : List α), (l.filterMap f).length = l.length →
    ∀ x ∈ l, ∃ y, f x = some y
  | [], _, _, hx => by cases hx
  | a :: l, h, x, hx => by
    cases hfa : f a with
    | none =>
      rw [List.filterMap_cons_none hfa] at h
      have := List.length_filterMap_le f l
      rw [List.length_cons] at h
      omega
    | some y =>
      rw [List.filterMap_cons_some hfa, List.length_cons, List.length_cons] at h
      rcases List.mem_cons.1 hx with rfl | hx'
      · exact ⟨y, hfa⟩
      · exact filterMap_all f l (by omega) x hx'

/-- Two elements of a list pairwise in a symmetric relation, at two positions, are in it. -/
theorem pairwise_at {α : Type} {A : α → α → Prop} (hs : ∀ x y, A x y → A y x) {l : List α} (h : l.Pairwise A)
    {k k' : Nat} (hk : k < l.length) (hk' : k' < l.length) (hne : k ≠ k') : A l[k] l[k'] := by
  rcases Nat.lt_or_gt_of_ne hne with hlt | hgt
  · exact List.pairwise_iff_getElem.1 h k k' hk hk' hlt
  · exact hs _ _ (List.pairwise_iff_getElem.1 h k' k hk' hk hgt)

/-- Elements of two elements' images, at two positions of a list whose `flatMap` is pairwise in a symmetric
relation, are in it. -/
theorem flatMap_at {α β : Type} {A : β → β → Prop} (hs : ∀ x y, A x y → A y x) {l : List α} {f : α → List β}
    (h : (l.flatMap f).Pairwise A) {k k' : Nat} (hk : k < l.length) (hk' : k' < l.length) (hne : k ≠ k') {x y : β}
    (hx : x ∈ f l[k]) (hy : y ∈ f l[k']) : A x y :=
  pairwise_at (A := fun a₁ a₂ => ∀ x ∈ f a₁, ∀ y ∈ f a₂, A x y) (fun _ _ hab x hx y hy => hs _ _ (hab y hy x hx))
    (List.pairwise_flatMap.1 h).2 hk hk' hne x hx y hy

/-- One element's image, in a list whose `flatMap` is pairwise in a relation, is pairwise in it. -/
theorem flatMap_in {α β : Type} {A : β → β → Prop} {l : List α} {f : α → List β} (h : (l.flatMap f).Pairwise A)
    {a : α} (ha : a ∈ l) : (f a).Pairwise A :=
  (List.pairwise_flatMap.1 h).1 a ha

section
variable {P : Prog} (h : ownRows P = true)
include h

/-- **A batch that is no input node maps every argument over its members.** -/
theorem batch_dims {n : Node} (hn : n ∈ unitsOf P) (hb : n.form = .batch) {a : Arg} (ha : a ∈ n.args) :
    a.dims.isSome = true := by
  rw [ownRows_eq, Bool.and_eq_true, Bool.and_eq_true] at h
  have h1 := List.all_eq_true.1 h.1.1 n hn
  rw [hb] at h1
  rcases Bool.or_eq_true_iff.1 h1 with hc | hc
  · cases (show false = true from hc)
  · exact Array.all_eq_true'.1 hc a ha

/-- **Every part of an argument of a node that is no input node is a run of leaves of an input node.** -/
theorem part_run {n : Node} (hn : n ∈ unitsOf P) {q : Part} (hq : q ∈ partsIn n) : ∃ r, runOf P q = some r := by
  rw [ownRows_eq, Bool.and_eq_true, Bool.and_eq_true] at h
  exact filterMap_all (runOf P) _ (by simpa using h.1.2) q (List.mem_flatMap.2 ⟨n, hn, hq⟩)

/-- The runs of the parts `ownRows` reads, those of some leaf, are apart: as parts, in the order they stand in. -/
theorem parts_apart : ((unitsOf P).flatMap partsIn).Pairwise fun q q' =>
    ∀ r, runOf P q = some r → ∀ r', runOf P q' = some r' → 0 < r.2.2 → 0 < r'.2.2 → Apart r r' := by
  rw [ownRows_eq, Bool.and_eq_true, Bool.and_eq_true] at h
  have hsorted := apart_of_sorted _ (List.pairwise_mergeSort (le := runLe) runLe_trans runLe_total _) h.2
  have hperm := (List.Perm.pairwise_iff (R := Apart) (fun hab => Apart.symm hab)
    (List.mergeSort_perm ((((unitsOf P).flatMap partsIn).filterMap (runOf P)).filter fun r => r.2.2 > 0) runLe)).1
    hsorted
  rw [List.pairwise_filter, List.pairwise_filterMap] at hperm
  exact hperm.imp fun {q q'} hqq r hr r' hr' hp hp' => hqq r hr r' hr' (by simpa using hp) (by simpa using hp')

end

/-- The relation `parts_apart` holds parts in is symmetric. -/
theorem parts_symm (P : Prog) (q q' : Part)
    (hqq : ∀ r, runOf P q = some r → ∀ r', runOf P q' = some r' → 0 < r.2.2 → 0 < r'.2.2 → Apart r r') :
    ∀ r, runOf P q' = some r → ∀ r', runOf P q = some r' → 0 < r.2.2 → 0 < r'.2.2 → Apart r r' :=
  fun r hr r' hr' hp hp' => (hqq r' hr' r hr hp' hp).symm

/-- **Two parts at two positions, each a run of some leaf, are runs that share none.** Positions are a root node
that is no input node, one of its arguments and one of that argument's parts. -/
theorem runs_apart {P : Prog} (h : ownRows P = true) {k i q k' i' q' : Nat}
    (hk : k < (P.defs[P.root]!).body.nodes.size)
    (hu : Desc.isInputNode P ((P.defs[P.root]!).body.nodes[k]!) = false)
    (hi : i < ((P.defs[P.root]!).body.nodes[k]!).args.size)
    (hq : q < ((((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.parts).size)
    (hk' : k' < (P.defs[P.root]!).body.nodes.size)
    (hu' : Desc.isInputNode P ((P.defs[P.root]!).body.nodes[k']!) = false)
    (hi' : i' < ((P.defs[P.root]!).body.nodes[k']!).args.size)
    (hq' : q' < ((((P.defs[P.root]!).body.nodes[k']!).args[i']!).seq.parts).size)
    {r r' : Nat × Nat × Nat}
    (hr : runOf P ((((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.parts[q]!) = some r)
    (hr' : runOf P ((((P.defs[P.root]!).body.nodes[k']!).args[i']!).seq.parts[q']!) = some r')
    (hp : 0 < r.2.2) (hp' : 0 < r'.2.2) (hne : ¬ (k = k' ∧ i = i' ∧ q = q')) : Apart r r' := by
  have hall := parts_apart h
  have hsym := parts_symm P
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
      exact this r hr r' hr' hp hp'
    · have hil : i < ((P.defs[P.root]!).body.nodes[k]!).args.toList.length := by simpa using hi
      have hil' : i' < ((P.defs[P.root]!).body.nodes[k]!).args.toList.length := by simpa using hi'
      refine flatMap_at hsym hnode hil hil' hii (x := (((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.parts[q]!)
        (y := (((P.defs[P.root]!).body.nodes[k]!).args[i']!).seq.parts[q']!) ?_ ?_ r hr r' hr' hp hp'
      · rw [Array.getElem_toList, ← getElem!_pos _ i hi, getElem!_pos _ q hq]
        exact Array.mem_toList_iff.2 (Array.getElem_mem hq)
      · rw [Array.getElem_toList, ← getElem!_pos _ i' hi', getElem!_pos _ q' hq']
        exact Array.mem_toList_iff.2 (Array.getElem_mem hq')
  · have hout' := pairwise_at (A := fun x y => (!Desc.isInputNode P x) = true → (!Desc.isInputNode P y) = true →
        ∀ a ∈ partsIn x, ∀ b ∈ partsIn y, ∀ r, runOf P a = some r → ∀ r', runOf P b = some r' → 0 < r.2.2 →
          0 < r'.2.2 → Apart r r')
      (fun x y hxy hy hx a ha b hb => hsym _ _ (hxy hx hy b hb a ha)) hout hkl hkl' hkk
    rw [hnk, hnk'] at hout'
    exact hout' (by simp [hu]) (by simp [hu']) _ (hmem _ i q hi hq) _ (hmem _ i' q' hi' hq') r hr r' hr' hp hp'

end NciCheck.Record.SecurityProofs
