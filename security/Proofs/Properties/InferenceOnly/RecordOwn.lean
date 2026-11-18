module

public import InferenceOnlyVerifier.Record

/-!
# What the record proofs read off the root

`OwnRuns` is the reading of the root the record proofs cover: `Record.ownRows` admits it and it has no edge
(`EdgeFree`). `Record.ownRows` admits a root whose argument reads another unit's outputs (an edge, `edgeOf`), and
`EdgeFreeReads` is the hypothesis that every record `read` accepts is of a root with none, false of any root with an
edge. On a root with none, `Record.rowRuns` reads every part of every argument of a root node that is no input node
(`rowRuns_own`). This module states what `OwnRuns` gives, by position: every such part is a run of leaves of one input
node (`part_run`), an affine part or a strided part whose reference k is the run's leaf k mod the run (`runOf_some`); a
batch node of those has `dims` on every argument (`batch_dims`); a strided part that reads its run more than once reads
a shared run (`part_shared`); and two parts at two positions, each of some leaf, are runs that share none, or one
shared run (`runs_shared`). The test sorts the runs and compares neighbours; sorted, neighbours the same or apart are
all the same or apart (`same_or_apart_of_sorted`), two the same are neighbours the same with all between
(`shared_of_sorted`), and the order they stood in does not matter (`List.Perm.pairwise_iff`).
-/

@[expose] public section

namespace InferenceOnlyVerifier.Record.SecurityProofs

open Flock.Program

/-- The run of leaves a part is, as `ownRows` reads it: its input node, its first leaf and its count. -/
def runOf (P : Prog) (q : Part) : Option (Nat × Nat × Nat) := (partRun P q).map (·.1)

/-- Two forms `==` holds of are one. -/
theorem form_eq {a b : Form} (h : (a == b) = true) : a = b := by
  cases a <;> cases b <;> first | rfl | cases (show false = true from h)

/-- **Node `i` that `inputAt` admits** is an input node of the root, a primitive or a batch. -/
theorem inputAt_some {P : Prog} {i : Nat} (hc : inputAt P i = true) :
    i < (P.defs[P.root]!).body.nodes.size ∧ Desc.isInputNode P ((P.defs[P.root]!).body.nodes[i]!) = true ∧
      (((P.defs[P.root]!).body.nodes[i]!).form = .prim ∨ ((P.defs[P.root]!).body.nodes[i]!).form = .batch) := by
  unfold inputAt at hc
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
    exact ⟨hi, by rw [hget]; exact hc.1, by rw [hget]; exact hc.2.imp form_eq form_eq⟩

/-- **A part that is a run, and whether it reads it more than once**: an affine part over an input node of the root,
once, or a strided part over one with outer stride 0 and inner stride 1 from the run's first leaf and a count a
multiple of its inner run, the run that inner run or the count where less, more than once where the count passes
it. -/
theorem partRun_some {P : Prog} {q : Part} {r : Nat × Nat × Nat} {f : Bool} (h : partRun P q = some (r, f)) :
    ((q = .affine false r.1 r.2.1 r.2.2 ∧ f = false) ∨
      ∃ inner count, q = .strided false r.1 (r.2.1 : Int) inner 0 1 count ∧ 0 < inner ∧ count % inner = 0 ∧
        r.2.2 = min inner count ∧ f = decide (inner < count)) ∧
    inputAt P r.1 = true := by
  cases q with
  | affine p i base count =>
    cases p with
    | true => cases h
    | false =>
      simp only [partRun] at h
      split at h
      · rename_i hc
        cases h
        exact ⟨Or.inl ⟨rfl, rfl⟩, hc⟩
      · cases h
  | strided p i base inner os is count =>
    cases p with
    | true => cases h
    | false =>
      simp only [partRun] at h
      split at h
      · rename_i hc
        simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hc
        obtain ⟨⟨⟨⟨⟨hin, hos⟩, his⟩, hb⟩, hpos⟩, hmod⟩ := hc
        cases h
        subst hos his
        refine ⟨Or.inr ⟨inner, count, ?_, hpos, hmod, rfl, rfl⟩, hin⟩
        simp only [Int.toNat_of_nonneg hb]
      · cases h
  | explicit _ => cases h
  | grid _ _ => cases h

/-- A part's run is its `partRun`'s, with whether it reads it more than once. -/
theorem partRun_of_runOf {P : Prog} {q : Part} {r : Nat × Nat × Nat} (h : runOf P q = some r) :
    ∃ f, partRun P q = some (r, f) := by
  unfold runOf at h
  obtain ⟨⟨r', f⟩, hp, hr⟩ := Option.map_eq_some_iff.1 h
  subst hr
  exact ⟨f, hp⟩

/-- **A part that is a run** is an affine part over an input node of the root, or a strided part over one with
outer stride 0 and inner stride 1 from the run's first leaf, whose reference k is the run's leaf k mod the run. -/
theorem runOf_some {P : Prog} {q : Part} {r : Nat × Nat × Nat} (h : runOf P q = some r) :
    (q = .affine false r.1 r.2.1 r.2.2 ∨
      ∃ inner count, q = .strided false r.1 (r.2.1 : Int) inner 0 1 count ∧ 0 < inner ∧ count % inner = 0 ∧
        r.2.2 = min inner count) ∧ r.1 < (P.defs[P.root]!).body.nodes.size ∧
      Desc.isInputNode P ((P.defs[P.root]!).body.nodes[r.1]!) = true ∧
      (((P.defs[P.root]!).body.nodes[r.1]!).form = .prim ∨ ((P.defs[P.root]!).body.nodes[r.1]!).form = .batch) := by
  obtain ⟨f, hp⟩ := partRun_of_runOf h
  obtain ⟨hq, hin⟩ := partRun_some hp
  refine ⟨hq.imp (fun h => h.1) (fun ⟨inner, count, h1, h2, h3, h4, _⟩ => ⟨inner, count, h1, h2, h3, h4⟩),
    inputAt_some hin⟩

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

/-- **Part `q` reads another unit's outputs**: an affine or strided part over a node of the root that is no input
node, an explicit part naming one, or a grid every lane of which reads one, as `Record.rowRuns` tells an
edge. -/
def edgeOf (P : Prog) : Part → Bool
  | .affine false i _ _ | .strided false i _ _ _ _ _ =>
    ((P.defs[P.root]!).body.nodes[i]?.map fun n => !Desc.isInputNode P n).getD false
  | .explicit it =>
    it.any fun r => !r.p && ((P.defs[P.root]!).body.nodes[r.i]?.map fun n => !Desc.isInputNode P n).getD false
  | .grid _ lanes =>
    lanes.all fun l => !l.p && ((P.defs[P.root]!).body.nodes[l.i]?.map fun n => !Desc.isInputNode P n).getD false
  | _ => false

/-- **The root has no edge**: no part of an argument of a root node that is no input node reads another such node. -/
def EdgeFree (P : Prog) : Prop := ∀ n ∈ unitsOf P, ∀ q ∈ partsIn n, edgeOf P q = false

/-- **Every record `read` accepts is of a root with no edge.** False of any root with an edge; the record proofs that
take it cover the roots with none, and restating them over `OwnRowsEdges` replaces it. -/
def EdgeFreeReads : Prop := ∀ (j tr : Lean.Json) (c : Case), read j tr = .ok c → EdgeFree c.f.P

/-- **The reading of the root the record proofs cover**: `own-rows` admits the root, which has no edge. -/
def OwnRuns (P : Prog) : Prop := ownRows P = true ∧ EdgeFree P

/-- **On a root with no edge, `own-rows` admits only roots the record proofs cover.** -/
theorem ownRuns_of_edgeFree {P : Prog} (h : ownRows P = true) (hE : EdgeFree P) : OwnRuns P := ⟨h, hE⟩

private theorem flatMap_congr' {α β : Type} {l : List α} {f g : α → List β} (h : ∀ a ∈ l, f a = g a) :
    l.flatMap f = l.flatMap g := by
  induction l with
  | nil => rfl
  | cons x xs ih =>
    simp only [List.flatMap_cons, h x List.mem_cons_self, ih fun a ha => h a (List.mem_cons_of_mem _ ha)]

/-- **On a root the record proofs cover, `rowRuns` reads every argument part of the root's nodes that are no input
nodes**, and every batch of those maps every argument. -/
theorem rowRuns_own {P : Prog} (h : OwnRuns P) :
    (unitsOf P).all (fun n => n.form != .batch || n.args.all (·.dims.isSome)) = true ∧
      ∃ runs, rowRuns P = some runs ∧ runsOf P ((unitsOf P).flatMap partsIn) = some runs := by
  obtain ⟨ho, hE⟩ := h
  have hp : ((unitsOf P).flatMap fun n => n.args.toList.flatMap fun a => a.seq.parts.toList.filter (!edgeOf P ·)) =
      (unitsOf P).flatMap partsIn := by
    refine flatMap_congr' fun n hn => ?_
    rw [← List.filter_flatMap]
    exact List.filter_eq_self.2 fun q hq => by simpa using hE n hn q hq
  have e : rowRuns P = if (unitsOf P).all (fun n => n.form != .batch || n.args.all (·.dims.isSome)) then
      runsOf P ((unitsOf P).flatMap fun n => n.args.toList.flatMap fun a => a.seq.parts.toList.filter (!edgeOf P ·))
    else none := rfl
  rw [hp] at e
  unfold ownRows at ho
  rw [e] at ho ⊢
  by_cases hb : (unitsOf P).all (fun n => n.form != .batch || n.args.all (·.dims.isSome)) = true
  · rw [if_pos hb] at ho ⊢
    obtain ⟨runs, hr⟩ := Option.isSome_iff_exists.1 ho
    exact ⟨hb, runs, hr, hr⟩
  · rw [if_neg hb] at ho
    cases ho

/-- The runs `runsOf` reads off the parts `parts`, sorted by node and first leaf. -/
def sortedOf (P : Prog) (parts : List Part) : List ((Nat × Nat × Nat) × Bool) :=
  ((parts.filterMap (partRun P)).filter fun r => r.1.2.2 > 0).mergeSort fun a b => runLe a.1 b.1

/-- The runs of neighbours of `l` that are the same run. -/
def adjShared (l : List ((Nat × Nat × Nat) × Bool)) : List (Nat × Nat × Nat) :=
  (l.zip l.tail).filterMap fun (a, b) => if a.1 == b.1 then some a.1 else none

/-- **What `runsOf` admits**: every part a run, neighbours sorted the same run or apart, and the shared runs those a
part reads more than once and those of neighbours the same. -/
theorem runsOf_some {P : Prog} {parts : List Part} {runs : Runs} (h : runsOf P parts = some runs) :
    (parts.filterMap (partRun P)).length = parts.length ∧
      ((sortedOf P parts).zip (sortedOf P parts).tail).all
        (fun (a, b) => a.1 == b.1 || a.1.1 != b.1.1 || a.1.2.1 + a.1.2.2 ≤ b.1.2.1) = true ∧
      runs.shared = (sortedOf P parts).filterMap (fun r => if r.2 then some r.1 else none) ++
        adjShared (sortedOf P parts) := by
  unfold runsOf at h
  dsimp only at h
  split at h
  · rename_i hc
    rw [Bool.and_eq_true] at hc
    obtain ⟨hlen, hall⟩ := hc
    cases h
    exact ⟨beq_iff_eq.1 hlen, hall, rfl⟩
  · cases h

theorem runLe_trans (a b c : Nat × Nat × Nat) (hab : runLe a b = true) (hbc : runLe b c = true) :
    runLe a c = true := by
  simp only [runLe, Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hab hbc ⊢
  omega

theorem runLe_total (a b : Nat × Nat × Nat) : (runLe a b || runLe b a) = true := by
  simp only [runLe, Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq]
  omega

theorem sortedOf_sorted (P : Prog) (parts : List Part) :
    (sortedOf P parts).Pairwise fun a b => runLe a.1 b.1 = true :=
  List.pairwise_mergeSort (le := fun (a b : (Nat × Nat × Nat) × Bool) => runLe a.1 b.1)
    (fun a b c => runLe_trans a.1 b.1 c.1) (fun a b => runLe_total a.1 b.1) _

theorem sortedOf_perm (P : Prog) (parts : List Part) :
    (sortedOf P parts).Perm ((parts.filterMap (partRun P)).filter fun r => r.1.2.2 > 0) :=
  List.mergeSort_perm _ _

theorem sortedOf_pos {P : Prog} {parts : List Part} {a : (Nat × Nat × Nat) × Bool} (ha : a ∈ sortedOf P parts) :
    0 < a.1.2.2 := by
  have := (List.mem_filter.1 ((sortedOf_perm P parts).mem_iff.1 ha)).2
  simpa using this

/-- Sorted by node and first leaf, runs whose neighbours are the same run or apart are all the same or apart. -/
theorem same_or_apart_of_sorted : ∀ (l : List ((Nat × Nat × Nat) × Bool)),
    l.Pairwise (fun a b => runLe a.1 b.1 = true) →
    ((l.zip l.tail).all fun (a, b) => a.1 == b.1 || a.1.1 != b.1.1 || a.1.2.1 + a.1.2.2 ≤ b.1.2.1) = true →
    l.Pairwise fun a b => a.1 = b.1 ∨ Apart a.1 b.1
  | [], _, _ => List.Pairwise.nil
  | [_], _, _ => List.pairwise_singleton _ _
  | x :: y :: rest, hs, ha => by
    rw [List.pairwise_cons] at hs
    obtain ⟨hx, hs'⟩ := hs
    simp only [List.tail_cons, List.zip_cons_cons, List.all_cons, Bool.and_eq_true] at ha
    obtain ⟨hxy, ha'⟩ := ha
    have ih := same_or_apart_of_sorted (y :: rest) hs' (by simpa using ha')
    refine List.pairwise_cons.2 ⟨fun z hz => ?_, ih⟩
    have hxy' : x.1 = y.1 ∨ x.1.1 ≠ y.1.1 ∨ x.1.2.1 + x.1.2.2 ≤ y.1.2.1 := by simpa [or_assoc] using hxy
    have hlxy : x.1.1 < y.1.1 ∨ (x.1.1 = y.1.1 ∧ x.1.2.1 ≤ y.1.2.1) := by
      simpa [runLe] using hx y List.mem_cons_self
    rcases hxy' with heq | hap
    · rcases List.mem_cons.1 hz with rfl | hz'
      · exact Or.inl heq
      · rw [heq]
        exact (List.pairwise_cons.1 ih).1 z hz'
    · right
      rcases List.mem_cons.1 hz with rfl | hz'
      · unfold Apart
        omega
      · have hlyz : y.1.1 < z.1.1 ∨ (y.1.1 = z.1.1 ∧ y.1.2.1 ≤ z.1.2.1) := by
          simpa [runLe] using (List.pairwise_cons.1 hs').1 z hz'
        unfold Apart
        omega

theorem adjShared_mono (x y : (Nat × Nat × Nat) × Bool) (rest : List ((Nat × Nat × Nat) × Bool))
    {v : Nat × Nat × Nat} (hv : v ∈ adjShared (y :: rest)) : v ∈ adjShared (x :: y :: rest) := by
  unfold adjShared at hv ⊢
  obtain ⟨a, ha, hfa⟩ := List.mem_filterMap.1 hv
  exact List.mem_filterMap.2 ⟨a, List.mem_cons_of_mem _ ha, hfa⟩

/-- Sorted by node and first leaf, runs none empty and each two the same or apart: two that are the same run are
neighbours the same. -/
theorem shared_of_sorted : ∀ (l : List ((Nat × Nat × Nat) × Bool)),
    l.Pairwise (fun a b => runLe a.1 b.1 = true) → l.Pairwise (fun a b => a.1 = b.1 ∨ Apart a.1 b.1) →
    (∀ a ∈ l, 0 < a.1.2.2) → l.Pairwise fun a b => a.1 = b.1 → a.1 ∈ adjShared l
  | [], _, _, _ => List.Pairwise.nil
  | [_], _, _, _ => List.pairwise_singleton _ _
  | x :: y :: rest, hs, hsa, hpos => by
    have ih := shared_of_sorted (y :: rest) (List.pairwise_cons.1 hs).2 (List.pairwise_cons.1 hsa).2
      (fun a ha => hpos a (List.mem_cons_of_mem _ ha))
    refine List.pairwise_cons.2 ⟨fun z hz hxz => ?_, ih.imp fun hab hq => adjShared_mono x y rest (hab hq)⟩
    have hxy : x.1 = y.1 := by
      rcases List.mem_cons.1 hz with rfl | hz'
      · exact hxz
      · have hlxy : x.1.1 < y.1.1 ∨ (x.1.1 = y.1.1 ∧ x.1.2.1 ≤ y.1.2.1) := by
          simpa [runLe] using (List.pairwise_cons.1 hs).1 y List.mem_cons_self
        have hlyz : y.1.1 < z.1.1 ∨ (y.1.1 = z.1.1 ∧ y.1.2.1 ≤ z.1.2.1) := by
          simpa [runLe] using (List.pairwise_cons.1 (List.pairwise_cons.1 hs).2).1 z hz'
        have h1 : x.1.1 = z.1.1 := congrArg (·.1) hxz
        have h2 : x.1.2.1 = z.1.2.1 := congrArg (·.2.1) hxz
        have hpx := hpos x List.mem_cons_self
        have hpy := hpos y (List.mem_cons_of_mem _ List.mem_cons_self)
        rcases (List.pairwise_cons.1 hsa).1 y List.mem_cons_self with h | h
        · exact h
        · unfold Apart at h
          omega
    unfold adjShared
    simp [hxy]

/-- Elements of a list pairwise in a relation, mapped by `filterMap` and filtered, are pairwise in it as the list's
elements. -/
private theorem pairwise_back {R : (Nat × Nat × Nat) × Bool → (Nat × Nat × Nat) × Bool → Prop}
    (hsym : ∀ a b, R a b → R b a) {P : Prog} {parts : List Part} (h : (sortedOf P parts).Pairwise R) :
    parts.Pairwise fun q q' => ∀ a, partRun P q = some a → ∀ b, partRun P q' = some b → 0 < a.1.2.2 →
      0 < b.1.2.2 → R a b := by
  have hperm := (List.Perm.pairwise_iff (fun _ _ => hsym _ _) (sortedOf_perm P parts)).1 h
  rw [List.pairwise_filter, List.pairwise_filterMap] at hperm
  exact hperm.imp fun {q q'} hqq a ha b hb hp hp' => hqq a ha b hb (by simpa using hp) (by simpa using hp')

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
variable {P : Prog} (h : OwnRuns P)
include h

/-- **A batch that is no input node maps every argument over its members.** -/
theorem batch_dims {n : Node} (hn : n ∈ unitsOf P) (hb : n.form = .batch) {a : Arg} (ha : a ∈ n.args) :
    a.dims.isSome = true := by
  have h1 := List.all_eq_true.1 (rowRuns_own h).1 n hn
  rw [hb] at h1
  rcases Bool.or_eq_true_iff.1 h1 with hc | hc
  · cases (show false = true from hc)
  · exact Array.all_eq_true'.1 hc a ha

/-- **Every part of an argument of a node that is no input node is a run of leaves of an input node.** -/
theorem part_run {n : Node} (hn : n ∈ unitsOf P) {q : Part} (hq : q ∈ partsIn n) : ∃ r, runOf P q = some r := by
  obtain ⟨-, runs, -, hr⟩ := rowRuns_own h
  obtain ⟨y, hy⟩ := filterMap_all (partRun P) _ (runsOf_some hr).1 q (List.mem_flatMap.2 ⟨n, hn, hq⟩)
  exact ⟨y.1, by simp [runOf, hy]⟩

/-- **A part that reads its run more than once reads a shared run.** -/
theorem part_shared {n : Node} (hn : n ∈ unitsOf P) {q : Part} (hq : q ∈ partsIn n) {r : Nat × Nat × Nat}
    (hr : partRun P q = some (r, true)) (hp : 0 < r.2.2) : ∃ runs, rowRuns P = some runs ∧ r ∈ runs.shared := by
  obtain ⟨-, runs, hrr, hro⟩ := rowRuns_own h
  refine ⟨runs, hrr, ?_⟩
  rw [(runsOf_some hro).2.2]
  refine List.mem_append_left _ (List.mem_filterMap.2 ⟨(r, true), ?_, by simp⟩)
  refine (sortedOf_perm P _).mem_iff.2 (List.mem_filter.2 ⟨List.mem_filterMap.2 ⟨q, ?_, hr⟩, by simpa using hp⟩)
  exact List.mem_flatMap.2 ⟨n, hn, hq⟩

/-- The runs of the parts `ownRows` reads, those of some leaf, are apart or one shared run: as parts, in the order they
stand in. -/
theorem parts_shared : ((unitsOf P).flatMap partsIn).Pairwise fun q q' =>
    ∀ r, runOf P q = some r → ∀ r', runOf P q' = some r' → 0 < r.2.2 → 0 < r'.2.2 →
      (r = r' ∧ ∃ runs, rowRuns P = some runs ∧ r ∈ runs.shared) ∨ Apart r r' := by
  obtain ⟨-, runs, hrr, hro⟩ := rowRuns_own h
  obtain ⟨-, hall, hsh⟩ := runsOf_some hro
  have hsa := same_or_apart_of_sorted _ (sortedOf_sorted P _) hall
  have hadj := shared_of_sorted _ (sortedOf_sorted P _) hsa fun a ha => sortedOf_pos ha
  have hboth := (hsa.and hadj).imp fun {a b} ⟨h1, h2⟩ => (show (a.1 = b.1 ∧ a.1 ∈ adjShared (sortedOf P
    ((unitsOf P).flatMap partsIn))) ∨ Apart a.1 b.1 from h1.imp (fun e => ⟨e, h2 e⟩) id)
  have hback := pairwise_back (R := fun a b => (a.1 = b.1 ∧ a.1 ∈ adjShared (sortedOf P
      ((unitsOf P).flatMap partsIn))) ∨ Apart a.1 b.1)
    (fun a b hab => hab.elim (fun ⟨e, hm⟩ => Or.inl ⟨e.symm, e ▸ hm⟩) (fun hap => Or.inr hap.symm)) hboth
  refine hback.imp fun {q q'} hqq r hr r' hr' hp hp' => ?_
  obtain ⟨f, hf⟩ := partRun_of_runOf hr
  obtain ⟨f', hf'⟩ := partRun_of_runOf hr'
  rcases hqq (r, f) hf (r', f') hf' hp hp' with ⟨e, hm⟩ | hap
  · exact Or.inl ⟨e, runs, hrr, by rw [hsh]; exact List.mem_append_right _ hm⟩
  · exact Or.inr hap

end

/-- **A relation every two parts at two positions are in**, where it is symmetric and the parts `ownRows` reads
are pairwise in it. Positions are a root node that is no input node, one of its arguments and one of that argument's
parts. -/
theorem parts_at {P : Prog} {R : Part → Part → Prop} (hsym : ∀ a b, R a b → R b a)
    (hall : ((unitsOf P).flatMap partsIn).Pairwise R) {k i q k' i' q' : Nat}
    (hk : k < (P.defs[P.root]!).body.nodes.size)
    (hu : Desc.isInputNode P ((P.defs[P.root]!).body.nodes[k]!) = false)
    (hi : i < ((P.defs[P.root]!).body.nodes[k]!).args.size)
    (hq : q < ((((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.parts).size)
    (hk' : k' < (P.defs[P.root]!).body.nodes.size)
    (hu' : Desc.isInputNode P ((P.defs[P.root]!).body.nodes[k']!) = false)
    (hi' : i' < ((P.defs[P.root]!).body.nodes[k']!).args.size)
    (hq' : q' < ((((P.defs[P.root]!).body.nodes[k']!).args[i']!).seq.parts).size)
    (hne : ¬ (k = k' ∧ i = i' ∧ q = q')) :
    R ((((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.parts[q]!)
      ((((P.defs[P.root]!).body.nodes[k']!).args[i']!).seq.parts[q']!) := by
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
      exact this
    · have hil : i < ((P.defs[P.root]!).body.nodes[k]!).args.toList.length := by simpa using hi
      have hil' : i' < ((P.defs[P.root]!).body.nodes[k]!).args.toList.length := by simpa using hi'
      refine flatMap_at hsym hnode hil hil' hii (x := (((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.parts[q]!)
        (y := (((P.defs[P.root]!).body.nodes[k]!).args[i']!).seq.parts[q']!) ?_ ?_
      · rw [Array.getElem_toList, ← getElem!_pos _ i hi, getElem!_pos _ q hq]
        exact Array.mem_toList_iff.2 (Array.getElem_mem hq)
      · rw [Array.getElem_toList, ← getElem!_pos _ i' hi', getElem!_pos _ q' hq']
        exact Array.mem_toList_iff.2 (Array.getElem_mem hq')
  · have hout' := pairwise_at (A := fun x y => (!Desc.isInputNode P x) = true → (!Desc.isInputNode P y) = true →
        ∀ a ∈ partsIn x, ∀ b ∈ partsIn y, R a b)
      (fun x y hxy hy hx a ha b hb => hsym _ _ (hxy hx hy b hb a ha)) hout hkl hkl' hkk
    rw [hnk, hnk'] at hout'
    exact hout' (by simp [hu]) (by simp [hu']) _ (hmem _ i q hi hq) _ (hmem _ i' q' hi' hq')

/-- **Two parts at two positions, each a run of some leaf, are runs that share none, or one shared run.** Positions
are a root node that is no input node, one of its arguments and one of that argument's parts. -/
theorem runs_shared {P : Prog} (h : OwnRuns P) {k i q k' i' q' : Nat}
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
    (hp : 0 < r.2.2) (hp' : 0 < r'.2.2) (hne : ¬ (k = k' ∧ i = i' ∧ q = q')) :
    (r = r' ∧ ∃ runs, rowRuns P = some runs ∧ r ∈ runs.shared) ∨ Apart r r' :=
  parts_at (R := fun q q' => ∀ r, runOf P q = some r → ∀ r', runOf P q' = some r' → 0 < r.2.2 → 0 < r'.2.2 →
      (r = r' ∧ ∃ runs, rowRuns P = some runs ∧ r ∈ runs.shared) ∨ Apart r r')
    (fun _ _ hab r hr r' hr' hp hp' => (hab r' hr' r hr hp' hp).elim
      (fun ⟨e, hs⟩ => Or.inl ⟨e.symm, e ▸ hs⟩) (fun hap => Or.inr hap.symm))
    (parts_shared h) hk hu hi hq hk' hu' hi' hq' hne r hr r' hr' hp hp'

end InferenceOnlyVerifier.Record.SecurityProofs
