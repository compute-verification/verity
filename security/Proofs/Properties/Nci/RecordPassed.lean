module

public import NciCheck.Record

/-!
# What each Definition passes on, as the record check computes it

`Record.passedOn` reads, per Definition of the Program, the parameters a leaf of its return may be, callees first.
This module states it one Definition at a time: a Definition's entry is `defOn` of the entries before it
(`passedOn_at`), a node's reach is `nodeOn` of the reaches before it (`reachOf_at`), and a reference of a part is
among what the part hands on (`partOn_get`). Both are folds that push one entry a step (`foldl_push`), so an entry,
once pushed, is the fold's.
-/

@[expose] public section

namespace NciCheck.Record.SecurityProofs

open Flock.Program

/-- A fold that pushes one entry a step: it adds an entry an element, keeps the entries it started from, and its
entry for element `i` is the step's on the fold of the elements before it. -/
theorem foldl_push {α β : Type} (f : Array β → α → β) : ∀ (l : List α) (init : Array β),
    (l.foldl (fun acc a => acc.push (f acc a)) init).size = init.size + l.length ∧
      (∀ i, i < init.size → (l.foldl (fun acc a => acc.push (f acc a)) init)[i]? = init[i]?) ∧
      ∀ i (h : i < l.length), (l.foldl (fun acc a => acc.push (f acc a)) init)[init.size + i]? =
        some (f ((l.take i).foldl (fun acc a => acc.push (f acc a)) init) l[i])
  | [], init => ⟨rfl, fun _ _ => rfl, fun _ h => by cases h⟩
  | a :: l, init => by
    obtain ⟨h1, h2, h3⟩ := foldl_push f l (init.push (f init a))
    rw [Array.size_push] at h1 h2 h3
    refine ⟨by rw [List.foldl_cons, h1, List.length_cons]; omega, fun i hi => ?_, fun i hi => ?_⟩
    · rw [List.foldl_cons, h2 i (by omega), Array.getElem?_push_lt hi, Array.getElem?_eq_getElem hi]
    · rw [List.foldl_cons]
      cases i with
      | zero =>
        rw [Nat.add_zero, h2 _ (by omega), Array.getElem?_push_size]
        rfl
      | succ j =>
        have := h3 j (by simpa using hi)
        rw [show init.size + (j + 1) = init.size + 1 + j by omega, this]
        rfl

/-- What the fold has at an entry it has pushed is what the fold of the elements before a later one has there. -/
theorem foldl_push_prefix {α β : Type} (f : Array β → α → β) (l : List α) (k i : Nat) (hi : i < k)
    (hk : k ≤ l.length) :
    ((l.take k).foldl (fun acc a => acc.push (f acc a)) #[])[i]? =
      (l.foldl (fun acc a => acc.push (f acc a)) #[])[i]? := by
  have hsz := (foldl_push f (l.take k) #[]).1
  have := (foldl_push f (l.drop k) ((l.take k).foldl (fun acc a => acc.push (f acc a)) #[])).2.1 i
    (by rw [hsz, List.length_take]; simp; omega)
  rw [← List.foldl_append, List.take_append_drop] at this
  exact this.symm

/-- The parameters a part's references may be leaves of: its parameter, or what the node it refers to reaches. -/
def partOn (every : List Nat) (reach : Array (List Nat)) : Part → List Nat
  | .affine p i _ count | .strided p i _ _ _ _ count =>
    if count == 0 then [] else if p then [i] else reach[i]?.getD every
  | .explicit items => items.toList.flatMap fun r => if r.p then [r.i] else reach[r.i]?.getD every

/-- The parameters a sequence's references may be leaves of. -/
def seqOn (every : List Nat) (reach : Array (List Nat)) (s : Seq) : List Nat :=
  s.parts.toList.flatMap (partOn every reach)

/-- The parameters a node's output leaves may be: of each argument its callee passes on (`acc`), the parameters
that argument's references may be. -/
def nodeOn (acc : Array (List Nat)) (every : List Nat) (reach : Array (List Nat)) (n : Node) : List Nat :=
  ((acc[n.fn]?.getD (List.range n.args.size)).flatMap fun i =>
    (n.args[i]?.map fun a => seqOn every reach a.seq).getD []).eraseDups

/-- Each node's reach, in the body's order. -/
def reachOf (acc : Array (List Nat)) (d : Defn) : Array (List Nat) :=
  d.body.nodes.foldl (init := #[]) fun reach n => reach.push (nodeOn acc (List.range d.params.size) reach n)

/-- The parameters a Definition passes on: none for a primitive, else those its return's references may be. -/
def defOn (acc : Array (List Nat)) (d : Defn) : List Nat :=
  if d.prim then [] else (seqOn (List.range d.params.size) (reachOf acc d) d.body.ret).eraseDups

/-- **`passedOn` is the fold of `defOn`** over the Program's Definitions. -/
theorem passedOn_eq (P : Prog) : passedOn P = P.defs.foldl (fun acc d => acc.push (defOn acc d)) #[] := by
  unfold passedOn
  congr 1
  funext acc d
  unfold defOn
  split <;> rfl

/-- **A Definition's entry**: `passedOn` has an entry a Definition, and Definition `d`'s is `defOn` of an array that
has `passedOn`'s entries of the Definitions before `d`. -/
theorem passedOn_at (P : Prog) :
    (passedOn P).size = P.defs.size ∧ ∀ d (h : d < P.defs.size), ∃ acc : Array (List Nat),
      (∀ c, c < d → acc[c]? = (passedOn P)[c]?) ∧ (passedOn P)[d]? = some (defOn acc P.defs[d]) := by
  rw [passedOn_eq, ← Array.foldl_toList]
  obtain ⟨h1, -, h3⟩ := foldl_push defOn P.defs.toList #[]
  refine ⟨by simpa using h1, fun d h =>
    ⟨(P.defs.toList.take d).foldl (fun acc a => acc.push (defOn acc a)) #[], fun c hc => ?_, ?_⟩⟩
  · exact foldl_push_prefix defOn P.defs.toList d c hc (by simpa using Nat.le_of_lt h)
  · have := h3 d (by simpa using h)
    simpa using this

/-- **A node's reach**: `reachOf` has an entry a node, and node `k`'s is `nodeOn` of an array that has `reachOf`'s
entries of the nodes before `k`. -/
theorem reachOf_at (acc : Array (List Nat)) (d : Defn) :
    (reachOf acc d).size = d.body.nodes.size ∧ ∀ k (h : k < d.body.nodes.size), ∃ reach : Array (List Nat),
      (∀ i, i < k → reach[i]? = (reachOf acc d)[i]?) ∧
      (reachOf acc d)[k]? = some (nodeOn acc (List.range d.params.size) reach d.body.nodes[k]) := by
  unfold reachOf
  rw [← Array.foldl_toList]
  obtain ⟨h1, -, h3⟩ := foldl_push (nodeOn acc (List.range d.params.size)) d.body.nodes.toList #[]
  refine ⟨by simpa using h1, fun k h =>
    ⟨(d.body.nodes.toList.take k).foldl
      (fun reach n => reach.push (nodeOn acc (List.range d.params.size) reach n)) #[], fun i hi => ?_, ?_⟩⟩
  · exact foldl_push_prefix _ d.body.nodes.toList k i hi (by simpa using Nat.le_of_lt h)
  · have := h3 k (by simpa using h)
    simpa using this

/-- **A reference of a part is among what the part hands on**: its parameter where it refers to one, else every
parameter the node it refers to reaches. -/
theorem partOn_get (every : List Nat) (reach : Array (List Nat)) (q : Part) (t : Nat) (ht : t < q.count) :
    ((q.get t).p = true → (q.get t).i ∈ partOn every reach q) ∧
      ((q.get t).p = false → ∀ j ∈ reach[(q.get t).i]?.getD every, j ∈ partOn every reach q) := by
  cases q with
  | affine p i b c =>
    have hc : (c == 0) = false := by simp [Part.count] at ht; simp; omega
    refine ⟨fun hp => ?_, fun hp j hj => ?_⟩
    · have hp' : p = true := hp
      simp [partOn, hc, hp']
      rfl
    · have hp' : p = false := hp
      simp only [partOn, hc, hp']
      exact hj
  | strided p i b inner os is c =>
    have hc : (c == 0) = false := by simp [Part.count] at ht; simp; omega
    refine ⟨fun hp => ?_, fun hp j hj => ?_⟩
    · have hp' : p = true := hp
      simp [partOn, hc, hp']
      rfl
    · have hp' : p = false := hp
      simp only [partOn, hc, hp']
      exact hj
  | explicit it =>
    have ht' : t < it.size := ht
    have hg : (Part.explicit it).get t = it[t] := by
      show it[t]! = it[t]
      exact getElem!_pos it t ht'
    rw [hg]
    refine ⟨fun hp => ?_, fun hp j hj => ?_⟩
    · exact List.mem_flatMap.2 ⟨it[t], by simp, by simp [hp]⟩
    · exact List.mem_flatMap.2 ⟨it[t], by simp, by simpa [hp] using hj⟩

/-- What `passedOn` holds as it reads Definition `d`: the entries of the Definitions before it. -/
def accAt (P : Prog) (d : Nat) : Array (List Nat) :=
  (P.defs.toList.take d).foldl (fun acc a => acc.push (defOn acc a)) #[]

/-- Each node's reach in Definition `d`, as `passedOn` computes it. -/
def reachAt (P : Prog) (d : Nat) : Array (List Nat) := reachOf (accAt P d) P.defs[d]!

/-- The parameters Definition `d` passes on, none where the Program has no such Definition. -/
def passOf (P : Prog) (d : Nat) : List Nat := (passedOn P)[d]?.getD []

/-- The parameters of Definition `d` a reference of its body may be a leaf of: its parameter, or every parameter
the node it refers to reaches. -/
def refOn (P : Prog) (d : Nat) (r : Ref) : List Nat :=
  if r.p then [r.i] else (reachAt P d)[r.i]?.getD (List.range (P.defs[d]!).params.size)

theorem accAt_size (P : Prog) {d : Nat} (h : d ≤ P.defs.size) : (accAt P d).size = d := by
  have := (foldl_push defOn (P.defs.toList.take d) #[]).1
  rw [List.length_take] at this
  simpa [accAt, Nat.min_eq_left h] using this

theorem accAt_lt (P : Prog) {d c : Nat} (h : d ≤ P.defs.size) (hc : c < d) : (accAt P d)[c]? = (passedOn P)[c]? := by
  rw [passedOn_eq, ← Array.foldl_toList]
  exact foldl_push_prefix defOn P.defs.toList d c hc (by simpa using h)

theorem passedOn_get (P : Prog) {d : Nat} (h : d < P.defs.size) :
    (passedOn P)[d]? = some (defOn (accAt P d) P.defs[d]) := by
  rw [passedOn_eq, ← Array.foldl_toList]
  have := (foldl_push defOn P.defs.toList #[]).2.2 d (by simpa using h)
  simpa [accAt] using this

/-- **A reference of the return is among what the Definition passes on**: its parameter where it refers to one,
else every parameter the node it refers to reaches. -/
theorem ret_passed (P : Prog) {d : Nat} (h : d < P.defs.size) (hp : (P.defs[d]!).prim = false) {q : Part}
    (hq : q ∈ (P.defs[d]!).body.ret.parts) {t : Nat} (ht : t < q.count) :
    ∀ x ∈ refOn P d (q.get t), x ∈ passOf P d := by
  intro x hx
  have hd : P.defs[d]! = P.defs[d] := getElem!_pos P.defs d h
  have hget : passOf P d =
      (seqOn (List.range (P.defs[d]!).params.size) (reachAt P d) (P.defs[d]!).body.ret).eraseDups := by
    unfold passOf
    rw [passedOn_get P h, Option.getD_some]
    unfold defOn reachAt
    rw [← hd, hp]
    rfl
  rw [hget, List.mem_eraseDups]
  refine List.mem_flatMap.2 ⟨q, Array.mem_toList_iff.2 hq, ?_⟩
  obtain ⟨h1, h2⟩ := partOn_get (List.range (P.defs[d]!).params.size) (reachAt P d) q t ht
  unfold refOn at hx
  split at hx
  · rename_i hrp
    rw [List.mem_singleton] at hx
    rw [hx]
    exact h1 hrp
  · rename_i hrp
    exact h2 (by simpa using hrp) x hx

/-- **A reference of an argument the callee passes on is among what the node reaches**: where node `k` of
Definition `d` hands its callee argument `i`, and the callee passes that parameter on or is no earlier Definition, a
reference of the argument to a parameter, or to an earlier node, is a leaf only of parameters node `k` reaches. -/
theorem arg_reached (P : Prog) {d : Nat} (h : d < P.defs.size) {k : Nat} (hk : k < (P.defs[d]!).body.nodes.size)
    {i : Nat} (hi : i < ((P.defs[d]!).body.nodes[k]!).args.size)
    (hpass : i ∈ passOf P ((P.defs[d]!).body.nodes[k]!).fn ∨ d ≤ ((P.defs[d]!).body.nodes[k]!).fn) {q : Part}
    (hq : q ∈ (((P.defs[d]!).body.nodes[k]!).args[i]!).seq.parts) {t : Nat} (ht : t < q.count)
    (hr : (q.get t).p = false → (q.get t).i < k) :
    ∀ x ∈ refOn P d (q.get t), x ∈ refOn P d ⟨false, k, 0⟩ := by
  intro x hx
  obtain ⟨hsz, hat⟩ := reachOf_at (accAt P d) (P.defs[d]!)
  obtain ⟨reach, hpre, hk'⟩ := hat k hk
  have hnk : (P.defs[d]!).body.nodes[k]! = (P.defs[d]!).body.nodes[k] := getElem!_pos _ k hk
  show x ∈ (reachAt P d)[k]?.getD (List.range (P.defs[d]!).params.size)
  unfold reachAt
  rw [hk', Option.getD_some, ← hnk]
  unfold nodeOn
  rw [List.mem_eraseDups]
  refine List.mem_flatMap.2 ⟨i, ?_, ?_⟩
  · rcases hpass with hpass | hpass
    · by_cases hfd : ((P.defs[d]!).body.nodes[k]!).fn < d
      · rw [accAt_lt P (Nat.le_of_lt h) hfd]
        unfold passOf at hpass
        cases hg : (passedOn P)[((P.defs[d]!).body.nodes[k]!).fn]? with
        | none => rw [hg] at hpass; cases hpass
        | some l => rw [hg] at hpass; exact hpass
      · rw [Array.getElem?_eq_none (by rw [accAt_size P (Nat.le_of_lt h)]; omega)]
        exact List.mem_range.2 hi
    · rw [Array.getElem?_eq_none (by rw [accAt_size P (Nat.le_of_lt h)]; omega)]
      exact List.mem_range.2 hi
  · rw [Array.getElem?_eq_getElem hi, Option.map_some, Option.getD_some, ← getElem!_pos _ i hi]
    refine List.mem_flatMap.2 ⟨q, Array.mem_toList_iff.2 hq, ?_⟩
    obtain ⟨h1, h2⟩ := partOn_get (List.range (P.defs[d]!).params.size) reach q t ht
    unfold refOn at hx
    split at hx
    · rename_i hrp
      rw [List.mem_singleton] at hx
      rw [hx]
      exact h1 hrp
    · rename_i hrp
      have hrp' : (q.get t).p = false := by simpa using hrp
      refine h2 hrp' x ?_
      rw [hpre _ (hr hrp')]
      exact hx

/-! ## Along a walk of resolution

A walk that starts at a leaf of an activation `U`'s return moves down into members of nodes and back up through
parameters. `Passing` is what it keeps: the reference it stands at is a leaf only of parameters its Definition
passes on, and so is every node it came down through, up to `U`. -/

theorem refOn_node (P : Prog) (d k j : Nat) : refOn P d ⟨false, k, j⟩ = refOn P d ⟨false, k, 0⟩ := by
  simp only [refOn]

theorem refOn_param (P : Prog) (d i j : Nat) : refOn P d ⟨true, i, j⟩ = [i] := by
  simp only [refOn, ite_true]

/-- From the activation up to `U`, every node the walk came down through reaches only parameters its Definition
passes on. -/
def Covered (P : Prog) (U : Scope) : Scope → Prop
  | .top _ _ _ => True
  | .child parent d off k m => U = .child parent d off k m ∨
      ((∀ x ∈ refOn P parent.d ⟨false, k, 0⟩, x ∈ passOf P parent.d) ∧ Covered P U parent)

/-- What a walk from a leaf of `U`'s return keeps. -/
def Passing (P : Prog) (U : Scope) (s : Scope × Ref) : Prop :=
  Covered P U s.1 ∧ ∀ x ∈ refOn P s.1.d s.2, x ∈ passOf P s.1.d

theorem covered_refl (P : Prog) : ∀ U : Scope, Covered P U U
  | .top _ _ _ => trivial
  | .child _ _ _ _ _ => Or.inl rfl

/-- The walk starts passing: at a reference of `U`'s return. -/
theorem passing_start (P : Prog) (U : Scope) (h : U.d < P.defs.size) (hp : (P.defs[U.d]!).prim = false) {q : Part}
    (hq : q ∈ (P.defs[U.d]!).body.ret.parts) {t : Nat} (ht : t < q.count) {r : Ref} (hr : r = q.get t) :
    Passing P U (U, r) :=
  ⟨covered_refl P U, by rw [hr]; exact ret_passed P h hp hq ht⟩

/-- A step down, from a node's reference into a member of the node at a reference of its callee's return. -/
theorem passing_down (P : Prog) (U sc : Scope) (k j off m : Nat) (h : Passing P U (sc, ⟨false, k, j⟩))
    (hfn : ((P.defs[sc.d]!).body.nodes[k]!).fn < P.defs.size)
    (hp : (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).prim = false) {q : Part}
    (hq : q ∈ (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).body.ret.parts) {t : Nat} (ht : t < q.count) {r : Ref}
    (hr : r = q.get t) : Passing P U (.child sc ((P.defs[sc.d]!).body.nodes[k]!).fn off k m, r) :=
  ⟨Or.inr ⟨fun x hx => h.2 x (by rw [refOn_node]; exact hx), h.1⟩, by rw [hr]; exact ret_passed P hfn hp hq ht⟩

/-- A step up, from a parameter's reference to a reference of the argument that fills it. -/
theorem passing_up (P : Prog) (U parent : Scope) (d off k m i j : Nat)
    (h : Passing P U (.child parent d off k m, ⟨true, i, j⟩)) (hU : U ≠ .child parent d off k m)
    (hd : parent.d < P.defs.size) (hk : k < (P.defs[parent.d]!).body.nodes.size)
    (hfn : ((P.defs[parent.d]!).body.nodes[k]!).fn = d)
    (hi : i < ((P.defs[parent.d]!).body.nodes[k]!).args.size) {q : Part}
    (hq : q ∈ (((P.defs[parent.d]!).body.nodes[k]!).args[i]!).seq.parts) {t : Nat} (ht : t < q.count) {r : Ref}
    (hr : r = q.get t) (hlt : r.p = false → r.i < k) : Passing P U (parent, r) := by
  obtain ⟨hc, hi'⟩ := h
  subst hr
  rcases hc with hc | ⟨hreach, hc⟩
  · exact absurd hc hU
  · refine ⟨hc, fun x hx => hreach x ?_⟩
    refine arg_reached P hd hk hi (Or.inl ?_) hq ht hlt x hx
    rw [hfn]
    exact hi' i (by rw [refOn_param]; exact List.mem_singleton.2 rfl)

/-- A step to another member of the node, at a reference of the callee's return. -/
theorem passing_back (P : Prog) (U parent : Scope) (d off k m off' m' : Nat) (r0 : Ref)
    (h : Passing P U (.child parent d off k m, r0)) (hU : U ≠ .child parent d off k m)
    (hd : d < P.defs.size) (hp : (P.defs[d]!).prim = false) {q : Part}
    (hq : q ∈ (P.defs[d]!).body.ret.parts) {t : Nat} (ht : t < q.count) {r : Ref} (hr : r = q.get t) :
    Passing P U (.child parent d off' k m', r) := by
  obtain ⟨hc, -⟩ := h
  rcases hc with hc | hc
  · exact absurd hc hU
  · exact ⟨Or.inr hc, by rw [hr]; exact ret_passed P hd hp hq ht⟩

/-- An argument that has a reference is one of the node's. -/
theorem arg_lt (args : Array Arg) (i t : Nat) (h : t < (args[i]!).seq.count) : i < args.size := by
  rcases Nat.lt_or_ge i args.size with h' | h'
  · exact h'
  · rw [getElem!_neg args i (by omega)] at h
    exact absurd h (Nat.not_lt_zero _)

/-- Where the walk leaves `U`, through a parameter, the parameter is one `U`'s Definition passes on. -/
theorem passing_exit (P : Prog) (U : Scope) (i j : Nat) (h : Passing P U (U, ⟨true, i, j⟩)) : i ∈ passOf P U.d :=
  h.2 i (by rw [refOn_param]; exact List.mem_singleton.2 rfl)

end NciCheck.Record.SecurityProofs
