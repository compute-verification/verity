module

public import Proofs.Properties.Nci.RecordJson

/-!
# The root's ranges

`Flock.TemplateQuery.evaluate` reads the root's nodes in order and gives one range a node that is no input node:
the node, its callee's id as the template, and its members as the count. `evaluate_spec` states it: the ranges'
nodes are the root's nodes that are no input node (`Desc.isInputNode`: the callee's id an input id and the first
Definition of that id a primitive), in order, each a call or a batch, a call with one unit and a batch with its `n`.
`ranges_spec` states it of a Program's own root: the query finds the root by its key (`Desc.def?`), which is the
Program's root where no Definition before it has that key (`RootFirst`, `ofProg_root`), and it passes over every
root node whose callee's key is an input id where the first Definition of such a key is a primitive (`InputsRead`),
so the ranges' nodes are the root's nodes whose callee's key is no input id.
-/

@[expose] public section

namespace NciCheck.Record.SecurityProofs

open Flock.TemplateQuery (Desc DDef DNode TRange isInput)

/-- A loop that never stops early keeps an invariant on the indices it has read. -/
theorem forIn_done {σ : Type} {f : Nat → σ → Except String (ForInStep σ)} (Inv : List Nat → σ → Prop)
    (hstep : ∀ k s x done, Inv done s → f k s = .ok x → ∃ s', x = .yield s' ∧ Inv (done ++ [k]) s') :
    ∀ (l done : List Nat) (s x : σ), Inv done s → forIn l s f = .ok x → Inv (done ++ l) x
  | [], done, s, x, hi, h => by
    rw [List.forIn_nil] at h
    cases h
    rwa [List.append_nil]
  | k :: l, done, s, x, hi, h => by
    rw [List.forIn_cons] at h
    cases hf : f k s with
    | error e =>
      rw [hf] at h
      cases h
    | ok st =>
      rw [hf] at h
      obtain ⟨s', rfl, hi'⟩ := hstep k s st done hi hf
      have := forIn_done Inv hstep l (done ++ [k]) s' x hi' h
      rwa [List.append_assoc, List.singleton_append] at this

/-- The id of the callee of the root's node `k`. -/
def tidAt (d : Desc) (root : DDef) (k : Nat) : String := d.callees.getD root.nodes[k]!.fn ""

/-- **The root's ranges are its nodes that are no input node, in order**: each a call or a batch of its callee's id,
a call with one unit and a batch with its `n`. -/
theorem evaluate_spec {d : Desc} {ts : Array String} {rs : Array TRange}
    (h : Flock.TemplateQuery.evaluate d ts = .ok rs) :
    ∃ root, d.def? d.root = some root ∧
      rs.toList.map (·.node) = (List.range root.nodes.size).filter (fun k => !d.isInputNode root.nodes[k]!) ∧
      ∀ r ∈ rs.toList, r.template = tidAt d root r.node ∧
        (root.nodes[r.node]!.form = "call" ∨ root.nodes[r.node]!.form = "batch") ∧
        r.count = (if root.nodes[r.node]!.form == "batch" then root.nodes[r.node]!.n else 1) := by
  unfold Flock.TemplateQuery.evaluate at h
  cases hroot : d.def? d.root with
  | none =>
    simp only [hroot] at h
    cases h
  | some root =>
    simp only [hroot] at h
    obtain ⟨r, hloop, h⟩ := Json.bind_ok h
    rw [Std.Legacy.Range.forIn_eq_forIn_range'] at hloop
    have hrange : List.range' [:root.nodes.size].start [:root.nodes.size].size [:root.nodes.size].step =
        List.range root.nodes.size := by
      simp [Std.Legacy.Range.size, List.range_eq_range']
    rw [hrange] at hloop
    have hinv := forIn_done
      (fun done (s : Array TRange × Nat) =>
        s.1.toList.map (·.node) = done.filter (fun k => !d.isInputNode root.nodes[k]!) ∧
          ∀ r ∈ s.1.toList, r.template = tidAt d root r.node ∧
            (root.nodes[r.node]!.form = "call" ∨ root.nodes[r.node]!.form = "batch") ∧
            r.count = (if root.nodes[r.node]!.form == "batch" then root.nodes[r.node]!.n else 1))
      ?_ (List.range root.nodes.size) [] (#[], 0) r ⟨rfl, fun _ hr => by simp at hr⟩ hloop
    · have hrs : rs = r.1 := by
        simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at h
        split at h
        · cases h
        · cases h
          rfl
      rw [List.nil_append] at hinv
      rw [hrs]
      exact ⟨root, rfl, hinv.1, hinv.2⟩
    · clear hloop hrange
      intro k s x done hi hx
      simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at hx
      split at hx
      · rename_i hin
        cases hx
        have hin' : d.isInputNode root.nodes[k]! = true := hin
        refine ⟨_, rfl, ?_, hi.2⟩
        rw [List.filter_append, hi.1]
        simp [hin']
      · rename_i hin
        have hin' : ¬ d.isInputNode root.nodes[k]! = true := hin
        split at hx
        · cases hx
        · cases hx
          rename_i hc
          refine ⟨_, rfl, ?_, ?_⟩
          · rw [List.filter_append, ← hi.1]
            simp [hin']
          · intro r hr
            simp only [Array.toList_push, List.mem_append, List.mem_singleton] at hr
            rcases hr with hr | rfl
            · exact hi.2 r hr
            · simp only [Bool.or_eq_true, Bool.not_eq_true', Bool.not_eq_false, not_or, beq_iff_eq] at hc
              exact ⟨rfl, hc.1, rfl⟩

/-- **Every range's template is one the query was asked for**: a node that is no input and whose callee's id is not
among the ids asked for is refused. -/
theorem evaluate_asked {d : Desc} {ts : Array String} {rs : Array TRange}
    (h : Flock.TemplateQuery.evaluate d ts = .ok rs) : ∀ r ∈ rs.toList, ts.contains r.template = true := by
  unfold Flock.TemplateQuery.evaluate at h
  cases hroot : d.def? d.root with
  | none =>
    simp only [hroot] at h
    cases h
  | some root =>
    simp only [hroot] at h
    obtain ⟨r, hloop, h⟩ := Json.bind_ok h
    rw [Std.Legacy.Range.forIn_eq_forIn_range'] at hloop
    have hinv := forIn_done
      (fun _ (s : Array TRange × Nat) => ∀ r ∈ s.1.toList, ts.contains r.template = true)
      ?_ _ [] (#[], 0) r (fun _ hr => by simp at hr) hloop
    · have hrs : rs = r.1 := by
        simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at h
        split at h
        · cases h
        · cases h
          rfl
      rw [hrs]
      exact hinv
    · clear hloop
      intro k s x done hi hx
      simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at hx
      split at hx
      · cases hx
        exact ⟨_, rfl, hi⟩
      · split at hx
        · cases hx
        · cases hx
          rename_i hc
          refine ⟨_, rfl, fun r hr => ?_⟩
          simp only [Array.toList_push, List.mem_append, List.mem_singleton] at hr
          rcases hr with hr | rfl
          · exact hi r hr
          · simp only [Bool.or_eq_true, Bool.not_eq_true', Bool.not_eq_false, not_or, beq_iff_eq] at hc
            exact hc.2

open Flock.Program (Prog Form Node isInputId) in
/-- No Definition before the root has the root's key: the root's ranges are read from the first Definition with its
key (`Desc.def?`). -/
def RootFirst (P : Prog) : Prop := ∀ i < P.root, (P.defs[i]!).key ≠ (P.defs[P.root]!).key

/-- A Definition as the template query reads it (`Desc.ofProg`): its key, whether it is a primitive, and its nodes'
forms, callees and counts. -/
def dOf (d : Flock.Program.Defn) : String × DDef :=
  (d.key, { prim := d.prim, nodes := d.body.nodes.map fun x => { form := x.form.name, fn := x.fn, n := x.n } })

open Flock.Program (Prog) in
/-- **The description's root is the Program's**, where no Definition before the root has its key. -/
theorem ofProg_root {P : Prog} (hr : P.root < P.defs.size) (hk : RootFirst P) :
    (Desc.ofProg P).def? (Desc.ofProg P).root = some (dOf (P.defs[P.root]!)).2 := by
  have hget : P.defs[P.root]! = P.defs[P.root] := getElem!_pos P.defs P.root hr
  have hdefs : (Desc.ofProg P).defs = P.defs.map dOf := rfl
  have hroot : (Desc.ofProg P).root = (P.defs[P.root]).key := by
    show (P.defs.getD P.root default).key = _
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hr, Option.getD_some]
  have hsz : P.root < (P.defs.map dOf).size := by rw [Array.size_map]; exact hr
  have hfind : (P.defs.map dOf).find? (·.1 == (P.defs[P.root]).key) = some (P.defs.map dOf)[P.root] := by
    rw [Array.find?_eq_some_iff_getElem]
    refine ⟨?_, P.root, hsz, rfl, fun j hj => ?_⟩
    · rw [Array.getElem_map]
      exact beq_self_eq_true _
    · have hj' : j < P.defs.size := Nat.lt_trans hj hr
      have := hk j hj
      rw [getElem!_pos P.defs j hj', hget] at this
      rw [Array.getElem_map]
      simpa [dOf] using this
  unfold Desc.def?
  rw [hdefs, hroot, hfind, Option.map_some, hget, Array.getElem_map]

open Flock.Program (Prog) in
/-- **A Definition is the description's Definition of its key**, where no Definition before it has that key. -/
theorem ofProg_def {P : Prog} {i : Nat} (hi : i < P.defs.size)
    (hk : ∀ j < i, (P.defs[j]!).key ≠ (P.defs[i]!).key) :
    (Desc.ofProg P).def? (P.defs[i]!).key = some (dOf (P.defs[i]!)).2 := by
  have hget : P.defs[i]! = P.defs[i] := getElem!_pos P.defs i hi
  have hdefs : (Desc.ofProg P).defs = P.defs.map dOf := rfl
  have hsz : i < (P.defs.map dOf).size := by rw [Array.size_map]; exact hi
  have hfind : (P.defs.map dOf).find? (·.1 == (P.defs[i]).key) = some (P.defs.map dOf)[i] := by
    rw [Array.find?_eq_some_iff_getElem]
    refine ⟨?_, i, hsz, rfl, fun j hj => ?_⟩
    · rw [Array.getElem_map]
      exact beq_self_eq_true _
    · have hj' : j < P.defs.size := Nat.lt_trans hj hi
      have := hk j hj
      rw [getElem!_pos P.defs j hj', hget] at this
      rw [Array.getElem_map]
      simpa [dOf] using this
  unfold Desc.def?
  rw [hdefs, hget, hfind, Option.map_some, Array.getElem_map]

open Flock.Program (Prog) in
/-- A node's callee's id, as the template query reads it, is the callee's key. -/
theorem callee_key (P : Prog) (fn : Nat) : (Desc.ofProg P).callees.getD fn "" = (P.defs[fn]!).key := by
  show (P.defs.map (·.key)).getD fn "" = _
  by_cases h : fn < P.defs.size
  · rw [Array.getD_eq_getD_getElem?, Array.getElem?_map, Array.getElem?_eq_getElem h, getElem!_pos P.defs fn h]
    rfl
  · rw [Array.getD_eq_getD_getElem?, Array.getElem?_map, Array.getElem?_eq_none (Nat.le_of_not_lt h),
      getElem!_neg P.defs fn h]
    rfl

open Flock.Program (Prog isInputId) in
/-- A node as the template query reads it in a Program's description: an input node where its callee's key is an
input id and the first Definition of that key is a primitive. -/
theorem ofProg_isInputNode (P : Prog) (x : DNode) :
    (Desc.ofProg P).isInputNode x =
      (isInputId (P.defs[x.fn]!).key && ((Desc.ofProg P).def? (P.defs[x.fn]!).key).any (·.prim)) := by
  show (isInput ((Desc.ofProg P).callees.getD x.fn "") &&
    ((Desc.ofProg P).def? ((Desc.ofProg P).callees.getD x.fn "")).any (·.prim)) = _
  rw [callee_key] <;> rfl

open Flock.Program (Prog isInputId) in
/-- The template query passes over every root node whose callee's key is an input id: the first Definition of such
a key is a primitive, so the node is an input node as the query reads one (`Desc.isInputNode`). -/
def InputsRead (P : Prog) : Prop :=
  ∀ k < (P.defs[P.root]!).body.nodes.size,
    isInputId (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).key = true →
      ((Desc.ofProg P).def? (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).key).any (·.prim) = true

open Flock.Program (Prog Form Node isInputId) in
/-- **The ranges are the root's nodes whose callee's key is no input id, in order**: each a call or a batch, its
template the callee's key and its count the node's members. -/
theorem ranges_spec {P : Prog} {ts : Array String} {rs : Array TRange} (hr : P.root < P.defs.size)
    (hk : RootFirst P) (hin : InputsRead P) (h : Flock.TemplateQuery.evaluate (Desc.ofProg P) ts = .ok rs) :
    rs.toList.map (·.node) = (List.range (P.defs[P.root]!).body.nodes.size).filter
        (fun k => !isInputId (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).key) ∧
      ∀ r ∈ rs.toList, r.node < (P.defs[P.root]!).body.nodes.size ∧
        r.template = (P.defs[((P.defs[P.root]!).body.nodes[r.node]!).fn]!).key ∧
        (((P.defs[P.root]!).body.nodes[r.node]!).form = .call ∨
          ((P.defs[P.root]!).body.nodes[r.node]!).form = .batch) ∧
        r.count = ((P.defs[P.root]!).body.nodes[r.node]!).members := by
  obtain ⟨root, hroot, hnodes, hall⟩ := evaluate_spec h
  rw [ofProg_root hr hk] at hroot
  cases hroot
  have hsize : (dOf (P.defs[P.root]!)).2.nodes.size = (P.defs[P.root]!).body.nodes.size := by
    show (Array.map _ _).size = _
    rw [Array.size_map]
  have hnode : ∀ k < (P.defs[P.root]!).body.nodes.size,
      (dOf (P.defs[P.root]!)).2.nodes[k]! =
        { form := ((P.defs[P.root]!).body.nodes[k]!).form.name, fn := ((P.defs[P.root]!).body.nodes[k]!).fn,
          n := ((P.defs[P.root]!).body.nodes[k]!).n } := by
    intro k hk'
    show (Array.map _ _)[k]! = _
    rw [getElem!_pos _ k (by rw [Array.size_map]; exact hk'), Array.getElem_map,
      getElem!_pos (P.defs[P.root]!).body.nodes k hk']
  have htid : ∀ k < (P.defs[P.root]!).body.nodes.size,
      tidAt (Desc.ofProg P) (dOf (P.defs[P.root]!)).2 k = (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).key := by
    intro k hk'
    unfold tidAt
    rw [hnode k hk', callee_key]
  have hlist : rs.toList.map (·.node) = (List.range (P.defs[P.root]!).body.nodes.size).filter
      (fun k => !isInputId (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).key) := by
    rw [hnodes, hsize]
    apply List.filter_congr
    intro k hk'
    have hk'' : k < (P.defs[P.root]!).body.nodes.size := List.mem_range.1 hk'
    have hq : (Desc.ofProg P).isInputNode (dOf (P.defs[P.root]!)).2.nodes[k]! =
        isInputId (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).key := by
      rw [ofProg_isInputNode, hnode k hk'']
      show (isInputId (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).key &&
        ((Desc.ofProg P).def? (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).key).any (·.prim)) = _
      cases hI : isInputId (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).key with
      | false => rfl
      | true => rw [hin k hk'' hI] <;> rfl
    rw [hq]
  refine ⟨hlist, fun r hr' => ?_⟩
  have hlt : r.node < (P.defs[P.root]!).body.nodes.size := by
    have : r.node ∈ rs.toList.map (·.node) := List.mem_map.2 ⟨r, hr', rfl⟩
    rw [hlist] at this
    exact List.mem_range.1 (List.mem_filter.1 this).1
  obtain ⟨h1, h2, h3⟩ := hall r hr'
  rw [hnode r.node hlt] at h2 h3
  rw [htid r.node hlt] at h1
  refine ⟨hlt, h1, ?_, ?_⟩
  · cases hf : ((P.defs[P.root]!).body.nodes[r.node]!).form <;> rw [hf] at h2 <;> simp [Form.name] at h2 ⊢
  · rw [h3]
    unfold Node.members
    cases hf : ((P.defs[P.root]!).body.nodes[r.node]!).form <;> rw [hf] at h2
    · simp [Form.name] at h2
    · rfl
    · rfl
    · simp [Form.name] at h2

end NciCheck.Record.SecurityProofs
