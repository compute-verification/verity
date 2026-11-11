module

public import Proofs.Properties.Nci.RecordJson

/-!
# The root's ranges

`Flock.TemplateQuery.evaluate` reads the root's nodes in order and gives one range a node that is no input: the
node, its callee's id as the template, and its members as the count. `evaluate_spec` states it: the ranges' nodes
are the root's nodes whose callee's id is no input id, in order, each a call or a batch, a call with one unit and a
batch with its `n`.
-/

@[expose] public section

namespace NciCheck.Record.SecurityProofs

open Flock.TemplateQuery (Desc DDef TRange isInput)

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

/-- **The root's ranges are its nodes that are no input, in order**: each a call or a batch of its callee's id, a
call with one unit and a batch with its `n`. -/
theorem evaluate_spec {d : Desc} {ts : Array String} {rs : Array TRange} (h : Flock.TemplateQuery.evaluate d ts = .ok rs) :
    ∃ root, d.def? d.root = some root ∧
      rs.toList.map (·.node) = (List.range root.nodes.size).filter (fun k => !isInput (tidAt d root k)) ∧
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
        s.1.toList.map (·.node) = done.filter (fun k => !isInput (tidAt d root k)) ∧
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
        have hin' : isInput (tidAt d root k) = true := hin
        refine ⟨_, rfl, ?_, hi.2⟩
        rw [List.filter_append, hi.1]
        simp [hin']
      · rename_i hin
        have hin' : ¬ isInput (tidAt d root k) = true := hin
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

end NciCheck.Record.SecurityProofs
