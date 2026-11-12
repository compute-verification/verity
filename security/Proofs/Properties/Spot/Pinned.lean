module

public import Properties.Spot

/-! # R7 holds each pinned value to its pinned root (`PinnedHeld`)

`Registration.checkRegistered` holds the record's entry for each of the verifier's own values it doesn't fill to the
first root `held` names for that value, and `Registration.check` hands it the pinned roots, then the log's, behind each
output value's root (`Registration.applyOutputs`). An output step leaves every other value's entries and first held
root as they are (`applyOutputs_keeps`), so where R7 holds, a value the verifier pinned and takes from no earlier run is
held to its pinned root (`pinnedHeld_of_R7`).
-/

@[expose] public section

open Lean (Json)
open Spot

namespace Properties.Spot.Pinned

/-- One output step leaves a value other than the output's: each of its entries stays, and so does its first held
root. -/
theorem applyOutput_keeps (log : Array Json) (u : String)
    (s : Array String × Array Registered.Entry × Array (String × String)) (o : String × String × Registration.Earlier)
    (ho : o.1 ≠ u) :
    (∀ want ∈ s.2.1, want.value = u → want ∈ (Registration.applyOutput log s o).2.1) ∧
      (Registration.applyOutput log s o).2.2.find? (·.1 == u) = s.2.2.find? (·.1 == u) := by
  obtain ⟨c, mine, held⟩ := s
  obtain ⟨v, d, p⟩ := o
  simp only at ho
  simp only [Registration.applyOutput]
  split
  · exact ⟨fun _ hw _ => hw, rfl⟩
  · split
    · refine ⟨fun want hw hu => Array.mem_map.2 ⟨want, hw, ?_⟩, ?_⟩
      · have : (want.value == v) = false := by simpa [hu] using Ne.symm ho
        simp [this]
      · have : (v == u) = false := by simpa using ho
        simp [Array.find?_append, this]
    · exact ⟨fun _ hw _ => hw, rfl⟩

theorem foldl_keeps (log : Array Json) (u : String) :
    ∀ (l : List (String × String × Registration.Earlier))
      (s : Array String × Array Registered.Entry × Array (String × String)), (∀ o ∈ l, o.1 ≠ u) →
      (∀ want ∈ s.2.1, want.value = u → want ∈ (l.foldl (Registration.applyOutput log) s).2.1) ∧
        (l.foldl (Registration.applyOutput log) s).2.2.find? (·.1 == u) = s.2.2.find? (·.1 == u)
  | [], _, _ => ⟨fun _ hw _ => hw, rfl⟩
  | o :: os, s, hl => by
    rw [List.foldl_cons]
    have hstep := applyOutput_keeps log u s o (hl o (List.mem_cons_self ..))
    obtain ⟨h1, h2⟩ := foldl_keeps log u os (Registration.applyOutput log s o)
      (fun o' ho' => hl o' (List.mem_cons_of_mem _ ho'))
    exact ⟨fun want hw hv => h1 want (hstep.1 want hw hv) hv, h2.trans hstep.2⟩

/-- The output steps leave a value no output names: each of its entries stays, and so does its first held root. -/
theorem applyOutputs_keeps (log : Array Json) (c : Array String) (mine : Array Registered.Entry)
    (held : Array (String × String)) (outputs : Array (String × String × Registration.Earlier)) (u : String)
    (hu : outputs.any (·.1 == u) = false) :
    (∀ want ∈ mine, want.value = u → want ∈ (Registration.applyOutputs c mine held outputs log).2.1) ∧
      (Registration.applyOutputs c mine held outputs log).2.2.find? (·.1 == u) = held.find? (·.1 == u) := by
  unfold Registration.applyOutputs
  rw [← Array.foldl_toList]
  refine foldl_keeps log u outputs.toList (c, mine, held) fun o ho heq => ?_
  have h : outputs.any (·.1 == u) = true := Array.any_eq_true'.2 ⟨o, Array.mem_toList_iff.1 ho, by simp [heq]⟩
  rw [hu] at h
  cases h

end Properties.Spot.Pinned

namespace Properties.Spot

/-- **R7 holds each pinned value to its pinned root**: where R7 holds, the record's entry for each of the verifier's own
values that it pinned, neither fills nor takes from an earlier run, has its pinned root (`PinnedHeld`). -/
theorem pinnedHeld_of_R7 {rec : Json} {v : Registration.Verifier} (h : R7 rec v) : PinnedHeld rec v := by
  obtain ⟨mine, es, mine', held', hown, hes, -, happ, -, -, hentry, -⟩ := h
  intro mine0 hmine want hw hs ho x hx e he
  rw [hown] at hmine
  cases hmine
  obtain ⟨hkeep, hfind⟩ := Pinned.applyOutputs_keeps v.log #[] mine
    (v.pinned ++ Registration.heldRoots v.log (Registration.ownEntries v)) v.outputs want.value ho
  rw [happ] at hkeep hfind
  dsimp only at hkeep hfind
  have he' : es.reverse.find? (fun e => jeq (getD e "value") (.str want.value)) = some e := by
    simpa only [Registration.entryOf, hes] using he
  have hroot := (hentry want (hkeep want hw rfl) e he').2.2.2.2.2
  have hheld : heldFor v.supplied held' want.value = held' := by
    simp only [heldFor, hs, Bool.false_eq_true, ↓reduceIte]
  have hx' : v.pinned.find? (·.1 == want.value) = some x := hx
  rw [hheld, hfind, Array.find?_append, hx', Option.some_or] at hroot
  exact hroot x rfl

end Properties.Spot
