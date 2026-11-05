import NetworkCheck.Circuit

/-!
# `keysUnique` accepts only distinct keys

`keysUnique` folds a set of the keys seen over the Program's Definitions; `keysUnique_foldl` reads the fold's flag as
the keys before it unseen and the rest distinct, and `keysUnique_nodup` states the acceptance as the Definitions' keys
`Nodup`, which a lookup by key needs to find the Definition at a node. It imports only the check, so a proof that
reads the scan's or a circuit record's acceptance imports it without the period's proofs.
-/

namespace Properties.NetworkPeriod

theorem keysUnique_foldl : ∀ (l : List Flock.Program.Defn) (seen : Std.HashMap String Unit) (ok : Bool),
    (l.foldl (fun (acc : Std.HashMap String Unit × Bool) d =>
      (acc.1.insert d.key (), acc.2 && !acc.1.contains d.key)) (seen, ok)).2 = true →
    ok = true ∧ (∀ d ∈ l, seen.contains d.key = false) ∧ (l.map (·.key)).Nodup
  | [], seen, ok, h => ⟨h, fun _ hd => by simp at hd, List.nodup_nil⟩
  | d :: l, seen, ok, h => by
    rw [List.foldl_cons] at h
    obtain ⟨hok, hl, hnd⟩ := keysUnique_foldl l _ _ h
    have hok' : ok = true ∧ seen.contains d.key = false := by simpa using hok
    have hl' : ∀ d' ∈ l, d.key ≠ d'.key ∧ seen.contains d'.key = false := fun d' hd' => by
      have := hl d' hd'
      rw [Std.HashMap.contains_insert, Bool.or_eq_false_iff] at this
      exact ⟨fun he => by simp [he] at this, this.2⟩
    refine ⟨hok'.1, fun d' hd' => ?_, ?_⟩
    · rcases List.mem_cons.1 hd' with rfl | hd'
      · exact hok'.2
      · exact (hl' d' hd').2
    · rw [List.map_cons, List.nodup_cons]
      refine ⟨fun hm => ?_, hnd⟩
      obtain ⟨d', hd', he⟩ := List.mem_map.1 hm
      exact (hl' d' hd').1 he.symm

/-- **`keysUnique`'s acceptance**: no two of the Program's Definitions share a key. -/
theorem keysUnique_nodup {P : Flock.Program.Prog} (h : NetworkCheck.Period.keysUnique P = true) :
    (P.defs.toList.map (·.key)).Nodup := by
  unfold NetworkCheck.Period.keysUnique at h
  rw [← Array.foldl_toList] at h
  exact (keysUnique_foldl _ _ _ h).2.2

end Properties.NetworkPeriod
