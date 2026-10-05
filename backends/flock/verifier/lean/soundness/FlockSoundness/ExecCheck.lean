import FlockSoundness.ExecPlacement
import FlockSoundness.Protocol.ExecCheck

/-!
# What `HmRow.check` accepted

The layout facts of an `hm96-sha512/row/v1` circuit that `HmRow.check` checks, as `ExecSetup.lean` uses them: the
ranges inside the block, pairwise apart, in slots of their nets' rows; the unit a text net with one range of
`vus_per_block × units_per_vu` slots and a group of 16-bit leaf ports; the row slots' ports; every net range's slots per
VU; one range per net; the wires' and leaf cuts' bounds; the leaf maps.
-/

open Flock FlockLevel3

namespace FlockSoundness

/-! ## Steps of an accepted check -/

/-! ## `HmRow.check`'s count of each port's wirings -/

theorem wcount_bump (wired : Array ((ℕ × ℕ × ℕ) × ℕ)) (key key' : ℕ × ℕ × ℕ) :
    wcount (bump wired key) key' = wcount wired key' + if key = key' then 1 else 0 := by
  unfold wcount bump
  split
  · rename_i i hi
    obtain ⟨hlt, hk, hmin⟩ := Array.findIdx?_eq_some_iff_getElem.1 hi
    have hl : i < wired.toList.length := by simpa using hlt
    have hsplit : wired.toList = wired.toList.take i ++ wired[i] :: wired.toList.drop (i + 1) := by
      rw [← Array.getElem_toList (h := hlt), List.getElem_cons_drop, List.take_append_drop]
    have hmod : (wired.modify i fun (k, v) => (k, v + 1)).toList =
        wired.toList.take i ++ (wired[i].1, wired[i].2 + 1) :: wired.toList.drop (i + 1) := by
      rw [Array.toList_modify, List.modify_eq_take_cons_drop hl]
      rfl
    have hpre : ∀ y ∈ wired.toList.take i, (y.1 == key) = false := by
      intro y hy
      obtain ⟨j, hj, rfl⟩ := List.getElem_of_mem hy
      have hj' : j < i := by simp at hj; omega
      rw [List.getElem_take]
      simpa using hmin j hj'
    have hki : wired[i].1 = key := by simpa using hk
    rw [← Array.find?_toList, ← Array.find?_toList, hmod]
    generalize wired.toList.take i = pre at hsplit hpre ⊢
    generalize wired.toList.drop (i + 1) = post at hsplit ⊢
    rw [hsplit]
    simp only [List.find?_append, List.find?_cons]
    by_cases he : key = key'
    · subst he
      rw [List.find?_eq_none.2 (fun y hy => by simp [hpre y hy])]
      simp [hki]
    · have hne : (wired[i].1 == key') = false := by simp [hki, he]
      simp [hne, he]
  · rename_i hn
    rw [Array.findIdx?_eq_none_iff] at hn
    rw [← Array.find?_toList, ← Array.find?_toList, Array.toList_push, List.find?_append]
    by_cases he : key = key'
    · subst he
      rw [List.find?_eq_none.2 (fun y hy => by simp [hn y (Array.mem_toList_iff.1 hy)])]
      simp
    · simp [he]

end FlockSoundness
