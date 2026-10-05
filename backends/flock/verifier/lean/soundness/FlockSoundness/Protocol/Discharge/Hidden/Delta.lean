import FlockSoundness.Protocol.Discharge.Hidden.Defs

/-!
# A hidden-output statement's Δ (`HiddenDelta`)

`HmOut.delta` is `HmRow.delta`'s pairs, then the output rows' message bits: per VU `g`, output port `j`, row word `k` and
bit `t < 16`, the pairs `outPairs` lists. Its loops are `forIn_range_append`'s, but for the port loop, which also carries
the VU's first output word of port `j`, `outAt c j` (`forIn_range_at`).
-/

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Hidden

theorem getElem!_eq_getD' {α : Type} [Inhabited α] (a : Array α) (i : ℕ) : a[i]! = a.getD i default := by
  by_cases hi : i < a.size
  · rw [getElem!_pos a i hi, Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hi]; rfl
  · rw [getElem!_neg a i hi, Array.getD_eq_getD_getElem?, Array.getElem?_eq_none (by omega)]; rfl

theorem outAt_succ (c : Circuit) {j : ℕ} (hj : j < c.outPorts.size) :
    outAt c (j + 1) = outAt c j + c.outPorts[j]!.2 := by
  unfold outAt
  rw [List.take_add_one, List.map_append, List.sum_append, List.getElem?_eq_getElem (by simpa using hj),
    getElem!_pos c.outPorts j hj]
  simp

theorem run_pair (m : Id (Array (ℕ × ℕ))) (da db : Array (ℕ × ℕ)) (L : List (ℕ × ℕ)) (hm : (Id.run m).toList = L) :
    (m >>= fun s => pure (da ++ s, db ++ s) : Id _).run = (da ++ L.toArray, db ++ L.toArray) := by
  show (da ++ Id.run m, db ++ Id.run m) = _
  rw [← hm, Array.toArray_toList]

theorem hidden_delta' : HiddenDelta := by
  intro c pin
  unfold HmOut.delta
  simp only
  refine run_pair _ _ _ _ ?_
  refine forIn_range_append _ _ (fun g _ d => ⟨_, rfl, ?_⟩) #[]
  refine forIn_range_append _ _ (fun j _ d => ⟨_, rfl, ?_⟩) d
  refine forIn_range_append _ _ (fun k _ d => ⟨_, rfl, ?_⟩) d
  refine forIn_range_append _ _ (fun t _ d => ⟨_, rfl, ?_⟩) d
  simp only [Array.toList_push, List.append_assoc, List.cons_append, List.nil_append]
  rfl

end FlockSoundness.Discharge.Hidden
