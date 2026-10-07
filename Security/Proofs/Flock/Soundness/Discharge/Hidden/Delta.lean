import Proofs.Flock.Soundness.Discharge.Hidden.Defs

/-!
# A hidden-output statement's Δ (`HiddenDelta`)

`HmOut.delta` is `HmRow.delta`'s pairs, then the output rows' message bits: per template instance `g`, output port `j`, row word `k` and
bit `t < 16`, the pairs `outPairs` lists. Its loops are `forIn_range_append`'s, but for the port loop, which also carries
the template instance's first output word of port `j`, `outAt c j` (`forIn_range_at`).
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

/-- A loop over `range' s m` whose state is an array and a counter, `A a` at step `a`: each step appends `G a` and adds
`W a` to the counter. -/
theorem forIn_list_at {γ : Type} (G : ℕ → List γ) (A W : ℕ → ℕ)
    {f : ℕ → Array γ × ℕ → Id (ForInStep (Array γ × ℕ))} :
    ∀ (m s : ℕ), (∀ a, s ≤ a → a < s + m → A (a + 1) = A a + W a) →
    (∀ a, s ≤ a → a < s + m → ∀ d, ∃ d', f a (d, A a) = ForInStep.yield (d', A a + W a) ∧
      d'.toList = d.toList ++ G a) →
    ∀ d : Array γ, (forIn (List.range' s m) (d, A s) f).1.toList = d.toList ++ (List.range' s m).flatMap G
  | 0, s, _, _, d => by simp only [List.range'_zero, List.forIn_nil, List.flatMap_nil, List.append_nil]; rfl
  | m + 1, s, hA, hf, d => by
    rw [List.range'_succ, List.forIn_cons]
    obtain ⟨d', hab, hd'⟩ := hf s le_rfl (by omega) d
    rw [hab, ← hA s le_rfl (by omega)]
    show (forIn (List.range' (s + 1) m) (d', A (s + 1)) f).1.toList = _
    rw [forIn_list_at G A W m (s + 1) (fun a h1 h2 => hA a (by omega) (by omega))
      (fun a h1 h2 d => hf a (by omega) (by omega) d) d', hd']
    simp

theorem forIn_range_at {γ : Type} (G : ℕ → List γ) (A W : ℕ → ℕ) (n : ℕ)
    {f : ℕ → Array γ × ℕ → Id (ForInStep (Array γ × ℕ))} (hA : ∀ a < n, A (a + 1) = A a + W a)
    (hf : ∀ a < n, ∀ d, ∃ d', f a (d, A a) = ForInStep.yield (d', A a + W a) ∧ d'.toList = d.toList ++ G a)
    (d : Array γ) : (forIn [:n] (d, A 0) f).1.toList = d.toList ++ (List.range n).flatMap G := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  have := forIn_list_at G A W [:n].size 0
    (fun a _ h2 => hA a (by simp only [Std.Legacy.Range.size] at h2; omega))
    (fun a _ h2 d => hf a (by simp only [Std.Legacy.Range.size] at h2; omega) d) d
  simpa [Std.Legacy.Range.size, List.range_eq_range'] using this

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
