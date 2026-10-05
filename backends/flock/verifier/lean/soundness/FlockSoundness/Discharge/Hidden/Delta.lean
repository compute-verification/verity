import FlockSoundness.Discharge.Hidden.Defs
import FlockSoundness.Protocol.Discharge.Hidden.Delta

/-!
# A hidden-output statement's Δ (`HiddenDelta`)

`HmOut.delta` is `HmRow.delta`'s pairs, then the output rows' message bits: per VU `g`, output port `j`, row word `k` and
bit `t < 16`, the pairs `outPairs` lists. Its loops are `forIn_range_append`'s, but for the port loop, which also carries
the VU's first output word of port `j`, `outAt c j` (`forIn_range_at`).
-/

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Hidden

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

end FlockSoundness.Discharge.Hidden
