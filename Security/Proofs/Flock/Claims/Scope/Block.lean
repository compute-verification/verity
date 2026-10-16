import Proofs.Flock.Claims.Scope.Wires

/-!
# One R1CS row over GF(2) as AND/XOR gates

`rowGates z off b N A B`: from wire `b` on, the XOR of the `N` inputs from `off` on that `A` selects, then of those `B`
selects, each chain starting from the zero wire `z`, then their AND. `row_wires`: after these `2N + 1` gates, wire
`b + 2N` holds `xsum A x N && xsum B x N`, where `xsum A x n` is the XOR of `A r && x r` over `r < n`.
-/

namespace Flock.Readable

open Claims

/-- The XOR of `A r && x r` over `r < n`. -/
def xsum (A x : ℕ → Bool) (n : ℕ) : Bool := (List.range n).foldl (fun acc r => xor acc (A r && x r)) false

theorem xsum_succ (A x : ℕ → Bool) (n : ℕ) : xsum A x (n + 1) = xor (xsum A x n) (A n && x n) := by
  simp [xsum, List.range_succ, List.foldl_append]

/-- **One R1CS row's gates**: from wire `b` on, the XOR chain of the inputs `off + r` that `A` selects, the XOR chain of
those `B` selects, each from the zero wire `z`, and their AND. -/
def rowGates (z off b N : ℕ) (A B : ℕ → Bool) : List Gate :=
  (List.range N).map (fun r => Gate.xor (if r = 0 then z else b + r - 1) (if A r then off + r else z)) ++
    (List.range N).map (fun s => Gate.xor (if s = 0 then z else b + N + s - 1) (if B s then off + s else z)) ++
    [Gate.and (b + N - 1) (b + 2 * N - 1)]

theorem rowGates_length (z off b N : ℕ) (A B : ℕ → Bool) : (rowGates z off b N A B).length = 2 * N + 1 := by
  simp [rowGates]
  omega

/-- What the row's wires hold: the wires before `b` as they are, the two chains' partial XORs, and the AND. -/
def rowVal (ws : List Bool) (b N : ℕ) (A B x : ℕ → Bool) (k : ℕ) : Bool :=
  if k < b then ws.getD k false
  else if k < b + N then xsum A x (k - b + 1)
  else if k < b + 2 * N then xsum B x (k - b - N + 1)
  else xsum A x N && xsum B x N

/-- **After one row's gates**, the wires before it are unchanged and wire `b + 2N` is the row's AND. -/
theorem row_wires {z off b N : ℕ} {A B x : ℕ → Bool} {ws : List Bool} (hN : 0 < N) (hb : ws.length = b) (hz : z < b)
    (hzv : ws.getD z false = false) (hoff : off + N ≤ b) (hx : ∀ r < N, ws.getD (off + r) false = x r) :
    ((rowGates z off b N A B).foldl step ws).length = b + 2 * N + 1 ∧
      (∀ k < b, ((rowGates z off b N A B).foldl step ws).getD k false = ws.getD k false) ∧
      ((rowGates z off b N A B).foldl step ws).getD (b + 2 * N) false = (xsum A x N && xsum B x N) := by
  refine ⟨by rw [foldl_step_length, rowGates_length, hb]; omega, fun k hk => foldl_step_prefix _ _ k (by omega), ?_⟩
  have hagree := foldl_step_agree (rowVal ws b N A B x) (rowGates z off b N A B) ws
    (fun k hk => by simp only [rowVal, hb ▸ hk, ↓reduceIte])
    (fun t ht => ?_) (b + 2 * N) (by rw [rowGates_length, hb]; omega)
  · rw [hagree]
    simp only [rowVal, show ¬ b + 2 * N < b by omega, show ¬ b + 2 * N < b + N by omega,
      show ¬ b + 2 * N < b + 2 * N by omega, ↓reduceIte]
  rw [rowGates_length] at ht
  rw [hb]
  have vz : rowVal ws b N A B x z = false := by simp only [rowVal, hz, ↓reduceIte, hzv]
  have vx : ∀ r < N, rowVal ws b N A B x (off + r) = x r := fun r hr => by
    simp only [rowVal, show off + r < b by omega, ↓reduceIte, hx r hr]
  have vsel : ∀ (S : ℕ → Bool) r, r < N → rowVal ws b N A B x (if S r then off + r else z) = (S r && x r) := by
    intro S r hr
    cases S r
    · simp [vz]
    · simp [vx r hr]
  rcases Nat.lt_or_ge t N with h1 | h1
  · have hg : (rowGates z off b N A B)[t]'(by rw [rowGates_length]; omega) =
        Gate.xor (if t = 0 then z else b + t - 1) (if A t then off + t else z) := by
      simp only [rowGates, List.append_assoc]
      rw [List.getElem_append_left (by simpa using h1)]
      simp
    rw [hg]
    refine ⟨⟨by split <;> omega, by split <;> omega⟩, ?_⟩
    simp only [gv, vsel A t h1]
    have hval : rowVal ws b N A B x (b + t) = xsum A x (t + 1) := by
      simp only [rowVal, show ¬ b + t < b by omega, show b + t < b + N by omega, ↓reduceIte]
      congr 1
      omega
    rw [hval, xsum_succ]
    split
    · subst t
      simp [vz, xsum]
    · rename_i ht0
      congr 1
      simp only [rowVal, show ¬ b + t - 1 < b by omega, show b + t - 1 < b + N by omega, ↓reduceIte]
      congr 1
      omega
  rcases Nat.lt_or_ge t (2 * N) with h2 | h2
  · have hg : (rowGates z off b N A B)[t]'(by rw [rowGates_length]; omega) =
        Gate.xor (if t - N = 0 then z else b + N + (t - N) - 1) (if B (t - N) then off + (t - N) else z) := by
      simp only [rowGates, List.append_assoc]
      rw [List.getElem_append_right (by simpa using h1)]
      rw [List.getElem_append_left (by simp; omega)]
      simp
    rw [hg]
    refine ⟨⟨by split <;> omega, by split <;> omega⟩, ?_⟩
    simp only [gv, vsel B (t - N) (by omega)]
    have hval : rowVal ws b N A B x (b + t) = xsum B x (t - N + 1) := by
      simp only [rowVal, show ¬ b + t < b by omega, show ¬ b + t < b + N by omega, show b + t < b + 2 * N by omega,
        ↓reduceIte]
      congr 1
      omega
    rw [hval, show t - N + 1 = (t - N) + 1 from rfl, xsum_succ]
    split
    · rename_i h0
      rw [h0]
      simp [vz, xsum]
    · rename_i h0
      congr 1
      simp only [rowVal, show ¬ b + N + (t - N) - 1 < b by omega, show ¬ b + N + (t - N) - 1 < b + N by omega,
        show b + N + (t - N) - 1 < b + 2 * N by omega, ↓reduceIte]
      congr 1
      omega
  · have ht2 : t = 2 * N := by omega
    subst ht2
    have hg : (rowGates z off b N A B)[2 * N]'(by rw [rowGates_length]; omega) = Gate.and (b + N - 1) (b + 2 * N - 1) := by
      simp only [rowGates, List.append_assoc]
      rw [List.getElem_append_right (by simp; omega)]
      rw [List.getElem_append_right (by simp; omega)]
      simp
    rw [hg]
    refine ⟨⟨by omega, by omega⟩, ?_⟩
    simp only [gv, rowVal, show ¬ b + 2 * N < b by omega, show ¬ b + 2 * N < b + N by omega,
      show ¬ b + 2 * N < b + 2 * N by omega, show ¬ b + N - 1 < b by omega, show b + N - 1 < b + N by omega,
      show ¬ b + 2 * N - 1 < b by omega, show ¬ b + 2 * N - 1 < b + N by omega, show b + 2 * N - 1 < b + 2 * N by omega,
      ↓reduceIte]
    rw [show b + N - 1 - b + 1 = N by omega, show b + 2 * N - 1 - b - N + 1 = N by omega]

end Flock.Readable
