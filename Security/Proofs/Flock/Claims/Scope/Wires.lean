import Definitions.Claims

/-!
# A circuit's wires, gate by gate

`Claims.Circuit.wires` appends one wire per gate. `foldl_step_agree`: when every gate reads only earlier wires and computes
what a valuation `v` says its wire holds, every wire is `v`'s. `foldl_step_prefix`: later gates leave earlier wires alone.
-/

namespace Flock.Readable

open Claims

/-- A gate's value under a valuation of the wires. -/
def gv (v : ℕ → Bool) : Gate → Bool
  | .and i j => v i && v j
  | .xor i j => xor (v i) (v j)

/-- A gate reads only wires below `n`. -/
def Reads (n : ℕ) : Gate → Prop
  | .and i j => i < n ∧ j < n
  | .xor i j => i < n ∧ j < n

/-- One gate of `Circuit.wires`: its value on the wires so far, appended. -/
def step (ws : List Bool) (g : Gate) : List Bool := ws ++ [gv (fun k => ws.getD k false) g]

theorem wires_eq (C : Circuit) (inp : List Bool) : C.wires inp = C.gates.foldl step inp := by
  unfold Circuit.wires step
  congr

theorem foldl_step_length : ∀ (G : List Gate) (ws : List Bool), (G.foldl step ws).length = ws.length + G.length
  | [], ws => by simp
  | g :: G, ws => by
    rw [List.foldl_cons, foldl_step_length G]
    simp [step]
    omega

theorem foldl_step_prefix : ∀ (G : List Gate) (ws : List Bool) (k : ℕ), k < ws.length →
    (G.foldl step ws).getD k false = ws.getD k false
  | [], _, _, _ => rfl
  | g :: G, ws, k, hk => by
    rw [List.foldl_cons, foldl_step_prefix G _ k (by simp [step]; omega)]
    simp only [step, List.getD_eq_getElem?_getD, List.getElem?_append_left hk]

theorem gv_congr {v v' : ℕ → Bool} {n : ℕ} {g : Gate} (hr : Reads n g) (h : ∀ k < n, v k = v' k) :
    gv v g = gv v' g := by
  cases g with
  | and i j => simp only [Reads] at hr; simp only [gv, h i hr.1, h j hr.2]
  | xor i j => simp only [Reads] at hr; simp only [gv, h i hr.1, h j hr.2]

/-- **Every wire is the valuation's**, when the wires so far are and every gate reads earlier wires and computes its own
wire's value. -/
theorem foldl_step_agree (v : ℕ → Bool) : ∀ (G : List Gate) (ws : List Bool),
    (∀ k < ws.length, ws.getD k false = v k) →
    (∀ t (ht : t < G.length), Reads (ws.length + t) G[t] ∧ gv v G[t] = v (ws.length + t)) →
    ∀ k < ws.length + G.length, (G.foldl step ws).getD k false = v k
  | [], ws, hws, _, k, hk => by simpa using hws k (by simpa using hk)
  | g :: G, ws, hws, hG, k, hk => by
    rw [List.foldl_cons]
    obtain ⟨hr, hv⟩ := hG 0 (by simp)
    simp only [List.getElem_cons_zero, Nat.add_zero] at hr hv
    have hstep : ∀ k < (step ws g).length, (step ws g).getD k false = v k := by
      intro k hk
      simp only [step, List.length_append, List.length_singleton] at hk
      rcases Nat.lt_succ_iff_lt_or_eq.1 hk with hk | rfl
      · simp only [step, List.getD_eq_getElem?_getD, List.getElem?_append_left hk]
        simpa [List.getD_eq_getElem?_getD] using hws k hk
      · simp only [step, List.getD_eq_getElem?_getD, List.getElem?_append_right (le_refl _), Nat.sub_self,
          List.getElem?_singleton]
        rw [← hv]
        exact gv_congr hr fun k hk => (hws k hk)
    have hlen : (step ws g).length = ws.length + 1 := by simp [step]
    refine foldl_step_agree v G (step ws g) hstep (fun t ht => ?_) k (by simp at hk; omega)
    obtain ⟨hr', hv'⟩ := hG (t + 1) (by simp; omega)
    simp only [List.getElem_cons_succ] at hr' hv'
    rw [hlen, Nat.add_assoc, Nat.add_comm 1 t]
    exact ⟨hr', hv'⟩

end Flock.Readable
