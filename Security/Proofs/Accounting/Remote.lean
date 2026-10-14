import Models.Accounting.Guarantees.Remote
import Proofs.Pouw.Prob
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

/-!
# The remote audit's compositions, from their routes

Each statement of `Models.Accounting.Guarantees.Remote` follows from the routes it takes, as hypotheses, by a union
bound over the routes' errors (`holds_and`) and, at each outcome, the γ arithmetic. The correct receipts' `(1 − ε) · C`
bits were paid by the node's reads, its state at the pass boundaries (at most `μ_chip · C`), its rebuilt bits (at most
`μ_rebuild · C`, by the rebuild's work at π_r and its fit beside `W_r`) and the guessing slack (`λ · C`); under
Identity nothing outside the node paid. `μ_host` then converts `C` into `C_true` (`muHost_mul`). Without Identity the
bits carried over the boundary and the bound `z` computed outside add `μ_link` and `μ_out`.

These are lemmas, not guarantees: none is listed in `lean-audit.json` until the routes are proved for the runs the
actual circuit realizes.
-/

namespace Accounting.SecurityProofs.Remote

open Accounting.Remote Pouw

variable {τ : Type} {N : Node τ} {a : Audit} {Ω : Type} {r : Run N a Ω}

section Holds

variable [Fintype Ω]

theorem holds_mono {δ : ℝ} {P Q : Ω → Prop} (h : ∀ ω, r.accepts ω → P ω → Q ω) (hp : r.Holds δ P) :
    r.Holds δ Q :=
  (pr_mono fun ω ⟨ha, hq⟩ => ⟨ha, fun hp => hq (h ω ha hp)⟩).trans hp

/-- The union bound: two routes hold together except with their errors summed. -/
theorem holds_and {δ₁ δ₂ : ℝ} {P Q : Ω → Prop} (h₁ : r.Holds δ₁ P) (h₂ : r.Holds δ₂ Q) :
    r.Holds (δ₁ + δ₂) fun ω => P ω ∧ Q ω := by
  refine (pr_mono (P' := fun ω => (r.accepts ω ∧ ¬ P ω) ∨ (r.accepts ω ∧ ¬ Q ω)) ?_).trans
    ((pr_or_le _ _).trans (add_le_add h₁ h₂))
  intro ω ⟨ha, hn⟩
  by_cases hp : P ω
  · exact Or.inr ⟨ha, fun hq => hn ⟨hp, hq⟩⟩
  · exact Or.inl ⟨ha, hp⟩

/-- A bound on every accepting outcome is a route at `δ = 0`. -/
theorem holds_of_forall {P : Ω → Prop} (h : ∀ ω, r.accepts ω → P ω) : r.Holds 0 P :=
  (pr_eq_zero fun ω ⟨ha, hn⟩ => hn (h ω ha)).le

/-- The rebuild's fit at `W_r = 0` is the Compute row alone. -/
theorem fit_zero (h : ∀ ω, r.rebuildWork ω ≤ N.rebuild * a.T) : r.Fit 0 0 :=
  holds_of_forall fun ω _ => by
    have : (r.rebuildWork ω : ℝ) ≤ N.rebuild * a.T := by exact_mod_cast h ω
    linarith

/-- The rebuild's fit beside `W_r`, from W at π_r and PoUW's certificate of `W_r`. -/
theorem fit_of_certified {Wr δ : ℝ} (hw : Assumptions.ComputeAdditive r) (hc : r.Certified Wr δ) : r.Fit Wr δ :=
  holds_mono (fun ω _ h => by
    have : (r.rebuildWork ω : ℝ) + r.pouwWork ω ≤ N.rebuild * a.T := by exact_mod_cast hw ω
    linarith) hc

end Holds

variable [Fintype τ]

theorem declared_le (N : Node τ) (T : ℕ) : N.declared T ≤ N.capacity T :=
  Finset.sum_le_sum fun m _ => Nat.mul_le_mul_right T (N.rate_le m)

theorem reads_le {ω : Ω} (h : ∀ m, r.read ω m ≤ N.ceil m * a.T) : r.reads ω ≤ N.capacity a.T :=
  Finset.sum_le_sum fun m _ => h m

/-- `μ_host` converts `C` into `C_true`. -/
theorem muHost_mul (hC : 0 < N.declared a.T) :
    muHost N a * N.capacity a.T = (N.capacity a.T : ℝ) - N.declared a.T := by
  have : (0 : ℝ) < N.capacity a.T := by exact_mod_cast hC.trans_le (declared_le N a.T)
  unfold muHost
  field_simp

/-- At an outcome meeting `ChipState`, `Rebuild` and `Fit W_r`: the state, the rebuilt bits and the guessing slack are
within `(μ_chip + μ_rebuild + λ) · C`, a nonnegative share. -/
theorem charges_le {Wr : ℝ} {ω : Ω} (hC : 0 < N.declared a.T) (hcw : 0 < a.cWord)
    (hm : r.state ω ≤ a.P * (N.chip + 1)) (hr : r.rebuilt ω * a.cWord ≤ a.b * r.rebuildWork ω)
    (hf : Wr + r.rebuildWork ω ≤ N.rebuild * a.T) :
    (r.state ω + r.rebuilt ω + a.P * a.lam : ℝ) ≤ (muChip N a + muRebuild N a Wr + muLambda N a) * N.declared a.T ∧
      0 ≤ muChip N a + muRebuild N a Wr + muLambda N a := by
  have hC0 : (0 : ℝ) < N.declared a.T := by exact_mod_cast hC
  have hcw' : (0 : ℝ) < a.cWord := by exact_mod_cast hcw
  have hb : (0 : ℝ) ≤ a.b := Nat.cast_nonneg _
  have hw0 : (0 : ℝ) ≤ r.rebuildWork ω := Nat.cast_nonneg _
  have f3 : (r.state ω : ℝ) ≤ a.P * (N.chip + 1) := by exact_mod_cast hm
  have f4 : (r.rebuilt ω : ℝ) * a.cWord ≤ a.b * r.rebuildWork ω := by exact_mod_cast hr
  have hB : (r.rebuilt ω : ℝ) ≤ a.b * (N.rebuild * a.T - Wr) / a.cWord := by
    rw [le_div_iff₀ hcw']
    have : (a.b : ℝ) * r.rebuildWork ω ≤ a.b * (N.rebuild * a.T - Wr) :=
      mul_le_mul_of_nonneg_left (by linarith) hb
    linarith
  have eC : muChip N a * N.declared a.T = a.P * (N.chip + 1) := by unfold muChip; field_simp
  have eR : muRebuild N a Wr * N.declared a.T = a.b * (N.rebuild * a.T - Wr) / a.cWord := by
    unfold muRebuild; field_simp
  have eΛ : muLambda N a * N.declared a.T = a.P * a.lam := by unfold muLambda; field_simp
  have gC : 0 ≤ muChip N a := by unfold muChip; positivity
  have gR : 0 ≤ muRebuild N a Wr := by
    unfold muRebuild; exact div_nonneg (mul_nonneg hb (by linarith)) (by positivity)
  have gΛ : 0 ≤ muLambda N a := by unfold muLambda; positivity
  refine ⟨?_, by linarith⟩
  rw [add_mul, add_mul, eC, eR, eΛ]
  linarith

/-- Under Identity the payments at every location are the node's reads, state and rebuilt bits. -/
theorem paid_of_identity (hid : Assumptions.Identity r) (ω : Ω) :
    (r.paid ω : ℝ) = r.reads ω + r.state ω + r.rebuilt ω := by
  obtain ⟨e1, e2, e3⟩ := hid ω
  simp [Run.paid, e1, e2, e3]

variable [Fintype Ω]

/-- **Conservation, at no location**, from `Sampled` and `Paid`. -/
theorem Conservation : Accounting.Guarantees.Remote.Conservation := by
  intro τ Ω _ _ N a r ε δs δp hC hs hp
  refine holds_mono ?_ (holds_and hs hp)
  intro ω _ ⟨h1, h2⟩
  refine ⟨h1, ?_⟩
  have : (0 : ℝ) < N.declared a.T := by exact_mod_cast hC
  have e : muLambda N a * N.declared a.T = a.P * a.lam := by unfold muLambda; field_simp
  have h2' : (r.correct ω * a.b : ℝ) ≤ r.paid ω + a.P * a.lam := by exact_mod_cast h2
  nlinarith [e, h1, h2']

/-- **The node-level guarantee under Identity**, from `Sampled`, `Paid`, `ChipState`, `Rebuild` and `Fit W_r`. -/
theorem NodeLevel : Accounting.Guarantees.Remote.NodeLevel := by
  intro τ Ω _ _ N a r ε Wr δs δp δm δr δf hC hε hcw hid hs hp hm hr hf
  refine holds_mono ?_ (holds_and (holds_and (holds_and (holds_and hs hp) hm) hr) hf)
  rintro ω - ⟨⟨⟨⟨h1, h2⟩, h3⟩, h4⟩, h5⟩
  obtain ⟨hk, hg⟩ := charges_le hC hcw h3 h4 h5
  have h2' : (r.correct ω * a.b : ℝ) ≤ r.paid ω + a.P * a.lam := by exact_mod_cast h2
  rw [paid_of_identity hid] at h2'
  have hCt : (N.declared a.T : ℝ) ≤ N.capacity a.T := by exact_mod_cast declared_le N a.T
  have eH := muHost_mul hC
  have := mul_le_mul_of_nonneg_left hCt (add_nonneg hε hg)
  unfold gamma
  linarith

/-- **The node-level guarantee's counting form under Identity**, from `Sampled`, `PaidJoint`, `ChipState`, `Rebuild`,
`Fit W_r` and `Ports`. -/
theorem NodeLevelJoint : Accounting.Guarantees.Remote.NodeLevelJoint := by
  intro τ Ω _ _ N a r ε Wr δs δp δm δr δf δq hC hε hcw hid hs hp hm hr hf hq
  refine holds_mono ?_ (holds_and (holds_and (holds_and (holds_and (holds_and hs hp) hm) hr) hf) hq)
  rintro ω - ⟨⟨⟨⟨⟨h1, h2⟩, h3⟩, h4⟩, h5⟩, h6⟩
  obtain ⟨hk, hg⟩ := charges_le hC hcw h3 h4 h5
  have h2' : (r.other ω + r.correct ω * a.b : ℝ) ≤ r.paid ω + a.P * a.lam := by exact_mod_cast h2
  rw [paid_of_identity hid] at h2'
  have h6' : (r.reads ω : ℝ) ≤ N.capacity a.T := by exact_mod_cast reads_le h6
  have hCt : (N.declared a.T : ℝ) ≤ N.capacity a.T := by exact_mod_cast declared_le N a.T
  have eH := muHost_mul hC
  have := mul_le_mul_of_nonneg_left hCt (add_nonneg hε hg)
  unfold gamma
  linarith

/-- **The node-level guarantee without Identity**, from the outside work's bound, PoUW's conjecture for `z`, and the
routes `Sampled`, `CoLocated`, `ChipState`, `Rebuild`, `Fit W_r` and `Links`. -/
theorem NodeLevelLinks : Accounting.Guarantees.Remote.NodeLevelLinks := by
  intro τ Ω _ _ N a r ε Wr Φout Wtile δs δc δm δr δf δl δz hC hε hcw hW hout hz hs hc hm hr hf hl
  refine holds_mono ?_
    (holds_and (holds_and (holds_and (holds_and (holds_and (holds_and hs hc) hm) hr) hf) hl) hz)
  rintro ω - ⟨⟨⟨⟨⟨⟨h1, h2⟩, h3⟩, h4⟩, h5⟩, ⟨h6i, h6o⟩⟩, h8⟩
  obtain ⟨hk, hg⟩ := charges_le hC hcw h3 h4 h5
  have hC0 : (0 : ℝ) < N.declared a.T := by exact_mod_cast hC
  have hb : (0 : ℝ) ≤ a.b := Nat.cast_nonneg _
  have f7i : (r.inBits ω : ℝ) ≤ N.linkIn * a.T := by exact_mod_cast h6i
  have f7o : (r.outBits ω : ℝ) ≤ N.linkOut * a.T := by exact_mod_cast h6o
  have f9 := hout ω
  have hΦT : 0 ≤ Φout * a.T := (Nat.cast_nonneg (α := ℝ) (r.workOut ω)).trans f9
  have hO : 0 ≤ a.b * Φout * a.T / Wtile := by
    rw [mul_assoc]; exact div_nonneg (mul_nonneg hb hΦT) hW.le
  have hA : ((r.outBits ω : ℝ) + r.zOut ω) * a.b / a.zb ≤
      N.linkOut * a.b / a.zb * a.T + a.b * Φout * a.T / Wtile := by
    rcases Nat.eq_zero_or_pos a.zb with h0 | hzb
    · simp [h0]; exact hO
    · have hzb' : (0 : ℝ) < a.zb := by exact_mod_cast hzb
      have hz' : (r.zOut ω : ℝ) ≤ a.zb * Φout * a.T / Wtile := by
        rw [le_div_iff₀ hW]
        have : (a.zb : ℝ) * r.workOut ω ≤ a.zb * (Φout * a.T) := mul_le_mul_of_nonneg_left f9 hzb'.le
        linarith
      have k1 : (r.outBits ω : ℝ) * a.b / a.zb ≤ N.linkOut * a.b / a.zb * a.T := by
        rw [div_mul_eq_mul_div, div_le_div_iff_of_pos_right hzb']
        have := mul_le_mul_of_nonneg_right f7o hb
        linarith
      have k2 : (r.zOut ω : ℝ) * a.b / a.zb ≤ a.b * Φout * a.T / Wtile := by
        have e2 : (a.zb * Φout * a.T / Wtile : ℝ) * a.b / a.zb = a.b * Φout * a.T / Wtile := by field_simp
        rw [← e2]
        exact div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_right hz' hb) hzb'.le
      have e1 : ((r.outBits ω : ℝ) + r.zOut ω) * a.b / a.zb =
          r.outBits ω * a.b / a.zb + r.zOut ω * a.b / a.zb := by ring
      rw [e1]; linarith
  have eL : muLink N a * N.declared a.T = N.linkIn * a.T + N.linkOut * a.b / a.zb * a.T := by
    unfold muLink; field_simp
  have eO : muOut N a Φout Wtile * N.declared a.T = a.b * Φout * a.T / Wtile := by unfold muOut; field_simp
  have gL : 0 ≤ muLink N a := by unfold muLink; positivity
  have gO : 0 ≤ muOut N a Φout Wtile := by
    unfold muOut; rw [mul_assoc]; exact div_nonneg (mul_nonneg hb hΦT) (by positivity)
  unfold Run.paidNode at h2
  have hCt : (N.declared a.T : ℝ) ≤ N.capacity a.T := by exact_mod_cast declared_le N a.T
  have eH := muHost_mul hC
  have hg' : 0 ≤ ε + (muChip N a + muRebuild N a Wr + muLambda N a) + muLink N a + muOut N a Φout Wtile := by
    linarith
  have := mul_le_mul_of_nonneg_left hCt hg'
  unfold gammaLinks gamma
  nlinarith

end Accounting.SecurityProofs.Remote
