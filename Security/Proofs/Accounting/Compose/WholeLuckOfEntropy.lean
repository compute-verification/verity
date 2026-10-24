import Definitions.Accounting.Guarantees.Compose

/-!
# A whole-tile luck per query from its tiles' entropy: proof

`Accounting.Guarantees.Compose.WholeLuckOfEntropy`. Per round take `WholeEntropyAt`'s `Wh`, `Lk` and `Break`, and
`g q a = Σ_{v ∈ Wh q a} w_v·((1 + κ₀)·[v ∈ Lk q a] − κ₀)`. Off `Break` at `sq a r = some q` the round's whole tiles
are `Wh q a` and `same` lands in `Lk q a`, so `roundGain ≤ g q a` (`w ≥ 0`, `1 + κ₀ ≥ 0`). `Wh q` doesn't read `q`'s
answer, so on `redraw a r q b` the sum is over one set `W`, and with `x_v = θ_r·w_v` (`moment_le`)
`exp(θ_r·g) = Π_{v ∈ W} ([v ∈ Lk]·c_v + e^{−κ₀x_v})`, `c_v = e^{x_v} − e^{−κ₀x_v} ≥ 0`, which `Finset.prod_add`
expands to `Σ_{S ⊆ W} [S ⊆ Lk]·Π_{v ∈ S} c_v·Π_{v ∈ W \ S} e^{−κ₀x_v}`. Summing over `b`, the entropy conjunct bounds
each `Σ_b [S ⊆ Lk]` by `|B|·e^{−Σ_S h}`, and the sum folds back to `|B|·Π_v (c_v·e^{−h_v} + e^{−κ₀x_v})`, each factor
`e^{−h_v}·e^{x_v} + (1 − e^{−h_v})·e^{−κ₀x_v}`, in `[0, 1]`.
-/

namespace Accounting.SecurityProofs.Compose.Whole

open Accounting.Compose Accounting.Assumptions.Compose

/-- One tile's term of the gain's exponential: `e^{x·((1 + κ)·[v ∈ L] − κ)} = [v ∈ L]·(e^x − e^{−κx}) + e^{−κx}`. -/
theorem exp_term {V : Type} [DecidableEq V] (L : Finset V) (v : V) (x κ : ℝ) :
    Real.exp (x * ((1 + κ) * (if v ∈ L then 1 else 0) - κ)) =
      (if v ∈ L then Real.exp x - Real.exp (-(κ * x)) else 0) + Real.exp (-(κ * x)) := by
  by_cases hv : v ∈ L
  · rw [ite_eq_left hv, ite_eq_left hv, show x * ((1 + κ) * 1 - κ) = x by ring]
    ring
  · rw [ite_eq_right hv, ite_eq_right hv, show x * ((1 + κ) * 0 - κ) = -(κ * x) by ring]
    ring

/-- A product over `S` of a term that is `c v` on `L` and `0` off it: `Π_S c` where `S ⊆ L`, `0` where not. -/
theorem prod_ite_mem {V : Type} [DecidableEq V] (S L : Finset V) (c : V → ℝ) :
    ∏ v ∈ S, (if v ∈ L then c v else 0) = if S ⊆ L then ∏ v ∈ S, c v else 0 := by
  by_cases hS : S ⊆ L
  · rw [ite_eq_left hS]
    exact Finset.prod_congr rfl fun v hv => ite_eq_left (hS hv)
  · rw [ite_eq_right hS]
    obtain ⟨v, hv, hvL⟩ := Finset.not_subset.mp hS
    exact Finset.prod_eq_zero hv (ite_eq_right hvL)

/-- **The gain's moment over one answer**: tiles `W` (the same for every answer `b`), each lucky where it is in
`L b`, at rates `x ≥ 0` and exchange rate `κ ≥ 0`. If every `S ⊆ W` is all lucky on at most a share `e^{−Σ_S h}` of
the answers, and each tile's factor `e^{−h}·e^x + (1 − e^{−h})·e^{−κx}` is at most `1`, the moment is at most `|B|`. -/
theorem moment_le {B V : Type} [Fintype B] [DecidableEq V] (W : Finset V) (L : B → Finset V) (x h : V → ℝ)
    (κ : ℝ) (hκ : 0 ≤ κ) (hx : ∀ v ∈ W, 0 ≤ x v)
    (hent : ∀ S ⊆ W, ((Finset.univ.filter fun b => S ⊆ L b).card : ℝ) ≤
      Fintype.card B * Real.exp (-∑ v ∈ S, h v))
    (hfac : ∀ v ∈ W, Real.exp (-h v) * Real.exp (x v) + (1 - Real.exp (-h v)) * Real.exp (-(κ * x v)) ≤ 1) :
    ∑ b, Real.exp (∑ v ∈ W, x v * ((1 + κ) * (if v ∈ L b then 1 else 0) - κ)) ≤ Fintype.card B := by
  set c : V → ℝ := fun v => Real.exp (x v) - Real.exp (-(κ * x v)) with hc
  set E : V → ℝ := fun v => Real.exp (-(κ * x v)) with hE
  have hc0 : ∀ v ∈ W, 0 ≤ c v := fun v hv => by
    simp only [hc, sub_nonneg, Real.exp_le_exp]
    nlinarith [hx v hv]
  have hE0 : ∀ v, 0 ≤ E v := fun v => (Real.exp_pos _).le
  have hexp : ∀ b, Real.exp (∑ v ∈ W, x v * ((1 + κ) * (if v ∈ L b then 1 else 0) - κ)) =
      ∑ S ∈ W.powerset, (if S ⊆ L b then ∏ v ∈ S, c v else 0) * ∏ v ∈ W \ S, E v := by
    intro b
    rw [Real.exp_sum, Finset.prod_congr rfl fun v _ => exp_term (L b) v (x v) κ, Finset.prod_add]
    exact Finset.sum_congr rfl fun S _ => by rw [prod_ite_mem]
  have hcard : ∀ S ∈ W.powerset, ∑ b, (if S ⊆ L b then ∏ v ∈ S, c v else 0) * ∏ v ∈ W \ S, E v ≤
      Fintype.card B * ((∏ v ∈ S, (c v * Real.exp (-h v))) * ∏ v ∈ W \ S, E v) := by
    intro S hS
    have hSW := Finset.mem_powerset.mp hS
    have hC : 0 ≤ (∏ v ∈ S, c v) * ∏ v ∈ W \ S, E v :=
      mul_nonneg (Finset.prod_nonneg fun v hv => hc0 v (hSW hv)) (Finset.prod_nonneg fun v _ => hE0 v)
    calc ∑ b, (if S ⊆ L b then ∏ v ∈ S, c v else 0) * ∏ v ∈ W \ S, E v
        = ((Finset.univ.filter fun b => S ⊆ L b).card : ℝ) * ((∏ v ∈ S, c v) * ∏ v ∈ W \ S, E v) := by
          rw [← Finset.sum_boole, Finset.sum_mul]
          exact Finset.sum_congr rfl fun b _ => by split_ifs <;> ring
      _ ≤ (Fintype.card B * Real.exp (-∑ v ∈ S, h v)) * ((∏ v ∈ S, c v) * ∏ v ∈ W \ S, E v) :=
          mul_le_mul_of_nonneg_right (hent S hSW) hC
      _ = Fintype.card B * ((∏ v ∈ S, (c v * Real.exp (-h v))) * ∏ v ∈ W \ S, E v) := by
          rw [← Finset.sum_neg_distrib, Real.exp_sum, Finset.prod_mul_distrib]
          ring
  calc ∑ b, Real.exp (∑ v ∈ W, x v * ((1 + κ) * (if v ∈ L b then 1 else 0) - κ))
      = ∑ S ∈ W.powerset, ∑ b, (if S ⊆ L b then ∏ v ∈ S, c v else 0) * ∏ v ∈ W \ S, E v := by
        rw [Finset.sum_congr rfl fun b _ => hexp b, Finset.sum_comm]
    _ ≤ ∑ S ∈ W.powerset, Fintype.card B * ((∏ v ∈ S, (c v * Real.exp (-h v))) * ∏ v ∈ W \ S, E v) :=
        Finset.sum_le_sum hcard
    _ = Fintype.card B * ∏ v ∈ W, (c v * Real.exp (-h v) + E v) := by
        rw [← Finset.mul_sum, Finset.prod_add]
    _ ≤ Fintype.card B * 1 := by
        refine mul_le_mul_of_nonneg_left (Finset.prod_le_one₀ (fun v hv => ?_) fun v hv => ?_) (Nat.cast_nonneg _)
        · exact add_nonneg (mul_nonneg (hc0 v hv) (Real.exp_pos _).le) (hE0 v)
        · have := hfac v hv
          simp only [hc, hE]
          linarith
    _ = Fintype.card B := mul_one _

open Classical in
theorem wholeLuckOfEntropy : Accounting.Guarantees.Compose.WholeLuckOfEntropy := by
  intro B V _ _ Tq M rd R w h κ0 θ sq δb hκ hw hθ hfac hE r
  obtain ⟨Wh, Lk, Br, hinv, hrd, hent, hB, hmem⟩ := hE r
  refine ⟨fun q a => ∑ v ∈ Wh q a, w v * ((1 + κ0) * (if v ∈ Lk q a then 1 else 0) - κ0), Br,
    fun q a => ?_, hB, fun a q hb hq => ?_⟩
  · have key := moment_le (Wh q a) (fun b => Lk q (redraw a r q b)) (fun v => θ r.val * w v) h κ0 hκ
      (fun v _ => mul_nonneg (hθ _) (hw v)) (fun S hS => hent q a S hS)
      (fun v hv => by simpa [hrd q a v hv] using hfac v)
    refine le_of_eq_of_le (Finset.sum_congr rfl fun b _ => ?_) key
    beta_reduce
    rw [hinv, Finset.mul_sum]
    exact congrArg Real.exp (Finset.sum_congr rfl fun v _ => by ring)
  · unfold roundGain
    rw [← Finset.sum_filter_add_sum_filter_not Finset.univ (· ∈ Wh q a)]
    have hin : (Finset.univ.filter (· ∈ Wh q a)) = Wh q a := by ext v; simp
    have hout : ∀ v ∈ Finset.univ.filter (· ∉ Wh q a),
        (if rd v = r.val ∧ R.mode a v = .whole then
          w v * ((1 + κ0) * (if R.same a v then 1 else 0) - κ0) else 0) = 0 := by
      intro v hv
      refine ite_eq_right fun ⟨hr, hm⟩ => ?_
      exact absurd (((hmem a q hb hq v hr).1).mp hm) (Finset.mem_filter.mp hv).2
    rw [Finset.sum_congr rfl hout, Finset.sum_const_zero, add_zero, hin]
    refine Finset.sum_le_sum fun v hv => ?_
    have hr := hrd q a v hv
    have hm := ((hmem a q hb hq v hr).1).mpr hv
    rw [ite_eq_left ⟨hr, hm⟩]
    refine mul_le_mul_of_nonneg_left ?_ (hw v)
    have h1 : (if R.same a v then (1 : ℝ) else 0) ≤ if v ∈ Lk q a then 1 else 0 := by
      by_cases hs : R.same a v
      · rw [ite_eq_left hs, ite_eq_left ((hmem a q hb hq v hr).2 hm hs)]
      · rw [ite_eq_right hs]
        split_ifs <;> norm_num
    nlinarith

end Accounting.SecurityProofs.Compose.Whole
