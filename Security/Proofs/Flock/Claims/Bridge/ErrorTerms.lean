import Definitions.Claims.Sampling

/-!
# The end-to-end error inside the claims form's (A2 of the bridge)

`Flock.Guarantees.EndToEnd` bounds `Pr[accept ∧ K₀ ≤ #wrong]` by `miss + ε_ks + δ_link` in `ℝ≥0∞`: the draw's miss
`C(n−K₀, kd)/C(n, kd)` (with `η` beside it in `EndToEndPerTest`), `ksAvgStrictZ` and the link term. `Claims.Sampling.KS`
bounds `PrBad` in `ℝ` by `κ D δ x + √(γ·Pr[B]) + Pr[B_ext]`. `e2e_error_le_ks` puts the first sum under the second at a
deployment `D` whose `δ` covers the draw's miss and whose `4·ε_table` covers the knowledge and link terms; the
square-root and `B_ext` terms are only required to be nonnegative. -/

namespace Claims.Sampling

open scoped ENNReal

theorem one_le_slack : 1 ≤ slack := by
  unfold slack
  first
    | exact Real.one_le_rpow (by norm_num) (by positivity)
    | calc (1 : ℝ) = (2 : ℝ) ^ (0 : ℝ) := (Real.rpow_zero 2).symm
        _ ≤ (2 : ℝ) ^ ((2 : ℝ) ^ (-40 : ℝ)) := Real.rpow_le_rpow_of_exponent_le (by norm_num) (by positivity)

section Deploy

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]

theorem four_εtable_le_κF (D : Deploy Coin Col T) (x : Msg) : 4 * D.εtable x ≤ κF D x := by
  unfold κF
  have h1 : (0 : ℝ) ≤ (2 : ℝ) ^ (-(D.innerWaits x : ℝ)) := by positivity
  have h2 : (0 : ℝ) ≤ 2 * (D.innerK x : ℝ) / D.innerCap x := by positivity
  have h3 : 0 ≤ Real.exp (-(D.innerK x : ℝ) / 8) := (Real.exp_pos _).le
  linarith

theorem δ_add_κF_le_κ (D : Deploy Coin Col T) (δ : ℝ) (x : Msg) (hδ : 0 ≤ δ) (hε : 0 ≤ D.εtable x) :
    δ + κF D x ≤ κ D δ x := by
  have hF : 0 ≤ κF D x := le_trans (by linarith) (four_εtable_le_κF D x)
  have hs : 1 ≤ slack := one_le_slack
  have h1 : δ + κF D x ≤ slack ^ 2 * (δ + κF D x) := le_mul_of_one_le_left (by linarith) (one_le_pow₀ hs)
  have h2 : 0 ≤ (1 + slack) * (2 * (D.k x : ℝ) / D.cap x + Real.exp (-(D.k x : ℝ) / 8)) :=
    mul_nonneg (by linarith) (add_nonneg (by positivity) (Real.exp_pos _).le)
  unfold κ
  exact le_add_of_le_of_nonneg h1 h2

theorem δ_add_εtable_le_κ (D : Deploy Coin Col T) (δ : ℝ) (x : Msg) (hδ : 0 ≤ δ) (hε : 0 ≤ D.εtable x) :
    δ + 4 * D.εtable x ≤ κ D δ x :=
  le_trans (by linarith [four_εtable_le_κF D x]) (δ_add_κF_le_κ D δ x hδ hε)

/-- **A2, the error terms**: an error `miss + ks + link` in `ℝ≥0∞` whose miss is at most `δ` and whose knowledge and
link terms are at most `4·ε_table` is at most `KS`'s error `κ D δ x + a + b`, at any nonnegative `a` (`√(γ·Pr[B])`)
and `b` (`Pr[B_ext]`). -/
theorem e2e_error_le_ks (D : Deploy Coin Col T) (δ : ℝ) (x : Msg) (hδ : 0 ≤ δ) (hε : 0 ≤ D.εtable x)
    {miss ks link : ℝ≥0∞} (hmiss : miss ≤ ENNReal.ofReal δ) (hkl : ks + link ≤ ENNReal.ofReal (4 * D.εtable x))
    {a b : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) :
    miss + ks + link ≤ ENNReal.ofReal (κ D δ x + a + b) := by
  have hκ := δ_add_εtable_le_κ D δ x hδ hε
  calc miss + ks + link = miss + (ks + link) := add_assoc _ _ _
    _ ≤ ENNReal.ofReal δ + ENNReal.ofReal (4 * D.εtable x) := add_le_add hmiss hkl
    _ = ENNReal.ofReal (δ + 4 * D.εtable x) := (ENNReal.ofReal_add hδ (by linarith)).symm
    _ ≤ ENNReal.ofReal (κ D δ x + a + b) := ENNReal.ofReal_le_ofReal (by linarith)

/-- `e2e_error_le_ks` in `ℝ`, the form `KSLinOn` bounds `PrBad` in. -/
theorem e2e_error_toReal_le_ks (D : Deploy Coin Col T) (δ : ℝ) (x : Msg) (hδ : 0 ≤ δ) (hε : 0 ≤ D.εtable x)
    {miss ks link : ℝ≥0∞} (hmiss : miss ≤ ENNReal.ofReal δ) (hkl : ks + link ≤ ENNReal.ofReal (4 * D.εtable x))
    {a b : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) :
    (miss + ks + link).toReal ≤ κ D δ x + a + b := by
  have hκ := δ_add_εtable_le_κ D δ x hδ hε
  exact ENNReal.toReal_le_of_le_ofReal (by linarith) (e2e_error_le_ks D δ x hδ hε hmiss hkl ha hb)

end Deploy

end Claims.Sampling
