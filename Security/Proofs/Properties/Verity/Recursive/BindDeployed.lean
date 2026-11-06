module

public import Proofs.Properties.Verity.Recursive
public import Proofs.Flock.Claims.Recursion.KappaEpsDeployed

/-!
# The binding term at the deployment's sizes

`Dep.delta`'s third term is the binding term `√(min(W, 2n)·E[bindBudget]/2^256)`, with
`bindBudget = (2 + r₁ + r₂)·t' + r₁ + r₂ + dColl` and `r = Dep.runs x = N_I·(1 + Σ_s waits_s·k_s)` the extractor's
expected reruns at a draw's statement. This module evaluates `r`, and with it the term, at the deployment's sizes
(`Dep.Sized`: `RecDeploy.SizesAt` at the brings the recursive record stages from). There the inner session and every
part wait `210` times, each wait reruns `k = ⌈4·N₀/e⌉` times with `2^10 ≤ N₀ ≤ 2^23` (one table), so
`1507 ≤ k ≤ 12343986` (`Dep.k_mem`), and at most `16` parts are staged:

* `316471 ≤ r ≤ R = 2592237061 · 41475792961` (`Dep.runs_le`, `Dep.runs_ge`), and `316471² ≤ r` where a part is
  staged: `N_I = 1 + 210·k_I` and `1 + Σ_s 210·k_s` are each between `316471` and their caps `2592237061` and
  `41475792961`. So `2 + 2r` is between `2^{37.54}` and `2^{67.54}`.
* At a session each of whose draws names a statement at those sizes that stages a part, `E[bindBudget]` is between
  `(2 + 2·316471²)·t' + 2·316471² + dColl` and `(2 + 2R)·t' + 2R + dColl` (`Dep.ex_bindBudget_mem`), and `Dep.delta` is
  between the error with each in place of `E[bindBudget]` (`Dep.delta_deployed`). The term still reads the registered
  circuit's wire bits `W` and units `n` and the strategy's calls `t'`, which are the deployment's and the strategy's,
  not fixed here.
-/

@[expose] public section

namespace Verity.Recursive

open Claims (Msg Out)
open FlockDeploy (Coin)
open RecDeploy
open Accounting (Device)

namespace Dep

variable (q : Dep)

/-- **The deployment's sizes at `x`**: `RecDeploy.SizesAt` at the brings `recordAt` stages from,
`{ deployedB q.b with tr := recTr … }`. -/
def Sized (x : Msg) : Prop :=
  RecDeploy.SizesAt { deployedB q.b with tr := recTr q.aux q.nOf q.hd q.recRest } x

/-- **The reruns per wait at the deployment's sizes**: the inner session's and each part's `k = ⌈4·N₀/e⌉` is between
`1507` and `12343986`, from one table's `2^10 ≤ N₀ ≤ 2^23`. -/
theorem k_mem {x : Msg} (h : q.Sized x) :
    (1507 ≤ q.I.k (Claims.Recursion.xI q.D x) ∧ q.I.k (Claims.Recursion.xI q.D x) ≤ 12343986) ∧
      ∀ j : Fin (q.D.nS x), 1507 ≤ (q.D.nums x j).k ∧ (q.D.nums x j).k ≤ 12343986 := by
  obtain ⟨s, spec, sts, hin, h1⟩ := h.1
  have hO := h.2.1
  refine ⟨?_, fun j => ?_⟩
  · obtain ⟨hlo, hhi⟩ := FlockDeploy.N₀_le hin.setup hin.size_pos fun st hst =>
      have := hin.tab st hst
      FlockDeploy.tableLen_plain_bounds st this.1 this.2.1
    rw [h1, Nat.cast_one, one_mul] at hhi
    have hk₀ := Claims.Flock.Numbers.k_ge hlo
    have hk₁ := k_le_one hhi
    exact ⟨hk₀, hk₁⟩
  · obtain ⟨i, spec', sts', hp, hp1⟩ := hO j j.isLt
    obtain ⟨hlo, hhi⟩ := N₀O_le hp
    rw [hp1, Nat.cast_one, one_mul] at hhi
    have hk₀ := Claims.Flock.Numbers.k_ge hlo
    have hk₁ := k_le_one hhi
    exact ⟨hk₀, hk₁⟩

/-- **At most `16` parts at the deployment's sizes** (`RecDeploy.SizesAt.nS_le`, the verifier's cap). -/
theorem nS_le_of_sized {x : Msg} (h : q.Sized x) : q.D.nS x ≤ 16 := by
  first
  | exact RecDeploy.SizesAt.nS_le h
  | exact h.2.2.1

/-- **The extractor's expected reruns at the deployment's sizes, from above**: `N_I = 1 + 210·k_I ≤ 2592237061` and
`1 + Σ_s 210·k_s ≤ 41475792961` over at most `16` parts. -/
theorem runs_le {x : Msg} (h : q.Sized x) : q.runs x ≤ 2592237061 * 41475792961 := by
  obtain ⟨⟨-, hI⟩, hP⟩ := q.k_mem h
  have hn := q.nS_le_of_sized h
  have hNI : Claims.Recursion.NI q.D q.I x ≤ 2592237061 := by
    show 1 + 210 * q.I.k (Claims.Recursion.xI q.D x) ≤ 2592237061
    omega
  have hsum : ∑ j : Fin (q.D.nS x), (q.D.nums x j).waits * (q.D.nums x j).k ≤ q.D.nS x * (210 * 12343986) :=
    sum_fin_le_nat _ fun j => by
      show 210 * (q.D.nums x j).k ≤ 210 * 12343986
      exact Nat.mul_le_mul_left 210 (hP j).2
  have hn' : q.D.nS x ≤ 16 := hn
  have hS : 1 + ∑ j : Fin (q.D.nS x), (q.D.nums x j).waits * (q.D.nums x j).k ≤ 41475792961 := by
    have := Nat.mul_le_mul_right (210 * 12343986) hn'
    omega
  unfold Dep.runs
  exact Nat.mul_le_mul hNI hS

/-- **The extractor's expected reruns at the deployment's sizes, from below**: `N_I = 1 + 210·k_I ≥ 316471`, and where
`x` stages a part, `1 + Σ_s 210·k_s ≥ 316471` too. -/
theorem runs_ge {x : Msg} (h : q.Sized x) : 316471 ≤ q.runs x ∧ (0 < q.D.nS x → 316471 ^ 2 ≤ q.runs x) := by
  obtain ⟨⟨hI, -⟩, hP⟩ := q.k_mem h
  have hNI : 316471 ≤ Claims.Recursion.NI q.D q.I x := by
    show 316471 ≤ 1 + 210 * q.I.k (Claims.Recursion.xI q.D x)
    omega
  have hj : ∀ j : Fin (q.D.nS x),
      210 * 1507 ≤ ∑ s : Fin (q.D.nS x), (q.D.nums x s).waits * (q.D.nums x s).k := fun j => by
    have h0 : 210 * 1507 ≤ (q.D.nums x j).waits * (q.D.nums x j).k := by
      show 210 * 1507 ≤ 210 * (q.D.nums x j).k
      exact Nat.mul_le_mul_left 210 (hP j).1
    exact h0.trans (Finset.single_le_sum (f := fun s : Fin (q.D.nS x) => (q.D.nums x s).waits * (q.D.nums x s).k)
      (fun _ _ => Nat.zero_le _) (Finset.mem_univ j))
  unfold Dep.runs
  refine ⟨Nat.le_trans hNI (Nat.le_mul_of_pos_right _ (by omega)), fun hn => ?_⟩
  have h1 : 316471 ≤ 1 + ∑ s : Fin (q.D.nS x), (q.D.nums x s).waits * (q.D.nums x s).k := by
    have := hj ⟨0, hn⟩
    omega
  calc 316471 ^ 2 = 316471 * 316471 := Nat.pow_two _
    _ ≤ _ := Nat.mul_le_mul hNI h1

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α] (V : Vocab d α) {k : Verity.Registration.Composite}
  {θ c : ByteArray} {F : Array ByteArray} {σ : Game.Strategy (V.game θ F c)}

/-- **The binding budget's average between two rerun counts**: where every draw's statement has between `r₀` and `r₁`
expected reruns, `E[bindBudget]` is between `(2 + 2r₀)·t' + 2r₀ + dColl` and `(2 + 2r₁)·t' + 2r₁ + dColl`. -/
theorem ex_bindBudget_mem (S : SessionIn V q k θ F c σ) (dColl r₀ r₁ : ℕ)
    (h : ∀ ξ, r₀ ≤ q.runs (S.x ξ) ∧ q.runs (S.x ξ) ≤ r₁) :
    (((2 + 2 * r₀) * q.tCost V S + 2 * r₀ + dColl : ℕ) : ℝ) ≤
        Claims.Ex (fun p : S.Ξ × S.Ξ => (q.bindBudget V S dColl p.1 p.2 : ℝ)) ∧
      Claims.Ex (fun p : S.Ξ × S.Ξ => (q.bindBudget V S dColl p.1 p.2 : ℝ)) ≤
        (((2 + 2 * r₁) * q.tCost V S + 2 * r₁ + dColl : ℕ) : ℝ) := by
  constructor
  · rw [← Claims.Ex_const (α := S.Ξ × S.Ξ) (((2 + 2 * r₀) * q.tCost V S + 2 * r₀ + dColl : ℕ) : ℝ)]
    refine Claims.Ex_mono fun p => Nat.cast_le.2 ?_
    obtain ⟨h₁, -⟩ := h p.1
    obtain ⟨h₂, -⟩ := h p.2
    have hm : (2 + 2 * r₀) * q.tCost V S ≤ (2 + q.runs (S.x p.1) + q.runs (S.x p.2)) * q.tCost V S :=
      Nat.mul_le_mul_right _ (by omega)
    unfold bindBudget
    omega
  · rw [← Claims.Ex_const (α := S.Ξ × S.Ξ) (((2 + 2 * r₁) * q.tCost V S + 2 * r₁ + dColl : ℕ) : ℝ)]
    refine Claims.Ex_mono fun p => Nat.cast_le.2 ?_
    obtain ⟨-, h₁⟩ := h p.1
    obtain ⟨-, h₂⟩ := h p.2
    have hm : (2 + q.runs (S.x p.1) + q.runs (S.x p.2)) * q.tCost V S ≤ (2 + 2 * r₁) * q.tCost V S :=
      Nat.mul_le_mul_right _ (by omega)
    unfold bindBudget
    omega

/-- **`Dep.delta` at the deployment's sizes**: at a session each of whose draws names a statement at the deployment's
sizes that stages a part, the error's number is between `δs + t'·2^{-λ} + √(min(W, 2n)·B₀/2^256)` and
`δs + t'·2^{-λ} + √(min(W, 2n)·B₁/2^256)`, with `B₀ = (2 + 2·316471²)·t' + 2·316471² + dColl` and
`B₁ = (2 + 2R)·t' + 2R + dColl` at `R = 2592237061 · 41475792961`: `2 + 2·316471²` is about `2^{37.54}` and `2 + 2R`
about `2^{67.54}`. -/
theorem delta_deployed (S : SessionIn V q k θ F c σ) (dColl : ℕ) (δs lam : ℝ) (hS : ∀ ξ, q.Sized (S.x ξ))
    (hparts : ∀ ξ, 0 < q.D.nS (S.x ξ)) :
    ENNReal.ofReal (δs + q.tCost V S * (2 : ℝ) ^ (-lam) +
        Real.sqrt (min ((∑ g, (V.circuitOf θ F).w g : ℕ) : ℝ) (2 * V.units θ F) *
          ((((2 + 2 * 316471 ^ 2) * q.tCost V S + 2 * 316471 ^ 2 + dColl : ℕ) : ℝ) / 2 ^ (256 : ℝ)))) ≤
        q.delta V S dColl δs lam ∧
      q.delta V S dColl δs lam ≤
        ENNReal.ofReal (δs + q.tCost V S * (2 : ℝ) ^ (-lam) +
          Real.sqrt (min ((∑ g, (V.circuitOf θ F).w g : ℕ) : ℝ) (2 * V.units θ F) *
            ((((2 + 2 * (2592237061 * 41475792961)) * q.tCost V S + 2 * (2592237061 * 41475792961) +
              dColl : ℕ) : ℝ) / 2 ^ (256 : ℝ)))) := by
  obtain ⟨h₀, h₁⟩ := q.ex_bindBudget_mem V S dColl (316471 ^ 2) (2592237061 * 41475792961)
    fun ξ => ⟨(q.runs_ge (hS ξ)).2 (hparts ξ), q.runs_le (hS ξ)⟩
  have h2 : (0 : ℝ) ≤ 2 ^ (256 : ℝ) := by positivity
  have hm : (0 : ℝ) ≤ min ((∑ g, (V.circuitOf θ F).w g : ℕ) : ℝ) (2 * V.units θ F) :=
    le_min (Nat.cast_nonneg _) (by positivity)
  unfold delta
  exact ⟨ENNReal.ofReal_le_ofReal (add_le_add le_rfl (Real.sqrt_le_sqrt
      (mul_le_mul_of_nonneg_left (div_le_div_of_nonneg_right h₀ h2) hm))),
    ENNReal.ofReal_le_ofReal (add_le_add le_rfl (Real.sqrt_le_sqrt
      (mul_le_mul_of_nonneg_left (div_le_div_of_nonneg_right h₁ h2) hm)))⟩

end Dep

end Verity.Recursive
