import Proofs.Flock.Claims.Recursion.BoundBridge
import Proofs.Flock.Claims.Recursion.DeployMargin
import Proofs.Flock.Claims.NumbersDeploy

/-!
# (P3) at the deployed parameters

`Bound.WeightCovers`, `√C_DE + N_I·√(k_I/32) ≤ √C_IVStar`, is a floor on `γ`'s weight at `IVStar`. Where `DE` takes
rounds it holds exactly where that weight is at least `coverWeight = (√C_DE + N_I·√(k_I/32))²`
(`Bound.weightCovers_iff`), and there `γ` at `IVStar` is at least `(√γ_DE + N_I·k_I)²` (`Bound.sq_le_γ_of_weightCovers`),
with equality at the floor (`Bound.sqrt_γ_cover`). It never holds where `IVStar` keeps a weight no larger than `DE`'s
(`Bound.not_weightCovers_of_le`): it is short there by `N_I·√(k_I/32) > 0` in `√C`.

At the deployed bound record (`RecDeploy.bdDeployed`) `γ`'s weight at `IVStar` (`w`) is the code side's, and the
deployed plain verifier takes rounds at every statement (`RecDeploy.V_k_pos_flock`), so:

* (P3) holds exactly where `w ≥ coverW wE` at every statement, `coverW wE = (√wE + (1 + 210·k)·√(k_I/32))²` at `DE`'s
  weight `wE`, `k = ⌈4·N₀/e⌉` at the deployed plain record's `N₀` and `k_I` its verifier's rounds
  (`RecDeploy.weightCovers_deployed_iff`), at `w = coverW wE` in particular (`RecDeploy.weightCovers_coverW`);
* it fails wherever `w ≤ wE` at some statement, at `w = wE` in particular (`RecDeploy.not_weightCovers_of_le`,
  `RecDeploy.not_weightCovers_self`): one weight for both records, the deployed plain record's, does not meet it;
* at the deployed plain weight and a plain session of `n` tables in scope, `coverW` is at most `n³·2^72`
  (`RecDeploy.coverW_plain_le`).

At one table of `N₀ = 2^23` (`k = 12343986`, `N_I = 1 + 210·k ≈ 2^31.27`) and at the bounds `k_I = 4402`,
`C_DE = 2^33`, `coverW` is about `2^69.65` against `C_DE`'s `2^33`, and `γ` at `IVStar` at the floor about `2^86.75`
against `γ_DE`'s `2^50.11`.
-/

namespace Claims.Recursion

open Claims

/-- `√(16·(√(2kC) + k)²) = 4·(√(2k)·√C + k)`, the form of Flock's `√γ`. -/
theorem sqrt_gamma_form {k : ℝ} (hk : 0 ≤ k) (C : ℝ) :
    Real.sqrt (16 * (Real.sqrt (2 * k * C) + k) ^ 2) = 4 * (Real.sqrt (2 * k) * Real.sqrt C + k) := by
  have h2k : (0 : ℝ) ≤ 2 * k := by linarith
  have hX : 0 ≤ 4 * (Real.sqrt (2 * k * C) + k) := mul_nonneg (by norm_num) (add_nonneg (Real.sqrt_nonneg _) hk)
  rw [show 16 * (Real.sqrt (2 * k * C) + k) ^ 2 = (4 * (Real.sqrt (2 * k * C) + k)) ^ 2 by ring,
    Real.sqrt_sq hX, Real.sqrt_mul h2k C]

/-- `4·√(2k)·√(k/32) = k`, so `N·√(k/32)` more in `√C` is `N·k` more in `√γ`. -/
theorem sqrt_gamma_unit {k : ℝ} (hk : 0 ≤ k) : 4 * (Real.sqrt (2 * k) * Real.sqrt (k / 32)) = k := by
  have h2k : (0 : ℝ) ≤ 2 * k := by linarith
  have hk4 : (0 : ℝ) ≤ k / 4 := by linarith
  rw [← Real.sqrt_mul h2k, show 2 * k * (k / 32) = (k / 4) ^ 2 by ring, Real.sqrt_sq hk4]
  ring

section Bound

variable {Coin Col T : Type} [Inhabited Coin]

omit [Inhabited Coin] in
/-- **(P3)'s floor on `γ`'s weight at `IVStar`**: `(√C_DE + N_I·√(k_I/32))²`. -/
noncomputable def Bound.coverWeight (Bd : Bound Coin Col T) (x : Msg) : ℝ :=
  (Real.sqrt (Bd.DE.weight x) + (1 + (Bd.DE.waits x : ℝ) * Bd.DE.k x) * Real.sqrt (Bd.DE.V.k x / 32)) ^ 2

omit [Inhabited Coin] in
/-- **A weight at or above the floor meets (P3).** -/
theorem Bound.weightCovers_of_le (Bd : Bound Coin Col T) (h : ∀ x, Bd.coverWeight x ≤ Bd.weight x) :
    Bd.WeightCovers :=
  fun x => Real.le_sqrt_of_sq_le (h x)

omit [Inhabited Coin] in
/-- **(P3) puts the weight at or above the floor** at a statement where `DE` takes rounds. -/
theorem Bound.le_of_weightCovers (Bd : Bound Coin Col T) (h : Bd.WeightCovers) (x : Msg) (hk : 0 < Bd.DE.V.k x) :
    Bd.coverWeight x ≤ Bd.weight x := by
  have h3 : 0 < (1 + (Bd.DE.waits x : ℝ) * Bd.DE.k x) * Real.sqrt (Bd.DE.V.k x / 32) :=
    mul_pos (by positivity) (Real.sqrt_pos.2 (div_pos (by exact_mod_cast hk) (by norm_num)))
  exact (Real.le_sqrt' (add_pos_of_nonneg_of_pos (Real.sqrt_nonneg _) h3)).1 (h x)

omit [Inhabited Coin] in
/-- **(P3) is the floor**, where `DE` takes rounds at every statement. -/
theorem Bound.weightCovers_iff (Bd : Bound Coin Col T) (hk : ∀ x, 0 < Bd.DE.V.k x) :
    Bd.WeightCovers ↔ ∀ x, Bd.coverWeight x ≤ Bd.weight x :=
  ⟨fun h x => Bd.le_of_weightCovers h x (hk x), Bd.weightCovers_of_le⟩

omit [Inhabited Coin] in
/-- **(P3) fails where `IVStar`'s weight is no larger than `DE`'s**, at a statement where `DE` takes rounds: it would
need `N_I·√(k_I/32) ≤ 0`. -/
theorem Bound.not_weightCovers_of_le (Bd : Bound Coin Col T) (x : Msg) (hk : 0 < Bd.DE.V.k x)
    (hw : Bd.weight x ≤ Bd.DE.weight x) : ¬ Bd.WeightCovers := fun h => by
  have h1 := h x
  have h2 := Real.sqrt_le_sqrt hw
  have h3 : 0 < (1 + (Bd.DE.waits x : ℝ) * Bd.DE.k x) * Real.sqrt (Bd.DE.V.k x / 32) :=
    mul_pos (by positivity) (Real.sqrt_pos.2 (div_pos (by exact_mod_cast hk) (by norm_num)))
  linarith

/-- **`γ` at `IVStar` where (P3) holds**: at least `(√γ_DE + N_I·k_I)²`. -/
theorem Bound.sq_le_γ_of_weightCovers (Bd : Bound Coin Col T) (h : Bd.WeightCovers) (x : Msg) :
    (Real.sqrt (Claims.Flock.γ Bd.DE x) + (1 + (Bd.DE.waits x : ℝ) * Bd.DE.k x) * Bd.DE.V.k x) ^ 2 ≤
      Claims.Flock.γ Bd.deploy x := by
  have hW := h x
  have e1 : Claims.Flock.γ Bd.deploy x =
      16 * (Real.sqrt (2 * (Bd.DE.V.k x : ℝ) * Bd.weight x) + Bd.DE.V.k x) ^ 2 := rfl
  have e2 : Claims.Flock.γ Bd.DE x =
      16 * (Real.sqrt (2 * (Bd.DE.V.k x : ℝ) * Bd.DE.weight x) + Bd.DE.V.k x) ^ 2 := rfl
  have hk : (0 : ℝ) ≤ Bd.DE.V.k x := Nat.cast_nonneg _
  have hN : (0 : ℝ) ≤ 1 + (Bd.DE.waits x : ℝ) * Bd.DE.k x := by positivity
  rw [e1, e2]
  generalize (Bd.DE.V.k x : ℝ) = k at hW hk ⊢
  generalize (1 + (Bd.DE.waits x : ℝ) * Bd.DE.k x) = N at hW hN ⊢
  generalize Bd.DE.weight x = C at hW ⊢
  generalize Bd.weight x = W at hW ⊢
  have hγ : 0 ≤ 16 * (Real.sqrt (2 * k * W) + k) ^ 2 := by positivity
  rw [← Real.sq_sqrt hγ, sqrt_gamma_form hk W, sqrt_gamma_form hk C]
  have h1 := mul_le_mul_of_nonneg_left hW (mul_nonneg (by norm_num : (0 : ℝ) ≤ 4) (Real.sqrt_nonneg (2 * k)))
  have h2 : N * (4 * (Real.sqrt (2 * k) * Real.sqrt (k / 32))) = N * k := by rw [sqrt_gamma_unit hk]
  have hle : 4 * (Real.sqrt (2 * k) * Real.sqrt C + k) + N * k ≤ 4 * (Real.sqrt (2 * k) * Real.sqrt W + k) := by
    linarith
  exact pow_le_pow_left₀ (add_nonneg (mul_nonneg (by norm_num)
    (add_nonneg (mul_nonneg (Real.sqrt_nonneg _) (Real.sqrt_nonneg _)) hk)) (mul_nonneg hN hk)) hle 2

/-- **At the floor `√γ` grows by exactly `N_I·k_I`**: `√γ_IVStar = √γ_DE + N_I·k_I`. -/
theorem Bound.sqrt_γ_cover (Bd : Bound Coin Col T) (x : Msg) (hw : Bd.weight x = Bd.coverWeight x) :
    Real.sqrt (Claims.Flock.γ Bd.deploy x) =
      Real.sqrt (Claims.Flock.γ Bd.DE x) + (1 + (Bd.DE.waits x : ℝ) * Bd.DE.k x) * Bd.DE.V.k x := by
  have hc : Real.sqrt (Bd.weight x) = Real.sqrt (Bd.DE.weight x) +
      (1 + (Bd.DE.waits x : ℝ) * Bd.DE.k x) * Real.sqrt (Bd.DE.V.k x / 32) := by
    rw [hw, Bound.coverWeight, Real.sqrt_sq (by positivity)]
  have e1 : Claims.Flock.γ Bd.deploy x =
      16 * (Real.sqrt (2 * (Bd.DE.V.k x : ℝ) * Bd.weight x) + Bd.DE.V.k x) ^ 2 := rfl
  have e2 : Claims.Flock.γ Bd.DE x =
      16 * (Real.sqrt (2 * (Bd.DE.V.k x : ℝ) * Bd.DE.weight x) + Bd.DE.V.k x) ^ 2 := rfl
  have hk : (0 : ℝ) ≤ Bd.DE.V.k x := Nat.cast_nonneg _
  rw [e1, e2]
  generalize (Bd.DE.V.k x : ℝ) = k at hc hk ⊢
  generalize (1 + (Bd.DE.waits x : ℝ) * Bd.DE.k x) = N at hc ⊢
  generalize Bd.DE.weight x = C at hc ⊢
  generalize Bd.weight x = W at hc ⊢
  rw [sqrt_gamma_form hk W, sqrt_gamma_form hk C, hc]
  linear_combination N * sqrt_gamma_unit hk

end Bound

end Claims.Recursion

namespace RecDeploy

open Claims (Msg Out)
open FlockDeploy (Coin)

/-- **(P3)'s floor at the deployment**, at `DE`'s weight `wE`: `(√wE + (1 + 210·k)·√(k_I/32))²`, with `k = ⌈4·N₀/e⌉` at
the deployed plain record's `N₀` and `k_I` the deployed plain verifier's rounds. -/
noncomputable def coverW (wE : Msg → ℝ) (x : Msg) : ℝ :=
  (Real.sqrt (wE x) + (1 + 210 * (⌈4 * FlockDeploy.N₀ false x / Real.exp 1⌉₊ : ℝ)) *
    Real.sqrt ((FlockDeploy.V false).k x / 32)) ^ 2

/-- **The deployed bound record's floor is `coverW`** at its `DE`'s weight, whatever else the code side picks. -/
theorem coverWeight_bdDeployed (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) :
    (bdDeployed rel nextE colsE wE join rowIn final w).coverWeight = coverW wE := by
  funext x
  show (Real.sqrt (wE x) + (1 + ((Claims.Flock.Numbers.waits : ℕ) : ℝ) *
    (⌈4 * FlockDeploy.N₀ false x / Real.exp 1⌉₊ : ℝ)) * Real.sqrt ((FlockDeploy.V false).k x / 32)) ^ 2 = _
  rw [show ((Claims.Flock.Numbers.waits : ℕ) : ℝ) = 210 by norm_num [Claims.Flock.Numbers.waits]]
  rfl

/-- **(P3) at the deployment is the floor**: the deployed bound record meets it exactly where `γ`'s weight at `IVStar`
is at least `coverW` at `DE`'s weight, at every statement. -/
theorem weightCovers_deployed_iff (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) :
    (bdDeployed rel nextE colsE wE join rowIn final w).WeightCovers ↔ ∀ x, coverW wE x ≤ w x := by
  have e := (bdDeployed rel nextE colsE wE join rowIn final w).weightCovers_iff fun x => V_k_pos_flock false x
  rw [coverWeight_bdDeployed] at e
  exact e

/-- **(P3) holds at the floor** at the deployment. -/
theorem weightCovers_coverW (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) :
    (bdDeployed rel nextE colsE wE join rowIn final (coverW wE)).WeightCovers :=
  (weightCovers_deployed_iff rel nextE colsE wE join rowIn final _).2 fun _ => le_rfl

/-- **(P3) fails at the deployment wherever `IVStar`'s weight is at most `DE`'s at some statement.** -/
theorem not_weightCovers_of_le (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) (x : Msg)
    (hw : w x ≤ wE x) : ¬ (bdDeployed rel nextE colsE wE join rowIn final w).WeightCovers :=
  (bdDeployed rel nextE colsE wE join rowIn final w).not_weightCovers_of_le x (V_k_pos_flock false x) hw

/-- **(P3) fails at one weight for both records**: `IVStar` keeping `DE`'s weight, the deployed plain record's among
them. -/
theorem not_weightCovers_self (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) :
    ¬ (bdDeployed rel nextE colsE wE join rowIn final wE).WeightCovers :=
  not_weightCovers_of_le rel nextE colsE wE join rowIn final wE [] le_rfl

/-- **The floor at the deployed plain weight**: at most `n³·2^72` at a plain session of `n` tables in scope
(`k ≤ n·2^24`, `k_I ≤ 4402·n`, `C_DE ≤ n·2^33`). -/
theorem coverW_plain_le (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) {x : Msg} {s spec sts}
    (h : FlockDeploy.PlainIn x s spec sts) :
    coverW (FlockDeploy.deployPlainTrim rel).weight x ≤ (sts.size : ℝ) ^ 3 * 2 ^ 72 := by
  have hn : (1 : ℝ) ≤ sts.size := FlockDeploy.size_one_le h.size_pos
  have hC : FlockDeploy.weight false x ≤ (sts.size : ℝ) * 2 ^ 33 := FlockDeploy.weight_plain_le rel h
  have hC0 : 0 ≤ FlockDeploy.weight false x := FlockDeploy.weight_nonneg false x
  have hV : ((FlockDeploy.V false).k x : ℝ) ≤ 2 + 4400 * (sts.size : ℝ) := by
    have := FlockDeploy.Vk_plain_le rel h
    exact_mod_cast this
  have hv0 : (0 : ℝ) ≤ (FlockDeploy.V false).k x := Nat.cast_nonneg _
  obtain ⟨hN₀, hN₁⟩ := FlockDeploy.N₀_le h.setup h.size_pos fun st hst =>
    FlockDeploy.tableLen_plain_bounds st (h.tab st hst).1 (h.tab st hst).2.1
  have hk := Claims.Flock.Numbers.k_le_mul hn (le_trans (by norm_num) hN₀) hN₁
  have hk0 : (0 : ℝ) ≤ (⌈4 * FlockDeploy.N₀ false x / Real.exp 1⌉₊ : ℝ) := Nat.cast_nonneg _
  show (Real.sqrt (FlockDeploy.weight false x) + (1 + 210 * (⌈4 * FlockDeploy.N₀ false x / Real.exp 1⌉₊ : ℝ)) *
    Real.sqrt ((FlockDeploy.V false).k x / 32)) ^ 2 ≤ (sts.size : ℝ) ^ 3 * 2 ^ 72
  generalize (⌈4 * FlockDeploy.N₀ false x / Real.exp 1⌉₊ : ℝ) = kc at hk hk0 ⊢
  generalize FlockDeploy.weight false x = C at hC hC0 ⊢
  generalize ((FlockDeploy.V false).k x : ℝ) = v at hV hv0 ⊢
  generalize (sts.size : ℝ) = n at hn hC hV hk ⊢
  have hv32 : 0 ≤ v / 32 := div_nonneg hv0 (by norm_num)
  have hN : 1 + 210 * kc ≤ n * 3523215361 := by linarith
  have hN2 : (1 + 210 * kc) ^ 2 ≤ (n * 3523215361) ^ 2 := pow_le_pow_left₀ (by linarith) hN 2
  have hprod : (1 + 210 * kc) ^ 2 * (v / 32) ≤ (n * 3523215361) ^ 2 * (n * 4402 / 32) :=
    mul_le_mul hN2 (by linarith) hv32 (by positivity)
  have hsq : (Real.sqrt C + (1 + 210 * kc) * Real.sqrt (v / 32)) ^ 2 ≤
      2 * Real.sqrt C ^ 2 + 2 * ((1 + 210 * kc) ^ 2 * Real.sqrt (v / 32) ^ 2) := by
    nlinarith [sq_nonneg (Real.sqrt C - (1 + 210 * kc) * Real.sqrt (v / 32))]
  rw [Real.sq_sqrt hC0, Real.sq_sqrt hv32] at hsq
  have hn3 : n ≤ n ^ 3 := le_self_pow₀ hn (by norm_num)
  linarith

end RecDeploy
