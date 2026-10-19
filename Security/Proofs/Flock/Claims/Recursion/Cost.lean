import Definitions.Claims.Recursion
import Proofs.Flock.Claims.Recursion.Compose

/-!
# The recursion's cost, from the outer extractor's

The recursion's extractor `E*` (`Claims.Recursion.E`) is the served `E^rep` of `Compose` (`served_eq`), so its cost
claim (`Claims.Recursion.Efficient`) is the composed extractor's at the outer extractor `E_O`'s own reruns:
`(1 + m′_I·K_I)·(1 + Σ_s m′_s·K_s)` in the worst case and `N_I·(1 + Σ_s m′_s·k_s)` in expectation, for `E_O` within
`1 + Σ_s m′_s·K_s` and `1 + Σ_s m′_s·k_s` (`efficient_of_EO`). `E_O`'s worst case holds on every coin (`EO_runs_le`),
so the worst-case half of `Efficient` is proved (`efficient_le`); `CostExp` proves `E_O`'s expectation, and so
`Efficient` (`efficient`).
-/

namespace Claims.Recursion

open Claims

theorem serve_eq {R₁ R₂ α : Type} (q : R₁ → RP R₂ Seen) : ∀ p : RP R₁ α, serve q p = RP.serve q p
  | .ret _ => rfl
  | .hash m k => by
      simp only [serve, RP.serve]
      exact congrArg (RP.hash m) (funext fun a => serve_eq q (k a))
  | .rerun r k => by
      simp only [serve, RP.serve]
      exact congrArg (RP.bind (q r)) (funext fun v => serve_eq q (k v))

/-- **`E^rep` served by `E₂` is the composed extractor of `Compose`.** -/
theorem served_eq {Coin W W₂ Op : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (V : Verifier Coin)
    (E₂ : Extractor Coin W₂ V) (dec : (x : Msg) → (Fin (V.k x) → Coin) → Option W₂ → Seen) (m k K : Msg → ℕ)
    (opens : Seen → List Op) (assemble : Msg → List Op → Option W) (check : Msg → W → Bool) :
    served V E₂ dec m k K opens assemble check = servedERep V E₂ dec m k K opens assemble check := by
  unfold served servedERep
  congr
  funext x r₀ c
  exact serve_eq _ _

/-- **The recursion's cost, from `E_O`'s**: `E_O` within `1 + Σ_s m′_s·K_s` reruns on every coin and
`1 + Σ_s m′_s·k_s` in expectation gives `Efficient`. -/
theorem efficient_of_EO {Coin W Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
    (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T)
    (hle : ExtractRunsLe H (EO D I DO decO) fun x =>
      1 + ∑ s : Fin (D.nS x), (D.nums x s).waits * (D.nums x s).cap)
    (hexp : ExtractRunsExpLe H (EO D I DO decO) fun x =>
      1 + ∑ s : Fin (D.nS x), (((D.nums x s).waits * (D.nums x s).k : ℕ) : ℝ)) :
    Efficient H D I DO decO := by
  unfold Efficient E
  rw [served_eq]
  refine ⟨servedERep_runs_le H _ _ _ _ _ _ _ _ _ hle, fun x Ω P ω => ?_⟩
  refine (servedERep_runs_exp H _ _ _ _ _ _ _ _ _ hexp x Ω P ω).trans (le_of_eq ?_)
  simp only [NI]
  push_cast
  ring

/-! ## `E_O`'s worst case -/

/-- `collect` reruns at most as often as its programs together. -/
theorem collect_runs_le {R α : Type} (H : Msg → Msg) (run : R → Trace) :
    ∀ ps : List (RP R (Option α)), ((collect ps).eval H run).2.2 ≤ (ps.map fun p => (p.eval H run).2.2).sum
  | [] => by simp [collect, RP.eval]
  | p :: ps => by
      simp only [collect, List.map_cons, List.sum_cons]
      rw [(RP.eval_bind H run p _).2]
      refine add_le_add le_rfl ?_
      rcases (p.eval H run).1 with _ | a
      · simp [RP.eval]
      · simp only
        rw [(RP.eval_bind H run _ _).2]
        simp only [RP.eval, add_zero]
        exact collect_runs_le H run ps

theorem viewing_runs {R : Type} (H : Msg → Msg) (run : R → Trace) (f : Seen → Seen) (r : R) :
    ((viewing f r).eval H run).2.2 = 1 := by
  simp [viewing, RP.eval]

/-- **`E_O`'s worst case**: the judged run, then at most `m′_s·K_s` reruns for each V* session `s`. -/
theorem EO_runs_le {Coin W Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
    (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) :
    ExtractRunsLe H (EO D I DO decO) fun x => 1 + ∑ s : Fin (D.nS x), (D.nums x s).waits * (D.nums x s).cap := by
  intro x Ω P ω r₀ c
  simp only [Extractor.eval, EO, RP.eval]
  split_ifs
  · rw [add_comm 1]
    refine Nat.add_le_add_right ((collect_runs_le H _ _).trans ?_) 1
    rw [List.map_map, Fin.sum_univ_def]
    refine List.sum_le_sum fun s _ => ?_
    simp only [Function.comp, serve_eq]
    refine (RP.serve_runs_le H _ _ _ ((firstWait_runs_le H _ _ _ _ _ _).trans (le_of_eq ?_))
      fun r => (viewing_runs H _ _ r).le).trans (le_of_eq (mul_one _))
    simp only [List.map_map, Function.comp_def, List.length_map, List.length_finRange, List.map_const',
      List.sum_replicate, smul_eq_mul]
  · simp [RP.eval]

/-- **The recursion's worst-case cost, proved**: `E*` reruns the prover at most
`(1 + m′_I·K_I)·(1 + Σ_s m′_s·K_s)` times, the first half of `Efficient`. -/
theorem efficient_le {Coin W Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
    (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) :
    ExtractRunsLe H (E H D I DO decO) fun x => (1 + I.waits (xI D x) * I.cap (xI D x)) *
      (1 + ∑ s : Fin (D.nS x), (D.nums x s).waits * (D.nums x s).cap) := by
  unfold E
  rw [served_eq]
  exact servedERep_runs_le H _ _ _ _ _ _ _ _ _ (EO_runs_le H D I DO decO)

end Claims.Recursion
