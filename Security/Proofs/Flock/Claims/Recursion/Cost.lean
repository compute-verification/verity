import Definitions.Claims.Recursion
import Proofs.Flock.Claims.Recursion.Compose

/-!
# The recursion's cost, from the outer extractor's

The recursion's extractor `E*` (`Claims.Recursion.E`) is the served `E^rep` of `Compose` (`served_eq`), so its cost
claim (`Claims.Recursion.Efficient`) is the composed extractor's at the outer extractor `E_O`'s own reruns:
`(1 + m′_I·K_I)·(1 + Σ_s m′_s·K_s)` in the worst case and `N_I·(1 + Σ_s m′_s·k_s)` in expectation, for `E_O` within
`1 + Σ_s m′_s·K_s` and `1 + Σ_s m′_s·k_s` (`efficient_of_EO`).
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
    (hle : ExtractRunsLe H (EO H D I DO decO) fun x =>
      1 + ∑ s : Fin (D.nS x), (D.nums x s).waits * (D.nums x s).cap)
    (hexp : ExtractRunsExpLe H (EO H D I DO decO) fun x =>
      1 + ∑ s : Fin (D.nS x), (((D.nums x s).waits * (D.nums x s).k : ℕ) : ℝ)) :
    Efficient H D I DO decO := by
  unfold Efficient E
  rw [served_eq]
  refine ⟨servedERep_runs_le H _ _ _ _ _ _ _ _ _ hle, fun x Ω P ω => ?_⟩
  refine (servedERep_runs_exp H _ _ _ _ _ _ _ _ _ hexp x Ω P ω).trans (le_of_eq ?_)
  simp only [NI]
  push_cast
  ring

end Claims.Recursion
