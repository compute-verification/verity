import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Data.Fintype.Pi

/-!
# The remote audit's opening draw (trusted)

The draw `audit.pooled` reads, as the verifier runs it. A session has `n` windows, in order. Window `t` registers its
`U_t ≤ m` receipts and their roots, which fix which of them are correct. Then the verifier releases window `t`'s openings
coin, live, and opens `s'_t = min(s_t, U_t)` distinct receipts of the `U_t`, uniform (`RemoteAudit.draws`, `Key.subset`).

* `Openings m s`: an openings coin, a uniform `min(s, U)`-subset of the first `U` receipts for each `U ≤ m`. A window reads
  the one at its own `U_t`, so its draw is uniform at whatever size the prover gave the window.
* `Coins n m s X`: a session's coins, window `t`'s other coins `X t` (the prover's, the world's, the earlier windows'
  anything) beside its openings coin.
* `Session`: each window's size `U_t` and correct receipts, read off the coins. That they are read before the window's
  openings, from the earlier windows' coins and the window's own other coins, is `Assumptions.Remote.Live`: the prover
  may adapt window `t` to every earlier window's openings.
* `pooled n U s' δ`: `audit.pooled`'s horizon reading, the most wrong receipts `Σ_t f_t·U_t` over the shares `f_t` with
  `0 ≤ f_t ≤ 1 − s'_t/U_t` and `Σ_t s'_t·f_t ≤ budget S δ = S·(1 − δ^(1/S))`, `S = Σ_t s'_t`. `audit.pooled` computes it
  greedily, the windows taken by `U_t/s'_t` falling (the fractional knapsack's optimum); that the greedy order reaches
  this supremum is not stated here.
-/

namespace Accounting.Remote.Draw

open Finset

/-- An openings coin: for each window size `U ≤ m`, a `min(s, U)`-subset of the first `U` receipts. -/
abbrev Openings (m s : ℕ) : Type := (U : Fin (m + 1)) → {S : Finset (Fin U) // S.card = min s U}

/-- A session's coins: window `t`'s other coins and its openings coin. -/
abbrev Coins (n m : ℕ) (s : Fin n → ℕ) (X : Fin n → Type) : Type := (t : Fin n) → X t × Openings m (s t)

/-- A session of `n` windows of at most `m` receipts, window `t` drawing `min(s_t, U_t)` of its `U_t`. -/
structure Session (n m : ℕ) (s : Fin n → ℕ) (X : Fin n → Type) where
  /-- Window `t`'s receipt count `U_t`. -/
  size : Fin n → Coins n m s X → Fin (m + 1)
  /-- Window `t`'s correct receipts, by position. -/
  correct : Fin n → Coins n m s X → Finset ℕ

/-- `audit.pooled`'s budget: `S·(1 − δ^(1/S))` (`-expm1(log δ / S)·S`). -/
noncomputable def budget (S : ℕ) (δ : ℝ) : ℝ := S * (1 - δ ^ ((S : ℝ)⁻¹))

/-- The pooled bound of windows of sizes `U` drawing `d`: the most `Σ_t f_t·U_t` over the shares `f` with
`0 ≤ f_t ≤ 1 − d_t/U_t` and `Σ_t d_t·f_t ≤ budget (Σ_t d_t) δ`. -/
noncomputable def pooled (n : ℕ) (U d : Fin n → ℕ) (δ : ℝ) : ℝ :=
  sSup ((fun f : Fin n → ℝ => ∑ t, f t * U t) ''
    {f | (∀ t, 0 ≤ f t ∧ f t ≤ 1 - (d t : ℝ) / U t) ∧ ∑ t, (d t : ℝ) * f t ≤ budget (∑ t, d t) δ})

namespace Session

variable {n m : ℕ} {s : Fin n → ℕ} {X : Fin n → Type} (D : Session n m s X)

/-- Window `t`'s draw count, `min(s_t, U_t)`. -/
def draws (t : Fin n) (ω : Coins n m s X) : ℕ := min (s t) (D.size t ω)

/-- The receipts window `t` opens: its openings coin at its own size. -/
def drawn (t : Fin n) (ω : Coins n m s X) : Finset ℕ := ((ω t).2 (D.size t ω)).1.map Fin.valEmbedding

/-- Window `t`'s correct receipts among its `U_t`. -/
def good (t : Fin n) (ω : Coins n m s X) : ℕ :=
  (univ.filter fun i : Fin (D.size t ω) => (i : ℕ) ∈ D.correct t ω).card

/-- The session's wrong receipts, `Σ_t (U_t − good_t)`. -/
def wrong (ω : Coins n m s X) : ℕ := ∑ t, ((D.size t ω : ℕ) - D.good t ω)

/-- The session's pooled bound at `δ`, at the sizes and draw counts of outcome `ω`. -/
noncomputable def pooled (δ : ℝ) (ω : Coins n m s X) : ℝ :=
  Draw.pooled n (fun t => D.size t ω) (fun t => D.draws t ω) δ

end Session

end Accounting.Remote.Draw
