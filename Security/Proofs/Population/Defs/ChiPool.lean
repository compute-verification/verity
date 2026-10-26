import Proofs.Population.Defs.Exfiltration

/-!
# The cheating pool with the verdict lag

`Proofs.Population.Defs.Exfiltration`'s game, with the halt `Δ` windows after the first caught window's commitment
instead of at it. A window's leak is the harm it does: what its egress carries beyond the honest run, or, in the
cheating pool, the declared work it skips. A caught window's verdict arrives only after the next `Δ` windows have run,
so a cheater keeps skipping work in each of them until the halt. `Strategy.cum` stops at the caught window, so it omits
those windows; `Strategy.cumLag Δ` counts them.

* `Strategy.lagLeak n Δ` is what the next `Δ` of the `n` remaining windows leak, whatever their coins: they all run
  before the halt.
* `Strategy.cumUpToLag Δ j` is what leaves through the halt that the `j`-th detection triggers, `Δ` windows after that
  window's commitment.
* `Strategy.cumLag Δ` is the first-detection case, `j = 1`.
-/

namespace Population.Exfil

variable {Ω : Type}

/-- **What the next `Δ` windows leak**, of the `n` that remain, from the history `h`: each window's leak whatever its
coin, since they all run before the halt. -/
noncomputable def Strategy.lagLeak (A : Strategy Ω) : (n Δ : ℕ) → List Ω → (Fin n → Ω) → ℝ
  | 0, _, _, _ => 0
  | _ + 1, 0, _, _ => 0
  | n + 1, Δ + 1, h, cs => A.leak h + A.lagLeak n Δ (cs 0 :: h) (Fin.tail cs)

/-- **What leaves through the halt the `j`-th detection triggers**, `Δ` windows after that window's commitment: each
window's leak up to and including the `j`-th window that is caught, then the next `Δ` windows' (`lagLeak`). With `j = 0`
that detection has happened, and only the `Δ` windows before the halt remain. -/
noncomputable def Strategy.cumUpToLag [DecidableEq Ω] (A : Strategy Ω) (Δ : ℕ) : (n j : ℕ) → List Ω → (Fin n → Ω) → ℝ
  | 0, _, _, _ => 0
  | n + 1, 0, h, cs => A.lagLeak (n + 1) Δ h cs
  | n + 1, j + 1, h, cs => A.leak h + if cs 0 ∈ A.esc h then A.cumUpToLag Δ n (j + 1) (cs 0 :: h) (Fin.tail cs)
      else A.cumUpToLag Δ n j (cs 0 :: h) (Fin.tail cs)

/-- **What leaves through the halt**, for a deployment that halts `Δ` windows after the first caught window's
commitment: `Strategy.cum` plus the leak of the windows that run while that window's verdict is pending. -/
noncomputable def Strategy.cumLag [DecidableEq Ω] (A : Strategy Ω) (Δ n : ℕ) (h : List Ω) (cs : Fin n → Ω) : ℝ :=
  A.cumUpToLag Δ n 1 h cs

end Population.Exfil
