module

public import Definitions.Pouw.Game.Deadline
public import Definitions.Pouw.PearlC.Game
public import Definitions.Pouw.PearlC.Tile

/-!
# The deadline in Pearl-C's game (trusted)

`Definitions.Pouw.Game.Deadline`'s timed run and verifier's rule are the game's; Pearl-C's adversaries are its own type, so
their budget by the deadline and what the deadline reads of them are restated here. The named assumption is
`Pouw.Assumptions.HonestClock`, the one both games read, at `CostModel.adversaries`. `SampledByDeadline` is what the
sampled audit gives at one adversary, timed run and deadline, the conclusion a statement takes the honest clock for.
-/

@[expose] public section

namespace Pouw.PearlC

variable {Q R S : Type}

/-- The run spends at most `T` after the salt's release and by the deadline `td`, and preprocessing and the online phase
together make at most `q` oracle calls, for every oracle and salt (surely). -/
def Adversary.BoundedBy {CM : CostModel Q R S} (A : Adversary CM) (U : Workload) (run : TimedRun Q R S) (td T : ℝ)
    (q : ℕ) : Prop :=
  ∀ (H : Q → R) (s : S), run.spentIn H s run.ts td ≤ T ∧
    A.pre.calls H + CM.calls A.online H U A.act (A.pre.eval H) s ≤ q

/-- The adversaries of Pearl-C's cost model `CM` on the workload `U`, as the deadline reads them. -/
def CostModel.adversaries (CM : CostModel Q R S) (U : Workload) : Adversaries Q R S (Adversary CM) Transcript where
  pre A := A.pre
  transcript A := A.transcript U
  cost A H s := CM.cost A.online H U A.act (A.pre.eval H) s
  calls A H s := CM.calls A.online H U A.act (A.pre.eval H) s

/-- **What the sampled audit gives by the deadline** at the adversary `A`, the timed run `run` and the deadline `td`, on
the workload `U`: for every run spending at most `T` after the salt's release and by `td` with at most `q ≤ 2⁶⁴` oracle
calls, and every sample weighted by the tiles' `W_ref`, the audit accepts a transcript received by `td` while `T` is
below `1 − g` of the claimed work less `ε_s` of the layout's with probability at most `(1 − ε_s)^t + δ(q, N)`. -/
def SampledByDeadline [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] {CM : CostModel Q R S} (P : Protocol Q R S)
    (TR : TileRules Q R S) (U : Workload) (A : Adversary CM) (run : TimedRun Q R S) (td g : ℝ) (δ : ℕ → ℕ → ℝ) :
    Prop :=
  ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 → A.BoundedBy U run td T q →
    ∀ (w : ℕ → ℕ) (κ : ℝ), 0 < κ → (∀ i < (TR.tiling U.layout).NT, (w i : ℝ) = κ * TR.Wref U.layout i) →
    ∀ (t : ℕ) (εs : ℝ), εs ≤ 1 →
      pr (fun x : (Q → R) × S × (Fin t → Ticket w (TR.tiling U.layout).NT) =>
          AcceptsBy run td x.1 x.2.1 (goodTiles P TR U x.1 x.2.1 (A.transcript U x.1 x.2.1) ∪
            blankTiles TR U (A.transcript U x.1 x.2.1)) x.2.2 ∧
          T < (1 - g) * (claimedWref TR U (A.transcript U x.1 x.2.1) - εs * tileWrefTotal TR U.layout))
        ≤ (1 - εs) ^ t + δ q U.layout.N

end Pouw.PearlC
