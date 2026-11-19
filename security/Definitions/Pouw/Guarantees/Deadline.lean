module

public import Definitions.Pouw.Assumptions.GameStatements
public import Definitions.Pouw.Assumptions.Deadline
public import Definitions.Pouw.Assumptions.Game
public import Definitions.Pouw.PearlC.Deadline
public import Definitions.Pouw.Assumptions.PearlC.TTOutRev1
public import Definitions.Pouw.Assumptions.PearlC.TTOutW1
public import Definitions.Pouw.Game.Params
public import Definitions.Pouw.PearlC.Deadline
public import Definitions.Pouw.PearlC.DeviceKernelWref
public import Definitions.Pouw.PearlC.Tile
public import Definitions.Pouw.Game.Deadline
public import Definitions.Assumptions.Clock
public import Definitions.Pouw.PearlC.Game
import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-!
# Work between the salt and the deadline (trusted)

A proof of compute utilization: acceptance means at least `W` of work after the salt and before a deadline. `EndToEnd`
gives the work and its freshness (preprocessing never reads the salt); the deadline is `Definitions.Pouw.Game.Deadline`'s
verifier's rule, which reads the auditor's device's two entries, the salt's and the docket's, and checks
`0 ≤ ra.elapsed rb ≤ Δ` on its clock and the Site's drift (`clockDrift`). The device's clock meets real time through
Core's `clock-within-drift` (`Verity.Assumptions.ClockWithin`), which puts the docket's entry within the real window
`realWindow site k Δ = Δ / (1 − q)` after the salt's; the real window meets the cost model through the honest clock
(`Pouw.Assumptions.HonestClock`). Every real time read is the device's own entries', `a`'s and `b`'s.

* `WorkBetweenSaltAndDeadline`: `EndToEnd` with the budget on the run's spending in the real window from the salt's
  entry, under the clock and the honest clock. Its hypotheses are `EndToEnd`'s, `Adversary.Bounded`'s online cost read
  in that window (`Adversary.BoundedBy`), and the two premises.
* `SpareCapacityAtDeadline`: remote's exhaustion. A declared fleet that can spend `C·w` in the real window `w` has at
  most `C·w − (1 − g)(1 − ε_s)·W` of it to spare for undeclared work, except with the same probability.
* `PearlCSampledByDeadline`: the same in Pearl-C's game, from `GGapSampled`, so at every device record from
  `PearlC.SampledDevRev1K`. Its instances at the served record's numbers, on the four layouts the sampled g is pinned
  at, are lemmas (`Proofs.Pouw.Lemmas.Deadline`).
-/

@[expose] public section

namespace Pouw.Guarantees

open Pouw.GameStatements Finset

/-- **Work between the salt and the deadline.** Under TT(g₀), the `Ω_max` bound, `clock-within-drift` on the
auditor's device registered under key `k` and an honest clock, for every in-domain workload, adversary, Site, docket
encoding, pair of the device's entries (the salt's `a`, the docket's `b`), run and deadline of `Δ` nanoseconds on the
device's clock, and budget: if the run spends at most `T` in the real window `w = realWindow site k Δ` from the salt's
entry, and makes at most `q` oracle calls, then the verifier accepts (the Site passes the clock's check, the docket is
bracketed by `a` and `b` with `0 ≤ ra.elapsed rb ≤ Δ`, and the audit with `t` work-weighted samples accepts) while
`T < (1 − g)(1 − ε_s)·W`, `g = 1 − (1 − g₀)/Ω_max`, and the run doesn't break SHA-512, with probability at most
`(1 − ε_s)^t + δ(q, N)`. That is, except with that probability, an accepted trace means at least `(1 − g)(1 − ε_s)·W`
of cost spent between the salt's entry and the end of its real window, or a collision among the run's own SHA-512
inputs. -/
def WorkBetweenSaltAndDeadline : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (P : Protocol Q R S) (D : Layout → Prop) (g₀ Ωmax : ℝ) (δ : ℕ → ℕ → ℝ),
    g₀ < 1 → 0 < Ωmax →
    (∀ U : Workload, U.InDomain D → ∀ u < U.layout.N,
      (P.Wref U.layout u : ℝ) ≤ Ωmax * ((U.layout.shape u).Wmm : ℝ)) →
    Assumptions.TT CM P D g₀ δ →
    ∀ U : Workload, U.InDomain D → ∀ (A : Adversary CM) (site : Verity.Site) (k : String) (dk : Docket S Trace k)
      (st : Stamps Q R S k) (run : TimedRun Q R S) (Δ : ℕ),
    Verity.Assumptions.ClockWithin site k st.real →
    Assumptions.HonestClock (CM.adversaries U) A dk st run (realWindow site k Δ) →
    ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 → A.BoundedBy U st run (realWindow site k Δ) T q →
    ∀ (t : ℕ) (εs : ℝ), εs ≤ 1 →
      pr (fun x : (Q → R) × S × (Fin t → Ticket (P.Wref U.layout) U.layout.N) =>
          AcceptsBy site dk st Δ x.1 x.2.1 (A.trace U x.1 x.2.1)
              (correctSet P U x.1 x.2.1 (A.trace U x.1 x.2.1)) x.2.2 ∧
            T < (1 - reductionGap g₀ Ωmax) * (1 - εs) * totalWref P U.layout ∧ ¬ A.Breaks U x.1 x.2.1)
        ≤ (1 - εs) ^ t + δ q U.layout.N

/-- **Remote's exhaustion: the spare capacity at the deadline.** A declared fleet of capacity `C` per unit of time can
spend `C·w` in the real window `w = realWindow site k Δ` from the salt's entry. Under `clock-within-drift` and an honest
clock, if the run's spending in that window and undeclared work `Y` together fit in it on every oracle and salt, then,
except with probability `(1 − ε_s)^t + δ(q, N)`, a trace the verifier accepts at the deadline of `Δ` nanoseconds on the
device's clock, from a run that doesn't break SHA-512, leaves `Y ≤ C·w − (1 − g)(1 − ε_s)·W`: the undeclared spare
capacity is at most the fleet's capacity less the work the audit certifies. -/
def SpareCapacityAtDeadline : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (P : Protocol Q R S) (D : Layout → Prop) (g₀ Ωmax : ℝ) (δ : ℕ → ℕ → ℝ),
    g₀ < 1 → 0 < Ωmax →
    (∀ U : Workload, U.InDomain D → ∀ u < U.layout.N,
      (P.Wref U.layout u : ℝ) ≤ Ωmax * ((U.layout.shape u).Wmm : ℝ)) →
    Assumptions.TT CM P D g₀ δ →
    ∀ U : Workload, U.InDomain D → ∀ (A : Adversary CM) (site : Verity.Site) (k : String) (dk : Docket S Trace k)
      (st : Stamps Q R S k) (run : TimedRun Q R S) (Δ : ℕ),
    Verity.Assumptions.ClockWithin site k st.real →
    Assumptions.HonestClock (CM.adversaries U) A dk st run (realWindow site k Δ) →
    ∀ C Y : ℝ, Y ≤ C * realWindow site k Δ → ∀ q : ℕ, q ≤ 2 ^ 64 →
    (∀ (H : Q → R) (s : S), run.spentIn H s (st.ts s) (st.ts s + realWindow site k Δ) + Y ≤ C * realWindow site k Δ ∧
      A.pre.calls H + CM.calls A.online H U A.act (A.pre.eval H) s ≤ q) →
    ∀ (t : ℕ) (εs : ℝ), εs ≤ 1 →
      pr (fun x : (Q → R) × S × (Fin t → Ticket (P.Wref U.layout) U.layout.N) =>
          AcceptsBy site dk st Δ x.1 x.2.1 (A.trace U x.1 x.2.1)
              (correctSet P U x.1 x.2.1 (A.trace U x.1 x.2.1)) x.2.2 ∧
            C * realWindow site k Δ - (1 - reductionGap g₀ Ωmax) * (1 - εs) * totalWref P U.layout < Y ∧
            ¬ A.Breaks U x.1 x.2.1)
        ≤ (1 - εs) ^ t + δ q U.layout.N

/-- **Pearl-C's sampled audit by the deadline.** `GGapSampled`, `clock-within-drift` and an honest clock give, for every
in-domain workload, adversary, Site, docket encoding, pair of the device's entries, run and deadline of `Δ` nanoseconds
on the device's clock, every run spending at most `T` in the real window `realWindow site k Δ` from the salt's entry with
at most `q` oracle calls, and every sample weighted by the tiles' `W_ref`: the verifier accepts (the clock's check, the
docket in time, the audit) while `T` is below `1 − g` of the claimed work less `ε_s` of the layout's with probability at
most `(1 − ε_s)^t + δ(q, N)`. -/
def PearlCSampledByDeadline : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : PearlC.CostModel Q R S)
    (P : PearlC.Protocol Q R S) (TR : PearlC.TileRules Q R S) (D : PearlC.Layout → Prop) (g : ℝ)
    (δ : ℕ → ℕ → ℝ), PearlC.GGapSampled CM P TR D g δ →
    ∀ U : PearlC.Workload, U.InDomain P D → ∀ (A : PearlC.Adversary CM) (site : Verity.Site) (k : String)
      (dk : Docket S PearlC.Trace k) (st : Stamps Q R S k) (run : TimedRun Q R S) (Δ : ℕ),
    Verity.Assumptions.ClockWithin site k st.real →
    Assumptions.HonestClock (CM.adversaries U) A dk st run (realWindow site k Δ) →
    ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 → A.BoundedBy U st run (realWindow site k Δ) T q →
    ∀ (w : ℕ → ℕ) (κ : ℝ), 0 < κ → (∀ i < (TR.tiling U.layout).NT, (w i : ℝ) = κ * TR.Wref U.layout i) →
    ∀ (t : ℕ) (εs : ℝ), εs ≤ 1 →
      pr (fun x : (Q → R) × S × (Fin t → Ticket w (TR.tiling U.layout).NT) =>
          AcceptsBy site dk st Δ x.1 x.2.1 (A.trace U x.1 x.2.1)
              (PearlC.goodTiles P TR U x.1 x.2.1 (A.trace U x.1 x.2.1) ∪
                PearlC.blankTiles TR U (A.trace U x.1 x.2.1)) x.2.2 ∧
            T < (1 - g) * (PearlC.claimedWref TR U (A.trace U x.1 x.2.1) - εs * PearlC.tileWrefTotal TR U.layout))
        ≤ (1 - εs) ^ t + δ q U.layout.N

end Pouw.Guarantees
