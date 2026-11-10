module

public import Definitions.Pouw.Game.Defs
public import Definitions.Pouw.NCP.RouteU
public import Definitions.Pouw.PearlC.Game

/-!
# The advantage form (trusted)

PoUW's security property in the form every protocol states: the adversary's advantage `ε`, with failure probability
`δ(q, N)`. `G_ε` says that what an adversary spends is at least the honest work of its correct units over `1 + ε`,
except with probability `δ` or by a break of SHA-512 (`Adversary.Breaks`), which is `GGap`'s property at the gap
`g = ε/(1 + ε)`: the gap is how the proofs carry it.
`AuditGε` is the audit's statement in ε form, `Protocol.WrefWithin` the bound `Ω_max` the reduction from TT reads,
and `Pouw.NCP.Params.InstanceU` route U's instance conditions.
-/

@[expose] public section

namespace Pouw

open Finset

variable {Q R S : Type}

/-- **The advantage property** `G_ε` at the adversary's advantage `ε` with failure probability `δ(q, N)`, in `GGap`'s
quantifier order: for every in-domain workload, every budget `T ≥ 0`, every `q ≤ 2^64` and every adversary within
`(T, q)`, `Pr_{H,s}[(1 + ε)·T < ∑_{u ∈ K} W_ref(u) ∧ 𝒜's run doesn't break SHA-512] ≤ δ`. What an adversary spends is
at least the honest work of its correct units over `1 + ε`, except with probability `δ` or by a break. -/
def Gε [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (P : Protocol Q R S)
    (D : Layout → Prop) (ε : ℝ) (δ : ℕ → ℕ → ℝ) : Prop :=
  ∀ U : Workload, U.InDomain D → ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 → ∀ A : Adversary CM, A.Bounded U T q →
    pr (fun x : (Q → R) × S =>
        (1 + ε) * T < ∑ u ∈ correctSet P U x.1 x.2 (A.transcript U x.1 x.2), (P.Wref U.layout u : ℝ) ∧
          ¬ A.Breaks U x.1 x.2)
      ≤ δ q U.layout.N

/-- **The audit's advantage** at the protocol's advantage `ε` and failure probability `δ(q, N)`: for every in-domain
workload, budget `T ≥ 0`, `q ≤ 2^64`, adversary within `(T, q)`, number of samples `t` and share `ε_s < 1` of the
reference work the samples may miss, the audit with `t` work-weighted samples accepts while
`(1 + ε)/(1 − ε_s)·T < W` and the run doesn't break SHA-512 with probability at most `(1 − ε_s)^t + δ(q, N)`: the
audit's own advantage is `(1 + ε)/(1 − ε_s) − 1`. -/
def AuditGε [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (P : Protocol Q R S)
    (D : Layout → Prop) (ε : ℝ) (δ : ℕ → ℕ → ℝ) : Prop :=
  ∀ U : Workload, U.InDomain D → ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 → ∀ A : Adversary CM, A.Bounded U T q →
    ∀ (t : ℕ) (εs : ℝ), εs < 1 →
      pr (fun x : (Q → R) × S × (Fin t → Ticket (P.Wref U.layout) U.layout.N) =>
          AuditAccepts (correctSet P U x.1 x.2.1 (A.transcript U x.1 x.2.1)) x.2.2 ∧
          (1 + ε) / (1 - εs) * T < totalWref P U.layout ∧ ¬ A.Breaks U x.1 x.2.1)
        ≤ (1 - εs) ^ t + δ q U.layout.N

/-- `Ω_max`'s bound on the domain `D`: every unit of every in-domain workload has `W_ref(u) ≤ Ω_max·W_mm(u)`. -/
def Protocol.WrefWithin (P : Protocol Q R S) (D : Layout → Prop) (Ωmax : ℝ) : Prop :=
  ∀ U : Workload, U.InDomain D → ∀ u < U.layout.N, (P.Wref U.layout u : ℝ) ≤ Ωmax * ((U.layout.shape u).Wmm : ℝ)

end Pouw

namespace Pouw.PearlC

open Finset

variable {Q R S : Type}

/-- **The advantage property** `G_ε` of Pearl-C's game at the adversary's advantage `ε` with failure probability
`δ(q, N)`, in `GGap`'s quantifier order: `Pr_{H,s}[(1 + ε)·T < ∑_{u ∈ K} W_ref(u)] ≤ δ`. What an adversary spends is at
least the honest work of its correct units over `1 + ε`, except with probability `δ`. -/
def Gε [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (P : Protocol Q R S)
    (D : Layout → Prop) (ε : ℝ) (δ : ℕ → ℕ → ℝ) : Prop :=
  ∀ U : Workload, U.InDomain P D → ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 → ∀ A : Adversary CM, A.Bounded U T q →
    pr (fun x : (Q → R) × S => (1 + ε) * T < ∑ u ∈ correctSet P U x.1 x.2 (A.transcript U x.1 x.2), P.Wref U.layout u)
      ≤ δ q U.layout.N

end Pouw.PearlC

namespace Pouw.NCP

/-- Route U's instance on the domain `D` at `f` forming units per position: `W_ref` is route U's, `wrefU f`; every
unit has at least `n₀` columns; and `g₀ ≥ d/(3k)` for every unit. -/
def Params.InstanceU {Q R S : Type} (P : Params Q R S) (f : ℕ) (D : Layout → Prop) (g₀ : ℝ) (n₀ : ℕ) : Prop :=
  (∀ (L : Layout) (u : ℕ), P.Wref L u = wrefU f L u) ∧
    (∀ U : Workload, U.InDomain D → ∀ u < U.layout.N, n₀ ≤ (U.layout.shape u).n) ∧
    ∀ U : Workload, U.InDomain D → ∀ u < U.layout.N, (P.d : ℝ) ≤ g₀ * (3 * (U.layout.shape u).k)

end Pouw.NCP
