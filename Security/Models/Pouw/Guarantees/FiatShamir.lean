import Definitions.Pouw.FiatShamir
import Models.Pouw.Assumptions.FiatShamir

/-!
# Relative security under Fiat–Shamir draws: the grinding premium (trusted)

Decision 102 derives each matmul's seed from SHA-512 of commitments the prover has already made (its operand roots, at
the epoch's live salt), in place of `ServiceSoundStaged`'s live per-stage coins, so `Accounting.Assumptions.TS.StageLuck`,
which `Accounting.Guarantees.TS.StageBudget` and `StageIdeal` rest on, no longer holds: the prover can re-roll a root
and try again. The statement here is its replacement. The grind is priced into the floor, as a premium `π` a seed group
in `F`; the tries reaching it go into the probability, `T·τ`. With GRIND's per-query moment (`SteerBoundOfMoment`) and
`π* = ln(T·M/δ)/θ`, that is `δ`. Their proofs are to be `Pouw.SecurityProofs.FiatShamir.FiatShamirFloor :
Pouw.Guarantees.FiatShamir.FiatShamirFloor` and `Pouw.SecurityProofs.FiatShamir.SteerBoundOfMoment`.

**Composition.**
* Gap (6)'s `Pouw.Guarantees.Window.WindowEps` (branch `cursor/pouw-window-eps-stmt-c7a9`) bounds the window audit's
  failure by `κ + coll(R, τ)` at the window's state, with no grinding term. It is the shape of the fixed-oracle premises
  here (`LiveFloorUnsteered`, `ServiceSoundStagedFS`: `κ + g(M, r, H)`), and their `g` is its `coll` at the oracle `H`.
  The Fiat–Shamir terms compose after it.
* Gap (4)'s `Pouw.Guarantees.EndToEndWindow` (branch `cursor/pouw-work-bound-stmt-c7a9`) reads one oracle `H`, SHA-512
  domain-separated for the commitments and the noise. `SeedExtraction` is its `StraightLineExtraction` at the seed's
  domain, on the same random-oracle card, and its `T` is gap (4)'s `q`. The premium enters gap (4) through the work
  step: `TileWork(γ_u)`'s EPS1 instance is gap (3)'s bridge from compute accounting's tensor–scalar model
  (`CircuitCredit`, not on main), whose sampled proofs rest on `StageLuck`. Under these draws `ε_FS = ε_live + (κ − 3)/4·ε_luck + π*/(units a group)` replaces
  the live `ε` in `γ_u`, and `coll + ε_id + δ` goes into `TileWork`'s `η(q, N)`. Neither bridge is stated: from this
  `Machine`/`QTree` game to `Pouw.Game`'s `CostModel` (RT4's change 4), and from the staged service to the TS model.
-/

namespace Pouw.Guarantees.FiatShamir

open Pouw.RelativeSecurity Pouw.Assumptions

/-- **Fiat–Shamir draws cost `κ + coll + ε_id + T·τ`, the premium priced into the floor** (proofs' gap (7), restated
after note:proofs/20261018T0532Z-report-redteam-fiatshamir). Take a service whose drawn units' proofs derive their
seeds in-circuit at the inputs its commitment names, read as a uniform oracle's answers there (`SeedExtraction`, the
random-oracle premise, failing with probability at most `ε_id(T, |J|)`), and whose other coins are live and drawn
after the commitment. Take an analysis's gain map `gain`, premium `π` and state term `g`. Suppose a try reaches the
premium with probability at most `τ` (`SteerBound`). Suppose too that at every oracle the live coins alone bound
accepted runs charged below `F` whose every committed seed realizes less than `π` by `κ + g(M, r, H)`
(`LiveFloorUnsteered`, the live statement with the premium priced into `F`). Then every adversary `M` within `T` oracle
calls, the inputs its commitment names counted, has its accepted runs charged below `F` with probability at most

  `κ + coll(M) + ε_id(T, |J|) + T·τ`

over the oracle, the live coins and its own coins (`ServiceFloorFS`), where `coll(M)` is its state term's mean.

At GRIND's instance (`SteerBoundOfMoment`: `τ = M·e^{−θπ}`) and the premium `π* = ln(T·M/δ)/θ`, the tries cost
`T·τ = δ`, and the grind sits in `F`, at `π*` a seed group. The live floor's accumulation over stages is kept in the
gain, whose moment multiplies over a seed group's units (`PrivateMoment`), not in the union over tries. A steer event
at the live floor itself (`π` at zero: any lucky saving) prices the grind into the probability alone, and RT4's C2
bounds that from below for every analysis: at `T = |J|`, one try a stage, the bound is at least `|J|/Σ_j p_j⁻¹`, one
stage's luck `p_j` rather than their product, which forces it to about 1. -/
def FiatShamirFloor : Prop :=
  ∀ (J N Live : Type) [Fintype J] [Fintype N] [Fintype Live] (Mc : Machine) (S : FSService J N Live) (T : ℕ)
    (εid : ℕ → ℕ → ℝ) (gain : GainMap Mc S) (g : StateTerm Mc S) (π τ F κ : ℝ),
    SeedExtraction Mc S T εid → SteerBound Mc S T gain π τ → LiveFloorUnsteered Mc S T gain π F κ g →
      ServiceFloorFS Mc S T F (fun (Ω : Type) [Fintype Ω] (M : FSAdv Mc S Ω) =>
        κ + coll g Ω M + εid T (Fintype.card J) + T * τ)

/-- **GRIND's moment is `SteerBound`'s instance** (`Accounting.Guarantees.TS.ExchangeTail` at each try): if a try's
gain has moment at most `M ≥ 0` at rate `θ ≥ 0` (`SteerMoment`), a try reaches every premium `π` with probability at
most `M·e^{−θπ}`. -/
def SteerBoundOfMoment : Prop :=
  ∀ (J N Live : Type) [Fintype J] [Fintype N] (Mc : Machine) (S : FSService J N Live) (T : ℕ) (gain : GainMap Mc S)
    (θ M π : ℝ), 0 ≤ θ → 0 ≤ M → SteerMoment Mc S T gain θ M → SteerBound Mc S T gain π (M * Real.exp (-(θ * π)))

end Pouw.Guarantees.FiatShamir
