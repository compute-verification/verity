import Definitions.Pouw.FiatShamir
import Models.Pouw.Assumptions.FiatShamir

/-!
# Relative security under Fiat–Shamir draws: the grinding premium (trusted)

Decision 102 derives each matmul's seed from SHA-512 of commitments the prover has already made (its operand roots, at
the epoch's live salt), in place of `ServiceSoundStaged`'s live per-stage coins, so
`Accounting.Assumptions.TS.StageLuck`, which `Accounting.Guarantees.TS.StageBudget` and `StageIdeal` rest on, no longer
holds: the prover can re-roll a root
and try again. The statement here is its replacement. The grind is priced into the floor, as a premium `π` a seed group
in `F`; the tries reaching it go into the probability, `T·τ`. With GRIND's per-query moment (`SteerBoundOfMoment`) and
`π* = ln(T·M/δ)/θ`, that is `δ`. Their proofs are to be `Pouw.SecurityProofs.FiatShamir.FiatShamirFloor :
Pouw.Guarantees.FiatShamir.FiatShamirFloor` and `Pouw.SecurityProofs.FiatShamir.SteerBoundOfMoment`.

**Composition.**
* The live floors per prover (`Models.Pouw.Guarantees.RelativeSecurity`, `StagedFloorKPerProver` and its siblings)
  bound each adversary's error by `breakBound`: constants plus weights times its collision finders, read through
  `ecr/sha-512` (`Claims.ecrBound`) at budgets computed from the adversary's time. The fixed-oracle premises here
  (`LiveFloorUnsteered`, `ServiceSoundStagedFS`) have that form at each oracle (`FindersFS`), and the floor reads them
  through the same `Claims.ecrBound` at the same budgets (`breakBoundFS`): one break term, two statements. Gap (6)'s `Pouw.Guarantees.Window.WindowEps` (branch
  `cursor/pouw-window-eps-stmt-c7a9`) bounds the window audit's failure by `κ + coll(R, τ)` at the window's state; its
  `coll` is to be read the same way. The Fiat–Shamir terms compose after it.
* Gap (4)'s `Pouw.Guarantees.EndToEndWindow` (branch `cursor/pouw-work-bound-stmt-c7a9`) reads one oracle `H`, SHA-512
  domain-separated for the commitments and the noise. `SeedExtraction` is its `StraightLineExtraction` at the seed's
  domain, on the same random-oracle card, and its `T` is gap (4)'s `q`. The premium enters gap (4) through the work
  step: `TileWork(γ_u)`'s EPS1 instance is gap (3)'s bridge from compute accounting's tensor–scalar model
  (`CircuitCredit`, not on main), whose sampled proofs rest on `StageLuck`. Under these draws
  `ε_FS = ε_live + (κₓ − 3)/4·ε_luck + π*/(units a group)` (`κₓ` GRIND's exchange rate) replaces the live `ε` in
  `γ_u`, and the break term (`breakBoundFS`) plus `ε_id + δ` goes into `TileWork`'s `η(q, N)`. Neither bridge is
  stated: from this `Machine`/`QTree` game to `Pouw.Game`'s `CostModel` (RT4's change 4), and from the staged service
  to the TS model.
-/

namespace Pouw.Guarantees.FiatShamir

open Pouw.RelativeSecurity Pouw.Assumptions

/-- **Fiat–Shamir draws cost the live break term plus `ε_id + T·τ`, the premium priced into the floor** (proofs' gap
(7), restated after note:proofs/20261018T0532Z-report-redteam-fiatshamir, and per prover after RT-DELTA r2,
note:compute-accounting/20261018T1625Z-draft-rt-delta). Take a service whose drawn units' proofs derive their seeds
in-circuit at the inputs its commitment names, read as a uniform oracle's answers there (`SeedExtraction`, the
random-oracle premise, failing with probability at most `ε_id(T, |J|)`), and whose other coins are live and drawn
after the commitment. Take an analysis's gain map `gain`, premium `π`, and collision finders `fb`, `fz` for `K` breaks
at weights `βb`, `βz` (`FindersFS`). Suppose a try reaches the premium with probability at most `τ` (`SteerBound`).
Suppose too that at every oracle the live coins alone bound accepted runs charged below `F` whose every committed seed
realizes less than `π` by `κ + βb · Σ_k Pr[fb finds] + βz · Σ_k Pr[fz finds]` (`LiveFloorUnsteered`, the live
statement with the premium priced into `F`). Then every adversary `M` within `T` oracle calls, the inputs its
commitment names counted, has its accepted runs charged below `F` with probability at most

  `𝔼_{H,r}[min(1, κ + βb · Σ_k ecr(fb M r H k, qb) + βz · Σ_k ecr(fz M r H k, qz))] + ε_id(T, |J|) + T·τ`

over the oracle, the live coins and its own coins (`ServiceFloorFS`), `ecr(F, q)` equal to `q / 2^256` where the finder
meets `ecr/sha-512` within `q` SHA-512 calls in expectation and `⊤` where it doesn't (`breakBoundFS`,
`Claims.ecrBound`, the break term of the live floors per prover). The budgets are computed from the adversary, never
the finders' declared costs: `qb = cb · time M + db` for the binding finders and `qz = cz · time M + dz` for the
zero-knowledge ones, at constants the analysis states (the fork finder's two runs and the verifier's checks for `cb`,
`db`; the link finders' `2(1 + k)` runs and the extractor's calls for `cz`, `dz`), with `time M` the machine's worst
case, which `M` can't set low. A finder that declares a large cost to meet `ecr/sha-512` trivially exceeds the budget
and reads `⊤`. With a free state term `g` in the finders' place the statement said nothing, since `g = 1` satisfies the
premise; through the finders at computed budgets, a machine that hard-codes a collision gets about 1 and every other
adversary `K (βb qb + βz qz) / 2^256` on top of `κ`.

At GRIND's instance (`SteerBoundOfMoment`: `τ = M·e^{−θπ}`) and the premium `π* = ln(T·M/δ)/θ`, the tries cost
`T·τ = δ`, and the grind sits in `F`, at `π*` a seed group. The live floor's accumulation over stages is kept in the
gain, whose moment multiplies over a seed group's units (`PrivateMoment`), not in the union over tries. A steer event
at the live floor itself (`π` at zero: any lucky saving) prices the grind into the probability alone, and RT4's C2
bounds that from below for every analysis: at `T = |J|`, one try a stage, the bound is at least `|J|/Σ_j p_j⁻¹`, one
stage's luck `p_j` rather than their product, which forces it to about 1. At `T = |J|`, one try a stage, there is
nothing to grind and every draw is as fresh as a live coin, so the break term is the live floors' at the oracle and the
statement's own terms are `ε_id(|J|, |J|) + |J|·τ`. -/
def FiatShamirFloor : Prop :=
  ∀ (J N Live : Type) [Fintype J] [Fintype N] [Fintype Live] (K : ℕ) (Mc : Machine) (S : FSService J N Live)
    (T : ℕ) (εid : ℕ → ℕ → ℝ) (gain : GainMap Mc S) (sha : Claims.Msg → Claims.Msg) (fb fz : FindersFS Mc S K)
    (cb db cz dz π τ F κ βb βz : ℝ),
    SeedExtraction Mc S T εid → SteerBound Mc S T gain π τ → LiveFloorUnsteered Mc S T gain π F κ sha fb fz βb βz →
      ServiceFloorFS Mc S T F (fun (Ω : Type) [Fintype Ω] (M : FSAdv Mc S Ω) =>
        breakBoundFS sha fb fz cb db cz dz κ βb βz Ω M + ENNReal.ofReal (εid T (Fintype.card J) + T * τ))

/-- **GRIND's moment is `SteerBound`'s instance** (`Accounting.Guarantees.TS.ExchangeTail` at each try): if a try's
gain has moment at most `M ≥ 0` at rate `θ ≥ 0` (`SteerMoment`), a try reaches every premium `π` with probability at
most `M·e^{−θπ}`. -/
def SteerBoundOfMoment : Prop :=
  ∀ (J N Live : Type) [Fintype J] [Fintype N] (Mc : Machine) (S : FSService J N Live) (T : ℕ) (gain : GainMap Mc S)
    (θ M π : ℝ), 0 ≤ θ → 0 ≤ M → SteerMoment Mc S T gain θ M → SteerBound Mc S T gain π (M * Real.exp (-(θ * π)))

end Pouw.Guarantees.FiatShamir
