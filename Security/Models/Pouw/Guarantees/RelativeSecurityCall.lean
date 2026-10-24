import Definitions.Pouw.RelativeSecurityCall
import Models.Pouw.Assumptions.RelativeSecurity
import Models.Pouw.Assumptions.RelativeSecurityCall

/-!
# Relative security's floors at J10-R's call scope (trusted)

RT-ALGO-17 (note:compute-accounting/20261019T0845Z-draft-rt-algo-r3). The floors of record charge a stage's
straight-line program over its own design bits (`ChargeVF3j16`, `ChargeVF3j16Overlap`). Under J10-R a call's units
share their noise, so these restate them for one straight-line program a call over J10-R's call coins
(`ChargeVF3j16Call`, `ChargeVF3j16CallOverlap`), on Theorem K's one-sided scope read by unit (`KScopeOneSidedCall`,
`KScopeOverlapOneSidedCall`), with the service premises unchanged at the calls' shapes. Each proof is
`Pouw.SecurityProofs.RelativeSecurity.G : Pouw.Guarantees.RelativeSecurity.G`.

The floors and their luck are the unit floors' summed over the call's units: a call of `mu × nu` units of `ur` rows
and `uc` columns at `B` segments of `L` is charged at least `κ L · mu nu · ur uc B`, `mu nu` times a unit's
`κ L ur uc B`, at the parity luck 0.854 and error `2^{−40} + R (δb + δz)`. The constant is unchanged: sharing the
coins across units costs the floor nothing on this scope, because on it no product is two-sided at two cell-blocks
whose words read it at odd scale, so none counts for two of them, whichever units they belong to. A product two-sided
at two units' cell-blocks is outside the scope, and the floors say nothing of a run that forms one (K1 is not claimed
there).

The per-prover forms (`StagedFloorKOneSidedCallPerProver`, `OverlapFloorKOneSidedCallPerProver`) are stated, not
proved, as the unit ones are; each follows from its unit form (`Proofs.Pouw.RelativeSecurityCall`).
-/

namespace Pouw.Guarantees.RelativeSecurity

open Pouw.RelativeSecurity Pouw.Assumptions

/-- **The staged window's service floor at J10-R's call scope**: `StagedFloorKOneSided` with each stage one call. In
every machine `Mc` and for every service `S` whose seed is `R` rounds of `St` stages, each stage one J10-R call of
`mu × nu` units of `ur` rows and `uc` columns at `B` segments of `L` (its coins J10-R's, both sides shared, on the
re-encoded design: 7 planes at `wRe`, integers mod `2^32` read mod 2 by `toF2`, plane 0 at an odd weight), under
`ServiceSoundStaged` at the call's shape, `KScopeOneSidedCall` and `ChargeVF3j16Call`, every adversary that commits
first has its accepted runs charged below `(R − 274 ln 2) · St · κ L · mu nu · ur uc B − Vmax` with probability at most
`2^{−40} + R (δb + δz)`: the unit floor, summed over the call's `mu nu` units. -/
def StagedFloorKOneSidedCall : Prop :=
  ∀ (Mc : Machine) (S : Service) (R St mu nu ur uc L B : ℕ) (e : S.Seed ≃ Paths (CallStage St mu nu ur uc L 7 B) R)
    (run : CallRunMap Mc S R St mu nu ur uc L 7 B (ZMod (2 ^ 32))) (δb δz κ Vmax : ℝ),
    0 < St → 0 < κ → 0 < L * (mu * nu * (ur * uc * B)) → 0 ≤ δb → 0 ≤ δz →
    ServiceSoundStaged Mc S run e δb δz → KScopeOneSidedCall Mc S run wRe toF2 0 →
      ChargeVF3j16Call Mc S run e κ Vmax →
    ServiceFloor Mc S ((R - 274 * Real.log 2) * (St * (κ * L * ((mu * nu * (ur * uc * B) : ℕ) : ℝ))) - Vmax)
      (((2 : ℝ) ^ 40)⁻¹ + R * (δb + δz))

/-- **The overlapped window's service floor at J10-R's call scope**: `OverlapFloorKOneSided` with each draw one call.
In every machine `Mc` and for every service `S` whose seed is `R` rounds of `St` draws, draw `t` one J10-R call of
`mu t × nu t` units of `ur` rows and `uc` columns at `B t` segments of `L` (as `StagedFloorKOneSidedCall`), overlapping
at any depth, its check's cell-block drawn uniformly among every call's (`ServiceSoundOverlap`'s deployment
requirement), under `ServiceSoundOverlap` at the calls' shapes, `KScopeOverlapOneSidedCall` and
`ChargeVF3j16CallOverlap`, every adversary that commits first has its accepted runs charged below
`(R − 274 ln 2) · κ L Σ_t mu_t nu_t · ur uc B_t − Vmax` with probability at most `2^{−40} + R (δb + δz)`. -/
def OverlapFloorKOneSidedCall : Prop :=
  ∀ (Mc : Machine) (S : Service) (R St ur uc L : ℕ) (mu nu B : Fin St → ℕ)
    (e : S.Seed ≃ Paths (OCallSpot St mu nu ur uc L 7 B) R)
    (run : OCallRunMap Mc S R St mu nu ur uc L 7 B (ZMod (2 ^ 32))) (δb δz κ Vmax : ℝ),
    0 < κ → 0 < L * ∑ t, mu t * nu t * (ur * uc * B t) → 0 ≤ δb → 0 ≤ δz →
    ServiceSoundOverlap Mc S run e δb δz → KScopeOverlapOneSidedCall Mc S run wRe toF2 0 →
      ChargeVF3j16CallOverlap Mc S run e κ Vmax →
    ServiceFloor Mc S ((R - 274 * Real.log 2) * (κ * L * ((∑ t, mu t * nu t * (ur * uc * B t) : ℕ) : ℝ)) - Vmax)
      (((2 : ℝ) ^ 40)⁻¹ + R * (δb + δz))

/-- `StagedFloorKOneSidedCall` per prover, at each adversary's own error: under `ServiceSoundStagedPerProver` at the
call's shape and the analysis's finders `fb` and `fz`, and `StagedFloorKOneSidedCall`'s other two premises, every
adversary `M` that commits first has its accepted runs charged below
`(R − 274 ln 2) · St · κ L · mu nu · ur uc B − Vmax` with probability at most
`breakBound sha fb fz cb db cz dz 2^{−40} δb βb δz βz Ω M`. -/
def StagedFloorKOneSidedCallPerProver : Prop :=
  ∀ (Mc : Machine) (S : Service) (R St mu nu ur uc L B : ℕ) (e : S.Seed ≃ Paths (CallStage St mu nu ur uc L 7 B) R)
    (run : CallRunMap Mc S R St mu nu ur uc L 7 B (ZMod (2 ^ 32))) (sha : Claims.Msg → Claims.Msg)
    (fb fz : Finders Mc S R) (cb db cz dz δb βb δz βz κ Vmax : ℝ),
    0 < St → 0 < κ → 0 < L * (mu * nu * (ur * uc * B)) → 0 ≤ δb → 0 ≤ βb → 0 ≤ δz → 0 ≤ βz →
    ServiceSoundStagedPerProver Mc S run e sha fb fz δb βb δz βz → KScopeOneSidedCall Mc S run wRe toF2 0 →
      ChargeVF3j16Call Mc S run e κ Vmax →
    ServiceFloorPerProver Mc S
      ((R - 274 * Real.log 2) * (St * (κ * L * ((mu * nu * (ur * uc * B) : ℕ) : ℝ))) - Vmax)
      (breakBound sha fb fz cb db cz dz ((2 : ℝ) ^ 40)⁻¹ δb βb δz βz)

/-- `OverlapFloorKOneSidedCall` per prover, at each adversary's own error: under `ServiceSoundOverlapPerProver` at the
calls' shapes and the analysis's finders `fb` and `fz`, and `OverlapFloorKOneSidedCall`'s other two premises, every
adversary `M` that commits first has its accepted runs charged below
`(R − 274 ln 2) · κ L Σ_t mu_t nu_t · ur uc B_t − Vmax` with probability at most
`breakBound sha fb fz cb db cz dz 2^{−40} δb βb δz βz Ω M`. -/
def OverlapFloorKOneSidedCallPerProver : Prop :=
  ∀ (Mc : Machine) (S : Service) (R St ur uc L : ℕ) (mu nu B : Fin St → ℕ)
    (e : S.Seed ≃ Paths (OCallSpot St mu nu ur uc L 7 B) R)
    (run : OCallRunMap Mc S R St mu nu ur uc L 7 B (ZMod (2 ^ 32))) (sha : Claims.Msg → Claims.Msg)
    (fb fz : Finders Mc S R) (cb db cz dz δb βb δz βz κ Vmax : ℝ),
    0 < κ → 0 < L * ∑ t, mu t * nu t * (ur * uc * B t) → 0 ≤ δb → 0 ≤ βb → 0 ≤ δz → 0 ≤ βz →
    ServiceSoundOverlapPerProver Mc S run e sha fb fz δb βb δz βz → KScopeOverlapOneSidedCall Mc S run wRe toF2 0 →
      ChargeVF3j16CallOverlap Mc S run e κ Vmax →
    ServiceFloorPerProver Mc S
      ((R - 274 * Real.log 2) * (κ * L * ((∑ t, mu t * nu t * (ur * uc * B t) : ℕ) : ℝ)) - Vmax)
      (breakBound sha fb fz cb db cz dz ((2 : ℝ) ^ 40)⁻¹ δb βb δz βz)

end Pouw.Guarantees.RelativeSecurity
