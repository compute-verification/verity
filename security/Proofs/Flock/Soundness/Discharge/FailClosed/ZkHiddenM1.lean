module

public import Proofs.Flock.Soundness.Discharge.ZkLink.M1
public import Proofs.Flock.Soundness.Discharge.ZkLink.StrictZ
public import Proofs.Flock.Soundness.Discharge.ZkLink.Link
public import Proofs.Flock.Soundness.Discharge.FailClosed.Numeric
public import Proofs.Flock.Soundness.Instance

/-!
# The hidden-output `--zk` tables are at Flock ZK's parameters, and `ε_ks` in closed form

`ZkLink/M1.lean` for the hidden-output plan of `J` tables (`ZkHidden.planZHJ`): every table it builds is at Flock
ZK's parameters, from what `verify --zk` checks at setup, so the knowledge term `ε_ks = ksAvgStrictZ` at that plan is at
most `ksAvgStrictZm1`, and at most one number that reads no table.

A refused hidden-output table carries the reference setup's regions and `contra`'s two (`SampleSetupZHJ.refused`), up to
1026 regions, where `TabZK.M1` reads `Stmt.InRange`'s 1024. So the tables meet `TabZK.M1W`, `M1` at
`Stmt.InRangeWide`, and the padded table error is checked there:

* `padTableError_le_m1_wide`: at 1026 regions, `padLinkError + padRepError² ≤ 3/2^206` at `m1Schedule m`,
  `25 ≤ m ≤ 35` (`m1_check_wide`, one kernel check per `m`, beside `m1_check`'s `2^-205` at 1024).
* `TabZK.bound_le_m1w`: a table's `bound` at `M1W` is at most `2^-203`, as `TabZK.bound_le_m1` at `M1`.
* `ksStrictZ_le_m1w`, `ksAvgStrictZ_le_m1w`: `ksStrictZ_le_m1` and `ksAvgStrictZ_le_m1` at `M1W`.
* `m1Ks_all`: the per-`m` kernel check for the hash-sum coefficient. At `m1Schedule m`, the sum over levels `ℓ ≥ 1`
  of `√(2^{log N_ℓ}·Q_ℓ)` is at most `25638` (each term below an integer `rootUp` whose square the kernel checks),
  and level 0's codeword is at most `2^23` long.
* `ksStrictZm1_le_closed`, `rateZ_le_m1w`, `ksAvgStrictZ_le_closed`: so at `M1W` tables
  $$\varepsilon_{ks}\le\frac{2^{-203}+(2\cdot25638+2^{14}+2)\sqrt{q_F^2/2^{513}}+q_S^2/2^{513}
  +\sqrt{k\,q_F^2/2^{513}}}{1-2^{23}/(ek)}$$
  (`ksClosedM1`, `rateM1`), whenever `2^23 < ek`.
* `SampleSetupZHJ.tabAt_M1W`, `table_M1W`, `refused_M1W`, `tableZHJ_M1W`, `planZHJ_M1W`: every table of `planZHJ` is at
  `M1W` (`Stmt.setupHidden`'s `checkInRange`, `Zk.schedule`'s `schedule_m1`, the inner code's `innerN_pow`), and
  `ksAvgStrictZ_planZHJ_le_m1` and `ksAvgStrictZ_planZHJ_le_closed` read the bounds at that plan (its rate cap is
  `ZkHidden.hr_planZHJ_23`).
* `qs_le_io`, `qs_le_mul`: `Q_s = Σ_c |Ω_c|/|Ω|`, the factor of `ε_bind` (`linkBoundZC`), is at most the largest
  `Σ_{u ∈ S} |io u|` over samples `S`, `io` the committed wires each unit reads or writes in the circuit's partition, so
  at most `k_d·r` at samples of `k_d` units with `|io u| ≤ r`.

The file sits under `FailClosed/` because it reads `Stmt.InRangeWide` and `piopNumMaxWide` (`FailClosed/Numeric.lean`),
which import the `ZkHidden` headlines.
-/

@[expose] public section

namespace FlockSoundness.Accounting

variable {wideLeaf : Bool}

/-! ## The hash-sum coefficient -/

/-- `⌊√n⌋` by bisection on `[lo, hi)`, within `f` steps. Nothing rests on it being right: `m1Ks_all` checks each
square it is used at. -/
def sqrtBis (n : ℕ) : ℕ → ℕ → ℕ → ℕ
  | 0, lo, _ => lo
  | f + 1, lo, hi =>
    if hi ≤ lo + 1 then lo
    else if (lo + hi) / 2 * ((lo + hi) / 2) ≤ n then sqrtBis n f ((lo + hi) / 2) hi
    else sqrtBis n f lo ((lo + hi) / 2)

end FlockSoundness.Accounting

namespace FlockSoundness.Discharge.ZkLink

open Game Model Accounting Spot.Sampling ZkSession StrictCR
open scoped ENNReal

variable {F : Type} [Field F] [Fintype F] [DecidableEq F] {mPts : ℕ}

/-- **A `--zk` table's knowledge bound at Flock ZK's parameters, as one number**: `ksStrictZm1` with the hash sum
at its largest coefficient (`hashSum_le_m1`) and the inner code's length at `2^13`. -/
noncomputable def ksClosedM1 (k : ℕ) (qF qS : ℝ) : ℝ :=
  2 ^ (-(203 : ℝ)) + (2 * 25638 + 2 ^ 14 + 2) * Real.sqrt (qF ^ 2 / 2 ^ (513 : ℝ)) + qS ^ 2 / 2 ^ (513 : ℝ) +
    Real.sqrt (k * (qF ^ 2 / 2 ^ (513 : ℝ)))

/-- The level-0 rate cap at Flock ZK's parameters: `2^23/(ek)`, level 0's codeword at most `2^23` long. -/
noncomputable def rateM1 (k : ℕ) : ℝ := 2 ^ 23 / (Real.exp 1 * k)

/-! ## `Q_s`, the expected number of commit strings the sampled proof blocks read -/

end FlockSoundness.Discharge.ZkLink

namespace FlockSoundness.Discharge.ZkHidden

open Game Spot.Sampling Exec Flock FlockLevel3 Model Accounting Refine ZkSession ZkLink ZkExec Composed

variable {n mPts : ℕ}

variable {I : Inputs} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
  (y₀ : SampleSetupZHJ (inputsJ I 0) mPts S₀ d₀) {Reg : Type}

end FlockSoundness.Discharge.ZkHidden
