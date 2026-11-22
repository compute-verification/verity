module

public import Proofs.Flock.Soundness.Discharge.ZkLink.M1
public import Proofs.Flock.Soundness.Discharge.ZkLink.StrictZ
public import Proofs.Flock.Soundness.Discharge.ZkLink.Link
public import Proofs.Flock.Soundness.Discharge.FailClosed.Numeric
public import Proofs.Flock.Soundness.Instance

/-!
# The knowledge term at Flock ZK's parameters, in closed form

* `ksClosedM1 k q_F q_S`: $2^{-203}+(2\cdot25638+2^{14}+2)\sqrt{q_F^2/2^{513}}+q_S^2/2^{513}+\sqrt{k\,q_F^2/2^{513}}$,
  the hash sum at its largest coefficient and the inner code's length at `2^13`, which `Bridge.deltaPub` reads over
  `1 − ρ`;
* `rateM1 k = 2^23/(ek)`: the level-0 rate cap there, level 0's codeword at most `2^23` long.
-/

@[expose] public section

namespace FlockSoundness.Discharge.ZkLink

/-- **A `--zk` table's knowledge bound at Flock ZK's parameters, as one number**: `ksStrictZm1` with the hash sum
at its largest coefficient and the inner code's length at `2^13`. -/
noncomputable def ksClosedM1 (k : ℕ) (qF qS : ℝ) : ℝ :=
  2 ^ (-(203 : ℝ)) + (2 * 25638 + 2 ^ 14 + 2) * Real.sqrt (qF ^ 2 / 2 ^ (513 : ℝ)) + qS ^ 2 / 2 ^ (513 : ℝ) +
    Real.sqrt (k * (qF ^ 2 / 2 ^ (513 : ℝ)))

/-- The level-0 rate cap at Flock ZK's parameters: `2^23/(ek)`, level 0's codeword at most `2^23` long. -/
noncomputable def rateM1 (k : ℕ) : ℝ := 2 ^ 23 / (Real.exp 1 * k)

end FlockSoundness.Discharge.ZkLink
