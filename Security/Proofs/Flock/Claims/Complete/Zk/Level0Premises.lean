module

public import Proofs.Flock.Claims.Complete.Zk.Level0Lanes
public import Proofs.Flock.Claims.Complete.Zk.Level0Run
public import Proofs.Flock.Claims.Complete.Zk.Level0Shape
public import Proofs.Flock.Claims.Complete.ZkTables.Level0
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Claims.Complete.Replay.LigeritoGoal
import Proofs.Flock.Claims.Complete.Zk.Count

/-!
# Level 0's masked Ligerito: its premises, proved

`ZkTables.Level0` states what the table-level forms need of level 0's masked Ligerito beyond plain's transfer. Proved
here:

* `proverRuns : ProverRuns`: the masked prover's Ligerito returns wherever its pair coins are independent, from
  `ligeritoPZk_ret` at a `--zk` schedule, which keeps fast100's `n ≠ 0` and `k_0 ≥ 1` (`zkSchedule_n`).
* `proverShape : ProverShape`: its proof has `LgShaped`'s lengths and `t_pad` padding coefficients, from
  `ligeritoPZk_shape` at the `--zk` schedule (`zkSchedule_lig`) and hm96-SHA-512's 64-byte digests, on level 0's
  masked commitment's shape (`commitPZk_shape`).
* `lanes : Lanes`: its lane phase in closed form, from `ligeritoPZk_lanes`.
-/

@[expose] public section

namespace FlockComplete.L0

open Flock Transcript FlockComplete Replay ZkTables.Level0

/-- The `--zk` schedule keeps fast100's `n`, which is not zero. -/
theorem zkSchedule_n {m : ℕ} {v : Schedule × ℕ} (h : Zk.schedule m = .ok v) : v.1.n ≠ 0 := by
  unfold Zk.schedule at h
  obtain ⟨q0, -, h⟩ := except_ok_bind h
  cases hf : fast100 m with
  | none => rw [hf] at h; cases h
  | some sch =>
    rw [hf] at h
    cases h
    exact (fast100_ligerito hf).1

/-- **`ZkTables.Level0.ProverRuns`**: the masked prover's Ligerito returns wherever its pair coins are independent. -/
theorem proverRuns : ProverRuns := by
  intro resp h m v T cm0 tb b lay hv rep extra lanePad salt nonce t l _ _ _ hv0 hrho
  exact ligeritoPZk_ret resp h _ v.1 T cm0 tb b lay hv rep v.2 extra lanePad salt nonce (zkSchedule_n hv0)
    (zkSchedule_shape hv0).1 t l hrho

/-- **`ZkTables.Level0.ProverShape`**: the masked prover's Ligerito proof has its lengths. -/
theorem proverShape : ProverShape := by
  intro resp h m v sh T cm0 tb b lay hv rep extra lanePad salt0 salt nonce t t' l l' P zl rc h64 h25 h35 hv0 hsch
    hhl hsl ht _ hcm hrun
  have hn : ∀ l r v, (MerkleScheme.hm96Sha512.opsWith (Hashes.ofSha512 h : Hashes V)).node l r = .ok v →
      v.size = sh.hashLen := fun _ _ _ e => by
    rw [hhl]
    exact hm96_node h64 _ _ e
  have hlf : ∀ row salt v, (MerkleScheme.hm96Sha512.opsWith (Hashes.ofSha512 h : Hashes V)).leaf row salt = .ok v →
      v.size = sh.hashLen := fun _ _ _ e => by
    rw [hhl]
    exact hm96_leaf h64 _ _ e
  have hlig := zkSchedule_lig h25 h35 hv0
  have hc0 := commitPZk_shape hn hlf (hlig.cap 0 (by have := hlig.two; omega)) hcm
  exact ligeritoPZk_shape resp h hn hlf (Or.inr (by rw [hsl]; rfl)) hsch hlig T hc0 ht b lay hv rep v.2 extra lanePad
    salt nonce hrun

/-- **`ZkTables.Level0.Lanes`**: the masked prover's lane phase in closed form. -/
theorem lanes : Lanes := by
  intro resp h m v T cm0 tb b lay hv rep extra lanePad salt nonce t t' l l' P zl rc _ _ hv0 _ _ hrun
  exact ligeritoPZk_lanes resp _ _ v.1 T cm0 tb b lay hv rep v.2 extra lanePad salt nonce (zkSchedule_n hv0)
    (zkSchedule_shape hv0).1 hrun

end FlockComplete.L0
