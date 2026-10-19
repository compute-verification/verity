import Proofs.Flock.Claims.Complete.ZkTables.Level0
import Proofs.Flock.Claims.Complete.Zk.Level0Run
import Proofs.Flock.Claims.Complete.Zk.Level0Lanes
import Proofs.Flock.Claims.Complete.Zk.Level0Shape
import Proofs.Flock.Claims.Complete.Zk.Count
import Proofs.Flock.Claims.Complete.Replay.LigeritoGoal

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
