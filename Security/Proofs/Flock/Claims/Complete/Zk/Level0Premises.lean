import Proofs.Flock.Claims.Complete.ZkTables.Level0
import Proofs.Flock.Claims.Complete.Zk.Level0Run
import Proofs.Flock.Claims.Complete.Zk.Count
import Proofs.Flock.Claims.Complete.Replay.LigeritoGoal

/-!
# Level 0's masked Ligerito: its premises, proved

`ZkTables.Level0` states what the table-level forms need of level 0's masked Ligerito beyond plain's transfer. Proved
here:

* `proverRuns : ProverRuns`: the masked prover's Ligerito returns wherever its pair coins are independent, from
  `ligeritoPZk_ret` at a `--zk` schedule, which keeps fast100's `n ≠ 0` and `k_0 ≥ 1` (`zkSchedule_n`).
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

end FlockComplete.L0
