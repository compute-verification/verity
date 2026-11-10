module

public import Proofs.Flock.SpotFlock.Facts
public import Proofs.Flock.SpotFlock.Opens
public import Proofs.Flock.SpotFlock.Sessions
public import Proofs.Flock.SpotFlock.RevealAt
public import Proofs.Flock.Claims.DeployTables

/-!
# An accepted run's checked unit's session has one table

`CheckedRun.count_le_one`: at every registration, the session statement of an accepted run's checked unit declares at
most one table (`r.ins.read.I.count ≤ 1`). Its setup makes `max 1 count` tables (`FlockDeploy.setupTables_size`), round
2 holds its tables' shapes to its template's reference statement's (`sessionOk`), and the design evaluates only where
every template's reference statement sets up in one table (`templateOk_of_two`).
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

variable {h : Held}

namespace CheckedRun

/-- **An accepted run's checked unit's session has one table.** -/
theorem count_le_one {r : CheckedRun} (hr : r.Ok h) : r.ins.read.I.count ≤ 1 := by
  obtain ⟨_, _, _, hx, -, h2, -, hok, hu, -, -, -, -, hset⟩ := hr
  have hs := sessionOk_of_stmtOkAt h (stmtOkAt_unit h hx hok hu)
  have ht := templateOk_of_two h h2 (templateOf_lt_of_checked h hx hu)
  simp only [sessionOk, Bool.and_eq_true, beq_iff_eq] at hs
  unfold templateOk at ht
  split at ht
  · simp only [Bool.and_eq_true, beq_iff_eq] at ht
    have hlen : (shapeOf (unitStmt h r.m₂ r.u)).length = 1 := by
      rw [hs.1.1.2]
      exact ht.1.2
    simp only [shapeOf, hset, List.length_map, Array.length_toList] at hlen
    have hsz : r.sts.size = max 1 r.ins.read.I.count :=
      FlockDeploy.setupTables_size (FlockDeploy.Ins.setup_read (FlockDeploy.setupOf_some hset).2)
    omega
  · cases ht

end CheckedRun

end SpotFlock
