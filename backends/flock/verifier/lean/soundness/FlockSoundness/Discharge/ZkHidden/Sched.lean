import FlockSoundness.Discharge.ZkHidden.Statements
import FlockLevel3.ArithFacts
import FlockSoundness.Protocol.Discharge.ZkHidden.Sched

/-!
# `ZkSchedOK`: `Zk.schedule`'s facts at a statement the `--zk` shape check passes

* The schedule half reads only `m`, which `Zk.schedule` accepts only from 25 to 35 (`schedule_range`); at each of
  those `zkSchedFacts` is checked by the kernel (`zkSchedFacts_all`, `decide +kernel`).
* `PadOK`'s radius: `100·(2^cols + t) ≤ 98·2^logLen` puts the padded rate at most `0.98 < 0.995²`.
* The inner code: `len` is at most `400` (`m ≤ 35`, `kLog ≤ m`, `k0 ≤ 6`), so `innerN ≤ 2^13`; it is a power of two,
  so `w.lo mod innerN` is uniform (level3's `lo_balanced`).
-/

namespace FlockSoundness.Discharge.ZkHidden

open Flock FlockLevel3 Game Model Refine ZkLink ZkExec Accounting
open scoped ENNReal NNReal

local notation "𝔽" => FlockLevel3.GF128

end FlockSoundness.Discharge.ZkHidden
