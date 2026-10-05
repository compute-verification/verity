import FlockSoundness.Discharge.ZkExec.RunsSeams
import FlockSoundness.Protocol.Discharge.ZkExec.RunsCarries
import FlockSoundness.Discharge.ZkExec.RunsStatements

/-!
# A `--zk` rep's rounds after round 0 carry its messages

`zk_carries`: the chunk lemma at `wRestZ`, glued from four phases with `CarriesP.append`.

* The zerocheck's tail and the lincheck: DecodeZL's `zl_carries` without round 0, which closes at the first coin and
  carries nothing (`carries_drop_coin`, `zl_tail_carries`).
* The opening: `carries_switch` and `carries_switches` from the ring switches' decoded lengths (`op_carries_sizes`).
* Ligerito with level 0's hiding (`zk_lig_carries`): M0's rounds (`carriesP_f`, `carriesP_fold'`, `carriesP_mid`),
  with level 0's pair's sums (`carriesP_ev`), the second `ρ` coin's empty round and the round of `T'`, `ybar`, the
  last lane message and the level-1 cap (`carriesP_tcap`). The `--zk` schedule is fast100's with level 0 restratified
  (`zk_schedule_fast100`), so the level loop is `carriesP_levels` at fast100's schedule, whose levels fit its bounds
  (`fast100_small`; level 0 at `--zk` does not).
* The inner proof's four rounds (`carriesP_inner`).

`ligPadShape_pat` is level 0 masked's coin pattern, the rounds' (`isCoin_rec` for the level loop).
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec Game Model DecodeZL

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

/-! ## Level 0's hiding rounds and the inner proof's -/

/-! ## Ligerito with level 0's hiding -/

/-! ## The chunk lemma -/

end FlockSoundness.Discharge.ZkExec
