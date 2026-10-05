import FlockSoundness.Discharge.Integrate.Bridge
import FlockSoundness.Discharge.DecodeZL.Statements
import FlockSoundness.Discharge.ZkExec.Statements
import FlockSoundness.Discharge.ZkSession.Session
import FlockSoundness.Discharge.ZkExec.Setup
import FlockSoundness.Discharge.ZkExec.Verify
import FlockSoundness.Discharge.Hidden.Wide.Block
import FlockSoundness.Discharge.ZkLink.Sched
import FlockSoundness.Protocol.Discharge.ZkLink.TablesAt
import FlockSoundness.Discharge.Hidden.Facts

/-!
# The `--zk` table at each draw, from flock-verify's setup (statements)

#828's `tablesAt` is the plain headline's table at draw `S`: `stmtOf` of the statement setupH sets up at the coin
server's draw `dj S`, wherever flock-verify accepts. Here is its `--zk` counterpart, `tablesAtZK`, a `TabZK` built from
what `verify --zk` sets up at the draw (`zkInputs I`):

* the statement `stmtOf st` at setupH's statement `st` and its regions, link layout at `PT_LOCAL`;
* `Zk.schedule st.m`'s schedule and `t_pad`, the padding constant `PAD_C`, two extra lanes;
* the pads at `PadLayout.new st.m st.kLog k0`'s positions (`padsOfLay`, `padsAOfLay`, `tripleOfLay`);
* the inner code `verifyInner` reads: `innerN` columns, column `j` at the point `j + 1` (`codeAt`), `K_PAD` padding
  symbols, `Q` queries at `w.lo mod innerN`, and the unique-decoding radius.

`DrawSetupZK` is a draw `verify --zk` sets up (`DrawSetup` at `zkInputs I`) with its schedule and the facts the table
needs of it; at a draw with none the table is a reference setup's statement with `contra`'s regions, which no witness
satisfies (as `tableAt`'s `refused`).

* `RecordsLiveZK`, `DrawOkZK`: #862's `RecordsLive` and `DrawOk` at `--zk` (`AcceptsZK`, `zkInputs I`).
* `TablesAtZKLive`: wherever flock-verify accepts what `wr` writes, the draw's `--zk` table is the executable's: its
  statement is `stmtOf` of setupH's, its dimensions and inner code are `Zk.verify`'s, and its extra claims at the
  record's link points are the executable's (`ExtraRealizes`, as `ofCircuit_extra`).
* `tablesAtZK_live`: its proof, with `0 < g` from setupH (`setupH_g_pos`, D9), the schedule's shape and padding from
  the m check (`schedule_zk`), the inner code's length (`innerN_pow`) and its query positions' uniformity
  (`prCoin_innerPos_le`), and the table the same at every `DrawSetupZK` of a draw (`tableZK_eq`).
-/

namespace FlockSoundness.Discharge.ZkLink

open Flock FlockLevel3 Game Model Refine Exec Integrate DecodeZL ZkSession ZkExec
open scoped ENNReal NNReal

local notation "𝔽" => FlockLevel3.GF128

/-! ## The pads and the inner code of a pads layout -/

/-! ## The inner code's facts -/

/-! ## A draw `verify --zk` sets up -/

variable {n : ℕ} (I : Inputs) (mPts : ℕ)

variable {I mPts}

namespace DrawSetupZK

variable {S : Finset (Fin n)} {d : Lean.Json} (y : DrawSetupZK I mPts S d)

end DrawSetupZK

section

variable {S : Finset (Fin n)} {d : Lean.Json}

end

variable (ht : I.tags.typed = false) {Reg : Type} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
  (y₀ : DrawSetupZK I mPts S₀ (dj S₀))

/-! ## The live session's record -/

/-! ## The statement -/

end FlockSoundness.Discharge.ZkLink
