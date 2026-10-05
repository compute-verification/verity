import FlockSoundness.Discharge.ZkExec.VerdictStatements
import FlockSoundness.Discharge.ZkLink.TablesAt
import FlockSoundness.Protocol.Discharge.ZkExec.LiveStatements

/-!
# `RunsZC` at the live `--zk` tables, from `TablesAtZKLive` (statement)

`RunsZC` at [wave2-zk-link]'s live tables `tablesAtZK` (`ZkLink/TablesAt.lean`): on every live outcome flock-verify
`--zk` accepts, each rep has a `RepRun` on its rounds along `repShapeZ` and the compiled table accepts at the two runs'
ends. Its pieces are proved or frozen apart: the per-rep runs (`zk_rep_plays`, `zk_rep_run`), the verdict
(`ZkTableVerdict`) and the table being the executable's (`TablesAtZKLive`, taken here as a hypothesis).

* `nXZ`, `decAtZ`, `opnAtZ`: M0's `nX`, `decAt` and `opnAt` at `--zk`: the table's number of extra claims, the decoder
  `decZ` at the table's dimensions, and the openings `opnZ` at the statement setupH sets up at the record's draw under
  the `--zk` tags and its `--zk` schedule, on the first two proof files.
* `RunsZCLive`: given `TablesAtZKLive`, the tags (`TagsOk`), `wr` the coin server's record and its draw at each of the
  law's draws, `RunsZC` holds at the law's draws (`RunsOnZC`) at `tablesAtZK` with those, no tail and `capOfRoot`.
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec Game Model DecodeZL ZkSession

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

variable {n mPts : ℕ} {Reg : Type}

/-- **`RunsZC` at the live `--zk` tables, at the law's draws** (R11b's message half at `--zk`). Given that wherever
flock-verify accepts the `--zk` table is the executable's (`TablesAtZKLive`), on untyped one-table inputs under the tags
`TagsOk` fixes, with `wr` the coin server's record of the live `--zk` session and each of the law's draws proving its
units at `mPts` link points (`DrawOkZK` at `L.draw ω`: the link points depend on `|S|`, so no `mPts` holds at every
draw), `RunsZC` holds at the law's draws (`RunsOnZC`) at the tables `tablesAtZK` under `hm96-sha512/v1` (`hs512`), with
the decoder `decAtZ`, the openings `opnAtZ`, no tail and `capOfRoot`. The reference setup `y₀` is data. -/
def RunsZCLive : Prop :=
  ∀ {n mPts : ℕ} {Reg : Type} (I : Inputs) (_ht : I.tags.typed = false) (dj : Finset (Fin n) → Lean.Json)
    {S₀ : Finset (Fin n)} (y₀ : ZkLink.DrawSetupZK I mPts S₀ (dj S₀))
    (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript) (L : Audit.Law n),
    ZkLink.TablesAtZKLive (n := n) (mPts := mPts) (Reg := Reg) → TagsOk I → ZkLink.RecordsLiveZK I dj wr →
    (∀ ω, ZkLink.DrawOkZK I dj mPts (L.draw ω)) →
    RunsOnZC (K := 𝕂) execArith hs512 (ZkLink.tablesAtZK (Reg := Reg) dj y₀) I wr
      (opnAtZ (ZkLink.tablesAtZK dj y₀) I dj) (decAtZ (ZkLink.tablesAtZK dj y₀)) (fun _ _ _ _ => [])
      (fun _ _ b _ => capOfRoot b) L

end FlockSoundness.Discharge.ZkExec
