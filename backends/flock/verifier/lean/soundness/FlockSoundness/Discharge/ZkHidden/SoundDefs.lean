import FlockSoundness.Discharge.ZkHidden.Statements
import FlockSoundness.Discharge.ZkExec.LiveStatements
import FlockSoundness.Discharge.HiddenDec.Statements
import FlockSoundness.Protocol.Discharge.ZkHidden.SoundDefs

/-!
# `RunsZC` at the hidden-output `--zk` tables (statement)

ZkExec's `RunsZCLive` (`ZkExec/LiveStatements.lean`) for the sessions production `--zk` writes, whose tables
`Stmt.setupTables` sets up with `Stmt.setupHidden` (hidden outputs, the default `--statement`). Its `opnAtZ` reads the
statement `Stmt.setupH` sets up, so the hidden case is a different `RunsZC` instance:

* `opnAtZH`: `opnAtZ` at the statement `Stmt.setupHidden` sets up at the record's draw under the `--zk` tags and coin
  tree.
* `RunsZCLiveH`: `RunsZCLive` with `tablesAtZKH`, `opnAtZH`, `TablesAtZKHLive`, `TagsOkH` and `DrawOkZKH` for
  `tablesAtZK`, `opnAtZ`, `TablesAtZKLive`, `TagsOk` and `DrawOkZK`, and `TablesAtZKHLive`'s hypotheses on the circuit
  (`HmOut.parse` accepts it at the `--zk` tags; its unit returns the outputs). The decoder `decAtZ`, the empty tail and
  `capOfRoot` are the public case's.
-/

namespace FlockSoundness.Discharge.ZkHidden

open Flock Refine Exec Game Model DecodeZL ZkSession ZkExec

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

variable {n mPts : ℕ} {Reg : Type}

/-- **`RunsZC` at the live hidden-output `--zk` tables, at the law's draws** (`RunsZCLive` under hidden outputs). Given
that wherever flock-verify accepts a hidden-output session the `--zk` table is the executable's (`TablesAtZKHLive`), on
untyped one-table inputs under the tags `TagsOkH` fixes, whose circuit file `HmOut.parse` accepts (at the `--zk` tags)
as a circuit whose unit returns the outputs, with `wr` the coin server's record of the live `--zk` session and each of
the law's draws proving its units at `mPts` link points (`DrawOkZKH` at `L.draw ω`), `RunsZC` holds at the law's draws
(`RunsOnZC`) at the tables `tablesAtZKH` under `hm96-sha512/v1` (`hs512`), with the decoder `decAtZ`, the openings
`opnAtZH`, no tail and `capOfRoot`. The reference setup `y₀` is data. -/
def RunsZCLiveH : Prop :=
  ∀ {n mPts : ℕ} {Reg : Type} (I : Inputs) (_ht : I.tags.typed = false) {c : Flock.Circuit}
    (_hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (_ho : c.outNet = c.unitNet)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (y₀ : DrawSetupZKH I mPts S₀ (dj S₀))
    (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript) (L : Audit.Law n),
    TablesAtZKHLive (n := n) (mPts := mPts) (Reg := Reg) → HiddenDec.TagsOkH I → ZkLink.RecordsLiveZK I dj wr →
    (∀ ω, DrawOkZKH I dj mPts (L.draw ω)) →
    RunsOnZC (K := 𝕂) execArith hs512 (tablesAtZKH (Reg := Reg) dj y₀) I wr
      (opnAtZH (tablesAtZKH dj y₀) I dj) (decAtZ (tablesAtZKH dj y₀)) (fun _ _ _ _ => [])
      (fun _ _ b _ => capOfRoot b) L

end FlockSoundness.Discharge.ZkHidden
