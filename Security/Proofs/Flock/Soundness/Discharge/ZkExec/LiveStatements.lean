import Proofs.Flock.Soundness.Discharge.ZkExec.VerdictStatements
import Proofs.Flock.Soundness.Discharge.ZkLink.TablesAt

/-!
# `RunsZC` at the live `--zk` tables, from `TablesAtZKLive` (statement)

`RunsZC` at `Discharge/ZkLink`'s live tables `tablesAtZK` (`ZkLink/TablesAt.lean`): on every live outcome flock-verify
`--zk` accepts, each rep has a `RepRun` on its rounds along `repShapeZ` and the compiled table accepts at the two runs'
ends. Its pieces are proved or frozen apart: the per-rep runs (`zk_rep_plays`, `zk_rep_run`), the verdict
(`ZkTableVerdict`) and the table being the executable's (`TablesAtZKLive`, taken here as a hypothesis).

* `nXZ`, `decAtZ`, `opnAtZ`: the non-ZK session's `nX`, `decAt` and `opnAt` at `--zk`: the table's number of extra
  claims, the decoder `decZ` at the table's dimensions, and the openings `opnZ` at the statement setupH sets up at the
  record's draw under the `--zk` tags and its `--zk` schedule, on the first two proof files.
* `RunsZCLive`: given `TablesAtZKLive`, the tags (`TagsOk`), `wr` the challenger's record and its draw at each of the
  law's draws, `RunsZC` holds at the law's draws (`RunsOnZC`) at `tablesAtZK` with those, no tail and `capOfRoot`.
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec Game Model DecodeZL ZkSession

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

variable {n mPts : ℕ} {Reg : Type}

/-- The number of the table's extra claims (two per region; the link points don't change it). -/
noncomputable def nXZ (T : Finset (Fin n) → Reg → TabZK 𝔽 mPts) (S : Finset (Fin n)) (R : Reg) : ℕ :=
  (Model.extraClaims (T S R).S (T S R).ptLocal mPts (fun _ _ => (0 : 𝔽))).length

/-- **The `--zk` decoder at draw `S`**: `decZ` on the wires after round 0 at the table's dimensions, number of claims,
schedule, padding columns and pads. -/
noncomputable def decAtZ (T : Finset (Fin n) → Reg → TabZK 𝔽 mPts) (S : Finset (Fin n)) (R : Reg) :
    ℕ → ByteArray → List Msg :=
  decZ (wRestZ (T S R).S.m (T S R).S.kLog (2 + nXZ T S R) (T S R).sch (T S R).t (T S R).np)

/-- **The `--zk` openings at draw `S`**: `opnZ` at the statement setupH sets up at the record's draw `dj S` under the
`--zk` tags and coin tree, and its `--zk` schedule, on the first two proof files. -/
noncomputable def opnAtZ (T : Finset (Fin n) → Reg → TabZK 𝔽 mPts) (I : Inputs) (dj : Finset (Fin n) → Lean.Json)
    (S : Finset (Fin n)) (R : Reg) (proofs : Array ByteArray) : ZkLink.OpensZ 𝔽 Dig (List UInt8) :=
  match Stmt.setupH (zkTags I) I.circuitFile I.publicFile (zkCoins I) I.tables (some (dj S)) I.partition I.program with
  | .ok st => match Zk.schedule st.m with
    | .ok (sch, _) => opnZ (Setup.ofCircuit st) sch (nXZ T S R) #[proofs[0]!, proofs[1]!]
    | .error _ => default
  | .error _ => default

/-- **`RunsZC` at the live `--zk` tables, at the law's draws** (R11b's message half at `--zk`). Given that wherever
flock-verify accepts the `--zk` table is the executable's (`TablesAtZKLive`), on untyped one-table inputs under the tags
`TagsOk` fixes, with `wr` the challenger's record of the live `--zk` session and each of the law's draws proving its
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
