import Proofs.Flock.Soundness.Discharge.ZkHidden.Statements
import Proofs.Flock.Soundness.Discharge.ZkExec.LiveStatements
import Proofs.Flock.Soundness.Discharge.HiddenDec.Statements

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
* `RunsZCLiveHSt`, `RunsZCLiveHISt`: `RunsZCLiveH` and `RunsZCLiveHI` at the statement circuit (`HmOut.parseStmt`, a
  templated file's template unrolled), from `TablesAtZKHLiveSt` and `TablesAtZKHLiveISt`.
-/

namespace FlockSoundness.Discharge.ZkHidden

open Flock Refine Exec Game Model DecodeZL ZkSession ZkExec

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

variable {n mPts : ℕ} {Reg : Type}

/-- **The `--zk` openings at draw `S` under hidden outputs** (`opnAtZ` with `setupHidden` for `setupH`): `opnZ` at the
statement setupHidden sets up at the record's draw `dj S` under the `--zk` tags and coin tree, and its `--zk` schedule,
on the first two proof files. -/
noncomputable def opnAtZH (T : Finset (Fin n) → Reg → TabZK 𝔽 mPts) (I : Inputs) (dj : Finset (Fin n) → Lean.Json)
    (S : Finset (Fin n)) (R : Reg) (proofs : Array ByteArray) : ZkLink.OpensZ 𝔽 Dig (List UInt8) :=
  match Stmt.setupHidden (zkTags I) I.circuitFile I.publicFile (zkCoins I) I.tables (some (dj S)) I.partition
      I.program with
  | .ok st => match Zk.schedule st.m with
    | .ok (sch, _) => opnZ (Setup.ofCircuit st) sch (nXZ T S R) #[proofs[0]!, proofs[1]!]
    | .error _ => default
  | .error _ => default

/-- **`RunsZC` at the live hidden-output `--zk` tables, at the law's draws** (`RunsZCLive` under hidden outputs). Given
that wherever flock-verify accepts a hidden-output session the `--zk` table is the executable's (`TablesAtZKHLive`), on
untyped one-table inputs under the tags `TagsOkH` fixes, whose circuit file `HmOut.parse` accepts (at the `--zk` tags)
as a circuit whose unit returns the outputs, with `wr` the challenger's record of the live `--zk` session and each of
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

/-- **`RunsZCLiveH` wherever the outputs are returned**, from `TablesAtZKHLiveI`. -/
def RunsZCLiveHI : Prop :=
  ∀ {n mPts : ℕ} {Reg : Type} (I : Inputs) (_ht : I.tags.typed = false) {c : Flock.Circuit}
    (_hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (y₀ : DrawSetupZKH I mPts S₀ (dj S₀))
    (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript) (L : Audit.Law n),
    TablesAtZKHLiveI (n := n) (mPts := mPts) (Reg := Reg) → HiddenDec.TagsOkH I → ZkLink.RecordsLiveZK I dj wr →
    (∀ ω, DrawOkZKH I dj mPts (L.draw ω)) →
    RunsOnZC (K := 𝕂) execArith hs512 (tablesAtZKH (Reg := Reg) dj y₀) I wr
      (opnAtZH (tablesAtZKH dj y₀) I dj) (decAtZ (tablesAtZKH dj y₀)) (fun _ _ _ _ => [])
      (fun _ _ b _ => capOfRoot b) L

/-- **`RunsZCLiveH` at the statement circuit**: on every circuit `HmOut.parseStmt` reads at the `--zk` tags (a templated
file's template unrolled) whose unit returns the outputs, from `TablesAtZKHLiveSt`. -/
def RunsZCLiveHSt : Prop :=
  ∀ {n mPts : ℕ} {Reg : Type} (I : Inputs) (_ht : I.tags.typed = false) {c : Flock.Circuit}
    (_hc : HmOut.parseStmt (zkTags I) I.circuitFile I.tables = .ok c) (_ho : c.outNet = c.unitNet)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (y₀ : DrawSetupZKH I mPts S₀ (dj S₀))
    (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript) (L : Audit.Law n),
    TablesAtZKHLiveSt (n := n) (mPts := mPts) (Reg := Reg) → HiddenDec.TagsOkH I → ZkLink.RecordsLiveZK I dj wr →
    (∀ ω, DrawOkZKH I dj mPts (L.draw ω)) →
    RunsOnZC (K := 𝕂) execArith hs512 (tablesAtZKH (Reg := Reg) dj y₀) I wr
      (opnAtZH (tablesAtZKH dj y₀) I dj) (decAtZ (tablesAtZKH dj y₀)) (fun _ _ _ _ => [])
      (fun _ _ b _ => capOfRoot b) L

/-- **`RunsZCLiveHI` at the statement circuit**, from `TablesAtZKHLiveISt`. -/
def RunsZCLiveHISt : Prop :=
  ∀ {n mPts : ℕ} {Reg : Type} (I : Inputs) (_ht : I.tags.typed = false) {c : Flock.Circuit}
    (_hc : HmOut.parseStmt (zkTags I) I.circuitFile I.tables = .ok c)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (y₀ : DrawSetupZKH I mPts S₀ (dj S₀))
    (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript) (L : Audit.Law n),
    TablesAtZKHLiveISt (n := n) (mPts := mPts) (Reg := Reg) → HiddenDec.TagsOkH I → ZkLink.RecordsLiveZK I dj wr →
    (∀ ω, DrawOkZKH I dj mPts (L.draw ω)) →
    RunsOnZC (K := 𝕂) execArith hs512 (tablesAtZKH (Reg := Reg) dj y₀) I wr
      (opnAtZH (tablesAtZKH dj y₀) I dj) (decAtZ (tablesAtZKH dj y₀)) (fun _ _ _ _ => [])
      (fun _ _ b _ => capOfRoot b) L

end FlockSoundness.Discharge.ZkHidden
