module

public import Proofs.Flock.Soundness.Discharge.ZkHidden.Statements
public import Proofs.Flock.Soundness.Discharge.ZkExec.LiveStatements
public import Proofs.Flock.Soundness.Discharge.HiddenDec.Statements
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt

/-!
# `RunsZC` at the hidden-output `--zk` tables (statement)

ZkExec's `RunsZCLive` (`ZkExec/LiveStatements.lean`) for the sessions the prover's `--zk` writes, whose tables
`Stmt.setupTables` sets up with `Stmt.setupHidden` (hidden outputs, the default `--statement`). Its `opnAtZ` reads the
statement `Stmt.setupH` sets up, so the hidden case is a different `RunsZC` instance:

* `opnAtZH`: `opnAtZ` at the statement `Stmt.setupHidden` sets up at the record's sample under the `--zk` tags and coin
  tree.
* `RunsZCLiveHI`: `RunsZCLive` with `tablesAtZKH`, `opnAtZH`, `TablesAtZKHLiveI`, `TagsOkH` and `SampleOkZKH` for
  `tablesAtZK`, `opnAtZ`, `TablesAtZKLive`, `TagsOk` and `SampleOkZK`, on every circuit `HmOut.parse` accepts at the
  `--zk` tags. The decoder `decAtZ`, the empty tail and `capOfRoot` are the public case's.
* `RunsZCLiveHSt`, `RunsZCLiveHISt`: the same at the statement circuit (`HmOut.parseStmt`, a templated file's template
  unrolled), from `TablesAtZKHLiveSt` and `TablesAtZKHLiveISt`; `RunsZCLiveHSt` where the unit returns the outputs.
-/

@[expose] public section

namespace FlockSoundness.Discharge.ZkHidden

open Flock Refine Exec Game Model DecodeZL ZkSession ZkExec

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

variable {n mPts : ℕ} {Reg : Type}

/-- **The `--zk` openings at sample `S` under hidden outputs** (`opnAtZ` with `setupHidden` for `setupH`): `opnZ` at the
statement setupHidden sets up at the record's sample `dj S` under the `--zk` tags and coin tree, and its `--zk` schedule,
on the first two proof files. -/
noncomputable def opnAtZH (T : Finset (Fin n) → Reg → TabZK 𝔽 mPts) (I : Inputs) (dj : Finset (Fin n) → Lean.Json)
    (S : Finset (Fin n)) (R : Reg) (proofs : Array ByteArray) : ZkLink.OpensZ 𝔽 Dig (List UInt8) :=
  match Stmt.setupHidden (zkTags I) I.circuitFile I.publicFile (zkCoins I) I.tables (some (dj S)) I.partition
      I.program with
  | .ok st => match Zk.schedule st.wideLeaf st.m with
    | .ok (sch, _) => opnZ (Setup.ofCircuit st) sch (nXZ T S R) #[proofs[0]!, proofs[1]!]
    | .error _ => default
  | .error _ => default

/-- **The hidden-output `RunsZCLive` wherever the outputs are returned**, from `TablesAtZKHLiveI`. -/
def RunsZCLiveHI : Prop :=
  ∀ {n mPts : ℕ} {Reg : Type} (I : Inputs) (_ht : I.tags.typed = false) {c : Flock.Circuit}
    (_hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (y₀ : SampleSetupZKH I mPts S₀ (dj S₀))
    (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript) (L : Spot.Sampling.SamplingDesign n),
    TablesAtZKHLiveI (n := n) (mPts := mPts) (Reg := Reg) → HiddenDec.TagsOkH I → ZkLink.RecordsLiveZK I dj wr →
    (∀ ω, SampleOkZKH I dj mPts (L.draw ω)) →
    RunsOnZC (K := 𝕂) execArith hs512 (tablesAtZKH (Reg := Reg) dj y₀) I wr
      (opnAtZH (tablesAtZKH dj y₀) I dj) (decAtZ (tablesAtZKH dj y₀)) (fun _ _ _ _ => [])
      (fun _ _ b _ => capOfRoot b) L

/-- **The hidden-output `RunsZCLive` at the statement circuit**: on every circuit `HmOut.parseStmt` reads at the
`--zk` tags (a templated file's template unrolled) whose unit returns the outputs, from `TablesAtZKHLiveSt`. -/
def RunsZCLiveHSt : Prop :=
  ∀ {n mPts : ℕ} {Reg : Type} (I : Inputs) (_ht : I.tags.typed = false) {c : Flock.Circuit}
    (_hc : HmOut.parseStmt (zkTags I) I.circuitFile I.tables = .ok c) (_ho : c.outNet = c.unitNet)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (y₀ : SampleSetupZKH I mPts S₀ (dj S₀))
    (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript) (L : Spot.Sampling.SamplingDesign n),
    TablesAtZKHLiveSt (n := n) (mPts := mPts) (Reg := Reg) → HiddenDec.TagsOkH I → ZkLink.RecordsLiveZK I dj wr →
    (∀ ω, SampleOkZKH I dj mPts (L.draw ω)) →
    RunsOnZC (K := 𝕂) execArith hs512 (tablesAtZKH (Reg := Reg) dj y₀) I wr
      (opnAtZH (tablesAtZKH dj y₀) I dj) (decAtZ (tablesAtZKH dj y₀)) (fun _ _ _ _ => [])
      (fun _ _ b _ => capOfRoot b) L

/-- **`RunsZCLiveHI` at the statement circuit**, from `TablesAtZKHLiveISt`. -/
def RunsZCLiveHISt : Prop :=
  ∀ {n mPts : ℕ} {Reg : Type} (I : Inputs) (_ht : I.tags.typed = false) {c : Flock.Circuit}
    (_hc : HmOut.parseStmt (zkTags I) I.circuitFile I.tables = .ok c)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (y₀ : SampleSetupZKH I mPts S₀ (dj S₀))
    (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript) (L : Spot.Sampling.SamplingDesign n),
    TablesAtZKHLiveISt (n := n) (mPts := mPts) (Reg := Reg) → HiddenDec.TagsOkH I → ZkLink.RecordsLiveZK I dj wr →
    (∀ ω, SampleOkZKH I dj mPts (L.draw ω)) →
    RunsOnZC (K := 𝕂) execArith hs512 (tablesAtZKH (Reg := Reg) dj y₀) I wr
      (opnAtZH (tablesAtZKH dj y₀) I dj) (decAtZ (tablesAtZKH dj y₀)) (fun _ _ _ _ => [])
      (fun _ _ b _ => capOfRoot b) L

end FlockSoundness.Discharge.ZkHidden
