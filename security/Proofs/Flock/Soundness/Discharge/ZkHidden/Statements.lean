module

public import Proofs.Flock.Soundness.Discharge.ZkHidden.Defs
public import Proofs.Flock.Soundness.Discharge.HiddenExec.Placed
public import Proofs.Flock.Soundness.Discharge.HiddenExec.Defs
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

/-! The statements of verity-verify-proof `--zk`'s acceptance and its tables under hidden outputs. -/

@[expose] public section

namespace FlockSoundness.Discharge.ZkHidden

open Flock FlockLevel3 Game Model Refine Exec DecodeZL ZkSession ZkExec ZkLink HiddenExec
open scoped ENNReal NNReal

local notation "𝔽" => FlockLevel3.GF128

def AcceptsZKSetupOne : Prop :=
  ∀ (I : Inputs) (t : Exec.Transcript), AcceptsZK I t →
    ∃ spec sts, setupOf (zkInputs I) t.record = .ok (spec, sts) ∧
      Flock.Zk.verify spec (sts.map Flock.Setup.ofCircuit) t.record t.proofs = .ok () ∧ 0 < sts.size ∧
      ∀ (j : ℕ) (hj : j < sts.size), Flock.Zk.shapeOk sts[j] = .ok () ∧ ∃ d,
        setupOne (tagsAt (zkTags I) I.count j) I.circuitFile I.publicFile (zkCoins I) I.tables d I.partition
          I.program = .ok sts[j] ∧ (I.count ≤ 1 → d = recordDraw t.record)

def AcceptsZKOneAny : Prop :=
  ∀ (I : Inputs) (t : Exec.Transcript), I.count ≤ 1 → AcceptsZK I t →
    ∃ st, setupOne (zkTags I) I.circuitFile I.publicFile (zkCoins I) I.tables (recordDraw t.record) I.partition
        I.program = .ok st ∧ Flock.Zk.shapeOk st = .ok () ∧
      Flock.Zk.verify st.spec #[Flock.Setup.ofCircuit st] t.record t.proofs = .ok ()

def AcceptsZKRetainedAny : Prop :=
  ∀ (I : Inputs) (t : Exec.Transcript), I.tags.retainRounds = true → AcceptsZK I t →
    ∃ spec sts sess, setupOf (zkInputs I) t.record = .ok (spec, sts) ∧ Record.decode spec t.record = .ok sess ∧
      ∀ s ∈ sess.streams.toList, AllRetained s.rounds

def AcceptsZKAcceptedH : Prop :=
  ∀ (I : Inputs) (t : Exec.Transcript), I.tags.hiddenOutputs = true → I.tags.typed = false → ∀ {c : Flock.Circuit},
    HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c → Hidden.Any.RowsFitS c →
    I.count ≤ 1 → AcceptsZK I t →
    ∃ st, Stmt.setupHidden (zkTags I) I.circuitFile I.publicFile (zkCoins I) I.tables (recordDraw t.record)
        I.partition I.program = .ok st ∧ Flock.Zk.shapeOk st = .ok () ∧
      ∀ regs : List (Model.Region st.c.kLog (st.m - st.c.kLog)),
        ∃ (T : Model.Statement) (a : AcceptedH T), a.st = st ∧ HEq a.regs regs ∧ a.tags = zkTags I ∧
          a.cf = I.circuitFile ∧ a.pf = I.publicFile ∧ a.coins = zkCoins I ∧ a.tables = I.tables ∧
          a.draw = recordDraw t.record

/-- It reads `st` only through `m`, `kLog` and `shapeOk`, so it is the same for `setupH`'s and `setupHidden`'s
statements. -/
def ZkSchedOK : Prop :=
  ∀ (st : Stmt) (sch : Flock.Schedule) (tPad : ℕ), StmtWF st → RegionsWF st → 13 ≤ st.m →
    Zk.shapeOk st = .ok () → Zk.schedule st.wideLeaf st.m = .ok (sch, tPad) →
    (2 ≤ (schedOf sch).levels.length ∧ ((schedOf sch).levels.headD default).lanes = (schedOf sch).k0 ∧
      (schedOf sch).m - 7 = (schedOf sch).k0 + ((schedOf sch).levels.headD default).logCols ∧
      levelsChained (schedOf sch).levels (schedOf sch).ext = true) ∧
    (schedOf sch).m = st.m ∧ Accounting.PadOK (schedOf sch).level₀ tPad ∧ innerN (layOf st sch) < 2 ^ 64 ∧
    (layOf st sch).len + Zk.K_PAD ≤ innerN (layOf st sch) ∧
    ∀ (hN : 0 < innerN (layOf st sch)) (D : Finset (Fin (innerN (layOf st sch)))),
      prCoin (fun w => innerPos _ hN w ∈ D) ≤ (D.card : ℝ≥0∞) / innerN (layOf st sch)

variable {n : ℕ} {mPts : ℕ} {Reg : Type}

def TablesAtZKHLive : Prop :=
  ∀ (I : Inputs) (_ht : I.tags.typed = false) {c : Flock.Circuit}
    (_hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (_ho : c.outNet = c.unitNet)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    (y₀ : SampleSetupZKH I mPts S₀ (dj S₀)) (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript),
    I.tags.hiddenOutputs = true → I.count ≤ 1 → RecordsLiveZK I dj wr →
    ∀ S R o, SampleOkZKH I dj mPts S → AcceptsZK I (wr S R o) →
      ∃ (st : Stmt) (sch : Flock.Schedule) (tPad : ℕ), StmtWF st ∧ RegionsWF st ∧
        Stmt.setupHidden (zkTags I) I.circuitFile I.publicFile (zkCoins I) I.tables (some (dj S)) I.partition
          I.program = .ok st ∧ Zk.schedule st.wideLeaf st.m = .ok (sch, tPad) ∧
        ∃ y : SampleSetupZKH I mPts S (dj S), y.x.st = st ∧ y.sch = sch ∧ y.tPad = tPad ∧
          tablesAtZKH (Reg := Reg) dj y₀ S R = y.table ∧
          ∀ points : Array (Array F128), points.size = 2 → (∀ p ∈ points.toList, p.size = mPts) →
            ExtraRealizes (Setup.ofCircuit st) y.table.S points
              (Model.extraClaims y.table.S y.table.ptLocal mPts (ptsOf mPts points))

def TablesAtZKHLiveI : Prop :=
  ∀ (I : Inputs) (_ht : I.tags.typed = false) {c : Flock.Circuit}
    (_hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    (y₀ : SampleSetupZKH I mPts S₀ (dj S₀)) (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript),
    I.tags.hiddenOutputs = true → I.count ≤ 1 → RecordsLiveZK I dj wr →
    ∀ S R o, SampleOkZKH I dj mPts S → AcceptsZK I (wr S R o) →
      ∃ (st : Stmt) (sch : Flock.Schedule) (tPad : ℕ), StmtWF st ∧ RegionsWF st ∧
        Stmt.setupHidden (zkTags I) I.circuitFile I.publicFile (zkCoins I) I.tables (some (dj S)) I.partition
          I.program = .ok st ∧ Zk.schedule st.wideLeaf st.m = .ok (sch, tPad) ∧
        ∃ y : SampleSetupZKH I mPts S (dj S), y.x.st = st ∧ y.sch = sch ∧ y.tPad = tPad ∧
          tablesAtZKH (Reg := Reg) dj y₀ S R = y.table ∧
          ∀ points : Array (Array F128), points.size = 2 → (∀ p ∈ points.toList, p.size = mPts) →
            ExtraRealizes (Setup.ofCircuit st) y.table.S points
              (Model.extraClaims y.table.S y.table.ptLocal mPts (ptsOf mPts points))

def TablesAtZKHLiveSt : Prop :=
  ∀ (I : Inputs) (_ht : I.tags.typed = false) {c : Flock.Circuit}
    (_hc : HmOut.parseStmt (zkTags I) I.circuitFile I.tables = .ok c) (_ho : c.outNet = c.unitNet)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    (y₀ : SampleSetupZKH I mPts S₀ (dj S₀)) (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript),
    I.tags.hiddenOutputs = true → I.count ≤ 1 → RecordsLiveZK I dj wr →
    ∀ S R o, SampleOkZKH I dj mPts S → AcceptsZK I (wr S R o) →
      ∃ (st : Stmt) (sch : Flock.Schedule) (tPad : ℕ), StmtWF st ∧ RegionsWF st ∧
        Stmt.setupHidden (zkTags I) I.circuitFile I.publicFile (zkCoins I) I.tables (some (dj S)) I.partition
          I.program = .ok st ∧ Zk.schedule st.wideLeaf st.m = .ok (sch, tPad) ∧
        ∃ y : SampleSetupZKH I mPts S (dj S), y.x.st = st ∧ y.sch = sch ∧ y.tPad = tPad ∧
          tablesAtZKH (Reg := Reg) dj y₀ S R = y.table ∧
          ∀ points : Array (Array F128), points.size = 2 → (∀ p ∈ points.toList, p.size = mPts) →
            ExtraRealizes (Setup.ofCircuit st) y.table.S points
              (Model.extraClaims y.table.S y.table.ptLocal mPts (ptsOf mPts points))

def TablesAtZKHLiveISt : Prop :=
  ∀ (I : Inputs) (_ht : I.tags.typed = false) {c : Flock.Circuit}
    (_hc : HmOut.parseStmt (zkTags I) I.circuitFile I.tables = .ok c)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    (y₀ : SampleSetupZKH I mPts S₀ (dj S₀)) (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript),
    I.tags.hiddenOutputs = true → I.count ≤ 1 → RecordsLiveZK I dj wr →
    ∀ S R o, SampleOkZKH I dj mPts S → AcceptsZK I (wr S R o) →
      ∃ (st : Stmt) (sch : Flock.Schedule) (tPad : ℕ), StmtWF st ∧ RegionsWF st ∧
        Stmt.setupHidden (zkTags I) I.circuitFile I.publicFile (zkCoins I) I.tables (some (dj S)) I.partition
          I.program = .ok st ∧ Zk.schedule st.wideLeaf st.m = .ok (sch, tPad) ∧
        ∃ y : SampleSetupZKH I mPts S (dj S), y.x.st = st ∧ y.sch = sch ∧ y.tPad = tPad ∧
          tablesAtZKH (Reg := Reg) dj y₀ S R = y.table ∧
          ∀ points : Array (Array F128), points.size = 2 → (∀ p ∈ points.toList, p.size = mPts) →
            ExtraRealizes (Setup.ofCircuit st) y.table.S points
              (Model.extraClaims y.table.S y.table.ptLocal mPts (ptsOf mPts points))

def TablesAtZKHLiveU : Prop :=
  ∀ (I : Inputs) (_ht : I.tags.typed = false)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    (y₀ : SampleSetupZKH I mPts S₀ (dj S₀)) (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript),
    I.tags.hiddenOutputs = true → I.count ≤ 1 → RecordsLiveZK I dj wr →
    ∀ S R o, SampleOkZKH I dj mPts S → AcceptsZK I (wr S R o) →
      ∃ (st : Stmt) (sch : Flock.Schedule) (tPad : ℕ), StmtWF st ∧ RegionsWF st ∧
        Stmt.setupHidden (zkTags I) I.circuitFile I.publicFile (zkCoins I) I.tables (some (dj S)) I.partition
          I.program = .ok st ∧ Zk.schedule st.wideLeaf st.m = .ok (sch, tPad) ∧
        ∃ y : SampleSetupZKH I mPts S (dj S), y.x.st = st ∧ y.sch = sch ∧ y.tPad = tPad ∧
          tablesAtZKH (Reg := Reg) dj y₀ S R = y.table ∧
          ∀ points : Array (Array F128), points.size = 2 → (∀ p ∈ points.toList, p.size = mPts) →
            ExtraRealizes (Setup.ofCircuit st) y.table.S points
              (Model.extraClaims y.table.S y.table.ptLocal mPts (ptsOf mPts points))

end FlockSoundness.Discharge.ZkHidden
