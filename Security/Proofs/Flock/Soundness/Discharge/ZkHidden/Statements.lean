import Proofs.Flock.Soundness.Discharge.ZkHidden.Defs
import Proofs.Flock.Soundness.Discharge.HiddenExec.Placed
import Proofs.Flock.Soundness.Discharge.Hidden.Wide.FitS

/-!
# The statements: flock-verify `--zk`'s acceptance and its tables under hidden outputs

ZkExec's acceptance statements and ZkLink's `TablesAtZKLive` take `hh : I.tags.hiddenOutputs = false`: they are about
`Stmt.setupH`'s statements, and under hidden outputs (the default `--statement`, PROTOCOL §16.13) `Stmt.setupTables`
calls `Stmt.setupHidden`. Each statement here is one of them for the sessions production `--zk` writes, as HiddenExec's
are for the public path. Where the original names `setupH`, this one names `setupOne` (without `hh`) or
`setupHidden` (under `hh = true`); otherwise the conclusion is the original's.

* `AcceptsZKSetupOne`, `AcceptsZKOneAny`, `AcceptsZKRetainedAny`: ZkExec's `AcceptsZKSetupH`, `AcceptsZKOne`,
  `AcceptsZKRetained` without `hh`.
* `TablesAtZKHLive`: ZkLink's `TablesAtZKLive` under `hh = true`, at `tablesAtZKH`, with HiddenExec's
  `TableAtHOfAccepts` hypotheses on the circuit (`HmOut.parse` accepts it, at the `--zk` tags; its unit returns the
  outputs). `TablesAtZKHLiveI` is it without the unit returning the outputs. `TablesAtZKHLiveSt` and
  `TablesAtZKHLiveISt` are them at the statement circuit (`HmOut.parseStmt`, a templated file's template unrolled), and
  `TablesAtZKHLiveU` is the conclusion with no circuit hypothesis, which the session's committed circuit discharges.
* `AcceptsZKAcceptedH`: lp-zkbind's `acceptsZK_accepted` (the placement every `--zk` table's binding is read at) for
  one hidden-output table, with HiddenExec's `AcceptedH` for `Placed.Accepted`.
* `ZkSchedOK`: the half of `TablesAtZKLive` that reads no setup: at any statement `Zk.shapeOk` passes, `Zk.schedule`'s
  schedule and `t_pad` have the facts `DrawSetupZK` and `DrawSetupZKH` record.

`zk_flock_countC_execOS_crH` (`Count.lean`) is `zk_flock_countC_execOS_cr` at the hidden-output plan.
-/

namespace FlockSoundness.Discharge.ZkHidden

open Flock FlockLevel3 Game Model Refine Exec DecodeZL ZkSession ZkExec ZkLink HiddenExec
open scoped ENNReal NNReal

local notation "𝔽" => FlockLevel3.GF128

/-! ## Acceptance -/

/-- **Each table of a `--zk` acceptance is `setupOne`'s statement** under the `--zk` tags and the coin tree, at some
draw (the record's, for one table), and passes the `--zk` shape check (`AcceptsZKSetupH` without `hh`). -/
def AcceptsZKSetupOne : Prop :=
  ∀ (I : Inputs) (t : Exec.Transcript), AcceptsZK I t →
    ∃ spec sts, setupOf (zkInputs I) t.record = .ok (spec, sts) ∧
      Flock.Zk.verify spec (sts.map Flock.Setup.ofCircuit) t.record t.proofs = .ok () ∧ 0 < sts.size ∧
      ∀ (j : ℕ) (hj : j < sts.size), Flock.Zk.shapeOk sts[j] = .ok () ∧ ∃ d,
        setupOne (tagsAt (zkTags I) I.count j) I.circuitFile I.publicFile (zkCoins I) I.tables d I.partition
          I.program = .ok sts[j] ∧ (I.count ≤ 1 → d = recordDraw t.record)

/-- **A one-table `--zk` acceptance is one `setupOne` statement's** (`AcceptsZKOne` without `hh`). -/
def AcceptsZKOneAny : Prop :=
  ∀ (I : Inputs) (t : Exec.Transcript), I.count ≤ 1 → AcceptsZK I t →
    ∃ st, setupOne (zkTags I) I.circuitFile I.publicFile (zkCoins I) I.tables (recordDraw t.record) I.partition
        I.program = .ok st ∧ Flock.Zk.shapeOk st = .ok () ∧
      Flock.Zk.verify st.spec #[Flock.Setup.ofCircuit st] t.record t.proofs = .ok ()

/-- **A `--zk` acceptance under retaining tags has every round's bytes** (`AcceptsZKRetained` without `hh`). -/
def AcceptsZKRetainedAny : Prop :=
  ∀ (I : Inputs) (t : Exec.Transcript), I.tags.retainRounds = true → AcceptsZK I t →
    ∃ spec sts sess, setupOf (zkInputs I) t.record = .ok (spec, sts) ∧ Record.decode spec t.record = .ok sess ∧
      ∀ s ∈ sess.streams.toList, AllRetained s.rounds

/-- **A one-table hidden-output `--zk` acceptance is placed** (lp-zkbind's `acceptsZK_accepted` at one table, with
`AcceptedH` for `Accepted`): the table is setupHidden's statement `st` at the record's draw, it passes the shape check,
and at any regions there is an `AcceptedH` of `st` under the `--zk` tags on the session's files, coins and tables. -/
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

/-! ## The schedule -/

/-- **`Zk.schedule`'s facts at a statement the `--zk` shape check passes**: its schedule has `DrawSetupZK`'s shape and
`m`, `t_pad` pads level 0, and the inner code at `layOf st sch` has fewer than `2^64` columns, at least the pads
message's, and uniform query positions. It reads `st` only through `m`, `kLog` and `shapeOk`, so it is the same for
`setupH`'s and `setupHidden`'s statements. -/
def ZkSchedOK : Prop :=
  ∀ (st : Stmt) (sch : Flock.Schedule) (tPad : ℕ), StmtWF st → RegionsWF st → 13 ≤ st.m →
    Zk.shapeOk st = .ok () → Zk.schedule st.m = .ok (sch, tPad) →
    (2 ≤ (schedOf sch).levels.length ∧ ((schedOf sch).levels.headD default).lanes = (schedOf sch).k0 ∧
      (schedOf sch).m - 7 = (schedOf sch).k0 + ((schedOf sch).levels.headD default).logCols ∧
      levelsChained (schedOf sch).levels (schedOf sch).ext = true) ∧
    (schedOf sch).m = st.m ∧ Accounting.PadOK (schedOf sch).level₀ tPad ∧ innerN (layOf st sch) < 2 ^ 64 ∧
    (layOf st sch).len + Zk.K_PAD ≤ innerN (layOf st sch) ∧
    ∀ (hN : 0 < innerN (layOf st sch)) (D : Finset (Fin (innerN (layOf st sch)))),
      prCoin (fun w => innerPos _ hN w ∈ D) ≤ (D.card : ℝ≥0∞) / innerN (layOf st sch)

/-! ## The tables -/

variable {n : ℕ} {mPts : ℕ} {Reg : Type}

/-- **Wherever flock-verify accepts a hidden-output session, the `--zk` table is the executable's**
(`TablesAtZKLive` under `hh = true`). On hidden-output one-table inputs whose `--statement` tags are untyped and whose
circuit file `HmOut.parse` accepts (at the `--zk` tags) as a circuit whose unit returns the outputs, with `wr` the coin
server's record (`RecordsLiveZK`): whenever the draw `S` is `DrawOkZKH` and `verify --zk` accepts what `wr` writes for
`o` at it, setupHidden's statement `st` at `dj S` under the `--zk` tags is well formed, `Zk.schedule st.m` is
`(sch, t_pad)`, the draw has a `DrawSetupZKH` at `st`, `sch` and `t_pad` whose table is `tablesAtZKH … S R`, and at the
record's link points `Setup.ofCircuit st`'s extra claims are that table's. -/
def TablesAtZKHLive : Prop :=
  ∀ (I : Inputs) (_ht : I.tags.typed = false) {c : Flock.Circuit}
    (_hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (_ho : c.outNet = c.unitNet)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    (y₀ : DrawSetupZKH I mPts S₀ (dj S₀)) (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript),
    I.tags.hiddenOutputs = true → I.count ≤ 1 → RecordsLiveZK I dj wr →
    ∀ S R o, DrawOkZKH I dj mPts S → AcceptsZK I (wr S R o) →
      ∃ (st : Stmt) (sch : Flock.Schedule) (tPad : ℕ), StmtWF st ∧ RegionsWF st ∧
        Stmt.setupHidden (zkTags I) I.circuitFile I.publicFile (zkCoins I) I.tables (some (dj S)) I.partition
          I.program = .ok st ∧ Zk.schedule st.m = .ok (sch, tPad) ∧
        ∃ y : DrawSetupZKH I mPts S (dj S), y.x.st = st ∧ y.sch = sch ∧ y.tPad = tPad ∧
          tablesAtZKH (Reg := Reg) dj y₀ S R = y.table ∧
          ∀ points : Array (Array F128), points.size = 2 → (∀ p ∈ points.toList, p.size = mPts) →
            ExtraRealizes (Setup.ofCircuit st) y.table.S points
              (Model.extraClaims y.table.S y.table.ptLocal mPts (ptsOf mPts points))

/-- **`TablesAtZKHLive` wherever the outputs are returned**: its conclusion on every circuit `HmOut.parse` accepts. -/
def TablesAtZKHLiveI : Prop :=
  ∀ (I : Inputs) (_ht : I.tags.typed = false) {c : Flock.Circuit}
    (_hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    (y₀ : DrawSetupZKH I mPts S₀ (dj S₀)) (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript),
    I.tags.hiddenOutputs = true → I.count ≤ 1 → RecordsLiveZK I dj wr →
    ∀ S R o, DrawOkZKH I dj mPts S → AcceptsZK I (wr S R o) →
      ∃ (st : Stmt) (sch : Flock.Schedule) (tPad : ℕ), StmtWF st ∧ RegionsWF st ∧
        Stmt.setupHidden (zkTags I) I.circuitFile I.publicFile (zkCoins I) I.tables (some (dj S)) I.partition
          I.program = .ok st ∧ Zk.schedule st.m = .ok (sch, tPad) ∧
        ∃ y : DrawSetupZKH I mPts S (dj S), y.x.st = st ∧ y.sch = sch ∧ y.tPad = tPad ∧
          tablesAtZKH (Reg := Reg) dj y₀ S R = y.table ∧
          ∀ points : Array (Array F128), points.size = 2 → (∀ p ∈ points.toList, p.size = mPts) →
            ExtraRealizes (Setup.ofCircuit st) y.table.S points
              (Model.extraClaims y.table.S y.table.ptLocal mPts (ptsOf mPts points))

/-- **`TablesAtZKHLive` at the statement circuit**: its conclusion on every circuit `HmOut.parseStmt` reads at the `--zk`
tags (a templated file's template unrolled, `HmOut.unrolled`) whose unit returns the outputs. -/
def TablesAtZKHLiveSt : Prop :=
  ∀ (I : Inputs) (_ht : I.tags.typed = false) {c : Flock.Circuit}
    (_hc : HmOut.parseStmt (zkTags I) I.circuitFile I.tables = .ok c) (_ho : c.outNet = c.unitNet)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    (y₀ : DrawSetupZKH I mPts S₀ (dj S₀)) (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript),
    I.tags.hiddenOutputs = true → I.count ≤ 1 → RecordsLiveZK I dj wr →
    ∀ S R o, DrawOkZKH I dj mPts S → AcceptsZK I (wr S R o) →
      ∃ (st : Stmt) (sch : Flock.Schedule) (tPad : ℕ), StmtWF st ∧ RegionsWF st ∧
        Stmt.setupHidden (zkTags I) I.circuitFile I.publicFile (zkCoins I) I.tables (some (dj S)) I.partition
          I.program = .ok st ∧ Zk.schedule st.m = .ok (sch, tPad) ∧
        ∃ y : DrawSetupZKH I mPts S (dj S), y.x.st = st ∧ y.sch = sch ∧ y.tPad = tPad ∧
          tablesAtZKH (Reg := Reg) dj y₀ S R = y.table ∧
          ∀ points : Array (Array F128), points.size = 2 → (∀ p ∈ points.toList, p.size = mPts) →
            ExtraRealizes (Setup.ofCircuit st) y.table.S points
              (Model.extraClaims y.table.S y.table.ptLocal mPts (ptsOf mPts points))

/-- **`TablesAtZKHLiveI` at the statement circuit**: its conclusion on every circuit `HmOut.parseStmt` reads. -/
def TablesAtZKHLiveISt : Prop :=
  ∀ (I : Inputs) (_ht : I.tags.typed = false) {c : Flock.Circuit}
    (_hc : HmOut.parseStmt (zkTags I) I.circuitFile I.tables = .ok c)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    (y₀ : DrawSetupZKH I mPts S₀ (dj S₀)) (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript),
    I.tags.hiddenOutputs = true → I.count ≤ 1 → RecordsLiveZK I dj wr →
    ∀ S R o, DrawOkZKH I dj mPts S → AcceptsZK I (wr S R o) →
      ∃ (st : Stmt) (sch : Flock.Schedule) (tPad : ℕ), StmtWF st ∧ RegionsWF st ∧
        Stmt.setupHidden (zkTags I) I.circuitFile I.publicFile (zkCoins I) I.tables (some (dj S)) I.partition
          I.program = .ok st ∧ Zk.schedule st.m = .ok (sch, tPad) ∧
        ∃ y : DrawSetupZKH I mPts S (dj S), y.x.st = st ∧ y.sch = sch ∧ y.tPad = tPad ∧
          tablesAtZKH (Reg := Reg) dj y₀ S R = y.table ∧
          ∀ points : Array (Array F128), points.size = 2 → (∀ p ∈ points.toList, p.size = mPts) →
            ExtraRealizes (Setup.ofCircuit st) y.table.S points
              (Model.extraClaims y.table.S y.table.ptLocal mPts (ptsOf mPts points))

/-- **`TablesAtZKHLiveI` from the session alone**: its conclusion with no hypothesis on the circuit, which the session's
own committed circuit discharges. -/
def TablesAtZKHLiveU : Prop :=
  ∀ (I : Inputs) (_ht : I.tags.typed = false)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    (y₀ : DrawSetupZKH I mPts S₀ (dj S₀)) (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript),
    I.tags.hiddenOutputs = true → I.count ≤ 1 → RecordsLiveZK I dj wr →
    ∀ S R o, DrawOkZKH I dj mPts S → AcceptsZK I (wr S R o) →
      ∃ (st : Stmt) (sch : Flock.Schedule) (tPad : ℕ), StmtWF st ∧ RegionsWF st ∧
        Stmt.setupHidden (zkTags I) I.circuitFile I.publicFile (zkCoins I) I.tables (some (dj S)) I.partition
          I.program = .ok st ∧ Zk.schedule st.m = .ok (sch, tPad) ∧
        ∃ y : DrawSetupZKH I mPts S (dj S), y.x.st = st ∧ y.sch = sch ∧ y.tPad = tPad ∧
          tablesAtZKH (Reg := Reg) dj y₀ S R = y.table ∧
          ∀ points : Array (Array F128), points.size = 2 → (∀ p ∈ points.toList, p.size = mPts) →
            ExtraRealizes (Setup.ofCircuit st) y.table.S points
              (Model.extraClaims y.table.S y.table.ptLocal mPts (ptsOf mPts points))

end FlockSoundness.Discharge.ZkHidden
