import FlockSoundness.Discharge.HiddenExec.Placed
import FlockSoundness.Protocol.Discharge.ZkHidden.Defs

/-!
# The statements: flock-verify `--zk`'s acceptance and its tables under hidden outputs

ZkExec's acceptance statements and ZkLink's `TablesAtZKLive` take `hh : I.tags.hiddenOutputs = false`: they are about
`Stmt.setupH`'s statements, and under hidden outputs (the default `--statement`, PROTOCOL §16.13) `Stmt.setupTables`
calls `Stmt.setupHidden`. Each statement here is one of them for the sessions production `--zk` writes, as HiddenExec's
are for the public path (#869). Where the original names `setupH`, this one names `setupOne` (without `hh`) or
`setupHidden` (under `hh = true`); otherwise the conclusion is the original's.

* `AcceptsZKSetupOne`, `AcceptsZKOneAny`, `AcceptsZKRetainedAny`: ZkExec's `AcceptsZKSetupH`, `AcceptsZKOne`,
  `AcceptsZKRetained` without `hh`.
* `TablesAtZKHLive`: ZkLink's `TablesAtZKLive` under `hh = true`, at `tablesAtZKH`, with #869's `TableAtHOfAccepts`
  hypotheses on the circuit (`HmOut.parse` accepts it, at the `--zk` tags; its unit returns the outputs).
* `AcceptsZKAcceptedH`: lp-zkbind's `acceptsZK_accepted` (the placement every `--zk` table's binding is read at) for
  one hidden-output table, with #869's `AcceptedH` for `Placed.Accepted`.
* `ZkSchedOK`: the half of `TablesAtZKLive` that reads no setup: at any statement `Zk.shapeOk` passes, `Zk.schedule`'s
  schedule and `t_pad` have the facts `DrawSetupZK` and `DrawSetupZKH` record.

`zk_flock_countC_execOS_crH` (`Count.lean`) is #874's `zk_flock_countC_execOS_cr` at the hidden-output plan.
-/

namespace FlockSoundness.Discharge.ZkHidden

open Flock FlockLevel3 Game Model Refine Exec DecodeZL ZkSession ZkExec ZkLink HiddenExec
open scoped ENNReal NNReal

local notation "𝔽" => FlockLevel3.GF128

/-! ## Acceptance -/

/-- **A one-table `--zk` acceptance is one `setupOne` statement's** (`AcceptsZKOne` without `hh`). -/
def AcceptsZKOneAny : Prop :=
  ∀ (I : Inputs) (t : Exec.Transcript), I.count ≤ 1 → AcceptsZK I t →
    ∃ st, setupOne (zkTags I) I.circuitFile I.publicFile (zkCoins I) I.tables (recordDraw t.record) I.partition
        I.program = .ok st ∧ Flock.Zk.shapeOk st = .ok () ∧
      Flock.Zk.verify st.spec #[Flock.Setup.ofCircuit st] t.record t.proofs = .ok ()

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

end FlockSoundness.Discharge.ZkHidden
