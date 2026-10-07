import Proofs.Flock.Soundness.Discharge.ZkLink.TablesAt
import Proofs.Flock.Soundness.Discharge.HiddenExec.Tables

/-!
# The `--zk` tables of a hidden-output session

ZkLink's `DrawSetupZK`, `tableAtZK`, `tablesAtZK` and `DrawOkZK` (`ZkLink/TablesAt.lean`) for `Stmt.setupHidden`, which
`Stmt.setupTables` calls under a tag with hidden outputs (`HiddenExec.setupOne`): the draw's setup is HiddenExec's
`DrawSetupH` at `zkInputs I` in place of Integrate's `DrawSetup`, and the table is built from it as `DrawSetupZK.tabAt`
builds it, field for field.

* `DrawSetupZKH I mPts S d`: a `DrawSetupH` at `zkInputs I`, `Zk.schedule`'s schedule and `t_pad` at its `m`, and
  `DrawSetupZK`'s facts of them.
* `DrawSetupZKH.table`, `.refused`, `tableAtZKH`, `tablesAtZKH`: as `DrawSetupZK`'s.
* `planAtZKH`, `tabAtZKH`: the one-table plan `zk_flock_countC_execOS_cr` takes.
* `DrawOkZKH`: `DrawOkZK` with `setupHidden` for `setupH`.
-/

namespace FlockSoundness.Discharge.ZkHidden

open Flock FlockLevel3 Game Model Refine Exec DecodeZL ZkSession ZkExec ZkLink HiddenExec
open scoped ENNReal NNReal

local notation "𝔽" => FlockLevel3.GF128

variable {n : ℕ} (I : Inputs) (mPts : ℕ)

/-- **A draw `verify --zk` sets up under hidden outputs**: `DrawSetupZK` with a `DrawSetupH` at `zkInputs I`
(setupHidden accepts the draw under the `--zk` tags and coin tree) for its `DrawSetup`. -/
structure DrawSetupZKH (S : Finset (Fin n)) (d : Lean.Json) where
  x : DrawSetupH (zkInputs I) mPts S d
  sch : Flock.Schedule
  tPad : ℕ
  hz : Zk.schedule x.st.m = .ok (sch, tPad)
  hshape : 2 ≤ (schedOf sch).levels.length ∧ ((schedOf sch).levels.headD default).lanes = (schedOf sch).k0 ∧
    (schedOf sch).m - 7 = (schedOf sch).k0 + ((schedOf sch).levels.headD default).logCols ∧
    levelsChained (schedOf sch).levels (schedOf sch).ext = true
  hmsch : (schedOf sch).m = x.st.m
  hpad : Accounting.PadOK (schedOf sch).level₀ tPad
  hN : innerN (layOf x.st sch) < 2 ^ 64
  hL : (layOf x.st sch).len + Zk.K_PAD ≤ innerN (layOf x.st sch)
  hpos : ∀ D : Finset (Fin (innerN (layOf x.st sch))),
    prCoin (fun w => innerPos _ (lt_of_lt_of_le (by simp [Zk.K_PAD]) hL) w ∈ D) ≤
      (D.card : ℝ≥0∞) / innerN (layOf x.st sch)

variable {I mPts}

namespace DrawSetupZKH

variable {S : Finset (Fin n)} {d : Lean.Json} (y : DrawSetupZKH I mPts S d)

theorem innerN_pos : 0 < innerN (layOf y.x.st y.sch) := lt_of_lt_of_le (by simp [Zk.K_PAD]) y.hL

/-- `codeAt`'s points, distinct below `2^64`. -/
def innerDom : Fin (innerN (layOf y.x.st y.sch)) ↪ 𝔽 :=
  ⟨fun j => innerPt j.val, fun a b h => by
    have hlt : ∀ j : Fin (innerN (layOf y.x.st y.sch)), j.val + 1 < 2 ^ 64 := fun j => by
      have := y.hN
      have := j.isLt
      omega
    have h' : ((a.val + 1).toUInt64 : UInt64) = (b.val + 1).toUInt64 := congrArg F128.lo h
    have := congrArg UInt64.toNat h'
    simp only [Nat.toUInt64_eq, UInt64.toNat_ofNat', Nat.mod_eq_of_lt (hlt a), Nat.mod_eq_of_lt (hlt b)] at this
    exact Fin.ext (by omega)⟩

/-- **The `--zk` table of a hidden-output draw**, at statement regions `regs` (`DrawSetupZK.tabAt`). -/
noncomputable def tabAt (regs : List (Model.Region y.x.st.c.kLog (y.x.st.m - y.x.st.c.kLog)))
    (hregs : ∀ R ∈ regs, R.free.Nodup ∧ R.free.length ≤ PT_LOCAL) : TabZK 𝔽 mPts where
  S := stmtOf y.x.st y.x.wf.1 regs
  sch := schedOf y.sch
  hshape := y.hshape
  hmsch := y.hmsch
  hm := y.x.hm
  t := y.tPad
  hpad := y.hpad
  c₀ := fOf Zk.PAD_C
  e := 2
  ptLocal := PT_LOCAL
  hlay := ⟨hregs, by
    have h := (stmtOf_linkLayout y.x.st y.x.wf.1 y.x.wf.2).2
    rw [y.x.hmPts] at h
    exact h⟩
  np := (layOf y.x.st y.sch).len
  pads := padsOfLay (layOf y.x.st y.sch)
  padsA := padsAOfLay (layOf y.x.st y.sch)
  tr := tripleOfLay (layOf y.x.st y.sch)
  N := innerN (layOf y.x.st y.sch)
  dom := y.innerDom
  nP := Zk.K_PAD
  q := Zk.Q
  hnP := ⟨by simp [Zk.K_PAD]⟩
  pos := innerPos _ y.innerN_pos
  hpos := y.hpos
  δ := Code.relativeUniqueDecodingRadius
    (ReedSolomon.code y.innerDom ((layOf y.x.st y.sch).len + Zk.K_PAD) : Set (Fin (innerN (layOf y.x.st y.sch)) → 𝔽))
  hL := y.hL
  hδ := le_rfl

/-- The draw's `--zk` table: setupHidden's statement at its regions. -/
noncomputable def table : TabZK 𝔽 mPts := y.tabAt (regionsOf y.x.st) (stmtOf_linkLayout y.x.st y.x.wf.1 y.x.wf.2).1

/-- A refused draw's `--zk` table: the statement with `contra`'s regions, which no witness satisfies. -/
noncomputable def refused : TabZK 𝔽 mPts :=
  y.tabAt (regionsOf y.x.st ++ Integrate.DrawSetup.contra y.x.st.c.kLog (y.x.st.m - y.x.st.c.kLog)) y.x.refused_wf

end DrawSetupZKH

variable {Reg : Type} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (y₀ : DrawSetupZKH I mPts S₀ (dj S₀))

open Classical in
/-- **The hidden-output `--zk` table at draw `S`** (`tableAtZK` under hidden outputs): the draw's table when
`verify --zk` sets up the draw's JSON `dj S` with setupHidden, and otherwise the reference setup `y₀`'s refused table. -/
noncomputable def tableAtZKH (S : Finset (Fin n)) : TabZK 𝔽 mPts :=
  if h : Nonempty (DrawSetupZKH I mPts S (dj S)) then (Classical.choice h).table else y₀.refused

/-- **The hidden-output `--zk` tables** (`tablesAtZK` under hidden outputs): `tableAtZKH` at every registration. -/
noncomputable abbrev tablesAtZKH : Finset (Fin n) → Reg → TabZK 𝔽 mPts := fun S _ => tableAtZKH dj y₀ S

/-- The one-table hidden-output `--zk` plan: at draw `S`, the table `tableAtZKH … S`, for every registration. -/
noncomputable abbrev planAtZKH : Finset (Fin n) → Reg → List (TabZK 𝔽 mPts) := fun S _ => [tableAtZKH dj y₀ S]

/-- Every drawn unit is in the plan's one table. -/
abbrev tabAtZKH : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planAtZKH (Reg := Reg) dj y₀ S R).length :=
  fun _ _ _ => ⟨0, Nat.zero_lt_one⟩

/-- **The challenger's draw at `S`, at `--zk` under hidden outputs** (`DrawOkZK` with `setupHidden` for `setupH`): it
proves `S`, and setupHidden sets up at it under the `--zk` tags and coin tree with `mPts` link points. -/
def DrawOkZKH (I : Inputs) (dj : Finset (Fin n) → Lean.Json) (mPts : ℕ) (S : Finset (Fin n)) : Prop :=
  (∀ D, Draw.UnitDraw.ofJson (dj S) = .ok D → ∀ u ∈ S, u.val ∈ D.proved) ∧
    ∀ st, Stmt.setupHidden (zkTags I) I.circuitFile I.publicFile (zkCoins I) I.tables (some (dj S)) I.partition
      I.program = .ok st → st.mPts = mPts

end FlockSoundness.Discharge.ZkHidden
