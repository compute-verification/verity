import FlockSoundness.Protocol.Discharge.DecodeZL.Statements
import FlockSoundness.Protocol.Discharge.ZkExec.Statements
import FlockSoundness.Protocol.Discharge.ZkSession.Session
import FlockSoundness.Protocol.Discharge.Hidden.Facts

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

/-! ## The pads and the inner code of a pads layout -/

/-- Pad `i` of a layout, as an index into its `len` pads. -/
def padIx (l : Zk.PadLayout) (i : ℕ) : Fin l.len := ⟨i % l.len, Nat.mod_lt _ (Nat.succ_pos _)⟩

/-- Level 0's pads at a layout's positions: the OOD value, the lane rounds' messages and the pair's sums. -/
def padsOfLay (l : Zk.PadLayout) : Pads0 l.len 2 :=
  ⟨padIx l (l.lgOod 0), fun j t => padIx l (l.lgMsg j t), fun i c => padIx l (l.lgE i c)⟩

/-- The algebraic phase's pads at a layout's positions. -/
def padsAOfLay (l : Zk.PadLayout) : PadsA l.len :=
  ⟨fun i => padIx l (l.r1ab i), fun i => padIx l (l.r1c i), fun j t => padIx l (l.ml j t), padIx l l.fc,
    fun j t => padIx l (l.lc j t), fun i => padIx l (l.zp i)⟩

/-- The sacrifice's pads at a layout's positions: `f_a, f_b, h_ab` and the triple `x, y, w`. -/
def tripleOfLay (l : Zk.PadLayout) : Triple l.len :=
  ⟨padIx l l.fa, padIx l l.fb, padIx l l.hab, padIx l l.tx, padIx l l.ty, padIx l l.tw⟩

/-- The `--zk` pads layout of a statement at a schedule. -/
def layOf (st : Stmt) (sch : Flock.Schedule) : Zk.PadLayout := Zk.PadLayout.new st.m st.c.kLog sch.k0

/-- `codeAt`'s evaluation point of column `j`: `j + 1` as a word. -/
def innerPt (j : ℕ) : 𝔽 := fOf ⟨(j + 1).toUInt64, 0⟩

/-- `verifyInner`'s query position of a coin word: `w.lo mod N`. -/
def innerPos (N : ℕ) (hN : 0 < N) (w : 𝔽) : Fin N := ⟨(GF128.val w).lo.toNat % N, Nat.mod_lt _ hN⟩

/-! ## The inner code's facts -/

/-! ## A draw `verify --zk` sets up -/

variable {n : ℕ} (I : Inputs) (mPts : ℕ)

/-- **A draw `verify --zk` sets up**: a `DrawSetup` at `zkInputs I` (setupH accepts the draw under the `--zk` tags and
coin tree), `Zk.schedule`'s schedule and `t_pad` at its `m`, and the facts of them the `--zk` table needs: the schedule's
shape and `m`, `t_pad`'s padding, the inner code's length (below `2^64`, so `codeAt`'s points are distinct, and at least
the pads message's), and its query positions' uniformity; and the statement passes the `--zk` shape check, as
`verify --zk` checks at setup (`Zk.shapeOk`). -/
structure DrawSetupZK (S : Finset (Fin n)) (d : Lean.Json) where
  x : DrawSetup (zkInputs I) mPts S d
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
  hzk : Zk.shapeOk x.st = .ok ()

variable {I mPts}

namespace DrawSetupZK

variable {S : Finset (Fin n)} {d : Lean.Json} (y : DrawSetupZK I mPts S d)

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

/-- **The `--zk` table of a draw `verify --zk` sets up**, at statement regions `regs`. -/
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

/-- The draw's `--zk` table: setupH's statement at its regions. -/
noncomputable def table : TabZK 𝔽 mPts := y.tabAt (regionsOf y.x.st) (stmtOf_linkLayout y.x.st y.x.wf.1 y.x.wf.2).1

/-- A refused draw's `--zk` table: the statement with `contra`'s regions, which no witness satisfies. -/
noncomputable def refused : TabZK 𝔽 mPts :=
  y.tabAt (regionsOf y.x.st ++ DrawSetup.contra y.x.st.c.kLog (y.x.st.m - y.x.st.c.kLog)) y.x.refused_wf

end DrawSetupZK

section

variable {S : Finset (Fin n)} {d : Lean.Json}

end

variable (ht : I.tags.typed = false) {Reg : Type} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
  (y₀ : DrawSetupZK I mPts S₀ (dj S₀))

open Classical in
/-- **The `--zk` table at draw `S`** (`tableAt` at `--zk`): the draw's table when `verify --zk` sets up the draw's JSON
`dj S`, and otherwise the reference setup `y₀`'s refused table. -/
noncomputable def tableAtZK (S : Finset (Fin n)) : TabZK 𝔽 mPts :=
  if h : Nonempty (DrawSetupZK I mPts S (dj S)) then (Classical.choice h).table else y₀.refused

/-- **The `--zk` tables** (`tablesAt` at `--zk`): `tableAtZK` at every registration. -/
noncomputable abbrev tablesAtZK : Finset (Fin n) → Reg → TabZK 𝔽 mPts := fun S _ => tableAtZK dj y₀ S

/-! ## The live session's record -/

/-- **`wr` is the coin server's record of the live `--zk` session** (#862's `RecordsLive` at `--zk`): whenever
flock-verify accepts what `wr` writes for the outcome `o` at draw `S`, the record carries the draw `dj S`, the proof
files are `o`'s, and the record decodes to a session whose table's `root_B`, link points and rep streams' rounds are
`o`'s. -/
def RecordsLiveZK (I : Inputs) (dj : Finset (Fin n) → Lean.Json)
    (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript) : Prop :=
  ∀ S R o, AcceptsZK I (wr S R o) →
    recordDraw (wr S R o).record = some (dj S) ∧ (wr S R o).proofs = o.proofs ∧
    ∀ spec sts sess, setupOf (zkInputs I) (wr S R o).record = .ok (spec, sts) →
      Record.decode spec (wr S R o).record = .ok sess →
      sess.rootB[0]! = o.rootB ∧ ptsOf mPts sess.points = o.pts ∧
      ∀ r : Fin 2, ∃ s : StreamRec,
        sess.streams.find? (fun x => x.name == toString spec.tables[0]! ++ toString "/rep" ++ toString r.val) =
          some s ∧ s.rounds.toList = pick (o.rep0, o.rep1) r

/-- **The coin server's draw at `S`, at `--zk`** (#862's `DrawOk` at `zkInputs I`): it proves `S`, and setupH sets up
at it under the `--zk` tags and coin tree with `mPts` link points. -/
def DrawOkZK (I : Inputs) (dj : Finset (Fin n) → Lean.Json) (mPts : ℕ) (S : Finset (Fin n)) : Prop :=
  (∀ D, Draw.UnitDraw.ofJson (dj S) = .ok D → ∀ u ∈ S, u.val ∈ D.proved) ∧
    ∀ st, Stmt.setupH (zkTags I) I.circuitFile I.publicFile (zkCoins I) I.tables (some (dj S)) I.partition I.program =
      .ok st → st.mPts = mPts

/-! ## The statement -/

end FlockSoundness.Discharge.ZkLink
