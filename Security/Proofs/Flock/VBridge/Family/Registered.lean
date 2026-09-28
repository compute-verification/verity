import Proofs.Flock.VBridge.Family.Rep
import Proofs.Flock.VBridge.Family.Openings

/-!
# VBridge G at V*'s family: what a session's acceptance says of its registered reads

A session's acceptance (`Accepted`: `LiveAcceptsZKR` and a `Good` layer) constrains its registered reads `zo.rd` in one
direction only: every registered read is a row of a port `own` names, at a drawn unit (`ZkOuter.hown`, `OwnReads`), and
the layer agrees with the registrant there (`RegAgree`). It never says that a row is read, and `rd` is a field of the
session the family builds, so which rows are registered reads, and what the registrant's value there is, are the
family's choice.

* `rd_isLeft`, `not_read_unit`, `not_regOut`: a registered read is a row of a unit's port (`read_row`: `rowsSh`'s rows
  are `posK`'s `.inl`), and a unit's computed gate is no row bit (`keyRowOf` is `some` only at input gates), so no
  output column is ever a registered read: `RegOut` at a column past the inputs is `False`. Hence `not_resZeroAt` and
  `not_opensAt`: `ResZeroAt` and `OpensAt`, whose last conjuncts are `RegOut` at the residuals' and `acc_out`'s output
  columns, never hold (so the family's parts and openings are read in the layers: `HResL`, `HUL`); `not_accOut`:
  `hAccRows_of`'s `hout` is never satisfiable.
* `oneAt_keyProg`: every unit's constant is the program's input gate `Layout.oneG`; `hOne_of`, `hOneOpen_of`: `HOne` and
  `HOneOpen` from `OneInLayer`, the layers carrying 1 at that gate. V* registers no row holding the constant: a row is an
  input port's words or an output port's (`circuit.py:16-18`), and the reads files name ports only
  (`rec_outer.py:236-238`, `rec_vstage.py:356-367`).
* `hRowKeys_of`: `HRowKeys` from the part circuits' keys (`RowColKeys`) and the family's registered reads covering the
  rows `own` names (`RowsRegistered`); `hRowsAgree_of`: `HRowsAgree` from those, the two parts' columns keying one bit
  of rows at one registered root and position (`SameRow`), and one registrant row there (`RowsShared`).
-/

namespace FlockVBridge.Family

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.Recursion FlockSoundness.Refine FlockSoundness.Model FlockSoundness.Audit
open Flock FlockLevel3 FlockVBridge.Compiled FlockVBridge.InnerAccepts FlockVBridge.Stages
open Flock.RecResiduals (portSizes groups)
open Flock.RecAlgebra (structureOf verifierRows el)
open FlockVBridge.OfVStar (ResZeroAt partBase)
open FlockVBridge.AccChain (RegOut OpensAt)

/-! ## A session's registered reads -/

section Reads

variable {mPts nv : ℕ} {Reg' : Type} {c : Flock.Circuit} {hU : Built c.unit} {hscope : Layout.scopeOk c = true}
  {Lv : Law nv} (zo : ZkOuter mPts nv Reg' c hU hscope Lv)

/-- **A registered read is a row** (`Reads.read_row`; `rowsSh`'s rows are the commit strings `.inl (u, q)`,
`Readers.ofRows`' `isRow`). -/
theorem rd_isLeft {p : PosK c hU nv} (h : zo.rd.isRead p) : p.isLeft = true :=
  zo.rd.read_row p h

/-- **A unit's computed gate is no row bit**: `keyRowOf` is `some` only below `2 + n · B`, where `uniform` places the
program's inputs, which are no unit's (`unit_uniform_input`). -/
theorem keyRowOf_unit {g : Fin (Layout.keyN c hU nv)} {u : Fin nv}
    (hu : ((Placed.keyProg c hU nv).partition (Layout.outsOf c hU nv)).unit g = some u) :
    Layout.keyRowOf c hU nv (Layout.portRowW c) g = none := by
  rcases h : Layout.keyRowOf c hU nv (Layout.portRowW c) g with _ | ⟨u', q, k⟩
  · rfl
  · obtain ⟨-, hlt, -, -⟩ := Layout.keyRowOf_spec h
    have hnone : (Placed.keyProg c hU nv).unit g = none :=
      Layout.unit_uniform_input (k := Placed.inGates nv (Layout.keyB c hU)) (R := Placed.circuitRows c hU)
        (ins := Placed.insOf nv (Layout.keyB c hU) (Aliased.keyIdx c (Layout.keyB c hU)))
        (one := Placed.oneGate nv (Layout.keyB c hU))
        (hone := Placed.insOf_ne_one nv (Layout.keyB c hU) (Aliased.keyIdx c (Layout.keyB c hU))) nv g
        (by show _ < 2 + nv * Layout.keyB c hU; omega)
    have hu' : (Placed.keyProg c hU nv).unit g = some u := hu
    rw [hnone] at hu'
    cases hu'

/-- **A unit's computed gate is no registered read**: its commit string is the single gate `.inr` (`keyRowOf_unit`,
`lay_pos`), and a registered read is a row (`rd_isLeft`). -/
theorem not_read_unit {g : Fin (Layout.keyN c hU nv)} {u : Fin nv}
    (hu : ((Placed.keyProg c hU nv).partition (Layout.outsOf c hU nv)).unit g = some u) :
    ¬ zo.rd.isRead (zo.lay.pos g) := by
  intro hread
  have hp : posK c hU nv g = .inr g := by
    unfold posK
    rw [keyRowOf_unit hu]
  have h : (zo.lay.pos g).isLeft = true := rd_isLeft zo hread
  rw [Decode.lay_pos zo, hp] at h
  exact Bool.false_ne_true h

/-- **No output column is a registered read.** Copy `j`'s gate at a column past the inputs is a computed gate of unit
`u` (`stackCol_comp`), so no registered read (`not_read_unit`), and `RegOut` there, which asks for one, never holds. -/
theorem not_regOut {u : Fin nv} {j : ℕ} (hj : j < c.upv) {col : ℕ} (hnIn : (Rows.ofNet hU).nIn ≤ col) (b : Bool) :
    ¬ RegOut zo u hj col b := by
  rintro ⟨hcol, -, hread, -⟩
  obtain ⟨i, hi⟩ := Decode.stackCol_comp (Rows.ofNet hU) c.upv hj ⟨col, hcol⟩ hnIn
  have hu : ((Placed.keyProg c hU nv).partition (Layout.outsOf c hU nv)).unit
      ((keyInst hU nv (Layout.outsOf c hU nv) u).wire ((Rows.ofNet hU).stackCol c.upv hj ⟨col, hcol⟩)) = some u := by
    rw [hi]
    exact (keyInst hU nv (Layout.outsOf c hU nv) u).mem i
  exact not_read_unit zo hu hread

/-- **`ResZeroAt` never holds**: its outputs are `RegOut` at the part's output columns, past the inputs
(`inRows_le_outStart`, as in `partCols_of_resZeroAt`), and there are `16 · RES_OUT_WORDS > 0` of them. -/
theorem not_resZeroAt (bits : Fin (Layout.keyN c hU nv) → Bool) (u : Fin nv) (j : ℕ) (sh : Shape)
    (gs : Array (Array ℕ × Array ℕ × Array ℕ)) (p : ℕ) (w : ℕ → F128) (rounds : Array (Array F128))
    (extras : Array Claim) (comb : Array F128) : ¬ ResZeroAt zo bits u j sh gs p w rounds extras comb := by
  rintro ⟨-, hj, -, hc, -, m, k, cl, hp, hsh, S, ns, nv', hpart, -, -, hz⟩
  obtain ⟨S1, ns1, nv1, hpart1, hn⟩ := OfVStar.check_part hc hp hsh
  rw [hpart] at hpart1
  cases hpart1
  have hbase : (Rows.ofNet hU).nIn ≤ partBase S ns nv' := by
    rw [Layout.ofNet_nIn_eq, OfVStar.inWords_unitLaid hn]
    exact Decode.inRows_le_outStart _ _ _
  exact not_regOut zo hj (by omega) false (hz 0 (by show 0 < 16 * 128; omega))

/-- **`OpensAt` never holds**: its last conjunct is `RegOut` at `acc_out`'s columns, past the inputs (as in
`unitSum_of_decode`). -/
theorem not_opensAt (bits : Fin (Layout.keyN c hU nv) → Bool) (u : Fin nv) (j LANES : ℕ) (r a cf A : Array F128) :
    ¬ OpensAt zo bits u j LANES r a cf A := by
  rintro ⟨hj, -, -, -, H, p, outBits, name, -, -, -, -, hl, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, haout⟩
  have hnIn : (Rows.ofNet hU).nIn ≤
      Hm.outStart ((recOpenNet LANES H p).run Flock.HmNets.St.empty).2 p.bits outBits := by
    rw [Layout.ofNet_nIn_eq, inWords_recOpen hl]
    exact Decode.inRows_le_outStart _ _ _
  exact not_regOut zo hj (by omega) _ (haout 0 (by omega))

end Reads

section Reg

variable {Q : VStmt} {Reg' : Type} (zo : Q.Outer Reg')

/-- **`hAccRows_of`'s `hout` is never satisfiable**: the `acc_out` columns of an opening are past its inputs (as in
`accOut_of_regOut`), so none is a registered read (`not_regOut`). -/
theorem not_accOut {u : Fin Q.nv} (hj : 0 < Q.c.upv) {LANES H : ℕ}
    (hp : Flock.RecOpen.params Q.c.unit.name = some (LANES, H)) (t : ℕ) (b : Bool) :
    ¬ AccChain.RegOut zo u hj (outCol LANES H + 1024 + t) b := by
  have hnIn : (Rows.ofNet Q.hU).nIn ≤ outCol LANES H := by
    rw [Layout.ofNet_nIn_eq, inWords_unit (Flock.RecOpen.check_ok (parse_facts zo.hc).recNet hp).2]
    exact Decode.inRows_le_outStart _ _ _
  exact not_regOut zo hj (by omega) b

end Reg

/-! ## The constant -/

/-- **Every unit's constant is the program's gate `oneG`** (`oneAt_uniform_val`). -/
theorem oneAt_keyProg (c : Flock.Circuit) (hU : Built c.unit) (nv : ℕ) (u : Fin nv) :
    Const.Prog.oneAt (Placed.keyProg c hU nv) u = Layout.oneG (c := c) (hU := hU) (n := nv) :=
  Fin.ext (Layout.oneAt_uniform_val (k := Placed.inGates nv (Layout.keyB c hU)) (R := Placed.circuitRows c hU)
    (ins := Placed.insOf nv (Layout.keyB c hU) (Aliased.keyIdx c (Layout.keyB c hU)))
    (one := Placed.oneGate nv (Layout.keyB c hU))
    (hone := Placed.insOf_ne_one nv (Layout.keyB c hU) (Aliased.keyIdx c (Layout.keyB c hU))) nv u)

section family

variable {nT nR : ℕ} {T : Fin nT → Type} {CT : Fin nR → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
variable {cls : HClass} {regs : List (Region cls.kLog (cls.m - cls.kLog))} {sch : Flock.Schedule}
variable {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => flockInnerCL T CT execArith H512 enc512 InnerAccepts.leafPlain cls regs (schedOf sch)

namespace VStarFamily

variable (F : VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂) (dg : ByteArray → ByteArray) (N k Rw : ℕ)

/-- Obligation: discharged by a check of the layer at `oneG` in `ZkOuter.Good` (`Recursive/Flock.lean`), since V*
registers no row holding the constant (`circuit.py:16-18`). **The layers carry 1 at the constant gate.** At every accepted outer proof, each session's layer is `true` at
`Layout.oneG`, the input gate every unit's constant is wired to (`oneAt_keyProg`). Acceptance does not give it: the
layer is `XplurZC`'s plurality of the prover's openings, `oneG` is no unit's gate and its commit string `.inr oneG` is
no registered read, so neither `wrong = ∅` nor `RegAgree` reads it; the public-file layer `Xpub` pins it
(`FlockPublic.lean`), the `--zk` layer does not. -/
def OneInLayer : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ j,
    layers F.zs dg N k Rw R ω t R₂ τ j (Layout.oneG (c := (F.Vs j).c) (hU := (F.Vs j).hU) (n := (F.Vs j).nv)) = true

/-- **`HOne` from `OneInLayer`**. -/
theorem hOne_of (h : F.OneInLayer dg N k Rw) : F.HOne dg N k Rw := by
  intro R ω t R₂ τ o ha r p _
  rw [oneAt_keyProg]
  exact h R ω t R₂ τ o ha (F.sess p)

/-- **`HOneOpen` from `OneInLayer`**. -/
theorem hOneOpen_of (h : F.OneInLayer dg N k Rw) : F.HOneOpen dg N k Rw := by
  intro R ω t R₂ τ o ha r ℓ _ j _
  rw [oneAt_keyProg]
  exact h R ω t R₂ τ o ha (F.sessO ℓ)

/-- Obligation: discharged by extraction at the committed trace τ* (an accepted registration's rows are the openings
the extractor gets), else by record custody. **The rows `own` names are registered reads.** At every accepted outer proof, in each session, the row of every
port the session's reads file `own` names is a registered read at every unit. Acceptance does not give it: `hown`
(`OwnReads`) says only that a registered read is such a row, and `rd` is the family's field. It says how the family's
sessions take `rd` from the registration: at each such row, the registrant's committed row, salt and path at the root
and position the verifier holds (`Reads.committed`), which only the registrant's knowledge of its openings supplies,
no file. -/
def RowsRegistered : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ j (u : Fin (F.Vs j).nv)
    (q : Fin (F.Vs j).c.ports.size), (∃ pt ∈ (F.zs R ω t R₂ j).own, pt.name = (F.Vs j).c.ports[q].name) →
      (F.zs R ω t R₂ j).rd.isRead (.inl (u, q))

/-- Obligation: discharged by `rowColKeys_of` (`Family/PartReads.lean`), from the verifier's check of the part
(`rowColOf_lt`), the parse of its circuit (`PartRowCols`) and the session's reads file (`OwnNamesRows`).
**Each part's row-port columns are keyed by a bit of a row `own` names.** At every accepted outer proof, every
row-port column `rowColOf` of part `p`'s unit is an input column whose Δ key (`Aliased.srcKey`, copy 0) is message bit
`k` of a row port `q'`, within its row, and the session's reads file names `q'`. A fact about V*'s `InnerRepCheck`
circuit (its row ports are the inputs from column 0, `rowCol`) and its reads file (`rec_vstage.py:356-357`: each row
port is a read of `regs[nm]`); established from the parse of the part's circuit (`keyProg_parse`, the lane that proves
`HStruct`) and the session's `own` (the lane that proves `HFiles`). -/
def RowColKeys : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ p, p < (partsOf F.sh).size →
    ∀ q < 128 * (nsOf F.sh p).sum, rowColOf F.sh p q < (Rows.ofNet (F.Vs (F.sess p)).hU).nIn ∧
      ∃ (q' : Fin (F.Vs (F.sess p)).c.ports.size) (k' : ℕ) (_ : k' < Layout.portRowW (F.Vs (F.sess p)).c q'),
        Aliased.srcKey (F.Vs (F.sess p)).c (0 * (Rows.ofNet (F.Vs (F.sess p)).hU).nIn + rowColOf F.sh p q) =
          .msg q'.val (k' / 8) (k' % 8) ∧
        ∃ pt ∈ (F.zs R ω t R₂ (F.sess p)).own, pt.name = (F.Vs (F.sess p)).c.ports[q'].name

/-- **`HRowKeys` from the circuit's keys and the registration's rows** (`RowColKeys`, `RowsRegistered`): the column's
key names a row `own` names, a registered read, and its bit is the registrant's. -/
theorem hRowKeys_of (hreg : F.RowsRegistered dg N k Rw) (hcol : F.RowColKeys dg N k Rw) : F.HRowKeys dg N k Rw := by
  intro R ω t R₂ τ o ha r p hp q hq
  obtain ⟨hlt, q', k', hk, hkey, hown⟩ := hcol R ω t R₂ τ o ha p hp q hq
  exact ⟨_, hlt, q', k', hk, hkey, hreg R ω t R₂ τ o ha (F.sess p) (F.un p r) q' hown, rfl⟩

/-- Obligation: discharged by `sameRow_of` (`Family/PartReads.lean`), from the parse of the parts' circuits
(`PartRowBits`), their reads files (`PartPortsOne`) and the family's registered reads (`ReadsOwn`).
**Two parts' columns of one element's bit key one bit of one registered row.** At every accepted outer proof, where
parts `p` and `p'` hold the same element of `w` at positions `a` and `a'`, the keys of their columns of bit `e` name the
same bit `b` of rows `q`, `q'`, and where both rows are registered reads at rep `r`'s units, they are at one registered
root and one position.
V*'s part sessions read element `i` of `w` as row `r` of one registration `regs[nm]` (`rec_vstage.py:356-357`);
established from the parse of the parts' circuits (the bit, the lane that proves `HStruct`) and the sessions' reads
files with `OwnReads` (the root and the position, the lane that proves `HFiles`). -/
def SameRow : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ (r : Fin 2) (p p' a a' e : ℕ),
    p < (partsOf F.sh).size → p' < (partsOf F.sh).size → ∀ (ha : a < (wIOf F.sh p).size)
    (ha' : a' < (wIOf F.sh p').size), (wIOf F.sh p)[a] = (wIOf F.sh p')[a'] → e < 128 →
    ∀ (q : Fin (F.Vs (F.sess p)).c.ports.size) (b : ℕ) (q' : Fin (F.Vs (F.sess p')).c.ports.size) (b' : ℕ),
      Aliased.srcKey (F.Vs (F.sess p)).c (0 * (Rows.ofNet (F.Vs (F.sess p)).hU).nIn + rowColOf F.sh p (128 * a + e)) =
        .msg q.val (b / 8) (b % 8) →
      Aliased.srcKey (F.Vs (F.sess p')).c
          (0 * (Rows.ofNet (F.Vs (F.sess p')).hU).nIn + rowColOf F.sh p' (128 * a' + e)) =
        .msg q'.val (b' / 8) (b' % 8) →
      b < Layout.portRowW (F.Vs (F.sess p)).c q → b' < Layout.portRowW (F.Vs (F.sess p')).c q' →
      (F.zs R ω t R₂ (F.sess p)).rd.isRead (.inl (F.un p r, q)) →
      (F.zs R ω t R₂ (F.sess p')).rd.isRead (.inl (F.un p' r, q')) →
      b = b' ∧ (F.zs R ω t R₂ (F.sess p)).rd.port (.inl (F.un p r, q)) =
          (F.zs R ω t R₂ (F.sess p')).rd.port (.inl (F.un p' r, q')) ∧
        (F.zs R ω t R₂ (F.sess p)).rd.readAt (.inl (F.un p r, q)) =
          (F.zs R ω t R₂ (F.sess p')).rd.readAt (.inl (F.un p' r, q'))

/-- Obligation: discharged by `SharedRows` (`rowsShared_of_sharedRows`, `rowsShared`); the transfer's `cr/sha-512` term
does not price it, since each of its finders sees one session's record.
**One registrant row per registered root and position.** At every accepted outer proof, two sessions' registered
reads at the same registered root and position carry the same registrant bits. Acceptance does not give it: each
session's `rd` is its own field, and `Reads.committed` makes both rows open one root at one position, which pins them
only up to a SHA-512 collision. It says that the family's sessions take the registrant's opening at a registered root
and position from one choice, the same in every session (as `RowsRegistered` says which rows they read). -/
def RowsShared : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ (s s' : Fin F.stmts.size)
    (u : Fin (F.Vs s).nv) (u' : Fin (F.Vs s').nv) (q : Fin (F.Vs s).c.ports.size) (q' : Fin (F.Vs s').c.ports.size),
    (F.zs R ω t R₂ s).rd.isRead (.inl (u, q)) → (F.zs R ω t R₂ s').rd.isRead (.inl (u', q')) →
    (F.zs R ω t R₂ s).rd.port (.inl (u, q)) = (F.zs R ω t R₂ s').rd.port (.inl (u', q')) →
    (F.zs R ω t R₂ s).rd.readAt (.inl (u, q)) = (F.zs R ω t R₂ s').rd.readAt (.inl (u', q')) →
    ∀ (b : ℕ) (hb : b < Layout.portRowW (F.Vs s).c q) (hb' : b < Layout.portRowW (F.Vs s').c q'),
      (Decode.regVal (F.zs R ω t R₂ s) (.inl (u, q))).2
          ⟨b, lt_of_lt_of_le hb ((Finset.le_sup (Finset.mem_univ q)).trans (Nat.le_succ _))⟩ =
        (Decode.regVal (F.zs R ω t R₂ s') (.inl (u', q'))).2
          ⟨b, lt_of_lt_of_le hb' ((Finset.le_sup (Finset.mem_univ q')).trans (Nat.le_succ _))⟩

/-- **`HRowsAgree` from the circuits' keys, the registered reads and one registrant row per root and position**
(`RowColKeys`, `RowsRegistered`, `SameRow`, `RowsShared`). -/
theorem hRowsAgree_of (hreg : F.RowsRegistered dg N k Rw) (hcol : F.RowColKeys dg N k Rw) (hsame : F.SameRow dg N k Rw)
    (hshared : F.RowsShared dg N k Rw) : F.HRowsAgree dg N k Rw := by
  intro R ω t R₂ τ o ha r p p' a a' e bit hp hp' ha₁ ha₁' heq he hrb
  obtain ⟨-, q, b, hb, hkey, hread, hval⟩ := hrb
  have hsz : (wIOf F.sh p').size = (nsOf F.sh p').sum := by rw [wIOf, nsOf, OfVStar.wIdx_size]
  obtain ⟨hlt', q', b', hb', hkey', hown'⟩ := hcol R ω t R₂ τ o ha p' hp' (128 * a' + e) (by omega)
  have hread' := hreg R ω t R₂ τ o ha (F.sess p') (F.un p' r) q' hown'
  obtain ⟨rfl, hport, hat⟩ :=
    hsame R ω t R₂ τ o ha r p p' a a' e hp hp' ha₁ ha₁' heq he q b q' b' hkey hkey' hb hb' hread hread'
  refine ⟨hlt', q', b, hb', hkey', hread', ?_⟩
  rw [← hval]
  exact (hshared R ω t R₂ τ o ha _ _ _ _ _ _ hread hread' hport hat b hb hb').symm

end VStarFamily

end family

end FlockVBridge.Family
