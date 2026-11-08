module

public import Proofs.Flock.Recursive.InnerRows
public import Proofs.Flock.Bridge.Lowers
public import Proofs.Flock.Soundness.Discharge.Const.Inst
public import Proofs.Flock.Soundness.Discharge.Placed.Keyed
public import Proofs.Flock.Soundness.Lowering
import Proofs.Flock.Claims.Scope.Fit
import Proofs.Flock.Claims.Scope.InScope
import Proofs.Flock.Claims.Scope.Setup
import Proofs.Flock.Soundness.Discharge.FailClosed.Scope
import Proofs.Flock.Soundness.Discharge.Hidden.Committed
import Proofs.Flock.Soundness.Discharge.Hidden.Delta
import Proofs.Flock.Soundness.Discharge.Hidden.Spec
import Proofs.Flock.Soundness.Discharge.Hidden.Unrolled
import Proofs.Flock.Soundness.Discharge.HiddenExec.Accepts
import Proofs.Flock.Soundness.Discharge.HiddenExec.SiteH
import Proofs.Flock.Soundness.Discharge.Placed.Setup
import Proofs.Flock.Soundness.Discharge.Placed.Wiring

@[expose] public section

set_option Elab.async false

/-!
# Each unit of a session's table, placed and lowered

`accepted_check`: at a statement the verifier parsed whose statement check passes (`Read.check`, the first half of its
scope refusal), with hidden outputs, each table `j` has an accepted statement of its own (`HiddenExec.AcceptedH`),
fixed by the statement alone, whose block R1CS every witness of the deployed relation satisfies with table `j`'s share
of its hidden values (`Readable.zOf`). It is `InnerRows.rows_commit_check`'s setup: each table's `Stmt.setupHidden`
run, its rows' fit, its layout, and `relS_relation`. `bit_block_zOf` reads that share's bits as `InnerRows.bitAt`, so
`unitRows_check`: each template instance's rows (`AcceptedH.rows`: the unit net's rows stacked `units_per_vu` times)
hold at table `j`'s bits, in every block, in the columns the statement places them (`AcceptedH.col`).

**A unit of the circuit's program** (`Placed.keyProg c hU n`, whose lowering is a member's program). Where the accepted
statement's circuit unrolls to `c`, `unitPlace` places unit `u` at template instance `g` of block `o`: its rows are the
statement's (`rows_keyProg`, W6 by definition), its instance the program's own (`Const.Prog.instC`), and its inputs on
one gate carry one value, since they have one source key (`aliased_keyProg`, as `SitesH.aliased_keyProg`). So at every
witness the table's statement accepts, the values the unit's block decodes satisfy each of its gates
(`UnitPlace.apply_decode`, with `UnitPlace.correct`'s premises), and Bridge's lowering computes them at each wire's
position from committed values that carry them where the unit reads from outside (`runIn_unitPlace`).
-/

namespace SpotFlock

open FlockSoundness FlockSoundness.Discharge Exec Flock Flock.Readable

/-- **Each table of a parsed, checked, hidden-output statement has an accepted statement its share satisfies**: table
`j` has an accepted statement `a` on `p.sts[j]` whose block R1CS `T` every witness of the deployed relation satisfies
with table `j`'s share of its hidden values. -/
theorem accepted_check {p : Parsed} (hP : Parses p) (hc : p.check = .ok ()) (hh : p.tags.hiddenOutputs = true)
    {j : ℕ} (hj : j < p.sts.size) :
    ∃ (T : Model.Statement) (a : HiddenExec.AcceptedH T), a.st = p.sts[j] ∧ T.kLog = p.sts[j].c.kLog ∧
      ∀ w : Claims.Flock.Wit, Claims.Flock.RelS (toReadable relation p) w → T.Satisfies (zOf p.sts.size w j T.m) := by
  obtain ⟨hRu, c0, -, hc0, -, -, hok⟩ := (check_ok_iff _ _ _ _ _ _ _ _).1 hc
  have hpi : p.publicInputs = none := Option.not_isSome_iff_eq_none.1 (by simp [hRu.2.2.2.2.2])
  have ht : p.tags.typed = false := hRu.2.1
  obtain ⟨spec, hset⟩ := setup_of_tablesOf (tablesOf_of_parses hP)
  unfold Read.setup at hset
  rw [hpi] at hset
  obtain ⟨-, hall⟩ := HiddenExec.setupTables_setupOne hset
  obtain ⟨d, hd, -⟩ := hall j hj
  have ht' : (tagsAt p.tags p.I.count j).typed = false := (tagsAt_typed _ _ _).trans ht
  unfold HiddenExec.setupOne at hd
  rw [tagsAt_hiddenOutputs, hh] at hd
  simp only [↓reduceIte] at hd
  have hsatAll : ∀ w : Claims.Flock.Wit, Claims.Flock.RelS (toReadable relation p) w →
      ∃ hf : (Tab.ofStmt p.sts[j]!).Fits (toReadable relation p).m (toReadable relation p).kLog,
        ((Tab.ofStmt p.sts[j]!).toModel hf.2.2).Satisfies (zOf p.sts.size w j p.sts[j]!.m) :=
    fun w hR => (relS_relation p w).1 hR j hj
  have e! : p.sts[j]! = p.sts[j] := getElem!_pos p.sts j hj
  revert hsatAll
  rw [e!]
  generalize hst : p.sts[j] = st at hd ⊢
  intro hsatAll
  have s := Hidden.setupHidden_spec' hd
  have hcst : HmOut.parseCommit p.tags p.I.circuitFile p.I.tables = .ok st.c :=
    (parseCommit_tagsAt ht _ _ _ _).symm.trans s.parse
  have hc0' : c0 = st.c := by
    unfold ProvedScope.circuitOf at hc0
    rw [hh] at hc0
    simp only [↓reduceIte] at hc0
    exact Except.ok.inj (hc0.symm.trans hcst)
  rw [hc0'] at hok
  have hrows : (Layout.v1Ports st.c || Layout.bitRows st.c) = true := by
    unfold FormOk at hok
    split at hok
    · unfold ProvedScope.scopeOkT at hok
      simp only [Bool.and_eq_true] at hok
      have e : ProvedScope.v2Row = Layout.v2Row := rfl
      have e' : ProvedScope.segRow = Layout.segRow := rfl
      have hr := hok.2
      simp only [ProvedScope.v1Ports, Layout.v1Ports, ProvedScope.v1Rows, Layout.v1Rows, ProvedScope.unpinned,
        Layout.unpinned, ProvedScope.bitRows, Layout.bitRows, ProvedScope.bitRowsU, Layout.bitRowsU, e, e'] at hr ⊢
      exact hr
    · rw [hh] at hok
      simp only [↓reduceIte] at hok
      rw [FailClosed.scopeOkI_eq] at hok
      unfold Layout.scopeOkI at hok
      simp only [Bool.and_eq_true] at hok
      exact hok.2.2
  have hC := (Hidden.setupHidden_specCC hd).committed
  have hrf : Hidden.Any.RowsFitS st.c := HiddenExec.rowsFit_of_committed ht' hC hrows
  have L : FlockSoundness.Layout st :=
    (Hidden.setupHidden_specCC hd).layout ht' (Hidden.setupHidden_unpinned ht' hd hrf.1.1)
  have hu : Hidden.unpin st = st :=
    Flock.InnerRows.unpin_self s.da s.db (Hidden.setupHidden_unpinned ht' hd hrf.1.1)
  have hT : (Hidden.unpin_layout ht' hd).model (Refine.regionsOf st) = L.model (Refine.regionsOf st) :=
    Flock.InnerRows.model_congr hu L _ _ _ HEq.rfl
  refine ⟨L.model (Refine.regionsOf st),
    ⟨tagsAt p.tags p.I.count j, p.I.circuitFile, p.I.publicFile, p.coins, p.I.tables, d, p.I.partition, p.I.program,
      st, ht', Hidden.setupHidden_specCC hd, Hidden.unpin_layout ht' hd,
      HiddenExec.unitRowsFlat_unpin ht' hd hrf.1.toP, Refine.regionsOf st, hT⟩, rfl, rfl, fun w hR => ?_⟩
  obtain ⟨_, hsat⟩ := hsatAll w hR
  exact hsat

/-- **The model's bit at a block's column is the witness's** (`InnerRows.bitAt`): table `j`'s share of the hidden values
(`Readable.zOf`), at block `o` and column `y`, wherever the statement's blocks are table `j`'s. -/
theorem bit_block_zOf {T : Model.Statement} (p : Parsed) (w : Claims.Flock.Wit) (j : ℕ)
    (hk : T.kLog = p.sts[j]!.c.kLog) (o : Fin (2 ^ T.nbl)) (y : Fin (2 ^ T.kLog)) :
    FlockSoundness.bit (T.block (zOf p.sts.size w j T.m) o y) = Flock.InnerRows.bitAt p w j o y := by
  show FlockSoundness.bit (toZ (w.hid.getD ((o.val * 2 ^ T.kLog + y.val) * p.sts.size + j) false)) = _
  rw [Flock.InnerRows.bit_toZ]
  unfold Flock.InnerRows.bitAt
  rw [congrArg (fun k => o.val * 2 ^ k + y.val) hk]

/-- **Each template instance's rows hold at the session witness's bits**: at a statement the verifier parsed whose
statement check passes, with hidden outputs, table `j` has an accepted statement `a` on `p.sts[j]`, and at every witness
of the deployed relation, every template instance `g` and every block `o`, the template instance rows hold at table `j`'s
bits of block `o` in the template instance's columns. -/
theorem unitRows_check {p : Parsed} (hP : Parses p) (hc : p.check = .ok ()) (hh : p.tags.hiddenOutputs = true)
    {j : ℕ} (hj : j < p.sts.size) :
    ∃ (T : Model.Statement) (a : HiddenExec.AcceptedH T), a.st = p.sts[j] ∧ T.kLog = p.sts[j].c.kLog ∧
      ∀ w : Claims.Flock.Wit, Claims.Flock.RelS (toReadable relation p) w →
        ∀ (g : ℕ) (hg : g < a.st.c.g) (o : Fin (2 ^ T.nbl)),
          a.rows.Sat fun y => Flock.InnerRows.bitAt p w j o (a.col hg y) := by
  obtain ⟨T, a, hst, hk, hsat⟩ := accepted_check hP hc hh hj
  refine ⟨T, a, hst, hk, fun w hR g hg o => ?_⟩
  have hk' : T.kLog = p.sts[j]!.c.kLog := by rw [hk, getElem!_pos p.sts j hj]
  simpa only [bit_block_zOf p w j hk'] using (a.placement hg).sat.rowsSat (hsat w hR) o

/-! ## A unit of the circuit's program, placed and lowered -/

/-- **The decoded values satisfy each of the unit's gates**: at a witness the statement accepts, each gate of a placed
unit holds its operation applied to the values the unit's block decodes. -/
theorem _root_.FlockSoundness.UnitPlace.apply_decode {C : Audit.Circuit} {n : ℕ} {P : Audit.Partition C n}
    {S : Model.Statement} {u : Fin n} (pl : UnitPlace P S u) {z : Model.Witness S.m} (hz : S.Satisfies z) (g : Fin C.N)
    (hg : P.unit g = some u) : pl.decode z g = (C.op g).apply (pl.decode z) := by
  obtain ⟨i, rfl⟩ := pl.inst.onto g hg
  have ha := pl.agree hz
  show pl.inst.decode _ _ = _
  rw [pl.inst.op i, pl.inst.decode_wire ha]
  refine (pl.placed.rowsSat hz pl.o i).trans ?_
  simp only [Audit.Op.apply, Audit.xorSum_map]
  congr 1 <;> exact Audit.xorSum_congr fun c _ => (pl.inst.decode_wire ha c).symm

variable {c : _root_.Flock.Circuit} (hU : Built c.unit) {n : ℕ}
  (outs : Finset (Fin (Placed.gates (Placed.inGates n (Placed.circuitRows c hU).nIn) (Placed.circuitRows c hU) n)))
  {T : Model.Statement} (a : HiddenExec.AcceptedH T) (hparse : HmOut.unrolled a.st.c = c)

include hparse in
/-- **W6 at one accepted statement**: each unit's rows in the circuit's program are the statement's template instance
rows. -/
theorem rows_keyProg (u : Fin n) : (Placed.keyProg c hU n).rows u = a.rows :=
  (Placed.rows_circuitProg c hU n _ _ u).trans (a.rows_eqU hparse hU).symm

include hparse in
/-- **A unit's inputs on one gate have one source key**, so they are aliased in the accepted statement's layout at any
template instance and block (`AcceptedH.aliased_of_key`, as `SitesH.aliased_keyProg`). -/
theorem aliased_keyProg (u : Fin n) {g : ℕ} (hg : g < a.st.c.g) (o : Fin (2 ^ T.nbl)) :
    a.L.Aliased a.regs (rows_keyProg hU a hparse u ▸ Const.Prog.instC (Placed.keyProg c hU n) outs u)
      (Fin.cast a.nbl_eq.symm o) hg := by
  have e := congrArg Rows.nIn (rows_keyProg hU a hparse u)
  refine a.aliased_of_key _ _ hg ?_
  intro x x' hw hx hx'
  have hw' := ((Placed.wire_cast (rows_keyProg hU a hparse u)
    (Const.Prog.instC (Placed.keyProg c hU n) outs u) x).symm.trans hw).trans
    (Placed.wire_cast (rows_keyProg hU a hparse u) (Const.Prog.instC (Placed.keyProg c hU n) outs u) x')
  have hk := Placed.keyProg_wire_key hU outs u _ _ (lt_of_lt_of_eq hx e.symm) (lt_of_lt_of_eq hx' e.symm) hw'
  have key : ∀ a' b : ℕ, Aliased.srcKey c a' = Aliased.srcKey c b →
      Aliased.srcKey (HmOut.unrolled a.st.c) a' = Aliased.srcKey (HmOut.unrolled a.st.c) b :=
    fun a' b hab => by rw [hparse]; exact hab
  exact Hidden.srcKey_eq_commit a.ht a.committed a.unitFacts.check (key _ _ hk)

/-- **Unit `u` of the circuit's program, placed at template instance `g` of block `o`** of one accepted statement whose
circuit unrolls to `c`. -/
noncomputable def unitPlace (u : Fin n) {g : ℕ} (hg : g < a.st.c.g) (o : Fin (2 ^ T.nbl)) :
    UnitPlace ((Placed.keyProg c hU n).partition outs) T u where
  R := (Placed.keyProg c hU n).rows u
  inst := Const.Prog.instC (Placed.keyProg c hU n) outs u
  o := o
  col x := a.col hg (Fin.cast (congrArg Rows.w (rows_keyProg hU a hparse u)) x)
  placed := (Placed.placement_rows (rows_keyProg hU a hparse u) (a.placement hg)).sat
  aliased := a.aliased_of_layout (rows_keyProg hU a hparse u) _ o hg (aliased_keyProg hU outs a hparse u hg o)

/-- **Bridge's lowering computes a placed unit's decoded values**: at a witness the statement accepts, on committed
values that carry the decoded values wherever the unit's lowered gates read from outside it, running the unit's lowered
gates gives each of its wires' decoded value at the wire's position. -/
theorem runIn_unitPlace {κ : Fin ((Placed.keyProg c hU n).circuit outs).N → Option Bool} (u : Fin n) {g : ℕ}
    (hg : g < a.st.c.g) (o : Fin (2 ^ T.nbl)) {z : Model.Witness T.m} (hz : T.Satisfies z) (ref : ℕ → Option Bool)
    (hread : ∀ p ∈ Bridge.unitGates ((Placed.keyProg c hU n).partition outs) u, ∀ x,
      (Bridge.progOf ((Placed.keyProg c hU n).circuit outs) κ).gates[p]? = some x →
      ∀ a' : Fin ((Placed.keyProg c hU n).circuit outs).N,
        Bridge.posOf ((Placed.keyProg c hU n).circuit outs) a' ∈ x.reads →
        Bridge.posOf ((Placed.keyProg c hU n).circuit outs) a' ∉
          Bridge.unitGates ((Placed.keyProg c hU n).partition outs) u →
        (Bridge.progOf ((Placed.keyProg c hU n).circuit outs) κ).read ref
            (Bridge.posOf ((Placed.keyProg c hU n).circuit outs) a') =
          some ((unitPlace hU outs a hparse u hg o).decode z a'))
    (w : Fin ((Placed.keyProg c hU n).circuit outs).N) (hw : ((Placed.keyProg c hU n).partition outs).unit w = some u) :
    (Bridge.progOf ((Placed.keyProg c hU n).circuit outs) κ).runIn ref
        (Bridge.unitGates ((Placed.keyProg c hU n).partition outs) u)
        (Bridge.posOf ((Placed.keyProg c hU n).circuit outs) w) =
      some ((unitPlace hU outs a hparse u hg o).decode z w) :=
  Bridge.runIn_posOf_of (κ := κ) _ u _ ((unitPlace hU outs a hparse u hg o).apply_decode hz) ref hread _ w rfl hw

end SpotFlock
