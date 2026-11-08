module

public import Proofs.Flock.Recursive.InnerRows
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

@[expose] public section

set_option Elab.async false

/-!
# Each template instance's rows hold at the session witness's bits

`unitRows_check`: at a statement the verifier parsed whose statement check passes (`Read.check`, the first half of its
scope refusal), with hidden outputs, each table `j` has an accepted statement of its own (`HiddenExec.AcceptedH`), fixed
by the statement alone, whose template instance rows (`AcceptedH.rows`: the unit net's rows stacked `units_per_vu` times)
hold at every witness of the deployed relation, in every block `o` and at every template instance `g`, at table `j`'s
bits there (`InnerRows.bitAt`) in the columns the statement places them (`AcceptedH.col`).

It is `InnerRows.rows_commit_check`'s setup (each table's `Stmt.setupHidden` run, its rows' fit, its layout, and the
witness's share satisfying the layout's block R1CS), followed by the placement (`AcceptedH.placement`,
`PlacedSat.rowsSat`) where that theorem follows the row hash. The rows are the statement circuit's
(`AcceptedH.rows_eqU`), so a decoder that reads a unit's wires at those columns (`decOf_bitAt`) reads values that
satisfy them.
-/

namespace SpotFlock

open FlockSoundness FlockSoundness.Discharge Exec Flock Flock.Readable

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
  unfold Flock.InnerRows.bitAt
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
  let a : HiddenExec.AcceptedH (L.model (Refine.regionsOf st)) :=
    ⟨tagsAt p.tags p.I.count j, p.I.circuitFile, p.I.publicFile, p.coins, p.I.tables, d, p.I.partition, p.I.program, st,
      ht', Hidden.setupHidden_specCC hd, Hidden.unpin_layout ht' hd, HiddenExec.unitRowsFlat_unpin ht' hd hrf.1.toP,
      Refine.regionsOf st, hT⟩
  refine ⟨L.model (Refine.regionsOf st), a, rfl, rfl, fun w hR g hg o => ?_⟩
  obtain ⟨_, hsat⟩ := hsatAll w hR
  have hpl := (a.placement hg).sat.rowsSat hsat o
  have hfun : (fun y => FlockSoundness.bit
      ((L.model (Refine.regionsOf st)).block (zOf p.sts.size w j st.m) o (a.col hg y))) =
      fun y => w.hid.getD ((o.val * 2 ^ st.c.kLog + (a.col hg y).val) * p.sts.size + j) false := by
    funext y
    show FlockSoundness.bit (toZ (w.hid.getD ((o.val * 2 ^ st.c.kLog + (a.col hg y).val) * p.sts.size + j) false)) = _
    exact Flock.InnerRows.bit_toZ _
  rw [← hfun]
  exact hpl

end SpotFlock
