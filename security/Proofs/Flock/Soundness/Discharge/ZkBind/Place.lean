module

public import Proofs.Flock.Soundness.Discharge.ZkSession.Session
public import Proofs.Flock.Soundness.Discharge.Const.Inst
public import Proofs.Flock.Soundness.Discharge.Exec.Setup
public import Proofs.Flock.Soundness.Discharge.Placed.Setup
public import Proofs.Flock.Soundness.Discharge.ZkExec.Event
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
import Proofs.Flock.Soundness.Discharge.Aliased.Keys
import Proofs.Flock.Soundness.Discharge.Placed.Keyed
import Proofs.Flock.Soundness.Discharge.Placed.Wiring
import Proofs.Flock.Soundness.Discharge.ZkExec.Setup
import Proofs.Flock.Soundness.Lowering
public import Proofs.Flock.Soundness.Discharge.ZkExec.RunsLive

/-! `--zk` unit placements: each sampled proof block's `UnitPlace` from setupH's accept. -/

@[expose] public section

open Flock FlockLevel3

namespace FlockSoundness.Discharge.ZkBind

open Model Audit Placed ZkSession Exec ZkExec

theorem acceptsZK_accepted {I : Inputs} {t : Exec.Transcript} (hh : I.tags.hiddenOutputs = false)
    (ht : I.tags.typed = false) (h : AcceptsZK I t) :
    ∃ spec sts, setupOf (zkInputs I) t.record = .ok (spec, sts) ∧
      ∀ (j : ℕ) (hj : j < sts.size), Flock.Zk.shapeOk sts[j] = .ok () ∧
        ∀ regs : List (Model.Region sts[j].c.kLog (sts[j].m - sts[j].c.kLog)),
          ∃ (T : Model.Statement) (a : Accepted T), a.st = sts[j] ∧ HEq a.regs regs ∧
            a.tags = tagsAt (zkTags I) I.count j ∧ a.cf = I.circuitFile ∧ a.tables = I.tables ∧
            (I.count ≤ 1 → a.draw = recordDraw t.record) := by
  obtain ⟨spec, sts, hs, -, -, hall⟩ := acceptsZK_setupH I t hh h
  refine ⟨spec, sts, hs, fun j hj => ⟨(hall j hj).1, fun regs => ?_⟩⟩
  obtain ⟨d, hd, hdr⟩ := (hall j hj).2
  have ht' : (tagsAt (zkTags I) I.count j).typed = false :=
    (tagsAt_typed _ _ _).trans ((zkTags_typed I).trans ht)
  exact ⟨_, ⟨tagsAt (zkTags I) I.count j, I.circuitFile, I.publicFile, zkCoins I, I.tables, d, I.partition,
    I.program, sts[j], ht', hd, regs, rfl⟩, rfl, HEq.rfl, rfl, rfl, rfl, hdr⟩

section Sites

variable {F : Type} [Field F] [Fintype F] [DecidableEq F] {n : ℕ} {Reg : Type} {mPts : ℕ}
  (planZ : Finset (Fin n) → Reg → List (TabZK F mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZ S R).length)

/-- **Each sampled proof block's site in its `--zk` table**: the acceptance whose statement is the statement of the `--zk` table
unit `u` is sampled into (`acc`), and the unit's template instance (`vu`, below the statement's `vus_per_block`) and block
(`o`). -/
structure SitesZ where
  acc : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n), u ∈ S → Accepted (tabOfZ planZ tab S R u).S
  vu : ∀ (S : Finset (Fin n)) (_ : Reg) (u : Fin n), u ∈ S → ℕ
  hvu : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S), vu S R u h < (acc S R u h).st.c.g
  o : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n), u ∈ S → Fin (2 ^ (tabOfZ planZ tab S R u).S.nbl)

variable {planZ tab} {c : Flock.Circuit} (hU : Built c.unit) (si : SitesZ planZ tab)
  (hparse : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
    HmRow.parse (si.acc S R u h).tags (si.acc S R u h).cf (si.acc S R u h).tables = .ok c)
  (outs : Finset (Fin (gates (inGates n (circuitRows c hU).nIn) (circuitRows c hU) n)))

include hparse in
theorem SitesZ.rowsOf_keyProg (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S) :
    (keyProg c hU n).rows u = (si.acc S R u h).rows := by
  exact (rows_circuitProg c hU n _ _ u).trans ((si.acc S R u h).rows_eq (hparse S R u h) hU).symm

/-- A sampled proof block's columns in its `--zk` table's block: its template instance's (`Accepted.col`). -/
noncomputable def SitesZ.colKey (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S) :
    Fin ((keyProg c hU n).rows u).w → Fin (2 ^ (tabOfZ planZ tab S R u).S.kLog) :=
  fun x => (si.acc S R u h).col (si.hvu S R u h) (Fin.cast (congrArg Rows.w (si.rowsOf_keyProg hU hparse S R u h)) x)

theorem SitesZ.placed_keyProg (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S) :
    Placement (tabOfZ planZ tab S R u).S ((keyProg c hU n).rows u) (si.colKey hU hparse S R u h) := by
  exact placement_rows (si.rowsOf_keyProg hU hparse S R u h) ((si.acc S R u h).placement (si.hvu S R u h))

theorem SitesZ.aliased_keyProg (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S) :
    ∀ x x', (Const.Prog.instC (keyProg c hU n) outs u).wire x = (Const.Prog.instC (keyProg c hU n) outs u).wire x' →
      x.val < ((keyProg c hU n).rows u).nIn → x'.val < ((keyProg c hU n).rows u).nIn →
      ∀ z : Model.Witness (tabOfZ planZ tab S R u).S.m, (tabOfZ planZ tab S R u).S.Satisfies z →
        (tabOfZ planZ tab S R u).S.block z (si.o S R u h) (si.colKey hU hparse S R u h x) =
          (tabOfZ planZ tab S R u).S.block z (si.o S R u h) (si.colKey hU hparse S R u h x') := by
  refine (si.acc S R u h).aliased_of_layout (si.rowsOf_keyProg hU hparse S R u h)
    (Const.Prog.instC (keyProg c hU n) outs u) (si.o S R u h) (si.hvu S R u h) ?_
  have hc := (si.acc S R u h).c_eq (hparse S R u h)
  have e := congrArg Rows.nIn (si.rowsOf_keyProg hU hparse S R u h)
  refine Aliased.setupH_aliased_of_key (si.acc S R u h).ht (si.acc S R u h).h (si.acc S R u h).regs _ _
    (si.hvu S R u h) ?_
  intro x x' hw hx hx'
  have hw' := ((wire_cast (si.rowsOf_keyProg hU hparse S R u h) (Const.Prog.instC (keyProg c hU n) outs u) x).symm.trans
    hw).trans (wire_cast (si.rowsOf_keyProg hU hparse S R u h) (Const.Prog.instC (keyProg c hU n) outs u) x')
  have hk := keyProg_wire_key hU outs u _ _ (lt_of_lt_of_eq hx e.symm) (lt_of_lt_of_eq hx' e.symm) hw'
  exact (congrArg (fun d => Aliased.srcKey d x.val = Aliased.srcKey d x'.val) hc).mpr hk

/-- **Each sampled proof block's place in its `--zk` table, for the circuit's program on Δ's source keys**: at its template
instance and block in its table's accepted statement, from the sites and the parse alone. -/
noncomputable def SitesZ.placeKey (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S) :
    UnitPlace ((keyProg c hU n).partition outs) (tabOfZ planZ tab S R u).S u where
  R := (keyProg c hU n).rows u
  inst := Const.Prog.instC (keyProg c hU n) outs u
  o := si.o S R u h
  col := si.colKey hU hparse S R u h
  placed := (si.placed_keyProg hU hparse S R u h).sat
  aliased := si.aliased_keyProg hU hparse outs S R u h

theorem SitesZ.placeKey_spec (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S) :
    (si.placeKey hU hparse outs S R u h).R = (keyProg c hU n).rows u ∧
      HEq (si.placeKey hU hparse outs S R u h).inst (Const.Prog.instC (keyProg c hU n) outs u) ∧
      (si.placeKey hU hparse outs S R u h).o = si.o S R u h := by
  exact ⟨rfl, HEq.rfl, rfl⟩

end Sites

end FlockSoundness.Discharge.ZkBind
