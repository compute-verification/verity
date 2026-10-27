import Proofs.Flock.Soundness.Discharge.HiddenExec.Placed
import Proofs.Flock.Soundness.Discharge.HiddenExec.PinnedLinear

/-!
# The sites of a one-table hidden-output session

Integrate's `sitesAt` and `sitesAt_parse` (`Integrate/Sites.lean`) at the hidden tables (`planAtH`, `tabAtH`). A drawn
table's acceptance (`TableH.acc`) is its `Stmt.setupHidden` run, with the layout `Hidden.unpin_layout` gives of its
statement without Δ's pinned copies, its unit slots' rows its flat twin's
(`unitRowsFlat_unpin`, at rows that fit), and the table's statement `stmtOf` of it, whose twin without the pinned copies
(`TableH.twinA`, `twinB`) is the layout's block R1CS (`Integrate.model_eq`) and holds at every witness (`TableH.sat`).

`sitesAtH` holds each drawn unit's acceptance, template instance and block; every acceptance parses the session's one
circuit file (`sitesAtH_parse`). So `ppH`, `hConstH` and `hZeroH` are `ProgPlaces`, `hConst` and `hZero` for
`Placed.keyProg` at the hidden tables.
-/

set_option Elab.async false

namespace FlockSoundness.Discharge.HiddenExec

open Flock FlockLevel3 Model Refine Audit
open FlockSoundness.Discharge.Exec (Inputs)

variable {n : ℕ} {I : Inputs} {mPts : ℕ}
  (ht : I.tags.typed = false) {c : Flock.Circuit} (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c)
  (hv1 : Hidden.Any.RowsFitS c)

/-- `stmtOf` of `Hidden.unpin st` is the twin, at matrices read off `unpin`, of a statement equal to `stmtOf st`. -/
theorem twin_of_eq {T : Model.Statement} {st : Stmt} {hwf : StmtWF st}
    {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} (hT : T = stmtOf st hwf regs) :
    stmtOf (Hidden.unpin st) (Hidden.unpin_wf hwf) regs =
      twin T (fun r j => matA (Hidden.unpin st) r.val j.val) (fun r j => matB (Hidden.unpin st) r.val j.val) := by
  subst hT
  rfl

/-- **With every pinned bit a linear row, a witness of a statement equal to `stmtOf st` satisfies its `unpin` twin**
(`sat_pinned`). -/
theorem twin_sat_of_linear {tags : Tags} {T : Model.Statement} {st : Stmt} {hwf : StmtWF st}
    {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} (hT : T = stmtOf st hwf regs)
    (L : FlockSoundness.Layout (Hidden.unpin st)) (F : Aliased.UnitFacts tags (Hidden.flatTwin st))
    (hleaf : st.leaf = default) (hda : st.da = (HmOut.delta st.c st.pin).1)
    (hdb : st.db = (HmOut.delta st.c st.pin).2) (hlin : ∀ y ∈ Hidden.pinnedCols st.c, LinearBit st.c y)
    {z : Model.Witness T.m} (hz : T.Satisfies z) :
    (twin T (fun r j => matA (Hidden.unpin st) r.val j.val) (fun r j => matB (Hidden.unpin st) r.val j.val)).Satisfies
      z := by
  subst hT
  exact sat_pinned L F hleaf hda hdb hwf hlin regs hz

namespace TableH

variable {S : Finset (Fin n)} (x : TableH I mPts S)

include hc in
/-- A drawn table's statement is on the session's one circuit. -/
theorem c_eq : x.st.c = c := Except.ok.inj ((setupHidden_parseC x.h).symm.trans hc)

/-- **A drawn table's statement without Δ's pinned copies** (`Hidden.unpin`), its matrices at the table's blocks. -/
noncomputable def twinA : Matrix (Fin (2 ^ x.T.S.kLog)) (Fin (2 ^ x.T.S.kLog)) (ZMod 2) :=
  fun r j => matA (Hidden.unpin x.st) r.val j.val

noncomputable def twinB : Matrix (Fin (2 ^ x.T.S.kLog)) (Fin (2 ^ x.T.S.kLog)) (ZMod 2) :=
  fun r j => matB (Hidden.unpin x.st) r.val j.val

/-- **A drawn table's acceptance**: its `Stmt.setupHidden` run, with the layout `Hidden.unpin_layout` gives and its
unit slots' rows its flat twin's (`unitRowsFlat_unpin`); the table's statement without Δ's pinned copies is the
layout's block R1CS at the table's regions (`Integrate.model_eq`, and `TableH.hT`). -/
noncomputable def acc : AcceptedH (twin x.T.S x.twinA x.twinB) where
  tags := I.tags
  cf := I.circuitFile
  pf := I.publicFile
  coins := I.coins
  tables := I.tables
  draw := some x.d
  partition := I.partition
  program := I.program
  st := x.st
  ht := ht
  h := x.h
  L := Hidden.unpin_layout ht x.h
  hR := unitRowsFlat_unpin ht x.h (by rw [x.c_eq hc]; exact hv1.1.toP)
  regs := x.regs
  hT := (Integrate.model_eq (st := Hidden.unpin x.st) (parse_facts (parse_untyped ht (setupHidden_parseC x.h)).1)
    (Hidden.unpin_layout ht x.h)
    (Hidden.setupHidden_spec x.h).leaf (Hidden.unpin_wf x.hwf) x.regs).trans (twin_of_eq x.hT)

include ht in
/-- **A witness of a drawn table satisfies its statement without Δ's pinned copies**, pinned outputs or not: every pinned
bit is a linear row (`pinned_linear`, what `HmOut.checkPinned` checks), so `twin_sat_of_linear` holds. -/
theorem sat (z : Model.Witness x.T.S.m) (hz : x.T.S.Satisfies z) : (twin x.T.S x.twinA x.twinB).Satisfies z :=
  have hp := parse_untyped ht (setupHidden_parseC x.h)
  have P := parse_facts hp.1
  twin_sat_of_linear x.hT (Hidden.unpin_layout ht x.h) (flat_unitFacts ht x.h) (Hidden.setupHidden_spec x.h).leaf
    (Hidden.setupHidden_spec x.h).da (Hidden.setupHidden_spec x.h).db
    (pinned_linear (Hidden.checkPinned_of_checkRows hp.2) P.typed P.nets) hz

end TableH

variable (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetupH I mPts S₀ (dj S₀)) {Reg : Type}

/-- **The sites of the one-table hidden plan**: each drawn unit's acceptance, template instance and block are its draw's
table's. -/
noncomputable def sitesAtH : SitesH (planAtH (Reg := Reg) dj x₀) (tabAtH dj x₀) where
  A S _ _ _ := (tableAtH dj x₀ S).twinA
  B S _ _ _ := (tableAtH dj x₀ S).twinB
  sat S _ _ _ := (tableAtH dj x₀ S).sat ht
  acc S _ _ _ := (tableAtH dj x₀ S).acc ht hc hv1
  vu S _ u h := (tableAtH dj x₀ S).vu u h
  hvu S _ u h := (tableAtH dj x₀ S).hvu u h
  o S _ u h := (tableAtH dj x₀ S).o u h

/-- **One circuit per session**: every site's acceptance parses the session's one circuit file to `c`. -/
theorem sitesAtH_parse :
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
      HmRow.parse ((sitesAtH ht hc hv1 dj x₀).acc S R u h).tags ((sitesAtH ht hc hv1 dj x₀).acc S R u h).cf
        ((sitesAtH ht hc hv1 dj x₀).acc S R u h).tables = .ok c :=
  fun _ _ _ _ => (parse_untyped ht hc).1

variable (hU : Built c.unit)
  (outs : Finset (Fin (Placed.gates (Placed.inGates n (Placed.circuitRows c hU).nIn) (Placed.circuitRows c hU) n)))

/-- **`pp` at the hidden tables**: `SitesH.progPlaces_keyProg` at `sitesAtH`. -/
noncomputable abbrev ppH : ProgPlaces (Placed.keyProg c hU n) outs (planAtH (Reg := Reg) dj x₀) (tabAtH dj x₀) :=
  SitesH.progPlaces_keyProg hU (sitesAtH_parse ht hc hv1 dj x₀) outs

/-- **`hConst` at the hidden tables** (`constCols_keyProgH`): only a unit's constant column sits on the one gate. -/
theorem hConstH :
    ((Placed.keyProg c hU n).partition outs).ConstCols (planAtH (Reg := Reg) dj x₀) (tabAtH dj x₀)
      (ppH ht hc hv1 dj x₀ hU outs).derived.place
      {Fin.castLE (Placed.le_gates _ _ n) (Placed.oneGate n (Placed.circuitRows c hU).nIn)} :=
  constCols_keyProgH hU (sitesAtH_parse ht hc hv1 dj x₀) outs

/-- **`hZero` at the hidden tables** (`zeroCols_keyProgH`): every column on the zero gate carries 0 in every satisfying
witness. -/
theorem hZeroH :
    ((Placed.keyProg c hU n).partition outs).ZeroCols (planAtH (Reg := Reg) dj x₀) (tabAtH dj x₀)
      (ppH ht hc hv1 dj x₀ hU outs).derived.place
      {Fin.castLE (Placed.le_gates _ _ n) (Placed.zeroGate n (Placed.circuitRows c hU).nIn)} :=
  zeroCols_keyProgH hU (sitesAtH_parse ht hc hv1 dj x₀) outs

end FlockSoundness.Discharge.HiddenExec
