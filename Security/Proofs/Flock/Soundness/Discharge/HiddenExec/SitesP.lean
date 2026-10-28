import Proofs.Flock.Soundness.Discharge.HiddenExec.TablesP
import Proofs.Flock.Soundness.Discharge.HiddenExec.PlacedP

set_option Elab.async false

/-!
# The sites of a one-table hidden-output session at public inputs

`Sites.lean` at the verifier's public inputs `own`. A drawn table's acceptance (`TableHP.acc`) is its
`Stmt.setupHidden … own` run with its `SpecP` (`Hidden.setupHidden_specP`), the layout `Hidden.hidden_layoutP` gives,
its unit slots' kept rows its flat twin's (`unitRowsFlatP`), and the table's statement `stmtOf` of it, the layout's
block R1CS (`Integrate.model_eq`).

`sitesAtHP` holds each drawn unit's acceptance, template instance and block; every acceptance parses the session's one
circuit file (`sitesAtHP_parse`). So `ppHP`, `hConstHP` and `hZeroHP` are `ProgPlaces`, `hConst` and `hZero` for
`Placed.keyProgP` at the hidden tables at `own`.
-/

namespace FlockSoundness.Discharge.HiddenExec

open Flock FlockLevel3 Model Refine Audit
open FlockSoundness.Discharge.Exec (Inputs)

variable {n : ℕ} {I : Inputs} {own : Option ByteArray} {mPts : ℕ}
  (ht : I.tags.typed = false) {c : Flock.Circuit} (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c)
  (hv1 : Hidden.Any.RowsFitS c)

namespace TableHP

variable {S : Finset (Fin n)} (x : TableHP I own mPts S)

/-- A drawn table's `SpecP`, at its circuit's public input ports. -/
theorem specP : Hidden.SpecP I.tags I.circuitFile I.publicFile I.coins I.tables (some x.d) own
    (Hidden.setupHidden_specP x.h).choose x.st :=
  (Hidden.setupHidden_specP x.h).choose_spec

include hc in
/-- A drawn table's statement is on the session's one circuit. -/
theorem c_eq : x.st.c = c := Except.ok.inj (x.specP.parse.symm.trans hc)

/-- **A drawn table's acceptance at `own`** (`TableH.acc`): its `Stmt.setupHidden … own` run and `SpecP`, with the
layout `Hidden.hidden_layoutP` gives and its unit slots' kept rows its flat twin's (`unitRowsFlatP`); the table's
statement is the layout's block R1CS at the table's regions (`Integrate.model_eq`, and `TableHP.hT`). -/
noncomputable def acc : AcceptedHP x.T.S where
  tags := I.tags
  cf := I.circuitFile
  pf := I.publicFile
  coins := I.coins
  tables := I.tables
  draw := some x.d
  partition := I.partition
  program := I.program
  own := own
  st := x.st
  ht := ht
  h := x.h
  pi := (Hidden.setupHidden_specP x.h).choose
  s := x.specP
  L := Hidden.hidden_layoutP ht x.specP (Hidden.unpinned_of_rep (Hidden.parse_untyped ht x.specP.parse).1
    (Hidden.parse_untyped ht x.specP.parse).2 (by rw [x.c_eq hc]; exact hv1.1.1))
  hR := unitRowsFlatP ht x.specP (by rw [x.c_eq hc]; exact hv1.1)
  regs := x.regs
  hregs := x.hregs
  hT := (Integrate.model_eq (parse_facts (Hidden.parse_untyped ht x.specP.parse).1) _ x.specP.leaf x.hwf
    x.regs).trans x.hT.symm

end TableHP

variable (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetupHP I own mPts S₀ (dj S₀))
  {Reg : Type}

/-- **The sites of the one-table hidden plan at `own`** (`sitesAtH`). -/
noncomputable def sitesAtHP : SitesHP (planAtHP (Reg := Reg) dj x₀) (tabAtHP dj x₀) where
  acc S _ _ _ := (tableAtHP dj x₀ S).acc ht hc hv1
  vu S _ u h := (tableAtHP dj x₀ S).vu u h
  hvu S _ u h := (tableAtHP dj x₀ S).hvu u h
  o S _ u h := (tableAtHP dj x₀ S).o u h

/-- **One circuit per session**: every site's acceptance parses the session's one circuit file to `c`. -/
theorem sitesAtHP_parse :
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
      HmRow.parse ((sitesAtHP ht hc hv1 dj x₀).acc S R u h).tags ((sitesAtHP ht hc hv1 dj x₀).acc S R u h).cf
        ((sitesAtHP ht hc hv1 dj x₀).acc S R u h).tables = .ok c :=
  fun _ _ _ _ => (Hidden.parse_untyped ht hc).1

variable (hU : Built c.unit) {pi : HmIn.Ports} (hpi : HmIn.parse c = .ok pi)
  (outs : Finset (Fin (Placed.gates (Placed.inGates n (Placed.circuitRows c hU).nIn) (Placed.circuitRows c hU) n)))

/-- **`pp` at the hidden tables at `own`**: `SitesHP.progPlaces_keyProgP` at `sitesAtHP`. -/
noncomputable abbrev ppHP :
    ProgPlaces (Placed.keyProgP c pi hU n) outs (planAtHP (Reg := Reg) dj x₀) (tabAtHP dj x₀) :=
  SitesHP.progPlaces_keyProgP hU (sitesAtHP_parse ht hc hv1 dj x₀) hpi outs

/-- **`hConst` at the hidden tables at `own`** (`constCols_keyProgHP`). -/
theorem hConstHP :
    ((Placed.keyProgP c pi hU n).partition outs).ConstCols (planAtHP (Reg := Reg) dj x₀) (tabAtHP dj x₀)
      (ppHP ht hc hv1 dj x₀ hU hpi outs).derived.place
      {Fin.castLE (Placed.le_gates _ _ n) (Placed.oneGate n (Placed.circuitRows c hU).nIn)} :=
  constCols_keyProgHP hU (sitesAtHP_parse ht hc hv1 dj x₀) hpi outs

/-- **`hZero` at the hidden tables at `own`** (`zeroCols_keyProgHP`). -/
theorem hZeroHP :
    ((Placed.keyProgP c pi hU n).partition outs).ZeroCols (planAtHP (Reg := Reg) dj x₀) (tabAtHP dj x₀)
      (ppHP ht hc hv1 dj x₀ hU hpi outs).derived.place
      {Fin.castLE (Placed.le_gates _ _ n) (Placed.zeroGate n (Placed.circuitRows c hU).nIn)} :=
  zeroCols_keyProgHP hU (sitesAtHP_parse ht hc hv1 dj x₀) hpi outs

end FlockSoundness.Discharge.HiddenExec
