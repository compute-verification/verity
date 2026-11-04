import Proofs.Flock.Soundness.Discharge.HiddenExec.TablesP
import Proofs.Flock.Soundness.Discharge.HiddenExec.PlacedP
import Proofs.Flock.Soundness.Discharge.HiddenExec.Sites

set_option Elab.async false

/-!
# The sites of a one-table hidden-output session at public inputs

`Sites.lean` at the verifier's public inputs `own`, at the session's statement circuit `c` (`HmOut.parseStmt`, a
templated file's committed circuit unrolled). A drawn table's acceptance (`TableHP.acc`) is its `Stmt.setupHidden … own`
run with its `SpecP` (`Hidden.setupHidden_specP`), the layout `Hidden.hidden_layoutP` gives, its unit slots' kept rows
its flat twin's (`unitRowsFlatP`), and the table's statement `stmtOf` of it, the layout's block R1CS
(`Integrate.model_eqW`).

`sitesAtHP` holds each drawn unit's acceptance, template instance and block; every acceptance's statement circuit is the
session's (`sitesAtHP_parse`). So `ppHP`, `hConstHP` and `hZeroHP` are `ProgPlaces`, `hConst` and `hZero` for
`Placed.keyProgP` at the hidden tables at `own`.
-/

namespace FlockSoundness.Discharge.HiddenExec

open Flock FlockLevel3 Model Refine Audit
open FlockSoundness.Discharge.Exec (Inputs)

variable {n : ℕ} {I : Inputs} {own : Option ByteArray} {mPts : ℕ}
  (ht : I.tags.typed = false) {c : Flock.Circuit} (hcS : HmOut.parseStmt I.tags I.circuitFile I.tables = .ok c)
  (hv1 : Hidden.Any.RowsFitS c)

namespace TableHP

variable {S : Finset (Fin n)} (x : TableHP I own mPts S)

/-- A drawn table's `SpecP`, at its circuit's public input ports. -/
theorem specP : Hidden.SpecP I.tags I.circuitFile I.publicFile I.coins I.tables (some x.d) own
    (Hidden.setupHidden_specP x.h).choose x.st :=
  (Hidden.setupHidden_specP x.h).choose_spec

/-- A drawn table's circuit is committed: `HmOut.parse`'s or `HmOut.parseTmpl`'s. -/
theorem committed : Hidden.Committed I.tags I.circuitFile I.tables x.st.c := Hidden.committed_of_commit x.specP.parse

include ht hcS in
/-- **A drawn table's statement circuit is the session's**: its committed circuit unrolls to `c`. -/
theorem unrolled_eq : HmOut.unrolled x.st.c = c :=
  Except.ok.inj ((FlockSoundness.parseStmt_of_commit ht x.specP.parse).symm.trans hcS)

include ht hcS hv1 in
/-- A drawn table's output rows are the statement circuit's (`outRowsS_of_unrolled`). -/
theorem outRows : Hidden.Any.OutRowsS x.st.c :=
  outRowsS_of_unrolled ht x.committed (by rw [x.unrolled_eq ht hcS]; exact hv1.1)

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
  L := Hidden.hidden_layoutP ht x.specP (x.committed.unpinned_of_rep ht (x.outRows ht hcS hv1).1)
  hR := unitRowsFlatP ht x.specP (x.outRows ht hcS hv1)
  regs := x.regs
  hregs := x.hregs
  hT := (Integrate.model_eqW (Hidden.Committed.netWidths ht x.committed) (Hidden.Committed.netKinds ht x.committed) _
    x.specP.leaf x.hwf x.regs).trans x.hT.symm

end TableHP

variable (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetupHP I own mPts S₀ (dj S₀))
  {Reg : Type}

/-- **The sites of the one-table hidden plan at `own`** (`sitesAtH`). -/
noncomputable def sitesAtHP : SitesHP (planAtHP (Reg := Reg) dj x₀) (tabAtHP dj x₀) where
  acc S _ _ _ := (tableAtHP dj x₀ S).acc ht hcS hv1
  vu S _ u h := (tableAtHP dj x₀ S).vu u h
  hvu S _ u h := (tableAtHP dj x₀ S).hvu u h
  o S _ u h := (tableAtHP dj x₀ S).o u h

/-- **One statement circuit per session**: every site's committed circuit unrolls to the session's `c`. -/
theorem sitesAtHP_parse :
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
      HmOut.unrolled ((sitesAtHP ht hcS hv1 dj x₀).acc S R u h).st.c = c :=
  fun S _ _ _ => (tableAtHP dj x₀ S).unrolled_eq ht hcS

variable (hU : Built c.unit) {pi : HmIn.Ports} (hpi : HmIn.parse c = .ok pi)
  (outs : Finset (Fin (Placed.gates (Placed.inGates n (Placed.circuitRows c hU).nIn) (Placed.circuitRows c hU) n)))

/-- **`pp` at the hidden tables at `own`**: `SitesHP.progPlaces_keyProgP` at `sitesAtHP`. -/
noncomputable abbrev ppHP :
    ProgPlaces (Placed.keyProgP c pi hU n) outs (planAtHP (Reg := Reg) dj x₀) (tabAtHP dj x₀) :=
  SitesHP.progPlaces_keyProgP hU (sitesAtHP_parse ht hcS hv1 dj x₀) hpi outs

/-- **`hConst` at the hidden tables at `own`** (`constCols_keyProgHP`). -/
theorem hConstHP :
    ((Placed.keyProgP c pi hU n).partition outs).ConstCols (planAtHP (Reg := Reg) dj x₀) (tabAtHP dj x₀)
      (ppHP ht hcS hv1 dj x₀ hU hpi outs).derived.place
      {Fin.castLE (Placed.le_gates _ _ n) (Placed.oneGate n (Placed.circuitRows c hU).nIn)} :=
  constCols_keyProgHP hU (sitesAtHP_parse ht hcS hv1 dj x₀) hpi outs

/-- **`hZero` at the hidden tables at `own`** (`zeroCols_keyProgHP`). -/
theorem hZeroHP :
    ((Placed.keyProgP c pi hU n).partition outs).ZeroCols (planAtHP (Reg := Reg) dj x₀) (tabAtHP dj x₀)
      (ppHP ht hcS hv1 dj x₀ hU hpi outs).derived.place
      {Fin.castLE (Placed.le_gates _ _ n) (Placed.zeroGate n (Placed.circuitRows c hU).nIn)} :=
  zeroCols_keyProgHP hU (sitesAtHP_parse ht hcS hv1 dj x₀) hpi outs

end FlockSoundness.Discharge.HiddenExec
