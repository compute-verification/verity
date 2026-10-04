import FlockSoundness.Discharge.HiddenExec.Sites
import FlockSoundness.Discharge.ScopeW.Scope

/-!
# The sites of a one-table hidden-output session, under the widened scope

`HiddenExec.sitesAtH` reads `sha512/row/v1` rows only to give a drawn table's acceptance its unit slots' rows
(`AcceptedH.hR`, `HiddenExec.unitRowsFlat`): every message column `Hidden.outPairs` copies an output bit into is no unit
slot's row. `unitRowsFlatW` is that at the four shapes: an output row's byte is in one of its chain's compressions
(`HmRow.byteAt`: a `sha512x3` slot, segment by segment for a segmented row) or in its `hm96` slot's block (a
`sha512/row/v2` row with no compression before it), and neither is a unit slot's row. The rest is `Sites.lean`'s, at
`accW`.
-/

set_option Elab.async false

namespace FlockSoundness.Discharge.ScopeW

open Flock FlockLevel3 Model Refine Audit
open FlockSoundness.Discharge.HiddenExec
open FlockSoundness.Discharge.Exec (Inputs)
open Lean (Json)

/-- **An untyped hidden statement reads its flat twin's rows at every unit slot's row**, at rows of the widened scope's
four shapes: `HiddenExec.unitRowsFlat` with `outPairs_ne_unit` at bit rows and segmented rows. -/
theorem unitRowsFlatW {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
    {draw partition : Option Json} {program : Option ByteArray} {st : Stmt} (ht : tags.typed = false)
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st)
    (hW : ∀ p ∈ st.c.ports.toList, rowShapeW p = true) : UnitRowsFlat st := sorry

variable {n : ℕ} {I : Inputs} {mPts : ℕ}
  (ht : I.tags.typed = false) {c : Flock.Circuit} (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c)
  (hW : ∀ p ∈ c.ports.toList, rowShapeW p = true)

/-- **A drawn table's acceptance, at the widened scope**: `TableH.acc` with its unit slots' rows from
`unitRowsFlatW`. -/
noncomputable def accW {S : Finset (Fin n)} (x : TableH I mPts S) : AcceptedH x.T.S where
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
  L := Hidden.hidden_layout ht x.h
  hR := unitRowsFlatW ht x.h (by rw [x.c_eq hc]; exact hW)
  regs := x.regs
  hT := (Integrate.model_eq (parse_facts (parse_untyped ht (setupHidden_parseC x.h)).1) _
    (Hidden.setupHidden_spec x.h).leaf x.hwf x.regs).trans x.hT.symm

variable (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetupH I mPts S₀ (dj S₀)) {Reg : Type}

/-- **The sites of the one-table hidden plan, at the widened scope**: `sitesAtH` at `accW`. -/
noncomputable def sitesAtHW : SitesH (planAtH (Reg := Reg) dj x₀) (tabAtH dj x₀) where
  acc S _ _ _ := accW ht hc hW (tableAtH dj x₀ S)
  vu S _ u h := (tableAtH dj x₀ S).vu u h
  hvu S _ u h := (tableAtH dj x₀ S).hvu u h
  o S _ u h := (tableAtH dj x₀ S).o u h

/-- **One circuit per session**, as `sitesAtH_parse`. -/
theorem sitesAtHW_parse :
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
      HmRow.parse ((sitesAtHW ht hc hW dj x₀).acc S R u h).tags ((sitesAtHW ht hc hW dj x₀).acc S R u h).cf
        ((sitesAtHW ht hc hW dj x₀).acc S R u h).tables = .ok c :=
  fun _ _ _ _ => (parse_untyped ht hc).1

variable (hU : Built c.unit)
  (outs : Finset (Fin (Placed.gates (Placed.inGates n (Placed.circuitRows c hU).nIn) (Placed.circuitRows c hU) n)))

/-- **`pp` at the hidden tables, at the widened scope**: `SitesH.progPlaces_keyProg` at `sitesAtHW`. -/
noncomputable abbrev ppHW : ProgPlaces (Placed.keyProg c hU n) outs (planAtH (Reg := Reg) dj x₀) (tabAtH dj x₀) :=
  SitesH.progPlaces_keyProg hU (sitesAtHW_parse ht hc hW dj x₀) outs

/-- **`hConst` at the hidden tables, at the widened scope** (`constCols_keyProgH`). -/
theorem hConstHW :
    ((Placed.keyProg c hU n).partition outs).ConstCols (planAtH (Reg := Reg) dj x₀) (tabAtH dj x₀)
      (ppHW ht hc hW dj x₀ hU outs).derived.place
      {Fin.castLE (Placed.le_gates _ _ n) (Placed.oneGate n (Placed.circuitRows c hU).nIn)} :=
  constCols_keyProgH hU (sitesAtHW_parse ht hc hW dj x₀) outs

/-- **`hZero` at the hidden tables, at the widened scope** (`zeroCols_keyProgH`). -/
theorem hZeroHW :
    ((Placed.keyProg c hU n).partition outs).ZeroCols (planAtH (Reg := Reg) dj x₀) (tabAtH dj x₀)
      (ppHW ht hc hW dj x₀ hU outs).derived.place
      {Fin.castLE (Placed.le_gates _ _ n) (Placed.zeroGate n (Placed.circuitRows c hU).nIn)} :=
  zeroCols_keyProgH hU (sitesAtHW_parse ht hc hW dj x₀) outs

end FlockSoundness.Discharge.ScopeW
