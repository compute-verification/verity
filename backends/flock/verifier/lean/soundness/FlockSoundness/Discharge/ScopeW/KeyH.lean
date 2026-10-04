import FlockSoundness.Discharge.HiddenExec.KeyH
import FlockSoundness.Discharge.ScopeW.RowEnc
import FlockSoundness.Discharge.ScopeW.Rows

set_option Elab.async false

/-!
# `lay` and `rs` at `keyProg`'s places in hidden-output tables, under the widened scope

`HiddenExec.keyReadersH` at rows of the four shapes `rowShapeW` admits. Two things change:

* an output row is a run of output ports (`HmOut.outGroup`), so the gate on an output word `o` is a bit of the output
  row whose run holds `o` (`outRunOf`), at its offset in the run (`outBitRowW`); on a circuit whose output rows are one
  port each it is `HiddenExec.outBitRow`'s bit;
* each row is hashed under its own schema (`encW`: `rowEncOf` of its port), with its extractor (`pairW`), so `rs` is an
  `HmRowsP` (`keyRowsHW`).

The readers' addresses are `keyReadersH`'s: `Typed.msgCol` places a row's byte in its chain's compression
(`HmRow.byteAt`, segment by segment) or, past them, in its `hm96` slot's block, so they hold at every shape.

What is left open, each a named obligation on a `sorry` stub:
* `keyProgHW_copies`: Δ's copies, at input rows (`copy_siteH`, unchanged) and at output rows of runs (`out_copyH` at
  `outBitRowW`);
* `keyProgHW_bcPinned`: `keyProgH_bcPinned` at `rowOfHW` (no row's shape is read);
* `keyProgHW_valPinned`: every computed committed gate is an output row's bit (`outBitRowW`, at runs);
* `keyRowsHW_computes`: `HmRowComputes` at each row's schema: the `b ‖ c` at the row's `hm96` output is the commit
  string of `rowEncOf` of the row's value (`bc_site_commitH` at `sha512/row/v2` and `sha512/row-seg/v1` rows).
-/

namespace FlockSoundness.Discharge.ScopeW

open Flock FlockLevel3 Model Audit Binding Placed
open FlockSoundness.Discharge.HiddenExec
open FlockSoundness.Discharge.Layout (keyN keyRowOf portRowW OutColAt outDecode OutsOut KeysRow BitVal rowEnc
  rowEnc_inj)

variable {c : Flock.Circuit} {hU : Built c.unit} {n : ℕ}

/-! ## Output rows of runs of output ports -/

variable (c) in
/-- **The output row holding output word `o`**: the last output row whose run of output ports starts at or before `o`
(`HmOut.outGroup`'s third component). -/
def outRunOf (o : ℕ) : ℕ := Nat.findGreatest (fun j => (HmOut.outGroup c j).2.2.1 ≤ o) (c.outRowPorts.size - 1)

variable (c) in
/-- **An output row's bit, at runs**: copy `d.1`'s output `d.2.1`, bit `d.2.2`, of unit `u` is bit
`16 (o - at) + d.2.2` of `u`'s output row `nIn + j`, where the copy's output word `o` is in output row `j`'s run
(`outRunOf`), which starts at output word `at`; none past the row. -/
def outBitRowW (u : Fin n) (d : ℕ × ℕ × ℕ) : Option (Fin n × (q : Fin c.ports.size) × Fin (portRowW c q)) :=
  if hq : HmOut.nIn c + outRunOf c (outWordOf c d) < c.ports.size then
    if hk : 16 * (outWordOf c d - (HmOut.outGroup c (outRunOf c (outWordOf c d))).2.2.1) + d.2.2 <
        portRowW c ⟨_, hq⟩ then
      some (u, ⟨⟨_, hq⟩, ⟨_, hk⟩⟩)
    else none
  else none

variable (c hU n) in
open Classical in
/-- **The output-row bit of a unit's computed gate, at runs**: `HiddenExec.outRowOf` at `outBitRowW`. -/
noncomputable def outRowOfW (outs : Finset (Fin (keyN c hU n))) (g : Fin (keyN c hU n)) :
    Option (Fin n × (q : Fin c.ports.size) × Fin (portRowW c q)) :=
  match (keyProg c hU n).unit g with
  | some u =>
    if h : ∃ x, (Const.Prog.instC (keyProg c hU n) outs u).wire x = g then
      outBitRowW c u (outDecode c hU (Classical.choose h).val)
    else none
  | none => none

variable (c hU n) in
/-- **The row bits of `keyProg`, at runs**: an input row's bit (`Layout.keyRowOf`), else an output row's
(`outRowOfW`). -/
noncomputable def rowOfHW (outs : Finset (Fin (keyN c hU n))) (g : Fin (keyN c hU n)) :
    Option (Fin n × (q : Fin c.ports.size) × Fin (portRowW c q)) :=
  match keyRowOf c hU n (portRowW c) g with
  | some r => some r
  | none => outRowOfW c hU n outs g

/-! ## The readers -/

section places

variable {Reg : Type} {mPts : ℕ} {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  {si : SitesH plan tab}
  (hparse : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
    HmRow.parse (si.acc S R u h).tags (si.acc S R u h).cf (si.acc S R u h).tables = .ok c)
  (outs : Finset (Fin (keyN c hU n))) (popOf : Reg → Public)

/-- **The readers at `keyProg`'s sites in hidden-output tables, at runs**: `HiddenExec.keyReadersH` at `rowOfHW`. -/
noncomputable def keyReadersHW : Layout.Readers ((keyProg c hU n).circuit outs) plan tab :=
  Layout.Readers.ofRows (keyPlaceH hparse outs) (Layout.keyN_pos c hU n) (portRowW c)
    (W := Finset.univ.sup (portRowW c) + 1)
    (fun q => (Finset.le_sup (Finset.mem_univ q)).trans (Nat.le_succ _)) (Nat.le_add_left 1 _) (rowOfHW c hU n outs)
    (keyRowAddrH si) (keySaltAddrH si (hU := hU)) (keyBcAddrH si)
    (fun R u q => ((popOf R).digest u.val q.val c.ports.size).data.toList)
    (Layout.keyGate (c := c) (hU := hU) (n := n) fun _ _ => false)

variable (ho : c.outNet = c.unitNet) (hW : ∀ p ∈ c.ports.toList, rowShapeW p = true)
  (houts : OutsOut c hU n outs)

include ho hW houts in
/-- **Δ's copies at `keyProg`'s sites, at the widened scope**: a column on an input row's bit holds the witness's bit at
its key's message column (`copy_siteH`); a column on an output row's bit (`outBitRowW`) holds the bit at the output
row's message column (`out_copyH` at runs of output ports and at the four shapes). -/
theorem keyProgHW_copies : (keyReadersHW hparse outs popOf).Copies (keyPlaceH hparse outs) := sorry

/-- **The rows' `b ‖ c` at `keyProg`'s sites is the registration's**, as `keyProgH_bcPinned` (`bc_site_ofH`). -/
theorem keyProgHW_bcPinned
    (hsite : ∀ S R u (h : u ∈ S) M, Binding.Sat plan tab S R u M →
      BcSiteH (si.acc S R u h) (popOf R) (si.o S R u h) (si.vu S R u h) u.val) :
    (keyReadersHW hparse outs popOf).BcPinned (P := (keyProg c hU n).partition outs) := sorry

include ho hW houts in
/-- **The single gates at `keyProg`'s sites are the constant and the zero**, as `keyProgH_valPinned`: a computed
committed gate is a returned output's, so a bit of the output row whose run holds its word (`outBitRowW`). -/
theorem keyProgHW_valPinned (hkeys : KeysRow c hU (portRowW c)) :
    (keyReadersHW hparse outs popOf).ValPinned (P := (keyProg c hU n).partition outs) := sorry

/-- **`lay` at `keyProg`'s places in hidden-output tables, at the widened scope**: `keyReadersHW`'s layout, with
`bitCode` for the single gates and Δ's copies from `keyProgHW_copies`. -/
noncomputable def keyLayoutHW :
    Binding.Layout ((keyProg c hU n).partition outs) plan tab
      (((keyProg c hU n).partition outs).placeDecoder plan tab (keyPlaceH hparse outs)) :=
  (keyReadersHW hparse outs popOf).layout (keyPlaceH hparse outs) Layout.bitCode
    (keyProgHW_copies hparse outs popOf ho hW houts)

/-- **`rs` at `keyProg`'s places in hidden-output tables, at the widened scope**: `keyReadersHW`'s rows, each under its
port's schema (`encW`, `pairW`), from `keyProgHW_copies`, `keyProgHW_bcPinned` and `keyProgHW_valPinned`. -/
noncomputable def keyRowsHW
    (hsite : ∀ S R u (h : u ∈ S) M, Binding.Sat plan tab S R u M →
      BcSiteH (si.acc S R u h) (popOf R) (si.o S R u h) (si.vu S R u h) u.val)
    (hkeys : KeysRow c hU (portRowW c)) :
    HmRowsP (keyLayoutHW hparse outs popOf ho hW houts) :=
  rowsP (keyReadersHW hparse outs popOf) (keyPlaceH hparse outs) (encW c n (Fin (keyN c hU n)))
    (pairW c n (Fin (keyN c hU n))) pairW_spec Layout.bitCode Layout.bitCode_inj
    (keyProgHW_copies hparse outs popOf ho hW houts)
    (keyProgHW_bcPinned hparse outs popOf hsite) (keyProgHW_valPinned hparse outs popOf ho hW houts hkeys)

/-- **`HmRowComputes` at `keyProg`'s rows, at the widened scope** (`hHm` for `keyRowsHW`): at every drawn unit's
committed wire held by a row, in a satisfying witness of the unit's table, the `b ‖ c` at the row's `hm96` output is the
commit string of `rowEncOf` of the row's value under its port's schema and the salt at `Hm.saltCol`. -/
theorem keyRowsHW_computes
    (hsite : ∀ S R u (h : u ∈ S) M, Binding.Sat plan tab S R u M →
      BcSiteH (si.acc S R u h) (popOf R) (si.o S R u h) (si.vu S R u h) u.val)
    (hkeys : KeysRow c hU (portRowW c)) :
    (keyRowsHW hparse outs popOf ho hW houts hsite hkeys).Computes := sorry

end places

end FlockSoundness.Discharge.ScopeW
