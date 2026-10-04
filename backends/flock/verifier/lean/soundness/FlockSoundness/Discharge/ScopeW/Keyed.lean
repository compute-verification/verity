import FlockSoundness.Discharge.ScopeW.RowEnc
import FlockSoundness.Discharge.ScopeW.Rows
import FlockSoundness.Discharge.Hm.Site

set_option Elab.async false

/-!
# `rs` at the production program's places, each row under its own schema

`Layout.keyedRowsOuts` hashes every row with `Layout.rowEnc`, the bytes `sha512/row/v1` hashes, so `hHm`
(`Hm.keyedRowsOuts_computes`) takes every row to be a `sha512/row/v1` row (`hv1`). `keyedRowsW` is its readers' rows
(`Layout.keyReaders`, the public outputs registered by `outBitOf`) under each row's own schema (`encW`, `pairW`), with
`keyedRowsOuts`' copies and pins: `Layout.keyProg_copies`, `keyProg_bcPinned` and `keyProg_valPinned` read no row's
shape. Its layout is `Layout.keyedLayout`, unchanged.

What is left open is `hHm` at the four shapes (`keyedRowsOutsW_computes`, on a `sorry` stub): the `b ‖ c` at a row's
`hm96` output is the commit string of `rowEncOf` of the row's value (`Hm.bc_site_commit` at `sha512/row/v2` and
`sha512/row-seg/v1` rows).
-/

namespace FlockSoundness.Discharge.ScopeW

open Flock FlockLevel3 Binding Placed
open FlockSoundness.Discharge.Layout (keyN portRowW KeysRow OutsOut PortReads BcSite keyedLayout keyReaders keyPlace
  outBitOf outsOf outsOut_outsOf keysRow_of_parse bitCode bitCode_inj)

variable {c : Flock.Circuit} {hU : Built c.unit} {n : ℕ} {Reg : Type} {mPts : ℕ}
  {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  {si : Placed.Sites plan tab}
  (hparse : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
    HmRow.parse (si.acc S R u h).tags (si.acc S R u h).cf (si.acc S R u h).tables = .ok c)
  (saltCol : Flock.Circuit → ℕ → ℕ → ℕ → ℕ → ℕ) {popOf : Reg → Public}

/-- **`rs` at the production program's places, each row under its own schema**: `Layout.keyedRows`' readers, copies and
pins, with `encW` and `pairW` in place of `rowEnc`. -/
noncomputable def keyedRowsW (outs : Finset (Fin (keyN c hU n))) (ho : c.outNet = c.unitNet)
    (hsite : ∀ S R u (h : u ∈ S) M, Binding.Sat plan tab S R u M →
      BcSite (si.acc S R u h) (popOf R) (si.o S R u h) (si.vu S R u h) u.val)
    (houts : OutsOut c hU n outs) (hkeys : KeysRow c hU (portRowW c)) :
    HmRowsP (keyedLayout hparse outs saltCol popOf) :=
  rowsP (keyReaders (portRowW c) hparse outs saltCol popOf (outBitOf c hU n outs popOf)) (keyPlace hparse outs)
    (encW c n (Fin (keyN c hU n))) (pairW c n (Fin (keyN c hU n))) pairW_spec bitCode bitCode_inj
    (Layout.keyProg_copies hparse outs saltCol popOf _)
    (Layout.keyProg_bcPinned hparse outs saltCol popOf _ hsite)
    (Layout.keyProg_valPinned hparse outs saltCol popOf _ hkeys (Const.constCols_keyProg hU hparse outs)
      (Zero.zeroCols_keyProg hU hparse outs) (Layout.keyProg_hout hparse ho hsite houts))

/-- **`keyedRowsW` at the returned outputs' bits** (`outsOf`), `KeysRow` from the parse: `Layout.keyedRowsOuts`'
form. -/
noncomputable def keyedRowsOutsW {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)}
    (hc : HmRow.parse tags bytes tables = .ok c) (h1 : c.unit.inGroups.size = 1) (hpr : PortReads c hU)
    (ho : c.outNet = c.unitNet)
    (hsite : ∀ S R u (h : u ∈ S) M, Binding.Sat plan tab S R u M →
      BcSite (si.acc S R u h) (popOf R) (si.o S R u h) (si.vu S R u h) u.val) :
    HmRowsP (keyedLayout hparse (outsOf c hU n) saltCol popOf) :=
  keyedRowsW hparse saltCol (outsOf c hU n) ho hsite (outsOut_outsOf c hU n) (keysRow_of_parse hc h1 hpr)

/-- **`HmRowComputes` at `keyedRowsOutsW`'s rows** (`hHm`), at the `hm96` slots' salt columns: in a satisfying witness
of a drawn unit's table, each row port's `b ‖ c` at the unit's site is the commit string of `rowEncOf` of the row's
value under its port's schema, at every row of the four shapes (`hW`, which `scopeOkW_ports` gives). -/
theorem keyedRowsOutsW_computes {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)}
    (hc : HmRow.parse tags bytes tables = .ok c) (h1 : c.unit.inGroups.size = 1) (hpr : PortReads c hU)
    (ho : c.outNet = c.unitNet) (hW : ∀ p ∈ c.ports.toList, rowShapeW p = true)
    (hsite : ∀ S R u (h : u ∈ S) M, Binding.Sat plan tab S R u M →
      BcSite (si.acc S R u h) (popOf R) (si.o S R u h) (si.vu S R u h) u.val) :
    (keyedRowsOutsW hparse Hm.saltCol hc h1 hpr ho hsite).Computes := sorry

end FlockSoundness.Discharge.ScopeW
