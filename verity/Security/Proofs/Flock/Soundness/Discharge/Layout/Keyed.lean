import Proofs.Flock.Soundness.Discharge.Layout.RowEnc
import Proofs.Flock.Soundness.Discharge.Layout.KeysRow
import Proofs.Flock.Soundness.Discharge.Layout.Outs
import Proofs.Flock.Soundness.Discharge.Const
import Proofs.Flock.Soundness.Discharge.Zero

set_option Elab.async false

/-!
# `lay` and `rs` at the production program's places

At [e2e-placed]'s places of the circuit's program on Δ's source keys (`Sites.progPlaces_keyProg hU hparse outs`), with
row widths `16 · words` (`portRowW`), `lay` is `keyLayout` with the registration's bit `outBitOf` (the population's
returned outputs) and `bitCode` for the single gates' commit strings, and `rs` is `keyRowsEnc` with `enc := rowEnc` (the
bytes `sha512/row/v1` hashes, `keyRow_enc`) and [e2e-const]'s `Const.constCols_keyProg` and [e2e-zero]'s
`Zero.zeroCols_keyProg` for the constant's and the zero's gates (`keyedRows`). With `KeysRow` from the parse
(`keysRow_of_parse`), `keyedRowsOfParse`; with the returned outputs' bits for the program's outputs (`outsOf`,
`outsOut_outsOf`), `keyedRowsOuts`.

What `rs` still takes: an untyped circuit whose unit returns the outputs (`c.outNet = c.unitNet`), the drawn units' sites
(`BcSite`), and `outs` the returned outputs' bits (`OutsOut`); `keyedRowsOfParse` a unit with only its leaf group, and
every input a unit row reads a port bit (`PortReads`); `keyedRowsOuts` no `OutsOut`.
-/

namespace FlockSoundness.Discharge.Layout

open Flock FlockLevel3 Binding Placed

variable {c : Flock.Circuit} {hU : Built c.unit} {n : ℕ} {Reg : Type} {mPts : ℕ}
  {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  {si : Placed.Sites plan tab}
  (hparse : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
    HmRow.parse (si.acc S R u h).tags (si.acc S R u h).cf (si.acc S R u h).tables = .ok c)
  (outs : Finset (Fin (keyN c hU n))) (saltCol : Flock.Circuit → ℕ → ℕ → ℕ → ℕ → ℕ) (popOf : Reg → Public)

/-- **`lay` at the production program's places**: `keyLayout` at row widths `16 · words`, with the population's
returned outputs for the registration's bit. -/
noncomputable abbrev keyedLayout :
    Binding.Layout ((keyProg c hU n).partition outs) plan tab
      (((keyProg c hU n).partition outs).placeDecoder plan tab (keyPlace hparse outs)) :=
  keyLayout (portRowW c) hparse outs saltCol popOf (outBitOf c hU n outs popOf) bitCode

variable {outs popOf} in
/-- **`rs` at the production program's places**: `keyRowsEnc` at row widths `16 · words`, with the constant's and the
zero's gates from [e2e-const] and [e2e-zero]. -/
noncomputable def keyedRows (ho : c.outNet = c.unitNet)
    (hsite : ∀ S R u (h : u ∈ S) M, Binding.Sat plan tab S R u M →
      BcSite (si.acc S R u h) (popOf R) (si.o S R u h) (si.vu S R u h) u.val)
    (houts : OutsOut c hU n outs) (hkeys : KeysRow c hU (portRowW c)) :
    Binding.HmRows (keyedLayout hparse outs saltCol popOf) :=
  keyRowsEnc (rowW := portRowW c) (hparse := hparse) (saltCol := saltCol) ho hsite hkeys houts
    (Const.constCols_keyProg hU hparse outs) (Zero.zeroCols_keyProg hU hparse outs)

/-- `keyedRows`' `enc` is `rowEnc`. -/
theorem keyedRows_enc (ho : c.outNet = c.unitNet)
    (hsite : ∀ S R u (h : u ∈ S) M, Binding.Sat plan tab S R u M →
      BcSite (si.acc S R u h) (popOf R) (si.o S R u h) (si.vu S R u h) u.val)
    (houts : OutsOut c hU n outs) (hkeys : KeysRow c hU (portRowW c)) :
    (keyedRows hparse saltCol ho hsite houts hkeys).enc = rowEnc := rfl

variable {outs popOf} in
/-- **`rs` at the production program's places, with `KeysRow` from the parse** (`keysRow_of_parse`): the circuit's unit
has only its leaf group, and every input a unit row reads is a port bit (`PortReads`). -/
noncomputable def keyedRowsOfParse {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)}
    (hc : HmRow.parse tags bytes tables = .ok c) (h1 : c.unit.inGroups.size = 1) (hpr : PortReads c hU)
    (ho : c.outNet = c.unitNet)
    (hsite : ∀ S R u (h : u ∈ S) M, Binding.Sat plan tab S R u M →
      BcSite (si.acc S R u h) (popOf R) (si.o S R u h) (si.vu S R u h) u.val)
    (houts : OutsOut c hU n outs) :
    Binding.HmRows (keyedLayout hparse outs saltCol popOf) :=
  keyedRows hparse saltCol ho hsite houts (keysRow_of_parse hc h1 hpr)

variable {popOf} in
/-- **`rs` at the production program's places, with the returned outputs' bits for the program's outputs**
(`outs := outsOf`, `outsOut_outsOf`): `keyedRowsOfParse` without `OutsOut`. -/
noncomputable def keyedRowsOuts {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)}
    (hc : HmRow.parse tags bytes tables = .ok c) (h1 : c.unit.inGroups.size = 1) (hpr : PortReads c hU)
    (ho : c.outNet = c.unitNet)
    (hsite : ∀ S R u (h : u ∈ S) M, Binding.Sat plan tab S R u M →
      BcSite (si.acc S R u h) (popOf R) (si.o S R u h) (si.vu S R u h) u.val) :
    Binding.HmRows (keyedLayout hparse (outsOf c hU n) saltCol popOf) :=
  keyedRowsOfParse hparse saltCol hc h1 hpr ho hsite (outsOut_outsOf c hU n)

end FlockSoundness.Discharge.Layout
