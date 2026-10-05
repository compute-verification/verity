import FlockSoundness.Discharge.Layout.RowEnc
import FlockSoundness.Discharge.Layout.KeysRow
import FlockSoundness.Discharge.Layout.Outs
import FlockSoundness.Discharge.Const
import FlockSoundness.Discharge.Zero
import FlockSoundness.Protocol.Discharge.Layout.Keyed

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

/-- `keyedRows`' `enc` is `rowEnc`. -/
theorem keyedRows_enc (ho : c.outNet = c.unitNet)
    (hsite : ∀ S R u (h : u ∈ S) M, Binding.Sat plan tab S R u M →
      BcSite (si.acc S R u h) (popOf R) (si.o S R u h) (si.vu S R u h) u.val)
    (houts : OutsOut c hU n outs) (hkeys : KeysRow c hU (portRowW c)) :
    (keyedRows hparse saltCol ho hsite houts hkeys).enc = rowEnc := rfl

end FlockSoundness.Discharge.Layout
