import FlockSoundness.Discharge.Layout.OutPin
import FlockSoundness.Protocol.Discharge.Layout.RowEnc
import FlockSoundness.Discharge.Layout.KeyProg

set_option Elab.async false

/-!
# The bytes `sha512/row/v1` hashes for a row

`Binding.HmRows.enc` is the byte string `HmRowComputes` says a row's `hm96` slot commits to: for port `p`'s row of
`words` u16 words, the circuit's `sha512x3` slots hash `rowPrefix words` and the row's `2 · words` bytes at
`Typed.msgCol` ([e2e-hm]'s `Discharge.Hm.port_commit`). The layout's values carry their width (`BitVal`), so one
injective `enc` recovers each port's words from its value:

* `rowEnc`: `rowPrefix w ‖ bytes` for a value of `16 w` bits with nothing set past them (`RowShaped`), and a longer code
  of every other value; `rowEnc_inj`;
* `rowEnc_row`: at a row of `16 k` bits it is `rowPrefix k` and the row's `2 k` bytes;
* `keyRow_enc`: at `keyProg`'s places, row port `q` of `16 · words` bits (`rowW q`) holds a value whose `rowEnc` is
  `rowPrefix words` and the `2 · words` bytes the witness holds at the drawn unit's message columns, `port_commit`'s
  row with the block's bits at the site;
* `keyRowsEnc`: `rs` there (`keyRowsOut`) with `enc := rowEnc` and `bitCode` for the other commit strings.
-/

namespace FlockSoundness.Discharge.Layout

open Flock Binding

/-! ## Bytes and their bits -/

/-! ## The row prefix -/

/-! ## The encoding -/

variable {W : ℕ}

/-! ## At `keyProg`'s rows -/

open Placed

variable {c : Flock.Circuit} {hU : Built c.unit} {n : ℕ} {rowW : Fin c.ports.size → ℕ} {Reg : Type} {mPts : ℕ}
  {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  {si : Placed.Sites plan tab}
  (hparse : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
    HmRow.parse (si.acc S R u h).tags (si.acc S R u h).cf (si.acc S R u h).tables = .ok c)
  (outs : Finset (Fin (keyN c hU n))) (saltCol : Flock.Circuit → ℕ → ℕ → ℕ → ℕ → ℕ) (popOf : Reg → Public)
  (outBit : Reg → Fin (keyN c hU n) → Bool)

/-- `keyRowsEnc`'s `enc` is `rowEnc`. -/
theorem keyRowsEnc_enc (ho : c.outNet = c.unitNet)
    (hsite : ∀ S R u (h : u ∈ S) M, Binding.Sat plan tab S R u M →
      BcSite (si.acc S R u h) (popOf R) (si.o S R u h) (si.vu S R u h) u.val)
    (hkeys : KeysRow c hU rowW) (houts : OutsOut c hU n outs)
    (hc1 : ((keyProg c hU n).partition outs).ConstCols plan tab (keyPlace hparse outs)
      {oneG (c := c) (hU := hU) (n := n)})
    (hz1 : ((keyProg c hU n).partition outs).ZeroCols plan tab (keyPlace hparse outs)
      {zeroG (c := c) (hU := hU) (n := n)}) :
    (keyRowsEnc (rowW := rowW) (hparse := hparse) (saltCol := saltCol) ho hsite hkeys houts hc1 hz1).enc =
      rowEnc := rfl

end FlockSoundness.Discharge.Layout
