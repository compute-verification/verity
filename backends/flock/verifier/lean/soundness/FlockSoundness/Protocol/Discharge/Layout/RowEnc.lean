import FlockSoundness.Discharge.Layout.OutPin

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

/-- Bit `i` of a value, `false` past `W`. -/
def BitVal.get (v : BitVal W) (i : ℕ) : Bool := if h : i < W then v.2 ⟨i, h⟩ else false

/-- **A row's value**: a width of `16 w` bits (`w` u16 words), and no bit set past it. -/
def RowShaped (v : BitVal W) : Prop := v.1.val % 16 = 0 ∧ ∀ i : Fin W, v.1.val ≤ i.val → v.2 i = false

instance : DecidablePred (RowShaped (W := W)) := fun _ => inferInstanceAs (Decidable (_ ∧ _))

/-- One byte per width and per bit: a code of fixed length `2 W + 1`. -/
def bitCode (v : BitVal W) : List UInt8 :=
  List.ofFn (fun i : Fin (W + 1) => if i = v.1 then (1 : UInt8) else 0) ++
    List.ofFn (fun i : Fin W => if v.2 i then (1 : UInt8) else 0)

/-- The `2 w` bytes of a row of `w` words: byte `j`'s bit `t` is bit `8 j + t`. -/
def rowBytes (v : BitVal W) : List UInt8 :=
  List.ofFn fun j : Fin (2 * (v.1.val / 16)) => byteOf fun t => v.get (8 * j.val + t.val)

/-- **The bytes `sha512/row/v1` hashes for a value**: a row of `w` words is `rowPrefix w` and its `2 w` bytes, as the
circuit's `sha512x3` slots hash it; any other value is `128 + W` zero bytes and `bitCode`, longer than every row. -/
def rowEnc (v : BitVal W) : List UInt8 :=
  if RowShaped v then (HmRow.rowPrefix (v.1.val / 16)).data.toList ++ rowBytes v
  else List.replicate (128 + W) 0 ++ bitCode v

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

variable {outs popOf} (rowW) in
/-- **`rs` at `keyProg`'s places, with the row bytes `sha512/row/v1` hashes**: `keyRowsOut` with `enc := rowEnc`
(`keyRow_enc`) and `bitCode` for the other commit strings. -/
noncomputable def keyRowsEnc (ho : c.outNet = c.unitNet)
    (hsite : ∀ S R u (h : u ∈ S) M, Binding.Sat plan tab S R u M →
      BcSite (si.acc S R u h) (popOf R) (si.o S R u h) (si.vu S R u h) u.val)
    (hkeys : KeysRow c hU rowW) (houts : OutsOut c hU n outs)
    (hc1 : ((keyProg c hU n).partition outs).ConstCols plan tab (keyPlace hparse outs)
      {oneG (c := c) (hU := hU) (n := n)})
    (hz1 : ((keyProg c hU n).partition outs).ZeroCols plan tab (keyPlace hparse outs)
      {zeroG (c := c) (hU := hU) (n := n)}) :
    Binding.HmRows (keyLayout rowW hparse outs saltCol popOf (outBitOf c hU n outs popOf) bitCode) :=
  keyRowsOut (outs := outs) (popOf := popOf) (hparse := hparse) (rowW := rowW) (saltCol := saltCol) rowEnc
    rowEnc_inj bitCode bitCode_inj ho hsite hkeys houts hc1 hz1

end FlockSoundness.Discharge.Layout
