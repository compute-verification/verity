import Proofs.Accounting.TS.Session.Lanes.Layout
import Proofs.Flock.Soundness.Discharge.Layout.KeyProg
import Proofs.Flock.Soundness.Discharge.Layout.KeysRow
import Proofs.Flock.Soundness.Discharge.HiddenExec.KeyH

/-!
# The leaf's decodes, over its statements' port layout

`Accounting.Guarantees.TS.LeafLanesSound` reads a session's registered rows through a decode per statement (`decL` for
a lane's, `decC` for the combine's). These are the decodes, from the statements' layout (`PouwWindow.Lanes`) and the
session's own circuit:

* `portBit s v name k`: bit `k` of the row of the port named `name`, at unit 0, in a layer value `v` (the registered reads
  `openedReg` or `rowsReg` give): `v` at a gate of the key program that carries it and is a wire of unit 0, by the
  hidden-output layout (`HiddenExec.rowOfH`) at the session's row widths (`portRowW`, the ones its rows are laid out
  at). At an input row (`keyRowOf`), input gate `2 + u·B + b` is bit `8·byte + bit` of unit `u`'s row port `q` where
  input `b`'s key is that message bit of the parsed circuit's row `q`; at an output row (`outRowOf`), the gate a copy of
  `u` returns on that bit. No circuit file is assumed: the gate is the parsed circuit's;
* `bitsN`, `sgnBits`: a field's bits as a natural number, least significant first, and its two's complement;
* `decLane T rd`, `decComb T rd`: lane `l`'s and the combine's view from a reader `rd` of ports' row bits, each field at
  its port and first bit in the layout (`fieldBit`, `PouwWindow.Lanes.fieldOf?`): an A row's codes (s8) and scale (its
  f32 encoding), the record's z (s32) and y (its bf16 encoding) cell by cell, a partials row's or the receipt's u32 words,
  the admit bits, salt, position and roots (`LanePub`), the index and `ok`;
* `decL T`, `decC T`: those at `portBit`.

A field's place in its port's row is the same in every lane (`lanePorts`' fields don't read the lane), so `decLane`
reads them at lane 0's ports.

`decL_congr` and `decC_congr`: a decode reads only the ports' row bits, so two sessions or layer values that agree on
them decode alike. The leaf `LeafLanesSound` opens is the one the data ports' rows decode to (`Lanes.Committed`); the
known values' content is flock-verify's to check (`Lanes.KnownRead`).
-/

namespace Accounting.TS.Lanes

open PouwWindow.Lanes Flock.SoundAtBridge Accounting.TS.Checked
open FlockSoundness.Discharge.Layout (keyN keyRowOf portRowW)

open Classical in
/-- **Bit `k` of the row of the port named `name`**, at unit 0, in the layer value `v`: `v` at a key program gate that
the hidden-output layout (`HiddenExec.rowOfH`, at the session's row widths `portRowW`) places on that bit and that is a
wire of unit 0 (`P.io`), or nothing where no such gate is: an input gate on an input row's bit (`keyRowOf`), which unit 0
reads, or a gate unit 0 returns on an output row's bit (`outRowOf`). Several input columns can carry one message bit;
only the first is wired, so only it is read. -/
noncomputable def portBit (s : RowsSession) (v : ℕ → ℕ → Option Bool) (name : String) (k : ℕ) : Option Bool :=
  if h : ∃ (g : Fin (keyN s.c s.hU s.n)) (u : Fin s.n) (q : Fin s.c.ports.size) (b : Fin (portRowW s.c q)),
      u.val = 0 ∧ s.c.ports[q].name = name ∧ b.val = k ∧
        FlockSoundness.Discharge.HiddenExec.rowOfH s.c s.hU s.n (FlockSoundness.Discharge.Layout.outsOf s.c s.hU s.n) g =
          some (u, ⟨q, b⟩) ∧
        (g : Fin s.Circ.N) ∈ s.P.io u
  then v 0 (Classical.choose h).val else none

/-- The natural number a reader's bits `[lo, lo + n)` spell, least significant first, a missing bit read as 0. -/
def bitsN (rd : ℕ → Option Bool) (lo n : ℕ) : ℕ :=
  ∑ i ∈ Finset.range n, if rd (lo + i) = some true then 2 ^ i else 0

/-- A field's `n` bits spell a number below `2^n`. -/
theorem bitsN_lt (rd : ℕ → Option Bool) (lo : ℕ) : ∀ n, bitsN rd lo n < 2 ^ n
  | 0 => by simp [bitsN]
  | n + 1 => by
    have ih := bitsN_lt rd lo n
    have e : bitsN rd lo (n + 1) = bitsN rd lo n + (if rd (lo + n) = some true then 2 ^ n else 0) := by
      unfold bitsN
      rw [Finset.sum_range_succ]
    rw [e, pow_succ]
    split <;> omega

/-- Two's complement at `n` bits. -/
def sgnBits (n x : ℕ) : ℤ := if x < 2 ^ (n - 1) then (x : ℤ) else (x : ℤ) - 2 ^ n

/-- The first bit of the field at `path` among the ports `ps` (`fieldOf?`), 0 where none holds it. -/
def fieldBit (ps : List Port) (path : String) : ℕ := ((fieldOf? ps path).map fun pf => pf.2.bit).getD 0

/-- **A lane's public inputs** as its statement reads them: its admit bits and the salt, position and roots, each
within its field's width (a digest's 512 bits, a u64, two digests), as the decode reads them from those bits. -/
structure LanePub where
  admit : Fin 32 → Bool
  salt : ℕ
  position : ℕ
  roots : ℕ
  salt_lt : salt < 2 ^ digestBits
  position_lt : position < 2 ^ positionBits
  roots_lt : roots < 2 ^ (2 * digestBits)

/-- **Lane `l`'s view from a reader of its ports' row bits**, each field at its port and first bit in the tile's layout:
an A row at row slot `s` from port `a<s>`, a B row at column slot `t` from `b<t>` (its `K` s8 codes, then its scale's
encoding), the record from `zy` (cell `(i, j)`'s s32 z and its y's encoding), the partials row's `nw` u32 words from
`partials`, the admit bits from `admit`, the salt, position, roots and index from `pub`, and `ok` from `o0`. -/
noncomputable def decLane (K nw : ℕ) (T : Tile) (rd : String → ℕ → Option Bool) : LaneView K nw LanePub where
  a s := ⟨fun k => sgnBits 8 (bitsN (rd s!"a{s.val}") (fieldBit (lanePorts T 0) s!"a[{s.val}].codes" + 8 * k.val) 8),
    (bitsN (rd s!"a{s.val}") (fieldBit (lanePorts T 0) s!"a[{s.val}].scale") 32 : ℤ)⟩
  b t := ⟨fun k => sgnBits 8 (bitsN (rd s!"b{t.val}") (fieldBit (lanePorts T 0) s!"b[{t.val}].codes" + 8 * k.val) 8),
    (bitsN (rd s!"b{t.val}") (fieldBit (lanePorts T 0) s!"b[{t.val}].scale") 32 : ℤ)⟩
  z i j := sgnBits 32 (bitsN (rd "zy") (fieldBit (lanePorts T 0) "zy.z" + 32 * (i.val * T.c + j.val)) 32)
  y i j := (bitsN (rd "zy") (fieldBit (lanePorts T 0) "zy.y" + 16 * (i.val * T.c + j.val)) 16 : ℤ)
  part w := BitVec.ofNat 32 (bitsN (rd "partials") (fieldBit (lanePorts T 0) "partials" + 32 * w.val) 32)
  pub := ⟨fun i => rd "admit" (fieldBit (lanePorts T 0) "admit" + i.val) == some true,
    bitsN (rd "pub") (fieldBit (lanePorts T 0) "salt") digestBits,
    bitsN (rd "pub") (fieldBit (lanePorts T 0) "position") positionBits,
    bitsN (rd "pub") (fieldBit (lanePorts T 0) "roots") (2 * digestBits), bitsN_lt _ _ _, bitsN_lt _ _ _,
    bitsN_lt _ _ _⟩
  lane := ⟨bitsN (rd "pub") (fieldBit (lanePorts T 0) "lane") laneBits % 32, Nat.mod_lt _ (by norm_num)⟩
  ok := rd "o0" 0 == some true

/-- **The combine's view from a reader of its ports' row bits**: row `x` of the partials from `p<x>`, the receipt from
`receipt`, each `nw` u32 words, and `ok` from `o0`. -/
noncomputable def decComb (nw : ℕ) (T : Tile) (rd : String → ℕ → Option Bool) : CombineView nw where
  parts x w := BitVec.ofNat 32
    (bitsN (rd s!"p{x.val}") (fieldBit (combinePorts T) s!"partials[{x.val}]" + 32 * w.val) 32)
  receipt w := BitVec.ofNat 32 (bitsN (rd "receipt") (fieldBit (combinePorts T) "receipt" + 32 * w.val) 32)
  ok := rd "o0" 0 == some true

/-- **A lane's decode**: `decLane` at the session's port bits (`portBit`). -/
noncomputable def decL (K nw : ℕ) (T : Tile) (s : RowsSession) (v : ℕ → ℕ → Option Bool) : LaneView K nw LanePub :=
  decLane K nw T (portBit s v)

/-- **The combine's decode**: `decComb` at the session's port bits (`portBit`). -/
noncomputable def decC (nw : ℕ) (T : Tile) (s : RowsSession) (v : ℕ → ℕ → Option Bool) : CombineView nw :=
  decComb nw T (portBit s v)

/-- A lane's decode reads only the ports' row bits. -/
theorem decL_congr {K nw : ℕ} (T : Tile) {s s' : RowsSession} {v v' : ℕ → ℕ → Option Bool}
    (h : ∀ name k, portBit s v name k = portBit s' v' name k) : decL K nw T s v = decL K nw T s' v' := by
  have e : portBit s v = portBit s' v' := funext fun name => funext fun k => h name k
  unfold decL
  rw [e]

/-- The combine's decode reads only the ports' row bits. -/
theorem decC_congr {nw : ℕ} (T : Tile) {s s' : RowsSession} {v v' : ℕ → ℕ → Option Bool}
    (h : ∀ name k, portBit s v name k = portBit s' v' name k) : decC nw T s v = decC nw T s' v' := by
  have e : portBit s v = portBit s' v' := funext fun name => funext fun k => h name k
  unfold decC
  rw [e]

end Accounting.TS.Lanes
