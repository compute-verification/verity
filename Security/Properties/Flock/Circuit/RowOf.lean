import Properties.Flock.Circuit.Outs
import Flock.HmRow
import Flock.Registered
import Flock.Hm96
import Flock.Public

/-!
# The row bits of the circuit a registered circuit file describes, and its ports' registered leaves

* `HiddenExec.rowOfH c hU n outs g`: the unit, port and bit of the port's row that gate `g` of `keyProg c hU n` is: an
  input row's bit (`Layout.keyRowOf`, within the rows' widths `Layout.portRowW`), else a returned output's bit of an
  output row (`HiddenExec.outRowOf`, through `outBitRow`, `outPortOf`, `Hidden.outAt` and `Layout.outDecode`); `none` at
  a gate no row holds.
* `Bridge.domOf`, `schemaOf`: a port's frame-v3 domain (its name) and leaf schema; `regLeafOf`, `leavesOf`: the
  registered tree leaves of a port's rows, C-Flock's hm96 tree leaves of the public file's `b ‖ c`, framed at the
  port's schema.

Moved unchanged from `Proofs.Flock.Soundness.Discharge.{Layout.KeysRow, Layout.KeyProg, Layout.OutPin, Hidden.Defs,
HiddenExec.Ret, HiddenExec.KeyH, Const.Inst}` and `Proofs.Flock.Bridge.{Deployed, Transports}`, so that the vocabulary's
positions (`Verity.FileRead.place`) read them; the lemmas about them stay there.
-/

namespace FlockSoundness.Discharge.Layout

open Flock

variable (c : Flock.Circuit) (hU : Built c.unit) (n : ℕ)

/-- **The row widths**: port `q`'s `16 · words` bits. -/
def portRowW (q : Fin c.ports.size) : ℕ := 16 * c.ports[q].words

variable (rowW : Fin c.ports.size → ℕ)

/-- **The row bits of `keyProg`**: input gate `2 + u·B + b` is bit `8·byte + bit` of unit `u`'s row port `q` when
input `b`'s Δ key is message bit `bit < 8` of byte `byte` of row port `q`, within the row's `rowW q` bits. -/
noncomputable def keyRowOf (g : Fin (keyN c hU n)) : Option (Fin n × (q : Fin c.ports.size) × Fin (rowW q)) :=
  if h : 2 ≤ g.val ∧ g.val < 2 + n * keyB c hU then
    match Aliased.srcKey c ((g.val - 2) % keyB c hU) with
    | .msg q byte bit =>
      if hq : q < c.ports.size then
        if hk : 8 * byte + bit < rowW ⟨q, hq⟩ ∧ bit < 8 then
          some (⟨(g.val - 2) / keyB c hU, (Nat.div_lt_iff_lt_mul (Nat.pos_of_ne_zero fun h0 => by
            rw [h0, Nat.mul_zero] at h; omega)).2 (by omega)⟩, ⟨⟨q, hq⟩, ⟨8 * byte + bit, hk.1⟩⟩)
        else none
      else none
    | _ => none
  else none

/-- The copy, output and bit a stacked computed column is on (`OutColAt` read back). -/
noncomputable def outDecode (x : ℕ) : ℕ × ℕ × ℕ :=
  ((x - c.upv * (Rows.ofNet hU).nIn) / ((Rows.ofNet hU).nComp + 1),
    ((Rows.ofNet hU).nIn + ((x - c.upv * (Rows.ofNet hU).nIn) % ((Rows.ofNet hU).nComp + 1) - 1) -
      (c.unit.outGroups.getD 0 default).col * WORD) / 16,
    ((Rows.ofNet hU).nIn + ((x - c.upv * (Rows.ofNet hU).nIn) % ((Rows.ofNet hU).nComp + 1) - 1) -
      (c.unit.outGroups.getD 0 default).col * WORD) % 16)

end FlockSoundness.Discharge.Layout

namespace FlockSoundness.Discharge.Hidden

open Flock

/-- Output port `j`'s first word among a template instance's output words (`HmOut.delta`'s `at_`). -/
def outAt (c : Flock.Circuit) (j : ℕ) : ℕ := ((c.outPorts.toList.take j).map (·.2)).sum

end FlockSoundness.Discharge.Hidden

namespace FlockSoundness.Discharge.Const.Prog

/-- **Unit `u`'s rows, wired as the program wires them**, as an instance in the program's circuit. -/
def instC {N n : ℕ} (p : FlockSoundness.Prog N n) (outs : Finset (Fin N)) (u : Fin n) :
    (p.partition outs).IsRowsUnit u (p.rows u) :=
  (rowsWireC p u).toIsRowsUnit outs

end FlockSoundness.Discharge.Const.Prog

namespace FlockSoundness.Discharge.HiddenExec

open Flock
open FlockSoundness.Discharge.Placed (keyProg)
open FlockSoundness.Discharge.Layout (keyN keyRowOf portRowW outDecode)

/-- The output port output word `o` is in: the last whose first word is at most `o`. -/
def outPortOf (c : Flock.Circuit) (o : ℕ) : ℕ := Nat.findGreatest (fun j => Hidden.outAt c j ≤ o) c.outPorts.size

variable {c : Flock.Circuit} {hU : Built c.unit} {n : ℕ}

variable (c) in
/-- The output word of copy `d.1`'s output `d.2.1`. -/
def outWordOf (d : ℕ × ℕ × ℕ) : ℕ := (c.leavesOut.getD d.1 #[])[d.2.1]!

variable (c) in
/-- **An output row's bit**: copy `d.1`'s output `d.2.1`, bit `d.2.2`, of unit `u` is bit `16 k + d.2.2` of `u`'s output
row `nIn + jo`, where the copy's output word is word `k` of output port `jo` (`outPortOf`); none past the row. -/
def outBitRow (u : Fin n) (d : ℕ × ℕ × ℕ) : Option (Fin n × (q : Fin c.ports.size) × Fin (portRowW c q)) :=
  if hq : HmOut.nIn c + outPortOf c (outWordOf c d) < c.ports.size then
    if hk : 16 * (outWordOf c d - Hidden.outAt c (outPortOf c (outWordOf c d))) + d.2.2 < portRowW c ⟨_, hq⟩ then
      some (u, ⟨⟨_, hq⟩, ⟨_, hk⟩⟩)
    else none
  else none

variable (c hU n) in
open Classical in
/-- **The output-row bit of a unit's computed gate**: the bit (`outBitRow`) of the returned output whose column of the
gate's unit's instance holds the gate (`Layout.outDecode`). -/
noncomputable def outRowOf (outs : Finset (Fin (keyN c hU n))) (g : Fin (keyN c hU n)) :
    Option (Fin n × (q : Fin c.ports.size) × Fin (portRowW c q)) :=
  match (keyProg c hU n).unit g with
  | some u =>
    if h : ∃ x, (Const.Prog.instC (keyProg c hU n) outs u).wire x = g then
      outBitRow c u (outDecode c hU (Classical.choose h).val)
    else none
  | none => none

variable (c hU n) in
/-- **The row bits of `keyProg` at a hidden-output statement**: an input row's bit (`Layout.keyRowOf`), else an output
row's (`outRowOf`). -/
noncomputable def rowOfH (outs : Finset (Fin (keyN c hU n))) (g : Fin (keyN c hU n)) :
    Option (Fin n × (q : Fin c.ports.size) × Fin (portRowW c q)) :=
  match keyRowOf c hU n (portRowW c) g with
  | some r => some r
  | none => outRowOf c hU n outs g

end FlockSoundness.Discharge.HiddenExec

namespace Flock.Bridge

/-- A port's frame-v3 domain: its name in the circuit file. -/
def domOf (c : Flock.Circuit) (q : ℕ) : ByteArray := ((c.ports[q]?.map (·.name)).getD "").toUTF8

/-- A port's leaf schema: `Registered.portSchema`'s, as C-Flock frames the port's rows (`HmRow.checkPublic`). -/
def schemaOf (c : Flock.Circuit) (q : ℕ) : String :=
  (c.ports[q]?.map Flock.Registered.portSchema).getD Flock.HmRow.ROW_SCHEMA_V2

/-- **The registered tree leaf of row `(u, q)`**: C-Flock's hm96 tree leaf of the public file's `b ‖ c` there. -/
def regLeafOf (c : Flock.Circuit) (pub : Flock.Public) (q u : ℕ) : ByteArray :=
  Flock.Hm96.Default512.treeLeaf (pub.digest u q c.ports.size)

/-- The leaves of port `q`'s registered tree. -/
def leavesOf (c : Flock.Circuit) (pub : Flock.Public) (nl q : ℕ) : Array ByteArray :=
  (Array.range nl).map fun u => Flock.Registered.leafAs (schemaOf c q) (domOf c q) u (regLeafOf c pub q u)

end Flock.Bridge
