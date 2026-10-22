/-
PROTOCOL.md §16.10 (registered reads), a value the verifier registers at known content (`Registered.KNOWN_KEY`): the known
content binds a statement's row only where the circuit reads it. `check` refuses a circuit one of whose input ports the
`--registered` file names at known content has a row bit that no computed row of the unit reads, at any copy: the
statement would then not depend on that bit, and its known content would bind nothing there.

For each such bit it finds a copy and an input column holding it (`cands`, a search the proof does not rely on) and accepts
the bit only where the parsed circuit itself says so: the copy is one of the unit's (`upv`), a computed row reads the
column (`RecResiduals.readIns`), and Δ's source key at the column is that bit of the port (`srcMsg`, which mirrors the
proof's `srcKey` at a message bit from a leaf). Its cost is the known bits times the unit's input ports, never the unit's gates.
-/
import Flock.RegKnown
import Flock.RecResiduals
import Flock.ProvedScope

namespace Flock
namespace KnownRead

/-- Each row port's first word among the message's words (`HmRow.delta`'s `starts`). -/
def starts (c : Circuit) : Array Nat :=
  (c.ports.foldl (fun (acc : Array Nat × Nat) p => (acc.1.push acc.2, acc.2 + p.words)) (#[], 0)).1

/-- The input port an input column `off` is a bit of: the first whose columns hold it. -/
def portOf (pc : Array (Nat × Nat)) (off : Nat) : Nat :=
  ((List.range pc.size).find? fun p => decide (pc[p]!.1 ≤ off ∧ off < pc[p]!.1 + pc[p]!.2)).getD 0

/-- **The message bit copy `k`'s input port `p`, bit `t` is copied from**, where Δ copies one into it from the leaf word
the copy's leaves name (`leavesIn`): bit `bit` of byte `byte` of row port `q`; none where Δ copies the zero, and at a port
past the leaf group (a cut's or a wire's, which a statement of one input group has none of: `scopeOkH`), where it counts
no bit as read. -/
def srcAtMsg (c : Circuit) (k p t : Nat) : Option (Nat × Nat × Nat) :=
  if p < (c.unit.inGroups.getD 0 default).bits.size then
    if (c.leavesIn.getD k #[])[p]! < 0 then none else
      let x := ((c.leavesIn.getD k #[])[p]!).toNat
      let q := ((List.range (starts c).size).reverse.find? (fun q => (starts c)[q]! ≤ x)).getD 0
      if 16 * (x - (starts c)[q]!) + t < c.ports[q]!.bits then
        some (q, 2 * (x - (starts c)[q]!) + t / 8, t % 8)
      else none
  else none

/-- **The message bit copy `u`'s input column `y` is copied from**, at the port bit it is (`mask`, the unit's
`ProvedScope.portMask`; `pc`, its `portCols`); none at an input padding column. -/
def srcMsg (c : Circuit) (mask : Array Bool) (pc : Array (Nat × Nat)) (u y : Nat) : Option (Nat × Nat × Nat) :=
  if mask.getD y false then srcAtMsg c u (portOf pc y) (y - pc[portOf pc y]!.1) else none

/-- **Where each message word sits among the copies' leaf inputs**: word `x`'s `(copy, input port)` pairs, every copy
below `upv` and every port of the unit's leaf group whose leaf is `x`. -/
def cands (c : Circuit) (words : Nat) : Array (List (Nat × Nat)) :=
  let nb := (c.unit.inGroups.getD 0 default).bits.size
  (List.range c.upv).foldl (fun m u =>
    (List.range nb).foldl (fun m p =>
      let l := (c.leavesIn.getD u #[]).getD p (-1)
      if l < 0 then m else m.modify l.toNat ((u, p) :: ·)) m) (Array.replicate words [])

/-- **Bit `k` of row port `q` is read**: at one of the candidates `cs`, copy `u` is a copy of the unit, a computed row
reads its input column `y` (`rd`, `RecResiduals.readIns`), and Δ copies bit `k` of `q` into it (`srcMsg`). -/
def bitRead (c : Circuit) (mask : Array Bool) (pc : Array (Nat × Nat)) (rd : Array Bool) (cs : List (Nat × Nat))
    (q k : Nat) : Bool :=
  cs.any fun (u, p) =>
    let y := pc[p]!.1 + k % 16
    decide (u < c.upv) && rd.getD y false && decide (srcMsg c mask pc u y = some (q, k / 8, k % 8))

/-- **Every bit of every known input port is read** (PROTOCOL.md §16.10): each port of the circuit's input rows
(`HmOut.nIn`) that the `--registered` file names at known content has each bit of its `16 · words`-bit row read by a
computed row of the unit, at some copy (`bitRead`). An output row the file names is the circuit's own output, not read. -/
def check (c : Circuit) (known : Array (String × Array (Array Nat × ByteArray))) : V Unit :=
  let mask := ProvedScope.portMask c.unit.inGroups (c.unit.inWords * WORD)
  let pc := portCols c.unit.inGroups
  let rd := RecResiduals.readIns c.unit
  let st := starts c
  let cs := cands c (c.ports.foldl (· + ·.words) 0)
  known.toList.forM fun (name, _) =>
    (List.range (HmOut.nIn c)).forM fun q =>
      if c.ports[q]!.name = name then
        match (List.range (16 * c.ports[q]!.words)).find?
            (fun k => !bitRead c mask pc rd (cs.getD (st[q]! + k / 16) []) q k) with
        | some k =>
          throw s!"registered reads: known port {name}: no computed row of the circuit reads bit {k} of its row, so its known content binds nothing there"
        | none => pure ()
      else pure ()

end KnownRead
end Flock
