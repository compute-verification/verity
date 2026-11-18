module

public import Flock.Compose
public import Flock.ProvedScope
public import Flock.Zk
meta import Flock.Compose
meta import Flock.ProvedScope
meta import Flock.Zk

/-!
`Compose.compose`'s file read back by the door's reader (`ProvedScope.circuitOf` under `Tags.circuit`'s `--zk` tags, so
`HmOut.parseCommit`: every check `HmRow.parse` makes, the `sha512x3` and `hm96` sections against the verifier's own nets,
and `checkRows`), then compared with the block it was written from: the unit's net, the input rows, the output ports,
and the pinned dummy `b ‖ c` of every row against the padding instance's (`HmOut.checkPublic`'s rule).
-/

@[expose] public section

namespace Flock.Test.Compose

open Flock.Compose

/-- Rows `x` (20 bits, two leaves) and `y` (16 bits), one output word: bit `i` is `x_i y_i ⊕ x_{16+i}`, bit 0 negated, so
the all-zero input gives 1. -/
def toyLaid : HmNets.Laid :=
  let p : HmNets.M (Array HmNets.Lin) := do
    let x := (← HmNets.input 48).toArray
    (Array.range 16).mapM fun i => do
      let a ← HmNets.and (x.getD i default) (x.getD (32 + i) default)
      pure (a ^^^ x.getD (16 + i) default ^^^ (if i == 0 then HmNets.Lin.one else HmNets.Lin.zero))
  let (outs, st) := p.run HmNets.St.empty
  HmNets.layout st #[16, 16, 16] #[16] outs

def toy : Block := { name := "toy", laid := toyLaid, ins := #[("x", 20), ("y", 16)], outs := #[("o", 1)] }

def hex128 : String := "".pushn 'a' 128

def read (b : Block) : V Circuit := do
  let f ← compose b hex128 hex128
  ProvedScope.circuitOf (Zk.tags Tags.circuit (Zk.maskLogOf f) 1) f []

/-- The parsed circuit is the block's: its unit, input rows, output ports, and dummy `b ‖ c`. -/
def agrees (b : Block) (c : Circuit) : Bool :=
  let k := b.ins.size
  c.name == b.name
    && (match HmNets.netOf b.name b.laid with
        | .ok n => decide ((c.nets.getD c.unitNet default).2 = n)
        | .error _ => false)
    && (c.ports.extract 0 k).map (fun p => (p.name, p.bits)) == b.ins
    && c.outPorts == b.outs
    && c.dummyOutputs == zeroOutputs b.laid
    && (List.range c.ports.size).all fun p =>
        HmOut.rowBcZeroSalt (c.ports.getD p default) (HmOut.dummyRow c p) == c.dummyDigests.getD p ByteArray.empty

#guard zeroOutputs toyLaid == #[1]
#guard match read toy with
  | .ok c => agrees toy c
  | .error _ => false
-- a unit whose leaves are not the rows' words
#guard (compose { toy with ins := #[("x", 16), ("y", 16)] } hex128 hex128).toOption.isNone

end Flock.Test.Compose
