module

public import Flock.Compose
public import Flock.ProvedScope
public import Flock.Zk
public import Flock.Test.Lower
meta import Flock.Compose
meta import Flock.Test.Lower

/-!
`Compose.compose`'s files read back by the door's reader (`ProvedScope.circuitOf` under the statement's `--zk` tags, so
`HmOut.parseCommit`: every check `HmRow.parse` makes, the `sha512x3` and `hm96` sections against the verifier's own nets,
`checkRows`, and for a template `HmOut.parseTmpl`'s), then compared with the block each was written from: the unit (its
net, or its template's text), the input rows, the output ports, the dummy outputs and every row's pinned dummy `b ‖ c`
against the padding instance's (`HmOut.checkPublic`'s rule). The blocks: a flat toy unit, and `Flock.Test.Lower`'s 2-bit
adder lowered to its template. `roundTrips` builds both row nets per file, which takes minutes in the interpreter, so
`flock-compose-check` runs it compiled (`tests/verity/protocols/proofs/flock/prover/test_lean_compose.py`); the guards
here are the parts that build no row net.
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

def toy : Block := { name := "toy", body := .flat toyLaid, ins := #[("x", 20), ("y", 16)], outs := #[("o", 1)] }

/-- The adder's block: its four input bits in one 16-bit row, its four output bits in one word. -/
def adder : V Block := do
  pure { name := "adder", body := .tmpl (← Test.Lower.tmpl) "iv", ins := #[("x", 16)], outs := #[("o", 1)] }

def hex128 : String := "".pushn 'a' 128

def tagsOf (b : Block) : Tags := match b.body with
  | .flat _ => Tags.circuit
  | .tmpl .. => Tags.circuitTmpl

def read (b : Block) : V Circuit := do
  let f ← compose b hex128 hex128
  ProvedScope.circuitOf (Zk.tags (tagsOf b) (Zk.maskLogOf f) 1) f []

/-- The parsed circuit is the block's: its unit, input rows, output ports, and dummy. -/
def agrees (b : Block) (c : Circuit) : V Bool := do
  let unitOk ← match b.body with
    | .flat l => do pure (decide ((c.nets.getD c.unitNet default).2 = (← HmNets.netOf b.name l)))
    | .tmpl t _ => pure ((c.tmpl.map fun tu => Lower.render tu.t) == some (Lower.render t))
  let u ← b.body.part b.name
  pure (unitOk && c.name == b.name
    && (c.ports.extract 0 b.ins.size).map (fun p => (p.name, p.bits)) == b.ins
    && c.outPorts == b.outs
    && c.dummyOutputs == u.zero
    && (List.range c.ports.size).all fun p =>
        HmOut.rowBcZeroSalt (c.ports.getD p default) (HmOut.dummyRow c p) == c.dummyDigests.getD p ByteArray.empty)

/-- Each block's file read back by the door's reader and agreeing with the block. -/
def roundTrips : V Unit := do
  for b in [toy, ← adder] do
    if !(← agrees b (← read b)) then throw s!"{b.name}: the file read back is not the block's"

/-! What builds no row net. -/

-- the flat unit's section reads back as its net
#guard match Net.parse ((Laid.write "toy" toyLaid).splitOn "\n").toArray.pop, HmNets.netOf "toy" toyLaid with
  | .ok a, .ok b => decide (a = b)
  | _, _ => false
#guard zeroOutputs toyLaid == #[1]
-- the adder on 0 and 0 is 8 (`Flock.Test.Lower.WORDS`)
#guard match Test.Lower.tmpl with
  | .ok t => zeroOutputsT t == #[8]
  | .error _ => false
-- the adder's template is read at its layout mode
#guard match adder with
  | .ok b => (b.body.part b.name).toOption.isSome
  | .error _ => false
-- a unit whose leaves are not the rows' words, or of a layout mode that isn't one
#guard (compose { toy with ins := #[("x", 16), ("y", 16)] } hex128 hex128).toOption.isNone
#guard match adder with
  | .ok b => (compose { b with body := match b.body with | .tmpl t _ => .tmpl t "flat" | x => x } hex128 hex128).toOption.isNone
  | .error _ => false

end Flock.Test.Compose
