/-
`Flock.Test.ToyPubFiles`' files parse to what it states, by running the verifier's functions on them: the circuit file
to `circuit` (`Desc.parse`), in the hidden-output headlines' scope (`ProvedScope.scopeOkH`), with each port named once
(`Registered.dupName`); the public file loads at it to `pub` (`HmOut.loadPublic`); and the draw is a draw of unit 0
(`Draw.UnitDraw.ofJson`) whose drawn statement (`HmOut.drawn`) is unit 0's (`Registered.unitIndices`); and the two
ports' registered roots differ. Each written field of `circuit` and `pub` is compared with the parse's; the rest are
the parse's by definition. `check` parses the 56 MB circuit file once: about nine minutes in the interpreter, so it
runs compiled, as `flock-witness toy-pub` (`test_toy_pub.py`), and not as a `#guard`.
-/
import Flock.Test.ToyPubFiles

namespace Flock.Test.ToyPub

open Lean Flock.Test.ToyPubFiles

def groupEq (a b : Group) : Bool := a.col == b.col && a.words == b.words && a.bits == b.bits

def groupsEq (a b : Array Group) : Bool := a.size == b.size && (a.zip b).all fun (x, y) => groupEq x y

def sparseEq (a b : Sparse) : Bool :=
  a.rows == b.rows && a.width == b.width && a.data.data == b.data.data && a.start == b.start && a.len == b.len

def netEq (a b : Net) : Bool :=
  a.name == b.name && a.useful == b.useful && a.constPos == b.constPos && groupsEq a.inGroups b.inGroups &&
    groupsEq a.outGroups b.outGroups && a.inWords == b.inWords && sparseEq a.a b.a && sparseEq a.b b.b &&
    a.unitLog == b.unitLog && a.lookup.isNone && b.lookup.isNone

def portEq (a b : Port) : Bool :=
  a.name == b.name && a.words == b.words && a.chunks == b.chunks && a.nb == b.nb && a.bits == b.bits &&
    a.v2 == b.v2 && a.segs == b.segs

/-- The written fields of `circuit` are `c`'s. -/
def circuitWritten (c : Circuit) : Bool :=
  let w := ToyPubFiles.circuit
  c.name == w.name && c.nets.size == 3 && (c.nets.getD 0 default).1 == "unit" &&
    netEq (c.nets.getD 0 default).2 unitNet &&
    c.ports.size == w.ports.size && (c.ports.zip w.ports).all (fun (x, y) => portEq x y) &&
    c.outPorts == w.outPorts && c.outRowPorts == w.outRowPorts && c.g == w.g && c.kLog == w.kLog &&
    c.upv == w.upv && c.unitNet == w.unitNet && c.leavesIn == w.leavesIn && c.leavesOut == w.leavesOut &&
    c.key.data == w.key.data && c.dummyOutputs == w.dummyOutputs && c.leafScheme == w.leafScheme &&
    c.perVu == w.perVu && c.shaNet == w.shaNet && c.hmNet == w.hmNet && c.outNet == w.outNet &&
    c.outNetPorts == w.outNetPorts && c.outNetSlots == w.outNetSlots && c.leafCuts == w.leafCuts &&
    c.typed.isNone && c.cls.isNone && c.tmpl.isNone

/-- The written fields of `pub` are `p`'s. -/
def pubWritten (p : Public) : Bool :=
  p.n == pub.n && p.digests.map (·.data) == pub.digests.map (·.data) && p.outputs == pub.outputs &&
    p.population.isNone && p.tables.isNone

/-- Every conjunct, in one pass over the files. -/
def check : Bool :=
  match Desc.parse tags circuitFile [] with
  | .error _ => false
  | .ok D =>
    circuitWritten D.unroll && ProvedScope.scopeOkH circuit && (Registered.dupName circuit).isNone &&
    (root 0).data != (root 1).data &&
    (match HmOut.loadPublic tags circuit publicFile with
      | .ok p => pubWritten p
      | .error _ => false) &&
    (match Draw.UnitDraw.ofJson draw with
      | .ok d => d.proved == #[0]
      | .error _ => false) &&
    (match HmOut.drawn circuit pub draw with
      | .ok dp => (Registered.unitIndices dp).toOption == some #[0]
      | .error _ => false)

-- the session's tags don't read the public file
example : tags = tagsAt ByteArray.empty := rfl

/-- `flock-witness toy-pub`: `check`'s verdict, exit code 0 exactly when it holds. -/
def cmd : IO UInt32 := do
  let ok := check
  IO.println s!"TOYPUB {ok}"
  pure (if ok then 0 else 1)

end Flock.Test.ToyPub
