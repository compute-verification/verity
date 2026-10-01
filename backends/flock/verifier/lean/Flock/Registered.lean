/-
PROTOCOL.md §16.10 (registered reads); `protocols/sampled_proofs/README.md`, "Registered values": a statement's rows of a
registered value (a program's weights, constants and tables, committed once at registration under `hm96-sha512/row/v1`
leaves), opened against the verifier's own registered roots at the positions its own program reads.
-/
import Flock.HmRow

namespace Flock
namespace Registered

open Lean (Json)

def FORMAT : String := "verity/registered-reads/v0"

/-- A port of the verifier's own `--registered` file: the value its rows are rows of, the value's registered root, leaf
count and frame-v3-sha512 domain, and the position every unit of the population reads (none: a unit of another
template). -/
structure Port where
  name : String
  value : String
  root : ByteArray
  leaves : Nat
  domain : ByteArray
  reads : Array (Option Nat)
  deriving Inhabited

def ofJson (j : Json) : V (Array Port) := do
  let ctx := "registered reads"
  if (← J.str j "format" ctx) != FORMAT then throw s!"{ctx}: not a {FORMAT} file"
  let .ok (.obj kvs) := j.getObjVal? "ports" | throw s!"{ctx}: ports is not an object"
  if kvs.toArray.isEmpty then throw s!"{ctx}: no port"
  kvs.toArray.mapM fun (name, e) => do
    let root ← unhex (← J.str e "root" ctx)
    let domain ← unhex (← J.str e "domain" ctx)
    let leaves ← J.nat e "leaves" ctx
    if root.size != 64 || domain.size != 64 || leaves == 0 then
      throw s!"{ctx}: port {name}: a 64-byte root and domain, and at least one leaf"
    let reads ← (← J.arr e "reads" ctx).mapM fun r => match r with
      | .null => pure none
      | r => some <$> J.asNat r s!"{ctx}: a read position"
    return { name, value := ← J.str e "value" ctx, root, leaves, domain, reads }

/-- The frame-v3-sha512 root a leaf at `i` climbs to through `path` (the siblings, leaf level first, 64 bytes each). -/
def climb (dom leaf : ByteArray) (i depth : Nat) (path : ByteArray) : ByteArray := Id.run do
  let mut cur := leaf
  let mut idx := i
  for d in [0:depth] do
    let sib := path.extract (64 * d) (64 * d + 64)
    cur := if idx % 2 == 0 then FV3.hash "node" [dom, FV3.uint d, FV3.uint (idx / 2), cur, sib] Sha512.hash
      else FV3.hash "node" [dom, FV3.uint d, FV3.uint (idx / 2), sib, cur] Sha512.hash
    idx := idx / 2
  return cur

/-- What a verdict reports: each registered port's value and the root its reads were opened against. -/
def finding (own : Array Port) : Json :=
  Json.mkObj (own.toList.map fun p => (p.name, Json.mkObj [("value", Json.str p.value), ("root", Json.str (hexOf p.root))]))

/-- The statement's registered reads against the verifier's own `own` (`--registered`). The header's `registered` must
name exactly `own`'s ports, each a port of the circuit and a table of a shared-row file: every table row's `b ‖ c`, as the
value's row at its stated position, opens the registered root through its path, and every instance reads, by its ref,
the position the verifier's program reads for its unit. Without `own`, a statement that names registered reads is
refused: its rows would be checked against no root. -/
def check (c : Circuit) (pub : Public) (own : Option (Array Port)) : V Unit := do
  let hdr := pub.header.getObjVal? "registered"
  let some own := own
    | if hdr.isOk then throw "registered reads: the statement reads registered values, and the verifier holds no registered roots (--registered)"
  let .ok (.obj kvs) := hdr
    | throw "registered reads: the statement names no registered rows: a registered value's rows committed afresh open no registered root"
  for (name, _) in kvs.toArray do
    if !own.any (·.name == name) then throw s!"registered reads: the statement names port {name} registered, and the verifier's registered reads do not"
  let some t := pub.tables | throw "registered reads: a registered port is a table of a shared-row file"
  let np := c.ports.size
  let ctx := "public file.registered"
  let indices ← (← J.arr (← J.get pub.header "units" "public file") "indices" "public file").mapM (J.asNat · "a unit index")
  for reg in own do
    let some p := (List.range np).find? (fun p => c.ports[p]!.name == reg.name)
      | throw s!"registered reads: {reg.name} is not an input port of the circuit"
    let .ok e := pub.header.getObjVal? "registered" >>= (·.getObjVal? reg.name)
      | throw s!"registered reads: the statement commits port {reg.name}'s rows of {reg.value} afresh, with no path to its registered root"
    if (← J.str e "value" ctx) != reg.value then throw s!"registered reads: port {reg.name} reads {reg.value}, not another value"
    let positions ← (← J.arr e "positions" ctx).mapM (J.asNat · "a registered position")
    let paths ← (← J.arr e "paths" ctx).mapM (J.asStr · "a registered path")
    let rows := t.counts[p]!
    if positions.size != rows || paths.size != rows then
      throw s!"registered reads: port {reg.name}: one position and one path per table row"
    let depth := log2ceil reg.leaves
    for r in [0:rows] do
      let pos := positions[r]!
      let path ← unhex paths[r]!
      let bc := pub.digests.getD (t.starts[p]! + r) ByteArray.empty
      let leaf := FV3.leaf reg.domain pos HmRow.ROW_SCHEMA (Hm96.Default512.treeLeaf bc) Sha512.hash
      if pos ≥ reg.leaves || path.size != 64 * depth || climb reg.domain leaf pos depth path != reg.root then
        throw s!"registered reads: port {reg.name}: table row {r} does not open {reg.value}'s registered root at position {pos}"
    for i in [0:pub.n] do
      let u := indices.getD i 0
      let got := positions.getD (t.refs.getD (i * np + p) 0) 0
      match reg.reads[u]? with
      | some (some want) =>
        if got != want then
          throw s!"registered reads: port {reg.name}: unit {u} reads position {got}, and the verifier's program reads {want}"
      | _ => throw s!"registered reads: port {reg.name}: the verifier's program reads no row of {reg.value} in unit {u}"

end Registered
end Flock
