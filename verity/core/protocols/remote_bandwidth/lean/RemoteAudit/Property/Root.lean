import Std.Data.HashMap
import Flock.Program
import RemoteAudit.Property.Pass

/-!
# property/v1's reading of a Program's root, at its step ends

What `stacking.analyze` reads of census/v1, computed from the Program itself at the step ends it is given, so the census
it reads is the Program's by construction. The root's first nodes are its inputs and every later node is a Call; step s
is root nodes `[ends[s − 1], ends[s])` (`ends[−1]` the input count). A Call touches a root value's leaf when one of its
gates takes the leaf as an operand or among a selection's candidates (`census._Analysis.reads`, both alike: a touch is a
touch), what a Definition reads of its parameters memoized per Definition and multiplied out through `batch` members
and `scan` iterations, as the census marks them (`_Marks`).

From those touches: each step's units (the pass's tallies of its Calls), the run-input bits it first reads (a run
input's leaf is born in the step before the first that touches it), and the longest any dynamic value is held (a
computed one from its producer's step, a run input from its birth, to the last step that touches it). A root Call that
passes one of its arguments through, whose readers the census would hand to the value passed, is refused (`Root`); a
Call's own repeats of a gate change no step's numbers.
-/

namespace RemoteAudit.Property

open Flock.Program

def MEMBER_LIMIT : Nat := 2 ^ 22

abbrev Marks := Std.HashMap Nat (Array (Nat × Nat))

private def addMark (into : Marks) (i : Nat) (iv : Ivs) : Marks :=
  if iv.isEmpty then into else into.insert i (into.getD i #[] ++ iv)

/-- `_Marks.walk`: the leaf intervals parts name in one space (`p` inside a Definition, `n` at the root). -/
def walkParts (ps : Array Part) (space : Bool) (into : Marks) : Except String Marks := do
  let mut into := into
  for ((p, i), iv) in ← partsIvs ps do
    if p == space then into := addMark into i iv
  return into

/-- `_Marks.mark`: what reading `sub` of parts reads. -/
def markParts (ps : Array Part) (sub : Ivs) (space : Bool) (into : Marks) : Except String Marks := do
  if total sub == partsCount ps then return ← walkParts ps space into
  let mut into := into
  for (lo, hi) in sub do into ← walkParts (sliceParts ps lo hi) space into
  return into

private def axisWalk : Ty → Nat → Option (Nat × Ty)
  | t, 0 => some (1, t)
  | .array n e, ax + 1 => (axisWalk e ax).map fun (o, i) => (n * o, i)
  | _, _ => none

/-- `Coll.take(axis, m)`'s parts: member `m` along array axis `ax`, one run of its leaves per index over the axes
before it. -/
def takeParts (ps : Array Part) (ty : Ty) (ax m : Nat) : Except String (Array Part) := do
  let some (outer, .array d e) := axisWalk ty ax | throw s!"axis {ax} requires nested Arrays"
  let L := e.leaves
  let stride := d * L
  return (Array.range outer).foldl (fun acc k => acc ++ sliceParts ps (k * stride + m * L) (k * stride + (m + 1) * L)) #[]

/-- `_Marks.mapped`: what `n` members (a batch's members, a scan's iterations), each reading `sub` of its part of `a`,
read of `a`. A census approximates past `MEMBER_LIMIT` runs; this reading refuses there. -/
def mapped (a : Arg) (ax : Option Nat) (n : Nat) (sub : Ivs) (space : Bool) (into : Marks) : Except String Marks := do
  if sub.isEmpty || n == 0 then return into
  let some ax := ax | return ← markParts a.seq.parts sub space into
  let L := a.ty.leaves / n
  if total sub == L then return ← walkParts a.seq.parts space into
  if n * sub.size > MEMBER_LIMIT then
    throw s!"a batch's {n} members each read {sub.size} runs of their part: past the {MEMBER_LIMIT} runs this reading follows"
  let mut into := into
  for m in [0:n] do
    if ax == 0 then
      for (lo, hi) in sub do into ← walkParts (sliceParts a.seq.parts (m * L + lo) (m * L + hi)) space into
    else into ← markParts (← takeParts a.seq.parts a.ty ax m) sub space into
  return into

/-- `_Analysis.node`: what a node reads, given what its callee reads of its parameters (`r`). -/
def markNode (nd : Node) (r : Array Ivs) (space : Bool) (into : Marks) : Except String Marks := do
  let mut into := into
  match nd.form with
  | .prim | .call => for (a, sub) in nd.args.zip r do into ← markParts a.seq.parts sub space into
  | .batch =>
    for i in [0:min nd.args.size r.size] do into ← mapped nd.args[i]! (nd.axes.getD i none) nd.n r[i]! space into
  | .scan =>
    let k := nd.nxs
    if let (some a, some sub) := (nd.args[0]?, r[0]?) then into ← markParts a.seq.parts sub space into
    for i in [1:min nd.args.size r.size] do
      into ← if i ≤ k then mapped nd.args[i]! (some 0) nd.n r[i]! space into
        else markParts nd.args[i]!.seq.parts r[i]! space into
  return into

abbrev RM := StateT (Std.HashMap Nat (Array Ivs)) (Except String)

/-- `_Analysis.reads`: what one activation of Definition `fi` reads of each parameter (a primitive all of them). A
callee's index is below its caller's, so `fi + 1` is fuel enough. -/
def reads (P : Prog) (fi : Nat) (fuel : Nat := fi + 1) : RM (Array Ivs) := do
  let fuel + 1 := fuel | throw DEEP
  if let some hit := (← get)[fi]? then return hit
  let d := P.defs[fi]!
  let r ← if d.prim then pure (d.params.map fun (_, t) => full t.leaves) else do
    let mut into : Marks := {}
    for nd in d.body.nodes do
      let sub ← reads P nd.fn fuel
      into ← markNode nd sub true into
    pure ((Array.range d.body.params.size).map fun i => norm (into.getD i #[]))
  modify (·.insert fi r)
  return r

/-- Whether Definition `fi` may return one of its parameters' leaves, passed through its nodes (refused at the root);
out of fuel, it may. -/
def mayPass (P : Prog) (fi : Nat) (fuel : Nat := fi + 1) : Bool :=
  match fuel with
  | 0 => true
  | fuel + 1 =>
    let d := P.defs[fi]!
    !d.prim && d.body.ret.parts.any fun q => match q with
      | .explicit it => it.any fun r => r.p || (match d.body.nodes[r.i]? with
        | some nd => mayPass P nd.fn fuel
        | none => false)
      | q => match partSpace? q with
        | some (true, _) => true
        | some (false, i) => match d.body.nodes[i]? with
          | some nd => mayPass P nd.fn fuel
          | none => false
        | none => false

/-- The number of input nodes, refused unless they come first, and the steps' ends, refused unless they increase past
them to the last node (`census.census`). -/
def stepsOf (P : Prog) (ends : Array Nat) : Except String Nat := do
  let nodes := P.defs[P.root]!.body.nodes
  let isIn := fun (nd : Node) => let c := P.defs[nd.fn]!; c.prim && c.binding.isSome
  let I := (nodes.filter isIn).size
  if (nodes.extract 0 I).any (!isIn ·) then throw "a root's input nodes come first (PROTOCOL.md §5.2)"
  let starts := #[I] ++ ends
  if (Array.range ends.size).any (fun s => ends[s]! ≤ starts[s]!) || (ends.size > 0 && ends.back! != nodes.size)
      || (ends.isEmpty && nodes.size > I) then
    throw s!"step ends {ends.extract 0 8} are not increasing root-node indices past the {I} input nodes ending at {nodes.size}"
  return I

def stepOf (ends : Array Nat) (k : Nat) : Nat := (ends.filter (· ≤ k)).size

/-- What property/v1 reads of a Program (`stacking.Analysis`). -/
structure Analysis where
  /-- per step, its units by `BUDGETED`, and the run-input bits it first reads -/
  units : Array (Array Nat)
  inputBits : Array Nat
  /-- each primitive's (and input's) class and operand bits, by id -/
  shapes : Array (String × Option String × Nat)
  /-- refused gates by (code, primitive or Definition) -/
  refused : Array ((String × String) × Nat)
  family : Nat
  macs : Nat
  affine : Nat
  unproven : Array (String × Nat)
  sites : Array ((String × Nat) × (Int × Int) × (Int × Int))
  maxLifetime : Nat

private def ltSS (a b : String × String) : Bool := a.1 < b.1 || (a.1 == b.1 && a.2 < b.2)

/-- `stacking.analyze` of the Program at `ends`. Ports on a Definition other than the root (`ported`) are refused. -/
def analyze (P : Prog) (ends : Array Nat) (ported : Array String) : Except String Analysis := do
  let rootKey := P.defs[P.root]!.key
  if let some k := ported.find? (· != rootKey) then
    throw s!"definition '{k}' declares ports: this verdict reads no port declarations yet"
  let I ← stepsOf P ends
  let S := ends.size
  let nodes := P.defs[P.root]!.body.nodes
  for j in [I:nodes.size] do
    if mayPass P nodes[j]!.fn then
      throw s!"root node {j} ({P.defs[nodes[j]!.fn]!.key}) may pass an argument through: not covered by this verdict"
  let ((_, _, ts, _), st) ← (walk P P.root #[] #[] rootKey (fuelOf P.root)).run {}
  let mut units : Array (Array Nat) := Array.replicate S #[0, 0, 0, 0]
  let mut whole : Tally := {}
  for (k, t) in ts do
    whole := whole.add t
    let s := stepOf ends k
    if s < S then units := units.modify s fun u => (Array.range 4).map fun c => u[c]! + t.units[c]!
  -- touches: per root node, per step, the leaves the step's Calls read
  let mut touched : Array (Array (Array (Nat × Nat))) := Array.replicate nodes.size (Array.replicate S #[])
  let mut memo : Std.HashMap Nat (Array Ivs) := {}
  for k in [I:nodes.size] do
    let nd := nodes[k]!
    let (r, m) ← (reads P nd.fn).run memo
    memo := m
    let into ← markNode nd r false {}
    let s := stepOf ends k
    for (j, iv) in into.toArray do
      if j < nodes.size && s < S then touched := touched.modify j fun row => row.modify s (· ++ iv)
  let mut inputBits := Array.replicate S 0
  let mut longest := 0
  for j in [0:nodes.size] do
    let rows := touched[j]!.map norm
    let nd := nodes[j]!
    if j < I then
      if P.defs[nd.fn]!.binding != some RUN then continue
      let some w := homogeneous nd.out | throw s!"input node {j}: an input's leaves have one width"
      let mut seen : Ivs := #[]
      let mut news : Array Ivs := #[]
      for s in [0:S] do
        let n := minus rows[s]! seen
        news := news.push n
        inputBits := inputBits.modify s (· + total n * w)
        seen := norm (seen ++ n)
        if !rows[s]!.isEmpty then
          match (Array.range (s + 1)).find? (fun i => !(isect rows[s]! news[i]!).isEmpty) with
          | some f => longest := max longest (s - f + 1)
          | none => pure ()
    else
      let birth := stepOf ends j
      let last := (Array.range S).foldl (fun acc s => if rows[s]!.isEmpty then acc else max acc s) birth
      longest := max longest (last - birth)
  return {
    units, inputBits
    shapes := st.shapes.toArray.qsort (fun a b => a.1 < b.1)
    refused := whole.refused.toArray.qsort (fun a b => ltSS a.1 b.1)
    family := whole.family, macs := whole.macs, affine := whole.affine
    unproven := whole.unproven.toArray.qsort (fun a b => a.1 < b.1)
    sites := st.sites.toArray.qsort (fun a b => a.1.1 < b.1.1 || (a.1.1 == b.1.1 && a.1.2 < b.1.2))
    maxLifetime := longest }

end RemoteAudit.Property
