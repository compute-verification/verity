module

public import Spot.OneStageZkStatement
public import Spot.Docket
public import Spot.Partition
public import Spot.Registered
public import Flock

/-!
The members' reads of C's input nodes, from C and the partition alone: the nodes each member's units read
(`memberInputs`), every port of a registered input held to its registration of record (`pinsHeld`), and a run input
node's one tree held to C's leaf count and the domain under the window's open root (`runTreeHeld`), which
`OneStageZk.runInputRootsAt` reads.
-/

@[expose] public section

namespace Spot.OneStageZk

open Lean (Json)
open Flock (V)

/-- The refusal of member `member`'s registered port `port` reading the registered input `value` at another `what` (root,
leaf count, domain) than the pin the docket's R8 accepted (`pinsHeld`). -/
def PINNED_READ_WHY (member port value what : String) : String :=
  s!"registered: member {member}'s port {port} reads the registered input {value} at another {what} than its pin of " ++
    "record: a registered input the weights' registration of record pins is read at the pin's root, leaf count and domain"

/-- The refusal of member `member`'s registered port `port` reading the registered input `value`, which the program marks,
R8 accepted no pin of and whose root the verifier holds none of (`pinsHeld`). -/
def UNPINNED_READ_WHY (member port value : String) : String :=
  s!"registered: member {member}'s port {port} reads the registered input {value}, which the program marks, R8 " ++
    "accepted no pin of and the verifier holds no root of: a marked input is read only at a pin R8 accepted or a " ++
    "root the verifier holds"

/-- The refusal of member `member`, whose units read its parameter `param` at the two input nodes `w0` and `w1` of C
(`memberInputs`). -/
def TWO_READS_WHY (member param : String) (w0 w1 : Nat) : String :=
  s!"registered reads: member {member}'s units read parameter {param} at C's input nodes {w0} and {w1}: a parameter " ++
    "that reads an input reads one node, each port one tree"

/-- The input nodes the members `[m0, m0 + n)` of root node `k` read their parameter `p` of type `ty` at
(`Flock.OneCircuit.regRows?`): the whole parameter's, else, for an array whose whole has no closed form, each row's,
which must be one node, an input at every row or at none; the whole parameter's refusal otherwise. -/
def paramInput (P : Flock.Program.Prog) (k m0 n : Nat) (p : String) (ty : Flock.Program.Ty) : V (Option (Nat × Nat)) :=
  match Flock.OneCircuit.regRows? P k m0 n p, ty with
  | .ok r, _ => pure r
  | .error e, .array c _ =>
    match (List.range c).map fun i => Flock.OneCircuit.regRows? P k m0 n s!"{p}[{i}]" with
    | .ok r0 :: rows => if rows.all (fun | .ok r => r == r0 | .error _ => false) then pure r0 else throw e
    | _ => throw e
  | .error e, _ => throw e

/-- The refusal of member `member`'s units meeting a range the partition gives at root node `k`, which C does not have
(`memberInputs`). -/
def RANGE_NODE_WHY (member : String) (k : Nat) : String :=
  s!"member {member}: its units meet a range the partition gives at C's root node {k}, which C does not have: a " ++
    "range is of C's root nodes"

/-- **The input nodes member `m`'s units read**, each `(parameter, node, row leaves)` once: over every range of `rs`
its units meet, each parameter of the range's node's Definition that its units there read at an input node of C,
registered or run (`paramInput`), all of them at once; a form `regRows?` refuses is refused, and so are two input nodes
at one parameter (`TWO_READS_WHY`), and a range at a root node C does not have (`RANGE_NODE_WHY`). Every proof block the member holds, from C and the partition, never its files. Its
work is the member's ranges times their Definitions' parameters. -/
def memberInputs (P : Flock.Program.Prog) (rs : Array Flock.TemplateQuery.TRange) (m : Member) :
    V (Array (String × Nat × Nat)) :=
  m.units.foldlM (init := #[]) fun acc (lo, hi) => rs.foldlM (init := acc) fun acc r => do
    let a := max lo r.first
    let b := min hi (r.first + r.count)
    if b ≤ a then return acc
    let some x := (Flock.Registered.rootNodes P)[r.node]? | throw (RANGE_NODE_WHY m.name r.node)
    let some f := P.defs[x.fn]? | throw (RANGE_NODE_WHY m.name r.node)
    f.params.foldlM (init := acc) fun acc (p, ty) => do
      match ← (paramInput P r.node (a - r.first) (b - a) p ty).mapError (s!"member {m.name}: {·}") with
      | none => pure acc
      | some (w, len) =>
        match acc.find? (·.1 == p) with
        | some (_, w0, _) => if w0 == w then pure acc else throw (TWO_READS_WHY m.name p w0 w)
        | none => pure (acc.push (p, w, len))

/-- The template parameter a registered port reads: its `param`, else its name, without a row index (`paramRow`). -/
def portParam (q : Flock.Registered.Port) : String :=
  (Flock.Registered.paramRow (if q.param.isEmpty then q.name else q.param)).1

/-- The registration of record's entry of the registered input `v`: R8's accepted pin (`rp.pins`), else the root the
verifier itself holds of it (`rp.roots`, `Docket.heldEntries`), else round 1's record's entry of a value a link names
(`rp.linked`); as `(root, leaves, domain)`. No arm reads a root the record names: a root the prover registers is held
to nothing but itself. -/
def pinOf (rp : Docket.ReadPins) (v : String) : Option (String × Nat × String) :=
  match rp.pins.find? (·.1.value == v) with
  | some (e, root) => some (root, e.leaves, e.domain)
  | none => match rp.roots.find? (·.1.value == v) with
    | some (e, root) => some (root, e.leaves, e.domain)
    | none => (rp.linked.find? (·.value == v)).bind fun l =>
      match getD l.entry "root", nat? (getD l.entry "leaves"), getD l.entry "domain" with
      | .str r, some n, .str d => some (r, n, d)
      | _, _, _ => none

/-- **Every port of a registered input is at its registration of record**: for each member, at each parameter its
units read at a registered input node of C (`memberInputs`) whose gates are a marked value `v`'s
(`Partition.heldValue` over `marks`), every port of its registered file at that parameter, a tree's or not, is at `v`'s
entry of record (`pinOf`): its root, leaf count and domain, refused by name otherwise (`PINNED_READ_WHY`, or
`UNPINNED_READ_WHY` where there is none); and every port naming a marked input the verifier does not expect
(`rp.loose`) is at its pin. So no port of a registered node is exempt as a tree port. Every member, sampled or not,
before any coin. -/
def pinsHeld (P : Flock.Program.Prog) (rs : Array Flock.TemplateQuery.TRange) (marks : Array (String × Nat × Nat))
    (rp : Docket.ReadPins) (ms : Array Member) : V Unit :=
  ms.forM fun m => do
    let own ← match m.own.registered with
      | none => pure #[]
      | some j => (Flock.Registered.ofJson j).mapError (s!"member {m.name}: {·}")
    let reads ← memberInputs P rs m
    own.forM fun q => do
      let v := (reads.find? (·.1 == portParam q)).bind fun (_, w, _) => Partition.heldValue P marks w
      let v := v.orElse fun _ => if rp.loose.contains q.value then some q.value else none
      match v with
      | none => pure ()
      | some v => match pinOf rp v with
        | none => throw (UNPINNED_READ_WHY m.name q.name v)
        | some (root, leaves, domain) =>
          if Flock.hexOf q.root != root then throw (PINNED_READ_WHY m.name q.name v "root")
          else if q.leaves != leaves then throw (PINNED_READ_WHY m.name q.name v "leaf count")
          else if Flock.hexOf q.domain != domain then throw (PINNED_READ_WHY m.name q.name v "domain")
          else pure ()

/-- The refusal of member `member` whose units read C's run input node `w` at `port` where the record holds no root of
its one tree `input/n<w>` (`runInputRootsAt`). -/
def RUN_TREE_MISSING_WHY (member port : String) (w : Nat) : String :=
  s!"member {member}: port {port} reads C's run input node {w}, and the record holds no root of " ++
    s!"{Flock.Registered.freshValue w}: a run input node's rows are committed once, in its one tree, before any coin"

/-- The refusal of the record's tree `input/n<w>` of run input node `w` at another `what` (leaf count, domain) than C's
node and the window's open root give it (`runTreeHeld`). -/
def RUN_TREE_WHY (w : Nat) (what : String) : String :=
  s!"registered: the record's {Flock.Registered.freshValue w} has another {what} than C's run input node {w} and the " ++
    "window's open root give it: a run input node's one tree binds its window"

/-- The refusal of run input node `w`'s tree under a verifier that names no window open root (`runTreeHeld`). -/
def RUN_TREE_UNBOUND_WHY (w : Nat) : String :=
  s!"registered: C's run input node {w} is read from its one tree {Flock.Registered.freshValue w}, and the verifier " ++
    "names no window open root (window_root) for its domain to bind"

/-- **A run input node's one tree binds its window**: the record's entry `r` of `input/n<w>` has the leaf count C's
node `w` gives it, its rows `outLeaves · members / len` for rows of `len` leaves (as `OneCircuit.pos` counts them),
and the domain the verifier computes itself (`Spot.Registered.domainAt`): the program's digest `prog`, the value
`input/n<w>`, a row's 16-bit words (`len` leaves of the node's leaf width, as the writer packs a port's row), those rows
and 16-bit words, beside the window's open root `R`, which the per-member form
reads only where the challenger's signed coins verify under it (`coinsOf`, `treesOf`). The domain's `schema` is a
constant of the domain (`hm96-sha512/row/v1`), not the rows' format. Refused by name otherwise (`RUN_TREE_WHY`), and
with no `R` (`RUN_TREE_UNBOUND_WHY`). -/
def runTreeHeld (P : Flock.Program.Prog) (prog : String) (R : Option String) (w len : Nat) (r : Json) : V Unit := do
  let some R := R | throw (RUN_TREE_UNBOUND_WHY w)
  let some wn := (Flock.Registered.rootNodes P)[w]? | throw (RUN_TREE_WHY w "leaf count")
  let some f := P.defs[wn.fn]? | throw (RUN_TREE_WHY w "leaf count")
  let rows := if len == 0 then 0 else f.outLeaves * wn.members / len
  if nat? (getD r "leaves") != some rows then throw (RUN_TREE_WHY w "leaf count")
  let words := if f.outLeaves == 0 then 0 else len * (f.ret.widths.foldl (· + ·) 0 / f.outLeaves) / 16
  let d ← Spot.Registered.domainAt prog (Flock.Registered.freshValue w) words rows 16 (some R)
  if getD r "domain" != .str (Flock.hexOf d) then throw (RUN_TREE_WHY w "domain")

end Spot.OneStageZk
