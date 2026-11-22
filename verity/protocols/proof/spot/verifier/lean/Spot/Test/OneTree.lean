module

public import Spot.OneStageZk
meta import Spot.OneStageZk

/-!
`OneStageZk.memberInputs`, `pinsHeld` and `runInputRootsAt` on small programs, each refusal by name, and the refusals'
texts, which `members` prints word for word.
-/

@[expose] public section

namespace Spot.Test.OneTree

open Lean (Json)
open Flock (V)
open Spot.OneStageZk

private def sharedT : Own :=
  { programJ := Json.null, program := .empty, partition := Json.null, pub := "", registered := none }

private def memberT (name : String) (units : Array (Nat × Nat)) : Member := { name, units, g := 1, own := sharedT }

private def whyT {α : Type} : V α → String
  | .error e => e
  | .ok _ => ""

private def refusal : V Unit → String
  | .error e => e
  | .ok () => ""

/-- A registered file of `ports`, each `(name, value, root's hex digit, leaves)`, its domain `d…`. -/
private def regT (ports : List (String × String × Char × Nat)) : Json :=
  Json.mkObj [("format", Json.str Flock.Registered.FORMAT),
    ("ports", Json.mkObj (ports.map fun (name, value, c, leaves) => (name, Json.mkObj [
      ("domain", Json.str ("".pushn 'd' 128)), ("leaves", Lean.toJson leaves),
      ("root", Json.str ("".pushn c 128)), ("value", Json.str value)])))]

/-- The member `name` with the registered file `reg`. -/
private def readerT (name : String) (reg : Json) : Member :=
  { memberT name #[(0, 1)] with own := { sharedT with registered := some reg } }

/-- The pin of `w` R8 accepts in these guards: root `a…`, 2 leaves, domain `d…`. -/
private def pinEntryT : Spot.Registered.Entry :=
  { value := "w", leaves := 2, words := 1, wordBits := 16, domain := "".pushn 'd' 128 }

private def pinT : Array (Spot.Registered.Entry × String) := #[(pinEntryT, "".pushn 'a' 128)]

/-- `pinsHeld` with nothing derived to read, `loose` marked and `pins` accepted. -/
private def heldNoReads (loose : Array String) (pins : Array (Spot.Registered.Entry × String)) (ms : Array Member) :
    V Unit :=
  pinsHeld default #[] #[] ⟨loose, #[], #[], pins, #[]⟩ ms

-- a member reading the pinned input at the pin's own entry, and one reading only other values: accepted
#guard refusal (heldNoReads #["w"] pinT #[readerT "a" (regT [("p", "w", 'a', 2)]), readerT "b" (regT [("q", "v", 'b', 3)]),
  memberT "c" #[(1, 2)]]) == ""
-- another root, another leaf count: refused by name
#guard whyT (heldNoReads #["w"] pinT #[readerT "a" (regT [("p", "w", 'b', 2)])]) == PINNED_READ_WHY "a" "p" "w" "root"
#guard whyT (heldNoReads #["w"] pinT #[readerT "a" (regT [("p", "w", 'a', 3)])]) ==
  PINNED_READ_WHY "a" "p" "w" "leaf count"
-- another domain: refused by name
#guard whyT (heldNoReads #["w"] #[({ pinEntryT with domain := "".pushn 'e' 128 }, "".pushn 'a' 128)]
  #[readerT "a" (regT [("p", "w", 'a', 2)])]) == PINNED_READ_WHY "a" "p" "w" "domain"
-- a marked input R8 accepted no pin of: refused by name; with nothing marked, no read is held
#guard whyT (heldNoReads #["w", "u"] pinT #[readerT "a" (regT [("p", "u", 'a', 2)])]) == UNPINNED_READ_WHY "a" "p" "u"
#guard refusal (heldNoReads #[] #[] #[readerT "a" (regT [("p", "u", 'b', 3)])]) == ""

/-- Two registered inputs `w0` (node 0) and `w1` (node 1) of four leaves each, and template `A`'s batch of four members
(node 2) mapping its parameter `w` (two leaves) over axis 0 of an argument whose first part is `w0`'s leaves and second
`w1`'s: members 0 and 1 read `w0`, members 2 and 3 `w1`. With `mixed`, node 1 is a run input instead. -/
private def batchToy (mixed : Bool := false) : Flock.Program.Prog :=
  let v16 : Flock.Program.Ty := .value 16
  let run : Flock.Program.Defn :=
    { key := "Input16_v1", id := "Input16_v1", prim := true, params := #[], ret := v16, gates := 1, paramLeaves := 0,
      outLeaves := 1, binding := some Flock.Program.RUN }
  let input : Flock.Program.Defn :=
    { key := "RegisteredInput16_v1", id := "RegisteredInput16_v1", prim := true, params := #[], ret := v16, gates := 1,
      paramLeaves := 0, outLeaves := 1, binding := some Flock.Program.REGISTRATION }
  let a : Flock.Program.Defn :=
    { key := "A", id := "A", prim := true, params := #[("w", .array 2 v16)], ret := v16, gates := 1, paramLeaves := 2,
      outLeaves := 1 }
  let w : Flock.Program.Node := { form := .batch, fn := 0, args := #[], out := .array 4 v16, n := 4 }
  let reader : Flock.Program.Node :=
    { form := .batch, fn := 1, out := .array 4 v16, n := 4, axes := #[some 0],
      args := #[{ ty := .array 4 (.array 2 v16), dims := some (4, 2),
                  seq := Flock.Program.Seq.ofParts #[.affine false 0 0 4, .affine false 1 0 4] }] }
  let root : Flock.Program.Defn :=
    { key := "R", id := "R", prim := false, params := #[], ret := v16, gates := 0, paramLeaves := 0, outLeaves := 1,
      body := { params := #[], retType := v16, nodes := #[w, if mixed then { w with fn := 3 } else w, reader],
                ret := default, offsets := #[0, 4, 8, 12] } }
  { defs := #[input, a, root, run], index := {}, root := 2, programFn := "", sha512 := .empty }

private def batchHeldT : Array (String × Nat × Nat) := #[("w0", 0, 4), ("w1", 4, 4)]

private def batchRangeT : Flock.TemplateQuery.TRange := ⟨2, "A", 0, 4⟩

/-- `w0` and `w1` at root `a…`, 2 leaves, domain `d…`. -/
private def batchPinsT : Array (Spot.Registered.Entry × String) :=
  #[({ pinEntryT with value := "w0" }, "".pushn 'a' 128), ({ pinEntryT with value := "w1" }, "".pushn 'a' 128)]

/-- `pinsHeld` over `batchToy` with `w0` and `w1` marked, R8's pins `pins` (by default `batchPinsT`) and the
verifier's own held roots `roots`. -/
private def batchHeld (ms : Array Member) (pins : Array (Spot.Registered.Entry × String) := batchPinsT)
    (roots : Array (Spot.Registered.Entry × String) := #[]) (loose : Array String := #["w0", "w1"]) : V Unit :=
  pinsHeld (batchToy) #[batchRangeT] batchHeldT ⟨loose, batchHeldT, #[batchRangeT], pins, roots⟩ ms

/-- The chunk `name` over the units `us` of `batchToy`'s range, its registered file `reg`. -/
private def chunkT (name : String) (lo hi : Nat) (reg : Option Json) (us : Array (Nat × Nat) := #[(lo, hi)]) : Member :=
  { memberT name us with own := { sharedT with registered := reg } }

-- every chunk reads the one node of its own units, whichever of them is the first
#guard (memberInputs (batchToy) #[batchRangeT] (chunkT "A/c0" 0 2 none)).toOption == some #[("w", 0, 2)]
#guard (memberInputs (batchToy) #[batchRangeT] (chunkT "A/c1" 2 4 none)).toOption == some #[("w", 1, 2)]
-- a chunk whose units read two input nodes at one parameter: refused by name, in one proof block's range or across two
#guard whyT (memberInputs (batchToy) #[batchRangeT] (chunkT "A/c" 1 3 none)) ==
  s!"member A/c: {Flock.OneCircuit.READ_FORM_WHY 2 "w" "its members read root nodes [0, 1]"}"
#guard whyT (memberInputs (batchToy) #[batchRangeT] (chunkT "A/c" 0 0 none #[(0, 1), (2, 3)])) == TWO_READS_WHY "A/c" "w" 0 1
-- a range at a root node C does not have: refused by name, not skipped
#guard whyT (memberInputs (batchToy) #[{ batchRangeT with node := 9 }] (chunkT "A/c0" 0 2 none)) == RANGE_NODE_WHY "A/c0" 9
-- a parameter mixing the marked weight with a run input: refused by name
#guard whyT (memberInputs (batchToy true) #[batchRangeT] (chunkT "A/c" 1 3 none)) ==
  s!"member A/c: {Flock.OneCircuit.READ_FORM_WHY 2 "w" "its members read root nodes [0, 1]"}"
-- every honest chunk's ports at the pin of the node its units read: accepted
#guard refusal (batchHeld #[chunkT "A/c0" 0 2 (regT [("w", "w0", 'a', 2)]),
  chunkT "A/c1" 2 4 (regT [("w", "w1", 'a', 2)])]) == ""
-- its own root for its value: refused by name
#guard whyT (batchHeld #[chunkT "A/c0" 0 2 (regT [("w", "w0", 'b', 2)])]) == PINNED_READ_WHY "A/c0" "w" "w0" "root"
-- any port of the parameter, whatever value it names, a tree's included, at another root than the node's pin: refused
-- by name; no port of a registered node is exempt as a tree port
#guard whyT (batchHeld #[chunkT "A/c0" 0 2 (regT [("w[0]", "w0", 'a', 2), ("w[1]", "w1", 'b', 2)])]) ==
  PINNED_READ_WHY "A/c0" "w[1]" "w0" "root"
#guard whyT (batchHeld #[chunkT "A/c0" 0 2 (regT [("w[0]", "w0", 'a', 2), ("w[1]", "e", 'b', 2)])]) ==
  PINNED_READ_WHY "A/c0" "w[1]" "w0" "root"
#guard whyT (batchHeld #[chunkT "A/c0" 0 2 (regT [("w[0]", "w0", 'a', 2), ("w[1]", "committed/A/w", 'b', 2)])]) ==
  PINNED_READ_WHY "A/c0" "w[1]" "w0" "root"
#guard whyT (batchHeld #[chunkT "A/c0" 0 2 (regT [("w[0]", "w0", 'a', 2), ("w[1]", "input/n1", 'b', 2)])]) ==
  PINNED_READ_WHY "A/c0" "w[1]" "w0" "root"
#guard whyT (batchHeld #[chunkT "A/c1" 2 4 (regT [("w", "committed/A/w", 'a', 3)])]) ==
  PINNED_READ_WHY "A/c1" "w" "w1" "leaf count"
-- a marked input the verifier expects by name, with no pin R8 accepted and no root it holds itself: refused by name,
-- whatever root the record names; at a root the verifier holds, accepted there and refused at another
#guard whyT (batchHeld #[chunkT "A/c0" 0 2 (regT [("w", "w0", 'a', 2)])] #[] #[] #[]) ==
  UNPINNED_READ_WHY "A/c0" "w" "w0"
#guard refusal (batchHeld #[chunkT "A/c0" 0 2 (regT [("w", "w0", 'a', 2)])] #[] batchPinsT #[]) == ""
#guard whyT (batchHeld #[chunkT "A/c0" 0 2 (regT [("w", "w0", 'b', 2)])] #[] batchPinsT #[]) ==
  PINNED_READ_WHY "A/c0" "w" "w0" "root"

/-- The window open root of these guards. -/
private def rootT : String := "".pushn 'f' 128

/-- The domain of `batchToy true`'s run input node 1 read in rows of 2 leaves, under `rootT`. -/
private def runDomainT : String :=
  ((Spot.Registered.domainAt "" "input/n1" 2 2 16 (some rootT)).toOption.map Flock.hexOf).getD ""

/-- A record holding `input/n1` at root `e…`, `leaves` leaves and domain `d`. -/
private def runRecT (leaves : Nat) (d : String) : Json :=
  Json.mkObj [("roots", Json.arr #[Json.mkObj [("domain", Json.str d), ("leaves", Lean.toJson leaves),
    ("port", Json.str "input/n1"), ("root", Json.str ("".pushn 'e' 128))]])]

/-- `runInputRootsAt` over `batchToy true`'s range at program digest `""`, window open root `R`, the member `m` holding
the roots `own`, and the tied gates `tied`. -/
private def runRootsT (R : Option String) (rec : Json) (m : Member) (own : Array Json := #[])
    (tied : Array (Nat × Nat) := #[]) : V (Array Json) :=
  runInputRootsAt (batchToy true) #[batchRangeT] "" R tied m own (rootsByPort rec)

-- a reader of the run input node opens the record's one tree, at C's leaves and the domain under the open root
#guard (runRootsT (some rootT) (runRecT 2 runDomainT) (chunkT "A/c1" 2 4 none)).toOption.map (·.map (getD · "tree")) ==
  some #[Json.str "input/n1"]
-- a reader of the registered node alone, or of a tied parameter's run input, is not read here
#guard (runRootsT (some rootT) (runRecT 2 runDomainT) (chunkT "A/c0" 0 2 none)).toOption.map (·.size) == some 0
#guard (runRootsT (some rootT) (Json.mkObj []) (chunkT "A/c1" 2 4 none) (tied := #[(4, 4)])).toOption.map (·.size) ==
  some 0
-- every reader of a tied node opens one root: a second root for it is refused by name, the same root is not
#guard (tiedRootsAt (batchToy true) #[batchRangeT] #[(4, 4)] (chunkT "A/c1" 2 4 none)
  #[Json.mkObj [("port", Json.str "w"), ("root", Json.str "a")]] #[]).toOption == some #[(1, Json.str "a")]
#guard (tiedRootsAt (batchToy true) #[batchRangeT] #[(4, 4)] (chunkT "A/c1" 2 4 none)
  #[Json.mkObj [("port", Json.str "w"), ("root", Json.str "a")]] #[(1, Json.str "a")]).toOption == some #[(1, Json.str "a")]
#guard whyT (tiedRootsAt (batchToy true) #[batchRangeT] #[(4, 4)] (chunkT "A/c1" 2 4 none)
  #[Json.mkObj [("port", Json.str "w"), ("root", Json.str "b")]] #[(1, Json.str "a")]) == TIED_ROOTS_WHY 1
#guard (tiedRootsAt (batchToy true) #[batchRangeT] #[] (chunkT "A/c1" 2 4 none)
  #[Json.mkObj [("port", Json.str "w"), ("root", Json.str "b")]] #[(1, Json.str "a")]).toOption == some #[(1, Json.str "a")]
-- no tree in the record, a root of the member's own, another leaf count, another domain, no open root: refused by name
#guard whyT (runRootsT (some rootT) (Json.mkObj [("roots", Json.arr #[])]) (chunkT "A/c1" 2 4 none)) ==
  RUN_TREE_MISSING_WHY "A/c1" "w" 1
#guard whyT (runRootsT (some rootT) (runRecT 2 runDomainT) (chunkT "A/c1" 2 4 none)
  #[Json.mkObj [("port", Json.str "w")]]) == RUN_OWN_ROOT_WHY "A/c1" "w" 1
#guard whyT (runRootsT (some rootT) (runRecT 3 runDomainT) (chunkT "A/c1" 2 4 none)) == RUN_TREE_WHY 1 "leaf count"
#guard whyT (runRootsT (some rootT) (runRecT 2 ("".pushn 'd' 128)) (chunkT "A/c1" 2 4 none)) == RUN_TREE_WHY 1 "domain"
#guard whyT (runRootsT (some ("".pushn '0' 128)) (runRecT 2 runDomainT) (chunkT "A/c1" 2 4 none)) ==
  RUN_TREE_WHY 1 "domain"
#guard whyT (runRootsT none (runRecT 2 runDomainT) (chunkT "A/c1" 2 4 none)) == RUN_TREE_UNBOUND_WHY 1
-- every proof block the member holds, not the first: a chunk whose first unit reads the registered node is read at its
-- later range too, which reads the run input node: refused by name
#guard whyT (runRootsT (some rootT) (Json.mkObj [("roots", Json.arr #[])]) (chunkT "A/c" 0 0 none #[(0, 1), (3, 4)])) ==
  TWO_READS_WHY "A/c" "w" 0 1
-- the refusals' text, which `members` prints word for word
#guard TIED_ROOTS_WHY 3 == "members: C's tied run input node 3 is read under two roots: every reader of a tied node " ++
  "opens one tree"
#guard TWO_READS_WHY "a" "p" 1 2 == "registered reads: member a's units read parameter p at C's input nodes 1 and 2: a " ++
  "parameter that reads an input reads one node, each port one tree"
#guard RUN_TREE_MISSING_WHY "a" "p" 3 == "member a: port p reads C's run input node 3, and the record holds no root of " ++
  "input/n3: a run input node's rows are committed once, in its one tree, before any coin"
#guard RUN_TREE_WHY 3 "domain" == "registered: the record's input/n3 has another domain than C's run input node 3 and " ++
  "the window's open root give it: a run input node's one tree binds its window"
#guard RUN_TREE_UNBOUND_WHY 3 == "registered: C's run input node 3 is read from its one tree input/n3, and the verifier " ++
  "names no window open root (window_root) for its domain to bind"
#guard Flock.OneCircuit.READ_FORM_WHY 2 "p" "x" == "registered reads: parameter p of root node 2: x, and its argument " ++
  "reads an input node: a parameter that reads an input reads one node, each row in one lane's run"
#guard Docket.WEIGHTS_QUERY_WHY == "registered: the program marks registered inputs the verifier does not expect, and " ++
  "the partition is no template query that applies to the program, whose ranges its members' reads of them are read " ++
  "off: a marked input is read only at a pin R8 accepted"
-- the refusals' text, which `members.unpinned_read_why` and `docket.WEIGHTS_UNREGISTERED_WHY` print word for word
#guard UNPINNED_READ_WHY "a" "p" "u" == "registered: member a's port p reads the registered input u, which the program " ++
  "marks, R8 accepted no pin of and the verifier holds no root of: a marked input is read only at a pin R8 accepted " ++
  "or a root the verifier holds"
#guard RANGE_NODE_WHY "a" 9 == "member a: its units meet a range the partition gives at C's root node 9, which C " ++
  "does not have: a range is of C's root nodes"
#guard Docket.WEIGHTS_UNREGISTERED_WHY == "registered: the program marks registered inputs the verifier does not expect, and " ++
  "the parameters carry no weights' registration of record R8 accepts: a marked input is read only at a pin R8 accepted"
-- the refusal's text, which `members.pinned_read_why` prints word for word
#guard PINNED_READ_WHY "a" "p" "w" "root" == "registered: member a's port p reads the registered input w at another " ++
  "root than its pin of record: a registered input the weights' registration of record pins is read at the pin's " ++
  "root, leaf count and domain"

end Spot.Test.OneTree
