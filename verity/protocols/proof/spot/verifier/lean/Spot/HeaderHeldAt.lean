module

public import Spot.OneStageZk
public import Flock.Test.WiredPorts
meta import Spot.OneStageZk
meta import Flock.Test.WiredPorts

/-!
A member's header held to its slice (`OneStageZk.headerHeldAt`), the difftest's `header_held_at` piece, and a
member's ports at C's run input nodes' trees (`OneStageZk.runInputRootsAt`) on `Flock.Test.WiredPorts`' programs: on
`runP`, unit 0 reads its parameter `w` from run input node 0, so its slice holds `w` at the record's root of
`input/n0`, with that tree, at the leaf count and the domain C and the window's open root give it; on `regP`, where
node 0 is a registered input, it holds none; a record holding no `input/n0`, a root of the member's own at `w`, another
domain or leaf count, and no window open root are each refused by name. At that slice, a header naming `w`, at the registration's
root or another, is held only by a read of `w` at `input/n0`: no read, a read at another run input's tree and a read
at the member's own committed tree each refuse by `SHARED_UNREAD`.
-/

@[expose] public section

namespace Spot.HeaderHeldAt

open Lean (Json)

/-- `OneStageZk.headerHeldAt` on `{reads, registered, slice, header}`, `reads` a list of `[port, value]` and
`registered` the verifier's registered file or null (no ports): the empty string where it holds, else its refusal. -/
def piece (j : Json) : Except String Json := do
  let .arr rs := getD j "reads" | throw "reads is not a list"
  let reads ← rs.mapM fun r => match r with
    | .arr #[.str p, .str v] => pure (p, v)
    | _ => throw "a read is not [port, value]"
  let held ← match getD j "registered" with
    | .null => pure #[]
    | f => Flock.Registered.ofJson f
  match OneStageZk.headerHeldAt reads held (getD j "slice") (getD j "header") with
  | .ok () => pure (Json.str "")
  | .error e => pure (Json.str e)

open Flock.Test.WiredPorts (runP regP)

def runR : String := "".pushn 'e' 128
-- node 0 of `runP`: 4 members of 2 leaves, read by T's units a row of 2 leaves each
def runDom : Json := match Spot.Registered.domainAt "pp" "input/n0" 2 4 16 (some runR) with
  | .ok d => .str (Flock.hexOf d)
  | .error _ => .null
def n0 : Json := Json.mkObj [("domain", runDom), ("leaves", (4 : Nat)), ("port", "input/n0"), ("root", "ee")]
def runRec : Json := Json.mkObj [("roots", Json.arr #[n0])]
def runRs : Array Flock.TemplateQuery.TRange := #[⟨1, "T", 0, 2⟩, ⟨2, "T", 2, 1⟩]
def runM : OneStageZk.Member :=
  { name := "T/c0", units := #[(0, 2)], g := 1,
    own := { programJ := Json.null, program := .empty, partition := Json.null, pub := "", registered := none } }
def runAt (P : Flock.Program.Prog) (own : Array Json) (rec : Json := runRec) (R : Option String := some runR) :
    Except String (Array Json) :=
  OneStageZk.runInputRootsAt P runRs "pp" R #[] runM own (OneStageZk.rootsByPort rec)
def wRun : Json :=
  Json.mkObj [("domain", runDom), ("leaves", (4 : Nat)), ("port", "w"), ("root", "ee"), ("tree", "input/n0")]
def runWhy (r : Except String (Array Json)) : String := match r with
  | .error e => e
  | .ok _ => ""

#guard runDom != .null
#guard (runAt runP #[]).toOption == some #[wRun]
#guard (runAt regP #[]).toOption == some #[]
#guard runWhy (runAt runP #[] (Json.mkObj [("roots", Json.arr #[])])) == OneStageZk.RUN_TREE_MISSING_WHY "T/c0" "w" 0
#guard runWhy (runAt runP #[Json.mkObj [("port", "w"), ("root", "aa")]]) == OneStageZk.RUN_OWN_ROOT_WHY "T/c0" "w" 0
#guard runWhy (runAt runP #[] (Json.mkObj [("roots", Json.arr #[n0.setObjVal! "domain" "dee"])])) ==
  OneStageZk.RUN_TREE_WHY 0 "domain"
#guard runWhy (runAt runP #[] (Json.mkObj [("roots", Json.arr #[n0.setObjVal! "leaves" (3 : Nat)])])) ==
  OneStageZk.RUN_TREE_WHY 0 "leaf count"
#guard runWhy (runAt runP #[] runRec none) == OneStageZk.RUN_TREE_UNBOUND_WHY 0
#guard OneStageZk.RUN_OWN_ROOT_WHY "T/c0" "w" 0 == "member T/c0: port w reads C's run input node 0, and the " ++
  "registration holds a root of its own for it: a run input node's rows are committed once, in input/n0"

-- the door's `OneCircuit.pos` takes the port at exactly the tree `runInputRootsAt` derives: both read the unit's root
-- node of C and its row there (`OneCircuit.regRow`)
#guard (runAt runP #[]).toOption.map (·.map (getD · "tree")) == some #[Json.str (Flock.Registered.freshValue 0)]
#guard (Flock.OneCircuit.pos runP #[0, 1] Flock.Test.WiredPorts.fresh).toOption.isSome
#guard (Flock.OneCircuit.pos runP #[0, 1]
  { Flock.Test.WiredPorts.fresh with value := Flock.Registered.freshValue 1 }).toOption.isNone

def runSlice : Json := Json.mkObj [("population", (2 : Nat)), ("roots", Json.arr #[wRun]), ("scheme", "s")]
def runHeader : Json := Json.mkObj [("instances", (2 : Nat)), ("scheme", "s"),
  ("frame_v3", Json.mkObj [("roots", Json.mkObj [("w", "77")]), ("domain_ids", Json.mkObj [("w", "d77")])])]
def heldRun (reads : Array (String × String)) : String :=
  match OneStageZk.headerHeldAt reads #[] runSlice runHeader with
  | .ok () => ""
  | .error e => e

#guard heldRun #[("w", "input/n0")] == ""
#guard heldRun #[] == OneStageZk.SHARED_UNREAD "w" "input/n0"
#guard heldRun #[("w", "input/n1")] == OneStageZk.SHARED_UNREAD "w" "input/n0"
#guard heldRun #[("w", "committed/w")] == OneStageZk.SHARED_UNREAD "w" "input/n0"
-- the node's own root and domain with no read: the rows C gives are not the tree's leaves in order, refused too
#guard (match OneStageZk.headerHeldAt #[] #[] runSlice (runHeader.setObjVal! "frame_v3" (Json.mkObj
    [("roots", Json.mkObj [("w", "ee")]), ("domain_ids", Json.mkObj [("w", runDom)])])) with
  | .ok () => ""
  | .error e => e) == OneStageZk.SHARED_UNREAD "w" "input/n0"
#guard OneStageZk.SHARED_UNREAD "w" "input/n0" == "public file: port w is held in the shared tree input/n0, and the " ++
  "member's registered reads open no row of it there: a member's rows of a tree other members' rows share are held " ++
  "to the registration's root by opening it at the leaves C gives, and the header's own root there is no commitment " ++
  "to them"

end Spot.HeaderHeldAt
