module

public import Spot.OneStageZk
public import Flock.Test.WiredPorts
meta import Spot.OneStageZk
meta import Flock.Test.WiredPorts

/-!
A member's header held to its slice (`OneStageZk.headerHeldAt`), the difftest's `header_held_at` piece, and a
member's ports at C's run input nodes' trees (`OneStageZk.runInputRootsAt`) on `Flock.Test.WiredPorts`' programs: on
`runP`, unit 0 reads its parameter `w` from run input node 0, so its slice holds `w` at the record's root of
`input/n0`, with that tree; on `regP`, where node 0 is a registered input, it holds none; a root of the member's own at
`w` is refused by name where the record holds `input/n0`, and kept where it does not. At that slice, a header whose root at `w` is not the registration's is held only by a read of
`w` at `input/n0`: no read, a read at another run input's tree and a read at the member's own committed tree each
refuse by `SHARED_UNREAD`.
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

def n0 : Json := Json.mkObj [("domain", "dee"), ("leaves", (4 : Nat)), ("port", "input/n0"), ("root", "ee")]
def runRec : Json := Json.mkObj [("roots", Json.arr #[n0])]
def runRs : Array Flock.TemplateQuery.TRange := #[⟨1, "T", 0, 2⟩, ⟨2, "T", 2, 1⟩]
def runAt (P : Flock.Program.Prog) (own : Array Json) (rec : Json := runRec) : Except String (Array Json) :=
  OneStageZk.runInputRootsAt P runRs "T/c0" 0 own (OneStageZk.rootsByPort rec)
def wRun : Json :=
  Json.mkObj [("domain", "dee"), ("leaves", (4 : Nat)), ("port", "w"), ("root", "ee"), ("tree", "input/n0")]

#guard (runAt runP #[]).toOption == some #[wRun]
#guard (runAt regP #[]).toOption == some #[]
#guard (runAt runP #[] (Json.mkObj [("roots", Json.arr #[])])).toOption == some #[]
#guard (runAt runP #[Json.mkObj [("port", "w"), ("root", "aa")]] (Json.mkObj [("roots", Json.arr #[])])).toOption ==
  some #[Json.mkObj [("port", "w"), ("root", "aa")]]
#guard match runAt runP #[Json.mkObj [("port", "w"), ("root", "aa")]] with
  | .error e => e == OneStageZk.RUN_OWN_ROOT_WHY "T/c0" "w" 0
  | .ok _ => false
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
#guard OneStageZk.SHARED_UNREAD "w" "input/n0" == "public file: port w is held in the shared tree input/n0, and the " ++
  "member's registered reads open no row of it there: a member's rows of a tree other members' rows share are held " ++
  "to the registration's root by opening it at the leaves C gives, and the header's own root there is no commitment " ++
  "to them"

end Spot.HeaderHeldAt
