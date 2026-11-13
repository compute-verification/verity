/-
The composite a configuration's subprotocols make (`verity run CONFIG`): onsite (`Core.Onsite.protocol`,
`verity/onsite/v0`) when they are onsite's, remote (`Core.Remote.V1.protocol`, `verity/remote/v1`) when they are
remote's, listed in any order. These two are the only composites a verifier runs: protocol soundness covers only these
two (`Verity.verifier`), so `ofIds` refuses any other set by name (an id no composite composes, one listed twice, none,
or a set that is no composite's). Remote's development composite, H1's `Core.Remote.protocol` (`verity/remote/v0`), is
no composite here: `ofIds` refuses its subprotocols, and it runs only as its own development executable (`remote`).

    product IDS id                                            IDS: the subprotocols' ids, comma-separated
    product IDS --pins PINS check PARAMS COMMITMENT EVIDENCE [BLOB ...] | serve

The command after `IDS` is the core CLI's (`CoreProtocol.Cli`), of the composite `ofIds IDS` under the auditor's pins
(`atPins`): PINS, `{"challenger": KEY | null, "sessions": [WINDOW_ROOT, ...]}`, is the auditor's registered challenger
key and designated sessions, and an acceptance whose sampled components name another key or window root is refused.
-/
module

public import CoreOnsite
public import CoreProtocol.Cli
public import CoreRemote
meta import CoreOnsite
meta import CoreProtocol.Cli
meta import CoreRemote

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace Core.Composites

open Composition

/-- The composites, each with its subprotocols' ids. -/
def composites : List (BlobProtocol × List String) :=
  [(Core.Onsite.protocol, Core.Onsite.components.map (·.id)),
    (Core.Remote.V1.protocol, Core.Remote.V1.components.map (·.id))]

/-- Every subprotocol of some composite, by id. -/
def known : List String := (composites.flatMap (·.2)).eraseDups

/-- The composite over exactly the subprotocols `ids` names, in any order, or why the list is refused. -/
def ofIds (ids : List String) : Except String BlobProtocol :=
  match ids.find? (fun i => !known.contains i) with
  | some i => .error s!"{i} is no composite's subprotocol ({", ".intercalate known})"
  | none =>
    match ids.find? (fun i => ids.count i > 1) with
    | some i => .error s!"{i} is listed twice"
    | none =>
      match composites.find? (fun (_, xs) => xs.length == ids.length && xs.all ids.contains) with
      | some (P, _) => .ok P
      | none =>
        let named := composites.map fun (P, xs) => s!"{P.id}: {", ".intercalate xs}"
        .error (s!"[{", ".intercalate ids}] is no composite's subprotocols ({"; ".intercalate named})" ++
          ": protocol soundness covers only those composites")

open Lean (Json)
open Flock (V)

/-- **The auditor's pins**, as the composite reads them: the auditor's challenger's key (none: no challenger is
pinned) and the window roots of the sessions the auditor designated, each as the bytes of its text. -/
structure Pins where
  challenger : Option Spot.Challenger.ChallengerKey := none
  sessions : List ByteArray := []

/-- The pins file: `{"challenger": KEY | null, "sessions": [WINDOW_ROOT, ...]}`, `KEY` as the sampled parameters
carry a challenger's key (`ChallengerKey.ofJson`). -/
def Pins.ofJson (j : Json) : V Pins := do
  let .obj kvs := j | throw "the pins are not an object"
  unless kvs.toArray.map (·.1) == #["challenger", "sessions"] do
    throw "the pins hold exactly challenger and sessions"
  let challenger ← match ← Flock.J.get j "challenger" "the pins" with
    | .null => pure none
    | k => some <$> Spot.Challenger.ChallengerKey.ofJson k
  let sessions ← (← Flock.J.arr j "sessions" "the pins").toList.mapM fun s => do
    pure (← Flock.J.asStr s "the pins' sessions").toUTF8
  pure { challenger, sessions }

/-- The one-stage parameters of a sampled-proofs component, `n`, or none for any other component or parameters that
don't read (which its own check refuses). -/
def sampledAt (n : Named) : Option Spot.OneStageZk.Params :=
  if n.id = VStar.Recursive.Sampled.PROTOCOL then
    match canonicalJson n.params.toUTF8 >>= VStar.Recursive.Sampled.paramsOfJson with
    | .ok (.inl p) => some p
    | _ => none
  else none

/-- One sampled component's parameters name only what `P` pins: the challenger its coins come from is the pinned
key, under the pinned LMS parameters, and its window root is a designated session's. -/
def spotPinned (P : Pins) (p : Spot.OneStageZk.Params) : Bool :=
  (p.base.challenger.all fun kl => P.challenger.any fun k =>
    kl.1.pub.data.toList == k.pub.data.toList && kl.1.params.N == k.params.N && kl.1.params.W == k.params.W &&
      kl.1.params.H == k.params.H) &&
  p.base.verifier.windowRoot.all fun w => P.sessions.any (·.data.toList == w.toUTF8.data.toList)

/-- The composite's parameters `j` name only what `P` pins, in every sampled component; parameters that don't read
as a composite's pass, since the composite refuses them itself. -/
def paramsPinned (P : Pins) (j : Json) : Bool :=
  match decodeParams j with
  | .ok ps => (ps.filterMap sampledAt).all (spotPinned P)
  | .error _ => true

def UNPINNED : String :=
  "a sampled component's challenger key or window root is not the auditor's pinned key or a designated session"

/-- `o`, except an acceptance under parameters `j` that name what `P` doesn't pin, which is refused. -/
def holdPins {α β : Type} (P : Pins) (j : Json) : Outcome α β → Outcome α β
  | .accept pub prof => if paramsPinned P j then .accept pub prof else .reject UNPINNED
  | o => o

/-- **`B` under the auditor's pins**: `B`'s check, whose acceptance stands only when the parameters name only what
`P` pins (`holdPins`); every other outcome, and every refusal before the check, is `B`'s. -/
def atPins (P : Pins) (B : BlobProtocol) : BlobProtocol where
  id := B.id
  Params := Json × B.Params
  Commitment := B.Commitment
  Evidence := B.Evidence
  Public := B.Public
  Profile := B.Profile
  params j := (j, ·) <$> B.params j
  commitment := B.commitment
  evidence := B.evidence
  digest := B.digest
  check p c e := holdPins P p.1 (B.check p.2 c e)
  publicJson := B.publicJson
  profileJson := B.profileJson
  checkBlobs p c e bs := holdPins P p.1 (B.checkBlobs p.2 c e bs)

private def USAGE : String :=
  "usage: product IDS id | product IDS --pins PINS check PARAMS COMMITMENT EVIDENCE [BLOB ...] | " ++
    "product IDS --pins PINS serve"

/-- `product IDS ...`: the core CLI of `ofIds IDS` under the pins file `PINS` (`atPins`); `id` reads no pins. 64 when
`IDS` or `PINS` is refused or missing. -/
def main (args : List String) : IO UInt32 := do
  match args with
  | ids :: rest =>
    match ofIds ((ids.splitOn ",").filter (· ≠ "")) with
    | .error e => IO.eprintln s!"product: {e}"; return 64
    | .ok P =>
      match rest with
      | ["id"] => P.main rest
      | "--pins" :: file :: rest =>
        match canonicalJson (← IO.FS.readBinFile file) >>= Pins.ofJson with
        | .ok pins => (atPins pins P).main rest
        | .error e => IO.eprintln s!"product: the pins {file}: {e}"; return 64
      | _ => IO.eprintln s!"product: the composite checks only under the auditor's pins\n{USAGE}"; return 64
  | [] => IO.eprintln USAGE; return 64

private def idOf (ids : List String) : Option String := (ofIds ids).toOption.map (·.id)

private def contains (s pat : String) : Bool := (s.splitOn pat).length > 1

private def onsiteIds : List String := Core.Onsite.components.map (·.id)
private def remoteIds : List String := Core.Remote.components.map (·.id)
private def remoteV1Ids : List String := Core.Remote.V1.components.map (·.id)

#guard composites.map (·.1.id) == ["verity/onsite/v0", "verity/remote/v1"]
#guard onsiteIds.all known.contains && remoteV1Ids.all known.contains
#guard known.length == (onsiteIds ++ remoteV1Ids).eraseDups.length

-- the composites by their subprotocols, in any order, and no other set
#guard idOf onsiteIds == some "verity/onsite/v0"
#guard idOf onsiteIds.reverse == some "verity/onsite/v0"
-- H1's development composite is none: its subprotocols are refused, by name
#guard idOf remoteIds == none
#guard idOf remoteIds.reverse == none
#guard contains (match ofIds remoteIds with | .error e => e | .ok _ => "")
  "is no composite's subprotocols (verity/onsite/v0: "
#guard idOf remoteV1Ids == some "verity/remote/v1"
#guard idOf remoteV1Ids.reverse == some "verity/remote/v1"
#guard idOf (remoteV1Ids ++ ["verity/pouw/period/v1"]) == none
#guard idOf [] == none
#guard idOf ["verity/sampled-proofs/one-stage/v1"] == none
#guard idOf ["verity/sampled-proofs/one-stage/v2"] == none
#guard idOf ["verity/sampled-proofs/v1", "verity/certifier/v0"] == none
#guard idOf (onsiteIds.drop 1) == none
#guard idOf ["verity/wipe/v0"] == none
#guard idOf ["verity/pouw/v1", "verity/pouw/v1", "verity/sampled-proofs/v1"] == none
#guard idOf (remoteIds ++ ["verity/pouw/period/v1"]) == none
#guard idOf (onsiteIds ++ ["verity/pouw/v1"]) == none
#guard contains (match ofIds ["verity/sampled-proofs/v1"] with | .error e => e | .ok _ => "")
  "protocol soundness covers only those composites"
#guard contains (match ofIds ["verity/sampled-proofs/one-stage/v1"] with | .error e => e | .ok _ => "")
  "is no composite's subprotocol"
#guard contains (match ofIds ["verity/sampled-proofs/one-stage/v2"] with | .error e => e | .ok _ => "")
  "is no composite's subprotocol"

private def pinsOf (s : String) : Except String Pins := Json.parse s >>= Pins.ofJson

-- the pins file: no challenger and no sessions pins nothing; any other key, or a key that doesn't read, is refused
#guard match pinsOf "{\"challenger\":null,\"sessions\":[\"ab\"]}" with
  | .ok P => P.challenger.isNone && P.sessions.map (·.data.toList) == ["ab".toUTF8.data.toList] | .error _ => false
#guard (pinsOf "{\"challenger\":null}").toOption.isNone
#guard (pinsOf "{\"challenger\":null,\"sessions\":[],\"extra\":1}").toOption.isNone
#guard (pinsOf "{\"challenger\":{\"H\":3},\"sessions\":[]}").toOption.isNone
#guard (pinsOf "{\"challenger\":null,\"sessions\":[1]}").toOption.isNone
-- parameters that aren't a composite's are the composite's to refuse
#guard paramsPinned {} Json.null

end Core.Composites
