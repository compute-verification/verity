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
(`atPins`): PINS (`Pins.ofJson`) is the auditor's registered device keys and designated sessions, and an acceptance
whose components name a device key or a session the pins don't hold is refused.
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

/-- A network certifier's entry as the auditor pins it: the link it is pinned for, its LMS key, its session and the
start of the stretch of the link's link-windows it signs, as a period's parameters list an entry
(`NetworkCheck.Period.Entry`). A link's pinned entries, in their order, are the list its periods list. -/
structure CertifierPin where
  link : String
  key : Spot.Challenger.ChallengerKey
  session : ByteArray
  start : Nat

/-- A memory challenger's pin: its LMS public key and the servers whose memory it is pinned to challenge. -/
structure MemoryPin where
  key : ByteArray
  servers : List ByteArray

/-- **The auditor's pins**, as the composite reads them: its challenger's key (none: no challenger is pinned) and the
sessions it designated for the challenger, sampled proofs' window roots and PoUW's session ids, each as the bytes of
its text; its memory challengers and the sessions it designated for them; and its network certifiers' entries. -/
structure Pins where
  challenger : Option Spot.Challenger.ChallengerKey := none
  sessions : List ByteArray := []
  memory : List MemoryPin := []
  memorySessions : List ByteArray := []
  certifiers : List CertifierPin := []

/-- `j` is an object with exactly the keys `ks`, in their order. -/
def pinKeys (j : Json) (ks : Array String) (what : String) : V Unit := do
  let .obj kvs := j | throw s!"{what} is not an object"
  unless kvs.toArray.map (·.1) == ks do throw s!"{what} holds exactly {", ".intercalate ks.toList}"

/-- The list of hex strings at `k` of `j`. -/
def pinHexes (j : Json) (k ctx : String) : V (List ByteArray) := do
  (← Flock.J.arr j k ctx).toList.mapM fun s => do Flock.unhex (← Flock.J.asStr s s!"{ctx}.{k}")

/-- A network certifier's pinned entry,
`{"H": h, "N": n, "W": w, "link": name, "public": hex, "session": hex, "start": a}`. -/
def certifierOfJson (j : Json) : V CertifierPin := do
  let ctx := "a certifier's entry"
  pinKeys j #["H", "N", "W", "link", "public", "session", "start"] ctx
  pure ⟨← Flock.J.str j "link" ctx,
    ⟨⟨← Flock.J.nat j "N" ctx, ← Flock.J.nat j "W" ctx, ← Flock.J.nat j "H" ctx⟩,
      ← Flock.unhex (← Flock.J.str j "public" ctx)⟩,
    ← Flock.unhex (← Flock.J.str j "session" ctx), ← Flock.J.nat j "start" ctx⟩

/-- A memory challenger's pin, `{"public": hex, "servers": [hex, ...]}`. -/
def memoryOfJson (j : Json) : V MemoryPin := do
  pinKeys j #["public", "servers"] "a memory challenger's pin"
  pure ⟨← Flock.unhex (← Flock.J.str j "public" "a memory challenger's pin"),
    ← pinHexes j "servers" "a memory challenger's pin"⟩

/-- The pins file: `{"certifiers": [CERTIFIER, ...], "challenger": KEY | null, "memory": [MEMORY, ...],
"memory_sessions": [hex, ...], "sessions": [SESSION, ...]}`, `KEY` as the sampled parameters carry a challenger's key
(`ChallengerKey.ofJson`), `SESSION` a sampled window root or a PoUW session id, `CERTIFIER` as `certifierOfJson` and
`MEMORY` as `memoryOfJson` read them. -/
def Pins.ofJson (j : Json) : V Pins := do
  pinKeys j #["certifiers", "challenger", "memory", "memory_sessions", "sessions"] "the pins"
  let challenger ← match ← Flock.J.get j "challenger" "the pins" with
    | .null => pure none
    | k => some <$> Spot.Challenger.ChallengerKey.ofJson k
  let sessions ← (← Flock.J.arr j "sessions" "the pins").toList.mapM fun s => do
    pure (← Flock.J.asStr s "the pins' sessions").toUTF8
  pure { challenger, sessions,
         memory := ← (← Flock.J.arr j "memory" "the pins").toList.mapM memoryOfJson,
         memorySessions := ← pinHexes j "memory_sessions" "the pins",
         certifiers := ← (← Flock.J.arr j "certifiers" "the pins").toList.mapM certifierOfJson }

/-- The one-stage parameters of a sampled-proofs component, `n`, or none for any other component or parameters that
don't read (which its own check refuses). -/
def sampledAt (n : Named) : Option Spot.OneStageZk.Params :=
  if n.id = VStar.Recursive.Sampled.PROTOCOL then
    match canonicalJson n.params.toUTF8 >>= VStar.Recursive.Sampled.paramsOfJson with
    | .ok (.inl p) => some p
    | _ => none
  else none

/-- The parameters of a component `n` of `B`'s id, as `B`'s own decoder reads them; none for any other component or
parameters that don't read (which its own check refuses). -/
def paramsAt (B : Protocol) (n : Named) : Option B.Params :=
  if n.id = B.id then (canonicalJson n.params.toUTF8 >>= B.params).toOption else none

/-- A PoUW period component's parameters (`PouwCheck.Period.protocol`'s decoder). -/
def pouwAt (n : Named) : Option PouwCheck.Period.Params := paramsAt PouwCheck.Period.protocol n

/-- A PoUS run component's parameters (`PousCheck.RunProtocol.protocol`'s decoder). -/
def pousAt (n : Named) : Option PousCheck.RunProtocol.Params := paramsAt PousCheck.RunProtocol.protocol n

/-- A wipe component's parameters (`MemoryChallenger.WipeProtocol.protocol`'s decoder). -/
def wipeAt (n : Named) : Option MemoryChallenger.WipeProtocol.Params := paramsAt MemoryChallenger.WipeProtocol.protocol n

/-- A network certifier component's parameters (`NetworkCheck.Period.protocol`'s decoder). -/
def networkAt (n : Named) : Option NetworkCheck.Period.Params := paramsAt NetworkCheck.Period.protocol n

/-- `k` is the LMS key `q`: the same public key under the same parameters. -/
def sameKey (k q : Spot.Challenger.ChallengerKey) : Bool :=
  k.pub.data.toList == q.pub.data.toList && k.params.N == q.params.N && k.params.W == q.params.W &&
    k.params.H == q.params.H

/-- One sampled component's parameters name only what `P` pins: the challenger its coins come from is the pinned
key, under the pinned LMS parameters, and its window root is a designated session's. -/
def spotPinned (P : Pins) (p : Spot.OneStageZk.Params) : Bool :=
  (p.base.challenger.all fun kl => P.challenger.any (sameKey kl.1)) &&
  p.base.verifier.windowRoot.all fun w => P.sessions.any (·.data.toList == w.toUTF8.data.toList)

/-- The bytes `a` are the bytes `b`. -/
def sameBytes (a b : ByteArray) : Bool := a.data.toList == b.data.toList

/-- One of `S` is the bytes `s`. -/
def heldBytes (S : List ByteArray) (s : ByteArray) : Bool := S.any (·.data.toList == s.data.toList)

/-- A PoUW period's challenger key, where it reads as one, is the pinned challenger's, and its parameters list the
sessions its windows' checks are held to (`registered`), each one designated for the challenger. A period run alone
lists none, so the sessions its challenger's ledger opens are held to nothing, and it is refused. -/
def pouwPinned (P : Pins) (p : PouwCheck.Period.Params) : Bool :=
  (match Spot.Challenger.ChallengerKey.ofJson p.key with
    | .ok k => P.challenger.any (sameKey k)
    | .error _ => true) &&
  p.registered.any fun r => r.challengePhases.all fun s => heldBytes P.sessions s.toUTF8

/-- Every key a PoUS run trusts is a pinned memory challenger's, and its session is one designated for the memory
challengers. A run's parameters name no server, so any pinned memory challenger's key passes. -/
def pousPinned (P : Pins) (p : PousCheck.RunProtocol.Params) : Bool :=
  (p.trusted.all fun t => P.memory.any fun m => sameBytes t m.key) && heldBytes P.memorySessions p.sessionId

/-- Every key a wipe trusts is the key of a memory challenger pinned for the wipe's server, and its session is one
designated for the memory challengers. -/
def wipePinned (P : Pins) (p : MemoryChallenger.WipeProtocol.Params) : Bool :=
  (p.trusted.all fun t => P.memory.any fun m => sameBytes t m.key && heldBytes m.servers p.server) &&
    heldBytes P.memorySessions p.session

/-- A period's entry `e`, its key under the period's LMS parameters `lms`, is the pinned entry `c`: the same key,
session and start. -/
def entryHeld (lms : Lms.Params) (c : CertifierPin) (e : NetworkCheck.Period.Entry) : Bool :=
  sameKey ⟨lms, e.key⟩ c.key && sameBytes e.session c.session && e.start == c.start

/-- The period's entries `es` are the pinned entries `cs`, one for one in their order. -/
def entriesHeld (lms : Lms.Params) : List CertifierPin → List NetworkCheck.Period.Entry → Bool
  | [], [] => true
  | c :: cs, e :: es => entryHeld lms c e && entriesHeld lms cs es
  | _, _ => false

/-- A network certifier period's entries, read by its own decoder (`NetworkCheck.Period.certifierEntries`'s), are the
entries pinned for its link, one for one: each window's entry (`NetworkCheck.Period.entryAt`) is then the pinned
entry whose stretch holds it, and a link with no pinned entry is refused, since a period lists at least one. -/
def networkPinned (P : Pins) (p : NetworkCheck.Period.Params) : Bool :=
  entriesHeld p.lms (P.certifiers.filter (·.link == p.link)) p.entries.toList

/-- The composite's parameters `j` name only what `P` pins, in every component that reads a device's key or session:
sampled proofs', PoUW's, PoUS's, the wipe's and the network certifier's. Parameters that don't read as a composite's
pass, since the composite refuses them itself. -/
def paramsPinned (P : Pins) (j : Json) : Bool :=
  match decodeParams j with
  | .ok ps => (ps.filterMap sampledAt).all (spotPinned P) && (ps.filterMap pouwAt).all (pouwPinned P) &&
      (ps.filterMap pousAt).all (pousPinned P) && (ps.filterMap wipeAt).all (wipePinned P) &&
      (ps.filterMap networkAt).all (networkPinned P)
  | .error _ => true

def UNPINNED : String :=
  "a component names a device key or a session the auditor's pins don't hold"

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
private def noPins (rest : String) : String :=
  "{\"certifiers\":[],\"challenger\":null,\"memory\":[],\"memory_sessions\":[],\"sessions\":[" ++ rest ++ "]}"
#guard match pinsOf (noPins "\"ab\"") with
  | .ok P => P.challenger.isNone && P.memory.isEmpty && P.certifiers.isEmpty && P.memorySessions.isEmpty &&
      P.sessions.map (·.data.toList) == ["ab".toUTF8.data.toList]
  | .error _ => false
private def entryPin (link pub session : String) (start : Nat) : String :=
  s!"\{\"H\":3,\"N\":64,\"W\":1,\"link\":\"{link}\",\"public\":\"{pub}\",\"session\":\"{session}\",\"start\":{start}}"
private def somePins : String :=
  "{\"certifiers\":[" ++ entryPin "eth0" "0a" "0c" 4 ++ "," ++ entryPin "eth1" "0f" "0c" 0 ++ "," ++
    entryPin "eth0" "1a" "1c" 9 ++ "],\"challenger\":null,\"memory\":[{\"public\":\"0b\"," ++
    "\"servers\":[\"0d\"]}],\"memory_sessions\":[\"0e\"],\"sessions\":[]}"
#guard match pinsOf somePins with
  | .ok P => P.memory.map (fun m => (m.key.data.toList, m.servers.map (·.data.toList))) == [([0x0b], [[0x0d]])] &&
      P.memorySessions.map (·.data.toList) == [[0x0e]] &&
      P.certifiers.map (fun c => (c.link, c.key.params.N, c.key.params.W, c.key.params.H, c.key.pub.data.toList,
        c.session.data.toList, c.start)) ==
        [("eth0", 64, 1, 3, [0x0a], [0x0c], 4), ("eth1", 64, 1, 3, [0x0f], [0x0c], 0),
          ("eth0", 64, 1, 3, [0x1a], [0x1c], 9)]
  | .error _ => false
-- a period's entries pass where they are its link's pinned entries one for one, in order: not a subset, a reordering,
-- another start, session or key, other LMS parameters, or another link's
#guard match pinsOf somePins with
  | .ok P =>
    let lms : Lms.Params := ⟨64, 1, 3⟩
    let e (k s : UInt8) (a : Nat) : NetworkCheck.Period.Entry := ⟨⟨#[k]⟩, ⟨#[s]⟩, a⟩
    let held (l : Lms.Params) (link : String) (es : List NetworkCheck.Period.Entry) :=
      entriesHeld l (P.certifiers.filter (·.link == link)) es
    held lms "eth0" [e 0x0a 0x0c 4, e 0x1a 0x1c 9] && held lms "eth1" [e 0x0f 0x0c 0] &&
      !held lms "eth0" [e 0x0a 0x0c 4] && !held lms "eth0" [e 0x1a 0x1c 9, e 0x0a 0x0c 4] &&
      !held lms "eth0" [e 0x0a 0x0c 3, e 0x1a 0x1c 9] && !held lms "eth0" [e 0x0a 0x1c 4, e 0x1a 0x1c 9] &&
      !held lms "eth0" [e 0x0f 0x0c 4, e 0x1a 0x1c 9] && !held ⟨64, 1, 2⟩ "eth0" [e 0x0a 0x0c 4, e 0x1a 0x1c 9] &&
      !held lms "eth2" [e 0x0a 0x0c 4] && !held lms "eth1" []
  | .error _ => false
#guard (pinsOf "{\"challenger\":null,\"sessions\":[]}").toOption.isNone
#guard (pinsOf "{\"certifiers\":[],\"challenger\":null,\"memory\":null,\"sessions\":[]}").toOption.isNone
#guard (pinsOf ((noPins "").replace "\"sessions\"" "\"extra\":1,\"sessions\"")).toOption.isNone
#guard (pinsOf ((noPins "").replace "\"certifiers\"" "\"certifier_sessions\":[],\"certifiers\"")).toOption.isNone
#guard (pinsOf ((noPins "").replace "\"challenger\":null" "\"challenger\":{\"H\":3}")).toOption.isNone
#guard (pinsOf (noPins "1")).toOption.isNone
#guard (pinsOf ((noPins "").replace "\"memory_sessions\":[]" "\"memory_sessions\":[1]")).toOption.isNone
#guard (pinsOf (somePins.replace "\"start\":4" "\"start\":4,\"stop\":9")).toOption.isNone
#guard (pinsOf (somePins.replace "\"start\":4" "\"start\":\"4\"")).toOption.isNone
#guard (pinsOf (somePins.replace "\"session\":\"0c\"" "\"session\":\"0\"")).toOption.isNone
#guard (pinsOf (somePins.replace "\"servers\":[\"0d\"]" "\"servers\":\"0d\"")).toOption.isNone
-- a PoUW period passes where its sessions are designated for the challenger; a period run alone lists none, and is
-- refused
#guard match pinsOf (noPins "\"s1\",\"s2\"") with
  | .ok P =>
    let p (r : Option PouwCheck.Period.Registered) : PouwCheck.Period.Params := ⟨.null, .null, "p", r⟩
    let reg (ss : List String) : Option PouwCheck.Period.Registered := some ⟨"h", "d", ss⟩
    pouwPinned P (p (reg ["s1"])) && pouwPinned P (p (reg ["s2", "s1"])) && pouwPinned P (p (reg [])) &&
      !pouwPinned P (p (reg ["s1", "s3"])) && !pouwPinned P (p none)
  | .error _ => false
-- parameters that aren't a composite's are the composite's to refuse
#guard paramsPinned {} Json.null

end Core.Composites
