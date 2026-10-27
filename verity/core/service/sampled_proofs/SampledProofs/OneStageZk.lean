/-
Sampled proofs' one-stage audit on the proofs themselves (`verity/sampled-proofs/one-stage/v2`), a `Core.BlobProtocol`: the
v1 audit (`OneStage`) with the verdicts computed here, from the statement's files, rather than read from the evidence.

* **params**: v1's, plus `proofs`, exactly `{"circuit", "registered"}`: the SHA-512 the auditor committed for the circuit
  file (lowercase hex of 64 bytes) and the verifier's own registered file (`flock-verify --registered`'s object, or null
  for a program that reads no registered value). The partition must not be null: the sessions are verified against the
  verifier's own partition object;
* **commitment**: v1's (the registration record);
* **evidence**: v1's without `verdicts`, plus `proofs`, exactly `{"public", "sessions"}`: the public file's SHA-512 and,
  for each of v1's `sessions` in order, `{"name", "record", "reps"}`: its name, its `session.json`'s SHA-512 and
  `{file: SHA-512}` naming at least each rep file the statement's session reads (`<stream with / as .>.bin`);
* **blobs**: the files those digests name (`Core.Blobs`), and the MUFU tables any lookup reads, found under their pinned
  SHA-512 as `flock-verify --archive` finds them;
* **check**: `flock-verify verify --zk --statement verity/flock-circuit` with `--circuit`, `--public`, `--partition`,
  `--program` and `--registered` the verifier's own and one `--session` a session, as `one_stage.proofs.verify` runs it
  (`verdicts`, the C-Flock library calls `verifyCmd` makes on that path); then v1's `check` on those verdicts, one of
  record a session (`flock-verify/<name>`). A session accepted under another program than the partition object's, proved
  units other than the evidence's `proved_units`, or a circuit, public or record file no blob holds, rejects; a rep
  file no blob holds is that session's reject, as a file `flock-verify` can't read is.

Without blobs (`toProtocol`) every check rejects: no blob holds the circuit file.
-/
import CoreProtocol.Blobs
import SampledProofs.OneStage
import SampledProofs.ChallengerRounds
import SampledProofs.Hello
import Flock

namespace SampledProofs.OneStageZk

open Lean (Json)
open Flock (V)

def PROTOCOL : String := "verity/sampled-proofs/one-stage/v2"
/-- The statement every drawn unit's session proves (`one_stage.proofs.STATEMENT`). -/
def STATEMENT : String := "verity/flock-circuit"

/-- The verifier's own inputs to `flock-verify verify --zk`: the program's descriptor (as JSON and as the bytes
`--program` reads), the partition object, the circuit file's committed SHA-512 and the registered file. -/
structure Own where
  programJ : Json
  program : ByteArray
  partition : Json
  circuit : String
  registered : Option Json

structure Params where
  base : OneStage.Params
  own : Own

structure Session where
  name : String
  record : String
  reps : Array (String × String)
  /-- the segments of the session's uses of the challenger's coins (`sessionUses`), in order -/
  coins : Array Json := #[]

structure Evidence where
  base : OneStage.Evidence
  pub : String
  sessions : Array Session

private def exactly (j : Json) (ks : Array String) (what : String) : V Unit :=
  if keys? j != some (sortStrings ks) then throw s!"the keys of {what} are not {", ".intercalate (sortStrings ks).toList}"
  else pure ()

private def digest? (s : String) : Bool := s.length == 128 && isHex s

private def without (j : Json) (k : String) : List (String × Json) :=
  match j with
  | .obj kvs => (kvs.toArray.filter (·.1 != k)).toList
  | _ => []

/-- `one_stage.proofs.reads_registered`'s name: `RegisteredInput<w>_v1`, `w` a positive decimal without a leading zero. -/
def registeredInputName (k : String) : Bool :=
  let cs := k.toList
  let p := "RegisteredInput".toList
  let s := "_v1".toList
  let mid := (cs.drop p.length).take (cs.length - p.length - s.length)
  cs.length > p.length + s.length && cs.take p.length == p && cs.drop (cs.length - s.length) == s &&
    mid.all Char.isDigit && mid.head? != some '0'

/-- `one_stage.proofs.reads_registered`: the descriptor has an input bound at registration, or no `definitions` object
to tell. -/
def readsRegistered (program : Json) : Bool :=
  match getD program "definitions" with
  | .obj kvs => kvs.toArray.any fun (k, _) => registeredInputName k
  | _ => true

private def circuitOf (pr : Json) : V (String × Option Json) := do
  let .str circuit := getD pr "circuit" | throw "proofs: circuit is not a SHA-512 digest (128 lowercase hex)"
  if !digest? circuit then throw "proofs: circuit is not a SHA-512 digest (128 lowercase hex)"
  let registered ← match getD pr "registered" with
    | .null => pure none
    | r@(.obj _) => pure (some r)
    | _ => throw "proofs: registered is not the verifier's registered file (an object) or null"
  pure (circuit, registered)

def paramsOfJson (j : Json) : V Params := do
  let some pr := field? j "proofs"
    | throw "the parameters name no proofs (the auditor's circuit digest and the verifier's registered reads)"
  exactly pr #["circuit", "registered"] "proofs"
  let (circuit, registered) ← circuitOf pr
  let rest := Json.mkObj (without j "proofs")
  let base ← OneStage.paramsOfJson rest
  let some partition := base.verifier.partition
    | throw "partition is null: the sessions are verified against the verifier's own partition object"
  let programJ := getD rest "program"
  pure { base, own := { programJ, program := (← Flock.canon programJ).toUTF8, partition, circuit, registered } }

/-- The `verdicts` command's own inputs: exactly `{"circuit", "partition", "program", "registered"}`, the program and
partition objects. -/
def ownOfJson (j : Json) : V Own := do
  exactly j #["circuit", "partition", "program", "registered"] "the verifier's own inputs"
  let (circuit, registered) ← circuitOf j
  let (programJ@(.obj _), partition@(.obj _)) := (getD j "program", getD j "partition")
    | throw "the program and the partition are not objects"
  pure { programJ, program := (← Flock.canon programJ).toUTF8, partition, circuit, registered }

private def sessionOf (j : Json) : V Session := do
  let withCoins := (field? j "coins").isSome
  exactly j (if withCoins then #["coins", "name", "record", "reps"] else #["name", "record", "reps"])
    "a session of the proofs"
  let (.str name, .str record, .obj _) := (getD j "name", getD j "record", getD j "reps")
    | throw "a session of the proofs is not a name, its record's digest and its rep files' digests"
  let reps ← (entries (getD j "reps")).mapM fun (f, d) => match d with
    | .str d => pure (f, d)
    | _ => throw s!"session {name}: rep file {f}'s digest is not a string"
  let coins ← match getD j "coins" with
    | .arr ss => pure ss
    | .null => pure #[]
    | _ => throw s!"session {name}: coins is not a list of segments"
  pure { name, record, reps, coins }

/-- The evidence's `proofs`: the public file's digest and the sessions. -/
def proofsOf (pr : Json) : V (String × Array Session) := do
  exactly pr #["public", "sessions"] "the evidence's proofs"
  let .str pub := getD pr "public" | throw "proofs: public is not a SHA-512 digest"
  let .arr ss := getD pr "sessions" | throw "proofs: sessions is not a list"
  pure (pub, ← ss.mapM sessionOf)

def evidenceOf (b : ByteArray) : V Evidence := do
  let j ← Core.canonicalJson b
  if !(j matches .obj _) then throw "the evidence is not an object"
  if (field? j "verdicts").isSome then
    throw "the evidence names verdicts: this audit's verdicts are its own run of the verifier on the proofs"
  let some pr := field? j "proofs" | throw "the evidence names no proofs (the public file and the sessions' files)"
  let base ← OneStage.evidenceOf (← Flock.canon (Json.mkObj (without j "proofs" ++ [("verdicts", Json.mkObj [])]))).toUTF8
  let (pub, sessions) ← proofsOf pr
  if sessions.map (·.name) != base.sessions then throw "proofs: the sessions are not the evidence's sessions, in order"
  pure { base, pub, sessions }

/-- `flock-verify`'s `checkIdentifier` without the verifier's own identifier: a public header that names one refuses. -/
def checkIdentifier (st : Flock.Stmt) : V Unit :=
  match st.pub.header.getObjVal? "identifier" with
  | .error _ => pure ()
  | .ok j => do
    let .obj _ := j | throw "identifier: the public header's identifier is not a JSON object"
    if let some e := Flock.Program.domainError j "identifier" then
      throw s!"identifier: the public header's identifier is outside canonical JSON's RFC 8785 domain: {e}"
    throw "identifier: the public header names an identifier; pass the verifier's own (--identifier F)"

/-- `flock-verify`'s `buildSessionOf` on an hm96-sha512 row statement with none of `--work`, `--stratified`,
`--closure`, `--public-inputs` or `--identifier`. -/
def buildSessionOf (parsed : V Flock.Circuit) (tags : Flock.Tags) (circuitB pubB : ByteArray) (coins : Flock.Coins)
    (tables : List (String × ByteArray)) (count : Nat) (draw partition : Option Json) (program : Option ByteArray)
    (registered : Option (Array Flock.Registered.Port)) : V (Flock.SessionSpec × Array Flock.Stmt) := do
  if !tags.hm96Rows then throw s!"{tags.name}: not an hm96-sha512 row statement"
  let (spec, sts) ← if tags.hiddenOutputs then
      Flock.Stmt.setupTablesOf parsed tags circuitB pubB coins count tables draw partition program
    else Flock.Stmt.setupTables tags circuitB pubB coins count tables draw partition program
  for st in sts do
    Flock.checkRegistered st registered
    checkIdentifier st
  Flock.Registered.checkFull registered draw
  return (spec, sts)

/-- `flock-verify`'s `programDigest`: SHA-512 of the canonical descriptor without `annotations`. -/
def programDigest (b : ByteArray) : Option String := do
  let j ← (Json.parse (← String.fromUTF8? b)).toOption
  let bare := match j with
    | .obj kvs => Json.mkObj (kvs.toArray.filter (·.1 != "annotations")).toList
    | x => x
  return Flock.hexOf (Flock.Sha512.hash (← (Flock.canon bare).toOption).toUTF8)

private def headerLine (b : ByteArray) : ByteArray := Id.run do
  for i in [0:b.size] do
    if b[i]! == 10 then return b.extract 0 i
  return b

/-- `one_stage.proofs.proved_units`: the public file's unit indices (`units.indices`) of the instances the session's
record drew (`unit_draw.units`), ascending. -/
def provedUnits (pubB : ByteArray) (record : String) : V (Array Nat) := do
  let some line := String.fromUTF8? (headerLine pubB) | throw "the public file's header is not UTF-8"
  let h ← Json.parse line
  let .arr is := getD (getD h "units") "indices" | throw "the public file's header names no units.indices"
  let .arr ds := getD (getD (← Json.parse record) "unit_draw") "units" | throw "the record's unit_draw names no units"
  let units ← ds.mapM fun d => match nat? d >>= fun i => is[i]? >>= nat? with
    | some u => pure u
    | none => throw "the record draws an instance the public file's units.indices does not name"
  return units.qsort (· < ·)

/-- One `flock-verify verify --zk` over the evidence's sessions on the blobs, as `one_stage.proofs.verify` runs it: each
session's verdict (`flock-verify/<name>`, of record), with the units an accepted one proved. Session `i`'s coins are the
window's challenger's at slot `i` (`Flock.Zk.onChallenger`, `--challenger-slot i`), each verdict `Flock.Zk.verifyChallenger`'s;
that its coins are the challenger's releases is `roundsWhy`'s check. A refusal for what
`proofs.verify` refuses: registered values read without the verifier's registered file, a file no blob holds, an
accepted session naming another program than the partition object's, or proved units other than the evidence's. -/
def verdicts (p : Own) (pub : String) (sessions : Array Session) (provedUnits? : Option (Array Nat))
    (bs : Core.Blobs) : V (Array (String × Json)) := do
  let some tags0 := Flock.Tags.find? STATEMENT | throw s!"the statement {STATEMENT} is not known"
  if tags0.plainLeaves then throw s!"{tags0.name}: a statement with plain SHA-512 leaves: --zk's hiding needs hm96 leaves"
  if p.registered.isNone && readsRegistered p.programJ then
    throw ("registered: the program reads registered values (a RegisteredInput among its definitions) and the " ++
      "verifier holds no registered reads: verify with its own registered file")
  let circuitB ← bs.get p.circuit "circuit"
  let pubB ← bs.get pub "public file"
  let tables := Flock.MUFU_TABLES.filterMap fun t =>
    (bs.entries.find? fun (d, _) => Flock.hexOf d == t.sha512).map fun (_, b) => (t.name, b)
  let count := 1
  let tags := Flock.Zk.tagsChallenger tags0 (Flock.Zk.maskLogOf circuitB) count
  let (registered, complete, known, regParams) ← match p.registered with
    | none => pure (none, false, #[], #[])
    | some j => do
      let own ← Flock.Registered.ofJson j
      let known ← Flock.Registered.knownOf j
      Flock.Registered.checkKnownAll own known
      pure (some own, ← Flock.Registered.completeOf j, known ++ Flock.KnownRead.fullRows own (← Flock.KnownRead.readOf j),
        ← Flock.Registered.paramsOf j)
  let partition := some p.partition
  let program := some p.program
  let parsed := Flock.ProvedScope.circuitOf tags circuitB tables
  let scope := Flock.ProvedScope.checkOf parsed tags true pubB count false registered.isSome
  let setupOf := fun (slot : Nat) (draw : Option Json) =>
    match (do
        Flock.Registered.checkProgram program registered.isSome
        Flock.RecSetup.checkInner none none (pure (none, none))
        let r ← buildSessionOf parsed tags circuitB pubB Flock.Coins.os tables count draw partition program registered
        let r := (← Flock.Zk.onChallenger r.1 slot, r.2)
        if complete then r.2.forM fun (st : Flock.Stmt) => Flock.Registered.checkComplete st.c (registered.getD #[])
        r.2.forM fun (st : Flock.Stmt) => Flock.Registered.checkCover st.c program partition registered regParams
        r.2.forM fun (st : Flock.Stmt) => Flock.KnownRead.check st.c known
        Flock.RecSetup.checks none none tags0 tables r.2 circuitB partition program registered none none none
        Flock.RecSetup.oneFull none r.2
        r.2.forM Flock.Zk.shapeOk
        r.2.forM (Flock.RecCoins.check · registered (Flock.RecCoins.Source.atC3 none none none)
          (Flock.RecSetup.c3 none none))
        r.2.forM (Flock.RecVStar.checkPubRows · registered none none)
        r.2.forM (Flock.RecVStar.checkTopRows · registered none)
        r.2.forM (Flock.RecCoins.checkLink · registered (Flock.RecCoins.Source.atC3 none none none) none)
        pure r) with
    | .ok (spec, sts) => (if draw.isSome then Flock.ProvedScope.drawLaw true tags false draw else .ok ()).map fun () =>
        (spec, sts.map Flock.Setup.ofCircuit, (sts[0]?.map (·.partition)).getD Json.null,
          Flock.registeredOpenings sts registered)
    | .error e => Except.error s!"setup: {e}"
  let programSha := programDigest p.program
  let want := getD p.partition "program"
  let mut out : Array (String × Json) := #[]
  let mut proved : Array Nat := #[]
  let mut all := true
  for (s, slot) in sessions.toList.zipIdx do
    let some record := String.fromUTF8? (← bs.get s.record s!"session {s.name}'s record")
      | throw s!"session {s.name}'s record is not UTF-8"
    let draw := Flock.Witness.recordDraw record
    let st : V (Flock.SessionSpec × Array Flock.Setup × Json × Json) := setupOf slot draw
    let files : V (Array ByteArray) := do
      let (spec, _, _) ← st
      spec.repStreams.mapM fun (_, _, w) => do
        let n := s!"{w.replace "/" "."}.bin"
        let some (_, d) := s.reps.find? (·.1 == n) | throw s!"session files: the session names no rep file {n}"
        (bs.get d s!"session files: {n}")
    let verdict : V Unit := do
      let proofs ← files
      let (spec, tabs, _) ← st
      Flock.Zk.verifyChallenger spec tabs record proofs
      scope
      Flock.ProvedScope.sizesOk true (tabs.map (·.m))
      if draw.isNone then Flock.ProvedScope.drawLaw true tags false draw
      Flock.RecSetup.fullDraw none false draw record
      Flock.RecSetup.innerDrawn none
      Flock.ProvedScope.programHeld program
      Flock.RecordText.check record
    let (ok, why) := match verdict with | .ok () => (true, "") | .error e => (false, e)
    all := all && ok
    if ok && (programSha.map Json.str) != some want then
      throw s!"session {s.name}'s verdict names program {programSha.getD "none"}, not the partition object's {want.compress}"
    let units ← if ok then provedUnits pubB record else pure #[]
    proved := proved ++ units
    let (part, opened) := match st with | .ok (_, _, p, o) => (p, o) | .error _ => (Json.null, Json.null)
    out := out.push (s!"flock-verify/{s.name}", Json.mkObj ([("accepted", Json.bool ok), ("of_record", Json.bool true),
      ("session", Json.str s.name), ("why", Json.str why), ("units", natsJson units)] ++
      (if part.isNull then [] else [("partition", part)]) ++
      (match programSha with | some h => [("program_sha512", Json.str h)] | none => []) ++
      (match registered with | some r => [("registered", Flock.Registered.finding r)] | none => []) ++
      (if opened.isNull then [] else [("openings", opened)])))
  if let some want := provedUnits? then
    if all && proved.qsort (· < ·) != want then throw "the units the sessions proved are not the evidence's proved_units"
  return out

/-- The session stream its link points are drawn on (`Firewall.Coins.SESSION`). -/
def POINTS : String := "session"

private def hexes (j : Json) (what : String) : V (Array String) := do
  let .arr xs := j | throw s!"{what} is not a list of hex strings"
  xs.mapM fun | .str s => pure s | _ => throw s!"{what} is not a list of hex strings"

/-- What a session's link points are released after: the SHA-512 of its tables' root_B digests
(`link.root_b_sha512`, 64 bytes each), in the record's key order. -/
def pointsAfter (link : Json) : V ByteArray := do
  let .obj _ := getD link "root_b_sha512" | throw "record: link.root_b_sha512 is not an object"
  let ds ← (entries (getD link "root_b_sha512")).mapM fun (t, d) => match d with
    | .str h => if h.length == 128 then Flock.unhex h else throw s!"record: link.root_b_sha512.{t} is not 64 bytes of hex"
    | _ => throw s!"record: link.root_b_sha512.{t} is not 64 bytes of hex"
  return Flock.Sha512.hash (ds.foldl (· ++ ·) .empty)

/-- Round k's request to the window's challenger (live PROTOCOL.md §9), all of it public: its stream's name (two bytes
big-endian of its UTF-8 length, then the name), the stream's chain at the round `a` (64 bytes), then the round's coin
count `n`, its F128s as the record's `coins` lists them (eight bytes little-endian). Its coins are released after its
SHA-512. -/
def roundRequest (stream : String) (a : ByteArray) (n : Nat) : ByteArray :=
  let s := stream.toUTF8
  ⟨#[(s.size / 256 % 256).toUInt8, (s.size % 256).toUInt8]⟩ ++ s ++ a ++ Flock.le64 n.toUInt64

/-- Session `slot`'s uses of the challenger's coins, each with the coins (as the record's F128 hex) its bytes must
begin with: its link points, of family `slot/<slot>/session` after `pointsAfter`, then each stream's rounds in the
record's order, of family `slot/<slot>/<stream>` after the SHA-512 of the round's request (`roundRequest`) at the
stream's chain `a_k = SHA-512(a_{k-1} ‖ msg_sha512_k)` from `a_{-1}` empty and the round's own coin count, the record's
F128s `n` (live PROTOCOL.md §9), so two rounds of equal bytes are two uses and a release after a request of another count
is no use; `⌈n/4⌉` coins of 64 bytes for `n` F128s. -/
def sessionUses (slot : Nat) (record : String) : V (List (Challenger.Use × Array String)) := do
  let j ← (Json.parse record).mapError (s!"record: not JSON: {·}")
  let link := getD j "link"
  let .arr ps := getD link "points" | throw "record: link.points is not a list"
  let pts ← ps.foldlM (fun acc p => return acc ++ (← hexes p "a link point")) #[]
  let mut out := [({ label := Hello.slotStream slot POINTS, after := ← pointsAfter link, coins := (pts.size + 3) / 4 },
    pts)]
  let .arr ss := getD j "streams" | throw "record: streams is not a list"
  for s in ss do
    let .str name := getD s "stream" | throw "record: a stream's name is not a string"
    if name == POINTS then throw s!"record: a stream is named {POINTS}, the link points' family"
    if name.toUTF8.size ≥ 65536 then throw s!"record: a stream's name is past 65535 bytes"
    let .arr rs := getD s "rounds" | throw s!"record: {name}'s rounds are not a list"
    let mut chain := ByteArray.empty
    for r in rs do
      let .str sha := getD r "msg_sha512" | throw s!"record: a round of {name} has no msg_sha512"
      if sha.length != 128 then throw s!"record: a round of {name}'s msg_sha512 is not 64 bytes of hex"
      let cs ← hexes (getD r "coins") s!"the coins of a round of {name}"
      chain := Flock.Sha512.hash (chain ++ (← Flock.unhex sha))
      out := out ++ [({ label := Hello.slotStream slot name, after := Flock.Sha512.hash (roundRequest name chain cs.size),
        coins := (cs.size + 3) / 4 }, cs)]
  return out

/-- Why the coins `ξ` don't begin with the F128s `want` (record hex, `{hi:016x}{lo:016x}`, 16 bytes each); none when
they do. -/
def coinsMismatch (want : Array String) (ξ : ByteArray) : Option String :=
  (want.toList.zipIdx.find? fun (h, i) => Flock.hexOf (ξ.extract (16 * i) (16 * i + 16)) != h).map fun (_, i) =>
    s!"coin {i} is not the challenger's"

def ROUNDS_NEED_CHALLENGER : String :=
  "v2 reads every session's link points and round coins from the auditor's challenger, and the parameters name " ++
  "none (challenger and window_root)"

/-- Why the window's coins are not all the challenger's (`Challenger.roundCoins`), one release a use; none when they
are. The uses are the draw's (its family, after the registration's digest, as many coins as its segment opens) and
each session's (`sessionUses`, slot `i` the session's index), each with its segment: the draw's `coins.segment`, then
each session's `coins`. The log must release the draw before any of a session's families, and a session's link
points before its rounds (`order`), and each use of a session's coins must be one run among that session's releases
(`groups`, one a session). Every coin a session's record holds must be the challenger's (`coinsMismatch`). -/
def roundsWhy (p : Params) (c : OneStage.Commitment) (e : Evidence) (bs : Core.Blobs) : Option String :=
  match p.base.challenger, p.base.verifier.windowRoot with
  | some (key, label), some w =>
    let r : V Unit := do
      let draw := getD e.base.coins "segment"
      let .arr opened := getD draw "openings" | throw Challenger.SEGMENT_FORM
      let mut uses : Array Challenger.Use := #[{ label, after := c.digest, coins := opened.size }]
      let mut wants : Array (Option (String × Array String)) := #[none]
      let mut segments : Array Json := #[draw]
      let mut order : Array (String × String) := #[]
      let mut groups : Array (List String) := #[]
      for (s, i) in e.sessions.toList.zipIdx do
        let some record := String.fromUTF8? (← bs.get s.record s!"session {s.name}'s record")
          | throw s!"session {s.name}'s record is not UTF-8"
        let us ← (sessionUses i record).mapError (s!"session {s.name}: {·}")
        unless s.coins.size == us.length do
          throw s!"session {s.name}: {s.coins.size} segments for its {us.length} uses of the challenger's coins"
        let points := Hello.slotStream i POINTS
        let fams := (us.map (·.1.label)).eraseDups
        order := order ++ (fams.map (label, ·)).toArray ++ ((fams.filter (· != points)).map (points, ·)).toArray
        groups := groups.push fams
        uses := uses ++ us.toArray.map (·.1)
        wants := wants ++ us.toArray.map fun (u, hs) => some (s!"{s.name}'s {u.label}", hs)
        segments := segments ++ s.coins
      let coins := Json.mkObj [("ledger", getD e.base.coins "ledger"), ("main", getD e.base.coins "main"),
        ("pins", getD e.base.coins "pins"), ("segments", Json.arr segments), ("session", getD e.base.coins "session")]
      let ξs ← Challenger.roundCoins key w uses.toList order.toList groups.toList coins
      for (ξ, want) in ξs.zip wants.toList do
        if let some (what, hs) := want then
          if let some why := coinsMismatch hs ξ then throw s!"session {what}: {why}"
    match r with
    | .ok () => none
    | .error why => some why
  | _, _ => some ROUNDS_NEED_CHALLENGER

/-- v2's check: the sessions' verdicts (`verdicts`), then v1's check at the draw's coins (`OneStage.coinsOf`,
`OneStage.checkAt`), and an accept only when every coin the window read is the challenger's, one release a use
(`roundsWhy`). -/
def checkBlobs (p : Params) (c : OneStage.Commitment) (e : Evidence) (bs : Core.Blobs) :
    Core.Outcome OneStage.Public Core.Integrity :=
  match verdicts p.own e.pub e.sessions (some e.base.provedUnits) bs with
  | .error err => .reject s!"proofs: {err}"
  | .ok vs =>
    match Core.atCoins (OneStage.coinsOf p.base c e.base)
        (fun ξ => OneStage.checkAt p.base c ξ { e.base with verdicts := vs }) with
    | .accept x y => match roundsWhy p c e bs with
      | some why => .reject s!"proofs: round coins: {why}"
      | none => .accept x y
    | o => o

def protocol : Core.BlobProtocol where
  id := PROTOCOL
  Params := Params
  Commitment := OneStage.Commitment
  Evidence := Evidence
  Public := OneStage.Public
  Profile := Core.Integrity
  params := paramsOfJson
  commitment := OneStage.commitmentOf
  evidence := evidenceOf
  digest c := c.digest
  check p c e := checkBlobs p c e Core.Blobs.empty
  publicJson := OneStage.publicJson
  profileJson := Core.Integrity.toJson
  checkBlobs := checkBlobs

/-- `protocol.main`'s commands, and `verdicts OWN PROOFS [BLOB ...]`: the sessions' verdicts alone (`verdicts`, with no
proved units to bind), from the verifier's own inputs (`ownOfJson`) and the evidence's `proofs` (`proofsOf`), each file
canonical JSON; one line, `{"verdicts": [verdict, ...]}` in session order or `{"refused": why}`, exit 0 or 1. The
difftest holds it to `flock-verify verify --zk` (`one_stage.proofs.verify`). `uses SLOT RECORD`: the session record's
uses of the challenger's coins at slot `SLOT` (`sessionUses`), one line `{"uses": [{after, coins, label, want}, ...]}`
or `{"refused": why}`, which `one_stage.proofs.session_uses` is held to. -/
def main (args : List String) : IO UInt32 := do
  match args with
  | "verdicts" :: own :: proofs :: blobs =>
    let bs ← blobs.toArray.mapM fun (f : String) => IO.FS.readBinFile f
    let ownB ← IO.FS.readBinFile own
    let proofsB ← IO.FS.readBinFile proofs
    let r : V (Array (String × Json)) := do
      let (pub, sessions) ← proofsOf (← Core.canonicalJson proofsB)
      verdicts (← ownOfJson (← Core.canonicalJson ownB)) pub sessions none (Core.Blobs.ofArray bs)
    let line := match r with
      | .ok vs => Json.mkObj [("verdicts", Json.arr (vs.map (·.2)))]
      | .error e => Json.mkObj [("refused", Json.str e)]
    IO.println ((Flock.canon line).toOption.getD line.compress)
    return if r.toBool then 0 else 1
  | ["uses", slot, record] =>
    let text ← IO.FS.readFile record
    let r : V (List (Challenger.Use × Array String)) := do
      let some i := slot.toNat? | throw "the slot is not a natural number"
      sessionUses i text
    let line := match r with
      | .ok us => Json.mkObj [("uses", Json.arr (us.toArray.map fun (u, hs) => Json.mkObj [("after",
          Json.str (Flock.hexOf u.after)), ("coins", Lean.toJson u.coins), ("label", Json.str u.label),
          ("want", Json.arr (hs.map Json.str))]))]
      | .error e => Json.mkObj [("refused", Json.str e)]
    IO.println ((Flock.canon line).toOption.getD line.compress)
    return if r.toBool then 0 else 1
  | _ => protocol.main args

#guard registeredInputName "RegisteredInput96_v1"
#guard registeredInputName "RegisteredInput1_v1"
#guard !registeredInputName "RegisteredInput0_v1"
#guard !registeredInputName "RegisteredInput01_v1"
#guard !registeredInputName "RegisteredInput_v1"
#guard !registeredInputName "RegisteredInput96_v2"
#guard !registeredInputName "XRegisteredInput96_v1"
#guard readsRegistered (Json.mkObj [])
#guard !readsRegistered (Json.mkObj [("definitions", Json.mkObj [("Rope_v1", Json.null)])])
#guard readsRegistered (Json.mkObj [("definitions", Json.mkObj [("RegisteredInput96_v1", Json.null)])])
#guard (provedUnits "{\"units\":{\"indices\":[7,3,9]}}\nrest".toUTF8 "{\"unit_draw\":{\"units\":[2,0]}}").toOption
  == some #[7, 9]
#guard (provedUnits "{\"units\":{\"indices\":[7]}}".toUTF8 "{\"unit_draw\":{\"units\":[1]}}").toOption.isNone
#guard ((sessionUses 1 ("{\"link\":{\"points\":[[\"" ++ "aa".pushn 'a' 30 ++ "\"]],\"root_b_sha512\":{\"t\":\"" ++
    "".pushn '0' 128 ++ "\"}},\"streams\":[{\"stream\":\"t/rep0\",\"rounds\":[{\"msg_sha512\":\"" ++ "".pushn '1' 128 ++
    "\",\"coins\":[\"a\",\"b\",\"c\",\"d\",\"e\"]}]}]}")).toOption.map
    (·.map fun (u, hs) => (u.label, u.coins, hs.size))) == some [("slot/1/session", 1, 1), ("slot/1/t/rep0", 2, 5)]
#guard (roundRequest "t/é" (ByteArray.mk (Array.replicate 64 7)) 261).data ==
  #[0, 4, 116, 47, 195, 169] ++ Array.replicate 64 7 ++ #[5, 1, 0, 0, 0, 0, 0, 0]
#guard coinsMismatch #["00".pushn '0' 30] (ByteArray.mk (Array.replicate 64 0)) == none
#guard (coinsMismatch #["01".pushn '0' 30] (ByteArray.mk (Array.replicate 64 0))).isSome

end SampledProofs.OneStageZk
