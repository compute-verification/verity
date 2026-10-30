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
  exactly j #["name", "record", "reps"] "a session of the proofs"
  let (.str name, .str record, .obj _) := (getD j "name", getD j "record", getD j "reps")
    | throw "a session of the proofs is not a name, its record's digest and its rep files' digests"
  let reps ← (entries (getD j "reps")).mapM fun (f, d) => match d with
    | .str d => pure (f, d)
    | _ => throw s!"session {name}: rep file {f}'s digest is not a string"
  pure { name, record, reps }

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
session's verdict (`flock-verify/<name>`, of record), with the units an accepted one proved. A refusal for what
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
  let coins := Flock.Coins.tree (Flock.Zk.coinSpec (tags0.tableNames count))
  let tags := Flock.Zk.tags (tags0.withCoins coins) (Flock.Zk.maskLogOf circuitB) count
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
  let setupOf := fun (draw : Option Json) =>
    match (do
        Flock.Registered.checkProgram program registered.isSome
        Flock.RecSetup.checkInner none none (pure (none, none))
        let r ← buildSessionOf parsed tags circuitB pubB coins tables count draw partition program registered
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
  let mut coinKeys : Array String := #[]
  let mut out : Array (String × Json) := #[]
  let mut proved : Array Nat := #[]
  let mut all := true
  for s in sessions do
    let some record := String.fromUTF8? (← bs.get s.record s!"session {s.name}'s record")
      | throw s!"session {s.name}'s record is not UTF-8"
    let draw := Flock.Witness.recordDraw record
    -- never another session's setup: every draw whose canonical JSON fails would share one key
    let st := setupOf draw
    let files : V (Array ByteArray) := do
      let (spec, _, _) ← st
      spec.repStreams.mapM fun (_, _, w) => do
        let n := s!"{w.replace "/" "."}.bin"
        let some (_, d) := s.reps.find? (·.1 == n) | throw s!"session files: the session names no rep file {n}"
        (bs.get d s!"session files: {n}")
    let coinKey : Option String := match st with
      | .ok (spec, _, _) =>
        if spec.coinTree.isNone then none
        else (Json.parse record).toOption.bind fun j =>
          (j.getObjVal? "coin_tree").toOption.bind fun ct => (ct.getObjValAs? String "key").toOption
      | .error _ => none
    let verdict : V Unit := do
      let proofs ← files
      if let some k := coinKey then Flock.CoinTree.checkFresh coinKeys k
      let (spec, tabs, _) ← st
      Flock.Zk.verify spec tabs record proofs
      scope
      Flock.ProvedScope.sizesOk true (tabs.map (·.m))
      if draw.isNone then Flock.ProvedScope.drawLaw true tags false draw
      Flock.RecSetup.fullDraw none false draw record
      Flock.RecSetup.innerDrawn none
      Flock.ProvedScope.programHeld program
      Flock.RecordText.check record
    if let some k := coinKey then coinKeys := coinKeys.push k
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

def checkBlobs (p : Params) (c : OneStage.Commitment) (e : Evidence) (bs : Core.Blobs) :
    Core.Outcome OneStage.Public Core.Integrity :=
  match verdicts p.own e.pub e.sessions (some e.base.provedUnits) bs with
  | .error err => .reject s!"proofs: {err}"
  | .ok vs => OneStage.check p.base c { e.base with verdicts := vs }

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
difftest holds it to `flock-verify verify --zk` (`one_stage.proofs.verify`). -/
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

end SampledProofs.OneStageZk
