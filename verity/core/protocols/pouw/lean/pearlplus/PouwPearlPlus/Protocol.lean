module

public import CoreProtocol.Blobs
public import PouwPearlPlus.Open
meta import CoreProtocol.Blobs
meta import PouwPearlPlus.Open

/-!
# Pearl+'s run verdict as a `Core.BlobProtocol`: `verity/pouw/pearl-plus/v1`

Remote's work component (`Core.Remote.V1.components`): `check` on a served run's retained passes, its pass files the
caller's blobs.

**Params**, canonical JSON with exactly these keys, the verifier's own:

    {"coins": epoch coins, "draws": t, "record": {"commitment": hex, "draw": draw, "schema": s, "source": hex}}

the epoch's coins (`EpochCoins.ofJson`: a live session's on the auditor host's challenger, or an in-process one's), the
number of draws, and the verifier's record of the coin it drew on this run's commitment (`CoinRecord`, as `verify_run`
writes it: its schema and its draw), which pins the source (`after-coin`): a bare source, which `check` also takes, is no
input here, so the coin is always the one drawn on the run. The verdict is of record (`ofRecord`, `--of-record` as
`served_pearlplus.sh` runs it): only a challenger draw on live coins is taken (`ofRecordReason`), so in-process coins, the
node's draw and a v1 record are refused as `of-record`.

**Commitment**, canonical JSON with exactly these keys:

    {"manifests": [manifest, …], "record": hex, "window": {"dropped": [s, …], "record": W, "rounds": [s, …] | null}}

the retained passes' manifests, a nonempty list; the SHA-512 digest (lowercase hex) of the registration record the
run is on, the record the policy and sampled proofs commit to, which remote ties it to (`Core.Remote.V1.run`); and the
window record W the passes are tied to (`served_passes.py retain --window-record`'s, the record that registration is
chained right after, which remote ties it to too, `Core.Remote.V1.window`), with the rounds the passes are, in their
order (`null`: pass i is round i), and the rounds dropped from the retention before the coin. The check holds the
passes to W (`tieOf`, `verify_run.tie`): W registers exactly each call's A rows and y and each fixed operand's B rows among its
``pearl-a/``, ``pearl-y/`` and ``pearl-b/`` values, beside the dropped rounds' calls', each B at the root the pass's
tiles open B against (`Pass.weightRoots`, which the fixed operands' root folds), so a run whose passes or whose rootB are not
the ones W registers is refused before its coin, a version-2 W that registers no fixed operand among them. That is a W at
version 2, its header's ``version`` (`wVersion`). A W that states none is version 1, written before W registered the
fixed operands: it is held to the calls alone, as it was then, and the verdict's tie names its fixed operands untied
(`WEIGHTS_UNTIED`), so its accept says that no fixed operand it credits is a registered value. Its digest is
SHA-512(`commitmentTag` ‖ 0 ‖ the commitment's bytes).

**Evidence**, canonical JSON `{"files": [files, …]}`, one object per pass in the commitment's order: each manifest path
the pass's openings read, mapped to the SHA-512 (lowercase hex) of the blob holding its bytes. A file is the blob under
that digest, held to its manifest entry (`held`); a path the object doesn't name, or a digest no blob has, is the tile's
`FileNotFoundError` or the opening's failure, which rejects the run.

**The check** is `check` on them: it accepts with the run's credit at Pearl's assumption at δ = `DELTA`, or rejects with
`check`'s reason. It never leaves a run undecided.

**Public**, on acceptance: the credit (`credited`, `credited_tiles`, `macs`, `rho`, `epsilon`) and `delta`.

**Profile**: the draws asked for, the tiles drawn and the passes.
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace PouwCheck.PearlPlus.Protocol

open Lean (Json toJson)
open Flock (V)
open PouwCommit

def protocolId : String := PROTOCOL
def commitmentTag : String := "verity/pouw/pearl-plus/manifests/v1"
def paramsKeys : List String := ["coins", "draws", "record"]
def recordKeys : List String := ["commitment", "draw", "schema", "source"]
def commitmentKeys : List String := ["manifests", "record", "window"]
def windowKeys : List String := ["dropped", "record", "rounds"]
def evidenceKeys : List String := ["files"]

structure Params where
  coins : EpochCoins
  draws : Nat
  record : CoinRecord

structure Commitment where
  manifests : List Json
  record : ByteArray
  /-- the window record W the passes are tied to, and its digest (`Flock.WindowRecord.digest`) -/
  window : Json
  windowDigest : ByteArray
  rounds : Option (List Nat)
  dropped : List Nat
  bytes : ByteArray

structure Evidence where
  files : List Json

structure Profile where
  draws : Nat
  drawn : Nat
  passes : Nat

def keys? : Json → Option (List String)
  | .obj kvs => some ((kvs.toArray.map (·.1)).qsort (· < ·)).toList
  | _ => none

def bytesOf (j : Json) (what : String) : V ByteArray := do
  let some s := str? j what | throw s!"{what} is not a hex string"
  let some b := unhex s | throw s!"{what} is not lowercase hex"
  pure b

def paramsOf (j : Json) : V Params := do
  unless keys? j == some paramsKeys do throw s!"keys are not {", ".intercalate paramsKeys}"
  let coins ← match EpochCoins.ofJson ((get? j "coins").getD .null) with
    | .ok c => pure c
    | .error e => throw s!"coins: {e}"
  let some draws := ((get? j "draws").getD .null).getNat?.toOption | throw "draws is not a nonnegative integer"
  let r := (get? j "record").getD .null
  unless keys? r == some recordKeys do throw s!"record: keys are not {", ".intercalate recordKeys}"
  let commitment ← bytesOf r "commitment"
  let source ← bytesOf r "source"
  let some schema := str? r "schema" | throw "record: schema is not a string"
  pure { coins, draws, record := { commitment, source, schema, draw := (get? r "draw").getD .null } }

def commitmentOf (b : ByteArray) : V Commitment := do
  let j ← Core.canonicalJson b
  unless keys? j == some commitmentKeys do throw s!"keys are not {", ".intercalate commitmentKeys}"
  let some (.arr ms) := get? j "manifests" | throw "the retained passes' manifests are not a list"
  if ms.isEmpty then throw "the run holds no pass"
  let record ← bytesOf j "record"
  unless record.size == 64 do throw "record is not a SHA-512 digest"
  let w := (get? j "window").getD .null
  unless keys? w == some windowKeys do throw s!"window: keys are not {", ".intercalate windowKeys}"
  let window := (get? w "record").getD .null
  unless window matches .obj _ do throw "window: record is not an object"
  let windowDigest ← match Flock.WindowRecord.digest window with
    | .ok d => pure d
    | .error e => throw s!"window: record: {e}"
  let nats (what : String) : V (Option (List Nat)) := match (get? w what).getD .null with
    | .null => pure none
    | .arr xs => match xs.toList.mapM (·.getNat?.toOption) with
      | some ns => pure (some ns)
      | none => throw s!"window: {what} is not a list of nonnegative integers"
    | _ => throw s!"window: {what} is not a list of nonnegative integers"
  let rounds ← nats "rounds"
  let some dropped ← nats "dropped" | throw "window: dropped is not a list of nonnegative integers"
  pure { manifests := ms.toList, record, window, windowDigest, rounds, dropped, bytes := b }

def Commitment.digest (c : Commitment) : ByteArray :=
  Flock.Sha512.hash (commitmentTag.toUTF8 ++ ByteArray.mk #[0] ++ c.bytes)

def evidenceOf (b : ByteArray) : V Evidence := do
  let j ← Core.canonicalJson b
  unless keys? j == some evidenceKeys do throw s!"keys are not {", ".intercalate evidenceKeys}"
  let some (.arr fs) := get? j "files" | throw "files is not a list"
  unless fs.all (keys? · |>.isSome) do throw "files is not a list of objects"
  pure { files := fs.toList }

def inputOf (p : Params) (c : Commitment) : Input :=
  { coins := p.coins, manifests := c.manifests, draws := p.draws, held := some p.record, tie := some c.window,
    rounds := c.rounds, dropped := c.dropped, ofRecord := true }

/-- The opener on the caller's blobs: pass `i`'s file at manifest entry `spec` is the blob its evidence names for the
entry's path, `held` to the entry. -/
def opener (c : Commitment) (e : Evidence) (bs : Core.Blobs) (pl : Plan) (i tid : Nat) : Except String TileOpening :=
  let names := e.files.getD i .null
  let file (spec : Json) : Except String ByteArray := do
    let f := (str? spec "path").getD ""
    let some d := str? names f | throw s!"FileNotFoundError: {f}"
    held spec (← bs.get d f)
  opening (c.manifests.getD i .null) pl.passes[i]! file tid

def checkBlobs (p : Params) (c : Commitment) (e : Evidence) (bs : Core.Blobs) : Core.Outcome Credit Profile :=
  let i := inputOf p c
  if e.files.length != c.manifests.length then
    .reject s!"evidence: {e.files.length} passes' files for {c.manifests.length} passes"
  else
    match plan i with
    | .error r => .reject r.reason
    | .ok pl =>
      let ans := check i (opener c e bs pl)
      match ans.accept, ans.work with
      | true, some w => .accept w { draws := p.draws, drawn := pl.drawn, passes := pl.passes.length }
      | _, _ => .reject (ans.reason.getD "the run is not accepted")

def publicJson (w : Credit) : Json :=
  Json.mkObj [("credited", toJson w.credited), ("credited_tiles", toJson w.creditedTiles), ("delta", .str (toString DELTA)),
    ("epsilon", .str (toString w.epsilon)), ("macs", toJson w.macs), ("rho", .str (toString w.rho))]

def profileJson (pr : Profile) : Json :=
  Json.mkObj [("draws", toJson pr.draws), ("drawn", toJson pr.drawn), ("passes", toJson pr.passes)]

def protocol : Core.BlobProtocol where
  id := protocolId
  Params := Params
  Commitment := Commitment
  Evidence := Evidence
  Public := Credit
  Profile := Profile
  params := paramsOf
  commitment := commitmentOf
  evidence := evidenceOf
  digest := Commitment.digest
  check p c e := checkBlobs p c e Core.Blobs.empty
  checkBlobs := checkBlobs
  publicJson := publicJson
  profileJson := profileJson

private def coins : String :=
  "{\"epoch\":0,\"root\":\"" ++ String.ofList (List.replicate 128 '0') ++ "\",\"scheme\":\"" ++ SCHEME ++ "\"}"
private def params (record : String) : ByteArray :=
  ("{\"coins\":" ++ coins ++ ",\"draws\":3,\"record\":" ++ record ++ "}").toUTF8
private def recordOf (schema : String) : String :=
  "{\"commitment\":\"" ++ String.ofList (List.replicate 128 '0') ++ "\",\"draw\":" ++ nodeDraw.compress ++
    ",\"schema\":" ++ schema ++ ",\"source\":\"" ++ String.ofList (List.replicate 64 '0') ++ "\"}"
private def record : String := recordOf ("\"" ++ COIN_SCHEMA ++ "\"")
private def reasonOf (p c e : ByteArray) : Option String := (protocol.answer p c e #[]).reason
private def windowRec : String := "{\"format\":\"verity/registration/v2\",\"registered\":[]}"
private def windowOf (w : String) : String := "{\"dropped\":[],\"record\":" ++ w ++ ",\"rounds\":null}"
private def runIn (manifests window : String) : ByteArray :=
  ("{\"manifests\":" ++ manifests ++ ",\"record\":\"" ++ String.ofList (List.replicate 128 '0') ++ "\",\"window\":" ++
    window ++ "}").toUTF8
private def run (manifests : String) : ByteArray := runIn manifests (windowOf windowRec)

-- every input gets a verdict: parameters, a commitment or evidence of another shape is refused, naming the input
#guard protocolId == "verity/pouw/pearl-plus/v1"
#guard reasonOf "{}".toUTF8 "[]".toUTF8 "{}".toUTF8 == some "params: keys are not coins, draws, record"
#guard reasonOf (params "{\"source\":\"00\"}") "[]".toUTF8 "{}".toUTF8 ==
  some "params: record: keys are not commitment, draw, schema, source"
#guard reasonOf (params (recordOf "1")) "[]".toUTF8 "{}".toUTF8 == some "params: record: schema is not a string"
#guard reasonOf (params record) "[{}]".toUTF8 "{}".toUTF8 == some "commitment: keys are not manifests, record, window"
#guard reasonOf (params record) (run "[]") "{}".toUTF8 == some "commitment: the run holds no pass"
#guard reasonOf (params record) (run "{}") "{}".toUTF8 ==
  some "commitment: the retained passes' manifests are not a list"
#guard reasonOf (params record) ("{\"manifests\":[{}],\"record\":\"00\",\"window\":" ++ windowOf windowRec ++ "}").toUTF8
    "{}".toUTF8 ==
  some "commitment: record is not a SHA-512 digest"
-- the window record the passes are tied to: W, the rounds the passes are and the dropped rounds
#guard reasonOf (params record) (runIn "[{}]" "{}") "{}".toUTF8 ==
  some "commitment: window: keys are not dropped, record, rounds"
#guard reasonOf (params record) (runIn "[{}]" (windowOf "[]")) "{}".toUTF8 ==
  some "commitment: window: record is not an object"
#guard reasonOf (params record) (runIn "[{}]" ("{\"dropped\":[],\"record\":" ++ windowRec ++ ",\"rounds\":[-1]}"))
    "{}".toUTF8 ==
  some "commitment: window: rounds is not a list of nonnegative integers"
#guard reasonOf (params record) (runIn "[{}]" ("{\"dropped\":null,\"record\":" ++ windowRec ++ ",\"rounds\":[0]}"))
    "{}".toUTF8 ==
  some "commitment: window: dropped is not a list of nonnegative integers"
#guard reasonOf (params record) (run "[{}]") "{}".toUTF8 == some "evidence: keys are not files"
#guard reasonOf (params record) (run "[{}]") "{\"files\":[]}".toUTF8 == some "evidence: 0 passes' files for 1 passes"
-- the check runs tied to the commitment's W, at its rounds and dropped rounds
#guard (do
    let p ← paramsOf (← Core.canonicalJson (params record))
    let c ← commitmentOf (runIn "[{}]" ("{\"dropped\":[2],\"record\":" ++ windowRec ++ ",\"rounds\":[1]}"))
    let i := inputOf p c
    pure (i.tie == some c.window && i.rounds == some [1] && i.dropped == [2] &&
      some (Flock.hexOf c.windowDigest) == (Flock.WindowRecord.digestHex c.window).toOption)).toOption == some true
-- the verdict is of record, on the record's schema and draw: in-process coins are no verdict of record
#guard (do
    let p ← paramsOf (← Core.canonicalJson (params record))
    let i := inputOf p (← commitmentOf (run "[{}]"))
    let some h := i.held | throw "no coin record"
    pure (i.ofRecord && h.schema == COIN_SCHEMA && fieldIs (get? h.draw "from") DRAW_NODE &&
      ((ofRecordReason i.coins i.source i.held).getD "").startsWith "the epoch's coins are an in-process")).toOption ==
  some true
-- well-formed inputs reach the run's verdict, whose refusal is the run's
#guard (reasonOf (params record) (run "[{}]") "{\"files\":[{}]}".toUTF8).isSome
#guard (protocol.answer (params record) (run "[{}]") "{\"files\":[{}]}".toUTF8 #[]).verdict == .reject

end PouwCheck.PearlPlus.Protocol
