import CoreProtocol.Blobs
import PouwPearlPlus.Open

/-!
# Pearl+'s run verdict as a `Core.BlobProtocol`: `verity/pouw/pearl-plus/v1`

Remote's work component (`Core.Remote.V1.components`): `check` on a served run's retained passes, its pass files the
caller's blobs.

**Params**, canonical JSON with exactly these keys, the verifier's own:

    {"coins": coin commitment, "draws": t, "record": {"commitment": hex, "source": hex}}

the epoch's coin commitment (`CoinCommitment.ofJson`), the number of draws, and the verifier's record of the coin it drew
on this run's commitment (`CoinRecord`), which pins the source (`after-coin`): a bare source, which `check` also takes, is
no input here, so the coin is always the one drawn on the run.

**Commitment**, canonical JSON with exactly these keys:

    {"manifests": [manifest, …], "record": hex}

the retained passes' manifests, a nonempty list, and the SHA-512 digest (lowercase hex) of the registration record the
run is on: the record the policy and sampled proofs commit to, which remote ties it to (`Core.Remote.V1.run`); the
check doesn't read it. Its digest is SHA-512(`commitmentTag` ‖ 0 ‖ the commitment's bytes).

**Evidence**, canonical JSON `{"files": [files, …]}`, one object per pass in the commitment's order: each manifest path
the pass's openings read, mapped to the SHA-512 (lowercase hex) of the blob holding its bytes. A file is the blob under
that digest, held to its manifest entry (`held`); a path the object doesn't name, or a digest no blob has, is the tile's
`FileNotFoundError` or the opening's failure, which rejects the run.

**The check** is `check` on them: it accepts with the run's credit at Pearl's assumption at δ = `DELTA`, or rejects with
`check`'s reason. It never leaves a run undecided.

**Public**, on acceptance: the credit (`credited`, `credited_tiles`, `macs`, `rho`, `epsilon`) and `delta`.

**Profile**: the draws asked for, the tiles drawn and the passes.
-/

namespace PouwCheck.PearlPlus.Protocol

open Lean (Json toJson)
open Flock (V)
open PouwCommit

def protocolId : String := PROTOCOL
def commitmentTag : String := "verity/pouw/pearl-plus/manifests/v1"
def paramsKeys : List String := ["coins", "draws", "record"]
def recordKeys : List String := ["commitment", "source"]
def commitmentKeys : List String := ["manifests", "record"]
def evidenceKeys : List String := ["files"]

structure Params where
  coins : CoinCommitment
  draws : Nat
  record : CoinRecord

structure Commitment where
  manifests : List Json
  record : ByteArray
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
  let coins ← match CoinCommitment.ofJson ((get? j "coins").getD .null) with
    | .ok c => pure c
    | .error e => throw s!"coins: {e}"
  let some draws := ((get? j "draws").getD .null).getNat?.toOption | throw "draws is not a nonnegative integer"
  let r := (get? j "record").getD .null
  unless keys? r == some recordKeys do throw s!"record: keys are not {", ".intercalate recordKeys}"
  let commitment ← bytesOf r "commitment"
  let source ← bytesOf r "source"
  pure { coins, draws, record := { commitment, source } }

def commitmentOf (b : ByteArray) : V Commitment := do
  let j ← Core.canonicalJson b
  unless keys? j == some commitmentKeys do throw s!"keys are not {", ".intercalate commitmentKeys}"
  let some (.arr ms) := get? j "manifests" | throw "the retained passes' manifests are not a list"
  if ms.isEmpty then throw "the run holds no pass"
  let record ← bytesOf j "record"
  unless record.size == 64 do throw "record is not a SHA-512 digest"
  pure { manifests := ms.toList, record, bytes := b }

def Commitment.digest (c : Commitment) : ByteArray :=
  Flock.Sha512.hash (commitmentTag.toUTF8 ++ ByteArray.mk #[0] ++ c.bytes)

def evidenceOf (b : ByteArray) : V Evidence := do
  let j ← Core.canonicalJson b
  unless keys? j == some evidenceKeys do throw s!"keys are not {", ".intercalate evidenceKeys}"
  let some (.arr fs) := get? j "files" | throw "files is not a list"
  unless fs.all (keys? · |>.isSome) do throw "files is not a list of objects"
  pure { files := fs.toList }

def inputOf (p : Params) (c : Commitment) : Input :=
  { coins := p.coins, manifests := c.manifests, draws := p.draws, held := some p.record }

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
private def record : String :=
  "{\"commitment\":\"" ++ String.ofList (List.replicate 128 '0') ++ "\",\"source\":\"" ++
    String.ofList (List.replicate 64 '0') ++ "\"}"
private def reasonOf (p c e : ByteArray) : Option String := (protocol.answer p c e #[]).reason
private def run (manifests : String) : ByteArray :=
  ("{\"manifests\":" ++ manifests ++ ",\"record\":\"" ++ String.ofList (List.replicate 128 '0') ++ "\"}").toUTF8

-- every input gets a verdict: parameters, a commitment or evidence of another shape is refused, naming the input
#guard protocolId == "verity/pouw/pearl-plus/v1"
#guard reasonOf "{}".toUTF8 "[]".toUTF8 "{}".toUTF8 == some "params: keys are not coins, draws, record"
#guard reasonOf (params "{\"source\":\"00\"}") "[]".toUTF8 "{}".toUTF8 == some "params: record: keys are not commitment, source"
#guard reasonOf (params record) "[{}]".toUTF8 "{}".toUTF8 == some "commitment: keys are not manifests, record"
#guard reasonOf (params record) (run "[]") "{}".toUTF8 == some "commitment: the run holds no pass"
#guard reasonOf (params record) (run "{}") "{}".toUTF8 ==
  some "commitment: the retained passes' manifests are not a list"
#guard reasonOf (params record) "{\"manifests\":[{}],\"record\":\"00\"}".toUTF8 "{}".toUTF8 ==
  some "commitment: record is not a SHA-512 digest"
#guard reasonOf (params record) (run "[{}]") "{}".toUTF8 == some "evidence: keys are not files"
#guard reasonOf (params record) (run "[{}]") "{\"files\":[]}".toUTF8 == some "evidence: 0 passes' files for 1 passes"
-- well-formed inputs reach the run's verdict, whose refusal is the run's
#guard (reasonOf (params record) (run "[{}]") "{\"files\":[{}]}".toUTF8).isSome
#guard (protocol.answer (params record) (run "[{}]") "{\"files\":[{}]}".toUTF8 #[]).verdict == .reject

end PouwCheck.PearlPlus.Protocol
