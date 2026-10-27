/-
Blobs: content-addressed bytes a check reads beside its three byte strings (a statement's circuit file, its public file,
each session's record and proofs), which are too large for canonical JSON.

* `Blobs`: the caller's bytes, each held under the SHA-512 the verifier computes itself, so a blob is found only by the
  digest of its own bytes: a lookup by a digest the parameters or the record name is the check that the bytes are the
  committed ones;
* `BlobProtocol`: a `Protocol` whose check also reads blobs (`checkBlobs`). Its `toProtocol` is the protocol without
  them, the same check on no blobs, so a composition of `Protocol`s takes it unchanged;
* `BlobProtocol.answer`: `Protocol.answer` with `checkBlobs` on the caller's blobs in place of `check`;
* `BlobProtocol.main`: `Protocol.main`'s command line, whose `check` takes the blobs' files after the evidence and whose
  `serve` case may carry `"blobs": [hex, ...]`.

A protocol without blobs needs none of this: `Protocol` and `Protocol.answer` are unchanged.
-/
import CoreProtocol.Cli
import Flock.Hash

namespace Core

open Lean (Json)
open Flock (V)

/-- The caller's blobs, each under the SHA-512 of its bytes, which `ofArray` computes. -/
structure Blobs where
  entries : Array (ByteArray × ByteArray)

def Blobs.ofArray (bs : Array ByteArray) : Blobs := ⟨bs.map fun b => (Flock.Sha512.hash b, b)⟩

def Blobs.empty : Blobs := ⟨#[]⟩

/-- The blob whose SHA-512 is `digest` (lowercase hex of 64 bytes), or a refusal naming `what`. -/
def Blobs.get (bs : Blobs) (digest : String) (what : String) : V ByteArray := do
  let d ← Flock.unhex digest
  if d.size != 64 || Flock.hexOf d != digest then throw s!"{what}: {digest} is not a SHA-512 digest (128 lowercase hex)"
  match bs.entries.find? (·.1 == d) with
  | some (_, b) => pure b
  | none => throw s!"{what}: no blob has the SHA-512 {digest}"

structure BlobProtocol extends Protocol where
  checkBlobs : Params → Commitment → Evidence → Blobs → Outcome Public Profile

/-- The protocol on the caller's blobs: `Protocol.answer` with `checkBlobs` on them. -/
def BlobProtocol.on (P : BlobProtocol) (blobs : Array ByteArray) : Protocol :=
  { P.toProtocol with check := fun p c e => P.checkBlobs p c e (Blobs.ofArray blobs) }

def BlobProtocol.answer (P : BlobProtocol) (params commitment evidence : ByteArray) (blobs : Array ByteArray) : Answer :=
  (P.on blobs).answer params commitment evidence

/-- One `serve` case: `Protocol.case`'s, with the blobs as hex under `blobs` (absent: none). -/
def BlobProtocol.case (P : BlobProtocol) (line : String) : Answer :=
  let split : V (Array ByteArray × String) := do
    let j ← canonicalJson line.toUTF8
    let .obj kvs := j | throw "the case is not an object"
    let bs ← match j.getObjVal? "blobs" with
      | .error _ => pure #[]
      | .ok (.arr bs) => bs.mapM fun b => do Flock.unhex (← Flock.J.asStr b "the case's blobs")
      | .ok _ => throw "the case's blobs are not a list of hex strings"
    pure (bs, ← Flock.canon (Json.mkObj (kvs.toArray.filter (·.1 != "blobs")).toList))
  match split with
  | .ok (bs, rest) => (P.on bs).case rest
  | .error e =>
    { protocol := P.id, verdict := .reject, reason := some s!"case: {e}", digest := none, published := none,
      profile := none }

private def exitOf : Verdict → UInt32
  | .accept => 0
  | .reject => 1
  | .undecided => 2

partial def BlobProtocol.serveLoop (P : BlobProtocol) (stdin : IO.FS.Stream) : IO UInt32 := do
  let line ← stdin.getLine
  if line.isEmpty then return 0
  let line := (line.dropEndWhile (· == '\n')).toString
  match (P.case line).canonical with
  | .ok out =>
    let o ← IO.getStdout
    o.putStrLn out
    o.flush
    P.serveLoop stdin
  | .error e =>
    IO.eprintln s!"{P.id}: the outcome is not canonical JSON: {e}"
    return 70

def BlobProtocol.main (P : BlobProtocol) (args : List String) : IO UInt32 := do
  match args with
  | ["id"] => IO.println P.id; return 0
  | "check" :: params :: commitment :: evidence :: blobs =>
    let bs ← blobs.toArray.mapM fun (f : String) => IO.FS.readBinFile f
    let a := P.answer (← IO.FS.readBinFile params) (← IO.FS.readBinFile commitment) (← IO.FS.readBinFile evidence) bs
    match a.canonical with
    | .ok line =>
      let out ← IO.getStdout
      out.putStrLn line
      out.flush
      return exitOf a.verdict
    | .error e =>
      IO.eprintln s!"{P.id}: the outcome is not canonical JSON: {e}"
      return 70
  | ["serve"] => P.serveLoop (← IO.getStdin)
  | _ =>
    IO.eprintln s!"usage: {P.id} id | check PARAMS COMMITMENT EVIDENCE [BLOB ...] | serve"
    return 64

end Core
