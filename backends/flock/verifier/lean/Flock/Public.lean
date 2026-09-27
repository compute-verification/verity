/-
PROTOCOL.md §16.5, §16.6: the verifier's public file (no rows) and frame-v3 hashing. The output roots are checked here;
the input ports' roots depend on the row leaf's schema and are the row leaf's check (`RowLeaf.checkPublic`).
-/
import Flock.Circuit

namespace Flock

open Lean (Json)

/-- The public file: its header, one row digest per instance and input port, the output words. -/
structure Public where
  header : Json
  n : Nat
  digests : Array ByteArray
  outputs : Array Nat
  /-- SHA-256 of the canonical header (`rows = false`) and every byte after it: what Σ binds -/
  sha : ByteArray
  deriving Inhabited

def Public.load (tags : Tags) (c : Circuit) (file : ByteArray) : V Public := do
  let some nl := findByte file 10 | throw "public file: no header line"
  let text ← match String.fromUTF8? (file.extract 0 nl) with
    | some s => pure s
    | none => throw "public file: header is not UTF-8"
  let header ← match Json.parse text with
    | .ok j => pure j
    | .error e => throw s!"public file: header: {e}"
  let ctx := "public file"
  if (← J.str header "format" ctx) != tags.publicFormat || (← J.str header "statement" ctx) != tags.statement then
    throw s!"public file: not a {tags.publicFormat} file"
  if (← J.str header tags.publicCircuitKey ctx) != hexOf c.sha then throw "public file: staged for another circuit"
  let n ← J.nat header "instances" ctx
  if (← J.bool header "rows" ctx) then throw "public file: it holds rows (the verifier's public file has none)"
  let np := c.ports.size
  let nOut := c.outWords
  let body := file.extract (nl + 1) file.size
  if body.size != 32 * n * np + 2 * n * nOut then throw "public file: length"
  let digests := (Array.range (n * np)).map fun i => body.extract (32 * i) (32 * i + 32)
  let o := 32 * n * np
  let outputs := (Array.range (n * nOut)).map fun i => (body.get! (o + 2 * i)).toNat + 256 * (body.get! (o + 2 * i + 1)).toNat
  let sha := Sha256.hash ((← canon header).toUTF8 ++ body)
  return { header, n, digests, outputs, sha }

def Public.digest (pub : Public) (i p np : Nat) : ByteArray := pub.digests.getD (i * np + p) ByteArray.empty

/- frame-v3 hashing (§16.6) -/
namespace FV3

def FRAME : ByteArray := "veritor/protocol/merkle/frame/v3".toUTF8.push 0

def hash (tag : String) (parts : List ByteArray) : ByteArray :=
  let t := tag.toUTF8
  Sha256.hash (parts.foldl (fun acc p => acc ++ be64 p.size.toUInt64 ++ p) (FRAME ++ be32 t.size.toUInt32 ++ t))

def uint (x : Nat) : ByteArray :=
  let b := be64 x.toUInt64
  let z := ((List.range 8).find? (fun i => b.get! i != 0)).getD 7
  b.extract z 8

def leaf (dom : ByteArray) (i : Nat) (schema : String) (payload : ByteArray) : ByteArray :=
  hash "leaf" [dom, uint i, uint i, schema.toUTF8, payload]

def root (dom : ByteArray) (leaves : Array ByteArray) : ByteArray := Id.run do
  let n := leaves.size
  let width := nextPow2 n
  let mut level := leaves ++ (Array.range (width - n)).map fun k => hash "pad" [dom, uint (n + k)]
  for depth in [0:log2ceil width] do
    level := (Array.range (level.size / 2)).map fun i => hash "node" [dom, uint depth, uint i, level[2 * i]!, level[2 * i + 1]!]
  return level.getD 0 ByteArray.empty

end FV3

namespace Public

def frameV3 (pub : Public) : V Json := J.get pub.header "frame_v3" "public file"

/-- `frame_v3.<k>.<port>` as bytes. -/
def fv3Bytes (pub : Public) (k port : String) : V ByteArray := do
  unhex (← J.str (← J.get (← pub.frameV3) k "public file") port s!"public file.frame_v3.{k}")

/-- The output roots: every output port's words (u16 big-endian leaves, instance-major) against the header's root. -/
def checkOutputs (c : Circuit) (pub : Public) : V Unit := do
  let nOut := c.outWords
  let mut at_ := 0
  for (name, w) in c.outPorts do
    let dom ← pub.fv3Bytes "domain_ids" name
    let words := (Array.range pub.n).foldl (fun acc i => acc ++ (Array.range w).map fun k => pub.outputs.getD (i * nOut + at_ + k) 0) #[]
    let leaves := (Array.range words.size).map fun r =>
      FV3.leaf dom r "u16" (ByteArray.mk #[(words[r]! / 256).toUInt8, (words[r]! % 256).toUInt8])
    if FV3.root dom leaves != (← pub.fv3Bytes "roots" name) then
      throw s!"output port {name}: the output words do not hash to the header's frame-v3 root"
    at_ := at_ + w

end Public

end Flock
