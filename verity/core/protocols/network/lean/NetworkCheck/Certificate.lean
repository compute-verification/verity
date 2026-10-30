import Flock.Hash
import Flock.Bytes
import Flock.Canon
import Identity.Tagged

/-!
# The network certificate's encodings (record v2)

Executable forms of `verity/core/protocols/certifier/certificate.py`, whose module docstring is the spec: the
frame tree (`frameLeaf`, `node`, `empty`, `depth`, `framesRoot`, `openFrame`, `verifyFrame`) over a window's real frames
(`realFrames`), the parameters digest (`parametersDigest`, the manifest's identity bytes through `Flock.canonUtf8`), the record head's
encoding and digest (`RecordHead.encode`, `RecordHead.digest`, `recordHead`), the digest a certificate signs, the record
digest under the certifier's salt (`certificateDigest`), and the message it signs over it (`certificateMessage`,
`verity.ml.crypto.lms.certificate_message`). A Python `ValueError` is an `Except.error` here.
`tagged_sha512` is core's `Identity.taggedSha512`, which refuses what Python refuses. Every other SHA-512 is
`Flock.Sha512.hash`, and what each such call hashes has a name of its own (`leafInput`, `nodeInput`, `emptyInput`,
`certificateInput`, `messageInput`), as the properties (`Security/Properties/NetworkCertifier.lean`) name the strings a run hashes. The tree climbs by structural recursion (`pairs`, `climb`, `top`). `NetworkCheck.Difftest` holds these to the
Python on generated inputs.
-/

namespace NetworkCheck

open Flock

abbrev E := Except String

/-- `x` as `width` big-endian bytes, refused unless `0 ≤ x < 2^(8·width)` (Python's `_u`). -/
def uBE (width : Nat) (x : Int) (name : String) : E ByteArray :=
  if 0 ≤ x ∧ x < 2 ^ (8 * width) then
    pure ⟨(List.range width).toArray.map fun i => (x.toNat >>> (8 * (width - 1 - i))).toUInt8⟩
  else throw s!"{name} is an unsigned {8 * width}-bit integer, not {x}"

/-- `u16be len(s) ‖ s` of `s`'s UTF-8 (Python's `_text`). -/
def text (s : String) (name : String) : E ByteArray := do
  let b := s.toUTF8
  pure ((← uBE 2 b.size s!"len({name})") ++ b)

def frameTag : ByteArray := "verity-network-certifier/frame/v2".toUTF8.push 0
def emptyTag : ByteArray := "verity-network-certifier/empty/v2".toUTF8.push 0
def nodeTag : ByteArray := "verity-network-certifier/node/v2".toUTF8.push 0
def recordTag : String := "verity-network-certifier/record/v2"
def parametersTag : String := "verity-network-certifier/parameters/v1"
def digestBytes : Nat := 64

/-- `block(x)`: x zero-padded to one 128-byte SHA-512 block, refused past 128 bytes. -/
def block (x : ByteArray) : E ByteArray :=
  if x.size ≤ 128 then pure (x ++ ⟨Array.replicate (128 - x.size) 0⟩)
  else throw "a tag block is one 128-byte SHA-512 block"

/-- The grid parameters a record reads (`grid.GridParams`): T, r, B, τ_b and Q. -/
structure Params where
  buckets : Nat
  allocation : Nat
  frameBytes : Nat
  bucketMs : Nat
  queueCapacity : Nat

/-- What a leaf hashes, `block(FRAME_TAG ‖ u32be B) ‖ f`, refused unless `f` has B bytes. -/
def leafInput (frame : ByteArray) (frameBytes : Nat) : E ByteArray := do
  if frame.size ≠ frameBytes then throw s!"frames are {frameBytes} bytes, not {frame.size}"
  pure ((← block (frameTag ++ (← uBE 4 frameBytes "B"))) ++ frame)

/-- `leaf(f) = SHA-512(block(FRAME_TAG ‖ u32be B) ‖ f)`, refused unless `f` has B bytes. -/
def frameLeaf (frame : ByteArray) (frameBytes : Nat) : E ByteArray := Sha512.hash <$> leafInput frame frameBytes

def nodeBlock : ByteArray := nodeTag ++ ⟨Array.replicate (128 - nodeTag.size) 0⟩

/-- What a node hashes, `block(NODE_TAG) ‖ a ‖ b`. -/
def nodeInput (left right : ByteArray) : ByteArray := nodeBlock ++ left ++ right

/-- `node(a, b) = SHA-512(block(NODE_TAG) ‖ a ‖ b)`. -/
def node (left right : ByteArray) : ByteArray := Sha512.hash (nodeInput left right)

/-- What `E_0` hashes, `block(EMPTY_TAG)`. -/
def emptyInput : ByteArray := emptyTag ++ ⟨Array.replicate (128 - emptyTag.size) 0⟩

/-- `E_level`: `E_0 = SHA-512(block(EMPTY_TAG))`, `E_{l+1} = node(E_l, E_l)`. -/
def empty : Nat → ByteArray
  | 0 => Sha512.hash emptyInput
  | l + 1 => let e := empty l; node e e

/-- Python's `int.bit_length` on a natural number. -/
def bitLength (n : Nat) : Nat := if n = 0 then 0 else Nat.log2 n + 1

/-- `D = ⌈log2(T·r)⌉`, computed as Python does: `(T·r − 1).bit_length()`. -/
def depth (p : Params) : Nat := bitLength (p.buckets * p.allocation - 1)

/-- The real frames of rows of `allocation` slots, row by row, refused at the first row of another size or that isn't
left-packed. -/
def rowFrames (allocation : Nat) : List (Array (Option ByteArray)) → E (Array ByteArray)
  | [] => pure #[]
  | row :: rows => do
    if row.size ≠ allocation then throw s!"a row has {allocation} slots, not {row.size}"
    let n := (row.filter Option.isSome).size
    unless (row.extract 0 n).all Option.isSome do throw "a row that isn't left-packed has no encoding (W4)"
    pure (row.filterMap id ++ (← rowFrames allocation rows))

/-- The window's real frames in `grid.fill`'s order, refused unless the grid has T rows of r slots, each left-packed. -/
def realFrames (grid : Array (Array (Option ByteArray))) (p : Params) : E (Array ByteArray) := do
  if grid.size ≠ p.buckets then throw s!"a grid has {p.buckets} rows, not {grid.size}"
  rowFrames p.allocation grid.toList

/-- One level up from level `lv`: the node of each pair, the last of an odd level paired with `E_lv`. -/
def pairs (lv : Nat) : List ByteArray → List ByteArray
  | a :: b :: rest => node a b :: pairs lv rest
  | [a] => [node a (empty lv)]
  | [] => []

/-- The levels from `level`, at level `lv`, up `k` levels. -/
def climb : Nat → Nat → List ByteArray → List (List ByteArray)
  | _, 0, level => [level]
  | lv, k + 1, level => level :: climb (lv + 1) k (pairs lv level)

/-- The top of `climb`: the level `k` levels up from `level`, at level `lv`. -/
def top : Nat → Nat → List ByteArray → List ByteArray
  | _, 0, level => level
  | lv, k + 1, level => top (lv + 1) k (pairs lv level)

/-- The leaves of the window's frames, refused past T·r frames or at a frame of other than B bytes. -/
def leaves (frames : Array ByteArray) (p : Params) : E (Array ByteArray) := do
  if frames.size > p.buckets * p.allocation then throw "more frames than the window has slots"
  frames.mapM (frameLeaf · p.frameBytes)

/-- The tree's non-empty prefix at each level, leaves first; every node to its right is the level's `E`. -/
def levels (frames : Array ByteArray) (p : Params) : E (Array (Array ByteArray)) := do
  pure ((climb 0 (depth p) (← leaves frames p).toList).map List.toArray).toArray

def framesRoot (frames : Array ByteArray) (p : Params) : E ByteArray := do
  pure ((top 0 (depth p) (← leaves frames p).toList).headD (empty (depth p)))

/-- Frame k's authentication path: D siblings, leaf level first. -/
def openFrame (frames : Array ByteArray) (p : Params) (k : Int) : E (Array ByteArray) := do
  if ¬(0 ≤ k ∧ k < frames.size) then throw s!"frame {k} is not among the window's {frames.size}"
  let ls ← levels frames p
  let mut k := k.toNat
  let mut path := #[]
  for lv in [0:ls.size - 1] do
    let level := ls[lv]!
    let sibling := k ^^^ 1
    path := path.push (if h : sibling < level.size then level[sibling] else empty lv)
    k := k >>> 1
  pure path

/-- Whether `frame` is leaf k under `root`. -/
def verifyFrame (root : ByteArray) (p : Params) (k : Int) (frame : ByteArray) (path : Array ByteArray) : Bool :=
  if path.size ≠ depth p ∨ ¬(0 ≤ k ∧ k < 2 ^ path.size) ∨ frame.size ≠ p.frameBytes then false
  else match frameLeaf frame p.frameBytes with
    | .error _ => false
    | .ok leaf => Id.run do
      let mut d := leaf
      let mut k := k.toNat
      for sibling in path do
        d := if k % 2 = 0 then node d sibling else node sibling d
        k := k >>> 1
      return d == root

/-- Σ_sync as the manifest reads it: its deltas, and its ρ as (numerator, denominator) in lowest terms. -/
structure SyncSet where
  deltas : Array Int
  rhos : Array (Int × Nat)

/-- `str(Fraction)`: `n` when the denominator is 1, else `n/d`. -/
def rhoText (r : Int × Nat) : String := if r.2 = 1 then toString r.1 else s!"{r.1}/{r.2}"

/-- The manifest `parameters_digest` hashes. -/
def parametersManifest (p : Params) (ss : Option SyncSet) : Lean.Json :=
  Lean.Json.mkObj [
    ("buckets", Lean.toJson p.buckets), ("allocation", Lean.toJson p.allocation),
    ("frame_bytes", Lean.toJson p.frameBytes), ("bucket_ms", Lean.toJson p.bucketMs),
    ("queue_capacity", Lean.toJson p.queueCapacity),
    ("sync_set", match ss with
      | none => .null
      | some s => Lean.Json.mkObj [
          ("deltas", .arr (s.deltas.map Lean.toJson)),
          ("rhos", .arr (s.rhos.map fun r => .str (rhoText r)))])]

/-- `parameters_digest`: `tagged_sha512(PARAMETERS_TAG, canonical_json_bytes(manifest))`. -/
def parametersDigest (p : Params) (ss : Option SyncSet) : E ByteArray := do
  Identity.taggedSha512 parametersTag (← Flock.canonUtf8 (parametersManifest p ss)).toUTF8

/-- `Direction`, by its value; its code is 0 for egress and 1 for ingress. -/
inductive Direction where
  | egress
  | ingress
  deriving DecidableEq, Inhabited

def Direction.value : Direction → String
  | .egress => "egress"
  | .ingress => "ingress"

def Direction.code : Direction → Nat
  | .egress => 0
  | .ingress => 1

def Direction.ofValue : String → E Direction
  | "egress" => pure .egress
  | "ingress" => pure .ingress
  | s => throw s!"not a direction: {s}"

/-- `STATUS_CODES`: complete (`none`) is 0, then each `Reason` by its value. -/
def statusCode : Option String → E Nat
  | none => pure 0
  | some "monitor-unavailable" => pure 1
  | some "overflow" => pure 2
  | some "malformed" => pure 3
  | some "dropped" => pure 4
  | some "delivery-refused" => pure 5
  | some "coverage-never-established" => pure 6
  | some s => throw s!"not a reason: {s}"

/-- `RecordHead`: a v2 record with its frames replaced by their root. Integers are `Int` so that a field Python would
refuse as out of range is refused here too. -/
structure RecordHead where
  link : String
  direction : Direction
  window : Int
  buckets : Int
  allocation : Int
  frameBytes : Int
  bucketMs : Int
  status : Option String
  deadline : Bool
  parameters : ByteArray
  counts : Array Int
  root : ByteArray

/-- `RecordHead.__post_init__`'s checks, then `encode`'s fields in order. -/
def RecordHead.encode (h : RecordHead) : E ByteArray := do
  if h.counts.size ≠ h.buckets ∨ h.counts.any (fun n => ¬(0 ≤ n ∧ n ≤ h.allocation)) then
    throw s!"a record has {h.buckets} counts of 0 … {h.allocation}"
  if h.parameters.size ≠ digestBytes ∨ h.root.size ≠ digestBytes then
    throw "the parameters digest and the root are 64 bytes"
  let mut out := (← text h.link "link") ++ (← uBE 1 h.direction.code "direction") ++ (← uBE 8 h.window "window")
    ++ (← uBE 4 h.buckets "T") ++ (← uBE 4 h.allocation "r") ++ (← uBE 4 h.frameBytes "B")
    ++ (← uBE 4 h.bucketMs "τ_b") ++ (← uBE 1 (← statusCode h.status) "status")
    ++ (← uBE 1 (if h.deadline then 1 else 0) "deadline") ++ h.parameters
  for n in h.counts do out := out ++ (← uBE 4 n "count")
  pure (out ++ h.root)

/-- The record digest, `tagged_sha512(RECORD_TAG, R)`. -/
def RecordHead.digest (h : RecordHead) : E ByteArray := do Identity.taggedSha512 recordTag (← h.encode)

/-- `record_head`: the head of a window's grid, its counts the occupancy and its root over the real frames. -/
def recordHead (link : String) (direction : Direction) (window : Int) (grid : Array (Array (Option ByteArray)))
    (status : Option String) (p : Params) (parameters : ByteArray) (deadline : Bool) : E RecordHead := do
  let frames ← realFrames grid p
  pure { link, direction, window, buckets := p.buckets, allocation := p.allocation, frameBytes := p.frameBytes,
         bucketMs := p.bucketMs, status, deadline, parameters,
         counts := grid.map fun row => ((row.filter Option.isSome).size : Int),
         root := ← framesRoot frames p }

/-- The signed digest's tag, `CERTIFICATE_TAG`. -/
def certificateTag : ByteArray := "verity/net-cert/v1".toUTF8

/-- The length of the certifier's salt for a record (`SALT_BYTES`). -/
def saltBytes : Nat := 64

/-- What `certificate_digest` hashes: `"verity/net-cert/v1" ‖ salt_c ‖ R`. -/
def certificateInput (salt record : ByteArray) : ByteArray := certificateTag ++ salt ++ record

/-- `certificate_digest`: the digest a certificate signs, the record digest `R` hidden under `salt_c`, the 64 bytes the
certifier drew for the record. -/
def certificateDigest (salt record : ByteArray) : ByteArray := Sha512.hash (certificateInput salt record)

/-- `[x] = u64be len(x) ‖ x` (`lms._field`). -/
def field (b : ByteArray) : ByteArray := be64 b.size.toUInt64 ++ b

/-- The message's domain-separation tag, LMS's network-certificate role tag (`lms.ROLES`): the message's own, distinct
from the ids of the checks that read it (`verity/network-certificate/v4`, `verity/network-records/v3`). -/
def messageTag : String := "verity/network-certificate-message/v2"

/-- The length of a certifier session id (`lms.SESSION_BYTES`). -/
def sessionBytes : Nat := 32

/-- What `certificate_message` hashes:
`[tag] ‖ I ‖ [[session] ‖ [link] ‖ [direction] ‖ u64be window] ‖ [signed digest]`, the tag `messageTag`
(`verity/network-certificate-message/v2`, LMS's network-certificate role tag, the message's own and distinct from the ids
of the checks that read it, `verity/network-certificate/v4` and `verity/network-records/v3`); refused unless the digest
is 64 bytes, the window a u64, the session 32 bytes and I 16. -/
def messageInput (keyId session : ByteArray) (link : String) (direction : Direction) (window : Int)
    (digest : ByteArray) : E ByteArray := do
  if digest.size ≠ digestBytes then throw "a signed digest is 64 bytes"
  let w ← uBE 8 window "window"
  if session.size ≠ sessionBytes then throw s!"a certifier session id is 32 bytes, not {session.size}"
  if keyId.size ≠ 16 then throw s!"an LMS identifier is 16 bytes, not {keyId.size}"
  let context := field session ++ field link.toUTF8 ++ field direction.value.toUTF8 ++ w
  pure (field messageTag.toUTF8 ++ keyId ++ field context ++ field digest)

/-- `certificate_message` (`certificate.py` and `verity.ml.crypto.lms`): SHA-512 of `messageInput`,
`[tag] ‖ I ‖ [[session] ‖ [link] ‖ [direction] ‖ u64be window] ‖ [signed digest]` with the tag
`verity/network-certificate-message/v2`; refused unless the digest is 64 bytes, the window a u64, the session 32 bytes
and I 16. -/
def certificateMessage (keyId session : ByteArray) (link : String) (direction : Direction) (window : Int)
    (digest : ByteArray) : E ByteArray :=
  Sha512.hash <$> messageInput keyId session link direction window digest

end NetworkCheck
