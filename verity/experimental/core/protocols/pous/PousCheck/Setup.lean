import PousCheck.Keys
import PousCheck.Commit

/-!
The public-encoder setup check (`setup.verifier_key_from_encoding`, `protocol.verifier_key(…, mode=PUBLIC)`): the verifier
decodes each segment of the server's `C*` with the keys of `Keys`, streams `Dec(C*)` into the commitment's framing
(`Commit.commitState`), and accepts iff the digest is the server's commitment to `W`. It never encodes and never holds `W`.

`decodeSegment` is `schemes/p2.py`'s `decode_segment`. The verification root over the blocks is not built here: `Tree`
takes the leaf and root functions as parameters (the verifier's leaves and its RFC 6962 tree), so the check is the same
whichever module supplies them.

The refusals come in Python's order: per segment, a segment past the parameters' count, a segment of another shape (block
count, a block's width), the salt's length (when the first segment's keys are expanded), and the first block refused by
`decodeBlock`; after the last segment, a missing segment, then the commitment.
-/

namespace PousCheck.Setup

open PousCheck.Prim

/-- Why setup refuses `C*`. -/
inductive Refusal where
  | tooMany (s : Nat)
  | shape (s : Nat)
  | salt (msg : String)
  | keys (msg : String)
  | block (s : Nat) (r : P2.Refusal)
  | count (n : Nat)
  | manifest (msg : String)
  | commitment
  deriving Repr, BEq

/-- The kind a refusal is reported as, the same word for Python's message (`scripts/setup_difftest.py`). -/
def Refusal.kind : Refusal → String
  | .tooMany _ | .count _ => "segments"
  | .shape _ => "shape"
  | .salt _ => "salt"
  | .keys _ => "keys"
  | .block _ r => r.kind
  | .manifest _ => "manifest"
  | .commitment => "commitment"

/-- Segment `s` of `W` from its blocks (`decode_segment`): the segment's index and block count, the salt (the keys'
prefix), then each block in order. -/
def decodeSegment (q : P2.Params) (salt : ByteArray) (s : Nat) (blocks : Array ByteArray) : Except Refusal ByteArray := do
  if s ≥ q.segments then throw (.tooMany s)
  if blocks.size != q.blocksPerSegment then throw (.shape s)
  let e := Keys.Expander.ofPrefix (← (Keys.packagePrefix q salt).mapError Refusal.salt)
  let mut out := ByteArray.emptyWithCapacity q.segmentBytes
  for h : j in [0:blocks.size] do
    let block := blocks[j]
    if block.size != q.blockBytes then throw (.block s (.width j))
    if Prim.natOfBE block ≥ q.p then throw (.block s (.noncanonical j))
    let (k, t, _) ← (e.keys q s j).mapError Refusal.keys
    out := out ++ (← (P2.decodeBlock q j k t block).mapError (Refusal.block s))
  return out

/-- The verifier's leaves of a segment's blocks, and its root over every leaf in block order. -/
structure Tree where
  leaves : P2.Params → ByteArray → Nat → Array ByteArray → Array ByteArray
  root : Array ByteArray → ByteArray

/-- `vk`: the root and the leaves. -/
structure VerifierKey where
  root : ByteArray
  leaves : Array ByteArray

/-- `verifier_key_from_encoding`: decode every segment of `C*`, check `commit(Dec(C*)) == commitment`, and build `vk` from
the blocks with `tree`. -/
def verifierKeyFromEncoding (q : P2.Params) (tree : Tree) (commitment salt : ByteArray)
    (encoded : Array (Array ByteArray)) : Except Refusal VerifierKey := do
  let mut h ← (Commit.commitState q).mapError Refusal.manifest
  let mut leaves : Array ByteArray := #[]
  for h' : s in [0:encoded.size] do
    let blocks := encoded[s]
    if s ≥ q.segments then throw (.tooMany s)
    if blocks.size != q.blocksPerSegment || blocks.any (·.size != q.blockBytes) then throw (.shape s)
    h := h.update (← decodeSegment q salt s blocks)
    leaves := leaves ++ tree.leaves q salt s blocks
  if encoded.size != q.segments then throw (.count encoded.size)
  if h.finalize != commitment then throw .commitment
  return ⟨tree.root leaves, leaves⟩

/-- Whether the decoded `C*` matches the commitment: the setup check with the tree left out. -/
def checkEncoding (q : P2.Params) (commitment salt : ByteArray) (encoded : Array (Array ByteArray)) :
    Except Refusal Unit := do
  let _ ← verifierKeyFromEncoding q ⟨fun _ _ _ _ => #[], fun _ => .empty⟩ commitment salt encoded
  return ()

end PousCheck.Setup
