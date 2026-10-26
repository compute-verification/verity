import RemoteAudit.Window

/-!
# The openings frame: `verity/core/protocols/remote_bandwidth/wire.py`'s `decode_openings`

The bytes a prover sends a window's openings in, decoded strictly: `none` for a body that ends early, runs past its
openings, or has a bound flag above 1. Every opening carries no tile.

~~~text
body      u32be n ‖ opening × n
opening   u32be pass ‖ u64be position ‖ block (4096) ‖ u16be len(leaf) ‖ leaf ‖ leaf salt (192) ‖ payload salt (192)
          ‖ u8 d ‖ d × 64 (receipt path) ‖ u8 d ‖ d × 64 (payload path) ‖ u8 bound (0 or 1) ‖ bound × 64 (d_j)
~~~
-/

namespace RemoteAudit.Wire

/-- A read position in a body. -/
structure Cursor where
  body : ByteArray
  pos : Nat

/-- The big-endian integer of the next `n` bytes. -/
def Cursor.int (c : Cursor) (n : Nat) : Option (Nat × Cursor) :=
  if c.pos + n ≤ c.body.size then
    some ((List.range n).foldl (fun acc k => acc * 256 + (c.body.get! (c.pos + k)).toNat) 0, { c with pos := c.pos + n })
  else none

def Cursor.bytes (c : Cursor) (n : Nat) : Option (ByteArray × Cursor) :=
  if c.pos + n ≤ c.body.size then some (c.body.extract c.pos (c.pos + n), { c with pos := c.pos + n }) else none

/-- `d` siblings of 64 bytes. -/
def Cursor.path (c : Cursor) (d : Nat) : Option (Array ByteArray × Cursor) := Id.run do
  let mut c := c
  let mut out := #[]
  for _ in [0:d] do
    match c.bytes DIGEST_BYTES with
    | some (x, c') => out := out.push x; c := c'
    | none => return none
  return some (out, c)

def Cursor.opening (c : Cursor) : Option (Opening × Cursor) := do
  let (i, c) ← c.int 4
  let (p, c) ← c.int 8
  let (block, c) ← c.bytes Codec.BLOCK_BYTES
  let (nl, c) ← c.int 2
  let (leaf, c) ← c.bytes nl
  let (leafSalt, c) ← c.bytes SALT_BYTES
  let (payloadSalt, c) ← c.bytes SALT_BYTES
  let (d, c) ← c.int 1
  let (receiptPath, c) ← c.path d
  let (d, c) ← c.int 1
  let (payloadPath, c) ← c.path d
  let (bound, c) ← c.int 1
  guard (bound ≤ 1)
  let (digest, c) ← c.bytes (DIGEST_BYTES * bound)
  pure ({ pass := i, position := p, block, leaf, leafSalt, receiptPath, payloadSalt, payloadPath,
          tile := ByteArray.empty, digest }, c)

/-- `decode_openings(body)`. -/
def decodeOpenings (body : ByteArray) : Option (Array Opening) := Id.run do
  let some (n, c) := Cursor.int ⟨body, 0⟩ 4 | return none
  let mut c := c
  let mut out := #[]
  for _ in [0:n] do
    match c.opening with
    | some (o, c') => out := out.push o; c := c'
    | none => return none
  return if c.pos == body.size then some out else none

/-- A window's openings from its frames, in the order they came: every frame's openings in turn, or none at all when
any frame doesn't decode, so that the window is rejected as malformed (`checkWindow` on a window with draws,
`Session.verify` on one without). The verifier's process reads a window's frames so (`Verifier.windowCheck`), and the
difftest a transcript's (`Difftest.eventOf`'s `frames`), as `wire.openings_of` does. -/
def openingsOf (frames : Array ByteArray) : Array Opening :=
  ((frames.mapM decodeOpenings).map (·.flatten)).getD #[]

end RemoteAudit.Wire
