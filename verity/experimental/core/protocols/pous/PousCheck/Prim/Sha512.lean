import Flock.Hash

/-!
Streaming SHA-512 over Flock's compression (`Flock.Sha512.compress`): absorb bytes in any pieces, then finalize. P2 needs
it twice: the key prefix's midstate, shared by every chunk of a setup (`p2/PROTOCOL.md` §4: the prefix is whole 128-byte
blocks), and the commitment to `W`, streamed one segment at a time as `setup.commit` does. `Flock.Sha512.hash m` is
`(init.update m).finalize`; nothing here is P2's, so a shared primitives package can take the module over as it is.
-/

namespace PousCheck.Prim.Sha512

/-- The chaining value after `len` bytes, of which the last `buf.size < 128` are not yet compressed. -/
structure State where
  h : Array UInt64
  buf : ByteArray
  len : Nat

def init : State := ⟨Flock.Sha512.iv, .empty, 0⟩

/-- Absorb `data`: complete the pending block, compress every whole block of `data` in place, keep the rest. -/
def State.update (s : State) (data : ByteArray) : State := Id.run do
  let mut h := s.h
  let mut buf := s.buf
  let mut off := 0
  if buf.size > 0 then
    let take := min (128 - buf.size) data.size
    buf := buf ++ data.extract 0 take
    off := take
    if buf.size == 128 then
      h := Flock.Sha512.compress h buf 0
      buf := .empty
  if buf.size == 0 then
    let blocks := (data.size - off) / 128
    for b in [0:blocks] do
      h := Flock.Sha512.compress h data (off + 128 * b)
    off := off + 128 * blocks
    buf := data.extract off data.size
  return ⟨h, buf, s.len + data.size⟩

/-- What `finalize` compresses after the whole blocks: the pending bytes then FIPS 180-4's padding (`0x80`, zeros, the
128-bit bit length), one block or two. -/
def State.padding (s : State) : ByteArray := Id.run do
  let mut data := s.buf.push 0x80
  data := data ++ ByteArray.mk (Array.replicate ((112 + 128 - data.size % 128) % 128) 0)
  let bits := s.len * 8
  for i in [0:16] do
    data := data.push ((bits >>> (8 * (15 - i))) % 256).toUInt8
  return data

/-- The chaining value as the digest: its eight words, big-endian. -/
def digest (h : Array UInt64) : ByteArray :=
  ⟨Array.ofFn (n := 64) fun i => (h[i.val / 8]! >>> (8 * (7 - i.val % 8)).toUInt64).toUInt8⟩

/-- The digest of everything absorbed. -/
def State.finalize (s : State) : ByteArray := Id.run do
  let data := s.padding
  let mut h := s.h
  for blk in [0:data.size / 128] do
    h := Flock.Sha512.compress h data (128 * blk)
  return digest h

def hash (msg : ByteArray) : ByteArray := (init.update msg).finalize

end PousCheck.Prim.Sha512
