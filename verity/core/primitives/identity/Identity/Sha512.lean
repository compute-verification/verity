import Flock.Bytes
import Flock.Hash

/-!
Streaming SHA-512 over Flock's compression (`Flock.Sha512.compress`, the only one): absorb bytes in any pieces, then
finalize. A state is a midstate: a prefix of whole 128-byte blocks absorbed once can be extended many times (P2's key
expander), and a long message can be absorbed a piece at a time (a commitment streamed segment by segment).
`Flock.Sha512.hash m` is `(init.update m).finalize`.
-/

namespace Identity.Sha512

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

-- FIPS 180-4's "abc", and a message absorbed in pieces across a block boundary
#guard Flock.hexOf (hash "abc".toUTF8) ==
  "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f"
#guard let m := ByteArray.mk ((Array.range 300).map (·.toUInt8))
  ((init.update (m.extract 0 7)).update (m.extract 7 300)).finalize == Flock.Sha512.hash m

end Identity.Sha512
