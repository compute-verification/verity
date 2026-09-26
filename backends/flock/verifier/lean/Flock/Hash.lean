/-
PROTOCOL.md §2.5: SHA-256 (FIPS 180-4) and BLAKE3 (compression, hash, keyed hash).
-/
namespace Flock

abbrev Digest := ByteArray

namespace Sha256

def k : Array UInt32 := #[
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2]

def iv : Array UInt32 := #[0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]

@[inline] def rotr (x : UInt32) (n : UInt32) : UInt32 := (x >>> n) ||| (x <<< (32 - n))

/-- One compression of the 64-byte block of `data` at `off`. -/
def compress (h : Array UInt32) (data : ByteArray) (off : Nat) : Array UInt32 := Id.run do
  let mut w : Array UInt32 := Array.mkEmpty 64
  for i in [0:16] do
    let b := fun j => (data.get! (off + 4 * i + j)).toUInt32
    w := w.push ((b 0 <<< 24) ||| (b 1 <<< 16) ||| (b 2 <<< 8) ||| b 3)
  for i in [16:64] do
    let x := w[i - 15]!
    let y := w[i - 2]!
    let s0 := rotr x 7 ^^^ rotr x 18 ^^^ (x >>> 3)
    let s1 := rotr y 17 ^^^ rotr y 19 ^^^ (y >>> 10)
    w := w.push (w[i - 16]! + s0 + w[i - 7]! + s1)
  let mut a := h[0]!
  let mut b := h[1]!
  let mut c := h[2]!
  let mut d := h[3]!
  let mut e := h[4]!
  let mut f := h[5]!
  let mut g := h[6]!
  let mut hh := h[7]!
  for i in [0:64] do
    let s1 := rotr e 6 ^^^ rotr e 11 ^^^ rotr e 25
    let ch := (e &&& f) ^^^ ((~~~e) &&& g)
    let t1 := hh + s1 + ch + k[i]! + w[i]!
    let s0 := rotr a 2 ^^^ rotr a 13 ^^^ rotr a 22
    let maj := (a &&& b) ^^^ (a &&& c) ^^^ (b &&& c)
    let t2 := s0 + maj
    hh := g
    g := f
    f := e
    e := d + t1
    d := c
    c := b
    b := a
    a := t1 + t2
  return #[h[0]! + a, h[1]! + b, h[2]! + c, h[3]! + d, h[4]! + e, h[5]! + f, h[6]! + g, h[7]! + hh]

/-- SHA-256 of `msg`. -/
def hash (msg : ByteArray) : Digest := Id.run do
  let len := msg.size
  let mut data := msg
  data := data.push 0x80
  while data.size % 64 != 56 do
    data := data.push 0
  let bits := len * 8
  for i in [0:8] do
    data := data.push ((bits >>> (8 * (7 - i))) % 256).toUInt8
  let mut h := iv
  for blk in [0:data.size / 64] do
    h := compress h data (64 * blk)
  let mut out := ByteArray.emptyWithCapacity 32
  for x in h do
    out := out.push (x >>> 24).toUInt8 |>.push (x >>> 16).toUInt8 |>.push (x >>> 8).toUInt8 |>.push x.toUInt8
  return out

end Sha256

namespace Blake3

def CHUNK_START : UInt32 := 1
def CHUNK_END : UInt32 := 2
def PARENT : UInt32 := 4
def ROOT : UInt32 := 8
def KEYED_HASH : UInt32 := 16

def iv : Array UInt32 := Sha256.iv
def perm : Array Nat := #[2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8]

@[inline] def rotr (x : UInt32) (n : UInt32) : UInt32 := (x >>> n) ||| (x <<< (32 - n))

@[inline] def g (s : Array UInt32) (a b c d : Nat) (mx my : UInt32) : Array UInt32 := Id.run do
  let mut s := s
  s := s.set! a (s[a]! + s[b]! + mx)
  s := s.set! d (rotr (s[d]! ^^^ s[a]!) 16)
  s := s.set! c (s[c]! + s[d]!)
  s := s.set! b (rotr (s[b]! ^^^ s[c]!) 12)
  s := s.set! a (s[a]! + s[b]! + my)
  s := s.set! d (rotr (s[d]! ^^^ s[a]!) 8)
  s := s.set! c (s[c]! + s[d]!)
  s := s.set! b (rotr (s[b]! ^^^ s[c]!) 7)
  return s

/-- The BLAKE3 compression function: the 16-word output state. -/
def compress (cv : Array UInt32) (m : Array UInt32) (counter : UInt64) (blockLen : UInt32) (flags : UInt32) : Array UInt32 := Id.run do
  let mut s : Array UInt32 := #[cv[0]!, cv[1]!, cv[2]!, cv[3]!, cv[4]!, cv[5]!, cv[6]!, cv[7]!,
    iv[0]!, iv[1]!, iv[2]!, iv[3]!, counter.toUInt32, (counter >>> 32).toUInt32, blockLen, flags]
  let mut m := m
  for r in [0:7] do
    s := g s 0 4 8 12 m[0]! m[1]!
    s := g s 1 5 9 13 m[2]! m[3]!
    s := g s 2 6 10 14 m[4]! m[5]!
    s := g s 3 7 11 15 m[6]! m[7]!
    s := g s 0 5 10 15 m[8]! m[9]!
    s := g s 1 6 11 12 m[10]! m[11]!
    s := g s 2 7 8 13 m[12]! m[13]!
    s := g s 3 4 9 14 m[14]! m[15]!
    if r < 6 then
      m := perm.map (fun i => m[i]!)
  let mut out := s
  for i in [0:8] do
    out := out.set! i (s[i]! ^^^ s[i + 8]!)
    out := out.set! (i + 8) (s[i + 8]! ^^^ cv[i]!)
  return out

def wordsOfBytes (b : ByteArray) (off : Nat) (n : Nat) : Array UInt32 := Id.run do
  let mut w : Array UInt32 := Array.mkEmpty n
  for i in [0:n] do
    let at_ := off + 4 * i
    let byte := fun j => if at_ + j < b.size then (b.get! (at_ + j)).toUInt32 else 0
    w := w.push (byte 0 ||| (byte 1 <<< 8) ||| (byte 2 <<< 16) ||| (byte 3 <<< 24))
  return w

def bytesOfWords (w : Array UInt32) (n : Nat) : ByteArray := Id.run do
  let mut out := ByteArray.emptyWithCapacity (4 * n)
  for i in [0:n] do
    let x := w[i]!
    out := out.push x.toUInt8 |>.push (x >>> 8).toUInt8 |>.push (x >>> 16).toUInt8 |>.push (x >>> 24).toUInt8
  return out

/-- Streaming hasher state (the reference implementation's chunk state and chaining-value stack). -/
structure Hasher where
  key : Array UInt32
  flags : UInt32
  cv : Array UInt32
  chunkCounter : UInt64
  block : ByteArray
  blocksCompressed : Nat
  stack : Array (Array UInt32)
  deriving Inhabited

def Hasher.new (key : Array UInt32) (flags : UInt32) : Hasher :=
  ⟨key, flags, key, 0, ByteArray.empty, 0, #[]⟩

def Hasher.plain : Hasher := Hasher.new iv 0
def Hasher.keyed (key : ByteArray) : Hasher := Hasher.new (wordsOfBytes key 0 8) KEYED_HASH

private def startFlag (h : Hasher) : UInt32 := if h.blocksCompressed == 0 then CHUNK_START else 0

private def chunkCv (h : Hasher) : Array UInt32 :=
  (compress h.cv (wordsOfBytes h.block 0 16) h.chunkCounter h.block.size.toUInt32 (h.flags ||| startFlag h ||| CHUNK_END)).extract 0 8

private def parentCv (h : Hasher) (l r : Array UInt32) : Array UInt32 :=
  (compress h.key (l ++ r) 0 64 (h.flags ||| PARENT)).extract 0 8

/-- Absorb `input`. -/
def Hasher.update (h0 : Hasher) (input : ByteArray) : Hasher := Id.run do
  let mut h := h0
  let mut pos := 0
  while pos < input.size do
    if h.blocksCompressed * 64 + h.block.size == 1024 then
      -- the chunk is full and more input follows: close it and merge
      let cvc := chunkCv h
      let total := h.chunkCounter + 1
      let mut newCv := cvc
      let mut t := total
      let mut st := h.stack
      while t % 2 == 0 do
        newCv := parentCv h (st.back!) newCv
        st := st.pop
        t := t / 2
      h := { h with stack := st.push newCv, cv := h.key, chunkCounter := total, block := ByteArray.empty, blocksCompressed := 0 }
    if h.block.size == 64 then
      let out := compress h.cv (wordsOfBytes h.block 0 16) h.chunkCounter 64 (h.flags ||| startFlag h)
      h := { h with cv := out.extract 0 8, blocksCompressed := h.blocksCompressed + 1, block := ByteArray.empty }
    let chunkLeft := 1024 - (h.blocksCompressed * 64 + h.block.size)
    let take := min (min (64 - h.block.size) chunkLeft) (input.size - pos)
    h := { h with block := input.copySlice pos h.block h.block.size take }
    pos := pos + take
  return h

/-- The 32-byte root output. -/
def Hasher.finalize (h : Hasher) : Digest := Id.run do
  -- the current chunk's output, then the stack's parents from the top
  let mut inCv := h.cv
  let mut words := wordsOfBytes h.block 0 16
  let mut counter := h.chunkCounter
  let mut blen := h.block.size.toUInt32
  let mut flags := h.flags ||| startFlag h ||| CHUNK_END
  let mut i := h.stack.size
  while i > 0 do
    i := i - 1
    let cvOut := (compress inCv words counter blen flags).extract 0 8
    inCv := h.key
    words := h.stack[i]! ++ cvOut
    counter := 0
    blen := 64
    flags := h.flags ||| PARENT
  return bytesOfWords (compress inCv words 0 blen (flags ||| ROOT)) 8

def hash (b : ByteArray) : Digest := (Hasher.plain.update b).finalize
def keyedHash (key : ByteArray) (b : ByteArray) : Digest := ((Hasher.keyed key).update b).finalize

end Blake3

end Flock
