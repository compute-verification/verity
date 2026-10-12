import Flock.Hash
import Flock.Bytes

/-!
# Streaming digests

A circuit file runs to hundreds of megabytes, so it is written line by line, never held whole: `Sha512S` and `Sha256S` take
it in pieces (`Flock.Sha512.compress` and `Flock.Sha256.compress` on each whole block) and give the digest of their
concatenation, as `Flock.Sha512.hash` and `Flock.Sha256.hash` of it would.
-/

namespace VStar

structure Sha512S where
  h : Array UInt64 := Flock.Sha512.iv
  buf : ByteArray := ByteArray.empty
  len : Nat := 0

def Sha512S.update (s : Sha512S) (b : ByteArray) : Sha512S := Id.run do
  let data := s.buf ++ b
  let n := data.size / 128
  let mut h := s.h
  for i in [0:n] do h := Flock.Sha512.compress h data (128 * i)
  return { h, buf := data.extract (128 * n) data.size, len := s.len + b.size }

/-- The digest: the buffered bytes, `0x80`, zeros and the 128-bit bit length, then the state big-endian. -/
def Sha512S.finish (s : Sha512S) : ByteArray := Id.run do
  let mut data := s.buf.push 0x80
  for _ in [0:(112 + 128 - data.size % 128) % 128] do data := data.push 0
  let bits := s.len * 8
  for i in [0:16] do data := data.push ((bits >>> (8 * (15 - i))) % 256).toUInt8
  let mut h := s.h
  for blk in [0:data.size / 128] do h := Flock.Sha512.compress h data (128 * blk)
  return ⟨Array.ofFn (n := 64) fun i => (h[i.val / 8]! >>> (8 * (7 - i.val % 8)).toUInt64).toUInt8⟩

structure Sha256S where
  h : Array UInt32 := Flock.Sha256.iv
  buf : ByteArray := ByteArray.empty
  len : Nat := 0

def Sha256S.update (s : Sha256S) (b : ByteArray) : Sha256S := Id.run do
  let data := s.buf ++ b
  let n := data.size / 64
  let mut h := s.h
  for i in [0:n] do h := Flock.Sha256.compress h data (64 * i)
  return { h, buf := data.extract (64 * n) data.size, len := s.len + b.size }

def Sha256S.finish (s : Sha256S) : ByteArray := Id.run do
  let mut data := s.buf.push 0x80
  for _ in [0:(56 + 64 - data.size % 64) % 64] do data := data.push 0
  let bits := s.len * 8
  for i in [0:8] do data := data.push ((bits >>> (8 * (7 - i))) % 256).toUInt8
  let mut h := s.h
  for blk in [0:data.size / 64] do h := Flock.Sha256.compress h data (64 * blk)
  return ⟨Array.ofFn (n := 32) fun i => (h[i.val / 4]! >>> (24 - 8 * (i.val % 4)).toUInt32).toUInt8⟩

/-- A text's SHA-512, SHA-256, length in bytes and newlines, taken piece by piece. -/
structure Digests where
  sha512 : Sha512S := {}
  sha256 : Sha256S := {}
  bytes : Nat := 0

def Digests.update (d : Digests) (s : String) : Digests :=
  let b := s.toUTF8
  { sha512 := d.sha512.update b, sha256 := d.sha256.update b, bytes := d.bytes + b.size }

/-- The pieces' digests as a pure computation: `put` each piece into the state. -/
abbrev Hashing := StateM Digests

def hashPut (s : String) : Hashing Unit := modify (·.update s)

end VStar
