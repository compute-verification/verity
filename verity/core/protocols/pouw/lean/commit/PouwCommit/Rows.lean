import Flock.Hm96
import Flock.HmRow
import PouwCommit.Frame
import PouwCommit.Bindings
import Identity.Domains

/-!
# The protocol's row and tile trees (`audit.rows_domain`, `Hm96Domain`, `commit_rows`, `transcript_domain`)

A scheme's trees are frame-v3 trees under its `commitment_hash`: `sha256` (the default), `sha512`, or `hm96-sha512`, which is
`frame-v3-sha512` with hm96-sha512 leaves. Under `hm96-sha512` a committed value is its preimage followed by a 192-byte salt
`y`; `x` is the preimage's `sha512/row-seg/v1` digest (a row) or, in a tile tree (`digests`), the 64-byte preimage itself;
and the frame leaf is the one over `hm96.SHA512.tree_leaf(b ‖ c)` under the schema `hm96-sha512/row-seg/v1`. Elsewhere a
leaf is the frame's plain leaf over the value under the tree's schema (`verity/pouw/row/v1`, `verity/pouw/tile/v1`).

`commit_rows` draws the hm96 salts from the prover's 32-byte key: leaf `i`'s salt is ChaCha20 blocks `3i, 3i+1, 3i+2` under
the key, with a 64-bit counter and the tree's 64-bit nonce (`hm96_salt`).
-/

namespace PouwCommit

open Flock

def ROW_SCHEMA : String := Domains.POUW_AUDIT_ROW
def TILE_SCHEMA : String := Domains.POUW_AUDIT_TILE
def HM96_SALT : Nat := 192

private def rotl (x : UInt32) (n : UInt32) : UInt32 := (x <<< n) ||| (x >>> (32 - n))

private def u32le (b : ByteArray) (o : Nat) : UInt32 :=
  (b.get! o).toUInt32 ||| ((b.get! (o + 1)).toUInt32 <<< 8) ||| ((b.get! (o + 2)).toUInt32 <<< 16) |||
    ((b.get! (o + 3)).toUInt32 <<< 24)

/-- ChaCha20's block function, 20 rounds, a 64-bit counter in words 12, 13 and a 64-bit nonce in words 14, 15
(`audit.chacha20_block`). -/
def chacha20Block (key : ByteArray) (counter nonce : Nat) : ByteArray := Id.run do
  let s : Array UInt32 := #[0x61707865, 0x3320646E, 0x79622D32, 0x6B206574] ++ (Array.range 8).map (u32le key <| 4 * ·) ++
    #[(counter % 2 ^ 32).toUInt32, (counter / 2 ^ 32 % 2 ^ 32).toUInt32, (nonce % 2 ^ 32).toUInt32,
      (nonce / 2 ^ 32 % 2 ^ 32).toUInt32]
  let mut x := s
  let qr (x : Array UInt32) (a b c d : Nat) : Array UInt32 := Id.run do
    let mut x := x
    for (p, q, r, n) in [(a, b, d, 16), (c, d, b, 12), (a, b, d, 8), (c, d, b, 7)] do
      x := x.set! p (x[p]! + x[q]!)
      x := x.set! r (rotl (x[r]! ^^^ x[p]!) n.toUInt32)
    return x
  for _ in [0:10] do
    x := qr x 0 4 8 12; x := qr x 1 5 9 13; x := qr x 2 6 10 14; x := qr x 3 7 11 15
    x := qr x 0 5 10 15; x := qr x 1 6 11 12; x := qr x 2 7 8 13; x := qr x 3 4 9 14
  let mut out := ByteArray.emptyWithCapacity 64
  for i in [0:16] do
    let w := x[i]! + s[i]!
    for j in [0:4] do out := out.push (w >>> (8 * j).toUInt32).toUInt8
  return out

/-- Leaf `i`'s 192-byte salt in the tree of salt nonce `nonce` (`audit.hm96_salt`). -/
def hm96Salt (key : ByteArray) (nonce i : Nat) : ByteArray :=
  chacha20Block key (3 * i) nonce ++ chacha20Block key (3 * i + 1) nonce ++ chacha20Block key (3 * i + 2) nonce

/-- A tree's frame and leaves: a plain frame-v3 tree, or `hm96-sha512`'s, of rows or (`digests`) of tile digests. -/
structure RowsDomain where
  frame : Domain
  hm96 : Bool
  digests : Bool

/-- `rows_domain(binding, owner, count, hash, digests)`, `hash` one of `sha256`, `sha512`, `hm96-sha512`. -/
def rowsDomain (binding : ByteArray) (owner : Int) (count : Nat) (hash : String) (digests : Bool) :
    Except String RowsDomain := do
  match hash with
  | "hm96-sha512" => pure ⟨← Domain.mk? .sha512 binding owner count, true, digests⟩
  | "sha256" => pure ⟨← Domain.mk? .sha256 binding owner count, false, digests⟩
  | "sha512" => pure ⟨← Domain.mk? .sha512 binding owner count, false, digests⟩
  | h => throw s!"the frame hash {h} is not one of sha256, sha512, hm96-sha512"

namespace RowsDomain

/-- The frame leaf over `value` at `rank` (`Hm96Domain.leaf` under `hm96-sha512`, which does not read `schema`). -/
def leaf (d : RowsDomain) (rank : Nat) (schema : String) (value : ByteArray) : Except String ByteArray := do
  if !d.hm96 then return d.frame.leaf rank schema value
  if value.size < HM96_SALT || (d.digests && value.size != 64 + HM96_SALT) then
    throw "an hm96-sha512 value is its preimage and a 192-byte salt"
  let pre := value.extract 0 (value.size - HM96_SALT)
  let y := value.extract (value.size - HM96_SALT) value.size
  let x := if d.digests then pre else HmRow.segDigest (8 * pre.size) pre
  return d.frame.leaf rank HmRow.ROW_SCHEMA_SEG (Hm96.Default512.leaf x y)

/-- The tree's root over its values in rank order, under `schema`. -/
def root (d : RowsDomain) (schema : String) (values : Array ByteArray) : Except String ByteArray := do
  let leaves ← (Array.range values.size).mapM fun i => d.leaf i schema values[i]!
  match d.frame.root leaves with
  | some r => pure r
  | none => throw s!"expected values for {d.frame.count} positions, got {values.size}"

/-- `verify_opening` of `value` at `rank` against `root`, under `schema`. -/
def verify (d : RowsDomain) (root : ByteArray) (rank : Nat) (schema : String) (value : ByteArray)
    (path : List ByteArray) : Except String Bool :=
  d.frame.verify root rank value path (d.leaf rank schema)

end RowsDomain

/-- `commit_rows(binding, owner, rows, hash, salt_key, nonce, digests).commitment.root`: under `hm96-sha512` each row is
followed by its salt `hm96Salt saltKey nonce i`, and the key is 32 bytes. A nonce of 2⁶⁴ or more, which Python's ChaCha20
words would carry past 32 bits, is refused. -/
def commitRows (binding : ByteArray) (owner : Int) (rows : Array ByteArray) (hash : String) (saltKey : Option ByteArray)
    (nonce : Nat) (digests : Bool) : Except String ByteArray := do
  let values ← if hash == "hm96-sha512" then
      match saltKey with
      | none => throw "an hm96-sha512 tree takes the prover's salt key"
      | some k =>
        if !rows.isEmpty && k.size != 32 then throw "the salt key is 32 bytes"
        if !rows.isEmpty && nonce ≥ 2 ^ 64 then throw "the salt nonce is a u64"
        pure ((Array.range rows.size).map fun i => rows[i]! ++ hm96Salt k nonce i)
    else pure rows
  (← rowsDomain binding owner rows.size hash digests).root ROW_SCHEMA values

/-- `transcript_domain(epoch, count, hash)`: the tile digests' tree, bound to the epoch's salt and weights' root. -/
def transcriptDomain (salt weightsRoot : ByteArray) (count : Nat) (hash : String) : Except String RowsDomain :=
  rowsDomain (transcriptBinding salt weightsRoot) (-1) count hash true

/-- The transcript's root over the tiles' digests in order (`Prover.prove`): under `hm96-sha512` digest `i` is followed by
`hm96Salt saltKey 2⁶³ i`. -/
def transcriptRoot (salt weightsRoot : ByteArray) (digests : Array ByteArray) (hash : String) (saltKey : ByteArray) :
    Except String ByteArray := do
  let values ← if hash == "hm96-sha512" then
      if !digests.isEmpty && saltKey.size != 32 then throw "the salt key is 32 bytes"
      pure ((Array.range digests.size).map fun i => digests[i]! ++ hm96Salt saltKey (2 ^ 63) i)
    else pure digests
  (← transcriptDomain salt weightsRoot digests.size hash).root TILE_SCHEMA values

end PouwCommit
