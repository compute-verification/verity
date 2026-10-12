import PouwWindow.Entry
import Identity.Tagged
import PouwCommit.Py

/-!
# The side trees (`window.py`, "The side trees")

Each side of an entry is one frame-v3-sha512 tree of 2^D leaves over the rows its tiles keep, in order, the rest
frame-v3's pad leaves, under the window's side domain: `binding = identity_digest_sha512(SIDE_DOMAIN, {window, side,
depth})`, owner −1, positions `0 … 2^D − 1`. A side row is its operand row's u16 words, then its place in the call as a u32
in one 128-bit word (`sideRow?`, `sideIndex?`). `sideLevels?` builds a tree's levels over its first leaves and `sidePath`
reads a leaf's siblings from them, each node past them a node over pad leaves only (`padNode`).
-/

namespace PouwWindow

open Lean (Json)
open PouwCommit (Domain)

def SIDE_DOMAIN : String := "verity/pouw/window/side-domain/v1"
/-- `OWNER`: the frame's owner, as C-Flock's statement domains have it. -/
def owner : Int := -1
/-- `SIDE_INDEX_WORDS`: a side row's index field, one 128-bit word. -/
def sideIndexWords : Nat := 8

/-- `side_row`: the operand row's words, then the index field: the u32 index as two u16 words, low first, and six zeros. -/
def sideRow? (words : List Int) (index : Int) : Except String (List Int) := do
  if words.length % sideIndexWords != 0 then
    throw s!"a side row's operand words are whole 128-bit words, not {words.length} u16 words"
  if !(0 ≤ index && index < 2 ^ 32) then throw s!"a side row's index is a u32, not {index}"
  pure (words ++ [index % 2 ^ 16, index / 2 ^ 16] ++ List.replicate (sideIndexWords - 2) 0)

/-- `side_index`: the index a side row holds, refused unless its index field is a u32 and zeros. -/
def sideIndex? (row : List Int) : Except String Nat := do
  let field := row.drop (row.length - sideIndexWords)
  if row.length ≤ sideIndexWords || (field.drop 2).any (· != 0) || (field.take 2).any (fun w => w < 0 || w ≥ 2 ^ 16) then
    throw "the row's last 128-bit word is not a u32 index and zeros"
  pure ((field[0]!).toNat + (field[1]!).toNat * 2 ^ 16)

/-- `side_domain`: the domain of every side-`side` tree of the window whose root's domain id is `window`. -/
def sideDomain? (window : ByteArray) (side : String) : Except String Domain := do
  let some s := Side.ofName? side | throw s!"a side is one of ['a', 'b'], not {PouwCommit.pyRepr side}"
  if window.size != 64 then throw "the window's domain id is 64 bytes"
  let binding ← Identity.identityDigestSha512 SIDE_DOMAIN
    (Json.mkObj [("window", Flock.hexOf window), ("side", s.name), ("depth", s.depth)])
  Domain.mk? .sha512 binding owner (2 ^ s.depth)

/-- `pad_node`: the node at (`level`, `index`) of a subtree of `dom` that holds pad leaves only. -/
def padNode (dom : Domain) : Nat → Nat → ByteArray
  | 0, index => dom.pad index
  | level + 1, index => dom.node level index (padNode dom level (2 * index)) (padNode dom level (2 * index + 1))

/-- `side_levels`: a tree of `dom` over `leaves` (1 to `dom.count`, a power of two), each later leaf frame-v3's pad: each
level's nodes that hold one of the leaves, the root last. -/
def sideLevels? (dom : Domain) (leaves : Array ByteArray) : Except String (Array (Array ByteArray)) := do
  let width := dom.count
  if !(1 ≤ leaves.size && leaves.size ≤ width) || width &&& (width - 1) != 0 then
    throw s!"a side tree holds 1 to {width} rows, not {leaves.size}"
  let depth := width.log2
  let mut level := leaves
  let mut levels := #[]
  for d in [0:depth] do
    levels := levels.push level
    level := (Array.range ((level.size + 1) / 2)).map fun y =>
      dom.node d y level[2 * y]! (if 2 * y + 1 < level.size then level[2 * y + 1]! else padNode dom d (2 * y + 1))
  pure (levels.push level)

/-- `side_path`: leaf `index`'s siblings, leaf level first, one per level below the root. -/
def sidePath (dom : Domain) (levels : Array (Array ByteArray)) (index : Nat) : List ByteArray := Id.run do
  let mut out := []
  let mut i := index
  for d in [0:levels.size - 1] do
    let level := levels[d]!
    let s := i ^^^ 1
    out := out ++ [if s < level.size then level[s]! else padNode dom d s]
    i := i / 2
  return out

end PouwWindow
