/-
`FrameV3Read_v1{D, SCHEMA}` on the verifier's builder (`Flock.HmNets`), as `verity/ml/commitments/gates/frame_v3.py`
builds it, row for row: the leaf part (`Frame.leafHash`, `FrameV3Leaf_v1{B, SCHEMA}`), a node part a level
(`Frame.nodeHash`, `FrameV3Node_v1{B}`) and the comparison with the root (`Frame.eqDigest`, `LmsEqual_v1{N = 64}`), each
a traced Definition whose ANDs are `HmNets.and`s in Python's order, and the read's body (`frameWith`), which takes its
calls as parameters (those three and `BL.constant`'s, for its constant bits) and makes them in Python's order. A message carrying a hidden `uint` is laid out once for each of its lengths
and selected bit by bit under flags that nest (`Frame.hashUint`); its SHA-512 is the chain of `HmNets.compress` from the
IV (`Frame.ivW`) over its words, padded by `padTail` (`shaW`). A byte is 8 forms least significant first, and a word 64
forms least significant first with its bytes big-endian.
-/
module

public import Flock.HmNets
public import Flock.ReadNet.Sha

@[expose] public section

namespace Flock.ReadNet

open Flock.HmNets hiding and

namespace Frame

/-! ## Bytes and words -/

/-- A byte of the circuit: its 8 bits' forms, least significant first. -/
abbrev Byte := List Lin

/-- `frame_v3._bytes`: words' bytes, eight a word, big-endian. -/
def bytesW (ws : List Word) : List Byte :=
  ws.flatMap fun w => (List.range 8).map fun b => (w.drop (8 * (7 - b))).take 8

/-- `frame_v3._words`: bytes read back as words, eight a word, big-endian. -/
def wordsB (data : List Byte) : List Word :=
  (List.range (data.length / 8)).map fun k => (List.range 8).flatMap fun b => data.getD (8 * k + 7 - b) []

/-- `forms.const_bits b 8`: a constant byte. -/
def constByte (b : UInt8) : Byte := (List.range 8).map fun i => Lin.const (b.toNat.testBit i)

/-- `frame_v3._const_bytes`. -/
def constBytes (m : List UInt8) : List Byte := m.map constByte

/-- `int.to_bytes(n, "big")`: `v`'s low `8 n` bits as `n` bytes, big-endian. -/
def beBytes (n v : Nat) : List UInt8 := (List.range n).map fun i => (v / 256 ^ (n - 1 - i) % 256).toUInt8

/-- `frame_v3._uint_bytes`: the low `8 n` bits of `bits` (least significant first, zero past their end) as `n` bytes,
big-endian. -/
def uintBytes (bits : List Lin) (n : Nat) : List Byte :=
  let p := bits.take (8 * n) ++ List.replicate (8 * n - min bits.length (8 * n)) Lin.zero
  (List.range n).map fun b => (p.drop (8 * (n - 1 - b))).take 8

/-- SHA-512's initial chaining value as constant words. -/
def ivW : List Word := Flock.Sha512.iv.toList.map fun v => constW v.toBitVec

/-! ## Messages over a hidden `uint` -/

/-- `merkle._FRAME`. -/
def FRAME : List UInt8 := "veritor/protocol/merkle/frame/v3".toUTF8.data.toList ++ [0]

/-- `merkle._LEAF`. -/
def LEAF : List UInt8 := "leaf".toUTF8.data.toList

/-- `merkle._NODE`. -/
def NODE : List UInt8 := "node".toUTF8.data.toList

/-- `frame_v3._message`: the framed message over parts of byte forms, padded, as SHA-512 words. -/
def message (tag : List UInt8) (parts : List (List Byte)) : List Word :=
  let data := constBytes (FRAME ++ beBytes 4 tag.length ++ tag) ++
    parts.flatMap fun p => constBytes (beBytes 8 p.length) ++ p
  wordsB (data ++ constBytes (padTail data.length))

/-- `forms.OR`: `a ⊕ b ⊕ a b`. -/
def orL (a b : Lin) : M Lin := do
  let r ← Flock.HmNets.and a b
  pure (a ^^^ b ^^^ r)

/-- `_flags`' chain over bits `t + 7, t + 6, …, 8` of `bits`, from the top down: `acc` the OR of the bits above, and the
OR at each bit `8 j` put before `out`. -/
def flagsGo (bits : List Lin) : Nat → Option Lin → List Lin → M (List Lin)
  | 0, _, out => pure out
  | t + 1, acc, out => do
    let b := bits.getD (t + 8) Lin.zero
    let a ← match acc with
      | none => pure b
      | some a => orL a b
    flagsGo bits t (some a) (if (t + 8) % 8 = 0 then a :: out else out)

/-- `frame_v3._flags`: `z_j` for `j = 1 … n - 1`, set iff `bits` (`8 n` of them) has a bit at or above `8 j` set. -/
def flags (bits : List Lin) (n : Nat) : M (List Lin) := flagsGo bits (8 * n - 8) none []

/-- `_select`'s sum for one bit: `m` and, for each flag `z` and adjacent layouts' bits `(a, b)`, `z (a ⊕ b)` added. -/
def selectGo : Lin → List Lin → List (Lin × Lin) → M Lin
  | m, z :: zs, (a, b) :: ps => do
    let d := a ^^^ b
    let m' ← if d.isConst then pure (if d.c then m ^^^ z else m) else do
      let r ← Flock.HmNets.and z d
      pure (m ^^^ r)
    selectGo m' zs ps
  | m, _, _ => pure m

/-- One bit of `_select`: the layouts' bits `col`, selected under the flags `fl`. -/
def selectBit (fl col : List Lin) : M Lin := selectGo (col.headD Lin.zero) fl (col.zip col.tail)

/-- `frame_v3._select`: bit by bit, `layouts[m]` where the first `m` flags are set and the next is not. -/
def select (fl : List Lin) (layouts : List (List Word)) : M (List Word) :=
  (List.range (layouts.headD []).length).mapM fun i =>
    (List.range 64).mapM fun j => selectBit fl (layouts.map fun l => (l.getD i []).getD j Lin.zero)

/-- `frame_v3._hash_uint`: SHA-512 of the message whose parts are `head`, then `tail (uint x)`, over every length of
`uint x` up to `n` bytes. -/
def hashUint (tag : List UInt8) (head : List (List Byte)) (x : List Lin) (n : Nat)
    (tail : List Byte → List (List Byte)) : M (List Word) := do
  let layouts := (List.range n).map fun L => message tag (head ++ tail (uintBytes x (L + 1)))
  let fl ← flags x n
  let ws ← select fl layouts
  shaW ((ws.length + 15) / 16) ivW ws

/-! ## The parts -/

/-- `FrameV3Leaf_v1{B = n, SCHEMA}` (`frame_v3.leaf_hash`): the leaf of `value` at rank = position = `x` (`8 n` bits)
under the domain id `domain`. -/
def leafHash (schema : String) (n : Nat) (domain value : List Word) (x : List Lin) : M (List Word) :=
  hashUint LEAF [bytesW domain] x n fun u => [u, u, constBytes schema.toUTF8.data.toList, bytesW value]

/-- `FrameV3Node_v1{B = n}` (`frame_v3.node_hash`): the node at `level` (8 bits) and `index` (`8 n` bits) over `cur` and
`sib`, `cur` the right child iff `d`: the swap is `t = d (cur ⊕ sib)`, left `cur ⊕ t`, right `sib ⊕ t`. -/
def nodeHash (n : Nat) (domain : List Word) (level : Byte) (index : List Lin) (d : Lin) (cur sib : List Word) :
    M (List Word) := do
  let t ← (cur.zip sib).mapM fun cs => (cs.1.zip cs.2).mapM fun xy => Flock.HmNets.and d (xy.1 ^^^ xy.2)
  let left := (cur.zip t).map fun ct => xorW ct.1 ct.2
  let right := (sib.zip t).map fun st => xorW st.1 st.2
  hashUint NODE [bytesW domain, [level]] index n fun u => [u, bytesW left, bytesW right]

/-- One pass of `forms.and_reduce`: ANDs of adjacent pairs, an odd last bit carried. -/
def andPairs : List Lin → M (List Lin)
  | a :: b :: rest => do
    let r ← Flock.HmNets.and a b
    let rs ← andPairs rest
    pure (r :: rs)
  | l => pure l

/-- `forms.and_reduce`'s passes, at most `k` of them. -/
def andTree : Nat → List Lin → M Lin
  | _, [] => pure Lin.one
  | _, [a] => pure a
  | 0, a :: _ => pure a
  | k + 1, l => do
    let l' ← andPairs l
    andTree k l'

/-- `forms.and_reduce`: the AND of `bits`, pairwise pass by pass. -/
def andReduce (bits : List Lin) : M Lin := andTree bits.length bits

/-- `LmsEqual_v1{N = 64}`: 1 iff the digests `x` and `y` agree, the AND of their bits' agreements. -/
def eqDigest (x y : List Word) : M Lin :=
  andReduce ((x.zip y).flatMap fun ab => (ab.1.zip ab.2).map fun uv => uv.1 ^^^ uv.2 ^^^ Lin.one)

/-- `frame_v3.index_bytes`: `B` of a field of `bits` bits. -/
def indexBytes (bits : Nat) : Nat := max 1 ((bits + 7) / 8)

/-- The body's constant bits so far: the output of its `BL.constant(0)` call and of its `BL.constant(1)` call, once made. -/
abbrev Consts := Option Lin × Option Lin

/-- `frame_v3._body.const(v, w)`: `v`'s low `w` bits, least significant first, each the output of the body's one
`BL.constant` call for its value, which `const` makes on the value's first use, 0 before 1. -/
def constBits (const : Bool → M Lin) (c : Consts) (v w : Nat) : M (Consts × List Lin) := do
  let need := fun b => (List.range w).any fun j => v.testBit j == b
  let c0 ← if need false && c.1.isNone then some <$> const false else pure c.1
  let c1 ← if need true && c.2.isNone then some <$> const true else pure c.2
  pure ((c0, c1), (List.range w).map fun j => if v.testBit j then c1.getD Lin.one else c0.getD Lin.zero)

end Frame

open Frame

/-- The read's levels `k, k + 1, …` from the digest `cur`, one a sibling, each a call of `node B`: level `k` swaps by
`pos`'s bit `k` and hashes the node at level `k` and index `pos >> (k + 1)`, zero-extended to `B` bytes, the padding's
and the level's constant bits from `Frame.constBits const` in that order. -/
def climbWith (const : Bool → M Lin)
    (node : Nat → List Word → Byte → List Lin → Lin → List Word → List Word → M (List Word))
    (domain : List Word) (pos : List Lin) : Nat → List (List Word) → Consts → List Word → M (List Word)
  | _, [], _, cur => pure cur
  | k, s :: ss, c, cur => do
    let hi := pos.drop (k + 1)
    let n := indexBytes hi.length
    let pad ← Frame.constBits const c 0 (8 * n - hi.length)
    let lv ← Frame.constBits const pad.1 k 8
    let cur ← node n domain lv.2 (hi ++ pad.2) (pos.getD k Lin.zero) cur s
    climbWith const node domain pos (k + 1) ss lv.1 cur

/-- `FrameV3Read_v1{D, SCHEMA}`'s body (`frame_v3._body`) over its calls: the constant bits (`const`, `BL.constant`), the
leaf `leaf B` of `value` at `pos` (`D` bits, zero-extended to `B = index_bytes(D)` bytes), a node a level
(`climbWith node`), and `eq` of the last node and `root`. -/
def frameWith (const : Bool → M Lin) (leaf : Nat → List Word → List Word → List Lin → M (List Word))
    (node : Nat → List Word → Byte → List Lin → Lin → List Word → List Word → M (List Word))
    (eq : List Word → List Word → M Lin) (D : Nat) (root domain value : List Word) (pos : List Lin)
    (path : List (List Word)) : M Lin := do
  let n := indexBytes D
  let pad ← Frame.constBits const (none, none) 0 (8 * n - D)
  let cur ← leaf n domain value (pos ++ pad.2)
  let cur ← climbWith const node domain pos 0 path pad.1 cur
  eq cur root

/-- `BL.constant(b)` in a flat net: the constant form. -/
def constForm (b : Bool) : M Lin := pure (Lin.const b)

/-- `FrameV3Read_v1{D, SCHEMA}`: 1 iff `path` opens `value` at position `pos` (`D` bits) to `root` in the depth-`D` dense
frame-v3-sha512 tree of domain id `domain`. -/
def frame (D : Nat) (schema : String) : List Word → List Word → List Word → List Lin → List (List Word) → M Lin :=
  frameWith constForm (leafHash schema) nodeHash eqDigest D

end Flock.ReadNet
