/-
`FrameV3Read_v1{D, SCHEMA}` on the verifier's builder (`Flock.HmNets`), part for part as `gates.frame_v3` builds it: the
leaf (`leafHash`, `FrameV3Leaf_v1`), one node a level (`nodeHash`, `FrameV3Node_v1`) and the comparison with the root
(`eqDigest`, `LmsEqual_v1{N = 64}`). A message carrying a hidden `uint` is laid out once for each of its lengths and
selected bit by bit under flags that nest (`hashUint`). It imports only the builder.
-/
module

public import Proofs.Accounting.ReadGates.Net

@[expose] public section

namespace Accounting.ReadGates

open Flock.HmNets hiding and

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

/-- `_flags`' chain over bits `t + 7, t + 6, …, 8` of `bits`, from the top down: `acc` the OR of the bits above,
and the OR at each bit `8 j` put before `out`. -/
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
  sha512W ws

/-- `frame_v3.leaf_hash`: the leaf of `value` at rank = position = `x` (`8 n` bits) under the domain id `domain`. -/
def leafHash (schema : String) (domain value : List Word) (x : List Lin) (n : Nat) : M (List Word) :=
  hashUint LEAF [bytesW domain] x n fun u => [u, u, constBytes schema.toUTF8.data.toList, bytesW value]

/-- `frame_v3.node_hash`: the node at `level` (8 bits) and `index` (`8 n` bits) over `cur` and `sib`, `cur` the right
child iff `d`: the swap is `t = d (cur ⊕ sib)`, left `cur ⊕ t`, right `sib ⊕ t`. -/
def nodeHash (domain : List Word) (level : Byte) (index : List Lin) (d : Lin) (cur sib : List Word) (n : Nat) :
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

/-- The constant level byte `k`, least significant bit first. -/
def levelByte (k : Nat) : Byte := (List.range 8).map fun j => Lin.const (k.testBit j)

/-- The read's levels `k, k + 1, …` from the digest `cur`, one per sibling: level `k` swaps by `pos`'s bit `k` and hashes
the node at index `pos >> (k + 1)`, zero-extended to whole bytes. -/
def climbW (domain : List Word) (pos : List Lin) : Nat → List (List Word) → List Word → M (List Word)
  | _, [], cur => pure cur
  | k, s :: ss, cur => do
    let hi := pos.drop (k + 1)
    let n := indexBytes hi.length
    let cur ← nodeHash domain (levelByte k) (hi ++ List.replicate (8 * n - hi.length) Lin.zero) (pos.getD k Lin.zero) cur
      s n
    climbW domain pos (k + 1) ss cur

/-- `FrameV3Read_v1{D, SCHEMA}`: 1 iff `path` opens `value` at position `pos` (`D` bits) to `root` in the depth-`D` dense
frame-v3-sha512 tree of domain id `domain`. -/
def frameRead (D : Nat) (schema : String) (root domain value : List Word) (pos : List Lin) (path : List (List Word)) :
    M Lin := do
  let n := indexBytes D
  let leaf ← leafHash schema domain value (pos ++ List.replicate (8 * n - D) Lin.zero) n
  let cur ← climbW domain pos 0 path leaf
  eqDigest cur root

end Accounting.ReadGates
