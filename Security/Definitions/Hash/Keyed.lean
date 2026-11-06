import Mathlib
import Definitions.Hash.Tree

/-!
# The keyed leaf over a hash (trusted)

A frame-v3 subtree whose top message is hashed under a window's secret key, with the hash a parameter `H` (SHA-512 is
the instance of record). For the window's key `ks` (64 OS bytes, fresh each window,
never published), its 64-byte live salt `salt` and an instance's top message `M_top`:

    d = H(ks ‖ salt ‖ M_top)

`ks ‖ salt` is exactly one 128-byte block, so the committer resumes `H` from that block's chaining value for every
digest of the window. `M_top` is the message whose unkeyed hash is the subtree's frame-v3 root: the root node
`frame "node" [id, uint (D − 1), uint 0, L, R]` at depth `D = merkle_depth count` for a count of at least 2, and the
frame leaf's message itself for a count of 1 (`Top`). Every message below `M_top` is hashed from the initial value, as
in `Tree`. `d` replaces the subtree's root: it is the instance's public digest, and no frame-v3 message.

An opening at `rank` is the frame leaf's message and the path's siblings from the leaf level up. `climb` hashes every
message below the top from the initial value and returns the top message unhashed, and the verifier accepts when
`rank < count`, the path has `merkle_depth count` siblings and `H(ks ‖ salt ‖ top) = d` (`accepts`). `hashed` lists the
messages it hashed: those below the top, bottom first, then the keyed message. `H` of the top message is the subtree's
unkeyed root, which the committer never computes.

The window's key is committed before its salt is drawn, as the hm96 leaf over `ks` under 192 OS salt bytes
(`Hm96.leaf` at `x = ks`), the window's key leaf, which a drawn unit's proof opens `ks` against.
-/

namespace Hash.Keyed

/-- The keyed message `ks ‖ salt ‖ m`: the key and the salt fill its first block. -/
def msg (ks salt : Digest) (m : Msg) : Msg := ks.bytes ++ salt.bytes ++ m

/-- The keyed digest `H(ks ‖ salt ‖ m)`. -/
def digest (H : Msg → Digest) (ks salt : Digest) (m : Msg) : Digest := H (msg ks salt m)

/-- An instance's top message as its data give it: the root node over the children `l` and `r` at level `lv`, or, for a
one-leaf subtree, the frame leaf over `value` under `schema`. -/
inductive Top where
  | node (lv : ℕ) (l r : Digest)
  | leaf (schema : String) (value : Msg)

/-- The top message under the domain `D`: the node at level `lv` and index 0, or the frame leaf at rank 0. -/
def Top.msg (D : Tree.Domain) : Top → Msg
  | .node lv l r => Tree.nodeMsg D lv 0 l r
  | .leaf schema value => Tree.leafMsg D 0 schema value

/-- From the message `m` at index `cur` of level `lv`, up the siblings, every message below the top hashed from the
initial value: the messages hashed on the way, bottom first, and the top message. -/
def climb (H : Msg → Digest) (D : Tree.Domain) : ℕ → ℕ → Msg → List Digest → List Msg × Msg
  | _, _, m, [] => ([], m)
  | lv, cur, m, s :: ss =>
    let up := climb H D (lv + 1) (cur / 2)
      (if cur % 2 = 0 then Tree.nodeMsg D lv (cur / 2) (H m) s else Tree.nodeMsg D lv (cur / 2) s (H m)) ss
    (m :: up.1, up.2)

/-- The verifier accepts the frame leaf's message `leaf` at `rank` with path `sibs` against the keyed digest `d`, under
the key `ks` and the salt `salt`. -/
def accepts (H : Msg → Digest) (ks salt : Digest) (D : Tree.Domain) (d : Digest) (rank : ℕ) (leaf : Msg)
    (sibs : List Digest) : Prop :=
  rank < D.count ∧ sibs.length = Tree.depth D.count ∧ digest H ks salt (climb H D 0 rank leaf sibs).2 = d

/-- The messages the verifier hashed checking that opening: those below the top, bottom first, then the keyed one. -/
def hashed (H : Msg → Digest) (ks salt : Digest) (D : Tree.Domain) (rank : ℕ) (leaf : Msg) (sibs : List Digest) :
    List Msg :=
  (climb H D 0 rank leaf sibs).1 ++ [msg ks salt (climb H D 0 rank leaf sibs).2]

end Hash.Keyed
