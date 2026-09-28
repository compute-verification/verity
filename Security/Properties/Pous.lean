import Definitions.Pous.P2.Cost
import PousCheck.SetupProtocol

/-!
# The PoUS setup check's properties, stated against the Lean code

What `verity/pous/p2/v4/setup` (`PousCheck.SetupProtocol.protocol`, package `pous_check`) accepts and publishes, stated
against its `check` and `Accepts` (`CoreProtocol/Protocol.lean`), and what its block decoder computes, stated against
the trusted model's (`Definitions/Pous/P2/Cost.lean`). No statement takes an assumption: the check is a function of its
three inputs, and these say which function. The check sees every commitment and every `C*` of any shape, at parameters
`P2.Params.validate` accepts (the protocol's `params` refuses any other before the check runs).

* `AcceptsIff`: the check accepts iff every block of `C*` decodes canonically (`Decodes`) and the decoded payload's
  commitment (`Commit.commit`) is the server's;
* `PublishesRoot`: on acceptance the published root is `Merkle.root` over the encoded blocks' leaves in block order
  (`leaves`), beside the server's commitment, the salt and the block count;
* `Deployed`: the registered point `P2.Params.deployed` is valid, with `p = 2^16448 − 21065`, 2,056-byte blocks of
  2,054 payload bytes, and `B = 2^19` blocks;
* `AcceptsIffDeployed`, `PublishesRootDeployed`: the first two at that point;
* `DecodeIsModel`: the code's `P2.decodeValue p K t c` is the trusted model's block decoder `Pous.decBlock` (P2 in the
  ring model `ringRO`) run at the identity labeling with block keys `(K, t)`, on every canonical `c < p`, read mod
  `2^b`; `DecodeIsModelDeployed` at the registered point. With `Decodes` (which reads `decodeValue`), it ties what the
  check accepts to `p2ring`'s `dec`, which the model's audit statements read.

`Decodes` and `leaves` are declarative: they name each block by its segment and index rather than by the check's loop,
and read only the code's primitives (the keys, `P2.decodeValue`, `Merkle.packageLeaf`). Only `Deployed` is proved
(`Properties.Proofs.Pous`); nothing is pinned yet.
-/

namespace PousCheck.Properties

/-- Block `j` of segment `s` decodes canonically to the payload `m`: the keys of `(s, j)` under `salt` exist, the block
is `blockBytes` bytes, its value `c` is below `p`, `D(c)` fits in the payload's bits, and `m` is `D(c)` as `payloadBytes`
big-endian bytes (`decode_segment`). -/
def BlockDecodes (q : P2.Params) (salt : ByteArray) (s j : Nat) (block m : ByteArray) : Prop :=
  ∃ pre k t a, Keys.packagePrefix q salt = .ok pre ∧ (Keys.Expander.ofPrefix pre).keys q s j = .ok (k, t, a) ∧
    block.size = q.blockBytes ∧ Prim.natOfBE block < q.p ∧
    P2.decodeValue q.p k t (Prim.natOfBE block) < 2 ^ (8 * q.payloadBytes) ∧
    m = Prim.beOfNat (P2.decodeValue q.p k t (Prim.natOfBE block)) q.payloadBytes

/-- `C*` decodes canonically to `W`: `segments` segments of `blocksPerSegment` blocks each, and `W` is the blocks' payloads
in block order, block `j` of segment `s` at payload offset `(s · blocksPerSegment + j) · payloadBytes`. -/
def Decodes (q : P2.Params) (salt : ByteArray) (C : Array (Array ByteArray)) (W : ByteArray) : Prop :=
  C.size = q.segments ∧ W.size = q.wBytes ∧
    ∀ s (hs : s < C.size), C[s].size = q.blocksPerSegment ∧
      ∀ j (hj : j < C[s].size),
        let at_ := (s * q.blocksPerSegment + j) * q.payloadBytes
        BlockDecodes q salt s j C[s][j] (W.extract at_ (at_ + q.payloadBytes))

/-- The leaves of `C*` (`verifier.segment_leaves`, segment by segment): `packageLeaf salt s v` of block `v` of segment
`s`, in block order. -/
def leaves (salt : ByteArray) (C : Array (Array ByteArray)) : Array ByteArray :=
  (C.toList.mapIdx fun s seg => seg.toList.mapIdx fun v block => Merkle.packageLeaf salt s v block).flatten.toArray

/-- At `q`, the check accepts iff `C*` decodes canonically to some `W` whose commitment is the server's. -/
def AcceptsIffAt (q : P2.Params) : Prop :=
  ∀ (c : SetupProtocol.Commitment) (C : Array (Array ByteArray)),
    SetupProtocol.protocol.Accepts q c C ↔ ∃ W, Decodes q c.salt C W ∧ Commit.commit q W = .ok c.w

/-- At `q`, an acceptance publishes `Merkle.root` over `C*`'s leaves (which it has: an accepted `C*` has a block), the
server's commitment, the salt and the block count `B`; its profile is the mode alone. -/
def PublishesRootAt (q : P2.Params) : Prop :=
  ∀ (c : SetupProtocol.Commitment) (C : Array (Array ByteArray)) (pub : SetupProtocol.Public) (prof : Unit),
    SetupProtocol.protocol.check q c C = .accept pub prof →
      Merkle.root (leaves c.salt C) = some pub.root ∧ pub.commitment = c.w ∧ pub.salt = c.salt ∧ pub.blocks = q.blocks

def AcceptsIff : Prop := ∀ q : P2.Params, q.validate = .ok q → AcceptsIffAt q

def PublishesRoot : Prop := ∀ q : P2.Params, q.validate = .ok q → PublishesRootAt q

/-- The registered point (`p2/PROTOCOL.md` v1.1, `p2-p16448-c21065`, profile `p2-2054B`): valid, its prime
`2^16448 − 21065`, 2,056-byte blocks of 2,054 payload bytes, `2^16` blocks per segment in 8 segments, `2^19` in all. -/
def Deployed : Prop :=
  let q := P2.Params.deployed
  q.validate = .ok q ∧ q.p = 2 ^ 16448 - 21065 ∧ q.p = P ∧ q.blockBytes = 2056 ∧ q.blockBytes = W_BYTES ∧
    q.payloadBytes = 2054 ∧ q.blocksPerSegment = 2 ^ 16 ∧ q.segments = 8 ∧ q.blocks = 2 ^ 19

def AcceptsIffDeployed : Prop := AcceptsIffAt P2.Params.deployed

def PublishesRootDeployed : Prop := PublishesRootAt P2.Params.deployed

/-- The identity labeling `φ = id` of `Z/p` (the model's `nonemptyΩ` witness), at which `p2ring` is
`decode_segment` for given keys. -/
def idLabeling (p : ℕ) [NeZero p] : Pous.OddLabeling p :=
  ⟨(ZMod.finEquiv p).symm.toEquiv, fun x => map_neg (ZMod.finEquiv p).symm x⟩

/-- At `(B, w, p, b)`: for every block `j` whose keys are the mask `K < 2^w` and the offset `t < p`, and every canonical
`c < p`, the model's `decBlock` run at the identity labeling reads `P2.decodeValue p K t c mod 2^b`. -/
def DecodeIsModelAt (B w p b : ℕ) [NeZero p] : Prop :=
  ∀ (keys : Fin B → Pous.Bits w × Fin p) (j : Fin B) (K c : ℕ),
    K < 2 ^ w → c < p → (keys j).1 = Pous.bitsOfNat w K →
      Pous.bitsVal ((Pous.decBlock B w p b j (Pous.bitsOfNat w c)).run
          ((Pous.ringRO B w p).answer (idLabeling p, keys)))
        = P2.decodeValue p K (keys j).2 c % 2 ^ b

def DecodeIsModel : Prop := ∀ (B w p b : ℕ) [NeZero p], p ≤ 2 ^ w → DecodeIsModelAt B w p b

/-- `DecodeIsModel` at the registered point: `2^19` blocks of `16,448` bits, `p = 2^16448 − 21065`, `8 · 2,054` payload
bits. -/
def DecodeIsModelDeployed : Prop := DecodeIsModelAt (2 ^ 19) 16448 (2 ^ 16448 - 21065) (8 * 2054)

end PousCheck.Properties
