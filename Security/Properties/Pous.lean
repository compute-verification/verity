import PousCheck.SetupProtocol

/-!
# The PoUS setup check's properties, stated against the Lean code

What `verity/pous/p2/v4/setup` (`PousCheck.SetupProtocol.protocol`, package `pous_check`) accepts and publishes, stated
against its `check` and `Accepts` (`CoreProtocol/Protocol.lean`) in the one shape a property takes: for every input, if
the check accepts, a named conclusion holds. The check is a function of its three inputs (the parameters, the server's
commitment and salt, and `C*` of any shape) and sees all of them, so no statement takes a premise. It refuses parameters
`P2.Params.validate` rejects, so their validity follows from acceptance and is no hypothesis.

* `AcceptsDecodes`: an acceptance means `DecodesToCommitment`: every block of `C*` decodes canonically (`Decodes`) to a
  payload `W` whose commitment (`Commit.commit`) is the server's;
* `PublishesRoot`: an acceptance publishes `Published`: `Merkle.root` over the encoded blocks' leaves in block order
  (`leaves`), beside the server's commitment, the salt and the block count;
* `Deployed`, a fixed-input evaluation: the registered point `P2.Params.deployed` is valid, with `p = 2^16448 − 21065`,
  2,056-byte blocks of 2,054 payload bytes, and `B = 2^19` blocks.

`Decodes` and `leaves` are declarative: they name each block by its segment and index rather than by the check's loop,
and read only the code's primitives (the keys, `P2.decodeValue`, `Merkle.packageLeaf`).

All three are proved and guarantees of this package (`lean-audit.json`): `Deployed` in `Proofs.Properties.Pous`,
`AcceptsDecodes` and `PublishesRoot` in `Proofs.Properties.Pous.Setup`. What is not a claim is stated in
`Proofs.Properties.Pous`, open: the converse of
`AcceptsDecodes` (completeness: a `C*` that decodes to the committed `W` is accepted), with its instance, and the code's
block decoder read as the trusted model's (`DecodeIsModel`).
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

/-- `C*` decodes canonically, under the server's salt, to some `W` whose commitment is the server's. -/
def DecodesToCommitment (q : P2.Params) (c : SetupProtocol.Commitment) (C : Array (Array ByteArray)) : Prop :=
  ∃ W, Decodes q c.salt C W ∧ Commit.commit q W = .ok c.w

/-- The check accepts only a `C*` that decodes to the committed `W`. -/
def AcceptsDecodes : Prop :=
  ∀ (q : P2.Params) (c : SetupProtocol.Commitment) (C : Array (Array ByteArray)),
    SetupProtocol.protocol.Accepts q c C → DecodesToCommitment q c C

/-- `pub` is `Merkle.root` over `C*`'s leaves (which it has: an accepted `C*` has a block), the server's commitment, the
salt and the block count `B`. -/
def Published (q : P2.Params) (c : SetupProtocol.Commitment) (C : Array (Array ByteArray))
    (pub : SetupProtocol.Public) : Prop :=
  Merkle.root (leaves c.salt C) = some pub.root ∧ pub.commitment = c.w ∧ pub.salt = c.salt ∧ pub.blocks = q.blocks

/-- An acceptance publishes `C*`'s root, the server's commitment, the salt and `B`; its profile is the mode alone. -/
def PublishesRoot : Prop :=
  ∀ (q : P2.Params) (c : SetupProtocol.Commitment) (C : Array (Array ByteArray)) (pub : SetupProtocol.Public)
    (prof : Unit), SetupProtocol.protocol.check q c C = .accept pub prof → Published q c C pub

/-- The registered point (P2 v4, `p2-p16448-c21065`, profile `p2-2054B`): valid, its prime
`2^16448 − 21065`, 2,056-byte blocks of 2,054 payload bytes, `2^16` blocks per segment in 8 segments, `2^19` in all. -/
def Deployed : Prop :=
  let q := P2.Params.deployed
  q.validate = .ok q ∧ q.p = 2 ^ 16448 - 21065 ∧ q.p = P ∧ q.blockBytes = 2056 ∧ q.blockBytes = W_BYTES ∧
    q.payloadBytes = 2054 ∧ q.blocksPerSegment = 2 ^ 16 ∧ q.segments = 8 ∧ q.blocks = 2 ^ 19

end PousCheck.Properties
