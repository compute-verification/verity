import PouwCommit.Rows
import PouwWindow.Rounds

/-!
# The lead's commitment premises that PoUW's code computes

Two of the commitment premises the lead's guarantee reads are about what PoUW's executable Lean computes, and are
stated here against its functions, for every input.

* **row-bound** (`LeafFirstBlock`, `RowBound`): each row digest binds its position. A SHA-512 frame-v3 leaf
  (`PouwCommit.Domain.leaf`) hashes `FRAME ‖ be32 4 ‖ "leaf" ‖ be64 64 ‖ id ‖ be64 |rank| ‖ rank ‖ …`, `rank` in
  minimal big-endian bytes, so its first 128-byte block, SHA-512's first compression input after the IV, holds the
  domain's 64-byte id at bytes 49 to 112 and the position from byte 121. A position below 2⁵⁶ fits, and the block
  determines the id and the position, with no hash premise (`LeafFirstBlock`): two rows at two positions, or in two
  domains, never share their first compression. Two leaves with one digest at two positions, or in two domains, are
  two different framed messages with one SHA-512 digest (`RowBound`). A domain's id binds its binding, owner and count
  by `Properties.Pouw.Commit.FrameInjective`. Both hold of every frame leaf, an `hm96-sha512` row tree's included:
  `RowsDomain.leaf` puts the row's hm96 leaf in `Domain.leaf` at the row's rank. Neither covers the row's own
  compressions there: `HmRow.segDigest` reads the row alone, so equal rows at two positions of an `hm96-sha512` tree
  share those messages.
* **g-private's order** (`RoundsOpenAfterRoots`): what a live window's round check (`PouwWindow.roundsReason`) accepts
  is one opening per recorded round root, opening `i` the epoch's round coin `i`, opened against the coin commitment
  fixed before the epoch and after the verifier recorded round `i`'s root. Round `i`'s noise key (`roundKey`) is
  derived from that coin and that root. That the coin is uniform and hidden from the prover until it is opened is the
  live coins' trust (`trusted-host`), which no property states.
-/

namespace Properties.Pouw.Lead.Commit

open PouwCommit
open SampledProofs.Challenger (LiveCommitment LiveOpening liveReason)

/-! ## row-bound -/

/-- The message `Domain.leaf d rank schema value` hashes. -/
def leafMessage (d : Domain) (rank : Nat) (schema : String) (value : ByteArray) : ByteArray :=
  frameBytes "leaf" [d.id, natBytes rank, natBytes rank, schema.toUTF8, value]

/-- A message's first 128-byte block: SHA-512's first compression input after the IV. -/
def firstBlock (m : ByteArray) : ByteArray := m.extract 0 128

/-- **row-bound, the first block**: the first compression input of a SHA-512 frame-v3 leaf's message determines its
domain's id and, below 2⁵⁶, its position, whatever the schemas and values. -/
def LeafFirstBlock : Prop :=
  ∀ binding owner count binding' owner' count' (d d' : Domain) rank rank' schema schema' (v v' b : ByteArray),
    Domain.mk? .sha512 binding owner count = .ok d → Domain.mk? .sha512 binding' owner' count' = .ok d' →
    firstBlock (leafMessage d rank schema v) = b → firstBlock (leafMessage d' rank' schema' v') = b →
    rank < 2 ^ 56 → rank' < 2 ^ 56 → d.id = d'.id ∧ rank = rank'

/-- **row-bound, the digest**: two SHA-512 frame-v3 leaves with one digest, at positions below 2⁵⁶, are at one position
of domains with one id, or their two framed messages, different, have one SHA-512 digest. -/
def RowBound : Prop :=
  ∀ binding owner count binding' owner' count' (d d' : Domain) rank rank' schema schema' (v v' x : ByteArray),
    Domain.mk? .sha512 binding owner count = .ok d → Domain.mk? .sha512 binding' owner' count' = .ok d' →
    d.leaf rank schema v = x → d'.leaf rank' schema' v' = x →
    rank < 2 ^ 56 → rank' < 2 ^ 56 →
      (d.id = d'.id ∧ rank = rank') ∨
        (leafMessage d rank schema v ≠ leafMessage d' rank' schema' v' ∧
          Flock.Sha512.hash (leafMessage d rank schema v) = Flock.Sha512.hash (leafMessage d' rank' schema' v'))

/-! ## g-private's order -/

/-- **The round coins open after their roots**: what `roundsReason` accepts is a live epoch's coin commitment `c` and
one opening per recorded round root, opening `i` being `c`'s round coin `i`, opened after `roots[i]`, which
`liveReason` accepts against `c`'s root. -/
def RoundsOpenAfterRoots : Prop :=
  ∀ (c : LiveCommitment) (roots : List ByteArray) (openings : List LiveOpening),
    PouwWindow.roundsReason c roots openings = .ok none →
      PouwWindow.liveEpochReason c = none ∧ openings.length = roots.length ∧
        ∀ i (h : i < openings.length) (h' : i < roots.length),
          openings[i].label = PouwWindow.ROUND ∧ openings[i].index = i ∧ openings[i].after = roots[i] ∧
            liveReason c openings[i] PouwWindow.ROUND i roots[i] = .ok none

end Properties.Pouw.Lead.Commit
