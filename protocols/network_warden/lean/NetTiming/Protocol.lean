import NetTiming.Protocol.Grid
import NetTiming.Protocol.Run

/-!
# The network warden's protocol: grids, the interaction, the Program and acceptance

`Grid`: one link's rows and grids, occupancy, left-packing, fill, and the FIFO warden. `Run`: statuses, observations,
the warden, the interaction and its runs, the Program, acceptance and the accepted observation sequences.

This module, `Assumptions` and `Guarantees` are the trusted text: they import only one another and Mathlib, so their
diff is everything a statement reviewer reads. `SecurityProofs` holds the proofs and every lemma, which nobody reviews
for meaning.
-/
