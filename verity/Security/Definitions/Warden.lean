import Definitions.Warden.Grid
import Definitions.Warden.Run
import Definitions.Warden.Ingress
import Definitions.Warden.Schedule

/-!
# The network warden's protocol: grids, the interaction, the Program and acceptance

`Grid`: one link's rows and grids, occupancy, left-packing, fill, and the FIFO warden. `Run`: statuses, observations,
the warden, the interaction and its runs, the Program, acceptance and the accepted observation sequences. `Ingress`:
the interaction read inward, a link's frames with the timing erased, and acceptance as it bears on ingress. `Schedule`:
the Program's logical schedule and clock-sync advice, and the constant-rate Program they define.

This module, `Assumptions` and `Guarantees` are the trusted text: they import only one another and Mathlib, so their
diff is everything a statement reviewer reads. This module is the Program and imports only Lean core (the audit's
`layers`): its counts and sums run over `List`, and `SecurityProofs.Reference` proves them equal to their `Finset`
forms. `SecurityProofs` holds the proofs and every lemma, which nobody reviews
for meaning.
-/
