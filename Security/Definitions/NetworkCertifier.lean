import Definitions.NetworkCertifier.Grid
import Definitions.NetworkCertifier.Run
import Definitions.NetworkCertifier.Ingress
import Definitions.NetworkCertifier.Schedule
import Definitions.NetworkCertifier.Premises

/-!
# The network certifier's protocol: grids, the interaction, the Program and acceptance

`Grid`: one link's rows and grids, occupancy, left-packing, fill, and the FIFO certifier. `Run`: statuses, observations,
the certifier, the interaction and its runs, the Program, acceptance and the accepted observation sequences. `Ingress`:
the interaction read inward, a link's frames with the timing erased, and acceptance as it bears on ingress. `Schedule`:
the Program's logical schedule and clock-sync advice, and the constant-rate Program they define. `Premises`: the one
premise a property of the network certifier takes, that the certifier device runs as built (`DeviceIntact`).

This module, `Assumptions` and `Guarantees` are the trusted text: they import only one another and Mathlib, so their
diff is everything a statement reviewer reads. `SecurityProofs` holds the proofs and every lemma, which nobody reviews
for meaning.
-/
