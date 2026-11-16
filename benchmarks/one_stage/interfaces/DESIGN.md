# E2E demo interfaces: design

How the replay bundle's step commitments are built. `README.md` says what the two documents are and checks each
fixture.

The step commitment is kernels' commit call (`verity/commit-slots/v1`): a step tree over slot trees over
proof-block trees over rows, only each proof block's outputs committed. An output row opens in its proof block's own commitment. An input
row opens where its producer committed it, and names the producer's coordinates (the fixture's `x` and `sx` open at
their layer's input quantizer). The bundle adds a block tree over each server block's step roots.
The bundle builds its step commitments with kernels' CPU reference, `verity/commitments/cuda/commit_slots/slots.py`,
imported from the tree. The suite pins the SHA-512 of their fixture, `commit-slots-v1.json`, which the reference must
reproduce; its roots and digests are those of the commit `REFERENCE` pins.

