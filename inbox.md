[2026-11-20T05:37Z] fc06bc2691df
From the proofs lead 9:42 PM PST, small add to your branch: circuits showed flock-lower-check lowering a 41-bit record with no refusal (it does not call admitted; it cannot, Flock does not import Spot). Add ONE sentence to FlockLowerCheck's compose-mode help/docstring: its output is not a verdict of the door; the door's lowering is lowerBlock with admitted and admittedBlock, and nothing may cite this executable's acceptance of a block as the door's. No behaviour change.

