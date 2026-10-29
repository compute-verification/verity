import Proofs.Pouw.Lead.PStar.AlphabetProofs
import Proofs.Pouw.Lead.PStar.ExpansionProofs
import Proofs.Pouw.Lead.PStar.MassFloorProofs

/-!
# The lead's p*: how often a coin lane leaves an entry unchanged

Its proofs are in `Lead/PStar/`, its new declarations in `Pouw.SecurityProofs.Lead.PStar`.

* `AlphabetProofs`: the per-entry and matrix hit counts for digits drawn after the data, for `uw6`, the coin lane
  `1 + b + s` and hw's row lane, alone or in a sum of lanes.
* `ExpansionProofs`: the affine expansion `u + t·v` is pairwise uniform on a prime modulus, with the two moments of
  the hit count over steps.
* `MassFloorProofs`: point masses at most `p` force a variance of at least `(1/p² − 1)/12`, with its mean over steps.
-/
