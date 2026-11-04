module

public import Properties.Flock.Nets

/-!
# The verifier's two fixed nets build (`Properties.Flock.Nets`): open

Not proved yet. The kernel builds neither net while `HmNets.symm` is well-founded recursion; a structurally recursive
`symm`, proved equal to the deployed one, lets `decide +kernel` build both and replaces this stub.
-/

@[expose] public section

namespace Properties.Flock.Nets

theorem fixedNetsBuild : FixedNetsBuild := by
  sorry

end Properties.Flock.Nets
