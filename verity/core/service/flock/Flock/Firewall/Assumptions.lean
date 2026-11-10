/-
# The ZK server's named assumption (D8 (a), zero knowledge only)

The contract derives what the schedule fixes, and checks every commitment, pad and coin itself. What the ZK server fixes by
its own computation it only reports: its level-0 recheck commitment (`By.recheck`), its opening of that commitment
(`By.shadow`), and the values it computes from its pads and the values that passed, `final_c'` and the inner proof
(`By.own`). The session's Lean finish computes them (`Flock.Firewall.Finish`: the recheck's `Link` commitment, the opening
`openLive`, the inner proof `innerLive` and the proof file `repFw`), and the rep's Lean frame releases the inner proof's
rounds itself (`Frame.innerSched`): `(ρ, σ)` and `Y` from its samples and the coins, `final_c'` and `τ`'s commitment from the
rounds the finish computed (`Frame.innerOwn` fixes the rest); the contract checks only that what leaves
is what the release log reports. `FirewallComputes spec t` names that
assumption: each such value is `spec`'s for its name. It bears on zero knowledge only: the verifier checks every outer
proof whatever the ZK server computed, so no soundness statement reads it.
-/
module

public import Flock.Firewall.Contract

@[expose] public section

namespace Flock.Firewall.Contract

/-- D8 (a): every value the ZK server reports fixing by its own computation is the specified one, `spec what`. The
decisions across sessions (whether to go on, whether to release) are outside this assumption, in Lean or in Rust. -/
def FirewallComputes (spec : String → Option ByteArray) (t : ReleaseLog) : Prop :=
  ∀ what src want sent ps, Item.fixed what src want sent ps ∈ t.items.toList →
    (src = .recheck ∨ src = .shadow ∨ src = .own) → spec what = some want

end Flock.Firewall.Contract
