/-
# The firewall's named assumption (D8 (a), zero knowledge only)

The contract derives what the schedule fixes, and checks every commitment, pad and coin itself. What the firewall fixes by
its own computation it only reports: its level-0 recheck commitment (`By.recheck`), its opening of that commitment
(`By.shadow`), and the values it computes from its pads and the values that passed, `final_c'` and the inner proof
(`By.own`). Today's Rust computes them (`firewall.rs`: the recheck's `commit_zk`, `ShadowLink`, `zkv::replay`, `prove_inner`),
and the contract checks only that what leaves is what the transcript reports. `FirewallComputes spec t` names that
assumption: each such value is `spec`'s for its name. It bears on zero knowledge only: the verifier checks every outer
proof whatever the firewall computed, so no soundness statement reads it.
-/
import Flock.Firewall.Contract

namespace Flock.Firewall.Contract

/-- D8 (a): every value the firewall reports fixing by its own computation is the specified one, `spec what`. The
decisions across sessions (whether to go on, whether to release) are outside this assumption, in Lean or in Rust. -/
def FirewallComputes (spec : String → Option ByteArray) (t : Transcript) : Prop :=
  ∀ what src want sent ps, Item.fixed what src want sent ps ∈ t.items.toList →
    (src = .recheck ∨ src = .shadow ∨ src = .own) → spec what = some want

end Flock.Firewall.Contract
