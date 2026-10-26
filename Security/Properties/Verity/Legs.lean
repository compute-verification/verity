import Properties.Verity.Game

/-!
# What each resource subprotocol's owner proves for its target: the component-level bound

`Properties.Verity.Game` leaves PoUW, PoUS, the bandwidth audit and the network certifier each a target,
`Vocab.Target a`, whose event is `a`'s verdict (`Vocab.Verdict`: the file check, the component's acceptance and every
release it read logged) while its claim fails, outside the break. `ComponentBound a` is the same bound without the
file check: the component accepts its case, every release its check read is in the device's log, the claim fails and
there is no break. It is what each owner states and proves of its component (ONE-16 for PoUW, ONE-10 for PoUS, ONE-11
for the bandwidth audit, ONE-12 for the network certifier, in `note:proofs/20261026T0332Z-report-onethm`); its target
follows by monotonicity (`Vocab.pouwSound_of`, `pousSound_of`, `poubSound_of`, `networkSound_of` in
`Proofs.Properties.Verity.Soundness.Legs`, ONE-18).
-/

namespace Verity

open Accounting Core
open scoped ENNReal

namespace Vocab

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α] (V : Vocab d α)

/-- **Subprotocol `a`'s component-level bound**: at all parameters `θ`, circuit files `F` and commitments `c`, and for
every strategy, the probability that `a`'s compiled component accepts its case in what the prover sent, every release
its check read is in the device's log, its claim fails, and there is no break, is at most its published error. -/
def ComponentBound (a : Audit) : Prop :=
  ∀ (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)),
    Game.prob (fun o => ComponentAccepts a o.sent ∧ (∀ r ∈ V.coinsRead a o.sent, r ∈ o.log) ∧
        ¬ V.Holds θ F c σ a o ∧ ¬ V.Break θ F c o) (V.game θ F c) σ ≤ V.delta θ a

end Vocab

end Verity
