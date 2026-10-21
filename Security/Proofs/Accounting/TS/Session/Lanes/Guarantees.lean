import Proofs.Accounting.TS.Session.Lanes.Assumptions

/-!
# A leaf's 33 linked sessions realize one unit (the statement)

`Accounting.Guarantees.TS.LeafLanesSound`: under option A a drawn 32 × 32 H1 leaf is proved as its 32 lanes
(`h1zy.PouwH1LaneZy_v1{TILE}`) and their combine (`PouwH1Combine_v1{TILE}`), 33 C-Flock sessions over the leaf's
committed rows and one post-draw partials record. Accepted together, they realize the one-unit proof layer
`Accounting.TS.Lanes.leafLayer`, whose relation is the unit's served checks on the leaf's 32 × 32 cells (`UnitServed`:
`z = A·Bᵀ` and `y = dq(z)` at every cell, the receipt the XOR of its lanes' honest partials), at
`CheckedSessionSound`'s hypotheses and flock-verify's complete check at each of the 33 sessions, and two named
premises: the circuits compute their Definitions at their port rows (`Accounting.Assumptions.TS.LeafChecks`) and the
sessions open one leaf, one partials record and the verifier's registrations (`Accounting.Assumptions.TS.LeafShared`).
Each proof is then accepted with a leaf that fails the unit's checks with probability at most `33 δ`: `δ` a session
from `CheckedSessionSound`, joined by a union bound.
It sits in the proofs package because the sessions do (`reads_exempt`).
-/

namespace Accounting.Guarantees.TS

open Accounting.TS Accounting.TS.Lanes Flock.SoundAtBridge
open Pouw.RelativeSecurity (Paths)
open scoped ENNReal

/-- **A leaf's 33 linked C-Flock sessions realize one unit, fully checked.** For any window whose proof `i` of round
`w` on the path `x` is a leaf proved as its lanes' sessions `sess x w i (some l)` and its combine's `sess x w i none`
(#2909's `RowsSession`), with outcomes `out x w i k`, and the leaf layer
`leafLayer sess out dq part pub receipt leaf right` (accepted where flock-verify `--zk --registered` accepts all 33
sessions, opening the leaf's values `leaf`, its relation the unit's served checks at the path's public inputs `pub` and
receipt `receipt`): given at every one of the 33 sessions `CheckedSessionSound`'s hypotheses (`RowsSession.CR`, the
window's law `Accounting.Assumptions.TS.WindowLaw`, full checking `kd = n`, a circuit that names each port once, every
unit's served roots registered, a port named `ok` in the circuit and in `own`, and
$$n \cdot (\varepsilon_{ks} + \delta_{link} + \delta_{rows} + q_R^2/2^{513}) \le \delta,$$
`δ ≥ 0`), at every one of them flock-verify's complete check (`Flock.Registered.checkComplete`, run when the
`--registered` file sets `complete`: every port of the circuit is a port of `own`, so every port the decodes read is
registered and opens at its registered root), the 33 circuits computing their Definitions at their port rows through
the decodes `decL`, `decC` (`Accounting.Assumptions.TS.LeafChecks`) and the sessions' committed rows being, through the
decodes, one leaf, one partials record `parts`, the receipt and the verifier's registrations
(`Accounting.Assumptions.TS.LeafShared`), each proof is accepted with a leaf that fails the unit's served checks on the
path with probability at most `33 δ`. -/
def LeafLanesSound : Prop :=
  ∀ {Ω : Type} [Fintype Ω] {S nπ K nw : ℕ} {Pub : Type}
    (sess : Paths Ω S → Fin S → Fin nπ → Option (Fin 32) → RowsSession)
    (out : ∀ x w i k, (sess x w i k).Out) (ok : String) (dq : List ℤ → ℤ)
    (part : Pub → Fin 32 → (Fin 4 → Row K) → (Fin 8 → Row K) → Fin nw → BitVec 32)
    (decL : RowsSession → (ℕ → ℕ → Option Bool) → LaneView K nw Pub)
    (decC : RowsSession → (ℕ → ℕ → Option Bool) → CombineView nw)
    (pub : Paths Ω S → Fin S → Fin nπ → Pub) (parts : Paths Ω S → Fin S → Fin nπ → Fin 32 → Fin nw → BitVec 32)
    (receipt : Paths Ω S → Fin S → Fin nπ → Fin nw → BitVec 32) (leaf : Paths Ω S → Fin S → Fin nπ → Leaf K)
    (right : Paths Ω S → (Fin S → Fin nπ → Leaf K) → Prop) (δ : ℝ),
    (∀ x w i k, (sess x w i k).CR) →
    (∀ k, Accounting.Assumptions.TS.WindowLaw (fun x w i => sess x w i k) (fun x w i => out x w i k)) →
    (∀ x w i k, (sess x w i k).kd = (sess x w i k).n) →
    (∀ x w i k (q q' : Fin (sess x w i k).c.ports.size),
      (sess x w i k).c.ports[q].name = (sess x w i k).c.ports[q'].name → q = q') →
    (∀ x w i k u, (sess x w i k).ServedRegistered u) →
    (∀ x w i k, (∃ q : Fin (sess x w i k).c.ports.size, (sess x w i k).c.ports[q].name = ok) ∧
      ∃ pt ∈ (sess x w i k).own, pt.name = ok) →
    (∀ x w i k, _root_.Flock.Registered.checkComplete (sess x w i k).c (sess x w i k).own = .ok ()) →
    (∀ x w i k, ((sess x w i k).n : ℝ≥0∞) * (sess x w i k).bound ≤ ENNReal.ofReal δ) → 0 ≤ δ →
    (∀ x w i, Accounting.Assumptions.TS.LeafChecks dq part (sess x w i) decL decC) →
    Accounting.Assumptions.TS.LeafShared sess decL decC pub parts receipt leaf →
    (leafLayer sess out dq part pub receipt leaf right).SoundAt (33 * δ)

end Accounting.Guarantees.TS
