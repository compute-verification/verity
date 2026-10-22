import Proofs.Accounting.TS.Session.Lanes.Assumptions
import Proofs.Accounting.TS.Session.Lanes.KnownOf

/-!
# A leaf's 33 linked sessions realize one unit (the statement)

`Accounting.Guarantees.TS.LeafLanesSound`: under option A a drawn 32 × 32 H1 leaf is proved as its 32 lanes
(`h1zy.PouwH1LaneZy_v1{TILE}`) and their combine (`PouwH1Combine_v1{TILE}`), 33 C-Flock sessions over the leaf's
committed rows and one post-draw partials record. Accepted together, they realize the one-unit proof layer
`Accounting.TS.Lanes.leafLayer`, whose relation is the unit's served checks on the leaf's 32 × 32 cells (`UnitServed`:
`z = A·Bᵀ` and `y = dq(z)` at every cell, the receipt the XOR of its lanes' honest partials) unless two of the
sessions' registrants' records exhibit a SHA-512 collision (`LeafColl`) or a session's known-content tie does
(`LeafKnownColl`), at `CheckedSessionSound`'s hypotheses at each
of the 33 sessions and two named premises: the circuits compute their Definitions
(`Accounting.Assumptions.TS.LeafChecks`) and, where the sessions' registrants' rows agree, the decodes read one leaf,
one partials record and the verifier's registrations (`Accounting.Assumptions.TS.LeafReads`, which may take the known
rows as holding: the proof derives them from flock-verify's known-content check, #3142's tie). The known content each
session is handed is the path's public inputs' rows (`Accounting.TS.Lanes.knownOf`, `PouwWindow.Lanes.knownRows`), so
that check binds them: lane `l`'s `pub` row its salt, position, roots and index `l`, its `admit` row the admit bits, and
every `o0` row `ok = 1`. The leaf is one unit of
the declared circuit, and the partials are proof-layer values (RT4-LOGIC-30's ruling). Each proof is then accepted
with a leaf that fails the unit's checks, and no collision in hand, with probability at most `33 δ`: `δ` a session
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
`leafLayer sess out known dq part pub receipt leaf right` (accepted where flock-verify `--zk --registered` accepts all 33
sessions, its known-content check included at `known x w i k = knownOf (pub x w i) (salts x w i k) k`, the rows of the
path's public inputs `pub x w i` (`PouwWindow.Lanes.knownRows`) under any salts, opening the leaf's values `leaf`, its relation the unit's
served checks at the path's public inputs `pub` and receipt `receipt`, or a clash of two of the sessions' registrants'
records, `LeafColl`, or a collision of a session's known-content tie, `LeafKnownColl`): given at every one of the 33 sessions `CheckedSessionSound`'s hypotheses (`RowsSession.CR`, the
window's law `Accounting.Assumptions.TS.WindowLaw`, full checking `kd = n`, a circuit that names each port once, every
unit's served roots registered, a port named `ok` in the circuit and in `own`, and
$$n \cdot (\varepsilon_{ks} + \delta_{link} + \delta_{rows} + q_R^2/2^{513}) \le \delta,$$
`δ ≥ 0`), the 33 circuits computing their Definitions at the decodes `decL`, `decC`
(`Accounting.Assumptions.TS.LeafChecks`) and, wherever the sessions' registrants' rows agree at one root and
position, their committed rows being, through the decodes, one leaf, one partials record `parts`, the receipt and the
verifier's registrations wherever, too, the known rows hold (`Accounting.Assumptions.TS.LeafReads`), each proof is
accepted with a leaf that fails the unit's served checks on the path, and no clash of its sessions' registrants'
records or of their known-content ties, with probability at most
`33 δ`. -/
def LeafLanesSound : Prop :=
  ∀ {Ω : Type} [Fintype Ω] {S nπ K nw : ℕ}
    (sess : Paths Ω S → Fin S → Fin nπ → Option (Fin 32) → RowsSession)
    (out : ∀ x w i k, (sess x w i k).Out) (salts : Paths Ω S → Fin S → Fin nπ → Option (Fin 32) → String → ByteArray)
    (ok : String) (dq : List ℤ → ℤ)
    (part : LanePub → Fin 32 → (Fin 4 → Row K) → (Fin 8 → Row K) → Fin nw → BitVec 32)
    (decL : RowsSession → (ℕ → ℕ → Option Bool) → LaneView K nw LanePub)
    (decC : RowsSession → (ℕ → ℕ → Option Bool) → CombineView nw)
    (pub : Paths Ω S → Fin S → Fin nπ → LanePub) (parts : Paths Ω S → Fin S → Fin nπ → Fin 32 → Fin nw → BitVec 32)
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
    (∀ x w i k, ((sess x w i k).n : ℝ≥0∞) * (sess x w i k).bound ≤ ENNReal.ofReal δ) → 0 ≤ δ →
    (∀ x w i, Accounting.Assumptions.TS.LeafChecks dq part (sess x w i) decL decC) →
    Accounting.Assumptions.TS.LeafReads sess (fun x w i k => knownOf (pub x w i) (salts x w i k) k) decL decC pub parts
      receipt leaf →
    (leafLayer sess out (fun x w i k => knownOf (pub x w i) (salts x w i k) k) dq part pub receipt leaf right).SoundAt
      (33 * δ)

end Accounting.Guarantees.TS
