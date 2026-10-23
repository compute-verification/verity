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
(`LeafKnownColl`). Its acceptance includes flock-verify's known-content, complete and known-read checks at each of
the 33 sessions, and it takes `CheckedSessionSound`'s hypotheses at each of them and two named premises: the circuits compute
their Definitions at their port rows, read through the layout's decodes (`Accounting.Assumptions.TS.LeafChecks` at
`Lanes.decL`, `Lanes.decC`; the complete check makes those rows the registered reads) and, where the sessions'
registrants' rows agree, their data ports' rows hold one leaf, one partials record and the receipt
(`Accounting.Assumptions.TS.LeafReads`, the verifier's registrations, which may take the known rows as holding: the
proof derives them from flock-verify's known-content check, #3142's tie). The known content each session is handed is
the path's public inputs' rows (`Accounting.TS.Lanes.knownOf`, `PouwWindow.Lanes.knownRows`), so that check and the
known-read check bind them, unasked: lane `l`'s `pub` row its salt, position, roots and index `l`, its `admit` row the
admit bits, and every `o0` row `ok = 1`. The leaf is one unit of the declared circuit, and the partials are proof-layer values (RT4-LOGIC-30's
ruling). Each proof is then accepted with a leaf that fails the unit's checks, and no collision in hand, with
probability at most `33 δ`: `δ` a session from `CheckedSessionSound`, joined by a union bound.
It sits in the proofs package because the sessions do (`reads_exempt`).
-/

namespace Accounting.Guarantees.TS

open Accounting.TS Accounting.TS.Lanes Flock.SoundAtBridge
open Pouw.RelativeSecurity (Paths)
open scoped ENNReal

/-- **A leaf's 33 linked C-Flock sessions realize one unit, fully checked.** For any window whose proof `i` of round
`w` on the path `x` is a leaf proved as its lanes' sessions `sess x w i (some l)` and its combine's `sess x w i none`
(#2909's `RowsSession`), with outcomes `out x w i k`, and the leaf layer
`leafLayer sess out known dq part pub receipt leaf right` (accepted where flock-verify `--zk --registered` accepts all
33 sessions, its known-content check at `known x w i k = knownOf (pub x w i) (salts x w i k) k`, the rows of the path's
public inputs `pub x w i` (`PouwWindow.Lanes.knownRows`) under any salts, and its complete check
(`Flock.Registered.checkComplete`, run when the `--registered` file sets `complete`: every port of the circuit is a port
of `own`) and its known-read check (`Flock.KnownRead.check`: each known port is a `sha512/row/v2` row of its known rows'
words, a computed row of the unit reads each bit of a known input row and a copy returns each bit of a known output row)
included, opening the leaf's values `leaf`, its relation the unit's served checks at the path's public inputs
`pub` and receipt `receipt`, or a clash of two of the sessions' registrants' records, `LeafColl`, or a collision of a
session's known-content tie, `LeafKnownColl`): given at every one of the 33 sessions `CheckedSessionSound`'s hypotheses (`RowsSession.CR`, the
window's law `Accounting.Assumptions.TS.WindowLaw`, full checking `kd = n`, a circuit that names each port once, every
unit's served roots registered, at least one unit, a port named `ok` in the circuit and in `own`, and
$$n \cdot (\varepsilon_{ks} + \delta_{link} + \delta_{rows} + q_R^2/2^{513}) \le \delta,$$
`δ ≥ 0`), the 33 circuits computing their Definitions at their port rows through the layout's decodes at the served
tile (`Accounting.Assumptions.TS.LeafChecks` at `decL K nw (servedTile K)`, `decC nw (servedTile K)`; under the complete
check every port the decodes read is registered and opens at its registered root) and, wherever the sessions'
registrants' rows agree at one root and position and the known rows hold, their committed rows at the data ports
holding one in-range leaf, one partials record `parts` and the receipt (`Accounting.Assumptions.TS.LeafReads`), each
proof is accepted with a leaf that
fails the unit's served checks on the path, and no clash of its sessions' registrants' records or of their
known-content ties, with probability at most `33 δ`. -/
def LeafLanesSound : Prop :=
  ∀ {Ω : Type} [Fintype Ω] {S nπ K nw : ℕ}
    (sess : Paths Ω S → Fin S → Fin nπ → Option (Fin 32) → RowsSession)
    (out : ∀ x w i k, (sess x w i k).Out) (salts : Paths Ω S → Fin S → Fin nπ → Option (Fin 32) → String → ByteArray)
    (ok : String) (dq : List ℤ → ℤ)
    (part : LanePub → Fin 32 → (Fin 4 → Row K) → (Fin 8 → Row K) → Fin nw → BitVec 32)
    (pub : Paths Ω S → Fin S → Fin nπ → LanePub) (parts : Paths Ω S → Fin S → Fin nπ → Fin 32 → Fin nw → BitVec 32)
    (receipt : Paths Ω S → Fin S → Fin nπ → Fin nw → BitVec 32) (leaf : Paths Ω S → Fin S → Fin nπ → Leaf K)
    (right : Paths Ω S → (Fin S → Fin nπ → Leaf K) → Prop) (δ : ℝ),
    (∀ x w i k, (sess x w i k).CR) →
    (∀ k, Accounting.Assumptions.TS.WindowLaw (fun x w i => sess x w i k) (fun x w i => out x w i k)) →
    (∀ x w i k, (sess x w i k).kd = (sess x w i k).n) →
    (∀ x w i k (q q' : Fin (sess x w i k).c.ports.size),
      (sess x w i k).c.ports[q].name = (sess x w i k).c.ports[q'].name → q = q') →
    (∀ x w i k u, (sess x w i k).ServedRegistered u) → (∀ x w i k, 0 < (sess x w i k).n) →
    (∀ x w i k, (∃ q : Fin (sess x w i k).c.ports.size, (sess x w i k).c.ports[q].name = ok) ∧
      ∃ pt ∈ (sess x w i k).own, pt.name = ok) →
    (∀ x w i k, ((sess x w i k).n : ℝ≥0∞) * (sess x w i k).bound ≤ ENNReal.ofReal δ) → 0 ≤ δ →
    (∀ x w i, Accounting.Assumptions.TS.LeafChecks dq part (sess x w i) (decL K nw (servedTile K))
      (decC nw (servedTile K))) →
    Accounting.Assumptions.TS.LeafReads (servedTile K) sess (fun x w i k => knownOf (pub x w i) (salts x w i k) k) parts
      receipt leaf →
    (leafLayer sess out (fun x w i k => knownOf (pub x w i) (salts x w i k) k) dq part pub receipt leaf right).SoundAt
      (33 * δ)

end Accounting.Guarantees.TS
