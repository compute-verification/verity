import Proofs.Accounting.TS.Session.Lanes.Assumptions
import Proofs.Accounting.TS.Session.Lanes.KnownOf
import Proofs.Accounting.TS.Session.Lanes.Committed
import PouwWindow.OwnPorts

/-!
# A leaf's 33 linked sessions realize one unit (the statement)

`Accounting.Guarantees.TS.LeafLanesSound`: under option A a drawn 32 × 32 H1 leaf is proved as its 32 lanes
(`h1zy.PouwH1LaneZy_v1{TILE}`) and their combine (`PouwH1Combine_v1{TILE}`), 33 C-Flock sessions over the leaf's
committed rows and one post-draw partials record. Accepted together, they realize the one-unit proof layer
`Accounting.TS.Lanes.leafLayer`, whose relation is the unit's served checks on the leaf's 32 × 32 cells (`UnitServed`:
`z = A·Bᵀ` and `y = dq(z)` at every cell, the receipt the XOR of its lanes' honest partials) unless two of the
sessions' registrants' records exhibit a SHA-512 collision (`LeafColl`) or a session's known-content tie does
(`LeafKnownColl`). Its acceptance includes flock-verify's known-content, complete and known-read checks at each of
the 33 sessions, and the leaf verifier's check of each session's registered file (`PouwWindow.Lanes.checkOwn`, run by
`pouw-leaf-own`), and it takes `CheckedSessionSound`'s hypotheses at each of them, but port uniqueness and the `ok` port,
which that acceptance gives (the known-read check's names, at known content naming `o0`), and one named premise: the
circuits compute their Definitions at their port rows, read through the layout's decodes
(`Accounting.Assumptions.TS.LeafChecks` at `Lanes.decL`, `Lanes.decC`; the complete check makes those rows the
registered reads). The leaf the layer opens is the one the lanes' committed rows decode to, and the receipt it checks
the one the combine's committed row decodes to (`Accounting.TS.Lanes.leafOf`, `receiptOf`): each A row as the lane
`4 (r mod 8)` reads it, each B row as the lane `⌊(j mod 8)/2⌋` reads it, each cell of the record as the one lane whose
cell it is, so no premise says which leaf the rows hold. The four lanes that read one A row, or one B row, read it in
full (`PouwWindow.Lanes.readFull`) at one registered tree, so they decode it alike up to a clash of their registrants'
records (`LeafColl`). The known content each session is handed is
the path's public inputs' rows (`Accounting.TS.Lanes.knownOf`, `PouwWindow.Lanes.knownRows`), so that check and the
known-read check bind them, unasked: lane `l`'s `pub` row its salt, position, roots and index `l`, its `admit` row the
admit bits, and every `o0` row `ok = 1`. That each session's registered file holds the layout's data ports
(`PouwWindow.Lanes.ownOf` at one tree per value, `roots`) is not asked either: acceptance checks it, at any roots, and
`pouw-leaf-own` takes them from the accepted registration and produced record alone (`PouwWindow.Lanes.roots?`), so no
premise is left on how the file was built. Each file marks its reads of the A, B and partials rows `read_in_full`
(`PouwWindow.Lanes.readFull`), which the known-read check holds to one row of the registered words: so lane `l`'s
partials row and the combine's `p<l>` are one row, the partials record is the rows the 33 statements read, and nothing
of it is asked. The leaf is one unit of the declared circuit, and the partials are proof-layer values (RT4-LOGIC-30's
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
`leafLayer sess out known full want dq part pub receiptOf leafOf right` (accepted where flock-verify `--zk --registered` accepts all
33 sessions, its known-content check at `known x w i k = knownOf (pub x w i) (salts x w i k) k`, the rows of the path's
public inputs `pub x w i` (`PouwWindow.Lanes.knownRows`) under any salts, and its complete check
(`Flock.Registered.checkComplete`, run when the `--registered` file sets `complete`: every port of the circuit is a port
of `own`) and its known-read check (`Flock.KnownRead.check`: each known port, and each port the file marks
`read_in_full`, `full x w i k = PouwWindow.Lanes.readFull (servedTile K) k`, the statement's reads of the partials, is
a `sha512/row/v2` row of its known rows' or registered row's words, a computed row of the unit reads each bit of such
an input row and a copy returns each bit of such an output row) included, and the leaf verifier's check of the file
(`PouwWindow.Lanes.checkOwn`: the file's first port of each name of `want x w i k = PouwWindow.Lanes.ownOf (roots x w i)
(servedTile K) k`, the statement's data ports at the layout's rows of one tree per value, any roots `roots`, is that
port), opening the leaf its lanes' committed rows decode to (`Accounting.TS.Lanes.leafOf` at the served tile: each A
row as lane `4 (r mod 8)` decodes it, each B row as lane `⌊(j mod 8)/2⌋` does, each cell of the record as the lane whose
cell it is, at the registrants' commit-time openings), its relation the unit's served checks at the path's public
inputs `pub` and the receipt the combine's committed row decodes to (`Accounting.TS.Lanes.receiptOf`), or a clash of two
of the sessions' registrants' records, `LeafColl`, or a collision of a session's known-content tie, `LeafKnownColl`):
given at every one of the 33 sessions `CheckedSessionSound`'s hypotheses (`RowsSession.CR`, the
window's law `Accounting.Assumptions.TS.WindowLaw`, full checking `kd = n`, every unit's served roots registered, at
least one unit, and
$$n \cdot (\varepsilon_{ks} + \delta_{link} + \delta_{rows} + q_R^2/2^{513}) \le \delta,$$
`δ ≥ 0`), the 33 circuits computing
their Definitions at their port rows through the layout's decodes at the served
tile (`Accounting.Assumptions.TS.LeafChecks` at `decL K nw (servedTile K)`, `decC nw (servedTile K)`; under the complete
check every port the decodes read is registered and opens at its registered root), each
proof is accepted with a committed leaf that
fails the unit's served checks on the path, and no clash of its sessions' registrants' records or of their
known-content ties, with probability at most `33 δ`. That each circuit names each port once and has a port `o0` that
`own` registers is not asked: an accepted proof's known-read check names every port of its known content, `o0`
among them, and its known-content check registers it (`Flock.KnownRead.namesOk`, `Flock.Registered.checkKnownAll`). -/
def LeafLanesSound : Prop :=
  ∀ {Ω : Type} [Fintype Ω] {S nπ K nw : ℕ}
    (sess : Paths Ω S → Fin S → Fin nπ → Option (Fin 32) → RowsSession)
    (out : ∀ x w i k, (sess x w i k).Out) (salts : Paths Ω S → Fin S → Fin nπ → Option (Fin 32) → String → ByteArray)
    (dq : List ℤ → ℤ)
    (part : LanePub → Fin 32 → (Fin 4 → Row K) → (Fin 8 → Row K) → Fin nw → BitVec 32)
    (pub : Paths Ω S → Fin S → Fin nπ → LanePub)
    (roots : Paths Ω S → Fin S → Fin nπ → String → _root_.Flock.Registered.Port)
    (right : Paths Ω S → (Fin S → Fin nπ → Leaf K) → Prop) (δ : ℝ),
    (∀ x w i k, (sess x w i k).CR) →
    (∀ k, Accounting.Assumptions.TS.WindowLaw (fun x w i => sess x w i k) (fun x w i => out x w i k)) →
    (∀ x w i k, (sess x w i k).kd = (sess x w i k).n) →
    (∀ x w i k u, (sess x w i k).ServedRegistered u) → (∀ x w i k, 0 < (sess x w i k).n) →
    (∀ x w i k, ((sess x w i k).n : ℝ≥0∞) * (sess x w i k).bound ≤ ENNReal.ofReal δ) → 0 ≤ δ →
    (∀ x w i, Accounting.Assumptions.TS.LeafChecks dq part (sess x w i) (decL K nw (servedTile K))
      (decC nw (servedTile K))) →
    (leafLayer sess out (fun x w i k => knownOf (pub x w i) (salts x w i k) k)
      (fun _ _ _ k => PouwWindow.Lanes.readFull (servedTile K) (k.map Fin.val))
      (fun x w i k => PouwWindow.Lanes.ownOf (roots x w i) (servedTile K) (k.map Fin.val)) dq part pub
      (fun x w i => receiptOf nw (servedTile K) (sess x w i)) (fun x w i => leafOf K (servedTile K) (sess x w i))
      right).SoundAt
      (33 * δ)

end Accounting.Guarantees.TS
