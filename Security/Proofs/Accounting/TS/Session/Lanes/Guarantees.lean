import Proofs.Accounting.TS.Session.Lanes.Assumptions
import Proofs.Accounting.TS.Session.Lanes.Lane
import Definitions.Pouw.H1Zy.Dequant
import Proofs.Accounting.TS.Session.Lanes.KnownOf
import Proofs.Accounting.TS.Session.Lanes.Committed
import Proofs.Accounting.TS.Session.Lanes.Canon
import PouwWindow.OwnPorts
import PouwWindow.FullDraw

/-!
# A leaf's 33 linked sessions realize one unit (the statement)

`Accounting.Guarantees.TS.LeafLanesSound`: under option A a drawn 32 × 32 H1 leaf is proved as its 32 lanes
(`h1zy.PouwH1LaneZy_v1{TILE}`) and their combine (`PouwH1Combine_v1{TILE}`), 33 C-Flock sessions over the leaf's
committed rows and one post-draw partials record. Accepted together, they realize the one-unit proof layer
`Accounting.TS.Lanes.leafLayer`, whose relation is the unit's served checks on the leaf's 32 × 32 cells (`UnitServed`:
`z = A·Bᵀ` and `y = dq(z)` at every cell, the receipt the XOR of its lanes' honest partials) unless two of the
sessions' registrants' records exhibit a SHA-512 collision (`LeafColl`) or a session's known-content tie does
(`LeafKnownColl`). Its acceptance includes flock-verify's known-content, complete and known-read checks at each of
the 33 sessions, and the leaf verifier's checks of each session's registered file (`PouwWindow.Lanes.checkOwn`) and
of its draw (`PouwWindow.Lanes.checkFull`: every unit of a non-empty population) and of the file's reads
(`PouwWindow.Lanes.checkOneRead`: one read at each port), all run by `pouw-leaf-own`. Each session is
`RowsSession.canon` of a session and its registrants' openings (`RowsSession.Openings`): its registered reads are built
from the verifier's own `own`, every circuit port's row that `own` names read through `own`'s port at its one position,
and only the rows, salts and paths the registrants committed come from them. It takes `CheckedSessionSound`'s
hypotheses at each session, but port uniqueness and the `ok` port, which that acceptance gives (the known-read check's
names, at known content naming `o0`), full checking (`kd = n`) and a unit (`0 < n`), which the draw check gives, and
registered served roots (`RowsSession.ServedRegistered`), which the reads built from `own` and the one-read check
give, and two named premises: the
lanes' and the combine's circuits compute their pieces' compositions on bits, at their port rows, read through the
layout's decodes (`Accounting.TS.Lanes.LaneLowers` and `CombineLowers` at `Lanes.decL`, `Lanes.decC`; the lane's dequant
is its Definition's word view, `Pouw.H1Zy.zyDequant` (`PouwZyDequant_v1`), and its chain words are `h1tile._chain`'s,
`Accounting.TS.Lanes.laneChain K nw` (`Pouw.H1Zy.h1Chain K` at the `pub` row's seeds); the complete check makes those
rows the registered reads). That the pieces'
lowerings compose to the Definitions is proved (`Accounting.TS.Lanes.laneComputes_of_lowers`,
`combineComputes_of_lowers`, `Pouw.H1Zy.laneBits_bv`, `combineBits_bitsOf`), and that
each Definition's `ok = 1` means the lane's or the combine's check
(`Accounting.TS.Lanes.leafChecks_of_computes`, which gives `Accounting.Assumptions.TS.LeafChecks`), at a `K`
the lanes bind at (`32 ∣ K`, `K < 2^17`), which the leaf verifier's tile check, part of the acceptance, gives
(`PouwWindow.Lanes.Tile.check`, `Accounting.TS.Lanes.k_of_check`). The leaf the layer opens is the one the lanes' committed rows decode to, and the receipt it checks
the one the combine's committed row decodes to (`Accounting.TS.Lanes.leafOf`, `receiptOf`): each A row as the lane
`4 (r mod 8)` reads it, each B row as the lane `⌊(j mod 8)/2⌋` reads it, each cell of the record as the one lane whose
cell it is, so no premise says which leaf the rows hold. The four lanes that read one A row, or one B row, read it in
full (`PouwWindow.Lanes.readFull`) at one registered tree, so they decode it alike up to a clash of their registrants'
records (`LeafColl`). The known content each session is handed is
the path's public inputs' rows (`Accounting.TS.Lanes.knownOf`, `PouwWindow.Lanes.knownRows`), so that check and the
known-read check bind them, unasked: lane `l`'s `pub` row its salt, position, roots and index `l`, its `admit` row the
admit bits, and every `o0` row `ok = 1`. That each session's registered file holds the layout's data ports
(`PouwWindow.Lanes.ownOf` at one table per value, `roots`, each value placed in its table from a base row, `place`) is
not asked either: acceptance checks it, at any roots and any placement, and `pouw-leaf-own` takes the roots from the
accepted registration and produced record alone (`PouwWindow.Lanes.roots?`) and, for a unit of a window Program, the
placement from the window's committed descriptor and round log (`PouwCheck.LeafOwn.placeOf`), so no premise is left
on how the file was built. Each file marks its reads of the A, B and partials rows `read_in_full`
(`PouwWindow.Lanes.readFull`), which the known-read check holds to one row of the registered words: so lane `l`'s
partials row and the combine's `p<l>` are one row, the partials record is the rows the 33 statements read, and nothing
of it is asked. The leaf is one unit of the declared circuit, and the partials are proof-layer values.
Each proof is then accepted with a leaf that fails the unit's checks, and no collision in hand, with
probability at most `33 δ`: `δ` a session from `CheckedSessionSound`, joined by a union bound.
It sits in the proofs package because the sessions do (`reads_exempt`).
-/

namespace Accounting.Guarantees.TS

open Accounting.TS Accounting.TS.Lanes Flock.SoundAtBridge
open Pouw.RelativeSecurity (Paths)
open scoped ENNReal

/-- **A leaf's 33 linked C-Flock sessions realize one unit, fully checked.** For any window whose proof `i` of round
`w` on the path `x` is a leaf proved as its lanes' sessions `sess x w i (some l)` and its combine's `sess x w i none`,
each a session `sess₀ x w i k` (a `RowsSession`) at its registrants' openings `op x w i k`, its registered reads
built from `own` (`RowsSession.canon`: each unit reads its row of every circuit port `own` names, through `own`'s first
port of that name at that port's one position, the registrants committing only the rows, salts and paths, each opening
`own`'s root there), with outcomes `out x w i k`, and the leaf layer
`leafLayer sess out known full want (dqOf Pouw.H1Zy.zyDequant) (partOf (laneChain K nw)) pub receiptOf leafOf right` (accepted where flock-verify `--zk --registered` accepts all
33 sessions, its known-content check at `known x w i k = knownOf (pub x w i) (salts x w i k) k`, the rows of the path's
public inputs `pub x w i` (`PouwWindow.Lanes.knownRows`) under any salts, and its complete check
(`Flock.Registered.checkComplete`, run on every statement whenever `--registered` is given: every port of the circuit is
a port of `own`) and its known-read check (`Flock.KnownRead.check`: each known port, and each port the file marks
`read_in_full`, `full x w i k = PouwWindow.Lanes.readFull (servedTile K) k`, the statement's reads of the partials, is
a `sha512/row/v2` row of its known rows' or registered row's words, a computed row of the unit reads each bit of such
an input row and a copy returns each bit of such an output row) included, and the leaf verifier's check of the file
(`PouwWindow.Lanes.checkOwn`: the file's first port of each name of `want x w i k = PouwWindow.Lanes.ownOf (roots x w i)
(place x w i) (servedTile K) k`, the statement's data ports at the layout's rows of one table per value, each value `v`
placed in the table `(place x w i v).1` from the base row `(place x w i v).2` and the layout's row read from that base,
any roots `roots` and any placement `place`, is that port), and the leaf verifier's check of the session's draw (`PouwWindow.Lanes.checkFull`, which `pouw-leaf-own` runs
on the session's record: `subset` of every unit of a non-empty population), and of the file's reads
(`PouwWindow.Lanes.checkOneRead`, which `pouw-leaf-own` runs too: every port of the file has one read), and of the
tile (`PouwWindow.Lanes.Tile.check` at 32 × 32 and `K`, which `pouw-leaf-own` runs in `lane?` and `combine?`: `32 ∣ K`
and `K < 2^17` among others), opening the leaf its lanes' committed rows
decode to (`Accounting.TS.Lanes.leafOf` at the served tile: each A
row as lane `4 (r mod 8)` decodes it, each B row as lane `⌊(j mod 8)/2⌋` does, each cell of the record as the lane whose
cell it is, at the registrants' commit-time openings), its relation the unit's served checks at the path's public
inputs `pub` and the receipt the combine's committed row decodes to (`Accounting.TS.Lanes.receiptOf`), or a clash of two
of the sessions' registrants' records, `LeafColl`, or a collision of a session's known-content tie, `LeafKnownColl`):
given at every one of the 33 sessions `CheckedSessionSound`'s hypotheses (`RowsSession.CR`, the
window's law `Accounting.Assumptions.TS.WindowLaw`, and
$$n \cdot (\varepsilon_{ks} + \delta_{link} + \delta_{rows} + q_R^2/2^{513}) \le \delta,$$
`δ ≥ 0`), the 33 circuits computing
their pieces' compositions on bits, at their port rows through the layout's decodes at
the served tile (`Accounting.TS.Lanes.LaneLowers` at `servedTile K`: the s8 multiply's Baugh–Wooley steps and clamp,
`PouwZyCell_v2`, `PouwMuxBit_v2`, `PouwMask32_v2` and `PouwEq32_v2` at their lowerings, `Pouw.H1Zy.laneBits`, at the
lanes' chain words `laneChain K nw` (`Pouw.H1Zy.h1Chain K`, the fx1 chain over the noise from σ_J keyed from σ_R, at
the `pub` row's salt, position and roots) and the dequant's Definition `Pouw.H1Zy.zyDequant` (`PouwZyDequant_v1`) on bits; and `Accounting.TS.Lanes.CombineLowers` at `decC nw (servedTile K)`: `PouwXor32_v2`, `PouwEq32_v2` and
`PouwAndBit_v2` at their lowerings, `Pouw.H1Zy.combineBits`; under the complete check every port the
decodes read is registered and opens at its registered root), with the unit's served checks at the Definition's own
dequant and partials (`dqOf Pouw.H1Zy.zyDequant`, `partOf (laneChain K nw)`), each
proof is accepted with a committed leaf that
fails the unit's served checks on the path, and no clash of its sessions' registrants' records or of their
known-content ties, with probability at most `33 δ`. That each circuit names each port once and has a port `o0` that
`own` registers is not asked: an accepted proof's known-read check names every port of its known content, `o0`
among them, and its known-content check registers it (`Flock.KnownRead.namesOk`, `Flock.Registered.checkKnownAll`).
Nor are full checking (`kd = n`) and a unit (`0 < n`): the draw check accepts only a record whose draw is `subset` of
its whole non-empty population, which the session's custody (`RecordCustodyZKJ`) makes the verifier's draw
`subset kd` over `n` units. Nor is every unit's served roots registered (`RowsSession.ServedRegistered`): the reads
built from `own` are its rows of the ports `own` names, through `own`'s ports, and with one read at each port
(`checkOneRead`) the position a port gives a unit at any draw is its one position
(`RowsSession.servedRegistered_canon`). -/
def LeafLanesSound : Prop :=
  ∀ {Ω : Type} [Fintype Ω] {S nπ K nw : ℕ}
    (sess₀ : Paths Ω S → Fin S → Fin nπ → Option (Fin 32) → RowsSession)
    (op : ∀ x w i k, (sess₀ x w i k).Openings),
    let sess := fun x w i k => (sess₀ x w i k).canon (op x w i k)
    ∀ (out : ∀ x w i k, (sess x w i k).Out)
    (salts : Paths Ω S → Fin S → Fin nπ → Option (Fin 32) → String → ByteArray)
    (pub : Paths Ω S → Fin S → Fin nπ → LanePub)
    (roots : Paths Ω S → Fin S → Fin nπ → String → _root_.Flock.Registered.Port)
    (place : Paths Ω S → Fin S → Fin nπ → String → String × ℕ)
    (right : Paths Ω S → (Fin S → Fin nπ → Leaf K) → Prop) (δ : ℝ),
    (∀ x w i k, (sess x w i k).CR) →
    (∀ k, Accounting.Assumptions.TS.WindowLaw (fun x w i => sess x w i k) (fun x w i => out x w i k)) →
    (∀ x w i k, ((sess x w i k).n : ℝ≥0∞) * (sess x w i k).bound ≤ ENNReal.ofReal δ) → 0 ≤ δ →
    (∀ x w i, LaneLowers (servedTile K) (laneChain K nw) Pouw.H1Zy.zyDequant (sess x w i)) →
    (∀ x w i, CombineLowers (sess x w i) (decC nw (servedTile K))) →
    (leafLayer sess out (fun x w i k => knownOf (pub x w i) (salts x w i k) k)
      (fun _ _ _ k => PouwWindow.Lanes.readFull (servedTile K) (k.map Fin.val))
      (fun x w i k => PouwWindow.Lanes.ownOf (roots x w i) (place x w i) (servedTile K) (k.map Fin.val))
      (dqOf Pouw.H1Zy.zyDequant) (partOf (laneChain K nw)) pub
      (fun x w i => receiptOf nw (servedTile K) (sess x w i)) (fun x w i => leafOf K (servedTile K) (sess x w i))
      right).SoundAt
      (33 * δ)

end Accounting.Guarantees.TS
