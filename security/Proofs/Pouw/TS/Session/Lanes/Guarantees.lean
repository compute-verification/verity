module

public import Proofs.Pouw.TS.Session.Group.Premises
public import Proofs.Pouw.TS.Session.Lanes.Lane
public import Definitions.Pouw.H1Zy.Dequant
public import Proofs.Pouw.TS.Session.Lanes.Committed
public import Proofs.Pouw.TS.Session.Lanes.Canon
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init

/-! The statement that a sampled 32 × 32 H1 leaf proved as its 32 lanes and their combine, 33 linked Flock sessions, is
accepted with a leaf that fails the unit's checks and no collision in hand with probability at most `33 δ`. -/

@[expose] public section

namespace ResourceAccounting.Guarantees.TS

open ResourceAccounting.TS ResourceAccounting.TS.Lanes Flock.SoundAtBridge
open Pouw.RelativeSecurity (Paths)
open scoped ENNReal

/-- **A leaf's 33 linked Flock sessions realize one unit, fully checked.** For any window whose proof `i` of round
`w` on the path `x` is a leaf proved as its lanes' sessions `sess x w i (some l)` and its combine's `sess x w i none`,
each a session `sess₀ x w i k` (a `RowsSession`) at its registrants' openings `op x w i k`, its registered reads
built from `own` (`RowsSession.canon`: each unit reads its row of every circuit port `own` names, through `own`'s first
port of that name at that port's one position, the registrants committing only the rows, salts and paths, each opening
`own`'s root there), with outcomes `out x w i k`, and the leaf layer
`leafLayer sess out known full want (dqOf Pouw.H1Zy.zyDequant) (partOf (laneChain K nw)) pub receiptOf leafOf right` (accepted where verity-verify-proof `--zk --registered` accepts all
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
any roots `roots` and any placement `place`, is that port), and the leaf verifier's check of the session's sample (`PouwWindow.Lanes.checkFull`, which `pouw-leaf-own` runs
on the session's record: `subset` of every unit of a non-empty population), and of the file's reads
(`PouwWindow.Lanes.checkOneRead`, which `pouw-leaf-own` runs too: every port of the file has one read), and of the
tile (`PouwWindow.Lanes.Tile.check` at 32 × 32 and `K`, which `pouw-leaf-own` runs in `lane?` and `combine?`: `32 ∣ K`
and `K < 2^17` among others), opening the leaf its lanes' committed rows
decode to (`ResourceAccounting.TS.Lanes.leafOf` at the served tile: each A
row as lane `4 (r mod 8)` decodes it, each B row as lane `⌊(j mod 8)/2⌋` does, each cell of the record as the lane whose
cell it is, at the registrants' commit-time openings), its relation the unit's served checks at the path's public
inputs `pub` and the receipt the combine's committed row decodes to (`ResourceAccounting.TS.Lanes.receiptOf`), or a clash of two
of the sessions' registrants' records, `LeafColl`, or a collision of a session's known-content tie, `LeafKnownColl`):
given at every one of the 33 sessions `CheckedSessionSound`'s hypotheses as the named premises
`ResourceAccounting.Premises.TS.SessionsCR` (`RowsSession.CR`) and `ResourceAccounting.Premises.TS.ExecutionPhaseDesigns` (the window's sampling design
`ResourceAccounting.Assumptions.TS.ExecutionPhaseDesign` at each statement), and
$$n \cdot (\varepsilon_{ks} + \delta_{link} + \delta_{rows} + q_R^2/2^{513}) \le \delta,$$
`δ ≥ 0`), the 33 circuits computing
their pieces' compositions on bits, at their port rows through the layout's decodes at
the served tile (`ResourceAccounting.TS.Lanes.LaneLowers` at `servedTile K`: the s8 multiply's Baugh–Wooley steps and clamp,
`PouwZyCell_v2`, `PouwMuxBit_v2`, `PouwMask32_v2` and `PouwEq32_v2` at their lowerings, `Pouw.H1Zy.laneBits`, at the
lanes' chain words `laneChain K nw` (`Pouw.H1Zy.h1Chain K`, the fx1 chain over the noise from σ_J keyed from σ_R, at
the `pub` row's salt, position and roots) and the dequant's Definition `Pouw.H1Zy.zyDequant` (`PouwZyDequant_v1`) on bits; and `ResourceAccounting.TS.Lanes.CombineLowers` at `decC nw (servedTile K)`: `PouwXor32_v2`, `PouwEq32_v2` and
`PouwAndBit_v2` at their lowerings, `Pouw.H1Zy.combineBits`; under the complete check every port the
decodes read is registered and opens at its registered root), with the unit's served checks at the Definition's own
dequant and partials (`dqOf Pouw.H1Zy.zyDequant`, `partOf (laneChain K nw)`), each
proof is accepted with a committed leaf that
fails the unit's served checks on the path, and no clash of its sessions' registrants' records or of their
known-content ties, with probability at most `33 δ`. That each circuit names each port once and has a port `o0` that
`own` registers is not asked: an accepted proof's known-read check names every port of its known content, `o0`
among them, and its known-content check registers it (`Flock.KnownRead.namesOk`, `Flock.Registered.checkKnownAll`).
Nor are full checking (`kd = n`) and a unit (`0 < n`): the sample check accepts only a record whose sample is `subset` of
its whole non-empty population, which the session's custody (`RecordCustodyZKJ`) makes the verifier's sample
`subset kd` over `n` units. Nor is every unit's served roots registered (`RowsSession.ServedRegistered`): the reads
built from `own` are its rows of the ports `own` names, through `own`'s ports, and with one read at each port
(`checkOneRead`) the position a port gives a unit at any sample is its one position
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
    ResourceAccounting.Premises.TS.SessionsCR sess →
    ResourceAccounting.Premises.TS.ExecutionPhaseDesigns sess out →
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

end ResourceAccounting.Guarantees.TS
