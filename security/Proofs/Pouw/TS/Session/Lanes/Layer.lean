module

public import Definitions.ResourceAccounting.Served
public import Proofs.Flock.C3.Compose.Honest
public import Proofs.Flock.Soundness.Discharge.HiddenDec.Assemble
public import Proofs.Flock.Soundness.Discharge.HiddenExec.SiteHP
public import Proofs.Flock.VBridge.Family.Registered
public import Proofs.Flock.EndToEndRegistered
public import Definitions.Pouw.TS.ProofLayer
public import Proofs.Flock.SoundAtBridge.Session
public import Proofs.Flock.C3.RegLevels.OpensRegs
public import Proofs.Flock.C3.RegLevels.ValueTie
public import Proofs.Flock.Recursive.ValueSrc
public import Proofs.Flock.Recursive.Registrants
public import Flock.KnownRead
public import PouwWindow.OwnPorts
public import PouwWindow.FullSample
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

/-!
# A leaf's proof layer through its 33 linked Flock sessions (option A)

Under option A a sampled 32 × 32 H1 leaf is proved as 33 statements (`h1zy`, SERVE-REG): its 32 lanes
`PouwH1LaneZy_v1{TILE}`, lane `l` reading its 4 A rows by row slot, its 8 B rows by column slot, the leaf's admit bits,
its whole z/y record, row `l` of the leaf's partials, the salt, position and roots, and the index `l`, and returning
`ok`; and their combine `PouwH1Combine_v1{TILE}`, reading the 32 partials rows and the leaf's receipt and returning `ok`.

The leaf is one unit of the registered circuit: the 33 statements' `ok` words are that unit's
`check` words, and its partials are proof-layer values, no values of the registered circuit. They are committed once,
after the sample, under hiding leaves, read only by the unit's own 33 statements, which all open that one commitment,
and they are what the unit's checks are proved through, not inputs it is credited on. So the unit's served checks
(`UnitServed`) read its A and B rows, its record and its receipt, and no partials row.

This file states, on words, what the 33 statements check and what one unit's served checks are:

* `rowOf`, `colOf`: lane `l = 4g + q`'s row slot `s` is row `g + 8s` and its column slot `t` column
  `2q + 8⌊t/2⌋ + (t mod 2)` (`h1tile.Layout.rows`, `cols` at `R = C = 32`), so its cells are `h1zy.lane_cells`;
* `LaneView`, `LaneOk`: what lane `l`'s statement reads, and its `ok = 1`: its partials row is the honest one
  (`part`, `h1zy.lane_partial`'s words, each u32, at its rows, the public inputs and `l`) and each of its 32 cells has
  `z = Σ_k a_k b_k` and `y = dq [z, s_a, s_w]` at its row's and column's scales (`PouwZyCheck_v1{4, 8}`);
* `CombineView`, `CombineOk`: what the combine reads, and its `ok = 1`: the XOR, in lane order, of the 32 rows is the
  receipt word by word (`xorAll`);
* `PortRow`, `portRead`: a wire assignment at every bit of every unit's port rows, registered or not, which the
  circuits' premise reads (`ResourceAccounting.Assumptions.TS.LeafChecks`);
* `UnitServed`: the leaf's served checks, `ResourceAccounting.Served`'s `ProductOf` and `DequantOf` at every cell (`CellOk`,
  with `dq` the Program's dequant on its words, `PouwZyDequant_v2` today) and its receipt the XOR of its lanes' honest
  partials (in the order `lane_partials` splits the tile unit's receipt);
* `SameTree`, `RowsAgree`, `LeafColl`: the 33 sessions' registrants' rows (`RowsSession.rdU`, each the commit-time
  opening its session registers) are one row's bytes wherever two of them read one registered tree (root, leaf count,
  domain and schema, under any port name) at one position; or two of the
  registrants' records clash (`ReadsX.Clash`: their openings, or hm96's leaves over their rows, are two different
  SHA-512 inputs with one digest, both in hand). Each opening opens its root there (`ReadsX.committed`), so one of the
  two holds (`ResourceAccounting.TS.Lanes.rowsAgree_or_leafColl`);
* `KnownHeld`, `LeafKnownColl`: the known-content tie (at every port the verifier registers at known content:
  each lane's `admit` and `pub`, every statement's `o0`). Where verity-verify-proof's known-content check
  (`Registered.checkKnownAll`) accepts a session's `own` at its known content, the registrant's row at every read of
  such a port is the bytes of the known row at its position (`KnownHeld`), or the registrant's openings exhibit a
  SHA-512 collision (`ResourceAccounting.TS.Held.OkColl` at that port's name and rows; `LeafKnownColl` at one of the 33);
* `leafLayer`: the unit's proof layer. Proof `i` of round `w` is accepted where verity-verify-proof `--zk --registered`
  accepts every one of its 33 sessions (`sess x w i (some l)` lane `l`'s, `sess x w i none` the combine's), its
  known-content check (`known x w i k`, the session's registered file's known content), its complete check
  (`Registered.checkComplete`, run whenever `--registered` is given) and its check that each known port, and each port the file marks
  `read_in_full` (`full x w i k`), is a row of its words whose every bit the circuit reads or returns
  (`KnownRead.check`) included, and the leaf verifier's checks of its registered file and of its session's sample
  (`pouw-leaf-own`: `PouwWindow.Lanes.checkOwn`, and `PouwWindow.Lanes.checkFull`, every unit sampled) pass, opens the
  leaf's values committed before σ_w (`Leaf`: its A and B rows with their scales and its z/y record), and its relation is
  `UnitServed` at the path's public inputs and receipt, which are no leaves of the round (`ProofLayer`), or `LeafColl`
  or `LeafKnownColl` at the proof's sessions.

Words are integers as `ResourceAccounting.Served` reads them (codes s8, z s32, scales and y their encodings); partial and
receipt words are `BitVec 32`. These definitions sit beside the checked layer's, under `reads_exempt`'s
`Proofs.Pouw.TS.Session`.
-/

@[expose] public section

namespace ResourceAccounting.TS.Lanes

open ResourceAccounting.TS Flock.SoundAtBridge FlockSoundness.Discharge.ZkReg
open Pouw.RelativeSecurity (Paths)

/-- An A or B row: its `K` codes, then its scale. -/
structure Row (K : ℕ) where
  codes : Fin K → ℤ
  scale : ℤ

/-- **What lane `l`'s statement reads**: its A rows by row slot, its B rows by column slot, the leaf's record, its
partials row, the public inputs (admit, salt, position, roots), its index and its `ok` word. -/
structure LaneView (K nw : ℕ) (Pub : Type) where
  a : Fin 4 → Row K
  b : Fin 8 → Row K
  z : Fin 32 → Fin 32 → ℤ
  y : Fin 32 → Fin 32 → ℤ
  part : Fin nw → BitVec 32
  pub : Pub
  lane : Fin 32
  ok : Bool

/-- **Wire `g` is a bit of one of unit `u`'s port rows**: its commit string is `u`'s row of some port `q` of the
circuit, an input port or an output row, not a single gate's own string (`Readers.ofRows`). -/
def PortRow (s : RowsSession) (u : Fin s.n) (g : Fin s.Circ.N) : Prop :=
  ∃ q : Fin s.c.ports.size, s.vb.pos g = Sum.inl (u, q)

open Classical in
/-- **An assignment's port rows**: `vals` at every wire of unit `a` that is a bit of one of `a`'s port rows (`PortRow`),
registered or not; `none` elsewhere. Where every port is registered it is `rowsReg`
(`RowsSession.rowsReg_eq_portRead`). -/
noncomputable def portRead (s : RowsSession) (vals : Fin s.Circ.N → Bool) : ℕ → ℕ → Option Bool :=
  fun a b =>
    if h : a < s.n ∧ b < s.Circ.N then
      if (⟨b, h.2⟩ : Fin s.Circ.N) ∈ s.P.io ⟨a, h.1⟩ ∧ PortRow s ⟨a, h.1⟩ ⟨b, h.2⟩ then some (vals ⟨b, h.2⟩)
      else none
    else none

end ResourceAccounting.TS.Lanes
