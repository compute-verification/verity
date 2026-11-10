module

public import Mathlib.Data.List.SplitLengths
public import Definitions.Claims.Sampling
public import Definitions.Flock.Circuit
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
# Flock's session as a sampling statement

The Flock session's units, read as the sampling claims' statement, partition, commitment and witness
(`Claims.Sampling`), so a game-form bound over the live audit can conclude in the claims' relation `RelS`:

* `progOf C κ`: the circuit's gates lowered to the Boolean profile's (`BGate`), in order; a row
  `(⊕ a) ∧ (⊕ b)` is three gates (the two XORs, then their AND), every other gate one. An input gate the constants
  `κ` fix to `b` is the constant `.const b`, public circuit structure, which a proof block reads with no opening. Wire
  `g` sits at `posOf C g`, its last gate.
* `partOf P κ ns`: one class, the population, in strata of sizes `ns`, each a run of consecutive slots
  (`List.splitLengths`). Slot `u` (`blockOf`) is unit `u`'s lowered gates (`unitGates`), then the position of each
  input row `u` owns (`insOf`): an input gate the constants `κ` don't fix, whose first reader is `u` (`owner`). **An
  input row is checked with the unit that owns it**, as the deployed verifier opens it: its session (`HmRow.drawn`)
  opens only the rows its sampled proof blocks read, and registration opens none. An input no unit reads is in no slot.
* `designOf r`: the population sampled by the one-stage design `r`.
* `stmtOf P κ r desc q roots`: the statement of the program at constants `κ`, the query `q`, that design and the
  roots.
* `commitOf pos bit val com cs`: an opening at wire `g`'s position passes when some opening of the registered commit
  string at `g`'s position (`cs (pos g)`) carries its value on `g` (`bit (val op) g`), as `RightRegZ` reads the
  registration's commitments. It reads neither the roots nor `H`.
* `witOf C X`: an opening of every wire at its position, of its value in `C`'s evaluation on `X`'s inputs.
-/

@[expose] public section

namespace Flock.Bridge

open FlockSoundness.Audit

/-- The Boolean-profile gates one Flock gate lowers to: three for a row, one for any other. -/
def gateSize {N : ℕ} : Verity.Boolean.Op N → ℕ
  | .row _ _ => 3
  | _ => 1

/-- Where wire `g`'s lowered gates start: the lowered sizes of the gates before it. -/
def off (C : Verity.Boolean.Circuit) (g : ℕ) : ℕ :=
  (((List.finRange C.N).filter fun h => decide (h.val < g)).map fun h => gateSize (C.op h)).sum

/-- **Wire `g`'s position** in the lowered program: its last lowered gate. -/
def posOf (C : Verity.Boolean.Circuit) (g : Fin C.N) : ℕ := off C g.val + gateSize (C.op g) - 1

/-- Gate `g` lowered: a row to the XOR of its first list, the XOR of its second, and their AND; an input the
constants `κ` fix to `b` to `.const b`. -/
def lower (C : Verity.Boolean.Circuit) (κ : Fin C.N → Option Bool) (g : Fin C.N) : List Claims.Sampling.BGate :=
  match C.op g with
  | .input => match κ g with
    | some b => [.const b]
    | none => [.input]
  | .and a b => [.and [posOf C a, posOf C b]]
  | .xor a b => [.xor [posOf C a, posOf C b]]
  | .not a => [.not (posOf C a)]
  | .row a b => [.xor (a.map (posOf C)), .xor (b.map (posOf C)), .and [off C g.val, off C g.val + 1]]

/-- **The circuit as a Boolean-profile program** at constants `κ`: its gates lowered in order, returning its outputs'
positions. -/
def progOf (C : Verity.Boolean.Circuit) (κ : Fin C.N → Option Bool) : Claims.Sampling.Prog where
  gates := (List.finRange C.N).flatMap (lower C κ)
  outs := ((List.finRange C.N).filter fun g => decide (g ∈ C.outputs)).map (posOf C)

/-- Unit `u`'s lowered gates. -/
def unitGates {C : Verity.Boolean.Circuit} {n : ℕ} (P : Partition C n) (u : Fin n) : List ℕ :=
  (List.finRange C.N).flatMap fun g =>
    if P.unit g = some u then (List.range (gateSize (C.op g))).map (off C g.val + ·) else []

/-- **Wire `g`'s owner**: the first unit that reads it; `none` when no unit does. -/
def owner {C : Verity.Boolean.Circuit} {n : ℕ} (P : Partition C n) (g : Fin C.N) : Option (Fin n) :=
  (List.finRange n).find? fun u => decide (∃ h, P.unit h = some u ∧ g ∈ (C.op h).args)

/-- **Unit `u`'s inputs**: the input gates the constants `κ` don't fix whose owner is `u`, in gate order. -/
def insOf {C : Verity.Boolean.Circuit} {n : ℕ} (P : Partition C n) (κ : Fin C.N → Option Bool) (u : Fin n) : List (Fin C.N) :=
  (List.finRange C.N).filter fun g => (C.op g).isInput && (κ g).isNone && decide (owner P g = some u)

/-- **Unit `u`'s proof block**: its lowered gates `unitGates P u`, then the position of each input it owns. -/
def blockOf {C : Verity.Boolean.Circuit} {n : ℕ} (P : Partition C n) (κ : Fin C.N → Option Bool) (u : Fin n) : List ℕ :=
  unitGates P u ++ (insOf P κ u).map (posOf C)

/-- **The partition**: the population, in strata of sizes `ns`, each a run of consecutive slots, with slot `u` unit
`u`'s proof block `blockOf P κ u`. -/
def partOf {C : Verity.Boolean.Circuit} {n : ℕ} (P : Partition C n) (κ : Fin C.N → Option Bool) (ns : List ℕ) :
    Claims.Sampling.Partition :=
  ⟨[ns.splitLengths ((List.finRange n).map (blockOf P κ))]⟩

/-- **The design**: the population sampled by the one-stage design `r`. -/
def designOf (r : Claims.Sampling.OneStage) : Claims.Sampling.Design := ⟨[r]⟩

/-- **The statement**: `progOf`'s program at constants `κ` under the query `q` and the descriptor bytes `desc`, the
design `designOf r`, and the roots. -/
def stmtOf {C : Verity.Boolean.Circuit} {n : ℕ} (_P : Partition C n) (κ : Fin C.N → Option Bool) (r : Claims.Sampling.OneStage)
    (desc : Claims.Msg) (q : Claims.Sampling.Query) (roots : List Claims.Msg) : Claims.Sampling.AuditStatement :=
  ⟨desc, progOf C κ, q, designOf r, roots⟩

open Classical in
/-- **The registration's commitment**: an opening at wire `g`'s position passes when some opening `op` of the
registered commit string at `g`'s position carries its value on `g`. -/
noncomputable def commitOf {C : Verity.Boolean.Circuit} {Pos Val Opn Com : Type} (pos : Fin C.N → Pos) (bit : Val → Fin C.N → Bool)
    (val : Opn → Val) (com : Opn → Com) (cs : Pos → Com) : Claims.Sampling.CommitmentScheme Bool where
  opens _ o := .ret (decide (∃ g : Fin C.N, posOf C g = o.pos ∧ ∃ op : Opn, com op = cs (pos g) ∧ bit (val op) g = o.val))
  layer _ _ _ := []

/-- **The witness**: an opening of every wire at its position, of its value in `C`'s evaluation on `X`'s inputs. -/
def witOf (C : Verity.Boolean.Circuit) (X : Fin C.N → Bool) : Claims.Sampling.Wit :=
  ⟨(List.finRange C.N).map fun g => ⟨posOf C g, C.eval X g, []⟩⟩

end Flock.Bridge
