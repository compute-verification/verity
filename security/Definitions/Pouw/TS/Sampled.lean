module

public import Definitions.Pouw.TS.ProofLayer
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

/-! A sampled window's proofs, what (I8) reads of its sampled proof blocks; `UnitFixed` is claimed only where a layer's `cut` is no later than the protocol's real commitment of each position and the modes are the run's. -/

@[expose] public section

namespace ResourceAccounting.TS

open Pouw.RelativeSecurity

/-- Round `w`'s coins on the path `x`. -/
def coinAt {Ω : Type} : {n : ℕ} → Paths Ω n → Fin n → Ω
  | 0, _, w => w.elim0
  | _ + 1, x, ⟨0, _⟩ => x.1
  | _ + 1, x, ⟨w + 1, h⟩ => coinAt x.2 ⟨w, Nat.lt_of_succ_lt_succ h⟩

/-- `x`'s coins before round `w`'s coin `k` (its rounds before `w`, and round `w`'s coins before `k`; `k = m`: all of
round `w`'s coins before its sample), then `z`'s: the run that shares what `x` committed before that coin and continues on
fresh coins. -/
def spliceAt {C B : Type} {m : ℕ} : {n : ℕ} → ℕ → ℕ → Paths (Paths C m × B) n → Paths (Paths C m × B) n →
    Paths (Paths C m × B) n
  | 0, _, _, _, _ => ()
  | _ + 1, 0, k, x, z => ((splice k x.1.1 z.1.1, z.1.2), z.2)
  | _ + 1, w + 1, k, x, z => (x.1, spliceAt w k x.2 z.2)

section Plurality

variable {X V : Type}

/-- `y` opens `v` and no other value. -/
def Alone (opens : X → V → Prop) (y : X) (v : V) : Prop := opens y v ∧ ∀ v', opens y v' → v' = v

open Classical in
/-- How many continuations `sp x z` of `x` open `v` alone. -/
noncomputable def tally [Fintype X] (sp : X → X → X) (opens : X → V → Prop) (x : X) (v : V) : ℕ :=
  (Finset.univ.filter fun z => Alone opens (sp x z) v).card

/-- **The plurality value**: a value that the most continuations of `x` open alone. -/
noncomputable def plural [Fintype X] [Nonempty V] (sp : X → X → X) (opens : X → V → Prop) (x : X) : V :=
  Classical.epsilon fun v => ∀ u, tally sp opens x u ≤ tally sp opens x v

end Plurality

/-- A unit's mode in a cheater's run: paid (the run computed it: its outputs are its own, from the operands it
committed, under its noise); folded (skipped and its words guessed, its committed words folded into the receipts); or
whole (skipped with its fold). -/
inductive Mode
  | paid
  | folded
  | whole
  deriving DecidableEq

/-- **A sampled window's proofs** over `S` rounds of coins on `Paths C m × B`, `s` samples a round of units `J`, each
unit reading its slots `Q`'s positions `Pos`, each position's value in `V`:
* `draw`: the units the sampling coin names;
* `pos w j k`: the position unit `j` of round `w` reads at slot `k`;
* `coin w j`: the coin of round `w` before its sample that unit `j`'s noise reads;
* `cut ℓ = (w, k)`: `ℓ` is committed before round `w`'s coin `k` (`k = m`: before its sample), after every coin before it,
  and no later than the protocol commits it;
* `src ℓ`: the unit whose output `ℓ` is, none for the circuit's inputs (its writer reads it, `WriterReads`);
* `accepted x w i`, `opened x w i`: on `x`, whether round `w`'s proof of its `i`-th sampled proof block was accepted, and the
  values its opening gives that unit's slots;
* `rel x w j`: unit `j`'s relation in round `w` on its slots' values, the public Program its proof passes, at the
  public coins of `x` its seed reads;
* `right x w j`: its rightness: its outputs are its own, from its operands, under the noise its seed derives. -/
structure SampledLayer (C B : Type) (S m s : ℕ) (J Q Pos V : Type) where
  draw : B → Fin s → J
  pos : Fin S → J → Q → Pos
  coin : Fin S → J → Fin m
  cut : Pos → Fin S × Fin (m + 1)
  src : Pos → Option (Fin S × J)
  accepted : Paths (Paths C m × B) S → Fin S → Fin s → Prop
  opened : Paths (Paths C m × B) S → Fin S → Fin s → Q → V
  rel : Paths (Paths C m × B) S → Fin S → J → (Q → V) → Prop
  right : Paths (Paths C m × B) S → Fin S → J → (Q → V) → Prop

namespace SampledLayer

variable {C B : Type} {S m s : ℕ} {J Q Pos V : Type} (pf : SampledLayer C B S m s J Q Pos V)

/-- The unit round `w`'s sample `i` names on `x`. -/
def drawnAt (x : Paths (Paths C m × B) S) (w : Fin S) (i : Fin s) : J := pf.draw (coinAt x w).2 i

/-- **`x` opens position `ℓ` to `v`**: in some round, an accepted proof of a sampled proof block opens one of its slots, at `ℓ`,
to `v`. -/
def Opens (x : Paths (Paths C m × B) S) (ℓ : Pos) (v : V) : Prop :=
  ∃ w i k, pf.pos w (pf.drawnAt x w i) k = ℓ ∧ pf.accepted x w i ∧ pf.opened x w i k = v

/-- The continuations from position `ℓ`'s cut. -/
def cont (ℓ : Pos) : Paths (Paths C m × B) S → Paths (Paths C m × B) S → Paths (Paths C m × B) S :=
  spliceAt (pf.cut ℓ).1 (pf.cut ℓ).2

/-- **Position `ℓ`'s value as committed before its cut**: the value its sampled readers' accepted proofs open alone on the
most continuations of `x` from `ℓ`'s cut. -/
noncomputable def committed [Fintype C] [Fintype B] [Nonempty V] (x : Paths (Paths C m × B) S) (ℓ : Pos) : V :=
  plural (pf.cont ℓ) (fun y v => pf.Opens y ℓ v) x

/-- Unit `j` of round `w`'s slots at their committed values. -/
noncomputable def view [Fintype C] [Fintype B] [Nonempty V] (x : Paths (Paths C m × B) S) (w : Fin S) (j : J) :
    Q → V :=
  fun k => pf.committed x (pf.pos w j k)

def Moved [Fintype C] [Fintype B] [Nonempty V] (x : Paths (Paths C m × B) S) (ℓ : Pos) : Prop :=
  ∃ v, pf.Opens x ℓ v ∧ v ≠ pf.committed x ℓ

/-- **Binding fails in round `w`**: an accepted proof of one of its sampled proof blocks opens the unit's slots to values other
than their committed ones. -/
def MovedIn [Fintype C] [Fintype B] [Nonempty V] (x : Paths (Paths C m × B) S) (w : Fin S) : Prop :=
  ∃ i, pf.accepted x w i ∧ pf.opened x w i ≠ pf.view x w (pf.drawnAt x w i)

/-- **Binding at `δ_bind`**: in every round, binding fails (`MovedIn`) with probability at most `δ_bind` over the
window's coins. -/
def BindingAt [Fintype C] [Fintype B] [Nonempty V] (δbind : ℝ) : Prop := ∀ w, prob (fun x => pf.MovedIn x w) ≤ δbind

/-- The openings are witnesses of zero-knowledge proofs, so a finder that runs the window twice from the cut takes them
with Flock's knowledge extractor, at `ε_ks` a proof, before `cr/sha-512` bounds the collision. -/
def Forked (ℓ : Pos) (x z : Paths (Paths C m × B) S) : Prop :=
  ∃ v v', pf.Opens x ℓ v ∧ pf.Opens (pf.cont ℓ x z) ℓ v' ∧ v ≠ v'

/-- Round `w`'s proof of its `i`-th sampled proof block is accepted, and its opening fails the unit's relation. -/
def Unsound (x : Paths (Paths C m × B) S) (w : Fin S) (i : Fin s) : Prop :=
  pf.accepted x w i ∧ ¬ pf.rel x w (pf.drawnAt x w i) (pf.opened x w i)

/-- **Proof soundness at `δ_ks`, jointly over a round's samples**: in each round, some sampled proof block's proof is accepted with
an opening that fails its relation with probability at most `δ_ks` over the window's coins. -/
def JointSound [Fintype C] [Fintype B] (δks : ℝ) : Prop := ∀ w, prob (fun x => ∃ i, pf.Unsound x w i) ≤ δks

/-- **Each unit's relation pins its rightness**: values of its slots that satisfy unit `j`'s relation are right. A
check with no absorbed word as witness (`PouwCallCheck_v1`, `PouwWarpCheck_v1`, `PouwSegCheck_v3`) recomputes
every word it absorbs from the unit's operands, its seed and its noise. -/
def Pins : Prop := ∀ x w j v, pf.rel x w j v → pf.right x w j v

/-- **A position's writer reads it**: the unit whose output a position is has it among its slots, so the writer's
rightness speaks of it. Protocol structure, checked for each layer. -/
def WriterReads : Prop := ∀ ℓ u, pf.src ℓ = some u → ∃ k, pf.pos u.1 u.2 k = ℓ

/-- **Unit `u` writes, after `v`'s coin, a position that unit `v` reads**: the position's cut is after the coin `v`'s
noise reads, so its value can be chosen after that coin (such as a K-chunk's running sum). A position committed before
`v`'s coin is fixed when `v`'s noise is sampled, whether its writer is right or not, so it is no edge. -/
def Late (u v : Fin S × J) : Prop :=
  ∃ k, pf.src (pf.pos v.1 v.2 k) = some u ∧
    (v.1 < (pf.cut (pf.pos v.1 v.2 k)).1 ∨
      v.1 = (pf.cut (pf.pos v.1 v.2 k)).1 ∧ (pf.coin v.1 v.2).castSucc < (pf.cut (pf.pos v.1 v.2 k)).2)

/-- **Unit `j` of round `w` is good on `x`** under the run's modes `mode`: the run paid for every unit of its late cone
(itself, and every unit that writes it a value after its coin, transitively). A skipped unit isn't good even when its
check passes by luck. -/
def Good (mode : Paths (Paths C m × B) S → Fin S × J → Mode) (x : Paths (Paths C m × B) S) (w : Fin S) (j : J) :
    Prop :=
  ∀ u : Fin S × J, Relation.ReflTransGen pf.Late u (w, j) → mode x u = .paid

open Classical in
noncomputable def coneWt [Fintype J] (wt : Fin S → J → ℝ) (u : Fin S × J) : ℝ :=
  ∑ v : Fin S × J, if Relation.ReflTransGen pf.Late u v then wt v.1 v.2 else 0

/-- **The play is its sampled proof blocks' proofs'** (𝒫_samp): the window is accepted only if every sampled proof block's proof is, and
the ideal game's check is the sampled proof blocks' committed values, so sampled proof blocks right at them are accepted by the ideal
game. -/
def Of [Fintype C] [Fintype B] [Nonempty V] {K T L U : Type} (P : Play (Paths C m × B) S K T L U) : Prop :=
  (∀ x, P.accepts x → ∀ w i, pf.accepted x w i) ∧
    ∀ x, (∀ w i, pf.right x w (pf.drawnAt x w i) (pf.view x w (pf.drawnAt x w i))) → P.game.accepts x

open Classical in
/-- A property of the share's definition, proved for each game, not assumed. -/
def Covers [Fintype J] {K T L U : Type} (P : Play (Paths C m × B) S K T L U)
    (mode : Paths (Paths C m × B) S → Fin S × J → Mode) (wt : Fin S → J → ℝ) (Nw : ℝ) : Prop :=
  ∀ x, P.accepts x → (∑ w, ∑ j, if pf.Good mode x w j then 0 else wt w j) ≤ P.game.deficiency x * Nw

end SampledLayer

end ResourceAccounting.TS
