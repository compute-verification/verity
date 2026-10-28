import Definitions.Accounting.TS.Drawn

/-!
# A fixed word (trusted)

Option A, as ruled: a unit's words are **fixed** when each of its multiplicative operands is bound
before the coin its noise reads, or is written by a unit that is always proved, or by one the service prices at its
public fan-out. Pricing is a property of the draw (`Priced`): some draw lands on the writer with probability at least
its late cone's weight over the mean round weight `N_w`, the unit `DrawnLayer.Covers` charges in. Additive operands
don't decide fixedness: a computed unit's work reads its multiplicative operands and its noise only
(`Accounting.Guarantees.TS.AdditiveFixed`, a lemma, not an assumption).
Option B (every operand bound before the coin) was refused because it drops K-chunk sums and glue-fed products, and
option C (`UnitFixed`'s own reading) because it admits a cheater who cancels the noise (`A = −noise`). This file is
that scoping with the slot roles the ruling needs and no
well-founded chain, so late cones may be cyclic.

`WordLayer` is a `DrawnLayer` with, per unit, its slots' roles, its noise, the words the run wrote, and its honest
derivation split along the roles: its work (`prod`, from its multiplicative operands and its noise) plus what its
additive operands add (`acc`, which reads neither the noise nor a multiplicative operand), and the values it writes.

* `Bound u k`: slot `k`'s position is committed no later than `u`'s coin, so its committed value is a function of the
  coins before that coin (σ_{<c}).
* `AlwaysProved`, `Priced`, `Fixed`: every draw names the unit; some draw lands on it at its late cone's weight
  (`DrawnLayer.coneWt`) over `N_w`; option A, per unit, a public property of the layer and the weights.
* `MulLate`, `Steers`: a unit writes one of `u`'s multiplicative operands after `u`'s coin; a unit reaches such a
  writer along `DrawnLayer.Late`, additive edges included.
* `mulView`, `addView`, `derive`, `work`: a unit's multiplicative and additive operands at their committed values, its
  honest words, and its words less what its additive operands add.
* `Computed`, `Folds`: per path, the run computed the unit (its words and the values it writes are its honest ones, from
  its committed slots, under its noise); its receipt commits the fold of its words.
* `FixedWords`: per path, the unit is fixed, computed, and so is every unit that steers it.

`FixedWords` asks more than `Fixed ∧ Computed`. A K-chunk's running sum `S₀`, read additively by the next chunk, whose
sum `S₁` feeds an always-proved glue `G = g(S₁)` that a later product reads, leaves that product `Fixed` (its operand's
writer is always proved) and computed, while `S₀`, written after the product's coin by a skipped chunk, steers `G`. So a
unit's words are fixed only when every unit steering it was computed too. `DrawnLayer.Good` asks that of the whole
late cone, so a good unit that is `Fixed` has fixed words.
-/

namespace Accounting.TS

open Pouw.RelativeSecurity

/-- A slot's role in its unit's derivation: `mul`, an operand its work reads (a product's operand, a glue's input:
anything the noise meets, or that meets another operand); `add`, an operand only added to its work (a K-chunk's
running sum from the chunk before it); `other`, no operand (an output, a receipt, a seed's input). -/
inductive SlotRole
  | mul
  | add
  | other
  deriving DecidableEq

/-- **A drawn layer with its units' words** (option A): a `DrawnLayer` and, per unit `u`,
* `role u k`: slot `k`'s role;
* `noise x u`: the noise `u`'s seed derives on `x`, in `N`;
* `words x u`: the words the run wrote for `u` on `x`, in `W`;
* `prod u a n`: `u`'s work from its multiplicative operands `a` under noise `n`;
* `acc u b`: what its additive operands `b` add to its work. It reads neither the noise nor a multiplicative operand,
  so an operand the noise meets is multiplicative;
* `output u v n k`: the value `u` writes at slot `k`, from its slots' values `v` under noise `n`;
* `receipt u`, `fold u`: the slot of `u`'s receipt, and the fold of its words that the receipt commits. -/
structure WordLayer (C B : Type) (S m s : ℕ) (J Q Pos V N W : Type) extends DrawnLayer C B S m s J Q Pos V where
  role : Fin S × J → Q → SlotRole
  noise : Paths (Paths C m × B) S → Fin S × J → N
  words : Paths (Paths C m × B) S → Fin S × J → W
  prod : (u : Fin S × J) → ({k : Q // role u k = .mul} → V) → N → W
  acc : (u : Fin S × J) → ({k : Q // role u k = .add} → V) → W
  output : Fin S × J → (Q → V) → N → Q → V
  receipt : Fin S × J → Q
  fold : Fin S × J → W → V

namespace WordLayer

variable {C B : Type} {S m s : ℕ} {J Q Pos V N W : Type} (pf : WordLayer C B S m s J Q Pos V N W)

/-- **Slot `k` of unit `u` is bound before `u`'s coin**: its position's cut is no later than the coin `u`'s noise reads
(round first, then coin), so its committed value is a function of the coins before that coin. It is `DrawnLayer.Late`'s
condition, negated. -/
def Bound (u : Fin S × J) (k : Q) : Prop :=
  (pf.cut (pf.pos u.1 u.2 k)).1 < u.1 ∨
    (pf.cut (pf.pos u.1 u.2 k)).1 = u.1 ∧ (pf.cut (pf.pos u.1 u.2 k)).2 ≤ (pf.coin u.1 u.2).castSucc

/-- **Unit `u` is always proved**: every draw names it, so its proof is among its round's on every path. -/
def AlwaysProved (u : Fin S × J) : Prop := ∀ b, ∃ i, pf.draw b i = u.2

section Priced

variable [Fintype B] [Fintype J]

/-- **Unit `v` is priced at its public fan-out** under the weights `wt` (`wt w j`: unit `j`'s words in round `w`): some
draw `i` lands on it with probability at least its late cone's weight (`DrawnLayer.coneWt`: itself and every unit it
writes a value after that unit's coin, transitively, so every credited unit a wrong `v` spoils) over the mean round
weight `N_w = Σ wt / S`, the unit `DrawnLayer.Covers` charges the game's deficiency in (not `v`'s own round's weight,
which differs from it by `wt(v.1)/N_w`). A skipped `v` is then charged its cone, `F` in
`skipped ≤ F·wt(B)/(1 − p) + slack`. A property of the draw, public.
A reader composing it with `DrawnLayer.Covers` passes `Covers` the same `Nw = Σ wt / S` that it reads. -/
def Priced (wt : Fin S → J → ℝ) (v : Fin S × J) : Prop :=
  ∃ i, pf.coneWt wt v ≤ prob (fun b : B => pf.draw b i = v.2) * ((∑ w, ∑ j, wt w j) / S)

/-- **Option A: unit `u` is fixed** under the weights `wt` (public, a property of the layer and the weights):
each of its multiplicative operands is bound before the coin its noise reads, or is written by a unit that is always
proved or priced at its public fan-out (`Priced`). Its additive operands don't enter. -/
def Fixed (wt : Fin S → J → ℝ) (u : Fin S × J) : Prop :=
  ∀ k, pf.role u k = .mul → pf.Bound u k ∨
    ∃ v, pf.src (pf.pos u.1 u.2 k) = some v ∧ (pf.AlwaysProved v ∨ pf.Priced wt v)

end Priced

/-- **Unit `v` writes one of `u`'s multiplicative operands after `u`'s coin.** -/
def MulLate (v u : Fin S × J) : Prop :=
  ∃ k, pf.role u k = .mul ∧ pf.src (pf.pos u.1 u.2 k) = some v ∧ ¬ pf.Bound u k

/-- **Unit `t` steers unit `u`**: `t` reaches, along `DrawnLayer.Late` (additive edges included, itself included), a
unit that writes one of `u`'s multiplicative operands after `u`'s coin. A K-chunk's running sum steers the glue its
chunks' last sum feeds, and through that glue the product that reads it. -/
def Steers (t u : Fin S × J) : Prop := ∃ v, Relation.ReflTransGen pf.Late t v ∧ pf.MulLate v u

variable [Fintype C] [Fintype B] [Nonempty V]

/-- Unit `u`'s multiplicative operands at their committed values on `x`. -/
noncomputable def mulView (x : Paths (Paths C m × B) S) (u : Fin S × J) : {k : Q // pf.role u k = .mul} → V :=
  fun k => pf.view x u.1 u.2 k.1

/-- Unit `u`'s additive operands at their committed values on `x`. -/
noncomputable def addView (x : Paths (Paths C m × B) S) (u : Fin S × J) : {k : Q // pf.role u k = .add} → V :=
  fun k => pf.view x u.1 u.2 k.1

/-- **Unit `u`'s receipt commits the fold of its words** on `x`. -/
def Folds (x : Paths (Paths C m × B) S) (u : Fin S × J) : Prop :=
  pf.committed x (pf.pos u.1 u.2 (pf.receipt u)) = pf.fold u (pf.words x u)

variable [AddCommGroup W]

/-- **Unit `u`'s honest words** from its slots' values `v` under noise `n`: its work plus what its additive operands
add. -/
def derive (u : Fin S × J) (v : Q → V) (n : N) : W := pf.prod u (fun k => v k.1) n + pf.acc u (fun k => v k.1)

/-- **Unit `u`'s work on `x`**: its words less what its additive operands add. -/
noncomputable def work (x : Paths (Paths C m × B) S) (u : Fin S × J) : W := pf.words x u - pf.acc u (pf.addView x u)

/-- **The run computed unit `u` on `x`**: its words are its honest words from its committed slots under its noise, and
every value it writes is its honest output there. -/
def Computed (x : Paths (Paths C m × B) S) (u : Fin S × J) : Prop :=
  pf.words x u = pf.derive u (pf.view x u.1 u.2) (pf.noise x u) ∧
    ∀ k, pf.src (pf.pos u.1 u.2 k) = some u →
      pf.committed x (pf.pos u.1 u.2 k) = pf.output u (pf.view x u.1 u.2) (pf.noise x u) k

variable [Fintype J] in
/-- **Unit `u`'s words on `x` are fixed words** under the weights `wt`: `u` is fixed (option A), the run computed it,
and it computed every unit that steers it. -/
def FixedWords (wt : Fin S → J → ℝ) (x : Paths (Paths C m × B) S) (u : Fin S × J) : Prop :=
  pf.Fixed wt u ∧ pf.Computed x u ∧ ∀ t, pf.Steers t u → pf.Computed x t

end WordLayer

end Accounting.TS
