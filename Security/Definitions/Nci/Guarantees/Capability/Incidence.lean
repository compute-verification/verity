import Definitions.Nci.Capability

/-!
# Local bounds: caps from incidence

How many counted products the held slots and operands support. `IncidenceCap` is pure combinatorics on any gates;
`OperandCapSeparate` and `OperandCapJoint` combine it with the budgets' feasible region. Under one joint budget the cap is
`r h²/(4 βw βx)` when `r h ≤ 2 βw aout` (pairs limit it) and `aout (h − βw aout / r)/βx` otherwise (operand supply and
fan-out limit it): two regimes of one calculation.
-/

namespace Nci.Guarantees

open Verity Finset Nci.Capability

/-- **The incidence cap.** With at most `r` counted products per slot-operand pair and at most `aout` per operand, any
gates `S` compute at most `min(r |U| |V|, aout |V|)` counted products whole, `U` and `V` the slots and operands those
products use. The rectangle `U × V` overcounts pairs that never meet in a product; a finer count of the incidence graph is
a structural improvement, not a new premise. -/
def IncidenceCap : Prop :=
  ∀ (d : RegisteredCircuit) (B : Nci.Covering.MulBlocks d) {κ : Type} [DecidableEq κ] (W : Items d.D (Fin B.nw))
    (O : Items d.D κ) (I : Incidence B W O) (r aout : ℝ),
    (∀ i o, ((B.P.filter fun q => q.1 = i ∧ I.occ q = o).card : ℝ) ≤ r) →
    (∀ o, ((B.P.filter fun q => I.occ q = o).card : ℝ) ≤ aout) →
    ∀ S : Finset (Fin d.D.N),
      ((completedIn B S).card : ℝ) ≤
        min (r * ((completedIn B S).image Prod.fst).card * ((completedIn B S).image I.occ).card)
          (aout * ((completedIn B S).image I.occ).card)

/-- **The operand cap, separate budgets.** One price `βw > 0` per slot and `βx > 0` per operand; every proof block holds
slots priced at most `hw` and operands priced at most `hx`. If `Q` bounds `min(r x y, aout y)` wherever `βw x ≤ hw` and
`βx y ≤ hx`, each proof block computes at most `Q` counted products whole. -/
def OperandCapSeparate : Prop :=
  ∀ (R : Committed) (B : Nci.Covering.MulBlocks R.d) {κ : Type} [DecidableEq κ] (W : Items R.d.D (Fin B.nw))
    (O : Items R.d.D κ) (I : Incidence B W O) (r aout βw βx hw hx Q : ℝ),
    0 < βw → 0 < βx → (∀ i, W.price i = βw) → (∀ o, O.price o = βx) →
    (∀ i o, ((B.P.filter fun q => q.1 = i ∧ I.occ q = o).card : ℝ) ≤ r) →
    (∀ o, ((B.P.filter fun q => I.occ q = o).card : ℝ) ≤ aout) →
    HeldPriceAtMost R W hw → HeldPriceAtMost R O hx →
    (∀ x y : ℝ, 0 ≤ x → 0 ≤ y → βw * x ≤ hw → βx * y ≤ hx → min (r * x * y) (aout * y) ≤ Q) →
    ∀ u, ((completedIn B (R.d.𝓘.gates u)).card : ℝ) ≤ Q

/-- **The operand cap, one joint budget.** As `OperandCapSeparate`, with every proof block holding slots and operands
priced at most `h` together (a budget on `W.sum O`), and `Q` bounding `min(r x y, aout y)` wherever `βw x + βx y ≤ h`. -/
def OperandCapJoint : Prop :=
  ∀ (R : Committed) (B : Nci.Covering.MulBlocks R.d) {κ : Type} [DecidableEq κ] (W : Items R.d.D (Fin B.nw))
    (O : Items R.d.D κ) (I : Incidence B W O) (r aout βw βx h Q : ℝ),
    0 < βw → 0 < βx → (∀ i, W.price i = βw) → (∀ o, O.price o = βx) →
    (∀ i o, ((B.P.filter fun q => q.1 = i ∧ I.occ q = o).card : ℝ) ≤ r) →
    (∀ o, ((B.P.filter fun q => I.occ q = o).card : ℝ) ≤ aout) →
    HeldPriceAtMost R (W.sum O) h →
    (∀ x y : ℝ, 0 ≤ x → 0 ≤ y → βw * x + βx * y ≤ h → min (r * x * y) (aout * y) ≤ Q) →
    ∀ u, ((completedIn B (R.d.𝓘.gates u)).card : ℝ) ≤ Q

end Nci.Guarantees
