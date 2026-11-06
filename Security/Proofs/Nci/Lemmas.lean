module

public import Definitions.Nci.Assumptions
public import Definitions.Nci.Threads
public import Definitions.Core.Circuit
public import Definitions.Core.RegisteredCircuit
public import Definitions.Core.Partition
public import Definitions.Nci

/-!
# The inference-only policy's lemmas

These are steps of proofs, not guarantees: they sit with the proofs, and no record of the lock pins them.

Each is a `def G : Prop` over `Protocol`'s definitions and the assumptions, proved as
`Nci.SecurityProofs.G : Nci.Guarantees.G`.
-/

@[expose] public section

namespace Nci.Guarantees

open Verity
open scoped ENNReal

/-- **The read bound**: a declaration that contains the reference circuit `r` (`M`) and complies at `(Γ, α)`
(`CompliesRate`), with its registered information free of the reference circuit's fresh input (`ThetaRegistered`), has
at least `Reads_Γ(r) / α` gates. Lower bounds on the fresh input a computation reads become lower bounds on size when the
declared circuit computes `r` the way `r` does. -/
def ReadBound : Prop :=
  ∀ (Γ : ℕ) (α : ℝ) (r d : RegisteredCircuit) (M : Containment r.D d.D), CompliesRate Γ α d →
    Assumptions.ThetaRegistered r d (PartialContainment.ofContainment M) →
    (Reads Γ r : ℝ) ≤ α * d.D.size d.D.computed

/-- **Inference complies as declared**: a Boolean declaration divided into sessions of exactly `c` gates, padding
included, each reading from outside itself only registered information and its own prompt `prompt u` of at most `k`
bits, complies at `(c, k)`, and its slowdown factor is at most 1. -/
def Inference : Prop :=
  ∀ (c k : ℕ) (d : RegisteredCircuit) (B : ℕ) (P : Partition d.D B) (prompt : Fin B → Finset (Fin d.D.N)),
    d.D.Boolean → Assumptions.Sessions d P prompt c k → Complies c k d ∧ slowdown (Complies c k) d ≤ 1

end Nci.Guarantees

namespace Nci.SecurityProofs
open Verity Finset

open Classical in
/-- **Lemma 1 (full positions)** (the draft, "Throttling updated matrix multiplications"): "If `R` uses `p` weight
entries and has `τ` full positions, then `R` computes at most `pτ` products at its full positions, and reads at least
`b_a d` bits of activations or partial sums for each of them." The reads are those
`Assumptions.ActivationsIncompressible` charges: at each full position, `R` uses at least `d` input entries that are not
linked to an entry whole in `R`, `d` the smallest inner dimension. -/
def _root_.Nci.Guarantees.FullPositions : Prop :=
  ∀ {D : Circuit} (c : Chain D) (R : Finset (Fin D.N)) (d : ℕ) (hd : ∀ ℓ, d ≤ c.d ℓ),
    (univ.filter fun g : (ℓ : Fin c.L) × Fin c.N × Fin (c.e ℓ) × Fin (c.d ℓ) =>
        c.prod g.1 g.2.1 g.2.2.1 g.2.2.2 ∈ R ∧ c.Full R g.2.1).card ≤
      (c.weightsUsed R).card * (univ.filter (c.Full R)).card ∧
    d * (univ.filter (c.Full R)).card ≤
      ((c.inputsUsed R).filter fun x => c.Full R x.2.1 ∧ ¬ c.LinkedWhole R x.1 x.2.1 x.2.2).card

end Nci.SecurityProofs

namespace Nci.SecurityProofs
open Finset

/-- **The discrete Loomis–Whitney inequality** in three dimensions: a finite set of triples, squared, has at most the
product of the sizes of its three coordinate-pair projections. -/
def _root_.Nci.Guarantees.LoomisWhitney : Prop :=
  ∀ {α β γ : Type*} [DecidableEq α] [DecidableEq β] [DecidableEq γ] (T : Finset (α × β × γ)),
    #T ^ 2 ≤ #(T.image fun x => (x.1, x.2.1)) * #(T.image fun x => (x.1, x.2.2)) *
    #(T.image fun x => (x.2.1, x.2.2))

end Nci.SecurityProofs

namespace Nci.SecurityProofs
open Verity Finset

/-- **Lemma 2 (other positions)** (the draft, "Throttling updated matrix multiplications"): "At positions that are not
full positions of `R`, every entry of `x^{(ℓ-1)}` that `R` uses costs `R` a read of at least `b_a` bits, and every entry
of `y^{(ℓ)}` that `R` contributes products to is split." `R` is an inference block of a division `P`; the read is the
one `Assumptions.ActivationsIncompressible` charges, since such an entry is not linked to an entry whole in `R`. -/
def _root_.Nci.Guarantees.OtherPositions : Prop :=
  ∀ {D : Circuit} (c : Chain D) {n : ℕ} (P : Partition D n) (u : Fin n),
    (∀ ℓ t j, ⟨ℓ, t, j⟩ ∈ c.inputsUsed (P.gates u) → ¬ c.Full (P.gates u) t →
      ¬ c.LinkedWhole (P.gates u) ℓ t j) ∧
    ∀ ℓ t i j, c.prod ℓ t i j ∈ P.gates u → ¬ c.Full (P.gates u) t → c.Split P ℓ t i

end Nci.SecurityProofs
