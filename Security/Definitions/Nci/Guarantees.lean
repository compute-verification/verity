import Definitions.Nci.Assumptions
import Definitions.Core.Circuit
import Definitions.Core.Declaration
import Definitions.Core.Partition
import Definitions.Nci

/-!
# NCI's guarantees

Each is a `def G : Prop` over `Protocol`'s definitions and the assumptions, proved as
`Nci.SecurityProofs.G : Nci.Guarantees.G`.
-/

namespace Nci.Guarantees

open Verity
open scoped ENNReal

/-- **The read bound**: a declaration that contains `c` (`M`) and complies at `(Γ, α)`, with its baseline registered
(`BaselineRegistered`), has at least `Reads_Γ(c) / α` gates. Lower bounds on NCI become lower bounds on size when the
declared circuit computes `c` the way `c` does. -/
def ReadBound : Prop :=
  ∀ (Γ : ℕ) (α : ℝ) (c d : Declaration) (M : Containment c.D d.D), Complies Γ α d →
    Assumptions.BaselineRegistered c d (PartialContainment.ofContainment M) →
    (Reads Γ c : ℝ) ≤ α * d.D.size d.D.computed

/-- **Inference complies as declared**: a declaration divided into sessions of at most `Γ` gates, each reading from
outside itself only baseline information and its own prompt `prompt u`, with at most `α` bits of prompt per gate of the
session, complies at `(Γ, α)`, and its slowdown factor is at most 1. -/
def Inference : Prop :=
  ∀ (Γ : ℕ) (α : ℝ) (d : Declaration) (n : ℕ) (P : Partition d.D n) (prompt : Fin n → Finset (Fin d.D.N)),
    (∀ u, d.D.size (P.gates u) ≤ Γ) → (∀ u, d.D.into (P.gates u) ⊆ d.θ ∪ prompt u) →
    (∀ u, (d.D.width (prompt u) : ℝ) ≤ α * d.D.size (P.gates u)) →
      Complies Γ α d ∧ slowdown (Complies Γ α) d ≤ 1

/-- **Updated matrix multiplications are throttled** (the draft's theorem, "Throttling updated matrix
multiplications"): under compute-bounded incompressibility, every declared circuit `D ⊨ 𝒫` that runs updated matrix
multiplications is larger than those multiplications by a factor of at least `F`. The multiplications have about `2G`
gates: `G` products, and about as many additions. `K = αΓ` is the most an IU reads, `b_w` and `b_a` are the widths of a
weight entry and an activation entry, and `d` is the smallest inner dimension (a smaller `d` gives a smaller `F`). -/
def UpdatedMatMul : Prop :=
  ∀ (Γ : ℕ) (α b_w b_a : ℝ) (D : Declaration) (c : Chain D.D) (d : ℕ),
    0 < α → 0 < b_w → 0 < b_a → (∀ ℓ, d ≤ c.d ℓ) → Complies Γ α D →
    Assumptions.ActivationsIncompressible D c Γ (α * Γ) b_w b_a →
    Assumptions.PartialSumsIncompressible D c Γ (α * Γ) b_w b_a →
      F α (α * Γ) b_w b_a d * (2 * c.G) ≤ D.D.size D.D.computed

/-- **Frontier training is throttled** (the draft, "From updated matrix multiplications to training"): let `T` be a
training circuit in the class `𝒯`, so that every declared circuit that implements `T` and complies runs updated matrix
multiplications, which take up a fraction `κ` of `size(T)` (the coverage `κ(T)`). If compute-bounded incompressibility
holds for the multiplications each of them runs, then `ρ_𝒫(T) ≥ κ F`. -/
def Training : Prop :=
  ∀ (Γ : ℕ) (α b_w b_a κ : ℝ) (T : Declaration) (d : ℕ),
    0 < α → 0 < b_w → 0 < b_a → 0 < T.D.size T.D.computed →
    (∀ D, T.Implements D → Complies Γ α D → ∃ c : Chain D.D,
      (∀ ℓ, d ≤ c.d ℓ) ∧ κ * T.D.size T.D.computed ≤ 2 * c.G ∧
      Assumptions.ActivationsIncompressible D c Γ (α * Γ) b_w b_a ∧
      Assumptions.PartialSumsIncompressible D c Γ (α * Γ) b_w b_a) →
    ENNReal.ofReal (κ * F α (α * Γ) b_w b_a d) ≤ slowdown (Complies Γ α) T

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
open Verity Finset

open Classical in
/-- **What an IU reads** (the draft's proof of its theorem, "Throttling updated matrix multiplications"): "By the lemmas
and incompressibility, `R` reads at least `b_w p_R + b_a (d τ_R + Q_R)` bits", for an IU `R` of at most `Γ` computed
gates that reads at most `K` bits of novel information, under `Assumptions.ActivationsSupport` at cap `K + b_w + b_a`
for some committed values `θv` of the baseline. `p_R` is the number of weight entries `R` uses, `τ_R` its number of full
positions, `Q_R` the number of input entries it uses at positions that are not full, and `d` the smallest inner
dimension. -/
def _root_.Nci.Guarantees.IuReads : Prop :=
  ∀ (D : Declaration) (c : Chain D.D) (Γ : ℕ) (K b_w b_a : ℝ) (d : ℕ) (hd : ∀ ℓ, d ≤ c.d ℓ) (hb_a : 0 ≤ b_a) (hb_w : 0 ≤ b_w) (θv : D.D.Transcript) (h : Assumptions.ActivationsSupport D θv c (K + b_w + b_a) b_w b_a) (R : Finset (Fin D.D.N)) (hR : R ⊆ D.D.computed) (hΓ : D.D.size R ≤ Γ) (hK : (novelReads D R : ℝ) ≤ K),
    b_w * (c.weightsUsed R).card +
      b_a * (d * (univ.filter (c.Full R)).card + ((c.inputsUsed R).filter fun x => ¬ c.Full R x.2.1).card) ≤
    novelReads D R

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
of `y^{(ℓ)}` that `R` contributes products to is split." `R` is an IU of a division `P`; the read is the one
`Assumptions.ActivationsIncompressible` charges, since such an entry is not linked to an entry whole in `R`. -/
def _root_.Nci.Guarantees.OtherPositions : Prop :=
  ∀ {D : Circuit} (c : Chain D) {n : ℕ} (P : Partition D n) (u : Fin n),
    (∀ ℓ t j, ⟨ℓ, t, j⟩ ∈ c.inputsUsed (P.gates u) → ¬ c.Full (P.gates u) t →
      ¬ c.LinkedWhole (P.gates u) ℓ t j) ∧
    ∀ ℓ t i j, c.prod ℓ t i j ∈ P.gates u → ¬ c.Full (P.gates u) t → c.Split P ℓ t i

end Nci.SecurityProofs
