import Definitions.Nci.Assumptions
import Definitions.Nci.Threads
import Definitions.Core.Circuit
import Definitions.Core.RegisteredCircuit
import Definitions.Core.Partition
import Definitions.Nci

/-!
# The inference-only policy's guarantees

Each is a `def G : Prop` over `Protocol`'s definitions and the assumptions, proved as
`Nci.SecurityProofs.G : Nci.Guarantees.G`.
-/

namespace Nci.Guarantees

open Verity
open scoped ENNReal

/-- **The read bound**: a declaration that contains the reference circuit `r` (`M`) and complies at `(c, k)`, with its
registered information free of the reference circuit's fresh input (`ThetaRegistered`), has at least `Reads_c(r) / α`
gates, at the rate `α = k / c`. Lower bounds on the fresh input a computation reads become lower bounds on size when the
declared circuit computes `r` the way `r` does. -/
def ReadBound : Prop :=
  ∀ (c k : ℕ) (r d : RegisteredCircuit) (M : Containment r.D d.D), Complies c k d →
    Assumptions.ThetaRegistered r d (PartialContainment.ofContainment M) →
    (Reads c r : ℝ) ≤ (k / c : ℝ) * d.D.size d.D.computed

/-- **Inference complies as declared**: a Boolean declaration divided into sessions of exactly `c` gates, padding
included, each reading from outside itself only registered information and its own prompt `prompt u` of at most `k`
bits, complies at `(c, k)`, and its slowdown factor is at most 1. -/
def Inference : Prop :=
  ∀ (c k : ℕ) (d : RegisteredCircuit) (B : ℕ) (P : Partition d.D B) (prompt : Fin B → Finset (Fin d.D.N)),
    d.D.Boolean → Assumptions.Sessions d P prompt c k → Complies c k d ∧ slowdown (Complies c k) d ≤ 1

/-- **Updated matrix multiplications are throttled** (the draft's theorem, "Throttling updated matrix
multiplications"): under compute-bounded incompressibility, every declared circuit `D ⊨ 𝒫` that runs updated matrix
multiplications is larger than those multiplications by a factor of at least `F`. The multiplications have about `2G`
gates: `G` products, and about as many additions. The policy is at `(c, k)`, `k` is the most an inference block reads,
`F` is taken at the rate `k / c`, `b_w` and `b_a` are the widths of a weight entry and an activation entry, and `d` is
the smallest inner dimension (a smaller `d` gives a smaller `F`). -/
def UpdatedMatMul : Prop :=
  ∀ (c k : ℕ) (b_w b_a : ℝ) (D : RegisteredCircuit) (ch : Chain D.D) (d : ℕ),
    0 < c → 0 < k → 0 < b_w → 0 < b_a → Assumptions.InnerAtLeast ch d → Complies c k D →
    Assumptions.ActivationsIncompressible D ch c k b_w b_a →
    Assumptions.PartialSumsIncompressible D ch c k b_w b_a →
      F (k / c) k b_w b_a d * (2 * ch.G) ≤ D.D.size D.D.computed

/-- **Frontier training is throttled** (the draft, "From updated matrix multiplications to training"): let `T` be a
training circuit in the class `𝒯`, so that every declared circuit that implements `T` and complies runs updated matrix
multiplications, which take up a fraction `κ` of `size(T)` (the coverage `κ(T)`). If compute-bounded incompressibility
holds for the multiplications each of them runs, then `ρ_𝒫(T) ≥ κ F`. -/
def Training : Prop :=
  ∀ (c k : ℕ) (b_w b_a κ : ℝ) (T : RegisteredCircuit) (d : ℕ),
    0 < c → 0 < k → 0 < b_w → 0 < b_a → 0 < T.D.size T.D.computed → Assumptions.InClass T c k b_w b_a κ d →
    ENNReal.ofReal (κ * F (k / c) k b_w b_a d) ≤ slowdown (Complies c k) T

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
/-- **What an inference block reads** (the draft's proof of its theorem, "Throttling updated matrix multiplications"):
"By the lemmas and incompressibility, `R` reads at least `b_w p_R + b_a (d τ_R + Q_R)` bits", for an inference block `R`
of at most `c` computed gates that reads at most `k` fresh input bits, under `Assumptions.ActivationsSupport` at cap
`k + b_w + b_a` for some committed values `θv` of the registered information. `p_R` is the number of weight entries `R`
uses, `τ_R` its number of full positions, `Q_R` the number of input entries it uses at positions that are not full, and
`d` the smallest inner dimension. -/
def _root_.Nci.Guarantees.IuReads : Prop :=
  ∀ (D : RegisteredCircuit) (ch : Chain D.D) (c : ℕ) (k b_w b_a : ℝ) (d : ℕ) (hd : Assumptions.InnerAtLeast ch d) (hb_a : 0 ≤ b_a) (hb_w : 0 ≤ b_w) (θv : D.D.Transcript) (h : Assumptions.ActivationsSupport D θv ch (k + b_w + b_a) b_w b_a) (R : Finset (Fin D.D.N)) (hR : Small D c k R),
    b_w * (ch.weightsUsed R).card +
      b_a * (d * (univ.filter (ch.Full R)).card + ((ch.inputsUsed R).filter fun x => ¬ ch.Full R x.2.1).card) ≤
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
of `y^{(ℓ)}` that `R` contributes products to is split." `R` is an inference block of a division `P`; the read is the
one `Assumptions.ActivationsIncompressible` charges, since such an entry is not linked to an entry whole in `R`. -/
def _root_.Nci.Guarantees.OtherPositions : Prop :=
  ∀ {D : Circuit} (c : Chain D) {n : ℕ} (P : Partition D n) (u : Fin n),
    (∀ ℓ t j, ⟨ℓ, t, j⟩ ∈ c.inputsUsed (P.gates u) → ¬ c.Full (P.gates u) t →
      ¬ c.LinkedWhole (P.gates u) ℓ t j) ∧
    ∀ ℓ t i j, c.prod ℓ t i j ∈ P.gates u → ¬ c.Full (P.gates u) t → c.Split P ℓ t i

end Nci.SecurityProofs
