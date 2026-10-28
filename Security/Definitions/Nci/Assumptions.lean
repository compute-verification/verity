import Definitions.Nci.Support
import Definitions.Nci.Assumptions.Baseline
import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-!
# NCI's named assumptions

The bound for updated matrix multiplications rests on one assumption, compute-bounded incompressibility (the draft,
"Updated matrix multiplications"): the activations, partial sums and weights of the chain cannot be passed between IUs
more cheaply than by sending them, and "an IU that uses several of these values, other than values it computes from its
own products alone, must read at least their combined width". The draft's proof uses it twice: for the weights and
activations an IU uses, and for the weights it uses and the partial sums it receives. Each is a `Prop` about one
declaration and one chain it runs, taken as a hypothesis by the guarantees that need it.

Which declaration matters. On a declared circuit, whose wiring the prover controls, both are false, an analysis
simplification, and the guarantees that apply them there (`UpdatedMatMul`,
`UpdatedMatMulReads`, `Training`) are falsified. That covers declared circuits only:
on a reference training circuit, the one a declared circuit contains, both stand, and with `WeightsIncompressible` they
are `TrainingIncompressible`, NCI's one adopted premise.

Both are assumed only for chains whose weights and first inputs `x^{(0)}` depend on more than `Γ` work of training since
the commitment (the draft, "Frontier training currently requires updated matrix multiplications"). Starting from a
committed checkpoint, one IU could compute weights that depend on less from the training data itself. If the layers
before the first multiplication are committed in `θ`, an IU computes `x^{(0)}` at many positions from `θ` alone, and
activation incompressibility fails at the first multiplication. Frontier runs pass this threshold at once, since
everything before the first multiplication is updated too.

Both are assumed only for IUs that read at most `K` bits, which under the policy is `αΓ`. The draft states its
assumption for every IU of at most `Γ` gates and uses it only for IUs that read at most `K`. Without the cap it fails for
honest training: an IU that runs a transformer's first attention layer and first MLP at many positions reads the
embeddings and attention weights once, and then computes the MLP's inputs at every position itself.

Every bound through a containment, total or partial (`PartialContainment`), also rests on `ThetaRegistered`
(`Definitions/Nci/Assumptions/Baseline.lean`), a premise about what the protocol lets the baseline `θ` hold rather than about
any circuit.
-/

namespace Nci.Assumptions

open Verity

open Classical in
/-- **Incompressibility of weights and activations** (the draft's weight and activation incompressibility, applied
jointly). Every IU `R` of `D` (a set of at most `Γ` computed gates) that reads at most `K` bits of novel information reads
at least `b_w` bits for each weight entry it uses and `b_a` bits for each input entry it uses, other than an input entry
linked to an entry whole in `R`, which `R` can compute from its own products alone:
`b_w · #(weight entries R uses) + b_a · #(input entries R uses, not linked to an entry whole in R) ≤ width(in(R) \ θ)`.
"Uses" means that `R` holds a product of the entry. `b_w` is the width of a weight entry and `b_a` that of an activation
entry; at FP4 both widths are 4 bits. These are widths, not support: in the support form below an E2M1 product takes
only 37 values, so there `b_w + b_a ≤ log₂ 37 ≈ 5.2` at FP4. An entry's share of its block scale is not counted, since
an IU can use an entry without reading its scale and leave the scaling to another.

The draft's rationale (Lemma 1's proof): `R` must read each input entry it uses, its linked output entry, or a partial
sum of that output entry; these are distinct for distinct input entries, since the links are injective, and each is at
least `b_a` wide. No IU of at most `Γ` gates can reconstruct entries of `W_ℓ`, or entries of the chain's input and output
vectors, from fewer bits of novel information than their width.

False when `D` is a declared circuit; it stands when `D` is a reference training circuit,
as the activations' half of `TrainingIncompressible`. `ActivationsSupport`'s counterexample does not
apply to it: an E2M1 product takes at most 37 values, but an IU that holds the product gate reads its weight and input
wires whole, at least `b_w + b_a` bits. -/
def ActivationsIncompressible (D : Registered) (c : Chain D.D) (Γ : ℕ) (K b_w b_a : ℝ) : Prop :=
  ∀ R ⊆ D.D.computed, D.D.size R ≤ Γ → (novelReads D R : ℝ) ≤ K →
    b_w * (c.weightsUsed R).card +
        b_a * ((c.inputsUsed R).filter fun x => ¬ c.LinkedWhole R x.1 x.2.1 x.2.2).card ≤ novelReads D R

/-- **Incompressibility of weights and activations, in support form**: `ActivationsIncompressible` restated over what
values determine rather than over wiring. For every IU `R` and every set `T` of its gates whose charge (`Chain.charge`,
`b_w` per weight entry and `b_a` per input entry of which `T` holds a product, other than one linked to an entry whole in
`R`) is at most `K`, the gates of `T` take at least `2^charge` distinct values over the runs of `D` with θ at
its committed values `θv`.

Wiring cannot falsify it: a weight held in a constant gate, or rebuilt from a narrower wire, has the support it has.
It fails when the chain's values are determined by less than they are charged for, as when the chain is not updated.
`b_w` and `b_a` are rates, in bits of support per entry, and are below an entry's width when products repeat values (at
FP4, a product's sign and scale symmetries), so they are measured, not read from the format.

Every value of `R` is a function of the values on `in(R)`, so `support(T) ≤ 2^novelReads(R)`; at cap `K + b_w + b_a`
this assumption gives `ActivationsIncompressible` at cap `K` (`activationsIncompressible_of_support`). It needs no
bound on `R`'s size: the support form says what the values are, not what an IU can compute.

**Falsified** at FP4, for every updated chain too: when a product
gate of the first multiplication holds an E2M1 × E2M1 product, `T` that one gate is charged `b_w + b_a` and takes at
most 37 values, so the assumption fails at every cap of at least `b_w + b_a` once `b_w + b_a > log₂ 37 ≈ 5.21` (the
calculator's 2.5 + 4 = 6.5). At FP8 (E4M3, 2011 products) this counterexample does not apply. -/
def ActivationsSupport (D : Registered) (θv : D.D.Transcript) (c : Chain D.D) (K b_w b_a : ℝ) : Prop :=
  ∀ R ⊆ D.D.computed, ∀ T ⊆ R, c.charge R b_w b_a T ≤ K → (2 : ℝ) ^ c.charge R b_w b_a T ≤ support D θv T

/-- **Incompressibility of partial sums** (the draft's partial-sum incompressibility, applied jointly with weight
incompressibility, and the count of partial sums in the proof of its theorem). A partial sum is a sum of some of the
products that make up an output entry. When several IUs contribute products to the same entry, each partial sum passed
between them costs its reader at least its width, at least `b_a`, and an entry whose products lie in `m ≥ 2` IUs needs
at least `m − 1` partial sums to pass between IUs.

For every division `P` of `D` into IUs of at most `Γ` gates, each reading at most `K` bits of novel information, the
partial sums passed between IUs can be charged to the IUs that receive them, `s u` to IU `u`, so that:
1. an entry whose products lie in `m` IUs is charged at least `m − 1` in total;
2. each IU reads at least `b_w` bits for each weight entry it uses and `b_a` bits for each partial sum charged to it.

This is the draft's statement with `s u` the number of partial sums IU `u` receives: the partial sums are not gates the
chain names, since `D` may add up each entry in any order.

False when `D` is a declared circuit; it stands when `D` is a reference training circuit,
as the partial sums' half of `TrainingIncompressible`. `PartialSumsSupport`'s counterexample does not
apply to it: an adder IU over a `k × k × k` block holds values determined by `2k²` entries, but it reads its `k³`
products' wires to form them. -/
def PartialSumsIncompressible (D : Registered) (c : Chain D.D) (Γ : ℕ) (K b_w b_a : ℝ) : Prop :=
  ∀ (n : ℕ) (P : Partition D.D n), (∀ u, D.D.size (P.gates u) ≤ Γ ∧ (novelReads D (P.gates u) : ℝ) ≤ K) →
    ∃ s : Fin n → ℕ,
      ∑ x : (ℓ : Fin c.L) × Fin c.N × Fin (c.e ℓ), (c.spread P x.1 x.2.1 x.2.2 - 1) ≤ ∑ u, s u ∧
      ∀ u, b_w * (c.weightsUsed (P.gates u)).card + b_a * s u ≤ novelReads D (P.gates u)

/-- **Incompressibility of partial sums, in support form**: `PartialSumsIncompressible` restated over what values
determine rather than over wiring. For every division `P` of `D` into IUs of at most `Γ` gates, each reading at most `K`
bits of novel information, the partial sums passed between IUs can be charged to the IUs that receive them, `s u` to IU
`u`, so that an entry whose products lie in `m` IUs is charged at least `m − 1` in total, and the gates of each IU take
at least `2^(b_w · #(weight entries it uses) + b_a · s u)` distinct values over the runs of `D` with θ at its
committed values `θv`.

Like `ActivationsSupport`, wiring cannot falsify it, and `b_w` and `b_a` are rates in bits of support per entry, below an
entry's width by whatever the updated weights and the partial sums compress to. It is meant for a reference training
circuit, the one a declared circuit contains (`Nci.Guarantees.ContainedMatMul`), where it is a claim about how
compressible training is. An IU's values are functions of the values on its novel reads, so it gives
`PartialSumsIncompressible` at the same caps (`partialSumsIncompressible_of_support`).

**Falsified** for one honest matrix multiplication, with binary
additions and inside the caps: divide it into product IUs that fix a column over a block of rows and positions, and IUs
that hold the adder tree over `k × k × k` blocks. Each entry's products then lie in `d` IUs, while each IU's values are
functions of the few weight and input entries it depends on. In the exact 1-bit toy (`d = e = N = 8`, `b_w = b_a = 1`)
the IUs can be charged at most 183 (192 bits of entries) against the 448 required, for every `Γ ≥ 448` and `K ≥ 512`.
At `d = 7168` and the deployed `K = 2·10⁷` such a division absorbs at most 8.4% of the charge at FP4 and 17.8% at FP8
(2.5 and 4 bits). -/
def PartialSumsSupport (D : Registered) (θv : D.D.Transcript) (c : Chain D.D) (Γ : ℕ) (K b_w b_a : ℝ) : Prop :=
  ∀ (n : ℕ) (P : Partition D.D n), (∀ u, D.D.size (P.gates u) ≤ Γ ∧ (novelReads D (P.gates u) : ℝ) ≤ K) →
    ∃ s : Fin n → ℕ,
      ∑ x : (ℓ : Fin c.L) × Fin c.N × Fin (c.e ℓ), (c.spread P x.1 x.2.1 x.2.2 - 1) ≤ ∑ u, s u ∧
      ∀ u, (2 : ℝ) ^ (b_w * (c.weightsUsed (P.gates u)).card + b_a * s u) ≤ support D θv (P.gates u)

end Nci.Assumptions
