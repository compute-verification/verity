import Definitions.Nci.Support
import Definitions.Nci.Assumptions.Baseline
import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-!
# The inference-only policy's named assumptions

The bound for updated matrix multiplications rests on one assumption, compute-bounded incompressibility (the draft,
"Updated matrix multiplications"): the activations, partial sums and weights of the chain cannot be passed between
inference blocks more cheaply than by sending them, and "an inference block that uses several of these values, other
than values it computes from its own products alone, must read at least their combined width". The draft's proof uses it
twice: for the weights and activations an inference block uses, and for the weights it uses and the partial sums it
receives. Each is a `Prop` about one declaration and one chain it runs, taken as a hypothesis by the guarantees that
need it.

Which declaration matters. On a declared circuit, whose wiring the prover controls, both are false, an analysis
simplification, and the guarantees that apply them there (`UpdatedMatMul`,
`UpdatedMatMulReads`, `Training`) are falsified. That covers declared circuits only:
on a reference training circuit, the one a declared circuit contains, both stand, and with `WeightsIncompressible` they
are `TrainingIncompressible`, the inference-only policy's one adopted premise.

Both are assumed only for chains whose weights and first inputs `x^{(0)}` depend on more than `c` gates of training
since the commitment (the draft, "Frontier training currently requires updated matrix multiplications"). Starting from a
committed checkpoint, one inference block could compute weights that depend on less from the training data itself. If
the layers before the first multiplication are committed in `θ`, an inference block computes `x^{(0)}` at many positions
from `θ` alone, and activation incompressibility fails at the first multiplication. Frontier runs pass this threshold at
once, since everything before the first multiplication is updated too.

Both are assumed only for inference blocks that read at most `k` bits, the policy's read cap. The draft states its
assumption for every inference block of at most `c` gates and uses it only for inference blocks that read at most `k`.
Without the cap it fails for honest training: an inference block that runs a transformer's first attention layer and
first MLP at many positions reads the embeddings and attention weights once, and then computes the MLP's inputs at every
position itself.

Every bound through a containment, total or partial (`PartialContainment`), also rests on `ThetaRegistered`
(`Definitions/Nci/Assumptions/Baseline.lean`), a premise about what the protocol lets the registered
information `θ` hold rather than about any circuit.
-/

namespace Nci.Assumptions

open Verity

open Classical in
/-- **Incompressibility of weights and activations** (the draft's weight and activation incompressibility, applied
jointly). Every inference block `R` of `D` (a set of at most `c` computed gates) that reads at most `k` fresh input bits
reads at least `b_w` bits for each weight entry it uses and `b_a` bits for each input entry it uses, other than an input
entry linked to an entry whole in `R`, which `R` can compute from its own products alone:
`b_w · #(weight entries R uses) + b_a · #(input entries R uses, not linked to an entry whole in R) ≤ width(in(R) \ θ)`.
"Uses" means that `R` holds a product of the entry. `b_w` is the width of a weight entry and `b_a` that of an activation
entry; at FP4 both widths are 4 bits. These are widths, not support: in the support form below an E2M1 product takes
only 37 values, so there `b_w + b_a ≤ log₂ 37 ≈ 5.2` at FP4. An entry's share of its block scale is not counted, since
an inference block can use an entry without reading its scale and leave the scaling to another.

The draft's rationale (Lemma 1's proof): `R` must read each input entry it uses, its linked output entry, or a partial
sum of that output entry; these are distinct for distinct input entries, since the links are injective, and each is at
least `b_a` wide. No inference block of at most `c` gates can reconstruct entries of `W_ℓ`, or entries of the chain's
input and output vectors, from fewer fresh input bits than their width.

False when `D` is a declared circuit; it stands when `D` is a reference training circuit,
as the activations' half of `TrainingIncompressible`. `ActivationsSupport`'s counterexample does not
apply to it: an E2M1 product takes at most 37 values, but an inference block that holds the product gate reads its
weight and input wires whole, at least `b_w + b_a` bits. -/
def ActivationsIncompressible (D : RegisteredCircuit) (ch : Chain D.D) (c : ℕ) (k b_w b_a : ℝ) : Prop :=
  ∀ R ⊆ D.D.computed, D.D.size R ≤ c → (novelReads D R : ℝ) ≤ k →
    b_w * (ch.weightsUsed R).card +
        b_a * ((ch.inputsUsed R).filter fun x => ¬ ch.LinkedWhole R x.1 x.2.1 x.2.2).card ≤ novelReads D R

/-- **Incompressibility of weights and activations, in support form**: `ActivationsIncompressible` restated over what
values determine rather than over wiring. For every inference block `R` and every set `T` of its gates whose charge
(`Chain.charge`, `b_w` per weight entry and `b_a` per input entry of which `T` holds a product, other than one linked to
an entry whole in `R`) is at most `k`, the gates of `T` take at least `2^charge` distinct values over the runs of `D`
with the registered information at its committed values `θv`.

Wiring cannot falsify it: a weight held in a constant gate, or rebuilt from a narrower wire, has the support it has.
It fails when the chain's values are determined by less than they are charged for, as when the chain is not updated.
`b_w` and `b_a` are rates, in bits of support per entry, and are below an entry's width when products repeat values (at
FP4, a product's sign and scale symmetries), so they are measured, not read from the format.

Every value of `R` is a function of the values on `in(R)`, so `support(T) ≤ 2^novelReads(R)`; at cap `k + b_w + b_a`
this assumption gives `ActivationsIncompressible` at cap `k` (`activationsIncompressible_of_support`). It needs no
bound on `R`'s size: the support form says what the values are, not what an inference block can compute.

**Falsified** at FP4, for every updated chain too: when a product
gate of the first multiplication holds an E2M1 × E2M1 product, `T` that one gate is charged `b_w + b_a` and takes at
most 37 values, so the assumption fails at every cap of at least `b_w + b_a` once `b_w + b_a > log₂ 37 ≈ 5.21` (the
calculator's 2.5 + 4 = 6.5). At FP8 (E4M3, 2011 products) this counterexample does not apply. -/
def ActivationsSupport (D : RegisteredCircuit) (θv : D.D.Transcript) (ch : Chain D.D) (k b_w b_a : ℝ) : Prop :=
  ∀ R ⊆ D.D.computed, ∀ T ⊆ R, ch.charge R b_w b_a T ≤ k → (2 : ℝ) ^ ch.charge R b_w b_a T ≤ support D θv T

/-- **Incompressibility of partial sums** (the draft's partial-sum incompressibility, applied jointly with weight
incompressibility, and the count of partial sums in the proof of its theorem). A partial sum is a sum of some of the
products that make up an output entry. When several inference blocks contribute products to the same entry, each partial
sum passed between them costs its reader at least its width, at least `b_a`, and an entry whose products lie in `m ≥ 2`
inference blocks needs at least `m − 1` partial sums to pass between inference blocks.

For every division `P` of `D` into inference blocks of at most `c` gates, each reading at most `k` fresh input bits, the
partial sums passed between inference blocks can be charged to the inference blocks that receive them, `s u` to
inference block `u`, so that:
1. an entry whose products lie in `m` inference blocks is charged at least `m − 1` in total;
2. each inference block reads at least `b_w` bits for each weight entry it uses and `b_a` bits for each partial sum
charged to it.

This is the draft's statement with `s u` the number of partial sums inference block `u` receives: the partial sums are
not gates the chain names, since `D` may add up each entry in any order.

False when `D` is a declared circuit; it stands when `D` is a reference training circuit,
as the partial sums' half of `TrainingIncompressible`. `PartialSumsSupport`'s counterexample does not
apply to it: an adder inference block over a `k × k × k` block holds values determined by `2k²` entries, but it reads
its `k³` products' wires to form them. -/
def PartialSumsIncompressible (D : RegisteredCircuit) (ch : Chain D.D) (c : ℕ) (k b_w b_a : ℝ) : Prop :=
  ∀ (n : ℕ) (P : Partition D.D n), (∀ u, D.D.size (P.gates u) ≤ c ∧ (novelReads D (P.gates u) : ℝ) ≤ k) →
    ∃ s : Fin n → ℕ,
      ∑ x : (ℓ : Fin ch.L) × Fin ch.N × Fin (ch.e ℓ), (ch.spread P x.1 x.2.1 x.2.2 - 1) ≤ ∑ u, s u ∧
      ∀ u, b_w * (ch.weightsUsed (P.gates u)).card + b_a * s u ≤ novelReads D (P.gates u)

/-- **Incompressibility of partial sums, in support form**: `PartialSumsIncompressible` restated over what values
determine rather than over wiring. For every division `P` of `D` into inference blocks of at most `c` gates, each
reading at most `k` fresh input bits, the partial sums passed between inference blocks can be charged to the inference
blocks that receive them, `s u` to inference block `u`, so that an entry whose products lie in `m` inference blocks is
charged at least `m − 1` in total, and the gates of each inference block take at least
`2^(b_w · #(weight entries it uses) + b_a · s u)` distinct values over the runs of `D` with the registered information
at its committed values `θv`.

Like `ActivationsSupport`, wiring cannot falsify it, and `b_w` and `b_a` are rates in bits of support per entry, below an
entry's width by whatever the updated weights and the partial sums compress to. It is meant for a reference training
circuit, the one a declared circuit contains (`Nci.Guarantees.ContainedMatMul`), where it is a claim about how
compressible training is. An inference block's values are functions of the values on its fresh input reads, so it gives
`PartialSumsIncompressible` at the same caps (`partialSumsIncompressible_of_support`).

**Falsified** for one honest matrix multiplication, with binary
additions and inside the caps: divide it into product inference blocks that fix a column over a block of rows and
positions, and inference blocks that hold the adder tree over `k × k × k` blocks. Each entry's products then lie in `d`
inference blocks, while each inference block's values are functions of the few weight and input entries it depends on.
In the exact 1-bit toy (`d = e = N = 8`, `b_w = b_a = 1`) the inference blocks can be charged at most 183 (192 bits of
entries) against the 448 required, for every `c ≥ 448` and `k ≥ 512`. At `d = 7168` and the deployed `k = 2·10⁷` such a
division absorbs at most 8.4% of the charge at FP4 and 17.8% at FP8 (2.5 and 4 bits). -/
def PartialSumsSupport (D : RegisteredCircuit) (θv : D.D.Transcript) (ch : Chain D.D) (c : ℕ) (k b_w b_a : ℝ) : Prop :=
  ∀ (n : ℕ) (P : Partition D.D n), (∀ u, D.D.size (P.gates u) ≤ c ∧ (novelReads D (P.gates u) : ℝ) ≤ k) →
    ∃ s : Fin n → ℕ,
      ∑ x : (ℓ : Fin ch.L) × Fin ch.N × Fin (ch.e ℓ), (ch.spread P x.1 x.2.1 x.2.2 - 1) ≤ ∑ u, s u ∧
      ∀ u, (2 : ℝ) ^ (b_w * (ch.weightsUsed (P.gates u)).card + b_a * s u) ≤ support D θv (P.gates u)

/-! The parameters the guarantees are taken at, named: what a chain's shape, a run's coverage and a division's caps must
be for a guarantee's factor to apply. Each is checkable on the circuit it reads, and none is about the world. -/

/-- **The chain's inner dimensions are at least `d`**: each of `ch`'s multiplications sums at least `d` products into
every output entry. A factor taken at `d` is taken at the smallest inner dimension or below it. -/
def InnerAtLeast {D : Circuit} (ch : Chain D) (d : ℕ) : Prop :=
  ∀ ℓ, d ≤ ch.d ℓ

/-- **Coverage** (the draft's `κ(T)`): the counted work `x` is at least a fraction `κ` of the reference circuit `r`'s
computed gates, `κ · size(r) ≤ x`. A chain of `G` products counts `x = 2G` (a product and an addition each), a threading's
counted multiplications `x = 2 |M|`, and the cover theorem's certified products `x = |P|`. A training run whose counted
work is a small share of its gates is slowed in proportion to that share. -/
def Coverage (κ : ℝ) (r : RegisteredCircuit) (x : ℝ) : Prop :=
  κ * r.D.size r.D.computed ≤ x

/-- **Small inference blocks**: each block of the division `P` of `D` has at most `c` gates and reads at most `k` fresh
input bits, the caps under which `PartialSumsIncompressible` charges a division. -/
def SmallBlocks (D : RegisteredCircuit) {n : ℕ} (P : Partition D.D n) (c : ℕ) (k : ℝ) : Prop :=
  ∀ u, D.D.size (P.gates u) ≤ c ∧ (novelReads D (P.gates u) : ℝ) ≤ k

/-- **Sessions of exactly `c` gates, each reading its own prompt of at most `k` bits**: each block `u` of the division
`P` of `d` has exactly `c` gates, padding included, and reads from outside itself only registered information and its
prompt `prompt u`, at most `k` bits wide. It is the layout an inference deployment declares, a session per block. -/
def Sessions (d : RegisteredCircuit) {B : ℕ} (P : Partition d.D B) (prompt : Fin B → Finset (Fin d.D.N)) (c k : ℕ) : Prop :=
  (∀ u, (P.gates u).card = c) ∧ (∀ u, d.D.into (P.gates u) ⊆ d.θ ∪ prompt u) ∧ (∀ u, d.D.width (prompt u) ≤ k)

/-- **The training circuit `T` is in the class `𝒯`** (the draft, "From updated matrix multiplications to training"):
every declared circuit `D` that implements `T` and complies at `(c, k)` runs a chain `ch` of updated matrix
multiplications whose inner dimensions are at least `d` and whose products cover a fraction `κ` of `T`'s size, with
compute-bounded incompressibility for `ch` on `D`. It carries `ActivationsIncompressible` and
`PartialSumsIncompressible` on declared circuits, and is false wherever they are. -/
def InClass (T : RegisteredCircuit) (c k : ℕ) (b_w b_a κ : ℝ) (d : ℕ) : Prop :=
  ∀ D, T.Implements D → Complies c k D → ∃ ch : Chain D.D,
    InnerAtLeast ch d ∧ Coverage κ T (2 * ch.G) ∧
    ActivationsIncompressible D ch c k b_w b_a ∧ PartialSumsIncompressible D ch c k b_w b_a

end Nci.Assumptions
