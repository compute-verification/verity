# The training slowdown

Training's one statement: a circuit that trains, in the sense of T1 and T2 below, and is also an inference circuit
must be far larger than its own work needs, by the slowdown ρ. The definitions come in order, from circuits up, each
with the Lean declaration that states it; the table at the end lists where the Lean differs. The statements are in
`security/Definitions/Slowdown/Cut.lean`, `security/Definitions/Slowdown/Guarantees/Cut.lean` and, for T1 and T2,
`security/Definitions/Assumptions.lean`; the proofs are in `security/Proofs/Slowdown/Cut.lean`.

## Circuits

**Circuit.** Let Σ be a finite alphabet. A *circuit* C over Σ is a finite directed acyclic graph whose vertices are
*wires*, each taking a value in Σ. A wire with no predecessors is an *input*; every other wire is a *gate*, which
computes a function of its predecessors' values. |C| is the number of gates, and for a set of gates S, in(S) is the set
of wires outside S that are predecessors of a gate of S. Lean: `Verity.Circuit`, whose `Circuit.computed` are the gates,
`C.size C.computed` is |C| and `Circuit.into` is in(S).

**Boolean circuit.** A circuit over {0, 1} in which every gate has at most two predecessors. Lean:
`Verity.Circuit.Boolean`.

**Word.** A *word* of C is a set of wires of C; its *width* is the number of wires in it. Lean: a `Finset (Fin C.N)`.

**Cut.** A set of wires *cuts* A from B if it meets every path from a wire of A to a wire of B, and κ(A → B) is the size
of the smallest such set. For a set θ of inputs of C, κ_θ(B) is κ(A → B) with A the inputs of C outside θ. A set of
words stands for the wires of its words. Once the values on a cut are known, the values on B follow from them and from θ
alone, so at most κ_θ(B) bits from outside θ reach B. Lean: `Slowdown.Cut.Separates`, `Slowdown.Cut.kappa`, and κ_θ of a
set of words is `Slowdown.Cut.kappaWords`, with θ the registered information `Verity.RegisteredCircuit.θ`.

## Matmuls

**Matmul structure and MAC.** A *matmul structure* 𝓜 on a Boolean circuit C is a set of *matmuls*. Each matmul m has
finite index sets for i, j and l, and consists of weights, words w^m_{i,l}; activations, words x^m_{l,j}; outputs, words
y^m_{i,j}; and for each triple (i, j, l) a *MAC* M^m_{i,j,l}, a connected set of gates that *reads* w^m_{i,l} and
x^m_{l,j} (every wire of both is a predecessor of one of its gates) and *feeds* y^m_{i,j} (one of its gates has a path to
a wire of it). Within a matmul, distinct indices give distinct words; MACs are pairwise disjoint, and outputs pairwise
distinct. Lean: `Slowdown.Cut.Work`, a family of `Slowdown.Cut.Matmul`s, with the MAC `Matmul.mac` (`Slowdown.Cut.Connected`)
and `Work.mac_disjoint`.

Notation: W_m and X_m are matmul m's weight words and activation words (`Work.wWords`, `Work.xWords`), and an *operand
word* is a word of some W_m or X_m. N is the number of MACs (`Work.N`) and O the total width of the outputs (`Work.O`).
For a set of gates A, E(A) is the set of outputs fed by a MAC contained in A that have no wire in A (`Work.clear`).

## Inference circuits

The developer registers θ in advance, for example the weights of the models it serves.

**Inference block.** For a Boolean circuit C, a set θ of its inputs and c, k ∈ ℕ, a set I of gates of C is a
*(c, k)-inference block* of (C, θ) if |I| ≤ c and |in(I) ∖ θ| ≤ k. Lean: a part of a `Verity.Partition` that meets
`Verity.Partition.IsCK`, with in(I) ∖ θ as `Verity.RegisteredCircuit.unregistered`.

**Inference circuit.** For c, k, n ∈ ℕ, (C, θ) is a *(c, k, n)-inference circuit* if the gates of C can be partitioned
into at most n sets, each a (c, k)-inference block of (C, θ). It has at most nc gates. Lean:
`InferenceOnly.CompliesAtMost c k n`, the verdict the inference-only policy proves of the registered circuit.

## Training circuits

Training uses updated parameters: bits that are new (not registered), large (many of them are read) and expensive to
compute (not cheaply decompressed from a much smaller description). Two properties of a matmul structure model their
use, one about the operands its MACs read and one about the outputs they feed.

**Property T1 (operands are hard).** For a matmul structure 𝓜 on a Boolean circuit C, a set θ of inputs of C, k ∈ ℕ and a
*price* β(u) ≥ 0 for each operand word u, 𝓜 *satisfies T1* with (k, β) relative to θ if every set S of operand words
with κ_θ(S) ≤ k has

$$ \sum_{u \in S} \beta(u) \le \kappa_\theta(S). $$

Cutting operand words off from the inputs outside θ costs at least the sum of their prices, and a word is charged once,
whichever side of a product it sits on. T1 fails for registered weights (κ_θ = 0) and for fast weights computed from θ
and a prompt of fewer than k bits. Lean: `Slowdown.Cut.Assumptions.Hardness`, named assumption `training-cut-hardness`.

T1 formalizes the three properties above:

1. New: κ_θ(S) is measured from the unregistered inputs. A word that can be computed from θ alone has κ_θ = 0, so T1
   rules it out.
2. Large: the left side grows by the price of each word. If each updated parameter has p bits and the update can be
   compressed by a factor of r, then its price is about p/r.
3. Expensive to compute: the bound holds for every set of words with κ_θ(S) ≤ k, so no k wires can determine many
   updated parameters. This fails if the unregistered information behind the update is small, e.g. if the entire
   training run reads fewer than k unregistered bits.

**Property T2 (exports are wide).** For β_y > 0, 𝓜 *satisfies T2* with β_y if every set A of gates of C has

$$ \kappa\big(A \to E(A)\big) \ge \beta_y\,|E(A)|. $$

Lean: `Slowdown.Cut.Assumptions.Exports`, named assumption `training-cut-exports`.

**Training circuit.** Call two operand words *joined* if some matmul has one among its weight words and the other among
its activation words. For k, ω ∈ ℕ and q, β_y > 0, (C, θ) is a *(k, q, ω, β_y)-training circuit* if C has a matmul
structure 𝓜, with a price β(u) ≥ 0 for each operand word u, such that:

1. 𝓜 satisfies T1 with (k, β) relative to θ;
2. 𝓜 satisfies T2 with β_y;
3. no word is both a weight word and an activation word of one matmul, and each joined pair is joined by only one
   matmul;
4. no ω + 1 operand words are pairwise joined;
5. β(u) β(v) ≥ q for all joined u and v.

Lean states the five conditions as separate premises of `Slowdown.Cut.Guarantees.HE` and
`Slowdown.Cut.Guarantees.HEShared`: conditions 3 to 5 over arrays of words (`Slowdown.Cut.NoSharedPair`,
`Slowdown.Cut.Arrays`, `Slowdown.Cut.CostsAtLeast`).

## The slowdown theorem

**Theorem (training slowdown).** Let (C, θ) be a (k, q, ω, β_y)-training circuit whose matmul structure has N ≥ 1 MACs
and output width O, and suppose (C, θ) is also a (c, k, n)-inference circuit. Let ν := N/|C| be the MACs per gate,
o := O/N the output width per MAC, F := √(2(1 − 1/ω)), and

$$ \alpha := \frac{F}{2}\sqrt{\frac{k^3}{q\,\beta_y}},\qquad \lambda := \frac{F^2 k^2\,o}{4q},\qquad
   \mu := \left(\frac{\sqrt{\lambda}+\sqrt{\lambda+4(k+\alpha)}}{2}\right)^{2}. $$

Then the *slowdown* ρ := nc/|C| satisfies ρ ≥ cν/μ.

cν is the number of MACs in c gates of the circuit, and μ the most of them an inference block can hold on average. F is
1 when every operand word is only ever a weight or only ever an activation, and 2/√3 with backpropagation, where
weights, activations and gradients are multiplied in all three pairs. Lean: `Slowdown.Cut.Guarantees.HE` (F = 1) and
`Slowdown.Cut.Guarantees.HEShared` (`Slowdown.Cut.shareF`), over every partition into blocks of at most k bits outside θ,
and `Slowdown.Cut.Guarantees.CountHE`, `HE` on `InferenceOnly.CompliesAtMost c k n`.

The lemmas read two quantities every matmul structure has. *Pairs within reach*: for a set S of operand words,
π(S) := Σ_m |S ∩ W_m| |S ∩ X_m|, and Π_θ(k) := max {π(S) : κ_θ(S) ≤ k} (`Work.pi`, `Slowdown.Cut.piMax`). *Export
width*: b, the minimum of κ(A → E(A))/|E(A)| over sets of gates A with E(A) nonempty, and ∞ if there is none
(`Work.bY`).

**Lemma (general bound).** Let (C, θ) be a (c, k, n)-inference circuit and 𝓜 a matmul structure on C with N ≥ 1 MACs,
output width O, Π := Π_θ(k) and export width b, ν, o and ρ as in the theorem, and

$$ \alpha_* := \sqrt{\frac{k\,\Pi}{b}},\qquad \lambda_* := \Pi\,o,\qquad
   \mu_* := \left(\frac{\sqrt{\lambda_*}+\sqrt{\lambda_*+4(k+\alpha_*)}}{2}\right)^{2}. $$

Then ρ ≥ cν/μ_*. It assumes nothing about training: T2 says b ≥ β_y, and T1 bounds Π by the third lemma. Lean:
`Slowdown.Cut.Guarantees.Slowdown`, with α_* as `Slowdown.Cut.alpha`.

**Lemma (MAC count).** In the setting of the general bound, N ≤ n(k + α_*) + √(Π n O). Lean:
`Slowdown.Cut.Guarantees.Bound`.

**Lemma (T1 bounds the pairs within reach).** For the matmul structure of a (k, q, ω, β_y)-training circuit,
Π_θ(k) ≤ F²k²/(4q). Lean: proved inside `HE` and `HEShared` (`security/Proofs/Slowdown/Cut.lean`), not stated on its own.

## Where the Lean differs

| The Lean | The statement | To close |
|---|---|---|
| T1 and T2 are named (H) and (E) | T1, T2 | rename; planned |
| `HE` and `HEShared` take T1 at two arrays, or at arrays of a graph, a price per array and a value counted once with its casts (`val`); `Hardness` at one-word arrays and `val = id` is T1 | a price per word, conditions 3 to 5 over words | the lemma on T1 and the theorem over words, for any pattern of joined words (F below √2); not planned |
| the training circuit is five separate premises | one definition | a structure for it, the theorem stated over it; planned |
| `Slowdown` is a weaker closed form, ρ ≥ (c/\|C\|)(N/(k + α_*) − √(Π O N)/(k + α_*)^{3/2}) | ρ ≥ cν/μ_*, and ρ ≥ cν/μ | the MAC count solved exactly for n; planned |
| an inference block is also `Slowdown.Covering.InferenceBlock`, on subcircuits, which `InferenceOnly.SecurityProofs.PartitionCovers` is stated over | one definition | it moves into `Cut` unchanged; one definition is not planned |
| a Boolean circuit holds one bit per gate, every gate reading at least one wire, with no bound on predecessors | at most two predecessors | none: the proofs use no bound |
| every vertex is a gate, the computed gates the statement's gates | wires, inputs and gates | none: Core's circuit, read by every protocol |
| outputs may repeat | outputs pairwise distinct | none: the Lean is the more general |
| `Work.bY` is 0 when no E(A) is nonempty | b = ∞ | none: both give α_* = 0 |
