import Mathlib.Analysis.Real.Sqrt
import Definitions.Nci.Products.System

/-!
# Products of two fresh operands, made whole through chains: the class bound, the cap and its bound on `ρ`

The theorems about a whole-making system (`Definitions.Nci.Products.System`), products by the standard algorithm whose
operands a block may pay at their width or make whole, proved in `Nci.Products.SecurityProofs`. `k` is the bits an
inference block reads beyond `θ`, `b` an operand's width, `μ` a post-fixed point of the system,
`r_j (1 + Σ_i A_ij μ_i)² ≤ μ_j` with `r_j = c_j k / (K_j² b)`, and `e_j = Σ_i A_ij μ_i` class `j`'s whole-making factor
(`wholeFactor`).

* **The class bound** (`ClassBound`): an inference block holds the whole chains of at most `μ_j k / b` output entries of
  class `j`'s products. By induction on nesting depth: a product's whole chains in a block are at most
  `|had A| |had B| / K²` by Loomis–Whitney, P-share sums them over the class's products at most nested `t` deep into
  `c_j |had values|² / K_j²`, and each had value is paid, at most `k / b` of them, or made whole from an output entry
  nested below `t`, at most `Σ_i A_ij μ_i k / b` of them by P-match and the bound below `t`.
* **The cap with operands made whole** (`WholeCap`): a product `q` of the system with its partial sums received at their
  width has, in every partition into `m` inference blocks, `N ≤ m (α' + k) + β' √(m O)`, with `N` its MACs, `O` its
  output entries, `α' = (k'/3) √(k'/3) / √(b b b_C)` and `β' = (k'/2) / b` at `k' = k (1 + e)`, `e` its class's
  whole-making factor. A block's whole MACs pay their operands through the block's reads of their cones or the values the
  block makes whole, at most `e k / b` by the class bound, so the cap without the word level
  (`Nci.Products.Guarantees.NoCellCap`) holds with `k'` for `k` in `α` and `β`; the split MACs stay at `k` a block.
* **The products kind's bound with operands made whole** (`WholeBound`): for any set `G` of the system's products with
  their partial sums received, `ρ ≥ max_{q ∈ G} chainBlocks N_q (α'_q + k) (β'_q √O_q) c / |C|` (`wholeBound`), a
  partition-level bound (`Nci.Formalism.PartitionBound`). It holds on any subset, so the counted set may leave out any
  product, the products that read a gather's rows before any token-mixing step among them. It bounds by one product at
  a time: the sum over products, several in one block (the note's `Λ`, `SHARE`), is the next lemma.

The premises are named as io_bound's: Values; P-share (`System.Shared`); P-width extended (`System.WidthExtended`),
which takes the place of widths as passed's operand side (`Standard.Passed`, P-width) for the system's products;
P-match (`System.Matched`); and the partial sums received at their width (`Standard.Received`, P-width's read side with
P-chain). None reads P-cell (`Standard.Unsplit`) or the chains' order (`Standard.Chained`). The post-fixed point is a
condition on numbers the theorems take, not a fact of the circuit.

Each has an honest and a nontrivial lemma (`Definitions.Nci.Guarantees.Products.WholeWitnesses`).
-/

namespace Nci.Products.Guarantees

open Finset
open Verity (Declaration Partition)
open Nci.Formalism Nci.Products

/-- **The class bound**: in a whole-making system with values, P-share at `c_j`, P-width extended at `b` and P-match at
`A_ij`, and `μ` a post-fixed point, `r_j (1 + Σ_i A_ij μ_i)² ≤ μ_j` with `r_j = c_j k / (K_j² b)`, every inference
block holds the whole chains of at most `μ_j k / b` output entries of class `j`'s products. -/
def ClassBound : Prop :=
  ∀ (d : Declaration) (ι κ : Type) [Fintype ι] [DecidableEq ι] [Fintype κ] [DecidableEq κ] (sys : System d ι κ)
    (c k b : ℕ) (cs : κ → ℝ) (A : κ → κ → ℝ) (μ : κ → ℝ) (I : Finset (Fin d.D.N)) (j : κ),
    (∀ p, (sys.prod p).Algorithm) → sys.Values → sys.Shared c k cs → sys.WidthExtended c k b → sys.Matched A →
    0 < b → (∀ j, 0 < sys.K j) → (∀ j, 0 ≤ cs j) → (∀ i j, 0 ≤ A i j) →
    (∀ j, cs j * k / ((sys.K j : ℝ) ^ 2 * b) * (1 + wholeFactor A μ j) ^ 2 ≤ μ j) → InferenceBlock d c k I →
    ((∑ p ∈ univ.filter (fun p => sys.cls p = j), ((sys.prod p).wholeOut I).card : ℕ) : ℝ) ≤ μ j * k / b

/-- **The cap with operands made whole**: a product `q` of a whole-making system, with the class bound's premises and
its partial sums received at their width, has in every partition into `m` inference blocks
`N ≤ m ((k'/3) √(k'/3) / √(b b b_C) + k) + (k'/2) / √(b b) · √(m O)` at `k' = k (1 + e)`, `N` its MACs, `O` its
output entries and `e` its class's whole-making factor. -/
def WholeCap : Prop :=
  ∀ (d : Declaration) (ι κ : Type) [Fintype ι] [DecidableEq ι] [Fintype κ] [DecidableEq κ] (sys : System d ι κ)
    (c k b bC : ℕ) (cs : κ → ℝ) (A : κ → κ → ℝ) (μ : κ → ℝ) (q : ι) (m : ℕ) (P : Partition d.D m),
    (∀ p, (sys.prod p).Algorithm) → sys.Values → sys.Shared c k cs → sys.WidthExtended c k b → sys.Matched A →
    (sys.prod q).Received c k bC → P.IsCK d c k → 0 < b → 0 < bC → (∀ j, 0 < sys.K j) → (∀ j, 0 ≤ cs j) →
    (∀ i j, 0 ≤ A i j) → (∀ j, cs j * k / ((sys.K j : ℝ) ^ 2 * b) * (1 + wholeFactor A μ j) ^ 2 ≤ μ j) →
    ((sys.ni q * sys.K (sys.cls q) * sys.nj q : ℕ) : ℝ) ≤
      m * ((k : ℝ) * (1 + wholeFactor A μ (sys.cls q)) / 3 * √((k : ℝ) * (1 + wholeFactor A μ (sys.cls q)) / 3) /
          √((b : ℝ) * b * bC) + k) +
        (k : ℝ) * (1 + wholeFactor A μ (sys.cls q)) / 2 / √((b : ℝ) * b) * √((m : ℝ) * (sys.ni q * sys.nj q : ℕ))

/-- **The products kind's bound with operands made whole**: for any set `G` of a whole-making system's products with
their partial sums received at their width, every partition `𝓘` into inference blocks has
`|𝓘| c / |C| ≥ max_{q ∈ G} chainBlocks N_q (α'_q + k) (β'_q √O_q) · c / |C|`, a partition-level bound. -/
def WholeBound : Prop :=
  ∀ (d : Declaration) (ι κ : Type) [Fintype ι] [DecidableEq ι] [Fintype κ] [DecidableEq κ] (sys : System d ι κ)
    (c k b bC : ℕ) (cs : κ → ℝ) (A : κ → κ → ℝ) (μ : κ → ℝ) (G : Finset ι),
    (∀ p, (sys.prod p).Algorithm) → sys.Values → sys.Shared c k cs → sys.WidthExtended c k b → sys.Matched A →
    (∀ q ∈ G, (sys.prod q).Received c k bC) → 0 < b → 0 < bC → (∀ j, 0 < sys.K j) → (∀ j, 0 ≤ cs j) →
    (∀ i j, 0 ≤ A i j) → (∀ j, cs j * k / ((sys.K j : ℝ) ^ 2 * b) * (1 + wholeFactor A μ j) ^ 2 ≤ μ j) →
    PartitionBound d c k (G.sup fun q => wholeBound d (sys.ni q * sys.K (sys.cls q) * sys.nj q)
      (sys.ni q * sys.nj q) c k b bC (wholeFactor A μ (sys.cls q)))

end Nci.Products.Guarantees
