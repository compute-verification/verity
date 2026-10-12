import Definitions.Core
import Mathlib.Analysis.Real.Sqrt

/-!
# Novel communication intensity: the policy

The developer declares `(D, θ)` (`Verity.Declaration`). The input gates in `θ` hold baseline information, committed
beforehand; every other value is novel information. A division of `D` into isolation units (IUs) is a
`Verity.Partition` of its computed gates. Input gates are in no IU, so an IU that uses an input reads it. The gates `R`
of an IU read `width(in(R) \ θ)` bits of novel information (`novelReads`): baseline information is free, and so is
anything `R` computes itself. `R`'s novel communication intensity is `NCI(R) = novelReads R / size R`, in bits per gate.

`Complies Γ α` is the NCI policy: `D` divides into IUs of at most `Γ` gates, each with NCI at most `α`. A policy is a
condition on declarations that the verifier checks from `D` and `θ` alone (`Policy`), a protocol with no rounds of its
own. `slowdown policy t` is the factor by which every declaration that complies with `policy` and implements `t` is
larger than `t`. `Reads Γ c` is the least total novel information that the IUs of a division of `c` into IUs of at most
`Γ` gates read.

The draft's necessary subcomputation of frontier training is an updated matrix multiplication chain (`Chain`): the
gates where a circuit computes the products, the complete output entries, and the links between consecutive
multiplications. For a set `R` of gates it gives the draft's vocabulary: the weight entries `R` uses, the input entries
it uses, the entries whole in `R`, `R`'s full positions, and in a division into IUs the entries that are split and the
number of IUs that hold products of each. `F` is the slowdown factor the draft proves for such chains.
-/

namespace Nci

open Verity
open scoped ENNReal

/-- The novel information that the gates `R` of `d`'s circuit read, in bits: the width of `in(R) \ θ`. -/
def novelReads (d : Declaration) (R : Finset (Fin d.D.N)) : ℕ := d.D.width (d.D.into R \ d.θ)

/-- A policy: a condition on declarations. -/
abbrev Policy : Type := Declaration → Prop

/-- **The NCI policy** at maximum IU size `Γ` and NCI limit `α`: `d`'s circuit divides into IUs of at most `Γ` gates,
each reading at most `α` bits of novel information per gate (`NCI(R) ≤ α`, multiplied out so that an IU with no gates
needs no division). -/
def Complies (Γ : ℕ) (α : ℝ) : Policy := fun d =>
  ∃ (n : ℕ) (P : Partition d.D n),
    ∀ u, d.D.size (P.gates u) ≤ Γ ∧ (novelReads d (P.gates u) : ℝ) ≤ α * d.D.size (P.gates u)

/-- `Reads_Γ(c)`: the least total novel information that the IUs of a division of `c` into IUs of at most `Γ` gates
read. -/
noncomputable def Reads (Γ : ℕ) (c : Declaration) : ℕ :=
  sInf {r | ∃ (n : ℕ) (P : Partition c.D n), (∀ u, c.D.size (P.gates u) ≤ Γ) ∧ r = ∑ u, novelReads c (P.gates u)}

/-- **The slowdown factor** `ρ_policy(t)`: the least ratio of `size(d)` to `size(t)` over the declarations `d` that
implement `t` and comply with `policy`, and `⊤` when none does. -/
noncomputable def slowdown (policy : Policy) (t : Declaration) : ℝ≥0∞ :=
  ⨅ (d : Declaration) (_ : t.Implements d ∧ policy d), (d.D.size d.D.computed : ℝ≥0∞) / t.D.size t.D.computed

/-- **An updated matrix multiplication chain that `D` runs** (the draft, "Updated matrix multiplications"): weight
matrices computed after the commitment and, at each of `N` token positions `t`, multiplications `y_t = W_ℓ x_t`. The
draft numbers the multiplications from 1 and writes `y_t^{(ℓ)} = W_ℓ x_t^{(ℓ-1)}`. Here they are numbered from 0, so the
inputs and inner dimension of multiplication `ℓ` are the draft's `x^{(ℓ)}` and `d_ℓ`, and its outputs, weights, output
dimension and link are the draft's `y^{(ℓ+1)}`, `W_{ℓ+1}`, `e_{ℓ+1}` and `π_{ℓ+1}`. The chain names gates and no values,
since the counting never reads values. `D` runs the chain when:
1. **Products:** it computes each product `W_ℓ[i, j] x_t^{(ℓ)}[j]` with its own gate, `prod ℓ t i j`;
2. **Sums:** it computes each entry `y_t^{(ℓ)}[i]` by adding up its products, in any order, so each of them feeds the
   gate `y ℓ t i` that holds the complete entry;
3. **Links:** it computes each entry `x_t^{(ℓ+1)}[j]` using the complete entry `y_t^{(ℓ)}[π ℓ j]`, possibly together
   with other values, where `π ℓ` is injective, so that gate feeds the products of `x_t^{(ℓ+1)}[j]`.

`D` may perform arbitrary additional computation. -/
structure Chain (D : Circuit) where
  /-- The depth: the number of multiplications. -/
  L : ℕ
  /-- The number of token positions. -/
  N : ℕ
  /-- The inner dimension of multiplication `ℓ`: the length of its input vectors, the number of columns of `W_ℓ`. -/
  d : Fin L → ℕ
  /-- The output dimension of multiplication `ℓ`: the length of its output vectors, the number of rows of `W_ℓ`. -/
  e : Fin L → ℕ
  /-- The gate that computes the product `W_ℓ[i, j] x_t^{(ℓ)}[j]`. -/
  prod : (ℓ : Fin L) → Fin N → Fin (e ℓ) → Fin (d ℓ) → Fin D.N
  prod_inj : Function.Injective fun p : (ℓ : Fin L) × Fin N × Fin (e ℓ) × Fin (d ℓ) => prod p.1 p.2.1 p.2.2.1 p.2.2.2
  prod_computed : ∀ ℓ t i j, D.isInput (prod ℓ t i j) = false
  /-- The gate that holds the complete entry `y_t^{(ℓ)}[i]`. -/
  y : (ℓ : Fin L) → Fin N → Fin (e ℓ) → Fin D.N
  sums : ∀ ℓ t i j, D.Feeds (prod ℓ t i j) (y ℓ t i)
  /-- The link into multiplication `ℓ + 1`: its input entry `j` is computed using the output entry `π ℓ h j` of
  multiplication `ℓ`. -/
  π : (ℓ : Fin L) → (h : ℓ.val + 1 < L) → Fin (d ⟨ℓ.val + 1, h⟩) → Fin (e ℓ)
  π_inj : ∀ ℓ h, Function.Injective (π ℓ h)
  links : ∀ ℓ h t i j, D.Feeds (y ℓ t (π ℓ h j)) (prod ⟨ℓ.val + 1, h⟩ t i j)

namespace Chain

open Finset

variable {D : Circuit} (c : Chain D)

/-- `G = ∑_ℓ G_ℓ`, the number of products of the chain: multiplication `ℓ` computes `G_ℓ = e_ℓ d_ℓ N` of them (the
draft's `G_ℓ = e_ℓ d_{ℓ-1} N`). -/
def G : ℕ := ∑ ℓ, c.e ℓ * c.d ℓ * c.N

/-- The weight entries `W_ℓ[i, j]` that `R` uses: those of which `R` holds a product. -/
def weightsUsed (R : Finset (Fin D.N)) : Finset ((ℓ : Fin c.L) × Fin (c.e ℓ) × Fin (c.d ℓ)) :=
  univ.filter fun w => ∃ t, c.prod w.1 t w.2.1 w.2.2 ∈ R

/-- The input entries `x_t^{(ℓ)}[j]` that `R` uses: those of which `R` holds a product. -/
def inputsUsed (R : Finset (Fin D.N)) : Finset ((ℓ : Fin c.L) × Fin c.N × Fin (c.d ℓ)) :=
  univ.filter fun x => ∃ i, c.prod x.1 x.2.1 i x.2.2 ∈ R

/-- The entry `y_t^{(ℓ)}[i]` is **whole** in `R` when all of its products lie in `R`. (The draft, "Throttling updated
matrix multiplications", calls an entry whole if all of its products lie in one IU, and split otherwise.) -/
def Whole (R : Finset (Fin D.N)) (ℓ : Fin c.L) (t : Fin c.N) (i : Fin (c.e ℓ)) : Prop :=
  ∀ j, c.prod ℓ t i j ∈ R

/-- The entry `y_t^{(ℓ)}[i]` is **split** in a division `P` into IUs: no IU holds all of its products. -/
def Split {n : ℕ} (P : Partition D n) (ℓ : Fin c.L) (t : Fin c.N) (i : Fin (c.e ℓ)) : Prop :=
  ¬ ∃ u, c.Whole (P.gates u) ℓ t i

/-- Position `t` is a **full position** of `R` if `R` computes some whole entry at position `t` (the draft). -/
def Full (R : Finset (Fin D.N)) (t : Fin c.N) : Prop :=
  ∃ ℓ i, c.Whole R ℓ t i

/-- The input entry `x_t^{(ℓ)}[j]` is **linked to an entry whole in `R`**: `ℓ > 0`, and the entry
`y_t^{(ℓ-1)}[π j]` it is computed from is whole in `R`, so that `R` can compute it from its own products. -/
def LinkedWhole (R : Finset (Fin D.N)) : (ℓ : Fin c.L) → Fin c.N → Fin (c.d ℓ) → Prop
  | ⟨0, _⟩, _, _ => False
  | ⟨ℓ + 1, h⟩, t, j => c.Whole R ⟨ℓ, by omega⟩ t (c.π ⟨ℓ, by omega⟩ h j)

open Classical in
/-- The number of IUs of a division `P` that hold products of the entry `y_t^{(ℓ)}[i]`. -/
noncomputable def spread {n : ℕ} (P : Partition D n) (ℓ : Fin c.L) (t : Fin c.N) (i : Fin (c.e ℓ)) : ℕ :=
  (univ.filter fun u => ∃ j, c.prod ℓ t i j ∈ P.gates u).card

end Chain

/-- **The slowdown factor for updated matrix multiplications** `F` (the draft's theorem, "Throttling updated matrix
multiplications"), at NCI limit `α`, read cap `K`, widths `b_w` of a weight entry and `b_a` of an activation entry, and
smallest inner dimension `d`:
`F = 1/(2α) · max_{0<μ<1} min{(1 + √μ)² b_w b_a d / K, √2 (1 + √μ) √(1 − μ) b_a √(b_w / K)}`,
the maximum written as a supremum. The first term covers IUs that complete entries themselves, the second IUs that split
entries with other IUs. -/
noncomputable def F (α K b_w b_a d : ℝ) : ℝ :=
  1 / (2 * α) * ⨆ μ : Set.Ioo (0 : ℝ) 1,
    min ((1 + √(μ : ℝ)) ^ 2 * b_w * b_a * d / K) (√2 * (1 + √(μ : ℝ)) * √(1 - μ) * b_a * √(b_w / K))

end Nci
