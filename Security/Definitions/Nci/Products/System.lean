import Definitions.Nci.Products.Received

/-!
# Products of two fresh operands, made whole through chains: the whole-making system

A block can have an operand entry without paying its width: it **makes it whole**, computing it through elementwise
steps from output entries of other products whose whole chains it runs itself. The **whole-making system** (`System`)
is the products whose two operands both depend on what the run produced, each by the standard algorithm (`Standard`):
the counted ones, attention's, the adapters' and the products of an array with itself. They are grouped in classes `j`
by chain length `K_j`, each product at a nesting depth, and each value an operand entry carries has the output entries
it is made from (`System.src`). A value is an operand entry's cone (`Standard.coneA`, `Standard.coneB`), so two products
that read one array have equal cones there.

For a block `I`: `Standard.wholeOut I`, the output entries whose whole chain `I` holds; `Standard.hadA I` and
`Standard.hadB I`, the operand entries it holds or reads every bit of; `Standard.hadCones I`, their values; and
`System.hadValues F I`, the values a set `F` of products has in `I`.

Four premises, facts of the given circuit, named as io_bound's are:

* **Values** (`System.Values`): two operand entries' cones are equal or disjoint, and none is empty.
* **P-share** (`System.Shared`) at constants `c_j`: in every inference block, any set `F` of class `j`'s products has
  `Σ_{p ∈ F} |hadA_p| |hadB_p| ≤ c_j |hadValues F|²`. It holds at `c_j = 1/4` where no two products read the same pair
  of values and no value is read on both sides of the class's products, and at `c_j = 1` where only the first holds (a
  product of an array with itself); io_bound's `c_j = 1/3` is Motzkin–Straus at clique number 3.
* **P-width, extended** (`System.WidthExtended`) at width `b`: every value of a system product that an inference block
  has, the block pays at its width, reading `b` bits of its cone beyond `θ`, or makes whole, holding the whole chain of
  one of the output entries it is made from. A value obtained for less any other way, a gather's rows repeated over the
  tokens that share an index among them, stays outside.
* **P-match** (`System.Matched`) at fan-outs `A_ij`: each value is made from output entries of products nested below
  every product that reads it, and each output entry of a class-`i` product makes at most `A_ij` of class `j`'s values,
  elementwise steps matching operand entries one to one (fan-out at most fan-in).

The theorems (`Nci.Products.Guarantees.ClassBound`, `WholeCap`, `WholeBound`) read the system at a **post-fixed point**
`μ ≥ 0`, `r_j (1 + Σ_i A_ij μ_i)² ≤ μ_j` for every class with `r_j = c_j k / (K_j² b)`, of which io_bound's
`whole_chain_of` computes the least, and a product's **whole-making factor** `e = Σ_i A_ij μ_i` at its class `j`
(`wholeFactor`).
`wholeBound` is the products kind's bound on `ρ` that the cap with operands made whole gives: `chainBlocks N
(α' + k) (β' √O) · c / |C|`, with `α' = (k'/3) √(k'/3) / √(b b b_C)` and `β' = (k'/2) / b` at `k' = k (1 + e)`.
-/

namespace Nci.Products

open Finset
open Verity (Declaration)
open Nci.Formalism
open scoped ENNReal

namespace Standard

variable {d : Declaration} {ni nl nj : ℕ} (S : Standard d ni nl nj)

/-- The output entries whose whole chain `I` holds. -/
def wholeOut (I : Finset (Fin d.D.N)) : Finset (Fin ni × Fin nj) :=
  univ.filter fun e => S.entry e.1 e.2 ⊆ I

/-- The entries of `A` that `I` holds or reads every bit of. -/
def hadA (I : Finset (Fin d.D.N)) : Finset (Fin ni × Fin nl) :=
  univ.filter fun e => S.opA e.1 e.2 ⊆ I ∪ d.D.into I

/-- The entries of `B` that `I` holds or reads every bit of. -/
def hadB (I : Finset (Fin d.D.N)) : Finset (Fin nl × Fin nj) :=
  univ.filter fun e => S.opB e.1 e.2 ⊆ I ∪ d.D.into I

/-- The values of the operand entries `I` has: their cones. -/
def hadCones (I : Finset (Fin d.D.N)) : Finset (Finset (Fin d.D.N)) :=
  (S.hadA I).image (fun e => S.coneA e.1 e.2) ∪ (S.hadB I).image fun e => S.coneB e.1 e.2

/-- The values of all its operand entries. -/
def values : Finset (Finset (Fin d.D.N)) :=
  univ.image (fun e : Fin ni × Fin nl => S.coneA e.1 e.2) ∪ univ.image fun e : Fin nl × Fin nj => S.coneB e.1 e.2

end Standard

/-- **A whole-making system** in `d`: products `p : ι`, each `C_p = A_p · B_p` by the standard algorithm with chains of
its class's length `K (cls p)`, at nesting depth `depth p`, and the output entries `src v` each value `v` is made from. -/
structure System (d : Declaration) (ι κ : Type) where
  /-- Each class's chain length. -/
  K : κ → ℕ
  /-- Each product's class. -/
  cls : ι → κ
  /-- Its output's rows. -/
  ni : ι → ℕ
  /-- Its output's columns. -/
  nj : ι → ℕ
  /-- The product. -/
  prod : (p : ι) → Standard d (ni p) (K (cls p)) (nj p)
  /-- Its nesting depth. -/
  depth : ι → ℕ
  /-- The output entries a value is made from. -/
  src : Finset (Fin d.D.N) → Finset ((q : ι) × Fin (ni q) × Fin (nj q))

namespace System

variable {d : Declaration} {ι κ : Type} [Fintype ι] [DecidableEq ι] (sys : System d ι κ)

/-- The values of class `j`'s products. -/
def classValues [DecidableEq κ] (j : κ) : Finset (Finset (Fin d.D.N)) :=
  (univ.filter fun p => sys.cls p = j).biUnion fun p => (sys.prod p).values

/-- The values the products `F` have in `I`. -/
def hadValues (F : Finset ι) (I : Finset (Fin d.D.N)) : Finset (Finset (Fin d.D.N)) :=
  F.biUnion fun p => (sys.prod p).hadCones I

/-- **Values**, a fact of the given circuit: two operand entries' cones are equal or disjoint, and none is empty. -/
def Values : Prop :=
  ∀ p q, ∀ v ∈ (sys.prod p).values, ∀ w ∈ (sys.prod q).values, v.Nonempty ∧ (v = w ∨ Disjoint v w)

/-- **P-share** at `c_j`, a fact of the given circuit at the policy's `(c, k)`: in every inference block, any set `F` of
class `j`'s products has `Σ_{p ∈ F} |hadA_p| |hadB_p| ≤ c_j |hadValues F|²`. -/
def Shared (c k : ℕ) (cs : κ → ℝ) : Prop :=
  ∀ I, InferenceBlock d c k I → ∀ j (F : Finset ι), (∀ p ∈ F, sys.cls p = j) →
    ((∑ p ∈ F, ((sys.prod p).hadA I).card * ((sys.prod p).hadB I).card : ℕ) : ℝ) ≤
      cs j * ((sys.hadValues F I).card : ℝ) ^ 2

/-- **P-width, extended** at width `b`, a fact of the given circuit at the policy's `(c, k)`: every value of a system
product that an inference block has, the block reads `b` bits of beyond `θ`, or it holds the whole chain of one of the
output entries the value is made from. -/
def WidthExtended (c k b : ℕ) : Prop :=
  ∀ I, InferenceBlock d c k I → ∀ p, ∀ v ∈ (sys.prod p).hadCones I,
    b ≤ (d.unregistered I ∩ v).card ∨ ∃ x ∈ sys.src v, (sys.prod x.1).entry x.2.1 x.2.2 ⊆ I

/-- **P-match** at fan-outs `A_ij`, a fact of the given circuit: each value is made from output entries of products
nested below every product that reads it, and each output entry of a class-`i` product makes at most `A_ij` of class
`j`'s values. -/
def Matched [DecidableEq κ] (A : κ → κ → ℝ) : Prop :=
  (∀ p, ∀ v ∈ (sys.prod p).values, ∀ x ∈ sys.src v, sys.depth x.1 < sys.depth p) ∧
    ∀ x j, (((sys.classValues j).filter fun v => x ∈ sys.src v).card : ℝ) ≤ A (sys.cls x.1) j

end System

/-- **The whole-making factor** of class `j` at fan-outs `A` and a post-fixed point `μ`: `Σ_i A_ij μ_i`, the values a
block makes whole for one of class `j`'s products, in units of `k / b`. -/
def wholeFactor {κ : Type} [Fintype κ] (A : κ → κ → ℝ) (μ : κ → ℝ) (j : κ) : ℝ := ∑ i, A i j * μ i

/-- **The products kind's bound with operands made whole**: `chainBlocks N (α' + k) (β' √O) · c / |C|` for a product of
`N` MACs and `O` output entries, with `α' = (k'/3) √(k'/3) / √(b b b_C)` and `β' = (k'/2) / √(b b)` at the bits
`k' = k (1 + e)` its blocks have for operands and partial sums, `e` its whole-making factor. -/
noncomputable def wholeBound (d : Declaration) (N O c k b bC : ℕ) (e : ℝ) : ℝ≥0∞ :=
  ENNReal.ofReal (chainBlocks N
      ((k : ℝ) * (1 + e) / 3 * √((k : ℝ) * (1 + e) / 3) / √((b : ℝ) * b * bC) + k)
      ((k : ℝ) * (1 + e) / 2 / √((b : ℝ) * b) * √(O : ℝ))) *
    (c : ℝ≥0∞) / (d.D.computed.card : ℝ≥0∞)

end Nci.Products
