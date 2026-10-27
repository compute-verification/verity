import Definitions.Accounting.FixedMachine.Instances

/-!
# A fixed machine's proof blocks are described within their budgets

The working draft's Lemma 1 (`Verity.Guarantees.BlockDescription`) for an actual circuit that is a fixed machine
(`Accounting.FixedMachine`): register `H` run `T` steps, fix a decoder and an evaluator that performs an extraction rule
on every proof block of that registration, and only then take an actual circuit `A` that is `H` run `T` steps, with each
proof block the region of one location over the run. Every proof block's extracted string is then described within the
location's initial budget plus the bits read into it over the run (`BlockDescribed`): its initial state and ingress are
all it starts from. The evaluator and the decoder are fixed by `H` before `A` is chosen, so a circuit built for one
candidate gets no bits for free.

* `BlockDescription`: the bound;
* `GateCount`: `A`'s work is `T · |H|` gates;
* `BlockDescription_honest`: every premise of the bound holds together at the honest machine (`FixedMachine.Instances`);
* `BlockDescription_nontrivial`: at the hard-coding circuit `Hard` the other premises hold, `Hard` is no fixed machine
  of `H₀` at any number of steps, and the bound fails.
-/

namespace Accounting.Guarantees.FixedMachine

open Verity Finset Accounting.FixedMachine

/-- **A fixed machine's proof blocks are described within their budgets.** Let `R` register `H` run `T` steps, `H`
Boolean, and let a fixed evaluator `e` of at most `a_blk` bits perform an extraction rule `y` within time `t` for every
proof block of `R`, at the decoder `M` with registered information `r`. Let `A` be an actual circuit that is `H` run
`T` steps (`FixedMachine`), its gates renumbered by `φ`, each proof block `u` of `R` the region of a location `X u` over
the run, with initial information `a u` within `D u` bits (`InitialBudget`, `R`'s registered gates as registered). Then
in every run of `A` that holds the registered values, each proof block's extracted string has local description length
at most `D u + in_{X u}([0, ∞)) + ⌈log₂ B⌉ + a_blk` (`BlockDescribed`). -/
def BlockDescription : Prop :=
  ∀ (H : Circuit) (S : H.StateWires) (T : ℕ) (R : Registration), R.d.D = H.unroll S T → H.Boolean →
    ∀ (ρ : Type) (M : Decoder ρ) (r : ρ) (t a_blk : ℕ) (e : List Bool)
      (y : Fin R.B → R.d.D.Transcript → List Bool), e.length ≤ a_blk → Evaluates M r t R e y →
      ∀ (L : Type) (A : Placed L), Accounting.FixedMachine H S T A →
      ∀ (φ : Circuit.Iso R.d.D A.toCircuit) (X : Fin R.B → Finset L)
        (a : Fin R.B → Finset (Fin A.N)) (D : Fin R.B → ℕ),
        (∀ u, (R.P.gates u).map φ.e.toEmbedding = A.region (X u) (Set.Ici 0)) →
        (∀ u, InitialBudget A (R.d.θ.map φ.e.toEmbedding) (a u) (X u) (D u)) →
        ∀ x : A.Transcript, R.Honest (φ.pull x) →
          ∀ u, BlockDescribed M r t A (X u) (D u) R.B a_blk (y u (φ.pull (A.eval x)))

/-- **A fixed machine's work is `T · |H|`**: an actual circuit that is `H` run `T` steps has `T` times `H`'s operation
gates. -/
def GateCount : Prop :=
  ∀ (H : Circuit) (S : H.StateWires) (T : ℕ) (L : Type) (A : Placed L), Accounting.FixedMachine H S T A →
    A.size A.computed = T * H.size H.computed

/-- **`BlockDescription`'s premises hold together at the honest machine**: `R₀` registers `H₀` run `T₀` steps, `H₀` is
Boolean, the empty evaluator is within `0` bits and performs the final-state extraction `y₀` on `R₀`'s proof block at
`M₀`, `A₀` is `H₀` run `T₀` steps, its one location's region is the proof block, the initial state is its initial
information within `D₀ = 1` bit, and every run of `A₀` holds `R₀`'s registered values. -/
def BlockDescription_honest : Prop :=
  R₀.d.D = H₀.unroll S₀ T₀ ∧ H₀.Boolean ∧ ([] : List Bool).length ≤ 0 ∧ Evaluates M₀ () 0 R₀ [] y₀ ∧
    Accounting.FixedMachine H₀ S₀ T₀ A₀ ∧
    (∀ u, (R₀.P.gates u).map φ₀.e.toEmbedding = A₀.region (X₀ u) (Set.Ici 0)) ∧
    (∀ u, InitialBudget A₀ (R₀.d.θ.map φ₀.e.toEmbedding) (a₀ u) (X₀ u) (D₀ u)) ∧
    ∀ x, R₀.Honest (φ₀.pull x)

/-- **Without `FixedMachine`, an explicit circuit beats `BlockDescription`'s bound**: at the honest machine's
registration, decoder and evaluator (`R₀` registers `H₀` run `T₀` steps, `H₀` Boolean, the empty evaluator within `0`
bits performing `y₀` on `R₀` at `M₀`), the hard-coding circuit `Hard` meets the other premises in its own gate
numbering (its one proof block, its operation gate, is the region of its one location; with gate `0` registered its
initial budget is `0` bits), it is no fixed machine of `H₀` at any number of steps, and its string `hardString` is not
described within its `0`-bit budget, no ingress, one proof block and the empty evaluator. The premises that read the
renumbering `φ` have none to read at `Hard`. -/
def BlockDescription_nontrivial : Prop :=
  R₀.d.D = H₀.unroll S₀ T₀ ∧ H₀.Boolean ∧ ([] : List Bool).length ≤ 0 ∧ Evaluates M₀ () 0 R₀ [] y₀ ∧
    (∀ T, ¬ Accounting.FixedMachine H₀ S₀ T Hard) ∧
    Hard.region univ (Set.Ici 0) = {⟨1, by decide⟩} ∧ InitialBudget Hard {⟨0, by decide⟩} ∅ univ 0 ∧
    ¬ BlockDescribed M₀ () 0 Hard univ 0 1 0 hardString

end Accounting.Guarantees.FixedMachine
