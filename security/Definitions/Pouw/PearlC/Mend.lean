module

public import Definitions.Pouw.PearlC.FormingPDefs

/-!
# The products a salt cannot make count (definitions only)

rev1's debit (`DeviceRev1`) charges a word's products that no salt of the support can make contribute, so that a program
which skips them gains nothing. A product's value reaches the word only through the step's first truncation (`aligned`):
the group's terms are aligned to its largest exponent by a truncating shift, and a product shifted out contributes 0.

* **The cut** (`cutFold`, `cuts`): per product of one step on the pipeline `p`, whether its aligned contribution is 0,
  each group's exponent `E` being the one `groupStepTotal` aligns to (`maxExp` over the incoming value's and the group's
  products' finite terms). A zero product is cut.
* **Along a word's chain** (`chainStep`, `chainCuts`): atom `t` steps from `+0` when `t` is a multiple of `G` (the first
  atom of a promotion group; with `G = 0` only atom 0), and otherwise from the word atom `t − 1` wrote, as
  `Accum.chainP p G` does; entry `32·t + r` is product `r` of atom `t`.
* **The support's ends** (`endRow`): a row's codes under §5's atom-mix (`FormingP.atomMixP p`) when every element's noise
  line is the flat pair `(flatLine 0, flatLineF s)` of `SaltDead.support`, with `s` clear at the plus end and set at the
  minus end; each end is a salt of the support for every element at once.
* **The forced zero** (`ForcedZero`): product `l` of the word of rows `x` and `w` is cut at both ends.
  `CutForEverySalt` is the property the debit stands for: the product is cut at every choice of support pairs for the
  two rows' elements (`saltRow`); it gives `ForcedZero`, since the two ends are such choices
  (`Proofs.Pouw.PearlC.Mend.forcedZero_of_cutForEverySalt`).

This module imports no proofs, so the named protocol may read it.
-/

@[expose] public section

namespace Pouw.PearlC.Mend

open Pouw.Fp8Atom

/-- The products of the groups `gs`, from the incoming value `v`, that the adder cuts: those whose aligned contribution
at their group's exponent is 0. -/
def cutFold (p : Pipeline) : List ℕ → Value → List (Code × Code) → List Bool
  | [], _, _ => []
  | size :: gs, v, ab =>
      let ps := (ab.take size).map fun x => productTotal x.1 x.2
      let E := maxExp p.zeroExp ((v :: ps).filterMap Value.finite?)
      (ab.take size).map (fun x => aligned p.width E (product x.1 x.2) == 0) ++
        cutFold p gs (groupStepTotal p.width p.zeroExp v ps) (ab.drop size)

/-- The products of one step of `p` from the accumulator word `acc` that the adder cuts, in `k` order. -/
def cuts (p : Pipeline) (acc : ℕ) (a b : List Code) : List Bool := cutFold p p.groups (fp32Value acc) (a.zip b)

/-- One atom of a word's chain: the atom's index, the word it writes, and the cut flags so far. -/
def chainStep (p : Pipeline) (G : ℕ) (st : ℕ × ℕ × List Bool) (x : AtomIn) : ℕ × ℕ × List Bool :=
  let w := if st.1 % G = 0 then 0 else st.2.1
  (st.1 + 1, p.step w (List.ofFn x.1) (List.ofFn x.2), st.2.2 ++ cuts p w (List.ofFn x.1) (List.ofFn x.2))

/-- **The cut flags of a word's chain** on `p`, promoted every `G` atoms: entry `32·t + r` is whether product `r` of atom
`t` is cut. -/
def chainCuts (p : Pipeline) (G : ℕ) (atoms : List AtomIn) : List Bool :=
  (atoms.foldl (chainStep p G) (0, 0, [])).2.2

/-- A row's codes at the support's plus end (`minus` clear) or minus end (`minus` set). -/
def endRow (p : Pipeline) (alpha beta : (k : ℕ) → (Fin k → ℚ) → ℚ) (k : ℕ) (x : Fin k → ℚ) (minus : Bool) :
    Fin k → Code :=
  fun l => (FormingP.atomMixP p).elem (alpha k x) (x l) (beta k x) (SaltDead.flatLine fun _ => false)
    (SaltDead.flatLineF fun _ => minus)

/-- Whether product `l` of the word of rows `x` and `w` is cut at one end of the support. -/
def cutAt (p : Pipeline) (G : ℕ) (alpha beta : (k : ℕ) → (Fin k → ℚ) → ℚ) (k : ℕ) (x w : Fin k → ℚ) (minus : Bool)
    (l : ℕ) : Bool :=
  (chainCuts p G (rowAtoms (endRow p alpha beta k x minus) (endRow p alpha beta k w minus))).getD l false

/-- **A forced zero**: product `l` of the word of rows `x` and `w` is cut at both ends of the support. -/
def ForcedZero (p : Pipeline) (G : ℕ) (alpha beta : (k : ℕ) → (Fin k → ℚ) → ℚ) (k : ℕ) (x w : Fin k → ℚ) (l : Fin k) :
    Prop :=
  cutAt p G alpha beta k x w false l = true ∧ cutAt p G alpha beta k x w true l = true

/-- A row's codes when element `l`'s noise line is the pair `σ l`. -/
def saltRow (p : Pipeline) (alpha beta : (k : ℕ) → (Fin k → ℚ) → ℚ) (k : ℕ) (x : Fin k → ℚ)
    (σ : Fin k → (Fin 32 → Code) × (Fin 32 → Code)) : Fin k → Code :=
  fun l => (FormingP.atomMixP p).elem (alpha k x) (x l) (beta k x) (σ l).1 (σ l).2

/-- Product `l` of the word of rows `x` and `w` is cut at every choice of support pairs for the two rows' elements. -/
def CutForEverySalt (p : Pipeline) (G : ℕ) (alpha beta : (k : ℕ) → (Fin k → ℚ) → ℚ) (k : ℕ) (x w : Fin k → ℚ)
    (l : Fin k) : Prop :=
  ∀ σ τ : Fin k → (Fin 32 → Code) × (Fin 32 → Code), (∀ t, σ t ∈ SaltDead.support) → (∀ t, τ t ∈ SaltDead.support) →
    (chainCuts p G (rowAtoms (saltRow p alpha beta k x σ) (saltRow p alpha beta k w τ))).getD l false = true

end Pouw.PearlC.Mend
