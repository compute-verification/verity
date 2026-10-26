import Definitions.Accounting.Gates.Codec
import Definitions.Accounting.Gates.Cut
import Models.Accounting.Assumptions.EncodeChain

/-!
# The min-cut bound: the words a circuit stacks are priced on any cut that holds its writes (trusted)

The encode bound (`EncodeChainHardness`, and `Guarantees.Gates.EncodeBudget` over a declared circuit) prices the
words a circuit computes by all of its budgeted units and run-input bits. A circuit unrolled over a carrier's lifetime
also does the served work, whose units would then pay for stacked words. The min-cut bound prices them by one side of
a cut instead:

* `MinCut`: for a circuit `C` of the restricted class with the codec's inputs that computes `q` distinct words of the
  codec's encoding on more than `ε` of the draws, and every set `D` of its gates that holds the gates it writes those
  words to (`out i l ∈ D`), under `EncodeChainHardness F k B E ε` inside its floor,

    `q · E · (n − 1) ≤ budgeted (cut C D) · (n − 1) + (boundaryBits C D + runBitsIn C D) · E`,

  that is `q · (n − 1) ≤ b(∂D) + r(D) + (n − 1) · W(D) / E`: the bits that cross into `D` from dynamic gates outside it,
  the run-input bits inside it, and `D`'s units over `E` words' worth. Static gates outside `D` (the weights, the
  payload and whatever is computed from them alone) cross free. It holds at every `D`, so at the least of them, and
  at a coarser cut than the least one it is a valid, looser bound. At `D` every gate it is `EncodeChainHardness`
  itself (`Whole`).
* `Whole`: the trivial cut, every gate in `D`, has no boundary, `C`'s units and `C`'s run-input bits.
* `Units`: the cut circuit's units are at most `D`'s units in `C` and, for each gather in `D`, its index bits again:
  a gather in `D` whose index an earlier gather outside `D` already paid for pays it again.

Every claim resting on `MinCut` is conditional on the named open conjecture `EncodeChainHardness`.
-/

namespace Accounting.Guarantees.Cut

open Accounting.Gates Accounting.Assumptions

/-- **The min-cut bound.** For a restricted circuit `C` (window proven, at most `B` operand bits a budgeted gate) whose
inputs are the codec's and that computes the `q` distinct words `words` (each below `p`) of the masked `k`-round
codec over `F`, byte for byte less their parity, on more than `ε = εnum/εden` of the draws, and every set `D` of its
gates that holds every gate it writes those bytes to, under `EncodeChainHardness F k B E ε` inside its floor
(`ε ≥ 2^−64`, `k ≥ 1`, `p ≥ 2^66`, `p ≡ 3 mod 4`):

  `q · E · (n − 1) ≤ budgeted (cut C D) · (n − 1) + (boundaryBits C D + runBitsIn C D) · E`. -/
def MinCut : Prop :=
  ∀ (F : Field) (k B E εnum εden : ℕ) {ι : Type} (C : Dag ι) (D : Finset ℕ) (m : ℕ → ℕ) (x : Draw k → ι → ℤ)
    (words : List ℕ) (out : ℕ → ℕ → ℕ),
    EncodeChainHardness F k B E εnum εden → εden ≤ 2 ^ 64 * εnum → 0 < k → 2 ^ 66 ≤ F.p → F.p % 4 = 3 →
    Restricted B C → CodecInputs F C (draws F k) m x → words.Nodup → (∀ i ∈ words, i < F.p) →
    ComputesWords F k C x m words out εnum εden → (∀ i ∈ words, ∀ l < F.bytes, out i l ∈ D) →
    words.length * E * (F.n - 1) ≤
      budgeted (cut C D) * (F.n - 1) + (boundaryBits C D + runBitsIn C D) * E

/-- **The trivial cut.** With every gate in `D`, nothing crosses, and the cut circuit has `C`'s units and `C`'s
run-input bits: `MinCut` there is `EncodeChainHardness`'s bound. -/
def Whole : Prop :=
  ∀ {ι : Type} (C : Dag ι),
    boundaryBits C (Finset.range C.length) = 0 ∧ budgeted (cut C (Finset.range C.length)) = budgeted C ∧
      runBitsIn C (Finset.range C.length) = runBits C

/-- **The cut circuit's units.** At most `D`'s units in `C`, and each gather's index bits in `D` again. -/
def Units : Prop :=
  ∀ {ι : Type} (C : Dag ι) (D : Finset ℕ),
    budgeted (cut C D) ≤ ∑ i ∈ (Finset.range C.length).filter (· ∈ D), (units C i + (C[i]?.map Gate.gatherBits).getD 0)

end Accounting.Guarantees.Cut
