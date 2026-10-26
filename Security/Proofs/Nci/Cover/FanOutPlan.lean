import Definitions.Nci.Guarantees.CoverFanOut

/-!
# The fan-out proof's plan: the statements of its pieces

`FanOutIntensity` and `FanOutReadIntensity` follow from these five, the count of products finished but not whole that
`intensity`'s proof makes (`finished_sdiff_whole_card_le`), and arithmetic: a sum over tokens, and AM–GM for the `f`
variant. The cover bounds follow from the intensities by summing over a cover (`cover_sum`). They are
lemmas, not guarantees, stated here so that each is proved against a fixed statement: `walk`, `readSet_card_le`,
`lookupCost` and `charge` in `FanOutWalk`, `rootCost` in `FanOutRootCost`.
-/

namespace Nci.Cover.FanOutPlan

/-- **Walk** (A1's Lemma 1): if a set `I` of gates holds an exit `h` of a value `v`, following keys from `v` reaches,
before any other value `I` reads, a value `I` reads, or a stop `I` holds at a computed exit no later than `h`.
Induction on `h`, with `entry` on each link. -/
def Walk : Prop :=
  ∀ (C : Circuit) {J L Tok Val : Type} [DecidableEq Val] (M : MulBlocks C J L) (d aOut c V f : ℕ)
    (K : Keyed M Tok Val d aOut c V f) (I : Finset (Fin C.N)),
    ∀ v ∈ K.vals, ∀ h ∈ C.exits (K.gates v), h ∈ I →
      ∃ u ∈ K.vals, K.tok u = K.tok v ∧ Follows K.key (K.stops ∪ K.readSet I) v u ∧
        (u ∈ K.readSet I ∨ (u ∈ K.stops ∩ K.heldSet I ∧ ∃ g ∈ C.exits (K.gates u) ∩ C.computed I, g ≤ h))

/-- **A root costs `d`** (A1's Lemma 2): a set of gates that holds a root spends values of its token, and holds lookups
of it, `d` in all. Strong induction on a bound `n` for a computed exit `g < n` of the held root: each leg's link puts the
leg's exit `h < g` in `I` (`entry` on the link's path inside the root), `Walk` takes it to a spent value, a held lookup
or a held root with a computed exit at most `h` (then the induction), and the legs' disjoint key paths make the ends
distinct (`Follows` with the larger stop set `stops ∪ readSet I` is `Follows` with `roots ∪ lookups`). -/
def RootCost : Prop :=
  ∀ (C : Circuit) {J L Tok Val : Type} [DecidableEq J] [DecidableEq Val] (M : MulBlocks C J L) (d aOut c V f : ℕ)
    (K : Keyed M Tok Val d aOut c V f) (I : Finset (Fin C.N)),
    ∀ r ∈ K.roots ∩ K.heldSet I, ∃ X Y : Finset Val,
      (∀ u ∈ X, K.Spent I u ∧ K.tok u = K.tok r) ∧
      (∀ u ∈ Y, u ∈ K.lookups ∩ K.heldSet I ∧ K.tok u = K.tok r) ∧
      d ≤ X.card + Y.card

/-- **Lookups** (A1's Lemma 3): under `ColumnsIncompressible` with `k < bcol · V · (E + 1)`, an inference block holds
at most `E` lookups of one token, since their columns are disjoint and `V` wide. -/
def LookupCost : Prop :=
  ∀ (C : Circuit) {J L Tok Val : Type} [DecidableEq L] [DecidableEq Val] (M : MulBlocks C J L) (d aOut c V f : ℕ)
    (K : Keyed M Tok Val d aOut c V f) (s k E : ℕ) (bcol : ℝ),
    Assumptions.ColumnsIncompressible K s bcol → (k : ℝ) < bcol * V * (E + 1) →
    ∀ I, C.InferenceBlock s k I → ∀ Y ⊆ K.lookups ∩ K.heldSet I, (∀ v ∈ Y, ∀ v' ∈ Y, K.tok v = K.tok v') →
      Y.card ≤ E

/-- **Charge** (A1's low-token case): if a set of gates holds no root of token `i`, the counted products of token `i` it
computes whole number at most `aOut` per value of token `i` it spends. Each is charged to the value `Walk` reaches from
its activation's exit; a held stop there would be a lookup (excluded by `reach_lookups`) or a root (none held). -/
def Charge : Prop :=
  ∀ (C : Circuit) {J L Tok Val : Type} [DecidableEq J] [DecidableEq Val] (M : MulBlocks C J L) (d aOut c V f : ℕ)
    (K : Keyed M Tok Val d aOut c V f) (I : Finset (Fin C.N)) (i : Tok),
    (∀ r ∈ K.roots ∩ K.heldSet I, K.tok r ≠ i) →
    ∀ B ⊆ M.whole I, (∀ p ∈ B, K.tok (K.act p) = i) →
      ∃ X : Finset Val, (∀ u ∈ X, K.Spent I u ∧ K.tok u = i) ∧ B.card ≤ aOut * X.card

/-- **A block reads at most its inputs' worth of values**: the values are disjoint and hold no registered parameter, and
each value a block reads has a gate among its inputs, so `#(readSet I) ≤ #(inputs I \ θ)`. `FanOutIntensity` needs it,
since without `ValueReadHardness` nothing else bounds the values a block reads. -/
def ReadSetCardLe : Prop :=
  ∀ (C : Circuit) {J L Tok Val : Type} [DecidableEq Val] (M : MulBlocks C J L) (d aOut c V f : ℕ)
    (K : Keyed M Tok Val d aOut c V f) (I : Finset (Fin C.N)),
    (K.readSet I).card ≤ (C.inputs I \ C.θ).card

end Nci.Cover.FanOutPlan
