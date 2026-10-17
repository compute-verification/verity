import Definitions.Accounting.Gates.Codec

/-!
# The encode bound's named conjecture: a word's encoding costs budgeted gates per squaring (trusted)

What the encode bound (`Models/Accounting/Guarantees/Gates.lean`, `EncodeBudget`) assumes, a named `Prop` taken as a
hypothesis, never an `axiom`. It is genuine math, a conjecture (Daniel, 8 Oct): it sits under `shape.conjectures` with
a falsification test that runs, and every claim resting on it is conditional.

* `EncodeChainHardness F k f w E ε`: in the circuit model, computing `q` distinct words of the codec's encoding (less
  parity) on more than `ε` of the draws costs `E` budgeted gates a word, less `E/(n−1)` for each bit of run input.
-/

namespace Accounting.Assumptions

open Accounting.Gates

/-- **`encode-chain-hardness`: computing a word of the codec's encoding costs `E` budgeted gates.** For every circuit
`C` of the model whose window is proven, whose budgeted gates but gathers read at most `f` operands and whose values lie
within `w` bits (`Restricted f w C`), whose registered inputs are bytes of the payload `m`, of the draw, or fixed before
the draw (`CodecInputs`), and that computes the `q` distinct words `words` (each below `p`) of the masked `k`-round
codec over `F`, byte for byte less their parity, on more than `ε = εnum/εden` of the uniform draws (`ComputesWords`):

  `q · E · (n − 1) ≤ budgeted C · (n − 1) + runBits C · E`,

that is `q ≤ budgeted C / E + runBits C / (n − 1)`: each word costs `E` budgeted gates, or `n − 1` bits of run input
(the word itself, sent after the draw).

**Why it should hold.** A word's encoding is `k` modular square roots, each `S = bitlen((p+1)/4) − 1` sequential
squarings mod `p` (1,021 at p1024). In the model products are free only as s8-family steps, whose operands must be
proven, by interval analysis, to be bytes; so after every product the column sums (about 2^21 each) must be returned
to `n` byte limbs, and a budgeted gate's value is one limb's worth. The carry bound proves, for one squaring in
isolation, that its canonical bytes need `2^s` joint values of the charged gates (`SquaringCarry`, `s = 63` at p1024:
63 bits, two 32-bit gates); the conjecture is that the bound composes along the chain and over words, with no shortcut
for the exponentiation (as Sloth's sequentiality) and no amortization across words. At `E = k · S · e`:

* `e = 2`, the proved per-squaring bound composed, is the weakest form that uses the carry bound:
  `E = 3 · 1021 · 2 = 6,126` a word at p1024/k3, `196,032` a 4 KiB block (32 words);
* `e = n − 2 = 126`, one carry a limb (the re-normalization step: returning the `n` limbs to the window after a
  product takes `n − O(1)` budgeted gates), is the headline: `E = 385,938` a word, `12,350,016` a block.

The honest circuit, and every construction the falsification test knows (f-ary carry look-ahead, the red team's IMMA
plus `I32Div`/`I32Rem` carries), spends at least `n` budgeted gates a squaring. It is false without `Restricted`: one
budgeted gate of unbounded fan-in computes an output byte from all the inputs, so `n` gates compute a word; and with
values of 576 bits (`PouwSqAcc_v1`), about ten gates do. -/
def EncodeChainHardness (F : Field) (k f w E εnum εden : ℕ) : Prop :=
  ∀ {ι : Type} (C : Circuit ι) (m : ℕ → ℕ) (x : Draw k → ι → ℤ) (words : List ℕ) (out : ℕ → ℕ → ℕ),
    Restricted f w C → CodecInputs F C (draws F k) m x → words.Nodup → (∀ i ∈ words, i < F.p) →
    ComputesWords F k C x m words out εnum εden →
    words.length * E * (F.n - 1) ≤ budgeted C * (F.n - 1) + runBits C * E

end Accounting.Assumptions
