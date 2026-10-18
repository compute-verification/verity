import Definitions.Accounting.Gates.Codec

/-!
# The encode bound's named conjecture: a word's encoding costs budgeted units per squaring (trusted)

What the encode bound (`Models/Accounting/Guarantees/Gates.lean`, `EncodeBudget`) assumes, a named `Prop` taken as a
hypothesis, never an `axiom`. It is genuine math, a conjecture (Daniel, 8 Oct): it sits under `shape.conjectures` with
a falsification test that runs, and every claim resting on it is conditional.

* `EncodeChainHardness F k B E ε`: in the circuit model, computing `q` distinct words of the codec's encoding (less
  parity) on more than `ε` of the draws costs `E` budgeted units a word, less `E/(n−1)` for each bit of run input.
-/

namespace Accounting.Assumptions

open Accounting.Gates

/-- **`encode-chain-hardness`: computing a word of the codec's encoding costs `E` budgeted units.** For every circuit
`C` of the model whose window is proven and whose budgeted gates each read at most `B` operand bits (`Restricted B C`),
whose registered inputs are fixed before the draw and whose coin inputs are bytes of the draw or fixed
(`CodecInputs`), and that computes the `q` distinct words `words` (each below `p`) of the masked `k`-round codec over
`F`, byte for byte less their parity, on more than `ε = εnum/εden` of the uniform draws, its inputs and budgeted gates
there in their declared intervals, so the whole circuit in its window (`ComputesWords`; claimed for `ε ≥ 2^−64`):

  `q · E · (n − 1) ≤ budgeted C · (n − 1) + runBits C · E`,

that is `q ≤ budgeted C / E + runBits C / (n − 1)`: each word costs `E` units (property/v1's, `units`), or `n − 1`
bits of run input (the word itself, sent after the draw).

**Why it should hold.** A word's encoding is `k` modular square roots, each `S = bitlen((p+1)/4) − 1` sequential
squarings mod `p` (1,021 at p1024). In the model products are free only as s8-family steps, whose operands must be
proven, by interval analysis, to be bytes; so each squaring's `L = n/8` input limbs are byte-format values, which a
free gate can't make from a wide one, and come from the previous squaring's budgeted gates. For one squaring the red
team's Lemma R (`Renormalization`, note:20261017T1859Z-report-property-redteam R3.3 and R4.1) puts at least
`L − d(ε)` units there, `d(ε) = d₀ + log₂(1/ε)/8`, at every `ε ≥ 2^−64`, whatever byte-format representation the
squaring outputs; the conjecture is that it composes along the chain and over words, with no shortcut for the
exponentiation (as Sloth's sequentiality) and no amortization across words. At `E = k · S · (L − d(ε))`:

* `d₀ = 4` is the value taken at p1024, and `ε`'s share is `log₂(1/ε)/8` limbs, rounded up. `d₀` is not absolute:
  circuits with no budgeted gate are exact on the `a` whose squares don't carry, which refute it below `ε = 2^−182`
  (`d₀ ≥ 4.16` at `p = 256^27 − 377`, `≥ 12.73` at p1024), so the claim is for `ε ≥ 2^−64` only, `Renormalization`'s
  floor;
* at p1024, `k = 3`, `ε = 2^−40`: `e = 128 − 4 − 5 = 119`, so `E = 364,497` a word and `11,663,904` a 4 KiB block
  (32 words). At `ε = 2^−20` (the property's carrier threshold) `e = 121` and `E = 370,623`.

**The restriction is `B`.** It is false without one: a budgeted gate that reads a word's bits computes a byte of its
root, so a word costs about `k·n/32 + n/8` gates (224 at p1024, `k = 3`). property/v1's listed primitives read at most
96 operand bits; the falsification test's constructions (ripple carries, the red team's IMMA plus `I32Div`/`I32Rem`,
and look-ahead carries) are within `B = 128`, and spend at least `L − d(0)` units a squaring. Which `B` is safe is
open: a test at toy `n` with `B/n` fixed would settle the trend.

**Units, and why a gather over run-time values pays one.** A gate that reads only registered values and constants is
free (it computes one value on every draw), and a gather pays its fresh index bits. Without the extra unit for a
gather whose column holds a draw-dependent value, a tree of gathers indexed by the draw's bytes, over constant leaves,
would compute any function of the draw (`(k + 2)·n` bits) for about that many units, a word for about 5,120 at p1024.

**The target is parity-free.** The dropped bit is the Legendre symbol of the last round's input, and
`ρ(p − x) = p − ρ(x)` (`Guarantees.Gates.RhoPairs`): the ideal model of a round is a random function on the pairs
`{x, p − x}` with that sign rule, and what the conjecture charges is the pair's root, not the sign. -/
def EncodeChainHardness (F : Field) (k B E εnum εden : ℕ) : Prop :=
  ∀ {ι : Type} (C : Dag ι) (m : ℕ → ℕ) (x : Draw k → ι → ℤ) (words : List ℕ) (out : ℕ → ℕ → ℕ),
    Restricted B C → CodecInputs F C (draws F k) m x → words.Nodup → (∀ i ∈ words, i < F.p) →
    ComputesWords F k C x m words out εnum εden →
    words.length * E * (F.n - 1) ≤ budgeted C * (F.n - 1) + runBits C * E

end Accounting.Assumptions
