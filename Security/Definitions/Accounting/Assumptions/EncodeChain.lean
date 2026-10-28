import Definitions.Accounting.Gates.Codec

/-!
# The encode bound's former conjecture, falsified: a word's encoding costs budgeted units per squaring (trusted)

**Falsified.** `Accounting.SecurityProofs.Gates.EncodeChainRefuted` (`Security/Proofs/Accounting/Gates/Bitwise.lean`)
proves it false inside its floor at every `E ≥ 1` and every `B`, p1024 at `k = 3` among the fields: an AND of two bits
is a free s8-family step, so a circuit of bits computes any number of words for the key's gather units
(`benchmarks/property_redteam/bitwise_codec.py` builds it, its test runs it). It stays listed, marked `falsified` under
`shape.conjectures`, so that anything that takes it as a hypothesis is refused by the Lean check; no guarantee does. The
encode bound now rests on its two parts in the model's nonlinear units (`Gates.nonlinear`, a data product a unit):
`SquaringFloor` and `ChainSum` (`Definitions/Accounting/Assumptions/SquaringChain.lean`).

* `EncodeChainHardness F k B E ε`: in the circuit model, computing `q` distinct words of the codec's encoding (less
  parity) on more than `ε` of the draws costs `E` budgeted units a word, less `E/(n−1)` for each bit of run input, at
  `ε ≥ 2^−64`, `k ≥ 1` and a field with `p ≥ 2^66` and `p ≡ 3 mod 4`, and claims nothing outside that floor.
-/

namespace Accounting.Assumptions

open Accounting.Gates

/-- **`encode-chain-hardness`: computing a word of the codec's encoding costs `E` budgeted units (falsified:
`Accounting.SecurityProofs.Gates.EncodeChainRefuted`; superseded by `SquaringFloor` and `ChainSum`).** For every circuit
`C` of the model whose window is proven and whose budgeted gates each read at most `B` operand bits (`Restricted B C`),
whose registered inputs are fixed before the draw and whose coin inputs are bytes of the draw or fixed
(`CodecInputs`), and that computes the `q` distinct words `words` (each below `p`) of the masked `k`-round codec over
`F`, byte for byte less their parity, on more than `ε = εnum/εden` of the uniform draws, its inputs and budgeted gates
but gathers there in their declared intervals, so the whole circuit in its window (`ComputesWords`):

  `q · E · (n − 1) ≤ budgeted C · (n − 1) + runBits C · E`,

that is `q ≤ budgeted C / E + runBits C / (n − 1)`: each word costs `E` units (property/v1's, `units`), or `n − 1`
bits of run input (the word itself, sent after the draw).

**The floor.** It is claimed at `ε ≥ 2^−64` (`εden ≤ 2^64 · εnum`), `k ≥ 1`, `p ≥ 2^66` and `p ≡ 3 mod 4`, and says
nothing outside them; every reader carries them. Below each, a circuit with no budgeted gate computes a word, so the
bound fails at every `E ≥ 1` (known counterexamples; the examples after `EncodeCost` in `Proofs/Accounting/Gates.lean`
put them outside the floor):

* at `ε = 0`, the constant bytes of one draw's target are right on that draw;
* at `k = 0` the encoding is the payload word itself, registered before the draw;
* at a field narrower than a byte the target has no bytes, and below `p = 2^66` a constant target can be right on
  more than `2^−64` of the draws: at `⟨8, 5⟩`, `k = 3`, one draw is (there are about `2^40`), and at `p = 3` every
  target is `0`. At a prime `p ≡ 3 mod 4` the last round's constant is uniform and `ρ` takes each value from at most
  two inputs, so a constant target is right on at most `4/p` of the draws, `≤ 2^−64` from `p = 2^66`;
* at `p ≢ 3 mod 4` `ρ` is no square root (at `p = 2^n` it is `0` on every even input).

The floor doesn't say `p` is prime, since p1024's primality has no Lean proof here: at a composite `p ≡ 3 mod 4`
(`3^43`, where `ρ` is `0` on a third of the inputs) the bound is false, and no reader instantiates one.

**Why it should hold.** A word's encoding is `k` modular square roots, each `S = bitlen((p+1)/4) − 1` sequential
squarings mod `p` (1,021 at p1024). In the model products are free only as s8-family steps, whose operands must be
proven, by interval analysis, to be bytes; so each squaring's `L = n/8` input limbs are byte-format values, which a
free gate can't make from a wide one, and come from the previous squaring's budgeted gates. For one squaring the red
team's Lemma R (`Renormalization`) puts at least
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
96 operand bits; the falsification test's constructions (ripple carries, IMMA plus `I32Div`/`I32Rem`, and look-ahead
carries) are within `B = 128`, and spend at least `L − d(0)` units a squaring. Which `B` is safe is open: a test at toy
`n` with `B/n` fixed would settle the trend.

**Units, and why a gather over run-time values pays one.** A gate that reads only registered values and constants is
free (it computes one value on every draw), and a gather pays its fresh index bits, `⌈log₂ (V + 1)⌉` for a column
of `V` words and the `0` outside it, `n` for a table read of its index's low `n` bits (`IntGate.indexBits`), each kind
counted apart as property/v1 counts them: a one-word gather of a run-time index is the comparison
`[i ≤ 0]`, and were it free, thresholds and affine maps would carry a square's column sums back to bytes and the
chain would cost nothing. Without the extra unit for a
gather whose column holds a draw-dependent value, a tree of gathers indexed by the draw's bytes, over constant leaves,
would compute any function of the draw (`(k + 2)·n` bits) for about that many units, a word for about 5,120 at p1024.

**The target is parity-free.** The dropped bit is the Legendre symbol of the last round's input, and
`ρ(p − x) = p − ρ(x)` (`Guarantees.Gates.RhoPairs`): the ideal model of a round is a random function on the pairs
`{x, p − x}` with that sign rule, and what the conjecture charges is the pair's root, not the sign. -/
def EncodeChainHardness (F : Field) (k B E εnum εden : ℕ) : Prop :=
  εden ≤ 2 ^ 64 * εnum → 0 < k → 2 ^ 66 ≤ F.p → F.p % 4 = 3 →
  ∀ {ι : Type} (C : IntDag ι) (m : ℕ → ℕ) (x : Draw k → ι → ℤ) (words : List ℕ) (out : ℕ → ℕ → ℕ),
    Restricted B C → CodecInputs F C (draws F k) m x → words.Nodup → (∀ i ∈ words, i < F.p) →
    ComputesWords F k C x m words out εnum εden →
    words.length * E * (F.n - 1) ≤ budgeted C * (F.n - 1) + runBits C * E

end Accounting.Assumptions
