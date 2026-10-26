import Mathlib.Data.Int.ModEq
import Definitions.Accounting.Gates.Codec

/-!
# The encode bound's conjectures in the repriced model: a floor per squaring, and a chain costs its sum (trusted)

What the encode bound (`Definitions/Accounting/Guarantees/Gates.lean`, `EncodeBudget`) assumes in place of the
falsified `EncodeChainHardness`: two named `Prop`s, each taken as a hypothesis, never an `axiom`. Both are genuine math,
conjectures (Daniel, 8 Oct): each sits under `shape.conjectures` with a falsification test that runs, and every claim
resting on them is conditional. Both are priced in the repriced model's capped units (`Gates.capped`: budgeted units,
and a compare-select unit for each thin lane, an s8-family lane whose two multiplicands are dynamic and one of them
two-valued), the units the stacking property caps (`Gates.Property`).

* `SquaringFloor F e ε`: one squaring mod `p`, from a word's canonical bytes to `L = n/8` byte-format limbs congruent
  to its square, right on more than `ε` of the words, costs at least `e` capped units, at `ε ≥ 2^−64` and `p ≥ 2^66`.
  The red team's Lemma R (`Guarantees.Gates.Renormalization`) at `c = 0`, in capped units.
* `ChainSum F k B e ε`: if one squaring costs `e`, the codec's word costs `k · S · e`, `S = Field.squarings F` (1,021 at
  p1024): its `k` roots are `k · S` sequential squarings, which cost their sum, and its words cost theirs
  (`Gates.EncodeBound F k B (k · S · e) ε`), at `ε ≥ 2^−64`, `k ≥ 1`, `p ≥ 2^66` and `p ≡ 3 mod 4`. Sloth's
  sequentiality (Lenstra–Wesolowski), stated in the model.

**Why two.** `EncodeChainHardness` was the two at once, priced in budgeted units, and the model refuted it
(`Accounting.SecurityProofs.Gates.EncodeChainRefuted`): an AND of two bits is a free MAC, so a bitwise codec computes
any number of words for the key's gather units. The repricing charges that AND a compare-select unit, and the claim
splits where its evidence does: a squaring's floor is a counting question about one squaring, which the selection-only
bound (≥ 159 units a squaring, note:20261026T0110Z-report-s5-lowerbounds, sketched) and the known constructions bear
on; the chain's additivity is a sequentiality question about representations between squarings, which they don't.

**Price.** At p1024 (`L = 128`), `e = L − d(ε)` with `d(ε) = d₀ + ⌈log₂(1/ε)/8⌉` and `d₀ = 4`: `e = 119` at the
headline `ε = 2^−40` and `121` at the property's carrier threshold `2^−20`, so at `k = 3`
`E = 3 · 1021 · 119 = 364,497` capped units a word (`11,663,904` a 4 KiB block of 32 words), and `370,623` at `2^−20`:
the numbers `EncodeChainHardness` priced, now in capped units.
-/

namespace Accounting.Assumptions

open Accounting.Gates

/-- **`squaring-floor`: a squaring mod `p` costs `e` capped units.** Take a circuit `C` of the model whose window is
proven (`Proven C`), with no run input, whose registered inputs are one value on every word and whose coin inputs are
bytes of the word `a` or one value on every word (so units count what reads `a`), that on more than `ε = εnum/εden` of
the words `a ∈ [0, p)` holds its declared intervals (so is in its window) and outputs `L = n/8` values (`out l`), each
within a byte format (`fmt l`), whose base-256 sum is `a²` mod `p`. Then `e ≤ capped C`: its budgeted units and thin
lanes are at least `e`.

**The floor.** It is claimed at `ε ≥ 2^−64` (`εden ≤ 2^64 · εnum`) and `p ≥ 2^66`, and says nothing outside them. Below
the floor, circuits with no budgeted gate are exact on the `a` whose squares don't carry (the column sums of `a²` as
s8-family steps, folded mod `p`), which refute every fixed `d₀` (`Renormalization`'s docstring: `d₀ ≥ 12.73` at p1024 at
`ε ≈ 2^−922`); none reaches `2^−64`, since a non-carrying `a` has digits below 16, `ε ≤ 16^L/p ≈ 2^−512`.

**Why it should hold.** The limbs out are byte-format values, which interval analysis proves a free gate makes only
from narrow values; a square's column sums are wide, so its carries pass through gates the caps count. The red team's
Lemma R puts `L − d(ε)` budgeted units there (note:20261017T1859Z-report-property-redteam R3.3, R4.1), and the
selection-only bound (note:20261026T0110Z-report-s5-lowerbounds, sketched: a circuit whose budgeted gates are
selections pays at least 159 a squaring at p1024) is above `e`. Every known construction pays more: ripple and
look-ahead carries at least `L − d(0) = 124`, a bitwise squaring about `n²/2 ≈ 5·10^5` thin lanes.

**What would falsify it.** A squaring circuit with fewer than `e` capped units on more than `ε ≥ 2^−64` of the words:
the falsification test (`test_squaring_floor_known_constructions`) builds each known construction and checks its capped
units against `e`. -/
def SquaringFloor (F : Field) (e εnum εden : ℕ) : Prop :=
  εden ≤ 2 ^ 64 * εnum → 2 ^ 66 ≤ F.p →
  ∀ {ι : Type} (C : Dag ι) (x : ℕ → ι → ℤ) (out : ℕ → ℕ) (fmt : ℕ → Int8Fmt) (good : Finset ℕ),
    Proven C → runBits C = 0 → good ⊆ Finset.range F.p → MoreThan (Finset.range F.p) good εnum εden →
    (∀ i, (∃ g ∈ C, g.op = .input .registered i) → ∀ a ∈ good, ∀ a' ∈ good, x a i = x a' i) →
    (∀ i, (∃ g ∈ C, g.op = .input .coin i) →
      (∃ l, ∀ a ∈ good, x a i = byte a l) ∨ ∀ a ∈ good, ∀ a' ∈ good, x a i = x a' i) →
    (∀ a ∈ good, DeclaredHold C (x a) ∧
      (∀ l < F.bytes, (fmt l).lo ≤ val C (x a) (out l) ∧ val C (x a) (out l) ≤ (fmt l).hi) ∧
      ∑ l ∈ Finset.range F.bytes, 256 ^ l * val C (x a) (out l) ≡ (a : ℤ) ^ 2 [ZMOD F.p]) →
    e ≤ capped C

/-- **`chain-sum`: chained squarings cost their sum.** If one squaring mod `p` costs `e` capped units
(`SquaringFloor F e ε`), then computing words of the masked `k`-round codec over `F` costs `k · S · e` capped units a
word, `S = Field.squarings F`: every restricted circuit (`Restricted B`) with the codec's inputs that computes `q`
distinct words (each below `p`), byte for byte less their parity, on more than `ε` of the draws has
`q · k · S · e · (n − 1) ≤ capped C · (n − 1) + runBits C · k · S · e` (`Gates.EncodeBound`). Each word costs its
`k · S` squarings, or its `n − 1` bits as run input.

**The floor.** It is claimed at `ε ≥ 2^−64`, `k ≥ 1`, `p ≥ 2^66` and `p ≡ 3 mod 4`, `EncodeChainHardness`'s floor, for
the reasons its docstring gives: below each a circuit with no unit computes a word. It doesn't say `p` is prime.

**Why it should hold.** A root is `S` squarings in sequence, each reading the last one's output, and the word's `k`
roots are in sequence too (`ρ(σ(x) + c_r + t)`), so no squaring starts before the one before it ends: Sloth's
sequentiality (Lenstra–Wesolowski, the slow-timed hash), whose best known shortcut for a square root mod a prime
`p ≡ 3 mod 4` is the exponentiation itself. Across words, word `i`'s tweak `τ·i mod p` makes the chains' inputs
pairwise unrelated, so a word shares no squaring with another. The conjecture is that the model's capped units, like
time, add along that chain: no representation between squarings (signed or redundant digits, bits, unreduced limbs)
makes a squaring cheaper than `e` on average, and none is shared across words.

**What would falsify it.** A codec circuit whose capped units a word fall below `k · S · e`: a chain amortizing its
renormalizations across squarings, or a block sharing work across words. The falsification test
(`test_chain_sum_known_chains`) prices the known chains: the byte-limb chain (each squaring a known construction,
measured over chained squarings at p1024) and the bitwise codec, whose thin lanes the red team's port measures at toy
fields (`benchmarks/property_redteam/tests/test_bitwise_codec.py`). -/
def ChainSum (F : Field) (k B e εnum εden : ℕ) : Prop :=
  εden ≤ 2 ^ 64 * εnum → 0 < k → 2 ^ 66 ≤ F.p → F.p % 4 = 3 →
    SquaringFloor F e εnum εden → EncodeBound F k B (k * F.squarings * e) εnum εden

end Accounting.Assumptions
