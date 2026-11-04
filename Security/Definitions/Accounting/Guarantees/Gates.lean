import Mathlib.Data.Int.ModEq
import Mathlib.Data.Nat.Prime.Defs
import Definitions.Accounting.Gates
import Definitions.Core.Prims
import Definitions.Accounting.Assumptions.SquaringChain

/-!
# The circuit model's window, and its lower bounds: the fooling bound, the carry bound and the encode budget (trusted)

The statements over the circuit model (`Definitions/Accounting/Gates`) that the stacking property's encode bound rests
on, from the most general to the one 6a's carrier lemma takes as its `q`:

* `ProvenWindow`: the property's interval analysis is sound. A circuit whose window is proven (`Proven`) is in its
  window on every input on which its inputs and budgeted gates but gathers hold their declared intervals
  (`DeclaredHold`; a gather's interval is derived from its column's): the window is a fact of the operand bounds, not
  a run-time hypothesis.
* `ClampOrWrap`, its corollary: in the window, an s8-family step's clamp and wrap give the same values, so the census
  keys the s8·s8 (`.satfinite`) and u8·s8 (wrapped) steps to one class.
* `Fooling`: a 2-adic fooling bound. Free gates (affine maps and s8-family steps in their window) see only the low `j`
  bits of their operands when their outputs' low `j` bits are asked for. So if two inputs agree mod `2^j` in every
  registered and coin input and every charged gate takes one value on both, every gate does mod `2^j`, and a set of
  inputs that pairwise differ mod `2^j` in some output needs that many joint values of the charged gates:
  `|F| ≤ volume C`.
* `SquaringCarry`: the carry bound (workstream G's §(7), made precise). A circuit that computes the `n` canonical bytes
  of `a² mod p`, in its window, on the members `G` of the sparse fooling family (`sparse`), has `|G| ≤ volume C`. On
  the whole family (`2^s` members) its charged gates have at least `s` bits: `s = 63` at p1024 (`256^127 ≤ p`). It is a
  bound in bits, not in gates: a budgeted gate of `b` declared bits carries `b` of them. "Computes" is "on every member
  of `G`": on the fraction `ε` of the family, `G` has `ε · 2^s` members.
* `SquaringCarryNibbles`: the same at `j = 4` on the nibble family (`nibbles`), whose bytes are all `8 mod 16` and
  whose squares first differ mod 16 two bytes above their first differing nibble: `4s` bits, 252 at p1024 (`s = 63`).
  Four free bits a byte is the most a fooling family of this kind gets (the squares' first difference must land in
  the low `j` bits of a byte, with `j` bits of each input byte fixed), and with no reduction mod `p` the input has at
  most `n/16` bytes.
* `UnitsFooling`: `Fooling` in units. With registered inputs fixed and each budgeted gate but a gather declaring at
  most `2^w` values, `|F| ≤ 2^(runBits C + w · budgeted C)`: a gate that reads only registered values is one value on
  all of `F`, a budgeted one that reads a run or coin value is a unit of `w` bits, and a gather's value is fixed by its
  index's class among its column's words and its fill outside it (a table read's, by its index's low bits), which
  its index's fresh bits of that kind count.
* `SquaringUnits`: the two together. A circuit that squares the nibble family, its registered inputs fixed and its
  coin inputs `a`'s bytes, has `4s ≤ runBits C + w · budgeted C`: at p1024 with no run input, 8 units at `w = 33` (or
  `w = 32`, s32's window), 32 were each budgeted gate a byte (`w = 8`).
* `FoolingCap`: why counting stops there. A set of inputs that pairwise differ mod `2^j` in some output of `out` has at
  most `2^(j·|out|)` members, so a fooling bound in units is at most `j·|out|/w`: one squaring's outputs cap what it
  can charge (252 nibble bits, about 8 units at `w = 33`), and a chain's intermediate squarings, which are not
  outputs, add nothing. A floor per squaring and a chain's sum are premises (`SquaringFloor`, `ChainSum`), not counts.
* `Renormalization`: the step the encode bound needs and no counting argument gives, stated and unproved (the red
  team's Lemma R): returning a square's limbs to byte format takes `L − c − d(ε)` budgeted units, `d(ε) =
  d₀ + log₂(1/ε)/8`, at every `ε` above a floor `2^−t` (claimed at p1024 with `d₀ = 4`, `t = 64`; false with no
  floor). `SquaringFloor` adds to it at `c = 0` the square's data products, and `ChainSum` composes it over the chain.
* `RhoPairs`: the parity-free target's symmetry. `ρ(p − x) = p − ρ(x)`, and `ρ(x)`'s parity, the bit the target
  drops, is the Legendre symbol of `x`.
* `EncodeBudget`: the connection. For a declared circuit `W` that satisfies the stacking property `P` and is in the
  restricted class, the words of fresh encodings `W`'s values hold on more than `ε` of the draws number at most
  `(budget)/E + (run-input bits)/(n − 1)`, under `SquaringFloor` at `e` and `ChainSum` with `E ≤ k · S · e`: the
  budget is the caps (rational nonlinear units per certified byte, data products among them) times the
  audit's certified reads, summed over the windows, and the run inputs are each window's `R_in` bound, summed over the
  windows.
* `EncodeCost`: under the same premises, the encode cost (`encodeCost`, in nonlinear units) of `q` words is at least
  `q · E` (`⊤` when no circuit computes them): the cost 6a's `encodeBitsCost` reads.
* `FmaxfRange`, `FminfRange`, `RzByteRange`: property/v1's interval rules for P's producers hold their evaluators'
  values (`Definitions/Accounting/Gates/Ranges`), the intervals `DeclaredHold` asks of those budgeted gates.
* `QuantRange`, `DitherRange`, `LutSoftmaxRange`: the quantizers', the dither's and the LUT softmax's values lie in
  their own intervals: `[lo, lo + w]` for a clamp, `[lo − m, lo + w + m + 1]` for a dither, and the hull of the table
  and 0 for the LUT.
* `Bf16ToF32Affine`: the registry's `Bf16ToF32_v1` is free-affine, and on every bf16 word its evaluator is that map,
  `×2^16` from the s16 reading to the s32 reading, which never wraps.
* `F32ToS8RziRange`, `Int8ClampRange`, `LutIndexRange`, `Exp2Lut127Range`, `S32LowByteRange`: the exact-int8
  profile's primitives' intervals: the cast's `[−128, 127]`, the quantizer's clamp chain's `[−127, 127]`, the LUT
  index's `[0, 255]`, the table's 256 values in `[0, 127]`, and a low byte's `[−128, 127]`, the identity on `[0, 127]`.
* `DeclaredWindow`: `ProvenWindow` with no hypothesis on a run input's value but its type (`Typed`). A circuit whose
  declarations are earned (`Declared`: an input declared at its type, a budgeted gate at an interval its evaluator
  provably keeps) holds them (`DeclaredHold`), and so is in its window.
* `Int8LinearWindow`, `LutPvWindow`: vLLM's served int8 linear (row quantizer into a satfinite s8·s8 IMMA chain, `K ≤
  131,040`) and P·V (LUT index, table gather, low byte into the same chain, `32·S ≤ 132,096`) are proven and declared,
  so they're in their window on every typed input.
-/

namespace Accounting.Guarantees.Gates

open Accounting.Gates Accounting.Assumptions

/-- **The window, proven, holds.** A circuit whose window is proven (`Proven C`: property/v1's interval analysis) is in
its window on every input `x` on which its inputs and budgeted gates but gathers hold their declared intervals. -/
def ProvenWindow : Prop :=
  ∀ {ι : Type} (C : IntDag ι) (x : ι → ℤ), Proven C → DeclaredHold C x → InWindow C x

/-- **Clamp or wrap, in the window.** For a circuit `C` whose window is proven, on an input on which its inputs and
budgeted gates but gathers hold their declared intervals: a circuit `C'` gate for gate like `C` but for its s8-family
steps' clamp or wrap (`IntGate.Reclamped`) has `C`'s values. -/
def ClampOrWrap : Prop :=
  ∀ {ι : Type} (C C' : IntDag ι) (x : ι → ℤ), Proven C → DeclaredHold C x → C'.length = C.length →
    (∀ i (h : i < C.length) (h' : i < C'.length), C[i].Reclamped C'[i]) → values C' x = values C x

/-- **The 2-adic fooling bound.** For a circuit `C`, a set `F` of inputs on each of which `C` is in its window, that
agree pairwise mod `2^j` in every registered and coin input, and that pairwise differ mod `2^j` in some output gate of
`out`: `|F| ≤ volume C`, the joint values the run inputs' and budgeted gates' declared intervals allow. -/
def Fooling : Prop :=
  ∀ {ι κ : Type} (C : IntDag ι) (F : Finset κ) (x : κ → ι → ℤ) (j : ℕ) (out : List ℕ),
    (∀ s ∈ F, InWindow C (x s)) →
    (∀ s ∈ F, ∀ s' ∈ F, ∀ i, FreeInput C i → x s i ≡ x s' i [ZMOD 2 ^ j]) →
    (∀ s ∈ F, ∀ s' ∈ F, s ≠ s' → ∃ o ∈ out, ¬ val C (x s) o ≡ val C (x s') o [ZMOD 2 ^ j]) →
    F.card ≤ volume C

/-- **The carry bound.** For `s ≤ 255` with `256^(2s+1) ≤ p` and `2s ≤ n`: a circuit `C` whose registered and coin
inputs read the bytes of `a` (`slot i` the byte input `i` reads), and that on every `a = sparse S` with
`S ∈ G ⊆ 2^{0..s−1}` is in its window and outputs (`out l`, `l < n`) the bytes of `a² mod p`, has `|G| ≤ volume C`. -/
def SquaringCarry : Prop :=
  ∀ {ι : Type} (C : IntDag ι) (p n s : ℕ) (G : Finset (Finset ℕ)) (x : Finset ℕ → ι → ℤ) (slot : ι → ℕ)
    (out : ℕ → ℕ),
    s ≤ 255 → 256 ^ (2 * s + 1) ≤ p → 2 * s ≤ n → G ⊆ (Finset.range s).powerset →
    (∀ S ∈ G, ∀ i, FreeInput C i → x S i = byte (sparse S) (slot i)) →
    (∀ S ∈ G, InWindow C (x S) ∧ ∀ l < n, val C (x S) (out l) = byte (sparse S ^ 2 % p) l) →
    G.card ≤ volume C

/-- **The carry bound, a nibble a byte.** For `(249·256^s)² ≤ p` and `s + 2 ≤ n`: a circuit `C` whose registered and
coin inputs read the bytes of `a` (`slot i` the byte input `i` reads), and that on every `a = nibbles s t` with
`t ∈ G` is in its window and outputs (`out l`, `l < n`) the bytes of `a² mod p`, has `|G| ≤ volume C`. On the whole
family (`16^s` members) its charged gates have at least `4s` bits: 252 at p1024 (`s = 63`). -/
def SquaringCarryNibbles : Prop :=
  ∀ {ι : Type} (C : IntDag ι) (p n s : ℕ) (G : Finset (Fin s → Fin 16)) (x : (Fin s → Fin 16) → ι → ℤ)
    (slot : ι → ℕ) (out : ℕ → ℕ),
    (249 * 256 ^ s) ^ 2 ≤ p → s + 2 ≤ n →
    (∀ t ∈ G, ∀ i, FreeInput C i → x t i = byte (nibbles s t) (slot i)) →
    (∀ t ∈ G, InWindow C (x t) ∧ ∀ l < n, val C (x t) (out l) = byte (nibbles s t ^ 2 % p) l) →
    G.card ≤ volume C

/-- **The fooling bound in units.** For a circuit `C` whose budgeted gates but gathers each declare at most `2^w`
values (`w ≥ 1`), and a set `F` of inputs on each of which `C` is in its window, that agree in every registered input,
agree pairwise mod `2^j` in every coin input, and pairwise differ mod `2^j` in some output gate of `out`:
`|F| ≤ 2^(runBits C + w · budgeted C)`. A budgeted gate that reads a run or coin value carries its value's `w` bits
for its unit; a gather's units carry its index's outcomes, the fresh index bits of `IntGate.indexBits`, shared by the
gathers of one index and kind; every other gate is free or reads only registered values. -/
def UnitsFooling : Prop :=
  ∀ {ι κ : Type} (C : IntDag ι) (F : Finset κ) (x : κ → ι → ℤ) (j w : ℕ) (out : List ℕ),
    1 ≤ w → (∀ g ∈ C, ∀ cls f, g.op = .budgeted cls f → g.size ≤ 2 ^ w) →
    (∀ s ∈ F, InWindow C (x s)) →
    (∀ s ∈ F, ∀ s' ∈ F, ∀ i, (∃ g ∈ C, g.op = .input .registered i) → x s i = x s' i) →
    (∀ s ∈ F, ∀ s' ∈ F, ∀ i, (∃ g ∈ C, g.op = .input .coin i) → x s i ≡ x s' i [ZMOD 2 ^ j]) →
    (∀ s ∈ F, ∀ s' ∈ F, s ≠ s' → ∃ o ∈ out, ¬ val C (x s) o ≡ val C (x s') o [ZMOD 2 ^ j]) →
    F.card ≤ 2 ^ (runBits C + w * budgeted C)

/-- **The carry bound in units.** For `(249·256^s)² ≤ p`, `s + 2 ≤ n` and `w ≥ 1`: a circuit `C` whose budgeted gates
but gathers each declare at most `2^w` values, whose registered inputs are fixed and whose coin inputs read the bytes
of `a` (`slot i` the byte input `i` reads), and that on every `a = nibbles s t` with `t ∈ G` is in its window and
outputs (`out l`, `l < n`) the bytes of `a² mod p`, has `|G| ≤ 2^(runBits C + w · budgeted C)`. On the whole family,
`4s ≤ runBits C + w · budgeted C`: at p1024 (`s = 63`), with no run input and `w = 33`, at least 8 units. -/
def SquaringUnits : Prop :=
  ∀ {ι : Type} (C : IntDag ι) (p n s w : ℕ) (G : Finset (Fin s → Fin 16)) (x : (Fin s → Fin 16) → ι → ℤ)
    (slot : ι → ℕ) (out : ℕ → ℕ),
    (249 * 256 ^ s) ^ 2 ≤ p → s + 2 ≤ n → 1 ≤ w →
    (∀ g ∈ C, ∀ cls f, g.op = .budgeted cls f → g.size ≤ 2 ^ w) →
    (∀ t ∈ G, ∀ t' ∈ G, ∀ i, (∃ g ∈ C, g.op = .input .registered i) → x t i = x t' i) →
    (∀ t ∈ G, ∀ i, (∃ g ∈ C, g.op = .input .coin i) → x t i = byte (nibbles s t) (slot i)) →
    (∀ t ∈ G, InWindow C (x t) ∧ ∀ l < n, val C (x t) (out l) = byte (nibbles s t ^ 2 % p) l) →
    G.card ≤ 2 ^ (runBits C + w * budgeted C)

/-- **The fooling bound's information cap.** For any circuit `C` and any set `F` of inputs that pairwise differ mod
`2^j` in some output gate of `out`: `|F| ≤ 2^(j · |out|)`. So the fooling bounds (`Fooling`, `UnitsFooling`) charge a
circuit at most `j · |out|` bits, `j · |out| / w` units, whatever it computes between its inputs and its outputs: a
squaring's floor and a chain's sum can't come from counting. -/
def FoolingCap : Prop :=
  ∀ {ι κ : Type} (C : IntDag ι) (F : Finset κ) (x : κ → ι → ℤ) (j : ℕ) (out : List ℕ),
    (∀ s ∈ F, ∀ s' ∈ F, s ≠ s' → ∃ o ∈ out, ¬ val C (x s) o ≡ val C (x s') o [ZMOD 2 ^ j]) →
    F.card ≤ 2 ^ (j * out.length)

/-- **Re-normalization** (Lemma R; unproved): a square's limbs come back to byte format only through budgeted gates,
about one a limb. Take a circuit whose window is proven, with no run input, whose registered inputs are fixed and whose
coin inputs are bytes of `a` or fixed (so units count what reads `a`), that on more than `ε ≥ 2^−t` of the `a ∈ [0, p)`
holds its declared intervals (so is in its window) and outputs `L + c` values (`out l`, `L = n/8`), each within a byte
format (`fmt l`), whose base-256 sum is `a²` mod `p`. Then `ε · 2^(8L) ≤ 2^(8·(budgeted C + d₀ + c))`, that is
`budgeted C ≥ L − c − d(ε)` with `d(ε) = d₀ + log₂(1/ε)/8`. The outputs are any byte-format representation (signed
digits, extra limbs, unreduced), since the chain's next squaring reads any; for one squaring no cap on operand bits is
needed. The s8 family forms the column sums of `a²` for free; what it can't do is carry, and `SquaringCarry` proves only
bits of that (63 at p1024), not units. The claimed instance is p1024 with `d₀ = 4` and `t = 64`.

**`d₀` is not absolute, so the floor.** A circuit with no budgeted gate (`a`'s bytes as coin inputs declared `[0, D_l]`,
`a²`'s column sums as s8-family steps, folded mod `p`) is exact on every `a` whose column sums don't carry, and such
families grow with `L` faster than any fixed `d₀` covers: `2^33.30` members at `p = 256^27 − 377` (`d₀ ≥ 4.16`),
`2^101.85` at p1024 (`d₀ ≥ 12.73`), and, sampled, at least `2^159.4` at p1024 (64 digits from a geometric law;
`d₀ ≥ 19.9`), which refutes the slope `1/7` too (`d = 4 + log₂(1/ε)/7` allows at most `2^156`). Every one sits at
`ε ≤ 2^−182`, and none can reach the floor at p1024: a column holds `a_l²`, so a non-carrying `a` has digits below 16
and `ε ≤ 16^L/p ≈ 2^−512`. Above the floor every family found is within `d₀ = 4`: the exhaustive toy fields (implied
`d₀ ≤ 2.18`), the non-carrying sets of the byte fields `L ≤ 10`, counted exactly (`≤ 2.48`), and at p1024 the
carry-guessing frontier (modeled, 10 to 15 bits a dropped carry: `d = 1 … 5` at `ε = 2^−9.8 … 2^−69.5`, against the
formula's `5.2 … 12.7`). `t = 64` keeps the claim inside the range that frontier reaches; the bound reads `ε = 2^−40`
(the headline) and `2^−20` (the property's carrier threshold). -/
def Renormalization (F : Field) (d₀ c t εnum εden : ℕ) : Prop :=
  ∀ {ι : Type} (C : IntDag ι) (x : ℕ → ι → ℤ) (out : ℕ → ℕ) (fmt : ℕ → Int8Fmt) (good : Finset ℕ),
    εden ≤ 2 ^ t * εnum →
    Proven C → runBits C = 0 → good ⊆ Finset.range F.p → MoreThan (Finset.range F.p) good εnum εden →
    (∀ i, (∃ g ∈ C, g.op = .input .registered i) → ∀ a ∈ good, ∀ a' ∈ good, x a i = x a' i) →
    (∀ i, (∃ g ∈ C, g.op = .input .coin i) →
      (∃ l, ∀ a ∈ good, x a i = byte a l) ∨ ∀ a ∈ good, ∀ a' ∈ good, x a i = x a' i) →
    (∀ a ∈ good, DeclaredHold C (x a) ∧
      (∀ l < F.bytes + c, (fmt l).lo ≤ val C (x a) (out l) ∧ val C (x a) (out l) ≤ (fmt l).hi) ∧
      ∑ l ∈ Finset.range (F.bytes + c), 256 ^ l * val C (x a) (out l) ≡ (a : ℤ) ^ 2 [ZMOD F.p]) →
    εnum * 2 ^ (8 * F.bytes) ≤ εden * 2 ^ (8 * (budgeted C + d₀ + c))

/-- **`ρ` on pairs.** At a prime `p ≡ 3 mod 4` and `0 < x < p`: `ρ(p − x) = p − ρ(x)`, and `ρ(x)` is odd exactly when
`x` is not a square mod `p`. So `ρ` maps the pair `{x, p − x}` onto the pair `{s, p − s}` of `±x`'s roots, taking the
member the Legendre symbol of `x` names: the ideal model of a round is a random function on such pairs with that sign
rule, not a random permutation, and the bit `lessParity` drops is the Legendre symbol of the last round's input. -/
def RhoPairs : Prop :=
  ∀ p x : ℕ, p.Prime → p % 4 = 3 → 0 < x → x < p →
    rho p (p - x) = p - rho p x ∧ (rho p x % 2 = 1 ↔ ¬ ∃ y, y ^ 2 % p = x)

/-- **The encode budget: what the stacking property bounds.** For a declared circuit `W` in the restricted class
(`Restricted B`) whose census satisfies the stacking property `P` (`Property caps A W`), whose registered and coin
inputs are the codec's (`CodecInputs`), and whose values hold, byte for byte less parity, the encodings of the `q`
distinct words `words` on more than `ε` of the fresh draws, its inputs and budgeted gates but gathers there in
their declared intervals (its window is then `ProvenWindow`'s, never a hypothesis): under `SquaringFloor F e ε` and
`ChainSum F k B e ε`, at `E ≤ k · S · e` (`S = Field.squarings F`) and inside their floor (`ε ≥ 2^−64`, `k ≥ 1`,
`p ≥ 2^66`, `p ≡ 3 mod 4`),

  `q · E · (n − 1) ≤ (c · Σ_w reads_w) · (n − 1) + 8 · (Σ_w R_in,w) · E`,

with `c` the caps' nonlinear units per certified byte read, a rational (property/v1's units: gathers by fresh index
bits, and a unit for each data product), and `R_in,w` window `w`'s run-input bytes:
`q ≤ c · reads / E + 8 · Σ_w R_in,w / (n − 1)`. -/
def EncodeBudget : Prop :=
  ∀ (F : Field) (k B E e εnum εden : ℕ) {ι : Type} (W : IntDag ι) (caps : Caps) (A : Audit) (m : ℕ → ℕ)
    (x : Draw k → ι → ℤ) (words : List ℕ) (out : ℕ → ℕ → ℕ),
    SquaringFloor F e εnum εden → ChainSum F k B e εnum εden → E ≤ k * F.squarings * e →
    εden ≤ 2 ^ 64 * εnum → 0 < k → 2 ^ 66 ≤ F.p → F.p % 4 = 3 →
    Property caps A W → Restricted B W → CodecInputs F W (draws F k) m x → words.Nodup →
    (∀ i ∈ words, i < F.p) → ComputesWords F k W x m words out εnum εden →
    ((words.length * E * (F.n - 1) : ℕ) : ℚ) ≤
      caps.total * (∑ v ∈ Finset.range A.windows, A.reads v : ℕ) * (F.n - 1 : ℕ) +
        (8 * (∑ v ∈ Finset.range A.windows, caps.runBytes v) * E : ℕ)

/-- **The encode cost, bounded.** Under `SquaringFloor F e ε` and `ChainSum F k B e ε`, at `E ≤ k · S · e` and inside
their floor (`ε ≥ 2^−64`, `k ≥ 1`, `p ≥ 2^66`, `p ≡ 3 mod 4`), with `n ≥ 2`, `q` distinct words cost at least `q · E`
nonlinear units: `q · E ≤ encodeCost F k B ε q`, which is `⊤` when no restricted circuit with no run input computes them
on more than `ε` of the draws. -/
def EncodeCost : Prop :=
  ∀ (F : Field) (k B E e εnum εden q : ℕ),
    SquaringFloor F e εnum εden → ChainSum F k B e εnum εden → E ≤ k * F.squarings * e →
    εden ≤ 2 ^ 64 * εnum → 0 < k → 2 ^ 66 ≤ F.p → F.p % 4 = 3 → 2 ≤ F.n →
      ((q * E : ℕ) : ℕ∞) ≤ encodeCost F k B εnum εden q

/-! ## Primitives' output intervals

What `DeclaredHold` asks of a budgeted gate, that its value lies in its declared interval, for the primitives whose
interval property/v1 derives (P's producers) and the ones it reads as declared: each a theorem over the primitive's
evaluator (`Definitions/Accounting/Gates/Ranges`). -/

/-- **`F32Fmaxf`'s rule holds.** Where property/v1's rule (`fmaxfRule`, `stacking._fmaxf`) gives an interval `o` for
operand ranges `a` and `b`, `fmaxf` of any s32 readings in them lies in `o`. -/
def FmaxfRange : Prop :=
  ∀ a b o, fmaxfRule a b = some o → ∀ x y, S32 x → S32 y → Rng.Holds a x → Rng.Holds b y → o.Mem (f32Fmaxf x y)

/-- **`F32Fminf`'s rule holds.** Where `fminfRule` (`stacking._fminf`) gives `o` for operand ranges `a` and `b`,
`fminf` of any s32 readings in them lies in `o`. -/
def FminfRange : Prop :=
  ∀ a b o, fminfRule a b = some o → ∀ x y, S32 x → S32 y → Rng.Holds a x → Rng.Holds b y → o.Mem (f32Fminf x y)

/-- **`F32ToU32RzByte`'s rule holds.** Where `rzByteRule` (`stacking._rz_byte`) gives `o` for an operand range `a`,
the low byte of `cvt.rzi.u32.f32` of any reading in it lies in `o`. -/
def RzByteRange : Prop :=
  ∀ a o, rzByteRule a = some o → ∀ x, Rng.Holds a x → o.Mem (f32RzByte x)

/-- **A clamped quantizer's interval.** `clamp(round(x/s), lo, lo + w)` lies in `[lo, lo + w]` whatever the rounding:
`[−127, 127]` for steps.py's Q, K, V, activations and weights at `top = 127`, `[0, p_top]` for P. -/
def QuantRange : Prop :=
  ∀ (lo : ℤ) (w : ℕ) (rnd : ℚ → ℤ) (v : ℚ), Iv.Mem (lo, lo + w) (quantize lo w rnd v)

/-- **The dither's interval.** A `b`-bit dither of `⌊v⌋` clamped to `[lo, lo + w]` lies in `[lo − m, lo + w + m + 1]`,
`m = 2^(b−1) − 1`: `[−128, 127]` for `floor_quant` at `top = 127` (`lo = m − 128`, `w = 254 − 2m`), and
`[1 − top, top]` for `act_quant` (`lo = −t`, `w = 2t`, `t = top − 1 − m`). -/
def DitherRange : Prop :=
  ∀ (lo : ℤ) (w m : ℕ) (v : ℚ) (R : Fin (2 * m + 2)), Iv.Mem (lo - m, lo + w + m + 1) (dither lo w m v R)

/-- **The LUT softmax's interval.** Its element lies in the hull of its table and 0 (its gather's fill, the value
outside its column): in `[0, 127]` for a table of values in `[0, 127]`. -/
def LutSoftmaxRange : Prop :=
  ∀ (T : List ℤ) (d : ℚ), (hull0 T).Mem (lutSoftmax T d)

/-- **`Bf16ToF32` is free-affine.** The registry classes `Bf16ToF32_v1` affine, and on every bf16 word `u` its
evaluator's f32 word (`Verity.Prims.bf16ToF32`), read as s32, is `bf16ToF32Op` on `u` read as s16: `×2^16` with no
wrap, its image `[−2^31, 2^31 − 2^16]` within s32. So a `Bf16ToF32_v1` gate is the free affine map property/v1 counts
it, and `Proven` proves its window. -/
def Bf16ToF32Affine : Prop :=
  classOf "Bf16ToF32_v1" = some .affine ∧
    ∀ u : Fin (2 ^ 16),
      signed 32 (Verity.Prims.bf16ToF32 u.val) = (bf16ToF32Op : IntOp Unit).raw (fun _ => 0) [signed 16 u.val]

/-- **`F32ToS8Rzi`'s interval.** Its code, read as s8, lies in `[−128, 127]`: its type, by saturation. -/
def F32ToS8RziRange : Prop :=
  ∀ a, Iv.Mem (-128, 127) (f32ToS8Rzi a)

/-- **The int8 quantizer's clamp.** `Int8QuantElement`'s `s8(clamp(r, −127, 127))` (`F32Fmaxf` against `−127.0`,
`F32Fminf` against `127.0`, `F32ToS8Rzi`) of any s32 reading (`wrapS32 z`, which is every s32 word for some `z`), a
NaN included, lies in `[−127, 127]`: the s8 family's static `A = (−127, 127)`. The interval analysis over s32
readings can't carry it from the clamp to the cast (a float in `[−127.0, 127.0]` reads as two s32 intervals), so a
gate-by-gate window takes the cast's type, `[−128, 127]`, as property/v1 does. -/
def Int8ClampRange : Prop :=
  ∀ z, Iv.Mem (-127, 127) (int8Clamp (wrapS32 z))

/-- **`LutIndex`'s interval.** `[0, 255]`, whatever its operands. -/
def LutIndexRange : Prop :=
  ∀ d a, Iv.Mem (0, 255) (lutIndex d a)

/-- **`Exp2Lut127`'s table.** 256 entries, and a read at any index lies in `[0, 127]` (past the table, 0). -/
def Exp2Lut127Range : Prop :=
  exp2Lut127.length = 256 ∧ ∀ t : ℕ, Iv.Mem (0, 127) (exp2Lut127.getD t 0)

/-- **`S32LowByte`'s interval.** `[−128, 127]`, its type; and a word in `[0, 127]` is its own low byte, so P's codes
keep the LUT's `[0, 127]`. -/
def S32LowByteRange : Prop :=
  (∀ z, Iv.Mem (-128, 127) (s32LowByte z)) ∧ ∀ t : Fin 128, s32LowByte (t.val : ℤ) = t.val

/-! ## Declared bounds, earned

The window from nothing declared on trust: every input at its type's whole range, every budgeted gate's interval a
theorem over its evaluator on its operands' derived intervals (`Definitions/Accounting/Gates/Declared`). -/

/-- **Declared bounds, earned, hold.** A circuit whose window is proven and whose every declared bound is earned
(`Declared`: an input's is its type's whole range, a budgeted gate's holds its evaluator's value on its operands'
derived intervals) holds its declared intervals and is in its window on every input that gives each slot a word of
its type (`Typed`): no hypothesis that a run input lies within a narrower bound. -/
def DeclaredWindow : Prop :=
  ∀ {ι : Type} (C : IntDag ι) (x : ι → ℤ), Proven C → Declared C → Typed C x → DeclaredHold C x ∧ InWindow C x

/-- **The int8 linear's window, by theorem.** For `K = 32·S ≤ 131,040` (`S ≤ 4095`; gemm's `S8_K_LIMIT`, `K < 2^17`),
the int8 linear's coordinate (`int8Linear`: `Int8RowQuant`'s codes, then `ScaledMmS8_v1`'s s8·s8 dot) is proven and
earns every declared bound, whatever its row scale and rounding compute: its codes and weights are s8 by type
(`F32ToS8Rzi`'s saturation, the weights' width), so each product is in `[−16256, 16384]` and the accumulator stays in
s32. So it is in its window on every input that gives each slot a word of its type. -/
def Int8LinearWindow : Prop :=
  ∀ (S : ℕ) (fs fr : List ℤ → ℤ), S ≤ 4095 →
    Proven (int8Linear S fs fr) ∧ Declared (int8Linear S fs fr) ∧
      ∀ x, Typed (int8Linear S fs fr) x → InWindow (int8Linear S fs fr) x

/-- **The LUT attention's `P V'` window, by theorem.** For `32·S ≤ 132,096` padded keys (`S ≤ 4128`), any number `n` of
them real, and a table of values in `[0, 127]` (`Exp2Lut127Range`'s), the `P V'` channel (`lutPv`) is proven and earns
every declared bound: P's codes are `S32LowByte` of a gather's value, which lies in the hull of the table and 0
(`[0, 127]`, derived), V's are s8 by type, so each product is in `[−16256, 16129]`. So it is in its window on every
input that gives each slot a word of its type. property/v1 reads `S32LowByte` at its type, `[−128, 127]`, and so
refuses past `K = 131,040` (the row's `row_refusal`). -/
def LutPvWindow : Prop :=
  ∀ (S n : ℕ) (T : List ℤ), S ≤ 4128 → (∀ t ∈ T, 0 ≤ t ∧ t ≤ 127) →
    Proven (lutPv S n T) ∧ Declared (lutPv S n T) ∧ ∀ x, Typed (lutPv S n T) x → InWindow (lutPv S n T) x

end Accounting.Guarantees.Gates
