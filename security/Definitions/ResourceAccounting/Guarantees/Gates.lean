module

public import Definitions.ResourceAccounting.Assumptions.SquaringChain
public import Definitions.ResourceAccounting.Gates
public import Definitions.ResourceAccounting.Gates.Declared
public import Definitions.ResourceAccounting.Gates.Registry
public import Definitions.Core.Prims
public import Mathlib.Data.Nat.Prime.Defs

/-! The circuit model's window and its lower bounds, the fooling, carry and encode bounds, with Lemma R (`Renormalization`) stated and unproved. -/

@[expose] public section

namespace ResourceAccounting.Guarantees.Gates

open ResourceAccounting.Gates ResourceAccounting.Assumptions

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
`budgeted C ≥ L − c − d(ε)`. Claimed at p1024 with `d₀ = 4` and `t = 64`; `d₀` is not absolute, so with no floor on
`ε` it is false. -/
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

/-- **The encode budget: what the stacking property bounds.** For a registered circuit `W` in the restricted class
(`Restricted B`) whose census satisfies the stacking property `P` (`Property caps A W`), whose registered and coin
inputs are the codec's (`CodecInputs`), and whose values hold, byte for byte less parity, the encodings of the `q`
distinct words `words` on more than `ε` of the fresh samples, its inputs and budgeted gates but gathers there in
their declared intervals (its window is then `ProvenWindow`'s, never a hypothesis): under `SquaringFloor F e ε` and
`ChainSum F k B e ε`, at `E ≤ k · S · e` (`S = Field.squarings F`) and inside their floor (`ε ≥ 2^−64`, `k ≥ 1`,
`p ≥ 2^66`, `p ≡ 3 mod 4`),

  `q · E · (n − 1) ≤ (c · Σ_w reads_w) · (n − 1) + 8 · (Σ_w R_in,w) · E`,

with `c` the caps' nonlinear units per certified byte read, a rational (property/v1's units: gathers by fresh index
bits, and a unit for each data product), and `R_in,w` window `w`'s run-input bytes:
`q ≤ c · reads / E + 8 · Σ_w R_in,w / (n − 1)`. -/
def EncodeBudget : Prop :=
  ∀ (F : Field) (k B E e εnum εden : ℕ) {ι : Type} (W : IntDag ι) (caps : Caps) (A : Audit) (m : ℕ → ℕ)
    (x : Sample k → ι → ℤ) (words : List ℕ) (out : ℕ → ℕ → ℕ),
    SquaringFloor F e εnum εden → ChainSum F k B e εnum εden → E ≤ k * F.squarings * e →
    εden ≤ 2 ^ 64 * εnum → 0 < k → 2 ^ 66 ≤ F.p → F.p % 4 = 3 →
    Property caps A W → Restricted B W → CodecInputs F W (samples F k) m x → words.Nodup →
    (∀ i ∈ words, i < F.p) → ComputesWords F k W x m words out εnum εden →
    ((words.length * E * (F.n - 1) : ℕ) : ℚ) ≤
      caps.total * (∑ v ∈ Finset.range A.windows, A.reads v : ℕ) * (F.n - 1 : ℕ) +
        (8 * (∑ v ∈ Finset.range A.windows, caps.runBytes v) * E : ℕ)

/-- **The encode cost, bounded.** Under `SquaringFloor F e ε` and `ChainSum F k B e ε`, at `E ≤ k · S · e` and inside
their floor (`ε ≥ 2^−64`, `k ≥ 1`, `p ≥ 2^66`, `p ≡ 3 mod 4`), with `n ≥ 2`, `q` distinct words cost at least `q · E`
nonlinear units: `q · E ≤ encodeCost F k B ε q`, which is `⊤` when no restricted circuit with no run input computes them
on more than `ε` of the samples. -/
def EncodeCost : Prop :=
  ∀ (F : Field) (k B E e εnum εden q : ℕ),
    SquaringFloor F e εnum εden → ChainSum F k B e εnum εden → E ≤ k * F.squarings * e →
    εden ≤ 2 ^ 64 * εnum → 0 < k → 2 ^ 66 ≤ F.p → F.p % 4 = 3 → 2 ≤ F.n →
      ((q * E : ℕ) : ℕ∞) ≤ encodeCost F k B εnum εden q

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

end ResourceAccounting.Guarantees.Gates
