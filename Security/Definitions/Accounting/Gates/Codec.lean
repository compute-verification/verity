import Mathlib.Data.Fintype.Pi
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Data.Finset.Prod
import Mathlib.Order.Interval.Finset.Nat
import Mathlib.Order.Lattice.Nat
import Mathlib.Data.ENat.Lattice
import Definitions.Accounting.Gates.Model

/-!
# The word codec in the circuit model: what computing a block's encoding means (trusted)

The codec of `verity/core/protocols/poub/codec.py` (masked, at `k` rounds over `p = 2^n − c`), and what a
circuit of the model (`Model.lean`) computing its words is:

* **The codec.** `rho p a` is `codec.rho`: `s = a^((p+1)/4) mod p`, the root of `a` or of `−a`, taken even when `a` is
  a square and odd when it isn't. `encodeWord p K t m` is `codec.encode_word`: `x ← m`, then for each round `r`,
  `x ← σ_K(x)` when `r ≥ 1` and `x ← ρ((x + c_r + t) mod p)`. Word `i`'s tweak is `t_i = τ·i mod p` (an unbound block,
  `b = 0`; word `i` of block `j` is `i = 32j + w` at p1024).
* **Draws.** A key is drawn after the circuit is declared: its round constants `c_r ∈ [0, p)`, its mask `K ∈ [0, 2^n)`
  and its tweak multiplier `τ ∈ [1, p)`, uniform (`draws`): the codec's seed expanded by an ideal XOF.
* **Inputs** (`CodecInputs`), by their binding. A registered input is bound before the draw, so it is one value on
  every draw: a byte of the payload `m`, or anything else fixed then. A coin input is a byte of the draw (a round
  constant's, the mask's, a word's tweak) or one value on every draw: a live verifier coin of another audit (a noise
  or fold key) is independent of the codec's draw. A run input may be anything, a function of the draw among them, and
  the bounds charge its bits. Units (`units`) count a gate that reads a coin or run input, so the draw costs where it
  enters; a registered input holding draw bytes would make the encode free, which is why it can't.
* **Targets.** Word `i`'s target is its encoding less its parity (`lessParity`: the low bit cleared), byte for byte in
  the store's layout (`byte`, little-endian). The dropped bit is the Legendre symbol of the last round's input: `ρ(x)`
  is even exactly when `x` is a square mod `p` (`p ≡ 3 mod 4`), and a circuit gets that symbol without the root.
  For the same reason `ρ(p − x) = p − ρ(x)` (`Guarantees.Gates.RhoPairs`): `ρ` maps the pair `{x, p − x}` to the
  pair `{s, p − s}` of `±x`'s roots and takes the member the symbol names, so the ideal model of a round is a random
  function on such pairs with that sign rule, not a random permutation of `[0, p)`.
* **Computing.** A circuit computes words `words` on a draw when its inputs and budgeted gates but gathers hold
  their declared intervals there (`DeclaredHold`, which with `Proven` puts it in its window) and its output gates
  `out i l` hold the targets' bytes (`Encodes`). `encodeCost` is the fewest nonlinear units (`nonlinear`: budgeted
  units and data products) of any circuit of the model, with its window proven, at most `B` operand bits a budgeted gate
  (`Restricted`) and no run input, that computes `q` distinct words on more than `ε` of the draws, and `⊤` when there
  is none: the cost 6a's `encodeBitsCost` reads.
* **The encode bound** (`EncodeBound F k B E ε`): every such circuit, run inputs allowed, pays `E` nonlinear units a
  word, or `n − 1` run-input bits. A root is `Field.squarings` sequential squarings (1,021 at p1024).
-/

namespace Accounting.Gates

/-- A codec's field, `p = 2^n − c`. -/
structure Field where
  n : ℕ
  c : ℕ

/-- The prime. -/
def Field.p (F : Field) : ℕ := 2 ^ F.n - F.c

/-- A word's bytes. -/
def Field.bytes (F : Field) : ℕ := F.n / 8

/-- The deployed field `p1024`: `2^1024 − 105`. -/
def p1024 : Field := ⟨1024, 105⟩

/-- Byte `l` of `N`, little-endian. -/
def byte (N l : ℕ) : ℕ := N / 256 ^ l % 256

/-- `codec.rho`: the root of `a` or of `−a`, even when `a` is a square mod `p` and odd when it isn't. -/
def rho (p a : ℕ) : ℕ :=
  let s := a ^ ((p + 1) / 4) % p
  if s % 2 = (if s * s % p = a then 0 else 1) then s else if s = 0 then 0 else p - s

/-- A key's round constants and its mask. -/
structure Key where
  c : List ℕ
  mask : ℕ

/-- `σ_K(x)`: `x xor K` when that is below `p`, else `x`. -/
def sigma (p mask x : ℕ) : ℕ := if x ^^^ mask < p then x ^^^ mask else x

/-- `codec.encode_word` of payload word `m` at tweak `t`, masked. -/
def encodeWord (p : ℕ) (K : Key) (t m : ℕ) : ℕ :=
  K.c.zipIdx.foldl (fun x e => rho p (((if e.2 = 0 then x else sigma p K.mask x) + e.1 + t) % p)) m

/-- A word less its parity: its low bit cleared. -/
def lessParity (y : ℕ) : ℕ := y / 2 * 2

/-- A draw of a `k`-round key: its round constants, its mask and its tweak multiplier. -/
abbrev Draw (k : ℕ) := (Fin k → ℕ) × ℕ × ℕ

/-- The draws, uniform: `c_r ∈ [0, p)`, `K ∈ [0, 2^n)`, `τ ∈ [1, p)`. -/
def draws (F : Field) (k : ℕ) : Finset (Draw k) :=
  Fintype.piFinset (fun _ => Finset.range F.p) ×ˢ (Finset.range (2 ^ F.n) ×ˢ Finset.Ico 1 F.p)

/-- A draw's key. -/
def Draw.key {k : ℕ} (d : Draw k) : Key := ⟨List.ofFn d.1, d.2.1⟩

/-- Word `i`'s tweak under a draw, `τ·i mod p`. -/
def Draw.tweak {k : ℕ} (F : Field) (d : Draw k) (i : ℕ) : ℕ := d.2.2 * i % F.p

/-- A registered input's canonical slot: a byte of a payload word, of a round constant, of the mask, or of a word's
tweak. -/
inductive Slot
  | payload (i l : ℕ)
  | const (r l : ℕ)
  | mask (l : ℕ)
  | tweak (i l : ℕ)

/-- A slot's value under a draw, for payload `m`. -/
def Slot.value {k : ℕ} (F : Field) (d : Draw k) (m : ℕ → ℕ) : Slot → ℤ
  | .payload i l => byte (m i) l
  | .const r l => if h : r < k then byte (d.1 ⟨r, h⟩) l else 0
  | .mask l => byte d.2.1 l
  | .tweak i l => byte (d.tweak F i) l

variable {ι : Type} {k : ℕ}

/-- The codec's inputs over draws `D`, by binding: a registered input is one value on every draw (a payload byte, or
anything fixed before the draw); a coin input is a canonical slot's byte on every draw, or one value on every draw (a
coin independent of the draw). -/
def CodecInputs (F : Field) (C : Dag ι) (D : Finset (Draw k)) (m : ℕ → ℕ) (x : Draw k → ι → ℤ) : Prop :=
  (∀ i, (∃ g ∈ C, g.op = .input .registered i) → ∀ d ∈ D, ∀ d' ∈ D, x d i = x d' i) ∧
    ∀ i, (∃ g ∈ C, g.op = .input .coin i) →
      (∃ s : Slot, ∀ d ∈ D, x d i = s.value F d m) ∨ ∀ d ∈ D, ∀ d' ∈ D, x d i = x d' i

/-- On draw `d`, `C`'s inputs and budgeted gates but gathers hold their declared intervals and its outputs `out i l`
hold the bytes of words `words`' encodings, less their parity. -/
def Encodes (F : Field) (C : Dag ι) (x : Draw k → ι → ℤ) (m : ℕ → ℕ) (words : List ℕ) (out : ℕ → ℕ → ℕ)
    (d : Draw k) : Prop :=
  DeclaredHold C (x d) ∧
    ∀ i ∈ words, ∀ l < F.bytes,
      val C (x d) (out i l) = byte (lessParity (encodeWord F.p d.key (d.tweak F i) (m i))) l

/-- `C` computes words `words` on more than `ε = εnum/εden` of the draws: on a set `good` of draws that large. -/
def ComputesWords (F : Field) (k : ℕ) (C : Dag ι) (x : Draw k → ι → ℤ) (m : ℕ → ℕ) (words : List ℕ)
    (out : ℕ → ℕ → ℕ) (εnum εden : ℕ) : Prop :=
  ∃ good ⊆ draws F k, MoreThan (draws F k) good εnum εden ∧ ∀ d ∈ good, Encodes F C x m words out d

/-- A circuit of the chain's restricted class: its window proven, and at most `B` operand bits a budgeted gate. -/
def Restricted (B : ℕ) (C : Dag ι) : Prop := Proven C ∧ OperandBits B C

/-- The nonlinear units (`nonlinear`: budgeted units and data products) of the restricted circuits with no run-input
bits that compute `q` distinct words (each below `p`) of the codec's encoding, less parity, on more than `ε` of the
draws, from the payload and whatever they registered before the draw. -/
def encodeCosts (F : Field) (k B εnum εden q : ℕ) : Set ℕ :=
  {c | ∃ (ι : Type) (C : Dag ι) (m : ℕ → ℕ) (x : Draw k → ι → ℤ) (words : List ℕ) (out : ℕ → ℕ → ℕ),
    words.length = q ∧ words.Nodup ∧ (∀ i ∈ words, i < F.p) ∧ Restricted B C ∧ runBits C = 0 ∧
      CodecInputs F C (draws F k) m x ∧ ComputesWords F k C x m words out εnum εden ∧ nonlinear C = c}

/-- **The encode cost**: the fewest nonlinear units of any circuit of `encodeCosts`, and `⊤` when there is none (6a's
`encodeBitsCost` at the words its bits fill). -/
noncomputable def encodeCost (F : Field) (k B εnum εden q : ℕ) : ℕ∞ :=
  ⨅ c ∈ encodeCosts F k B εnum εden q, (c : ℕ∞)

/-- **A root's squarings**: `ρ` raises to `(p + 1)/4`, `bitlen((p + 1)/4) − 1` sequential squarings mod `p` (and a
multiply for each other set bit), 1,021 at p1024. -/
def Field.squarings (F : Field) : ℕ := Nat.log 2 ((F.p + 1) / 4)

/-- **The encode bound at `E`**: every restricted circuit (`Restricted B`) whose inputs are the codec's (`CodecInputs`)
and that computes the `q` distinct words `words` (each below `p`) of the masked `k`-round codec over `F`, byte for byte
less their parity, on more than `ε = εnum/εden` of the draws (`ComputesWords`), has

  `q · E · (n − 1) ≤ nonlinear C · (n − 1) + runBits C · E`,

that is `q ≤ nonlinear C / E + runBits C / (n − 1)`: each word costs `E` nonlinear units, or `n − 1` bits of run
input (the word itself, sent after the draw). What the encode budget reads; `Assumptions.ChainSum` concludes it. -/
def EncodeBound (F : Field) (k B E εnum εden : ℕ) : Prop :=
  ∀ {ι : Type} (C : Dag ι) (m : ℕ → ℕ) (x : Draw k → ι → ℤ) (words : List ℕ) (out : ℕ → ℕ → ℕ),
    Restricted B C → CodecInputs F C (draws F k) m x → words.Nodup → (∀ i ∈ words, i < F.p) →
    ComputesWords F k C x m words out εnum εden →
    words.length * E * (F.n - 1) ≤ nonlinear C * (F.n - 1) + runBits C * E

/-- The carry bound's fooling family: `a_S = 16 · Σ_{i ∈ S} 256^i`, bit 4 of byte `i` set for `i ∈ S`, so every byte
of `a_S` is even and byte `2i + 1` of `a_S²` is odd exactly when `i ∈ S`. -/
def sparse (S : Finset ℕ) : ℕ := 16 * ∑ i ∈ S, 256 ^ i

/-- The nibble family: `a_t = 8 + Σ_{i<s} (16·t_i + 8)·256^(i+1)` for `t : Fin s → Fin 16`, byte 0 `8` and byte `i + 1`
`16·t_i + 8`, so every byte of `a_t` is `8 mod 16` and the high nibbles are `t`. For `t ≠ t'` first apart at `m`,
`a_t − a_t' = 16·256^(m+1)·D` with `D ≢ 0 mod 16` and `a_t + a_t' = 16·O` with `O` odd, so the squares agree below
byte `m + 2` and differ there mod 16. -/
def nibbles (s : ℕ) (t : Fin s → Fin 16) : ℕ := 8 + ∑ i : Fin s, (16 * (t i : ℕ) + 8) * 256 ^ (i.1 + 1)

end Accounting.Gates
