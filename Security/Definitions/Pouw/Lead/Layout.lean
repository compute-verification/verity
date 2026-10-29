import Mathlib.Data.Fintype.BigOperators
import Mathlib.Data.Rat.Cast.Order
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Order.Group.Unbundled.Abs

/-!
# The lead's layout, coins and committed words (trusted definitions)

The lead is `tag-blc4a-t64-rot/served/s8/k32/w/l1w/l1a/min`: served sparse noise with a row tag (`ssn-1l-tag`) and
three coin lanes a later step in 64-row tiles with 7-bit odd coins (`blc4a-t64-e7`), the lanes' row weights `A′` set by
row parity (`-rot`), served noisy with the coin lanes folded exactly, k32 steps, words chunks of 256 KiB (`l1w`) and
hiding L1 + (a).

* **The layout** (`Layout`) is the call's shape (`m` rows, `n` columns, `S` k32 steps a word) and the design's
  constants: the coin tile's rows, the coin lanes' bits, the words chunk's size, the egress cap the network term reads
  and the node's memory.
  `lead` fixes the design's constants; `pt512x64`, `pt512x512` and `pt8192` are the three evaluated points.
* **The coins** (`Coins`) are what the certifier draws after it holds the call's `A`: the tag bits `b`, the step-1
  bits `a₁`, the tag weights `G ∈ [−3, 3]^{4×n}` and the coin lanes `s ∈ {±1, ±3, …, ±(2^e − 1)}`, one a step, tile,
  lane and column. They are a finite type, and every bound averages over them uniformly.
* **The committed word** (`word`) of row `i`, step `t`, column `j` is `−β_j` plus, over the steps `τ ≤ t`, the data's
  step-`τ` dot product `κ i τ j` (fixed before the coins) and the step's coin term (`coinTerm`):
  - step 0: the row tag against `G` and `a₁` against the column code `C`;
  - step `τ ≥ 1`: the tag lane `1 + b_τ[i]`, and the three coin lanes `Σ_ℓ A′[i, ℓ] · s_τ[⌊i/64⌋, ℓ, j]`, with
    `A′ = (127, 1, 1)` on even rows and `(1, 127, 1)` on odd rows.

  The words are integers. That the served 32-bit register holds each one exactly is the TAGS lane's exactness piece.
-/

namespace Pouw.Lead

/-- **The lead's layout**: the call's shape and the design's constants. -/
structure Layout where
  /-- Tokens a step (the evaluated points' `T`). -/
  T : ℕ
  /-- Rows of the call. -/
  m : ℕ
  /-- Columns of the call. -/
  n : ℕ
  /-- Inner dimension of the call. -/
  K : ℕ
  /-- k32 steps a committed word: step 0 holds 24 data lanes, each later step 28 (three lanes are coins, one the tag). -/
  S : ℕ
  /-- Rows a coin tile: 64. -/
  tile : ℕ
  /-- Bits a coin lane's value: 7, so `s ∈ {±1, ±3, …, ±127}`. -/
  e : ℕ
  /-- 32-bit words a words chunk: 65,536 (256 KiB, `l1w`). -/
  chunkWords : ℕ
  /-- The egress cap the network term reads, in bits a step's word (32 or 320). -/
  cap : ℕ
  /-- The node's memory in MMA tiles of 512 bytes, its cards' and hosts': what pre-coin tables can hold. -/
  mem : ℕ

/-- Coin tiles a call: `⌈m / tile⌉`. -/
def Layout.tiles (L : Layout) : ℕ := (L.m + L.tile - 1) / L.tile

/-- **The call's coins**, drawn by the certifier after it holds the call's `A`:
`(b, a₁, g, u)` with tag bits `b i t`, step-1 bits `a₁ i e`, tag weights `G = g − 3` and coin lanes `s = 2u − (2^e − 1)`
(`u` a step, tile, lane and column). -/
abbrev Coins (L : Layout) : Type :=
  (Fin L.m → Fin L.S → Fin 2) × (Fin L.m → Fin 4 → Fin 2) × (Fin 4 → Fin L.n → Fin 7) ×
    (Fin L.S → Fin L.tiles → Fin 3 → Fin L.n → Fin (2 ^ L.e))

/-- A coin lane's value: `s = 2u − (2^e − 1)`, odd in `[−(2^e − 1), 2^e − 1]`. -/
def Coins.s {L : Layout} (ω : Coins L) (t : Fin L.S) (k : Fin L.tiles) (ℓ : Fin 3) (j : Fin L.n) : ℤ :=
  2 * ((ω.2.2.2 t k ℓ j : ℕ) : ℤ) - (2 ^ L.e - 1)

/-- A tag weight: `G = g − 3 ∈ [−3, 3]`. -/
def Coins.G {L : Layout} (ω : Coins L) (q : Fin 4) (j : Fin L.n) : ℤ := ((ω.2.2.1 q j : ℕ) : ℤ) - 3

/-- Digit `q` of `z` in balanced base 13, in `[−6, 6]`, for `|z| ≤ (13⁴ − 1)/2 = 14,280`: the plain digits of
`z + 14,280`, less 6. -/
def bal13 (z : ℤ) (q : Fin 4) : ℤ := (z + 14280) / 13 ^ (q : ℕ) % 13 - 6

/-- Row `i`'s tag: the balanced base-13 digits of `(⌊i/2⌋ + 1)·(−1)^i`. -/
def rowTag (i : ℕ) : Fin 4 → ℤ := bal13 (if i % 2 = 0 then (i / 2 + 1 : ℕ) else -((i / 2 + 1 : ℕ) : ℤ))

/-- Column `j`'s code `C[:, j]`: the balanced base-13 digits of `j + 1`. -/
def colCode (j : ℕ) : Fin 4 → ℤ := bal13 (j + 1 : ℕ)

/-- `-rot`'s row weights: `A′ = (127, 1, 1)` on even rows and `(1, 127, 1)` on odd rows. -/
def rotWeight (i : ℕ) (ℓ : Fin 3) : ℤ :=
  if (ℓ : ℕ) = i % 2 then 127 else 1

/-- The column bias `β_j = ⌊(Σ_q C[q, j] + 3(S − 1) + 1) / 2⌋`, a public constant a column. -/
def bias (S j : ℕ) : ℤ := ((∑ q : Fin 4, colCode j q) + 3 * ((S : ℤ) - 1) + 1) / 2

/-- **Step `t`'s coin term** in row `i`, column `j`. Step 0: `T(i)·G[:, j] + a₁[i]·C[:, j]`. Step `t ≥ 1`: the tag lane
`1 + b_t[i]` and the coin lanes `Σ_ℓ A′[i, ℓ] · s_t[⌊i/tile⌋, ℓ, j]`. -/
def coinTerm (L : Layout) (ω : Coins L) (i : Fin L.m) (t : Fin L.S) (j : Fin L.n) : ℤ :=
  if (t : ℕ) = 0 then
    (∑ q : Fin 4, rowTag i q * ω.G q j) + ∑ q : Fin 4, ((ω.2.1 i q : ℕ) : ℤ) * colCode j q
  else
    (1 + ((ω.1 i t : ℕ) : ℤ)) +
      ∑ ℓ : Fin 3, rotWeight i ℓ *
        (if h : (i : ℕ) / L.tile < L.tiles then ω.s t ⟨(i : ℕ) / L.tile, h⟩ ℓ j else 0)

/-- **The committed word** of row `i`, step `t`, column `j`: `−β_j + Σ_{τ ≤ t} (κ i τ j + coinTerm i τ j)`, with `κ`
the data's step dot products, fixed before the coins. -/
def word (L : Layout) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (ω : Coins L) (i : Fin L.m) (t : Fin L.S)
    (j : Fin L.n) : ℤ :=
  -bias L.S j + ∑ τ : Fin L.S, if τ ≤ t then κ i τ j + coinTerm L ω i τ j else 0

/-- Data the served layout can carry: each step's dot product is of at most 28 int8 lanes, `|κ| ≤ 28 · 127²`. -/
def DataInRange (L : Layout) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) : Prop :=
  ∀ i t j, -(28 * 127 ^ 2) ≤ κ i t j ∧ κ i t j ≤ 28 * 127 ^ 2

/-- k32 steps for inner dimension `K`: step 0 holds 24 data lanes and each later step 28, `1 + ⌈(K − 24)/28⌉`. -/
def stepsFor (K : ℕ) : ℕ := 1 + (K - 24 + 27) / 28

/-- **The lead** at `T` tokens and a call of `m × K × n`: 64-row coin tiles, 7-bit coins, 256 KiB words chunks, the
network term at cap 32, and 8 TiB of node memory (`2^34` tiles). -/
def lead (T m n K : ℕ) : Layout where
  T := T
  m := m
  n := n
  K := K
  S := stepsFor K
  tile := 64
  e := 7
  chunkWords := 65536
  cap := 32
  mem := 2 ^ 34

/-- The lead at cap 320. -/
def lead320 (T m n K : ℕ) : Layout := { lead T m n K with cap := 320 }

/-- The point T 512 × 64: 512 tokens a step, 64 rows a call. -/
def pt512x64 (n K : ℕ) : Layout := lead 512 64 n K

/-- The point T 512 × 512: 512 tokens a step, 512 rows a call. -/
def pt512x512 (n K : ℕ) : Layout := lead 512 512 n K

/-- The point T 8,192: 8,192 tokens a step and rows a call. -/
def pt8192 (n K : ℕ) : Layout := lead 8192 8192 n K

end Pouw.Lead
