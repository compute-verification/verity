import Definitions.Pouw.RelativeSecurity

/-!
# The filler's cell check (trusted)

What the filler's per-check luck is about, as `benchmarks/pouw/filler/pfill_ref.py` computes a tile and the service checks it:

* **The noise** (`Noise`): a round's A rows, `K` bits each, and its B columns. The tile's noise is SHA-512 of the
  round's live coins (`pfill_ref.digest`, one counter per row block and per column block), read here as uniform and
  independent per row and per column, as every PoUW luck lemma reads its coins.
* **The word** (`cellWord`): a cell's committed word `Σ_k x_k·c_k(y)`, A's row bits `x` against the column's entries
  `c y`, in the word ring `A`: the s32 word `ZMod (2 ^ 32)` (the tile's `& M32`) or `ℤ`. pfill-bits (`PF_BITS`,
  `bitsCol`): B's entries are the column's bits, the word `popcount(a & b)`. pfill-bU (`PF_BU`, `buCol`): B's entries
  are `uOf` of the column's digest bytes, on `{±1, …, ±64}`, the word `A·u`. The ±1 tiles (s8, e2m1) have constant
  parity, and nothing here is about them.
* **The products** (`Products`, `CellProducts`, `Short`, `shortShare`): a run's word for a cell with `R` two-sided
  products in the cell projection, every other row's and column's noise fixed (VI.2.1), short when `R < K`, and a
  round's share of short cells.
* **The parity** (`parity`, `ParityBalanced`): the column's entries mod 2 (`φ : A →+* ZMod 2`), uniform (Lemma P2's
  hypothesis on `π`).
* **The bound**: `luck d = ½ + 2^(−d/2−1)` at `d = K − R` products short.
* **The checks** (`CellRight`, `FullFill`, `SpotFill`, `FillBlocks`): a window whose every round checks every cell
  (𝒫_full), one uniform cell (the spot check), or one uniform cell beside another block's one-draw check on
  independent coins (Theorem S with each block at its own `p`, the budget `eps.py`'s cheater LP reads).
* **Pinned receipts** (`wrongShare`, `FullPinnedFill`, `SpotPinnedFill`, `PinnedFillBlocks`): a filler whose check
  recomputes each tile's receipt from the round's committed data and live coins and compares it with the committed one
  (lfill's tile receipt, `lf_ref`; a lane chain absorbed into the fold's receipt; keyed rounds in the fold), checked in
  full, at one uniform tile, or at one uniform tile beside another block. The round's share is the share of tiles whose
  committed receipt isn't the recomputed one.
-/

namespace Accounting.Filler

open Finset Pouw.RelativeSecurity

section Cell

variable {K : ℕ} {Y A : Type} [CommRing A]

/-- A bit's value in the word ring: `0` or `1`. -/
def bit (z : ZMod 2) : A := (z.val : A)

/-- pfill-bU's B entry from its digest byte `t` (`pfill_ref.u_of`, `(t & 0x7F) − 64 + ((t >> 6) & 1)`), on
`{±1, …, ±64}`. -/
def uOf (t : Fin 256) : ℤ := (((t : ℕ) % 128 : ℕ) : ℤ) - 64 + (((t : ℕ) / 64 % 2 : ℕ) : ℤ)

/-- **A cell's word**: A's row bits `x` against B's column, whose entries are `c y`: `Σ_k x_k·c_k(y)`. -/
def cellWord (c : Y → Fin K → A) (x : Fin K → ZMod 2) (y : Y) : A := ∑ k, bit (x k) * c y k

/-- pfill-bits' column (`PF_BITS`): B's entries are the column's bits, so the word is `popcount(a & b)`. -/
def bitsCol : (Fin K → ZMod 2) → Fin K → A := fun y k => bit (y k)

/-- pfill-bU's column (`PF_BU`): B's entries are `uOf` of the column's digest bytes, so the word is `A·u`. -/
def buCol : (Fin K → Fin 256) → Fin K → A := fun t k => ((uOf (t k) : ℤ) : A)

/-- The column's parity vector: its entries read mod 2. -/
def parity (φ : A →+* ZMod 2) (c : Y → Fin K → A) (y : Y) : Fin K → ZMod 2 := fun k => φ (c y k)

/-- **The column's parity is uniform** (Lemma P2's hypothesis on `π`): every parity vector has probability `2^(−K)`. -/
def ParityBalanced [Fintype Y] (φ : A →+* ZMod 2) (c : Y → Fin K → A) : Prop :=
  ∀ z : Fin K → ZMod 2, prob (fun y => parity φ c y = z) = 1 / 2 ^ K

/-- **A value with `R` two-sided products** (RS-E-REF r6 VI.2.1): `P(x, y) = Σ_{j<R} f_j(x)·g_j(y) + a(x) + b(y)` over
`A`, with `f_j` and `a` any functions of the row's bits and `g_j` and `b` any functions of the column's noise. -/
def Products (R : ℕ) (P : (Fin K → ZMod 2) → Y → A) : Prop :=
  ∃ (f : Fin R → (Fin K → ZMod 2) → A) (g : Fin R → Y → A) (a : (Fin K → ZMod 2) → A) (b : Y → A),
    ∀ x y, P x y = ∑ j, f j x * g j y + a x + b y

/-- **The filler's per-check luck** at `d` products short: `½ + 2^(−d/2−1)` (`0.8536` at `d = 1`, `¾` at `d = 2`). -/
noncomputable def luck (d : ℕ) : ℝ := 1 / 2 + (2 : ℝ) ^ (-(d : ℝ) / 2 - 1)

end Cell

section Tile

variable {K : ℕ} {ρ ν Y A : Type} [CommRing A]

/-- **A round's noise on a tile**: each A row's `K` bits and each B column's noise. -/
abbrev Noise (K : ℕ) (ρ ν Y : Type) : Type := (ρ → Fin K → ZMod 2) × (ν → Y)

variable [DecidableEq ρ] [DecidableEq ν]

/-- **Cell `(r, s)` takes `R` products in the cell projection**: at every setting of the other rows' and columns'
noise, the run's word `W`, as a function of row `r`'s bits and column `s`'s noise, has `R` two-sided products. -/
def CellProducts (W : Noise K ρ ν Y → A) (r : ρ) (s : ν) (R : ℕ) : Prop :=
  ∀ ω : Noise K ρ ν Y, Products R fun x y => W (Function.update ω.1 r x, Function.update ω.2 s y)

/-- **A cell short of products**: its word takes fewer than `K` products in the cell projection. -/
def Short (W : Noise K ρ ν Y → A) (r : ρ) (s : ν) : Prop := ∃ R < K, CellProducts W r s R

/-- **A round's short share**: the share of the tile's cells short of products, for the run's words `W`. -/
noncomputable def shortShare [Fintype ρ] [Fintype ν] (W : ρ → ν → Noise K ρ ν Y → A) : ℝ :=
  prob fun rs : ρ × ν => Short (W rs.1 rs.2) rs.1 rs.2

/-- **A cell's check**: the run's word at cell `(r, s)` is the cell's word at the round's noise. -/
def CellRight (c : Y → Fin K → A) (W : ρ → ν → Noise K ρ ν Y → A) (ω : Noise K ρ ν Y) (r : ρ) (s : ν) : Prop :=
  W r s ω = cellWord c (ω.1 r) (ω.2 s)

end Tile

section Checks

variable {K : ℕ} {ρ ν Y A : Type} [CommRing A] [Fintype ρ] [Fintype ν] [DecidableEq ρ] [DecidableEq ν]

/-- **A full-checking filler window** (𝒫_full): each round's words `W` are fixed before the round's noise, the
round's share is their short share, and it passes only where every cell's check does. -/
def FullFill (c : Y → Fin K → A) : {n : ℕ} → RoundGame (Noise K ρ ν Y) n → Prop
  | _, .done => True
  | _, .round d pass next =>
    (∃ W : ρ → ν → Noise K ρ ν Y → A, (∀ ω, d ω = shortShare W) ∧ ∀ ω, pass ω → ∀ r s, CellRight c W ω r s) ∧
      ∀ ω, FullFill c (next ω)

/-- **A spot-checked filler window**: each round's coins are the round's noise and the verifier's uniform cell; the
round's share is the short share of the words `W`, fixed before them, and it passes only where the drawn cell's
check does. -/
def SpotFill (c : Y → Fin K → A) : {n : ℕ} → RoundGame (Noise K ρ ν Y × (ρ × ν)) n → Prop
  | _, .done => True
  | _, .round d pass next =>
    (∃ W : ρ → ν → Noise K ρ ν Y → A, (∀ a, d a = shortShare W) ∧ ∀ a, pass a → CellRight c W a.1 a.2.1 a.2.2) ∧
      ∀ a, SpotFill c (next a)

/-- **A window of two blocks** (Theorem S with each block at its own `p`): each round's coins are another block's
(`Ω₀`, the words' spot check) and the filler's spot check's, independent. The other block's share `d₀ ≥ 0` is fixed
before its coins and its check passes with probability at most `1 − (1 − p₀)·d₀`; the filler's words `W` are fixed
before the round's noise. The round's share is `(1 − p₀)·d₀ + (1 − 0.854)·shortShare W`, `eps.py`'s budget row, and
the round passes only where both blocks' checks do. -/
def FillBlocks {Ω₀ : Type} [Fintype Ω₀] (p₀ : ℝ) (c : Y → Fin K → A) :
    {n : ℕ} → RoundGame (Ω₀ × (Noise K ρ ν Y × (ρ × ν))) n → Prop
  | _, .done => True
  | _, .round d pass next =>
    (∃ (d₀ : ℝ) (pass₀ : Ω₀ → Prop) (W : ρ → ν → Noise K ρ ν Y → A), 0 ≤ d₀ ∧
      prob pass₀ ≤ 1 - (1 - p₀) * d₀ ∧ (∀ a, d a = (1 - p₀) * d₀ + (1 - 427 / 500) * shortShare W) ∧
      ∀ a, pass a → pass₀ a.1 ∧ CellRight c W a.2.1 a.2.2.1 a.2.2.2) ∧
      ∀ a, FillBlocks p₀ c (next a)

end Checks

section Pinned

variable {D C τ V : Type} [Fintype τ]

/-- **A round's wrong share**: the share of tiles whose committed receipt `R t` isn't the tile's receipt `F w c t`, a
function of the round's committed data `w` and live coins `c` that the verifier recomputes. -/
noncomputable def wrongShare (F : D → C → τ → V) (w : D) (c : C) (R : τ → V) : ℝ := prob fun t => R t ≠ F w c t

/-- **A full-checking window of pinned receipts**: each round's committed data `w` and receipts `R` are any functions of
its live coins, the round's share is their wrong share, and it passes only where every tile's committed receipt is the
recomputed one. -/
def FullPinnedFill (F : D → C → τ → V) : {n : ℕ} → RoundGame C n → Prop
  | _, .done => True
  | _, .round d pass next =>
    (∃ (w : C → D) (R : C → τ → V), (∀ c, d c = wrongShare F (w c) c (R c)) ∧
        ∀ c, pass c → ∀ t, R c t = F (w c) c t) ∧
      ∀ c, FullPinnedFill F (next c)

/-- **A spot-checked window of pinned receipts**: each round's coins are its live coins and the verifier's uniform tile,
drawn after the receipts are committed; the committed data and receipts are any functions of the live coins, the
round's share is their wrong share, and it passes only where the drawn tile's committed receipt is the recomputed
one. -/
def SpotPinnedFill (F : D → C → τ → V) : {n : ℕ} → RoundGame (C × τ) n → Prop
  | _, .done => True
  | _, .round d pass next =>
    (∃ (w : C → D) (R : C → τ → V), (∀ a, d a = wrongShare F (w a.1) a.1 (R a.1)) ∧
        ∀ a, pass a → R a.1 a.2 = F (w a.1) a.1 a.2) ∧
      ∀ a, SpotPinnedFill F (next a)

/-- **A window of a block beside pinned receipts** (Theorem S with the filler at `p = 0`): each round's coins are
another block's (`Ω₀`) and the pinned receipts' spot check's, independent. The other block's share `d₀ ≥ 0` is fixed
before its coins and its check passes with probability at most `1 − (1 − p₀)·d₀`; the committed data and receipts are
any functions of the coins but the tile. The round's share is `(1 − p₀)·d₀ + wrongShare`, `eps.py`'s budget row at the
filler's `p = 0`, and the round passes only where both blocks' checks do. -/
def PinnedFillBlocks {Ω₀ : Type} [Fintype Ω₀] (p₀ : ℝ) (F : D → C → τ → V) :
    {n : ℕ} → RoundGame (Ω₀ × (C × τ)) n → Prop
  | _, .done => True
  | _, .round d pass next =>
    (∃ (d₀ : ℝ) (pass₀ : Ω₀ → Prop) (w : Ω₀ → C → D) (R : Ω₀ → C → τ → V), 0 ≤ d₀ ∧
      prob pass₀ ≤ 1 - (1 - p₀) * d₀ ∧
      (∀ a, d a = (1 - p₀) * d₀ + wrongShare F (w a.1 a.2.1) a.2.1 (R a.1 a.2.1)) ∧
      ∀ a, pass a → pass₀ a.1 ∧ R a.1 a.2.1 a.2.2 = F (w a.1 a.2.1) a.2.1 a.2.2) ∧
      ∀ a, PinnedFillBlocks p₀ F (next a)

end Pinned

end Accounting.Filler
