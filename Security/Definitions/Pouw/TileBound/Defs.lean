import Mathlib.LinearAlgebra.Matrix.Rank
import Definitions.Pouw.PearlC.Fp4

/-!
# The FP4-tile model (trusted definitions)

`docs/pouw/fp4-tile-model.md`. The adversary's only fast multiplier is the tile gate of
`BLACKWELL_SM120_NVF4` (`Pouw.PearlC.Fp4.sm120Nvf4`: `mma.sync ... kind::mxf4nvf4 ... m16n8k64`), priced per slot.
Every other multiplier runs at its measured price, and elementwise work and data movement are bounded. Outputs are
bit-exact. A cheaper computation of an exact matrix product in this model is a program in the rewrite class below.

* **A firing of the tile gate** (`Firing`): 16 × 64 codes and scales for `P`, 64 × 8 for `Q`, and 16 × 8 accumulator
  words. Output `(i, j)` is one firing of `sm120Nvf4.step` on row `i` of `P` and column `j` of `Q`. It has
  `gateSlots = 8192` slots `(i, l, j)`, the product `P[i, l]·Q[l, j]`. The k-index `l` is shared by all 16 rows and all
  8 columns.
  - **Per-slot pricing:** `price .fp4 = 1/2` FP8-MAC units per slot, so a firing costs 4,096 units, dense or not.
  - **The exact regime** (`Firing.Exact`): every output word's value is the accumulator plus the exact sum of its 64 slot
    values. That is the atom's behaviour on in-domain operands, where NVFP4 chains are exact. Its faithfulness is
    `tc-model/sm120-fp4`, measured and never proved.
* **The rewrite class.**
  - A *quadratic program* (`QuadProg`) is a list of products of two linear forms of the inputs, `ℓ t = a t·X + b t·Y`
    and `ℓ' t = a' t·X + b' t·Y`, combined linearly into the outputs: `eval X Y o = Σ_t w t o · ℓ t · ℓ' t`.
  - A *bilinear program* (`BilinProg`) has `ℓ` on `X` only and `ℓ'` on `Y` only. Strassen and Winograd's matrix
    algorithms are bilinear; Winograd's inner-product trick and the square trick are quadratic.
  - Linear pre- and post-processing are free here, which is generous to the adversary: the model bounds elementwise
    work, it doesn't make it free.
  - Correctness is on the **live** entries (`CrossOn`). A branch that knows some entries are zero is only asked for the
    products of live entries. That is the debit: dead blocks, zero blocks and 2:4 structure are removed by the domain and
    the replay, and the bound is on the live count `nLive`.
* **Blocks.** An X-block is a set of X-entries in one row (`XBlocks`), a Y-block a set of Y-entries in one column
  (`YBlocks`). NVFP4's blocks are 16 consecutive k-positions under one UE4M3 scale (`nvfp4X`, `nvfp4Y`).
  `touches f` counts the blocks the linear form `f` reads.
* **Slot programs** (`SlotProg`): each product is one slot of a tile of some format `Fmt`, at `price`. The
  *representability rule* (`SlotProg.Representable`) says each input of a slot is one exactly representable value of
  its format, so it reads at most `capacity` blocks (§2.3 of the model page). The tile-only claim itself,
  `fp4-tile-only/sm120`, is that this list of formats and prices is complete; it is hardware faithfulness, measured, and
  is not a Lean statement.
* **Tile programs** (`TileProg`): FP4 programs built from gate firings. Their products are the gates' slots, so each
  one's `P`-input reads one X-block and its `Q`-input, shared across rows, one Y-block. `RealizesAt` says which firings
  carry the program at one input, and `outAt` is what they output.
* **Statement content kept here, not in the proofs:** the per-block index sets `LiveIn` and `SideIn`, the unit form
  `unitForm`, and the honest program `honest`.
-/

namespace Pouw.TileBound

open Finset Pouw.PearlC.Fp4 Pouw.Fp8Atom

/-! ## Formats, per-slot prices and capacities -/

/-- The multiplier formats of the RTX PRO 6000 (sm_120) that the model prices. `fp32` is the scalar pipe (FFMA), `int8`
the integer tile (`IMMA`), `fp16` the FP16 tile (BF16's price, FP16's 11 significand bits), `dp4a` the integer dot
product on the generic cores. -/
inductive Fmt where
  | fp4
  | fp8
  | bf16
  | tf32
  | fp32
  | int8
  | fp16
  | dp4a
  deriving DecidableEq, Repr

/-- The price of one slot (one product) in FP8-MAC units, measured at locked clocks. NVFP4 0.5 (the 2:4-sparse form is
0.25 per nominal MAC, so the same 0.5 per nonzero product), FP8 1, int8 1, BF16 2, FP16 2, TF32 4, `dp4a` 4.05 per int8
MAC, FFMA 8.46 (the model's value). GPU 4 has since measured the FP32 and F16 CUDA-core multipliers at 8.0–8.15, that is
16.0–16.3 FP4 slots per product (F1). Every pin that reads `price .fp32` holds across that range; FFMA's 16-block
closure is a tie there, not a margin: equality at 8.0, 2% at 8.15 (`fp32_closed_upto`). -/
def price : Fmt → ℚ
  | .fp4 => 1 / 2
  | .fp8 => 1
  | .bf16 => 2
  | .tf32 => 4
  | .fp32 => 846 / 100
  | .int8 => 1
  | .fp16 => 2
  | .dp4a => 405 / 100

/-- The capacity of a format: how many input blocks one exactly representable value of it can combine. A scaled E2M1
value has at most 6 significant bits (2 from the code, 4 from the UE4M3 mantissa), and a sum of `b` of them at one
exponent needs `6 + ⌈log₂ b⌉`. So FP4 (one value, one scale) holds 1 block, E4M3 holds 2 blocks of codes under one
equal scale, BF16 (8 bits) 4, TF32 and FP16 (11 bits) 32, and FP32 (24 bits) 2¹⁸. The integer formats hold codes under
one equal scale, as twice the code (at most 12): a sum of `b` needs `12·b ≤ 127`, so int8 and `dp4a`'s int8 operands
hold 10. These are the model page's §2.3 table: full-width values. Narrower values pack more; that is the measured
merge rate `fp4-merge-rate/sm120`, not this table. -/
def capacity : Fmt → ℕ
  | .fp4 => 1
  | .fp8 => 2
  | .bf16 => 4
  | .tf32 => 32
  | .fp32 => 2 ^ 18
  | .int8 => 10
  | .fp16 => 32
  | .dp4a => 10

/-! ## Programs in the rewrite class -/

/-- A quadratic program over `K` for `X·Y`, `X : m × k`, `Y : k × n`, with products indexed by `τ`: product `t` is
`(a t·X + b t·Y)·(a' t·X + b' t·Y)`, and it enters output `o` with coefficient `w t o`. -/
structure QuadProg (K : Type*) (m k n : ℕ) (τ : Type*) where
  a : τ → Fin m × Fin k → K
  b : τ → Fin k × Fin n → K
  a' : τ → Fin m × Fin k → K
  b' : τ → Fin k × Fin n → K
  w : τ → Fin m × Fin n → K

/-- A bilinear program: product `t` is `(u t·X)·(v t·Y)`. -/
structure BilinProg (K : Type*) (m k n : ℕ) (τ : Type*) where
  u : τ → Fin m × Fin k → K
  v : τ → Fin k × Fin n → K
  w : τ → Fin m × Fin n → K

variable {K : Type*} [Field K] {m k n : ℕ} {τ : Type*}

/-- A bilinear program as a quadratic one: `ℓ = u·X`, `ℓ' = v·Y`. -/
def BilinProg.toQuad (P : BilinProg K m k n τ) : QuadProg K m k n τ := ⟨P.u, 0, 0, P.v, P.w⟩

/-- The exact matrix product, `X` and `Y` as functions on their entries. -/
def matMul (X : Fin m × Fin k → K) (Y : Fin k × Fin n → K) (o : Fin m × Fin n) : K :=
  ∑ l, X (o.1, l) * Y (l, o.2)

/-- A quadratic program's output on inputs `X`, `Y`. -/
def QuadProg.eval [Fintype τ] (P : QuadProg K m k n τ) (X : Fin m × Fin k → K) (Y : Fin k × Fin n → K)
    (o : Fin m × Fin n) : K :=
  ∑ t, P.w t o * ((∑ e, P.a t e * X e) + ∑ e, P.b t e * Y e) * ((∑ e, P.a' t e * X e) + ∑ e, P.b' t e * Y e)

/-- **Correct on the live entries.** For every live X-entry `x`, live Y-entry `y` and output `o`, the coefficient of
`X x · Y y` in output `o` is that of `X·Y`: 1 if `x = (o.1, l)` and `y = (l, o.2)`, else 0. It follows from `eval = X·Y`
on every input whose dead entries are zero (`crossOn_of_eval`). -/
def QuadProg.CrossOn [Fintype τ] (P : QuadProg K m k n τ) (liveX : Fin m × Fin k → Prop)
    (liveY : Fin k × Fin n → Prop) : Prop :=
  ∀ x, liveX x → ∀ y, liveY y → ∀ o : Fin m × Fin n,
    ∑ t, P.w t o * (P.a t x * P.b' t y + P.a' t x * P.b t y) =
      if x.1 = o.1 ∧ x.2 = y.1 ∧ y.2 = o.2 then 1 else 0

open Classical in
/-- The live products: triples `(i, l, j)` with `X (i, l)` and `Y (l, j)` both live. With every entry live it is
`m·k·n` (`nLive_all`). -/
noncomputable def nLive (liveX : Fin m × Fin k → Prop) (liveY : Fin k × Fin n → Prop) : ℕ :=
  Fintype.card {x : Fin m × Fin k × Fin n // liveX (x.1, x.2.1) ∧ liveY (x.2.1, x.2.2)}

/-! ## Blocks -/

/-- A partition of X's entries into blocks, each inside one row. -/
structure XBlocks (m k : ℕ) (β : Type*) where
  blk : Fin m × Fin k → β
  inRow : ∀ e e', blk e = blk e' → e.1 = e'.1

/-- A partition of Y's entries into blocks, each inside one column. -/
structure YBlocks (k n : ℕ) (β : Type*) where
  blk : Fin k × Fin n → β
  inCol : ∀ e e', blk e = blk e' → e.2 = e'.2

variable {β : Type*}

/-- The linear form `f` reads block `b`. -/
def XBlocks.Touches (B : XBlocks m k β) (f : Fin m × Fin k → K) (b : β) : Prop := ∃ e, B.blk e = b ∧ f e ≠ 0

/-- The linear form `f` reads block `b`. -/
def YBlocks.Touches (B : YBlocks k n β) (f : Fin k × Fin n → K) (b : β) : Prop := ∃ e, B.blk e = b ∧ f e ≠ 0

open Classical in
/-- The number of X-blocks the linear form `f` reads. -/
noncomputable def XBlocks.touches [Fintype β] (B : XBlocks m k β) (f : Fin m × Fin k → K) : ℕ :=
  #{b | B.Touches f b}

open Classical in
/-- The number of Y-blocks the linear form `f` reads. -/
noncomputable def YBlocks.touches [Fintype β] (B : YBlocks k n β) (f : Fin k × Fin n → K) : ℕ :=
  #{b | B.Touches f b}

/-- The first k-position of `l`'s block of 16. -/
def blockStart (l : Fin k) : Fin k := ⟨l.val / 16 * 16, lt_of_le_of_lt (Nat.div_mul_le_self _ _) l.isLt⟩

/-- NVFP4's X-blocks: row `i`, 16 consecutive k-positions (one UE4M3 scale), named by the row and the block's first
position. -/
def nvfp4X (m k : ℕ) : XBlocks m k (Fin m × Fin k) := ⟨fun e => (e.1, blockStart e.2), fun _ _ h => (Prod.mk.inj h).1⟩

/-- NVFP4's Y-blocks: column `j`, 16 consecutive k-positions. -/
def nvfp4Y (k n : ℕ) : YBlocks k n (Fin k × Fin n) := ⟨fun e => (blockStart e.1, e.2), fun _ _ h => (Prod.mk.inj h).2⟩

/-! ## The per-block index sets and the honest program -/

/-- The live products whose X-entry lies in block `b`. -/
def LiveIn (B : XBlocks m k β) (liveX : Fin m × Fin k → Prop) (liveY : Fin k × Fin n → Prop) (b : β)
    (x : Fin m × Fin k × Fin n) : Prop :=
  B.blk (x.1, x.2.1) = b ∧ liveX (x.1, x.2.1) ∧ liveY (x.2.1, x.2.2)

/-- The (product, side) pairs whose X-part reads block `b`: side `false` is `ℓ`'s `a`, `true` is `ℓ'`'s `a'`. -/
def SideIn (P : QuadProg K m k n τ) (B : XBlocks m k β) (b : β) (q : τ × Bool) : Prop :=
  if q.2 then B.Touches (P.a' q.1) b else B.Touches (P.a q.1) b

/-- The unit linear form on entry `e₀`, and the unit matrix on it. -/
def unitForm {ι : Type*} [DecidableEq ι] (e₀ : ι) : ι → K := fun e => if e = e₀ then 1 else 0

/-! ## Slot programs: per-slot pricing -/

/-- A quadratic program whose products are slots of tiles: product `t` runs in format `fmt t`. -/
structure SlotProg (K : Type*) (m k n : ℕ) (τ : Type*) extends QuadProg K m k n τ where
  fmt : τ → Fmt

/-- A slot program's cost: its slots at their formats' prices. -/
def SlotProg.cost [Fintype τ] (P : SlotProg K m k n τ) : ℚ := ∑ t, price (P.fmt t)

/-- **The representability rule.** Each input of each slot is one exactly representable value of the slot's format,
so it reads at most `capacity` blocks, X-blocks and Y-blocks together. For FP4 slots this is Lemma 1 of the model
page: a combination of two input blocks lies on the E2M1 grid under one UE4M3 scale only on a thin set, which the
measured merge rate `fp4-merge-rate/sm120` bounds. -/
def SlotProg.Representable [Fintype β] {γ : Type*} [Fintype γ] (P : SlotProg K m k n τ) (BX : XBlocks m k β)
    (BY : YBlocks k n γ) : Prop :=
  ∀ t, BX.touches (P.a t) + BY.touches (P.b t) ≤ capacity (P.fmt t) ∧
    BX.touches (P.a' t) + BY.touches (P.b' t) ≤ capacity (P.fmt t)

/-- The honest dense program: one FP4 slot per triple `(i, l, j)`, `X (i, l)·Y (l, j)` into output `(i, j)`. -/
def honest (K : Type*) [Field K] (m k n : ℕ) : SlotProg K m k n (Fin m × Fin k × Fin n) where
  a t := unitForm (t.1, t.2.1)
  b _ := 0
  a' _ := 0
  b' t := unitForm (t.2.1, t.2.2)
  w t := unitForm (t.1, t.2.2)
  fmt _ := .fp4

/-! ## The tile gate of `BLACKWELL_SM120_NVF4` -/

/-- Slots per firing of the m16n8k64 gate. -/
def gateSlots : ℕ := 16 * 64 * 8

/-- A firing's price: every slot, whether its product is zero or not. -/
def gatePrice : ℚ := gateSlots * price .fp4

/-- One firing of the m16n8k64 tile gate: `P` (16 × 64 codes, 4 UE4M3 scale bytes per row), `Q` (64 × 8 codes, 4
scale bytes per column) and the 16 × 8 accumulator words. -/
structure Firing where
  p : Fin 16 → Fin 64 → Nib
  ps : Fin 16 → Fin 4 → Fin 256
  q : Fin 64 → Fin 8 → Nib
  qs : Fin 8 → Fin 4 → Fin 256
  acc : Fin 16 → Fin 8 → ℕ

/-- The UE4M3 scale byte of k-position `l`'s block of 16. -/
def scaleIdx (l : Fin 64) : Fin 4 := ⟨l.val / 16, by omega⟩

/-- Output word `(i, j)`: one firing of `sm120Nvf4.step`, row `i` of `P` against column `j` of `Q`. -/
def Firing.word (g : Firing) (i : Fin 16) (j : Fin 8) : Option ℕ :=
  sm120Nvf4.step (g.acc i j) (List.ofFn (g.p i)) (List.ofFn fun l => g.q l j) (List.ofFn (g.ps i)) (List.ofFn (g.qs j))

/-- A code's exact value under its block's UE4M3 scale (an invalid scale byte reads 0; the domain excludes it). -/
def elemValue (c : Nib) (s : Fin 256) : ℚ := decode c * (scaleValue .ue4m3 s).getD 0

/-- An FP32 word's exact value; `none` for infinities and NaNs. -/
def wordValue (w : ℕ) : Option ℚ := (fp32Value w).finite?.map Term.value

/-- Slot `(i, l, j)`'s value: `P[i, l]·Q[l, j]` under both blocks' scales. -/
def Firing.slotValue (g : Firing) (i : Fin 16) (l : Fin 64) (j : Fin 8) : ℚ :=
  elemValue (g.p i l) (g.ps i (scaleIdx l)) * elemValue (g.q l j) (g.qs j (scaleIdx l))

/-- **The exact regime:** every one of the firing's 128 output words is the accumulator plus the exact sum of its 64
slot values. It asks this of all 128 words, including words the program doesn't use. -/
def Firing.Exact (g : Firing) : Prop :=
  ∀ i j, (g.word i j).bind wordValue = some ((wordValue (g.acc i j)).getD 0 + ∑ l, g.slotValue i l j)

/-! ## Tile programs: FP4 programs built from gate firings -/

/-- A tile program with `G` firings. Firing `g` takes the linear form `u g i l` of `X` into `P[i, l]` and `v g l j` of `Y`
into `Q[l, j]`. Its output `(i, j)` enters output `o` with coefficient `w g i j o`. -/
structure TileProg (K : Type*) (m k n G : ℕ) where
  u : Fin G → Fin 16 → Fin 64 → Fin m × Fin k → K
  v : Fin G → Fin 64 → Fin 8 → Fin k × Fin n → K
  w : Fin G → Fin 16 → Fin 8 → Fin m × Fin n → K

/-- A tile program's slots as a bilinear program: slot `(g, i, l, j)` is `(u g i l·X)·(v g l j·Y)`, into `w g i j`. -/
def TileProg.toBilin {G : ℕ} (P : TileProg K m k n G) : BilinProg K m k n (Fin G × Fin 16 × Fin 64 × Fin 8) :=
  ⟨fun s => P.u s.1 s.2.1 s.2.2.1, fun s => P.v s.1 s.2.2.1 s.2.2.2, fun s => P.w s.1 s.2.1 s.2.2.2⟩

/-- A tile program's cost: its firings at `gatePrice`. -/
def TileProg.cost {G : ℕ} (_ : TileProg K m k n G) : ℚ := G * gatePrice

/-- The representability rule for FP4 tiles: each `P`-input reads at most one X-block. -/
def TileProg.Representable {G : ℕ} [Fintype β] (P : TileProg K m k n G) (BX : XBlocks m k β) : Prop :=
  ∀ g i l, BX.touches (P.u g i l) ≤ 1

/-- **Realization at one input.** The firings `gs` realize the tile program at the inputs `X`, `Y`: each firing is
exact, its accumulators start at `+0`, and its slot inputs carry the program's linear forms at that input,
`elemValue P[i, l] = u g i l·X` and `elemValue Q[l, j] = v g l j·Y`. It is pointwise by design: a code under a scale
byte takes finitely many values, so only finitely many inputs can be carried, and a statement over every input would
force the forms to vanish.

Two consequences of this form:
* **Unused words.** Every firing must be `Firing.Exact` on all 128 words, including those no `w g i j o` reads. A firing
  whose unused words overflow or aren't exact is therefore excluded, though its used words still behave bilinearly.
* **Starting at `+0` loses nothing.** An exact firing adds its incoming accumulator exactly. So a program that chains
  firings through the accumulator has the same output, at the same cost, as the same firings started at `+0` with
  their partial sums added in `outAt`'s linear combination. -/
def TileProg.RealizesAt {G : ℕ} (P : TileProg ℚ m k n G) (gs : Fin G → Firing) (X : Fin m × Fin k → ℚ)
    (Y : Fin k × Fin n → ℚ) : Prop :=
  ∀ g, (gs g).Exact ∧ (∀ i j, (gs g).acc i j = 0) ∧
    (∀ i l, elemValue ((gs g).p i l) ((gs g).ps i (scaleIdx l)) = ∑ e, P.u g i l e * X e) ∧
    (∀ l j, elemValue ((gs g).q l j) ((gs g).qs j (scaleIdx l)) = ∑ e, P.v g l j e * Y e)

/-- A tile program's output from the firings `gs`: their words, decoded exactly and combined linearly. -/
def TileProg.outAt {G : ℕ} (P : TileProg ℚ m k n G) (gs : Fin G → Firing) (o : Fin m × Fin n) : ℚ :=
  ∑ g, ∑ i, ∑ j, P.w g i j o * (((gs g).word i j).bind wordValue).getD 0

end Pouw.TileBound
