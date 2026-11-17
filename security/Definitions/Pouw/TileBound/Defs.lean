module

public import Definitions.Pouw.PearlC.Fp4
public import Mathlib.Algebra.BigOperators.Group.Finset.Defs
public import Mathlib.Algebra.Field.Defs
public import Mathlib.Data.Fintype.Prod
public import Mathlib.Data.Rat.Defs
import Mathlib.Data.Nat.Totient
import Mathlib.Data.Sym.Sym2.Init
import Mathlib.Tactic.ContinuousFunctionalCalculus
import Mathlib.Tactic.NormNum.GCD

/-! The FP4-tile model, whose adversary's only fast multiplier is the m16n8k64 NVFP4 tile gate priced per slot, and whose faithfulness (`tc-model/sm120-fp4`, `fp4-tile-only/sm120`) is measured, not proved. -/

@[expose] public section

namespace Pouw.TileBound

open Finset Pouw.PearlC.Fp4 Pouw.Fp8Atom

/-- `fp32` is the scalar pipe (FFMA), `int8` the integer tile (`IMMA`), `fp16` the FP16 tile (BF16's price, FP16's 11
significand bits), `dp4a` the integer dot product on the generic cores. -/
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

/-- In FP8-MAC units at locked clocks; NVFP4's 2:4-sparse form is the same 0.5 per nonzero product. -/
def price : Fmt → ℚ
  | .fp4 => 1 / 2
  | .fp8 => 1
  | .bf16 => 2
  | .tf32 => 4
  | .fp32 => 846 / 100
  | .int8 => 1
  | .fp16 => 2
  | .dp4a => 405 / 100

/-- For full-width values: a sum of `b` scaled E2M1 values at one exponent needs `6 + ⌈log₂ b⌉` bits. Narrower values pack
more, which the measured merge rate `fp4-merge-rate/sm120` bounds, not this table. -/
def capacity : Fmt → ℕ
  | .fp4 => 1
  | .fp8 => 2
  | .bf16 => 4
  | .tf32 => 32
  | .fp32 => 2 ^ 18
  | .int8 => 10
  | .fp16 => 32
  | .dp4a => 10

structure QuadProg (K : Type*) (m k n : ℕ) (τ : Type*) where
  a : τ → Fin m × Fin k → K
  b : τ → Fin k × Fin n → K
  a' : τ → Fin m × Fin k → K
  b' : τ → Fin k × Fin n → K
  w : τ → Fin m × Fin n → K

structure BilinProg (K : Type*) (m k n : ℕ) (τ : Type*) where
  u : τ → Fin m × Fin k → K
  v : τ → Fin k × Fin n → K
  w : τ → Fin m × Fin n → K

variable {K : Type*} [Field K] {m k n : ℕ} {τ : Type*}

def BilinProg.toQuad (P : BilinProg K m k n τ) : QuadProg K m k n τ := ⟨P.u, 0, 0, P.v, P.w⟩

def matMul (X : Fin m × Fin k → K) (Y : Fin k × Fin n → K) (o : Fin m × Fin n) : K :=
  ∑ l, X (o.1, l) * Y (l, o.2)

def QuadProg.eval [Fintype τ] (P : QuadProg K m k n τ) (X : Fin m × Fin k → K) (Y : Fin k × Fin n → K)
    (o : Fin m × Fin n) : K :=
  ∑ t, P.w t o * ((∑ e, P.a t e * X e) + ∑ e, P.b t e * Y e) * ((∑ e, P.a' t e * X e) + ∑ e, P.b' t e * Y e)

def QuadProg.CrossOn [Fintype τ] (P : QuadProg K m k n τ) (liveX : Fin m × Fin k → Prop)
    (liveY : Fin k × Fin n → Prop) : Prop :=
  ∀ x, liveX x → ∀ y, liveY y → ∀ o : Fin m × Fin n,
    ∑ t, P.w t o * (P.a t x * P.b' t y + P.a' t x * P.b t y) =
      if x.1 = o.1 ∧ x.2 = y.1 ∧ y.2 = o.2 then 1 else 0

open Classical in
noncomputable def nLive (liveX : Fin m × Fin k → Prop) (liveY : Fin k × Fin n → Prop) : ℕ :=
  Fintype.card {x : Fin m × Fin k × Fin n // liveX (x.1, x.2.1) ∧ liveY (x.2.1, x.2.2)}

structure XBlocks (m k : ℕ) (β : Type*) where
  blk : Fin m × Fin k → β
  inRow : ∀ e e', blk e = blk e' → e.1 = e'.1

structure YBlocks (k n : ℕ) (β : Type*) where
  blk : Fin k × Fin n → β
  inCol : ∀ e e', blk e = blk e' → e.2 = e'.2

variable {β : Type*}

def XBlocks.Touches (B : XBlocks m k β) (f : Fin m × Fin k → K) (b : β) : Prop := ∃ e, B.blk e = b ∧ f e ≠ 0

def YBlocks.Touches (B : YBlocks k n β) (f : Fin k × Fin n → K) (b : β) : Prop := ∃ e, B.blk e = b ∧ f e ≠ 0

open Classical in
noncomputable def XBlocks.touches [Fintype β] (B : XBlocks m k β) (f : Fin m × Fin k → K) : ℕ :=
  #{b | B.Touches f b}

open Classical in
noncomputable def YBlocks.touches [Fintype β] (B : YBlocks k n β) (f : Fin k × Fin n → K) : ℕ :=
  #{b | B.Touches f b}

def blockStart (l : Fin k) : Fin k := ⟨l.val / 16 * 16, lt_of_le_of_lt (Nat.div_mul_le_self _ _) l.isLt⟩

def nvfp4X (m k : ℕ) : XBlocks m k (Fin m × Fin k) := ⟨fun e => (e.1, blockStart e.2), fun _ _ h => (Prod.mk.inj h).1⟩

def nvfp4Y (k n : ℕ) : YBlocks k n (Fin k × Fin n) := ⟨fun e => (blockStart e.1, e.2), fun _ _ h => (Prod.mk.inj h).2⟩

def LiveIn (B : XBlocks m k β) (liveX : Fin m × Fin k → Prop) (liveY : Fin k × Fin n → Prop) (b : β)
    (x : Fin m × Fin k × Fin n) : Prop :=
  B.blk (x.1, x.2.1) = b ∧ liveX (x.1, x.2.1) ∧ liveY (x.2.1, x.2.2)

/-- Side `false` is `ℓ`'s `a`, `true` is `ℓ'`'s `a'`. -/
def SideIn (P : QuadProg K m k n τ) (B : XBlocks m k β) (b : β) (q : τ × Bool) : Prop :=
  if q.2 then B.Touches (P.a' q.1) b else B.Touches (P.a q.1) b

def unitForm {ι : Type*} [DecidableEq ι] (e₀ : ι) : ι → K := fun e => if e = e₀ then 1 else 0

structure SlotProg (K : Type*) (m k n : ℕ) (τ : Type*) extends QuadProg K m k n τ where
  fmt : τ → Fmt

def SlotProg.cost [Fintype τ] (P : SlotProg K m k n τ) : ℚ := ∑ t, price (P.fmt t)

/-- For FP4 slots a combination of two input blocks lies on the E2M1 grid under one UE4M3 scale only on a thin set, which
the measured merge rate `fp4-merge-rate/sm120` bounds. -/
def SlotProg.Representable [Fintype β] {γ : Type*} [Fintype γ] (P : SlotProg K m k n τ) (BX : XBlocks m k β)
    (BY : YBlocks k n γ) : Prop :=
  ∀ t, BX.touches (P.a t) + BY.touches (P.b t) ≤ capacity (P.fmt t) ∧
    BX.touches (P.a' t) + BY.touches (P.b' t) ≤ capacity (P.fmt t)

def honest (K : Type*) [Field K] (m k n : ℕ) : SlotProg K m k n (Fin m × Fin k × Fin n) where
  a t := unitForm (t.1, t.2.1)
  b _ := 0
  a' _ := 0
  b' t := unitForm (t.2.1, t.2.2)
  w t := unitForm (t.1, t.2.2)
  fmt _ := .fp4

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

def scaleIdx (l : Fin 64) : Fin 4 := ⟨l.val / 16, by omega⟩

def Firing.word (g : Firing) (i : Fin 16) (j : Fin 8) : Option ℕ :=
  sm120Nvf4.step (g.acc i j) (List.ofFn (g.p i)) (List.ofFn fun l => g.q l j) (List.ofFn (g.ps i)) (List.ofFn (g.qs j))

/-- An invalid scale byte reads 0; the domain excludes it. -/
def elemValue (c : Nib) (s : Fin 256) : ℚ := decode c * (scaleValue .ue4m3 s).getD 0

/-- `none` for infinities and NaNs. -/
def wordValue (w : ℕ) : Option ℚ := (fp32Value w).finite?.map Term.value

def Firing.slotValue (g : Firing) (i : Fin 16) (l : Fin 64) (j : Fin 8) : ℚ :=
  elemValue (g.p i l) (g.ps i (scaleIdx l)) * elemValue (g.q l j) (g.qs j (scaleIdx l))

/-- It asks this of all 128 words, including words the program doesn't use. -/
def Firing.Exact (g : Firing) : Prop :=
  ∀ i j, (g.word i j).bind wordValue = some ((wordValue (g.acc i j)).getD 0 + ∑ l, g.slotValue i l j)

structure TileProg (K : Type*) (m k n G : ℕ) where
  u : Fin G → Fin 16 → Fin 64 → Fin m × Fin k → K
  v : Fin G → Fin 64 → Fin 8 → Fin k × Fin n → K
  w : Fin G → Fin 16 → Fin 8 → Fin m × Fin n → K

def TileProg.toBilin {G : ℕ} (P : TileProg K m k n G) : BilinProg K m k n (Fin G × Fin 16 × Fin 64 × Fin 8) :=
  ⟨fun s => P.u s.1 s.2.1 s.2.2.1, fun s => P.v s.1 s.2.2.1 s.2.2.2, fun s => P.w s.1 s.2.1 s.2.2.2⟩

def TileProg.cost {G : ℕ} (_ : TileProg K m k n G) : ℚ := G * gatePrice

def TileProg.Representable {G : ℕ} [Fintype β] (P : TileProg K m k n G) (BX : XBlocks m k β) : Prop :=
  ∀ g i l, BX.touches (P.u g i l) ≤ 1

/-- Pointwise by design: a code under a scale byte takes finitely many values, so a statement over every input would force
the forms to vanish. Every firing must be `Firing.Exact` on all 128 words, so one whose unused words overflow or aren't
exact is excluded. -/
def TileProg.RealizesAt {G : ℕ} (P : TileProg ℚ m k n G) (gs : Fin G → Firing) (X : Fin m × Fin k → ℚ)
    (Y : Fin k × Fin n → ℚ) : Prop :=
  ∀ g, (gs g).Exact ∧ (∀ i j, (gs g).acc i j = 0) ∧
    (∀ i l, elemValue ((gs g).p i l) ((gs g).ps i (scaleIdx l)) = ∑ e, P.u g i l e * X e) ∧
    (∀ l j, elemValue ((gs g).q l j) ((gs g).qs j (scaleIdx l)) = ∑ e, P.v g l j e * Y e)

def TileProg.outAt {G : ℕ} (P : TileProg ℚ m k n G) (gs : Fin G → Firing) (o : Fin m × Fin n) : ℚ :=
  ∑ g, ∑ i, ∑ j, P.w g i j o * (((gs g).word i j).bind wordValue).getD 0

end Pouw.TileBound
