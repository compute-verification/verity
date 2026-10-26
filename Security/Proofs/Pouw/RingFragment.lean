import Mathlib
import Definitions.Pouw.RelativeSecurity
import Proofs.Pouw.RelativeSecurity

/-!
# Theorem K's count for the ring fragment (statement; the proof waits on its red team)

BET-K-DISCHARGE (CMP-123, ACCEL): every program of the ring fragment pays at least Theorem K's count for Dither's
transcript, with `theoremK` reused as it stands. This module states that count (`RingKCount`) and its cost form
(`RingKCost`, from the count by `ringKCost_of_count`), and gives the honest naive kernel as its non-vacuity witness
(`HonestWitness`, rule I). It is the base O-CHARGE (CMP-111) builds its Lean on. MODEL-SPEC's machine
(`Verity.Accel.Machine`, CMP-109) is a run-level model of the card that isn't on `main` yet, and it has no
straight-line layer, so the fragment is stated here on Theorem K's own programs (`SlpG`). The map from a run of
MODEL-SPEC's machine on ring-fragment opcodes to an `SlpG` with a `Flow` is O-CHARGE's.

**The fragment.** A stage's run is an `SlpG` over a ring `R` with a map `φ : R →+* F₂` (`Z/2³²` for s32 wrap, with
`toF2`). Its product slots are MMA products or an `IMAD`'s `a·b` (`AffOps`: two bit-affine operands, mixed included),
and its linear slots are accumulators (MMA's `C + Σ`, `IMAD`'s `+ c`, `IADD3`, copies) over `Z/2³²`. P2 is the `Flow`:

* every slot has at most one free reader (`Flow.free`), a later linear slot;
* a linear slot has at most one linear slot among its free inputs: the accumulator it updates in place;
* every other nonzero read of a slot by a linear slot is a delivery (`deliveries`), priced `d` a word. That covers a
  product a second accumulator reads, an accumulator read by anything but its in-place successor (a merge, an
  `IADD3`'s second and third inputs, a copy out), and a second free reader, which the flow can't give.

**The count** (`RingKCount`). If the plane `p₀`'s weight maps to 1 and every word the check reads is an accumulator,
the certified cell-blocks (`certified`: the committed prefix word right at every bit point with at most two ones)
split into `a` blocks that each have `L` products of their own and `b` blocks that each have a delivery of their own:
`a + b = #certified`, `L a ≤ #products`, `b ≤ #deliveries`. With products at `κ` and deliveries at `d`
(`RingKCost`): `min(κ L, d) · #certified ≤ κ · #products + d · #deliveries`. No row purity: the delivery charge
replaces it.

**What it discharges.** For degree-2 ring programs, `KScopeAffine`'s third conjunct (row purity), and
`ChargeVF3j16`'s cost line in this form: a machine that charges `κ` a product and `d` a delivery charges at least
`min(κ L, d)` a certified cell-block, so at `κ_eff = min(κ, d/L)` a product of the floor. `KScopeAffine`'s first
conjunct (`AffProd` products) is the fragment's degree-2 restriction and stays. `KScopeOneSided` isn't discharged: the
parity luck (0.854) reads uncertified words that are short, not exact, so the difference argument below doesn't reach
them.

**What is excluded**, each a hypothesis of `RingKCount`:

* **Products of products** (degree above 2; `AffOps`). With operand factors that are themselves products, the count
  is false at level 2: the index trick, `A · S_b · Π_k (1 − (X_k − Y_k)²)` with `A = Σ_p w_p β_p(i, ·)`, `S_b` the
  block's coin sum and `X_k`, `Y_k` index-bit sums, agrees with the target at every bit point with at most two ones
  with `2⌈log₂ L⌉ + 1` products (11 at `L = 32`), and RS-E-REF's operand factor `Π_u (1 − z_u)` (VIII) is the same
  warning. Such a word is wrong at uniform points, so for degree above 2 the question is luck (`SlpG.cubeQuad_err`
  needs degree-2 errors), not this count.
* **Free fan-out of an accumulator.** An MMA that reads its `C` out of place and leaves `C` live (sm_120's
  `mma.sync`, MODEL-SPEC's `D = C + A·B` with `D ≠ C`) gives one word two free readers. `Flow.WF` refuses that, so
  such a program is in the fragment only if the second read is charged as a delivery. H200's `wgmma` accumulates in
  place (no `C` operand) and is in it.
* **Non-ring instructions** (O-BITS): `IMAD.HI`, `IMAD.WIDE`'s high word, shifts, `LOP3`, `PRMT`, compares, selects
  and the float formats.
* **Operands are free.** A product's bit-affine operands are given, not computed by slots, so the loads and adds that
  form them aren't charged, which only helps the program.

**The price obligation** (MODEL-SPEC and O-CHARGE), for a run's charge to be at least `κ · #products + d · #deliveries`
less its noise reads: an in-place MMA at least `κ` a product; `IMAD` on bit-affine `a`, `b` at least `κ + d` (its
product, and `c` when that is a delivery); `IADD3` at least `d` per input that isn't its in-place accumulator (so
`d ≤ price(IADD3)/2`, or `/3` when no input is in place); a copy of an accumulator word, and a shared-memory round
trip, at least `d` a word. The commitment's read of a word is outside the program and is no delivery.

**The proof plan** (after the red team). Distinct certified cell-blocks have distinct word slots (`theoremK` at no
products against `PrefixDiffRank`). For a certified `e`, walk back from its word slot along the in-place chain to the
nearest certified word slot `s'` (scale `c`, the chain's coefficients' product; `c = 0` if the chain starts first). If
no delivery lands on the walk, the word is `c` times `s'`'s plus the products attached to the walk plus a bit-affine
word, so `theoremK` with target `dPrefix e − c · dPrefix e'` and `PrefixDiffRank` give the walk at least `L` products.
Walks of distinct certified blocks are disjoint (the chains are paths), so their products and their deliveries are
too.
-/

namespace Pouw.RingFragment

open Pouw.RelativeSecurity Finset

section Flow

variable {V R : Type} [Fintype V] [CommRing R] {N : ℕ}

/-- **The accumulator flow** (P2 with in-place accumulation): each slot's one free reader, if any. -/
structure Flow (N : ℕ) where
  free : Fin N → Option (Fin N)

/-- A flow fits `P`: a free reader is a later linear slot, and a linear slot has at most one linear slot among its
free inputs (the accumulator it updates in place). Products are each read freely by at most one accumulator (P2). -/
def Flow.WF (P : SlpG V R N) (F : Flow N) : Prop :=
  (∀ k j, F.free k = some j → k < j ∧ P.isProd j = false) ∧
    ∀ j k k', F.free k = some j → F.free k' = some j → P.isProd k = false → P.isProd k' = false → k = k'

open Classical in
/-- **The deliveries**: the nonzero reads `(j, k)` of slot `k` by linear slot `j` that aren't `k`'s free read. -/
noncomputable def deliveries (P : SlpG V R N) (F : Flow N) : ℕ :=
  (univ.filter fun e : Σ j : Fin N, Fin j.val =>
    P.isProd e.1 = false ∧ P.lin e.1 e.2 ≠ 0 ∧ F.free ⟨e.2.val, e.2.isLt.trans e.1.isLt⟩ ≠ some e.1).card

/-- The program's products. -/
def products (P : SlpG V R N) : ℕ := (univ.filter fun j => P.isProd j = true).card

end Flow

section Ops

variable {Vl Vr R : Type} [Fintype Vl] [Fintype Vr] [DecidableEq Vl] [DecidableEq Vr] [CommRing R] {N : ℕ}

/-- **Degree 2**: every product multiplies two bit-affine operands (`AffProd`, mixed included). -/
def AffOps (P : SlpG (Vl ⊕ Vr) R N) : Prop := ∀ j, P.isProd j = true → AffProd (P.prod j)

end Ops

section Design

variable {R : Type} [CommRing R] {n L np B n' : ℕ}

/-- Cell-block `c` is **certified**: its committed prefix word is right at every bit point with at most two ones. For
an error of degree at most 2 that is right everywhere on the cube, and an uncertified block fails the check at luck ¾
(`SlpG.cubeQuad_err`). -/
def Certified (pc : Fin n → Fin B → Fin L → R) (vq : Fin B → Fin L → Fin n' → R) (w : Fin np → R) {N : ℕ}
    (P : SlpG (DBits n L np B n') R N) (word : DCells n n' B → Fin N) (c : DCells n n' B) : Prop :=
  ∀ U : Finset (DBits n L np B n'), U.card ≤ 2 →
    P.val (bitInd U) (word c) = dPrefix pc vq w c.1 c.2.1 c.2.2.val (bitInd U)

open Classical in
/-- The certified cell-blocks. -/
noncomputable def certified (pc : Fin n → Fin B → Fin L → R) (vq : Fin B → Fin L → Fin n' → R) (w : Fin np → R)
    {N : ℕ} (P : SlpG (DBits n L np B n') R N) (word : DCells n n' B → Fin N) : Finset (DCells n n' B) :=
  univ.filter fun c => Certified pc vq w P word c

end Design

/-! ## The statement -/

/-- **Theorem K's count for the ring fragment.** Over any ring with a map `φ` to `F₂` and plane weights whose plane
`p₀` maps to 1, a stage's degree-2 program (`AffOps`) with an accumulator flow (`Flow.WF`) whose check reads
accumulators (`word`) has its certified cell-blocks split into `a` with `L` products each and `b` with a delivery
each: `a + b = #certified`, `L a ≤ #products` and `b ≤ #deliveries`. Whatever the committed `pc`, `vq`, the sharing
across rows and columns, and the merges. -/
def RingKCount : Prop :=
  ∀ {R : Type} [CommRing R] (φ : R →+* ZMod 2) {n L np B n' : ℕ} (w : Fin np → R) (p₀ : Fin np), φ (w p₀) = 1 →
    ∀ (pc : Fin n → Fin B → Fin L → R) (vq : Fin B → Fin L → Fin n' → R) {N : ℕ}
      (P : SlpG (DBits n L np B n') R N) (F : Flow N), F.WF P → AffOps P →
    ∀ word : DCells n n' B → Fin N, (∀ c, P.isProd (word c) = false) →
      ∃ a b : ℕ, a + b = (certified pc vq w P word).card ∧ L * a ≤ products P ∧ b ≤ deliveries P F

/-- **The count's cost form**: products at `κ` and deliveries at `d` pay at least `min(κ L, d)` a certified
cell-block. -/
def RingKCost : Prop :=
  ∀ {R : Type} [CommRing R] (φ : R →+* ZMod 2) {n L np B n' : ℕ} (w : Fin np → R) (p₀ : Fin np), φ (w p₀) = 1 →
    ∀ (pc : Fin n → Fin B → Fin L → R) (vq : Fin B → Fin L → Fin n' → R) {N : ℕ}
      (P : SlpG (DBits n L np B n') R N) (F : Flow N), F.WF P → AffOps P →
    ∀ word : DCells n n' B → Fin N, (∀ c, P.isProd (word c) = false) → ∀ κ d : ℝ, 0 ≤ κ → 0 ≤ d →
      min (κ * L) d * ((certified pc vq w P word).card : ℝ) ≤ κ * (products P : ℝ) + d * (deliveries P F : ℝ)

theorem ringKCost_of_count (h : RingKCount) : RingKCost := by
  intro R _ φ n L np B n' w p₀ hw pc vq N P F hF hA word hword κ d hκ hd
  obtain ⟨a, b, hab, ha, hb⟩ := h φ w p₀ hw pc vq P F hF hA word hword
  have ha' : (L : ℝ) * a ≤ products P := by exact_mod_cast ha
  have hb' : (b : ℝ) ≤ deliveries P F := by exact_mod_cast hb
  have h1 : min (κ * L) d * a ≤ κ * L * a := mul_le_mul_of_nonneg_right (min_le_left _ _) (Nat.cast_nonneg a)
  have h2 : min (κ * L) d * b ≤ d * b := mul_le_mul_of_nonneg_right (min_le_right _ _) (Nat.cast_nonneg b)
  have h3 : κ * L * a ≤ κ * products P := by
    rw [mul_assoc]
    exact mul_le_mul_of_nonneg_left ha' hκ
  have h4 : d * b ≤ d * deliveries P F := mul_le_mul_of_nonneg_left hb' hd
  rw [← hab, Nat.cast_add, mul_add]
  linarith

/-- **The prefix-difference rank**, the proof's one new lemma: a cell-block's committed prefix word has `F₂` rank at
least `2L`, and so has its difference with any multiple of another cell-block's (the newest block of the two, or of
the one whose scale maps to 0, carries the identity of `rank_projF2_dIncr`). -/
def PrefixDiffRank : Prop :=
  ∀ {R : Type} [CommRing R] (φ : R →+* ZMod 2) {n L np B n' : ℕ} (w : Fin np → R) (p₀ : Fin np), φ (w p₀) = 1 →
    ∀ (pc : Fin n → Fin B → Fin L → R) (vq : Fin B → Fin L → Fin n' → R),
      (∀ e : DCells n n' B, 2 * L ≤ (projF2 φ (dPrefix pc vq w e.1 e.2.1 e.2.2.val)).rank) ∧
      ∀ e e' : DCells n n' B, e ≠ e' → ∀ c : R,
        2 * L ≤ (projF2 φ fun z =>
          dPrefix pc vq w e.1 e.2.1 e.2.2.val z - c * dPrefix pc vq w e'.1 e'.2.1 e'.2.2.val z).rank

/-! ## The honest naive kernel (rule I) -/

section Honest

variable {R : Type} [CommRing R] {n L np B n' : ℕ}

/-- Cell-block `(i, l, t)` as the index `(i n' + l) B + t`. -/
def cellIdx : DCells n n' B ≃ Fin (n * (n' * B)) :=
  (Equiv.prodCongr (Equiv.refl _) finProdFinEquiv).trans finProdFinEquiv

/-- The honest kernel's slots: cell-block `r`'s `L` products, then its accumulator, at `(L + 1) r + j`. -/
def slotIdx : DCells n n' B × Fin (L + 1) ≃ Fin (n * (n' * B) * (L + 1)) :=
  (Equiv.prodCongr cellIdx (Equiv.refl _)).trans finProdFinEquiv

/-- The honest kernel's slot count. -/
abbrev honestN (n L B n' : ℕ) : ℕ := n * (n' * B) * (L + 1)

/-- Slot `a` (an accumulator) reads slot `b`: its own cell-block's products, and its cell's previous accumulator. -/
def honestReads (a b : DCells n n' B × Fin (L + 1)) : Bool :=
  decide (a.2 = Fin.last L ∧ ((b.1 = a.1 ∧ b.2.val < L) ∨
    (b.2 = Fin.last L ∧ b.1.1 = a.1.1 ∧ b.1.2.1 = a.1.2.1 ∧ b.1.2.2.val + 1 = a.1.2.2.val)))

/-- **The honest naive kernel**: cell-block `(i, l, t)`'s `L` products `P̃_(i, tL+j) X_(tL+j, l)` (MMA products of
the design's operands), each read once by the block's accumulator, which adds them to its cell's previous accumulator,
updated in place. -/
def honestProg (pc : Fin n → Fin B → Fin L → R) (vq : Fin B → Fin L → Fin n' → R) (w : Fin np → R) :
    SlpG (DBits n L np B n') R (honestN n L B n') where
  isProd s := decide ((slotIdx.symm s).2.val < L)
  prod s z :=
    if h : (slotIdx.symm s).2.val < L then
      (dP pc w (slotIdx.symm s).1.1 (slotIdx.symm s).1.2.2 ⟨_, h⟩).eval z *
        (dX vq (slotIdx.symm s).1.2.2 ⟨_, h⟩ (slotIdx.symm s).1.2.1).eval z
    else 0
  lin s k := if honestReads (slotIdx.symm s) (slotIdx.symm ⟨k.val, k.isLt.trans s.isLt⟩) then 1 else 0
  off _ := ⟨0, 0⟩

/-- The honest kernel's flow: a product to its block's accumulator, an accumulator to its cell's next one. -/
def honestFlow : Flow (honestN n L B n') where
  free s :=
    if (slotIdx.symm s).2.val < L then some (slotIdx ((slotIdx.symm s).1, Fin.last L))
    else if h : (slotIdx.symm s).1.2.2.val + 1 < B then
      some (slotIdx (((slotIdx.symm s).1.1, (slotIdx.symm s).1.2.1, ⟨_, h⟩), Fin.last L))
    else none

/-- The honest kernel's committed words: the accumulators. -/
def honestWord (c : DCells n n' B) : Fin (honestN n L B n') := slotIdx (c, Fin.last L)

/-- **The honest witness** (rule I): the honest naive kernel meets every hypothesis of `RingKCount` (an accumulator
flow, degree-2 products, accumulators as words), has no delivery and `n n' B L` products, and its every word is its
target exactly, so every cell-block is certified and the count holds with equality. -/
def HonestWitness : Prop :=
  ∀ {R : Type} [CommRing R] {n L np B n' : ℕ} (pc : Fin n → Fin B → Fin L → R) (vq : Fin B → Fin L → Fin n' → R)
    (w : Fin np → R),
    (honestFlow (n := n) (L := L) (B := B) (n' := n')).WF (honestProg pc vq w) ∧ AffOps (honestProg pc vq w) ∧
    (∀ c, (honestProg pc vq w).isProd (honestWord c) = false) ∧
    deliveries (honestProg pc vq w) honestFlow = 0 ∧
    products (honestProg pc vq w) = n * n' * B * L ∧
    ∀ c z, (honestProg pc vq w).val z (honestWord c) = dPrefix pc vq w c.1 c.2.1 c.2.2.val z

/-- Under the honest witness every cell-block is certified, and `RingKCount`'s bound `L · #certified ≤ #products`
(with no delivery) is an equality: the count is tight. -/
theorem honest_tight (h : HonestWitness) (pc : Fin n → Fin B → Fin L → R) (vq : Fin B → Fin L → Fin n' → R)
    (w : Fin np → R) :
    (certified pc vq w (honestProg pc vq w) honestWord).card = n * n' * B ∧
      L * (certified pc vq w (honestProg pc vq w) honestWord).card = products (honestProg pc vq w) := by
  obtain ⟨-, -, -, -, hprod, hval⟩ := h pc vq w
  have hall : certified pc vq w (honestProg pc vq w) honestWord = univ := by
    refine Finset.eq_univ_of_forall fun c => ?_
    simp only [certified, Finset.mem_filter, Finset.mem_univ, true_and]
    exact fun U _ => hval c _
  have hcard : (certified pc vq w (honestProg pc vq w) honestWord).card = n * n' * B := by
    rw [hall, Finset.card_univ]
    simp [DCells, Fintype.card_prod, mul_assoc]
  refine ⟨hcard, ?_⟩
  rw [hcard, hprod]
  ring

end Honest

end Pouw.RingFragment
