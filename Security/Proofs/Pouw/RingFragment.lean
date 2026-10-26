import Mathlib
import Definitions.Pouw.RelativeSecurity
import Proofs.Pouw.RelativeSecurity

/-!
# Theorem K's count for the ring fragment (statement, red-teamed; the proof in progress)

BET-K-DISCHARGE (CMP-123, ACCEL): every program of the ring fragment pays at least Theorem K's count for Dither's
transcript, with `theoremK` reused as it stands. This module states that count (`RingKCount`) and its cost form
(`RingKCost`, from the count by `ringKCost_of_count`), and proves the honest naive kernel its non-vacuity witness
(`honestWitness : HonestWitness`, rule I; `honest_tight`: the count holds there with equality). It is the base O-CHARGE
(CMP-111) builds its Lean on. MODEL-SPEC's machine
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

**What it would discharge**, once `RingKCount` is proved, given the price obligation below and a bridge (not yet
written) from `RingKCost` to `RoundsHypKSt`'s cost line. For degree-2 ring programs, `KScopeAffine`'s third conjunct
(row purity), and `ChargeVF3j16`'s cost line in this form: a machine that charges `κ` a product and `d` a delivery
charges at least `min(κ L, d)` a certified cell-block, so at `κ_eff = min(κ, d/L)` a product of the floor. That is
Theorem K's full count only when `d ≥ κ L`. `KScopeAffine`'s first
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
  such a program is in the fragment only if the second read is charged as a delivery. The card charges nothing for
  it (an out-of-place `mma.sync` costs what an in-place one does), so the count with free forks is open: rank alone
  allows a fork's shared prefix to serve several blocks. H200's `wgmma` accumulates in place (no `C` operand) and is
  in it.
* **Non-ring instructions** (O-BITS): `IMAD.HI`, `IMAD.WIDE`'s high word, shifts, `LOP3`, `PRMT`, compares, selects
  and the float formats.
* **Operands are free.** A product's bit-affine operands are given, not computed by slots, so the loads and adds that
  form them aren't charged, which only helps the program.

**The price obligation** (MODEL-SPEC and O-CHARGE), for a run's charge to be at least `κ · #products + d · #deliveries`
less its noise reads: an in-place MMA at least `κ` a product; `IMAD` on bit-affine `a`, `b` at least `κ + d` (its
product, and `c` when that is a delivery); `IADD3` at least `d` per input that isn't its in-place accumulator (so
`d ≤ price(IADD3)/2`, or `/3` when no input is in place); a copy of an accumulator word, and a shared-memory round
trip, at least `d` a word. The commitment's read of a word is outside the program and is no delivery.

**Open conjuncts** (the coordinator's red team, 26 Oct). Two of `RingKCount`'s hypotheses are scope this module
can't keep, since R-ACCEL-4 allows no premise about algorithms:

* **K1, forks.** MODEL-SPEC models an out-of-place MMA as physical and free, so `Flow.WF`'s one-free-reader rule
  must go. With forks the accumulators form a forest, and the count becomes a Steiner bound in the rank metric.
  Strassen is no forest, since its output sets aren't suffix-closed along any tree, but the general bound is open.
* **K2, degree.** `AffOps` is to be replaced by certifying at the live uniform coin. One word's cone needs `L`
  products at any degree when it is right with probability above `3/4` (mod 2, fix one input of each product). The
  count across words with products of products is open.

**The proof plan.** Distinct certified cell-blocks have distinct word slots (`word_injOn`: `theoremK` at no products
against `PrefixDiffRank`). For a certified `e`, walk back from its word slot along the in-place chain to the
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

/-- What linear slot `j` reads from products. -/
def prodPart (P : SlpG V R N) (j : Fin N) (z : V → R) : R :=
  ∑ k : Fin j.val, if P.isProd ⟨k.val, k.isLt.trans j.isLt⟩ = true then
    P.lin j k * P.val z ⟨k.val, k.isLt.trans j.isLt⟩ else 0

/-- **One accumulator step**: a linear slot with no delivery reads its products, at most one linear slot (the
accumulator it updates in place, unique by `Flow.WF`) and its offset. -/
theorem val_step (P : SlpG V R N) (F : Flow N) (hF : F.WF P) (j : Fin N) (hj : P.isProd j = false)
    (hnd : ∀ k : Fin j.val, P.lin j k ≠ 0 → F.free ⟨k.val, k.isLt.trans j.isLt⟩ = some j) :
    (∃ k : Fin j.val, P.isProd ⟨k.val, k.isLt.trans j.isLt⟩ = false ∧ P.lin j k ≠ 0 ∧ ∀ z,
      P.val z j = P.lin j k * P.val z ⟨k.val, k.isLt.trans j.isLt⟩ + prodPart P j z + (P.off j).eval z) ∨
    ∀ z, P.val z j = prodPart P j z + (P.off j).eval z := by
  classical
  have hnp : ¬P.isProd j = true := by simp [hj]
  have hv : ∀ z, P.val z j =
      (∑ k : Fin j.val, P.lin j k * P.val z ⟨k.val, k.isLt.trans j.isLt⟩) + (P.off j).eval z := fun z => by
    rw [SlpG.val, ite_eq_right hnp]
  have hsplit : ∀ z, (∑ k : Fin j.val, P.lin j k * P.val z ⟨k.val, k.isLt.trans j.isLt⟩) =
      (∑ k : Fin j.val, if P.isProd ⟨k.val, k.isLt.trans j.isLt⟩ = true then 0 else
        P.lin j k * P.val z ⟨k.val, k.isLt.trans j.isLt⟩) + prodPart P j z := fun z => by
    rw [prodPart, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun k _ => ?_
    split_ifs <;> simp
  by_cases hex : ∃ k : Fin j.val, P.isProd ⟨k.val, k.isLt.trans j.isLt⟩ = false ∧ P.lin j k ≠ 0
  · obtain ⟨k, hk, hl⟩ := hex
    left
    refine ⟨k, hk, hl, fun z => ?_⟩
    rw [hv, hsplit]
    have hk2 : ¬P.isProd ⟨k.val, k.isLt.trans j.isLt⟩ = true := by simp [hk]
    rw [Finset.sum_eq_single k, ite_eq_right hk2]
    · intro k' _ hk'
      split_ifs with hp
      · rfl
      · by_cases hl' : P.lin j k' = 0
        · simp [hl']
        · exfalso
          apply hk'
          have e := hF.2 j _ _ (hnd k hl) (hnd k' hl') hk (by simpa using hp)
          exact Fin.ext (congrArg Fin.val e).symm
    · intro h
      exact absurd (Finset.mem_univ k) h
  · right
    intro z
    rw [hv, hsplit, Finset.sum_eq_zero, zero_add]
    intro k _
    split_ifs with hp
    · rfl
    · have hl : P.lin j k = 0 := by
        by_contra hl
        exact hex ⟨k, by simpa using hp, hl⟩
      simp [hl]

end Flow

section Ops

variable {Vl Vr R : Type} [Fintype Vl] [Fintype Vr] [DecidableEq Vl] [DecidableEq Vr] [CommRing R] {N : ℕ}

/-- **Degree 2**: every product multiplies two bit-affine operands (`AffProd`, mixed included). -/
def AffOps (P : SlpG (Vl ⊕ Vr) R N) : Prop := ∀ j, P.isProd j = true → AffProd (P.prod j)

omit [DecidableEq Vl] [DecidableEq Vr] in
/-- **An accumulator's product reads in `theoremK`'s form**: with degree-2 products, what linear slot `j` reads from
products is a combination of products of bit-affine operands. -/
theorem prodPart_eq (P : SlpG (Vl ⊕ Vr) R N) (hP : AffOps P) (j : Fin N) :
    ∃ A Bo : Fin j.val → BitAff (Vl ⊕ Vr) R, ∀ z, prodPart P j z =
      ∑ k ∈ univ.filter (fun k : Fin j.val => P.isProd ⟨k.val, k.isLt.trans j.isLt⟩ = true),
        P.lin j k * ((A k).eval z * (Bo k).eval z) := by
  classical
  have h : ∀ k : Fin j.val, ∃ AB : BitAff (Vl ⊕ Vr) R × BitAff (Vl ⊕ Vr) R,
      P.isProd ⟨k.val, k.isLt.trans j.isLt⟩ = true →
        ∀ z, P.val z ⟨k.val, k.isLt.trans j.isLt⟩ = AB.1.eval z * AB.2.eval z := by
    intro k
    by_cases hk : P.isProd ⟨k.val, k.isLt.trans j.isLt⟩ = true
    · obtain ⟨A, Bo, hAB⟩ := hP _ hk
      exact ⟨(A, Bo), fun _ z => by rw [SlpG.val, ite_eq_left hk]; exact hAB z⟩
    · exact ⟨(⟨0, 0⟩, ⟨0, 0⟩), fun h => absurd h hk⟩
  choose AB hAB using h
  refine ⟨fun k => (AB k).1, fun k => (AB k).2, fun z => ?_⟩
  rw [prodPart, Finset.sum_filter]
  refine Finset.sum_congr rfl fun k _ => ?_
  split_ifs with hk
  · rw [hAB k hk z]
  · rfl

end Ops

section Walk

variable {V R : Type} [Fintype V] [CommRing R] {N : ℕ}

/-- The first slot of `stop` on slot `t`'s forward free path, `t` included. -/
def owner (F : Flow N) (stop : Finset (Fin N)) (t : Fin N) : Option (Fin N) :=
  if t ∈ stop then some t else
    match F.free t with
    | none => none
    | some j => if _h : t < j then owner F stop j else none
termination_by N - t.val
decreasing_by have := j.isLt; have : t.val < j.val := _h; omega

theorem owner_of_mem (F : Flow N) {stop : Finset (Fin N)} {t : Fin N} (h : t ∈ stop) :
    owner F stop t = some t := by
  rw [owner, ite_eq_left h]

theorem owner_of_free (F : Flow N) {stop : Finset (Fin N)} {t j : Fin N} (h : t ∉ stop) (hj : F.free t = some j)
    (hlt : t < j) : owner F stop t = owner F stop j := by
  rw [owner, ite_eq_right h]
  simp [hj, hlt]

/-- A slot's owner is no earlier than the slot. -/
theorem owner_le (F : Flow N) (stop : Finset (Fin N)) {o : Fin N} (t : Fin N) (h : owner F stop t = some o) :
    t ≤ o := by
  suffices H : ∀ m (t : Fin N), N - t.val = m → owner F stop t = some o → t ≤ o from H _ t rfl h
  intro m
  induction m using Nat.strong_induction_on with
  | _ m ih =>
    intro t hm h
    rw [owner] at h
    by_cases hs : t ∈ stop
    · rw [ite_eq_left hs] at h
      exact le_of_eq (Option.some_injective _ h)
    · rw [ite_eq_right hs] at h
      rcases hf : F.free t with _ | j
      · simp [hf] at h
      · by_cases hlt : t < j
        · simp only [hf, hlt, dite_true] at h
          have hm' : N - j.val < m := by have := j.isLt; have : t.val < j.val := hlt; omega
          exact le_trans hlt.le (ih _ hm' j rfl h)
        · simp [hf, hlt] at h

/-- Slot `s`'s value is `base` plus a combination of the slots in `S` plus a bit-affine word, uniformly in the
bits. -/
def ValForm (P : SlpG V R N) (S : Finset (Fin N)) (s : Fin N) (base : (V → R) → R) : Prop :=
  ∃ (lam : Fin N → R) (r : BitAff V R), ∀ z, P.val z s = base z + ∑ k ∈ S, lam k * P.val z k + r.eval z

/-- What linear slot `s` reads from products, as a combination over a set holding every product it reads. -/
theorem prodPart_sum (P : SlpG V R N) (S : Finset (Fin N)) (s : Fin N)
    (hS : ∀ k : Fin s.val, P.isProd ⟨k.val, k.isLt.trans s.isLt⟩ = true → P.lin s k ≠ 0 →
      (⟨k.val, k.isLt.trans s.isLt⟩ : Fin N) ∈ S) :
    ∃ mu : Fin N → R, ∀ z, prodPart P s z = ∑ k ∈ S, mu k * P.val z k := by
  classical
  let mu : Fin N → R := fun k =>
    if h : k.val < s.val then (if P.isProd k = true then P.lin s ⟨k.val, h⟩ else 0) else 0
  refine ⟨mu, fun z => ?_⟩
  let G : Fin N → R := fun k => if k ∈ S then mu k * P.val z k else 0
  have e1 : prodPart P s z = ∑ k ∈ univ.map (Fin.castLEEmb s.isLt.le), G k := by
    rw [Finset.sum_map, prodPart]
    refine Finset.sum_congr rfl fun k _ => ?_
    have hk : (Fin.castLEEmb s.isLt.le k) = ⟨k.val, k.isLt.trans s.isLt⟩ := rfl
    rw [hk]
    simp only [G, mu, dite_eq_left k.isLt]
    by_cases hp : P.isProd ⟨k.val, k.isLt.trans s.isLt⟩ = true
    · by_cases hl : P.lin s k = 0
      · simp [hl]
      · rw [ite_eq_left (hS k hp hl)]
        simp [hp]
    · simp [hp]
  have e2 : ∑ k ∈ univ.map (Fin.castLEEmb s.isLt.le), G k = ∑ k, G k := by
    refine Finset.sum_subset (Finset.subset_univ _) fun k _ hk => ?_
    have hks : ¬k < s := fun hks => hk (Finset.mem_map.mpr ⟨⟨k.val, hks⟩, Finset.mem_univ _, rfl⟩)
    simp [G, mu, hks]
  rw [e1, e2, Finset.sum_ite_mem, Finset.univ_inter]

theorem valForm_of (P : SlpG V R N) (S : Finset (Fin N)) (s : Fin N) (base : (V → R) → R) (lam0 : Fin N → R)
    (r0 : BitAff V R)
    (hS : ∀ k : Fin s.val, P.isProd ⟨k.val, k.isLt.trans s.isLt⟩ = true → P.lin s k ≠ 0 →
      (⟨k.val, k.isLt.trans s.isLt⟩ : Fin N) ∈ S)
    (hv : ∀ z, P.val z s =
      base z + (∑ k ∈ S, lam0 k * P.val z k + r0.eval z) + prodPart P s z + (P.off s).eval z) :
    ValForm P S s base := by
  obtain ⟨mu, hmu⟩ := prodPart_sum P S s hS
  refine ⟨fun k => lam0 k + mu k, ⟨r0.c + (P.off s).c, fun v => r0.α v + (P.off s).α v⟩, fun z => ?_⟩
  rw [hv, hmu]
  simp only [BitAff.eval, add_mul, Finset.sum_add_distrib]
  ring

/-- One step back along the chain: a value form of the in-place predecessor gives one of the slot. -/
theorem valForm_step (P : SlpG V R N) (S : Finset (Fin N)) (s k : Fin N) (l : R) (base : (V → R) → R)
    (hk : ValForm P S k base)
    (hS : ∀ k : Fin s.val, P.isProd ⟨k.val, k.isLt.trans s.isLt⟩ = true → P.lin s k ≠ 0 →
      (⟨k.val, k.isLt.trans s.isLt⟩ : Fin N) ∈ S)
    (hv : ∀ z, P.val z s = l * P.val z k + prodPart P s z + (P.off s).eval z) :
    ValForm P S s fun z => l * base z := by
  obtain ⟨lam, r, hr⟩ := hk
  refine valForm_of P S s _ (fun k => l * lam k) ⟨l * r.c, fun v => l * r.α v⟩ hS fun z => ?_
  rw [hv, hr]
  simp only [BitAff.eval, mul_add, Finset.mul_sum, mul_assoc]
  ring

open Classical in
/-- The deliveries, as a set of reads (`deliveries` is its size). -/
noncomputable def delSet (P : SlpG V R N) (F : Flow N) : Finset (Σ j : Fin N, Fin j.val) :=
  univ.filter fun e : Σ j : Fin N, Fin j.val =>
    P.isProd e.1 = false ∧ P.lin e.1 e.2 ≠ 0 ∧ F.free ⟨e.2.val, e.2.isLt.trans e.1.isLt⟩ ≠ some e.1

omit [Fintype V] in
theorem deliveries_eq (P : SlpG V R N) (F : Flow N) : deliveries P F = (delSet P F).card := rfl

open Classical in
/-- The products whose free path first meets `stop` at `o`. -/
noncomputable def ownProds (P : SlpG V R N) (F : Flow N) (stop : Finset (Fin N)) (o : Fin N) : Finset (Fin N) :=
  univ.filter fun k => P.isProd k = true ∧ owner F stop k = some o

/-- **The walk back** from a linear slot `s` owned by `o`, along its in-place chain: it meets a delivery on a slot
`o` owns, or `s`'s value is a combination of `o`'s own products plus a bit-affine word, plus a multiple of the
first earlier slot of `stop` it meets, if any. -/
theorem walk (P : SlpG V R N) (F : Flow N) (hF : F.WF P) (stop : Finset (Fin N))
    (hstop : ∀ s ∈ stop, P.isProd s = false) (o s : Fin N) (hs : P.isProd s = false)
    (ho : owner F stop s = some o) :
    (∃ e ∈ delSet P F, owner F stop e.1 = some o) ∨ ValForm P (ownProds P F stop o) s (fun _ => 0) ∨
      ∃ a ∈ stop, a ≠ o ∧ ∃ c : R, ValForm P (ownProds P F stop o) s fun z => c * P.val z a := by
  classical
  suffices H : ∀ m (s : Fin N), s.val = m → P.isProd s = false → owner F stop s = some o →
      (∃ e ∈ delSet P F, owner F stop e.1 = some o) ∨ ValForm P (ownProds P F stop o) s (fun _ => 0) ∨
        ∃ a ∈ stop, a ≠ o ∧ ∃ c : R, ValForm P (ownProds P F stop o) s fun z => c * P.val z a from
    H _ s rfl hs ho
  intro m
  induction m using Nat.strong_induction_on with
  | _ m ih =>
  intro s hm hs ho
  subst hm
  by_cases hd : ∃ k : Fin s.val, P.lin s k ≠ 0 ∧ F.free ⟨k.val, k.isLt.trans s.isLt⟩ ≠ some s
  · obtain ⟨k, hl, hf⟩ := hd
    exact Or.inl ⟨⟨s, k⟩, by simp [delSet, hs, hl, hf], ho⟩
  simp only [not_exists, not_and, not_not] at hd
  have hS : ∀ k : Fin s.val, P.isProd ⟨k.val, k.isLt.trans s.isLt⟩ = true → P.lin s k ≠ 0 →
      (⟨k.val, k.isLt.trans s.isLt⟩ : Fin N) ∈ ownProds P F stop o := by
    intro k hp hl
    have hns : (⟨k.val, k.isLt.trans s.isLt⟩ : Fin N) ∉ stop := fun h => by simp [hstop _ h] at hp
    simp only [ownProds, Finset.mem_filter, Finset.mem_univ, true_and]
    exact ⟨hp, (owner_of_free F hns (hd k hl) k.isLt).trans ho⟩
  rcases val_step P F hF s hs hd with ⟨k, hk, hl, hv⟩ | hv
  · by_cases hks : (⟨k.val, k.isLt.trans s.isLt⟩ : Fin N) ∈ stop
    · refine Or.inr (Or.inr ⟨_, hks, fun h => ?_, P.lin s k, valForm_of P _ s _ 0 ⟨0, 0⟩ hS fun z => ?_⟩)
      · have h1 := owner_le F stop s ho
        rw [← h] at h1
        exact absurd k.isLt (not_lt.mpr h1)
      · rw [hv z]
        simp [BitAff.eval]
    · have hko : owner F stop ⟨k.val, k.isLt.trans s.isLt⟩ = some o :=
        (owner_of_free F hks (hd k hl) k.isLt).trans ho
      rcases ih k.val k.isLt _ rfl hk hko with h | h | ⟨a, ha, hao, c, h⟩
      · exact Or.inl h
      · refine Or.inr (Or.inl ?_)
        simpa using valForm_step P _ s _ (P.lin s k) _ h hS hv
      · refine Or.inr (Or.inr ⟨a, ha, hao, P.lin s k * c, ?_⟩)
        simpa [mul_assoc] using valForm_step P _ s _ (P.lin s k) _ h hS hv
  · exact Or.inr (Or.inl (valForm_of P _ s _ 0 ⟨0, 0⟩ hS fun z => by rw [hv z]; simp [BitAff.eval]))

end Walk

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

theorem cellIdx_val (c : DCells n n' B) : (cellIdx c).val = c.2.2.val + B * c.2.1.val + n' * B * c.1.val := rfl

theorem slotIdx_val (x : DCells n n' B × Fin (L + 1)) : (slotIdx x).val = x.2.val + (L + 1) * (cellIdx x.1).val := rfl

theorem honestProg_isProd (pc : Fin n → Fin B → Fin L → R) (vq : Fin B → Fin L → Fin n' → R) (w : Fin np → R)
    (x : DCells n n' B × Fin (L + 1)) : (honestProg pc vq w).isProd (slotIdx x) = decide (x.2.val < L) := by
  show decide ((slotIdx.symm (slotIdx x)).2.val < L) = _
  rw [Equiv.symm_apply_apply]

theorem honestProg_val_prod (pc : Fin n → Fin B → Fin L → R) (vq : Fin B → Fin L → Fin n' → R) (w : Fin np → R)
    (z : DBits n L np B n' → R) (c : DCells n n' B) (j : Fin L) :
    (honestProg pc vq w).val z (slotIdx (c, j.castSucc)) = (dP pc w c.1 c.2.2 j).eval z * (dX vq c.2.2 j c.2.1).eval z := by
  have hp : (honestProg pc vq w).isProd (slotIdx (c, j.castSucc)) = true := by
    rw [honestProg_isProd]
    simp
  rw [SlpG.val, ite_eq_left hp]
  simp [honestProg]

theorem honestFlow_prod (c : DCells n n' B) (a : Fin (L + 1)) (h : a.val < L) :
    (honestFlow (n := n) (L := L) (B := B) (n' := n')).free (slotIdx (c, a)) = some (slotIdx (c, Fin.last L)) := by
  simp [honestFlow, h]

theorem honestFlow_acc (i : Fin n) (l : Fin n') (t : Fin B) :
    (honestFlow (n := n) (L := L) (B := B) (n' := n')).free (slotIdx ((i, l, t), Fin.last L)) =
      if h : t.val + 1 < B then some (slotIdx ((i, l, ⟨t.val + 1, h⟩), Fin.last L)) else none := by
  simp [honestFlow]

/-- An accumulator reads only earlier slots. -/
theorem honestReads_lt (a b : DCells n n' B × Fin (L + 1)) (h : honestReads a b = true) : slotIdx b < slotIdx a := by
  obtain ⟨⟨i, l, t⟩, a2⟩ := a
  obtain ⟨⟨i', l', t'⟩, b2⟩ := b
  simp only [honestReads, decide_eq_true_eq, Prod.mk.injEq] at h
  obtain ⟨rfl, h⟩ := h
  rcases h with ⟨⟨hi, hl, ht⟩, hb⟩ | ⟨rfl, hi, hl, ht⟩
  · simp only [Fin.lt_def, slotIdx_val, cellIdx_val, Fin.val_last]
    rw [hi, hl, ht]
    omega
  · simp only [Fin.lt_def, slotIdx_val, cellIdx_val, Fin.val_last]
    rw [hi, hl, ← ht]
    have e : (L + 1) * (t'.val + 1 + B * l.val + n' * B * i.val) =
        (L + 1) * (t'.val + B * l.val + n' * B * i.val) + (L + 1) := by ring
    omega

theorem honest_wf (pc : Fin n → Fin B → Fin L → R) (vq : Fin B → Fin L → Fin n' → R) (w : Fin np → R) :
    (honestFlow (n := n) (L := L) (B := B) (n' := n')).WF (honestProg pc vq w) := by
  refine ⟨fun k j h => ?_, fun j k k' h h' hk hk' => ?_⟩
  · obtain ⟨⟨⟨i, l, t⟩, a⟩, rfl⟩ := slotIdx.surjective k
    by_cases ha : a.val < L
    · rw [honestFlow_prod _ _ ha] at h
      obtain rfl := Option.some_injective _ h
      refine ⟨?_, by rw [honestProg_isProd]; simp⟩
      simp only [Fin.lt_def, slotIdx_val, Fin.val_last]
      omega
    · have ha' : a = Fin.last L := Fin.ext (by have := a.isLt; simp only [Fin.val_last]; omega)
      subst ha'
      rw [honestFlow_acc] at h
      by_cases ht : t.val + 1 < B
      · rw [dite_eq_left ht] at h
        obtain rfl := Option.some_injective _ h
        refine ⟨?_, by rw [honestProg_isProd]; simp⟩
        simp only [Fin.lt_def, slotIdx_val, cellIdx_val, Fin.val_last]
        have e : (L + 1) * (t.val + 1 + B * l.val + n' * B * i.val) =
            (L + 1) * (t.val + B * l.val + n' * B * i.val) + (L + 1) := by ring
        omega
      · rw [dite_eq_right ht] at h
        exact absurd h (by simp)
  · obtain ⟨⟨⟨i, l, t⟩, a⟩, rfl⟩ := slotIdx.surjective k
    obtain ⟨⟨⟨i', l', t'⟩, a'⟩, rfl⟩ := slotIdx.surjective k'
    rw [honestProg_isProd, decide_eq_false_iff_not] at hk hk'
    dsimp only at hk hk'
    have ha : a = Fin.last L := Fin.ext (by have := a.isLt; simp only [Fin.val_last]; omega)
    have ha' : a' = Fin.last L := Fin.ext (by have := a'.isLt; simp only [Fin.val_last]; omega)
    subst ha ha'
    rw [honestFlow_acc] at h h'
    by_cases ht : t.val + 1 < B
    · by_cases ht' : t'.val + 1 < B
      · rw [dite_eq_left ht] at h
        rw [dite_eq_left ht'] at h'
        have e := slotIdx.injective (Option.some_injective _ (h.trans h'.symm))
        have hi : i = i' := congrArg (fun x : DCells n n' B × Fin (L + 1) => x.1.1) e
        have hl : l = l' := congrArg (fun x : DCells n n' B × Fin (L + 1) => x.1.2.1) e
        have htt : t = t' := Fin.ext (by
          have h2 := congrArg (fun x : DCells n n' B × Fin (L + 1) => x.1.2.2.val) e
          dsimp only at h2
          omega)
        subst hi hl htt
        rfl
      · rw [dite_eq_right ht'] at h'
        exact absurd h' (by simp)
    · rw [dite_eq_right ht] at h
      exact absurd h (by simp)

theorem honest_affOps (pc : Fin n → Fin B → Fin L → R) (vq : Fin B → Fin L → Fin n' → R) (w : Fin np → R) :
    AffOps (honestProg pc vq w) := by
  intro j hj
  have hj' : (slotIdx.symm j).2.val < L := of_decide_eq_true hj
  exact ⟨dP pc w (slotIdx.symm j).1.1 (slotIdx.symm j).1.2.2 ⟨_, hj'⟩,
    dX vq (slotIdx.symm j).1.2.2 ⟨_, hj'⟩ (slotIdx.symm j).1.2.1, fun z => dite_eq_left hj'⟩

theorem honest_deliveries (pc : Fin n → Fin B → Fin L → R) (vq : Fin B → Fin L → Fin n' → R) (w : Fin np → R) :
    deliveries (honestProg pc vq w) (honestFlow (n := n) (L := L) (B := B) (n' := n')) = 0 := by
  classical
  rw [deliveries, Finset.card_eq_zero, Finset.filter_eq_empty_iff]
  rintro ⟨s, k⟩ - ⟨-, hlin, hfree⟩
  try dsimp only at hlin hfree
  apply hfree
  have hl : honestReads (slotIdx.symm s) (slotIdx.symm ⟨k.val, k.isLt.trans s.isLt⟩) = true := by
    by_contra hc
    exact hlin (ite_eq_right hc)
  generalize (⟨k.val, k.isLt.trans s.isLt⟩ : Fin (honestN n L B n')) = k' at hl ⊢
  obtain ⟨⟨⟨i, l, t⟩, a⟩, rfl⟩ := slotIdx.surjective s
  obtain ⟨⟨⟨i', l', t'⟩, b⟩, rfl⟩ := slotIdx.surjective k'
  simp only [Equiv.symm_apply_apply, honestReads, decide_eq_true_eq, Prod.mk.injEq] at hl
  obtain ⟨rfl, h⟩ := hl
  rcases h with ⟨⟨rfl, rfl, rfl⟩, hb⟩ | ⟨rfl, rfl, rfl, ht⟩
  · exact honestFlow_prod _ _ hb
  · have hlt : t'.val + 1 < B := ht ▸ t.isLt
    rw [honestFlow_acc, dite_eq_left hlt, show (⟨t'.val + 1, hlt⟩ : Fin B) = t from Fin.ext ht]

theorem honest_products (pc : Fin n → Fin B → Fin L → R) (vq : Fin B → Fin L → Fin n' → R) (w : Fin np → R) :
    products (honestProg pc vq w) = n * n' * B * L := by
  rw [products, Finset.card_filter, ← Equiv.sum_comp slotIdx, Fintype.sum_prod_type]
  have hin : ∀ c : DCells n n' B,
      (∑ a : Fin (L + 1), if (honestProg pc vq w).isProd (slotIdx (c, a)) = true then 1 else 0) = L := by
    intro c
    rw [Fin.sum_univ_castSucc]
    simp [honestProg_isProd]
  rw [Finset.sum_congr rfl fun c _ => hin c, Finset.sum_const, Finset.card_univ, smul_eq_mul]
  simp only [DCells, Fintype.card_prod, Fintype.card_fin]
  ring

/-- An accumulator's read sum, over the slots as pairs: its own block's products and its cell's previous
accumulator, each with coefficient 1. -/
theorem honest_lin_sum (pc : Fin n → Fin B → Fin L → R) (vq : Fin B → Fin L → Fin n' → R) (w : Fin np → R)
    (z : DBits n L np B n' → R) (s : Fin (honestN n L B n')) :
    ∑ k : Fin s.val, (honestProg pc vq w).lin s k * (honestProg pc vq w).val z ⟨k.val, k.isLt.trans s.isLt⟩ =
      ∑ y : DCells n n' B × Fin (L + 1),
        (if honestReads (slotIdx.symm s) y then (1 : R) else 0) * (honestProg pc vq w).val z (slotIdx y) := by
  classical
  let G : Fin (honestN n L B n') → R := fun k =>
    (if honestReads (slotIdx.symm s) (slotIdx.symm k) then (1 : R) else 0) * (honestProg pc vq w).val z k
  have e1 : ∑ k : Fin s.val, (honestProg pc vq w).lin s k * (honestProg pc vq w).val z ⟨k.val, k.isLt.trans s.isLt⟩ =
      ∑ k ∈ univ.map (Fin.castLEEmb s.isLt.le), G k := by
    rw [Finset.sum_map]
    rfl
  have e2 : ∑ k ∈ univ.map (Fin.castLEEmb s.isLt.le), G k = ∑ k, G k := by
    refine Finset.sum_subset (Finset.subset_univ _) fun k _ hk => ?_
    have hks : ¬k < s := fun hks => hk (Finset.mem_map.mpr ⟨⟨k.val, hks⟩, Finset.mem_univ _, rfl⟩)
    have hr : honestReads (slotIdx.symm s) (slotIdx.symm k) ≠ true := fun h =>
      hks (by simpa using honestReads_lt _ _ h)
    simp only [G, ite_eq_right hr, zero_mul]
  rw [e1, e2, ← Equiv.sum_comp slotIdx]
  simp only [G, Equiv.symm_apply_apply]

/-- An accumulator's value: its block's increment plus its cell's previous accumulator. -/
theorem honest_val_acc (pc : Fin n → Fin B → Fin L → R) (vq : Fin B → Fin L → Fin n' → R) (w : Fin np → R)
    (z : DBits n L np B n' → R) (i : Fin n) (l : Fin n') (t : Fin B) :
    (honestProg pc vq w).val z (slotIdx ((i, l, t), Fin.last L)) =
      dIncr pc vq w i l t z + if h : 0 < t.val then
        (honestProg pc vq w).val z (slotIdx ((i, l, ⟨t.val - 1, by have := t.isLt; omega⟩), Fin.last L)) else 0 := by
  classical
  have hnp : ¬(honestProg pc vq w).isProd (slotIdx ((i, l, t), Fin.last L)) = true := by
    rw [honestProg_isProd]
    simp
  have hoff : ((honestProg pc vq w).off (slotIdx ((i, l, t), Fin.last L))).eval z = 0 := by
    simp [honestProg, BitAff.eval]
  have hr1 : ∀ (c : DCells n n' B) (j : Fin L) (v : R),
      (if honestReads ((i, l, t), Fin.last L) (c, j.castSucc) then (1 : R) else 0) * v =
        if c = (i, l, t) then v else 0 := by
    intro c j v
    by_cases hc : c = (i, l, t) <;> simp [honestReads, hc, Fin.castSucc_ne_last]
  rw [SlpG.val, ite_eq_right hnp, hoff, add_zero, honest_lin_sum, Equiv.symm_apply_apply, Fintype.sum_prod_type]
  simp only [Fin.sum_univ_castSucc]
  rw [Finset.sum_add_distrib]
  congr 1
  · simp only [hr1]
    rw [Finset.sum_comm]
    simp only [Finset.sum_ite_eq', Finset.mem_univ, ↓reduceIte, honestProg_val_prod]
    rfl
  · by_cases ht : 0 < t.val
    · rw [dite_eq_left ht, Finset.sum_eq_single (i, l, ⟨t.val - 1, by have := t.isLt; omega⟩)]
      · rw [ite_eq_left, one_mul]
        simp [honestReads]
        omega
      · intro c _ hc
        rw [ite_eq_right, zero_mul]
        intro hr
        apply hc
        obtain ⟨i', l', t'⟩ := c
        simp [honestReads] at hr
        obtain ⟨hi, hl, h3⟩ := hr
        rw [hi, hl, show t' = ⟨t.val - 1, by have := t.isLt; omega⟩ from Fin.ext (by show t'.val = t.val - 1; omega)]
      · intro h
        exact absurd (Finset.mem_univ _) h
    · rw [dite_eq_right ht]
      refine Finset.sum_eq_zero fun c _ => ?_
      rw [ite_eq_right, zero_mul]
      intro hr
      obtain ⟨i', l', t'⟩ := c
      simp [honestReads] at hr
      obtain ⟨-, -, h3⟩ := hr
      omega

/-- Every honest word is its target, everywhere. -/
theorem honest_val (pc : Fin n → Fin B → Fin L → R) (vq : Fin B → Fin L → Fin n' → R) (w : Fin np → R)
    (z : DBits n L np B n' → R) (c : DCells n n' B) :
    (honestProg pc vq w).val z (honestWord c) = dPrefix pc vq w c.1 c.2.1 c.2.2.val z := by
  obtain ⟨i, l, t⟩ := c
  suffices H : ∀ m (t : Fin B), t.val = m →
      (honestProg pc vq w).val z (slotIdx ((i, l, t), Fin.last L)) = dPrefix pc vq w i l t.val z from H _ t rfl
  intro m
  induction m with
  | zero =>
    intro t ht
    have h0 : ¬0 < t.val := by omega
    have hB : 0 < B := by have := t.isLt; omega
    obtain rfl : t = ⟨0, hB⟩ := Fin.ext ht
    rw [honest_val_acc, dite_eq_right h0, add_zero, dPrefix_zero pc vq w i l hB]
  | succ m ih =>
    intro t ht
    have hlt : m + 1 < B := ht ▸ t.isLt
    obtain rfl : t = ⟨m + 1, hlt⟩ := Fin.ext ht
    rw [honest_val_acc, dite_eq_left (Nat.succ_pos m)]
    show dIncr pc vq w i l ⟨m + 1, hlt⟩ z +
        (honestProg pc vq w).val z (slotIdx ((i, l, ⟨m, by omega⟩), Fin.last L)) = dPrefix pc vq w i l (m + 1) z
    rw [ih ⟨m, by omega⟩ rfl, ← dPrefix_succ pc vq w i l hlt z]
    exact sub_add_cancel _ _

/-- **The honest witness, proved** (rule I). -/
theorem honestWitness : HonestWitness := by
  intro R _ n L np B n' pc vq w
  exact ⟨honest_wf pc vq w, honest_affOps pc vq w, fun c => by simp [honestWord, honestProg_isProd],
    honest_deliveries pc vq w, honest_products pc vq w, fun c z => honest_val pc vq w z c⟩

/-- Every honest cell-block is certified, and `RingKCount`'s bound `L · #certified ≤ #products` (with no delivery) is
an equality: the count is tight. -/
theorem honest_tight (pc : Fin n → Fin B → Fin L → R) (vq : Fin B → Fin L → Fin n' → R) (w : Fin np → R) :
    (certified pc vq w (honestProg pc vq w) honestWord).card = n * n' * B ∧
      L * (certified pc vq w (honestProg pc vq w) honestWord).card = products (honestProg pc vq w) := by
  obtain ⟨-, -, -, -, hprod, hval⟩ := honestWitness pc vq w
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

/-! ## Toward the count (the degree-2, fork-free fragment) -/

section Count

variable {R : Type} [CommRing R] {n L np B n' : ℕ}

/-- **Distinct certified cell-blocks have distinct word slots** (`theoremK` at no product): two on one slot would
make the difference of their targets vanish at every bit point with at most two ones, so its rank would be 0, against
`PrefixDiffRank`. -/
theorem word_injOn (hR : PrefixDiffRank) (φ : R →+* ZMod 2) (w : Fin np → R) (p₀ : Fin np) (hp₀ : φ (w p₀) = 1)
    (hL : 0 < L) (pc : Fin n → Fin B → Fin L → R) (vq : Fin B → Fin L → Fin n' → R) {N : ℕ}
    (P : SlpG (DBits n L np B n') R N) (word : DCells n n' B → Fin N) :
    Set.InjOn word (certified pc vq w P word : Set (DCells n n' B)) := by
  intro e he e' he' h
  by_contra hne
  have hc : Certified pc vq w P word e := by simpa [certified] using he
  have hc' : Certified pc vq w P word e' := by simpa [certified] using he'
  have hrank := (hR φ w p₀ hp₀ pc vq).2 e e' hne 1
  have hle := theoremK φ (∅ : Finset ℕ) (fun _ => (0 : R)) (fun _ => (⟨0, 0⟩ : BitAff (DBits n L np B n') R))
    (fun _ => ⟨0, 0⟩) ⟨0, 0⟩
    (fun z => dPrefix pc vq w e.1 e.2.1 e.2.2.val z - 1 * dPrefix pc vq w e'.1 e'.2.1 e'.2.2.val z)
    fun U hU => by
      rw [← hc U hU, ← hc' U hU, h]
      simp [BitAff.eval]
  rw [Finset.card_empty, mul_zero] at hle
  omega

/-- **A walk that starts its chain owns `L` products** (`theoremK`): if a certified block's word is `S`'s products
plus a bit-affine word, `S` has at least `L` products. -/
theorem own_card_ge_start (hR : PrefixDiffRank) (φ : R →+* ZMod 2) (w : Fin np → R) (p₀ : Fin np)
    (hp₀ : φ (w p₀) = 1) (pc : Fin n → Fin B → Fin L → R) (vq : Fin B → Fin L → Fin n' → R) {N : ℕ}
    (P : SlpG (DBits n L np B n') R N) (word : DCells n n' B → Fin N) {e : DCells n n' B}
    (he : Certified pc vq w P word e) {ι : Type} (S : Finset ι) (lam : ι → R)
    (A Bo : ι → BitAff (DBits n L np B n') R) (r : BitAff (DBits n L np B n') R)
    (hval : ∀ z, P.val z (word e) = ∑ k ∈ S, lam k * ((A k).eval z * (Bo k).eval z) + r.eval z) :
    L ≤ S.card := by
  have hle := theoremK φ S lam A Bo r (dPrefix pc vq w e.1 e.2.1 e.2.2.val) fun U hU =>
    (he U hU).symm.trans (hval _)
  have := (hR φ w p₀ hp₀ pc vq).1 e
  omega

/-- **A walk between certified blocks owns `L` products** (`theoremK` with a scale): if a certified block's word is
`c` times another certified block's word plus `S`'s products plus a bit-affine word, `S` has at least `L` products. -/
theorem own_card_ge_step (hR : PrefixDiffRank) (φ : R →+* ZMod 2) (w : Fin np → R) (p₀ : Fin np)
    (hp₀ : φ (w p₀) = 1) (pc : Fin n → Fin B → Fin L → R) (vq : Fin B → Fin L → Fin n' → R) {N : ℕ}
    (P : SlpG (DBits n L np B n') R N) (word : DCells n n' B → Fin N) {e e' : DCells n n' B} (hne : e ≠ e')
    (he : Certified pc vq w P word e) (he' : Certified pc vq w P word e') (c : R) {ι : Type} (S : Finset ι)
    (lam : ι → R) (A Bo : ι → BitAff (DBits n L np B n') R) (r : BitAff (DBits n L np B n') R)
    (hval : ∀ z, P.val z (word e) =
      c * P.val z (word e') + ∑ k ∈ S, lam k * ((A k).eval z * (Bo k).eval z) + r.eval z) :
    L ≤ S.card := by
  have hle := theoremK φ S lam A Bo r
    (fun z => dPrefix pc vq w e.1 e.2.1 e.2.2.val z - c * dPrefix pc vq w e'.1 e'.2.1 e'.2.2.val z)
    fun U hU => by
      rw [← he U hU, ← he' U hU, hval]
      ring
  have := (hR φ w p₀ hp₀ pc vq).2 e e' hne c
  omega

/-- **The split from its owners**: if the certified cell-blocks in `A` each own `L` products, pairwise disjoint, and
the rest each own a delivery (one whose `key` is the block's `f`, with `f` injective), then `RingKCount`'s split
holds. -/
theorem split_of_owners {E X D K : Type} [DecidableEq E] [DecidableEq X] [DecidableEq K] (L : ℕ)
    (cert A : Finset E) (hA : A ⊆ cert)
    (prods : Finset X) (own : E → Finset X) (hown : ∀ e ∈ A, L ≤ (own e).card ∧ own e ⊆ prods)
    (hdisj : (A : Set E).PairwiseDisjoint own) (dels : Finset D) (key : D → K) (f : E → K)
    (hdel : ∀ e ∈ cert \ A, ∃ d ∈ dels, key d = f e) (hinj : Set.InjOn f (cert \ A : Finset E)) :
    ∃ a b : ℕ, a + b = cert.card ∧ L * a ≤ prods.card ∧ b ≤ dels.card := by
  refine ⟨A.card, (cert \ A).card, ?_, ?_, ?_⟩
  · have h := Finset.card_sdiff_add_card_eq_card hA
    omega
  · calc L * A.card = ∑ _e ∈ A, L := by rw [Finset.sum_const, smul_eq_mul, mul_comm]
      _ ≤ ∑ e ∈ A, (own e).card := Finset.sum_le_sum fun e he => (hown e he).1
      _ = (A.biUnion own).card := (Finset.card_biUnion hdisj).symm
      _ ≤ prods.card := Finset.card_le_card (Finset.biUnion_subset.mpr fun e he => (hown e he).2)
  · calc (cert \ A).card = ((cert \ A).image f).card := (Finset.card_image_of_injOn hinj).symm
      _ ≤ (dels.image key).card := Finset.card_le_card fun k hk => by
          obtain ⟨e, he, rfl⟩ := Finset.mem_image.mp hk
          obtain ⟨d, hd, hdk⟩ := hdel e he
          exact Finset.mem_image.mpr ⟨d, hd, hdk⟩
      _ ≤ dels.card := Finset.card_image_le

/-- **`RingKCount` from `PrefixDiffRank`.** Each certified block's word slot is a slot of `stop` (the certified
word slots), owned by itself. Walking back from it, either it meets a delivery on a slot it owns (a delivery of its
own, since owners are functions and word slots are distinct), or its word is its own products plus a bit-affine word
plus a multiple of an earlier certified block's word, so `theoremK` and `PrefixDiffRank` give it `L` products of its
own. -/
theorem ringKCount_of_rank (hR : PrefixDiffRank) : RingKCount := by
  intro R _ φ n L np B n' w p₀ hw pc vq N P F hF hA word hword
  classical
  rcases Nat.eq_zero_or_pos L with hL | hL
  · exact ⟨(certified pc vq w P word).card, 0, rfl, by subst hL; simp, Nat.zero_le _⟩
  set cert := certified pc vq w P word with hcert
  have hcertE : ∀ e ∈ cert, Certified pc vq w P word e := fun e he => by simpa [hcert, certified] using he
  have hinj := word_injOn hR φ w p₀ hw hL pc vq P word
  set stop := cert.image word with hstopdef
  have hstop : ∀ s ∈ stop, P.isProd s = false := by
    intro s hs
    obtain ⟨e, -, rfl⟩ := Finset.mem_image.mp hs
    exact hword e
  have hAB : ∀ k : Fin N, ∃ AB : BitAff (DBits n L np B n') R × BitAff (DBits n L np B n') R,
      P.isProd k = true → ∀ z, P.val z k = AB.1.eval z * AB.2.eval z := by
    intro k
    by_cases hk : P.isProd k = true
    · obtain ⟨A, Bo, h⟩ := hA k hk
      exact ⟨(A, Bo), fun _ z => by rw [SlpG.val, ite_eq_left hk]; exact h z⟩
    · exact ⟨(⟨0, 0⟩, ⟨0, 0⟩), fun h => absurd h hk⟩
  choose AB hAB using hAB
  have hsum : ∀ (lam : Fin N → R) (o : Fin N) z, ∑ k ∈ ownProds P F stop o, lam k * P.val z k =
      ∑ k ∈ ownProds P F stop o, lam k * ((AB k).1.eval z * (AB k).2.eval z) := by
    intro lam o z
    refine Finset.sum_congr rfl fun k hk => ?_
    simp only [ownProds, Finset.mem_filter, Finset.mem_univ, true_and] at hk
    rw [hAB k hk.1 z]
  rw [deliveries_eq]
  refine split_of_owners L cert (cert.filter fun e => ¬∃ d ∈ delSet P F, owner F stop d.1 = some (word e))
    (Finset.filter_subset _ _) (univ.filter fun k => P.isProd k = true) (fun e => ownProds P F stop (word e))
    ?_ ?_ (delSet P F) (fun d => owner F stop d.1) (fun e => some (word e)) ?_ ?_
  · intro e he
    rw [Finset.mem_filter] at he
    refine ⟨?_, fun k hk => ?_⟩
    · have hwe : word e ∈ stop := Finset.mem_image_of_mem word he.1
      rcases walk P F hF stop hstop (word e) (word e) (hword e) (owner_of_mem F hwe) with
        h | ⟨lam, r, hv⟩ | ⟨a, ha, hao, c, lam, r, hv⟩
      · exact absurd h he.2
      · exact own_card_ge_start hR φ w p₀ hw pc vq P word (hcertE e he.1) _ lam (fun k => (AB k).1)
          (fun k => (AB k).2) r fun z => by rw [hv z, ← hsum]; simp
      · obtain ⟨e', he', rfl⟩ := Finset.mem_image.mp ha
        have hne : e ≠ e' := fun h => hao (h ▸ rfl)
        exact own_card_ge_step hR φ w p₀ hw pc vq P word hne (hcertE e he.1) (hcertE e' he') c _ lam
          (fun k => (AB k).1) (fun k => (AB k).2) r fun z => by rw [hv z, ← hsum]
    · simp only [ownProds, Finset.mem_filter, Finset.mem_univ, true_and] at hk ⊢
      exact hk.1
  · intro e he e' he' hne
    simp only [Function.onFun]
    rw [Finset.disjoint_left]
    intro k hk hk'
    simp only [ownProds, Finset.mem_filter, Finset.mem_univ, true_and] at hk hk'
    exact hne (hinj (Finset.mem_filter.mp (Finset.mem_coe.mp he)).1 (Finset.mem_filter.mp (Finset.mem_coe.mp he')).1
      (Option.some_injective _ (hk.2.symm.trans hk'.2)))
  · intro e he
    obtain ⟨he1, he2⟩ := Finset.mem_sdiff.mp he
    rw [Finset.mem_filter, not_and, not_not] at he2
    exact he2 he1
  · intro e he e' he' h
    exact hinj (Finset.mem_sdiff.mp (Finset.mem_coe.mp he)).1 (Finset.mem_sdiff.mp (Finset.mem_coe.mp he')).1
      (Option.some_injective _ h)

end Count

end Pouw.RingFragment
