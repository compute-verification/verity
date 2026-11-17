module

public import Definitions.Pouw.TileBound.Defs
public import Mathlib.Algebra.Field.Rat
public import Mathlib.Data.Matrix.Mul
import Mathlib.LinearAlgebra.Matrix.Rank

/-! The FP4-tile model's block-support count `quad_support` and the cost bounds it gives, down to programs of
`sm120Nvf4` gate firings. -/

@[expose] public section

namespace Pouw.TileBound

open Finset Pouw.Fp8Atom Pouw.PearlC.Fp4

variable {K : Type*} [Field K] {m k n : ℕ} {τ : Type*} [Fintype τ] {β : Type*} [Fintype β] {γ : Type*}
  [Fintype γ]

omit [Fintype τ] [Fintype β] [Fintype γ] in
theorem card_le_of_mul_eq_one {ι κ : Type*} [Fintype ι] [Fintype κ] [DecidableEq ι] (A : Matrix ι κ K)
    (B : Matrix κ ι K) (h : A * B = 1) : Fintype.card ι ≤ Fintype.card κ := by
  have := Matrix.rank_mul_le_left A B
  rw [h, Matrix.rank_one] at this
  exact this.trans (Matrix.rank_le_card_width A)

omit [Fintype β] [Fintype γ] in
open Classical in
theorem block_count (P : QuadProg K m k n τ) {liveX : Fin m × Fin k → Prop} {liveY : Fin k × Fin n → Prop}
    (h : P.CrossOn liveX liveY) (B : XBlocks m k β) (b : β) :
    Fintype.card {x // LiveIn B liveX liveY b x} ≤ Fintype.card {q // SideIn P B b q} := by
  let A : Matrix {x // LiveIn B liveX liveY b x} {q // SideIn P B b q} K := fun x q =>
    (if q.1.2 then P.a' q.1.1 (x.1.1, x.1.2.1) else P.a q.1.1 (x.1.1, x.1.2.1)) * P.w q.1.1 (x.1.1, x.1.2.2)
  let C : Matrix {q // SideIn P B b q} {x // LiveIn B liveX liveY b x} K := fun q y =>
    if q.1.2 then P.b q.1.1 (y.1.2.1, y.1.2.2) else P.b' q.1.1 (y.1.2.1, y.1.2.2)
  refine card_le_of_mul_eq_one A C ?_
  ext x y
  obtain ⟨⟨i, l, j'⟩, hxb, hxX, hxY⟩ := x
  obtain ⟨⟨i2, l', j⟩, hyb, hyX, hyY⟩ := y
  have hi : i = i2 := B.inRow _ _ (hxb.trans hyb.symm)
  subst hi
  have hext : ∀ q : τ × Bool, ¬ SideIn P B b q →
      (if q.2 then P.a' q.1 (i, l) else P.a q.1 (i, l)) = 0 := by
    intro q hq
    rcases q with ⟨t, s⟩
    cases s <;> simp only [SideIn, XBlocks.Touches, ite_true, ite_false, Bool.false_eq_true] at hq ⊢ <;>
      · by_contra hne
        exact hq ⟨(i, l), hxb, hne⟩
  rw [Matrix.mul_apply]
  let F : τ × Bool → K := fun q => (if q.2 then P.a' q.1 (i, l) else P.a q.1 (i, l)) * P.w q.1 (i, j') *
    (if q.2 then P.b q.1 (l', j) else P.b' q.1 (l', j))
  have hsum : (∑ q : {q // SideIn P B b q}, F q.1) = ∑ q : τ × Bool, F q := by
    rw [← Finset.sum_subtype (univ.filter (SideIn P B b)) (by simp) F]
    apply Finset.sum_subset (Finset.filter_subset _ _)
    intro q _ hq
    simp only [F]
    rw [hext q (by simpa using hq)]
    ring
  change (∑ q : {q // SideIn P B b q}, F q.1) = _
  rw [hsum, Fintype.sum_prod_type]
  simp only [F, Fintype.sum_bool, ite_true, ite_false, Bool.false_eq_true]
  have hc := h (i, l) hxX (l', j) hyY (i, j')
  have hre : (∑ t, (P.a' t (i, l) * P.w t (i, j') * P.b t (l', j) +
      P.a t (i, l) * P.w t (i, j') * P.b' t (l', j))) =
      ∑ t, P.w t (i, j') * (P.a t (i, l) * P.b' t (l', j) + P.a' t (i, l) * P.b t (l', j)) := by
    apply Finset.sum_congr rfl
    intro t _
    ring
  rw [hre, hc, Matrix.one_apply]
  by_cases hl : l = l' <;> by_cases hj : j = j' <;>
    simp [hl, hj, Subtype.ext_iff, Prod.ext_iff, eq_comm (a := j')]

omit [Fintype γ] in
theorem crossX_count (P : QuadProg K m k n τ) {liveX : Fin m × Fin k → Prop} {liveY : Fin k × Fin n → Prop}
    (h : P.CrossOn liveX liveY) (B : XBlocks m k β) :
    nLive liveX liveY ≤ ∑ t, (B.touches (P.a t) + B.touches (P.a' t)) := by
  classical
  have hfib : nLive liveX liveY = ∑ b, Fintype.card {x // LiveIn B liveX liveY b x} := by
    unfold nLive
    rw [Fintype.card_subtype, Finset.card_eq_sum_card_fiberwise (f := fun x => B.blk (x.1, x.2.1)) (t := univ)
      (fun _ _ => mem_coe.2 (mem_univ _))]
    apply Finset.sum_congr rfl
    intro b _
    rw [Fintype.card_subtype]
    congr 1
    ext x
    simp [LiveIn]
    tauto
  have hdc : ∑ b, Fintype.card {q // SideIn P B b q} =
      ∑ t, (B.touches (P.a t) + B.touches (P.a' t)) := by
    simp only [Fintype.card_subtype, Finset.card_filter]
    rw [Finset.sum_comm, Fintype.sum_prod_type]
    apply Finset.sum_congr rfl
    intro t _
    rw [Fintype.sum_bool, add_comm]
    simp only [SideIn, XBlocks.touches, Finset.card_filter, ite_true, ite_false, Bool.false_eq_true]
  rw [hfib, ← hdc]
  exact Finset.sum_le_sum fun b _ => block_count P h B b

/-- The transposed program, for `Yᵀ·Xᵀ = (X·Y)ᵀ`: Y's forms become X's. -/
def QuadProg.transpose (P : QuadProg K m k n τ) : QuadProg K n k m τ where
  a t e := P.b' t (e.2, e.1)
  b t e := P.a' t (e.2, e.1)
  a' t e := P.b t (e.2, e.1)
  b' t e := P.a t (e.2, e.1)
  w t o := P.w t (o.2, o.1)

omit [Fintype β] [Fintype γ] in
theorem QuadProg.crossOn_transpose (P : QuadProg K m k n τ) {liveX : Fin m × Fin k → Prop}
    {liveY : Fin k × Fin n → Prop} (h : P.CrossOn liveX liveY) :
    P.transpose.CrossOn (fun e => liveY (e.2, e.1)) (fun e => liveX (e.2, e.1)) := by
  intro x hx y hy o
  have hc := h (y.2, y.1) hy (x.2, x.1) hx (o.2, o.1)
  simp only [QuadProg.transpose]
  have hre : (∑ t, P.w t (o.2, o.1) * (P.b' t (x.2, x.1) * P.a t (y.2, y.1) + P.b t (x.2, x.1) * P.a' t (y.2, y.1))) =
      ∑ t, P.w t (o.2, o.1) * (P.a t (y.2, y.1) * P.b' t (x.2, x.1) + P.a' t (y.2, y.1) * P.b t (x.2, x.1)) := by
    apply Finset.sum_congr rfl
    intro t _
    ring
  rw [hre, hc]
  apply if_congr _ rfl rfl
  exact ⟨fun ⟨h1, h2, h3⟩ => ⟨h3, h2.symm, h1⟩, fun ⟨h1, h2, h3⟩ => ⟨h3, h2.symm, h1⟩⟩

/-- Y-blocks read as the transposed program's X-blocks. -/
def YBlocks.toXT (B : YBlocks k n β) : XBlocks n k β := ⟨fun e => B.blk (e.2, e.1), fun _ _ h => B.inCol _ _ h⟩

omit [Fintype τ] [Fintype γ] in
theorem YBlocks.touches_toXT (B : YBlocks k n β) (f : Fin k × Fin n → K) :
    B.toXT.touches (fun e => f (e.2, e.1)) = B.touches f := by
  classical
  have hP : ∀ b, B.toXT.Touches (fun e => f (e.2, e.1)) b ↔ B.Touches f b := fun b =>
    ⟨fun ⟨e, h1, h2⟩ => ⟨(e.2, e.1), h1, h2⟩, fun ⟨e, h1, h2⟩ => ⟨(e.2, e.1), h1, h2⟩⟩
  simp only [XBlocks.touches, YBlocks.touches, hP]

omit [Fintype τ] [Fintype β] [Fintype γ] in
theorem nLive_transpose (liveX : Fin m × Fin k → Prop) (liveY : Fin k × Fin n → Prop) :
    nLive (fun e => liveY (e.2, e.1)) (fun e => liveX (e.2, e.1)) = nLive liveX liveY := by
  classical
  unfold nLive
  let e : Fin n × Fin k × Fin m ≃ Fin m × Fin k × Fin n :=
    ⟨fun x => (x.2.2, x.2.1, x.1), fun x => (x.2.2, x.2.1, x.1), fun _ => rfl, fun _ => rfl⟩
  exact Fintype.card_congr (Equiv.subtypeEquiv e fun x => by simp [e, and_comm])

omit [Fintype β] in
theorem crossY_count (P : QuadProg K m k n τ) {liveX : Fin m × Fin k → Prop} {liveY : Fin k × Fin n → Prop}
    (h : P.CrossOn liveX liveY) (B : YBlocks k n γ) :
    nLive liveX liveY ≤ ∑ t, (B.touches (P.b t) + B.touches (P.b' t)) := by
  have hx := crossX_count P.transpose (P.crossOn_transpose h) B.toXT
  rw [nLive_transpose] at hx
  refine hx.trans (le_of_eq (Finset.sum_congr rfl fun t _ => ?_))
  change B.toXT.touches (fun e => P.b' t (e.2, e.1)) + B.toXT.touches (fun e => P.b t (e.2, e.1)) = _
  rw [B.touches_toXT, B.touches_toXT, add_comm]

theorem quad_support (P : QuadProg K m k n τ) {liveX : Fin m × Fin k → Prop} {liveY : Fin k × Fin n → Prop}
    (h : P.CrossOn liveX liveY) (BX : XBlocks m k β) (BY : YBlocks k n γ) :
    2 * nLive liveX liveY ≤ ∑ t, ((BX.touches (P.a t) + BY.touches (P.b t)) +
      (BX.touches (P.a' t) + BY.touches (P.b' t))) := by
  have hx := crossX_count P h BX
  have hy := crossY_count P h BY
  have : ∑ t, ((BX.touches (P.a t) + BY.touches (P.b t)) + (BX.touches (P.a' t) + BY.touches (P.b' t))) =
      ∑ t, (BX.touches (P.a t) + BX.touches (P.a' t)) + ∑ t, (BY.touches (P.b t) + BY.touches (P.b' t)) := by
    rw [← Finset.sum_add_distrib]
    apply Finset.sum_congr rfl
    intro t _
    ring
  rw [this]
  omega

omit [Fintype τ] [Fintype γ] in
theorem XBlocks.touches_zero (B : XBlocks m k β) : B.touches (0 : Fin m × Fin k → K) = 0 := by
  classical
  simp [XBlocks.touches, XBlocks.Touches]

omit [Fintype γ] in
theorem bilin_support (P : BilinProg K m k n τ) {liveX : Fin m × Fin k → Prop} {liveY : Fin k × Fin n → Prop}
    (h : P.toQuad.CrossOn liveX liveY) (BX : XBlocks m k β) :
    nLive liveX liveY ≤ ∑ t, BX.touches (P.u t) := by
  have hx := crossX_count P.toQuad h BX
  simpa [BilinProg.toQuad, XBlocks.touches_zero] using hx

theorem cost_ge_supportAt (P : QuadProg K m k n τ) {liveX : Fin m × Fin k → Prop} {liveY : Fin k × Fin n → Prop}
    (h : P.CrossOn liveX liveY) (BX : XBlocks m k β) (BY : YBlocks k n γ) (s : τ → ℕ)
    (hs : ∀ t, BX.touches (P.a t) + BY.touches (P.b t) ≤ s t ∧ BX.touches (P.a' t) + BY.touches (P.b' t) ≤ s t)
    (c : τ → ℚ) (p₀ : ℚ) (hp₀ : 0 ≤ p₀) (hf : ∀ t, (s t : ℚ) * p₀ ≤ c t) :
    p₀ * nLive liveX liveY ≤ ∑ t, c t := by
  have hq := quad_support P h BX BY
  have hcap : ∑ t, ((BX.touches (P.a t) + BY.touches (P.b t)) + (BX.touches (P.a' t) + BY.touches (P.b' t))) ≤
      ∑ t, 2 * s t :=
    Finset.sum_le_sum fun t _ => by have := hs t; omega
  have hN : nLive liveX liveY ≤ ∑ t, s t := by
    have h2 := hq.trans hcap
    rw [← Finset.mul_sum] at h2
    omega
  calc p₀ * (nLive liveX liveY : ℚ) ≤ p₀ * ∑ t, (s t : ℚ) := by
        gcongr
        exact_mod_cast hN
    _ = ∑ t, (s t : ℚ) * p₀ := by
        rw [Finset.mul_sum]
        exact Finset.sum_congr rfl fun t _ => mul_comm _ _
    _ ≤ ∑ t, c t := Finset.sum_le_sum fun t _ => hf t

theorem cost_ge_support (P : SlotProg K m k n τ) {liveX : Fin m × Fin k → Prop} {liveY : Fin k × Fin n → Prop}
    (h : P.toQuadProg.CrossOn liveX liveY) (BX : XBlocks m k β) (BY : YBlocks k n γ) (s : τ → ℕ)
    (hs : ∀ t, BX.touches (P.a t) + BY.touches (P.b t) ≤ s t ∧ BX.touches (P.a' t) + BY.touches (P.b' t) ≤ s t)
    (hf : ∀ t, (s t : ℚ) * price .fp4 ≤ price (P.fmt t)) :
    price .fp4 * nLive liveX liveY ≤ P.cost :=
  cost_ge_supportAt P.toQuadProg h BX BY s hs (fun t => price (P.fmt t)) (price .fp4) (by norm_num [price]) hf

theorem cost_ge (P : SlotProg K m k n τ) {liveX : Fin m × Fin k → Prop} {liveY : Fin k × Fin n → Prop}
    (h : P.toQuadProg.CrossOn liveX liveY) (BX : XBlocks m k β) (BY : YBlocks k n γ) (hr : P.Representable BX BY)
    (hf : ∀ t, (capacity (P.fmt t) : ℚ) * price .fp4 ≤ price (P.fmt t)) :
    price .fp4 * nLive liveX liveY ≤ P.cost :=
  cost_ge_support P h BX BY (fun t => capacity (P.fmt t)) hr hf

theorem fmt_closed_iff (f : Fmt) :
    (capacity f : ℚ) * price .fp4 ≤ price f ↔ f = .fp4 ∨ f = .fp8 ∨ f = .bf16 := by
  cases f <;> norm_num [capacity, price]

theorem open_tf32 : price .tf32 < capacity .tf32 * price .fp4 := by norm_num [capacity, price]

theorem open_fp32 : price .fp32 < capacity .fp32 * price .fp4 := by norm_num [capacity, price]

theorem tf32_closed_upto : (8 : ℚ) * price .fp4 ≤ price .tf32 := by norm_num [price]

theorem fp32_closed_upto : (16 : ℚ) * price .fp4 ≤ price .fp32 := by norm_num [price]

theorem tf32_open_at : price .tf32 < (9 : ℕ) * price .fp4 := by norm_num [price]

theorem fp32_open_at : price .fp32 < (17 : ℕ) * price .fp4 := by norm_num [price]

theorem fp16_closed_upto : (4 : ℚ) * price .fp4 ≤ price .fp16 := by norm_num [price]

theorem fp16_open_at : price .fp16 < (5 : ℕ) * price .fp4 := by norm_num [price]

theorem int8_closed_upto : (2 : ℚ) * price .fp4 ≤ price .int8 := by norm_num [price]

theorem int8_open_at : price .int8 < (3 : ℕ) * price .fp4 := by norm_num [price]

theorem dp4a_closed_upto : (8 : ℚ) * price .fp4 ≤ price .dp4a := by norm_num [price]

theorem dp4a_open_at : price .dp4a < (9 : ℕ) * price .fp4 := by norm_num [price]

theorem open_fp16 : price .fp16 < capacity .fp16 * price .fp4 := by norm_num [capacity, price]

theorem open_int8 : price .int8 < capacity .int8 * price .fp4 := by norm_num [capacity, price]

theorem open_dp4a : price .dp4a < capacity .dp4a * price .fp4 := by norm_num [capacity, price]

omit [Fintype τ] [Fintype β] [Fintype γ] in
/-- Wherever `c < s · p₀`, numbers that meet the support count at a total price below `p₀` per live product: not an
exhibited program. -/
theorem support_admits_openAt {c p₀ : ℚ} {s : ℕ} (hf : c < s * p₀) (M : ℕ) (hM : 0 < M) :
    2 * (s * M) ≤ ∑ _t : Fin M, (s + s) ∧ (∑ _t : Fin M, c) < p₀ * ((s * M : ℕ) : ℚ) := by
  refine ⟨by simp; ring_nf; omega, ?_⟩
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, Nat.cast_mul]
  have hM' : (0 : ℚ) < M := by exact_mod_cast hM
  nlinarith

omit [Fintype τ] [Fintype β] [Fintype γ] in
/-- The support count can't rule out a cost below FP4's where `price f < s · price .fp4`: the pin exhibits numbers that
pass the count, not a program. Ruling them out needs a matrix-multiplication rank bound of order
`m·k·n·price .fp4 / price f`; the exactness-depth argument that closes known recursive algorithms is not in Lean. -/
theorem support_admits_open {f : Fmt} {s : ℕ} (hf : price f < s * price .fp4) (M : ℕ) (hM : 0 < M) :
    2 * (s * M) ≤ ∑ _t : Fin M, (s + s) ∧ (∑ _t : Fin M, price f) < price .fp4 * ((s * M : ℕ) : ℚ) :=
  support_admits_openAt hf M hM

omit [Fintype τ] [Fintype β] [Fintype γ] in
theorem honest_crossOn : (honest K m k n).toQuadProg.CrossOn (fun _ => True) (fun _ => True) := by
  intro x _ y _ o
  simp only [honest, unitForm, Pi.zero_apply, mul_zero, add_zero]
  rw [Finset.sum_eq_single (x.1, x.2, y.2)]
  · by_cases h1 : x.1 = o.1 <;> by_cases h2 : x.2 = y.1 <;> by_cases h3 : y.2 = o.2 <;>
      simp [h1, h2, h3, Prod.ext_iff, eq_comm (a := o.1), eq_comm (a := o.2), eq_comm (a := y.1)]
  · intro t _ ht
    by_cases hx : x = (t.1, t.2.1)
    · by_cases hy : y = (t.2.1, t.2.2)
      · exfalso
        apply ht
        rw [hx, hy]
      · simp [hy]
    · simp [hx]
  · intro h
    exact absurd (mem_univ _) h

omit [Fintype τ] [Fintype β] [Fintype γ] in
theorem nLive_all : nLive (m := m) (k := k) (n := n) (fun _ => True) (fun _ => True) = m * k * n := by
  classical
  simp [nLive, Fintype.card_subtype, Fintype.card_prod, Fintype.card_fin, mul_assoc]

omit [Fintype τ] [Fintype β] [Fintype γ] in
theorem honest_cost : (honest K m k n).cost = price .fp4 * (m * k * n : ℕ) := by
  simp [SlotProg.cost, honest, Finset.sum_const, Finset.card_univ, Fintype.card_prod, Fintype.card_fin]
  ring

omit [Fintype τ] [Fintype γ] in
theorem XBlocks.touches_unit [DecidableEq β] (B : XBlocks m k β) (e₀ : Fin m × Fin k) :
    B.touches (unitForm (K := K) e₀) = 1 := by
  classical
  unfold XBlocks.touches
  rw [Finset.card_eq_one]
  refine ⟨B.blk e₀, Finset.eq_singleton_iff_unique_mem.2 ⟨Finset.mem_filter.2 ⟨mem_univ _, e₀, rfl, by simp [unitForm]⟩,
    fun b hb => ?_⟩⟩
  obtain ⟨_, e, he, hne⟩ := Finset.mem_filter.1 hb
  by_cases h : e = e₀
  · rw [← he, h]
  · exact absurd (by simp [unitForm, h]) hne

omit [Fintype τ] [Fintype β] in
theorem YBlocks.touches_unit [DecidableEq γ] (B : YBlocks k n γ) (e₀ : Fin k × Fin n) :
    B.touches (unitForm (K := K) e₀) = 1 := by
  classical
  unfold YBlocks.touches
  rw [Finset.card_eq_one]
  refine ⟨B.blk e₀, Finset.eq_singleton_iff_unique_mem.2 ⟨Finset.mem_filter.2 ⟨mem_univ _, e₀, rfl, by simp [unitForm]⟩,
    fun b hb => ?_⟩⟩
  obtain ⟨_, e, he, hne⟩ := Finset.mem_filter.1 hb
  by_cases h : e = e₀
  · rw [← he, h]
  · exact absurd (by simp [unitForm, h]) hne

omit [Fintype τ] [Fintype β] in
theorem YBlocks.touches_zero (B : YBlocks k n γ) : B.touches (0 : Fin k × Fin n → K) = 0 := by
  classical
  simp [YBlocks.touches, YBlocks.Touches]

omit [Fintype τ] in
theorem honest_representable [DecidableEq β] [DecidableEq γ] (BX : XBlocks m k β) (BY : YBlocks k n γ) :
    (honest K m k n).Representable BX BY := by
  intro t
  simp only [honest, XBlocks.touches_unit, YBlocks.touches_unit, XBlocks.touches_zero, YBlocks.touches_zero,
    capacity]
  omega

omit [Fintype τ] [Fintype γ] in
theorem tile_count {G : ℕ} (P : TileProg K m k n G) {liveX : Fin m × Fin k → Prop} {liveY : Fin k × Fin n → Prop}
    (h : P.toBilin.toQuad.CrossOn liveX liveY) (BX : XBlocks m k β) (hr : P.Representable BX) :
    nLive liveX liveY ≤ G * gateSlots := by
  have hs := bilin_support P.toBilin h BX
  have hle : ∑ s, BX.touches (P.toBilin.u s) ≤ ∑ _s : Fin G × Fin 16 × Fin 64 × Fin 8, 1 :=
    Finset.sum_le_sum fun s _ => hr s.1 s.2.1 s.2.2.1
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_prod, Fintype.card_fin, smul_eq_mul, mul_one] at hle
  unfold gateSlots
  omega

omit [Fintype τ] [Fintype γ] in
theorem tile_cost_ge {G : ℕ} (P : TileProg K m k n G) {liveX : Fin m × Fin k → Prop}
    {liveY : Fin k × Fin n → Prop} (h : P.toBilin.toQuad.CrossOn liveX liveY) (BX : XBlocks m k β)
    (hr : P.Representable BX) : price .fp4 * nLive liveX liveY ≤ P.cost := by
  have hc := tile_count P h BX hr
  have hc' : (nLive liveX liveY : ℚ) ≤ G * gateSlots := by exact_mod_cast hc
  unfold TileProg.cost gatePrice
  have hp : (0 : ℚ) ≤ price .fp4 := by norm_num [price]
  nlinarith

theorem wordValue_zero : wordValue 0 = some 0 := by
  unfold wordValue fp32Value
  norm_num [Value.finite?, Term.value, Term.zero]

omit [Fintype τ] [Fintype β] [Fintype γ] in
theorem realizes_out_at {G : ℕ} (P : TileProg ℚ m k n G) (gs : Fin G → Firing) (X : Fin m × Fin k → ℚ)
    (Y : Fin k × Fin n → ℚ) (hR : P.RealizesAt gs X Y) (o : Fin m × Fin n) :
    P.outAt gs o = P.toBilin.toQuad.eval X Y o := by
  have hw : ∀ g i j, (((gs g).word i j).bind wordValue).getD 0 =
      ∑ l, (∑ e, P.u g i l e * X e) * (∑ e, P.v g l j e * Y e) := by
    intro g i j
    obtain ⟨hex, hacc, hu, hv⟩ := hR g
    rw [hex i j, hacc i j, wordValue_zero]
    simp only [Option.getD_some, zero_add]
    exact Finset.sum_congr rfl fun l _ => by rw [Firing.slotValue, hu, hv]
  simp only [TileProg.outAt, hw, QuadProg.eval, BilinProg.toQuad, TileProg.toBilin, Pi.zero_apply, zero_mul,
    Finset.sum_const_zero, add_zero, zero_add, Fintype.sum_prod_type]
  refine Finset.sum_congr rfl fun g _ => Finset.sum_congr rfl fun i _ => ?_
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [Finset.mul_sum]
  exact Finset.sum_congr rfl fun l _ => by ring

omit [Fintype β] [Fintype γ] in
/-- The three inputs are FP4-representable (E2M1 has 0 and 1, and UE4M3 a scale 1), so a tile program's firings can
carry them. -/
theorem crossOn_of_eval (P : QuadProg K m k n τ) {liveX : Fin m × Fin k → Prop} {liveY : Fin k × Fin n → Prop}
    (h : ∀ x, liveX x → ∀ y, liveY y → ∀ o,
      P.eval (unitForm x) (unitForm y) o = matMul (unitForm x) (unitForm y) o ∧
      P.eval (unitForm x) 0 o = matMul (unitForm x) 0 o ∧
      P.eval 0 (unitForm y) o = matMul 0 (unitForm y) o) :
    P.CrossOn liveX liveY := by
  classical
  intro x hx y hy o
  obtain ⟨e11, e10, e01⟩ := h x hx y hy o
  have hsx : ∀ f : Fin m × Fin k → K, ∑ e, f e * unitForm x e = f x := fun f => by simp [unitForm]
  have hsy : ∀ f : Fin k × Fin n → K, ∑ e, f e * unitForm y e = f y := fun f => by simp [unitForm]
  simp only [QuadProg.eval, hsx, hsy, Pi.zero_apply, mul_zero, Finset.sum_const_zero, add_zero, zero_add] at e11 e10 e01
  have hm10 : matMul (unitForm (K := K) x) (0 : Fin k × Fin n → K) o = 0 := by simp [matMul]
  have hm01 : matMul (0 : Fin m × Fin k → K) (unitForm (K := K) y) o = 0 := by simp [matMul]
  have hm11 : matMul (unitForm (K := K) x) (unitForm (K := K) y) o =
      if x.1 = o.1 ∧ x.2 = y.1 ∧ y.2 = o.2 then 1 else 0 := by
    unfold matMul unitForm
    rw [Finset.sum_eq_single x.2]
    · by_cases h1 : x.1 = o.1 <;> by_cases h2 : x.2 = y.1 <;> by_cases h3 : y.2 = o.2 <;>
        simp [h1, h2, h3, Prod.ext_iff, eq_comm (a := o.1), eq_comm (a := o.2)]
    · intro l _ hl
      have : (o.1, l) ≠ x := fun he => hl (by rw [← he])
      simp [this]
    · intro h'
      exact absurd (mem_univ _) h'
  rw [hm10] at e10
  rw [hm01] at e01
  rw [hm11] at e11
  rw [← e11]
  have : ∑ t, P.w t o * (P.a t x * P.b' t y + P.a' t x * P.b t y) =
      (∑ t, P.w t o * (P.a t x + P.b t y) * (P.a' t x + P.b' t y)) - (∑ t, P.w t o * P.a t x * P.a' t x) -
        (∑ t, P.w t o * P.b t y * P.b' t y) := by
    rw [← Finset.sum_sub_distrib, ← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun t _ => by ring
  rw [this, e10, e01]
  ring

omit [Fintype τ] [Fintype γ] in
theorem tile_cost_ge_of_gates {G : ℕ} (P : TileProg ℚ m k n G) {liveX : Fin m × Fin k → Prop}
    {liveY : Fin k × Fin n → Prop} (BX : XBlocks m k β) (hr : P.Representable BX)
    (hg : ∀ x, liveX x → ∀ y, liveY y → ∃ g11 g10 g01 : Fin G → Firing,
      P.RealizesAt g11 (unitForm x) (unitForm y) ∧ P.RealizesAt g10 (unitForm x) 0 ∧
        P.RealizesAt g01 0 (unitForm y) ∧
      ∀ o, P.outAt g11 o = matMul (unitForm x) (unitForm y) o ∧ P.outAt g10 o = matMul (unitForm x) 0 o ∧
        P.outAt g01 o = matMul 0 (unitForm y) o) :
    price .fp4 * nLive liveX liveY ≤ P.cost := by
  refine tile_cost_ge P (crossOn_of_eval _ fun x hx y hy o => ?_) BX hr
  obtain ⟨g11, g10, g01, h11, h10, h01, hout⟩ := hg x hx y hy
  obtain ⟨o11, o10, o01⟩ := hout o
  exact ⟨(realizes_out_at P g11 _ _ h11 o).symm.trans o11, (realizes_out_at P g10 _ _ h10 o).symm.trans o10,
    (realizes_out_at P g01 _ _ h01 o).symm.trans o01⟩

/-- A one-gate witness: code 1.0 (nibble 2) at `P[0, 0]` and at `Q[0, 0]`, every other code 0, every scale byte
`0x38` (UE4M3 1.0), accumulators `+0`. -/
def oneGate : Firing where
  p i l := if i.val = 0 ∧ l.val = 0 then 2 else 0
  ps _ _ := 0x38
  q l j := if l.val = 0 ∧ j.val = 0 then 2 else 0
  qs _ _ := 0x38
  acc _ _ := 0

theorem oneGate_exact : oneGate.Exact := by
  unfold Firing.Exact
  decide +kernel

theorem oneGate_p : ∀ i l, elemValue (oneGate.p i l) (oneGate.ps i (scaleIdx l)) =
    if (i, l) = ((0 : Fin 16), (0 : Fin 64)) then 1 else 0 := by decide +kernel

theorem oneGate_q : ∀ l j, elemValue (oneGate.q l j) (oneGate.qs j (scaleIdx l)) =
    if (l, j) = ((0 : Fin 64), (0 : Fin 8)) then 1 else 0 := by decide +kernel

omit [Fintype τ] [Fintype β] [Fintype γ] in
theorem sum_unitForm_mul {ι : Type*} [Fintype ι] [DecidableEq ι] (a b : ι) :
    ∑ e, unitForm (K := ℚ) a e * unitForm b e = if a = b then 1 else 0 := by
  by_cases h : a = b <;> simp [unitForm, Finset.sum_ite_eq', h, Ne.symm]

def oneGateProg : TileProg ℚ 16 64 8 1 :=
  ⟨fun _ i l => unitForm (i, l), fun _ l j => unitForm (l, j), fun _ i j => unitForm (i, j)⟩

theorem oneGateProg_realizesAt :
    oneGateProg.RealizesAt (fun _ => oneGate) (unitForm (0, 0)) (unitForm (0, 0)) := by
  intro _
  refine ⟨oneGate_exact, fun _ _ => rfl, fun i l => ?_, fun l j => ?_⟩
  · rw [oneGate_p]; simp only [oneGateProg, sum_unitForm_mul]
  · rw [oneGate_q]; simp only [oneGateProg, sum_unitForm_mul]

theorem realizesAt_nonvacuous : ∃ (P : TileProg ℚ 16 64 8 1) (gs : Fin 1 → Firing),
    P.RealizesAt gs (unitForm (0, 0)) (unitForm (0, 0)) ∧ P.outAt gs (0, 0) = 1 := by
  refine ⟨oneGateProg, fun _ => oneGate, oneGateProg_realizesAt, ?_⟩
  rw [realizes_out_at _ _ _ _ oneGateProg_realizesAt]
  simp [QuadProg.eval, BilinProg.toQuad, TileProg.toBilin, oneGateProg, unitForm, Fintype.sum_prod_type]

end Pouw.TileBound
