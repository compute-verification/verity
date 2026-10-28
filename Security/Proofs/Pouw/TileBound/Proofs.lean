import Definitions.Pouw.TileBound.Defs

/-!
# The FP4-tile model: the block-support count and what it bounds (proofs)

`docs/pouw/fp4-tile-model.md` §2.

* `crossX_count`: for every quadratic program correct on the live entries, the live count is at most the number of
  (product, side, X-block) incidences, `Σ_t (touches (a t) + touches (a' t))`. Per X-block the matrix-product tensor
  restricted to that block is a vector–matrix product of rank `|block|·(live columns)`, and each product touching the
  block adds rank at most one (`block_count`); the proof is `card ι ≤ card κ` from `A * B = 1` over `K`.
* `crossY_count`: the same for Y-blocks, by transposing the program (`QuadProg.transpose`).
* `quad_support`: the two together, `2·nLive ≤ Σ_t (touches ℓ t + touches ℓ' t)`. Quadratic programs lose nothing
  against bilinear ones in this count.
* `cost_ge_support`: a slot program whose slots' inputs read at most `s t` blocks costs at least `price .fp4 · nLive`
  whenever each slot's price covers `s t` FP4 slots. `cost_ge` is its instance at the representability rule's
  capacities, which `fmt_closed_iff` says closes exactly FP4, FP8 and BF16. The others close up to a support per input
  (the `*_closed_upto` pins): int8 2 blocks, FP16 4, TF32 8, `dp4a` 8, FFMA 16 (Strassen depth 1, 2, 3, 3, 4). FFMA's 16
  is a tie at the bottom of its measured price range (8.0–8.15 units), not a margin.
* `honest_crossOn`, `honest_cost`: the honest dense program is correct and meets the bound (`nLive_all`).
* `tile_cost_ge`: a tile program costs at least `price .fp4 · nLive`. `realizes_out_at` says that at an input where
  exact `sm120Nvf4` firings realize it, its decoded output is its slots' bilinear program. `realizesAt_nonvacuous` is a
  kernel-checked firing that realizes a program with a nonzero output.
* `crossOn_of_eval`: correct outputs at three unit and zero inputs give `CrossOn`. `tile_cost_ge_of_gates` chains them:
  realizing firings with correct outputs there give the cost bound.
* The shared cost lemma `cost_ge_supportAt` and `support_admits_openAt` are here; `cost_ge_support`
  and `support_admits_open` are their instances at FP4's price.
* `support_admits_open`: why wider pre-adds aren't closed. From 9 TF32 blocks, 5 FP16, 3 int8, 9 `dp4a` or 17 FFMA blocks
  per input (the `*_open_at` pins), the support count can't rule out a cost profile below FP4's. The pin exhibits
  numbers that pass the count, not a program.
-/

namespace Pouw.TileBound

open Finset Pouw.Fp8Atom Pouw.PearlC.Fp4

variable {K : Type*} [Field K] {m k n : ℕ} {τ : Type*} [Fintype τ] {β : Type*} [Fintype β] {γ : Type*}
  [Fintype γ]

/-! ## The rank step -/

omit [Fintype τ] [Fintype β] [Fintype γ] in
/-- If `A * B = 1` then `A`'s row count is at most its column count. -/
theorem card_le_of_mul_eq_one {ι κ : Type*} [Fintype ι] [Fintype κ] [DecidableEq ι] (A : Matrix ι κ K)
    (B : Matrix κ ι K) (h : A * B = 1) : Fintype.card ι ≤ Fintype.card κ := by
  have := Matrix.rank_mul_le_left A B
  rw [h, Matrix.rank_one] at this
  exact this.trans (Matrix.rank_le_card_width A)

/-! ## The block count, X-side -/

omit [Fintype β] [Fintype γ] in
open Classical in
/-- **Per block.** The live products in X-block `b` are at most the (product, side) pairs whose X-part reads `b`. -/
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
/-- **The X-side count.** A quadratic program correct on the live entries has at least `nLive` (product, side, X-block)
incidences. -/
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

/-! ## The Y-side, by transposition -/

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
/-- **The Y-side count.** -/
theorem crossY_count (P : QuadProg K m k n τ) {liveX : Fin m × Fin k → Prop} {liveY : Fin k × Fin n → Prop}
    (h : P.CrossOn liveX liveY) (B : YBlocks k n γ) :
    nLive liveX liveY ≤ ∑ t, (B.touches (P.b t) + B.touches (P.b' t)) := by
  have hx := crossX_count P.transpose (P.crossOn_transpose h) B.toXT
  rw [nLive_transpose] at hx
  refine hx.trans (le_of_eq (Finset.sum_congr rfl fun t _ => ?_))
  change B.toXT.touches (fun e => P.b' t (e.2, e.1)) + B.toXT.touches (fun e => P.b t (e.2, e.1)) = _
  rw [B.touches_toXT, B.touches_toXT, add_comm]

/-! ## Both sides: the support count -/

/-- **Theorem 1, quadratic form.** A quadratic program correct on the live entries reads, over its products' two
inputs, at least `2·nLive` blocks. No factor of 2 is lost against bilinear programs. -/
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
/-- **Theorem 1, bilinear form.** A bilinear program correct on the live entries reads at least `nLive` X-blocks over its
products. -/
theorem bilin_support (P : BilinProg K m k n τ) {liveX : Fin m × Fin k → Prop} {liveY : Fin k × Fin n → Prop}
    (h : P.toQuad.CrossOn liveX liveY) (BX : XBlocks m k β) :
    nLive liveX liveY ≤ ∑ t, BX.touches (P.u t) := by
  have hx := crossX_count P.toQuad h BX
  simpa [BilinProg.toQuad, XBlocks.touches_zero] using hx

/-! ## Cost under per-slot pricing -/

/-- **The shared cost lemma** (moved here from its FP8-tile files). A quadratic program correct on the live
entries, whose product `t` has both inputs reading at most `s t` blocks, costs at least `p₀` per live product under
any per-product prices `c` with `s t · p₀ ≤ c t`. -/
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

/-- **The cost bound, per-slot supports.** A slot program correct on the live entries whose slot `t` has both inputs
reading at most `s t` blocks costs at least FP4's price for every live product, provided each slot's price covers
`s t` FP4 slots. The shared lemma at `p₀ = price .fp4`. -/
theorem cost_ge_support (P : SlotProg K m k n τ) {liveX : Fin m × Fin k → Prop} {liveY : Fin k × Fin n → Prop}
    (h : P.toQuadProg.CrossOn liveX liveY) (BX : XBlocks m k β) (BY : YBlocks k n γ) (s : τ → ℕ)
    (hs : ∀ t, BX.touches (P.a t) + BY.touches (P.b t) ≤ s t ∧ BX.touches (P.a' t) + BY.touches (P.b' t) ≤ s t)
    (hf : ∀ t, (s t : ℚ) * price .fp4 ≤ price (P.fmt t)) :
    price .fp4 * nLive liveX liveY ≤ P.cost :=
  cost_ge_supportAt P.toQuadProg h BX BY s hs (fun t => price (P.fmt t)) (price .fp4) (by norm_num [price]) hf

/-- **The cost bound.** A slot program correct on the live entries, under the representability rule, costs at least
FP4's price for every live product, provided every format it uses has `capacity f · price .fp4 ≤ price f`. -/
theorem cost_ge (P : SlotProg K m k n τ) {liveX : Fin m × Fin k → Prop} {liveY : Fin k × Fin n → Prop}
    (h : P.toQuadProg.CrossOn liveX liveY) (BX : XBlocks m k β) (BY : YBlocks k n γ) (hr : P.Representable BX BY)
    (hf : ∀ t, (capacity (P.fmt t) : ℚ) * price .fp4 ≤ price (P.fmt t)) :
    price .fp4 * nLive liveX liveY ≤ P.cost :=
  cost_ge_support P h BX BY (fun t => capacity (P.fmt t)) hr hf

/-- **Which formats the representability rule closes:** exactly FP4, FP8 and BF16. FP16 and int8 at their prices, and
`dp4a`, hold more blocks than they pay for (`open_fp16`, `open_int8`, `open_dp4a`). -/
theorem fmt_closed_iff (f : Fmt) :
    (capacity f : ℚ) * price .fp4 ≤ price f ↔ f = .fp4 ∨ f = .fp8 ∨ f = .bf16 := by
  cases f <;> norm_num [capacity, price]

/-- TF32 is not closed: 32 blocks per value at 8 FP4 slots' price. -/
theorem open_tf32 : price .tf32 < capacity .tf32 * price .fp4 := by norm_num [capacity, price]

/-- FP32 (FFMA) is not closed: 2¹⁸ blocks per value at about 16–17 FP4 slots' price (16.92 at the model's 8.46; 16.0–16.3
measured). -/
theorem open_fp32 : price .fp32 < capacity .fp32 * price .fp4 := by norm_num [capacity, price]

/-- **TF32 pre-adds reading at most 8 blocks are closed** (at a tie): 8 FP4 slots' price is TF32's. A Strassen level-3
pre-add reads 8 blocks. -/
theorem tf32_closed_upto : (8 : ℚ) * price .fp4 ≤ price .tf32 := by norm_num [price]

/-- **FP32 (FFMA) pre-adds reading at most 16 blocks are closed, as a tie:** 16 FP4 slots cost 8 units, FFMA 8.46 in the
model and 8.0–8.15 measured (16.0–16.3 FP4 slots, GPU 4 F1). At the bottom of that range the two are equal; the
statement is `≤`, so it holds across it. A Strassen level-4 pre-add reads 16 blocks. -/
theorem fp32_closed_upto : (16 : ℚ) * price .fp4 ≤ price .fp32 := by norm_num [price]

/-- From 9 blocks per input, TF32's price is below the count (`support_admits_open`). -/
theorem tf32_open_at : price .tf32 < (9 : ℕ) * price .fp4 := by norm_num [price]

/-- From 17 blocks per input, FFMA's price is below the count. That holds for any price below 8.5 units (17 FP4 slots),
so for the model's 8.46 and the measured 8.0–8.15 alike. At 8.5 or more the closure would reach 17 blocks; below 8.0
(16 slots) `fp32_closed_upto` would fail and the closure would stop at 15. -/
theorem fp32_open_at : price .fp32 < (17 : ℕ) * price .fp4 := by norm_num [price]

/-- FP16 is closed up to 4 blocks per input (a tie): FP16 at BF16's price but with 11 significand bits. Strassen depth 2. -/
theorem fp16_closed_upto : (4 : ℚ) * price .fp4 ≤ price .fp16 := by norm_num [price]

/-- From 5 blocks per input, FP16's price is below the count. -/
theorem fp16_open_at : price .fp16 < (5 : ℕ) * price .fp4 := by norm_num [price]

/-- int8 is closed up to 2 blocks per input (a tie). Strassen depth 1. -/
theorem int8_closed_upto : (2 : ℚ) * price .fp4 ≤ price .int8 := by norm_num [price]

/-- From 3 blocks per input, int8's price is below the count. -/
theorem int8_open_at : price .int8 < (3 : ℕ) * price .fp4 := by norm_num [price]

/-- `dp4a` is closed up to 8 blocks per input: 8 FP4 slots cost 4, `dp4a` 4.05. -/
theorem dp4a_closed_upto : (8 : ℚ) * price .fp4 ≤ price .dp4a := by norm_num [price]

/-- From 9 blocks per input, `dp4a`'s price is below the count. -/
theorem dp4a_open_at : price .dp4a < (9 : ℕ) * price .fp4 := by norm_num [price]

/-- FP16 is not closed at its capacity (32 blocks). -/
theorem open_fp16 : price .fp16 < capacity .fp16 * price .fp4 := by norm_num [capacity, price]

/-- int8 is not closed at its capacity (10 blocks). -/
theorem open_int8 : price .int8 < capacity .int8 * price .fp4 := by norm_num [capacity, price]

/-- `dp4a` is not closed at its capacity (10 blocks). -/
theorem open_dp4a : price .dp4a < capacity .dp4a * price .fp4 := by norm_num [capacity, price]

omit [Fintype τ] [Fintype β] [Fintype γ] in
/-- **The support count can't rule out cheaper cost profiles, at any base price**. Wherever
`c < s · p₀`, `M` products whose inputs each read `s` blocks meet the support count with equality for `s·M` live
products, at a total price below `p₀` per live product. These are numbers that pass the count, not an exhibited
program. -/
theorem support_admits_openAt {c p₀ : ℚ} {s : ℕ} (hf : c < s * p₀) (M : ℕ) (hM : 0 < M) :
    2 * (s * M) ≤ ∑ _t : Fin M, (s + s) ∧ (∑ _t : Fin M, c) < p₀ * ((s * M : ℕ) : ℚ) := by
  refine ⟨by simp; ring_nf; omega, ?_⟩
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, Nat.cast_mul]
  have hM' : (0 : ℚ) < M := by exact_mod_cast hM
  nlinarith

omit [Fintype τ] [Fintype β] [Fintype γ] in
/-- **Why wider pre-adds can't be closed by this argument.** Take a format and a support `s` with
`price f < s · price .fp4`: TF32 from 9 blocks, FP16 from 5, int8 from 3, `dp4a` from 9, FFMA from 17. The support
count and the per-slot prices can't rule out a cost below FP4's there. The pin exhibits numbers that pass the count,
not a program. Ruling them out needs a lower bound on the number of products: a matrix-multiplication rank bound of
order `m·k·n·price .fp4 / price f`, far above the best known (about `3n²` for `⟨n, n, n⟩`). For the known recursive
algorithms the exactness-depth argument of `docs/pouw/fp4-tile-model.md` §2.5(a) closes them instead (Derived, not in
Lean). -/
theorem support_admits_open {f : Fmt} {s : ℕ} (hf : price f < s * price .fp4) (M : ℕ) (hM : 0 < M) :
    2 * (s * M) ≤ ∑ _t : Fin M, (s + s) ∧ (∑ _t : Fin M, price f) < price .fp4 * ((s * M : ℕ) : ℚ) :=
  support_admits_openAt hf M hM

/-! ## The honest program meets the bound -/

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
/-- The honest program costs exactly FP4's price for every product: the bound is tight. -/
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
/-- The honest program satisfies the representability rule on any blocks: each slot input is one entry. -/
theorem honest_representable [DecidableEq β] [DecidableEq γ] (BX : XBlocks m k β) (BY : YBlocks k n γ) :
    (honest K m k n).Representable BX BY := by
  intro t
  simp only [honest, XBlocks.touches_unit, YBlocks.touches_unit, XBlocks.touches_zero, YBlocks.touches_zero,
    capacity]
  omega

/-! ## Tile programs of `sm120Nvf4` gate firings -/

omit [Fintype τ] [Fintype γ] in
/-- **The tile count.** A tile program correct on the live entries, whose `P`-inputs each read at most one X-block,
fires at least `nLive / 8192` gates. -/
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
/-- **The tile cost bound.** Such a tile program costs at least FP4's per-slot price for every live product. -/
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
/-- **The bridge to the gate, at one input.** When exact `sm120Nvf4` firings `gs` realize a tile program at `X`, `Y`, its
output from them (the gate words decoded exactly and combined linearly) is its slots' bilinear program at `X`, `Y`. -/
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

/-! ## From correct outputs to the cross identity -/

omit [Fintype β] [Fintype γ] in
/-- **Correct outputs at three inputs give `CrossOn`.** A quadratic program whose output is `X·Y` at the unit and zero
inputs `(e_x, e_y)`, `(e_x, 0)` and `(0, e_y)`, for every live X-entry `x` and live Y-entry `y`, satisfies `CrossOn` on the
live entries. Those inputs are FP4-representable (E2M1 has 0 and 1, and a UE4M3 scale 1), so a tile program's firings
can carry them. The fourth input, `(0, 0)`, isn't needed. -/
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

/-! ## The chain from gate firings to the cost bound -/

omit [Fintype τ] [Fintype γ] in
/-- **From gate firings to the cost bound.** Take a tile program whose `P`-inputs each read at most one X-block. Suppose
that, for every live X-entry `x` and live Y-entry `y`, exact firings realize it at `(e_x, e_y)`, `(e_x, 0)` and
`(0, e_y)`, and output `X·Y` there. Then it costs at least FP4's per-slot price for every live product. -/
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

/-! ## `RealizesAt` is not vacuous: one exact firing, machine-checked -/

/-- A one-gate witness: code 1.0 (nibble 2) at `P[0, 0]` and at `Q[0, 0]`, every other code 0, every scale byte
`0x38` (UE4M3 1.0), accumulators `+0`. -/
def oneGate : Firing where
  p i l := if i.val = 0 ∧ l.val = 0 then 2 else 0
  ps _ _ := 0x38
  q l j := if l.val = 0 ∧ j.val = 0 then 2 else 0
  qs _ _ := 0x38
  acc _ _ := 0

/-- Every output word of the one-gate witness is exact: `sm120Nvf4.step` evaluated by the kernel on all 128 outputs. -/
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

/-- The honest one-gate program for a 16 × 64 × 8 product. -/
def oneGateProg : TileProg ℚ 16 64 8 1 :=
  ⟨fun _ i l => unitForm (i, l), fun _ l j => unitForm (l, j), fun _ i j => unitForm (i, j)⟩

theorem oneGateProg_realizesAt :
    oneGateProg.RealizesAt (fun _ => oneGate) (unitForm (0, 0)) (unitForm (0, 0)) := by
  intro _
  refine ⟨oneGate_exact, fun _ _ => rfl, fun i l => ?_, fun l j => ?_⟩
  · rw [oneGate_p]; simp only [oneGateProg, sum_unitForm_mul]
  · rw [oneGate_q]; simp only [oneGateProg, sum_unitForm_mul]

/-- **`RealizesAt` is satisfiable with a nonzero output.** An exact firing of `sm120Nvf4` realizes a tile program at the
unit inputs, and the program's output there is 1. -/
theorem realizesAt_nonvacuous : ∃ (P : TileProg ℚ 16 64 8 1) (gs : Fin 1 → Firing),
    P.RealizesAt gs (unitForm (0, 0)) (unitForm (0, 0)) ∧ P.outAt gs (0, 0) = 1 := by
  refine ⟨oneGateProg, fun _ => oneGate, oneGateProg_realizesAt, ?_⟩
  rw [realizes_out_at _ _ _ _ oneGateProg_realizesAt]
  simp [QuadProg.eval, BilinProg.toQuad, TileProg.toBilin, oneGateProg, unitForm, Fintype.sum_prod_type]

end Pouw.TileBound
