import FlockSoundness.LigeritoPhases

/-!
# From one level to the next (PROTOCOL.md §13.3 steps 5–8)

After a level's folds, the prover commits the next table, the verifier binds it by two
out-of-domain samples, and checks the fold against it at the level's queries. From `PostFold` on
the level, this leaves `PreFold` on the next level except with probability
* `pairs(L')·(μ/|F|)²` (two list elements agree at both OOD points, `value_twoOod_collide_le`),
* `2L'/|F|` (the OOD claims' batching coins),
* the level's `queryError` (the queries miss the unique OOD-consistent message's mismatch),
* `⌈log2 Q⌉/|F|` (the consistency coins cancel it) and `L'/|F|` (its batching coin).
-/

namespace FlockSoundness

open Game Model Accounting
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

/-! ## Batching coins over `K`-valued differences -/

/-- A batching coin `β ∈ F` makes `c₀ + β·c₁` vanish, unless both vanish, for at most one value. -/
theorem prCoin_embed_affine_le (A : Arith F K) (c₀ c₁ : K) :
    prCoin (fun β : F => c₀ + A.embed β * c₁ = 0 ∧ ¬ (c₀ = 0 ∧ c₁ = 0)) ≤
      1 / Fintype.card F := by
  classical
  by_cases h1 : c₁ = 0
  · refine le_trans (prCoin_mono (q := fun _ => False) fun β ⟨h, hn⟩ =>
      hn ⟨by simpa [h1] using h, h1⟩) ?_
    simp [prCoin, avg]
  · rw [prCoin_eq_card]
    gcongr
    refine Nat.cast_le_one.2 (Finset.card_le_one.2 fun a ha b hb => ?_)
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at ha hb
    have : A.embed a * c₁ = A.embed b * c₁ := by linear_combination ha.1 - hb.1
    exact A.embed.injective (mul_right_cancel₀ h1 this)

theorem prCoin_embed_affine_union_le (A : Arith F K) {ι : Type} (S : Finset ι) (c₀ c₁ : ι → K) :
    prCoin (fun β : F => ∃ M ∈ S, c₀ M + A.embed β * c₁ M = 0 ∧ ¬ (c₀ M = 0 ∧ c₁ M = 0)) ≤
      S.card / Fintype.card F := by
  calc _ ≤ ∑ M ∈ S, prCoin (fun β : F => c₀ M + A.embed β * c₁ M = 0 ∧ ¬ (c₀ M = 0 ∧ c₁ M = 0)) :=
        prCoin_exists_le S _
    _ ≤ ∑ _M ∈ S, (1 : ℝ≥0∞) / Fintype.card F :=
        Finset.sum_le_sum fun M _ => prCoin_embed_affine_le A (c₀ M) (c₁ M)
    _ = S.card / Fintype.card F := by rw [Finset.sum_const, nsmul_eq_mul, mul_one_div]

theorem sum_map_zipIdx {M : Type} [AddCommMonoid M] (h : ℕ × ℕ → M) : ∀ (l : List ℕ) (k : ℕ),
    ((l.zipIdx k).map h).sum = ∑ i ∈ Finset.range l.length, h (l.getD i 0, k + i)
  | [], _ => by simp
  | a :: l, k => by
      rw [List.zipIdx_cons, List.map_cons, List.sum_cons, sum_map_zipIdx h l (k + 1),
        List.length_cons, Finset.sum_range_succ']
      simp only [List.getD_cons_succ, List.getD_cons_zero, add_zero]
      rw [add_comm]
      congr 1
      exact Finset.sum_congr rfl fun i _ => by rw [show k + 1 + i = k + (i + 1) by ring]

/-- **The consistency coins over `K`.** A nonzero `K` vector of mismatches, weighted by `eq(α)`
over `k ≥ ⌈log2 Q⌉` coordinates of `F`, sums to zero with probability at most `k/|F|`. -/
theorem prCoin_embed_eqList_le (A : Arith F K) (hA : A.Correct) {k Q : ℕ} (hQ : Q ≤ 2 ^ k)
    (v : ℕ → K) (j₀ : ℕ) (hj₀ : j₀ < Q) (hv : v j₀ ≠ 0) :
    prCoin (fun α : Fin k → F =>
        ∑ j ∈ Finset.range Q, A.embed (eqList (List.ofFn α) j) * v j = 0) ≤
      (k : ℝ≥0∞) / Fintype.card F := by
  haveI : CharP F 2 := hA.charTwo
  set a := fun j => (limbs A hA (v j)).1
  set c := fun j => (limbs A hA (v j)).2
  have hsum : ∀ α : Fin k → F, ∑ j ∈ Finset.range Q, A.embed (eqList (List.ofFn α) j) * v j =
      A.ofLimbs (∑ j ∈ Finset.range Q, eqList (List.ofFn α) j * a j)
        (∑ j ∈ Finset.range Q, eqList (List.ofFn α) j * c j) := by
    intro α
    rw [← ofLimbs_sum]
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [← ofLimbs_limbs A hA (v j), mul_comm, ofLimbs_mul_embed]
    simp only [a, c]
    congr 1 <;> ring
  have hzero : ∀ α : Fin k → F,
      ∑ j ∈ Finset.range Q, A.embed (eqList (List.ofFn α) j) * v j = 0 →
        (∑ j ∈ Finset.range Q, eqList (List.ofFn α) j * a j = 0) ∧
          ∑ j ∈ Finset.range Q, eqList (List.ofFn α) j * c j = 0 := by
    intro α h
    rw [hsum, ← ofLimbs_zero A] at h
    exact ofLimbs_inj A hA h
  by_cases ha : a j₀ = 0
  · have hc : c j₀ ≠ 0 := by
      intro hc
      apply hv
      rw [← ofLimbs_limbs A hA (v j₀)]
      simp only [a, c] at ha hc
      rw [ha, hc, ofLimbs_zero]
    exact (prCoin_mono fun α h => (hzero α h).2).trans (prCoin_eqList_le hQ c j₀ hj₀ hc)
  · exact (prCoin_mono fun α h => (hzero α h).1).trans (prCoin_eqList_le hQ a j₀ hj₀ ha)

theorem le_two_pow_ceilLog2 (Q : ℕ) (hQ : Q ≤ 2 ^ 64) : Q ≤ 2 ^ ceilLog2 Q := by
  unfold ceilLog2
  cases h : (List.range 65).find? (fun k => decide (Q ≤ 2 ^ k)) with
  | none =>
      have := List.find?_eq_none.1 h 64 (List.mem_range.2 (by norm_num))
      simp at this
      omega
  | some k =>
      have := List.find?_some h
      simpa using this

/-! ## The two OOD samples -/

/-- A table's multilinear extension over `K` at an `F` point. -/
def mleK (A : Arith F K) (dim : ℕ) (g : ℕ → F) (z : List F) : K :=
  ∑ y ∈ Finset.range (2 ^ dim), A.embed (g y) * eqList (z.map A.embed) y

theorem map_eqList (A : Arith F K) (z : List F) (y : ℕ) :
    A.embed (eqList z y) = eqList (z.map A.embed) y := by
  unfold eqList
  rw [map_list_prod, List.map_map, List.zipIdx_map, List.map_map]
  congr 1
  refine List.map_congr_left fun ai _ => ?_
  obtain ⟨a, i⟩ := ai
  simp only [Function.comp_apply, Prod.map_apply, id_eq]
  by_cases h : y.testBit i <;> simp [h]

theorem mleK_ofFn (A : Arith F K) (dim : ℕ) (g : ℕ → F) (z : Fin dim → F) :
    mleK A dim g (List.ofFn z) = A.embed (mleL (fun y : Fin (2 ^ dim) => g y.val) (List.ofFn z)) := by
  unfold mleK mleL
  rw [map_sum, ← Fin.sum_univ_eq_sum_range (fun y => A.embed (g y) *
    eqList ((List.ofFn z).map A.embed) y)]
  refine Finset.sum_congr rfl fun y _ => ?_
  rw [map_mul, map_eqList, mul_comm]

/-- `twoOod` with the prover's answers kept. -/
structure OodRec (F K : Type) (dim : ℕ) where
  z₁ : Fin dim → F
  y₁ : F
  m₁ : K × K
  β₁ : F
  z₂ : Fin dim → F
  y₂ : F
  m₂ : K × K
  β₂ : F

def twoOodR (dim : ℕ) : Game (OodRec F K dim) :=
  Game.coin (Fin dim → F) fun z₁ => Game.send F fun y₁ => Game.send (K × K) fun m₁ =>
    Game.coin F fun β₁ => Game.coin (Fin dim → F) fun z₂ => Game.send F fun y₂ =>
      Game.send (K × K) fun m₂ => Game.coin F fun β₂ =>
        Game.ret ⟨z₁, y₁, m₁, β₁, z₂, y₂, m₂, β₂⟩

/-- `twoOod`'s output from the record. -/
def OodRec.out (A : Arith F K) (split : ℕ) (s : K) (q : Quad K) {dim : ℕ} (r : OodRec F K dim) :
    K × Quad K × List (OodCtx F) :=
  (s + A.embed r.β₁ * A.embed r.y₁ + A.embed r.β₂ * A.embed r.y₂,
    (q.addScaled (fromMsg r.m₁ (A.embed r.y₁)) (A.embed r.β₁)).addScaled
      (fromMsg r.m₂ (A.embed r.y₂)) (A.embed r.β₂),
    [⟨List.ofFn r.z₁, r.β₁, some split⟩, ⟨List.ofFn r.z₂, r.β₂, some split⟩])

theorem twoOod_eq_map (A : Arith F K) (dim split : ℕ) (s : K) (q : Quad K) :
    twoOod A dim split s q = (twoOodR dim).map (OodRec.out A split s q) := rfl

theorem Quad.addScaled_eval (q p : Quad K) (β x : K) :
    (q.addScaled p β).eval x = q.eval x + β * p.eval x := by
  simp only [Quad.addScaled, Quad.eval]; ring

theorem Quad.addScaled_sum [CharP K 2] (q : Quad K) (msg : K × K) (y β : K) :
    (q.addScaled (fromMsg msg y) β).eval 0 + (q.addScaled (fromMsg msg y) β).eval 1 =
      q.eval 0 + q.eval 1 + β * y := by
  have h := fromMsg_sum msg y
  rw [Quad.addScaled_eval, Quad.addScaled_eval]
  linear_combination β * h

/-- The batched claim minus a message's value, after the OOD claims `y.2.2`. -/
def oodDiff (A : Arith F K) (dim : ℕ) (g : ℕ → F) (base : K) (y : K × Quad K × List (OodCtx F)) :
    K :=
  y.1 - (base + (y.2.2.map fun o => A.embed o.β * mleK A dim g o.z).sum)

/-- **After two OOD samples**: the output has the expected shape, and at most one message keeps a
zero difference, for which the previous claim `s` is its value `base`. -/
def OodGood (A : Arith F K) {ι : Type} (S : Finset ι) (g : ι → ℕ → F) (base : ι → K)
    (dim a : ℕ) (s : K) (y : K × Quad K × List (OodCtx F)) : Prop :=
  y.2.1.eval 0 + y.2.1.eval 1 = y.1 ∧ (∀ o ∈ y.2.2, o.split = some a ∧ o.z.length = dim) ∧
    ∀ M ∈ S, oodDiff A dim (g M) (base M) y = 0 →
      s = base M ∧ ∀ M' ∈ S, oodDiff A dim (g M') (base M') y = 0 → M' = M

open Classical in
/-- **The OOD samples bind the next table.** -/
theorem value_twoOod_good (A : Arith F K) (hA : A.Correct) {ι : Type} (S : Finset ι)
    (g : ι → ℕ → F) (dim : ℕ)
    (hg : ∀ M ∈ S, ∀ M' ∈ S, (∀ y < 2 ^ dim, g M y = g M' y) → M = M')
    (base : ι → K) (a : ℕ) (s : K) (q : Quad K) (hq : q.eval 0 + q.eval 1 = s) :
    value (fun y => ¬ OodGood A S g base dim a s y) (twoOod A dim a s q) ≤
      (S.card.choose 2 : ℝ≥0∞) * ((dim : ℝ≥0∞) / Fintype.card F) ^ 2 +
        2 * (S.card / Fintype.card F) := by
  haveI : CharP F 2 := hA.charTwo
  haveI : CharP K 2 := hA.charTwoK
  let G : ι → Fin (2 ^ dim) → F := fun M y => g M y.val
  let mF : ι → (Fin dim → F) → F := fun M z => mleL (G M) (List.ofFn z)
  let d₁ : OodRec F K dim → ι → K := fun r M =>
    (s - base M) + A.embed r.β₁ * (A.embed r.y₁ - A.embed (mF M r.z₁))
  let collide : OodRec F K dim → Prop := fun r =>
    OodCollide (S.image G) (List.ofFn r.z₁) (List.ofFn r.z₂)
  let bad₁ : OodRec F K dim → Prop := fun r => ∃ M ∈ S,
    (s - base M) + A.embed r.β₁ * (A.embed r.y₁ - A.embed (mF M r.z₁)) = 0 ∧
      ¬ (s - base M = 0 ∧ A.embed r.y₁ - A.embed (mF M r.z₁) = 0)
  let bad₂ : OodRec F K dim → Prop := fun r => ∃ M ∈ S,
    d₁ r M + A.embed r.β₂ * (A.embed r.y₂ - A.embed (mF M r.z₂)) = 0 ∧
      ¬ (d₁ r M = 0 ∧ A.embed r.y₂ - A.embed (mF M r.z₂) = 0)
  have hdiff : ∀ r M, oodDiff A dim (g M) (base M) (OodRec.out A a s q r) =
      d₁ r M + A.embed r.β₂ * (A.embed r.y₂ - A.embed (mF M r.z₂)) := by
    intro r M
    simp only [oodDiff, OodRec.out, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil,
      mleK_ofFn, d₁, mF, G]
    ring
  have himp : ∀ r, ¬ OodGood A S g base dim a s (OodRec.out A a s q r) →
      collide r ∨ bad₁ r ∨ bad₂ r := by
    intro r hno
    by_contra hall
    simp only [not_or] at hall
    obtain ⟨hc, h1, h2⟩ := hall
    have hzero : ∀ M ∈ S, oodDiff A dim (g M) (base M) (OodRec.out A a s q r) = 0 →
        s - base M = 0 ∧ A.embed r.y₁ = A.embed (mF M r.z₁) ∧ A.embed r.y₂ = A.embed (mF M r.z₂) := by
      intro M hM h0
      rw [hdiff] at h0
      have e2 : d₁ r M = 0 ∧ A.embed r.y₂ - A.embed (mF M r.z₂) = 0 := by
        by_contra hn; exact h2 ⟨M, hM, h0, hn⟩
      have e1 : s - base M = 0 ∧ A.embed r.y₁ - A.embed (mF M r.z₁) = 0 := by
        by_contra hn; exact h1 ⟨M, hM, e2.1, hn⟩
      exact ⟨e1.1, sub_eq_zero.1 e1.2, sub_eq_zero.1 e2.2⟩
    refine hno ⟨?_, ?_, fun M hM h0 => ⟨sub_eq_zero.1 (hzero M hM h0).1, fun M' hM' h0' => ?_⟩⟩
    · simp only [OodRec.out, Quad.addScaled_sum, hq]
    · intro o ho
      simp only [OodRec.out, List.mem_cons, List.mem_nil_iff, or_false] at ho
      rcases ho with rfl | rfl <;> simp
    · obtain ⟨_, a1, a2⟩ := hzero M hM h0
      obtain ⟨_, b1, b2⟩ := hzero M' hM' h0'
      by_contra hne
      apply hc
      refine ⟨G M', Finset.mem_image_of_mem G hM', G M, Finset.mem_image_of_mem G hM, ?_,
        A.embed.injective (b1.symm.trans a1), A.embed.injective (b2.symm.trans a2)⟩
      intro hG
      exact hne (hg M' hM' M hM fun y hy => congrFun hG ⟨y, hy⟩)
  -- the collision
  have hcol : value collide (twoOodR dim) ≤
      (S.card.choose 2 : ℝ≥0∞) * ((dim : ℝ≥0∞) / Fintype.card F) ^ 2 := by
    have h := value_twoOod_collide_le A (S.image G) a s q
    rw [twoOod_eq_map, value_map] at h
    refine le_trans (value_mono (fun r hr => ⟨_, _, rfl, hr⟩) _) (h.trans ?_)
    gcongr
    exact Finset.card_image_le (s := S) (f := G)
  -- the first batching coin
  have hb₁ : value bad₁ (twoOodR dim) ≤ S.card / Fintype.card F := by
    refine value_coin_le_of_forall _ _ _ _ fun z₁ => ?_
    rw [value_send]; refine iSup_le fun y₁ => ?_
    rw [value_send]; refine iSup_le fun m₁ => ?_
    refine (value_coin_le bad₁ F _ (fun β₁ => ∃ M ∈ S,
        (s - base M) + A.embed β₁ * (A.embed y₁ - A.embed (mF M z₁)) = 0 ∧
          ¬ (s - base M = 0 ∧ A.embed y₁ - A.embed (mF M z₁) = 0))
      (S.card / Fintype.card F) 0
      (prCoin_embed_affine_union_le A S _ _) fun β₁ hβ₁ => ?_).trans (by rw [add_zero])
    refine value_coin_le_of_forall _ _ _ _ fun z₂ => ?_
    rw [value_send]; refine iSup_le fun y₂ => ?_
    rw [value_send]; refine iSup_le fun m₂ => ?_
    refine value_coin_le_of_forall _ _ _ _ fun β₂ => ?_
    rw [value_ret, if_neg hβ₁]
  -- the second batching coin
  have hb₂ : value bad₂ (twoOodR dim) ≤ S.card / Fintype.card F := by
    refine value_coin_le_of_forall _ _ _ _ fun z₁ => ?_
    rw [value_send]; refine iSup_le fun y₁ => ?_
    rw [value_send]; refine iSup_le fun m₁ => ?_
    refine value_coin_le_of_forall _ _ _ _ fun β₁ => ?_
    refine value_coin_le_of_forall _ _ _ _ fun z₂ => ?_
    rw [value_send]; refine iSup_le fun y₂ => ?_
    rw [value_send]; refine iSup_le fun m₂ => ?_
    rw [value_coin_ret]
    exact prCoin_embed_affine_union_le A S
      (fun M => (s - base M) + A.embed β₁ * (A.embed y₁ - A.embed (mF M z₁)))
      (fun M => A.embed y₂ - A.embed (mF M z₂))
  rw [twoOod_eq_map, value_map]
  calc value (fun r => ¬ OodGood A S g base dim a s (OodRec.out A a s q r)) (twoOodR dim)
      ≤ value (fun r => collide r ∨ bad₁ r ∨ bad₂ r) (twoOodR dim) := value_mono himp _
    _ ≤ value collide (twoOodR dim) + (value bad₁ (twoOodR dim) + value bad₂ (twoOodR dim)) :=
        (value_or_le _ _ _).trans (add_le_add le_rfl (value_or_le _ _ _))
    _ ≤ _ := by
        rw [two_mul]
        exact add_le_add hcol (add_le_add hb₁ hb₂)

end FlockSoundness
