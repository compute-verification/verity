import Proofs.Flock.Soundness.LigeritoPhases

/-!
# From one level to the next (PROTOCOL.md §13.3 steps 5–8)

After a level's folds, the prover commits the next table, the verifier binds it by two
out-of-domain samples, and checks the fold against it at the level's queries. From `PostFold` on
the level, this leaves `PreFold` on the next level except with probability
* `pairs(L')·(μ/|F|)²` (two list elements agree at both OOD points, `value_twoOod_collide_le`),
* `2L'/|F|` (the OOD claims' batching coins),
* the level's `queryError` (the queries miss the unique OOD-consistent message's mismatch),
* `⌈log2 Q⌉/|F|` (the consistency coins cancel it) and `L'/|F|` (its batching coin).

`phaseA` takes the level's doom at any radius `δ` within its own (`PostFoldAt`): its queries then
miss a `δ`-far fold with probability at most `(1 − δ)^Q`, which costs `ε` beyond `queryError`.
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

/-! ## The queries and the consistency claim -/

theorem queryAndBatch_true (A : Arith F K) (lv : Level) (C : Oracle F) (ch : List K) (start : ℕ)
    (st : LigState F K) :
    queryAndBatch A lv C ch start st true =
      Game.coin (Fin lv.queries → F) fun w => Game.coin (Fin (ceilLog2 lv.queries) → F) fun α =>
        Game.send (K × K) fun msg => Game.coin F fun β =>
          Game.ret { st with
            s := st.s + A.embed β *
              consValue A (List.ofFn α) ch C (positions A lv.queries lv.logLen w)
            q := st.q.addScaled (fromMsg msg
              (consValue A (List.ofFn α) ch C (positions A lv.queries lv.logLen w))) (A.embed β)
            cons := st.cons ++
              [⟨lv.logCols, positions A lv.queries lv.logLen w, List.ofFn α, β, start⟩] } := rfl

theorem queryAndBatch_false (A : Arith F K) (lv : Level) (C : Oracle F) (ch : List K)
    (start : ℕ) (st : LigState F K) :
    queryAndBatch A lv C ch start st false =
      Game.coin (Fin lv.queries → F) fun w => Game.coin (Fin (ceilLog2 lv.queries) → F) fun α =>
        Game.coin F fun β =>
          Game.ret { st with
            s := st.s + A.embed β *
              consValue A (List.ofFn α) ch C (positions A lv.queries lv.logLen w)
            cons := st.cons ++
              [⟨lv.logCols, positions A lv.queries lv.logLen w, List.ofFn α, β, start⟩] } := rfl

/-- The consistency claim's error against a claimed fold `G`:
`Σ_j eq(α)_j·(fold(C)[p_j] − Enc(G)[p_j])`. -/
def consGap (A : Arith F K) (cols : ℕ) (C : Oracle F) (ch : List K) (G : ℕ → K) (α : List F)
    (pos : List ℕ) : K :=
  ((pos.zipIdx).map fun pj => A.embed (eqList α pj.2) *
    (foldColumn A ch (fun ℓ => C ℓ pj.1) - encK A cols G pj.1)).sum

open Classical in
/-- The positions where the fold of `C` by `ch` differs from the encoding of `G`. -/
noncomputable def mismatch (A : Arith F K) (cols d : ℕ) (C : Oracle F) (ch : List K)
    (G : ℕ → K) : Finset ℕ :=
  (Finset.range (2 ^ d)).filter fun p => foldColumn A ch (fun ℓ => C ℓ p) ≠ encK A cols G p

theorem positions_length (A : Arith F K) (Q d : ℕ) (w : Fin Q → F) :
    (positions A Q d w).length = Q := by
  rw [positions_eq]; simp

open Classical in
/-- **The queries and the consistency coins.** If at most one message has a zero difference `D`
and its claimed fold's mismatch survives the queries with probability at most `ε`, then
after the queries, `α` and `β`, every difference is nonzero, except with probability
`ε + ⌈log2 Q⌉/|F| + |S|/|F|`. -/
theorem value_queryAndBatch_le (A : Arith F K) (hA : A.Correct) (l : Level) (hd : l.logLen ≤ 64)
    (hQ : l.queries ≤ 2 ^ 64) (C : Oracle F) (ch : List K) (start : ℕ) (st : LigState F K)
    (wm : Bool) {ι : Type} (S : Finset ι) (D : ι → K) (G : ι → ℕ → K) (ε : ℝ)
    (huniq : ∀ M ∈ S, D M = 0 → ∀ M' ∈ S, D M' = 0 → M' = M)
    (hfar : ∀ M ∈ S, D M = 0 →
      (1 - ((mismatch A l.logCols l.logLen C ch (G M)).card : ℝ) / 2 ^ l.logLen) ^ l.queries ≤ ε)
    (P : LigState F K → Prop)
    (hP : ∀ (w : Fin l.queries → F) (α : Fin (ceilLog2 l.queries) → F) (β : F) (st' : LigState F K),
      st'.s = st.s + A.embed β * consValue A (List.ofFn α) ch C (positions A l.queries l.logLen w) →
      st'.cons = st.cons ++ [⟨l.logCols, positions A l.queries l.logLen w, List.ofFn α, β, start⟩] →
      st'.levels = st.levels → st'.oods = st.oods → st'.prev = st.prev →
      (wm = true → st'.q.eval 0 + st'.q.eval 1 = st.q.eval 0 + st.q.eval 1 +
        A.embed β * consValue A (List.ofFn α) ch C (positions A l.queries l.logLen w)) →
      P st' → ∃ M ∈ S, D M + A.embed β *
        consGap A l.logCols C ch (G M) (List.ofFn α) (positions A l.queries l.logLen w) = 0) :
    value P (queryAndBatch A l C ch start st wm) ≤
      ENNReal.ofReal ε + ((ceilLog2 l.queries : ℝ≥0∞) / Fintype.card F + S.card / Fintype.card F) := by
  haveI : CharP K 2 := hA.charTwoK
  set Q := l.queries
  set pos := fun w : Fin Q → F => positions A Q l.logLen w
  set gap := fun (M : ι) (w : Fin Q → F) (α : Fin (ceilLog2 Q) → F) =>
    consGap A l.logCols C ch (G M) (List.ofFn α) (pos w)
  let Bq : (Fin Q → F) → Prop := fun w =>
    ∃ M ∈ S, D M = 0 ∧ ∀ p ∈ pos w, p ∉ mismatch A l.logCols l.logLen C ch (G M)
  let Bα : (Fin Q → F) → (Fin (ceilLog2 Q) → F) → Prop := fun w α =>
    ∃ M ∈ S, D M = 0 ∧ gap M w α = 0
  let Bβ : (Fin Q → F) → (Fin (ceilLog2 Q) → F) → F → Prop := fun w α β =>
    ∃ M ∈ S, D M + A.embed β * gap M w α = 0 ∧ ¬ (D M = 0 ∧ gap M w α = 0)
  have hBq : prCoin Bq ≤ ENNReal.ofReal ε := by
    by_cases hex : ∃ M ∈ S, D M = 0
    · obtain ⟨M, hM, hD⟩ := hex
      refine (prCoin_mono (q := fun w => ∀ p ∈ positions A Q l.logLen w,
          p ∉ mismatch A l.logCols l.logLen C ch (G M)) fun w ⟨M', hM', hD', h⟩ => by
        rw [← huniq M hM hD M' hM' hD']; exact h).trans ?_
      refine (prCoin_positions_miss_le A hA Q l.logLen hd _ (Finset.filter_subset _ _)).trans ?_
      exact ENNReal.ofReal_le_ofReal (hfar M hM hD)
    · refine (prCoin_mono (q := fun _ => False) fun w ⟨M, hM, hD, _⟩ => hex ⟨M, hM, hD⟩).trans ?_
      simp [prCoin, avg]
  have hBα : ∀ w, ¬ Bq w → prCoin (Bα w) ≤ (ceilLog2 Q : ℝ≥0∞) / Fintype.card F := by
    intro w hw
    by_cases hex : ∃ M ∈ S, D M = 0
    · obtain ⟨M, hM, hD⟩ := hex
      obtain ⟨p, hp, hpm⟩ : ∃ p ∈ pos w, p ∈ mismatch A l.logCols l.logLen C ch (G M) := by
        by_contra hno
        push Not at hno
        exact hw ⟨M, hM, hD, hno⟩
      obtain ⟨j₀, hj₀, hpj⟩ := List.getElem_of_mem hp
      have hlen : (pos w).length = Q := positions_length A Q l.logLen w
      set v : ℕ → K := fun j => foldColumn A ch (fun ℓ => C ℓ ((pos w).getD j 0)) -
        encK A l.logCols (G M) ((pos w).getD j 0)
      have hgap : ∀ α, gap M w α = ∑ j ∈ Finset.range Q, A.embed (eqList (List.ofFn α) j) * v j := by
        intro α
        simp only [gap, consGap, List.zipIdx]
        rw [sum_map_zipIdx (fun pj => A.embed (eqList (List.ofFn α) pj.2) *
          (foldColumn A ch (fun ℓ => C ℓ pj.1) - encK A l.logCols (G M) pj.1)) (pos w) 0, hlen]
        simp only [zero_add, v]
      have hv : v j₀ ≠ 0 := by
        simp only [v, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hj₀, Option.getD_some,
          hpj]
        simp only [mismatch, Finset.mem_filter] at hpm
        exact sub_ne_zero.2 hpm.2
      refine (prCoin_mono (q := fun α => ∑ j ∈ Finset.range Q,
          A.embed (eqList (List.ofFn α) j) * v j = 0) fun α ⟨M', hM', hD', h⟩ => by
        rw [← hgap, huniq M hM hD M' hM' hD'] at *; exact h).trans ?_
      exact prCoin_embed_eqList_le A hA (le_two_pow_ceilLog2 Q hQ) v j₀ (by omega) hv
    · refine (prCoin_mono (q := fun _ => False) fun α ⟨M, hM, hD, _⟩ => hex ⟨M, hM, hD⟩).trans ?_
      simp [prCoin, avg]
  have hBβ : ∀ w α, prCoin (Bβ w α) ≤ S.card / Fintype.card F := fun w α =>
    prCoin_embed_affine_union_le A S D fun M => gap M w α
  have hleaf : ∀ w α β st', ¬ Bα w α → ¬ Bβ w α β →
      st'.s = st.s + A.embed β * consValue A (List.ofFn α) ch C (pos w) →
      st'.cons = st.cons ++ [⟨l.logCols, pos w, List.ofFn α, β, start⟩] →
      st'.levels = st.levels → st'.oods = st.oods → st'.prev = st.prev →
      (wm = true → st'.q.eval 0 + st'.q.eval 1 = st.q.eval 0 + st.q.eval 1 +
        A.embed β * consValue A (List.ofFn α) ch C (pos w)) → ¬ P st' := by
    intro w α β st' hα hβ h1 h2 h3 h4 h5 h6 hPst
    obtain ⟨M, hM, h0⟩ := hP w α β st' h1 h2 h3 h4 h5 h6 hPst
    have : D M = 0 ∧ gap M w α = 0 := by
      by_contra hn; exact hβ ⟨M, hM, h0, hn⟩
    exact hα ⟨M, hM, this.1, this.2⟩
  cases wm with
  | true =>
      rw [queryAndBatch_true]
      refine (value_coin_le P _ _ Bq _ _ hBq fun w hw => ?_)
      refine (value_coin_le P _ _ (Bα w) _ _ (hBα w hw) fun α hα => ?_)
      rw [value_send]
      refine iSup_le fun msg => ?_
      refine (value_coin_le P _ _ (Bβ w α) _ 0 (hBβ w α) fun β hβ => ?_).trans (by rw [add_zero])
      rw [value_ret]
      split_ifs with hPst
      · exact absurd hPst (hleaf w α β _ hα hβ (by rfl) (by rfl) (by rfl) (by rfl) (by rfl)
          fun _ => Quad.addScaled_sum _ _ _ _)
      · exact le_rfl
  | false =>
      rw [queryAndBatch_false]
      refine (value_coin_le P _ _ Bq _ _ hBq fun w hw => ?_)
      refine (value_coin_le P _ _ (Bα w) _ _ (hBα w hw) fun α hα => ?_)
      refine (value_coin_le P _ _ (Bβ w α) _ 0 (hBβ w α) fun β hβ => ?_).trans (by rw [add_zero])
      rw [value_ret]
      split_ifs with hPst
      · exact absurd hPst (hleaf w α β _ hα hβ (by rfl) (by rfl) (by rfl) (by rfl) (by rfl)
          fun h => by cases h)
      · exact le_rfl

/-! ## One level to the next -/

/-- A split table's `K` message on the previous level's columns: `g(2x) + u·g(2x+1)`. -/
def unsplit (A : Arith F K) (g : ℕ → F) (x : ℕ) : K :=
  A.embed (g (2 * x)) + A.u * A.embed (g (2 * x + 1))

theorem flat_inj {lanes cols : ℕ} (M M' : Message F lanes cols)
    (h : ∀ y < 2 ^ (lanes + cols), flat M y = flat M' y) : M = M' := by
  funext ℓ x
  have hy : ℓ.val + 2 ^ lanes * x.val < 2 ^ (lanes + cols) := by
    rw [pow_add]
    calc ℓ.val + 2 ^ lanes * x.val < 2 ^ lanes + 2 ^ lanes * x.val := by have := ℓ.isLt; omega
      _ = 2 ^ lanes * (x.val + 1) := by ring
      _ ≤ 2 ^ lanes * 2 ^ cols := Nat.mul_le_mul_left _ x.isLt
  have := h _ hy
  rwa [flat_add_mul, flat_add_mul] at this

theorem sum_mul_list_sum {ι κ : Type} (s : Finset ι) (u : ι → K) (l : List κ) (f : ι → κ → K) :
    ∑ x ∈ s, u x * (l.map (f x)).sum = (l.map fun j => ∑ x ∈ s, u x * f x j).sum := by
  induction l with
  | nil => simp
  | cons a l ih =>
      simp only [List.map_cons, List.sum_cons, mul_add, Finset.sum_add_distrib, ih]

theorem list_sum_map_sub {κ : Type} (l : List κ) (f g : κ → K) :
    (l.map fun j => f j - g j).sum = (l.map f).sum - (l.map g).sum := by
  induction l with
  | nil => simp
  | cons a l ih => simp only [List.map_cons, List.sum_cons, ih]; ring

/-- **The next level's claim, unfolded.** Against the weight on the next split table, a table's
value is its unsplit value on the previous stage's weight, plus the new consistency claim on the
unsplit message's encoding, plus the new OOD claims. -/
theorem sum_preW_new (A : Arith F K) (n : ℕ) (b : ℕ → F) (initial : List K)
    (levels : List (List K)) (oods oods₂ : List (OodCtx F)) (cons : List (ConsCtx F)) (e : ℕ)
    (cc : ConsCtx F) (hold : ∀ o ∈ oods, OodOld levels o)
    (hnew : ∀ o ∈ oods₂, o.split = some levels.length) (hcc : cc.start = levels.length)
    (g : ℕ → F) :
    ∑ y ∈ Finset.range (2 ^ (e + 1)),
        A.embed (g y) * preW A n b initial levels (oods ++ oods₂) (cons ++ [cc]) e y =
      ∑ x ∈ Finset.range (2 ^ e), unsplit A g x * resAt A n b initial levels oods cons e x
        + A.embed cc.β * ((cc.positions.zipIdx).map fun pj =>
            A.embed (eqList cc.α pj.2) * encK A e (unsplit A g) pj.1).sum
        + (oods₂.map fun o => A.embed o.β * mleK A (e + 1) g o.z).sum := by
  have hf1 : (oods ++ oods₂).filter (fun o => o.split ≠ some levels.length) = oods := by
    rw [List.filter_append, List.filter_eq_self.2, List.filter_eq_nil_iff.2, List.append_nil]
    · intro o ho; simp [hnew o ho]
    · intro o ho
      rcases hold o ho with h | ⟨a, ha, h⟩ <;> simp [h]; omega
  have hf2 : (oods ++ oods₂).filter (fun o => o.split = some levels.length) = oods₂ := by
    rw [List.filter_append, List.filter_eq_nil_iff.2, List.filter_eq_self.2, List.nil_append]
    · intro o ho; simp [hnew o ho]
    · intro o ho
      rcases hold o ho with h | ⟨a, ha, h⟩ <;> simp [h]; omega
  rw [sum_preW, hf1, hf2]
  simp only [resAt_cons_append, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil,
    add_zero, consTerm_new A levels e cc hcc, mul_add, Finset.sum_add_distrib]
  have hY : ∑ x ∈ Finset.range (2 ^ e), (A.embed (g (2 * x)) + A.u * A.embed (g (2 * x + 1))) *
      (A.embed cc.β * ((cc.positions.zipIdx).map fun pj =>
        A.embed (eqList cc.α pj.2) * A.embed (A.Xhat x e (A.omega pj.1))).sum) =
      A.embed cc.β * ((cc.positions.zipIdx).map fun pj =>
        A.embed (eqList cc.α pj.2) * encK A e (unsplit A g) pj.1).sum := by
    rw [show (fun x => (A.embed (g (2 * x)) + A.u * A.embed (g (2 * x + 1))) *
        (A.embed cc.β * ((cc.positions.zipIdx).map fun pj =>
          A.embed (eqList cc.α pj.2) * A.embed (A.Xhat x e (A.omega pj.1))).sum)) =
        fun x => A.embed cc.β * (unsplit A g x * ((cc.positions.zipIdx).map fun pj =>
          A.embed (eqList cc.α pj.2) * A.embed (A.Xhat x e (A.omega pj.1))).sum) from
        funext fun x => by simp only [unsplit]; ring, ← Finset.mul_sum, sum_mul_list_sum]
    congr 2
    refine List.map_congr_left fun pj _ => ?_
    unfold encK
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl fun x _ => by simp only [unsplit]; ring
  rw [hY]
  rfl

/-- The next commitment, its OOD samples and the level's queries, then the continuation (as in
`recursiveRounds` and `ligerito`). -/
def midRound (A : Arith F K) (lv : Level) (dim : ℕ) (ch : List K) (st : LigState F K)
    (k : Oracle F → LigState F K → Game Bool) : Game Bool := do
  let C' ← recv (Oracle F)
  let (s, q, oods) ← twoOod A dim st.levels.length st.s st.q
  let st := { st with s := s, q := q, oods := st.oods ++ oods }
  let st ← queryAndBatch A lv st.prev ch st.levels.length st true
  k C' st

/-- Everything of `midLevelError` after the level's folds. -/
noncomputable def midA (l l' : Level) : ℝ :=
  pairs l'.listBound * (((l'.logCols + l'.lanes : ℕ) : ℝ) / qF) ^ 2 + 2 * l'.listBound / qF
    + l.queryError + l'.listBound * ceilLog2 l.queries / qF + l'.listBound / qF

theorem midLevelError_eq (l l' : Level) :
    midLevelError l l' = l.foldError (max l.listBound l'.listBound) + midA l l' := by
  unfold midLevelError midA; ring

theorem ofReal_nat_div_qF' (N : ℕ) :
    ENNReal.ofReal ((N : ℝ) / qF) = (N : ℝ≥0∞) / 2 ^ 128 := by
  unfold qF
  rw [ENNReal.ofReal_div_of_pos (by positivity), ENNReal.ofReal_natCast,
    ENNReal.ofReal_pow (by norm_num), ENNReal.ofReal_ofNat]

theorem card_filter_fin (N : ℕ) (p : ℕ → Prop) [DecidablePred p] :
    (Finset.univ.filter fun q : Fin N => p q.val).card = ((Finset.range N).filter p).card := by
  rw [← Finset.card_map Fin.valEmbedding]
  congr 1
  ext x
  simp only [Finset.mem_map, Finset.mem_filter, Finset.mem_univ, true_and, Fin.valEmbedding_apply,
    Finset.mem_range]
  constructor
  · rintro ⟨q, hq, rfl⟩; exact ⟨q.isLt, hq⟩
  · rintro ⟨hx, hp⟩; exact ⟨⟨x, hx⟩, hp, rfl⟩

theorem midA_bound (l l' : Level) (L : ℕ) (hL : L ≤ l'.listBound) (hF : Fintype.card F = 2 ^ 128) :
    (L.choose 2 : ℝ≥0∞) * ((((l'.logCols + l'.lanes : ℕ) : ℕ) : ℝ≥0∞) / Fintype.card F) ^ 2 +
        2 * ((L : ℝ≥0∞) / Fintype.card F) +
      (ENNReal.ofReal l.queryError + ((ceilLog2 l.queries : ℝ≥0∞) / Fintype.card F +
        (L : ℝ≥0∞) / Fintype.card F)) ≤ ENNReal.ofReal (midA l l') := by
  have hL1 : 1 ≤ l'.listBound := by
    unfold Level.listBound; have := Nat.one_le_two_pow (n := l'.logInvRate); omega
  have hqE := queryError_nonneg l
  have hqF : (0 : ℝ) < qF := by unfold qF; positivity
  have hp := pairs_nonneg l'.listBound
  set t1 := pairs l'.listBound * (((l'.logCols + l'.lanes : ℕ) : ℝ) / qF) ^ 2
  set t2 := 2 * (l'.listBound : ℝ) / qF
  set t4 := (l'.listBound : ℝ) * ceilLog2 l.queries / qF
  set t5 := (l'.listBound : ℝ) / qF
  have h1n : 0 ≤ t1 := mul_nonneg hp (sq_nonneg _)
  have h2n : 0 ≤ t2 := by positivity
  have h4n : 0 ≤ t4 := by positivity
  have h5n : 0 ≤ t5 := by positivity
  have hsplit : ENNReal.ofReal (midA l l') = ENNReal.ofReal t1 + ENNReal.ofReal t2 +
      ENNReal.ofReal l.queryError + ENNReal.ofReal t4 + ENNReal.ofReal t5 := by
    unfold midA
    rw [ENNReal.ofReal_add (by positivity) h5n, ENNReal.ofReal_add (by positivity) h4n,
      ENNReal.ofReal_add (by positivity) hqE, ENNReal.ofReal_add h1n h2n]
  rw [hsplit, hF, Nat.cast_pow, Nat.cast_ofNat]
  have e1 : (L.choose 2 : ℝ≥0∞) * ((((l'.logCols + l'.lanes : ℕ) : ℕ) : ℝ≥0∞) / 2 ^ 128) ^ 2 ≤
      ENNReal.ofReal t1 := by
    simp only [t1]
    rw [ENNReal.ofReal_mul hp, ENNReal.ofReal_pow (by positivity), ofReal_nat_div_qF']
    gcongr
    rw [pairs, ← Nat.cast_choose_two, ENNReal.ofReal_natCast]
    exact_mod_cast Nat.choose_le_choose 2 hL
  have e2 : 2 * ((L : ℝ≥0∞) / 2 ^ 128) ≤ ENNReal.ofReal t2 := by
    simp only [t2]
    rw [show (2 : ℝ) * l'.listBound = ((2 * l'.listBound : ℕ) : ℝ) by push_cast; ring,
      ofReal_nat_div_qF', ← mul_div_assoc]
    gcongr
    exact_mod_cast Nat.mul_le_mul_left 2 hL
  have e4 : (ceilLog2 l.queries : ℝ≥0∞) / 2 ^ 128 ≤ ENNReal.ofReal t4 := by
    simp only [t4]
    rw [show (l'.listBound : ℝ) * ceilLog2 l.queries =
      ((l'.listBound * ceilLog2 l.queries : ℕ) : ℝ) by push_cast; ring, ofReal_nat_div_qF']
    gcongr
    exact_mod_cast Nat.le_mul_of_pos_left _ hL1
  have e5 : (L : ℝ≥0∞) / 2 ^ 128 ≤ ENNReal.ofReal t5 := by
    simp only [t5]
    rw [ofReal_nat_div_qF']
    gcongr
  calc _ ≤ ENNReal.ofReal t1 + ENNReal.ofReal t2 +
          (ENNReal.ofReal l.queryError + (ENNReal.ofReal t4 + ENNReal.ofReal t5)) :=
        add_le_add (add_le_add e1 e2) (add_le_add le_rfl (add_le_add e4 e5))
    _ = _ := by ring

theorem card_agree_add_mismatch (A : Arith F K) (cols d : ℕ) (C : Oracle F) (ch : List K)
    (G : ℕ → K) :
    (Finset.univ.filter fun q : Fin (2 ^ d) =>
        foldColumn A ch (fun ℓ => C ℓ q.val) = encK A cols G q.val).card +
      (mismatch A cols d C ch G).card = 2 ^ d := by
  classical
  rw [card_filter_fin (2 ^ d) fun p => foldColumn A ch (fun ℓ => C ℓ p) = encK A cols G p]
  have h := Finset.card_filter_add_card_filter_not (s := Finset.range (2 ^ d))
    (fun p => foldColumn A ch (fun ℓ => C ℓ p) = encK A cols G p)
  rw [Finset.card_range] at h
  convert h using 3
  unfold mismatch
  exact Finset.filter_congr_decidable _ _ _

/-- **The far fold at a radius `δ`.** A claimed fold `G` whose value is the claim, for a claim
false for every fold of a message `δ`-close to the table (`δ` within the level's radius), is
`δ`-far from the level's fold: its mismatch survives the queries with probability at most
`(1 − δ)^Q`. -/
theorem query_bound_at (A : Arith F K) (hA : A.Correct) (n : ℕ) (b : ℕ → F)
    (initial : List K) (l : Level) (hf : l.Fits) (lanes : ℕ) (C : Oracle F) (ch : List K)
    (st : LigState F K) (δ : ℝ) (hδ : δ ≤ l.radius)
    (hpost : PostFoldAt A hA n b initial l hf lanes C ch δ st) (G : ℕ → K)
    (hs : st.s = ∑ x ∈ Finset.range (2 ^ l.logCols),
      G x * resAt A n b initial st.levels st.oods st.cons l.logCols x) :
    (1 - ((mismatch A l.logCols l.logLen C ch G).card : ℝ) / 2 ^ l.logLen) ^ l.queries ≤
      (1 - δ) ^ l.queries := by
  classical
  have hmis : (mismatch A l.logCols l.logLen C ch G).card ≤ 2 ^ l.logLen := by
    have := card_agree_add_mismatch A l.logCols l.logLen C ch G
    omega
  have h2 : (0 : ℝ) < 2 ^ l.logLen := by positivity
  have hle : 1 - ((mismatch A l.logCols l.logLen C ch G).card : ℝ) / 2 ^ l.logLen ≤ 1 - δ := by
    by_contra hgt
    push Not at hgt
    set T := Finset.univ.filter fun q : Fin (2 ^ l.logLen) =>
      foldColumn A ch (fun ℓ => C ℓ q.val) = encK A l.logCols G q.val
    have hT := card_agree_add_mismatch A l.logCols l.logLen C ch G
    have hTr : (T.card : ℝ) = 2 ^ l.logLen - (mismatch A l.logCols l.logLen C ch G).card := by
      rw [eq_sub_iff_add_eq]; exact_mod_cast hT
    have hTc : (T.card : ℝ) ≥ Fintype.card (Fin (2 ^ l.logLen)) * (1 - δ) := by
      have hlt : ((mismatch A l.logCols l.logLen C ch G).card : ℝ) / 2 ^ l.logLen < δ := by linarith
      rw [div_lt_iff₀ h2] at hlt
      rw [Fintype.card_fin, hTr]
      push_cast
      linarith
    obtain ⟨M₀, hM₀, hG⟩ := decode A hA l hf lanes C ch hpost.len hpost.mca δ hδ G T hTc
      fun q hq => (Finset.mem_filter.1 hq).2
    refine hpost.doom M₀ hM₀ ?_
    rw [hs]
    exact Finset.sum_congr rfl fun x hx => by rw [hG x (Finset.mem_range.1 hx)]
  have hnn : 0 ≤ 1 - ((mismatch A l.logCols l.logLen C ch G).card : ℝ) / 2 ^ l.logLen := by
    rw [sub_nonneg, div_le_one h2]; exact_mod_cast hmis
  exact pow_le_pow_left₀ hnn hle _

theorem queryError_eq (l : Level) : l.queryError = (1 - l.radius) ^ l.queries := by
  unfold Level.queryError Level.radius
  congr 1
  ring

/-- **The far fold** (`query_bound_at` at the level's radius): its mismatch survives the queries
with probability at most `queryError`. -/
theorem query_bound_of_doom (A : Arith F K) (hA : A.Correct) (n : ℕ) (b : ℕ → F)
    (initial : List K) (l : Level) (hf : l.Fits) (lanes : ℕ) (C : Oracle F) (ch : List K)
    (st : LigState F K) (hpost : PostFold A hA n b initial l hf lanes C ch st) (G : ℕ → K)
    (hs : st.s = ∑ x ∈ Finset.range (2 ^ l.logCols),
      G x * resAt A n b initial st.levels st.oods st.cons l.logCols x) :
    (1 - ((mismatch A l.logCols l.logLen C ch G).card : ℝ) / 2 ^ l.logLen) ^ l.queries ≤
      l.queryError :=
  (query_bound_at A hA n b initial l hf lanes C ch st l.radius le_rfl hpost G hs).trans_eq
    (queryError_eq l).symm

/-- **A failed `PreFold` is a zero batched difference.** After the OOD claims `oods₂` and the
consistency claim at `(α, positions, β)`, if the next level's claim is true for a message `M`
of its list, then `M`'s OOD difference plus `β` times its consistency gap vanishes. -/
theorem exists_zero_of_not_preFold (A : Arith F K) (n : ℕ) (b : ℕ → F) (initial : List K)
    (l l' : Level) (C : Oracle F) (ch : List K) (st : LigState F K)
    (hok : PostOK initial st.levels st.oods st.cons l.logCols)
    (hdim : l.logCols + 1 = l'.lanes + l'.logCols) (C' : Oracle F)
    (s₂ : K) (q₂ : Quad K) (oods₂ : List (OodCtx F)) (hsum₂ : q₂.eval 0 + q₂.eval 1 = s₂)
    (hshape : ∀ o ∈ oods₂, o.split = some st.levels.length ∧ o.z.length = l'.logCols + l'.lanes)
    (pos : List ℕ) (α : List F) (β : F) (st' : LigState F K)
    (h1 : st'.s = s₂ + A.embed β * consValue A α ch C pos)
    (h2 : st'.cons = st.cons ++ [⟨l.logCols, pos, α, β, st.levels.length⟩])
    (h3 : st'.levels = st.levels) (h4 : st'.oods = st.oods ++ oods₂)
    (h6 : st'.q.eval 0 + st'.q.eval 1 = q₂.eval 0 + q₂.eval 1 + A.embed β * consValue A α ch C pos)
    (hno : ¬ PreFold A n b initial l' C' l.logCols st') :
    ∃ M ∈ closeL A l'.radius l'.lanes l'.logCols l'.logLen C',
      oodDiff A (l'.logCols + l'.lanes) (flat M)
          (∑ x ∈ Finset.range (2 ^ l.logCols), unsplit A (flat M) x *
            resAt A n b initial st.levels st.oods st.cons l.logCols x) (s₂, q₂, oods₂) +
        A.embed β * consGap A l.logCols C ch (unsplit A (flat M)) α pos = 0 := by
  have hdim' : l'.logCols + l'.lanes = l.logCols + 1 := by omega
  have hq' : st'.q.eval 0 + st'.q.eval 1 = st'.s := by rw [h6, h1, hsum₂]
  have hok' : PreOK initial st'.levels st'.oods st'.cons l.logCols := by
    rw [h2, h3, h4]
    exact preOK_of_postOK initial st.levels st.oods st.cons l.logCols hok oods₂
      (fun o ho => ⟨(hshape o ho).1, by rw [(hshape o ho).2, hdim']⟩) _ rfl
  obtain ⟨M, hM, hM'⟩ : ∃ M ∈ closeL A l'.radius l'.lanes l'.logCols l'.logLen C',
      st'.s = ∑ y ∈ Finset.range (2 ^ (l.logCols + 1)),
        A.embed (flat M y) * preW A n b initial st'.levels st'.oods st'.cons l.logCols y := by
    by_contra hall
    push Not at hall
    exact hno ⟨hq', hok', hdim, hall⟩
  refine ⟨M, hM, ?_⟩
  rw [h2, h3, h4, sum_preW_new A n b initial st.levels st.oods oods₂ st.cons l.logCols _
    (fun o ho => (hok.ood o ho).1) (fun o ho => (hshape o ho).1) rfl, h1] at hM'
  have hcv : consValue A α ch C pos = ((pos.zipIdx).map fun pj =>
      A.embed (eqList α pj.2) * foldColumn A ch (fun ℓ => C ℓ pj.1)).sum := rfl
  have hgap : consGap A l.logCols C ch (unsplit A (flat M)) α pos =
      consValue A α ch C pos - ((pos.zipIdx).map fun pj =>
        A.embed (eqList α pj.2) * encK A l.logCols (unsplit A (flat M)) pj.1).sum := by
    rw [hcv, consGap, ← list_sum_map_sub]
    exact congrArg List.sum (List.map_congr_left fun pj _ => by ring)
  rw [hgap, oodDiff, hdim']
  dsimp only
  linear_combination hM'

open Classical in
/-- **From one level to the next.** From `PostFoldAt` on level `l` at a radius `δ` within its own,
the commitment of `l'` (on `dim` variables), its two OOD samples and `l`'s queries leave `PreFold`
on `l'` except with probability `midA l l' + ε`, where the queries miss a fold `δ`-far from `l`'s
table with probability at most `(1 − δ)^Q ≤ queryError + ε`. -/
theorem phaseA (A : Arith F K) (hA : A.Correct) (n : ℕ) (b : ℕ → F) (initial : List K)
    (l l' : Level) (hf : l.Fits) (hf' : l'.Fits) (hQ : l.queries ≤ 2 ^ 64) (lanes : ℕ)
    (ch : List K) (st : LigState F K) (δ : ℝ) (hδ : δ ≤ l.radius) (ε : ℝ)
    (hε : (1 - δ) ^ l.queries ≤ l.queryError + ε)
    (hpost : PostFoldAt A hA n b initial l hf lanes st.prev ch δ st)
    (hdim : l.logCols + 1 = l'.lanes + l'.logCols) (dim : ℕ) (hd : dim = l'.logCols + l'.lanes)
    (k : Oracle F → LigState F K → Game Bool) (εk : ℝ≥0∞)
    (hk : ∀ C' st', PreFold A n b initial l' C' l.logCols st' →
      value (fun r => r = true) (k C' st') ≤ εk) :
    value (fun r => r = true) (midRound A l dim ch st k) ≤
      ENNReal.ofReal (midA l l') + ENNReal.ofReal ε + εk := by
  subst hd
  haveI : CharP K 2 := hA.charTwoK
  have hdim' : l'.logCols + l'.lanes = l.logCols + 1 := by omega
  show value _ (Game.send (Oracle F) fun C' => _) ≤ _
  rw [value_send]
  refine iSup_le fun C' => ?_
  set S' := closeL A l'.radius l'.lanes l'.logCols l'.logLen C'
  have hS' : S'.card ≤ l'.listBound :=
    card_close_le A hA l' hf' l'.lanes (tbl l'.lanes l'.logLen C')
  let base : Message F l'.lanes l'.logCols → K := fun M =>
    ∑ x ∈ Finset.range (2 ^ l.logCols), unsplit A (flat M) x *
      resAt A n b initial st.levels st.oods st.cons l.logCols x
  refine (value_bind_le _ _ _ (OodGood A S' (fun M => flat M) base (l'.logCols + l'.lanes)
      st.levels.length st.s) (ENNReal.ofReal (l.queryError + ε) + ((ceilLog2 l.queries : ℝ≥0∞) /
        Fintype.card F + S'.card / Fintype.card F) + εk) fun y hy => ?_).trans ?_
  · obtain ⟨s₂, q₂, oods₂⟩ := y
    obtain ⟨hsum₂, hshape, hgood⟩ := hy
    show value _ ((queryAndBatch A l st.prev ch st.levels.length
      { st with s := s₂, q := q₂, oods := st.oods ++ oods₂ } true).bind (k C')) ≤ _
    refine (value_bind_le _ _ _ (PreFold A n b initial l' C' l.logCols) εk (hk C')).trans
      (add_le_add ?_ le_rfl)
    refine value_queryAndBatch_le A hA l hf.2 hQ st.prev ch st.levels.length
      { st with s := s₂, q := q₂, oods := st.oods ++ oods₂ } true S'
      (fun M => oodDiff A (l'.logCols + l'.lanes) (flat M) (base M) (s₂, q₂, oods₂))
      (fun M => unsplit A (flat M)) (l.queryError + ε)
      (fun M hM hD M' hM' hD' => (hgood M hM hD).2 M' hM' hD')
      (fun M hM hD => (query_bound_at A hA n b initial l hf lanes st.prev ch st δ hδ hpost _
        (hgood M hM hD).1).trans hε) _ ?_
    intro w α β st' h1 h2 h3 h4 h5 h6 hno
    exact exists_zero_of_not_preFold A n b initial l l' st.prev ch st hpost.ok hdim C'
      s₂ q₂ oods₂ hsum₂ hshape _ _ β st' h1 h2 h3 h4 (h6 rfl) hno
  · calc value (fun y => ¬ OodGood A S' (fun M => flat M) base (l'.logCols + l'.lanes)
            st.levels.length st.s y) (twoOod A (l'.logCols + l'.lanes) st.levels.length st.s st.q) +
          (ENNReal.ofReal (l.queryError + ε) + ((ceilLog2 l.queries : ℝ≥0∞) / Fintype.card F +
            S'.card / Fintype.card F) + εk)
        ≤ ((S'.card.choose 2 : ℝ≥0∞) * (((l'.logCols + l'.lanes : ℕ) : ℝ≥0∞) / Fintype.card F) ^ 2 +
            2 * (S'.card / Fintype.card F)) + (ENNReal.ofReal l.queryError + ENNReal.ofReal ε +
              ((ceilLog2 l.queries : ℝ≥0∞) / Fintype.card F + S'.card / Fintype.card F) + εk) :=
          add_le_add (value_twoOod_good A hA S' (fun M => flat M) (l'.logCols + l'.lanes)
            (fun M _ M' _ h => flat_inj M M' (fun y hy => h y (by rw [hdim', hdim]; exact hy)))
            base st.levels.length st.s st.q hpost.sum)
            (add_le_add (add_le_add ENNReal.ofReal_add_le le_rfl) le_rfl)
      _ = ((S'.card.choose 2 : ℝ≥0∞) * (((l'.logCols + l'.lanes : ℕ) : ℝ≥0∞) /
            Fintype.card F) ^ 2 + 2 * (S'.card / Fintype.card F) + (ENNReal.ofReal l.queryError +
              ((ceilLog2 l.queries : ℝ≥0∞) / Fintype.card F + S'.card / Fintype.card F))) +
            ENNReal.ofReal ε + εk := by
          ring
      _ ≤ ENNReal.ofReal (midA l l') + ENNReal.ofReal ε + εk :=
          add_le_add (add_le_add (midA_bound l l' S'.card hS' hA.card_F) le_rfl) le_rfl

end FlockSoundness
