import Proofs.Flock.Soundness.LigeritoRes
import Proofs.Flock.Soundness.LigeritoLevels
import Proofs.Flock.Soundness.ListSize

/-!
# Decoding a fold near a codeword (PROTOCOL.md §13.4, §13.5)

A level's queries check the fold of its table against the next level's claimed fold. When the
folds were no MCA event (`MCAGood`) and the fold agrees with the encoding of a `K` message `G` on
at least a `1 − γ` fraction of the positions, `G` is the fold of a message in the level's list
(`decode`). Three facts carry it: every row then agrees with a `K` codeword on those positions
(`rowsAgreeOn_of_foldAll`); a `K` codeword agreeing with `F` values is an `F` codeword there, by
the limbs `K = F ⊕ F·u` (`poly_limbs`) and the novel basis spanning (`xP_span`); and the `K`
encoding is injective on that many positions (`encK_eq_zero`).
-/

namespace FlockSoundness

open Game Model Accounting Polynomial
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

/-- A table of `2^lanes` lanes and `2^d` positions, read from an oracle. -/
def tbl (lanes d : ℕ) (C : Oracle F) : Table F lanes d := fun ℓ q => C ℓ.val q.val

open Classical in
/-- **A level's list**: the messages whose encoding is `γ`-close to the table. -/
noncomputable def closeL (A : Arith F K) (γ : ℝ) (lanes cols d : ℕ) (C : Oracle F) :
    Finset (Message F lanes cols) :=
  Finset.univ.filter fun M => Close γ (tbl lanes d C) (encode A d M)

/-- A message's table, lanes lowest: `g(ℓ + 2^lanes·c) = M[ℓ][c]`. -/
def flat {lanes cols : ℕ} (M : Message F lanes cols) (y : ℕ) : F :=
  if h : y / 2 ^ lanes < 2 ^ cols then
    M ⟨y % 2 ^ lanes, Nat.mod_lt _ (Nat.two_pow_pos _)⟩ ⟨y / 2 ^ lanes, h⟩
  else 0

theorem flat_add_mul {lanes cols : ℕ} (M : Message F lanes cols) (ℓ : Fin (2 ^ lanes))
    (x : Fin (2 ^ cols)) : flat M (ℓ.val + 2 ^ lanes * x.val) = M ℓ x := by
  have hd : (ℓ.val + 2 ^ lanes * x.val) / 2 ^ lanes = x.val := by
    rw [Nat.add_mul_div_left _ _ (Nat.two_pow_pos _), Nat.div_eq_of_lt ℓ.isLt, zero_add]
  have hm : (ℓ.val + 2 ^ lanes * x.val) % 2 ^ lanes = ℓ.val := by
    rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt ℓ.isLt]
  unfold flat
  rw [dif_pos (by rw [hd]; exact x.isLt)]
  congr 1 <;> exact Fin.ext (by simp [hd, hm])

/-- A table's rows over `K`. -/
def rowsK (A : Arith F K) (lanes d : ℕ) (C : Oracle F) : Fin (2 ^ lanes) → Fin (2 ^ d) → K :=
  fun ℓ q => A.embed (C ℓ.val q.val)

/-- The encoding over `K` of a `K`-valued message on `2^cols` columns. -/
def encK (A : Arith F K) (cols : ℕ) (G : ℕ → K) (q : ℕ) : K :=
  ∑ x ∈ Finset.range (2 ^ cols), G x * A.embed (A.Xhat x cols (A.omega q))

/-- A message's lanes folded by `ch`: the `K` message the next level commits (split). -/
def foldMsg (A : Arith F K) {lanes cols : ℕ} (ch : List K) (M : Message F lanes cols) : ℕ → K :=
  foldLoL ch (fun y => A.embed (flat M y))

theorem foldMsg_eq (A : Arith F K) {lanes cols : ℕ} (ch : List K) (hch : ch.length = lanes)
    (M : Message F lanes cols) (x : Fin (2 ^ cols)) :
    foldMsg A ch M x.val = ∑ ℓ : Fin (2 ^ lanes), eqList ch ℓ.val * A.embed (M ℓ x) := by
  unfold foldMsg
  rw [foldLoL_eq_sum, ← Fin.sum_univ_eq_sum_range (fun i => eqList ch i *
    A.embed (flat M (i + 2 ^ ch.length * x.val))) (2 ^ ch.length)]
  subst hch
  exact Finset.sum_congr rfl fun ℓ _ => by rw [flat_add_mul]

/-! ## The novel basis spans; `K` polynomials split into limbs -/

theorem xP_span (A : Arith F K) (hA : A.Correct) (L : ℕ) (hL : L ≤ 40) (P : F[X])
    (hP : P.natDegree < 2 ^ L) :
    ∃ m : Fin (2 ^ L) → F, ∑ c, Polynomial.C (m c) * xP A hA L hL c = P := by
  classical
  haveI : Nonempty (Fin (2 ^ L)) := ⟨⟨0, Nat.two_pow_pos L⟩⟩
  have hdeg : ∀ p : F[X], p.natDegree < 2 ^ L → p ∈ Polynomial.degreeLT F (2 ^ L) := fun p hp =>
    Polynomial.mem_degreeLT.2 (lt_of_le_of_lt Polynomial.degree_le_natDegree
      (WithBot.coe_lt_coe.2 hp))
  let b : Fin (2 ^ L) → Polynomial.degreeLT F (2 ^ L) := fun c => ⟨xP A hA L hL c,
    hdeg _ (xP_natDegree A hA L hL c)⟩
  have hli : LinearIndependent F b :=
    LinearIndependent.of_comp (Polynomial.degreeLT F (2 ^ L)).subtype
      (xP_linearIndependent A hA L hL)
  have hcard : Fintype.card (Fin (2 ^ L)) = Module.finrank F (Polynomial.degreeLT F (2 ^ L)) := by
    rw [Fintype.card_fin, (Polynomial.degreeLTEquiv F (2 ^ L)).finrank_eq, Module.finrank_fin_fun]
  have hspan := hli.span_eq_top_of_card_eq_finrank hcard
  have hmem : (⟨P, hdeg P hP⟩ : Polynomial.degreeLT F (2 ^ L)) ∈ Submodule.span F (Set.range b) := by
    rw [hspan]; exact Submodule.mem_top
  obtain ⟨m, hm⟩ := (Submodule.mem_span_range_iff_exists_fun F).1 hmem
  refine ⟨m, ?_⟩
  have := congrArg Subtype.val hm
  simpa [b, Polynomial.smul_eq_C_mul] using this

theorem ofLimbs_add (A : Arith F K) (a₀ a₁ b₀ b₁ : F) :
    A.ofLimbs a₀ a₁ + A.ofLimbs b₀ b₁ = A.ofLimbs (a₀ + b₀) (a₁ + b₁) := by
  simp only [Arith.ofLimbs, map_add]; ring

theorem ofLimbs_mul_embed (A : Arith F K) (a₀ a₁ t : F) :
    A.ofLimbs a₀ a₁ * A.embed t = A.ofLimbs (a₀ * t) (a₁ * t) := by
  simp only [Arith.ofLimbs, map_mul]; ring

theorem ofLimbs_sum (A : Arith F K) {ι : Type} (s : Finset ι) (a₀ a₁ : ι → F) :
    ∑ i ∈ s, A.ofLimbs (a₀ i) (a₁ i) = A.ofLimbs (∑ i ∈ s, a₀ i) (∑ i ∈ s, a₁ i) := by
  simp only [Arith.ofLimbs, map_sum, Finset.sum_add_distrib, Finset.sum_mul]

theorem ofLimbs_zero (A : Arith F K) : A.ofLimbs 0 0 = 0 := by simp [Arith.ofLimbs]

theorem ofLimbs_embed (A : Arith F K) (a : F) : A.ofLimbs a 0 = A.embed a := by
  simp [Arith.ofLimbs]

theorem ofLimbs_inj (A : Arith F K) (hA : A.Correct) {a₀ a₁ b₀ b₁ : F}
    (h : A.ofLimbs a₀ a₁ = A.ofLimbs b₀ b₁) : a₀ = b₀ ∧ a₁ = b₁ := by
  have := hA.ofLimbs_bijective.1 (a₁ := (a₀, a₁)) (a₂ := (b₀, b₁)) h
  exact ⟨congrArg Prod.fst this, congrArg Prod.snd this⟩

/-- The limbs of a `K` element. -/
noncomputable def limbs (A : Arith F K) (hA : A.Correct) (a : K) : F × F :=
  (Equiv.ofBijective _ hA.ofLimbs_bijective).symm a

theorem ofLimbs_limbs (A : Arith F K) (hA : A.Correct) (a : K) :
    A.ofLimbs (limbs A hA a).1 (limbs A hA a).2 = a :=
  (Equiv.ofBijective _ hA.ofLimbs_bijective).apply_symm_apply a

/-- **A `K` polynomial read on `F` is two `F` polynomials**: `P(ω) = P₀(ω) + u·P₁(ω)`. -/
theorem poly_limbs (A : Arith F K) (hA : A.Correct) (P : K[X]) :
    ∃ P₀ P₁ : F[X], P₀.natDegree ≤ P.natDegree ∧
      ∀ ω : F, P.eval (A.embed ω) = A.ofLimbs (P₀.eval ω) (P₁.eval ω) := by
  classical
  let a := fun i => limbs A hA (P.coeff i)
  refine ⟨∑ i ∈ Finset.range (P.natDegree + 1), Polynomial.C (a i).1 * Polynomial.X ^ i,
    ∑ i ∈ Finset.range (P.natDegree + 1), Polynomial.C (a i).2 * Polynomial.X ^ i, ?_, fun ω => ?_⟩
  · refine Polynomial.natDegree_sum_le_of_forall_le _ _ fun i hi => ?_
    refine (Polynomial.natDegree_C_mul_le _ _).trans ?_
    rw [Polynomial.natDegree_X_pow]
    exact Nat.lt_succ_iff.1 (Finset.mem_range.1 hi)
  · rw [Polynomial.eval_eq_sum_range]
    simp only [Polynomial.eval_finset_sum, Polynomial.eval_mul, Polynomial.eval_C,
      Polynomial.eval_pow, Polynomial.eval_X]
    rw [← ofLimbs_sum]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [← map_pow, ← ofLimbs_mul_embed, ofLimbs_limbs]

/-! ## The `K` encoding -/

theorem encK_mem (A : Arith F K) (hA : A.Correct) (l : Level) (hf : l.Fits) (G : ℕ → K) :
    (fun q : Fin (2 ^ l.logLen) => encK A l.logCols G q.val) ∈ codeK A hA l hf := by
  refine ReedSolomon.mem_code_of_polynomial_of_natDegree_lt_of_eval
    (∑ x : Fin (2 ^ l.logCols), Polynomial.C (G x) * (xP A hA l.logCols hf.1 x).map A.embed) ?_
    fun q => ?_
  · have h : (∑ x : Fin (2 ^ l.logCols), Polynomial.C (G x) *
        (xP A hA l.logCols hf.1 x).map A.embed).natDegree ≤ 2 ^ l.logCols - 1 := by
      refine Polynomial.natDegree_sum_le_of_forall_le _ _ fun x _ => ?_
      refine (Polynomial.natDegree_C_mul_le _ _).trans ?_
      rw [Polynomial.natDegree_map]
      have := xP_natDegree A hA l.logCols hf.1 x
      omega
    have : 0 < 2 ^ l.logCols := Nat.two_pow_pos _
    omega
  · simp only [encK, domainK, Function.Embedding.trans_apply, Function.Embedding.coeFn_mk,
      domainEmb_apply, Polynomial.eval_finset_sum, Polynomial.eval_mul, Polynomial.eval_C,
      Polynomial.eval_map_apply]
    rw [← Fin.sum_univ_eq_sum_range (fun x => G x * A.embed (A.Xhat x l.logCols (A.omega q.val)))]
    exact Finset.sum_congr rfl fun x _ => by rw [xP_eval A hA l.logCols hf.1]

theorem card_gt_of_agree (l : Level) (T : Finset (Fin (2 ^ l.logLen)))
    (hT : (T.card : ℝ) ≥ Fintype.card (Fin (2 ^ l.logLen)) * (1 - l.radius)) :
    2 ^ l.logCols < T.card := by
  have hr : 0 < l.rate := by unfold Level.rate; positivity
  have hr1 : l.rate ≤ 1 := by
    unfold Level.rate; exact pow_le_one₀ (by norm_num) (by norm_num)
  have hsq : l.rate ≤ Real.sqrt l.rate :=
    calc l.rate = Real.sqrt (l.rate ^ 2) := (Real.sqrt_sq hr.le).symm
      _ ≤ Real.sqrt l.rate := Real.sqrt_le_sqrt (by nlinarith)
  have hlen : (2 : ℝ) ^ l.logLen * l.rate = 2 ^ l.logCols := by
    unfold Level.rate Level.logLen
    rw [pow_add, inv_pow, mul_assoc, mul_inv_cancel₀ (pow_ne_zero _ two_ne_zero), mul_one]
  rw [Fintype.card_fin] at hT
  push_cast at hT
  have h1 : 1 - l.radius = Real.sqrt l.rate + Accounting.eta := by unfold Level.radius; ring
  rw [h1] at hT
  have heta : (0 : ℝ) < Accounting.eta := by unfold Accounting.eta; norm_num
  have : (2 : ℝ) ^ l.logCols < T.card := by
    have h2 : (0 : ℝ) < 2 ^ l.logLen := by positivity
    nlinarith
  exact_mod_cast this

/-- **The `K` encoding is injective** on more than `2^cols` positions. -/
theorem encK_eq_zero (A : Arith F K) (hA : A.Correct) (l : Level) (hf : l.Fits) (a : ℕ → K)
    (T : Finset (Fin (2 ^ l.logLen))) (hT : 2 ^ l.logCols ≤ T.card)
    (h : ∀ q ∈ T, encK A l.logCols a q.val = 0) : ∀ x < 2 ^ l.logCols, a x = 0 := by
  classical
  set a₀ : Fin (2 ^ l.logCols) → F := fun x => (limbs A hA (a x)).1
  set a₁ : Fin (2 ^ l.logCols) → F := fun x => (limbs A hA (a x)).2
  have henc : ∀ q ∈ T, A.encodeAt l.logCols a₀ q.val = 0 ∧ A.encodeAt l.logCols a₁ q.val = 0 := by
    intro q hq
    have h0 := h q hq
    unfold encK at h0
    rw [← Fin.sum_univ_eq_sum_range (fun x => a x * A.embed (A.Xhat x l.logCols (A.omega q.val)))]
      at h0
    have h1 : ∑ x : Fin (2 ^ l.logCols), a x * A.embed (A.Xhat x l.logCols (A.omega q.val)) =
        A.ofLimbs (A.encodeAt l.logCols a₀ q.val) (A.encodeAt l.logCols a₁ q.val) := by
      unfold Arith.encodeAt
      rw [← ofLimbs_sum]
      refine Finset.sum_congr rfl fun x _ => ?_
      rw [← ofLimbs_mul_embed, ofLimbs_limbs]
    rw [h1, ← ofLimbs_zero A] at h0
    exact ofLimbs_inj A hA h0
  have hzero : ∀ b : Fin (2 ^ l.logCols) → F, (∀ q ∈ T, A.encodeAt l.logCols b q.val = 0) → b = 0 := by
    intro b hb
    have hp : rowPoly A hA hf.1 b = 0 := by
      refine Polynomial.eq_zero_of_natDegree_lt_card_of_eval_eq_zero' (rowPoly A hA hf.1 b)
        (T.image fun q => A.omega q.val) (fun ω hω => ?_) ?_
      · obtain ⟨q, hq, rfl⟩ := Finset.mem_image.1 hω
        rw [rowPoly_eval A hA hf.1]; exact hb q hq
      · rw [Finset.card_image_of_injOn fun q _ q' _ hqq' =>
          Fin.ext (hA.omega_injective l.logLen hf.2 q.isLt q'.isLt hqq')]
        exact lt_of_lt_of_le (rowPoly_natDegree A hA hf.1 b) hT
    have hsum : ∑ c, b c • xP A hA l.logCols hf.1 c = 0 := by
      rw [← hp]; simp only [rowPoly, Polynomial.smul_eq_C_mul]
    funext c
    exact Fintype.linearIndependent_iff.1 (xP_linearIndependent A hA l.logCols hf.1) _ hsum c
  intro x hx
  have e0 := congrFun (hzero a₀ fun q hq => (henc q hq).1) ⟨x, hx⟩
  have e1 := congrFun (hzero a₁ fun q hq => (henc q hq).2) ⟨x, hx⟩
  rw [← ofLimbs_limbs A hA (a x)]
  simp only [a₀, a₁, Pi.zero_apply] at e0 e1
  rw [e0, e1, ofLimbs_zero]

/-! ## Decoding -/

/-- **Decoding a fold.** If the folds of a level's table by `ch` were no MCA event and the fold
agrees with the encoding of `G` on a set `T` of at least `(1 − γ)·2^d` positions, then `G` is the
fold of a message in the level's list. -/
theorem decode (A : Arith F K) (hA : A.Correct) (l : Level) (hf : l.Fits) (lanes : ℕ)
    (C : Oracle F) (ch : List K) (hch : ch.length = lanes)
    (hgood : MCAGood (codeK A hA l hf) l.radius lanes ch (rowsK A lanes l.logLen C))
    (G : ℕ → K) (T : Finset (Fin (2 ^ l.logLen)))
    (hT : (T.card : ℝ) ≥ Fintype.card (Fin (2 ^ l.logLen)) * (1 - l.radius))
    (hag : ∀ q ∈ T, foldColumn A ch (fun ℓ => C ℓ q.val) = encK A l.logCols G q.val) :
    ∃ M ∈ closeL A l.radius lanes l.logCols l.logLen C,
      ∀ x < 2 ^ l.logCols, G x = foldMsg A ch M x := by
  classical
  haveI : CharP K 2 := hA.charTwoK
  have hfold : ∀ q : Fin (2 ^ l.logLen), foldAllRows lanes ch (rowsK A lanes l.logLen C) q =
      foldColumn A ch (fun ℓ => C ℓ q.val) := by
    intro q
    rw [foldAllRows_eq_sum lanes ch hch, foldColumn, hch,
      ← Fin.sum_univ_eq_sum_range (fun ℓ => eqList ch ℓ * A.embed (C ℓ q.val))]
    rfl
  -- every row agrees with a `K` codeword on `T`
  have hrows := rowsAgreeOn_of_foldAll (codeK A hA l hf) l.radius lanes ch _ hgood T hT
    ⟨_, encK_mem A hA l hf G, fun q hq => by rw [hfold q, hag q hq]⟩
  -- which is an `F` codeword there
  have hmsg : ∀ ℓ : Fin (2 ^ lanes), ∃ m : Fin (2 ^ l.logCols) → F,
      ∀ q ∈ T, A.encodeAt l.logCols m q.val = C ℓ q.val := by
    intro ℓ
    obtain ⟨c, hc, hcT⟩ := hrows ℓ
    haveI : NeZero (2 ^ l.logCols) := ⟨(Nat.two_pow_pos _).ne'⟩
    obtain ⟨P, hPdeg, hPc⟩ := ReedSolomon.mem_code_iff_eval_of_ne_zero.1 hc
    obtain ⟨P₀, P₁, hP₀, hPl⟩ := poly_limbs A hA P
    obtain ⟨m, hm⟩ := xP_span A hA l.logCols hf.1 P₀ (lt_of_le_of_lt hP₀ hPdeg)
    refine ⟨m, fun q hq => ?_⟩
    have h1 := hcT q hq
    rw [← hPc q] at h1
    simp only [domainK, Function.Embedding.trans_apply, Function.Embedding.coeFn_mk,
      domainEmb_apply] at h1
    rw [hPl, rowsK, ← ofLimbs_embed] at h1
    rw [← rowPoly_eval A hA hf.1, rowPoly, hm]
    exact (ofLimbs_inj A hA h1).1
  choose m hm using hmsg
  let M : Message F lanes l.logCols := fun ℓ x => m ℓ x
  have hclose : M ∈ closeL A l.radius lanes l.logCols l.logLen C := by
    rw [closeL, Finset.mem_filter]
    refine ⟨Finset.mem_univ _, ?_⟩
    unfold Close
    refine le_trans ?_ ((Nat.cast_le (α := ℝ)).2 (Finset.card_le_card (s := T) fun q hq => ?_))
    · rw [Fintype.card_fin] at hT; push_cast at hT; linarith
    · simp only [Finset.mem_filter, Finset.mem_univ, true_and]
      intro ℓ
      simp only [tbl, encode]
      exact (hm ℓ q hq).symm
  refine ⟨M, hclose, ?_⟩
  -- `G` and the fold of `M` have the same encoding on `T`
  have hdiff : ∀ q ∈ T, encK A l.logCols (fun x => G x - foldMsg A ch M x) q.val = 0 := by
    intro q hq
    have h1 : encK A l.logCols (foldMsg A ch M) q.val = foldColumn A ch (fun ℓ => C ℓ q.val) := by
      unfold encK
      rw [← Fin.sum_univ_eq_sum_range (fun x => foldMsg A ch M x *
        A.embed (A.Xhat x l.logCols (A.omega q.val)))]
      simp only [foldMsg_eq A ch hch M, Finset.sum_mul]
      rw [Finset.sum_comm, foldColumn, hch,
        ← Fin.sum_univ_eq_sum_range (fun ℓ => eqList ch ℓ * A.embed (C ℓ q.val))]
      refine Finset.sum_congr rfl fun ℓ _ => ?_
      rw [← hm ℓ q hq, Arith.encodeAt, map_sum, Finset.mul_sum]
      exact Finset.sum_congr rfl fun x _ => by simp only [map_mul, M]; ring
    unfold encK at h1 ⊢
    simp only [sub_mul, Finset.sum_sub_distrib]
    rw [h1, hag q hq]
    unfold encK
    exact sub_self _
  intro x hx
  exact sub_eq_zero.1 (encK_eq_zero A hA l hf _ T (card_gt_of_agree l T hT).le hdiff x hx)

end FlockSoundness
