import Proofs.Flock.Claims.KS.ZkUntrim
import Proofs.Flock.Claims.GS.Decode
import Proofs.Flock.Claims.GS.ZkRadius

/-!
# `ExtendJ` at the untrimmed masked instance

`Claims.Flock.ExtendJ D N Ok` asks, at each table `j` of a statement: a table at table `j`'s word's positions whose
columns are `Ok` or `0` (a position no rerun opened) and which does not extract (`GoodTableAt`) has a table of `Ok`
columns that keeps its nonzero ones and does not extract either. At the untrimmed masked instance
(`FlockDeploy.deployZk`), `Ok` is a column of table `j`'s witness lanes (`wideJ`) and an unopened position reads `[]`,
which no trace's column equals.

The unopened positions are filled one at a time (`zk_fill_step`), each with a column `c :: 0…` whose first lane `c`
no trace one agreement short of its radius has there (`Claims.GS.exists_avoid`); such a trace keeps its agreement,
and every other trace gains at most the one position, so none reaches its radius. That `c` exists because those
traces' first lanes are few: at one index of the family, distinct first-lane polynomials of degree below `k` agree at
fewer than `k` of the `n` distinct points (`Claims.GS.card_agree_le`), and each agrees with the word's first lane at
`t ≥ (1 − θ)·n − 1` of them, so Johnson's bound caps them at `n` once `(k + 1)·n < t²` (`Claims.GS.card_lane0_le`).
At the masked code that holds at every `m` served, since `n·k > 10^4` and `n ≥ 2^12` (`zk_reg`), and the indices'
`n = 2^d` summed stay below `|GF(2^128)|` (`zk_card`).

* `Claims.GS.agreeW`, `Claims.GS.famEnc_getElem_zero`, `Claims.GS.card_lane0_le`, `Claims.GS.exists_avoid`: the
  family's traces near a word, by their first lane.
* `Claims.GS.agree_fill`: a word changed at one position gains at most that position.
* `Claims.GS.reg_of_nk`, `FlockDeploy.KS.zk_reg`, `FlockDeploy.KS.zk_card`, `FlockDeploy.KS.zk_inj`,
  `FlockDeploy.KS.zk_avoid`: the masked code's regime, size and points, and the first lane to fill with.
* `FlockDeploy.KS.zk_fill_step`, `FlockDeploy.KS.zk_extendJ`: `ExtendJ` at `deployZk rel`, at every relation.
* `FlockDeploy.KS.zk_KS_untrim`: so `Flock.KS` at the untrimmed masked instance at SHA-512 at the deployed relation
  holds with no premise (`zk_KS_untrim_of_extendJ`).
-/

namespace Claims.GS

open Polynomial

section Avoid

variable {F ι : Type} [Field F] [Fintype F] [DecidableEq F] [Fintype ι] {s : ι → ℕ} {T : ι → Type}
  [∀ i, Fintype (T i)]

/-- **How many of its `n i` positions a trace's encoding agrees with the word `W` at.** -/
def agreeW (C : (i : ι) → IRS F (T i) (s i)) (n : ι → ℕ) (W : ℕ → List F) (M : Σ i, T i) : ℕ :=
  ((Finset.univ : Finset (Fin (n M.1))).filter fun q => IRS.famEnc C M q.val = W q.val).card

omit [Fintype F] [DecidableEq F] [Fintype ι] [∀ i, Fintype (T i)] in
/-- **A trace's column's first lane** is its first row at the position's point. -/
theorem famEnc_getElem_zero (C : (i : ι) → IRS F (T i) (s i)) (M : Σ i, T i) (q : ℕ) (h : 0 < s M.1) :
    (IRS.famEnc C M q)[0]? = some (((C M.1).rows M.2 ⟨0, h⟩).eval ((C M.1).dom q)) := by
  unfold IRS.famEnc
  rw [List.getElem?_ofFn]
  simp [h, IRS.enc]

omit [Fintype F] [Fintype ι] [∀ i, Fintype (T i)] in
/-- **The first lanes of an index's traces one agreement short of the radius number at most `n`**: Johnson's bound on
where each agrees with the word's first lane, distinct ones agreeing at fewer than `k` of the `n` distinct points. -/
theorem card_lane0_le (C : (i : ι) → IRS F (T i) (s i)) (hs : ∀ i, 0 < s i) (n : ι → ℕ) (θ : ι → ℝ)
    (hinj : ∀ i, Set.InjOn (C i).dom {q | q < n i})
    (hreg : ∀ i (t : ℕ), (1 - θ i) * n i - 1 ≤ t → ((C i).k + 1) * n i < t * t) (W : ℕ → List F) (i : ι)
    (S : Finset F[X])
    (hS : ∀ f ∈ S, ∃ m : T i, (1 - θ i) * n i ≤ agreeW C n W ⟨i, m⟩ + 1 ∧ f = (C i).rows m ⟨0, hs i⟩) :
    S.card ≤ n i := by
  classical
  rcases S.eq_empty_or_nonempty with h0 | ⟨f₀, hf₀⟩
  · rw [h0]
    exact Nat.zero_le _
  let A : F[X] → Finset (Fin (n i)) := fun f =>
    Finset.univ.filter fun q => (W q.val)[0]? = some (f.eval ((C i).dom q.val))
  let t := ⌈(1 - θ i) * n i - 1⌉₊
  have hAt : ∀ f ∈ S, t ≤ (A f).card := by
    intro f hf
    obtain ⟨m, hm, rfl⟩ := hS f hf
    have hsub : ((Finset.univ : Finset (Fin (n i))).filter fun q =>
        IRS.famEnc C ⟨i, m⟩ q.val = W q.val) ⊆ A ((C i).rows m ⟨0, hs i⟩) := by
      intro q hq
      simp only [Finset.mem_filter, Finset.mem_univ, true_and, A] at hq ⊢
      rw [← hq]
      exact famEnc_getElem_zero C ⟨i, m⟩ q.val (hs i)
    have hle : (agreeW C n W ⟨i, m⟩ : ℝ) ≤ (A ((C i).rows m ⟨0, hs i⟩)).card := by
      exact_mod_cast Finset.card_le_card hsub
    exact Nat.ceil_le.2 (by linarith)
  have hAI : ∀ f ∈ S, ∀ g ∈ S, f ≠ g → (A f ∩ A g).card ≤ (C i).k - 1 := by
    intro f hf g hg hfg
    obtain ⟨m, -, rfl⟩ := hS f hf
    obtain ⟨m', -, rfl⟩ := hS g hg
    have hx : Function.Injective fun q : Fin (n i) => (C i).dom q.val :=
      fun q q' h => Fin.ext (hinj i q.isLt q'.isLt h)
    refine le_trans (Finset.card_le_card fun q hq => ?_)
      (card_agree_le _ hx hfg ((C i).natDegree_rows m _) ((C i).natDegree_rows m' _))
    simp only [Finset.mem_inter, Finset.mem_filter, Finset.mem_univ, true_and, A] at hq ⊢
    exact Option.some_injective _ (hq.1.symm.trans hq.2)
  have hk1 : 1 ≤ (C i).k := by
    obtain ⟨m, -, -⟩ := hS f₀ hf₀
    have := (C i).natDegree_rows m ⟨0, hs i⟩
    omega
  have hreg' : ((C i).k - 1 + 2) * n i < t * t := by
    rw [show (C i).k - 1 + 2 = (C i).k + 1 by omega]
    exact hreg i t (Nat.le_ceil _)
  have htN : t ≤ n i := (hAt f₀ hf₀).trans ((Finset.card_le_univ _).trans_eq (Fintype.card_fin _))
  exact johnson_card_le S A hAt hAI hreg' htN

omit [Fintype ι] in
/-- **A first lane no trace one agreement short of its radius has at `p`**, at any word, when the indices' lengths
summed stay below the field's size. -/
theorem exists_avoid (C : (i : ι) → IRS F (T i) (s i)) (hs : ∀ i, 0 < s i) (n : ι → ℕ) (θ : ι → ℝ)
    (hinj : ∀ i, Set.InjOn (C i).dom {q | q < n i})
    (hreg : ∀ i (t : ℕ), (1 - θ i) * n i - 1 ≤ t → ((C i).k + 1) * n i < t * t) [Fintype ι]
    (hcard : ∑ i, n i < Fintype.card F) (W : ℕ → List F) (p : ℕ) :
    ∃ c : F, ∀ M : Σ i, T i, (1 - θ M.1) * n M.1 ≤ agreeW C n W M + 1 → (IRS.famEnc C M p)[0]? ≠ some c := by
  classical
  let S : (i : ι) → Finset F[X] := fun i =>
    (Finset.univ.filter fun m : T i => (1 - θ i) * n i ≤ agreeW C n W ⟨i, m⟩ + 1).image
      fun m => (C i).rows m ⟨0, hs i⟩
  let Bad : Finset F := Finset.univ.biUnion fun i => (S i).image fun f => f.eval ((C i).dom p)
  have hB : Bad.card < (Finset.univ : Finset F).card := by
    rw [Finset.card_univ]
    calc Bad.card ≤ ∑ i, ((S i).image fun f => f.eval ((C i).dom p)).card := Finset.card_biUnion_le
      _ ≤ ∑ i, (S i).card := Finset.sum_le_sum fun i _ => Finset.card_image_le
      _ ≤ ∑ i, n i := Finset.sum_le_sum fun i _ => card_lane0_le C hs n θ hinj hreg W i (S i) fun f hf => by
          obtain ⟨m, hm, rfl⟩ := Finset.mem_image.1 hf
          exact ⟨m, (Finset.mem_filter.1 hm).2, rfl⟩
      _ < Fintype.card F := hcard
  obtain ⟨c, -, hc⟩ := Finset.exists_mem_notMem_of_card_lt_card hB
  refine ⟨c, fun ⟨i, m⟩ hM he => hc ?_⟩
  rw [famEnc_getElem_zero C ⟨i, m⟩ p (hs i)] at he
  exact Finset.mem_biUnion.2 ⟨i, Finset.mem_univ _, Finset.mem_image.2
    ⟨_, Finset.mem_image.2 ⟨m, Finset.mem_filter.2 ⟨Finset.mem_univ _, hM⟩, rfl⟩, Option.some_injective _ he⟩⟩

end Avoid

/-- **A word changed at one position `p`** agrees with `e` at most there more, and nowhere more when it differs from
`e` there: so a bound `r` on its agreement, at most one short of the old word's or else with `e p ≠ W' p`, bounds the
old word's. -/
theorem agree_fill {α : Type} [DecidableEq α] {n : ℕ} (e W W' : ℕ → α) (p : ℕ) (h : ∀ q, q ≠ p → W' q = W q)
    (r : ℝ) (hr : r ≤ ((Finset.univ : Finset (Fin n)).filter fun q => e q.val = W' q.val).card)
    (hav : r ≤ ((Finset.univ : Finset (Fin n)).filter fun q => e q.val = W q.val).card + 1 → e p ≠ W' p) :
    r ≤ ((Finset.univ : Finset (Fin n)).filter fun q => e q.val = W q.val).card := by
  have hle : ((Finset.univ : Finset (Fin n)).filter fun q => e q.val = W' q.val).card ≤
      ((Finset.univ : Finset (Fin n)).filter fun q => e q.val = W q.val).card + 1 := by
    calc _ ≤ (((Finset.univ : Finset (Fin n)).filter fun q => e q.val = W q.val) ∪
          (Finset.univ.filter fun q : Fin n => q.val = p)).card := by
          refine Finset.card_le_card fun q hq => ?_
          rw [Finset.mem_filter] at hq
          by_cases hqp : q.val = p
          · exact Finset.mem_union_right _ (Finset.mem_filter.2 ⟨Finset.mem_univ _, hqp⟩)
          · exact Finset.mem_union_left _ (Finset.mem_filter.2 ⟨Finset.mem_univ _, hq.2.trans (h _ hqp)⟩)
      _ ≤ _ + (Finset.univ.filter fun q : Fin n => q.val = p).card := Finset.card_union_le _ _
      _ ≤ _ + 1 := Nat.add_le_add_left (Finset.card_le_one.2 fun a ha b hb =>
          Fin.ext ((Finset.mem_filter.1 ha).2.trans (Finset.mem_filter.1 hb).2.symm)) _
  by_cases hcrit : r ≤ ((Finset.univ : Finset (Fin n)).filter fun q => e q.val = W q.val).card + 1
  · have hp := hav hcrit
    have hsub : ((Finset.univ : Finset (Fin n)).filter fun q => e q.val = W' q.val) ⊆
        (Finset.univ : Finset (Fin n)).filter fun q => e q.val = W q.val := by
      intro q hq
      rw [Finset.mem_filter] at hq ⊢
      by_cases hqp : q.val = p
      · exact (hp (by rw [← hqp]; exact hq.2)).elim
      · exact ⟨Finset.mem_univ _, hq.2.trans (h _ hqp)⟩
    exact hr.trans (by exact_mod_cast Finset.card_le_card hsub)
  · have hle' : (((Finset.univ : Finset (Fin n)).filter fun q => e q.val = W' q.val).card : ℝ) ≤
        ((Finset.univ : Finset (Fin n)).filter fun q => e q.val = W q.val).card + 1 := by
      exact_mod_cast hle
    exact absurd (hr.trans hle') hcrit

/-- **The regime one agreement short of the radius**: at `n ≥ 2^12` and `n·k > 10^4`, every `t` from
`(√(k/n) + 1/200)·n − 1` on has `(k + 1)·n < t²`. -/
theorem reg_of_nk {k n t : ℕ} (hn : 4096 ≤ n) (hnk : 10000 < n * k)
    (ht : (Real.sqrt ((k : ℝ) / n) + 1 / 200) * n - 1 ≤ t) : (k + 1) * n < t * t := by
  have hn' : (4096 : ℝ) ≤ n := by exact_mod_cast hn
  have hnk' : (10000 : ℝ) < n * k := by exact_mod_cast hnk
  have hnpos : (0 : ℝ) < n := by linarith
  set s := Real.sqrt ((k : ℝ) / n) with hs
  have hs0 : 0 ≤ s := Real.sqrt_nonneg _
  have hnss : (n : ℝ) * s ^ 2 = k := by
    rw [hs, Real.sq_sqrt (by positivity)]
    field_simp
  have hns : 100 < (n : ℝ) * s := by
    have h2 : (10000 : ℝ) < ((n : ℝ) * s) ^ 2 := by
      calc (10000 : ℝ) < n * k := hnk'
        _ = ((n : ℝ) * s) ^ 2 := by rw [← hnss]; ring
    nlinarith [mul_nonneg hnpos.le hs0]
  have ht' : (n : ℝ) * s + n / 200 - 1 ≤ t := by linarith
  have hR : 0 < (n : ℝ) * s + n / 200 - 1 := by linarith
  have hT : ((n : ℝ) * s + n / 200 - 1) ^ 2 ≤ (t : ℝ) * t := by nlinarith
  have key : ((k : ℝ) + 1) * n < ((n : ℝ) * s + n / 200 - 1) ^ 2 := by
    have e : ((n : ℝ) * s + n / 200 - 1) ^ 2 =
        n * (n * s ^ 2) + 2 * (n * s) * (n / 200 - 1) + (n / 200 - 1) ^ 2 := by ring
    rw [e, hnss]
    have h1 : (n : ℝ) - 200 ≤ 2 * (n * s) * (n / 200 - 1) := by nlinarith
    have h2 : (200 : ℝ) < (n / 200 - 1) ^ 2 := by nlinarith
    nlinarith
  exact_mod_cast key.trans_le hT

end Claims.GS

namespace FlockDeploy.KS

open Flock

local notation "𝔽" => FlockLevel3.GF128

/-- Every masked level 0 the verifier serves has at least `2^12` positions (the least is `m = 25`'s `2^13`). -/
theorem zkLen_ge12 : ∀ i : Claims.Flock.ZkIdx, 12 ≤ Claims.Flock.zkLen i := by
  decide +kernel

/-- **The masked code's regime one agreement short of its radius**, at every `m` served. -/
theorem zk_reg (i : Claims.Flock.ZkIdx) (t : ℕ)
    (ht : (1 - (1 - Real.sqrt (((2 : ℝ) ^ Claims.Flock.zkCols i + Claims.Flock.zkPad i) /
      2 ^ Claims.Flock.zkLen i) - 1 / 200)) * ((2 ^ Claims.Flock.zkLen i : ℕ) : ℝ) - 1 ≤ t) :
    ((Claims.Flock.flockZkLevelZero i).k + 1) * 2 ^ Claims.Flock.zkLen i < t * t := by
  show (2 ^ Claims.Flock.zkCols i + Claims.Flock.zkPad i + 1) * 2 ^ Claims.Flock.zkLen i < t * t
  have hn : 4096 ≤ 2 ^ Claims.Flock.zkLen i :=
    le_trans (by norm_num) (Nat.pow_le_pow_right (by norm_num) (zkLen_ge12 i))
  refine Claims.GS.reg_of_nk hn (Claims.GS.zkShape_nk i) (le_of_eq_of_le ?_ ht)
  push_cast
  ring

/-- **The masked code's points are distinct** at each index's `2^d` positions. -/
theorem zk_inj (i : Claims.Flock.ZkIdx) :
    Set.InjOn (Claims.Flock.flockZkLevelZero i).dom {q | q < 2 ^ Claims.Flock.zkLen i} :=
  FlockSoundness.execArith_correct.omega_injective (Claims.Flock.zkLen i)
    ((Claims.Flock.zkShape_fits i).1.trans (by norm_num))

/-- **The indices' codeword lengths summed stay below `|GF(2^128)|`**: each is at most `2^40`. -/
theorem zk_card : ∑ i : Claims.Flock.ZkIdx, 2 ^ Claims.Flock.zkLen i < Fintype.card 𝔽 := by
  rw [FlockLevel3.card_F]
  calc ∑ i : Claims.Flock.ZkIdx, 2 ^ Claims.Flock.zkLen i ≤ ∑ _i : Claims.Flock.ZkIdx, 2 ^ 40 :=
        Finset.sum_le_sum fun i _ => Nat.pow_le_pow_right (by norm_num) (Claims.Flock.zkShape_fits i).1
    _ < 2 ^ 128 := by simp

namespace Gen

variable [S : ZkSetup]

/-- **A first lane no masked trace one agreement short of its radius has at `p`**, at any word `W`, read at the
untrimmed masked instance (`Claims.GS.exists_avoid` at the deployed masked code). -/
theorem zk_avoid (rel : Readable.Parsed → Readable.Relation) (W : ℕ → List 𝔽) (p : ℕ) :
    ∃ c : 𝔽, ∀ M : Claims.Flock.ZkTrace,
      (1 - (deployZk rel).θ M) * ((deployZk rel).len M : ℝ) ≤
        (((Finset.univ : Finset (Fin ((deployZk rel).len M))).filter fun q =>
          (deployZk rel).enc M q.val = W q.val).card : ℝ) + 1 →
      ((deployZk rel).enc M p)[0]? ≠ some c :=
  Claims.GS.exists_avoid Claims.Flock.flockZkLevelZero (fun _ => pow_pos (by norm_num) _)
    (fun i => 2 ^ Claims.Flock.zkLen i)
    (fun i => 1 - Real.sqrt (((2 : ℝ) ^ Claims.Flock.zkCols i + Claims.Flock.zkPad i) / 2 ^ Claims.Flock.zkLen i) -
      1 / 200) zk_inj zk_reg zk_card W p

end Gen

/-- **A first lane no masked trace one agreement short of its radius has at `p`**, at any word `W`, read at the
untrimmed masked instance (`Claims.GS.exists_avoid` at the deployed masked code). -/
theorem zk_avoid (rel : Readable.Parsed → Readable.Relation) (W : ℕ → List 𝔽) (p : ℕ) :
    ∃ c : 𝔽, ∀ M : Claims.Flock.ZkTrace,
      (1 - (deployZk rel).θ M) * ((deployZk rel).len M : ℝ) ≤
        (((Finset.univ : Finset (Fin ((deployZk rel).len M))).filter fun q =>
          (deployZk rel).enc M q.val = W q.val).card : ℝ) + 1 →
      ((deployZk rel).enc M p)[0]? ≠ some c :=
  Gen.zk_avoid (S := S₀) (rel := rel) (W := W) (p := p)

theorem extTab_update_self {n : ℕ} (C : Fin n → List 𝔽) (p : Fin n) (u : List 𝔽) :
    Claims.Flock.extTab (Function.update C p u) p.val = u := by
  simp [Claims.Flock.extTab, p.isLt]

theorem extTab_update_ne {n : ℕ} (C : Fin n → List 𝔽) (p : Fin n) (u : List 𝔽) {q : ℕ} (hq : q ≠ p.val) :
    Claims.Flock.extTab (Function.update C p u) q = Claims.Flock.extTab C q := by
  unfold Claims.Flock.extTab
  split_ifs with h
  · exact Function.update_of_ne (fun e => hq (congrArg Fin.val e)) _ _
  · rfl

/-- **One unopened position filled**: at a table of table `j`'s word that does not extract, any position takes a
full-width column, `c :: 0…` with `c` from `zk_avoid`, after which the table still does not extract
(`Claims.GS.agree_fill`). -/
theorem zk_fill_step (rel : Readable.Parsed → Readable.Relation) (x : Claims.Msg) (j : ℕ)
    (C : Fin (wordLenJ true x j) → List 𝔽)
    (hg : ¬ Claims.Flock.GoodTableAt (deployZk rel) (wordLenJ true) x j C) (p : Fin (wordLenJ true x j)) :
    ∃ u, wideJ true x j u ∧ u ≠ 0 ∧
      ¬ Claims.Flock.GoodTableAt (deployZk rel) (wordLenJ true) x j (Function.update C p u) := by
  obtain ⟨w, hw0, hwOk⟩ : ∃ w, 0 < w ∧ ∀ u : List 𝔽, u.length = w → wideJ true x j u := by
    have hp := p.isLt
    cases h : (tables true x)[j]? with
    | none => simp [wordLenJ, h] at hp
    | some a =>
      obtain ⟨_, st⟩ := a
      exact ⟨2 ^ (Claims.Flock.zkLevel0 st.m).lanes, by positivity, fun u hu => by simp [wideJ, h, hu]⟩
  obtain ⟨c, hc⟩ := zk_avoid rel (Claims.Flock.extTab C) p.val
  refine ⟨c :: List.replicate (w - 1) 0, hwOk _ (by simp; omega), List.cons_ne_nil _ _, ?_⟩
  rintro ⟨M, t, ht, hrel, hle, hag⟩
  refine hg ⟨M, t, ht, hrel, hle, Claims.GS.agree_fill ((deployZk rel).enc M) (Claims.Flock.extTab C)
    (Claims.Flock.extTab (Function.update C p (c :: List.replicate (w - 1) 0))) p.val
    (fun q hq => extTab_update_ne C p _ hq) _ hag fun hcrit he => hc M hcrit ?_⟩
  rw [extTab_update_self] at he
  rw [he]
  simp


namespace Gen

variable [S : ZkSetup]

/-- **One unopened position filled**: at a table of table `j`'s word that does not extract, any position takes a
full-width column, `c :: 0…` with `c` from `zk_avoid`, after which the table still does not extract
(`Claims.GS.agree_fill`). -/
theorem zk_fill_step (rel : Readable.Parsed → Readable.Relation) (x : Claims.Msg) (j : ℕ)
    (C : Fin (wordLenJ true x j) → List 𝔽)
    (hg : ¬ Claims.Flock.GoodTableAt (deployZk rel) (wordLenJ true) x j C) (p : Fin (wordLenJ true x j)) :
    ∃ u, wideJ true x j u ∧ u ≠ 0 ∧
      ¬ Claims.Flock.GoodTableAt (deployZk rel) (wordLenJ true) x j (Function.update C p u) := by
  obtain ⟨w, hw0, hwOk⟩ : ∃ w, 0 < w ∧ ∀ u : List 𝔽, u.length = w → wideJ true x j u := by
    have hp := p.isLt
    cases h : (tables true x)[j]? with
    | none => simp [wordLenJ, h] at hp
    | some a =>
      obtain ⟨_, st⟩ := a
      exact ⟨2 ^ (Claims.Flock.zkLevel0 st.m).lanes, by positivity, fun u hu => by simp [wideJ, h, hu]⟩
  obtain ⟨c, hc⟩ := zk_avoid rel (Claims.Flock.extTab C) p.val
  refine ⟨c :: List.replicate (w - 1) 0, hwOk _ (by simp; omega), List.cons_ne_nil _ _, ?_⟩
  rintro ⟨M, t, ht, hrel, hle, hag⟩
  refine hg ⟨M, t, ht, hrel, hle, Claims.GS.agree_fill ((deployZk rel).enc M) (Claims.Flock.extTab C)
    (Claims.Flock.extTab (Function.update C p (c :: List.replicate (w - 1) 0))) p.val
    (fun q hq => extTab_update_ne C p _ hq) _ hag fun hcrit he => hc M hcrit ?_⟩
  rw [extTab_update_self] at he
  rw [he]
  simp

end Gen

/-- **`ExtendJ` at the untrimmed masked instance**, at every relation: a table of table `j`'s word with unopened
positions that does not extract is filled one position at a time (`zk_fill_step`) to one of full-width columns that
keeps its opened ones and does not extract. -/
theorem zk_extendJ (rel : Readable.Parsed → Readable.Relation) :
    Claims.Flock.ExtendJ (deployZk rel) (wordLenJ true) (wideJ true) := by
  intro x j _ C hC hg
  suffices H : ∀ k (C : Fin (wordLenJ true x j) → List 𝔽), (Finset.univ.filter fun p => C p = 0).card = k →
      (∀ p, Claims.Flock.orZeroJ (wideJ true) x j (C p)) →
      ¬ Claims.Flock.GoodTableAt (deployZk rel) (wordLenJ true) x j C →
      ∃ C' : Fin (wordLenJ true x j) → List 𝔽, (∀ p, wideJ true x j (C' p)) ∧ (∀ p, C p ≠ 0 → C' p = C p) ∧
        ¬ Claims.Flock.GoodTableAt (deployZk rel) (wordLenJ true) x j C' from H _ C rfl hC hg
  intro k
  induction k with
  | zero =>
    intro C hk hC hg
    refine ⟨C, fun p => ?_, fun _ _ => rfl, hg⟩
    rcases hC p with h0 | h
    · have hmem : p ∈ Finset.univ.filter fun q => C q = 0 := Finset.mem_filter.2 ⟨Finset.mem_univ p, h0⟩
      rw [Finset.card_eq_zero.1 hk] at hmem
      exact absurd hmem (Finset.notMem_empty p)
    · exact h
  | succ k ih =>
    intro C hk hC hg
    have hpos : 0 < (Finset.univ.filter fun q => C q = 0).card := by
      rw [hk]
      exact Nat.succ_pos k
    obtain ⟨p, hpm⟩ := Finset.card_pos.1 hpos
    have hp : C p = 0 := (Finset.mem_filter.1 hpm).2
    obtain ⟨u, hu, hu0, hg'⟩ := zk_fill_step rel x j C hg p
    have hk' : (Finset.univ.filter fun q => Function.update C p u q = 0).card = k := by
      have e : (Finset.univ.filter fun q => Function.update C p u q = 0) =
          (Finset.univ.filter fun q => C q = 0).erase p := by
        ext q
        by_cases hq : q = p
        · subst hq
          simp [hu0]
        · simp [hq]
      rw [e, Finset.card_erase_of_mem hpm, hk]
      simp
    have hC' : ∀ q, Claims.Flock.orZeroJ (wideJ true) x j (Function.update C p u q) := by
      intro q
      by_cases hq : q = p
      · subst hq
        rw [Function.update_self]
        exact .inr hu
      · rw [Function.update_of_ne hq]
        exact hC q
    obtain ⟨C', hw, hkeep, hg''⟩ := ih _ hk' hC' hg'
    refine ⟨C', hw, fun q hq => ?_, hg''⟩
    have hqp : q ≠ p := fun h => hq (h ▸ hp)
    rw [hkeep q (by rwa [Function.update_of_ne hqp]), Function.update_of_ne hqp]


namespace Gen

variable [S : ZkSetup]

/-- **`ExtendJ` at the untrimmed masked instance**, at every relation: a table of table `j`'s word with unopened
positions that does not extract is filled one position at a time (`zk_fill_step`) to one of full-width columns that
keeps its opened ones and does not extract. -/
theorem zk_extendJ (rel : Readable.Parsed → Readable.Relation) :
    Claims.Flock.ExtendJ (deployZk rel) (wordLenJ true) (wideJ true) := by
  intro x j _ C hC hg
  suffices H : ∀ k (C : Fin (wordLenJ true x j) → List 𝔽), (Finset.univ.filter fun p => C p = 0).card = k →
      (∀ p, Claims.Flock.orZeroJ (wideJ true) x j (C p)) →
      ¬ Claims.Flock.GoodTableAt (deployZk rel) (wordLenJ true) x j C →
      ∃ C' : Fin (wordLenJ true x j) → List 𝔽, (∀ p, wideJ true x j (C' p)) ∧ (∀ p, C p ≠ 0 → C' p = C p) ∧
        ¬ Claims.Flock.GoodTableAt (deployZk rel) (wordLenJ true) x j C' from H _ C rfl hC hg
  intro k
  induction k with
  | zero =>
    intro C hk hC hg
    refine ⟨C, fun p => ?_, fun _ _ => rfl, hg⟩
    rcases hC p with h0 | h
    · have hmem : p ∈ Finset.univ.filter fun q => C q = 0 := Finset.mem_filter.2 ⟨Finset.mem_univ p, h0⟩
      rw [Finset.card_eq_zero.1 hk] at hmem
      exact absurd hmem (Finset.notMem_empty p)
    · exact h
  | succ k ih =>
    intro C hk hC hg
    have hpos : 0 < (Finset.univ.filter fun q => C q = 0).card := by
      rw [hk]
      exact Nat.succ_pos k
    obtain ⟨p, hpm⟩ := Finset.card_pos.1 hpos
    have hp : C p = 0 := (Finset.mem_filter.1 hpm).2
    obtain ⟨u, hu, hu0, hg'⟩ := zk_fill_step rel x j C hg p
    have hk' : (Finset.univ.filter fun q => Function.update C p u q = 0).card = k := by
      have e : (Finset.univ.filter fun q => Function.update C p u q = 0) =
          (Finset.univ.filter fun q => C q = 0).erase p := by
        ext q
        by_cases hq : q = p
        · subst hq
          simp [hu0]
        · simp [hq]
      rw [e, Finset.card_erase_of_mem hpm, hk]
      simp
    have hC' : ∀ q, Claims.Flock.orZeroJ (wideJ true) x j (Function.update C p u q) := by
      intro q
      by_cases hq : q = p
      · subst hq
        rw [Function.update_self]
        exact .inr hu
      · rw [Function.update_of_ne hq]
        exact hC q
    obtain ⟨C', hw, hkeep, hg''⟩ := ih _ hk' hC' hg'
    refine ⟨C', hw, fun q hq => ?_, hg''⟩
    have hqp : q ≠ p := fun h => hq (h ▸ hp)
    rw [hkeep q (by rwa [Function.update_of_ne hqp]), Function.update_of_ne hqp]

end Gen

namespace Gen

variable [S : ZkSetup]

/-- **Masked knowledge soundness at the untrimmed instance**, at SHA-512 at the deployed relation, with no premise:
`zk_KS_untrim_of_extendJ` with `ExtendJ` (`zk_extendJ`). -/
theorem zk_KS_untrim : Claims.Flock.KS sha512 (deployZk Readable.relation) :=
  zk_KS_untrim_of_extendJ (zk_extendJ Readable.relation)

end Gen

/-- **Masked knowledge soundness at the untrimmed instance**, at SHA-512 at the deployed relation, with no premise:
`zk_KS_untrim_of_extendJ` with `ExtendJ` (`zk_extendJ`). -/
theorem zk_KS_untrim : Claims.Flock.KS sha512 (deployZk Readable.relation) :=
  Gen.zk_KS_untrim (S := S₀)

end FlockDeploy.KS
