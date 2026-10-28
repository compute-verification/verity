import Proofs.Flock.Claims.KS.JointJ
import Proofs.Flock.Claims.GS.Decode
import Proofs.Flock.Claims.DecodeFamily

/-!
# A table with unopened positions extends, at any code

KS at an untrimmed record (`FlockDeploy.deployZk`, `FlockDeploy.deployPlain`) quantifies over tables with unopened
positions (`0`, which no full-width column is), and the table bound is proved over full-width tables. What closes the
gap is that such a table extends: when it does not extract, filling its unopened positions with full-width columns
keeps it from extracting (`Claims.Flock.ExtendJ`; one table, `Claims.Flock.Extend`). This file proves that at any
record from one fact about its code, `Claims.Flock.Fillable`: at every word and position, some full-width column
other than `0` differs there from the column of every trace one agreement short of its radius. Each record supplies
the fact at its code, and the rest is one argument:

* `Claims.Flock.fill_step`: one unopened position filled keeps the table from extracting, since a trace short of its
  radius keeps its agreement and every other trace gains at most the one position (`Claims.GS.agree_fill`).
* `Claims.Flock.extendAt_of_fill`, `Claims.Flock.extendJ_of_fill`, `Claims.Flock.extend_of_fill`: so the positions
  filled one at a time give `ExtendJ`, and `Extend` at a one-table record.
* `Claims.Flock.tableBoundθ_of_extend`: the one-table bound at tables with unopened positions from the bound at
  full-width tables and `Extend` (`Claims.Flock.tableBoundJR_of_extend` is the per-table bound's).
* `Claims.Flock.fillable_of_avoid`: `Fillable` at a code of list columns from a first lane no trace short of its radius
  has at the position (`c :: 0…` at the column's width).
* `Claims.GS.exists_avoid`: that first lane, at an interleaved Reed–Solomon family whose traces short of the radius are
  few by Johnson's bound (`Claims.GS.card_lane0_le`), when the family's lengths summed stay below the field's size.
  `Claims.GS.reg_of_rate` gives Johnson's regime at rate at least `1/2` and `n ≥ 2^10`, which both Flock codes meet.
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

/-- **The regime one agreement short of the radius**: at `n ≥ 2^10` and rate `k/n ≥ 1/2`, every `t` from
`(√(k/n) + 1/200)·n − 1` on has `(k + 1)·n < t²`. -/
theorem reg_of_rate {k n t : ℕ} (hn : 1024 ≤ n) (hk : n ≤ 2 * k)
    (ht : (Real.sqrt ((k : ℝ) / n) + 1 / 200) * n - 1 ≤ t) : (k + 1) * n < t * t := by
  have hn' : (1024 : ℝ) ≤ n := by exact_mod_cast hn
  have hk' : (n : ℝ) ≤ 2 * k := by exact_mod_cast hk
  have hnpos : (0 : ℝ) < n := by linarith
  set s := Real.sqrt ((k : ℝ) / n) with hs
  have hs0 : 0 ≤ s := Real.sqrt_nonneg _
  have hnss : (n : ℝ) * s ^ 2 = k := by
    rw [hs, Real.sq_sqrt (by positivity)]
    field_simp
  have hs2 : (1 : ℝ) / 4 ≤ s ^ 2 := by
    have : (n : ℝ) / 4 ≤ n * s ^ 2 := by rw [hnss]; linarith
    nlinarith
  have hs1 : (1 : ℝ) / 2 ≤ s := by nlinarith
  have ht' : (n : ℝ) * s + n / 200 - 1 ≤ t := by linarith
  have hR : 0 < (n : ℝ) * s + n / 200 - 1 := by nlinarith
  have hT : ((n : ℝ) * s + n / 200 - 1) ^ 2 ≤ (t : ℝ) * t := by nlinarith
  have key : ((k : ℝ) + 1) * n < ((n : ℝ) * s + n / 200 - 1) ^ 2 := by
    have e : ((n : ℝ) * s + n / 200 - 1) ^ 2 =
        n * (n * s ^ 2) + 2 * (n * s) * (n / 200 - 1) + (n / 200 - 1) ^ 2 := by ring
    rw [e, hnss]
    have h1 : (n : ℝ) ≤ 2 * (n * s) * (n / 200 - 1) := by nlinarith
    have h2 : (0 : ℝ) < (n / 200 - 1) ^ 2 := by nlinarith
    nlinarith
  exact_mod_cast key.trans_le hT

end Claims.GS

namespace Claims.Flock

open FlockSoundness FlockSoundness.Game FlockSoundness.Extract FlockSoundness.Rewinding

/-! ## Filling unopened positions -/

section Fill

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]

/-- **Every position can be filled**: at table `j`'s word, at every word `W` and position `p` of it, some `Ok` column
other than `0` differs from the column there of every trace one agreement short of its radius at `W`. -/
def Fillable (D : Deploy Coin Col T) (N : Msg → ℕ → ℕ) (Ok : Msg → ℕ → Col → Prop) : Prop :=
  ∀ (x : Msg) (j : ℕ) (W : ℕ → Col) (p : ℕ), p < N x j → ∃ u, Ok x j u ∧ u ≠ 0 ∧ ∀ M : T,
    (1 - D.θ M) * D.len M ≤
        (((Finset.univ : Finset (Fin (D.len M))).filter fun q => D.enc M q.val = W q.val).card : ℝ) + 1 →
      D.enc M p ≠ u

/-- **A table with unopened positions extends**, at a one-table record: when it does not extract, some table of `Ok`
columns that keeps its opened ones does not extract either. -/
def Extend (D : Deploy Coin Col T) (N : Msg → ℕ) (Ok : Msg → Col → Prop) : Prop :=
  ∀ (x : Msg) (C : Fin (N x) → Col), (∀ p, orZero Ok x (C p)) → ¬ GoodTable D N x C →
    ∃ C' : Fin (N x) → Col, (∀ p, Ok x (C' p)) ∧ (∀ p, C p ≠ 0 → C' p = C p) ∧ ¬ GoodTable D N x C'

theorem extTab_update_self {n : ℕ} (C : Fin n → Col) (p : Fin n) (u : Col) :
    extTab (Function.update C p u) p.val = u := by
  simp [extTab, p.isLt]

theorem extTab_update_ne {n : ℕ} (C : Fin n → Col) (p : Fin n) (u : Col) {q : ℕ} (hq : q ≠ p.val) :
    extTab (Function.update C p u) q = extTab C q := by
  unfold extTab
  split_ifs with h
  · exact Function.update_of_ne (fun e => hq (congrArg Fin.val e)) _ _
  · rfl

/-- **One unopened position filled**: at a table of table `j`'s word that does not extract, any position takes an
`Ok` column other than `0` after which the table still does not extract (`Claims.GS.agree_fill`). -/
theorem fill_step {D : Deploy Coin Col T} {N : Msg → ℕ → ℕ} {Ok : Msg → ℕ → Col → Prop} (hfill : Fillable D N Ok)
    (x : Msg) (j : ℕ) (C : Fin (N x j) → Col) (hg : ¬ GoodTableAt D N x j C) (p : Fin (N x j)) :
    ∃ u, Ok x j u ∧ u ≠ 0 ∧ ¬ GoodTableAt D N x j (Function.update C p u) := by
  obtain ⟨u, hu, hu0, hav⟩ := hfill x j (extTab C) p.val p.isLt
  refine ⟨u, hu, hu0, ?_⟩
  rintro ⟨M, t, ht, hrel, hle, hag⟩
  refine hg ⟨M, t, ht, hrel, hle, Claims.GS.agree_fill (D.enc M) (extTab C)
    (extTab (Function.update C p u)) p.val (fun q hq => extTab_update_ne C p u hq) _ hag fun hcrit he => hav M hcrit ?_⟩
  rwa [extTab_update_self] at he

/-- **A table with unopened positions extends**, at table `j`: filled one position at a time (`fill_step`), it becomes
a table of `Ok` columns that keeps its opened ones and does not extract. -/
theorem extendAt_of_fill {D : Deploy Coin Col T} {N : Msg → ℕ → ℕ} {Ok : Msg → ℕ → Col → Prop}
    (hfill : Fillable D N Ok) (x : Msg) (j : ℕ) (C : Fin (N x j) → Col) (hC : ∀ p, orZeroJ Ok x j (C p))
    (hg : ¬ GoodTableAt D N x j C) :
    ∃ C' : Fin (N x j) → Col, (∀ p, Ok x j (C' p)) ∧ (∀ p, C p ≠ 0 → C' p = C p) ∧ ¬ GoodTableAt D N x j C' := by
  suffices H : ∀ k (C : Fin (N x j) → Col), (Finset.univ.filter fun p => C p = 0).card = k →
      (∀ p, orZeroJ Ok x j (C p)) → ¬ GoodTableAt D N x j C →
      ∃ C' : Fin (N x j) → Col, (∀ p, Ok x j (C' p)) ∧ (∀ p, C p ≠ 0 → C' p = C p) ∧
        ¬ GoodTableAt D N x j C' from H _ C rfl hC hg
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
    obtain ⟨u, hu, hu0, hg'⟩ := fill_step hfill x j C hg p
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
    have hC' : ∀ q, orZeroJ Ok x j (Function.update C p u q) := by
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

/-- **`ExtendJ` from `Fillable`.** -/
theorem extendJ_of_fill {D : Deploy Coin Col T} {N : Msg → ℕ → ℕ} {Ok : Msg → ℕ → Col → Prop}
    (hfill : Fillable D N Ok) : ExtendJ D N Ok :=
  fun x j _ C hC hg => extendAt_of_fill hfill x j C hC hg

/-- **`Extend` from `Fillable`** at table `0`. -/
theorem extend_of_fill {D : Deploy Coin Col T} {N : Msg → ℕ} {Ok : Msg → Col → Prop}
    (hfill : Fillable D (fun x _ => N x) (fun x _ => Ok x)) : Extend D N Ok :=
  fun x C hC hg => extendAt_of_fill hfill x 0 C hC hg

/-- **The fixed-table bound at tables with unopened positions**, from the bound at tables of `Ok` columns and
`Extend`, when accepted runs open only `Ok` columns and `Ok` admits no `0` on a nonempty word. -/
theorem tableBoundθ_of_extend (H : Msg → Msg) (D : Deploy Coin Col T) (N : Msg → ℕ) (Ok : Msg → Col → Prop)
    (hne : ∀ x u, 0 < N x → Ok x u → u ≠ 0) (hok : ColsOk H D N Ok) (hext : Extend D N Ok)
    (htab : TableBoundθ H D N Ok) : TableBoundθ H D N (orZero Ok) := by
  intro x Ω _ _ P
  obtain ⟨e, hfit, hb⟩ := htab x Ω P
  refine ⟨e, hfit, fun ω C hC hg => ?_⟩
  obtain ⟨C', hC', hkeep, hg'⟩ := hext x C hC hg
  refine (hb ω C' hC' hg').trans
    (add_le_add le_rfl (Pr_mono fun r ⟨ha, p, u, hu, hne'⟩ => ⟨ha, p, u, hu, ?_⟩))
  by_cases h0 : C p = 0
  · rw [h0]
    exact hne x u (Nat.lt_of_le_of_lt (Nat.zero_le _) p.isLt) (hok x Ω P ω r p.val u ha p.isLt hu)
  · rwa [← hkeep p h0]

end Fill

/-- **`Fillable` at a code of list columns**, from a first lane no trace one agreement short of its radius has at the
position, and a width of `Ok` columns at each nonempty word: the column `c :: 0…`. -/
theorem fillable_of_avoid {Coin F T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq F] [Zero F]
    (D : Deploy Coin (List F) T) (N : Msg → ℕ → ℕ) (Ok : Msg → ℕ → List F → Prop)
    (havoid : ∀ (W : ℕ → List F) (p : ℕ), ∃ c : F, ∀ M : T,
      (1 - D.θ M) * D.len M ≤
          (((Finset.univ : Finset (Fin (D.len M))).filter fun q => D.enc M q.val = W q.val).card : ℝ) + 1 →
        (D.enc M p)[0]? ≠ some c)
    (hw : ∀ x j, 0 < N x j → ∃ w, 0 < w ∧ ∀ u : List F, u.length = w → Ok x j u) : Fillable D N Ok := by
  intro x j W p hp
  obtain ⟨w, hw0, hwOk⟩ := hw x j (Nat.lt_of_le_of_lt (Nat.zero_le _) hp)
  obtain ⟨c, hc⟩ := havoid W p
  refine ⟨c :: List.replicate (w - 1) 0, hwOk _ (by simp; omega), List.cons_ne_nil _ _, fun M hM he => hc M hM ?_⟩
  rw [he]
  simp

end Claims.Flock
