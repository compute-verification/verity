import Definitions.Pous
import Models.Pous.Assumptions
import Proofs.Pous.Statements
import Proofs.Pous.Sanity
import Proofs.Pous.Accounting.Numbers
import Definitions.Pous.Model.BandGraph

/-!
# Band-graph combinatorics (game-independent)

The band relation `Band k a b` (`a` is one of the `k` nodes before `b`), the **starts** of a set (no node of the set
among the `k` predecessors) and its **ends** (none among the `k` successors), the band graph `bandGraph n k`
(parents `v − k, …, v − 1`, a `basePath`), and the counting lemmas the column-saving certificate uses. Nothing here
refers to the column game.
-/

namespace PousP3Column

open Pous Pous.ColumnGame

variable {n : ℕ}

/-! ### Chains with a strictly increasing weight -/

theorem length_le_of_isChain {α : Type} {R : α → α → Prop} (f : α → ℕ) (hf : ∀ a b, R a b → f a < f b) (d : ℕ) :
    ∀ (a : α) (t : List α), (a :: t).IsChain R → (∀ v ∈ a :: t, f v ≤ d) → t.length + f a ≤ d := by
  intro a t
  induction t generalizing a with
  | nil => intro _ h; simpa using h a (List.mem_cons_self)
  | cons b t ih =>
    intro hc h
    rw [List.isChain_cons_cons] at hc
    have hab := hf a b hc.1
    have := ih b hc.2 fun v hv => h v (List.mem_cons_of_mem a hv)
    simp only [List.length_cons]
    omega

/-! ### The band relation, starts and windows -/

/-- `a` is one of the `k` nodes before `b`. -/
def Band (k : ℕ) (a b : Fin n) : Prop := a.val < b.val ∧ b.val ≤ a.val + k

instance instDecidableBand (k : ℕ) (a b : Fin n) : Decidable (Band k a b) := by unfold Band; infer_instance

/-- The nodes of `S` with no node of `S` among their `k` predecessors. -/
def starts (k : ℕ) (S : Finset (Fin n)) : Finset (Fin n) := S.filter fun v => ∀ u ∈ S, ¬ Band k u v

/-- The `k` predecessors of `v`. -/
def window (k : ℕ) (v : Fin n) : Finset (Fin n) := Finset.univ.filter fun u => Band k u v

theorem starts_subset (k : ℕ) (S : Finset (Fin n)) : starts k S ⊆ S := Finset.filter_subset _ _

theorem exists_band_of_not_start {k : ℕ} {S : Finset (Fin n)} {v : Fin n} (hv : v ∈ S) (hns : v ∉ starts k S) :
    ∃ u ∈ S, Band k u v := by
  by_contra hne
  push Not at hne
  exact hns (Finset.mem_filter.2 ⟨hv, fun u hu h => hne u hu h⟩)

theorem band_isChain_nodup {k : ℕ} {p : List (Fin n)} (hp : p.IsChain (Band k)) : p.Nodup := by
  have h2 : p.IsChain (· < ·) := hp.imp fun a b h => Fin.lt_def.2 h.1
  exact (List.isChain_iff_pairwise.1 h2).imp ne_of_lt

/-- A chain whose nodes carry strictly increasing ranks in `[1, r]` has at most `r` nodes. -/
theorem chain_length_le_of_rank {α : Type} {R : α → α → Prop} (A : α → Prop) (f : α → ℕ) (r : ℕ)
    (hf : ∀ a b, R a b → A a → A b → f a < f b) (hrange : ∀ a, A a → 1 ≤ f a ∧ f a ≤ r) :
    ∀ p : List α, p.IsChain R → (∀ x ∈ p, A x) → p.length ≤ r := by
  intro p hp hpA
  cases p with
  | nil => exact Nat.zero_le _
  | cons a t =>
    have hchain : (a :: t).IsChain (fun x y => R x y ∧ A x ∧ A y) := by
      clear hrange
      induction t generalizing a with
      | nil => exact List.isChain_singleton a
      | cons b t ih =>
        rw [List.isChain_cons_cons] at hp ⊢
        exact ⟨⟨hp.1, hpA a List.mem_cons_self, hpA b (List.mem_cons_of_mem a List.mem_cons_self)⟩,
          ih b hp.2 fun z hz => hpA z (List.mem_cons_of_mem a hz)⟩
    have := length_le_of_isChain f (fun x y h => hf x y h.1 h.2.1 h.2.2) r a t hchain
      fun v hv => (hrange v (hpA v hv)).2
    have h1 := (hrange a (hpA a List.mem_cons_self)).1
    simp only [List.length_cons]
    omega

/-! ### Nodes per start -/

/-- **Nodes per start.** If every chain of `S` has at most `ℓ` nodes, `S` has at most `ℓ` nodes per start. -/
theorem card_le_mul_starts (k : ℕ) : ∀ (ℓ : ℕ) (S : Finset (Fin n)),
    (∀ p : List (Fin n), p.IsChain (Band k) → (∀ x ∈ p, x ∈ S) → p.length ≤ ℓ) →
    S.card ≤ ℓ * (starts k S).card := by
  classical
  intro ℓ
  induction ℓ with
  | zero =>
    intro S h
    have : S = ∅ := by
      rw [Finset.eq_empty_iff_forall_notMem]
      intro v hv
      have := h [v] (List.isChain_singleton v) (by simpa using hv)
      simp at this
    simp [this]
  | succ ℓ ih =>
    intro S h
    set T := starts k S with hTdef
    set S' := S \ T with hS'def
    have hS' : ∀ p : List (Fin n), p.IsChain (Band k) → (∀ x ∈ p, x ∈ S') → p.length ≤ ℓ := by
      intro p hp hpS
      cases p with
      | nil => exact Nat.zero_le _
      | cons v t =>
        have hv := Finset.mem_sdiff.1 (hpS v List.mem_cons_self)
        obtain ⟨u, huS, huv⟩ := exists_band_of_not_start hv.1 hv.2
        have := h (u :: v :: t) (List.isChain_cons_cons.2 ⟨huv, hp⟩) (by
          intro x hx
          rcases List.mem_cons.1 hx with rfl | hx
          · exact huS
          · exact (Finset.mem_sdiff.1 (hpS x hx)).1)
        simp only [List.length_cons] at this ⊢
        omega
    have h1 := ih S' hS'
    -- every start of `S'` has a start of `S` among its predecessors, and distinct ones have distinct ones
    have h2 : (starts k S').card ≤ T.card := by
      let f : Fin n → Fin n := fun v => if hv : ∃ u ∈ S, Band k u v then Classical.choose hv else v
      have hf : ∀ v ∈ starts k S', f v ∈ S ∧ Band k (f v) v ∧ f v ∈ T := by
        intro v hv
        have hvS' := starts_subset k S' hv
        have hv' := Finset.mem_sdiff.1 hvS'
        have hex := exists_band_of_not_start hv'.1 hv'.2
        have hfv : f v = Classical.choose hex := by simp only [f, hex, ↓reduceDIte]
        obtain ⟨hS, hB⟩ := Classical.choose_spec hex
        rw [← hfv] at hS hB
        refine ⟨hS, hB, ?_⟩
        by_contra hT
        exact (Finset.mem_filter.1 hv).2 (f v) (Finset.mem_sdiff.2 ⟨hS, hT⟩) hB
      refine Finset.card_le_card_of_injOn f (fun v hv => (hf v hv).2.2) ?_
      intro a ha b hb hab
      obtain ⟨-, hBa, -⟩ := hf a ha
      obtain ⟨-, hBb, -⟩ := hf b hb
      rw [hab] at hBa
      unfold Band at hBa hBb
      by_contra hne
      rcases lt_or_gt_of_ne (Fin.val_ne_of_ne hne) with hlt | hlt
      · exact (Finset.mem_filter.1 hb).2 a (starts_subset k S' ha) ⟨hlt, by omega⟩
      · exact (Finset.mem_filter.1 ha).2 b (starts_subset k S' hb) ⟨hlt, by omega⟩
    have h3 : S.card = S'.card + T.card := by
      have hsub : T ⊆ S := starts_subset k S
      rw [hS'def, Finset.card_sdiff_of_subset hsub]
      have := Finset.card_le_card hsub
      omega
    calc S.card = S'.card + T.card := h3
      _ ≤ ℓ * (starts k S').card + T.card := by omega
      _ ≤ ℓ * T.card + T.card := by gcongr
      _ = (ℓ + 1) * T.card := by ring

/-! ### Windows of starts -/

theorem window_card (k : ℕ) (v : Fin n) (hv : k ≤ v.val) : k ≤ (window k v).card := by
  let f : ℕ → Fin n := fun j => if h : v.val - k + j < n then ⟨v.val - k + j, h⟩ else v
  have hmem : ∀ j ∈ Finset.range k, f j ∈ window k v := by
    intro j hj
    rw [Finset.mem_range] at hj
    have h : v.val - k + j < n := by omega
    have hfj : f j = ⟨v.val - k + j, h⟩ := by simp only [f, h, ↓reduceDIte]
    rw [hfj, window, Finset.mem_filter]
    refine ⟨Finset.mem_univ _, ?_⟩
    unfold Band
    simp only
    omega
  have hinj : Set.InjOn f (Finset.range k) := by
    intro a ha b hb hab
    rw [Finset.coe_range, Set.mem_Iio] at ha hb
    have ha' : v.val - k + a < n := by omega
    have hb' : v.val - k + b < n := by omega
    have hfa : f a = ⟨v.val - k + a, ha'⟩ := by simp only [f, ha', ↓reduceDIte]
    have hfb : f b = ⟨v.val - k + b, hb'⟩ := by simp only [f, hb', ↓reduceDIte]
    rw [hfa, hfb, Fin.mk.injEq] at hab
    omega
  simpa using Finset.card_le_card_of_injOn f hmem hinj

/-- **Windows of starts.** If every start of `S` has its window inside `Z`, then `|starts|·k ≤ |Z| + k`. -/
theorem starts_card_mul_le (k : ℕ) (S Z : Finset (Fin n)) (hZ : ∀ v ∈ starts k S, window k v ⊆ Z) :
    (starts k S).card * k ≤ Z.card + k := by
  classical
  set T := starts k S with hTdef
  set Tb := T.filter fun v => k ≤ v.val with hTb
  have hsmall : (T.filter fun v => ¬ k ≤ v.val).card ≤ 1 := by
    rw [Finset.card_le_one]
    intro a ha b hb
    rw [Finset.mem_filter] at ha hb
    by_contra hne
    rcases lt_or_gt_of_ne (Fin.val_ne_of_ne hne) with hlt | hlt
    · exact (Finset.mem_filter.1 hb.1).2 a (starts_subset k S ha.1) ⟨hlt, by omega⟩
    · exact (Finset.mem_filter.1 ha.1).2 b (starts_subset k S hb.1) ⟨hlt, by omega⟩
  have hsplit : Tb.card + (T.filter fun v => ¬ k ≤ v.val).card = T.card :=
    Finset.card_filter_add_card_filter_not (s := T) (fun v : Fin n => k ≤ v.val)
  have hdisj : (Tb : Set (Fin n)).PairwiseDisjoint (window k) := by
    intro a ha b hb hne
    rw [Finset.coe_filter] at ha hb
    rw [Function.onFun, Finset.disjoint_left]
    intro u hua hub
    simp only [window, Finset.mem_filter, Finset.mem_univ, true_and] at hua hub
    unfold Band at hua hub
    rcases lt_or_gt_of_ne (Fin.val_ne_of_ne hne) with hlt | hlt
    · have := (Finset.mem_filter.1 hb.1).2 a (starts_subset k S ha.1)
      exact this ⟨hlt, by omega⟩
    · have := (Finset.mem_filter.1 ha.1).2 b (starts_subset k S hb.1)
      exact this ⟨hlt, by omega⟩
  have hbig : Tb.card * k ≤ Z.card := by
    calc Tb.card * k = ∑ _v ∈ Tb, k := by rw [Finset.sum_const, smul_eq_mul]
      _ ≤ ∑ v ∈ Tb, (window k v).card :=
          Finset.sum_le_sum fun v hv => window_card k v (Finset.mem_filter.1 hv).2
      _ = (Tb.biUnion (window k)).card := (Finset.card_biUnion hdisj).symm
      _ ≤ Z.card := Finset.card_le_card (Finset.biUnion_subset.2 fun v hv => hZ v (Finset.mem_filter.1 hv).1)
  have : T.card ≤ Tb.card + 1 := by omega
  calc T.card * k ≤ (Tb.card + 1) * k := Nat.mul_le_mul_right _ this
    _ = Tb.card * k + k := by ring
    _ ≤ Z.card + k := by omega

/-! ### 1. Depth robustness -/

/-- **The counting core.** If every chain of `S` has at most `ℓ` nodes, then `k·|S| ≤ ℓ·(n − |S| + k)`. -/
theorem band_card_le (k : ℕ) (S : Finset (Fin n)) (ℓ : ℕ)
    (hℓ : ∀ p : List (Fin n), p.IsChain (Band k) → (∀ x ∈ p, x ∈ S) → p.length ≤ ℓ) :
    k * S.card ≤ ℓ * (n - S.card + k) := by
  classical
  have hA := card_le_mul_starts k ℓ S hℓ
  have hB := starts_card_mul_le k S (Finset.univ \ S) (by
    intro v hv u hu
    rw [Finset.mem_sdiff]
    refine ⟨Finset.mem_univ u, fun huS => ?_⟩
    exact (Finset.mem_filter.1 hv).2 u huS (Finset.mem_filter.1 hu).2)
  rw [Finset.card_sdiff_of_subset (Finset.subset_univ S), Finset.card_univ, Fintype.card_fin] at hB
  calc k * S.card ≤ k * (ℓ * (starts k S).card) := Nat.mul_le_mul_left _ hA
    _ = ℓ * ((starts k S).card * k) := by ring
    _ ≤ ℓ * (n - S.card + k) := Nat.mul_le_mul_left _ hB

/-- **Long paths in the band graph.** Every `S` contains a path of `p` nodes with `k·|S| ≤ p·(n − |S| + k)`. -/
theorem band_long_path (k : ℕ) (S : Finset (Fin n)) :
    ∃ p : List (Fin n), p.IsChain (Band k) ∧ p.Nodup ∧ (∀ x ∈ p, x ∈ S) ∧
      k * S.card ≤ p.length * (n - S.card + k) := by
  classical
  let P : ℕ → Prop := fun ℓ => ∀ p : List (Fin n), p.IsChain (Band k) → (∀ x ∈ p, x ∈ S) → p.length ≤ ℓ
  have hex : ∃ ℓ, P ℓ := by
    refine ⟨S.card, fun p hp hpS => ?_⟩
    rw [← List.toFinset_card_of_nodup (band_isChain_nodup hp)]
    exact Finset.card_le_card fun x hx => hpS x (List.mem_toFinset.1 hx)
  set ℓ := Nat.find hex with hℓ
  have hP : P ℓ := Nat.find_spec hex
  have hbound := band_card_le k S ℓ hP
  -- a chain attains `ℓ`
  obtain ⟨p, hp, hpS, hlen⟩ : ∃ p : List (Fin n), p.IsChain (Band k) ∧ (∀ x ∈ p, x ∈ S) ∧ ℓ ≤ p.length := by
    rcases Nat.eq_zero_or_pos ℓ with h0 | hpos
    · exact ⟨[], List.isChain_nil, by simp, by simp [h0]⟩
    · have hnot : ¬ P (ℓ - 1) := Nat.find_min hex (by omega)
      simp only [P, not_forall, not_le] at hnot
      obtain ⟨p, hp, hpS, hlt⟩ := hnot
      exact ⟨p, hp, hpS, by omega⟩
  exact ⟨p, hp, band_isChain_nodup hp, hpS, hbound.trans (Nat.mul_le_mul_right _ hlen)⟩

/-! ### The band graph -/

theorem mem_par_band {n k : ℕ} (hk : 1 ≤ k) (u v : Fin n) : u ∈ (bandGraph n k).par v ↔ Band k u v := by
  simp only [bandGraph, basePath, bandExtra, Finset.mem_union, Finset.mem_filter, Finset.mem_univ, true_and]
  unfold Band
  omega

/-- **Depth robustness.** The band graph is `(1/2, k/(n + 2k))`-depth-robust. -/
theorem band_depthRobust (n k : ℕ) (hk : 1 ≤ k) :
    DRG.DepthRobust n (fun a b => a ∈ (bandGraph n k).par b) (1 / 2) ((k : ℝ) / (n + 2 * k)) := by
  intro S hS
  obtain ⟨p, hp, hnd, hpS, hlen⟩ := band_long_path k S
  refine ⟨p, hp.imp fun a b h => (mem_par_band hk a b).2 h, hnd, hpS, ?_⟩
  have hSn : S.card ≤ n := by
    have := Finset.card_le_univ S
    rwa [Fintype.card_fin] at this
  have hlenR : (k : ℝ) * S.card ≤ p.length * ((n : ℝ) - S.card + k) := by
    have : ((k * S.card : ℕ) : ℝ) ≤ ((p.length * (n - S.card + k) : ℕ) : ℝ) := by exact_mod_cast hlen
    push_cast [Nat.cast_sub hSn] at this
    linarith
  have hk' : (1 : ℝ) ≤ k := by exact_mod_cast hk
  have hSn' : (S.card : ℝ) ≤ n := by exact_mod_cast hSn
  have hpos : (0 : ℝ) < n + 2 * k := by positivity
  rw [div_mul_eq_mul_div, div_le_iff₀ hpos]
  by_contra hlt
  push Not at hlt
  have hgap : (0 : ℝ) < (n : ℝ) - S.card + k := by linarith
  have hlen0 : (0 : ℝ) ≤ p.length := Nat.cast_nonneg _
  nlinarith [mul_lt_mul_of_pos_right hlt hgap, mul_le_mul_of_nonneg_right hlenR hpos.le,
    mul_nonneg (by linarith : (0 : ℝ) ≤ 2 * S.card - n) (by linarith : (0 : ℝ) ≤ n + k)]

/-- The nodes of `U` with no node of `U` among their `k` successors. -/
def ends (k : ℕ) (U : Finset (Fin n)) : Finset (Fin n) := U.filter fun a => ∀ b ∈ U, ¬ Band k a b

theorem starts_sdiff_ends_subset (k : ℕ) (U : Finset (Fin n)) : starts k (U \ ends k U) ⊆ starts k U := by
  intro v hv
  rw [starts, Finset.mem_filter] at hv ⊢
  refine ⟨(Finset.mem_sdiff.1 hv.1).1, fun u hu huv => ?_⟩
  by_cases hue : u ∈ ends k U
  · exact (Finset.mem_filter.1 hue).2 v (Finset.mem_sdiff.1 hv.1).1 huv
  · exact hv.2 u (Finset.mem_sdiff.2 ⟨hu, hue⟩) huv

/-- There are no more ends than starts: the next node after an end, if any, is a start, and the first node is a
start that no end reaches. -/
theorem ends_card_le_starts (k : ℕ) (U : Finset (Fin n)) : (ends k U).card ≤ (starts k U).card := by
  classical
  rcases U.eq_empty_or_nonempty with h | hne
  · simp [h, ends]
  set M := U.max' hne
  set m0 := U.min' hne
  let nxt : Fin n → Fin n := fun a => if h : (U.filter fun b => a < b).Nonempty then (U.filter fun b => a < b).min' h
    else a
  have hnxt : ∀ a ∈ U, a ≠ M → (U.filter fun b => a < b).Nonempty := by
    intro a ha haM
    refine ⟨M, Finset.mem_filter.2 ⟨U.max'_mem hne, lt_of_le_of_ne (U.le_max' a ha) haM⟩⟩
  have hnxt_spec : ∀ a ∈ U, a ≠ M → nxt a ∈ U ∧ a < nxt a ∧ ∀ b ∈ U, a < b → nxt a ≤ b := by
    intro a ha haM
    have h := hnxt a ha haM
    have he : nxt a = (U.filter fun b => a < b).min' h := by simp only [nxt, h, ↓reduceDIte]
    have hm := Finset.min'_mem _ h
    rw [Finset.mem_filter] at hm
    rw [he]
    exact ⟨hm.1, hm.2, fun b hb hab => Finset.min'_le _ _ (Finset.mem_filter.2 ⟨hb, hab⟩)⟩
  have hm0 : m0 ∈ starts k U := by
    refine Finset.mem_filter.2 ⟨U.min'_mem hne, fun u hu huv => ?_⟩
    have := U.min'_le u hu
    exact absurd huv.1 (not_lt.2 (Fin.le_def.1 this))
  have himg : ∀ a ∈ (ends k U).erase M, nxt a ∈ (starts k U).erase m0 := by
    intro a ha
    obtain ⟨haM, haE⟩ := Finset.mem_erase.1 ha
    have haU := (Finset.mem_filter.1 haE).1
    obtain ⟨hnU, hlt, hmin⟩ := hnxt_spec a haU haM
    have hfar : a.val + k < (nxt a).val := by
      by_contra hc
      exact (Finset.mem_filter.1 haE).2 (nxt a) hnU ⟨Fin.lt_def.1 hlt, by omega⟩
    refine Finset.mem_erase.2 ⟨?_, Finset.mem_filter.2 ⟨hnU, fun u hu huv => ?_⟩⟩
    · intro heq
      have : m0 ≤ a := U.min'_le a haU
      rw [← heq] at this
      exact absurd (Fin.lt_def.1 hlt) (not_lt.2 (Fin.le_def.1 this))
    · unfold Band at huv
      have := hmin u hu (Fin.lt_def.2 (by omega))
      exact absurd huv.1 (not_lt.2 (Fin.le_def.1 this))
  have hinj : Set.InjOn nxt ((ends k U).erase M) := by
    intro a ha b hb hab
    obtain ⟨haM, haE⟩ := Finset.mem_erase.1 ha
    obtain ⟨hbM, hbE⟩ := Finset.mem_erase.1 hb
    have haU := (Finset.mem_filter.1 haE).1
    have hbU := (Finset.mem_filter.1 hbE).1
    obtain ⟨-, hla, hma⟩ := hnxt_spec a haU haM
    obtain ⟨-, hlb, hmb⟩ := hnxt_spec b hbU hbM
    by_contra hne'
    rcases lt_or_gt_of_ne hne' with h | h
    · have := hma b hbU h
      rw [hab] at this
      exact absurd hlb (not_lt.2 this)
    · have := hmb a haU h
      rw [← hab] at this
      exact absurd hla (not_lt.2 this)
  have h1 := Finset.card_le_card_of_injOn nxt himg hinj
  have h2 : (ends k U).card ≤ ((ends k U).erase M).card + 1 := by
    rw [Finset.card_erase_eq_ite]; split_ifs <;> omega
  have h3 : ((starts k U).erase m0).card + 1 = (starts k U).card := Finset.card_erase_add_one hm0
  omega

/-- **Ranks per segment.** For a top set `U` with ranks `f ∈ [1, d]` increasing along band edges, and any
`p, q`: `q·Σ (f − 1) ≤ p·|U| + A(d)·|starts U|` with `A(d) = Σ_{i < d} (q·i − p)⁺`. Peel the ends: an end has
rank at most `d`, and the rest has rank at most `d − 1` and no more starts. -/
theorem band_rank_sum (k p q : ℕ) : ∀ (d : ℕ) (U : Finset (Fin n)) (f : Fin n → ℕ),
    (∀ a b, Band k a b → a ∈ U → b ∈ U → f a < f b) → (∀ a ∈ U, 1 ≤ f a ∧ f a ≤ d) →
    q * ∑ a ∈ U, (f a - 1) ≤ p * U.card + (∑ i ∈ Finset.range d, (q * i - p)) * (starts k U).card := by
  classical
  intro d
  induction d with
  | zero =>
    intro U f _ hr
    have : U = ∅ := Finset.eq_empty_of_forall_notMem fun a ha => by have := hr a ha; omega
    simp [this]
  | succ d ih =>
    intro U f hf hr
    set E := ends k U with hE
    set U' := U \ E with hU'
    have hEU : E ⊆ U := Finset.filter_subset _ _
    have hr' : ∀ a ∈ U', 1 ≤ f a ∧ f a ≤ d := by
      intro a ha
      obtain ⟨haU, haE⟩ := Finset.mem_sdiff.1 ha
      have : ∃ b ∈ U, Band k a b := by
        by_contra hne
        push Not at hne
        exact haE (Finset.mem_filter.2 ⟨haU, fun b hb h => hne b hb h⟩)
      obtain ⟨b, hb, hab⟩ := this
      have := hf a b hab haU hb
      have := hr b hb
      exact ⟨(hr a haU).1, by omega⟩
    have hIH := ih U' f (fun a b h ha hb => hf a b h (Finset.mem_sdiff.1 ha).1 (Finset.mem_sdiff.1 hb).1) hr'
    have hsplit : ∑ a ∈ U, (f a - 1) = ∑ a ∈ E, (f a - 1) + ∑ a ∈ U', (f a - 1) := by
      rw [hU', ← Finset.sum_sdiff hEU, add_comm]
    have hEsum : q * ∑ a ∈ E, (f a - 1) ≤ (p + (q * d - p)) * E.card := by
      rw [Finset.mul_sum]
      calc ∑ a ∈ E, q * (f a - 1) ≤ ∑ _a ∈ E, (p + (q * d - p)) :=
            Finset.sum_le_sum fun a ha => by
              have := hr a (hEU ha)
              have : q * (f a - 1) ≤ q * d := Nat.mul_le_mul_left _ (by omega)
              omega
        _ = (p + (q * d - p)) * E.card := by rw [Finset.sum_const, smul_eq_mul, mul_comm]
    have hcard : U.card = E.card + U'.card := by
      rw [hU', Finset.card_sdiff_of_subset hEU]
      have := Finset.card_le_card hEU
      omega
    have hEs : E.card ≤ (starts k U).card := ends_card_le_starts k U
    have hSs : (starts k U').card ≤ (starts k U).card := Finset.card_le_card (starts_sdiff_ends_subset k U)
    rw [Finset.sum_range_succ, hsplit, Nat.mul_add]
    have h1 : (∑ i ∈ Finset.range d, (q * i - p)) * (starts k U').card ≤
        (∑ i ∈ Finset.range d, (q * i - p)) * (starts k U).card := Nat.mul_le_mul_left _ hSs
    have h2 : (q * d - p) * E.card ≤ (q * d - p) * (starts k U).card := Nat.mul_le_mul_left _ hEs
    nlinarith [hEsum, hIH, h1, h2, hcard]

end PousP3Column
