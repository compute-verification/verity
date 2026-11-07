module

public import Mathlib.Data.Fin.Tuple.Take
public import Mathlib.Data.Int.CardIntervalMod
public import Mathlib.Data.Nat.Dist
public import Proofs.Flock.Soundness.ZK.RealLeaves
public import Proofs.Properties.Verity.Sampled.CoinMap
public import Proofs.Properties.Verity.Sampled.OpenEmpty

/-!
# Sampled proofs' coupling: one hiding swap per opened leaf

An opening's node costs the coupling two terms: the opening's gap and its empty class (`openGap`, `openEmpty`, in
`Proofs.Properties.Verity.Sampled.CoinMap`). Both are read off one partition, the rounds of the leaf's fiber
(`TreeOpenable`) by the coins they release, and one bound covers their sum: the partition's class error (`classErr`).
Over the sample of the leaf, the class error at the release the prover picks after seeing the leaf is within twice the
real leaf's distance from an ideal one, so one hiding swap per opened leaf pays for both terms.

* **A pick's preimages** (`pickIn_eq_iff`, `card_pickIn_eq`, `sum_tsub_card_pickIn_le`): `pickIn S` hits each element
  of `S` from `|C| / |S|` coins, or one more.
* **The class error** (`classNum`, `classErr`, `shortfall_pickIn_add_le`): over a uniform release `u` of `Q` and the
  classes `S u` of `N` rounds, `J` of them nonempty, a pick's shortfall in the class of `u` plus the probability that
  the class is empty is at most `classErr`: each class's excess `J·|S u| - N`, plus `N` per empty class, over `|Q|·N`.
* **Within the leaf's distance** (`core_le`, `classNum_le`, `mul_classErr_le`): a fiber's class error, weighted by
  the fiber's size, is at most the fiber's distance from the ideal fiber, whose classes are uniform.
* **One hiding swap** (`avg_iSup_classErr_le`, `avg_iSup_classErr_le_hm96`): over the leaf's sample, the class error at
  the worst release is at most `2·δ₁`, where `δ₁` bounds the real leaf's distance from an ideal one at every event of
  the leaf and the rest of the sample (for hm96's leaves, `Claims.Hm96Hiding` through `ideal_leaf_swap`).
* **At an opening's node** (`openGap_add_openEmpty_le`): the node's gap plus its empty class is at most its class
  error, wherever every round of the fiber releases some `bits u`.
* **At the deployed releases** (`injective_roundBits`, `injective_linkBits`, `openGap_add_openEmpty_le_round`): a
  round's bits determine its `m` challenges and a link's bits its two rounds', at every `m`, and a sample's first `m`
  coins are a round's release (`opened_eq_roundBits`, in `Proofs.Properties.Verity.Sampled.OpenEmpty`), so a round's
  opening is within its class error.
-/

@[expose] public section

namespace Verity.Game

open scoped ENNReal

/-! ## A pick's preimages -/

/-- **Where a pick lands**: `d` picks `e ∈ S` exactly where `d`'s index modulo `|S|` is `e`'s index in `S`. -/
theorem pickIn_eq_iff {C : Type} [Fintype C] {S : Finset C} (h : S.Nonempty) (d : C) {e : C} (he : e ∈ S) :
    pickIn S d = e ↔ (Fintype.equivFin C d).1 % S.card = (S.equivFin ⟨e, he⟩).1 := by
  unfold pickIn
  rw [dite_eq_left h]
  constructor
  · intro hd
    have h2 : S.equivFin.symm ⟨(Fintype.equivFin C d).1 % S.card, Nat.mod_lt _ (Finset.card_pos.2 h)⟩ = ⟨e, he⟩ :=
      Subtype.ext hd
    rw [Equiv.symm_apply_eq] at h2
    exact congrArg Fin.val h2
  · intro hd
    rw [show (⟨(Fintype.equivFin C d).1 % S.card, Nat.mod_lt _ (Finset.card_pos.2 h)⟩ : Fin S.card) =
      S.equivFin ⟨e, he⟩ from Fin.ext hd, Equiv.symm_apply_apply]

open Classical in
/-- **A pick's preimages**: `e ∈ S` is picked by `|C| / |S|` coins, and one more where `e`'s index in `S` is below
`|C| % |S|`. -/
theorem card_pickIn_eq {C : Type} [Fintype C] {S : Finset C} (h : S.Nonempty) {e : C} (he : e ∈ S) :
    (Finset.univ.filter fun d => pickIn S d = e).card =
      Fintype.card C / S.card + if (S.equivFin ⟨e, he⟩).1 % S.card < Fintype.card C % S.card then 1 else 0 := by
  have hk : 0 < S.card := Finset.card_pos.2 h
  have hj : (S.equivFin ⟨e, he⟩).1 < S.card := (S.equivFin ⟨e, he⟩).2
  have hto : ∀ d : C, pickIn S d = e → (Fintype.equivFin C d).1 ≡ (S.equivFin ⟨e, he⟩).1 [MOD S.card] :=
    fun d hd => by
      unfold Nat.ModEq
      rw [Nat.mod_eq_of_lt hj]
      exact (pickIn_eq_iff h d he).1 hd
  have hinv : ∀ (m : ℕ) (hm : m < Fintype.card C), m ≡ (S.equivFin ⟨e, he⟩).1 [MOD S.card] →
      pickIn S ((Fintype.equivFin C).symm ⟨m, hm⟩) = e := fun m hm hmod => by
    unfold Nat.ModEq at hmod
    rw [Nat.mod_eq_of_lt hj] at hmod
    rw [pickIn_eq_iff h _ he, Equiv.apply_symm_apply]
    exact hmod
  have hN : 0 < Fintype.card C := lt_of_lt_of_le hk (Finset.card_le_univ S)
  rw [← Nat.count_modEq_card (Fintype.card C) hk, Nat.count_eq_card_filter_range]
  refine Finset.card_nbij' (fun d => (Fintype.equivFin C d).1)
    (fun m => (Fintype.equivFin C).symm ⟨m % Fintype.card C, Nat.mod_lt _ hN⟩) ?_ ?_ ?_ ?_
  · intro d hd
    simp only [Finset.coe_filter, Finset.mem_univ, true_and, Set.mem_ofPred_eq, Finset.mem_range] at hd ⊢
    exact ⟨(Fintype.equivFin C d).2, hto d hd⟩
  · intro m hm
    simp only [Finset.coe_filter, Finset.mem_range, Set.mem_ofPred_eq, Finset.mem_univ, true_and] at hm ⊢
    have h := hinv m hm.1 hm.2
    rwa [show (⟨m % Fintype.card C, Nat.mod_lt _ hN⟩ : Fin (Fintype.card C)) = ⟨m, hm.1⟩ from
      Fin.ext (Nat.mod_eq_of_lt hm.1)]
  · intro d _
    simp only [Nat.mod_eq_of_lt (Fintype.equivFin C d).2, Fin.eta, Equiv.symm_apply_apply]
  · intro m hm
    simp only [Finset.coe_filter, Finset.mem_range, Set.mem_ofPred_eq] at hm
    simp only [Equiv.apply_symm_apply, Nat.mod_eq_of_lt hm.1]

open Classical in
theorem card_pickIn_bounds {C : Type} [Fintype C] {S : Finset C} (h : S.Nonempty) {e : C} (he : e ∈ S) :
    Fintype.card C / S.card ≤ (Finset.univ.filter fun d => pickIn S d = e).card ∧
      (Finset.univ.filter fun d => pickIn S d = e).card ≤ Fintype.card C / S.card + 1 := by
  rw [card_pickIn_eq h he]
  split_ifs <;> omega

open Classical in
/-- **Every coin picks once**: the preimages of `S`'s elements are all the coins. -/
theorem sum_card_pickIn {C : Type} [Fintype C] {S : Finset C} (h : S.Nonempty) :
    ∑ e ∈ S, (Finset.univ.filter fun d => pickIn S d = e).card = Fintype.card C := by
  rw [← Finset.card_univ]
  exact (Finset.card_eq_sum_card_fiberwise fun d _ => pickIn_mem h d).symm

open Classical in
/-- **A pick's deficits**: the elements of `S` picked by fewer than `J` coins miss at most `J·|S| - |C|` in all. -/
theorem sum_tsub_card_pickIn_le {C : Type} [Fintype C] {S : Finset C} (h : S.Nonempty) (J : ℕ) :
    ∑ e ∈ S, (J - (Finset.univ.filter fun d => pickIn S d = e).card) ≤ J * S.card - Fintype.card C := by
  by_cases hJ : J ≤ Fintype.card C / S.card
  · rw [Finset.sum_eq_zero fun e he => by have := (card_pickIn_bounds h he).1; omega]
    exact Nat.zero_le _
  · have hle : ∀ e ∈ S, (Finset.univ.filter fun d => pickIn S d = e).card ≤ J := fun e he => by
      have := (card_pickIn_bounds h he).2
      omega
    calc ∑ e ∈ S, (J - (Finset.univ.filter fun d => pickIn S d = e).card)
        = ∑ _e ∈ S, J - ∑ e ∈ S, (Finset.univ.filter fun d => pickIn S d = e).card :=
          Finset.sum_tsub_distrib S hle
      _ = J * S.card - Fintype.card C := by
          rw [Finset.sum_const, smul_eq_mul, sum_card_pickIn h, mul_comm]
      _ ≤ J * S.card - Fintype.card C := le_rfl

/-! ## The class error -/

/-- **A partition's class count**: for classes of sizes `k` over `Q`, `J` of them nonempty, of `N` elements, each
class's excess `J·k u - N`, plus `N` per empty class. -/
def classNum {Q : Type} [Fintype Q] (N : ℕ) (k : Q → ℕ) : ℕ :=
  ∑ u, ((Finset.univ.filter fun v => 0 < k v).card * k u - N) +
    N * (Fintype.card Q - (Finset.univ.filter fun v => 0 < k v).card)

/-- **A partition's class error**: its class count over `|Q|·N`. -/
noncomputable def classErr {Q : Type} [Fintype Q] (N : ℕ) (k : Q → ℕ) : ℝ≥0∞ :=
  (classNum N k : ℝ≥0∞) / (Fintype.card Q * N)

open Classical in
/-- **A pick in a uniform release's class**: where the classes `S u` partition `C` (`cls`), the shortfall of a pick
in the class of a uniform `u`, plus the probability that the class is empty, is at most the class error. -/
theorem shortfall_pickIn_add_le {Q C : Type} [Fintype Q] [Nonempty Q] [Fintype C] [Nonempty C] (S : Q → Finset C)
    (cls : C → Q) (hS : ∀ u d, d ∈ S u ↔ cls d = u) :
    shortfall (fun u => pickIn (S u)) + prCoin (fun u => ¬ (S u).Nonempty) ≤
      classErr (Fintype.card C) fun u => (S u).card := by
  have hK0 : (Fintype.card Q : ℝ≥0∞) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
  have hN0 : (Fintype.card C : ℝ≥0∞) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
  have hKt : (Fintype.card Q : ℝ≥0∞) ≠ ∞ := ENNReal.natCast_ne_top _
  have hNt : (Fintype.card C : ℝ≥0∞) ≠ ∞ := ENNReal.natCast_ne_top _
  obtain ⟨c, hc⟩ : ∃ c, (Fintype.card Q : ℝ≥0∞)⁻¹ * (Fintype.card C : ℝ≥0∞)⁻¹ = c := ⟨_, rfl⟩
  have hct : c ≠ ∞ := hc ▸ ENNReal.mul_ne_top (ENNReal.inv_ne_top.2 hK0) (ENNReal.inv_ne_top.2 hN0)
  have hNc : (Fintype.card C : ℝ≥0∞)⁻¹ = Fintype.card Q * c := by
    rw [← hc, ← mul_assoc, ENNReal.mul_inv_cancel hK0 hKt, one_mul]
  obtain ⟨J, hJ⟩ : ∃ J, (Finset.univ.filter fun v => 0 < (S v).card).card = J := ⟨_, rfl⟩
  have hJE : J + (Finset.univ.filter fun v => ¬ 0 < (S v).card).card = Fintype.card Q := by
    rw [← hJ, Finset.card_filter_add_card_filter_not, Finset.card_univ]
  have hA : ∀ e, (avg fun u : Q => avg fun d : C => if pickIn (S u) d = e then (1 : ℝ≥0∞) else 0) =
      (∑ u, ((Finset.univ.filter fun d => pickIn (S u) d = e).card : ℝ≥0∞)) * c := fun e => by
    simp only [avg_ite_eq_prCoin, prCoin_eq_card]
    simp only [avg]
    rw [Finset.sum_mul, ← hc]
    refine Finset.sum_congr rfl fun u _ => ?_
    rw [div_eq_mul_inv]
    ring
  have hpre : ∀ e, (Finset.univ.filter fun v => ¬ 0 < (S v).card).card +
      (Finset.univ.filter fun d => pickIn (S (cls e)) d = e).card ≤
        ∑ u, (Finset.univ.filter fun d => pickIn (S u) d = e).card := fun e => by
    have h1 : ∀ u ∈ Finset.univ.filter (fun v => ¬ 0 < (S v).card),
        (Finset.univ.filter fun d => pickIn (S u) d = e).card = 1 := fun u hu => by
      have h0 : S u = ∅ := Finset.card_eq_zero.1 (Nat.eq_zero_of_not_pos (Finset.mem_filter.1 hu).2)
      have hne : ¬ (S u).Nonempty := by
        rw [h0]
        exact Finset.not_nonempty_empty
      have hp : ∀ d, pickIn (S u) d = d := fun d => by
        unfold pickIn
        rw [dite_eq_right hne]
      rw [Finset.card_eq_one]
      exact ⟨e, Finset.ext fun d => by simp [hp]⟩
    have hn : cls e ∉ Finset.univ.filter (fun v => ¬ 0 < (S v).card) := fun hm =>
      (Finset.mem_filter.1 hm).2 (Finset.card_pos.2 ⟨e, (hS _ _).2 rfl⟩)
    calc (Finset.univ.filter fun v => ¬ 0 < (S v).card).card +
          (Finset.univ.filter fun d => pickIn (S (cls e)) d = e).card
        = ∑ u ∈ Finset.univ.filter (fun v => ¬ 0 < (S v).card),
              (Finset.univ.filter fun d => pickIn (S u) d = e).card +
            (Finset.univ.filter fun d => pickIn (S (cls e)) d = e).card := by
          rw [Finset.sum_congr rfl h1, Finset.sum_const, smul_eq_mul, mul_one]
      _ = ∑ u ∈ insert (cls e) (Finset.univ.filter fun v => ¬ 0 < (S v).card),
            (Finset.univ.filter fun d => pickIn (S u) d = e).card := by
          rw [Finset.sum_insert hn, add_comm]
      _ ≤ ∑ u, (Finset.univ.filter fun d => pickIn (S u) d = e).card :=
          Finset.sum_le_sum_of_subset (Finset.subset_univ _)
  have hsf : shortfall (fun u => pickIn (S u)) ≤
      ((∑ e, (J - (Finset.univ.filter fun d => pickIn (S (cls e)) d = e).card) : ℕ) : ℝ≥0∞) * c := by
    simp only [shortfall]
    rw [Nat.cast_sum, Finset.sum_mul]
    refine Finset.sum_le_sum fun e _ => ?_
    rw [hA e]
    calc (Fintype.card C : ℝ≥0∞)⁻¹ - (∑ u, ((Finset.univ.filter fun d => pickIn (S u) d = e).card : ℝ≥0∞)) * c
        ≤ (Fintype.card C : ℝ≥0∞)⁻¹ - (((Finset.univ.filter fun v => ¬ 0 < (S v).card).card +
            (Finset.univ.filter fun d => pickIn (S (cls e)) d = e).card : ℕ) : ℝ≥0∞) * c := by
          gcongr
          rw [← Nat.cast_sum]
          exact_mod_cast hpre e
      _ = ((Fintype.card Q - ((Finset.univ.filter fun v => ¬ 0 < (S v).card).card +
            (Finset.univ.filter fun d => pickIn (S (cls e)) d = e).card) : ℕ) : ℝ≥0∞) * c := by
          rw [hNc, ← ENNReal.sub_mul fun _ _ => hct, ENNReal.natCast_sub]
      _ = ((J - (Finset.univ.filter fun d => pickIn (S (cls e)) d = e).card : ℕ) : ℝ≥0∞) * c := by
          congr 2
          omega
  have hcls : ∑ e, (J - (Finset.univ.filter fun d => pickIn (S (cls e)) d = e).card) ≤
      ∑ u, (J * (S u).card - Fintype.card C) := by
    rw [← Finset.sum_fiberwise Finset.univ cls]
    refine Finset.sum_le_sum fun u _ => ?_
    have hSu : Finset.univ.filter (fun e => cls e = u) = S u := Finset.ext fun d => by simp [hS]
    calc ∑ e ∈ Finset.univ.filter (fun e => cls e = u),
          (J - (Finset.univ.filter fun d => pickIn (S (cls e)) d = e).card)
        = ∑ e ∈ S u, (J - (Finset.univ.filter fun d => pickIn (S u) d = e).card) :=
          Finset.sum_congr hSu fun e he => by rw [(hS u e).1 he]
      _ ≤ J * (S u).card - Fintype.card C := by
          by_cases hne : (S u).Nonempty
          · exact sum_tsub_card_pickIn_le hne J
          · rw [Finset.not_nonempty_iff_eq_empty.1 hne, Finset.sum_empty]
            exact Nat.zero_le _
  have hemp : prCoin (fun u => ¬ (S u).Nonempty) =
      ((Fintype.card C * (Fintype.card Q - J) : ℕ) : ℝ≥0∞) * c := by
    have hcnt : (Finset.univ.filter fun u => ¬ (S u).Nonempty).card = Fintype.card Q - J := by
      rw [← hJE, Nat.add_sub_cancel_left]
      exact congrArg Finset.card (Finset.filter_congr fun u _ => by rw [Finset.card_pos])
    rw [prCoin_eq_card, hcnt, Nat.cast_mul, div_eq_mul_inv, ← hc]
    calc ((Fintype.card Q - J : ℕ) : ℝ≥0∞) * (Fintype.card Q : ℝ≥0∞)⁻¹
        = ((Fintype.card Q - J : ℕ) : ℝ≥0∞) * (Fintype.card Q : ℝ≥0∞)⁻¹ *
            ((Fintype.card C : ℝ≥0∞) * (Fintype.card C : ℝ≥0∞)⁻¹) := by
          rw [ENNReal.mul_inv_cancel hN0 hNt, mul_one]
      _ = _ := by ring
  have hce : classErr (Fintype.card C) (fun u => (S u).card) =
      (classNum (Fintype.card C) (fun u => (S u).card) : ℝ≥0∞) * c := by
    rw [classErr, div_eq_mul_inv, ENNReal.mul_inv (Or.inl hK0) (Or.inl hKt), hc]
  calc shortfall (fun u => pickIn (S u)) + prCoin (fun u => ¬ (S u).Nonempty)
      ≤ ((∑ e, (J - (Finset.univ.filter fun d => pickIn (S (cls e)) d = e).card) : ℕ) : ℝ≥0∞) * c +
          ((Fintype.card C * (Fintype.card Q - J) : ℕ) : ℝ≥0∞) * c := add_le_add hsf hemp.le
    _ ≤ ((∑ u, (J * (S u).card - Fintype.card C) : ℕ) : ℝ≥0∞) * c +
          ((Fintype.card C * (Fintype.card Q - J) : ℕ) : ℝ≥0∞) * c :=
        add_le_add (mul_le_mul_left (Nat.cast_le.2 hcls) c) le_rfl
    _ = (classNum (Fintype.card C) (fun u => (S u).card) : ℝ≥0∞) * c := by
        rw [← add_mul, ← Nat.cast_add, classNum, hJ]
    _ = classErr (Fintype.card C) (fun u => (S u).card) := hce.symm

/-! ## The class error within the leaf's distance -/

theorem cast_tsub (a b : ℕ) : ((a - b : ℕ) : ℝ) = max ((a : ℝ) - b) 0 := by
  rcases le_total b a with h | h
  · rw [Nat.cast_sub h, max_eq_left (sub_nonneg.2 (Nat.cast_le.2 h))]
  · rw [Nat.sub_eq_zero_of_le h, Nat.cast_zero, max_eq_right (sub_nonpos.2 (Nat.cast_le.2 h))]

theorem cast_dist (a b : ℕ) : (Nat.dist a b : ℝ) = |(a : ℝ) - b| := by
  rw [Nat.dist, Nat.cast_add, cast_tsub, cast_tsub]
  rcases le_total (a : ℝ) b with h | h
  · rw [max_eq_right (sub_nonpos.2 h), max_eq_left (sub_nonneg.2 h), abs_of_nonpos (sub_nonpos.2 h)]
    ring
  · rw [max_eq_left (sub_nonneg.2 h), max_eq_right (sub_nonpos.2 h), abs_of_nonneg (sub_nonneg.2 h)]
    ring

theorem core_arith {K J W X₀ N A V T : ℝ} (hJK : J ≤ K) (hJ : 0 ≤ J) (hK : 0 < K) (hX : 0 ≤ X₀)
    (hT : K * W * N = T + J * X₀) (hA : 2 * (K * W * A) ≤ J * (V + |T|)) (hY : |T| ≤ V) :
    W * (A + N * (K - J)) ≤ V + (K - J) * X₀ := by
  have hT2 : K * W * N * (K - J) = (T + J * X₀) * (K - J) := by rw [hT]
  have h1 : 0 ≤ (K - J) * (V - T) := mul_nonneg (by linarith) (by linarith [le_abs_self T])
  have h2 : 0 ≤ (K - J) * (K - J) * X₀ := mul_nonneg (mul_self_nonneg _) hX
  have h3 : 0 ≤ J * (V - |T|) := mul_nonneg hJ (by linarith)
  have key : K * (W * (A + N * (K - J))) ≤ K * (V + (K - J) * X₀) := by nlinarith
  exact le_of_mul_le_mul_left key hK

/-- **A partition's excess within its distance from uniform**, in the reals: for classes of weights `k` supported in
`S`, `W` times the class count is at most the distance of `|Q|·W·k` from the constant `X₀`. -/
theorem core_le {Q : Type} [Fintype Q] [DecidableEq Q] (S : Finset Q) (k : Q → ℝ) (hoff : ∀ u ∉ S, k u = 0)
    {W X₀ : ℝ} (hX : 0 ≤ X₀) (hN : 0 ≤ ∑ v, k v) (hK : 0 < (Fintype.card Q : ℝ)) :
    W * (∑ u, max ((S.card : ℝ) * k u - ∑ v, k v) 0 + (∑ v, k v) * ((Fintype.card Q : ℝ) - S.card)) ≤
      ∑ u, |(Fintype.card Q : ℝ) * W * k u - X₀| := by
  have hNS : ∑ u ∈ S, k u = ∑ v, k v :=
    Finset.sum_subset (Finset.subset_univ S) fun u _ hu => hoff u hu
  have hA : ∑ u, max ((S.card : ℝ) * k u - ∑ v, k v) 0 = ∑ u ∈ S, max ((S.card : ℝ) * k u - ∑ v, k v) 0 :=
    (Finset.sum_subset (Finset.subset_univ S) fun u _ hu => by
      rw [hoff u hu, mul_zero, zero_sub]
      exact max_eq_right (by linarith)).symm
  have hoffS : ∀ u ∈ Sᶜ, |(Fintype.card Q : ℝ) * W * k u - X₀| = X₀ := fun u hu => by
    rw [hoff u (Finset.mem_compl.1 hu), mul_zero, zero_sub, abs_neg, abs_of_nonneg hX]
  have hR : ∑ u, |(Fintype.card Q : ℝ) * W * k u - X₀| =
      ∑ u ∈ S, |(Fintype.card Q : ℝ) * W * k u - X₀| + ((Fintype.card Q : ℝ) - S.card) * X₀ := by
    rw [← Finset.sum_add_sum_compl S, Finset.sum_congr rfl hoffS, Finset.sum_const, Finset.card_compl,
      nsmul_eq_mul, Nat.cast_sub (Finset.card_le_univ S)]
  have hJK : (S.card : ℝ) ≤ Fintype.card Q := by exact_mod_cast Finset.card_le_univ S
  have hJ0 : (0 : ℝ) ≤ S.card := Nat.cast_nonneg _
  have hTs : ∑ u ∈ S, ((Fintype.card Q : ℝ) * W * k u - X₀) =
      (Fintype.card Q : ℝ) * W * (∑ v, k v) - S.card * X₀ := by
    rw [Finset.sum_sub_distrib, ← Finset.mul_sum, hNS, Finset.sum_const, nsmul_eq_mul]
  rw [hA, hR]
  generalize (Fintype.card Q : ℝ) = K at hK hJK hTs ⊢
  generalize hJd : (S.card : ℝ) = J at hJK hJ0 hTs ⊢
  generalize ∑ v, k v = N at hN hTs ⊢
  have hper : ∀ u ∈ S, 2 * (K * W * max (J * k u - N) 0) ≤
      (J * (K * W * k u - X₀) - (K * W * N - J * X₀)) + J * |K * W * k u - X₀| + |K * W * N - J * X₀| := by
    intro u _
    have hJa : J * (K * W * k u - X₀) ≤ J * |K * W * k u - X₀| := mul_le_mul_of_nonneg_left (le_abs_self _) hJ0
    have hJa' : 0 ≤ J * (K * W * k u - X₀) + J * |K * W * k u - X₀| := by
      rw [← mul_add]
      exact mul_nonneg hJ0 (by linarith [neg_abs_le (K * W * k u - X₀)])
    have hT1 := neg_le_abs (K * W * N - J * X₀)
    have hT2 := le_abs_self (K * W * N - J * X₀)
    rcases le_total 0 (J * k u - N) with h | h
    · rw [max_eq_left h]
      linarith
    · rw [max_eq_right h]
      linarith
  refine core_arith (T := K * W * N - J * X₀) hJK hJ0 hK hX (by ring) ?_ ?_
  · calc 2 * (K * W * ∑ u ∈ S, max (J * k u - N) 0) = ∑ u ∈ S, 2 * (K * W * max (J * k u - N) 0) := by
          rw [Finset.mul_sum, Finset.mul_sum]
      _ ≤ ∑ u ∈ S, ((J * (K * W * k u - X₀) - (K * W * N - J * X₀)) + J * |K * W * k u - X₀| +
            |K * W * N - J * X₀|) := Finset.sum_le_sum hper
      _ = J * (∑ u ∈ S, (K * W * k u - X₀)) - J * (K * W * N - J * X₀) + J * (∑ u ∈ S, |K * W * k u - X₀|) +
            J * |K * W * N - J * X₀| := by
          rw [Finset.sum_add_distrib, Finset.sum_add_distrib, Finset.sum_sub_distrib, Finset.mul_sum,
            Finset.mul_sum, Finset.sum_const, Finset.sum_const, nsmul_eq_mul, nsmul_eq_mul, hJd]
      _ = J * (∑ u ∈ S, |K * W * k u - X₀| + |K * W * N - J * X₀|) := by
          rw [hTs]
          ring
  · rw [← hTs]
    exact Finset.abs_sum_le_sum_abs _ _

/-- **The class count within the distance from uniform**, in the reals. -/
theorem classNum_cast_le {Q : Type} [Fintype Q] [Nonempty Q] (k : Q → ℕ) (z : ℕ) {X₀ : ℝ} (hX : 0 ≤ X₀) :
    ((z * classNum (∑ u, k u) k : ℕ) : ℝ) ≤ ∑ u, |(Fintype.card Q : ℝ) * z * k u - X₀| := by
  classical
  have hJK := Finset.card_le_univ (Finset.univ.filter fun v => 0 < k v)
  have h := core_le (Finset.univ.filter fun v => 0 < k v) (fun u => (k u : ℝ))
    (fun u hu => by
      simp only [Finset.mem_filter, Finset.mem_univ, true_and, not_lt, Nat.le_zero] at hu
      simp [hu])
    (W := z) hX (Finset.sum_nonneg fun _ _ => Nat.cast_nonneg _) (by exact_mod_cast Fintype.card_pos)
  refine Eq.trans_le ?_ h
  rw [classNum, Nat.cast_mul, Nat.cast_add, Nat.cast_sum, Nat.cast_mul, Nat.cast_sub hJK, Nat.cast_sum]
  simp only [cast_tsub, Nat.cast_mul, Nat.cast_sum]

/-- **A fiber's class count within its distance from the ideal fiber**: where `Ω`'s classes `cl` are of one size,
weights `n` summed per class have the class count, scaled by `z`, at most `|Q|` times the distance of `z·n` from the
constant `c`. -/
theorem classNum_le {Ω Q : Type} [Fintype Ω] [Fintype Q] [Nonempty Q] [DecidableEq Q] (cl : Ω → Q)
    (hs : ∀ u, Fintype.card Q * (Finset.univ.filter fun ω => cl ω = u).card = Fintype.card Ω) (n : Ω → ℕ)
    (z c : ℕ) :
    z * classNum (∑ ω, n ω) (fun u => ∑ ω ∈ Finset.univ.filter (fun ω => cl ω = u), n ω) ≤
      Fintype.card Q * ∑ ω, Nat.dist (z * n ω) c := by
  have hNk : ∑ ω, n ω = ∑ u, ∑ ω ∈ Finset.univ.filter (fun ω => cl ω = u), n ω :=
    (Finset.sum_fiberwise Finset.univ cl n).symm
  have hsR : ∀ u, (Fintype.card Q : ℝ) * ((Finset.univ.filter fun ω => cl ω = u).card : ℝ) = Fintype.card Ω :=
    fun u => by exact_mod_cast hs u
  have hrhs : ∑ u, |(Fintype.card Q : ℝ) * z * ((∑ ω ∈ Finset.univ.filter (fun ω => cl ω = u), n ω : ℕ) : ℝ) -
      (Fintype.card Ω : ℝ) * c| ≤ (Fintype.card Q : ℝ) * ∑ ω, |(z : ℝ) * n ω - c| := by
    rw [← Finset.sum_fiberwise Finset.univ cl (fun ω => |(z : ℝ) * n ω - c|), Finset.mul_sum]
    refine Finset.sum_le_sum fun u _ => ?_
    have he : (Fintype.card Q : ℝ) * z * ((∑ ω ∈ Finset.univ.filter (fun ω => cl ω = u), n ω : ℕ) : ℝ) -
        (Fintype.card Ω : ℝ) * c =
          (Fintype.card Q : ℝ) * ∑ ω ∈ Finset.univ.filter (fun ω => cl ω = u), ((z : ℝ) * n ω - c) := by
      rw [Finset.sum_sub_distrib, ← Finset.mul_sum, Finset.sum_const, nsmul_eq_mul, Nat.cast_sum, ← hsR u]
      ring
    rw [he, abs_mul, abs_of_nonneg (Nat.cast_nonneg _ : (0 : ℝ) ≤ Fintype.card Q)]
    exact mul_le_mul_of_nonneg_left (Finset.abs_sum_le_sum_abs _ _) (Nat.cast_nonneg _)
  rw [hNk, ← Nat.cast_le (α := ℝ)]
  refine (classNum_cast_le _ z (X₀ := (Fintype.card Ω : ℝ) * c) (by positivity)).trans (hrhs.trans (le_of_eq ?_))
  push_cast [cast_dist]
  rfl

/-- **The weighted class error within a bound**: `N` times the class error is at most `S / z` wherever `z` times
the class count is at most `|Q|·S`. -/
theorem mul_classErr_le {Q : Type} [Fintype Q] [Nonempty Q] {N z S : ℕ} {k : Q → ℕ} (hz : z ≠ 0)
    (h : z * classNum N k ≤ Fintype.card Q * S) : (N : ℝ≥0∞) * classErr N k ≤ (S : ℝ≥0∞) / z := by
  rcases Nat.eq_zero_or_pos N with hN | hN
  · subst hN
    simp
  have hN0 : (N : ℝ≥0∞) ≠ 0 := by exact_mod_cast hN.ne'
  have hK0 : (Fintype.card Q : ℝ≥0∞) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
  have hz0 : (z : ℝ≥0∞) ≠ 0 := by exact_mod_cast hz
  calc (N : ℝ≥0∞) * classErr N k = (classNum N k : ℝ≥0∞) * N / (Fintype.card Q * N) := by
        rw [classErr, ← mul_div_assoc, mul_comm (N : ℝ≥0∞)]
    _ = (classNum N k : ℝ≥0∞) / Fintype.card Q := ENNReal.mul_div_mul_right _ _ hN0 (ENNReal.natCast_ne_top N)
    _ = z * (classNum N k : ℝ≥0∞) / (z * Fintype.card Q) :=
        (ENNReal.mul_div_mul_left _ _ hz0 (ENNReal.natCast_ne_top z)).symm
    _ ≤ (Fintype.card Q * S : ℝ≥0∞) / (z * Fintype.card Q) := ENNReal.div_le_div_right (by exact_mod_cast h) _
    _ = S / z := by
        rw [mul_comm (Fintype.card Q : ℝ≥0∞) (S : ℝ≥0∞)]
        exact ENNReal.mul_div_mul_right _ _ hK0 (ENNReal.natCast_ne_top _)

/-! ## One hiding swap -/

/-- **A distance within twice a bound on every event**: where each side's mass on every event `G` is within `δ` of
the other's, their distance is within `2·δ`. -/
theorem sum_dist_div_le {ι : Type} (s : Finset ι) (a b : ι → ℕ) {D : ℝ≥0∞} (hD : D ≠ 0) {δ : ℝ≥0∞}
    (h₁ : ∀ (G : ι → Prop) [DecidablePred G],
      ((∑ i ∈ s.filter G, a i : ℕ) : ℝ≥0∞) / D ≤ ((∑ i ∈ s.filter G, b i : ℕ) : ℝ≥0∞) / D + δ)
    (h₂ : ∀ (G : ι → Prop) [DecidablePred G],
      ((∑ i ∈ s.filter G, b i : ℕ) : ℝ≥0∞) / D ≤ ((∑ i ∈ s.filter G, a i : ℕ) : ℝ≥0∞) / D + δ) :
    ((∑ i ∈ s, Nat.dist (a i) (b i) : ℕ) : ℝ≥0∞) / D ≤ 2 * δ := by
  have hpart : ∀ a b : ι → ℕ, (∀ (G : ι → Prop) [DecidablePred G],
      ((∑ i ∈ s.filter G, a i : ℕ) : ℝ≥0∞) / D ≤ ((∑ i ∈ s.filter G, b i : ℕ) : ℝ≥0∞) / D + δ) →
      ((∑ i ∈ s, (a i - b i) : ℕ) : ℝ≥0∞) / D ≤ δ := fun a b h => by
    have e : ∑ i ∈ s, (a i - b i) =
        ∑ i ∈ s.filter (fun i => b i < a i), a i - ∑ i ∈ s.filter (fun i => b i < a i), b i := by
      rw [← Finset.sum_tsub_distrib (s.filter fun i => b i < a i) (f := a) (g := b)
        fun i hi => (Finset.mem_filter.1 hi).2.le, Finset.sum_filter]
      refine Finset.sum_congr rfl fun i _ => ?_
      split_ifs with hi
      · rfl
      · omega
    rw [e, ENNReal.natCast_sub, ENNReal.sub_div fun _ _ => hD, tsub_le_iff_left]
    exact h fun i => b i < a i
  have e : ∑ i ∈ s, Nat.dist (a i) (b i) = ∑ i ∈ s, (a i - b i) + ∑ i ∈ s, (b i - a i) := by
    rw [← Finset.sum_add_distrib]
    rfl
  rw [e, Nat.cast_add, ← ENNReal.div_add_div_same, two_mul]
  exact add_le_add (hpart a b h₁) (hpart b a h₂)

/-- **An event's complement the other way**: a bound on the complements' probabilities bounds the events' the other
way round. -/
theorem prCoin_swap_back {Xs Zs : Type} [Fintype Xs] [Nonempty Xs] [Fintype Zs] [Nonempty Zs] (E₁ : Xs → Prop)
    (E₂ : Zs → Prop) {δ : ℝ≥0∞} (h : prCoin (fun x => ¬ E₁ x) ≤ prCoin (fun z => ¬ E₂ z) + δ) :
    prCoin E₂ ≤ prCoin E₁ + δ := by
  have h1 := FlockSoundness.ZK.prCoin_add_not E₁
  have h2 := FlockSoundness.ZK.prCoin_add_not E₂
  rw [← ENNReal.add_le_add_iff_right (FlockSoundness.ZK.prCoin_ne_top fun z => ¬ E₂ z), h2]
  calc (1 : ℝ≥0∞) = prCoin E₁ + prCoin fun x => ¬ E₁ x := h1.symm
    _ ≤ prCoin E₁ + (prCoin (fun z => ¬ E₂ z) + δ) := add_le_add le_rfl h
    _ = prCoin E₁ + δ + prCoin fun z => ¬ E₂ z := by ring

/-- **A pair's event, counted by its second coordinate's map**: the pairs `(ω, y)` whose `(ω, L (ω, y))` satisfies
`G`, counted over the values `L` takes. -/
theorem card_filter_pair {Ω Ys Λ : Type} [Fintype Ω] [Fintype Ys] [DecidableEq Λ] (L : Ω × Ys → Λ) (t : Finset Λ)
    (ht : ∀ p, L p ∈ t) (G : Ω × Λ → Prop) [DecidablePred G] :
    (Finset.univ.filter fun p : Ω × Ys => G (p.1, L p)).card =
      ∑ i ∈ (Finset.univ ×ˢ t).filter G, (Finset.univ.filter fun y => L (i.1, y) = i.2).card := by
  rw [Finset.card_filter, Fintype.sum_prod_type, Finset.sum_filter, Finset.sum_product]
  dsimp only
  refine Finset.sum_congr rfl fun ω _ => ?_
  rw [← Finset.sum_fiberwise_of_maps_to (s := Finset.univ) (t := t) (g := fun y => L (ω, y))
    fun y _ => ht (ω, y)]
  refine Finset.sum_congr rfl fun ℓ _ => ?_
  have hc : ∀ y ∈ Finset.univ.filter fun y => L (ω, y) = ℓ,
      (if G (ω, L (ω, y)) then (1 : ℕ) else 0) = if G (ω, ℓ) then 1 else 0 := fun y hy => by
    rw [(Finset.mem_filter.1 hy).2]
  rw [Finset.sum_congr rfl hc, Finset.sum_const, smul_eq_mul]
  split_ifs <;> simp

theorem prCoin_pair_eq {Ω Ys Λ : Type} [Fintype Ω] [Fintype Ys] [DecidableEq Λ] (L : Ω × Ys → Λ) (t : Finset Λ)
    (ht : ∀ p, L p ∈ t) (G : Ω × Λ → Prop) [DecidablePred G] :
    prCoin (fun p : Ω × Ys => G (p.1, L p)) =
      ((∑ i ∈ (Finset.univ ×ˢ t).filter G, (Finset.univ.filter fun y => L (i.1, y) = i.2).card : ℕ) : ℝ≥0∞) /
        Fintype.card (Ω × Ys) := by
  rw [prCoin_eq_card, card_filter_pair L t ht G]

/-- **Events of a release under its sampling design**: a map whose means are those of `bits` gives every event of it the
probability of the event of `bits`. -/
theorem prCoin_law {Ω Q R : Type} [Fintype Ω] [Fintype Q] {P : Ω → R} {bits : Q → R}
    (hlaw : ∀ g : R → ℝ≥0∞, avg (fun ω => g (P ω)) = avg fun u => g (bits u)) (A : R → Prop) :
    prCoin (fun ω => A (P ω)) = prCoin fun u => A (bits u) := by
  classical
  unfold prCoin
  exact hlaw fun r => if A r then 1 else 0

/-- **A mean through a map, bounded on each value**: where each value `ℓ` of `L`, weighted by its preimage's size,
has `F ℓ` within `B ℓ`, the mean of `F ∘ L` is within the values' `B` over `|Xs|`. -/
theorem avg_comp_le {Xs Λ : Type} [Fintype Xs] [DecidableEq Λ] (L : Xs → Λ) (F B : Λ → ℝ≥0∞)
    (h : ∀ ℓ, ((Finset.univ.filter fun x => L x = ℓ).card : ℝ≥0∞) * F ℓ ≤ B ℓ) :
    avg (fun x => F (L x)) ≤ ∑ ℓ ∈ Finset.univ.image L, (Fintype.card Xs : ℝ≥0∞)⁻¹ * B ℓ := by
  unfold avg
  rw [← Finset.sum_fiberwise_of_maps_to (s := Finset.univ) (t := Finset.univ.image L) (g := L)
    fun x _ => Finset.mem_image_of_mem L (Finset.mem_univ x)]
  refine Finset.sum_le_sum fun ℓ _ => ?_
  have hc : ∀ x ∈ Finset.univ.filter fun x => L x = ℓ,
      (Fintype.card Xs : ℝ≥0∞)⁻¹ * F (L x) = (Fintype.card Xs : ℝ≥0∞)⁻¹ * F ℓ :=
    fun x hx => by rw [(Finset.mem_filter.1 hx).2]
  rw [Finset.sum_congr rfl hc, Finset.sum_const, nsmul_eq_mul, mul_left_comm]
  exact mul_le_mul_right (h ℓ) _

/-- **One hiding swap per leaf**: a leaf `L (ω, y)` of the sample `ω` and the salt `y`, within `δ₁` of an ideal leaf
`LI z` at every event of the sample and the leaf (`hswap`), and releases `P i ω` of the sample, each distributed as an
injective `bits i` of a uniform `u` (`hlaw`). Over the sample and the salt, the class error of the leaf's fiber
partitioned by the release, at the worst release `i`, is at most `2·δ₁`. -/
theorem avg_iSup_classErr_le {Ω Ys Zs Λ ι : Type} [Fintype Ω] [Nonempty Ω] [Fintype Ys] [Nonempty Ys] [Fintype Zs]
    [Nonempty Zs] [DecidableEq Λ] (L : Ω × Ys → Λ) (LI : Zs → Λ) {δ₁ : ℝ≥0∞}
    (hswap : ∀ G : Ω → Λ → Prop,
      prCoin (fun p : Ω × Ys => G p.1 (L p)) ≤ prCoin (fun p : Ω × Zs => G p.1 (LI p.2)) + δ₁)
    {R : ι → Type} [∀ i, DecidableEq (R i)] (P : (i : ι) → Ω → R i) {Q : ι → Type} [∀ i, Fintype (Q i)]
    [∀ i, Nonempty (Q i)] (bits : (i : ι) → Q i → R i) (hinj : ∀ i, Function.Injective (bits i))
    (hlaw : ∀ i (g : R i → ℝ≥0∞), avg (fun ω => g (P i ω)) = avg fun u => g (bits i u)) :
    avg (fun p₀ : Ω × Ys => ⨆ i, classErr (Finset.univ.filter fun p => L p = L p₀).card
      fun u : Q i => (Finset.univ.filter fun p : Ω × Ys => L p = L p₀ ∧ P i p.1 = bits i u).card) ≤ 2 * δ₁ := by
  classical
  have hcov : ∀ i ω, ∃ u, P i ω = bits i u := by
    intro i ω
    by_contra hne
    have h0 : prCoin (fun ω => ¬ ∃ u, P i ω = bits i u) = 0 := by
      rw [prCoin_law (hlaw i) (fun r => ¬ ∃ u, r = bits i u)]
      exact le_antisymm ((prCoin_mono fun v hv => hv ⟨v, rfl⟩).trans_eq prCoin_false) zero_le
    rw [prCoin_eq_card, ENNReal.div_eq_zero_iff] at h0
    rcases h0 with h0 | h0
    · rw [Nat.cast_eq_zero, Finset.card_eq_zero, Finset.filter_eq_empty_iff] at h0
      exact h0 (Finset.mem_univ ω) hne
    · exact ENNReal.natCast_ne_top _ h0
  choose cl hcl using hcov
  have hiff : ∀ i ω u, P i ω = bits i u ↔ cl i ω = u := fun i ω u =>
    ⟨fun h => hinj i ((hcl i ω).symm.trans h), fun h => by rw [hcl i ω, h]⟩
  have hsz : ∀ i u, Fintype.card (Q i) * (Finset.univ.filter fun ω => cl i ω = u).card = Fintype.card Ω := by
    intro i u
    have hQ0 : (Fintype.card (Q i) : ℝ≥0∞) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
    have hΩ0 : (Fintype.card Ω : ℝ≥0∞) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
    have h := prCoin_law (hlaw i) (fun r => r = bits i u)
    have h1 : (Finset.univ.filter fun v : Q i => bits i v = bits i u).card = 1 := by
      rw [Finset.card_eq_one]
      exact ⟨u, Finset.ext fun v => by simp [(hinj i).eq_iff]⟩
    have hf : (Finset.univ.filter fun ω => P i ω = bits i u) = Finset.univ.filter fun ω => cl i ω = u :=
      Finset.filter_congr fun ω _ => hiff i ω u
    rw [prCoin_eq_card, prCoin_eq_card, h1, hf, Nat.cast_one,
      ENNReal.div_eq_div_iff hQ0 (ENNReal.natCast_ne_top _) hΩ0 (ENNReal.natCast_ne_top _), mul_one] at h
    exact_mod_cast h
  have hN : ∀ ℓ, (Finset.univ.filter fun p : Ω × Ys => L p = ℓ).card =
      ∑ ω, (Finset.univ.filter fun y => L (ω, y) = ℓ).card := by
    intro ℓ
    rw [Finset.card_filter, Fintype.sum_prod_type]
    simp only [Finset.card_filter]
  have hk : ∀ i ℓ (u : Q i), (Finset.univ.filter fun p : Ω × Ys => L p = ℓ ∧ P i p.1 = bits i u).card =
      ∑ ω ∈ Finset.univ.filter (fun ω => cl i ω = u), (Finset.univ.filter fun y => L (ω, y) = ℓ).card := by
    intro i ℓ u
    rw [Finset.card_filter, Fintype.sum_prod_type, Finset.sum_filter]
    dsimp only
    refine Finset.sum_congr rfl fun ω _ => ?_
    by_cases hω : cl i ω = u
    · rw [ite_eq_left hω, Finset.card_filter]
      refine Finset.sum_congr rfl fun y _ => ?_
      by_cases hy : L (ω, y) = ℓ
      · rw [ite_eq_left hy, ite_eq_left (show L (ω, y) = ℓ ∧ P i ω = bits i u from ⟨hy, (hiff i ω u).2 hω⟩)]
      · rw [ite_eq_right hy, ite_eq_right fun h : L (ω, y) = ℓ ∧ P i ω = bits i u => hy h.1]
    · rw [ite_eq_right hω]
      exact Finset.sum_eq_zero fun y _ =>
        ite_eq_right fun h : L (ω, y) = ℓ ∧ P i ω = bits i u => hω ((hiff i ω u).1 h.2)
  have hleaf : ∀ ℓ, ((Finset.univ.filter fun p : Ω × Ys => L p = ℓ).card : ℝ≥0∞) *
      (⨆ i, classErr (Finset.univ.filter fun p : Ω × Ys => L p = ℓ).card
        fun u : Q i => (Finset.univ.filter fun p : Ω × Ys => L p = ℓ ∧ P i p.1 = bits i u).card) ≤
      ((∑ ω, Nat.dist (Fintype.card Zs * (Finset.univ.filter fun y => L (ω, y) = ℓ).card)
        ((Finset.univ.filter fun z => LI z = ℓ).card * Fintype.card Ys) : ℕ) : ℝ≥0∞) / Fintype.card Zs := by
    intro ℓ
    rw [ENNReal.mul_iSup]
    refine iSup_le fun i => mul_classErr_le Fintype.card_ne_zero ?_
    rw [hN ℓ]
    simp only [hk i ℓ]
    exact classNum_le (cl i) (hsz i) (fun ω => (Finset.univ.filter fun y => L (ω, y) = ℓ).card) _ _
  obtain ⟨t, ht⟩ : ∃ t : Finset Λ, t = Finset.univ.image L ∪ Finset.univ.image LI := ⟨_, rfl⟩
  have htL : ∀ p, L p ∈ t := fun p => by
    rw [ht]
    exact Finset.mem_union_left _ (Finset.mem_image_of_mem L (Finset.mem_univ p))
  have htI : ∀ z, LI z ∈ t := fun z => by
    rw [ht]
    exact Finset.mem_union_right _ (Finset.mem_image_of_mem LI (Finset.mem_univ z))
  have hsub : Finset.univ.image L ⊆ t := by
    rw [ht]
    exact Finset.subset_union_left
  have hZ0 : (Fintype.card Zs : ℝ≥0∞) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
  have hY0 : (Fintype.card Ys : ℝ≥0∞) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
  obtain ⟨D, hD⟩ : ∃ D : ℝ≥0∞, D = ((Fintype.card Zs * Fintype.card (Ω × Ys) : ℕ) : ℝ≥0∞) := ⟨_, rfl⟩
  have hD0 : D ≠ 0 := by
    rw [hD]
    exact_mod_cast (Nat.mul_pos Fintype.card_pos Fintype.card_pos).ne'
  have hDI : D = (Fintype.card (Ω × Zs) : ℝ≥0∞) * Fintype.card Ys := by
    rw [hD, ← Nat.cast_mul]
    congr 1
    simp only [Fintype.card_prod]
    ring
  have hterm : ∀ x : ℕ, (Fintype.card (Ω × Ys) : ℝ≥0∞)⁻¹ * ((x : ℝ≥0∞) / Fintype.card Zs) = (x : ℝ≥0∞) / D :=
    fun x => by
      rw [hD, Nat.cast_mul, div_eq_mul_inv, div_eq_mul_inv,
        ENNReal.mul_inv (Or.inl hZ0) (Or.inl (ENNReal.natCast_ne_top _))]
      ring
  have hsum : ∑ ℓ ∈ t, ∑ ω, Nat.dist (Fintype.card Zs * (Finset.univ.filter fun y => L (ω, y) = ℓ).card)
        ((Finset.univ.filter fun z => LI z = ℓ).card * Fintype.card Ys) =
      ∑ i ∈ Finset.univ ×ˢ t, Nat.dist (Fintype.card Zs * (Finset.univ.filter fun y => L (i.1, y) = i.2).card)
        ((Finset.univ.filter fun z => LI z = i.2).card * Fintype.card Ys) := by
    simp only [Finset.sum_product_right]
  have hreal : ∀ (G : Ω × Λ → Prop) [DecidablePred G],
      ((∑ i ∈ (Finset.univ ×ˢ t).filter G,
        Fintype.card Zs * (Finset.univ.filter fun y => L (i.1, y) = i.2).card : ℕ) : ℝ≥0∞) / D =
        prCoin (fun p : Ω × Ys => G (p.1, L p)) := by
    intro G _
    rw [prCoin_pair_eq L t htL G, ← Finset.mul_sum, Nat.cast_mul, hD, Nat.cast_mul,
      ENNReal.mul_div_mul_left _ _ hZ0 (ENNReal.natCast_ne_top _)]
  have hideal : ∀ (G : Ω × Λ → Prop) [DecidablePred G],
      ((∑ i ∈ (Finset.univ ×ˢ t).filter G,
        (Finset.univ.filter fun z => LI z = i.2).card * Fintype.card Ys : ℕ) : ℝ≥0∞) / D =
        prCoin (fun p : Ω × Zs => G (p.1, LI p.2)) := by
    intro G _
    rw [prCoin_pair_eq (fun p : Ω × Zs => LI p.2) t (fun p => htI p.2) G, ← Finset.sum_mul, Nat.cast_mul, hDI]
    exact ENNReal.mul_div_mul_right _ _ hY0 (ENNReal.natCast_ne_top _)
  refine (avg_comp_le L _ _ hleaf).trans ((Finset.sum_le_sum_of_subset hsub).trans ?_)
  simp only [hterm]
  rw [← ENNReal.sum_div, ← Nat.cast_sum, hsum]
  refine sum_dist_div_le _ _ _ hD0 (fun G _ => ?_) (fun G _ => ?_)
  · rw [hreal G, hideal G]
    exact hswap fun ω ℓ => G (ω, ℓ)
  · rw [hreal G, hideal G]
    exact prCoin_swap_back (fun p : Ω × Ys => G (p.1, L p)) (fun p : Ω × Zs => G (p.1, LI p.2))
      (hswap fun ω ℓ => ¬ G (ω, ℓ))

/-- **One hiding swap per hm96 leaf**: `avg_iSup_classErr_le` at the leaf `H (x ω + My y, c y)` of an hm96 salt `y`,
whose ideal leaf is `H (U, c y)` at a fresh uniform `U` (`ideal_leaf_swap`, from `Claims.Hm96Hiding`). -/
theorem avg_iSup_classErr_le_hm96 {Ω Ys B C Λ ι : Type} [Fintype Ω] [Nonempty Ω] [Fintype Ys] [Nonempty Ys]
    [AddCommGroup B] [Fintype B] [DecidableEq Λ] (My : Ys → B) (c : Ys → C) {δ₁ : ℝ≥0∞}
    (hT1 : Claims.Hm96Hiding My c δ₁) (H : B × C → Λ) (x : Ω → B) {R : ι → Type} [∀ i, DecidableEq (R i)]
    (P : (i : ι) → Ω → R i) {Q : ι → Type} [∀ i, Fintype (Q i)] [∀ i, Nonempty (Q i)] (bits : (i : ι) → Q i → R i)
    (hinj : ∀ i, Function.Injective (bits i))
    (hlaw : ∀ i (g : R i → ℝ≥0∞), avg (fun ω => g (P i ω)) = avg fun u => g (bits i u)) :
    avg (fun p₀ : Ω × Ys => ⨆ i, classErr
      (Finset.univ.filter fun p : Ω × Ys => H (x p.1 + My p.2, c p.2) = H (x p₀.1 + My p₀.2, c p₀.2)).card
      fun u : Q i => (Finset.univ.filter fun p : Ω × Ys =>
        H (x p.1 + My p.2, c p.2) = H (x p₀.1 + My p₀.2, c p₀.2) ∧ P i p.1 = bits i u).card) ≤ 2 * δ₁ := by
  have : Nonempty B := ⟨0⟩
  exact avg_iSup_classErr_le (fun p : Ω × Ys => H (x p.1 + My p.2, c p.2)) (fun z : B × Ys => H (z.1, c z.2))
    (fun G => FlockSoundness.ZK.ideal_leaf_swap My c hT1 H x G) P bits hinj hlaw

end Verity.Game

namespace Verity.Sampled

open Verity.Game
open scoped ENNReal

/-- **An opening's gap and empty class within its class error**: at an opening's node whose canonical description
is `o`, where every round of the fiber releases `bits u` for some `u`, the gap plus the empty class is at most the
class error of the fiber partitioned by `bits`. -/
theorem openGap_add_openEmpty_le {β Q : Type} [Fintype Q] [Nonempty Q] {E : List Release → Game β}
    {g' : Game β} (o : OpeningAt E g') (ho : openingOf E g' = some o) (bits : Q → List Bool)
    (hinj : Function.Injective bits) (hcov : ∀ d : o.Openable, ∃ u, TreeDraw.opened d.1 o.n = bits u) :
    openGap E g' bits + openEmpty E g' bits ≤
      classErr (Fintype.card o.Openable) fun u => (o.classOf (bits u)).card := by
  choose cls hcls using hcov
  have hS : ∀ u d, d ∈ o.classOf (bits u) ↔ cls d = u := fun u d => by
    rw [OpeningAt.classOf, Finset.mem_filter, hcls d, hinj.eq_iff]
    exact and_iff_right (Finset.mem_univ d)
  have hG : openGap E g' bits = shortfall fun u => pickIn (o.classOf (bits u)) := by
    unfold openGap
    rw [ho]
    rfl
  have hE : openEmpty E g' bits = prCoin fun u => ¬ (o.classOf (bits u)).Nonempty := by
    unfold openEmpty EmptyAt
    congr 1
    funext u
    apply propext
    constructor
    · rintro ⟨o', ho', hne⟩
      rw [ho] at ho'
      cases ho'
      exact hne
    · exact fun hne => ⟨o, ho, hne⟩
  rw [hG, hE]
  exact shortfall_pickIn_add_le (fun u => o.classOf (bits u)) cls hS

/-! ## At the deployed releases -/

local notation "𝔽" => FlockLevel3.GF128

variable (e : 𝔽 ≃ (Fin 128 → Bool))

/-- Each of a round's challenges is released as its 128 bits. -/
theorem map_length_roundBits {m : ℕ} (c : Fin m → 𝔽) :
    (List.ofFn fun i => List.ofFn (e (c i))).map List.length = List.ofFn fun _ : Fin m => 128 := by
  rw [List.map_ofFn]
  exact congrArg List.ofFn (funext fun i => List.length_ofFn)

/-- **A round's bits determine its challenges**, at every count `m`: each challenge is released as its 128 bits, in
order. -/
theorem injective_roundBits {m : ℕ} : Function.Injective (roundBits e : (Fin m → 𝔽) → List Bool) := by
  intro c c' h
  have h' := List.eq_iff_flatten_eq.2 ⟨h, (map_length_roundBits e c).trans (map_length_roundBits e c').symm⟩
  funext i
  exact e.injective (List.ofFn_injective (congrFun (List.ofFn_injective h') i))

/-- A round's bits are 128 per challenge, under every encoding. -/
theorem length_roundBits_enc {m : ℕ} (c : Fin m → 𝔽) : (roundBits e c).length = 128 * m := by
  rw [roundBits, List.length_flatten, map_length_roundBits, List.ofFn_const, List.sum_replicate_nat, Nat.mul_comm]

/-- **A link's bits determine its two rounds' challenges**, at every count `m`: its first round's bits have the same
length at every release (`length_roundBits_enc`). -/
theorem injective_linkBits {m : ℕ} : Function.Injective (linkBits e : (Fin 2 → Fin m → 𝔽) → List Bool) := by
  intro p p' h
  obtain ⟨h0, h1⟩ := List.append_inj h ((length_roundBits_enc e (p 0)).trans (length_roundBits_enc e (p' 0)).symm)
  exact funext (Fin.forall_fin_two.2 ⟨injective_roundBits e h0, injective_roundBits e h1⟩)

/-- **At a round's opening**: where the opening releases `m` coins of a block of at least `m`, its gap plus its empty
class at a round's release is at most its class error (`openGap_add_openEmpty_le`, with `injective_roundBits` and
`opened_eq_roundBits`). -/
theorem openGap_add_openEmpty_le_round {β : Type} {m : ℕ} {E : List Release → Game β} {g' : Game β}
    (o : OpeningAt E g') (ho : openingOf E g' = some o) (hn : o.n = m) (hm : m ≤ o.T.spec.block) :
    openGap E g' (roundBits e : (Fin m → 𝔽) → List Bool) + openEmpty E g' (roundBits e : (Fin m → 𝔽) → List Bool) ≤
      classErr (Fintype.card o.Openable) fun c : Fin m → 𝔽 => (o.classOf (roundBits e c)).card := by
  subst hn
  exact openGap_add_openEmpty_le o ho (roundBits e) (injective_roundBits e) fun d =>
    ⟨_, opened_eq_roundBits e hm d.1⟩

end Verity.Sampled
