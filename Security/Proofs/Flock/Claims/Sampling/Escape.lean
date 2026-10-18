import Definitions.Claims.Sampling
import Proofs.Flock.Claims.Sampling.Draws
import Proofs.Flock.Claims.Sampling.Wait

/-!
# The designs' acceptance probabilities, as their draws' probabilities

`Definitions.Claims.Sampling` states each design's acceptance probability `P_a(B)` in closed form, over slots in `ℕ`.
Here each is the `Claims.Pr` that a model draw misses `B`:

* `Pr_pi_forall`, `Pr_and_prod`, `Ex_pi_prod`: independent coordinates.
* `mem_window`, `window_subset`, `window_zero`, `window_window`: `window o n B` is `B` on `[o, o + n)`, shifted to
  `[0, n)`; `finOf N B`, `disjoint_image_val_iff`: slots below `N` as units of a draw over `Fin N`.
* `stratum_miss_eq`, `strataPa_eq_Pr`: **independent uniform `k_s`-subsets of consecutive strata** (`StrataΩ`,
  `strataDraw`) miss `B` with probability `strataPa`; `strataPa_append`, `strataPa_window`.
* `OneStage.Pa_all`, `Pa_subset`, `Pa_bernoulli`: the `all`, uniform-subset and Bernoulli draws miss `B` with
  probability `P_a(B)`; `OneStage.Pa_strata`: every design but Bernoulli's is a strata draw (`OneStage.strata`: the
  stratified and work laws by definition, `all` and `subset` one stratum), with `OneStage.strata_sum`,
  `OneStage.strata_le`.
* `nested_escape`, `Rule.Pa_nested`: **the nested escape**: a Bernoulli first stage over a class's replay units, then a
  uniform `min(k, n_v)`-subset of each drawn one's `n_v` proof units, misses a faulty proof unit of every member of `B`
  with probability `∏_{v ∈ B} (1 − p·min(k, n_v)/n_v)`, whichever faulty proof unit each first draw fixes (the interior
  the prover commits after it); `Rule.Pa_none`: checking every proof unit, the first stage's `P_a`.
* `classesPa_eq_prod`, `classesPa_eq_Pr`, `Design.Pa_eq_Pr`: **the classes' product**: `P_a` on the partition is the
  product over its classes of each rule's `P̄_a` on the class's window (`classPa`), the probability that independent
  draws of the classes all miss; `classesPa_strata`, `Design.Pa_strata`: classes drawn as strata are one strata draw.

`Live.lean` proves these distributions of the executable draw on uniform bytes, and `Select` of the deployed sampled
verifier's (`drawMisses`, and `DrawLaw` from it, `drawLaw_of_slots`).
-/

namespace Claims.Sampling

open Finset

/-! ## Independent coordinates -/

/-- **Independent coordinates of a dependent product**: `Pr[∀ i ∈ D, q_i(ω_i)] = ∏_{i ∈ D} Pr[q_i]`. -/
theorem Pr_pi_forall {ι : Type} [Fintype ι] [DecidableEq ι] {Ω : ι → Type} [∀ i, Fintype (Ω i)]
    [∀ i, Nonempty (Ω i)] (D : Finset ι) (q : ∀ i, Ω i → Prop) :
    Pr (fun ω : (i : ι) → Ω i => ∀ i ∈ D, q i (ω i)) = ∏ i ∈ D, Pr (q i) := by
  rw [Pr_eq_toReal_prCoin, FlockSoundness.Game.prCoin_pi_forall, ENNReal.toReal_prod]
  exact prod_congr rfl fun i _ => (Pr_eq_toReal_prCoin (q i)).symm

/-- Two independent coordinates: `Pr[p(a) ∧ q(b)] = Pr[p]·Pr[q]`. -/
theorem Pr_and_prod {α β : Type} [Fintype α] [Fintype β] (p : α → Prop) (q : β → Prop) :
    Pr (fun s : α × β => p s.1 ∧ q s.2) = Pr p * Pr q := by
  classical
  unfold Pr
  rw [div_mul_div_comm, Fintype.card_prod, Nat.cast_mul]
  congr 1
  rw [← Nat.cast_mul, ← card_product]
  congr 2
  ext s
  simp

/-- The expectation of a product of functions of independent coordinates is the product of their expectations. -/
theorem Ex_pi_prod {ι : Type} [Fintype ι] [DecidableEq ι] {Ω : ι → Type} [∀ i, Fintype (Ω i)]
    (g : ∀ i, Ω i → ℝ) : Ex (fun ω : (i : ι) → Ω i => ∏ i, g i (ω i)) = ∏ i, Ex (g i) := by
  unfold Ex
  rw [prod_div_distrib, Fintype.card_pi, Nat.cast_prod, prod_univ_sum, Fintype.piFinset_univ]

/-! ## Windows -/

theorem mem_window {o n u : ℕ} {B : Finset ℕ} (hu : u < n) : u ∈ window o n B ↔ o + u ∈ B := by
  simp only [window, mem_image, mem_filter]
  constructor
  · rintro ⟨v, ⟨hv, hov, -⟩, rfl⟩
    rwa [Nat.add_sub_cancel' hov]
  · intro h
    exact ⟨o + u, ⟨h, by omega, by omega⟩, by omega⟩

theorem window_subset (o n : ℕ) (B : Finset ℕ) : window o n B ⊆ range n := by
  intro u hu
  simp only [window, mem_image, mem_filter] at hu
  obtain ⟨v, ⟨-, h1, h2⟩, rfl⟩ := hu
  exact mem_range.2 (by omega)

theorem window_zero {n : ℕ} {B : Finset ℕ} (hB : B ⊆ range n) : window 0 n B = B := by
  ext u
  constructor
  · intro h
    rw [mem_window (mem_range.1 (window_subset 0 n B h)), zero_add] at h
    exact h
  · intro h
    rw [mem_window (mem_range.1 (hB h)), zero_add]
    exact h

theorem window_window {o o' n n' : ℕ} (h : o' + n' ≤ n) (B : Finset ℕ) :
    window o' n' (window o n B) = window (o + o') n' B := by
  ext u
  by_cases hu : u < n'
  · rw [mem_window hu, mem_window hu, mem_window (by omega), add_assoc]
  · exact iff_of_false (fun h => hu (mem_range.1 (window_subset _ _ _ h)))
      fun h => hu (mem_range.1 (window_subset _ _ _ h))

/-- The slots of `B`, as units of a draw over `Fin N`. -/
def finOf (N : ℕ) (B : Finset ℕ) : Finset (Fin N) := univ.filter fun u => u.val ∈ B

theorem map_finOf {N : ℕ} {B : Finset ℕ} (hB : B ⊆ range N) : (finOf N B).map Fin.valEmbedding = B := by
  ext a
  simp only [finOf, mem_map, mem_filter, mem_univ, true_and, Fin.valEmbedding_apply]
  constructor
  · rintro ⟨u, hu, rfl⟩
    exact hu
  · intro ha
    exact ⟨⟨a, mem_range.1 (hB ha)⟩, ha, rfl⟩

theorem card_finOf {N : ℕ} {B : Finset ℕ} (hB : B ⊆ range N) : (finOf N B).card = B.card := by
  rw [← card_map Fin.valEmbedding, map_finOf hB]

/-- A set of units of a stratum from offset `o`, as slots, misses `B` exactly when it misses the stratum's window. -/
theorem disjoint_shift_iff {n : ℕ} (o : ℕ) (S : Finset (Fin n)) (B : Finset ℕ) :
    Disjoint (S.image fun u => o + u.val) B ↔ Disjoint S (finOf n (window o n B)) := by
  simp only [disjoint_left, mem_image, finOf, mem_filter, mem_univ, true_and]
  constructor
  · intro h u hu hw
    exact h ⟨u, hu, rfl⟩ ((mem_window u.isLt).1 hw)
  · rintro h _ ⟨u, hu, rfl⟩ hB
    exact h hu ((mem_window u.isLt).2 hB)

theorem disjoint_image_val_iff {N : ℕ} (S : Finset (Fin N)) {B : Finset ℕ} (hB : B ⊆ range N) :
    Disjoint (S.image Fin.val) B ↔ Disjoint S (finOf N B) := by
  rw [← window_zero hB, ← disjoint_shift_iff 0 S B, window_zero hB]
  simp only [zero_add]

/-! ## Strata draws -/

/-- **A stratum's uniform `k`-subset misses `B`**: a uniform `k`-subset of `n` slots from `o` misses `B` with
probability `C(n − |B ∩ [o, o + n)|, k)/C(n, k)`. -/
theorem stratum_miss_eq (n k o : ℕ) (B : Finset ℕ) :
    Pr (fun S : {S : Finset (Fin n) // S.card = k} => Disjoint (S.1.image fun u => o + u.val) B) =
      ((n - (window o n B).card).choose k : ℝ) / (n.choose k : ℝ) := by
  rcases le_or_gt k n with hk | hk
  · rw [Pr_congr (q := fun S : {S : Finset (Fin n) // S.card = k} => Disjoint S.1 (finOf n (window o n B)))
      fun S => disjoint_shift_iff o S.1 B, subset_miss_eq hk, card_finOf (window_subset o n B)]
  · have hE : IsEmpty {S : Finset (Fin n) // S.card = k} :=
      ⟨fun S => by have := S.1.card_le_univ; rw [Fintype.card_fin, S.2] at this; omega⟩
    rw [Nat.choose_eq_zero_of_lt hk, Nat.cast_zero, div_zero]
    unfold Pr
    rw [Fintype.card_eq_zero, Nat.cast_zero, div_zero]

/-- The coins of independent uniform subsets of consecutive strata `(n_s, k_s)`: a `k_s`-subset of each. -/
def StrataΩ : List (ℕ × ℕ) → Type
  | [] => Unit
  | (n, k) :: ss => {S : Finset (Fin n) // S.card = k} × StrataΩ ss

instance instFintypeStrataΩ : (ss : List (ℕ × ℕ)) → Fintype (StrataΩ ss)
  | [] => inferInstanceAs (Fintype Unit)
  | (n, k) :: ss =>
    have := instFintypeStrataΩ ss
    inferInstanceAs (Fintype ({S : Finset (Fin n) // S.card = k} × StrataΩ ss))

/-- The slots the strata's subsets draw, stratum `s` from offset `o + n_0 + ⋯ + n_{s−1}`. -/
def strataDraw : (ss : List (ℕ × ℕ)) → ℕ → StrataΩ ss → Finset ℕ
  | [], _, _ => ∅
  | (n, k) :: ss, o, ω =>
    (ω : {S : Finset (Fin n) // S.card = k} × StrataΩ ss).1.1.image (fun u => o + u.val) ∪ strataDraw ss (o + n) ω.2

/-- **`strataPa` is the strata draw's miss probability**: independent uniform `k_s`-subsets of consecutive strata of
`n_s` slots, from `o`, miss `B` with probability `strataPa ss o B`. -/
theorem strataPa_eq_Pr : ∀ (ss : List (ℕ × ℕ)) (o : ℕ) (B : Finset ℕ),
    strataPa ss o B = Pr (fun ω : StrataΩ ss => Disjoint (strataDraw ss o ω) B)
  | [], o, B => by
    rw [strataPa]
    unfold Pr
    simp only [strataDraw, disjoint_empty_left]
    rw [filter_true_of_mem fun _ _ => trivial, card_univ,
      div_self (Nat.cast_ne_zero.2 (Fintype.card_ne_zero (α := Unit)))]
  | (n, k) :: ss, o, B => by
    rw [strataPa, strataPa_eq_Pr ss (o + n) B, ← stratum_miss_eq, ← Pr_and_prod]
    exact Pr_congr fun ω => by simp only [strataDraw, disjoint_union_left]

theorem strataPa_append : ∀ (ss ss' : List (ℕ × ℕ)) (o : ℕ) (B : Finset ℕ),
    strataPa (ss ++ ss') o B = strataPa ss o B * strataPa ss' (o + (ss.map Prod.fst).sum) B
  | [], ss', o, B => by simp [strataPa]
  | (n, k) :: ss, ss', o, B => by
    rw [List.cons_append, strataPa, strataPa, strataPa_append ss ss' (o + n) B, List.map_cons, List.sum_cons,
      add_assoc, mul_assoc]

/-- A strata draw inside a window reads `B` shifted by the window's offset. -/
theorem strataPa_window : ∀ (ss : List (ℕ × ℕ)) {o o' n : ℕ} (B : Finset ℕ), o' + (ss.map Prod.fst).sum ≤ n →
    strataPa ss o' (window o n B) = strataPa ss (o + o') B
  | [], _, _, _, _, _ => rfl
  | (m, k) :: ss, o, o', n, B, h => by
    simp only [List.map_cons, List.sum_cons] at h
    rw [strataPa, strataPa, window_window (by omega), strataPa_window ss B (by omega), add_assoc]

/-! ## One-stage designs -/

/-- **The `all` draw misses `B`** (it draws every slot below `N = Σ ns`) with probability `P_a(B)`: one exactly when
`B` is empty. -/
theorem OneStage.Pa_all (ns : List ℕ) {B : Finset ℕ} (hB : B ⊆ range ns.sum) :
    OneStage.all.Pa ns B = Pr (fun _ : Unit => Disjoint (range ns.sum) B) := by
  have hiff : Disjoint (range ns.sum) B ↔ B = ∅ :=
    ⟨fun h => disjoint_self_iff_empty _ |>.1 (h.mono_left hB), fun h => h ▸ disjoint_empty_right _⟩
  rw [OneStage.Pa]
  unfold Pr
  by_cases hB0 : B = ∅ <;> simp [hB0, hiff]

/-- **The uniform `K`-subset misses `B`** with probability `P_a(B) = C(N − |B|, K)/C(N, K)`. -/
theorem OneStage.Pa_subset (ns : List ℕ) {K : ℕ} (hK : K ≤ ns.sum) {B : Finset ℕ} (hB : B ⊆ range ns.sum) :
    (OneStage.subset K).Pa ns B =
      Pr (fun S : {S : Finset (Fin ns.sum) // S.card = K} => Disjoint (S.1.image Fin.val) B) := by
  rw [Pr_congr (q := fun S : {S : Finset (Fin ns.sum) // S.card = K} => Disjoint S.1 (finOf ns.sum B))
    fun S => disjoint_image_val_iff S.1 hB, subset_miss_eq hK, card_finOf hB]
  rfl

/-- **The Bernoulli draw misses `B`** (slot `u` drawn when its coin in `[0, den)` is below `num`) with probability
`P_a(B) = ((den − num)/den)^|B|`. -/
theorem OneStage.Pa_bernoulli (ns : List ℕ) {num den : ℕ} [NeZero den] (h : num ≤ den) {B : Finset ℕ}
    (hB : B ⊆ range ns.sum) :
    (OneStage.bernoulli num den).Pa ns B =
      Pr (fun ω : Fin ns.sum → Fin den => Disjoint ((univ.filter fun u => (ω u).val < num).image Fin.val) B) := by
  rw [Pr_congr (q := fun ω : Fin ns.sum → Fin den =>
      Disjoint (univ.filter fun u => (ω u).val < num) (finOf ns.sum B)) fun ω => disjoint_image_val_iff _ hB,
    bernoulli_miss_eq h, card_finOf hB]
  rfl

/-- The strata `(n_s, k_s)` a one-stage design draws uniform subsets of, all but Bernoulli's: the stratified and work
laws' own, and one stratum of every slot for `all` (drawing all of it) and `subset`. -/
def OneStage.strata (ns : List ℕ) : OneStage → Option (List (ℕ × ℕ))
  | .all => some [(ns.sum, ns.sum)]
  | .subset K => some [(ns.sum, K)]
  | .bernoulli _ _ => none
  | .stratified K => some (ns.map fun n => (n, max 1 ((2 * K * n + ns.sum) / (2 * ns.sum))))
  | .work K w f =>
    let W := ((ns.zip w).map fun p => p.1 * p.2).sum
    some ((ns.zip (w.zip f)).map fun p => (p.1, min p.1 (max p.2.2 ((K * p.2.1 * p.1 + W - 1) / W))))

/-- **Every design but Bernoulli's is a strata draw**: its `P_a` is `strataPa` of its strata. -/
theorem OneStage.Pa_strata {ns : List ℕ} {d : OneStage} {ss : List (ℕ × ℕ)} (h : d.strata ns = some ss)
    {B : Finset ℕ} (hB : B ⊆ range ns.sum) : d.Pa ns B = strataPa ss 0 B := by
  cases d with
  | all =>
    simp only [OneStage.strata, Option.some.injEq] at h
    subst h
    rw [strataPa, strataPa, window_zero hB, Nat.choose_self, Nat.cast_one, div_one, mul_one, OneStage.Pa]
    by_cases hB0 : B = ∅
    · simp [hB0]
    · have hpos : 0 < B.card := card_pos.2 (nonempty_iff_ne_empty.2 hB0)
      have hle := card_le_card hB
      rw [card_range] at hle
      simp only [hB0, ite_false]
      rw [Nat.choose_eq_zero_of_lt (by omega : ns.sum - B.card < ns.sum), Nat.cast_zero]
  | subset K =>
    simp only [OneStage.strata, Option.some.injEq] at h
    subst h
    rw [strataPa, strataPa, window_zero hB, mul_one]
    rfl
  | bernoulli num den => exact absurd h (by simp [OneStage.strata])
  | stratified K =>
    simp only [OneStage.strata, Option.some.injEq] at h
    subst h
    rfl
  | work K w f =>
    simp only [OneStage.strata, Option.some.injEq] at h
    subst h
    rfl

/-- A well-formed design's strata cover its slots. -/
theorem OneStage.strata_sum {ns : List ℕ} {d : OneStage} {ss : List (ℕ × ℕ)} (h : d.strata ns = some ss)
    (hw : d.WellFormed ns) : (ss.map Prod.fst).sum = ns.sum := by
  cases d with
  | all =>
    simp only [OneStage.strata, Option.some.injEq] at h
    subst h
    simp
  | subset K =>
    simp only [OneStage.strata, Option.some.injEq] at h
    subst h
    simp
  | bernoulli num den => exact absurd h (by simp [OneStage.strata])
  | stratified K =>
    simp only [OneStage.strata, Option.some.injEq] at h
    subst h
    simp [Function.comp_def]
  | work K w f =>
    simp only [OneStage.strata, Option.some.injEq] at h
    subst h
    obtain ⟨hw, hf, -⟩ := hw
    rw [List.map_map]
    have : (Prod.fst ∘ fun p : ℕ × ℕ × ℕ => (p.1, min p.1 (max p.2.2
        ((K * p.2.1 * p.1 + ((ns.zip w).map fun p => p.1 * p.2).sum - 1) /
          ((ns.zip w).map fun p => p.1 * p.2).sum)))) = Prod.fst := rfl
    rw [this, List.map_fst_zip (by simp [hw, hf])]

/-- A well-formed design's strata each draw at most their size. -/
theorem OneStage.strata_le {ns : List ℕ} {d : OneStage} {ss : List (ℕ × ℕ)} (h : d.strata ns = some ss)
    (hw : d.WellFormed ns) : ∀ p ∈ ss, p.2 ≤ p.1 := by
  cases d with
  | all =>
    simp only [OneStage.strata, Option.some.injEq] at h
    subst h
    simp
  | subset K =>
    simp only [OneStage.strata, Option.some.injEq] at h
    subst h
    simpa using hw.2
  | bernoulli num den => exact absurd h (by simp [OneStage.strata])
  | stratified K =>
    simp only [OneStage.strata, Option.some.injEq] at h
    subst h
    obtain ⟨-, hK, hn⟩ := hw
    intro p hp
    obtain ⟨n, hnm, rfl⟩ := List.mem_map.1 hp
    have h1 := hn n hnm
    have hN : n ≤ ns.sum := List.le_sum_of_mem hnm
    refine max_le h1 (Nat.lt_succ_iff.1 ((Nat.div_lt_iff_lt_mul (by omega)).2 ?_))
    nlinarith
  | work K w f =>
    simp only [OneStage.strata, Option.some.injEq] at h
    subst h
    intro p hp
    obtain ⟨q, -, rfl⟩ := List.mem_map.1 hp
    exact min_le_left _ _

/-! ## The nested escape -/

/-- A uniform `m`-subset of `n` slots holds a given slot with probability `m/n`: it misses it with `1 − m/n`. -/
theorem subset_miss_one {n m : ℕ} (hm : m ≤ n) (u : Fin n) :
    Pr (fun S : {S : Finset (Fin n) // S.card = m} => u ∉ S.1) = 1 - (m : ℝ) / n := by
  have h := subset_miss_eq hm {u}
  rw [card_singleton] at h
  rw [Pr_congr fun S => disjoint_singleton_right.symm, h]
  obtain ⟨n', rfl⟩ : ∃ n', n = n' + 1 := ⟨n - 1, by have := u.isLt; omega⟩
  have hc := Nat.choose_mul_succ_eq n' m
  have hpos : (0 : ℝ) < ((n' + 1).choose m : ℕ) := by exact_mod_cast Nat.choose_pos hm
  have hc' : ((n'.choose m : ℕ) : ℝ) * ((n' : ℝ) + 1) = ((n' + 1).choose m : ℕ) * ((n' : ℝ) + 1 - m) := by
    have := congrArg (Nat.cast (R := ℝ)) hc
    push_cast [Nat.cast_sub hm] at this
    linarith
  rw [Nat.add_sub_cancel, div_eq_iff hpos.ne']
  push_cast
  field_simp
  linarith

/-- The coins of the nested draw's second stage: a uniform `min(k, n_v)`-subset of each replay unit's `n_v` proof
units. -/
instance instNonemptyMinSubset (n k : ℕ) : Nonempty {S : Finset (Fin n) // S.card = min k n} := by
  obtain ⟨t, -, ht⟩ := exists_subset_card_eq (s := (univ : Finset (Fin n))) (n := min k n)
    (by rw [card_univ, Fintype.card_fin]; exact min_le_right k n)
  exact ⟨⟨t, ht⟩⟩

/-- **The nested escape**, the interior fixed by the shared first stage: a Bernoulli first stage at rate `num/den` over
`m` replay units, then a uniform `min(k, n_v)`-subset of each replay unit's `n_v` proof units, independently. Whichever
proof unit `f ω₁ v` of each replay unit the first draw `ω₁` fixes as faulty (the interior the prover commits after
it), the checks miss it at every member of `B` with probability `∏_{v ∈ B} (1 − (num/den)·min(k, n_v)/n_v)`. -/
theorem nested_escape {m num den k : ℕ} [NeZero den] (h : num ≤ den) (sz : Fin m → ℕ)
    (f : (Fin m → Fin den) → (v : Fin m) → Fin (sz v)) (B : Finset (Fin m)) :
    Pr (fun p : (Fin m → Fin den) × ((v : Fin m) → {S : Finset (Fin (sz v)) // S.card = min k (sz v)}) =>
      ∀ v ∈ B, ¬ ((p.1 v).val < num ∧ f p.1 v ∈ (p.2 v).1)) =
      ∏ v ∈ B, (1 - (num : ℝ) / den * (min k (sz v) : ℕ) / (sz v : ℝ)) := by
  classical
  set c : Fin m → ℝ := fun v => ((min k (sz v) : ℕ) : ℝ) / (sz v : ℝ) with hc
  have hfix : ∀ a : Fin m → Fin den,
      Pr (fun S : (v : Fin m) → {S : Finset (Fin (sz v)) // S.card = min k (sz v)} =>
        ∀ v ∈ B, ¬ ((a v).val < num ∧ f a v ∈ (S v).1)) =
      ∏ v, (if v ∈ B then 1 - (if (a v).val < num then (1 : ℝ) else 0) * c v else 1) := by
    intro a
    refine (Pr_pi_forall (Ω := fun v => {S : Finset (Fin (sz v)) // S.card = min k (sz v)}) B
      (fun v S => ¬ ((a v).val < num ∧ f a v ∈ S.1))).trans ?_
    rw [prod_ite_mem, univ_inter]
    refine prod_congr rfl fun v _ => ?_
    by_cases hv : (a v).val < num
    · simp only [hv, true_and, ite_true, one_mul]
      exact subset_miss_one (min_le_right k (sz v)) (f a v)
    · simp only [hv, false_and, not_false_eq_true, ite_false, zero_mul, sub_zero]
      exact Pr_true
  rw [Pr_prod_eq_Ex]
  refine (congrArg Ex (funext hfix)).trans ?_
  rw [Ex_pi_prod (Ω := fun _ => Fin den) fun v x => if v ∈ B then 1 - (if x.val < num then (1 : ℝ) else 0) * c v
    else 1, ← prod_subset (subset_univ B) fun v _ hv => by simp only [hv, ite_false]; exact Ex_const 1]
  refine prod_congr rfl fun v hv => ?_
  have hden : (0 : ℝ) < den := by exact_mod_cast Nat.pos_of_ne_zero (NeZero.ne den)
  unfold Ex
  simp only [hv, ite_true, sum_sub_distrib, sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul, mul_one,
    ← sum_mul, sum_boole]
  rw [Fin.card_filter_val_lt, min_eq_right h, hc]
  field_simp

/-- **The two-stage rule's `P̄_a` is the nested escape**: under a Bernoulli first stage checking `k` proof units of
each drawn replay unit, `P̄_a(B)` is the probability that the checks miss a faulty proof unit `f ω₁ v` of every member
of `B`, whichever the first draw `ω₁` fixes. -/
theorem Rule.Pa_nested {num den k : ℕ} [NeZero den] (h : num ≤ den) (ns sz : List ℕ) {B : Finset ℕ}
    (hB : B ⊆ range sz.length) (f : (Fin sz.length → Fin den) → (v : Fin sz.length) → Fin (sz.getD v 0)) :
    (⟨.bernoulli num den, some k⟩ : Rule).Pa ns sz B =
      Pr (fun p : (Fin sz.length → Fin den) ×
          ((v : Fin sz.length) → {S : Finset (Fin (sz.getD v 0)) // S.card = min k (sz.getD v 0)}) =>
        ∀ v ∈ finOf sz.length B, ¬ ((p.1 v).val < num ∧ f p.1 v ∈ (p.2 v).1)) := by
  rw [nested_escape h (fun v => sz.getD v 0) f]
  show (∏ v ∈ B, _) = _
  conv_lhs => rw [← map_finOf hB]
  rw [prod_map]
  rfl

/-- **Checking every proof unit, a rule's `P̄_a` is its first stage's `P_a`.** -/
theorem Rule.Pa_none {r : Rule} (h : r.check = none) (ns sz : List ℕ) (B : Finset ℕ) :
    r.Pa ns sz B = r.first.Pa ns B := by
  obtain ⟨first, check⟩ := r
  simp only at h
  subst h
  cases first <;> rfl

/-! ## The classes' product -/

/-- The slot at which class `i` starts: the replay units of the classes before it. -/
def classOff (cs : List (List (List RU))) (i : ℕ) : ℕ := ((cs.take i).map fun c => c.flatten.length).sum

/-- Class `i`'s factor: its rule's `P̄_a` on the members of `B` in its window. -/
noncomputable def classPa (rs : List Rule) (cs : List (List (List RU))) (o : ℕ) (B : Finset ℕ) (i : ℕ) : ℝ :=
  (rs.getD i ⟨.all, none⟩).Pa (strataSizes (cs.getD i [])) (ruSizes (cs.getD i []))
    (window (o + classOff cs i) (cs.getD i []).flatten.length B)

/-- **The classes' product**: `classesPa` multiplies each class's factor on its own window. -/
theorem classesPa_eq_prod : ∀ (rs : List Rule) (cs : List (List (List RU))) (o : ℕ) (B : Finset ℕ),
    rs.length = cs.length → classesPa rs cs o B = ∏ i ∈ range rs.length, classPa rs cs o B i
  | [], [], _, _, _ => by simp [classesPa]
  | [], _ :: _, _, _, h => absurd h (by simp)
  | _ :: _, [], _, _, h => absurd h (by simp)
  | r :: rs, c :: cs, o, B, h => by
    rw [classesPa, classesPa_eq_prod rs cs (o + c.flatten.length) B (by simpa using h), List.length_cons,
      prod_range_succ', mul_comm]
    congr 1
    refine prod_congr rfl fun i _ => ?_
    simp [classPa, classOff, add_assoc]

/-- **The classes drawn independently**: when each class's factor is the probability that its own draw misses, the
classes' product is the probability that independent draws of every class all miss. -/
theorem classesPa_eq_Pr (rs : List Rule) (cs : List (List (List RU))) (o : ℕ) (B : Finset ℕ)
    (hl : rs.length = cs.length) {Ω : Fin rs.length → Type} [∀ i, Fintype (Ω i)] [∀ i, Nonempty (Ω i)]
    (E : ∀ i, Ω i → Prop) (hE : ∀ i : Fin rs.length, classPa rs cs o B i = Pr (E i)) :
    classesPa rs cs o B = Pr (fun ω : (i : Fin rs.length) → Ω i => ∀ i, E i (ω i)) := by
  classical
  rw [classesPa_eq_prod rs cs o B hl, ← Fin.prod_univ_eq_prod_range,
    Pr_congr (q := fun ω : (i : Fin rs.length) → Ω i => ∀ i ∈ univ, E i (ω i)) (by simp)]
  exact (prod_congr rfl fun i _ => hE i).trans (Pr_pi_forall univ E).symm

/-- **The design's `P_a` on the partition is the probability that the classes' independent draws all miss.** -/
theorem Design.Pa_eq_Pr (d : Design) (P : Partition) (B : Finset ℕ) (hl : d.rules.length = P.classes.length)
    {Ω : Fin d.rules.length → Type} [∀ i, Fintype (Ω i)] [∀ i, Nonempty (Ω i)] (E : ∀ i, Ω i → Prop)
    (hE : ∀ i : Fin d.rules.length, classPa d.rules P.classes 0 B i = Pr (E i)) :
    d.Pa P B = Pr (fun ω : (i : Fin d.rules.length) → Ω i => ∀ i, E i (ω i)) :=
  classesPa_eq_Pr d.rules P.classes 0 B hl E hE

/-- **Classes drawn as strata are one strata draw**: when each class's `P̄_a` is a strata draw's over its slots, the
classes' product is the strata draw of all their strata, in order. -/
theorem classesPa_strata : ∀ (rs : List Rule) (cs : List (List (List RU))) (sss : List (List (ℕ × ℕ))) (o : ℕ)
    (B : Finset ℕ), rs.length = cs.length → sss.length = cs.length →
    (∀ p ∈ (rs.zip cs).zip sss, (p.2.map Prod.fst).sum = p.1.2.flatten.length ∧
      ∀ B' ⊆ range p.1.2.flatten.length, p.1.1.Pa (strataSizes p.1.2) (ruSizes p.1.2) B' = strataPa p.2 0 B') →
    classesPa rs cs o B = strataPa sss.flatten o B
  | [], [], [], _, _, _, _, _ => rfl
  | [], [], _ :: _, _, _, _, h, _ => absurd h (by simp)
  | [], _ :: _, _, _, _, h, _, _ => absurd h (by simp)
  | _ :: _, [], _, _, _, h, _, _ => absurd h (by simp)
  | _ :: _, _ :: _, [], _, _, _, h, _ => absurd h (by simp)
  | r :: rs, c :: cs, ss :: sss, o, B, hl, hs, h => by
    obtain ⟨hsum, hPa⟩ := h ((r, c), ss) (by simp)
    dsimp only at hsum hPa
    rw [classesPa, hPa _ (window_subset _ _ _), strataPa_window ss B (by omega), add_zero, List.flatten_cons,
      strataPa_append, hsum, classesPa_strata rs cs sss (o + c.flatten.length) B (by simpa using hl)
        (by simpa using hs) fun p hp => h p (by simp [hp])]

/-- **A design whose classes are drawn as strata is one strata draw over the partition's replay units.** -/
theorem Design.Pa_strata (d : Design) (P : Partition) (sss : List (List (ℕ × ℕ)))
    (hl : d.rules.length = P.classes.length) (hs : sss.length = P.classes.length)
    (h : ∀ p ∈ (d.rules.zip P.classes).zip sss, (p.2.map Prod.fst).sum = p.1.2.flatten.length ∧
      ∀ B' ⊆ range p.1.2.flatten.length, p.1.1.Pa (strataSizes p.1.2) (ruSizes p.1.2) B' = strataPa p.2 0 B')
    (B : Finset ℕ) : d.Pa P B = strataPa sss.flatten 0 B :=
  classesPa_strata d.rules P.classes sss 0 B hl hs h

end Claims.Sampling
