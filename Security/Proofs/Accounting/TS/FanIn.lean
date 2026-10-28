import Proofs.Accounting.TS.Price

/-!
# The fan-in count, and FOLD-OPT and FILL-OPT from it

* `FanInCount`: every input an output sees feeds it (`feeds_of_sees`: two runs that differ in one input agree on every
  vertex it doesn't feed, `eq_of_not_feeds`, and every input has a run, `runs_eval`). The inputs a set `S` of vertices
  is fed by number at most `|S| + Σ_{i < m} (|src i| − 1)` when `S`'s nodes are below `m` (`card_fed_le`, by induction
  on `m`: replace node `m − 1` by its sources).
* `SeesOfInjective`: a run's outputs are `f` of its inputs, and `f` moves with each input.
* `ThinFoldInjective`: each step is a bijection of the state and of each word (`thinStep_inj_state`,
  `thinStep_inj_left`, `thinStep_inj_right`), and a prefix only moves the state (`thinFold_append`).
* `FoldOptOfFanIn` and `FillOptOfFanIn`: `|ι| ≤ |O| + Σ_k m_k n_k` (`card_le_issued`), and the menu's fan-in routes
  make `m_k·s ≤ π·a_k` for `s = φ*` (`χ*/N_x`), so `(|ι| − |O|)·s ≤ Σ_k n_k π·a_k = π·n` (`price_of_fanIn`).
-/

namespace Accounting.SecurityProofs.TS

open Accounting.TS Accounting.Assumptions.TS Finset

namespace FanIn

variable {ι W : Type} (D : Dataflow ι W)

theorem eq_of_feeds_inl {j j' : ι} (h : D.Feeds j (.inl j')) : j' = j := by
  cases h
  rfl

theorem feeds_inr {j : ι} {i : Fin D.n} (h : D.Feeds j (.inr i)) : ∃ s ∈ D.src i, D.Feeds j s := by
  cases h with
  | read hs hf => exact ⟨_, hs, hf⟩

/-- Two runs whose inputs differ only at `j` agree on every vertex `j` doesn't feed. -/
theorem eq_of_not_feeds {j : ι} {x x' : ι → W} {v v' : ι ⊕ Fin D.n → W} (hv : D.Runs x v) (hv' : D.Runs x' v')
    (hx : ∀ j', j' ≠ j → x j' = x' j') : ∀ s, ¬ D.Feeds j s → v s = v' s := by
  have inp : ∀ j', ¬ D.Feeds j (.inl j') → v (.inl j') = v' (.inl j') := fun j' h => by
    rw [hv.1, hv'.1]
    exact hx j' fun e => h (e ▸ Dataflow.Feeds.input)
  have node : ∀ k : ℕ, ∀ i : Fin D.n, i.val = k → ¬ D.Feeds j (.inr i) → v (.inr i) = v' (.inr i) := by
    intro k
    induction k using Nat.strong_induction_on with
    | _ k ih =>
      intro i hik hf
      rw [hv.2 i, hv'.2 i]
      refine D.reads i v v' fun s hs => ?_
      have hfs : ¬ D.Feeds j s := fun h => hf (Dataflow.Feeds.read hs h)
      rcases s with j' | k'
      · exact inp j' hfs
      · exact ih k'.val (hik ▸ Fin.lt_def.1 (D.earlier i k' hs)) k' rfl hfs
  intro s hs
  rcases s with j' | i
  · exact inp j' hs
  · exact node _ i rfl hs

/-- The run of inputs `x`, node by node. -/
noncomputable def eval [Inhabited W] (x : ι → W) : ι ⊕ Fin D.n → W :=
  Sum.elim x <| (wellFounded_lt (α := Fin D.n)).fix (C := fun _ => W) fun i rec =>
    D.op i (Sum.elim x fun k => if h : k < i then rec k h else default)

theorem runs_eval [Inhabited W] (x : ι → W) : D.Runs x (eval D x) := by
  refine ⟨fun _ => rfl, fun i => ?_⟩
  show (wellFounded_lt (α := Fin D.n)).fix (C := fun _ => W) _ i = _
  rw [WellFounded.fix_eq]
  refine D.reads i _ _ fun s hs => ?_
  rcases s with j | k
  · rfl
  · simp [eval, D.earlier i k hs]

/-- An input the outputs see feeds one of them. -/
theorem feeds_of_sees [Nontrivial W] [Inhabited W] {O : Finset (ι ⊕ Fin D.n)} (hO : D.Sees O) (j : ι) :
    ∃ o ∈ O, D.Feeds j o := by
  classical
  obtain ⟨a, b, hab⟩ := exists_pair_ne W
  let x : ι → W := fun _ => a
  obtain ⟨o, ho, hne⟩ := hO j x (Function.update x j b) (eval D x) (eval D (Function.update x j b))
    (runs_eval D x) (runs_eval D _) (fun j' hj' => (Function.update_of_ne hj' b x).symm)
    (by simpa [x] using hab)
  by_contra hno
  exact hne (eq_of_not_feeds D (runs_eval D x) (runs_eval D _)
    (fun j' hj' => (Function.update_of_ne hj' b x).symm) o fun hf => hno ⟨o, ho, hf⟩)

/-- The inputs feeding `S`, whose nodes are below `m`, are at most `|S| + Σ_{i < m} (|src i| − 1)`. -/
theorem card_fed_le (m : ℕ) : ∀ (S : Finset (ι ⊕ Fin D.n)), (∀ i, .inr i ∈ S → i.val < m) →
    ∀ T : Finset ι, (∀ j ∈ T, ∃ s ∈ S, D.Feeds j s) →
      T.card ≤ S.card + ∑ i ∈ univ.filter (fun i : Fin D.n => i.val < m), D.merges i := by
  classical
  induction m with
  | zero =>
    intro S hS T hT
    have hsub : T.map ⟨Sum.inl, Sum.inl_injective⟩ ⊆ S := by
      intro y hy
      obtain ⟨j, hj, rfl⟩ := mem_map.1 hy
      obtain ⟨s, hs, hf⟩ := hT j hj
      rcases s with j' | i
      · rw [eq_of_feeds_inl D hf] at hs
        exact hs
      · exact absurd (hS i hs) (Nat.not_lt_zero _)
    have := card_le_card hsub
    rw [card_map] at this
    omega
  | succ m ih =>
    intro S hS T hT
    by_cases hm : m < D.n
    · obtain ⟨a, ha⟩ : ∃ a : Fin D.n, a.val = m := ⟨⟨m, hm⟩, rfl⟩
      have hsplit : univ.filter (fun i : Fin D.n => i.val < m + 1) =
          insert a (univ.filter fun i : Fin D.n => i.val < m) := by
        ext i
        simp only [mem_filter, mem_univ, true_and, mem_insert, Fin.ext_iff, ha]
        omega
      have hna : a ∉ univ.filter fun i : Fin D.n => i.val < m := by simp [ha]
      rw [hsplit, sum_insert hna]
      by_cases haS : Sum.inr a ∈ S
      · have hS' : ∀ i, .inr i ∈ (S.erase (.inr a) ∪ D.src a) → i.val < m := by
          intro i hi
          rcases mem_union.1 hi with h | h
          · obtain ⟨hne, hmem⟩ := mem_erase.1 h
            have h1 := hS i hmem
            have h2 : i.val ≠ m := fun e => hne (by rw [show i = a from Fin.ext (e.trans ha.symm)])
            omega
          · have := Fin.lt_def.1 (D.earlier a i h)
            omega
        have hT' : ∀ j ∈ T, ∃ s ∈ S.erase (.inr a) ∪ D.src a, D.Feeds j s := by
          intro j hj
          obtain ⟨s, hs, hf⟩ := hT j hj
          by_cases e : s = .inr a
          · rw [e] at hf
            obtain ⟨s', hs', hf'⟩ := feeds_inr D hf
            exact ⟨s', mem_union_right _ hs', hf'⟩
          · exact ⟨s, mem_union_left _ (mem_erase.2 ⟨e, hs⟩), hf⟩
        have h1 := ih _ hS' T hT'
        have h2 := card_union_le (S.erase (.inr a)) (D.src a)
        have h3 := card_erase_of_mem haS
        have h4 := card_pos.2 ⟨_, haS⟩
        have h5 : D.merges a = (D.src a).card - 1 := rfl
        omega
      · have hS' : ∀ i, .inr i ∈ S → i.val < m := by
          intro i hi
          have h1 := hS i hi
          have h2 : i.val ≠ m := fun e => haS (by rw [show a = i from Fin.ext (ha.trans e.symm)]; exact hi)
          omega
        have := ih S hS' T hT
        omega
    · have heq : univ.filter (fun i : Fin D.n => i.val < m + 1) = univ.filter fun i : Fin D.n => i.val < m := by
        ext i
        simp only [mem_filter, mem_univ, true_and]
        have := i.isLt
        omega
      rw [heq]
      exact ih S (fun i hi => by have := i.isLt; have := hS i hi; omega) T hT

end FanIn

theorem FanInCount : Accounting.Guarantees.TS.FanInCount := by
  intro ι W _ _ _ D O hO
  classical
  have h := FanIn.card_fed_le D D.n O (fun i _ => i.isLt) univ fun j _ => FanIn.feeds_of_sees D hO j
  rwa [filter_true_of_mem fun i _ => i.isLt, card_univ] at h

theorem SeesOfInjective : Accounting.Guarantees.TS.SeesOfInjective := by
  intro ι W D O f hf hinj j x x' v v' hv hv' hx hne
  by_contra hno
  exact hinj j x x' hx hne (funext fun o => by
    rw [← hf x v hv, ← hf x' v' hv']
    exact not_not.1 fun h => hno ⟨o, o.2, h⟩)

theorem thinStep_inj_state {xor : Bool} {s s' w w' : BitVec 32} (h : thinStep xor s w w' = thinStep xor s' w w') :
    s = s' := by
  cases xor <;> simp_all [thinStep]

theorem thinStep_inj_left {xor : Bool} {s w₁ w₁' w₂ : BitVec 32} (h : thinStep xor s w₁ w₂ = thinStep xor s w₁' w₂) :
    w₁ = w₁' := by
  cases xor <;> simp_all [thinStep]

theorem thinStep_inj_right {xor : Bool} {s w₁ w₂ w₂' : BitVec 32}
    (h : thinStep xor s w₁ w₂ = thinStep xor s w₁ w₂') : w₂ = w₂' := by
  cases xor <;> simp_all [thinStep]

theorem thinFold_inj_state : ∀ (ws : List (BitVec 32 × BitVec 32)) (xor : Bool) (s s' : BitVec 32),
    thinFold xor s ws = thinFold xor s' ws → s = s'
  | [], _, _, _, h => h
  | (_, _) :: ws, _, _, _, h => thinStep_inj_state (thinFold_inj_state ws _ _ _ h)

theorem thinFold_append : ∀ (pre : List (BitVec 32 × BitVec 32)) (xor : Bool) (s : BitVec 32),
    ∃ xor' s', ∀ post, thinFold xor s (pre ++ post) = thinFold xor' s' post
  | [], xor, s => ⟨xor, s, fun _ => rfl⟩
  | (w, w') :: pre, xor, s => by
    obtain ⟨xor', s', h⟩ := thinFold_append pre (!xor) (thinStep xor s w w')
    exact ⟨xor', s', fun post => h post⟩

theorem ThinFoldInjective : Accounting.Guarantees.TS.ThinFoldInjective := by
  intro xor s pre post w₁ w₁' w₂ w₂'
  obtain ⟨xor', s', h⟩ := thinFold_append pre xor s
  refine ⟨fun hne e => hne ?_, fun hne e => hne ?_, fun s'' hne e => hne (thinFold_inj_state _ _ _ _ e)⟩
  · rw [h, h] at e
    exact thinStep_inj_left (thinFold_inj_state post _ _ _ e)
  · rw [h, h] at e
    exact thinStep_inj_right (thinFold_inj_state post _ _ _ e)

section Price

variable {K R : Type} [Fintype K] [Fintype R]

theorem price_eq_sum (G : FacetCosts K R) (lam : R → ℝ) (n : K → ℝ) :
    G.price lam n = ∑ k, n k * ∑ r, lam r * G.a r k := by
  unfold FacetCosts.price FacetCosts.load
  simp_rw [mul_sum]
  rw [sum_comm]
  exact sum_congr rfl fun k _ => sum_congr rfl fun r _ => by ring

/-- `c ≤ Σ_k m_k n_k` merges, and `m_k·s ≤ π·a_k` a class: `c·s ≤ π·n`. -/
theorem price_of_fanIn (G : FacetCosts K R) (lam : R → ℝ) (m n : K → ℝ) (c s : ℝ) (hs : 0 ≤ s)
    (hn : ∀ k, 0 ≤ n k) (hc : c ≤ ∑ k, m k * n k) (hk : ∀ k, m k * s ≤ ∑ r, lam r * G.a r k) :
    c * s ≤ G.price lam n := by
  rw [price_eq_sum]
  calc c * s ≤ (∑ k, m k * n k) * s := mul_le_mul_of_nonneg_right hc hs
    _ = ∑ k, n k * (m k * s) := by rw [sum_mul]; exact sum_congr rfl fun k _ => by ring
    _ ≤ ∑ k, n k * ∑ r, lam r * G.a r k := sum_le_sum fun k _ => mul_le_mul_of_nonneg_left (hk k) (hn k)

end Price

/-- The fan-in count read through `FanIn`: `|ι| ≤ |O| + Σ_k m_k n_k` for any `n` above the class counts. -/
theorem card_le_issued {K : Type} [Fintype K] [DecidableEq K] {ι W J : Type} [Fintype ι] [Nontrivial W]
    [Inhabited W] [Fintype J] [DecidableEq J] (D : Dataflow ι W) (O : Finset (ι ⊕ Fin D.n)) (ins : Fin D.n → J)
    (cls : J → K) (m n : K → ℝ) (hO : D.Sees O) (hF : FanIn D ins cls m) (hm : ∀ k, 0 ≤ m k)
    (hn : ∀ k, ((univ.filter fun j => cls j = k).card : ℝ) ≤ n k) :
    (Fintype.card ι : ℝ) ≤ O.card + ∑ k, m k * n k := by
  have h1 : (Fintype.card ι : ℝ) ≤ O.card + ∑ i, (D.merges i : ℝ) := by
    exact_mod_cast FanInCount D O hO
  have h2 : ∑ i, (D.merges i : ℝ) = ∑ j, (D.issued ins j : ℝ) := by
    unfold Dataflow.issued
    push_cast
    exact (sum_fiberwise univ ins fun i => (D.merges i : ℝ)).symm
  have h3 : ∑ j, (D.issued ins j : ℝ) ≤ ∑ j, m (cls j) := sum_le_sum fun j _ => hF j
  have h4 : ∑ j, m (cls j) = ∑ k, ((univ.filter fun j => cls j = k).card : ℝ) * m k := by
    rw [← sum_fiberwise univ cls fun j => m (cls j)]
    refine sum_congr rfl fun k _ => ?_
    rw [sum_congr rfl fun j hj => by rw [(mem_filter.1 hj).2], sum_const, nsmul_eq_mul]
  have h5 : ∑ k, ((univ.filter fun j => cls j = k).card : ℝ) * m k ≤ ∑ k, m k * n k :=
    sum_le_sum fun k _ => by rw [mul_comm]; exact mul_le_mul_of_nonneg_left (hn k) (hm k)
  linarith

/-- A route at `a_k/m_k` a word prices `m_k·s ≤ π·a_k` for any `s` it bounds; `m_k = 0` needs only `a, λ ≥ 0`. -/
theorem class_bound {K R : Type} [Fintype R] (G : FacetCosts K R) (lam : R → ℝ) (hl : ∀ r, 0 ≤ lam r)
    (ha : ∀ r k, 0 ≤ G.a r k) (m : K → ℝ) (hm : ∀ k, 0 ≤ m k) (s : ℝ)
    (hs : ∀ k, 0 < m k → s ≤ (∑ r, lam r * G.a r k) / m k) (k : K) : m k * s ≤ ∑ r, lam r * G.a r k := by
  rcases (hm k).lt_or_eq with hk | hk
  · rw [mul_comm, ← le_div_iff₀ hk]
    exact hs k hk
  · rw [← hk, zero_mul]
    exact sum_nonneg fun r _ => mul_nonneg (hl r) (ha r k)

theorem FoldOptOfFanIn : Accounting.Guarantees.TS.FoldOptOfFanIn := by
  intro K R T L U F X _ _ _ _ _ _ _ _ _ ι W J _ _ _ _ _ _ Z P A D O ins cls m hZ hO hF hm hn hN hOμ hroute lam hlam
  have ha : ∀ r k, 0 ≤ Z.G.a r k := hZ.1
  have hphi := (stars_nonneg Z P lam hZ hlam).2.2.1
  have hc := card_le_issued D O ins cls m A.nfold hO hF hm hn
  refine price_of_fanIn Z.G lam m A.nfold _ _ hphi
    (fun k => le_trans (Nat.cast_nonneg _) (hn k)) (by linarith) ?_
  refine class_bound Z.G lam hlam.1 ha m hm _ (fun k hk => ?_)
  obtain ⟨φ, hφ⟩ := hroute k hk
  calc Z.phiStar lam ≤ demandPrice lam (Z.fold φ) :=
        ciInf_le (Set.finite_range fun φ => demandPrice lam (Z.fold φ)).bddBelow φ
    _ ≤ ∑ r, lam r * (Z.G.a r k / m k) := sum_le_sum fun r _ => mul_le_mul_of_nonneg_left (hφ r) (hlam.1 r)
    _ = (∑ r, lam r * Z.G.a r k) / m k := by rw [sum_div]; exact sum_congr rfl fun r _ => by ring

theorem FillOptOfFanIn : Accounting.Guarantees.TS.FillOptOfFanIn := by
  intro K R T L U F X _ _ _ _ _ _ _ _ _ ι W J _ _ _ _ _ _ _ Z P A D O ins cls m hZ hO hF hm hn hOμ hroute lam hlam
  have ha : ∀ r k, 0 ≤ Z.G.a r k := hZ.1
  have hchi := (stars_nonneg Z P lam hZ hlam).2.2.2
  have hc := card_le_issued D O ins cls m A.nfill hO hF hm hn
  have hNx : (0 : ℝ) < Fintype.card ι := Nat.cast_pos.2 Fintype.card_pos
  have hk : ∀ k, m k * (Z.chiStar lam / Fintype.card ι) ≤ ∑ r, lam r * Z.G.a r k := by
    refine class_bound Z.G lam hlam.1 ha m hm _ (fun k hk => ?_)
    obtain ⟨ψ, hψ⟩ := hroute k hk
    rw [div_le_iff₀ hNx]
    calc Z.chiStar lam ≤ demandPrice lam (Z.fill ψ) :=
          ciInf_le (Set.finite_range fun ψ => demandPrice lam (Z.fill ψ)).bddBelow ψ
      _ ≤ ∑ r, lam r * (Fintype.card ι * Z.G.a r k / m k) :=
          sum_le_sum fun r _ => mul_le_mul_of_nonneg_left (hψ r) (hlam.1 r)
      _ = (∑ r, lam r * Z.G.a r k) / m k * Fintype.card ι := by
          rw [div_mul_eq_mul_div, sum_mul, sum_div]; exact sum_congr rfl fun r _ => by ring
  have key := price_of_fanIn Z.G lam m A.nfill ((1 - Z.muX) * Fintype.card ι) (Z.chiStar lam / Fintype.card ι)
    (div_nonneg hchi hNx.le) (fun k => le_trans (Nat.cast_nonneg _) (hn k)) (by linarith) hk
  calc (1 - Z.muX) * Z.chiStar lam = (1 - Z.muX) * Fintype.card ι * (Z.chiStar lam / Fintype.card ι) := by
        field_simp
    _ ≤ _ := key

end Accounting.SecurityProofs.TS
