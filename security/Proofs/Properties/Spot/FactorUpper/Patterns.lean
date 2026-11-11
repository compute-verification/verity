module

public import Proofs.Properties.Spot.FactorUpper.Sets

/-!
# Rankin's factor bounded above: the patterns' form over requests

`FactorUpperPatternsSound`. Each pattern `c = (r ↦ lightOf |r| U_r)`'s supremum is at most `φ(Σ_r ŷ(c_r))^θ` over the
patterns that meet no cell sampled whole, and `(1 − q)^θ` over those that do (`pat_bound`). The escape is a function
of the whole replay block, which is how it couples the requests: they read disjoint proof blocks, so
`y(U) ≥ Σ_r y(U_r)`, and `φ` is decreasing. The first sum is a product over the requests (`Finset.prod_univ_sum`) of
`1 + Light + Heavy`, Heavy at `y_S`, the least `y` of a set of `S` bits or more (`heavyY_le`), raised to the at most
`N` requests that read in the replay block (`chord_term`). The second is a union bound over the request that meets the
whole cell (`touch_union`, `whole_term`).
-/

@[expose] public section

namespace Properties.Spot.SecurityProofs.FactorUpperProof

section

open _root_.Spot.Interval
open _root_.Spot.FactorUpper
open Claims.Sampling
open Nat

theorem head_mono (m : ℕ) {z z' : ℝ} (h0 : 0 ≤ z) (h : z ≤ z') : head m z ≤ head m z' :=
  Finset.sum_le_sum fun _ _ => div_le_div_of_nonneg_right (pow_le_pow_left₀ h0 h _) (by positivity)

theorem foldl_min_le_init {α : Type*} [LinearOrder α] : ∀ (l : List α) (a : α), l.foldl min a ≤ a
  | [], _ => le_rfl
  | x :: l, a => (foldl_min_le_init l (min a x)).trans (min_le_left a x)

theorem foldl_min_le {α : Type*} [LinearOrder α] : ∀ (l : List α) (a : α) {x : α}, x ∈ l → l.foldl min a ≤ x
  | [], _, _, h => absurd h List.not_mem_nil
  | y :: l, a, x, h => by
    rcases List.mem_cons.1 h with rfl | h
    · exact (foldl_min_le_init l (min a x)).trans (min_le_right a x)
    · exact foldl_min_le l (min a y) h

theorem le_foldl_min {α : Type*} [LinearOrder α] {c : α} :
    ∀ (l : List α) (a : α), c ≤ a → (∀ x ∈ l, c ≤ x) → c ≤ l.foldl min a
  | [], _, ha, _ => ha
  | y :: l, a, ha, hl => le_foldl_min l (min a y) (le_min ha (hl y List.mem_cons_self))
      fun x hx => hl x (List.mem_cons_of_mem y hx)

/-- The largest value a `LogUB` bound allows: `0` at `none`, `e^v` at `some v`. -/
noncomputable def ubVal : Option ℚ → ℝ
  | none => 0
  | some v => Real.exp v

theorem ubVal_nonneg (o : Option ℚ) : 0 ≤ ubVal o := by
  cases o <;> simp [ubVal, (Real.exp_pos _).le]

theorem LogUB.at_ubVal (o : Option ℚ) : LogUB o (ubVal o) := by
  cases o
  · rfl
  · exact le_rfl

theorem LogUB.le_ubVal {o : Option ℚ} {X : ℝ} (h : LogUB o X) : X ≤ ubVal o := by
  cases o
  · exact le_of_eq h
  · exact h

theorem prod_le_pow_filter {ι : Type*} (s : Finset ι) (P : ι → Prop) [DecidablePred P] (f : ι → ℝ) {A : ℝ}
    (hf0 : ∀ i ∈ s, 0 ≤ f i) (hA : ∀ i ∈ s, P i → f i ≤ A) (h1 : ∀ i ∈ s, ¬ P i → f i = 1) :
    ∏ i ∈ s, f i ≤ A ^ (s.filter P).card := by
  rw [← Finset.prod_filter_mul_prod_filter_not s P,
    Finset.prod_eq_one (s := s.filter fun i => ¬ P i)
      (fun i hi => h1 i (Finset.mem_filter.1 hi).1 (Finset.mem_filter.1 hi).2), mul_one, ← Finset.prod_const]
  exact Finset.prod_le_prod₀ (fun i hi => hf0 i (Finset.mem_filter.1 hi).1)
    (fun i hi => hA i (Finset.mem_filter.1 hi).1 (Finset.mem_filter.1 hi).2)

/-- **The union bound** over a finite index. -/
theorem sum_filter_exists_le {ι κ : Type*} [Fintype κ] (s : Finset ι) (P : κ → ι → Prop)
    [∀ r, DecidablePred (P r)] [DecidablePred fun c => ∃ r, P r c] (f : ι → ℝ) (hf : ∀ c ∈ s, 0 ≤ f c) :
    ∑ c ∈ s.filter (fun c => ∃ r, P r c), f c ≤ ∑ r, ∑ c ∈ s.filter (P r), f c := by
  simp only [Finset.sum_filter]
  rw [Finset.sum_comm]
  refine Finset.sum_le_sum fun c hc => ?_
  split_ifs with h
  · obtain ⟨r, hr⟩ := h
    have := Finset.single_le_sum (f := fun r => if P r c then f c else 0)
      (fun r _ => ite_nonneg (hf c hc) le_rfl) (Finset.mem_univ r)
    simpa [hr] using this
  · exact Finset.sum_nonneg fun r _ => ite_nonneg (hf c hc) le_rfl

theorem card_mul_pow_le {k N : ℕ} (hk : k ≤ N) {T A : ℝ} (hT : 0 ≤ T) (hA : 1 ≤ A) :
    k * T * A ^ (k - 1) ≤ N * T * A ^ (N - 1) :=
  mul_le_mul (mul_le_mul_of_nonneg_right (by exact_mod_cast hk) hT)
    (pow_le_pow_right₀ hA (Nat.sub_le_sub_right hk 1)) (by positivity) (by positivity)

/-- **The sets meeting a part**: of the sets `A ⊆ D` under caps `m` on the keys `K`, those with a proof block of key
outside `F`. -/
theorem capped_split {G : Type*} [DecidableEq G] (K : Finset G) (κ : ℕ → G) (x : G → ℝ) (D : Finset ℕ)
    (hD : ∀ v ∈ D, κ v ∈ K) (m : G → ℕ) (F : G → Prop) [DecidablePred F] :
    ∑ A ∈ D.powerset with (∀ k ∈ K, cntOf κ A k ≤ m k) ∧ ¬ (∀ v ∈ A, F (κ v)), ∏ v ∈ A, x (κ v) =
      (∏ k ∈ K.filter F, capP (m k) (cntOf κ D k) (x k)) *
        (∏ k ∈ K.filter (fun k => ¬ F k), capP (m k) (cntOf κ D k) (x k) - 1) := by
  set Df := D.filter fun v => F (κ v)
  have hall := capped_sum_eq K κ x D hD m
  have hfin := capped_sum_eq K κ x Df (fun v hv => hD v (Finset.filter_subset _ _ hv)) m
  have hsplit := Finset.sum_filter_add_sum_filter_not (D.powerset.filter fun A => ∀ k ∈ K, cntOf κ A k ≤ m k)
    (fun A => ∀ v ∈ A, F (κ v)) fun A => ∏ v ∈ A, x (κ v)
  have hset : ((D.powerset.filter fun A => ∀ k ∈ K, cntOf κ A k ≤ m k).filter fun A => ∀ v ∈ A, F (κ v)) =
      Df.powerset.filter fun A => ∀ k ∈ K, cntOf κ A k ≤ m k := by
    ext A
    simp only [Finset.mem_filter, Finset.mem_powerset, Df]
    constructor
    · rintro ⟨⟨hA, hm⟩, hF⟩
      exact ⟨fun v hv => Finset.mem_filter.2 ⟨hA hv, hF v hv⟩, hm⟩
    · rintro ⟨hA, hm⟩
      exact ⟨⟨fun v hv => (Finset.mem_filter.1 (hA hv)).1, hm⟩, fun v hv => (Finset.mem_filter.1 (hA hv)).2⟩
  rw [hset, hfin, Finset.filter_filter, hall] at hsplit
  have hcf : ∀ k ∈ K.filter F, cntOf κ Df k = cntOf κ D k := by
    intro k hk
    unfold cntOf
    congr 1
    ext v
    simp only [Finset.mem_filter, Df]
    constructor
    · rintro ⟨⟨hv, -⟩, he⟩; exact ⟨hv, he⟩
    · rintro ⟨hv, he⟩; exact ⟨⟨hv, he ▸ (Finset.mem_filter.1 hk).2⟩, he⟩
  have hci : ∀ k ∈ K.filter (fun k => ¬ F k), cntOf κ Df k = 0 := by
    intro k hk
    unfold cntOf
    rw [Finset.card_eq_zero, Finset.filter_eq_empty_iff]
    intro v hv he
    exact (Finset.mem_filter.1 hk).2 (he ▸ (Finset.mem_filter.1 hv).2)
  have h1 : ∏ k ∈ K.filter (fun k => ¬ F k), capP (m k) (cntOf κ Df k) (x k) = 1 :=
    Finset.prod_eq_one fun k hk => by rw [hci k hk, capP_zero_right]
  have h2 : ∏ k ∈ K.filter F, capP (m k) (cntOf κ Df k) (x k) =
      ∏ k ∈ K.filter F, capP (m k) (cntOf κ D k) (x k) :=
    Finset.prod_congr rfl fun k hk => by rw [hcf k hk]
  have hfin' : ∏ k ∈ K, capP (m k) (cntOf κ Df k) (x k) = ∏ k ∈ K.filter F, capP (m k) (cntOf κ D k) (x k) := by
    rw [← Finset.prod_filter_mul_prod_filter_not K F, h1, h2, mul_one]
  have hall' := Finset.prod_filter_mul_prod_filter_not K F (fun k => capP (m k) (cntOf κ D k) (x k))
  rw [hfin'] at hsplit
  rw [mul_sub, mul_one]
  linarith

/-- A request's classes: `none`, the sets of `X` (each `some (some A)`) and the classes `Y ⊆ {some none}`. -/
theorem sum_classes (X : Finset (Finset ℕ)) (Y : Finset (Option (Option (Finset ℕ)))) (hY : Y ⊆ {some none})
    (f : Option (Option (Finset ℕ)) → ℝ) :
    ∑ a ∈ X.image (fun A => some (some A)) ∪ Y, f a = ∑ A ∈ X, f (some (some A)) + ∑ a ∈ Y, f a := by
  rw [Finset.sum_union, Finset.sum_image fun _ _ _ _ h => by simpa using h]
  rw [Finset.disjoint_left]
  intro a ha haY
  obtain ⟨A, -, rfl⟩ := Finset.mem_image.1 ha
  simpa using hY haY

theorem sum_classes_none (X : Finset (Finset ℕ)) (Y : Finset (Option (Option (Finset ℕ)))) (hY : Y ⊆ {some none})
    (f : Option (Option (Finset ℕ)) → ℝ) :
    ∑ a ∈ insert none (X.image (fun A => some (some A)) ∪ Y), f a =
      f none + ∑ A ∈ X, f (some (some A)) + ∑ a ∈ Y, f a := by
  rw [Finset.sum_insert, sum_classes X Y hY, add_assoc]
  intro h
  rcases Finset.mem_union.1 h with h | h
  · obtain ⟨A, -, h⟩ := Finset.mem_image.1 h; cases h
  · simpa using hY h

theorem heavy_sub (P : Prop) {inst : Decidable P} :
    (if P then ({some none} : Finset (Option (Option (Finset ℕ)))) else ∅) ⊆ {some none} := by
  split_ifs <;> simp

theorem sum_heavy (P : Prop) {inst : Decidable P} (f : Option (Option (Finset ℕ)) → ℝ) :
    ∑ a ∈ (if P then ({some none} : Finset (Option (Option (Finset ℕ)))) else ∅), f a =
      if P then f (some none) else 0 := by
  split_ifs <;> simp

end

section

open _root_.Spot.Interval
open _root_.Spot.FactorUpper
open Claims.Sampling
open Nat

section Request

variable {s : AuditStatement} {Pt : Partition} {d : Design.Two} {ρ : ℕ} {cs : List Cell}
  {reads : List (List (ℕ × ℕ))} {R : Type} {req : Fin s.prog.outs.length → R}

/-- The proof blocks of replay block `ρ` that request `r` reads. -/
noncomputable def Dr (s : AuditStatement) (Pt : Partition) (d : Design.Two) {R : Type}
    (req : Fin s.prog.outs.length → R) (ρ : ℕ) (r : R) : Finset ℕ :=
  blocksTo s.prog Pt req (blocksIn Pt d.part ρ) r

/-- The proof blocks of `D` in cells the bound samples at a rate. -/
noncomputable def finOf (Pt : Partition) (cs : List Cell) (D : Finset ℕ) : Finset ℕ :=
  D.filter fun v => (cs.getD (Pt.stratumOf v) default).rate.isSome

/-- The light sets of `D`: nonempty, fewer than `S` bits. -/
noncomputable def lsets (s : AuditStatement) (Pt : Partition) (S : ℕ) (D : Finset ℕ) : Finset (Finset ℕ) :=
  D.powerset.filter fun A => A.Nonempty ∧ bitsOf s Pt A < S

theorem blocksTo_subset {U B : Finset ℕ} (h : U ⊆ B) (r : R) :
    blocksTo s.prog Pt req U r ⊆ blocksTo s.prog Pt req B r := by
  classical
  unfold blocksTo; exact Finset.filter_subset_filter _ h

theorem blocksTo_sub_self (U : Finset ℕ) (r : R) : blocksTo s.prog Pt req U r ⊆ U := by
  classical
  unfold blocksTo; exact Finset.filter_subset _ _

theorem bitsOf_eq (A : Finset ℕ) : bitsOf s Pt A = ∑ v ∈ A, wid s Pt v := rfl

/-- Request `r`'s keys and counts: every proof block it reads has a key of `reads`', each at most its count. -/
theorem reads_keys (hc : Properties.Spot.Covers s Pt d ρ cs) {r : R}
    (hr : Properties.Spot.ReadsCover s Pt d req ρ reads r) {D : Finset ℕ}
    (hD : D ⊆ Dr s Pt d req ρ r) :
    (∀ v ∈ D, keyOf s Pt v ∈ keysOf cs reads) ∧
      ∀ k, cntOf (keyOf s Pt) D k ≤ countIn (reads.getD k.1 []) k.2 := by
  classical
  have hmem : ∀ v ∈ D, v ∈ blocksIn Pt d.part ρ ∧ v ∈ blocksTo s.prog Pt req (cell Pt d.part ρ (Pt.stratumOf v)) r := by
    intro v hv
    have h := hD hv
    unfold Dr blocksTo at h
    obtain ⟨hB, hp⟩ := Finset.mem_filter.1 h
    refine ⟨hB, ?_⟩
    unfold blocksTo
    exact Finset.mem_filter.2 ⟨(mem_blocksIn hB).1, hp⟩
  refine ⟨fun v hv => ?_, fun k => ?_⟩
  · obtain ⟨hB, hto⟩ := hmem v hv
    refine mem_keysOf.2 ⟨hc.1 ▸ (mem_blocksIn hB).2, ?_⟩
    show 0 < countIn (reads.getD (Pt.stratumOf v) []) (wid s Pt v)
    refine lt_of_lt_of_le ?_ (hr (Pt.stratumOf v) (wid s Pt v))
    exact Finset.card_pos.2 ⟨v, Finset.mem_filter.2 ⟨hto, rfl⟩⟩
  · rw [cntOf_keyOf]
    refine le_trans (Finset.card_le_card ?_) (hr k.1 k.2)
    intro v hv
    simp only [Finset.mem_filter] at hv
    obtain ⟨⟨hvD, hk1⟩, hk2⟩ := hv
    obtain ⟨-, hto⟩ := hmem v hvD
    rw [hk1] at hto
    exact Finset.mem_filter.2 ⟨hto, hk2⟩

theorem finKeys_nodup (cs : List Cell) (reads : List (List (ℕ × ℕ))) : (finKeys cs reads).Nodup :=
  (keysOf_nodup cs reads).filter _

theorem infKeys_nodup (cs : List Cell) (reads : List (List (ℕ × ℕ))) : (infKeys cs reads).Nodup :=
  (keysOf_nodup cs reads).filter _

/-- The read keys of the cells sampled at a rate, then of those sampled whole: each read key once. -/
theorem allKeys_nodup (cs : List Cell) (reads : List (List (ℕ × ℕ))) :
    (finKeys cs reads ++ infKeys cs reads).Nodup := by
  refine List.Nodup.append (finKeys_nodup cs reads) (infKeys_nodup cs reads) ?_
  intro k hf hi
  have h1 : (cs.getD k.1 default).rate.isSome = true := (List.mem_filter.1 hf).2
  have h2 : (cs.getD k.1 default).rate.isNone = true := (List.mem_filter.1 hi).2
  rw [Option.isNone_iff_eq_none] at h2
  rw [h2] at h1
  simp at h1

theorem mem_allKeys {k : ℕ × ℕ} (hk : k ∈ keysOf cs reads) : k ∈ finKeys cs reads ++ infKeys cs reads := by
  rcases h : (cs.getD k.1 default).rate with _ | _
  · exact List.mem_append_right _ (key_mem_inf hk h)
  · exact List.mem_append_left _ (key_mem_fin hk (by rw [h]; rfl))

/-- The caps a light set's proof blocks meet: at most `n` of a width `0`, at most `S/b` of a width `b > 0`. -/
def mcap (reads : List (List (ℕ × ℕ))) (S : ℕ) (k : ℕ × ℕ) : ℕ :=
  if k.2 = 0 then countIn (reads.getD k.1 []) k.2 else S / k.2

theorem light_caps {D A : Finset ℕ} {S : ℕ} (hA : A ⊆ D)
    (hn : ∀ k, cntOf (keyOf s Pt) D k ≤ countIn (reads.getD k.1 []) k.2) (hb : bitsOf s Pt A < S) (k : ℕ × ℕ) :
    cntOf (keyOf s Pt) A k ≤ mcap reads S k := by
  unfold mcap
  split_ifs with h0
  · exact (cntOf_mono hA k).trans (hn k)
  · rw [Nat.le_div_iff_mul_le (Nat.pos_of_ne_zero h0)]
    have : cntOf (keyOf s Pt) A k * k.2 = ∑ v ∈ A.filter (keyOf s Pt · = k), wid s Pt v := by
      rw [Finset.sum_congr rfl fun v hv => show wid s Pt v = k.2 from
        congrArg Prod.snd (Finset.mem_filter.1 hv).2, Finset.sum_const, smul_eq_mul]
      rfl
    rw [this]
    exact ((Finset.sum_le_sum_of_subset (Finset.filter_subset _ A)).trans (bitsOf_eq A).ge).trans hb.le

theorem capP_le_capVal {k : ℕ × ℕ} {S p c : ℕ} (hc : c ≤ countIn (reads.getD k.1 []) k.2) {x : ℝ} (hx : 0 ≤ x) :
    capP (mcap reads S k) c x ≤ capVal (grp cs reads S p k) S x := by
  have hb : (grp cs reads S p k).b = k.2 := rfl
  have hn : (grp cs reads S p k).n = countIn (reads.getD k.1 []) k.2 := rfl
  unfold mcap capVal
  rw [hb, hn]
  by_cases h0 : k.2 = 0
  · rw [if_pos h0, if_pos h0]; exact (capP_mono hc hx).trans (le_of_eq (capP_of_le le_rfl x))
  · rw [if_neg h0, if_neg h0]; exact capP_mono hc hx

theorem prod_bits_exp (A : Finset ℕ) (c : ℝ) :
    (2 : ℝ) ^ bitsOf s Pt A * Real.exp (-(c * yOf Pt cs A)) =
      ∏ v ∈ A, ((2 : ℝ) ^ (keyOf s Pt v).2 * Real.exp (-(c * (rateOf (cs.getD (keyOf s Pt v).1 default) : ℝ)))) := by
  rw [Finset.prod_mul_distrib, Finset.prod_pow_eq_pow_sum, ← Real.exp_sum]
  unfold yOf
  rw [Finset.mul_sum, ← Finset.sum_neg_distrib]
  rfl

/-- **The light sets' sum** of request `r` in the cells sampled at a rate, at weight `e^{−θs r}`. -/
theorem light_req (hc : Properties.Spot.Covers s Pt d ρ cs) {r : R}
    (hr : Properties.Spot.ReadsCover s Pt d req ρ reads r) (S p : ℕ) (θs : ℚ) :
    LogUB (lightPatterns (Ln2.of p) (((finKeys cs reads).map (grp cs reads S p)).map fun g => (g, -(θs * g.r))) S p)
      (∑ A ∈ lsets s Pt S (finOf Pt cs (Dr s Pt d req ρ r)),
        (2 : ℝ) ^ bitsOf s Pt A * Real.exp (-((θs : ℝ) * yOf Pt cs A))) := by
  set D := finOf Pt cs (Dr s Pt d req ρ r)
  set Kf := finKeys cs reads
  set G := grp cs reads S p
  set n : ℕ × ℕ → ℕ := fun k => countIn (reads.getD k.1 []) k.2
  have hkeys := reads_keys hc hr (D := D) (Finset.filter_subset _ _)
  have hD : ∀ v ∈ D, keyOf s Pt v ∈ Kf := fun v hv => key_mem_fin (hkeys.1 v hv) (Finset.mem_filter.1 hv).2
  set x : ℕ × ℕ → ℝ := fun k => (2 : ℝ) ^ k.2 * Real.exp (-((θs : ℝ) * (rateOf (cs.getD k.1 default) : ℝ)))
  have hx : ∀ k, 0 ≤ x k := fun k => by positivity
  have hsum : ∑ A ∈ lsets s Pt S D, (2 : ℝ) ^ bitsOf s Pt A * Real.exp (-((θs : ℝ) * yOf Pt cs A)) =
      ∑ A ∈ lsets s Pt S D, ∏ v ∈ A, x (keyOf s Pt v) :=
    Finset.sum_congr rfl fun A _ => prod_bits_exp A _
  have hg : ∀ y ∈ (Kf.map G).map fun g => (g, -(θs * g.r)), groupOk y.1 S := by
    intro y hy
    simp only [List.map_map, List.mem_map, Function.comp] at hy
    obtain ⟨k, hk, rfl⟩ := hy
    exact grp_ok (List.mem_filter.1 hk).1
  refine lightPatterns_sound (ln2Ok_of p) _ hg p (Finset.sum_nonneg fun _ _ => by positivity) ?_ ?_
  · rw [hsum]
    have hcap := capped_le_keys Kf (finKeys_nodup cs reads) D (keyOf s Pt) hD n (fun k _ => hkeys.2 k) x hx
      (mcap reads S)
    have hsub : lsets s Pt S D ⊆
        (D.powerset.filter fun A => ∀ k ∈ Kf.toFinset, cntOf (keyOf s Pt) A k ≤ mcap reads S k).erase ∅ := by
      intro A hA
      obtain ⟨hAD, hne, hb⟩ := Finset.mem_filter.1 hA
      refine Finset.mem_erase.2 ⟨hne.ne_empty, Finset.mem_filter.2 ⟨hAD, fun k _ => ?_⟩⟩
      exact light_caps (Finset.mem_powerset.1 hAD) hkeys.2 hb k
    have h0 : (∅ : Finset ℕ) ∈ D.powerset.filter fun A => ∀ k ∈ Kf.toFinset, cntOf (keyOf s Pt) A k ≤ mcap reads S k :=
      Finset.mem_filter.2 ⟨Finset.empty_mem_powerset D, fun k _ => by simp [cntOf]⟩
    have := Finset.add_sum_erase _ (fun A => ∏ v ∈ A, x (keyOf s Pt v)) h0
    rw [Finset.prod_empty] at this
    refine (Finset.sum_le_sum_of_subset_of_nonneg hsub fun _ _ _ => Finset.prod_nonneg fun _ _ => hx _).trans ?_
    refine le_trans (by linarith) (sub_le_sub_right (hcap.trans (le_of_eq ?_)) 1)
    simp only [List.map_map, Function.comp_def]
    congr 1
    refine List.map_congr_left fun k hk => ?_
    unfold mcap capVal
    simp only [G, grp, Group.r, x, rateOf, n]
    split_ifs with h0
    · rw [capP_of_le le_rfl]; push_cast; ring_nf
    · push_cast; ring_nf
  · intro hb0
    set bmin := bminOf ((Kf.map G).map fun g => (g, -(θs * g.r))) S
    have hbmin : bmin = (Kf.map Prod.snd).foldl min (S + 1) := by
      simp only [bmin, bminOf, List.map_map, Function.comp_def, G, grp]
    have hbv : ∀ v ∈ D, bmin ≤ wid s Pt v := fun v hv => by
      rw [hbmin]; exact foldl_min_le _ _ (List.mem_map.2 ⟨keyOf s Pt v, hD v hv, rfl⟩)
    set e : ℕ → ℝ := fun v => Real.exp (-((θs : ℝ) * (rateOf (cs.getD (keyOf s Pt v).1 default) : ℝ)))
    have he : ∀ v, 0 ≤ e v := fun v => (Real.exp_pos _).le
    set M := S / bmin
    have hsub : lsets s Pt S D ⊆ (D.powerset.filter fun A => A.card ≤ M).erase ∅ := by
      intro A hA
      obtain ⟨hAD, hne, hb⟩ := Finset.mem_filter.1 hA
      refine Finset.mem_erase.2 ⟨hne.ne_empty, Finset.mem_filter.2 ⟨hAD, ?_⟩⟩
      rw [Nat.le_div_iff_mul_le hb0]
      have : A.card * bmin = ∑ v ∈ A, bmin := by rw [Finset.sum_const, smul_eq_mul]
      rw [this]
      exact ((Finset.sum_le_sum fun v hv => hbv v (Finset.mem_powerset.1 hAD hv)).trans (bitsOf_eq A).ge).trans hb.le
    have hterm : ∀ A ∈ lsets s Pt S D, (2 : ℝ) ^ bitsOf s Pt A * Real.exp (-((θs : ℝ) * yOf Pt cs A)) ≤
        2 ^ S * ∏ v ∈ A, e v := by
      intro A hA
      rw [prod_bits_exp, Finset.prod_mul_distrib, Finset.prod_pow_eq_pow_sum]
      refine mul_le_mul_of_nonneg_right (pow_le_pow_right₀ one_le_two ?_) (Finset.prod_nonneg fun _ _ => he _)
      exact (Finset.mem_filter.1 hA).2.2.le
    have h0 : (∅ : Finset ℕ) ∈ D.powerset.filter fun A => A.card ≤ M :=
      Finset.mem_filter.2 ⟨Finset.empty_mem_powerset D, by simp⟩
    have hsmall := small_sum_le e he D M
    have := Finset.add_sum_erase _ (fun A => ∏ v ∈ A, e v) h0
    rw [Finset.prod_empty] at this
    have hW : ∑ v ∈ D, e v ≤ (((Kf.map G).map fun g => (g, -(θs * g.r))).map
        fun y => (y.1.n : ℝ) * Real.exp y.2).sum := by
      have := sum_le_keys Kf (finKeys_nodup cs reads) D (keyOf s Pt) hD n (fun k _ => hkeys.2 k)
        (fun k => Real.exp (-((θs : ℝ) * (rateOf (cs.getD k.1 default) : ℝ)))) (fun k => (Real.exp_pos _).le)
      refine this.trans (le_of_eq ?_)
      simp only [List.map_map, Function.comp_def]
      congr 1
      refine List.map_congr_left fun k _ => ?_
      simp only [G, grp, Group.r, rateOf, n]
      push_cast; ring_nf
    calc _ ≤ ∑ A ∈ lsets s Pt S D, 2 ^ S * ∏ v ∈ A, e v := Finset.sum_le_sum hterm
      _ ≤ ∑ A ∈ (D.powerset.filter fun A => A.card ≤ M).erase ∅, 2 ^ S * ∏ v ∈ A, e v :=
        Finset.sum_le_sum_of_subset_of_nonneg hsub fun _ _ _ => by positivity
      _ = 2 ^ S * ∑ A ∈ (D.powerset.filter fun A => A.card ≤ M).erase ∅, ∏ v ∈ A, e v := by
        rw [Finset.mul_sum]
      _ ≤ 2 ^ S * (expT M (∑ v ∈ D, e v) - 1) := mul_le_mul_of_nonneg_left (by linarith) (by positivity)
      _ = 2 ^ S * head M (∑ v ∈ D, e v) := by rw [expT_eq]; ring
      _ ≤ _ := mul_le_mul_of_nonneg_left (head_mono M (Finset.sum_nonneg fun v _ => he v) hW) (by positivity)

end Request

end

section

open _root_.Spot.Interval
open _root_.Spot.FactorUpper
open Claims.Sampling
open Nat

section Request

variable {s : AuditStatement} {Pt : Partition} {d : Design.Two} {ρ : ℕ} {cs : List Cell}
  {reads : List (List (ℕ × ℕ))} {R : Type} {req : Fin s.prog.outs.length → R}

/-- The product of the groups' capped sums at weight `1`, `∏_g L_g`-style, as `touchCount` reads it. -/
noncomputable def capProd (gs : List Group) (S : ℕ) : ℝ :=
  (gs.map fun g => capVal g S (2 ^ g.b * Real.exp ((0 : ℚ) : ℝ))).prod

theorem one_le_capProd (gs : List Group) (S : ℕ) : 1 ≤ capProd gs S := by
  unfold capProd
  induction gs with
  | nil => simp
  | cons g gs ih =>
    rw [List.map_cons, List.prod_cons]
    exact one_le_mul_of_one_le_of_one_le (one_le_capVal g S (by positivity)) ih

theorem filter_fin_eq (cs : List Cell) (reads : List (List (ℕ × ℕ))) :
    (finKeys cs reads ++ infKeys cs reads).toFinset.filter (fun k => (cs.getD k.1 default).rate.isSome = true) =
      (finKeys cs reads).toFinset := by
  ext k
  simp only [Finset.mem_filter, List.mem_toFinset, List.mem_append]
  constructor
  · rintro ⟨hk | hk, h⟩
    · exact hk
    · have h2 : (cs.getD k.1 default).rate.isNone = true := (List.mem_filter.1 hk).2
      rw [Option.isNone_iff_eq_none] at h2
      rw [h2] at h; cases h
  · intro hk
    exact ⟨Or.inl hk, (List.mem_filter.1 hk).2⟩

theorem filter_inf_eq (cs : List Cell) (reads : List (List (ℕ × ℕ))) :
    (finKeys cs reads ++ infKeys cs reads).toFinset.filter (fun k => ¬ (cs.getD k.1 default).rate.isSome = true) =
      (infKeys cs reads).toFinset := by
  ext k
  simp only [Finset.mem_filter, List.mem_toFinset, List.mem_append]
  constructor
  · rintro ⟨hk | hk, h⟩
    · exact absurd (List.mem_filter.1 hk).2 h
    · exact hk
  · intro hk
    refine ⟨Or.inr hk, ?_⟩
    have h2 : (cs.getD k.1 default).rate.isNone = true := (List.mem_filter.1 hk).2
    rw [Option.isNone_iff_eq_none] at h2
    rw [h2]; simp

theorem prod_capP_le {K : List (ℕ × ℕ)} (hK : K.Nodup) (S p : ℕ) (c : ℕ × ℕ → ℕ)
    (hc : ∀ k ∈ K, c k ≤ countIn (reads.getD k.1 []) k.2) :
    ∏ k ∈ K.toFinset, capP (mcap reads S k) (c k) ((2 : ℝ) ^ k.2) ≤ capProd (K.map (grp cs reads S p)) S := by
  unfold capProd
  rw [List.map_map, ← List.prod_toFinset _ hK]
  refine Finset.prod_le_prod₀ (fun k _ => capP_nonneg _ _ (by positivity)) fun k hk => ?_
  have := capP_le_capVal (cs := cs) (p := p) (S := S) (hc k (List.mem_toFinset.1 hk)) (x := (2 : ℝ) ^ k.2)
    (by positivity)
  refine this.trans (le_of_eq ?_)
  simp only [Function.comp_apply, Rat.cast_zero, Real.exp_zero, mul_one]
  rfl

theorem pow_bits (A : Finset ℕ) : (2 : ℝ) ^ bitsOf s Pt A = ∏ v ∈ A, (2 : ℝ) ^ (keyOf s Pt v).2 := by
  rw [Finset.prod_pow_eq_pow_sum]; rfl

/-- **A request's light sets**, all of them: at most `∏_F L_g ∏_I L_g − 1`. -/
theorem all_req (hc : Properties.Spot.Covers s Pt d ρ cs) {r : R}
    (hr : Properties.Spot.ReadsCover s Pt d req ρ reads r) (S p : ℕ) :
    ∑ A ∈ lsets s Pt S (Dr s Pt d req ρ r), (2 : ℝ) ^ bitsOf s Pt A ≤
      capProd ((finKeys cs reads).map (grp cs reads S p)) S * capProd ((infKeys cs reads).map (grp cs reads S p)) S
        - 1 := by
  set D := Dr s Pt d req ρ r
  have hkeys := reads_keys hc hr (D := D) le_rfl
  set K := (finKeys cs reads ++ infKeys cs reads).toFinset
  have hD : ∀ v ∈ D, keyOf s Pt v ∈ K := fun v hv => List.mem_toFinset.2 (mem_allKeys (hkeys.1 v hv))
  set x : ℕ × ℕ → ℝ := fun k => (2 : ℝ) ^ k.2
  have heq := capped_sum_eq K (keyOf s Pt) x D hD (mcap reads S)
  have hsub : lsets s Pt S D ⊆
      (D.powerset.filter fun A => ∀ k ∈ K, cntOf (keyOf s Pt) A k ≤ mcap reads S k).erase ∅ := by
    intro A hA
    obtain ⟨hAD, hne, hb⟩ := Finset.mem_filter.1 hA
    exact Finset.mem_erase.2 ⟨hne.ne_empty,
      Finset.mem_filter.2 ⟨hAD, fun k _ => light_caps (Finset.mem_powerset.1 hAD) hkeys.2 hb k⟩⟩
  have h0 : (∅ : Finset ℕ) ∈ D.powerset.filter fun A => ∀ k ∈ K, cntOf (keyOf s Pt) A k ≤ mcap reads S k :=
    Finset.mem_filter.2 ⟨Finset.empty_mem_powerset D, fun k _ => by simp [cntOf]⟩
  have hadd := Finset.add_sum_erase _ (fun A => ∏ v ∈ A, x (keyOf s Pt v)) h0
  rw [Finset.prod_empty] at hadd
  have hsplit := Finset.prod_filter_mul_prod_filter_not K (fun k => (cs.getD k.1 default).rate.isSome = true)
    (fun k => capP (mcap reads S k) (cntOf (keyOf s Pt) D k) (x k))
  rw [filter_fin_eq, filter_inf_eq] at hsplit
  have hf := prod_capP_le (cs := cs) (finKeys_nodup cs reads) S p (cntOf (keyOf s Pt) D) fun k _ => hkeys.2 k
  have hi := prod_capP_le (cs := cs) (infKeys_nodup cs reads) S p (cntOf (keyOf s Pt) D) fun k _ => hkeys.2 k
  have hi0 : 0 ≤ ∏ k ∈ (infKeys cs reads).toFinset, capP (mcap reads S k) (cntOf (keyOf s Pt) D k) (x k) :=
    Finset.prod_nonneg fun _ _ => capP_nonneg _ _ (by positivity)
  calc ∑ A ∈ lsets s Pt S D, (2 : ℝ) ^ bitsOf s Pt A
      = ∑ A ∈ lsets s Pt S D, ∏ v ∈ A, x (keyOf s Pt v) := Finset.sum_congr rfl fun A _ => pow_bits A
    _ ≤ ∑ A ∈ (D.powerset.filter fun A => ∀ k ∈ K, cntOf (keyOf s Pt) A k ≤ mcap reads S k).erase ∅,
        ∏ v ∈ A, x (keyOf s Pt v) :=
      Finset.sum_le_sum_of_subset_of_nonneg hsub fun _ _ _ => Finset.prod_nonneg fun _ _ => by positivity
    _ = ∏ k ∈ K, capP (mcap reads S k) (cntOf (keyOf s Pt) D k) (x k) - 1 := by rw [← heq]; linarith
    _ ≤ _ := by
      have hP1 := one_le_capProd ((finKeys cs reads).map (grp cs reads S p)) S
      rw [← hsplit]
      exact sub_le_sub_right (mul_le_mul hf hi hi0 (by linarith)) 1

/-- **A request's light sets meeting a cell sampled whole**: at most `∏_F L_g (∏_I L_g − 1)`. -/
theorem touch_req (hc : Properties.Spot.Covers s Pt d ρ cs) {r : R}
    (hr : Properties.Spot.ReadsCover s Pt d req ρ reads r) (S p : ℕ) :
    ∑ A ∈ (lsets s Pt S (Dr s Pt d req ρ r)).filter (fun A => ¬ A ⊆ finOf Pt cs (Dr s Pt d req ρ r)),
        (2 : ℝ) ^ bitsOf s Pt A ≤
      capProd ((finKeys cs reads).map (grp cs reads S p)) S *
        (capProd ((infKeys cs reads).map (grp cs reads S p)) S - 1) := by
  set D := Dr s Pt d req ρ r
  have hkeys := reads_keys hc hr (D := D) le_rfl
  set K := (finKeys cs reads ++ infKeys cs reads).toFinset
  have hD : ∀ v ∈ D, keyOf s Pt v ∈ K := fun v hv => List.mem_toFinset.2 (mem_allKeys (hkeys.1 v hv))
  set x : ℕ × ℕ → ℝ := fun k => (2 : ℝ) ^ k.2
  have heq := capped_split K (keyOf s Pt) x D hD (mcap reads S) (fun k => (cs.getD k.1 default).rate.isSome = true)
  rw [filter_fin_eq, filter_inf_eq] at heq
  have hsub : (lsets s Pt S D).filter (fun A => ¬ A ⊆ finOf Pt cs D) ⊆
      D.powerset.filter fun A => (∀ k ∈ K, cntOf (keyOf s Pt) A k ≤ mcap reads S k) ∧
        ¬ ∀ v ∈ A, (cs.getD (keyOf s Pt v).1 default).rate.isSome = true := by
    intro A hA
    obtain ⟨hA, hnot⟩ := Finset.mem_filter.1 hA
    obtain ⟨hAD, -, hb⟩ := Finset.mem_filter.1 hA
    refine Finset.mem_filter.2 ⟨hAD, fun k _ => light_caps (Finset.mem_powerset.1 hAD) hkeys.2 hb k, ?_⟩
    intro hall
    exact hnot fun v hv => Finset.mem_filter.2 ⟨Finset.mem_powerset.1 hAD hv, hall v hv⟩
  have hf := prod_capP_le (cs := cs) (finKeys_nodup cs reads) S p (cntOf (keyOf s Pt) D) fun k _ => hkeys.2 k
  have hi := prod_capP_le (cs := cs) (infKeys_nodup cs reads) S p (cntOf (keyOf s Pt) D) fun k _ => hkeys.2 k
  have hi1 : 1 ≤ ∏ k ∈ (infKeys cs reads).toFinset, capP (mcap reads S k) (cntOf (keyOf s Pt) D k) (x k) :=
    Finset.one_le_prod₀ fun _ _ => one_le_capP _ _ (by positivity)
  calc _ = ∑ A ∈ (lsets s Pt S D).filter (fun A => ¬ A ⊆ finOf Pt cs D), ∏ v ∈ A, x (keyOf s Pt v) :=
        Finset.sum_congr rfl fun A _ => pow_bits A
    _ ≤ _ := Finset.sum_le_sum_of_subset_of_nonneg hsub fun _ _ _ => Finset.prod_nonneg fun _ _ => by positivity
    _ = _ := heq
    _ ≤ _ := mul_le_mul hf (sub_le_sub_right hi 1) (by linarith)
        (by linarith [one_le_capProd ((finKeys cs reads).map (grp cs reads S p)) S])

/-- A set's bits are at most its groups' `Σ n b`. -/
theorem bits_le_total {K : List (ℕ × ℕ)} (hK : K.Nodup) {D : Finset ℕ} (hD : ∀ v ∈ D, keyOf s Pt v ∈ K)
    (hn : ∀ k, cntOf (keyOf s Pt) D k ≤ countIn (reads.getD k.1 []) k.2) {A : Finset ℕ} (hA : A ⊆ D) (S p : ℕ) :
    bitsOf s Pt A ≤ ((K.map (grp cs reads S p)).map fun g => g.n * g.b).sum := by
  have h := sum_le_keys K hK A (keyOf s Pt) (fun v hv => hD v (hA hv)) (fun k => countIn (reads.getD k.1 []) k.2)
    (fun k _ => (cntOf_mono hA k).trans (hn k)) (fun k => (k.2 : ℝ)) (fun k => Nat.cast_nonneg _)
  have hl : ((bitsOf s Pt A : ℕ) : ℝ) = ∑ v ∈ A, ((keyOf s Pt v).2 : ℝ) := by
    rw [bitsOf_eq]; push_cast; rfl
  have hr : ((((K.map (grp cs reads S p)).map fun g => g.n * g.b).sum : ℕ) : ℝ) =
      (K.map fun k => ((countIn (reads.getD k.1 []) k.2 : ℕ) : ℝ) * (k.2 : ℝ)).sum := by
    rw [Nat.cast_list_sum, List.map_map, List.map_map]
    congr 1
    refine List.map_congr_left fun k _ => ?_
    simp [grp]
  have : ((bitsOf s Pt A : ℕ) : ℝ) ≤ ((((K.map (grp cs reads S p)).map fun g => g.n * g.b).sum : ℕ) : ℝ) := by
    rw [hl, hr]; exact h
  exact_mod_cast this

theorem heavyY_nonneg {K : List (ℕ × ℕ)} {S p : ℕ} {y : ℚ} (h : heavyY (K.map (grp cs reads S p)) S = some y) :
    0 ≤ y := by
  unfold heavyY at h
  split_ifs at h with hlt
  split at h
  · cases h
  rename_i y0 ys hl
  have hy : ys.foldl min y0 = y := Option.some.inj h
  have hall : ∀ c ∈ y0 :: ys, 0 ≤ c := by
    rw [← hl]
    intro c hc
    obtain ⟨g, hg, hgc⟩ := List.mem_filterMap.1 hc
    obtain ⟨k, -, rfl⟩ := List.mem_map.1 hg
    split_ifs at hgc
    obtain rfl := Option.some.inj hgc
    exact mul_nonneg (rateOf_nonneg (cs.getD k.1 default)) (le_max_of_le_left zero_le_one)
  rw [← hy]
  exact le_foldl_min ys y0 (hall y0 List.mem_cons_self) fun c hc => hall c (List.mem_cons_of_mem _ hc)

/-- **No heavy set without `y_S`**: when `heavyY` finds none, every nonempty set of `D` has fewer than `S` bits. -/
theorem heavyY_none {K : List (ℕ × ℕ)} (hK : K.Nodup) {D : Finset ℕ} (hD : ∀ v ∈ D, keyOf s Pt v ∈ K)
    (hn : ∀ k, cntOf (keyOf s Pt) D k ≤ countIn (reads.getD k.1 []) k.2) {S p : ℕ}
    (h : heavyY (K.map (grp cs reads S p)) S = none) {A : Finset ℕ} (hA : A ⊆ D) (hne : A.Nonempty) :
    bitsOf s Pt A < S := by
  unfold heavyY at h
  split_ifs at h with hlt
  · exact lt_of_le_of_lt (bits_le_total hK hD hn hA S p) hlt
  · split at h
    · rename_i hnil
      have hcond : ∀ w ∈ A, wid s Pt w = 0 ∧ 0 < S := by
        intro w hw
        have hg : grp cs reads S p (keyOf s Pt w) ∈ K.map (grp cs reads S p) :=
          List.mem_map.2 ⟨_, hD w (hA hw), rfl⟩
        have h1 := (List.filterMap_eq_nil_iff.1 hnil) _ hg
        split_ifs at h1 with hb
        exact hb
      obtain ⟨v, hv⟩ := hne
      rw [bitsOf_eq, Finset.sum_eq_zero fun w hw => (hcond w hw).1]
      exact (hcond v hv).2
    · cases h

/-- **A heavy set's `y` is at least `y_S`**: every set of `D` of at least `S` bits. -/
theorem heavyY_le {K : List (ℕ × ℕ)} {D : Finset ℕ} (hD : ∀ v ∈ D, keyOf s Pt v ∈ K) {S p : ℕ} {y : ℚ}
    (h : heavyY (K.map (grp cs reads S p)) S = some y) {A : Finset ℕ} (hA : A ⊆ D) (hne : A.Nonempty)
    (hS : S ≤ bitsOf s Pt A) : (y : ℝ) ≤ yOf Pt cs A := by
  have hy0 := heavyY_nonneg h
  unfold heavyY at h
  split_ifs at h with hlt
  split at h
  · cases h
  rename_i y0 ys hl
  have hy : ys.foldl min y0 = y := Option.some.inj h
  have hcand : ∀ w ∈ A, ¬ (wid s Pt w = 0 ∧ 0 < S) →
      y ≤ rateOf (cs.getD (Pt.stratumOf w) default) * max 1 ((S : ℚ) / (wid s Pt w : ℚ)) := by
    intro w hw hc
    have hg : grp cs reads S p (keyOf s Pt w) ∈ K.map (grp cs reads S p) := List.mem_map.2 ⟨_, hD w (hA hw), rfl⟩
    have hmem : rateOf (cs.getD (Pt.stratumOf w) default) * max 1 ((S : ℚ) / (wid s Pt w : ℚ)) ∈ y0 :: ys := by
      rw [← hl]
      refine List.mem_filterMap.2 ⟨_, hg, ?_⟩
      rw [show (grp cs reads S p (keyOf s Pt w)).b = wid s Pt w from rfl, if_neg hc]
      rfl
    rw [← hy]
    rcases List.mem_cons.1 hmem with h0 | h0
    · rw [h0]; exact foldl_min_le_init ys y0
    · exact foldl_min_le ys y0 h0
  have hr0 : ∀ w, (0 : ℝ) ≤ rateOf (cs.getD (Pt.stratumOf w) default) := fun w => by
    exact_mod_cast rateOf_nonneg _
  have hsingle : ∀ w ∈ A, (rateOf (cs.getD (Pt.stratumOf w) default) : ℝ) ≤ yOf Pt cs A := fun w hw =>
    Finset.single_le_sum (f := fun w => (rateOf (cs.getD (Pt.stratumOf w) default) : ℝ)) (fun w _ => hr0 w) hw
  by_cases hS0 : S = 0
  · obtain ⟨w, hw⟩ := hne
    have := hcand w hw (by simp [hS0])
    rw [hS0, Nat.cast_zero, zero_div, max_eq_left zero_le_one, mul_one] at this
    exact (by exact_mod_cast this : (y : ℝ) ≤ _).trans (hsingle w hw)
  by_cases hbig : ∃ w ∈ A, S ≤ wid s Pt w
  · obtain ⟨w, hw, hSw⟩ := hbig
    have := hcand w hw fun h => by omega
    have hbpos : (0 : ℚ) < wid s Pt w := by exact_mod_cast (Nat.pos_of_ne_zero hS0).trans_le hSw
    rw [max_eq_left (div_le_one_of_le₀ (by exact_mod_cast hSw) hbpos.le), mul_one] at this
    exact (by exact_mod_cast this : (y : ℝ) ≤ _).trans (hsingle w hw)
  simp only [not_exists, not_and, not_le] at hbig
  have hper : ∀ w ∈ A, (y : ℝ) * (wid s Pt w : ℝ) ≤ (S : ℝ) * rateOf (cs.getD (Pt.stratumOf w) default) := by
    intro w hw
    by_cases hb0 : wid s Pt w = 0
    · rw [hb0, Nat.cast_zero, mul_zero]; exact mul_nonneg (Nat.cast_nonneg _) (hr0 w)
    · have := hcand w hw fun h => hb0 h.1
      have hbpos : (0 : ℚ) < wid s Pt w := by exact_mod_cast Nat.pos_of_ne_zero hb0
      rw [max_eq_right ((one_le_div hbpos).2 (by exact_mod_cast (hbig w hw).le))] at this
      have hq : y * (wid s Pt w : ℚ) ≤ (S : ℚ) * rateOf (cs.getD (Pt.stratumOf w) default) := by
        rw [mul_div_assoc', le_div_iff₀ hbpos] at this
        linarith
      exact_mod_cast hq
  have hsum : (y : ℝ) * (bitsOf s Pt A : ℝ) ≤ S * yOf Pt cs A := by
    rw [bitsOf_eq]; push_cast; rw [Finset.mul_sum]; unfold yOf; rw [Finset.mul_sum]
    exact Finset.sum_le_sum hper
  have hSpos : (0 : ℝ) < S := by exact_mod_cast Nat.pos_of_ne_zero hS0
  have hyb : (y : ℝ) * S ≤ y * bitsOf s Pt A :=
    mul_le_mul_of_nonneg_left (by exact_mod_cast hS) (by exact_mod_cast hy0)
  exact le_of_mul_le_mul_right (by linarith : (y : ℝ) * S ≤ yOf Pt cs A * S) hSpos

end Request

end

section

open _root_.Spot.Interval
open _root_.Spot.FactorUpper
open Claims.Sampling
open Nat

/-- The requests reading a proof block of replay block `ρ`. -/
noncomputable def actOf (s : AuditStatement) (Pt : Partition) (d : Design.Two) {R : Type} [Fintype R]
    (req : Fin s.prog.outs.length → R) (ρ : ℕ) : Finset R :=
  Finset.univ.filter fun r => (Dr s Pt d req ρ r).Nonempty

open Classical in
/-- The classes `some (some A)` of the sets of `X`, with `some none` when `P`. -/
noncomputable def clsOf (X : Finset (Finset ℕ)) (P : Prop) : Finset (Option (Option (Finset ℕ))) :=
  X.image (fun A => some (some A)) ∪ if P then {some none} else ∅

/-- Request `r`'s classes avoiding the cells sampled whole: heavy when `y_S` exists. -/
noncomputable def firstOf (s : AuditStatement) (Pt : Partition) (d : Design.Two) {R : Type} [Fintype R]
    (req : Fin s.prog.outs.length → R) (ρ : ℕ) (cs : List Cell) (yS : Option ℚ) (S : ℕ) (r : R) :
    Finset (Option (Option (Finset ℕ))) :=
  insert none (clsOf (lsets s Pt S (finOf Pt cs (Dr s Pt d req ρ r))) (r ∈ actOf s Pt d req ρ ∧ yS.isSome))

/-- Request `r`'s classes, heavy when `H` (the groups hold `S` bits or more). -/
noncomputable def allOf (s : AuditStatement) (Pt : Partition) (d : Design.Two) {R : Type} [Fintype R]
    (req : Fin s.prog.outs.length → R) (ρ : ℕ) (H : Prop) (S : ℕ) (r : R) : Finset (Option (Option (Finset ℕ))) :=
  insert none (clsOf (lsets s Pt S (Dr s Pt d req ρ r)) (r ∈ actOf s Pt d req ρ ∧ H))

/-- Request `r`'s classes meeting a cell sampled whole. -/
noncomputable def touchOf (s : AuditStatement) (Pt : Partition) (d : Design.Two) {R : Type} [Fintype R]
    (req : Fin s.prog.outs.length → R) (ρ : ℕ) (cs : List Cell) (H : Prop) (S : ℕ) (r : R) :
    Finset (Option (Option (Finset ℕ))) :=
  clsOf ((lsets s Pt S (Dr s Pt d req ρ r)).filter fun A => ¬ A ⊆ finOf Pt cs (Dr s Pt d req ρ r))
    (r ∈ actOf s Pt d req ρ ∧ H ∧ (Dr s Pt d req ρ r \ finOf Pt cs (Dr s Pt d req ρ r)).Nonempty)

/-- `U`'s pattern. -/
noncomputable def patOf (s : AuditStatement) (Pt : Partition) {R : Type} [DecidableEq R]
    (req : Fin s.prog.outs.length → R) (U : Finset ℕ) (r : R) : Option (Option (Finset ℕ)) :=
  lightOf s Pt (servedBits req r) (blocksTo s.prog Pt req U r)

/-- A class's `y`: `0`, `y_S` heavy, `y(A)` light. -/
noncomputable def yhat (Pt : Partition) (cs : List Cell) (yS : Option ℚ) : Option (Option (Finset ℕ)) → ℝ
  | none => 0
  | some none => ((yS.getD 0 : ℚ) : ℝ)
  | some (some A) => yOf Pt cs A

theorem mem_clsOf_some {X : Finset (Finset ℕ)} {P : Prop} {A : Finset ℕ} :
    some (some A) ∈ clsOf X P ↔ A ∈ X := by
  classical
  unfold clsOf
  constructor
  · intro h
    rcases Finset.mem_union.1 h with h | h
    · obtain ⟨B, hB, hBA⟩ := Finset.mem_image.1 h
      cases hBA; exact hB
    · split_ifs at h <;> simp at h
  · intro h
    exact Finset.mem_union_left _ (Finset.mem_image.2 ⟨A, h, rfl⟩)

theorem mem_clsOf_heavy {X : Finset (Finset ℕ)} {P : Prop} : some none ∈ clsOf X P ↔ P := by
  classical
  unfold clsOf
  constructor
  · intro h
    rcases Finset.mem_union.1 h with h | h
    · obtain ⟨B, -, hB⟩ := Finset.mem_image.1 h; cases hB
    · by_contra hP; rw [if_neg hP] at h; simp at h
  · intro hP
    exact Finset.mem_union_right _ (by rw [if_pos hP]; simp)

theorem none_not_mem_clsOf {X : Finset (Finset ℕ)} {P : Prop} : none ∉ clsOf X P := by
  classical
  unfold clsOf
  intro h
  rcases Finset.mem_union.1 h with h | h
  · obtain ⟨B, -, hB⟩ := Finset.mem_image.1 h; cases hB
  · split_ifs at h <;> simp at h

theorem lightOf_cases (s : AuditStatement) (Pt : Partition) (n : ℕ) (U : Finset ℕ) :
    (U = ∅ ∧ lightOf s Pt n U = none) ∨ (U.Nonempty ∧ n ≤ bitsOf s Pt U ∧ lightOf s Pt n U = some none) ∨
      (U.Nonempty ∧ bitsOf s Pt U < n ∧ lightOf s Pt n U = some (some U)) := by
  unfold lightOf
  by_cases h0 : U = ∅
  · exact Or.inl ⟨h0, if_pos h0⟩
  · have hne := Finset.nonempty_iff_ne_empty.2 h0
    by_cases h1 : n ≤ bitsOf s Pt U
    · exact Or.inr (Or.inl ⟨hne, h1, by rw [if_neg h0, if_pos h1]⟩)
    · exact Or.inr (Or.inr ⟨hne, Nat.lt_of_not_le h1, by rw [if_neg h0, if_neg h1]⟩)

section Patterns

variable {s : AuditStatement} {Pt : Partition} {d : Design.Two} {ρ : ℕ} {cs : List Cell}
  {R : Type} [Fintype R] [DecidableEq R] {req : Fin s.prog.outs.length → R} {S : ℕ}

omit [DecidableEq R] in
theorem mem_actOf {r : R} : r ∈ actOf s Pt d req ρ ↔ (Dr s Pt d req ρ r).Nonempty := by
  simp [actOf]

theorem finOf_subset (D : Finset ℕ) : finOf Pt cs D ⊆ D := Finset.filter_subset _ _

theorem lsets_empty : lsets s Pt S ∅ = ∅ := by
  ext A
  simp only [lsets, Finset.mem_filter, Finset.mem_powerset, Finset.subset_empty, Finset.notMem_empty, iff_false]
  rintro ⟨rfl, hne, -⟩
  exact Finset.not_nonempty_empty hne

omit [Fintype R] [DecidableEq R] in
theorem patOf_sub {U : Finset ℕ} (hU : U ⊆ blocksIn Pt d.part ρ) (r : R) :
    blocksTo s.prog Pt req U r ⊆ Dr s Pt d req ρ r := blocksTo_subset hU r

/-- Every set's class at `r` is one of `r`'s classes. -/
theorem pat_mem_all (hreq : ∀ r, (Dr s Pt d req ρ r).Nonempty → servedBits req r = S) {H : Prop}
    (hH : ∀ r, ∀ A ⊆ Dr s Pt d req ρ r, A.Nonempty → S ≤ bitsOf s Pt A → H) {U : Finset ℕ}
    (hU : U ⊆ blocksIn Pt d.part ρ) (r : R) : patOf s Pt req U r ∈ allOf s Pt d req ρ H S r := by
  have hA := patOf_sub (req := req) hU r
  unfold patOf allOf
  rcases lightOf_cases s Pt (servedBits req r) (blocksTo s.prog Pt req U r) with ⟨-, h⟩ | ⟨hne, hb, h⟩ | ⟨hne, hb, h⟩
  · rw [h]; exact Finset.mem_insert_self _ _
  · have hact := hne.mono hA
    rw [hreq r hact] at hb
    rw [h]
    exact Finset.mem_insert_of_mem (mem_clsOf_heavy.2 ⟨mem_actOf.2 hact, hH r _ hA hne hb⟩)
  · have hact := hne.mono hA
    rw [hreq r hact] at hb
    rw [h]
    exact Finset.mem_insert_of_mem (mem_clsOf_some.2 (Finset.mem_filter.2 ⟨Finset.mem_powerset.2 hA, hne, hb⟩))

/-- **A pattern not avoiding the cells sampled whole** meets one, at a request whose class meets one. -/
theorem pat_touch (hreq : ∀ r, (Dr s Pt d req ρ r).Nonempty → servedBits req r = S) {H : Prop}
    (hH : ∀ r, ∀ A ⊆ Dr s Pt d req ρ r, A.Nonempty → S ≤ bitsOf s Pt A → H) {yS : Option ℚ}
    (hnone : yS = none → ∀ r, ∀ A ⊆ finOf Pt cs (Dr s Pt d req ρ r), A.Nonempty → bitsOf s Pt A < S)
    {U : Finset ℕ} (hU : U ⊆ blocksIn Pt d.part ρ)
    (hPF : patOf s Pt req U ∉ Fintype.piFinset (firstOf s Pt d req ρ cs yS S)) :
    patOf s Pt req U ∈ (Fintype.piFinset (allOf s Pt d req ρ H S)).filter
        (fun c => ∃ r, c r ∈ touchOf s Pt d req ρ cs H S r) ∧
      ∃ v ∈ U, (cs.getD (Pt.stratumOf v) default).rate = none := by
  rw [Fintype.mem_piFinset] at hPF
  simp only [not_forall] at hPF
  obtain ⟨r, hr⟩ := hPF
  have hA := patOf_sub (req := req) hU r
  have hmeet : ∀ A ⊆ Dr s Pt d req ρ r, ¬ A ⊆ finOf Pt cs (Dr s Pt d req ρ r) →
      (Dr s Pt d req ρ r \ finOf Pt cs (Dr s Pt d req ρ r)).Nonempty ∧
        ∃ v ∈ A, (cs.getD (Pt.stratumOf v) default).rate = none := by
    intro A hAD hnot
    obtain ⟨v, hvA, hvf⟩ := Finset.not_subset.1 hnot
    have hvr : (cs.getD (Pt.stratumOf v) default).rate = none := by
      by_contra h
      exact hvf (Finset.mem_filter.2 ⟨hAD hvA, Option.isSome_iff_ne_none.2 h⟩)
    exact ⟨⟨v, Finset.mem_sdiff.2 ⟨hAD hvA, hvf⟩⟩, v, hvA, hvr⟩
  have hall := Fintype.mem_piFinset.2 (pat_mem_all hreq hH hU)
  unfold patOf at hr ⊢
  unfold firstOf at hr
  rcases lightOf_cases s Pt (servedBits req r) (blocksTo s.prog Pt req U r) with ⟨-, h⟩ | ⟨hne, hb, h⟩ | ⟨hne, hb, h⟩
  · rw [h] at hr; exact absurd (Finset.mem_insert_self _ _) hr
  · have hact := hne.mono hA
    rw [hreq r hact] at hb
    rw [h] at hr
    have hyS : yS = none := by
      by_contra hy
      exact hr (Finset.mem_insert_of_mem (mem_clsOf_heavy.2 ⟨mem_actOf.2 hact, Option.isSome_iff_ne_none.2 hy⟩))
    have hnot : ¬ blocksTo s.prog Pt req U r ⊆ finOf Pt cs (Dr s Pt d req ρ r) := fun hsub =>
      absurd hb (not_le.2 (hnone hyS r _ hsub hne))
    obtain ⟨hDi, v, hv, hvr⟩ := hmeet _ hA hnot
    refine ⟨Finset.mem_filter.2 ⟨hall, r, ?_⟩, v, blocksTo_sub_self U r hv, hvr⟩
    show lightOf s Pt (servedBits req r) (blocksTo s.prog Pt req U r) ∈ _
    rw [h]
    exact mem_clsOf_heavy.2 ⟨mem_actOf.2 hact, hH r _ hA hne hb, hDi⟩
  · have hact := hne.mono hA
    rw [hreq r hact] at hb
    rw [h] at hr
    have hnot : ¬ blocksTo s.prog Pt req U r ⊆ finOf Pt cs (Dr s Pt d req ρ r) := fun hsub =>
      hr (Finset.mem_insert_of_mem (mem_clsOf_some.2 (Finset.mem_filter.2 ⟨Finset.mem_powerset.2 hsub, hne, hb⟩)))
    obtain ⟨-, v, hv, hvr⟩ := hmeet _ hA hnot
    refine ⟨Finset.mem_filter.2 ⟨hall, r, ?_⟩, v, blocksTo_sub_self U r hv, hvr⟩
    show lightOf s Pt (servedBits req r) (blocksTo s.prog Pt req U r) ∈ _
    rw [h]
    exact mem_clsOf_some.2 (Finset.mem_filter.2 ⟨Finset.mem_filter.2 ⟨Finset.mem_powerset.2 hA, hne, hb⟩, hnot⟩)

theorem yhat_nonneg {yS : Option ℚ} (hy : ∀ y, yS = some y → 0 ≤ y) (a : Option (Option (Finset ℕ))) :
    0 ≤ yhat Pt cs yS a := by
  rcases a with _ | _ | A
  · exact le_rfl
  · show (0 : ℝ) ≤ ((yS.getD 0 : ℚ) : ℝ)
    rcases h : yS with _ | y
    · simp
    · simpa using hy y h
  · exact yOf_nonneg A

/-- **`y(U) ≥ Σ_r ŷ(c_r)`** for a set avoiding the cells sampled whole whose pattern `c` avoids them too. -/
theorem pat_y_le (hreq : ∀ r, (Dr s Pt d req ρ r).Nonempty → servedBits req r = S)
    (hdisj : ∀ r r', r ≠ r' → Disjoint (Dr s Pt d req ρ r) (Dr s Pt d req ρ r')) {yS : Option ℚ}
    (hsome : ∀ y, yS = some y → ∀ r, ∀ A ⊆ finOf Pt cs (Dr s Pt d req ρ r), A.Nonempty → S ≤ bitsOf s Pt A →
      (y : ℝ) ≤ yOf Pt cs A)
    {U : Finset ℕ} (hU : U ⊆ blocksIn Pt d.part ρ) (hUf : ∀ v ∈ U, (cs.getD (Pt.stratumOf v) default).rate.isSome)
    (hPF : patOf s Pt req U ∈ Fintype.piFinset (firstOf s Pt d req ρ cs yS S)) :
    ∑ r, yhat Pt cs yS (patOf s Pt req U r) ≤ yOf Pt cs U := by
  have hper : ∀ r, yhat Pt cs yS (patOf s Pt req U r) ≤ yOf Pt cs (blocksTo s.prog Pt req U r) := by
    intro r
    have hA := patOf_sub (req := req) hU r
    have hr := Fintype.mem_piFinset.1 hPF r
    unfold patOf at hr ⊢
    unfold firstOf at hr
    rcases lightOf_cases s Pt (servedBits req r) (blocksTo s.prog Pt req U r) with ⟨-, h⟩ | ⟨hne, hb, h⟩ |
      ⟨-, -, h⟩
    · rw [h]; exact yOf_nonneg _
    · have hact := hne.mono hA
      rw [hreq r hact] at hb
      rw [h] at hr ⊢
      rcases Finset.mem_insert.1 hr with h' | h'
      · cases h'
      obtain ⟨-, hy⟩ := mem_clsOf_heavy.1 h'
      obtain ⟨y, hy⟩ := Option.isSome_iff_exists.1 hy
      show ((yS.getD 0 : ℚ) : ℝ) ≤ _
      rw [hy, Option.getD_some]
      refine hsome y hy r _ (fun v hv => Finset.mem_filter.2 ⟨hA hv, ?_⟩) hne hb
      exact hUf v (blocksTo_sub_self U r hv)
    · rw [h]; exact le_rfl
  refine (Finset.sum_le_sum fun r _ => hper r).trans ?_
  unfold yOf
  rw [← Finset.sum_biUnion]
  · exact Finset.sum_le_sum_of_subset_of_nonneg (Finset.biUnion_subset.2 fun r _ => blocksTo_sub_self U r)
      fun v _ _ => by exact_mod_cast rateOf_nonneg _
  · intro r _ r' _ hne
    exact Finset.disjoint_of_subset_left (patOf_sub hU r)
      (Finset.disjoint_of_subset_right (patOf_sub hU r') (hdisj r r' hne))

/-- **A set's pass probability, by its pattern**: `φ(Σ_r ŷ(c_r))^θ` when its pattern avoids the cells sampled whole,
else `(1 − q)^θ`, the pattern then meeting one. -/
theorem pat_bound (hc : Properties.Spot.Covers s Pt d ρ cs)
    (hreq : ∀ r, (Dr s Pt d req ρ r).Nonempty → servedBits req r = S)
    (hdisj : ∀ r r', r ≠ r' → Disjoint (Dr s Pt d req ρ r) (Dr s Pt d req ρ r')) {H : Prop}
    (hH : ∀ r, ∀ A ⊆ Dr s Pt d req ρ r, A.Nonempty → S ≤ bitsOf s Pt A → H) {yS : Option ℚ}
    (hnone : yS = none → ∀ r, ∀ A ⊆ finOf Pt cs (Dr s Pt d req ρ r), A.Nonempty → bitsOf s Pt A < S)
    (hsome : ∀ y, yS = some y → ∀ r, ∀ A ⊆ finOf Pt cs (Dr s Pt d req ρ r), A.Nonempty → S ≤ bitsOf s Pt A →
      (y : ℝ) ≤ yOf Pt cs A)
    {q θ : ℝ} (hq0 : 0 ≤ q) (hq1 : q ≤ 1) (hθ : 0 ≤ θ) {U : Finset ℕ} (hU : U ⊆ blocksIn Pt d.part ρ) :
    d.passes Pt q (counts Pt d.part U) ρ ^ θ ≤
      (if patOf s Pt req U ∈ Fintype.piFinset (firstOf s Pt d req ρ cs yS S) then
          phi q (∑ r, yhat Pt cs yS (patOf s Pt req U r)) ^ θ else 0) +
        (if patOf s Pt req U ∈ (Fintype.piFinset (allOf s Pt d req ρ H S)).filter
            (fun c => ∃ r, c r ∈ touchOf s Pt d req ρ cs H S r) then (1 - q) ^ θ else 0) := by
  have hpq : 0 ≤ (1 - q) ^ θ := Real.rpow_nonneg (by linarith) θ
  by_cases hPF : patOf s Pt req U ∈ Fintype.piFinset (firstOf s Pt d req ρ cs yS S)
  · rw [if_pos hPF]
    refine le_add_of_le_of_nonneg ?_ (by split_ifs <;> [exact hpq; exact le_rfl])
    refine Real.rpow_le_rpow (passes_nonneg hq1 _) ?_ hθ
    by_cases hUf : ∀ v ∈ U, (cs.getD (Pt.stratumOf v) default).rate.isSome
    · exact (passes_le_phi hc hq0 hU).trans (phi_anti hq0 (pat_y_le hreq hdisj hsome hU hUf hPF))
    · simp only [not_forall] at hUf
      obtain ⟨v, hv, hvr⟩ := hUf
      rw [passes_whole hc q hU hv (Option.not_isSome_iff_eq_none.1 hvr)]
      unfold phi
      nlinarith [Real.exp_pos (-(∑ r, yhat Pt cs yS (patOf s Pt req U r)))]
  · obtain ⟨hTS, v, hv, hvr⟩ := pat_touch hreq hH hnone hU hPF
    rw [if_neg hPF, if_pos hTS, zero_add, passes_whole hc q hU hv hvr]

end Patterns

end

section

open _root_.Spot.Interval
open _root_.Spot.FactorUpper
open Claims.Sampling
open Nat

/-- A pattern's weight, `2^{Σ_r b(c_r)}`. -/
noncomputable def Wof (s : AuditStatement) (Pt : Partition) {R : Type} [Fintype R] [DecidableEq R]
    (req : Fin s.prog.outs.length → R) (c : R → Option (Option (Finset ℕ))) : ℝ :=
  (2 : ℝ) ^ ∑ r, lightBits s Pt (servedBits req r) (c r)

/-- A pattern's `y`, `Σ_r ŷ(c_r)`. -/
noncomputable def Yof (Pt : Partition) (cs : List Cell) (yS : Option ℚ) {R : Type} [Fintype R]
    (c : R → Option (Option (Finset ℕ))) : ℝ :=
  ∑ r, yhat Pt cs yS (c r)

section Sums

variable {s : AuditStatement} {Pt : Partition} {d : Design.Two} {ρ : ℕ} {cs : List Cell}
  {R : Type} [Fintype R] [DecidableEq R] {req : Fin s.prog.outs.length → R} {S : ℕ}

omit [DecidableEq R] in
theorem Dr_empty {r : R} (hr : r ∉ actOf s Pt d req ρ) : Dr s Pt d req ρ r = ∅ := by
  rw [mem_actOf, Finset.not_nonempty_iff_eq_empty] at hr; exact hr

theorem finOf_empty : finOf Pt cs ∅ = ∅ := by simp [finOf]

theorem first_le {yS : Option ℚ} (t : ℝ) {r : R} (hS : servedBits req r = S) {Lt Ht : ℝ}
    (hlight : ∑ A ∈ lsets s Pt S (finOf Pt cs (Dr s Pt d req ρ r)),
      (2 : ℝ) ^ bitsOf s Pt A * Real.exp (-(t * yOf Pt cs A)) ≤ Lt)
    (hheavy : ∀ y, yS = some y → (2 : ℝ) ^ S * Real.exp (-(t * y)) ≤ Ht) (hHt : 0 ≤ Ht) :
    ∑ a ∈ firstOf s Pt d req ρ cs yS S r,
        (2 : ℝ) ^ lightBits s Pt (servedBits req r) a * Real.exp (-(t * yhat Pt cs yS a)) ≤ 1 + Lt + Ht := by
  unfold firstOf clsOf
  rw [sum_classes_none _ _ (heavy_sub _), sum_heavy]
  simp only [lightBits, yhat, pow_zero, mul_zero, neg_zero, Real.exp_zero, mul_one, hS]
  refine _root_.add_le_add (_root_.add_le_add le_rfl hlight) ?_
  split_ifs with h
  · obtain ⟨-, hy⟩ := h
    obtain ⟨y, rfl⟩ := Option.isSome_iff_exists.1 hy
    simpa using hheavy y rfl
  · exact hHt

theorem first_one {yS : Option ℚ} (t : ℝ) {r : R} (hr : r ∉ actOf s Pt d req ρ) :
    ∑ a ∈ firstOf s Pt d req ρ cs yS S r,
        (2 : ℝ) ^ lightBits s Pt (servedBits req r) a * Real.exp (-(t * yhat Pt cs yS a)) = 1 := by
  unfold firstOf clsOf
  rw [sum_classes_none _ _ (heavy_sub _), sum_heavy, if_neg fun h => hr h.1, Dr_empty hr, finOf_empty, lsets_empty]
  simp [lightBits, yhat]

theorem all_le {H : Prop} {r : R} (hS : servedBits req r = S) {La Ha : ℝ}
    (hlight : ∑ A ∈ lsets s Pt S (Dr s Pt d req ρ r), (2 : ℝ) ^ bitsOf s Pt A ≤ La)
    (hheavy : H → (2 : ℝ) ^ S ≤ Ha) (hHa : 0 ≤ Ha) :
    ∑ a ∈ allOf s Pt d req ρ H S r, (2 : ℝ) ^ lightBits s Pt (servedBits req r) a ≤ 1 + La + Ha := by
  unfold allOf clsOf
  rw [sum_classes_none _ _ (heavy_sub _), sum_heavy]
  simp only [lightBits, pow_zero, hS]
  refine _root_.add_le_add (_root_.add_le_add le_rfl hlight) ?_
  split_ifs with h
  · exact hheavy h.2
  · exact hHa

theorem all_one {H : Prop} {r : R} (hr : r ∉ actOf s Pt d req ρ) :
    ∑ a ∈ allOf s Pt d req ρ H S r, (2 : ℝ) ^ lightBits s Pt (servedBits req r) a = 1 := by
  unfold allOf clsOf
  rw [sum_classes_none _ _ (heavy_sub _), sum_heavy, if_neg fun h => hr h.1, Dr_empty hr, lsets_empty]
  simp [lightBits]

theorem touch_le {H : Prop} {r : R} (hS : servedBits req r = S) {Lt Ha : ℝ}
    (hlight : ∑ A ∈ (lsets s Pt S (Dr s Pt d req ρ r)).filter (fun A => ¬ A ⊆ finOf Pt cs (Dr s Pt d req ρ r)),
      (2 : ℝ) ^ bitsOf s Pt A ≤ Lt)
    (hheavy : H → (2 : ℝ) ^ S ≤ Ha) (hHa : 0 ≤ Ha) :
    ∑ a ∈ touchOf s Pt d req ρ cs H S r, (2 : ℝ) ^ lightBits s Pt (servedBits req r) a ≤ Lt + Ha := by
  unfold touchOf clsOf
  rw [sum_classes _ _ (heavy_sub _), sum_heavy]
  simp only [lightBits, hS]
  refine _root_.add_le_add hlight ?_
  split_ifs with h
  · exact hheavy h.2.1
  · exact hHa

theorem touch_zero {H : Prop} {r : R} (hr : r ∉ actOf s Pt d req ρ) :
    ∑ a ∈ touchOf s Pt d req ρ cs H S r, (2 : ℝ) ^ lightBits s Pt (servedBits req r) a = 0 := by
  unfold touchOf clsOf
  rw [sum_classes _ _ (heavy_sub _), sum_heavy, if_neg fun h => hr h.1, Dr_empty hr, lsets_empty]
  simp

theorem W_eq_prod (c : R → Option (Option (Finset ℕ))) :
    Wof s Pt req c = ∏ r, (2 : ℝ) ^ lightBits s Pt (servedBits req r) (c r) :=
  (Finset.prod_pow_eq_pow_sum _ _ _).symm

/-- **A chord's weight factors over the requests.** -/
theorem first_prod (yS : Option ℚ) (t : ℝ) :
    ∑ c ∈ Fintype.piFinset (firstOf s Pt d req ρ cs yS S), Wof s Pt req c * Real.exp (-(t * Yof Pt cs yS c)) =
      ∏ r, ∑ a ∈ firstOf s Pt d req ρ cs yS S r,
        (2 : ℝ) ^ lightBits s Pt (servedBits req r) a * Real.exp (-(t * yhat Pt cs yS a)) := by
  rw [Finset.prod_univ_sum]
  refine Finset.sum_congr rfl fun c _ => ?_
  unfold Yof
  rw [W_eq_prod, Finset.mul_sum, ← Finset.sum_neg_distrib, Real.exp_sum, ← Finset.prod_mul_distrib]

theorem W_sum_pi (t : R → Finset (Option (Option (Finset ℕ)))) :
    ∑ c ∈ Fintype.piFinset t, Wof s Pt req c = ∏ r, ∑ a ∈ t r, (2 : ℝ) ^ lightBits s Pt (servedBits req r) a := by
  rw [Finset.prod_univ_sum]
  exact Finset.sum_congr rfl fun c _ => W_eq_prod c

/-- **The patterns' factor**, by the patterns avoiding the cells sampled whole and those meeting one. -/
theorem patterns_le (hc : Properties.Spot.Covers s Pt d ρ cs)
    (hreq : ∀ r, (Dr s Pt d req ρ r).Nonempty → servedBits req r = S)
    (hdisj : ∀ r r', r ≠ r' → Disjoint (Dr s Pt d req ρ r) (Dr s Pt d req ρ r')) {H : Prop}
    (hH : ∀ r, ∀ A ⊆ Dr s Pt d req ρ r, A.Nonempty → S ≤ bitsOf s Pt A → H) {yS : Option ℚ}
    (hnone : yS = none → ∀ r, ∀ A ⊆ finOf Pt cs (Dr s Pt d req ρ r), A.Nonempty → bitsOf s Pt A < S)
    (hsome : ∀ y, yS = some y → ∀ r, ∀ A ⊆ finOf Pt cs (Dr s Pt d req ρ r), A.Nonempty → S ≤ bitsOf s Pt A →
      (y : ℝ) ≤ yOf Pt cs A)
    {q θ : ℝ} (hq0 : 0 ≤ q) (hq1 : q ≤ 1) (hθ : 0 ≤ θ) :
    rankinFactorPatterns s Pt d req q θ ρ ≤
      ∑ c ∈ Fintype.piFinset (firstOf s Pt d req ρ cs yS S), Wof s Pt req c * phi q (Yof Pt cs yS c) ^ θ +
        (1 - q) ^ θ * ∑ c ∈ (Fintype.piFinset (allOf s Pt d req ρ H S)).filter
          (fun c => ∃ r, c r ∈ touchOf s Pt d req ρ cs H S r), Wof s Pt req c := by
  set PF := Fintype.piFinset (firstOf s Pt d req ρ cs yS S)
  set TS := (Fintype.piFinset (allOf s Pt d req ρ H S)).filter (fun c => ∃ r, c r ∈ touchOf s Pt d req ρ cs H S r)
  have hpq : 0 ≤ (1 - q) ^ θ := Real.rpow_nonneg (by linarith) θ
  have hW : ∀ c, 0 ≤ Wof s Pt req c := fun c => by unfold Wof; positivity
  unfold rankinFactorPatterns
  set Img := (blocksIn Pt d.part ρ).powerset.image
    (fun U r => lightOf s Pt (servedBits req r) (blocksTo s.prog Pt req U r))
  have h1 : ∀ c ∈ Img, (2 : ℝ) ^ (∑ r, lightBits s Pt (servedBits req r) (c r)) *
      (⨆ U : ((blocksIn Pt d.part ρ).powerset.filter
          fun U => (fun r => lightOf s Pt (servedBits req r) (blocksTo s.prog Pt req U r)) = c),
        d.passes Pt q (counts Pt d.part U.1) ρ ^ θ) ≤
      Wof s Pt req c * ((if c ∈ PF then phi q (Yof Pt cs yS c) ^ θ else 0) + (if c ∈ TS then (1 - q) ^ θ else 0)) := by
    intro c hcI
    obtain ⟨U0, hU0, hc0⟩ := Finset.mem_image.1 hcI
    refine mul_le_mul_of_nonneg_left ?_ (hW c)
    have : Nonempty ((blocksIn Pt d.part ρ).powerset.filter
        fun U => (fun r => lightOf s Pt (servedBits req r) (blocksTo s.prog Pt req U r)) = c) :=
      ⟨⟨U0, Finset.mem_filter.2 ⟨hU0, hc0⟩⟩⟩
    refine ciSup_le fun U => ?_
    obtain ⟨hUB, hUc⟩ := Finset.mem_filter.1 U.2
    have := pat_bound hc hreq hdisj hH hnone hsome hq0 hq1 hθ (Finset.mem_powerset.1 hUB)
    have hpat : patOf s Pt req U.1 = c := hUc
    rw [hpat] at this
    exact this
  have hsplit : ∀ c, Wof s Pt req c *
      ((if c ∈ PF then phi q (Yof Pt cs yS c) ^ θ else 0) + (if c ∈ TS then (1 - q) ^ θ else 0)) =
      (if c ∈ PF then Wof s Pt req c * phi q (Yof Pt cs yS c) ^ θ else 0) +
        (1 - q) ^ θ * (if c ∈ TS then Wof s Pt req c else 0) := by
    intro c; split_ifs <;> ring
  calc _ ≤ ∑ c ∈ Img, Wof s Pt req c *
        ((if c ∈ PF then phi q (Yof Pt cs yS c) ^ θ else 0) + (if c ∈ TS then (1 - q) ^ θ else 0)) :=
        Finset.sum_le_sum h1
    _ = ∑ c ∈ Img ∩ PF, Wof s Pt req c * phi q (Yof Pt cs yS c) ^ θ +
        (1 - q) ^ θ * ∑ c ∈ Img ∩ TS, Wof s Pt req c := by
      rw [Finset.sum_congr rfl fun c _ => hsplit c, Finset.sum_add_distrib, ← Finset.mul_sum,
        Finset.sum_ite_mem, Finset.sum_ite_mem]
    _ ≤ _ := _root_.add_le_add
        (Finset.sum_le_sum_of_subset_of_nonneg Finset.inter_subset_right fun c _ _ =>
          mul_nonneg (hW c) (Real.rpow_nonneg (phi_pos hq0 hq1 _).le θ))
        (mul_le_mul_of_nonneg_left
          (Finset.sum_le_sum_of_subset_of_nonneg Finset.inter_subset_right fun c _ _ => hW c) hpq)

end Sums

end

section

open _root_.Spot.Interval
open _root_.Spot.FactorUpper
open Claims.Sampling
open Nat

theorem erase_filter_card {R : Type} [Fintype R] [DecidableEq R] {act : Finset R} {r : R} (hr : r ∈ act) :
    ((Finset.univ.erase r).filter (· ∈ act)).card = act.card - 1 := by
  have : (Finset.univ.erase r).filter (· ∈ act) = act.erase r := by
    ext x; simp [and_comm]
  rw [this, Finset.card_erase_of_mem hr]

/-- **The union bound over the requests**: of the patterns of classes `All r`, those with a request in `Touch r`. -/
theorem touch_union {R T : Type} [Fintype R] [DecidableEq R] [DecidableEq T] (All Touch : R → Finset T)
    (w : R → T → ℝ) (hw : ∀ r a, 0 ≤ w r a) (act : Finset R) {N : ℕ} (hN : act.card ≤ N) {A Tb : ℝ}
    (hA1 : 1 ≤ A) (hT0 : 0 ≤ Tb) (hall : ∀ r ∈ act, ∑ a ∈ All r, w r a ≤ A)
    (hall1 : ∀ r ∉ act, ∑ a ∈ All r, w r a = 1) (htouch : ∀ r ∈ act, ∑ a ∈ Touch r, w r a ≤ Tb)
    (htouch0 : ∀ r ∉ act, ∑ a ∈ Touch r, w r a = 0) :
    (∑ c ∈ (Fintype.piFinset All).filter (fun c => ∃ r, c r ∈ Touch r), ∏ r, w r (c r) ≤
        N * Tb * A ^ (N - 1)) ∧
      ∑ c ∈ (Fintype.piFinset All).filter (fun c => ∃ r, c r ∈ Touch r), ∏ r, w r (c r) ≤ A ^ N := by
  have hP : ∀ c : R → T, 0 ≤ ∏ r, w r (c r) := fun c => Finset.prod_nonneg fun r _ => hw r _
  have hall0 : ∀ r, 0 ≤ ∑ a ∈ All r, w r a := fun r => Finset.sum_nonneg fun a _ => hw r a
  constructor
  · calc _ ≤ ∑ r, ∑ c ∈ (Fintype.piFinset All).filter (fun c => c r ∈ Touch r), ∏ r', w r' (c r') :=
          sum_filter_exists_le (Fintype.piFinset All) (fun r c => c r ∈ Touch r) (fun c => ∏ r', w r' (c r'))
            fun c _ => hP c
      _ ≤ ∑ r, ∑ c ∈ Fintype.piFinset (Function.update All r (Touch r)), ∏ r', w r' (c r') := by
          refine Finset.sum_le_sum fun r _ => Finset.sum_le_sum_of_subset_of_nonneg ?_ fun c _ _ => hP c
          intro c hc
          obtain ⟨hcA, hcT⟩ := Finset.mem_filter.1 hc
          refine Fintype.mem_piFinset.2 fun r' => ?_
          by_cases h : r' = r
          · subst h; rw [Function.update_self]; exact hcT
          · rw [Function.update_of_ne h]; exact Fintype.mem_piFinset.1 hcA r'
      _ = ∑ r, (∑ a ∈ Touch r, w r a) * ∏ r' ∈ Finset.univ.erase r, ∑ a ∈ All r', w r' a := by
          refine Finset.sum_congr rfl fun r _ => ?_
          rw [← Finset.prod_univ_sum, ← Finset.mul_prod_erase _ _ (Finset.mem_univ r), Function.update_self]
          congr 1
          exact Finset.prod_congr rfl fun r' hr' => by rw [Function.update_of_ne (Finset.ne_of_mem_erase hr')]
      _ ≤ ∑ r, (if r ∈ act then Tb * A ^ (act.card - 1) else 0) := by
          refine Finset.sum_le_sum fun r _ => ?_
          split_ifs with hr
          · refine mul_le_mul (htouch r hr) ?_ (Finset.prod_nonneg fun r' _ => hall0 r') hT0
            rw [← erase_filter_card hr]
            exact prod_le_pow_filter _ (· ∈ act) (fun r' => ∑ a ∈ All r', w r' a) (fun r' _ => hall0 r')
              (fun r' _ hr' => hall r' hr') fun r' _ hr' => hall1 r' hr'
          · rw [htouch0 r hr, zero_mul]
      _ = act.card * Tb * A ^ (act.card - 1) := by
          rw [Finset.sum_ite_mem, Finset.univ_inter, Finset.sum_const, nsmul_eq_mul, mul_assoc]
      _ ≤ _ := card_mul_pow_le hN hT0 hA1
  · calc _ ≤ ∑ c ∈ Fintype.piFinset All, ∏ r, w r (c r) :=
          Finset.sum_le_sum_of_subset_of_nonneg (Finset.filter_subset _ _) fun c _ _ => hP c
      _ = ∏ r, ∑ a ∈ All r, w r a := (Finset.prod_univ_sum _ _).symm
      _ ≤ A ^ (Finset.univ.filter (· ∈ act)).card :=
          prod_le_pow_filter _ (· ∈ act) (fun r => ∑ a ∈ All r, w r a) (fun r _ => hall0 r)
            (fun r _ hr => hall r hr) fun r _ hr => hall1 r hr
      _ ≤ A ^ N := by
          rw [Finset.filter_mem_eq_inter, Finset.univ_inter]
          exact pow_le_pow_right₀ hA1 hN

section Core

variable {s : AuditStatement} {Pt : Partition} {d : Design.Two} {ρ : ℕ} {cs : List Cell}
  {reads : List (List (ℕ × ℕ))} {R : Type} [Fintype R] [DecidableEq R] {req : Fin s.prog.outs.length → R}
  {S N p : ℕ}

omit [Fintype R] [DecidableEq R] in
theorem fin_keys_req (hc : Properties.Spot.Covers s Pt d ρ cs) {r : R}
    (hr : Properties.Spot.ReadsCover s Pt d req ρ reads r) :
    (∀ v ∈ finOf Pt cs (Dr s Pt d req ρ r), keyOf s Pt v ∈ finKeys cs reads) ∧
      ∀ k, cntOf (keyOf s Pt) (finOf Pt cs (Dr s Pt d req ρ r)) k ≤ countIn (reads.getD k.1 []) k.2 := by
  have hk := reads_keys hc hr (D := finOf Pt cs (Dr s Pt d req ρ r)) (finOf_subset _)
  exact ⟨fun v hv => key_mem_fin (hk.1 v hv) (Finset.mem_filter.1 hv).2, hk.2⟩

/-- **The chords' sum**: `(requestFac)^N − 1`, the patterns avoiding the cells sampled whole at slope `θs`. -/
theorem chord_term (hc : Properties.Spot.Covers s Pt d ρ cs) (hN : (actOf s Pt d req ρ).card ≤ N)
    (hreq : ∀ r, (Dr s Pt d req ρ r).Nonempty →
      servedBits req r = S ∧ Properties.Spot.ReadsCover s Pt d req ρ reads r) (θs : ℚ) :
    LogUB (((requestFac (Ln2.of p) ((finKeys cs reads).map (grp cs reads S p))
        (heavyY ((finKeys cs reads).map (grp cs reads S p)) S) θs S p).pow N (lnUpper N p) p).l)
      (∑ c ∈ (Fintype.piFinset (firstOf s Pt d req ρ cs
          (heavyY ((finKeys cs reads).map (grp cs reads S p)) S) S)).erase (fun _ => none),
        Wof s Pt req c *
          Real.exp (-((θs : ℝ) * Yof Pt cs (heavyY ((finKeys cs reads).map (grp cs reads S p)) S) c))) := by
  set fin := (finKeys cs reads).map (grp cs reads S p)
  set yS := heavyY fin S
  set L := Ln2.of p
  have hL : ln2Ok L := ln2Ok_of p
  set Light := ubVal (lightPatterns L (fin.map fun g => (g, -(θs * g.r))) S p)
  set Heavy := ubVal (yS.map fun y => (S : ℚ) * L.up - θs * y)
  have hL0 : 0 ≤ Light := ubVal_nonneg _
  have hH0 : 0 ≤ Heavy := ubVal_nonneg _
  have hF := requestFac_bounds (fin := fin) (yS := yS) (LogUB.at_ubVal _) hL0 (LogUB.at_ubVal _) hH0
  have hpow := pow_bounds p hF (n := N) (lnN := lnUpper N p) fun hN0 => by
    have := log_le_lnUpper (y := (N : ℚ)) (by exact_mod_cast hN0) p
    simpa using this
  set PF := Fintype.piFinset (firstOf s Pt d req ρ cs yS S)
  set f : R → ℝ := fun r => ∑ a ∈ firstOf s Pt d req ρ cs yS S r,
    (2 : ℝ) ^ lightBits s Pt (servedBits req r) a * Real.exp (-((θs : ℝ) * yhat Pt cs yS a))
  have hc0 : (fun _ : R => (none : Option (Option (Finset ℕ)))) ∈ PF :=
    Fintype.mem_piFinset.2 fun r => Finset.mem_insert_self _ _
  have hadd := Finset.add_sum_erase PF (fun c => Wof s Pt req c * Real.exp (-((θs : ℝ) * Yof Pt cs yS c))) hc0
  have h0 : Wof s Pt req (fun _ : R => (none : Option (Option (Finset ℕ)))) *
      Real.exp (-((θs : ℝ) * Yof Pt cs yS (fun _ : R => (none : Option (Option (Finset ℕ)))))) = 1 := by
    simp [Wof, Yof, lightBits, yhat]
  rw [h0, first_prod] at hadd
  have hheavy : ∀ y, yS = some y → (2 : ℝ) ^ S * Real.exp (-((θs : ℝ) * y)) ≤ Heavy := by
    intro y hy
    simp only [Heavy, hy, Option.map_some, ubVal]
    push_cast
    rw [sub_eq_add_neg, Real.exp_add]
    exact mul_le_mul_of_nonneg_right (two_pow_le_exp hL S) (Real.exp_pos _).le
  have hprod : ∏ r, f r ≤ (1 + (Light + Heavy)) ^ N := by
    refine (prod_le_pow_filter Finset.univ (· ∈ actOf s Pt d req ρ) f (A := 1 + (Light + Heavy))
      (fun r _ => Finset.sum_nonneg fun a _ => by positivity) (fun r _ hr => ?_)
      (fun r _ hr => first_one _ hr)).trans ?_
    · have hra := hreq r (mem_actOf.1 hr)
      have := first_le (yS := yS) (θs : ℝ) hra.1 (light_req hc hra.2 S p θs).le_ubVal hheavy hH0
      linarith
    · rw [Finset.filter_mem_eq_inter, Finset.univ_inter]
      exact pow_le_pow_right₀ (by linarith) hN
  refine hpow.2.2.mono ?_ (by linarith)
  refine Finset.sum_nonneg fun c _ => mul_nonneg ?_ (Real.exp_pos _).le
  unfold Wof; positivity

/-- **The patterns meeting a cell sampled whole**: `touchCount`'s union bound. -/
theorem whole_term (hc : Properties.Spot.Covers s Pt d ρ cs) (hN : (actOf s Pt d req ρ).card ≤ N)
    (hreq : ∀ r, (Dr s Pt d req ρ r).Nonempty →
      servedBits req r = S ∧ Properties.Spot.ReadsCover s Pt d req ρ reads r)
    {q θ : ℚ} (hq1 : (q : ℝ) ≤ 1) (hθ : (0 : ℝ) ≤ θ) :
    LogUB (if ((infKeys cs reads).map (grp cs reads S p)).isEmpty then none
        else oadd (lnPassWhole q θ p) (touchCount (Ln2.of p) ((finKeys cs reads).map (grp cs reads S p))
          ((infKeys cs reads).map (grp cs reads S p)) N S p))
      ((1 - (q : ℝ)) ^ (θ : ℝ) * ∑ c ∈ (Fintype.piFinset (allOf s Pt d req ρ
          (¬ ((((finKeys cs reads).map (grp cs reads S p)) ++ ((infKeys cs reads).map (grp cs reads S p))).map
            fun g => g.n * g.b).sum < S) S)).filter
          (fun c => ∃ r, c r ∈ touchOf s Pt d req ρ cs
            (¬ ((((finKeys cs reads).map (grp cs reads S p)) ++ ((infKeys cs reads).map (grp cs reads S p))).map
              fun g => g.n * g.b).sum < S) S r), Wof s Pt req c) := by
  set fin := (finKeys cs reads).map (grp cs reads S p)
  set inf := (infKeys cs reads).map (grp cs reads S p)
  set H : Prop := ¬ ((fin ++ inf).map fun g => g.n * g.b).sum < S
  set L := Ln2.of p
  have hL : ln2Ok L := ln2Ok_of p
  set All := allOf s Pt d req ρ H S
  set Touch := touchOf s Pt d req ρ cs H S
  set TS := (Fintype.piFinset All).filter (fun c => ∃ r, c r ∈ Touch r)
  have hW : ∀ c, 0 ≤ Wof s Pt req c := fun c => by unfold Wof; positivity
  have hpq : (0 : ℝ) ≤ (1 - (q : ℝ)) ^ (θ : ℝ) := Real.rpow_nonneg (by linarith) _
  by_cases hKe : infKeys cs reads = []
  · have hinf : inf.isEmpty = true := by simp [inf, hKe]
    rw [if_pos hinf]
    show _ = (0 : ℝ)
    have hsub : ∀ r, Dr s Pt d req ρ r ⊆ finOf Pt cs (Dr s Pt d req ρ r) := by
      intro r v hv
      have hk := reads_keys hc (hreq r ⟨v, hv⟩).2 (D := Dr s Pt d req ρ r) le_rfl
      have := mem_allKeys (hk.1 v hv)
      rw [hKe, List.append_nil] at this
      exact Finset.mem_filter.2 ⟨hv, (List.mem_filter.1 this).2⟩
    have hTS : TS = ∅ := by
      rw [Finset.filter_eq_empty_iff]
      rintro c - ⟨r, hr⟩
      simp only [Touch, touchOf, clsOf] at hr
      rcases Finset.mem_union.1 hr with h | h
      · obtain ⟨A, hA, -⟩ := Finset.mem_image.1 h
        obtain ⟨hA1, hA2⟩ := Finset.mem_filter.1 hA
        exact hA2 ((Finset.mem_powerset.1 (Finset.mem_filter.1 hA1).1).trans (hsub r))
      · split_ifs at h with hcond
        · obtain ⟨-, -, v, hv⟩ := hcond
          exact (Finset.mem_sdiff.1 hv).2 (hsub r (Finset.mem_sdiff.1 hv).1)
        · simp at h
    rw [hTS, Finset.sum_empty, mul_zero]
  have hinf : inf.isEmpty = false := by
    cases h : infKeys cs reads with
    | nil => exact absurd h hKe
    | cons _ _ => simp [inf, h]
  rw [if_neg (by rw [hinf]; simp)]
  refine LogUB.oadd (lnPassWhole_sound hq1 hθ p) ?_ (Finset.sum_nonneg fun c _ => hW c)
  set Pf := capProd fin S
  set Pi := capProd inf S
  set Hv : ℝ := if ((fin ++ inf).map fun g => g.n * g.b).sum < S then 0 else 2 ^ S
  have hHv0 : 0 ≤ Hv := by simp only [Hv]; split_ifs <;> positivity
  have hHv : H → (2 : ℝ) ^ S ≤ Hv := fun h => by simp only [Hv, if_neg h, le_rfl]
  have hPf1 : 1 ≤ Pf := one_le_capProd _ _
  have hPi1 : 1 ≤ Pi := one_le_capProd _ _
  have hf : FacBounds (Fac.prod (fin.map (cappedFac L · 0 S p)) p) (Pf - 1) :=
    prod_map_bounds' fin (cappedFac L · 0 S p) (fun g => capVal g S (2 ^ g.b * Real.exp ((0 : ℚ) : ℝ)))
      (fun g hg => by
        obtain ⟨k, hk, rfl⟩ := List.mem_map.1 hg
        exact cappedFac_bounds hL (grp_ok (List.mem_filter.1 hk).1) 0 p) p
  have hi : FacBounds (Fac.prod (inf.map (cappedFac L · 0 S p)) p) (Pi - 1) :=
    prod_map_bounds' inf (cappedFac L · 0 S p) (fun g => capVal g S (2 ^ g.b * Real.exp ((0 : ℚ) : ℝ)))
      (fun g hg => by
        obtain ⟨k, hk, rfl⟩ := List.mem_map.1 hg
        exact cappedFac_bounds hL (grp_ok (List.mem_filter.1 hk).1) 0 p) p
  have hH : LogUB (if ((fin ++ inf).map fun g => g.n * g.b).sum < S then none else some ((S : ℚ) * L.up)) Hv := by
    simp only [Hv]
    split_ifs
    · rfl
    · show (2 : ℝ) ^ S ≤ Real.exp (((S : ℚ) * L.up : ℚ) : ℝ)
      push_cast
      exact two_pow_le_exp hL S
  set A := Pf * Pi + Hv
  set T := Pf * (Pi - 1) + Hv
  have hA1 : 1 ≤ A := by nlinarith
  have hT0 : 0 ≤ T := by nlinarith
  have hsum : ∑ c ∈ TS, Wof s Pt req c =
      ∑ c ∈ TS, ∏ r, (2 : ℝ) ^ lightBits s Pt (servedBits req r) (c r) :=
    Finset.sum_congr rfl fun c _ => W_eq_prod c
  have hu := touch_union All Touch (fun r a => (2 : ℝ) ^ lightBits s Pt (servedBits req r) a)
    (fun r a => by positivity) (actOf s Pt d req ρ) hN hA1 hT0
    (fun r hr => by
      have hra := hreq r (mem_actOf.1 hr)
      have := all_le hra.1 (all_req hc hra.2 S p) hHv hHv0
      linarith)
    (fun r hr => all_one hr)
    (fun r hr => by
      have hra := hreq r (mem_actOf.1 hr)
      exact touch_le hra.1 (touch_req hc hra.2 S p) hHv hHv0)
    (fun r hr => touch_zero hr)
  rw [hsum]
  exact touchCount_sound hf hi hH hHv0 (Finset.sum_nonneg fun c _ => Finset.prod_nonneg fun r _ => by positivity)
    hu.1 hu.2

end Core

end

section

open _root_.Spot.Interval
open _root_.Spot.FactorUpper
open Claims.Sampling
open Nat

section Main

variable {s : AuditStatement} {Pt : Partition} {d : Design.Two} {ρ : ℕ} {cs : List Cell}
  {reads : List (List (ℕ × ℕ))} {R : Type} [Fintype R] [DecidableEq R] {req : Fin s.prog.outs.length → R}
  {S N p : ℕ}

/-- **The patterns' factor at valid parameters `P`** (`P` with `y_S` a breakpoint, or `P` itself). -/
theorem patterns_core (hc : Properties.Spot.Covers s Pt d ρ cs) (hN : (actOf s Pt d req ρ).card ≤ N)
    (hreq : ∀ r, (Dr s Pt d req ρ r).Nonempty →
      servedBits req r = S ∧ Properties.Spot.ReadsCover s Pt d req ρ reads r)
    (hdisj : ∀ r r', r ≠ r' → Disjoint (Dr s Pt d req ρ r) (Dr s Pt d req ρ r'))
    {P : Params} (hP : P.valid = true) {v : ℚ}
    (hv : bound P (fun σ => ((requestFac (Ln2.of p) ((finKeys cs reads).map (grp cs reads S p))
        (heavyY ((finKeys cs reads).map (grp cs reads S p)) S) (P.θ * σ) S p).pow N (lnUpper N p) p).l)
      (if ((infKeys cs reads).map (grp cs reads S p)).isEmpty then none
        else oadd (lnPassWhole P.q P.θ p) (touchCount (Ln2.of p) ((finKeys cs reads).map (grp cs reads S p))
          ((infKeys cs reads).map (grp cs reads S p)) N S p)) = some v) :
    rankinFactorPatterns s Pt d req (P.q : ℝ) (P.θ : ℝ) ρ ≤ Real.exp v := by
  obtain ⟨hq0, hq1, hθ, -⟩ := Params.valid_parts hP
  set fin := (finKeys cs reads).map (grp cs reads S p)
  set inf := (infKeys cs reads).map (grp cs reads S p)
  set yS := heavyY fin S
  set H : Prop := ¬ ((fin ++ inf).map fun g => g.n * g.b).sum < S
  have hreqS : ∀ r, (Dr s Pt d req ρ r).Nonempty → servedBits req r = S := fun r hr => (hreq r hr).1
  have hH : ∀ r, ∀ A ⊆ Dr s Pt d req ρ r, A.Nonempty → S ≤ bitsOf s Pt A → H := by
    intro r A hA hne hS
    have hk := reads_keys hc (hreq r (hne.mono hA)).2 (D := Dr s Pt d req ρ r) le_rfl
    have := bits_le_total (cs := cs) (allKeys_nodup cs reads) (fun v hv => mem_allKeys (hk.1 v hv)) hk.2 hA S p
    rw [List.map_append] at this
    exact not_lt.2 (hS.trans this)
  have hnone : yS = none → ∀ r, ∀ A ⊆ finOf Pt cs (Dr s Pt d req ρ r), A.Nonempty → bitsOf s Pt A < S := by
    intro hy r A hA hne
    have hk := fin_keys_req hc (hreq r (hne.mono (hA.trans (finOf_subset _)))).2
    exact heavyY_none (finKeys_nodup cs reads) hk.1 hk.2 hy hA hne
  have hsome : ∀ y, yS = some y → ∀ r, ∀ A ⊆ finOf Pt cs (Dr s Pt d req ρ r), A.Nonempty →
      S ≤ bitsOf s Pt A → (y : ℝ) ≤ yOf Pt cs A := by
    intro y hy r A hA hne hS
    have hk := fin_keys_req hc (hreq r (hne.mono (hA.trans (finOf_subset _)))).2
    exact heavyY_le hk.1 hy hA hne hS
  have hy0 : ∀ y, yS = some y → 0 ≤ y := fun y hy => heavyY_nonneg hy
  refine (patterns_le hc hreqS hdisj hH hnone hsome hq0 hq1 hθ).trans ?_
  set PF := Fintype.piFinset (firstOf s Pt d req ρ cs yS S)
  have hc0 : (fun _ : R => (none : Option (Option (Finset ℕ)))) ∈ PF :=
    Fintype.mem_piFinset.2 fun r => Finset.mem_insert_self _ _
  have hW : ∀ c, 0 ≤ Wof s Pt req c := fun c => by unfold Wof; positivity
  have hY : ∀ c : R → Option (Option (Finset ℕ)), 0 ≤ Yof Pt cs yS c := fun c =>
    Finset.sum_nonneg fun r _ => yhat_nonneg hy0 _
  have hb := bound_sound hP hv (PF.erase fun _ => none) (Wof s Pt req) (Yof Pt cs yS)
    (fun c => phi P.q (Yof Pt cs yS c) ^ (P.θ : ℝ)) (fun c _ => hW c) (fun c _ => hY c) (fun c _ => le_rfl)
    (fun σ => by
      have := chord_term (p := p) hc hN hreq (P.θ * σ)
      simp only [Rat.cast_mul] at this
      exact this)
    (whole_term hc hN hreq hq1 hθ)
  have hadd := Finset.add_sum_erase PF (fun c => Wof s Pt req c * phi P.q (Yof Pt cs yS c) ^ (P.θ : ℝ)) hc0
  have h0 : Wof s Pt req (fun _ : R => none) * phi P.q (Yof Pt cs yS fun _ : R => none) ^ (P.θ : ℝ) = 1 := by
    simp [Wof, Yof, lightBits, yhat, phi]
  rw [h0] at hadd
  linarith

end Main

/-- **The bound is above the patterns' factor.** -/
theorem factorUpperPatterns_sound : Properties.Spot.FactorUpperPatternsSound := by
  intro cs reads N S P v hv s Pt d R _ _ req ρ hc hN hreq hdisj
  by_cases hP : P.valid = true
  swap
  · simp [lnFactorUpperPatterns, hP] at hv
  unfold lnFactorUpperPatterns at hv
  rw [if_pos hP] at hv
  simp only [split_eq] at hv
  split at hv
  · rename_i y _
    by_cases hP' : ({ P with ys := insertY y P.ys } : Params).valid = true
    · rw [if_pos hP'] at hv
      exact patterns_core (p := P.p) hc hN hreq hdisj hP' hv
    · rw [if_neg hP'] at hv
      exact patterns_core (p := P.p) hc hN hreq hdisj hP hv
  · exact patterns_core (p := P.p) hc hN hreq hdisj hP hv

end

end Properties.Spot.SecurityProofs.FactorUpperProof
