import Definitions.NetworkCertifier.Guarantees.HorizonLeak
import Proofs.Flock.Claims.Sampling.Draws
import Proofs.Flock.Soundness.Audit.Periods
import Proofs.NetworkCertifier.Horizon
import Proofs.NetworkCertifier.Period
import Proofs.NetworkCertifier.Tap

/-!
# The leak through the network certifier over a monitoring period of sampled audits

The probability half: each window's audit, given the earlier windows' coins, accepts with probability at most
`(1 - q)^|W_i| + e_i`, and the escapes multiply over the windows with the errors adding (`pr_adaptive_add_le`). That is
the monitoring-period step (`FlockSoundness.Audit.Periods.prCoin_forall_prod_le`) at each window's bound capped at `1`,
whose product is at most the product of the escapes plus the errors' sum (`prod_min_add_le`); a union bound adds the
leak relations' errors `e' i`.

The count half: a family of faulty sets with fewer than `Fmax` units in total is the units listed by some
`Fin Fmax → Option (Σ i, Fin (n i))` (`exists_famOf_eq`), so there are at most `(N + 1)^Fmax` of them. At one family,
each window's observations number at most `2^#(freeInputs \ reg) · 2^(faultyBits) · Dj^k` (`encard_leakAt_le`, from
`encard_acceptedFaulty_le` at `AsReg`), and the faulty bits sum to at most `b · Fmax`. Under `trusted-host` the cells
the client receives read back from the network certifier's record under the schedule, one its record format holds
(`tap_cells_read`), so they
number at most `#Ss` times those records, which bounds the secrets they decode to (`encard_horizonGood_le`);
`∏_c (cap_c + 1)` is room, as in the period's bound.
-/

namespace NetTiming

open Claims Claims.Sampling Finset

/-! ### The escapes multiply, each with an additive error -/

/-- Bounds capped at `1`, each an escape `a i ≤ 1` plus an error `e i`, multiply to at most the escapes' product plus
the errors' sum. -/
theorem prod_min_add_le : ∀ {m : ℕ} (a e : Fin m → ℝ), (∀ i, 0 ≤ a i) → (∀ i, a i ≤ 1) → (∀ i, 0 ≤ e i) →
    ∏ i, min 1 (a i + e i) ≤ ∏ i, a i + ∑ i, e i
  | 0, _, _, _, _, _ => by simp
  | m + 1, a, e, ha0, ha1, he => by
    have ih : ∏ i : Fin m, min 1 (a i.succ + e i.succ) ≤ ∏ i : Fin m, a i.succ + ∑ i : Fin m, e i.succ :=
      prod_min_add_le _ _ (fun i => ha0 _) (fun i => ha1 _) (fun i => he _)
    have hm0 : ∀ i : Fin m, 0 ≤ min 1 (a i.succ + e i.succ) := fun i => le_min zero_le_one (add_nonneg (ha0 _) (he _))
    have hX0 : 0 ≤ ∏ i : Fin m, min 1 (a i.succ + e i.succ) := Finset.prod_nonneg fun i _ => hm0 i
    have hX1 : ∏ i : Fin m, min 1 (a i.succ + e i.succ) ≤ 1 :=
      Finset.prod_le_one₀ (fun i _ => hm0 i) fun i _ => min_le_left _ _
    have hE0 : 0 ≤ ∑ i : Fin m, e i.succ := Finset.sum_nonneg fun i _ => he _
    rw [Fin.prod_univ_succ, Fin.prod_univ_succ, Fin.sum_univ_succ]
    calc min 1 (a 0 + e 0) * ∏ i : Fin m, min 1 (a i.succ + e i.succ)
        ≤ (a 0 + e 0) * ∏ i : Fin m, min 1 (a i.succ + e i.succ) := mul_le_mul_of_nonneg_right (min_le_right _ _) hX0
      _ = a 0 * ∏ i : Fin m, min 1 (a i.succ + e i.succ) + e 0 * ∏ i : Fin m, min 1 (a i.succ + e i.succ) :=
          add_mul _ _ _
      _ ≤ a 0 * (∏ i : Fin m, a i.succ + ∑ i : Fin m, e i.succ) + e 0 * 1 :=
          add_le_add (mul_le_mul_of_nonneg_left ih (ha0 0)) (mul_le_mul_of_nonneg_left hX1 (he 0))
      _ ≤ a 0 * ∏ i : Fin m, a i.succ + (e 0 + ∑ i : Fin m, e i.succ) := by
          rw [mul_add, mul_one]
          linarith [mul_le_of_le_one_left hE0 (ha1 0)]

open FlockSoundness.Game in
/-- **The escapes multiply with additive errors**: windows whose audits, given the earlier windows' coins, accept with
probability at most `r^(f i) + e i`, `f i` fixed before window `i`'s coins, all accept with `Σ_i f i ≥ F` with
probability at most `r^F + Σ_i e i`: the monitoring-period step at the bounds capped at `1`. -/
theorem pr_adaptive_add_le {Ω : Type} [Fintype Ω] [Nonempty Ω] {r : ℝ} (hr0 : 0 ≤ r) (hr1 : r ≤ 1)
    (m : ℕ) (A : Fin m → (Fin m → Ω) → Ω → Prop) (f : Fin m → (Fin m → Ω) → ℕ) (e : Fin m → ℝ)
    (hA : ∀ i ω ω', (∀ j, j < i → ω j = ω' j) → A i ω = A i ω')
    (hf : ∀ i ω ω', (∀ j, j < i → ω j = ω' j) → f i ω = f i ω') (he : ∀ i, 0 ≤ e i)
    (hesc : ∀ i ω, Pr (A i ω) ≤ r ^ f i ω + e i) (F : ℕ) :
    Pr (fun ω => (∀ i, A i ω (ω i)) ∧ F ≤ ∑ i, f i ω) ≤ r ^ F + ∑ i, e i := by
  have hle1 : ∀ p : Ω → Prop, prCoin p ≤ 1 := fun p =>
    (prCoin_mono (q := fun _ => True) fun _ _ => trivial).trans_eq (by simp [prCoin, avg_const])
  have hstep : ∀ (t : Fin m) (ω : Fin m → Ω), prCoin (fun a : Ω => A t (Function.update ω t a)
      (Function.update ω t a t)) ≤ ENNReal.ofReal (min 1 (r ^ f t ω + e t)) := fun t ω => by
    have hupd : (fun a : Ω => A t (Function.update ω t a) (Function.update ω t a t)) = A t ω := funext fun a => by
      rw [Function.update_self, hA t (Function.update ω t a) ω fun j hj => Function.update_of_ne (ne_of_lt hj) _ _]
    rw [hupd, ← ENNReal.ofReal_toReal (ne_top_of_le_ne_top ENNReal.one_ne_top (hle1 _)),
      ← Claims.Sampling.Pr_eq_toReal_prCoin]
    exact ENNReal.ofReal_le_ofReal (le_min (Pr_le_one _) (hesc t ω))
  have key := FlockSoundness.Audit.Periods.prCoin_forall_prod_le (Ω := fun _ : Fin m => Ω)
    (fun t ω => A t ω (ω t)) (fun t ω => ENNReal.ofReal (min 1 (r ^ f t ω + e t)))
    (fun t ω ω' h => by
      show A t ω (ω t) = A t ω' (ω' t)
      rw [hA t ω ω' fun j hj => h j hj.le, h t le_rfl])
    (fun t ω ω' h => by simp only [hf t ω ω' h]) hstep (ENNReal.ofReal (r ^ F + ∑ i, e i))
  have he0 : 0 ≤ r ^ F + ∑ i, e i := add_nonneg (pow_nonneg hr0 F) (Finset.sum_nonneg fun i _ => he i)
  have hprod : ∀ ω : Fin m → Ω, F ≤ ∑ i, f i ω →
      ∏ t, ENNReal.ofReal (min 1 (r ^ f t ω + e t)) ≤ ENNReal.ofReal (r ^ F + ∑ i, e i) := fun ω hF => by
    rw [← ENNReal.ofReal_prod_of_nonneg fun i _ => le_min zero_le_one (add_nonneg (pow_nonneg hr0 _) (he i))]
    refine ENNReal.ofReal_le_ofReal ((prod_min_add_le (fun i => r ^ f i ω) e (fun i => pow_nonneg hr0 _)
      (fun i => pow_le_one₀ hr0 hr1) he).trans ?_)
    rw [Finset.prod_pow_eq_pow_sum]
    exact add_le_add (pow_le_pow_of_le_one hr0 hr1 hF) le_rfl
  have hmono : prCoin (fun ω : Fin m → Ω => (∀ i, A i ω (ω i)) ∧ F ≤ ∑ i, f i ω) ≤
      ENNReal.ofReal (r ^ F + ∑ i, e i) :=
    (prCoin_mono (q := fun ω : Fin m → Ω => (∀ t, A t ω (ω t)) ∧
      ∏ t, ENNReal.ofReal (min 1 (r ^ f t ω + e t)) ≤ ENNReal.ofReal (r ^ F + ∑ i, e i))
      fun ω hω => ⟨hω.1, hprod ω hω.2⟩).trans key
  rw [Claims.Sampling.Pr_eq_toReal_prCoin]
  exact (ENNReal.toReal_mono ENNReal.ofReal_ne_top hmono).trans_eq (ENNReal.toReal_ofReal he0)

/-! ### The families of faulty sets -/

/-- The units that `g` lists in window `i`. -/
def famOf {m : ℕ} {n : Fin m → ℕ} {F : ℕ} (g : Fin F → Option (Σ i : Fin m, Fin (n i))) (i : Fin m) : Finset ℕ :=
  (univ.filter fun v : Fin (n i) => ∃ t, g t = some ⟨i, v⟩).image Fin.val

theorem famOf_subset {m : ℕ} {n : Fin m → ℕ} {F : ℕ} (g : Fin F → Option (Σ i : Fin m, Fin (n i))) (i : Fin m) :
    famOf g i ⊆ range (n i) := by
  intro x hx
  simp only [famOf, mem_image] at hx
  obtain ⟨v, -, rfl⟩ := hx
  exact mem_range.2 v.2

theorem exists_famOf_eq {m : ℕ} (n : Fin m → ℕ) (Fmax : ℕ) (W : Fin m → Finset ℕ) (hW : ∀ i, W i ⊆ range (n i))
    (hF : ∑ i, (W i).card ≤ Fmax) : ∃ g : Fin Fmax → Option (Σ i : Fin m, Fin (n i)), famOf g = W := by
  classical
  let T : Finset (Σ i : Fin m, Fin (n i)) := univ.sigma fun i => univ.filter fun v : Fin (n i) => v.val ∈ W i
  have hfib : ∀ i, (univ.filter fun v : Fin (n i) => v.val ∈ W i).card = (W i).card := by
    intro i
    rw [← Finset.card_map Fin.valEmbedding]
    congr 1
    ext x
    simp only [Finset.mem_map, Finset.mem_filter, Finset.mem_univ, true_and, Fin.valEmbedding_apply]
    constructor
    · rintro ⟨v, hv, rfl⟩
      exact hv
    · intro hx
      exact ⟨⟨x, Finset.mem_range.1 (hW i hx)⟩, hx, rfl⟩
  have hTc : T.card ≤ Fmax := by
    rw [Finset.card_sigma]
    simp only [hfib]
    exact hF
  let g : Fin Fmax → Option (Σ i : Fin m, Fin (n i)) := fun t =>
    if h : t.val < T.card then some (T.equivFin.symm ⟨t.val, h⟩).val else none
  have hg : ∀ p, (∃ t, g t = some p) ↔ p ∈ T := by
    intro p
    constructor
    · rintro ⟨t, ht⟩
      by_cases h : t.val < T.card
      · simp only [g, h, dite_true, Option.some.injEq] at ht
        rw [← ht]
        exact (T.equivFin.symm _).2
      · simp only [g, h, dite_false, reduceCtorEq] at ht
    · intro hp
      refine ⟨⟨(T.equivFin ⟨p, hp⟩).val, lt_of_lt_of_le (T.equivFin ⟨p, hp⟩).2 hTc⟩, ?_⟩
      simp only [g, (T.equivFin ⟨p, hp⟩).2, dite_true, Fin.eta, Equiv.symm_apply_apply]
  refine ⟨g, funext fun i => ?_⟩
  ext x
  simp only [famOf, mem_image, mem_filter, mem_univ, true_and, hg]
  constructor
  · rintro ⟨v, hv, rfl⟩
    simpa [T] using hv
  · intro hx
    exact ⟨⟨x, mem_range.1 (hW i hx)⟩, by simpa [T] using hx, rfl⟩

theorem faultyBits_le_card_mul (C : Prog) (P : Partition) {n b : ℕ} {F : Finset ℕ} (hF : F ⊆ range n)
    (hb : ∀ v < n, faultyBits C P {v} ≤ b) : faultyBits C P F ≤ F.card * b := by
  have h : faultyBits C P F = ∑ v ∈ F, faultyBits C P {v} := by simp only [faultyBits, Finset.sum_singleton]
  rw [h]
  calc ∑ v ∈ F, faultyBits C P {v} ≤ ∑ _v ∈ F, b := sum_le_sum fun v hv => hb v (mem_range.1 (hF hv))
    _ = F.card * b := by rw [sum_const, smul_eq_mul]

/-! ### The count -/

theorem encard_leakAt_le {L K S : ℕ} (lay : AuditStatement → Layout L S) (k : AuditStatement → ℕ) (reg : AuditStatement → Finset ℕ) (Dj : ℕ)
    (s : AuditStatement) (P : Partition) (ρ : ℕ → Bool) (op : ℕ → List (Obs L K S) → List Bool) (F : Finset ℕ) :
    {o | LeakAt lay k reg Dj s P ρ op F o}.encard ≤
      ((2 ^ (freeInputs s.prog (lay s) (k s) \ reg s).card * (2 ^ faultyBits s.prog P F * Dj ^ k s) : ℕ) : ℕ∞) := by
  by_cases hfits : (lay s).Fits s.prog (k s)
  · calc _ ≤ {o : List (Obs L K S) | o.length = k s ∧ (∀ x ∈ o, x.jitter < Dj) ∧
            AcceptedFaulty s.prog P (lay s) (k s) (AsReg s.prog (lay s) (k s) (reg s) ρ) F op o}.encard :=
          Set.encard_le_encard fun o h => h.2
      _ ≤ (AsReg s.prog (lay s) (k s) (reg s) ρ).encard * ((2 ^ faultyBits s.prog P F * Dj ^ k s : ℕ) : ℕ∞) :=
          encard_acceptedFaulty_le _ _ _ _ hfits _ _ _ _
      _ ≤ ((2 ^ (freeInputs s.prog (lay s) (k s) \ reg s).card : ℕ) : ℕ∞) *
            ((2 ^ faultyBits s.prog P F * Dj ^ k s : ℕ) : ℕ∞) :=
          mul_le_mul' (encard_asReg_le _ _ _ _ _) le_rfl
      _ = _ := (Nat.cast_mul _ _).symm
  · have : {o | LeakAt lay k reg Dj s P ρ op F o} = ∅ :=
      Set.eq_empty_iff_forall_notMem.2 fun o h => hfits h.1
    rw [this, Set.encard_empty]
    exact zero_le

theorem encard_horizonGood_le {L K S : ℕ} (lay : AuditStatement → Layout L S) (k : AuditStatement → ℕ) (reg : AuditStatement → Finset ℕ) (Dj : ℕ)
    {m : ℕ} (sw : Fin m → AuditStatement) (Pw : Fin m → Partition) (ρ : Fin m → ℕ → Bool)
    (op : Fin m → ℕ → List (Obs L K S) → List Bool) (n : Fin m → ℕ) (b : ℕ)
    (hb : ∀ i, ∀ v < n i, faultyBits (sw i).prog (Pw i) {v} ≤ b)
    {Conn Cell Srv Dev Z : Type} [Fintype Conn] (arr : Srv → Conn → Arrivals Cell)
    (f : TapRecord Conn Cell (Fin m → List (Obs L K S))) (tap : Dev)
    (run : Dev → WardenSchedule Conn × Srv → (Conn → List (ℕ × CellView Cell)) × (Fin m → List (Obs L K S)))
    (htrust : Verity.Assumptions.TrustedHost tap run fun p => tapCode f p.1 (arr p.2))
    (cap : Conn → ℕ) (Ss : Set (WardenSchedule Conn)) (hfit : ∀ s ∈ Ss, f.fits s)
    (Zs : Set Z) (sched : Z → WardenSchedule Conn) (srv : Z → Srv) (hs : ∀ z ∈ Zs, sched z ∈ Ss)
    (dec : (Conn → List (ℕ × CellView Cell)) → Z) (hdz : ∀ z ∈ Zs, dec (run tap (sched z, srv z)).1 = z)
    (W : Z → Fin m → Finset ℕ) (hWn : ∀ z i, W z i ⊆ range (n i)) (Fmax : ℕ) :
    {z | z ∈ Zs ∧ HorizonGood lay k reg Dj sw Pw ρ op Fmax (W z) (run tap (sched z, srv z)).2}.encard ≤
      Ss.encard * (((∏ c, (cap c + 1)) * ((∑ i, n i) + 1) ^ Fmax * 2 ^ (b * Fmax) *
        ∏ i, (2 ^ (freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \ reg (sw i)).card * Dj ^ k (sw i)) : ℕ) : ℕ∞) := by
  classical
  set G := {z | z ∈ Zs ∧ HorizonGood lay k reg Dj sw Pw ρ op Fmax (W z) (run tap (sched z, srv z)).2} with hG
  set c : Fin m → ℕ := fun i => (freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \ reg (sw i)).card with hc
  set B : ℕ := 2 ^ (b * Fmax) * ∏ i, (2 ^ c i * Dj ^ k (sw i)) with hB
  let s : Finset (Fin Fmax → Option (Σ i : Fin m, Fin (n i))) := univ.filter fun g => ∑ i, (famOf g i).card < Fmax
  let X : (Fin Fmax → Option (Σ i : Fin m, Fin (n i))) → Set (Fin m → List (Obs L K S)) := fun g =>
    Set.pi Set.univ fun i => {o | LeakAt lay k reg Dj (sw i) (Pw i) (ρ i) (op i) (famOf g i) o}
  let Y : Set (Fin m → List (Obs L K S)) := ⋃ g ∈ s, X g
  let V : Set (Conn → List (ℕ × CellView Cell)) :=
    (fun p : WardenSchedule Conn × (Fin m → List (Obs L K S)) => f.read p.1 p.2) '' (Ss ×ˢ Y)
  have hgood : ∀ z ∈ G, (run tap (sched z, srv z)).2 ∈ Y := by
    rintro z ⟨-, hlt, hleak⟩
    obtain ⟨g, hg⟩ := exists_famOf_eq n Fmax (W z) (hWn z) hlt.le
    refine Set.mem_biUnion (x := g) ?_ ?_
    · simp only [s, Finset.coe_filter, Finset.mem_univ, true_and, Set.mem_ofPred_eq, hg]
      exact hlt
    · intro i _
      simp only [Set.mem_ofPred_eq, hg]
      exact hleak i
  have hsub : G ⊆ dec '' V := fun z hz =>
    ⟨(run tap (sched z, srv z)).1, ⟨(sched z, (run tap (sched z, srv z)).2), ⟨hs z hz.1, hgood z hz⟩,
      (tap_cells_read f arr tap run htrust _ (hfit _ (hs z hz.1)) _).symm⟩, hdz z hz.1⟩
  have hX : ∀ g ∈ s, (X g).encard ≤ ((B : ℕ) : ℕ∞) := by
    intro g hg
    have hlt : ∑ i, (famOf g i).card < Fmax := (Finset.mem_filter.1 hg).2
    have hbits : ∑ i, faultyBits (sw i).prog (Pw i) (famOf g i) ≤ b * Fmax := by
      calc ∑ i, faultyBits (sw i).prog (Pw i) (famOf g i) ≤ ∑ i, (famOf g i).card * b :=
            sum_le_sum fun i _ => faultyBits_le_card_mul _ _ (famOf_subset g i) (hb i)
        _ = (∑ i, (famOf g i).card) * b := (sum_mul _ _ _).symm
        _ ≤ Fmax * b := Nat.mul_le_mul_right _ hlt.le
        _ = b * Fmax := mul_comm _ _
    have hnat : ∏ i, (2 ^ c i * (2 ^ faultyBits (sw i).prog (Pw i) (famOf g i) * Dj ^ k (sw i))) ≤ B := by
      calc ∏ i, (2 ^ c i * (2 ^ faultyBits (sw i).prog (Pw i) (famOf g i) * Dj ^ k (sw i)))
          = (∏ i, 2 ^ faultyBits (sw i).prog (Pw i) (famOf g i)) * ∏ i, (2 ^ c i * Dj ^ k (sw i)) := by
            rw [← Finset.prod_mul_distrib]
            exact Finset.prod_congr rfl fun i _ => by ring
        _ = 2 ^ (∑ i, faultyBits (sw i).prog (Pw i) (famOf g i)) * ∏ i, (2 ^ c i * Dj ^ k (sw i)) := by
            rw [Finset.prod_pow_eq_pow_sum]
        _ ≤ B := Nat.mul_le_mul_right _ (Nat.pow_le_pow_right (by norm_num) hbits)
    calc (X g).encard = ∏ i, {o | LeakAt lay k reg Dj (sw i) (Pw i) (ρ i) (op i) (famOf g i) o}.encard :=
          Set.encard_pi_eq_prod_encard
      _ ≤ ∏ i, ((2 ^ c i * (2 ^ faultyBits (sw i).prog (Pw i) (famOf g i) * Dj ^ k (sw i)) : ℕ) : ℕ∞) :=
          Finset.prod_le_prod fun i _ => encard_leakAt_le _ _ _ _ _ _ _ _ _
      _ = ((∏ i, (2 ^ c i * (2 ^ faultyBits (sw i).prog (Pw i) (famOf g i) * Dj ^ k (sw i))) : ℕ) : ℕ∞) :=
          (Nat.cast_prod _ _).symm
      _ ≤ ((B : ℕ) : ℕ∞) := by exact_mod_cast hnat
  have hY : Y.encard ≤ ((((∑ i, n i) + 1) ^ Fmax * B : ℕ) : ℕ∞) := by
    calc Y.encard ≤ ∑ g ∈ s, (X g).encard := encard_biUnion_le s X
      _ ≤ ∑ _g ∈ s, ((B : ℕ) : ℕ∞) := sum_le_sum hX
      _ = ((s.card * B : ℕ) : ℕ∞) := by rw [sum_const, nsmul_eq_mul]; push_cast; ring
      _ ≤ ((((∑ i, n i) + 1) ^ Fmax * B : ℕ) : ℕ∞) := by
          have : s.card ≤ ((∑ i, n i) + 1) ^ Fmax := by
            refine (Finset.card_le_univ s).trans_eq ?_
            rw [Fintype.card_fun, Fintype.card_option, Fintype.card_sigma, Fintype.card_fin]
            simp only [Fintype.card_fin]
          exact_mod_cast Nat.mul_le_mul_right B this
  have hcap1 : 0 < ∏ c, (cap c + 1) := Finset.prod_pos fun c _ => Nat.succ_pos _
  have hroom : ((∑ i, n i) + 1) ^ Fmax * B ≤ (∏ c, (cap c + 1)) * ((∑ i, n i) + 1) ^ Fmax * 2 ^ (b * Fmax) *
      ∏ i, (2 ^ c i * Dj ^ k (sw i)) := by
    calc ((∑ i, n i) + 1) ^ Fmax * B ≤ (∏ c, (cap c + 1)) * (((∑ i, n i) + 1) ^ Fmax * B) :=
          Nat.le_mul_of_pos_left _ hcap1
      _ = _ := by rw [hB]; ring
  calc G.encard ≤ (dec '' V).encard := Set.encard_le_encard hsub
    _ ≤ V.encard := Set.encard_image_le _ _
    _ ≤ (Ss ×ˢ Y).encard := Set.encard_image_le _ _
    _ = Ss.encard * Y.encard := Set.encard_prod
    _ ≤ _ := by
        gcongr
        exact hY.trans (by exact_mod_cast hroom)

/-! ### The monitoring period -/

/-- **The leak through the network certifier over a monitoring period**: `Guarantees.HorizonLeakAccepted`, from
`pr_adaptive_add_le` at the conditional escape, a union bound over the leak relations, and `encard_horizonGood_le` at
each draw of the coins. -/
theorem horizonLeak : Guarantees.HorizonLeakAccepted := by
  intro L K S lay k reg Dj m sw Pw ρ op n b hb Ω _ _ Conn Cell Srv Dev Z _ arr f tap run htrust cap Ss hcap hfit Zs
    schedOf srvOf hs dec hdz acc hacc W hW hWn e e' he q hq0 hq1 hesc hleak δ hδ Fmax hF
  refine ⟨fun z hz => ?_, fun ω => encard_horizonGood_le lay k reg Dj sw Pw ρ op n b hb arr f tap run htrust cap Ss
    hfit Zs (fun z => schedOf z ω) (fun z => srvOf z ω) (fun z hz => hs z hz ω) (dec ω) (fun z hz => hdz z hz ω)
    (fun z => W z ω) (fun z i => hWn z ω i) Fmax⟩
  classical
  have hmiss := pr_adaptive_add_le (Ω := Ω) (r := 1 - q) (by linarith) (by linarith) m (acc z)
    (fun i ω => (W z ω i).card) e (fun i ω ω' h => hacc z i ω ω' h)
    (fun i ω ω' h => by simp only [hW z i ω ω' h]) he (fun i ω => hesc z hz i ω) Fmax
  have hpow : (1 - q) ^ Fmax ≤ δ := by
    have h1 : 1 - q ≤ Real.exp (-q) := by linarith [Real.add_one_le_exp (-q)]
    calc (1 - q) ^ Fmax ≤ Real.exp (-q) ^ Fmax := pow_le_pow_left₀ (by linarith) h1 Fmax
      _ = Real.exp (Fmax * (-q)) := (Real.exp_nat_mul _ _).symm
      _ ≤ Real.exp (-Real.log (1 / δ)) := Real.exp_le_exp.2 (by nlinarith)
      _ = δ := by rw [one_div, Real.log_inv, neg_neg, Real.exp_log hδ]
  calc Pr (fun ω => (∀ i, acc z i ω (ω i)) ∧
        ¬HorizonGood lay k reg Dj sw Pw ρ op Fmax (W z ω) (run tap (schedOf z ω, srvOf z ω)).2)
      ≤ Pr (fun ω => ((∀ i, acc z i ω (ω i)) ∧ Fmax ≤ ∑ i, (W z ω i).card) ∨
          ∃ i ∈ (univ : Finset (Fin m)), acc z i ω (ω i) ∧
            ¬LeakAt lay k reg Dj (sw i) (Pw i) (ρ i) (op i) (W z ω i) ((run tap (schedOf z ω, srvOf z ω)).2 i)) := by
        refine Pr_mono fun ω h => ?_
        obtain ⟨hall, hng⟩ := h
        unfold HorizonGood at hng
        by_cases hlt : ∑ i, (W z ω i).card < Fmax
        · have : ¬∀ i, LeakAt lay k reg Dj (sw i) (Pw i) (ρ i) (op i) (W z ω i)
              ((run tap (schedOf z ω, srvOf z ω)).2 i) := fun hl => hng ⟨hlt, hl⟩
          push Not at this
          obtain ⟨i, hi⟩ := this
          exact Or.inr ⟨i, mem_univ i, hall i, hi⟩
        · exact Or.inl ⟨hall, not_lt.1 hlt⟩
    _ ≤ _ := Pr_or_le _ _
    _ ≤ ((1 - q) ^ Fmax + ∑ i, e i) + ∑ i, e' i :=
        add_le_add hmiss ((Pr_exists_mem_le_sum univ _).trans (sum_le_sum fun i _ => hleak z hz i))
    _ ≤ δ + ∑ i, e i + ∑ i, e' i := by linarith

end NetTiming
