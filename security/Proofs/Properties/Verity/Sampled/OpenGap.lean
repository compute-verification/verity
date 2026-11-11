module

public import Proofs.Properties.Verity.Sampled.OpenEmpty

/-!
# An opening's gap at one hm96 leaf

An opening's node writes the live coins' release `bits u` into the tape as a round of its class at `bits u`, picked by
`pickIn` (`writeOpening`), and `openGap` is how far the written round falls short of uniform over the rounds the node
flips among (`shortfall`). This file bounds it at one leaf, averaged over the device's sample of the leaf's round, beside
`openEmpty`'s bound.

* **One class** (`prCoin_pickIn_ge`, `deficit_le`, `shortfall_pickIn_le`): `pickIn` picks the class member at the coin's
  index modulo the class's size, so it is uniform on the class only up to rounding. The shortfall, plus the live
  coins' probability of an over-weighted release (one the class releases more often than the live coins do), is at
  most the class's probability of an over-weighted release, plus the probability that the live bits are an independent
  member's release.
* **Over the leaf's sample** (`avg_fiber`, `Overweight`, `avg_shortfall_fiber_le`): averaged over a uniform sample `d₀`, a
  class with the sampling design of releases of `d₀`'s fiber of the leaf's hash gives the bound with the fibers' over-weighted
  releases.
* **At an hm96 leaf** (`release_swap_le`, `avg_shortfall_leaf_le`, `prCoin_diag`): at a leaf `H(x(ω) + M·y, c(y))`, T1
  swaps the sample's own release for the live bits within `2·δ₁` in an event that reads the leaf and the release, so the
  over-weighted terms cancel within `2·δ₁`. What remains is the collision term, the probability that two independent
  live coins' bits agree: the shortfall averaged over the sample is at most `2·δ₁` plus it.
* **At an opening's node** (`OpeningAt.avg_openable_eq`, `openGap_avg_le`): where every sample `d₀` of one leaf describes
  the node, the node's rounds have the sampling design of `d₀`'s fiber, so the bound holds for `openGap`.
* **At a round's release** (`roundBits_injective`, `prCoin_roundBits_eq`, `openGap_avg_le_round`): the bits of a
  round's `m` live challenges collide with probability `2^-(128·m)`, so `openGap` averaged over the sample is at most
  `2·δ₁ + 2^-(128·m)`.
-/

@[expose] public section

namespace Verity.Game

open scoped ENNReal

/-! ## One class -/

/-- **`pickIn S` picks each member of `S` for at least `⌊|C| / |S|⌋` of the `|C|` coins**: it picks the member at the
coin's index modulo `|S|`. -/
theorem prCoin_pickIn_ge {C : Type} [Fintype C] {S : Finset C} {e : C} (he : e ∈ S) :
    ((Fintype.card C / S.card : ℕ) : ℝ≥0∞) / Fintype.card C ≤ prCoin fun d => pickIn S d = e := by
  classical
  rw [prCoin_eq_card]
  refine ENNReal.div_le_div_right (Nat.cast_le.2 ?_) _
  have hne : S.Nonempty := ⟨e, he⟩
  have hk : 0 < S.card := Finset.card_pos.2 hne
  have hj : (S.equivFin ⟨e, he⟩).1 < S.card := (S.equivFin ⟨e, he⟩).2
  have hpick : ∀ (m : ℕ) (hm : m < Fintype.card C), m % S.card = (S.equivFin ⟨e, he⟩).1 →
      pickIn S ((Fintype.equivFin C).symm ⟨m, hm⟩) = e := by
    intro m hm hmj
    unfold pickIn
    rw [dite_eq_left hne]
    have hfin : ∀ (a : ℕ) (ha : a < S.card), a = (S.equivFin ⟨e, he⟩).1 → (S.equivFin.symm ⟨a, ha⟩).1 = e := by
      rintro a ha rfl
      simp
    exact hfin _ _ (by simpa using hmj)
  have hlt : ∀ t : Fin (Fintype.card C / S.card), t.1 * S.card + (S.equivFin ⟨e, he⟩).1 < Fintype.card C := by
    intro t
    have h1 : t.1 * S.card + (S.equivFin ⟨e, he⟩).1 < (t.1 + 1) * S.card := by
      rw [Nat.add_mul, Nat.one_mul]
      omega
    exact Nat.lt_of_lt_of_le h1 (le_trans (Nat.mul_le_mul_right _ t.2) (Nat.div_mul_le_self _ _))
  let φ : Fin (Fintype.card C / S.card) → {d // pickIn S d = e} := fun t =>
    ⟨(Fintype.equivFin C).symm ⟨t.1 * S.card + (S.equivFin ⟨e, he⟩).1, hlt t⟩, hpick _ _ (by
      rw [Nat.mul_comm, Nat.mul_add_mod, Nat.mod_eq_of_lt hj])⟩
  have hinj : Function.Injective φ := by
    intro a b hab
    have h1 := congrArg (fun x : {d // pickIn S d = e} => ((Fintype.equivFin C) x.1).1) hab
    simp only [φ, Equiv.apply_symm_apply] at h1
    exact Fin.ext (Nat.eq_of_mul_eq_mul_right hk (by omega))
  calc Fintype.card C / S.card = Fintype.card (Fin (Fintype.card C / S.card)) := (Fintype.card_fin _).symm
    _ ≤ Fintype.card {d // pickIn S d = e} := Fintype.card_le_of_injective φ hinj
    _ = (Finset.univ.filter fun d => pickIn S d = e).card := Fintype.card_subtype _

open Classical in
/-- **The written coin's deficit at one member `e`**: picking within the class `rel⁻¹(bits u)` of the live bits, the
written coin falls below `1/|C|` at `e` by at most `1/|C| - μ/k`, plus `μ/|C|`, where `μ` is the live coins'
probability of `e`'s release and `k` the size of its class (`prCoin_pickIn_ge`). -/
theorem deficit_le {C Q R : Type} [Fintype C] [Nonempty C] [Fintype Q] [Nonempty Q] [DecidableEq R]
    (rel : C → R) (bits : Q → R) (e : C) :
    (Fintype.card C : ℝ≥0∞)⁻¹ - (avg fun u : Q => avg fun d : C =>
        if pickIn (Finset.univ.filter fun d => rel d = bits u) d = e then 1 else 0)
      ≤ ((Fintype.card C : ℝ≥0∞)⁻¹ - prCoin (fun u : Q => bits u = rel e) *
          ((Finset.univ.filter fun d => rel d = rel e).card : ℝ≥0∞)⁻¹)
        + prCoin (fun u : Q => bits u = rel e) * (Fintype.card C : ℝ≥0∞)⁻¹ := by
  classical
  have he : e ∈ Finset.univ.filter fun d => rel d = rel e := by simp
  have hk : 0 < (Finset.univ.filter fun d => rel d = rel e).card := Finset.card_pos.2 ⟨e, he⟩
  have hk0 : ((Finset.univ.filter fun d => rel d = rel e).card : ℝ≥0∞) ≠ 0 := by exact_mod_cast hk.ne'
  have hkt : ((Finset.univ.filter fun d => rel d = rel e).card : ℝ≥0∞) ≠ ∞ := ENNReal.natCast_ne_top _
  have hN0 : (Fintype.card C : ℝ≥0∞) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
  have hNt : (Fintype.card C : ℝ≥0∞) ≠ ∞ := ENNReal.natCast_ne_top _
  have hhit : prCoin (fun u : Q => bits u = rel e) *
        prCoin (fun d => pickIn (Finset.univ.filter fun d => rel d = rel e) d = e) ≤
      avg fun u : Q => avg fun d : C =>
        if pickIn (Finset.univ.filter fun d => rel d = bits u) d = e then 1 else 0 := by
    rw [prCoin, prCoin, ← avg_mul_const]
    refine avg_mono fun u => ?_
    by_cases h : bits u = rel e
    · rw [ite_eq_left h, one_mul, h]
    · rw [ite_eq_right h, zero_mul]
      exact zero_le
  have hNk : Fintype.card C ≤ (Finset.univ.filter fun d => rel d = rel e).card *
      (Fintype.card C / (Finset.univ.filter fun d => rel d = rel e).card + 1) := (Nat.lt_mul_div_succ _ hk).le
  have hinv : ((Finset.univ.filter fun d => rel d = rel e).card : ℝ≥0∞)⁻¹ ≤
      ((Fintype.card C / (Finset.univ.filter fun d => rel d = rel e).card : ℕ) : ℝ≥0∞) / Fintype.card C +
        (Fintype.card C : ℝ≥0∞)⁻¹ := by
    calc ((Finset.univ.filter fun d => rel d = rel e).card : ℝ≥0∞)⁻¹
        = ((Finset.univ.filter fun d => rel d = rel e).card : ℝ≥0∞)⁻¹ * Fintype.card C * (Fintype.card C : ℝ≥0∞)⁻¹ := by
          rw [mul_assoc, ENNReal.mul_inv_cancel hN0 hNt, mul_one]
      _ ≤ ((Finset.univ.filter fun d => rel d = rel e).card : ℝ≥0∞)⁻¹ *
            ((Finset.univ.filter fun d => rel d = rel e).card *
              (((Fintype.card C / (Finset.univ.filter fun d => rel d = rel e).card : ℕ) : ℝ≥0∞) + 1)) *
            (Fintype.card C : ℝ≥0∞)⁻¹ := by
          gcongr
          exact_mod_cast hNk
      _ = (((Fintype.card C / (Finset.univ.filter fun d => rel d = rel e).card : ℕ) : ℝ≥0∞) + 1) *
            (Fintype.card C : ℝ≥0∞)⁻¹ := by
          rw [← mul_assoc, ENNReal.inv_mul_cancel hk0 hkt, one_mul]
      _ = _ := by rw [add_mul, one_mul, div_eq_mul_inv]
  have hp := prCoin_pickIn_ge (C := C) he
  have hw : prCoin (fun u : Q => bits u = rel e) * ((Finset.univ.filter fun d => rel d = rel e).card : ℝ≥0∞)⁻¹ ≤
      (avg fun u : Q => avg fun d : C =>
        if pickIn (Finset.univ.filter fun d => rel d = bits u) d = e then 1 else 0) +
        prCoin (fun u : Q => bits u = rel e) * (Fintype.card C : ℝ≥0∞)⁻¹ := by
    calc prCoin (fun u : Q => bits u = rel e) * ((Finset.univ.filter fun d => rel d = rel e).card : ℝ≥0∞)⁻¹
        ≤ prCoin (fun u : Q => bits u = rel e) *
            (((Fintype.card C / (Finset.univ.filter fun d => rel d = rel e).card : ℕ) : ℝ≥0∞) / Fintype.card C +
              (Fintype.card C : ℝ≥0∞)⁻¹) := by gcongr
      _ = prCoin (fun u : Q => bits u = rel e) *
            (((Fintype.card C / (Finset.univ.filter fun d => rel d = rel e).card : ℕ) : ℝ≥0∞) / Fintype.card C) +
            prCoin (fun u : Q => bits u = rel e) * (Fintype.card C : ℝ≥0∞)⁻¹ := mul_add _ _ _
      _ ≤ prCoin (fun u : Q => bits u = rel e) *
            prCoin (fun d => pickIn (Finset.univ.filter fun d => rel d = rel e) d = e) +
            prCoin (fun u : Q => bits u = rel e) * (Fintype.card C : ℝ≥0∞)⁻¹ := by gcongr
      _ ≤ _ := by gcongr
  rw [tsub_le_iff_right]
  calc (Fintype.card C : ℝ≥0∞)⁻¹
      ≤ ((Fintype.card C : ℝ≥0∞)⁻¹ - prCoin (fun u : Q => bits u = rel e) *
            ((Finset.univ.filter fun d => rel d = rel e).card : ℝ≥0∞)⁻¹) +
          prCoin (fun u : Q => bits u = rel e) * ((Finset.univ.filter fun d => rel d = rel e).card : ℝ≥0∞)⁻¹ :=
        le_tsub_add
    _ ≤ _ := by
        rw [add_assoc]
        gcongr
        rw [add_comm]
        exact hw

open Classical in
/-- **A class's written coin falls short of uniform by at most its over-weighted releases' excess, plus a collision
term.** With `A` the releases the class releases more often than the live coins do (`hA`), the shortfall of `pickIn`
within the class at the live bits, plus the live coins' probability of a release in `A`, is at most the class's
probability of a release in `A`, plus the probability that the live bits are an independent member's release
(`deficit_le`, summed over the members). -/
theorem shortfall_pickIn_le {C Q R : Type} [Fintype C] [Nonempty C] [Fintype Q] [Nonempty Q] [DecidableEq R]
    (rel : C → R) (bits : Q → R) (A : R → Prop)
    (hA : ∀ r, A r ↔ prCoin (fun u : Q => bits u = r) < prCoin fun d : C => rel d = r) :
    shortfall (fun u : Q => pickIn (Finset.univ.filter fun d => rel d = bits u)) + prCoin (fun u : Q => A (bits u))
      ≤ prCoin (fun d : C => A (rel d)) + prCoin fun p : C × Q => bits p.2 = rel p.1 := by
  classical
  have hν : ∀ r, prCoin (fun d : C => rel d = r) =
      ((Finset.univ.filter fun d => rel d = r).card : ℝ≥0∞) / Fintype.card C := fun r => prCoin_eq_card _
  have hk0 : ∀ e : C, ((Finset.univ.filter fun d => rel d = rel e).card : ℝ≥0∞) ≠ 0 := fun e => by
    exact_mod_cast (Finset.card_pos.2 ⟨e, by simp⟩).ne'
  have hkN : ∀ e : C, ((Finset.univ.filter fun d => rel d = rel e).card : ℝ≥0∞) / Fintype.card C *
      ((Finset.univ.filter fun d => rel d = rel e).card : ℝ≥0∞)⁻¹ = (Fintype.card C : ℝ≥0∞)⁻¹ := fun e => by
    rw [div_eq_mul_inv, mul_right_comm, ENNReal.mul_inv_cancel (hk0 e) (ENNReal.natCast_ne_top _), one_mul]
  have h1 : shortfall (fun u : Q => pickIn (Finset.univ.filter fun d => rel d = bits u)) ≤
      ∑ e : C, (((Fintype.card C : ℝ≥0∞)⁻¹ - prCoin (fun u : Q => bits u = rel e) *
          ((Finset.univ.filter fun d => rel d = rel e).card : ℝ≥0∞)⁻¹)
        + prCoin (fun u : Q => bits u = rel e) * (Fintype.card C : ℝ≥0∞)⁻¹) :=
    Finset.sum_le_sum fun e _ => deficit_le rel bits e
  have hu : ∀ u : Q, (if A (bits u) then (1 : ℝ≥0∞) else 0) ≤ ∑ e : C, if bits u = rel e then
      (if A (rel e) then ((Finset.univ.filter fun d => rel d = rel e).card : ℝ≥0∞)⁻¹ else 0) else 0 := by
    intro u
    by_cases hAu : A (bits u)
    · rw [ite_eq_left hAu]
      have hpos : 0 < (Finset.univ.filter fun d => rel d = bits u).card := by
        rcases Nat.eq_zero_or_pos (Finset.univ.filter fun d => rel d = bits u).card with h0 | h0
        · have h := (hA (bits u)).1 hAu
          rw [hν, h0, Nat.cast_zero, ENNReal.zero_div] at h
          exact absurd h (ENNReal.not_lt_zero)
        · exact h0
      have hc0 : ((Finset.univ.filter fun d => rel d = bits u).card : ℝ≥0∞) ≠ 0 := by exact_mod_cast hpos.ne'
      calc (1 : ℝ≥0∞) = ((Finset.univ.filter fun d => rel d = bits u).card : ℝ≥0∞) *
            ((Finset.univ.filter fun d => rel d = bits u).card : ℝ≥0∞)⁻¹ :=
            (ENNReal.mul_inv_cancel hc0 (ENNReal.natCast_ne_top _)).symm
        _ = ∑ e : C, if rel e = bits u then ((Finset.univ.filter fun d => rel d = bits u).card : ℝ≥0∞)⁻¹ else 0 := by
            rw [Finset.sum_ite, Finset.sum_const_zero, add_zero, Finset.sum_const, nsmul_eq_mul]
        _ ≤ _ := by
            refine Finset.sum_le_sum fun e _ => ?_
            by_cases h : rel e = bits u
            · rw [ite_eq_left h, ite_eq_left h.symm, h, ite_eq_left hAu]
            · rw [ite_eq_right h]
              exact zero_le
    · rw [ite_eq_right hAu]
      exact zero_le
  have h2 : prCoin (fun u : Q => A (bits u)) ≤ ∑ e : C, if A (rel e) then prCoin (fun u : Q => bits u = rel e) *
      ((Finset.univ.filter fun d => rel d = rel e).card : ℝ≥0∞)⁻¹ else 0 := by
    calc prCoin (fun u : Q => A (bits u)) = avg fun u : Q => if A (bits u) then (1 : ℝ≥0∞) else 0 := rfl
      _ ≤ avg fun u : Q => ∑ e : C, if bits u = rel e then
            (if A (rel e) then ((Finset.univ.filter fun d => rel d = rel e).card : ℝ≥0∞)⁻¹ else 0) else 0 :=
          avg_mono hu
      _ = ∑ e : C, avg fun u : Q => if bits u = rel e then
            (if A (rel e) then ((Finset.univ.filter fun d => rel d = rel e).card : ℝ≥0∞)⁻¹ else 0) else 0 :=
          avg_sum_comm _
      _ = _ := by
          refine Finset.sum_congr rfl fun e _ => ?_
          by_cases hAe : A (rel e)
          · rw [ite_eq_left hAe, ite_eq_left hAe, prCoin, ← avg_mul_const]
            refine congrArg (@avg Q _) (funext fun u => ?_)
            by_cases h : bits u = rel e
            · rw [ite_eq_left h, ite_eq_left h, one_mul]
            · rw [ite_eq_right h, ite_eq_right h, zero_mul]
          · rw [ite_eq_right hAe, ite_eq_right hAe]
            simp only [ite_self]
            exact avg_const 0
  have h3 : ∀ e : C, ((Fintype.card C : ℝ≥0∞)⁻¹ - prCoin (fun u : Q => bits u = rel e) *
        ((Finset.univ.filter fun d => rel d = rel e).card : ℝ≥0∞)⁻¹) +
      (if A (rel e) then prCoin (fun u : Q => bits u = rel e) *
        ((Finset.univ.filter fun d => rel d = rel e).card : ℝ≥0∞)⁻¹ else 0)
      ≤ if A (rel e) then (Fintype.card C : ℝ≥0∞)⁻¹ else 0 := by
    intro e
    by_cases hAe : A (rel e)
    · rw [ite_eq_left hAe, ite_eq_left hAe]
      have hlt := (hA (rel e)).1 hAe
      rw [hν] at hlt
      have := hlt.le
      have hle : prCoin (fun u : Q => bits u = rel e) * ((Finset.univ.filter fun d => rel d = rel e).card : ℝ≥0∞)⁻¹
          ≤ (Fintype.card C : ℝ≥0∞)⁻¹ := by
        rw [← hkN e]
        gcongr
      exact (tsub_add_cancel_of_le hle).le
    · rw [ite_eq_right hAe, ite_eq_right hAe, add_zero]
      have hge := mt (hA (rel e)).2 hAe
      rw [hν, not_lt] at hge
      have hle : (Fintype.card C : ℝ≥0∞)⁻¹ ≤ prCoin (fun u : Q => bits u = rel e) *
          ((Finset.univ.filter fun d => rel d = rel e).card : ℝ≥0∞)⁻¹ := by
        rw [← hkN e]
        gcongr
      exact (tsub_eq_zero_of_le hle).le
  calc shortfall (fun u : Q => pickIn (Finset.univ.filter fun d => rel d = bits u)) + prCoin (fun u : Q => A (bits u))
      ≤ (∑ e : C, (((Fintype.card C : ℝ≥0∞)⁻¹ - prCoin (fun u : Q => bits u = rel e) *
            ((Finset.univ.filter fun d => rel d = rel e).card : ℝ≥0∞)⁻¹)
          + prCoin (fun u : Q => bits u = rel e) * (Fintype.card C : ℝ≥0∞)⁻¹)) +
        ∑ e : C, (if A (rel e) then prCoin (fun u : Q => bits u = rel e) *
          ((Finset.univ.filter fun d => rel d = rel e).card : ℝ≥0∞)⁻¹ else 0) := add_le_add h1 h2
    _ = (∑ e : C, (((Fintype.card C : ℝ≥0∞)⁻¹ - prCoin (fun u : Q => bits u = rel e) *
            ((Finset.univ.filter fun d => rel d = rel e).card : ℝ≥0∞)⁻¹) +
          (if A (rel e) then prCoin (fun u : Q => bits u = rel e) *
            ((Finset.univ.filter fun d => rel d = rel e).card : ℝ≥0∞)⁻¹ else 0))) +
        ∑ e : C, prCoin (fun u : Q => bits u = rel e) * (Fintype.card C : ℝ≥0∞)⁻¹ := by
        simp only [Finset.sum_add_distrib]
        ring
    _ ≤ (∑ e : C, if A (rel e) then (Fintype.card C : ℝ≥0∞)⁻¹ else 0) +
        ∑ e : C, prCoin (fun u : Q => bits u = rel e) * (Fintype.card C : ℝ≥0∞)⁻¹ := by
        gcongr with e
        exact h3 e
    _ = prCoin (fun d : C => A (rel d)) + prCoin fun p : C × Q => bits p.2 = rel p.1 := by
        congr 1
        · show _ = ∑ e : C, (Fintype.card C : ℝ≥0∞)⁻¹ * (if A (rel e) then 1 else 0)
          refine Finset.sum_congr rfl fun e _ => ?_
          by_cases h : A (rel e)
          · rw [ite_eq_left h, ite_eq_left h, mul_one]
          · rw [ite_eq_right h, ite_eq_right h, mul_zero]
        · rw [FlockSoundness.ZK.prCoin_prod_split (fun (d : C) (u : Q) => bits u = rel d)]
          exact Finset.sum_congr rfl fun e _ => mul_comm _ _

/-- An average reads only the type's elements: it doesn't depend on the `Fintype` instance. -/
theorem avg_congr_inst {C : Type} (i₁ i₂ : Fintype C) (f : C → ℝ≥0∞) : @avg C i₁ f = @avg C i₂ f := by
  rw [Subsingleton.elim i₁ i₂]

/-! ## Over the leaf's sample -/

/-- **Resampling within the fiber**: a uniform `d₀`, then a uniform `d` with `d₀`'s value `L d₀`, is a uniform `d`. -/
theorem avg_fiber {D Dg : Type} [Fintype D] [DecidableEq Dg] (L : D → Dg) (g : Dg → D → ℝ≥0∞) :
    avg (fun d₀ => avg fun d : {d // L d = L d₀} => g (L d₀) d.1) = avg fun d => g (L d) d := by
  classical
  have hin : ∀ d₀ : D, (avg fun d : {d // L d = L d₀} => g (L d₀) d.1) =
      ∑ d : D, if L d = L d₀ then ((Finset.univ.filter fun d' => L d' = L d₀).card : ℝ≥0∞)⁻¹ * g (L d₀) d else 0 := by
    intro d₀
    unfold avg
    rw [Fintype.card_subtype, ← Finset.sum_filter]
    exact (Finset.sum_subtype (Finset.univ.filter fun d => L d = L d₀) (fun d => by simp)
      (fun d => ((Finset.univ.filter fun d' => L d' = L d₀).card : ℝ≥0∞)⁻¹ * g (L d₀) d)).symm
  show ∑ d₀ : D, (Fintype.card D : ℝ≥0∞)⁻¹ * (avg fun d : {d // L d = L d₀} => g (L d₀) d.1) =
    ∑ d : D, (Fintype.card D : ℝ≥0∞)⁻¹ * g (L d) d
  simp only [hin, Finset.mul_sum]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun d _ => ?_
  have hK0 : ((Finset.univ.filter fun d' => L d' = L d).card : ℝ≥0∞) ≠ 0 := by
    exact_mod_cast (Finset.card_pos.2 ⟨d, by simp⟩).ne'
  calc ∑ d₀ : D, (Fintype.card D : ℝ≥0∞)⁻¹ * (if L d = L d₀ then
          ((Finset.univ.filter fun d' => L d' = L d₀).card : ℝ≥0∞)⁻¹ * g (L d₀) d else 0)
      = ∑ d₀ : D, if L d₀ = L d then (Fintype.card D : ℝ≥0∞)⁻¹ *
          (((Finset.univ.filter fun d' => L d' = L d).card : ℝ≥0∞)⁻¹ * g (L d) d) else 0 := by
        refine Finset.sum_congr rfl fun d₀ _ => ?_
        by_cases h : L d = L d₀
        · rw [ite_eq_left h, ite_eq_left h.symm, h]
        · rw [ite_eq_right h, ite_eq_right (Ne.symm h), mul_zero]
    _ = ((Finset.univ.filter fun d' => L d' = L d).card : ℝ≥0∞) * ((Fintype.card D : ℝ≥0∞)⁻¹ *
          (((Finset.univ.filter fun d' => L d' = L d).card : ℝ≥0∞)⁻¹ * g (L d) d)) := by
        rw [Finset.sum_ite, Finset.sum_const_zero, add_zero, Finset.sum_const, nsmul_eq_mul]
    _ = (Fintype.card D : ℝ≥0∞)⁻¹ * g (L d) d := by
        rw [mul_left_comm, ← mul_assoc ((Finset.univ.filter fun d' => L d' = L d).card : ℝ≥0∞),
          ENNReal.mul_inv_cancel hK0 (ENNReal.natCast_ne_top _), one_mul]

/-- **The fiber of `ℓ` releases `r` more often than the live coins do**: a uniform member `d` of `L`'s fiber at `ℓ` has
the release `rel d = r` with a higher probability than the live bits are `r`. -/
def Overweight {D Dg Q R : Type} [Fintype D] [DecidableEq Dg] [Fintype Q] (L : D → Dg) (rel : D → R)
    (bits : Q → R) (ℓ : Dg) (r : R) : Prop :=
  prCoin (fun u : Q => bits u = r) < prCoin fun d : {d // L d = ℓ} => rel d.1 = r

open Classical in
/-- **The written coin's shortfall, averaged over the sample**: where the class at each sample `d₀` has the sampling design of releases
of `d₀`'s fiber of `L` (`hG`), its shortfall `G d₀` averaged over the sample, plus the probability that the live bits are
over-weighted at the sample's fiber, is at most the probability that the sample's own release is, plus the probability that
the live bits are the sample's release (`shortfall_pickIn_le` in each fiber, `avg_fiber`). -/
theorem avg_shortfall_fiber_le {D Dg Q R : Type} [Fintype D] [DecidableEq Dg] [Fintype Q] [Nonempty Q]
    [DecidableEq R] (L : D → Dg) (rel : D → R) (bits : Q → R) (G : D → ℝ≥0∞)
    (hG : ∀ d₀, ∃ (C : Type) (_ : Fintype C) (_ : Nonempty C) (rel' : C → R),
      (∀ φ : R → ℝ≥0∞, avg (fun c => φ (rel' c)) = avg fun d : {d // L d = L d₀} => φ (rel d.1)) ∧
      G d₀ = shortfall fun u : Q => pickIn (Finset.univ.filter fun c => rel' c = bits u)) :
    avg G + prCoin (fun p : D × Q => Overweight L rel bits (L p.1) (bits p.2))
      ≤ prCoin (fun d : D => Overweight L rel bits (L d) (rel d)) + prCoin fun p : D × Q => bits p.2 = rel p.1 := by
  classical
  have hper : ∀ d₀, G d₀ + prCoin (fun u : Q => Overweight L rel bits (L d₀) (bits u))
      ≤ prCoin (fun d : {d // L d = L d₀} => Overweight L rel bits (L d₀) (rel d.1))
        + avg fun d : {d // L d = L d₀} => prCoin fun u : Q => bits u = rel d.1 := by
    intro d₀
    obtain ⟨C, iC, nC, rel', hlaw, hGd⟩ := hG d₀
    have key := shortfall_pickIn_le rel' bits (Overweight L rel bits (L d₀)) fun r => by
      unfold Overweight
      rw [show (prCoin fun c : C => rel' c = r) = prCoin fun d : {d // L d = L d₀} => rel d.1 = r from
        hlaw fun r' => @ite _ (r' = r) (Classical.propDecidable _) 1 0]
    rw [hGd]
    refine key.trans_eq ?_
    congr 1
    · exact hlaw fun r => @ite _ (Overweight L rel bits (L d₀) r) (Classical.propDecidable _) 1 0
    · rw [FlockSoundness.ZK.prCoin_prod_split (fun (c : C) (u : Q) => bits u = rel' c)]
      exact hlaw fun r => prCoin fun u : Q => bits u = r
  have hsum := avg_mono hper
  rw [avg_add, avg_add, ← FlockSoundness.ZK.prCoin_prod_split
    (fun (d₀ : D) (u : Q) => Overweight L rel bits (L d₀) (bits u))] at hsum
  refine hsum.trans_eq ?_
  rw [FlockSoundness.ZK.prCoin_prod_split (fun (d : D) (u : Q) => bits u = rel d)]
  exact congrArg₂ (· + ·)
    (avg_fiber L fun ℓ d => @ite _ (Overweight L rel bits ℓ (rel d)) (Classical.propDecidable _) 1 0)
    (avg_fiber L fun _ d => prCoin fun u : Q => bits u = rel d)

/-! ## At an hm96 leaf -/

/-- **An event of the leaf and the release swaps the sample's release for the live bits within `2·δ₁`** at one hm96
leaf `H(x(ω) + M·y, c(y))`: T1 makes the leaf ideal, where the event no longer reads the coins `ω` and the release
`P ω` may be swapped for the live bits, whose sampling design it has (`hlaw`); T1 on the complement makes the leaf real again. -/
theorem release_swap_le {Ω Q Salt B C D R : Type} [Fintype Ω] [Nonempty Ω] [Fintype Q] [Nonempty Q] [Fintype Salt]
    [Nonempty Salt] [AddCommGroup B] [Fintype B] (My : Salt → B) (c : Salt → C) {δ₁ : ℝ≥0∞}
    (hT1 : Claims.Hm96Hiding My c δ₁) (H : B × C → D) (x : Ω → B) (P : Ω → R) (bits : Q → R)
    (hlaw : ∀ g : R → ℝ≥0∞, avg (fun ω => g (P ω)) = avg fun u => g (bits u)) (A : D → R → Prop) :
    prCoin (fun p : (Ω × Q) × Salt => A (H (x p.1.1 + My p.2, c p.2)) (P p.1.1))
      ≤ prCoin (fun p : (Ω × Q) × Salt => A (H (x p.1.1 + My p.2, c p.2)) (bits p.1.2)) + 2 * δ₁ := by
  have : Nonempty B := ⟨0⟩
  have h1 := FlockSoundness.ZK.ideal_leaf_swap My c hT1 H (fun w : Ω × Q => x w.1) (fun w ℓ => A ℓ (P w.1))
  have h2 := FlockSoundness.ZK.ideal_leaf_swap My c hT1 H (fun w : Ω × Q => x w.1) (fun w ℓ => ¬ A ℓ (bits w.2))
  have hl : avg (fun w : Ω × Q => prCoin fun u : B × Salt => A (H (u.1, c u.2)) (bits w.2))
      = avg fun q : Q => prCoin fun u : B × Salt => A (H (u.1, c u.2)) (bits q) :=
    (FlockSoundness.ZK.avg_prod_split _).trans (avg_const
      (avg fun q : Q => prCoin fun u : B × Salt => A (H (u.1, c u.2)) (bits q)))
  have hr : avg (fun w : Ω × Q => prCoin fun u : B × Salt => A (H (u.1, c u.2)) (P w.1))
      = avg fun ω : Ω => prCoin fun u : B × Salt => A (H (u.1, c u.2)) (P ω) :=
    (FlockSoundness.ZK.avg_prod_split _).trans (congrArg avg (funext fun ω => avg_const
      (prCoin fun u : B × Salt => A (H (u.1, c u.2)) (P ω))))
  have hswap : prCoin (fun p : (Ω × Q) × (B × Salt) => A (H (p.2.1, c p.2.2)) (P p.1.1))
      = prCoin (fun p : (Ω × Q) × (B × Salt) => A (H (p.2.1, c p.2.2)) (bits p.1.2)) := by
    rw [FlockSoundness.ZK.prCoin_prod_split (fun (w : Ω × Q) (u : B × Salt) => A (H (u.1, c u.2)) (P w.1)),
      FlockSoundness.ZK.prCoin_prod_split (fun (w : Ω × Q) (u : B × Salt) => A (H (u.1, c u.2)) (bits w.2))]
    exact hr.trans ((hlaw fun r => prCoin fun u : B × Salt => A (H (u.1, c u.2)) r).trans hl.symm)
  have hideal : prCoin (fun p : (Ω × Q) × (B × Salt) => A (H (p.2.1, c p.2.2)) (bits p.1.2))
      ≤ prCoin (fun p : (Ω × Q) × Salt => A (H (x p.1.1 + My p.2, c p.2)) (bits p.1.2)) + δ₁ := by
    have a := FlockSoundness.ZK.prCoin_add_not
      (fun p : (Ω × Q) × (B × Salt) => A (H (p.2.1, c p.2.2)) (bits p.1.2))
    have b := FlockSoundness.ZK.prCoin_add_not
      (fun p : (Ω × Q) × Salt => A (H (x p.1.1 + My p.2, c p.2)) (bits p.1.2))
    have hfin := FlockSoundness.ZK.prCoin_ne_top
      (fun p : (Ω × Q) × (B × Salt) => ¬ A (H (p.2.1, c p.2.2)) (bits p.1.2))
    refine (ENNReal.add_le_add_iff_right hfin).1 ?_
    calc _ = (1 : ℝ≥0∞) := a
      _ = _ := b.symm
      _ ≤ prCoin (fun p : (Ω × Q) × Salt => A (H (x p.1.1 + My p.2, c p.2)) (bits p.1.2)) +
          (prCoin (fun p : (Ω × Q) × (B × Salt) => ¬ A (H (p.2.1, c p.2.2)) (bits p.1.2)) + δ₁) := by
          gcongr
      _ = _ := by ring
  calc prCoin (fun p : (Ω × Q) × Salt => A (H (x p.1.1 + My p.2, c p.2)) (P p.1.1))
      ≤ prCoin (fun p : (Ω × Q) × (B × Salt) => A (H (p.2.1, c p.2.2)) (P p.1.1)) + δ₁ := h1
    _ = prCoin (fun p : (Ω × Q) × (B × Salt) => A (H (p.2.1, c p.2.2)) (bits p.1.2)) + δ₁ := by rw [hswap]
    _ ≤ (prCoin (fun p : (Ω × Q) × Salt => A (H (x p.1.1 + My p.2, c p.2)) (bits p.1.2)) + δ₁) + δ₁ := by
        gcongr
    _ = _ := by rw [add_assoc, ← two_mul]

/-- **The written coin's shortfall at an hm96 leaf is at most `2·δ₁` plus a collision term, averaged over the sample**:
the leaf `L(ω, y) = H(x(ω) + M·y, c(y))` (`hL`) of uniform coins `ω` and salt `y` releases `P ω` (`hrel`), and live
coins `u` sampled independently have bits `bits u` with the release's law (`hlaw`). Where the class at each sample `d₀` has
the sampling design of releases of `d₀`'s fiber (`hG`), its shortfall `G d₀` averaged over the sample is at most `2·δ₁` plus the
probability that two independent live coins' bits agree (`avg_shortfall_fiber_le`, with `release_swap_le` cancelling
the over-weighted terms). -/
theorem avg_shortfall_leaf_le {Ω Q Salt B C D R : Type} [Fintype Ω] [Nonempty Ω] [Fintype Q] [Nonempty Q]
    [Fintype Salt] [Nonempty Salt] [AddCommGroup B] [Fintype B] [DecidableEq D] [DecidableEq R]
    (My : Salt → B) (c : Salt → C) {δ₁ : ℝ≥0∞} (hT1 : Claims.Hm96Hiding My c δ₁) (H : B × C → D) (x : Ω → B)
    (P : Ω → R) (bits : Q → R) (hlaw : ∀ g : R → ℝ≥0∞, avg (fun ω => g (P ω)) = avg fun u => g (bits u))
    (L : Ω × Salt → D) (hL : ∀ ω y, L (ω, y) = H (x ω + My y, c y)) (rel : Ω × Salt → R)
    (hrel : ∀ ω y, rel (ω, y) = P ω) (G : Ω × Salt → ℝ≥0∞)
    (hG : ∀ d₀, ∃ (C' : Type) (_ : Fintype C') (_ : Nonempty C') (rel' : C' → R),
      (∀ φ : R → ℝ≥0∞, avg (fun c => φ (rel' c)) = avg fun d : {d // L d = L d₀} => φ (rel d.1)) ∧
      G d₀ = shortfall fun u : Q => pickIn (Finset.univ.filter fun c => rel' c = bits u)) :
    avg G ≤ 2 * δ₁ + prCoin fun p : Q × Q => bits p.1 = bits p.2 := by
  have key := avg_shortfall_fiber_le L rel bits G hG
  let e : (Ω × Q) × Salt ≃ (Ω × Salt) × Q :=
    ⟨fun q => ((q.1.1, q.2), q.1.2), fun p => ((p.1.1, p.2), p.1.2), fun _ => rfl, fun _ => rfl⟩
  have hsw := release_swap_le My c hT1 H x P bits hlaw fun ℓ r' => Overweight L rel bits ℓ r'
  have hY : prCoin (fun d : Ω × Salt => Overweight L rel bits (L d) (rel d)) ≤
      prCoin (fun p : (Ω × Salt) × Q => Overweight L rel bits (L p.1) (bits p.2)) + 2 * δ₁ := by
    calc prCoin (fun d : Ω × Salt => Overweight L rel bits (L d) (rel d))
        = prCoin (fun p : (Ω × Salt) × Q => Overweight L rel bits (L p.1) (rel p.1)) :=
          (FlockSoundness.ZK.prCoin_fst _).symm
      _ = prCoin (fun q : (Ω × Q) × Salt => Overweight L rel bits (L (e q).1) (rel (e q).1)) :=
          (FlockSoundness.ZK.prCoin_congr_equiv e fun p : (Ω × Salt) × Q =>
            Overweight L rel bits (L p.1) (rel p.1)).symm
      _ = prCoin (fun q : (Ω × Q) × Salt => Overweight L rel bits (H (x q.1.1 + My q.2, c q.2)) (P q.1.1)) := by
          congr 1
          funext q
          show Overweight L rel bits (L (q.1.1, q.2)) (rel (q.1.1, q.2)) = _
          rw [hL, hrel]
      _ ≤ prCoin (fun q : (Ω × Q) × Salt => Overweight L rel bits (H (x q.1.1 + My q.2, c q.2)) (bits q.1.2)) +
            2 * δ₁ := hsw
      _ = prCoin (fun q : (Ω × Q) × Salt => Overweight L rel bits (L (e q).1) (bits (e q).2)) + 2 * δ₁ := by
          congr 2
          funext q
          show _ = Overweight L rel bits (L (q.1.1, q.2)) (bits q.1.2)
          rw [hL]
      _ = prCoin (fun p : (Ω × Salt) × Q => Overweight L rel bits (L p.1) (bits p.2)) + 2 * δ₁ := by
          rw [FlockSoundness.ZK.prCoin_congr_equiv e fun p : (Ω × Salt) × Q =>
            Overweight L rel bits (L p.1) (bits p.2)]
  have hcoll : prCoin (fun p : (Ω × Salt) × Q => bits p.2 = rel p.1) =
      prCoin (fun p : Q × Q => bits p.1 = bits p.2) := by
    rw [FlockSoundness.ZK.prCoin_prod_split (fun (d : Ω × Salt) (u : Q) => bits u = rel d),
      FlockSoundness.ZK.prCoin_prod_split (fun (u' : Q) (u : Q) => bits u' = bits u)]
    calc avg (fun d : Ω × Salt => prCoin fun u : Q => bits u = rel d)
        = avg (fun ω : Ω => avg fun y : Salt => prCoin fun u : Q => bits u = rel (ω, y)) :=
          FlockSoundness.ZK.avg_prod_split _
      _ = avg (fun ω : Ω => prCoin fun u : Q => bits u = P ω) :=
          congrArg (@avg Ω _) (funext fun ω => by
            simp only [hrel]
            exact avg_const _)
      _ = avg (fun u' : Q => prCoin fun u : Q => bits u = bits u') := hlaw fun r' => prCoin fun u : Q => bits u = r'
      _ = avg (fun u' : Q => prCoin fun u : Q => bits u' = bits u) :=
          congrArg (@avg Q _) (funext fun u' => congrArg (@prCoin Q _) (funext fun u => propext eq_comm))
  have hX : prCoin (fun p : (Ω × Salt) × Q => Overweight L rel bits (L p.1) (bits p.2)) ≠ ∞ :=
    FlockSoundness.ZK.prCoin_ne_top _
  refine (ENNReal.add_le_add_iff_right hX).1 ?_
  calc avg G + prCoin (fun p : (Ω × Salt) × Q => Overweight L rel bits (L p.1) (bits p.2))
      ≤ prCoin (fun d : Ω × Salt => Overweight L rel bits (L d) (rel d)) +
          prCoin (fun p : (Ω × Salt) × Q => bits p.2 = rel p.1) := key
    _ ≤ (prCoin (fun p : (Ω × Salt) × Q => Overweight L rel bits (L p.1) (bits p.2)) + 2 * δ₁) +
          prCoin (fun p : Q × Q => bits p.1 = bits p.2) := by
        rw [hcoll]
        exact add_le_add hY le_rfl
    _ = (2 * δ₁ + prCoin fun p : Q × Q => bits p.1 = bits p.2) +
          prCoin (fun p : (Ω × Salt) × Q => Overweight L rel bits (L p.1) (bits p.2)) := by ring

/-- **Two uniform coins agree with probability `1/|Q|`.** -/
theorem prCoin_diag {Q : Type} [Fintype Q] [Nonempty Q] :
    prCoin (fun p : Q × Q => p.1 = p.2) = (Fintype.card Q : ℝ≥0∞)⁻¹ := by
  classical
  rw [FlockSoundness.ZK.prCoin_prod_split (fun (u' u : Q) => u' = u)]
  have h : ∀ u' : Q, prCoin (fun u : Q => u' = u) = (Fintype.card Q : ℝ≥0∞)⁻¹ := fun u' => by
    rw [prCoin_eq_card, Finset.filter_eq, ite_eq_left (Finset.mem_univ u'), Finset.card_singleton, Nat.cast_one,
      one_div]
  exact (congrArg (@avg Q _) (funext h)).trans (avg_const _)

end Verity.Game

namespace Verity.Sampled

open Verity.Game
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

/-! ## The gap at an opening's node -/

namespace OpeningAt

variable {β : Type} {E : List Release → Game β} {g' : Game β}

/-- **A node's rounds have the sampling design of its leaf's fiber**: where the play ends in a close that records its log and `o`
describes the node at the sample `d₀`'s leaf, the releases of the rounds any description `o₀` flips among have the sampling design of
the releases of the samples with `d₀`'s leaf (`opened_eq`). -/
theorem avg_openable_eq {lg : β → List Release} (hE : RecordsLog lg E) (o₀ o : OpeningAt E g') {ν key : ByteArray}
    {s r n block : ℕ} {d₀ : TreeDraw block} (h : o.AtLeaf ν key s r n d₀) (φ : List Bool → ℝ≥0∞) :
    avg (fun d : o₀.Openable => φ (TreeDraw.opened d.1 o₀.n)) =
      avg fun d : {d : TreeDraw block // (TreeDraw.leaf ν key s r d).data = (TreeDraw.leaf ν key s r d₀).data} =>
        φ (TreeDraw.opened d.1 n) := by
  calc avg (fun d : o₀.Openable => φ (TreeDraw.opened d.1 o₀.n))
      = avg (fun d : o₀.Openable => φ (TreeDraw.opened (cast (openable_eq o₀ o) d).1 o.n)) :=
        congrArg (@avg o₀.Openable _) (funext fun d => congrArg φ (opened_eq hE o₀ o d).symm)
    _ = avg (fun d : o.Openable => φ (TreeDraw.opened d.1 o.n)) :=
        avg_equiv (Equiv.cast (openable_eq o₀ o)) fun d => φ (TreeDraw.opened d.1 o.n)
    _ = _ := by
        obtain ⟨rfl, rfl, rfl, rfl, rfl, rfl, hd⟩ := h
        obtain rfl := eq_of_heq hd
        exact avg_congr_inst _ _ _

end OpeningAt

/-- **An opening's gap is at most `2·δ₁` plus the live releases' collision probability, averaged over the device's
sample** at one leaf. Where the play ends in a close that records its log and, at each sample `d₀` of stream `s`'s round
`r`, the node `g' d₀` has a description at `d₀`'s leaf (`ho`), the written round's shortfall at live coins whose bits
`bits u` have the sampling design of a sample's first `n` coins (`hlaw`), averaged over the sample, is at most `2·δ₁` plus the
probability that two independent live coins' bits agree (`avg_shortfall_leaf_le`). -/
theorem openGap_avg_le {β : Type} {lg : β → List Release} {E : List Release → Game β} (hE : RecordsLog lg E)
    (ν key : ByteArray) (s r n : ℕ) {block : ℕ} (g' : TreeDraw block → Game β)
    (ho : ∀ d₀, ∃ o : OpeningAt E (g' d₀), o.AtLeaf ν key s r n d₀) {Q : Type} [Fintype Q] [Nonempty Q]
    (bits : Q → List Bool) {δ₁ : ℝ≥0∞} (hT1 : Claims.Hm96Hiding (treeMy key) (treeC ν) δ₁)
    (hlaw : ∀ G : List Bool → ℝ≥0∞,
      avg (fun ω : Fin block → Fin 128 → Bool => G (TreeDraw.opened (ω, fun _ => false) n)) = avg fun u => G (bits u)) :
    avg (fun d₀ : TreeDraw block => openGap E (g' d₀) bits) ≤
      2 * δ₁ + prCoin fun p : Q × Q => bits p.1 = bits p.2 := by
  refine avg_shortfall_leaf_le (treeMy key) (treeC ν) hT1 (fun p => (treeH ν key p).data) (treeX ν s r)
    (fun ω => TreeDraw.opened (ω, fun _ => false) n) bits hlaw (fun d => (TreeDraw.leaf ν key s r d).data)
    (fun ω y => congrArg ByteArray.data (leaf_eq_treeH ν key s r ω y)) (fun d => TreeDraw.opened d n)
    (fun _ _ => rfl) (fun d₀ => openGap E (g' d₀) bits) fun d₀ => ?_
  obtain ⟨o, h⟩ := ho d₀
  obtain ⟨o₀, h₀⟩ := openingOf_some o
  refine ⟨o₀.Openable, inferInstance, inferInstance, fun d => TreeDraw.opened d.1 o₀.n,
    OpeningAt.avg_openable_eq hE o₀ o h, ?_⟩
  unfold openGap
  rw [h₀]
  rfl

/-! ## The gap at a round's release -/

variable (e : 𝔽 ≃ (Fin 128 → Bool))

/-- **A round's bits determine its challenges.** -/
theorem roundBits_injective {m : ℕ} : Function.Injective (roundBits e : (Fin m → 𝔽) → List Bool) := by
  intro c c' h
  have hl : ∀ c : Fin m → 𝔽, (List.ofFn fun i => List.ofFn (e (c i))).map List.length = List.ofFn fun _ => 128 :=
    fun c => by
      rw [List.map_ofFn]
      exact congrArg List.ofFn (funext fun i => List.length_ofFn)
  have h' := List.eq_iff_flatten_eq.2 ⟨h, (hl c).trans (hl c').symm⟩
  funext i
  exact e.injective (List.ofFn_injective (congrFun (List.ofFn_injective h') i))

/-- **Two rounds' bits agree with probability `2^-(128·m)`**: those of `m` uniform live challenges each
(`roundBits_injective`, `prCoin_diag`). -/
theorem prCoin_roundBits_eq {m : ℕ} :
    prCoin (fun p : (Fin m → 𝔽) × (Fin m → 𝔽) => roundBits e p.1 = roundBits e p.2) =
      ((2 : ℝ≥0∞) ^ (128 * m))⁻¹ := by
  calc prCoin (fun p : (Fin m → 𝔽) × (Fin m → 𝔽) => roundBits e p.1 = roundBits e p.2)
      = prCoin (fun p : (Fin m → 𝔽) × (Fin m → 𝔽) => p.1 = p.2) :=
        congrArg (@prCoin ((Fin m → 𝔽) × (Fin m → 𝔽)) _) (funext fun p => propext (roundBits_injective e).eq_iff)
    _ = (Fintype.card (Fin m → 𝔽) : ℝ≥0∞)⁻¹ := prCoin_diag
    _ = ((2 : ℝ≥0∞) ^ (128 * m))⁻¹ := by
        rw [Fintype.card_fun, Fintype.card_congr e, Fintype.card_fun, Fintype.card_fin, Fintype.card_fin,
          Fintype.card_bool, ← pow_mul, Nat.cast_pow, Nat.cast_ofNat]

/-- **An opening's gap at a round's release is at most `2·δ₁ + 2^-(128·m)`, averaged over the device's sample**
(`openGap_avg_le` at `avg_opened_roundBits`, with `prCoin_roundBits_eq`'s collision): the release is a round's `m`
live challenges as bits, and the opening's count `m` fits in the block. -/
theorem openGap_avg_le_round {β : Type} {lg : β → List Release} {E : List Release → Game β}
    (hE : RecordsLog lg E) (ν key : ByteArray) (s r : ℕ) {block m : ℕ} (hm : m ≤ block)
    (g' : TreeDraw block → Game β) (ho : ∀ d₀, ∃ o : OpeningAt E (g' d₀), o.AtLeaf ν key s r m d₀)
    {δ₁ : ℝ≥0∞} (hT1 : Claims.Hm96Hiding (treeMy key) (treeC ν) δ₁) :
    avg (fun d₀ : TreeDraw block => openGap E (g' d₀) (roundBits e : (Fin m → 𝔽) → List Bool)) ≤
      2 * δ₁ + ((2 : ℝ≥0∞) ^ (128 * m))⁻¹ :=
  (openGap_avg_le hE ν key s r m g' ho (roundBits e) hT1 (avg_opened_roundBits e hm)).trans_eq
    (congrArg (2 * δ₁ + ·) (prCoin_roundBits_eq e))

end Verity.Sampled
