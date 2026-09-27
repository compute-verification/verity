import FlockSoundness.RepSound
import FlockSoundness.LinkSound

/-!
# The phases before Ligerito, over any list of messages

`zerocheck_phase_of_card`, `lincheck_phase_of_card`, `opening_phase_of_card` and
`link_sound_of_card` pay a union bound over the messages near the committed table
(`closeMsgs`) and use nothing else about that list. This file states them for any set `W` of
level-0 messages of at most `L` elements (`…_of_list`), with the doomed states over `W`
(`RepDoomedW`, `DZcW`, `DLcW`, `DOpW`). The padded level 0 (`SoundnessPad`) takes `W` to be the
witnesses its padded code's list packs.
-/

namespace FlockSoundness

open Game Model Accounting
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

/-- A rep starts doomed over `W`: no message of `W` packs a witness that satisfies the block
R1CS and the pin and makes every opened region claim true. -/
def RepDoomedW (A : Arith F K) (S : Statement) (sch : Schedule)
    (W : Set (Message F sch.k0 sch.level₀.logCols)) (extra : List (Claim F S.m)) : Prop :=
  ∀ M ∈ W, let z := witnessOf A S.m sch.k0 sch.level₀.logCols M
    ¬ ((∀ o i, (S.A₀.mulVec (S.block z o)) i * (S.B₀.mulVec (S.block z o)) i = S.block z o i)
        ∧ (∀ o, S.block z o S.pin = 1) ∧ ∀ c ∈ extra, c.Holds z)

/-- Doomed after the zerocheck, over `W`. -/
def DZcW (A : Arith F K) (S : Statement) (sch : Schedule)
    (W : Set (Message F sch.k0 sch.level₀.logCols)) (extra : List (Claim F S.m)) :
    Option (ZcOut F S.m) → Prop
  | none => True
  | some zc => ∀ M ∈ W,
      ¬ ((⟨zc.va, A.lagS zc.z, zc.ρ⟩ : Claim F S.m).Holds (aVec S (wit A S sch M))
          ∧ (⟨zc.vb, A.lagS zc.z, zc.ρ⟩ : Claim F S.m).Holds (bVec S (wit A S sch M))
          ∧ (cClaim A zc).Holds (wit A S sch M)
          ∧ pinMLE S (wit A S sch M) (zc.ρOut S) = 1
          ∧ ∀ e ∈ extra, e.Holds (wit A S sch M))

/-- Doomed after the lincheck, over `W`. -/
def DLcW (A : Arith F K) (S : Statement) (sch : Schedule)
    (W : Set (Message F sch.k0 sch.level₀.logCols)) (extra : List (Claim F S.m))
    (zc : ZcOut F S.m) : Option (Claim F S.m) → Prop
  | none => True
  | some ab => ∀ M ∈ W,
      ¬ (ab.Holds (wit A S sch M) ∧ (cClaim A zc).Holds (wit A S sch M)
          ∧ ∀ e ∈ extra, e.Holds (wit A S sch M))

/-- Doomed after the opening, over `W`: no message of `W` has the batched value. -/
def DOpW (S : Statement) (sch : Schedule) (W : Set (Message F sch.k0 sch.level₀.logCols)) :
    Option (OpenOut F) → Prop
  | none => True
  | some op => ∀ M ∈ W,
      (∑ y ∈ Finset.range (2 ^ (S.m - 7)), packedAt sch M y * op.b y) ≠ op.T

open Classical in
/-- **The zerocheck phase** over a list `W` of at most `L` messages: as `zerocheck_phase_of_card`. -/
theorem zerocheck_phase_of_list (A : Arith F K) (hA : A.Correct) (S : Statement) (sch : Schedule)
    (hm : 13 ≤ S.m) (W : Set (Message F sch.k0 sch.level₀.logCols)) (extra : List (Claim F S.m))
    (hdoom : RepDoomedW A S sch W extra) (L : ℕ)
    (hL : (Finset.univ.filter fun M => M ∈ W).card ≤ L) :
    value (fun o => ¬ DZcW A S sch W extra o) (zerocheck A S.m) ≤
      ((S.m - 13 : ℕ) : ℝ≥0∞) / Fintype.card F +
        L * (((S.m - 13 + 127 + 2 * (S.m - 6) + (S.m - S.kLog) : ℕ) : ℝ≥0∞) / Fintype.card F) := by
  classical
  have := hA.charTwo
  have h6 : 6 ≤ S.m := le_trans (by norm_num) hm
  set s := Finset.univ.filter fun M => M ∈ W
  let E : Message F sch.k0 sch.level₀.logCols → Option (ZcOut F S.m) → Prop := fun M o =>
    ∃ zc, o = some zc ∧ ((⟨zc.va, A.lagS zc.z, zc.ρ⟩ : Claim F S.m).Holds (aVec S (wit A S sch M))
      ∧ (⟨zc.vb, A.lagS zc.z, zc.ρ⟩ : Claim F S.m).Holds (bVec S (wit A S sch M))
      ∧ (cClaim A zc).Holds (wit A S sch M) ∧ pinMLE S (wit A S sch M) (zc.ρOut S) = 1
      ∧ ∀ e ∈ extra, e.Holds (wit A S sch M))
  let G : (Fin (S.m - 13) → F) → Prop := fun rs => ∃ j, rs j = 1
  let B3 : Message F sch.k0 sch.level₀.logCols → (Fin (S.m - 13) → F) → Prop := fun M rs =>
    R1CSFails S (wit A S sch M) ∧ (∀ e ∈ extra, e.Holds (wit A S sch M)) ∧
      ∀ s₀ : Fin 64, ∑ x : Fin (2 ^ (S.m - 6)), eqAt (zcVector A rs) x.val *
        ((errBit S (wit A S sch M) (addr64 h6 x s₀)).val : F) = 0
  set εr : ℝ≥0∞ := ((127 + 2 * (S.m - 6) + (S.m - S.kLog) : ℕ) : ℝ≥0∞) / Fintype.card F
  have hB3 : ∀ M, prCoin (B3 M) ≤ ((S.m - 13 : ℕ) : ℝ≥0∞) / Fintype.card F := by
    intro M
    by_cases hr : R1CSFails S (wit A S sch M)
    · exact (prCoin_mono fun rs h => h.2.2).trans (prCoin_zc3_le A hA S hm _ hr)
    · calc prCoin (B3 M) ≤ prCoin (fun _ : Fin (S.m - 13) → F => False) :=
            prCoin_mono fun rs h => hr h.1
        _ = 0 := by simp [prCoin, avg]
        _ ≤ _ := zero_le
  calc value (fun o => ¬ DZcW A S sch W extra o) (zerocheck A S.m)
      ≤ value (fun o => ∃ M ∈ s, E M o) (zerocheck A S.m) := by
        refine value_mono (fun o h => ?_) _
        cases o with
        | none => exact absurd trivial h
        | some zc =>
          simp only [DZcW, not_forall, not_not] at h
          obtain ⟨M, hM, hall⟩ := h
          exact ⟨M, by simp [s, hM], zc, rfl, hall⟩
    _ ≤ avg (fun rs : Fin (S.m - 13) → F =>
          (if G rs then 1 else 0) + ∑ M ∈ s, ((if B3 M rs then 1 else 0) + εr)) := by
        rw [zerocheck_eq]
        refine value_coin_le_of_forall _ _ _ _ fun _ => ?_
        refine value_coin_le_avg _ _ _ _ fun rs => ?_
        by_cases hG : G rs
        · simp only [hG, if_true]; exact (value_le_one _ _).trans le_self_add
        · simp only [hG, if_false, zero_add]
          refine (value_exists_finset_le _ _ _).trans (Finset.sum_le_sum fun M hM => ?_)
          by_cases hB : B3 M rs
          · simp only [hB, if_true]; exact (value_le_one _ _).trans le_self_add
          · simp only [hB, if_false, zero_add]
            by_cases hext : ∀ e ∈ extra, e.Holds (wit A S sch M)
            · have hMc : M ∈ W := by simpa [s] using hM
              have hdM := hdoom M hMc
              by_cases hr : R1CSFails S (wit A S sch M)
              · obtain ⟨s₀, hs₀⟩ : ∃ s₀ : Fin 64, ∑ x : Fin (2 ^ (S.m - 6)),
                    eqAt (zcVector A rs) x.val *
                      ((errBit S (wit A S sch M) (addr64 h6 x s₀)).val : F) ≠ 0 := by
                  by_contra hno
                  push Not at hno
                  exact hB ⟨hr, hext, hno⟩
                have hreq : ∀ j, zcVector A rs j ≠ 1 := by
                  intro j
                  simp only [zcVector]
                  split_ifs with hj
                  · exact hA.pinned_ne_one _
                  · exact fun h1 => hG ⟨_, h1⟩
                refine (value_mono (fun o ⟨zc, ho, hA', hB', hC', _, _⟩ =>
                    ⟨zc, ho, hA', hB', hC'⟩) _).trans
                  ((value_zcRest_r1cs_le A hA S hm _ rs hreq s₀ hs₀).trans ?_)
                exact ENNReal.div_le_div_right (by exact_mod_cast (by omega :
                  127 + 2 * (S.m - 6) ≤ 127 + 2 * (S.m - 6) + (S.m - S.kLog))) _
              · obtain ⟨o₀, ho₀⟩ : ∃ o₀, S.block (wit A S sch M) o₀ S.pin ≠ 1 := by
                  by_contra hno
                  push Not at hno
                  exact hdM ⟨r1cs_of_not_fails S _ hr, hno, hext⟩
                refine (value_mono (fun o ⟨zc, ho, _, _, _, hp, _⟩ => ⟨zc, ho, hp⟩) _).trans
                  ((value_zcRest_pin_le A hA S _ rs o₀ ho₀).trans ?_)
                exact ENNReal.div_le_div_right (by exact_mod_cast (by simp only [Statement.nbl]; omega :
                  S.nbl ≤ 127 + 2 * (S.m - 6) + (S.m - S.kLog))) _
            · calc value (E M) _ ≤ value (fun _ => False) _ :=
                  value_mono (fun o ⟨_, _, _, _, _, _, he⟩ => hext he) _
                _ = 0 := value_false _
                _ ≤ εr := zero_le
    _ = prCoin G + ∑ M ∈ s, (prCoin (B3 M) + εr) := by
        rw [avg_add, avg_sum]
        have h1 : avg (fun rs : Fin (S.m - 13) → F => if G rs then (1 : ℝ≥0∞) else 0) =
            prCoin G := by unfold prCoin; congr; funext rs; congr
        have h2 : ∀ M, avg (fun rs : Fin (S.m - 13) → F =>
            (if B3 M rs then (1 : ℝ≥0∞) else 0) + εr) = prCoin (B3 M) + εr := fun M => by
          rw [avg_add, avg_const]; unfold prCoin; congr; funext rs; congr
        rw [h1]
        simp only [h2]
    _ ≤ ((S.m - 13 : ℕ) : ℝ≥0∞) / Fintype.card F +
          ∑ _M ∈ s, (((S.m - 13 : ℕ) : ℝ≥0∞) / Fintype.card F + εr) := by
        gcongr with M hM
        · exact prCoin_req_one_le S.m
        · exact hB3 M
    _ ≤ ((S.m - 13 : ℕ) : ℝ≥0∞) / Fintype.card F +
          L * (((S.m - 13 + 127 + 2 * (S.m - 6) + (S.m - S.kLog) : ℕ) : ℝ≥0∞) /
            Fintype.card F) := by
        rw [Finset.sum_const, nsmul_eq_mul]
        gcongr
        rw [ENNReal.div_add_div_same]
        push_cast
        ring_nf
        exact le_rfl

open Classical in
/-- **The lincheck phase** over a list `W`: as `lincheck_phase_of_card`. -/
theorem lincheck_phase_of_list (A : Arith F K) (hA : A.Correct) (S : Statement) (sch : Schedule)
    (W : Set (Message F sch.k0 sch.level₀.logCols)) (extra : List (Claim F S.m)) (zc : ZcOut F S.m)
    (hzc : DZcW A S sch W extra (some zc)) (L : ℕ)
    (hL : (Finset.univ.filter fun M => M ∈ W).card ≤ L) :
    value (fun o => ¬ DLcW A S sch W extra zc o) (lincheck A S zc) ≤
      L * (((2 + 2 * (S.kLog - 6) + 63 : ℕ) : ℝ≥0∞) / Fintype.card F) := by
  classical
  set s := Finset.univ.filter fun M => M ∈ W
  let E : Message F sch.k0 sch.level₀.logCols → Option (Claim F S.m) → Prop := fun M o =>
    ∃ ab, o = some ab ∧ ab.Holds (wit A S sch M) ∧ (cClaim A zc).Holds (wit A S sch M)
      ∧ ∀ e ∈ extra, e.Holds (wit A S sch M)
  calc value (fun o => ¬ DLcW A S sch W extra zc o) (lincheck A S zc)
      ≤ value (fun o => ∃ M ∈ s, E M o) (lincheck A S zc) := by
        refine value_mono (fun o h => ?_) _
        cases o with
        | none => exact absurd trivial h
        | some ab =>
          simp only [DLcW, not_forall, not_not] at h
          obtain ⟨M, hM, hab⟩ := h
          exact ⟨M, by simp [s, hM], ab, rfl, hab⟩
    _ ≤ ∑ M ∈ s, value (E M) (lincheck A S zc) := value_exists_finset_le _ _ _
    _ ≤ ∑ _M ∈ s, ((2 + 2 * (S.kLog - 6) + 63 : ℕ) : ℝ≥0∞) / Fintype.card F := by
        refine Finset.sum_le_sum fun M hM => ?_
        have hMc : M ∈ W := by simpa [s] using hM
        by_cases hce : (cClaim A zc).Holds (wit A S sch M) ∧ ∀ e ∈ extra, e.Holds (wit A S sch M)
        · refine (value_mono (fun o ⟨ab, hab, hh, _⟩ => ⟨ab, hab, hh⟩) _).trans
            (value_lincheck_le A hA S zc _ fun ⟨ha, hb, hp⟩ => hzc M hMc ⟨ha, hb, hce.1, hp, hce.2⟩)
        · calc value (E M) (lincheck A S zc) ≤ value (fun _ => False) (lincheck A S zc) :=
                value_mono (fun o ⟨_, _, _, hc, he⟩ => hce ⟨hc, he⟩) _
            _ = 0 := value_false _
            _ ≤ _ := zero_le
    _ = s.card * (((2 + 2 * (S.kLog - 6) + 63 : ℕ) : ℝ≥0∞) / Fintype.card F) := by
        rw [Finset.sum_const, nsmul_eq_mul]
    _ ≤ L * (((2 + 2 * (S.kLog - 6) + 63 : ℕ) : ℝ≥0∞) / Fintype.card F) := by
        gcongr

open Classical in
/-- **The opening phase** over a list `W`: as `opening_phase_of_card`. -/
theorem opening_phase_of_list (A : Arith F K) (hA : A.Correct) (S : Statement) (sch : Schedule)
    (h7 : 7 ≤ S.m) (W : Set (Message F sch.k0 sch.level₀.logCols)) (claims : List (Claim F S.m))
    (hdoom : ∀ M ∈ W, ∃ c ∈ claims, ¬ c.Holds (wit A S sch M)) (L : ℕ)
    (hL : (Finset.univ.filter fun M => M ∈ W).card ≤ L) :
    value (fun o => ¬ DOpW S sch W o) (opening A h7 claims) ≤
      L * (((7 * claims.length + 1 : ℕ) : ℝ≥0∞) / Fintype.card F) := by
  classical
  set s := Finset.univ.filter fun M => M ∈ W
  calc value (fun o => ¬ DOpW S sch W o) (opening A h7 claims)
      ≤ value (fun o => ∃ M ∈ s, EM S sch M o) (opening A h7 claims) := by
        refine value_mono (fun o h => ?_) _
        cases o with
        | none => exact absurd trivial h
        | some op =>
          simp only [DOpW, not_forall, not_not] at h
          obtain ⟨M, hM, hsum⟩ := h
          exact ⟨M, by simp [s, hM], op, rfl, hsum⟩
    _ ≤ ∑ M ∈ s, value (EM S sch M) (opening A h7 claims) := value_exists_finset_le _ _ _
    _ ≤ ∑ _M ∈ s, ((7 * claims.length + 1 : ℕ) : ℝ≥0∞) / Fintype.card F :=
        Finset.sum_le_sum fun M hM => value_EM_le A hA S sch h7 claims M
          (hdoom M (by simpa [s] using hM))
    _ = s.card * (((7 * claims.length + 1 : ℕ) : ℝ≥0∞) / Fintype.card F) := by
        rw [Finset.sum_const, nsmul_eq_mul]
    _ ≤ L * (((7 * claims.length + 1 : ℕ) : ℝ≥0∞) / Fintype.card F) := by
        gcongr

open Classical in
/-- **The link points** over a list `W` of which no message packs a satisfying witness: as
`link_sound_of_card`. -/
theorem link_sound_of_list (A : Arith F K) (hA : A.Correct) (S : Statement) (sch : Schedule)
    (ptLocal mPts : ℕ) (W : Set (Message F sch.k0 sch.level₀.logCols))
    (hC : ¬ ∃ M ∈ W, S.Satisfies (wit A S sch M))
    (hlay : S.LinkLayout ptLocal mPts) (L : ℕ)
    (hL : (Finset.univ.filter fun M => M ∈ W).card ≤ L) :
    prCoin (fun pts : Fin 2 → Fin mPts → F =>
        ¬ RepDoomedW A S sch W (extraClaims S ptLocal mPts pts)) ≤
      L * (S.regions.length * ((mPts : ℝ≥0∞) / Fintype.card F) ^ 2) := by
  classical
  set s := Finset.univ.filter fun M => M ∈ W
  calc prCoin (fun pts : Fin 2 → Fin mPts → F =>
          ¬ RepDoomedW A S sch W (extraClaims S ptLocal mPts pts))
      ≤ prCoin (fun pts => ∃ M ∈ s, ∃ R ∈ S.regions.toFinset,
          LinkBad S R ptLocal mPts (wit A S sch M) pts) := by
        refine prCoin_mono fun pts h => ?_
        simp only [RepDoomedW, not_forall, not_not] at h
        obtain ⟨M, hMW, hr1cs, hpin, hext⟩ := h
        have hsat : ¬ S.Satisfies (wit A S sch M) := fun hs => hC ⟨M, hMW, hs⟩
        have hreg : ¬ ∀ R ∈ S.regions, ∀ (o : Fin (2 ^ S.nbl)) (i : ℕ)
            (_hi : i < 2 ^ R.free.length), ∀ h : R.inBlock i < 2 ^ S.kLog,
              S.block (wit A S sch M) o ⟨R.inBlock i, h⟩ = R.bits o i :=
          fun hr => hsat ⟨hr1cs, hpin, hr⟩
        push Not at hreg
        obtain ⟨R, hR, o, i, hi, hib, hne⟩ := hreg
        refine ⟨M, by simp only [s, Finset.mem_filter, Finset.mem_univ, true_and]; exact hMW,
          R, List.mem_toFinset.2 hR, ⟨o, i, hi, hib, hne⟩, hext _ ?_, hext _ ?_⟩
        · exact List.mem_flatMap.2 ⟨R, hR, by simp⟩
        · exact List.mem_flatMap.2 ⟨R, hR, by simp⟩
    _ ≤ ∑ M ∈ s, prCoin (fun pts => ∃ R ∈ S.regions.toFinset,
          LinkBad S R ptLocal mPts (wit A S sch M) pts) :=
        prCoin_exists_le _ _
    _ ≤ ∑ M ∈ s, ∑ R ∈ S.regions.toFinset,
          prCoin (LinkBad S R ptLocal mPts (wit A S sch M)) :=
        Finset.sum_le_sum fun M _ => prCoin_exists_le _ _
    _ ≤ ∑ _M ∈ s, ∑ _R ∈ S.regions.toFinset, ((mPts : ℝ≥0∞) / Fintype.card F) ^ 2 :=
        Finset.sum_le_sum fun M _ => Finset.sum_le_sum fun R hR =>
          prCoin_linkBad_le A hA S R ptLocal mPts hlay (List.mem_toFinset.1 hR) _
    _ ≤ L * (S.regions.length * ((mPts : ℝ≥0∞) / Fintype.card F) ^ 2) := by
        simp only [Finset.sum_const, nsmul_eq_mul]
        gcongr
        all_goals first | exact_mod_cast hL | exact_mod_cast List.toFinset_card_le _

end FlockSoundness
