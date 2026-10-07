import Proofs.Flock.Soundness.Discharge.ZkSession.Inner
import ArkLib.Data.CodingTheory.ProximityGap.BCIKS20.AffineLines.UniqueDecoding
import VCVio.OracleComp.Constructions.SampleableType.NativeMeasure

/-!
# The `--zk` inner proof is sound: the code lemmas and `inner_sound`

`Inner` states the inner proof (`innerGame`, `InnerGood`, `εInner`) on Mathlib. Its soundness reads the pads code as
ArkLib's Reed–Solomon code and uses BCIKS20's correlated agreement on lines in the unique-decoding regime.
-/

namespace FlockSoundness.Discharge.ZkSession

open Game
open scoped ENNReal NNReal

variable {F : Type} [Field F]

/-! ## The code -/

section Code

open Polynomial

variable {N L : ℕ} (dom : Fin N ↪ F)

theorem enc_eq_evalOnPoints (m : Fin L → F) :
    enc dom m = ReedSolomon.evalOnPoints dom ((degreeLTEquiv F L).symm m : F[X]) := by
  funext j
  simp [enc, ReedSolomon.evalOnPoints, degreeLTEquiv, eval_monomial]

theorem enc_mem (m : Fin L → F) : enc dom m ∈ ReedSolomon.code dom L := by
  rw [enc_eq_evalOnPoints]
  exact Submodule.mem_map.2 ⟨_, ((degreeLTEquiv F L).symm m).2, rfl⟩

theorem exists_enc {v : Fin N → F} (hv : v ∈ ReedSolomon.code dom L) : ∃ m : Fin L → F, enc dom m = v := by
  obtain ⟨p, hp, rfl⟩ := Submodule.mem_map.1 hv
  refine ⟨degreeLTEquiv F L ⟨p, hp⟩, ?_⟩
  rw [enc_eq_evalOnPoints, LinearEquiv.symm_apply_apply]

theorem natDegree_lt_of_mem [DecidableEq F] {p : F[X]} (hp : p ∈ degreeLT F L) (hL : 0 < L) :
    p.natDegree < L := by
  by_cases h0 : p = 0
  · simp [h0, hL]
  · exact (natDegree_lt_iff_degree_lt h0).2 (mem_degreeLT.1 hp)

theorem enc_injective [DecidableEq F] (hL : L ≤ N) {m m' : Fin L → F} (h : enc dom m = enc dom m') : m = m' := by
  rcases Nat.eq_zero_or_pos L with rfl | hL0
  · exact Subsingleton.elim _ _
  apply (degreeLTEquiv F L).symm.injective
  apply Subtype.ext
  apply eq_of_natDegree_lt_card_of_eval_eq _ _ dom.injective
  · intro j
    have := congrFun h j
    rw [enc_eq_evalOnPoints, enc_eq_evalOnPoints] at this
    exact this
  · rw [Fintype.card_fin]
    exact max_lt ((natDegree_lt_of_mem ((degreeLTEquiv F L).symm m).2 hL0).trans_le hL)
      ((natDegree_lt_of_mem ((degreeLTEquiv F L).symm m').2 hL0).trans_le hL)

theorem enc_add_mul (m m' : Fin L → F) (β : F) :
    enc dom (fun i => m i + β * m' i) = fun j => enc dom m j + β * enc dom m' j := by
  funext j
  simp only [enc, add_mul, Finset.sum_add_distrib, Finset.mul_sum, mul_assoc]

end Code

/-! ## Distance, queries and coins -/

section Sound

variable [Fintype F] [DecidableEq F]

open Code ProximityGap

omit [Fintype F] in
/-- Agreement on `(1 − δ)·N` positions with a codeword puts a word within relative distance `δ` of the code. -/
theorem relDist_le_of_agree {N L : ℕ} [NeZero N] (dom : Fin N ↪ F) (δ : ℝ≥0) {u v : Fin N → F}
    (hv : v ∈ ReedSolomon.code dom L) (h : Agree δ u v) :
    δᵣ(u, (ReedSolomon.code dom L : Set (Fin N → F))) ≤ δ := by
  obtain ⟨S, hS, hSv⟩ := h
  have h1 : (relHammingDist u v : ℝ≥0) ≤ δ := by
    rw [relCloseToWord_iff_exists_agreementCols]
    refine ⟨S, ?_, fun j => ⟨hSv j, fun hne hj => hne (hSv j hj)⟩⟩
    rw [relDist_floor_bound_iff_complement_bound]
    simpa using hS
  calc δᵣ(u, (ReedSolomon.code dom L : Set (Fin N → F))) ≤ (relHammingDist u v : ℝ≥0∞) :=
        relDistFromCode_le_relDist_to_mem u v hv
    _ ≤ δ := by exact_mod_cast h1

/-- **BCIKS20 in the unique-decoding regime, as a coin bound.** If two words are not jointly `δ`-close to the code, a
uniform `z` puts `U₀ + z·U₁` within `δ` of a codeword with probability at most `N/|F|`. -/
theorem prCoin_close_le {N L : ℕ} [NeZero N] (dom : Fin N ↪ F) (δ : ℝ≥0)
    (hδ : δ ≤ relativeUniqueDecodingRadius (ReedSolomon.code dom L : Set (Fin N → F)))
    (U : Fin 2 → Fin N → F) (hU : ¬ jointAgreement (ReedSolomon.code dom L : Set (Fin N → F)) δ U) :
    prCoin (fun z : F => ∃ v ∈ ReedSolomon.code dom L, Agree δ (fun j => U 0 j + z * U 1 j) v) ≤
      (N : ℝ≥0∞) / Fintype.card F := by
  classical
  let _ : SampleableType F := SampleableType.ofFintype F
  have hca := RS_correlatedAgreement_affineLines_uniqueDecodingRegime (deg := L) (domain := dom) hδ
  have hle : Pr{let z ← $ᵗ F}[δᵣ(U 0 + z • U 1, (ReedSolomon.code dom L : Set (Fin N → F))) ≤ δ] ≤
      (errorBound δ L dom : ℝ≥0∞) := not_lt.1 fun hgt => hU (hca U hgt)
  rw [errorBound_eq_n_div_q_of_le_relUDR hδ, SampleableType.prEvent_uniformSample, ← prCoin_eq_card,
    Fintype.card_fin] at hle
  refine le_trans (prCoin_mono fun z hz => ?_) (hle.trans (le_of_eq ?_))
  · obtain ⟨v, hv, hz⟩ := hz
    have hU' : U 0 + z • U 1 = fun j => U 0 j + z * U 1 j := by
      funext j
      simp
    rw [hU']
    exact relDist_le_of_agree dom δ hv hz
  · rw [ENNReal.coe_div (by exact_mod_cast Fintype.card_ne_zero)]
    simp

omit [DecidableEq F] in
/-- `q` independent query words land in the positions satisfying `P` with probability at most `(|P|/N)^q`. -/
theorem prCoin_queries_le {N q : ℕ} (pos : F → Fin N)
    (hpos : ∀ D : Finset (Fin N), prCoin (fun x => pos x ∈ D) ≤ (D.card : ℝ≥0∞) / N)
    (P : Fin N → Prop) [DecidablePred P] :
    prCoin (fun w : Fin q → F => ∀ j, P (pos (w j))) ≤
      (((Finset.univ.filter P).card : ℝ≥0∞) / N) ^ q := by
  rw [FlockSoundness.prCoin_forall_fin q (fun _ x => P (pos x))]
  calc ∏ _j : Fin q, prCoin (fun x => P (pos x))
      ≤ ∏ _j : Fin q, (((Finset.univ.filter P).card : ℝ≥0∞) / N) :=
        Finset.prod_le_prod fun _ _ =>
          (prCoin_mono (q := fun x => pos x ∈ Finset.univ.filter P)
            fun x hx => Finset.mem_filter.2 ⟨Finset.mem_univ _, hx⟩).trans (hpos (Finset.univ.filter P))
    _ = _ := by rw [Finset.prod_const, Finset.card_univ, Fintype.card_fin]

theorem prCoin_false {C : Type} [Fintype C] : prCoin (fun _ : C => False) = 0 := by
  simp [prCoin, avg]

omit [DecidableEq F] in
/-- **The queries.** If `C(y)` agrees with `Ch + β·Cμ` on at most `(1 − δ)·N` positions, a check that needs every query
to land on an agreeing position passes with probability at most `(1 − δ)^q`. -/
theorem value_queries_le {N L q : ℕ} [NeZero N] (dom : Fin N ↪ F) (pos : F → Fin N)
    (hpos : ∀ D : Finset (Fin N), prCoin (fun x => pos x ∈ D) ≤ (D.card : ℝ≥0∞) / N) (δ : ℝ≥0)
    (y : Fin L → F) (Ch Cμ : Fin N → F) (β : F) (acc : (Fin q → F) → Prop) [DecidablePred acc]
    (hacc : ∀ w, acc w → ∀ j, enc dom y (pos (w j)) = Ch (pos (w j)) + β * Cμ (pos (w j)))
    [DecidableEq F]
    (hfar : (((Finset.univ.filter fun j => enc dom y j = Ch j + β * Cμ j).card : ℕ) : ℝ≥0) ≤ (1 - δ) * N) :
    value (· = true) (Game.coin (Fin q → F) fun w => .ret (decide (acc w))) ≤
      ((1 - δ : ℝ≥0) : ℝ≥0∞) ^ q := by
  rw [value_coin_ret]
  refine (prCoin_mono fun w hw => hacc w (of_decide_eq_true hw)).trans ?_
  refine (prCoin_queries_le pos hpos (fun j => enc dom y j = Ch j + β * Cμ j)).trans ?_
  gcongr
  rw [ENNReal.div_le_iff (by simp [NeZero.ne N]) (by simp)]
  exact_mod_cast hfar

/-- A check that never passes. -/
theorem value_coin_ret_false {C : Type} [Fintype C] [Nonempty C] (acc : C → Prop) [DecidablePred acc]
    (h : ∀ c, ¬ acc c) : value (· = true) (Game.coin C fun c => .ret (decide (acc c))) ≤ 0 := by
  rw [value_coin_ret]
  refine (prCoin_mono fun c hc => h c (of_decide_eq_true hc)).trans (le_of_eq prCoin_false)

omit [Fintype F] [DecidableEq F] in
/-- The unique-decoding radius of the pads code: `2δN ≤ N − L`. -/
theorem udr_bound {N L : ℕ} [NeZero L] [DecidableEq F] (dom : Fin N ↪ F) (hL : L ≤ N) (δ : ℝ≥0)
    (hδ : δ ≤ relativeUniqueDecodingRadius (ReedSolomon.code dom L : Set (Fin N → F))) :
    2 * (δ : ℝ) * N ≤ N - L := by
  have hN : (0 : ℝ) < N := by
    have := NeZero.ne L
    exact_mod_cast (by omega : 0 < N)
  unfold relativeUniqueDecodingRadius at hδ
  rw [ReedSolomon.dist_eq_of_le (by simpa using hL), Fintype.card_fin, Nat.cast_add, Nat.cast_one,
    add_tsub_cancel_right] at hδ
  have h' : (δ : ℝ) ≤ ((N - L : ℕ) : ℝ) / 2 / N := by exact_mod_cast hδ
  rw [Nat.cast_sub hL, div_div, le_div_iff₀ (by positivity)] at h'
  linarith

omit [Field F] [DecidableEq F] in
theorem εInner_caseA (N : ℕ) (q : ℕ) (δ : ℝ≥0) :
    1 / (Fintype.card F : ℝ≥0∞) + (1 / Fintype.card F + (1 / Fintype.card F + ((1 - δ : ℝ≥0) : ℝ≥0∞) ^ q)) ≤
      εInner N F q δ := by
  unfold εInner
  rw [← add_assoc, ← add_assoc, ENNReal.div_add_div_same, ENNReal.div_add_div_same]
  refine add_le_add_left (ENNReal.div_le_div_right ?_ _) _
  calc (1 + 1 + 1 : ℝ≥0∞) = 3 := by norm_num
    _ ≤ (N : ℝ≥0∞) + 3 := le_add_self

omit [Field F] [DecidableEq F] in
theorem εInner_caseB (N : ℕ) (q : ℕ) (δ : ℝ≥0) :
    0 + (0 + ((N : ℝ≥0∞) / Fintype.card F + ((1 - δ : ℝ≥0) : ℝ≥0∞) ^ q)) ≤ εInner N F q δ := by
  unfold εInner
  rw [zero_add, zero_add]
  gcongr
  exact le_self_add

omit [DecidableEq F] in
theorem card_le_of_real {N a : ℕ} {δ : ℝ≥0} (hδ1 : δ ≤ 1) (h : (a : ℝ) ≤ (1 - (δ : ℝ)) * N) :
    (a : ℝ≥0) ≤ (1 - δ) * N := by
  rw [← NNReal.coe_le_coe]
  push_cast [NNReal.coe_sub hδ1]
  exact h

omit [DecidableEq F] in
theorem real_le_of_card {N a : ℕ} {δ : ℝ≥0} (hδ1 : δ ≤ 1) (h : (1 - δ) * (N : ℝ≥0) ≤ a) :
    (1 - (δ : ℝ)) * N ≤ a := by
  have := NNReal.coe_le_coe.2 h
  push_cast [NNReal.coe_sub hδ1] at this
  exact this

theorem innerGame_eq {N : ℕ} (dom : Fin N ↪ F) (s nP q : ℕ) (pos : F → Fin N) {nA : ℕ}
    (rows : Fin nA → Lin F s) (tr : Triple s) (Ch Cμ : Fin N → F) :
    innerGame dom s nP q pos rows tr Ch Cμ =
      .coin F fun ε => .send (F × F) fun ρσ => .coin (Fin (nA + 3) → F) fun γ => .send F fun τ =>
        .coin F fun β => .send (Fin (s + nP) → F) fun y => .coin (Fin q → F) fun w =>
          .ret (decide (innerCheck dom pos rows tr Ch Cμ ε ρσ γ τ β y w)) := rfl

/-! ## Soundness -/

/-- **The coin `β` and the queries, decoded case.** The words `Ch, Cμ` agree with the codewords `C(mh), C(mμ)` on `S`, and
the batch at `mh`'s pads is nonzero (`hD`). Except for one `β`, the decoded answer `mh + β·mμ` fails the batched check,
and every other `y` disagrees with `Ch + β·Cμ` on more than `δN` positions. -/
theorem step_beta_close [CharP F 2] {N : ℕ} [NeZero N] (dom : Fin N ↪ F) (s nP q : ℕ) (pos : F → Fin N)
    (hpos : ∀ D : Finset (Fin N), prCoin (fun x => pos x ∈ D) ≤ (D.card : ℝ≥0∞) / N)
    {nA : ℕ} (rows : Fin nA → Lin F s) (tr : Triple s) (Ch Cμ : Fin N → F) (δ : ℝ≥0) (hL : s + nP ≤ N)
    (hδ1 : δ ≤ 1) (hud : 2 * (δ : ℝ) * N ≤ N - (s + nP : ℕ))
    (S : Finset (Fin N)) (hSr : (1 - (δ : ℝ)) * N ≤ S.card) (mh mμ : Fin (s + nP) → F)
    (hSh : ∀ j ∈ S, enc dom mh j = Ch j) (hSμ : ∀ j ∈ S, enc dom mμ j = Cμ j)
    (ε : F) (ρσ : F × F) (γ : Fin (nA + 3) → F)
    (hD : ∑ i, batchCoef γ (Fin.append rows (tripleRows tr ε ρσ.1 ρσ.2)) i * head mh i -
      batchT γ (Fin.append rows (tripleRows tr ε ρσ.1 ρσ.2)) ≠ 0) :
    value (· = true) (.send F fun τ => .coin F fun β => .send (Fin (s + nP) → F) fun y =>
        .coin (Fin q → F) fun w => .ret (decide (innerCheck dom pos rows tr Ch Cμ ε ρσ γ τ β y w))) ≤
      1 / Fintype.card F + ((1 - δ : ℝ≥0) : ℝ≥0∞) ^ q := by
  classical
  rw [value_send]
  refine iSup_le fun τ => ?_
  set R := Fin.append rows (tripleRows tr ε ρσ.1 ρσ.2) with hR
  set c := batchCoef γ R with hc
  set t := batchT γ R with ht
  refine value_coin_le (· = true) F _
    (fun β => (∑ i, c i * head mh i - t) + β * (∑ i, c i * head mμ i - τ) = 0) _ _ ?_ ?_
  · by_cases he : ∑ i, c i * head mμ i - τ = 0
    · refine ((prCoin_mono fun β hβ => hD ?_).trans (le_of_eq prCoin_false)).trans zero_le
      rw [he, mul_zero, add_zero] at hβ
      exact hβ
    · exact (prCoin_mono (q := fun β => (∑ i, c i * head mμ i - τ) * β + (∑ i, c i * head mh i - t) = 0)
        fun β hβ => by linear_combination hβ).trans (FlockSoundness.prCoin_linear_le _ _ he)
  intro β hβ
  rw [value_send]
  refine iSup_le fun y => ?_
  by_cases hB : ∑ i, c i * head y i = t + β * τ
  · -- `y` passes the batched check, so it is not the decoded `mh + β·mμ`.
    have hne : y ≠ fun i => mh i + β * mμ i := by
      rintro rfl
      apply hβ
      have : ∑ i, c i * (head mh i + β * head mμ i) = t + β * τ := hB
      simp only [mul_add, Finset.sum_add_distrib] at this
      rw [show ∑ i, c i * (β * head mμ i) = β * ∑ i, c i * head mμ i by
        rw [Finset.mul_sum]; exact Finset.sum_congr rfl fun i _ => by ring] at this
      linear_combination this
    refine value_queries_le dom pos hpos δ y Ch Cμ β _ (fun w hw => hw.2) ?_
    set P := Finset.univ.filter fun j => enc dom y j = Ch j + β * Cμ j with hP
    have hPS : (P ∩ S).card < s + nP := by
      by_contra hge
      push Not at hge
      apply hne
      apply enc_injective dom hL
      rw [enc_add_mul]
      refine ReedSolomon.eq_of_agree_of_card_le (enc_mem dom y)
        ((ReedSolomon.code dom (s + nP)).add_mem (enc_mem dom mh)
          ((ReedSolomon.code dom (s + nP)).smul_mem β (enc_mem dom mμ))) hge fun j hj => ?_
      obtain ⟨hjP, hjS⟩ := Finset.mem_inter.1 hj
      rw [(Finset.mem_filter.1 hjP).2]
      simp [hSh j hjS, hSμ j hjS]
    have hPd : (P \ S).card ≤ N - S.card := by
      calc (P \ S).card ≤ (Finset.univ \ S).card :=
            Finset.card_le_card (Finset.sdiff_subset_sdiff (Finset.subset_univ _) le_rfl)
        _ = N - S.card := by
            rw [Finset.card_sdiff_of_subset (Finset.subset_univ _), Finset.card_univ, Fintype.card_fin]
    have hSN : S.card ≤ N := by simpa using Finset.card_le_univ S
    have hsum := Finset.card_inter_add_card_sdiff P S
    apply card_le_of_real hδ1
    have h1 : (P.card : ℝ) + 1 ≤ (s + nP : ℕ) + (N - S.card : ℝ) := by
      have : P.card + 1 ≤ (s + nP) + (N - S.card) := by omega
      have h' : ((P.card + 1 : ℕ) : ℝ) ≤ ((s + nP + (N - S.card) : ℕ) : ℝ) := by exact_mod_cast this
      push_cast [Nat.cast_sub hSN] at h'
      push_cast
      linarith
    nlinarith
  · exact (value_coin_ret_false _ fun w hw => hB hw.1).trans zero_le

/-- **Decoded case.** The words are jointly `δ`-close to the code, so the decoded pads fail a row or the product: the
sacrifice's `ε`, the batch's `γ` and `β` each leave that visible except with probability `1/|F|`. -/
theorem inner_close [CharP F 2] {N : ℕ} [NeZero N] (dom : Fin N ↪ F) (s nP q : ℕ) (pos : F → Fin N)
    (hpos : ∀ D : Finset (Fin N), prCoin (fun x => pos x ∈ D) ≤ (D.card : ℝ≥0∞) / N)
    {nA : ℕ} (rows : Fin nA → Lin F s) (tr : Triple s) (Ch Cμ : Fin N → F) (δ : ℝ≥0) (hL : s + nP ≤ N)
    (hδ1 : δ ≤ 1) (hud : 2 * (δ : ℝ) * N ≤ N - (s + nP : ℕ))
    (hJ : jointAgreement (ReedSolomon.code dom (s + nP) : Set (Fin N → F)) δ ![Ch, Cμ])
    (h : ¬ InnerGood dom s nP rows tr δ Ch) :
    value (· = true) (innerGame dom s nP q pos rows tr Ch Cμ) ≤ εInner N F q δ := by
  classical
  rw [innerGame_eq]
  obtain ⟨S, hS, v, hv⟩ := hJ
  obtain ⟨mh, hmh⟩ := exists_enc dom (hv 0).1
  obtain ⟨mμ, hmμ⟩ := exists_enc dom (hv 1).1
  have hSh : ∀ j ∈ S, enc dom mh j = Ch j := fun j hj => by
    have := (Finset.mem_filter.1 ((hv 0).2 hj)).2
    rw [hmh]
    simpa using this
  have hSμ : ∀ j ∈ S, enc dom mμ j = Cμ j := fun j hj => by
    have := (Finset.mem_filter.1 ((hv 1).2 hj)).2
    rw [hmμ]
    simpa using this
  have hS' : (1 - δ) * (N : ℝ≥0) ≤ S.card := by simpa using hS
  have hSr := real_le_of_card hδ1 hS'
  set hh := head mh with hhh
  have hfail : ¬ ((∀ k, (rows k).eval hh = 0) ∧ hh tr.ab = hh tr.fa * hh tr.fb) := fun hg =>
    h ⟨mh, ⟨S, hS', fun j hj => (hSh j hj).symm⟩, hg⟩
  refine le_trans ?_ (εInner_caseA N q δ)
  -- The sacrifice's coin `ε`.
  refine value_coin_le (· = true) F _ (fun ε => (∀ k, (rows k).eval hh = 0) ∧
      hh tr.ab ≠ hh tr.fa * hh tr.fb ∧
      (hh tr.ab + hh tr.fa * hh tr.fb) * ε + (hh tr.tw - hh tr.tx * hh tr.ty) = 0) _ _ ?_ ?_
  · by_cases hc : (∀ k, (rows k).eval hh = 0) ∧ hh tr.ab ≠ hh tr.fa * hh tr.fb
    · have ha : hh tr.ab + hh tr.fa * hh tr.fb ≠ 0 := fun h0 =>
        hc.2 (by rw [eq_neg_of_add_eq_zero_left h0, CharTwo.neg_eq])
      exact (prCoin_mono fun ε hε => hε.2.2).trans (FlockSoundness.prCoin_linear_le _ _ ha)
    · exact ((prCoin_mono fun ε hε => hc ⟨hε.1, hε.2.1⟩).trans (le_of_eq prCoin_false)).trans zero_le
  intro ε hε
  rw [value_send]
  refine iSup_le fun ρσ => ?_
  have hrow : ∃ k, (Fin.append rows (tripleRows tr ε ρσ.1 ρσ.2) k).eval hh ≠ 0 := by
    by_contra hall
    push Not at hall
    have h1 : ∀ k, (rows k).eval hh = 0 := fun k => by
      simpa [Fin.append_left] using hall (Fin.castAdd 3 k)
    have ht : ∀ i : Fin 3, (tripleRows tr ε ρσ.1 ρσ.2 i).eval hh = 0 := fun i => by
      simpa [Fin.append_right] using hall (Fin.natAdd nA i)
    have hsac := sacrifice_eq tr ε ρσ.1 ρσ.2 hh (ht 0) (ht 1)
    rw [ht 2] at hsac
    have hab : hh tr.ab ≠ hh tr.fa * hh tr.fb := fun hab => hfail ⟨h1, hab⟩
    exact hε ⟨h1, hab, by linear_combination -hsac⟩
  obtain ⟨k₀, hk₀⟩ := hrow
  -- The batch's coins `γ`.
  refine value_coin_le (· = true) (Fin (nA + 3) → F) _
    (fun γ => ∑ k, γ k * (Fin.append rows (tripleRows tr ε ρσ.1 ρσ.2) k).eval hh = 0)
    (1 / Fintype.card F) _
    (FlockSoundness.prCoin_batch_le (fun k => (Fin.append rows (tripleRows tr ε ρσ.1 ρσ.2) k).eval hh) k₀ hk₀)
    fun γ hγ => ?_
  refine step_beta_close dom s nP q pos hpos rows tr Ch Cμ δ hL hδ1 hud S hSr mh mμ hSh hSμ ε ρσ γ ?_
  rw [batch_eval]
  exact hγ

/-- **Far case.** The words are not jointly `δ`-close to the code: by BCIKS20, `Ch + β·Cμ` is `δ`-far from it except
with probability `N/|F|`, and then every `y` fails a query with probability at least `δ`. -/
theorem inner_far {N : ℕ} [NeZero N] (dom : Fin N ↪ F) (s nP q : ℕ) (pos : F → Fin N)
    (hpos : ∀ D : Finset (Fin N), prCoin (fun x => pos x ∈ D) ≤ (D.card : ℝ≥0∞) / N)
    {nA : ℕ} (rows : Fin nA → Lin F s) (tr : Triple s) (Ch Cμ : Fin N → F) (δ : ℝ≥0)
    (hδ : δ ≤ relativeUniqueDecodingRadius (ReedSolomon.code dom (s + nP) : Set (Fin N → F)))
    (hJ : ¬ jointAgreement (ReedSolomon.code dom (s + nP) : Set (Fin N → F)) δ ![Ch, Cμ]) :
    value (· = true) (innerGame dom s nP q pos rows tr Ch Cμ) ≤ εInner N F q δ := by
  classical
  rw [innerGame_eq]
  refine le_trans ?_ (εInner_caseB N q δ)
  refine value_coin_le (· = true) F _ (fun _ => False) 0 _ (le_of_eq prCoin_false) fun ε _ => ?_
  rw [value_send]
  refine iSup_le fun ρσ => ?_
  refine value_coin_le (· = true) (Fin (nA + 3) → F) _ (fun _ => False) 0 _ (le_of_eq prCoin_false)
    fun γ _ => ?_
  rw [value_send]
  refine iSup_le fun τ => ?_
  refine value_coin_le (· = true) F _
    (fun β => ∃ v ∈ ReedSolomon.code dom (s + nP), Agree δ (fun j => Ch j + β * Cμ j) v) _ _
    (prCoin_close_le dom δ hδ ![Ch, Cμ] hJ) ?_
  intro β hβ
  rw [value_send]
  refine iSup_le fun y => ?_
  refine value_queries_le dom pos hpos δ y Ch Cμ β _ (fun w hw => hw.2) ?_
  by_contra hlt
  push Not at hlt
  exact hβ ⟨enc dom y, enc_mem dom y, _, hlt.le, fun j hj => ((Finset.mem_filter.1 hj).2).symm⟩

/-- **The inner proof is sound.** Against committed pads words `Ch, Cμ`, unless some pads message within relative
distance `δ` of `Ch` has every row vanish at its pads and `h_ab = h_fa·h_fb`, the verifier accepts with probability at
most `(N + 3)/|F| + (1 − δ)^q`, for `δ` up to the code's unique-decoding radius. -/
theorem inner_sound [CharP F 2] {N : ℕ} (dom : Fin N ↪ F) (s nP q : ℕ) [NeZero (s + nP)]
    (pos : F → Fin N) (hpos : ∀ D : Finset (Fin N), prCoin (fun x => pos x ∈ D) ≤ (D.card : ℝ≥0∞) / N)
    {nA : ℕ} (rows : Fin nA → Lin F s) (tr : Triple s) (Ch Cμ : Fin N → F) (δ : ℝ≥0) (hL : s + nP ≤ N)
    (hδ : δ ≤ relativeUniqueDecodingRadius (ReedSolomon.code dom (s + nP) : Set (Fin N → F)))
    (h : ¬ InnerGood dom s nP rows tr δ Ch) :
    value (· = true) (innerGame dom s nP q pos rows tr Ch Cμ) ≤ εInner N F q δ := by
  have : NeZero N := ⟨by have := NeZero.ne (s + nP); omega⟩
  have hud := udr_bound dom hL δ hδ
  have hNpos : (0 : ℝ) < N := by exact_mod_cast Nat.pos_of_ne_zero (NeZero.ne N)
  have hδ1 : δ ≤ 1 := by
    rw [← NNReal.coe_le_coe, NNReal.coe_one]
    nlinarith
  by_cases hJ : jointAgreement (ReedSolomon.code dom (s + nP) : Set (Fin N → F)) δ ![Ch, Cμ]
  · exact inner_close dom s nP q pos hpos rows tr Ch Cμ δ hL hδ1 (by exact_mod_cast hud) hJ h
  · exact inner_far dom s nP q pos hpos rows tr Ch Cμ δ hδ hJ

end Sound

end FlockSoundness.Discharge.ZkSession
