import Proofs.Flock.Claims.Sampling.Draws
import Proofs.Pouw.Credit
import SampledProofs.HiddenDraw

/-!
# Credit over hidden live rows (the hidden-position draw)

A stratum's `M` positions are one per live (row, column tile) of its calls, and the verifier fixes `M` from public values
alone (top-k × the window's live tokens × the stratum's MoE layers × tiles per row), never from the prover's calls table.
Before the draw coin the prover commits the table and every unit's values, so which (call, row, tile) units are correct
is fixed before the coin. The verifier draws positions from `[0, M)`, and a drawn position is accepted only on a
witness, a correct unit the opening locates (`SampledProofs.HiddenDraw.witness`). No two positions share a witness
(`witness_inj`, for any table, overlapping or out of order), so the good positions inject into the good live units, and
the credit `M` is at most `good/(1 − ε)` but with the draw's miss probability. A table that marks more rows live changes
no credit, which is `M`.

* `good_work_le`, `good_card_le`: **the injection**. Positions with a witness in `P`, no two sharing one, are at most
  `|P|`, and their work at most `P`'s.
* `subset_credit`, `subset_credit_pow`: **one stratum**. `k` positions drawn uniformly without replacement from `[0, M)`
  are all good while fewer than `(1 − ε)·M` units are good with probability at most
  `C(M − ⌈εM⌉, k)/C(M, k) ≤ (1 − ε)^k` (`Claims.Sampling.subset_miss_le`).
* `stratified_credit`: the same for stratum `s` of the stratified draw (`Claims.Sampling.stratified_miss_eq`).
* `work_credit`: **the window under the work law**. Every drawn position good while the good units do less than
  `(1 − ε)` of the work `W`, with probability at most `(1 − ε)^K` (`Claims.Sampling.work_miss_le`).
* `subset_credit_epsAt`: one stratum at the ε `PouwWindow.epsAt K δ M M 1` answers: at most `δ`.
* `credit_of_miss`, `drawn_credit`: **the served draw**, at `Properties.Pouw.Credit.EpsAtSound`'s form. Over the strata
  `PouwCheck.Credit.law` gives, at the ε `epsAt K δ C W 1` answers, every drawn position of `U` good while the good
  units' work falls short of `U`'s by more than `ε·C`, with probability at most `δ`
  (`Properties.Pouw.Credit.SecurityProofs.epsAtSound`).
* `hidden_credit`, `hidden_credit_pow`, `hidden_credit_epsAt`: the one-stratum bounds at HiddenDraw's witnesses
  (`Proves`), for any table `cs`: the good live units are the correct ones of `liveUnits width cs`, where
  `witness_bounds` puts every witness.

What a correct unit is, and that the opening's `ok` means the staged rows are the committed leaves at the private index,
are the opening's meaning and the commitments' binding, outside this file.
-/

namespace Pouw.SecurityProofs.HiddenCredit

open Lean (Json toJson)
open Finset FlockSoundness.Audit
open Properties.Pouw.Strata (unitWork)
open Properties.Pouw.Credit (drawnOn)
open Pouw.RelativeSecurity (prob)
open SampledProofs.HiddenDraw (Call witness witness_inj witness_bounds)

variable {T : Type}

/-! ## The injection -/

/-- **The good positions' work is at most the good units'**: when every position of `G` has a witness in `P`, no two
positions of `G` share a witness, and each unit does the work of a position it proves, `G`'s work is at most `P`'s. -/
theorem good_work_le {ι : Type} (G : Finset ι) (Wit : ι → T → Prop)
    (hinj : ∀ u ∈ G, ∀ u' ∈ G, ∀ x, Wit u x → Wit u' x → u = u') (P : Finset T) (hG : ∀ u ∈ G, ∃ x ∈ P, Wit u x)
    (wt : ι → ℕ) (pw : T → ℕ) (hpw : ∀ u ∈ G, ∀ x, Wit u x → pw x = wt u) :
    ∑ u ∈ G, wt u ≤ ∑ x ∈ P, pw x := by
  classical
  choose f hfP hfW using hG
  let g : G → T := fun a => f a.1 a.2
  have hg : Set.InjOn g (G.attach : Set G) := fun a _ b _ hab =>
    Subtype.ext (hinj a.1 a.2 b.1 b.2 (g a) (hfW a.1 a.2) (by rw [hab]; exact hfW b.1 b.2))
  calc ∑ u ∈ G, wt u = ∑ a ∈ G.attach, wt a.1 := (Finset.sum_attach G wt).symm
    _ = ∑ a ∈ G.attach, pw (g a) := Finset.sum_congr rfl fun a _ => (hpw a.1 a.2 _ (hfW a.1 a.2)).symm
    _ = ∑ x ∈ G.attach.image g, pw x := (Finset.sum_image hg).symm
    _ ≤ ∑ x ∈ P, pw x := Finset.sum_le_sum_of_subset fun x hx => by
        obtain ⟨a, -, rfl⟩ := Finset.mem_image.1 hx
        exact hfP a.1 a.2

/-- **The injection, by count**: at most `|P|` positions of `G`. -/
theorem good_card_le {ι : Type} (G : Finset ι) (Wit : ι → T → Prop)
    (hinj : ∀ u ∈ G, ∀ u' ∈ G, ∀ x, Wit u x → Wit u' x → u = u') (P : Finset T) (hG : ∀ u ∈ G, ∃ x ∈ P, Wit u x) :
    G.card ≤ P.card := by
  simpa using good_work_le G Wit hinj P hG (fun _ => 1) (fun _ => 1) fun _ _ _ _ => rfl

/-- The count behind every bound here: of `u` units (or units of work) split into `g` good and `b` bad, with the good
at most `p < (1 − ε)·u`, at least `⌈ε·u⌉` are bad. -/
theorem ceil_le_of_deficit {g b u p : ℕ} {ε : ℝ} (hsplit : g + b = u) (hg : g ≤ p) (hd : (p : ℝ) < (1 - ε) * u) :
    ⌈ε * u⌉₊ ≤ b := by
  rw [Nat.ceil_le]
  have h1 : (g : ℝ) ≤ p := by exact_mod_cast hg
  have h2 : (g : ℝ) + b = u := by exact_mod_cast hsplit
  linarith

/-- `C(M − t, k)/C(M, k) ≤ ((M − t)/M)^k`, read in `ℝ` (`Law.choose_div_le_pow`). -/
theorem choose_div_le {M k t : ℕ} (hk : k ≤ M) (hM : 0 < M) :
    ((M - t).choose k : ℝ) / (M.choose k : ℝ) ≤ (((M - t : ℕ) : ℝ) / M) ^ k := by
  have h := NNReal.coe_le_coe.2 (Law.choose_div_le_pow (m := t) hk hM)
  simpa only [NNReal.coe_div, NNReal.coe_pow, NNReal.coe_natCast] using h

/-- `(M − ⌈ε·M⌉)/M ≤ 1 − ε` for `ε ≤ 1`. -/
theorem sub_ceil_div_le {M : ℕ} {ε : ℝ} (hε1 : ε ≤ 1) (hM : 0 < M) : ((M - ⌈ε * M⌉₊ : ℕ) : ℝ) / M ≤ 1 - ε := by
  have hMr : (0 : ℝ) < M := by exact_mod_cast hM
  have hc : ⌈ε * M⌉₊ ≤ M := Nat.ceil_le.2 (mul_le_of_le_one_left hMr.le hε1)
  rw [div_le_iff₀ hMr, Nat.cast_sub hc]
  have := Nat.le_ceil (ε * M)
  nlinarith

/-! ## One stratum -/

/-- **Credit over hidden positions, one stratum.** `k` positions drawn uniformly without replacement from the
verifier's `[0, M)` are all good (each has a witness in `P`, the good live units) while fewer than `(1 − ε)·M` units are
good, with probability at most `C(M − ⌈εM⌉, k)/C(M, k)`, for any witness relation in which no two positions share a
witness. So the credit `M` is at most `|P|/(1 − ε)` but with that probability. -/
theorem subset_credit {M k : ℕ} (hk : k ≤ M) (Wit : Fin M → T → Prop) (hinj : ∀ r r' x, Wit r x → Wit r' x → r = r')
    (P : Finset T) (ε : ℝ) :
    Claims.Pr (fun S : {S : Finset (Fin M) // S.card = k} =>
        (∀ r ∈ S.1, ∃ x ∈ P, Wit r x) ∧ (P.card : ℝ) < (1 - ε) * M) ≤
      ((M - ⌈ε * M⌉₊).choose k : ℝ) / (M.choose k : ℝ) := by
  classical
  by_cases hd : (P.card : ℝ) < (1 - ε) * M
  · have hsplit := Finset.card_filter_add_card_filter_not (s := (univ : Finset (Fin M)))
      (fun r => ∃ x ∈ P, Wit r x)
    rw [card_univ, Fintype.card_fin] at hsplit
    have hG := good_card_le (univ.filter fun r : Fin M => ∃ x ∈ P, Wit r x) Wit (fun u _ u' _ => hinj u u') P
      (fun u hu => (mem_filter.1 hu).2)
    refine (Claims.Pr_mono fun S hS => ?_).trans
      (Claims.Sampling.subset_miss_le hk (ceil_le_of_deficit hsplit hG hd))
    exact Finset.disjoint_left.2 fun r hr hrB => (mem_filter.1 hrB).2 (hS.1 r hr)
  · rw [Claims.Pr_congr fun S => iff_false_intro fun h => hd h.2, Claims.Pr_false]
    positivity

/-- **Credit over hidden positions, one stratum, at `(1 − ε)^k`.** -/
theorem subset_credit_pow {M k : ℕ} (hk : k ≤ M) (Wit : Fin M → T → Prop)
    (hinj : ∀ r r' x, Wit r x → Wit r' x → r = r') (P : Finset T) {ε : ℝ} (hε1 : ε ≤ 1) :
    Claims.Pr (fun S : {S : Finset (Fin M) // S.card = k} =>
        (∀ r ∈ S.1, ∃ x ∈ P, Wit r x) ∧ (P.card : ℝ) < (1 - ε) * M) ≤ (1 - ε) ^ k := by
  refine (subset_credit hk Wit hinj P ε).trans ?_
  rcases Nat.eq_zero_or_pos M with h0 | hM
  · subst h0
    obtain rfl : k = 0 := by omega
    simp
  · exact (choose_div_le hk hM).trans (pow_le_pow_left₀ (by positivity) (sub_ceil_div_le hε1 hM) k)

/-! ## A stratum of the stratified draw, and the window under the work law -/

/-- **Credit over hidden positions in stratum `s` of the stratified draw**: every drawn position of stratum `s` good
while fewer than `(1 − ε)·M_s` units are, with probability at most `C(M_s − ⌈εM_s⌉, k_s)/C(M_s, k_s)`, whatever the
other strata draw. -/
theorem stratified_credit {n m : ℕ} (σ : Fin n → Fin m) (k : Fin m → ℕ) (hk : ∀ s, k s ≤ (Law.stratum σ s).card)
    (s : Fin m) (Wit : Fin n → T → Prop) (hinj : ∀ r r' x, σ r = s → σ r' = s → Wit r x → Wit r' x → r = r')
    (P : Finset T) (ε : ℝ) :
    Claims.Pr (fun ω : (Law.stratified σ k hk).Ω =>
        (∀ r ∈ (Law.stratified σ k hk).draw ω, σ r = s → ∃ x ∈ P, Wit r x) ∧
          (P.card : ℝ) < (1 - ε) * (Law.stratum σ s).card) ≤
      (((Law.stratum σ s).card - ⌈ε * (Law.stratum σ s).card⌉₊).choose (k s) : ℝ) /
        ((Law.stratum σ s).card.choose (k s) : ℝ) := by
  classical
  by_cases hd : (P.card : ℝ) < (1 - ε) * (Law.stratum σ s).card
  · set B := (Law.stratum σ s).filter fun r => ¬ ∃ x ∈ P, Wit r x with hBdef
    have hsplit := Finset.card_filter_add_card_filter_not (s := Law.stratum σ s) (fun r => ∃ x ∈ P, Wit r x)
    have hG := good_card_le ((Law.stratum σ s).filter fun r => ∃ x ∈ P, Wit r x) Wit
      (fun u hu u' hu' x => hinj u u' x (Law.mem_stratum.1 (mem_filter.1 hu).1)
        (Law.mem_stratum.1 (mem_filter.1 hu').1)) P (fun u hu => (mem_filter.1 hu).2)
    have ht := ceil_le_of_deficit hsplit hG hd
    have hBs : B.filter (σ · = s) = B := Finset.filter_true_of_mem fun r hr => Law.mem_stratum.1 (mem_filter.1 hr).1
    set F : Fin m → ℝ := fun s' => (((Law.stratum σ s').card - (B.filter (σ · = s')).card).choose (k s') : ℝ) /
      ((Law.stratum σ s').card.choose (k s') : ℝ) with hF
    have hF0 : ∀ s', 0 ≤ F s' := fun _ => by positivity
    have hF1 : ∀ s', F s' ≤ 1 := fun s' => div_le_one_of_le₀
      (by exact_mod_cast Nat.choose_le_choose _ (Nat.sub_le _ _)) (by positivity)
    refine (Claims.Pr_mono fun ω hω => ?_).trans ((Claims.Sampling.stratified_miss_eq σ k hk B).trans_le ?_)
    · exact Finset.disjoint_left.2 fun r hr hrB =>
        (mem_filter.1 hrB).2 (hω.1 r hr (Law.mem_stratum.1 (mem_filter.1 hrB).1))
    · calc ∏ s', F s' = F s * ∏ s' ∈ univ.erase s, F s' := (Finset.mul_prod_erase univ F (mem_univ s)).symm
        _ ≤ F s * 1 := mul_le_mul_of_nonneg_left (Finset.prod_le_one₀ (fun s' _ => hF0 s') fun s' _ => hF1 s') (hF0 s)
        _ = F s := mul_one _
        _ ≤ _ := by
          simp only [hF, hBs]
          gcongr
  · rw [Claims.Pr_congr fun ω => iff_false_intro fun h => hd h.2, Claims.Pr_false]
    positivity

/-- **Credit over hidden positions, the window under the work law**: every drawn position good while the good units
do less than `(1 − ε)` of the population's work `W` (each unit doing the work of the positions it proves), with
probability at most `(1 − ε)^K` (`Claims.Sampling.work_miss_le`, `Law.work_escape_le`'s weighted AM-GM). -/
theorem work_credit {n m : ℕ} (σ : Fin n → Fin m) (w f : Fin m → ℕ) (K : ℕ) (hf : ∀ s, 1 ≤ f s)
    (hW : 0 < Law.totalWork σ w) (Wit : Fin n → T → Prop) (hinj : ∀ r r' x, Wit r x → Wit r' x → r = r')
    (P : Finset T) (pw : T → ℕ) (hpw : ∀ r x, Wit r x → pw x = w (σ r)) {ε : ℝ} (hε1 : ε ≤ 1) :
    Claims.Pr (fun ω : (Law.work σ w f K).Ω =>
        (∀ r ∈ (Law.work σ w f K).draw ω, ∃ x ∈ P, Wit r x) ∧
          ((∑ x ∈ P, pw x : ℕ) : ℝ) < (1 - ε) * Law.totalWork σ w) ≤ (1 - ε) ^ K := by
  classical
  by_cases hd : ((∑ x ∈ P, pw x : ℕ) : ℝ) < (1 - ε) * Law.totalWork σ w
  · set B := univ.filter fun r : Fin n => ¬ ∃ x ∈ P, Wit r x with hBdef
    have hsplit : Law.workOf σ w (univ.filter fun r => ∃ x ∈ P, Wit r x) + Law.workOf σ w B = Law.totalWork σ w :=
      Finset.sum_filter_add_sum_filter_not univ _ _
    have hG := good_work_le (univ.filter fun r : Fin n => ∃ x ∈ P, Wit r x) Wit (fun u _ u' _ => hinj u u') P
      (fun u hu => (mem_filter.1 hu).2) (fun r => w (σ r)) pw (fun r _ x hx => hpw r x hx)
    have ht := ceil_le_of_deficit hsplit hG hd
    refine (Claims.Pr_mono fun ω hω => ?_).trans ((Claims.Sampling.work_miss_le σ w f K hf hW ht).trans ?_)
    · exact Finset.disjoint_left.2 fun r hr hrB => (mem_filter.1 hrB).2 (hω.1 r hr)
    · exact pow_le_pow_left₀ (by positivity) (sub_ceil_div_le hε1 hW) K
  · rw [Claims.Pr_congr fun ω => iff_false_intro fun h => hd h.2, Claims.Pr_false]
    exact pow_nonneg (by linarith) K

/-! ## At the ε `epsAt` reports -/

/-- **Credit over hidden positions at the ε `epsAt` reports, one stratum**: where `PouwWindow.epsAt K δ M M 1` answers
`ε` (the stratum's `M` positions, each credited and of unit work), `K ≤ M` positions drawn uniformly without
replacement are all good while fewer than `(1 − ε)·M` units are, with probability at most `δ`. -/
theorem subset_credit_epsAt {M K : ℕ} (hK : K ≤ M) (Wit : Fin M → T → Prop)
    (hinj : ∀ r r' x, Wit r x → Wit r' x → r = r') (P : Finset T) {δ e : ℚ}
    (he : PouwWindow.epsAt K δ M M 1 = .ok e) :
    Claims.Pr (fun S : {S : Finset (Fin M) // S.card = K} =>
        (∀ r ∈ S.1, ∃ x ∈ P, Wit r x) ∧ (P.card : ℝ) < (1 - (e : ℝ)) * M) ≤ δ := by
  obtain ⟨k', hk', hk'K⟩ := Properties.Pouw.Strata.epsAt_ok he
  obtain ⟨he0, he1, hδ0, -, hM1, -⟩ := Properties.Pouw.Window.kPrimeRefuses _ _ _ _ _ _ hk'
  have hq := Properties.Pouw.Window.kPrimeLeast _ _ _ _ _ _ hk'
  dsimp only at hq
  obtain ⟨-, hk'M, hlt, -⟩ := hq
  simp only [Int.toNat_natCast] at hk'M hlt
  have hM : 0 < M := by omega
  have hk'K' : k' ≤ K := by exact_mod_cast hk'K
  have he0r : (0 : ℝ) < e := by exact_mod_cast he0
  have he1r : (e : ℝ) ≤ 1 := by exact_mod_cast he1
  rcases Nat.lt_or_ge k' M with hlt' | hge'
  · have hs : PouwWindow.missShare e M M 1 = e := by
      have : (M : ℚ) ≠ 0 := by exact_mod_cast hM.ne'
      unfold PouwWindow.missShare
      push_cast
      field_simp
    have hδq := hlt hlt'
    rw [hs] at hδq
    have hδ : (1 - (e : ℝ)) ^ k' ≤ δ := by exact_mod_cast hδq
    calc _ ≤ (1 - (e : ℝ)) ^ K := subset_credit_pow hK Wit hinj P he1r
      _ ≤ (1 - (e : ℝ)) ^ k' := pow_le_pow_of_le_one (by linarith) (by linarith) hk'K'
      _ ≤ δ := hδ
  · have hMr : (0 : ℝ) < M := by exact_mod_cast hM
    have hc : 0 < ⌈(e : ℝ) * M⌉₊ := Nat.ceil_pos.2 (mul_pos he0r hMr)
    calc _ ≤ _ := subset_credit hK Wit hinj P e
      _ = 0 := by rw [Nat.choose_eq_zero_of_lt (by omega), Nat.cast_zero, zero_div]
      _ ≤ δ := by exact_mod_cast hδ0.le

/-- **A draw's miss bound is a credit bound**: when the draw `D` misses a set of units doing at least `ε·C` of the work
with probability at most `δ`, every drawn position of `U` is good while the good units' work falls short of `U`'s by
more than `ε·C`, with probability at most `δ`, for any witness relation in which no two positions of `U` share a
witness, each unit doing the work of the positions it proves. -/
theorem credit_of_miss {Ω : Type} [Fintype Ω] (D : Ω → Option (Array ℕ)) (wt : ℕ → ℕ) (e δ : ℚ) (C : ℕ)
    (hδ : 0 ≤ δ)
    (hmiss : ∀ B : Finset ℕ, e * C ≤ ((∑ u ∈ B, wt u : ℕ) : ℚ) →
      prob (fun g => ∃ d, D g = some d ∧ ∀ u ∈ B, u ∉ d) ≤ (δ : ℝ))
    (U : Finset ℕ) (Wit : ℕ → T → Prop) (hinj : ∀ u ∈ U, ∀ u' ∈ U, ∀ x, Wit u x → Wit u' x → u = u')
    (P : Finset T) (pw : T → ℕ) (hpw : ∀ u ∈ U, ∀ x, Wit u x → pw x = wt u) :
    prob (fun g => (∃ d, D g = some d ∧ ∀ u ∈ U, u ∈ d → ∃ x ∈ P, Wit u x) ∧
        ((∑ x ∈ P, pw x : ℕ) : ℚ) + e * C < ((∑ u ∈ U, wt u : ℕ) : ℚ)) ≤ (δ : ℝ) := by
  classical
  by_cases hd : ((∑ x ∈ P, pw x : ℕ) : ℚ) + e * C < ((∑ u ∈ U, wt u : ℕ) : ℚ)
  · set B := U.filter fun u => ¬ ∃ x ∈ P, Wit u x with hBdef
    have hsplit := Finset.sum_filter_add_sum_filter_not U (fun u => ∃ x ∈ P, Wit u x) wt
    have hG := good_work_le (U.filter fun u => ∃ x ∈ P, Wit u x) Wit
      (fun u hu u' hu' => hinj u (mem_filter.1 hu).1 u' (mem_filter.1 hu').1) P (fun u hu => (mem_filter.1 hu).2)
      wt pw (fun u hu => hpw u (mem_filter.1 hu).1)
    have hB : e * C ≤ ((∑ u ∈ B, wt u : ℕ) : ℚ) := by
      have h1 : ((∑ u ∈ U.filter fun u => ∃ x ∈ P, Wit u x, wt u : ℕ) : ℚ) ≤ ((∑ x ∈ P, pw x : ℕ) : ℚ) := by
        exact_mod_cast hG
      have h2 : ((∑ u ∈ U.filter fun u => ∃ x ∈ P, Wit u x, wt u : ℕ) : ℚ) + ((∑ u ∈ B, wt u : ℕ) : ℚ) =
          ((∑ u ∈ U, wt u : ℕ) : ℚ) := by
        exact_mod_cast hsplit
      linarith
    refine le_trans ?_ (hmiss B hB)
    rw [Properties.Served.SecurityProofs.prob_eq_pr, Properties.Served.SecurityProofs.prob_eq_pr]
    refine Claims.Pr_mono fun g hg => ?_
    obtain ⟨⟨d, hD, hgood⟩, -⟩ := hg
    exact ⟨d, hD, fun u hu hud => (mem_filter.1 hu).2 (hgood u (mem_filter.1 hu).1 hud)⟩
  · rw [Properties.Served.SecurityProofs.prob_eq_pr, Claims.Pr_congr fun g => iff_false_intro fun h => hd h.2,
      Claims.Pr_false]
    exact_mod_cast hδ

/-- **Credit over hidden positions at the served draw, at the ε `epsAt` reports** (`EpsAtSound`'s form): wherever
`strata` accepts a layout with strata `ss`, `law` puts them under `work:K`, and `epsAt K δ C W 1` (`W` the strata's
work) answers `ε`, the draw over uniform streams (`drawnOn`) draws only good positions of `U` while the good units'
work falls short of `U`'s by more than `ε·C`, with probability at most `δ`. -/
theorem drawn_credit (layout : Json) (ss : Array Flock.Draw.WorkStratum) (K : ℕ) (strata : Array Flock.Draw.Stratum)
    (δ e : ℚ) (C : ℕ) (hss : PouwCheck.Credit.strata layout = .ok ss)
    (hlaw : PouwCheck.Credit.law ss (toJson K) = .ok strata)
    (he : PouwWindow.epsAt K δ C (Flock.Draw.totalWork ss) 1 = .ok e) (U : Finset ℕ) (Wit : ℕ → T → Prop)
    (hinj : ∀ u ∈ U, ∀ u' ∈ U, ∀ x, Wit u x → Wit u' x → u = u') (P : Finset T) (pw : T → ℕ)
    (hpw : ∀ u ∈ U, ∀ x, Wit u x → pw x = unitWork ss u) :
    prob (fun g => (∃ d, drawnOn strata g = some d ∧ ∀ u ∈ U, u ∈ d → ∃ x ∈ P, Wit u x) ∧
        ((∑ x ∈ P, pw x : ℕ) : ℚ) + e * C < ((∑ u ∈ U, unitWork ss u : ℕ) : ℚ)) ≤ (δ : ℝ) := by
  obtain ⟨k', hk', -⟩ := Properties.Pouw.Strata.epsAt_ok he
  obtain ⟨-, -, hδ0, -⟩ := Properties.Pouw.Window.kPrimeRefuses _ _ _ _ _ _ hk'
  exact credit_of_miss (drawnOn strata) (unitWork ss) e δ C hδ0.le
    (Properties.Pouw.Credit.SecurityProofs.epsAtSound layout ss K strata δ e C hss hlaw he) U Wit hinj P pw hpw

/-! ## At HiddenDraw's witnesses -/

/-- The live units of a calls table: a call `c` of `cs`, one of its live rows `j < rows`, and a tile `t < width`. -/
def liveUnits (width : ℕ) (cs : List Call) : Finset (ℕ × ℕ × ℕ) :=
  (range cs.length).biUnion fun c => (range (cs.getD c ⟨0, 0⟩).rows ×ˢ range width).image fun jt => (c, jt)

/-- Position `r` is proved by the unit `x` under the table `cs`: the prover names a call that covers `r`, and `x` is
its witness (`SampledProofs.HiddenDraw.witness`). -/
def Proves (width : ℕ) (cs : List Call) (r : ℕ) (x : ℕ × ℕ × ℕ) : Prop := ∃ c, witness width cs c r = some x

/-- Two positions never share a unit, whichever calls the prover names (`witness_inj`). -/
theorem proves_inj {width : ℕ} {cs : List Call} {r r' : ℕ} {x : ℕ × ℕ × ℕ} (h : Proves width cs r x)
    (h' : Proves width cs r' x) : r = r' := by
  obtain ⟨c, hc⟩ := h
  obtain ⟨c', hc'⟩ := h'
  exact witness_inj width cs c c' r r' x hc hc'

/-- A unit that proves a position is a live unit of the table (`witness_bounds`). -/
theorem proves_live {width : ℕ} {cs : List Call} {r : ℕ} {x : ℕ × ℕ × ℕ} (h : Proves width cs r x) :
    x ∈ liveUnits width cs := by
  obtain ⟨c, hc⟩ := h
  obtain ⟨call, hcall, h1, hlen, hj, ht⟩ := witness_bounds width cs c r x hc
  obtain ⟨c₀, j, t⟩ := x
  simp only at h1 hj ht
  subst h1
  have hget : cs.getD c₀ ⟨0, 0⟩ = call := by rw [List.getD_eq_getElem?_getD, hcall, Option.getD_some]
  refine Finset.mem_biUnion.2 ⟨c₀, Finset.mem_range.2 hlen, Finset.mem_image.2 ⟨(j, t), ?_, rfl⟩⟩
  rw [hget]
  exact Finset.mem_product.2 ⟨Finset.mem_range.2 hj, Finset.mem_range.2 ht⟩

/-- A drawn position with a correct unit has a witness among the table's correct live units. -/
theorem hidden_good {width : ℕ} {cs : List Call} {correct : ℕ × ℕ × ℕ → Prop} [DecidablePred correct] {r : ℕ}
    (h : ∃ x, Proves width cs r x ∧ correct x) : ∃ x ∈ (liveUnits width cs).filter correct, Proves width cs r x := by
  obtain ⟨x, hx, hc⟩ := h
  exact ⟨x, Finset.mem_filter.2 ⟨proves_live hx, hc⟩, hx⟩

/-- **Credit over hidden live rows** (HiddenDraw's witnesses), for any calls table `cs` the prover commits before the
draw coin and any `correct` its committed values fix: `k` positions drawn uniformly without replacement from the
verifier's `[0, M)` each have a witness `(c, j, t)` whose unit is correct, while fewer than `(1 − ε)·M` of the table's
live (call, row, tile) units are correct, with probability at most `C(M − ⌈εM⌉, k)/C(M, k)`. -/
theorem hidden_credit {M k : ℕ} (hk : k ≤ M) (width : ℕ) (cs : List Call) (correct : ℕ × ℕ × ℕ → Prop)
    [DecidablePred correct] (ε : ℝ) :
    Claims.Pr (fun S : {S : Finset (Fin M) // S.card = k} =>
        (∀ r ∈ S.1, ∃ x, Proves width cs r x ∧ correct x) ∧
          (((liveUnits width cs).filter correct).card : ℝ) < (1 - ε) * M) ≤
      ((M - ⌈ε * M⌉₊).choose k : ℝ) / (M.choose k : ℝ) :=
  (Claims.Pr_mono fun _ hS => ⟨fun r hr => hidden_good (hS.1 r hr), hS.2⟩).trans
    (subset_credit hk (fun r x => Proves width cs r x) (fun _ _ _ h h' => Fin.ext (proves_inj h h')) _ ε)

/-- **Credit over hidden live rows, at `(1 − ε)^k`.** -/
theorem hidden_credit_pow {M k : ℕ} (hk : k ≤ M) (width : ℕ) (cs : List Call) (correct : ℕ × ℕ × ℕ → Prop)
    [DecidablePred correct] {ε : ℝ} (hε1 : ε ≤ 1) :
    Claims.Pr (fun S : {S : Finset (Fin M) // S.card = k} =>
        (∀ r ∈ S.1, ∃ x, Proves width cs r x ∧ correct x) ∧
          (((liveUnits width cs).filter correct).card : ℝ) < (1 - ε) * M) ≤ (1 - ε) ^ k :=
  (Claims.Pr_mono fun _ hS => ⟨fun r hr => hidden_good (hS.1 r hr), hS.2⟩).trans
    (subset_credit_pow hk (fun r x => Proves width cs r x) (fun _ _ _ h h' => Fin.ext (proves_inj h h')) _ hε1)

/-- **Credit over hidden live rows, at the ε `PouwWindow.epsAt K δ M M 1` reports**: at most `δ`. -/
theorem hidden_credit_epsAt {M K : ℕ} (hK : K ≤ M) (width : ℕ) (cs : List Call) (correct : ℕ × ℕ × ℕ → Prop)
    [DecidablePred correct] {δ e : ℚ} (he : PouwWindow.epsAt K δ M M 1 = .ok e) :
    Claims.Pr (fun S : {S : Finset (Fin M) // S.card = K} =>
        (∀ r ∈ S.1, ∃ x, Proves width cs r x ∧ correct x) ∧
          (((liveUnits width cs).filter correct).card : ℝ) < (1 - (e : ℝ)) * M) ≤ δ :=
  (Claims.Pr_mono fun _ hS => ⟨fun r hr => hidden_good (hS.1 r hr), hS.2⟩).trans
    (subset_credit_epsAt hK (fun r x => Proves width cs r x) (fun _ _ _ h h' => Fin.ext (proves_inj h h')) _ he)

end Pouw.SecurityProofs.HiddenCredit
