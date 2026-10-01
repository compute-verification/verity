import Pous.P2.Assumptions
import PousProofs.P2.RingTight

/-!
# P2 in the ring model: the guarantee, from the joint bound

`a3_prime_on` is `PousTrackA.a3_prime` over the ring bound, with the good event `True`: Lemma A for every history,
a union bound and the chain rule give `Pr[pass] ≤ ((g − 1)/B)^k + |Hist|·Γ`. With `RingLemmaA` (the open joint bound,
`Pous/P2/Assumptions.lean`) at an effective domain of `⌊p/2^λ⌋` points, a row of the certificate gives `MeetsOn` for
every `(D, L)` that satisfies `c·p + (D + 1)·L²·2^(D+2) ≤ 2^λ·p` (`meetsOn_of_row`, parametric). At P2's parameters
the row is `k = 88`, `λ = 6`, `c = 2`, through `PousP2p.gammaP_le` at `⌊p/64⌋ = 2^16442 − 330` and the M1 row machinery
at `ℓ = 16,442` (`row_b19_ring`), and the largest `D` at `L = 2^64` is `16,309` (`meetsOn_b19_k88`).
-/

namespace PousP2Ring

open Pous PousP2 PousP2p PousThm1 Pous.P2.Assumptions
open scoped ENNReal

/-! ### A3′ over the ring bound -/

/-- **A3′ over the ring bound** (good event `True`): `PousTrackA.a3_prime`'s proof with `SeqResponderOn` and
`LemmaAOn`. -/
theorem a3_prime_on {B w p : ℕ} [NeZero p] (sch : Scheme (ringRO B w p)) (S D Q L k g : ℕ) (Γ : ℝ≥0∞)
    (hA : LemmaAOn sch S D Q L g Γ) (W : Bits sch.n) (A₁ : Preprocessor sch S) (A₂ : SeqResponderOn sch S D Q L) :
    pr (seqAcceptsOn sch k W A₁ A₂) ≤
      (((g - 1 : ℕ) : ℝ≥0∞) / sch.B) ^ k + (∑ j ∈ Finset.range k, (sch.B : ℝ≥0∞) ^ j) * Γ := by
  classical
  let f := fun ω : (ringRO B w p).Ω => (ringRO B w p).answer ω
  let st : List (Fin sch.B) → Bits sch.n → Bits sch.t → sch.Code → (ringRO B w p).Ω → Bits S :=
    fun h W pp C ω => seqState (f ω) (A₂.step W pp) 0 h (A₁ W pp C ω)
  let pred : List (Fin sch.B) → ProdResponderOn sch S D Q L := fun h =>
    ⟨fun W pp σ i => progMap Prod.fst (A₂.step W pp h.length σ i),
      fun W pp σ i => ringBounded_progMap _ (A₂.bounded _ _ _ _ _)⟩
  let good : List (Fin sch.B) → (ringRO B w p).Ω × sch.Coins → Fin sch.B → Prop := fun h a i =>
    ((pred h).prog W (sch.enc W a.1 a.2).1 (st h W (sch.enc W a.1 a.2).1 (sch.enc W a.1 a.2).2 a.1) i).run
      (f a.1) = (sch.enc W a.1 a.2).2 i
  let E : (ringRO B w p).Ω × sch.Coins → Prop := fun a =>
    ∃ j ∈ Finset.range k, ∃ v ∈ (Finset.univ : Finset (Fin j → Fin sch.B)),
      g ≤ (Finset.univ.filter fun i => good (List.ofFn v) a i).card
  have hre : pr (seqAcceptsOn sch k W A₁ A₂) =
      pr (fun x : ((ringRO B w p).Ω × sch.Coins) × (Fin k → Fin sch.B) =>
        seqAnswers (f x.1.1) (A₂.step W (sch.enc W x.1.1 x.1.2).1) 0 (List.ofFn x.2)
          (A₁ W (sch.enc W x.1.1 x.1.2).1 (sch.enc W x.1.1 x.1.2).2 x.1.1) =
        List.ofFn fun t => (sch.enc W x.1.1 x.1.2).2 (x.2 t)) :=
    pr_equiv (Equiv.prodAssoc _ _ _).symm (fun x : ((ringRO B w p).Ω × sch.Coins) × (Fin k → Fin sch.B) =>
      seqAnswers (f x.1.1) (A₂.step W (sch.enc W x.1.1 x.1.2).1) 0 (List.ofFn x.2)
        (A₁ W (sch.enc W x.1.1 x.1.2).1 (sch.enc W x.1.1 x.1.2).2 x.1.1) =
      List.ofFn fun t => (sch.enc W x.1.1 x.1.2).2 (x.2 t))
  rw [hre]
  have : Nonempty (Fin sch.B) := ⟨⟨0, sch.B_pos⟩⟩
  refine (pr_prod_le (α := (ringRO B w p).Ω × sch.Coins) (β := Fin k → Fin sch.B)
    (fun a I => seqAnswers (f a.1) (A₂.step W (sch.enc W a.1 a.2).1) 0 (List.ofFn I)
        (A₁ W (sch.enc W a.1 a.2).1 (sch.enc W a.1 a.2).2 a.1) =
      List.ofFn fun t => (sch.enc W a.1 a.2).2 (I t))
    E ((((g - 1 : ℕ) : ℝ≥0∞) / sch.B) ^ k) fun a ha => ?_).trans ?_
  · refine pr_seq_le _ _ _ _ k 0 _ fun h hh => ?_
    have hlt : (Finset.univ.filter fun i => good h a i).card ≤ g - 1 := by
      have : ¬ g ≤ (Finset.univ.filter fun i => good h a i).card := by
        intro hg
        refine ha ⟨h.length, Finset.mem_range.2 hh, h.get, Finset.mem_univ _, ?_⟩
        rwa [List.ofFn_get]
      omega
    have hpr : pr (good h a) ≤ ((g - 1 : ℕ) : ℝ≥0∞) / sch.B := by
      rw [pr_def, Fintype.card_fin]
      exact ENNReal.div_le_div_right (by exact_mod_cast hlt) _
    simpa only [good, pred, st, run_progMap, Nat.zero_add] using hpr
  · have hE : pr E ≤ (∑ j ∈ Finset.range k, (sch.B : ℝ≥0∞) ^ j) * Γ := by
      calc pr E
          ≤ ∑ j ∈ Finset.range k, pr (fun a => ∃ v ∈ (Finset.univ : Finset (Fin j → Fin sch.B)),
            g ≤ (Finset.univ.filter fun i => good (List.ofFn v) a i).card) := pr_exists_le _ _
        _ ≤ ∑ j ∈ Finset.range k, ∑ _v : Fin j → Fin sch.B, Γ := by
            refine Finset.sum_le_sum fun j _ => (pr_exists_le _ _).trans (Finset.sum_le_sum fun v _ => ?_)
            have := hA W (st (List.ofFn v)) (pred (List.ofFn v))
            convert this using 3
        _ = (∑ j ∈ Finset.range k, (sch.B : ℝ≥0∞) ^ j) * Γ := by
            rw [Finset.sum_mul]
            refine Finset.sum_congr rfl fun j _ => ?_
            rw [Finset.sum_const, Finset.card_univ, Fintype.card_fun, Fintype.card_fin, Fintype.card_fin,
              nsmul_eq_mul, Nat.cast_pow]
    calc pr E + (((g - 1 : ℕ) : ℝ≥0∞) / sch.B) ^ k
        ≤ (∑ j ∈ Finset.range k, (sch.B : ℝ≥0∞) ^ j) * Γ + (((g - 1 : ℕ) : ℝ≥0∞) / sch.B) ^ k :=
          add_le_add hE le_rfl
      _ = _ := add_comm _ _

/-! ### The guarantee, parametric in `(D, L)` -/

theorem histCount_cast (B k : ℕ) : (histCount B k : ℝ≥0∞) = ∑ j ∈ Finset.range k, (B : ℝ≥0∞) ^ j := by
  rw [histCount]
  push_cast
  rfl

/-- **The guarantee in the ring model, from the joint bound.** For every prime `p ≡ 3 mod 4` with `2^b ≤ p ≤ 2^w`
and `20·w ≤ 21·b`, if `RingLemmaA` holds with constant `c`, then for every `(D, L)` with
`c·p + (D + 1)·L²·2^(D+2) ≤ 2^λ·p`, every key-query budget `Q` and every `k` for which the certificate's row holds at
the effective domain `⌊p/2^λ⌋`, `p2ring` meets the requirement over the ring bound at `(D, Q, L)`, sequential reveal,
`k` challenges and `ε = 2^-128`. Hardware enters only through `D`, `Q` and `L`. -/
theorem meetsOn_of_row (B w p b c : ℕ) [NeZero B] [NeZero p] (hp : p.Prime) (h4 : p % 4 = 3) (hb : 2 ^ b ≤ p)
    (hp2 : p ≤ 2 ^ w) (hsp : 20 * w ≤ 21 * b) (hA : RingLemmaA B w p b c) (D Q L k lam : ℕ)
    (hDL : c * p + (D + 1) * L ^ 2 * 2 ^ (D + 2) ≤ 2 ^ lam * p)
    (hrow : RingRow B (p / 2 ^ lam) ((p2ring B w p b).stateBound Params.ρ) Q k) :
    MeetsOn (p2ring B w p b) .sequential D Q L k Params.εMax := by
  obtain ⟨g, -, -, hpow, hΓ⟩ := hrow
  refine ⟨le_rfl, p2ring_correct hp h4 hb hp2, p2ring_codeHoldsW (le_of_two_pow_le (hb.trans hp2)),
    p2ring_spaceBound hsp, ?_⟩
  intro W A₁ A₂
  refine (a3_prime_on (p2ring B w p b) _ D Q L k g _ (hA _ D Q L g lam hDL) W A₁ A₂).trans (add_le_add ?_ ?_)
  · have hB0 : (0 : ℝ) < B := by exact_mod_cast Nat.pos_of_ne_zero (NeZero.ne B)
    change ((((g - 1 : ℕ) : ℝ≥0∞) / (B : ℝ≥0∞)) ^ k) ≤ ENNReal.ofReal ((1 / 100 : ℚ) : ℝ)
    rw [← ENNReal.ofReal_natCast, ← ENNReal.ofReal_natCast B, ← ENNReal.ofReal_div_of_pos hB0,
      ← ENNReal.ofReal_pow (by positivity)]
    refine ENNReal.ofReal_le_ofReal ?_
    have := (Rat.cast_le (K := ℝ)).2 hpow
    push_cast at this ⊢
    exact this
  · rw [← histCount_cast]
    exact hΓ

/-! ### The row at `k = 88`, `λ = 6`, `B = 2^19` -/

/-- `S = ⌊(18/19)·2^19·16448⌋`, whatever `p` and `b`. -/
theorem stateBound_p2ring_b19 (p b : ℕ) [NeZero p] :
    (p2ring (2 ^ 19) 16448 p b).stateBound Params.ρ = 8169621180 := by
  rw [Scheme.stateBound, Nat.floor_eq_iff (mul_nonneg (by norm_num [Params.ρ]) (Nat.cast_nonneg _))]
  change ((8169621180 : ℕ) : ℚ) ≤ 18 / 19 * ((2 ^ 19 * 16448 : ℕ) : ℚ) ∧
    18 / 19 * ((2 ^ 19 * 16448 : ℕ) : ℚ) < (8169621180 : ℕ) + 1
  norm_num

open Real in
/-- The certificate's log bound at the effective domain `2^16442`: `log₂(|Hist|·Γ) ≤ −16,088` at `k = 88`,
`g = 497,557`, `T = 2^20` and one extra state bit. -/
theorem log_b19_ring : logHistGammaM1 (2 ^ 19) (8169621180 + 1) (2 ^ 20) 497557 88 (2 ^ 16442) ≤ -16088 := by
  refine (logHistGammaM1_le _ _ _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num)).trans ?_
  rw [logb_two_pow_nat, logb_two_pow_nat]
  have hl := log_two_gt_d9
  have h : ((497557 : ℝ) + 1) / log 2 ≤ (497557 + 1) / 0.6931471803 :=
    div_le_div_of_nonneg_left (by positivity) (by norm_num) hl.le
  push_cast at h ⊢
  norm_num at h ⊢
  linarith

set_option exponentiation.threshold 20000 in
/-- `⌊p/64⌋ = 2^16442 − 330` at P2's modulus, so `2^16442 ≤ ⌊p/64⌋ + 330`. -/
theorem p16448_div64 : 0 < (2 ^ 16448 - 21065) / 2 ^ 6 ∧ 2 ^ 16442 ≤ (2 ^ 16448 - 21065) / 2 ^ 6 + 330 := by
  decide +kernel

/-- **The row at `k = 88`** for `2^19` blocks of P2's modulus at `λ = 6` (an effective domain of `⌊p/64⌋` points),
`T = 2^20` key queries, with `g = 497,557`, the largest `g` the 1% inequality allows. -/
theorem row_b19_ring : RingRow (2 ^ 19) ((2 ^ 16448 - 21065) / 2 ^ 6) 8169621180 (2 ^ 20) 88 := by
  refine ⟨497557, by norm_num, by norm_num, by norm_num, ?_⟩
  have hg : gammaM1 (2 ^ 19) 8169621180 (2 ^ 20) 497557 ((2 ^ 16448 - 21065) / 2 ^ 6) ≤
      gammaA2 (2 ^ 19) 16442 (8169621180 + 1) (2 ^ 20) 497557 :=
    gammaP_le _ 16442 _ 330 _ _ _ p16448_div64.1 p16448_div64.2 (le_two_pow_of _ 29 16442 (by norm_num) (by norm_num))
  rw [histCount_cast]
  exact (mul_le_mul_right hg _).trans
    (hist_gamma_le _ 16442 _ _ _ 88 (by norm_num) (by norm_num) (log_b19_ring.trans (by norm_num)))

set_option exponentiation.threshold 20000 in
/-- `D_cert = 16,309` at `L = 2^64` and `c = 2`: the inequality holds at `16,309` and fails at `16,310`. -/
theorem dcert_b19 : 2 * (2 ^ 16448 - 21065) + (16309 + 1) * (2 ^ 64) ^ 2 * 2 ^ (16309 + 2) ≤
      2 ^ 6 * (2 ^ 16448 - 21065) ∧
    ¬ 2 * (2 ^ 16448 - 21065) + (16310 + 1) * (2 ^ 64) ^ 2 * 2 ^ (16310 + 2) ≤ 2 ^ 6 * (2 ^ 16448 - 21065) := by
  decide +kernel

/-- **P2's guarantee at `k = 88`, from the joint bound.** At `2^19` blocks of P2's modulus `p = 2^16448 − 21065`,
for every payload width `b` with `20·16448 ≤ 21·b` and `b ≤ 16,447`, if `p` is prime (`PrimeP16448`) and the joint
bound holds with `c = 2` (`RingLemmaAP2`): `p2ring` meets the requirement over the ring bound, sequential reveal,
`k = 88`, `2^20` key queries and `ε = 2^-128`, at every `(D, L)` with `2p + (D + 1)·L²·2^(D+2) ≤ 64p`; in particular
at `D = 16,309` and `L = 2^64`, the largest `D` the inequality allows there (`dcert_b19`). Recomputation breaks it at
`D ≥ 32,894` (`tight_p16448`); between the two it is open. -/
theorem meetsOn_b19_k88 (hP : PrimeP16448) (hA : RingLemmaAP2) (b : ℕ) (hsp : 20 * 16448 ≤ 21 * b)
    (hb : b ≤ 16447) :
    (∀ D L, 2 * (2 ^ 16448 - 21065) + (D + 1) * L ^ 2 * 2 ^ (D + 2) ≤ 2 ^ 6 * (2 ^ 16448 - 21065) →
      MeetsOn (p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) b) .sequential D (2 ^ 20) L 88 Params.εMax) ∧
    MeetsOn (p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) b) .sequential 16309 (2 ^ 20) (2 ^ 64) 88 Params.εMax := by
  have h : ∀ D L, 2 * (2 ^ 16448 - 21065) + (D + 1) * L ^ 2 * 2 ^ (D + 2) ≤ 2 ^ 6 * (2 ^ 16448 - 21065) →
      MeetsOn (p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) b) .sequential D (2 ^ 20) L 88 Params.εMax := by
    intro D L hDL
    refine meetsOn_of_row (2 ^ 19) 16448 _ b 2 hP p16448_mod_four (two_pow_le_of_le hb hyps_16448.1)
      hyps_16448.2.1 hsp (hA b hsp hb) D (2 ^ 20) L 88 6 hDL ?_
    rw [stateBound_p2ring_b19]
    exact row_b19_ring
  exact ⟨h, h 16309 (2 ^ 64) dcert_b19.1⟩

/-! ### The joint bound is satisfiable -/

/-- `Γ ≥ 1` once the domain has at most one point and `g ≤ B`. -/
theorem one_le_gammaM1 (B S T g M : ℕ) (hgB : g ≤ B) (hM : M ≤ 1) : 1 ≤ gammaM1 B S T g M := by
  unfold gammaM1
  have h1 : (1 : ℝ≥0∞) ≤ 2 ^ S := one_le_pow₀ one_le_two
  have h2 : (1 : ℝ≥0∞) ≤ (B.choose g : ℝ≥0∞) := by exact_mod_cast Nat.choose_pos hgB
  have h3 : (1 : ℝ≥0∞) ≤ ∑ h ∈ Finset.range (g + 1), (((g + 1) * T).choose h : ℝ≥0∞) :=
    le_trans (by simp) (Finset.single_le_sum (fun _ _ => zero_le) (Finset.mem_range.2 (Nat.succ_pos g)))
  have hden : (M : ℝ≥0∞) ^ g ≤ 1 := pow_le_one₀ (zero_le) (by exact_mod_cast hM)
  have hnum : (1 : ℝ≥0∞) ≤ 2 ^ S * (B.choose g) * ∑ h ∈ Finset.range (g + 1), (((g + 1) * T).choose h : ℝ≥0∞) :=
    one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le h1 h2) h3
  exact hnum.trans ((div_one _).symm.le.trans (ENNReal.div_le_div_left hden _))

/-- **`RingLemmaA` is satisfiable** (at `c = p`): its hypothesis then forces `p ≤ 2^λ`, so the domain `⌊p/2^λ⌋` has at
most one point, and `Γ ≥ 1` whenever `g ≤ B`, while no responder answers more than `B` blocks. -/
theorem ringLemmaA_sat (B w p b : ℕ) [NeZero B] [NeZero p] : RingLemmaA B w p b p := by
  intro S D Q L g lam hDL W A₁ A₂
  have hp0 : 0 < p := Nat.pos_of_ne_zero (NeZero.ne p)
  have hpl : p ≤ 2 ^ lam := Nat.le_of_mul_le_mul_right (le_of_add_le_left hDL) hp0
  have hM : p / 2 ^ lam ≤ 1 := Nat.div_le_of_le_mul (by rw [mul_one]; exact hpl)
  by_cases hgB : g ≤ B
  · exact (pr_le_one _).trans (one_le_gammaM1 B S Q g _ hgB hM)
  · refine (pr_mono fun x hx => ?_).trans (pr_false.trans_le zero_le)
    exact absurd ((hx.trans (Finset.card_filter_le _ _)).trans_eq (Finset.card_fin _)) hgB
