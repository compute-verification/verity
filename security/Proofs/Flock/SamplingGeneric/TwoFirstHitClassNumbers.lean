module

public import Proofs.Flock.SamplingGeneric.TwoFirstHitClassAt
public import Proofs.Flock.Claims.NumbersFinder
import Proofs.Flock.SamplingGeneric.Fails
import Proofs.Flock.Claims.Prob

/-! The by-class first-hit bound at numbers; that the budget, `e_s`, the three counts and `n_PU·openCalls` are within its numbers is each instance's hypothesis. -/

@[expose] public section

namespace Claims.Flock.Numbers

def hashBound (T e Ns Ks Km c : ℚ) : ℚ :=
  (2 * (T * (1 + e * Ns) + c) + 4 * firstHitCalls T e Ns Ks Km c) / 2 ^ 256

theorem hashBound_cast (T e Ns Ks Km c : ℚ) :
    ((hashBound T e Ns Ks Km c : ℚ) : ℝ) =
      (2 * ((T : ℝ) * (1 + (e : ℝ) * (Ns : ℝ)) + (c : ℝ)) +
        4 * ((T : ℝ) * (1 + (e : ℝ) * (Ns : ℝ) + (Ks : ℝ) + (e : ℝ) * (Km : ℝ)) + (1 + (Ks : ℝ)) * (c : ℝ))) /
        2 ^ (256 : ℝ) := by
  rw [show (256 : ℝ) = ((256 : ℕ) : ℝ) by norm_num, Real.rpow_natCast]
  simp only [hashBound, firstHitCalls]
  push_cast
  all_goals ring

theorem real_hash_le {T e Ns Ks Km c : ℝ} {T' e' Ns' Ks' Km' c' b : ℚ} (hT0 : 0 ≤ T) (he0 : 0 ≤ e) (hNs0 : 0 ≤ Ns)
    (hKs0 : 0 ≤ Ks) (hKm0 : 0 ≤ Km) (hc0 : 0 ≤ c) (hT : T ≤ T') (he : e ≤ e') (hNs : Ns ≤ Ns') (hKs : Ks ≤ Ks')
    (hKm : Km ≤ Km') (hc : c ≤ c') (h : hashBound T' e' Ns' Ks' Km' c' ≤ b) :
    (2 * (T * (1 + e * Ns) + c) + 4 * (T * (1 + e * Ns + Ks + e * Km) + (1 + Ks) * c)) / 2 ^ (256 : ℝ) ≤ (b : ℝ) := by
  have hT' : (0 : ℝ) ≤ T' := hT0.trans hT
  have he' : (0 : ℝ) ≤ e' := he0.trans he
  have hKs' : (0 : ℝ) ≤ Ks' := hKs0.trans hKs
  have h1 : e * Ns ≤ (e' : ℝ) * (Ns' : ℝ) := mul_le_mul he hNs hNs0 he'
  have h2 : e * Km ≤ (e' : ℝ) * (Km' : ℝ) := mul_le_mul he hKm hKm0 he'
  have h3 : 0 ≤ e * Ns := mul_nonneg he0 hNs0
  have h4 : 0 ≤ e * Km := mul_nonneg he0 hKm0
  have h5 : T * (1 + e * Ns + Ks + e * Km) ≤
      (T' : ℝ) * (1 + (e' : ℝ) * (Ns' : ℝ) + (Ks' : ℝ) + (e' : ℝ) * (Km' : ℝ)) :=
    mul_le_mul hT (by linarith) (by linarith) hT'
  have h6 : (1 + Ks) * c ≤ (1 + (Ks' : ℝ)) * (c' : ℝ) := mul_le_mul (by linarith) hc hc0 (by linarith)
  have h8 : T * (1 + e * Ns) ≤ (T' : ℝ) * (1 + (e' : ℝ) * (Ns' : ℝ)) :=
    mul_le_mul hT (by linarith) (by linarith) hT'
  have h7 : 2 * (T * (1 + e * Ns) + c) + 4 * (T * (1 + e * Ns + Ks + e * Km) + (1 + Ks) * c) ≤
      2 * ((T' : ℝ) * (1 + (e' : ℝ) * (Ns' : ℝ)) + (c' : ℝ)) +
        4 * ((T' : ℝ) * (1 + (e' : ℝ) * (Ns' : ℝ) + (Ks' : ℝ) + (e' : ℝ) * (Km' : ℝ)) +
          (1 + (Ks' : ℝ)) * (c' : ℝ)) := by
    linarith
  refine le_trans ?_ (Rat.cast_le.2 h)
  rw [hashBound_cast]
  exact div_le_div_of_nonneg_right h7 (by positivity)

theorem hashBound_table1 : hashBound (2 ^ 80) 2592237061 (2 ^ 30) (2 ^ 42) (2 ^ 42) (2 ^ 80) ≤ 1 / 2 ^ 100 := by
  norm_num [hashBound, firstHitCalls]

theorem hashBound_table1_513 :
    hashBound (513 * 2 ^ 80) 2592237061 (2 ^ 30) (2 ^ 33) (2 ^ 33) (513 * 2 ^ 80) ≤ 1 / 2 ^ 100 := by
  norm_num [hashBound, firstHitCalls]

theorem hashBound_table1_513_k35 :
    hashBound (513 * 2 ^ 80) 2592237061 (2 ^ 30) (2 ^ 35) (2 ^ 35) (513 * 2 ^ 80) ≤ 1 / 2 ^ 98 := by
  norm_num [hashBound, firstHitCalls]

theorem hashBound_table1_513_k36 :
    hashBound (513 * 2 ^ 80) 2592237061 (2 ^ 30) (2 ^ 36) (2 ^ 36) (513 * 2 ^ 80) ≤ 1 / 2 ^ 97 := by
  norm_num [hashBound, firstHitCalls]

theorem hashBound_table1_513_k37 :
    hashBound (513 * 2 ^ 80) 2592237061 (2 ^ 30) (2 ^ 37) (2 ^ 37) (513 * 2 ^ 80) ≤ 1 / 2 ^ 96 := by
  norm_num [hashBound, firstHitCalls]

theorem hashBound_table1_513_narrow :
    hashBound (513 * 2 ^ 80) 2592237061 (2 ^ 27) (14 * 10 ^ 9) (14 * 10 ^ 9) (513 * 2 ^ 80) ≤ 1 / 2 ^ 100 := by
  norm_num [hashBound, firstHitCalls]

theorem cast_two_pow_neg_98 : ((1 / 2 ^ 98 : ℚ) : ℝ) = 2 ^ (-(98 : ℝ)) := by
  rw [Real.rpow_neg (by norm_num), show (98 : ℝ) = ((98 : ℕ) : ℝ) by norm_num, Real.rpow_natCast]
  push_cast
  all_goals ring

end Claims.Flock.Numbers

namespace Claims.SamplingGeneric

open Claims Finset
open Claims.Sampling (AuditStatement Partition InScopeTwo)
open Claims.Flock.Numbers (hashBound real_hash_le hashBound_table1 hashBound_table1_513 hashBound_table1_513_k35
  hashBound_table1_513_narrow cast_two_pow_neg_100 cast_two_pow_neg_98)

namespace CompiledTwo

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

theorem εclsAt_le_of_hashBound (H : Msg → Msg) (L : CompiledTwo Coin A) (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) {T : ℕ} {Ns Ks Km : ℝ} {T' e' Ns' Ks' Km' c' b : ℚ} (hNs0 : 0 ≤ Ns) (hKs0 : 0 ≤ Ks)
    (hKm0 : 0 ≤ Km) (hT : (T : ℝ) ≤ T') (he : L.toCompiled.es x ≤ e') (hNs : Ns ≤ Ns') (hKs : Ks ≤ Ks')
    (hKm : Km ≤ Km') (hown : (L.toCompiled.nPU x : ℝ) * L.toCompiled.openCalls x ≤ c')
    (h : hashBound T' e' Ns' Ks' Km' c' ≤ b) :
    L.εclsAt H x P T Ns Ks Km ≤ L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) + (b : ℝ) := by
  have he0 : 0 ≤ L.toCompiled.es x := maxBelow_nonneg _ _
  have hc0 : (0 : ℝ) ≤ (L.toCompiled.nPU x : ℝ) * L.toCompiled.openCalls x :=
    mul_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _)
  unfold εclsAt Compiled.runCost Compiled.firstHitCost
  exact add_le_add (le_refl _) (real_hash_le (Nat.cast_nonneg T) he0 hNs0 hKs0 hKm0 hc0 hT he hNs hKs hKm hown h)

theorem εclsAt_le_table1 (H : Msg → Msg) (L : CompiledTwo Coin A) (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) {T : ℕ} {Ns Ks Km : ℝ} (hNs0 : 0 ≤ Ns) (hKs0 : 0 ≤ Ks) (hKm0 : 0 ≤ Km) (hT : T ≤ 2 ^ 80)
    (he : L.toCompiled.es x ≤ 2592237061) (hNs : Ns ≤ 2 ^ 30) (hKs : Ks ≤ 2 ^ 42) (hKm : Km ≤ 2 ^ 42)
    (hown : L.toCompiled.nPU x * L.toCompiled.openCalls x ≤ 2 ^ 80) :
    L.εclsAt H x P T Ns Ks Km ≤ L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) + 2 ^ (-(100 : ℝ)) := by
  rw [← cast_two_pow_neg_100]
  refine εclsAt_le_of_hashBound H L x P (T' := 2 ^ 80) (e' := 2592237061) (Ns' := 2 ^ 30) (Ks' := 2 ^ 42)
    (Km' := 2 ^ 42) (c' := 2 ^ 80) hNs0 hKs0 hKm0 ?_ ?_ ?_ ?_ ?_ ?_ hashBound_table1
  · exact_mod_cast hT
  · rw [Rat.cast_ofNat]
    exact he
  · rw [Rat.cast_pow, Rat.cast_ofNat]
    exact hNs
  · rw [Rat.cast_pow, Rat.cast_ofNat]
    exact hKs
  · rw [Rat.cast_pow, Rat.cast_ofNat]
    exact hKm
  · exact_mod_cast hown

theorem εclsAt_le_table1_513 (H : Msg → Msg) (L : CompiledTwo Coin A) (x : Msg) {Ω : Type} [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) {T : ℕ} {Ns Ks Km : ℝ} (hNs0 : 0 ≤ Ns) (hKs0 : 0 ≤ Ks) (hKm0 : 0 ≤ Km)
    (hT : T ≤ 513 * 2 ^ 80) (he : L.toCompiled.es x ≤ 2592237061) (hNs : Ns ≤ 2 ^ 30) (hKs : Ks ≤ 2 ^ 33)
    (hKm : Km ≤ 2 ^ 33) (hown : L.toCompiled.nPU x * L.toCompiled.openCalls x ≤ 513 * 2 ^ 80) :
    L.εclsAt H x P T Ns Ks Km ≤ L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) + 2 ^ (-(100 : ℝ)) := by
  rw [← cast_two_pow_neg_100]
  refine εclsAt_le_of_hashBound H L x P (T' := 513 * 2 ^ 80) (e' := 2592237061) (Ns' := 2 ^ 30) (Ks' := 2 ^ 33)
    (Km' := 2 ^ 33) (c' := 513 * 2 ^ 80) hNs0 hKs0 hKm0 ?_ ?_ ?_ ?_ ?_ ?_ hashBound_table1_513
  · exact_mod_cast hT
  · rw [Rat.cast_ofNat]
    exact he
  · rw [Rat.cast_pow, Rat.cast_ofNat]
    exact hNs
  · rw [Rat.cast_pow, Rat.cast_ofNat]
    exact hKs
  · rw [Rat.cast_pow, Rat.cast_ofNat]
    exact hKm
  · exact_mod_cast hown

theorem εclsAt_le_table1_513_narrow (H : Msg → Msg) (L : CompiledTwo Coin A) (x : Msg) {Ω : Type} [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) {T : ℕ} {Ns Ks Km : ℝ} (hNs0 : 0 ≤ Ns) (hKs0 : 0 ≤ Ks) (hKm0 : 0 ≤ Km)
    (hT : T ≤ 513 * 2 ^ 80) (he : L.toCompiled.es x ≤ 2592237061) (hNs : Ns ≤ 2 ^ 27) (hKs : Ks ≤ 14 * 10 ^ 9)
    (hKm : Km ≤ 14 * 10 ^ 9) (hown : L.toCompiled.nPU x * L.toCompiled.openCalls x ≤ 513 * 2 ^ 80) :
    L.εclsAt H x P T Ns Ks Km ≤ L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) + 2 ^ (-(100 : ℝ)) := by
  rw [← cast_two_pow_neg_100]
  refine εclsAt_le_of_hashBound H L x P (T' := 513 * 2 ^ 80) (e' := 2592237061) (Ns' := 2 ^ 27) (Ks' := 14 * 10 ^ 9)
    (Km' := 14 * 10 ^ 9) (c' := 513 * 2 ^ 80) hNs0 hKs0 hKm0 ?_ ?_ ?_ ?_ ?_ ?_ hashBound_table1_513_narrow
  · exact_mod_cast hT
  · rw [Rat.cast_ofNat]
    exact he
  · rw [Rat.cast_pow, Rat.cast_ofNat]
    exact hNs
  · exact_mod_cast hKs
  · exact_mod_cast hKm
  · exact_mod_cast hown

theorem εclsAt_le_table1_513_k35 (H : Msg → Msg) (L : CompiledTwo Coin A) (x : Msg) {Ω : Type} [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) {T : ℕ} {Ns Ks Km : ℝ} (hNs0 : 0 ≤ Ns) (hKs0 : 0 ≤ Ks) (hKm0 : 0 ≤ Km)
    (hT : T ≤ 513 * 2 ^ 80) (he : L.toCompiled.es x ≤ 2592237061) (hNs : Ns ≤ 2 ^ 30) (hKs : Ks ≤ 2 ^ 35)
    (hKm : Km ≤ 2 ^ 35) (hown : L.toCompiled.nPU x * L.toCompiled.openCalls x ≤ 513 * 2 ^ 80) :
    L.εclsAt H x P T Ns Ks Km ≤ L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) + 2 ^ (-(98 : ℝ)) := by
  rw [← cast_two_pow_neg_98]
  refine εclsAt_le_of_hashBound H L x P (T' := 513 * 2 ^ 80) (e' := 2592237061) (Ns' := 2 ^ 30) (Ks' := 2 ^ 35)
    (Km' := 2 ^ 35) (c' := 513 * 2 ^ 80) hNs0 hKs0 hKm0 ?_ ?_ ?_ ?_ ?_ ?_ hashBound_table1_513_k35
  · exact_mod_cast hT
  · rw [Rat.cast_ofNat]
    exact he
  · rw [Rat.cast_pow, Rat.cast_ofNat]
    exact hNs
  · rw [Rat.cast_pow, Rat.cast_ofNat]
    exact hKs
  · rw [Rat.cast_pow, Rat.cast_ofNat]
    exact hKm
  · exact_mod_cast hown

theorem eventTwoX_le_cls_of_hashBound {H : Msg → Msg} {L : CompiledTwo Coin A} (hA : A.KS H)
    (hbind : L.commit.Binds H) (hS : L.Sessions H) (hopen : L.toCompiled.OpenCallsLe H) {x : Msg}
    {s : AuditStatement} {Pt : Partition} {d : Sampling.Design.Two} (hs : L.parse x = some s)
    (hPt : L.partition s = some Pt) (hd : L.two s = some d) (hin : InScopeTwo Pt d s) {Ω : Type} [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) (needs : ℕ → Finset ℕ)
    (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.2.pos ∈ needs u) (cls : Fin (L.nG x) → Fin (L.nG x))
    (hrep : ∀ u (g : Fin (L.nG x)), g.val ∈ needs u → (cls g).val ∈ needs u)
    (hallX : ∀ (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (g g' : Fin (L.nG x)), cls g = cls g' →
      (L.opensAt H P x a.1 a.2.1 a.2.2 g).isSome → (L.opensAt H P x a.1 a.2.1 a.2.2 g').isSome)
    (hallI : ∀ (t : Ω × (Fin (L.V.k x) → Coin)) (ec : (Fin (L.V.k x) → Coin) × L.SlotEC x) (g g' : Fin (L.nG x)),
      cls g = cls g' → (L.iopTable H P x t.1 d.part t.2 ec g).isSome →
        (L.iopTable H P x t.1 d.part t.2 ec g').isSome)
    (T : ℕ) (hT : ∀ ω r, (run H P L.V x ω r).pLog.length + (run H P L.V x ω r).vLog.length ≤ T) {Ns Ks Km : ℝ}
    (hNs : Ex (fun r : Fin (L.V.k x) → Coin => ((L.toCompiled.firstSlots x (L.checked x r)).card : ℝ)) ≤ Ns)
    (hKs : Ex (fun r : Fin (L.V.k x) → Coin =>
      (((L.toCompiled.sampleNeeds x needs (L.checked x r)).image cls).card : ℝ)) ≤ Ks)
    (hKm : Ex (fun r : Fin (L.V.k x) → Coin =>
      (L.toCompiled.sessCountIn x needs (fun l => (L.toCompiled.sampleNeeds x needs l).image cls)
        (L.checked x r) : ℝ)) ≤ Km)
    (hB : ExpectedCR H (Compiled.BconfX H L.toCompiled Ω P x))
    (hX : ∀ K : ℕ, ExpectedCR H (L.toCompiled.finderXCls H P x s needs cls K))
    (hI : ∀ K : ℕ, ExpectedCR H (L.finderICls H P x s d.part needs cls K))
    {T' e' Ns' Ks' Km' c' b : ℚ} (hTcap : (T : ℝ) ≤ T') (he : L.toCompiled.es x ≤ e') (hNscap : Ns ≤ Ns')
    (hKscap : Ks ≤ Ks') (hKmcap : Km ≤ Km')
    (hown : (L.toCompiled.nPU x : ℝ) * L.toCompiled.openCalls x ≤ c') (h : hashBound T' e' Ns' Ks' Km' c' ≤ b) :
    Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked x p.2, u ∈ L.FregTwoX H P x p.1 p.2) ≤
      L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) + (b : ℝ) := by
  have hNs0 : 0 ≤ Ns := le_trans (Ex_nonneg (by intro r; exact Nat.cast_nonneg _)) hNs
  have hKs0 : 0 ≤ Ks := le_trans (Ex_nonneg (by intro r; exact Nat.cast_nonneg _)) hKs
  have hKm0 : 0 ≤ Km := le_trans (Ex_nonneg (by intro r; exact Nat.cast_nonneg _)) hKm
  exact (eventTwoX_le_cls_at_budget hA hbind hS hopen hs hPt hd hin P needs hpos cls hrep hallX hallI T hT hNs hKs
    hKm hB hX hI).trans (εclsAt_le_of_hashBound H L x P hNs0 hKs0 hKm0 hTcap he hNscap hKscap hKmcap hown h)

theorem eventTwoX_le_cls_table1_513 {H : Msg → Msg} {L : CompiledTwo Coin A} (hA : A.KS H)
    (hbind : L.commit.Binds H) (hS : L.Sessions H) (hopen : L.toCompiled.OpenCallsLe H) {x : Msg}
    {s : AuditStatement} {Pt : Partition} {d : Sampling.Design.Two} (hs : L.parse x = some s)
    (hPt : L.partition s = some Pt) (hd : L.two s = some d) (hin : InScopeTwo Pt d s) {Ω : Type} [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) (needs : ℕ → Finset ℕ)
    (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.2.pos ∈ needs u) (cls : Fin (L.nG x) → Fin (L.nG x))
    (hrep : ∀ u (g : Fin (L.nG x)), g.val ∈ needs u → (cls g).val ∈ needs u)
    (hallX : ∀ (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (g g' : Fin (L.nG x)), cls g = cls g' →
      (L.opensAt H P x a.1 a.2.1 a.2.2 g).isSome → (L.opensAt H P x a.1 a.2.1 a.2.2 g').isSome)
    (hallI : ∀ (t : Ω × (Fin (L.V.k x) → Coin)) (ec : (Fin (L.V.k x) → Coin) × L.SlotEC x) (g g' : Fin (L.nG x)),
      cls g = cls g' → (L.iopTable H P x t.1 d.part t.2 ec g).isSome →
        (L.iopTable H P x t.1 d.part t.2 ec g').isSome)
    (T : ℕ) (hT : ∀ ω r, (run H P L.V x ω r).pLog.length + (run H P L.V x ω r).vLog.length ≤ T) {Ns Ks Km : ℝ}
    (hNs : Ex (fun r : Fin (L.V.k x) → Coin => ((L.toCompiled.firstSlots x (L.checked x r)).card : ℝ)) ≤ Ns)
    (hKs : Ex (fun r : Fin (L.V.k x) → Coin =>
      (((L.toCompiled.sampleNeeds x needs (L.checked x r)).image cls).card : ℝ)) ≤ Ks)
    (hKm : Ex (fun r : Fin (L.V.k x) → Coin =>
      (L.toCompiled.sessCountIn x needs (fun l => (L.toCompiled.sampleNeeds x needs l).image cls)
        (L.checked x r) : ℝ)) ≤ Km)
    (hB : ExpectedCR H (Compiled.BconfX H L.toCompiled Ω P x))
    (hX : ∀ K : ℕ, ExpectedCR H (L.toCompiled.finderXCls H P x s needs cls K))
    (hI : ∀ K : ℕ, ExpectedCR H (L.finderICls H P x s d.part needs cls K))
    (hTcap : T ≤ 513 * 2 ^ 80) (he : L.toCompiled.es x ≤ 2592237061) (hNscap : Ns ≤ 2 ^ 30) (hKscap : Ks ≤ 2 ^ 33)
    (hKmcap : Km ≤ 2 ^ 33) (hown : L.toCompiled.nPU x * L.toCompiled.openCalls x ≤ 513 * 2 ^ 80) :
    Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked x p.2, u ∈ L.FregTwoX H P x p.1 p.2) ≤
      L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) + 2 ^ (-(100 : ℝ)) := by
  have hNs0 : 0 ≤ Ns := le_trans (Ex_nonneg (by intro r; exact Nat.cast_nonneg _)) hNs
  have hKs0 : 0 ≤ Ks := le_trans (Ex_nonneg (by intro r; exact Nat.cast_nonneg _)) hKs
  have hKm0 : 0 ≤ Km := le_trans (Ex_nonneg (by intro r; exact Nat.cast_nonneg _)) hKm
  exact (eventTwoX_le_cls_at_budget hA hbind hS hopen hs hPt hd hin P needs hpos cls hrep hallX hallI T hT hNs hKs
    hKm hB hX hI).trans (εclsAt_le_table1_513 H L x P hNs0 hKs0 hKm0 hTcap he hNscap hKscap hKmcap hown)

end CompiledTwo

end Claims.SamplingGeneric
