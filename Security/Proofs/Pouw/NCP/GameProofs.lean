import Definitions.Pouw.Assumptions.NCP.GamePinned
import Proofs.Pouw.NCP.Proofs
import Proofs.Pouw.Barrier.Proofs
import Proofs.Pouw.Reduction
import Proofs.Pouw.Witness
import Proofs.Pouw.Sanity

/-!
# NCP's game instance: proofs

* `threeBlock`: every running sum is in int32 for `k ≤ 107,524` and noise in `[−31, 31]`, a sanity
  bound; `lastChecked`, `leadComplete` (checked values mod `2^32`, NA2).
* `gammaFromTTNCP`: the reduction with credited work `3·m·k·n`; `ttncpWitness`, in `outputPriced3`.
* `oneSegment`: the zero-input adversary of class (c), in `Pouw.Barrier.gemmModel`.
-/

namespace Pouw.NCP.GameProofs

open Finset Pouw.Proofs QTree

variable {Q R S : Type}

/-! ## The three-block int32 bound -/

theorem mul_bound {a b c : ℤ} (ha : -c ≤ a ∧ a ≤ c) (hb : -c ≤ b ∧ b ≤ c) : -(c * c) ≤ a * b ∧ a * b ≤ c * c := by
  obtain ⟨a1, a2⟩ := ha; obtain ⟨b1, b2⟩ := hb
  constructor <;> nlinarith [mul_nonneg (sub_nonneg.2 a1) (sub_nonneg.2 b1), mul_nonneg (sub_nonneg.2 a2) (sub_nonneg.2 b2),
    mul_nonneg (sub_nonneg.2 a1) (sub_nonneg.2 b2), mul_nonneg (sub_nonneg.2 a2) (sub_nonneg.2 b1)]

theorem x12 {k : ℕ} {A E1 R : Mat} (hR : SignedPerm k R) {i : ℕ} (hA : ∀ l < k, InInt7 (A i l))
    (hE : ∀ l < k, InSmall (E1 i l)) {l : ℕ} (hl : l < 2 * k) :
    -95 ≤ stackX k A E1 R i l ∧ stackX k A E1 R i l ≤ 95 := by
  simp only [stackX, hcat3, Pi.add_apply]
  split_ifs with h1
  · have := hA l h1; have := hE l h1; unfold InInt7 InSmall at *; constructor <;> linarith
  · have := hA (l - k) (by omega); have := NCP.Proofs.e2_bound hR hE (show l - k < k by omega)
    unfold InInt7 at *; constructor <;> linarith

theorem x3 {k : ℕ} {A E1 R : Mat} (hR : SignedPerm k R) {i : ℕ} (hA : ∀ l < k, InInt7 (A i l))
    (hE : ∀ l < k, InSmall (E1 i l)) {l : ℕ} (hl1 : 2 * k ≤ l) (hl : l < 3 * k) :
    -126 ≤ stackX k A E1 R i l ∧ stackX k A E1 R i l ≤ 126 := by
  simp only [stackX, hcat3, E3, Pi.add_apply, Pi.neg_apply, show ¬ l < k by omega, show ¬ l < 2 * k by omega,
    ite_false]
  have := hA (l - 2 * k) (by omega); have := hE (l - 2 * k) (by omega)
  have := NCP.Proofs.e2_bound hR hE (show l - 2 * k < k by omega)
  unfold InInt7 InSmall at *; constructor <;> linarith

theorem y12 {k : ℕ} {B F1 R : Mat} (hR : SignedPerm k R) {j : ℕ} (hB : ∀ l < k, InInt7 (B l j))
    (hF : ∀ l < k, InSmall (F1 l j)) {l : ℕ} (hl : l < 2 * k) :
    -95 ≤ stackY k B R F1 l j ∧ stackY k B R F1 l j ≤ 95 := by
  simp only [stackY, vcat3, Pi.add_apply]
  split_ifs with h1
  · have := hB l h1; have := hF l h1; unfold InInt7 InSmall at *; constructor <;> linarith
  · have := hB (l - k) (by omega); have := NCP.Proofs.f2_bound hR hF (show l - k < k by omega)
    unfold InInt7 at *; constructor <;> linarith

theorem y3 {k : ℕ} {B F1 R : Mat} (hR : SignedPerm k R) {j : ℕ} (hB : ∀ l < k, InInt7 (B l j))
    (hF : ∀ l < k, InSmall (F1 l j)) {l : ℕ} (hl1 : 2 * k ≤ l) (hl : l < 3 * k) :
    -126 ≤ stackY k B R F1 l j ∧ stackY k B R F1 l j ≤ 126 := by
  simp only [stackY, vcat3, F3, Pi.add_apply, show ¬ l < k by omega, show ¬ l < 2 * k by omega, ite_false]
  have := hB (l - 2 * k) (by omega); have := hF (l - 2 * k) (by omega)
  have := NCP.Proofs.f2_bound hR hF (show l - 2 * k < k by omega)
  unfold InInt7 InSmall at *; constructor <;> linarith

/-- **The three-block int32 bound.** -/
theorem threeBlock : GamePinned.ThreeBlock := by
  intro k A B E1 F1 R i j τ hk hR hA hB hE hF hτ
  have hk' : (k : ℤ) ≤ 107524 := by exact_mod_cast hk
  have h31 : (2 : ℤ) ^ 31 = 2147483648 := by norm_num
  show InInt32 (∑ l ∈ range τ, stackX k A E1 R i l * stackY k B R F1 l j)
  unfold InInt32
  rw [h31]
  by_cases h2 : τ ≤ 2 * k
  · have hb := sum_bounds τ (fun l => stackX k A E1 R i l * stackY k B R F1 l j) (-(95 * 95)) (95 * 95)
      fun l hl => mul_bound (x12 hR hA hE (by omega)) (y12 hR hB hF (by omega))
    have hτ' : (τ : ℤ) ≤ 2 * k := by exact_mod_cast h2
    constructor <;> nlinarith [hb.1, hb.2]
  · have hsplit := Finset.sum_range_add_sum_Ico (fun l => stackX k A E1 R i l * stackY k B R F1 l j) hτ
    have htot : ∑ l ∈ range (3 * k), stackX k A E1 R i l * stackY k B R F1 l j = ∑ l ∈ range k, A i l * B l j :=
      congrFun (congrFun (NCP.Proofs.stacked k A B E1 F1 R).2 i) j
    have hAB := sum_bounds k (fun l => A i l * B l j) (-4032) 4096 fun l hl => mul_bounds_int7 (hA l hl) (hB l hl)
    have hterm : ∀ l ∈ Ico τ (3 * k), -(126 * 126) ≤ stackX k A E1 R i l * stackY k B R F1 l j ∧
        stackX k A E1 R i l * stackY k B R F1 l j ≤ 126 * 126 := fun l hl => by
      obtain ⟨hl1, hl2⟩ := Finset.mem_Ico.1 hl
      exact mul_bound (x3 hR hA hE (by omega) hl2) (y3 hR hB hF (by omega) hl2)
    have hs1 := Finset.card_nsmul_le_sum (Ico τ (3 * k)) (fun l => stackX k A E1 R i l * stackY k B R F1 l j)
      (-(126 * 126)) fun l hl => (hterm l hl).1
    have hs2 := Finset.sum_le_card_nsmul (Ico τ (3 * k)) (fun l => stackX k A E1 R i l * stackY k B R F1 l j)
      (126 * 126) fun l hl => (hterm l hl).2
    rw [Nat.card_Ico, nsmul_eq_mul] at hs1 hs2
    have hc : ((3 * k - τ : ℕ) : ℤ) ≤ k := by exact_mod_cast (show 3 * k - τ ≤ k by omega)
    have hc0 : (0 : ℤ) ≤ ((3 * k - τ : ℕ) : ℤ) := by positivity
    constructor <;> nlinarith [hAB.1, hAB.2]

/-! ## Usefulness and completeness -/

/-- **Usefulness.** -/
theorem lastChecked : GamePinned.LastChecked := by
  intro Q R S P H s U u A i j hd hk hA hB
  have he : D1.blockEnd P.d (3 * (U.layout.shape u).k) (D1.nBlocks P.d (3 * (U.layout.shape u).k) - 1)
      = 3 * (U.layout.shape u).k := by
    rcases Nat.eq_zero_or_pos (U.layout.shape u).k with h0 | h0
    · simp [D1.blockEnd, h0]
    · exact Sanity.blockEnd_last hd (by omega)
  simp only [Params.Y, he, Params.opX, Params.opY]
  exact NCP.Proofs.wrapInt7 _ _ _ _ _ _ i j hk hA hB

/-- **Completeness.** -/
theorem leadComplete : GamePinned.LeadComplete := by
  intro Q R S P U hU u hu H s A hA
  obtain ⟨-, ⟨hsh, -⟩, hwt⟩ := hU
  have hk : (U.layout.shape u).k ≤ 2 ^ 19 - 1 := le_trans (hsh u hu).2.2.2.1 (by norm_num)
  refine ⟨fun x _ => rfl, fun i hi j hj => ?_⟩
  simp only [Params.toProtocol, Params.refOut, Params.opX, Params.opY]
  exact NCP.Proofs.wrapInt7 _ _ _ _ _ _ i j hk (fun l hl => hA _ hi l hl) (fun l hl => hwt u hu l hl _ hj)

/-! ## `G_γ` from TT_NCP -/

theorem wins_imp_ttncp (P : Protocol Q R S) (U : Workload) (H : Q → R) (s : S) (τ : CommitLog) {γ₀ Ωmax T : ℝ}
    (hΩ : 0 < Ωmax)
    (hW : ∀ u < U.layout.N, (P.Wref U.layout u : ℝ) ≤ Ωmax * (3 * ((U.layout.shape u).Wmm : ℝ)))
    (hwin : Wins P U (reductionGamma γ₀ Ωmax) T τ H s) :
    T / (1 - γ₀) < ∑ u ∈ checkedSet P U H s τ, 3 * ((U.layout.shape u).Wmm : ℝ) := by
  unfold Wins at hwin
  rw [one_sub_reductionGamma, div_div_eq_mul_div] at hwin
  have h1 : ∑ u ∈ correctSet P U H s τ, (P.Wref U.layout u : ℝ)
      ≤ Ωmax * ∑ u ∈ correctSet P U H s τ, 3 * ((U.layout.shape u).Wmm : ℝ) := by
    rw [Finset.mul_sum]
    exact Finset.sum_le_sum fun u hu => hW u (Finset.mem_range.1 (correctSet_subset_range P U H s τ hu))
  have h2 : ∑ u ∈ correctSet P U H s τ, 3 * ((U.layout.shape u).Wmm : ℝ)
      ≤ ∑ u ∈ checkedSet P U H s τ, 3 * ((U.layout.shape u).Wmm : ℝ) :=
    Finset.sum_le_sum_of_subset_of_nonneg (correctSet_subset_checkedSet P U H s τ) fun _ _ _ => by positivity
  have h3 : T * Ωmax / (1 - γ₀) = Ωmax * (T / (1 - γ₀)) := by ring
  rw [h3] at hwin
  have h4 : Ωmax * (T / (1 - γ₀)) < Ωmax * ∑ u ∈ checkedSet P U H s τ, 3 * ((U.layout.shape u).Wmm : ℝ) :=
    lt_of_lt_of_le hwin (h1.trans (mul_le_mul_of_nonneg_left h2 hΩ.le))
  exact lt_of_mul_lt_mul_left h4 hΩ.le

/-- **`G_γ` from TT_NCP.** -/
theorem gammaFromTTNCP : GamePinned.GammaFromTTNCP := by
  intro Q R S _ _ _ _ CM P d D γ₀ Ωmax ε hγ₀ hΩ hW hside hTT U hU T hT q hq A hA
  refine le_trans (pr_mono ?_) ((hTT hside).2 U hU T hT q hq A hA)
  rintro ⟨H, s⟩ hwin
  exact wins_imp_ttncp P U H s _ hΩ (hW U hU) hwin

/-! ## TT_NCP's witness -/

section Honest

variable (P : Params Q R S)

/-- Unit `u`'s honest commitment from the answers `a` to `qAct` and `b` to `qWt`. -/
def commitAns (U : Workload) (u : ℕ) (A : Mat) (a b : List R) : Commit :=
  let sh := U.layout.shape u
  let X := stackX sh.k A (P.expAct sh a) (P.expWt sh b).2
  let Y := stackY sh.k (U.weight (U.layout.wid u)) (P.expWt sh b).2 (P.expWt sh b).1
  ⟨A, fun t => mmW (D1.blockEnd P.d (3 * sh.k) t) X Y, mmW (3 * sh.k) X Y⟩

theorem commitAns_eq (H : Q → R) (s : S) (U : Workload) (u : ℕ) (A : Mat) :
    commitAns P U u A ((P.qAct s u A (U.layout.wid u)).map H) ((P.qWt s (U.layout.wid u)).map H)
      = ⟨A, P.toProtocol.refChecked H s U u A, P.toProtocol.refOut H s U u A⟩ := rfl

/-- Unit `u`'s honest program: query its noise, then commit. -/
def unitProg (U : Workload) (act : ℕ → Mat) (s : S) (u : ℕ) : QTree Q R Commit :=
  (queryAll (P.qAct s u (act u) (U.layout.wid u))).bind fun a =>
    (queryAll (P.qWt s (U.layout.wid u))).bind fun b => done (commitAns P U u (act u) a b)

/-- NCP's honest reference as an oracle program. -/
def honestProg (U : Workload) (act : ℕ → Mat) (s : S) : QTree Q R CommitLog :=
  upTo (unitProg P U act s) U.layout.N

theorem eval_honestProg (H : Q → R) (U : Workload) (act : ℕ → Mat) (s : S) :
    (honestProg P U act s).eval H = honestTranscript P.toProtocol H s U act := by
  unfold honestProg honestTranscript
  rw [eval_upTo]
  funext u
  simp only [unitProg, eval_bind, eval_queryAll, eval, commitAns_eq]

theorem calls_honestProg (H : Q → R) (U : Workload) (act : ℕ → Mat) (s : S) {c : ℕ} (hc : P.QueryBound c) :
    (honestProg P U act s).calls H ≤ U.layout.N * c := by
  refine calls_upTo_le H _ _ c fun u _ => ?_
  simp only [unitProg, calls_bind, calls_queryAll, eval_queryAll, calls]
  have := hc s u (act u) (U.layout.wid u)
  omega

end Honest

theorem outputPriced3_queries : (outputPriced3 Q R S).Queries :=
  fun prog U act σ s => ⟨prog U act σ s, fun _ => ⟨rfl, rfl⟩⟩

theorem outputPriced3_cost_le (prog : (outputPriced3 Q R S).Prog) (H : Q → R) (U : Workload) (act : ℕ → Mat)
    (σ : State) (s : S) :
    (outputPriced3 Q R S).cost prog H U act σ s ≤ ∑ u ∈ range U.layout.N, 3 * ((U.layout.shape u).Wmm : ℝ) := by
  simp only [outputPriced3]
  exact Finset.sum_le_sum_of_subset_of_nonneg (Finset.filter_subset _ _) fun _ _ _ => by positivity

theorem checked_le_outputPriced3_cost (P : Protocol Q R S) (prog : (outputPriced3 Q R S).Prog) (H : Q → R)
    (U : Workload) (act : ℕ → Mat) (σ : State) (s : S) :
    ∑ u ∈ checkedSet P U H s ((outputPriced3 Q R S).run prog H U act σ s), 3 * ((U.layout.shape u).Wmm : ℝ)
      ≤ (outputPriced3 Q R S).cost prog H U act σ s := by
  simp only [outputPriced3]
  refine Finset.sum_le_sum_of_subset_of_nonneg (fun u hu => ?_) fun _ _ _ => by positivity
  simp only [checkedSet, Finset.mem_filter] at hu ⊢
  obtain ⟨hu, c, hc, -⟩ := hu
  exact ⟨hu, by rw [hc]; rfl⟩

theorem ttncp_outputPriced3 [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (P : Protocol Q R S) (d : ℕ)
    (D : UnitLayout → Prop) {γ₀ : ℝ} (h0 : 0 ≤ γ₀) (h1 : γ₀ < 1) :
    Assumptions.TTNCP (outputPriced3 Q R S) P d D γ₀ (fun _ _ => 0) := by
  refine fun _ => ⟨outputPriced3_queries, fun U _ T hT q _ A hA => le_of_eq (pr_eq_zero fun x hx => ?_)⟩
  have e1 := checked_le_outputPriced3_cost P A.online x.1 U A.act (A.pre.eval x.1) x.2
  have e2 := (hA x.1 x.2).1
  have e3 : T ≤ T / (1 - γ₀) := by
    rw [le_div_iff₀ (by linarith)]
    nlinarith
  exact absurd hx (not_lt.2 (e1.trans (e2.trans e3)))

/-- **TT_NCP's satisfiability witness.** -/
theorem ttncpWitness : GamePinned.TTNCPWitness := by
  intro Q R _ _ _ P c γ₀ h0 h1 hc hW
  refine ⟨ttncp_outputPriced3 _ _ _ h0 h1, fun U act _ s => honestProg P U act s, fun U hU act _ H s =>
    ⟨eval_honestProg P H U act s, ?_, calls_honestProg P H U act s hc⟩⟩
  refine (outputPriced3_cost_le _ H U act [] s).trans (Finset.sum_le_sum fun u hu => ?_)
  exact_mod_cast hW U hU u (Finset.mem_range.1 hu)

/-! ## The one-segment adversary -/

/-- A unit's one-segment plan: `segPlan` with one register per block of the stacked product. -/
def onePlan (d : ℕ) (sh : Shape) : Barrier.Plan := Barrier.Proofs.segPlan (D1.nBlocks d (3 * sh.k)) d sh

theorem nBlocks_mul {d K : ℕ} (hd : 0 < d) (h : d ∣ K) : D1.nBlocks d K * d = K := by
  obtain ⟨c, rfl⟩ := h
  unfold D1.nBlocks
  rw [show d * c + d - 1 = d - 1 + d * c by omega, Nat.add_mul_div_left _ _ hd, Nat.div_eq_of_lt (by omega),
    zero_add, mul_comm]

theorem nBlocks_pos {d K : ℕ} (hd : 0 < d) (hK : 0 < K) : 1 ≤ D1.nBlocks d K :=
  (Nat.le_div_iff_mul_le hd).2 (by omega)

theorem e2_abs {k : ℕ} {E1 R : Mat} (hR : SignedPerm k R) {i : ℕ} {c : ℤ} (hE : ∀ l < k, |E1 i l| ≤ c) {l : ℕ}
    (hl : l < k) : |E2 k E1 R i l| ≤ c := by
  obtain ⟨π, σ, hσ, hR⟩ := hR
  have hc : ∀ x : Fin k, R x l = if x = π ⟨l, hl⟩ then σ ⟨l, hl⟩ else 0 := fun x => hR x ⟨l, hl⟩
  have e : E2 k E1 R i l = E1 i (π ⟨l, hl⟩) * σ ⟨l, hl⟩ := by
    simp only [E2, mm]
    rw [Finset.sum_range (fun x => E1 i x * R x l)]
    simp only [hc, mul_ite, mul_zero, Finset.sum_ite_eq', Finset.mem_univ, ite_true]
  rw [e, abs_mul]
  rcases hσ ⟨l, hl⟩ with h | h <;> rw [h] <;> simpa using hE _ (π ⟨l, hl⟩).isLt

theorem f2_abs {k : ℕ} {R F1 : Mat} (hR : SignedPerm k R) {j : ℕ} {c : ℤ} (hF : ∀ l < k, |F1 l j| ≤ c) {l : ℕ}
    (hl : l < k) : |F2 k R F1 l j| ≤ c := by
  obtain ⟨π, σ, hσ, hR⟩ := hR
  have hc : ∀ x : Fin k, R l x = if x = π.symm ⟨l, hl⟩ then σ x else 0 := fun x => by
    have hiff : ((⟨l, hl⟩ : Fin k) = π x) ↔ x = π.symm ⟨l, hl⟩ :=
      ⟨fun h => by rw [h, Equiv.symm_apply_apply], fun h => by rw [h, Equiv.apply_symm_apply]⟩
    exact (hR ⟨l, hl⟩ x).trans (if_congr hiff rfl rfl)
  have e : F2 k R F1 l j = -(σ (π.symm ⟨l, hl⟩) * F1 (π.symm ⟨l, hl⟩) j) := by
    simp only [F2, mm]
    rw [Finset.sum_range (fun x => R l x * F1 x j)]
    simp only [hc, ite_mul, zero_mul, Finset.sum_ite_eq', Finset.mem_univ, ite_true]
  rw [e, abs_neg, abs_mul]
  rcases hσ (π.symm ⟨l, hl⟩) with h | h <;> rw [h] <;> simpa using hF _ (π.symm ⟨l, hl⟩).isLt

theorem zero_x {k : ℕ} {E1 R : Mat} (hR : SignedPerm k R) {i : ℕ} (hE : ∀ l < k, |E1 i l| ≤ 32) {l : ℕ}
    (hl : l < 3 * k) : |stackX k 0 E1 R i l| ≤ 64 := by
  simp only [stackX, hcat3, E3, Pi.add_apply, Pi.neg_apply, Pi.zero_apply, zero_add]
  split_ifs with h1 h2
  · linarith [hE l h1]
  · linarith [e2_abs hR hE (show l - k < k by omega)]
  · have h3 := abs_le.1 (hE (l - 2 * k) (by omega))
    have h4 := abs_le.1 (e2_abs hR hE (show l - 2 * k < k by omega))
    rw [abs_le]
    constructor <;> linarith [h3.1, h3.2, h4.1, h4.2]

theorem zero_y {k : ℕ} {F1 R : Mat} (hR : SignedPerm k R) {j : ℕ} (hF : ∀ l < k, |F1 l j| ≤ 32) {l : ℕ}
    (hl : l < 3 * k) : |stackY k 0 R F1 l j| ≤ 64 := by
  simp only [stackY, vcat3, F3, Pi.add_apply, Pi.zero_apply, zero_add]
  split_ifs with h1 h2
  · linarith [hF l h1]
  · linarith [f2_abs hR hF (show l - k < k by omega)]
  · exact (abs_add_le _ _).trans (by linarith [hF (l - 2 * k) (by omega), f2_abs hR hF (show l - 2 * k < k by omega)])

/-- At zero input, noise of magnitude at most `32` and `k ≤ 2^16`, every running sum is at most `3k·64² < 2^31`. -/
theorem zero_int32 {k τ : ℕ} {E1 F1 R : Mat} (hR : SignedPerm k R) {i j : ℕ} (hE : ∀ l < k, |E1 i l| ≤ 32)
    (hF : ∀ l < k, |F1 l j| ≤ 32) (hk : k ≤ 2 ^ 16) (hτ : τ ≤ 3 * k) :
    InInt32 (mm τ (stackX k 0 E1 R) (stackY k 0 R F1) i j) := by
  have hs := Finset.sum_le_card_nsmul (range τ) (fun l => |stackX k 0 E1 R i l * stackY k 0 R F1 l j|) 4096
    fun l hl => by
      have hl' := Finset.mem_range.1 hl
      rw [abs_mul]
      exact (mul_le_mul (zero_x hR hE (by omega)) (zero_y hR hF (by omega)) (abs_nonneg _) (by norm_num)).trans
        (by norm_num)
  rw [Finset.card_range, nsmul_eq_mul] at hs
  have h := (Finset.abs_sum_le_sum_abs (fun l => stackX k 0 E1 R i l * stackY k 0 R F1 l j) (range τ)).trans hs
  have hτ' : (τ : ℤ) ≤ 3 * 2 ^ 16 := by exact_mod_cast (show τ ≤ 3 * 2 ^ 16 by omega)
  rw [abs_le] at h
  show -2 ^ 31 ≤ ∑ l ∈ range τ, stackX k 0 E1 R i l * stackY k 0 R F1 l j ∧
    ∑ l ∈ range τ, stackX k 0 E1 R i l * stackY k 0 R F1 l j ≤ 2 ^ 31 - 1
  constructor <;> linarith [h.1, h.2]

/-- The stacked product at zero input is in int32 for NCP's noise on a default-domain shape. -/
theorem nz_int32 (P : Params Q R S) (hN : P.NoiseRange) (hR : P.RSigned) (Lay : UnitLayout) (H : Q → R) (s : S) (u : ℕ)
    (hsh : Shape.InDefaultDomain (Lay.shape u)) {τ : ℕ} (hτ : τ ≤ 3 * (Lay.shape u).k) {a b : ℕ}
    (ha : a < (Lay.shape u).m) (hb : b < (Lay.shape u).n) :
    InInt32 (mm τ (P.opX H s ⟨Lay, fun _ => 0⟩ u 0) (P.opY H s ⟨Lay, fun _ => 0⟩ u) a b) := by
  have habs : ∀ {x : ℤ}, InNoise6 x → |x| ≤ 32 := fun {x} h => abs_le.2 ⟨by linarith [h.1], h.2⟩
  exact zero_int32 (hR _ _ hsh) (fun l hl => habs (hN.1 _ _ _ ha l hl)) (fun l hl => habs (hN.2 _ _ l hl _ hb))
    hsh.2.2.2.1 hτ

/-- **The int8 range at the lead's noise.** -/
theorem rangeLead : GamePinned.RangeLead := by
  intro k A B E1 F1 R hR hE hF hA hB
  refine ⟨fun i l hl => ?_, fun l j hl => ?_⟩
  · have hE' : ∀ l < k, |E1 i l| ≤ 32 := fun l hl => abs_le.2 ⟨by linarith [(hE i l hl).1], (hE i l hl).2⟩
    simp only [stackX, hcat3, E3, Pi.add_apply, Pi.neg_apply]
    split_ifs with h1 h2
    · have := hA i l h1; have := hE i l h1; unfold InInt7 InNoise6 at *; constructor <;> linarith
    · have := hA i (l - k) (by omega); have := abs_le.1 (e2_abs hR hE' (show l - k < k by omega))
      unfold InInt7 at *; constructor <;> linarith
    · have := hA i (l - 2 * k) (by omega); have := hE i (l - 2 * k) (by omega)
      have := abs_le.1 (e2_abs hR hE' (show l - 2 * k < k by omega))
      unfold InInt7 InNoise6 at *; constructor <;> linarith
  · have hF' : ∀ l < k, |F1 l j| ≤ 32 := fun l hl => abs_le.2 ⟨by linarith [(hF l hl j).1], (hF l hl j).2⟩
    simp only [stackY, vcat3, F3, Pi.add_apply]
    split_ifs with h1 h2
    · have := hB l h1 j; have := hF l h1 j; unfold InInt7 InNoise6 at *; constructor <;> linarith
    · have := hB (l - k) (by omega) j; have := abs_le.1 (f2_abs hR hF' (show l - k < k by omega))
      unfold InInt7 at *; constructor <;> linarith
    · have := hB (l - 2 * k) (by omega) j; have := hF (l - 2 * k) (by omega) j
      have := abs_le.1 (f2_abs hR hF' (show l - 2 * k < k by omega))
      unfold InInt7 InNoise6 at *; constructor <;> linarith

theorem onePlan_correct (P : Params Q R S) (hd : 0 < P.d) (Lay : UnitLayout) (H : Q → R) (s : S) (u : ℕ)
    (hk : 0 < (Lay.shape u).k)
    (hI : ∀ τ ≤ 3 * (Lay.shape u).k, ∀ a < (Lay.shape u).m, ∀ b < (Lay.shape u).n,
      InInt32 (mm τ (P.opX H s ⟨Lay, fun _ => 0⟩ u 0) (P.opY H s ⟨Lay, fun _ => 0⟩ u) a b)) :
    Correct P.toProtocol ⟨Lay, fun _ => 0⟩ H s u
      ((onePlan P.d (Lay.shape u)).commit P.nz H s ⟨Lay, fun _ => 0⟩ (fun _ => 0) u) := by
  have hX : P.opX H s ⟨Lay, fun _ => 0⟩ u 0 = P.nz H s ⟨Lay, fun _ => 0⟩ u 0 0 := by simp [Params.opX, Params.nz]
  have hY : P.opY H s ⟨Lay, fun _ => 0⟩ u = P.nz H s ⟨Lay, fun _ => 0⟩ u 0 1 := by simp [Params.opY, Params.nz]
  refine ⟨⟨Barrier.Proofs.zero_actOK _, fun x hx => ?_⟩, fun i hi j hj => ?_⟩
  · simp only [Params.toProtocol, Params.checkIdx, Finset.mem_product, Finset.mem_range] at hx
    obtain ⟨hxt, ha, hb⟩ := hx
    simp only [Barrier.Plan.commit, onePlan, Barrier.Proofs.segPlan_act, Option.getD_some, Params.toProtocol,
      Params.Y]
    rw [mmW_eq_of_mem (hI _ (blockEnd_le _ _ _) _ ha _ hb)]
    by_cases hlast : x.1 + 1 < D1.nBlocks P.d (3 * (Lay.shape u).k)
    · rw [Barrier.Proofs.segPlan_regs _ _ _ _ _ _ ha hb x.1 hlast, Sanity.blockEnd_full hd hlast, hX, hY]
      rfl
    · have hx1 : x.1 = D1.nBlocks P.d (3 * (Lay.shape u).k) - 1 := by omega
      rw [hx1, Barrier.Proofs.regs_eq, Barrier.Proofs.segPlan_nReg, ite_eq_left (by omega),
        Barrier.Proofs.segPlan_reg, ite_eq_right (by omega), Sanity.blockEnd_last hd (by omega)]
      simp only [Barrier.MExpr.eval, Pi.zero_apply, Params.opX, Params.opY]
      rw [(NCP.Proofs.stacked _ _ _ _ _ _).2]
      simp [mm]
  · simp [Barrier.Plan.commit, onePlan, Barrier.Proofs.segPlan, Barrier.MExpr.eval, mm]

/-- **The one-segment adversary.** -/
theorem oneSegment : GamePinned.OneSegment := by
  intro Q R S P Lay hd hN hR hdk hU
  have hsh : ∀ u < Lay.N, Shape.InDefaultDomain (Lay.shape u) := hU.2.1.1
  have hk : ∀ u < Lay.N, 0 < (Lay.shape u).k := fun u hu => lt_of_lt_of_le (by norm_num) (hsh u hu).2.2.1
  have hcost : ∀ u ∈ range Lay.N, (((onePlan P.d (Lay.shape u)).cost : ℕ) : ℝ)
      = 3 * ((Lay.shape u).Wmm : ℝ) - (Lay.shape u).m * P.d * (Lay.shape u).n := fun u hu => by
    have hu' := Finset.mem_range.1 hu
    have hL := nBlocks_pos hd (show 0 < 3 * (Lay.shape u).k by have := hk u hu'; omega)
    have hm := nBlocks_mul hd (Dvd.dvd.mul_left (hdk u hu') 3)
    have hm' : (D1.nBlocks P.d (3 * (Lay.shape u).k) : ℝ) * P.d = 3 * (Lay.shape u).k := by exact_mod_cast hm
    rw [onePlan, Barrier.Proofs.segPlan_cost _ _ _ hL]
    push_cast [Nat.cast_sub hL, Shape.Wmm]
    linear_combination ((Lay.shape u).m * (Lay.shape u).n : ℝ) * hm'
  have hB := Barrier.Proofs.planAdv_bounded P.nz (onePlan P.d) ⟨Lay, fun _ => 0⟩
  rw [Finset.sum_congr rfl hcost] at hB
  have hall := Barrier.Proofs.planAdv_correct P.nz (onePlan P.d) P.toProtocol ⟨Lay, fun _ => 0⟩
    fun H s u hu => onePlan_correct P hd Lay H s u (hk u hu) fun _ hτ _ ha _ hb =>
      nz_int32 P hN hR Lay H s u (hsh u hu) hτ ha hb
  refine ⟨Barrier.Proofs.planAdv P.nz (onePlan P.d), ⟨hU, ?_, by norm_num, hB, hall⟩, ?_⟩
  · refine Finset.sum_nonneg fun u hu => ?_
    rw [← hcost u hu]
    positivity
  · intro γ₀ hγ hN hγk H s
    have hchk : checkedSet P.toProtocol ⟨Lay, fun _ => 0⟩ H s
        ((Barrier.Proofs.planAdv P.nz (onePlan P.d)).transcript ⟨Lay, fun _ => 0⟩ H s) = range Lay.N := by
      refine Finset.Subset.antisymm (fun u hu => ?_) ?_
      · simp only [checkedSet, Finset.mem_filter] at hu; exact hu.1
      · rw [← hall H s]; exact correctSet_subset_checkedSet _ _ _ _ _
    rw [hchk, div_lt_iff₀ (by linarith), Finset.sum_mul]
    refine Finset.sum_lt_sum_of_nonempty (Finset.nonempty_range_iff.2 (by omega)) fun u hu => ?_
    have hu' := Finset.mem_range.1 hu
    have h1 := hγk u hu'
    have hmn : (0 : ℝ) < (Lay.shape u).m * (Lay.shape u).n := by
      have := (hsh u hu').1
      have : 0 < (Lay.shape u).n := lt_of_lt_of_le (by norm_num : (0 : ℕ) < 2 ^ 10) (hsh u hu').2.2.2.2.2.1
      positivity
    simp only [Shape.Wmm]
    push_cast
    nlinarith [mul_lt_mul_of_pos_left h1 hmn]

/-! ## The lead's parameters are satisfiable -/

theorem shiftBy_signed {c k : ℕ} (hc : c < k) : SignedPerm k (shiftBy c k) := by
  obtain ⟨n, rfl⟩ : ∃ n, k = n + 1 := ⟨k - 1, by omega⟩
  refine ⟨Equiv.addRight (⟨c, hc⟩ : Fin (n + 1)), fun _ => 1, fun _ => Or.inl rfl, fun i j => ?_⟩
  have h : ((Equiv.addRight (⟨c, hc⟩ : Fin (n + 1)) j : Fin (n + 1)) : ℕ)
      = if (j : ℕ) + c < n + 1 then (j : ℕ) + c else (j : ℕ) + c - (n + 1) := by
    simp only [Equiv.coe_addRight, Fin.val_add]
    split_ifs with hj
    · exact Nat.mod_eq_of_lt hj
    · rw [Nat.mod_eq_sub_mod (by omega), Nat.mod_eq_of_lt (by omega)]
  by_cases hij : i = Equiv.addRight (⟨c, hc⟩ : Fin (n + 1)) j
  · simp only [shiftBy, j.isLt, true_and, hij, h, ite_true]
  · have : ¬ ((i : ℕ) = if (j : ℕ) + c < n + 1 then (j : ℕ) + c else (j : ℕ) + c - (n + 1)) :=
      fun e => hij (Fin.ext (e.trans h.symm))
    simp only [shiftBy, j.isLt, true_and, this, hij, ite_false]

/-- The signed permutation matrix of `f` with signs `σ` on `{0, …, k − 1}`: column `j < k` has `σ j` in row `f j`. -/
def permMat (k : ℕ) (f : ℕ → ℕ) (σ : ℕ → ℤ) : Mat := fun i j => if j < k ∧ i = f j then σ j else 0

theorem permMat_signed {k : ℕ} {f : ℕ → ℕ} {σ : ℕ → ℤ} (hf : ∀ j < k, f j < k) (hff : ∀ j < k, f (f j) = j)
    (hσ : ∀ j, σ j = 1 ∨ σ j = -1) : SignedPerm k (permMat k f σ) := by
  let g : Fin k → Fin k := fun j => ⟨f j, hf j j.isLt⟩
  have hg : Function.Involutive g := fun j => Fin.ext (hff j j.isLt)
  refine ⟨Function.Involutive.toPerm g hg, fun j => σ j, fun j => hσ j, fun i j => ?_⟩
  have h : ((Function.Involutive.toPerm g hg j : Fin k) : ℕ) = f j := rfl
  by_cases hij : i = Function.Involutive.toPerm g hg j
  · simp only [permMat, j.isLt, true_and, hij, h, ite_true]
  · have : ¬ ((i : ℕ) = f j) := fun e => hij (Fin.ext (e.trans h.symm))
    simp only [permMat, j.isLt, true_and, this, hij, ite_false]

/-- `R²`'s column `j` for `R = permMat k f σ`: `σ (f j)·σ j` in row `f (f j)`. -/
theorem permMat_sq {k : ℕ} {f : ℕ → ℕ} {σ : ℕ → ℤ} (hf : ∀ j < k, f j < k) {i j : ℕ} (hj : j < k) :
    mm k (permMat k f σ) (permMat k f σ) i j = if i = f (f j) then σ (f j) * σ j else 0 := by
  have hcol : ∀ l, permMat k f σ l j = if l = f j then σ j else 0 := fun l => by
    simp only [permMat, hj, true_and]
  simp only [mm, hcol, mul_ite, mul_zero, Finset.sum_ite_eq', Finset.mem_range, hf j hj, ite_true]
  simp only [permMat, hf j hj, true_and, ite_mul, zero_mul]

/-- The shift by `24` passes the rejection rule at `d = 16`: step `t`'s image `[16t + 24, 16t + 40) mod k` is disjoint
from it, columns `16t + 7` and `16t + 8` land in different steps, and `R²` (the shift by `48`) has `0` at `(0, 0)`. -/
theorem shiftBy24_admissible {k : ℕ} (hk : 1024 ≤ k) (h16 : 16 ∣ k) : Admissible k 16 (shiftBy 24 k) := by
  have h00 : mm k (shiftBy 24 k) (shiftBy 24 k) 0 0 = 0 := by
    have e : shiftBy 24 k = permMat k (fun j => if j + 24 < k then j + 24 else j + 24 - k) (fun _ => 1) := rfl
    have hf : ∀ j < k, (if j + 24 < k then j + 24 else j + 24 - k) < k := fun j _ => by split_ifs <;> omega
    rw [e, permMat_sq hf (show 0 < k by omega)]
    simp [show 24 < k by omega, show 48 < k by omega]
  refine ⟨shiftBy_signed (by omega), fun t ht => ⟨fun j hj1 hj2 i hi1 hi2 => ?_, fun t' ht' h => ?_⟩,
    fun h => ?_, fun h => ?_⟩
  · simp only [shiftBy]
    split_ifs <;> omega
  · obtain ⟨m, rfl⟩ := h16
    obtain ⟨i1, hi1, hi1', hr1⟩ := h (t * 16 + 7) (by omega) (by omega)
    obtain ⟨i2, hi2, hi2', hr2⟩ := h (t * 16 + 8) (by omega) (by omega)
    simp only [shiftBy] at hr1 hr2
    split_ifs at hr1 hr2 <;> omega
  · have := h 0 (by omega) 0 (by omega)
    rw [h00] at this
    norm_num at this
  · have := h 0 (by omega) 0 (by omega)
    rw [h00] at this
    norm_num at this

/-- The cyclic shift by one (the previous witness). -/
def shiftR (k : ℕ) : Mat := fun i j => if j < k ∧ i = (if j + 1 < k then j + 1 else 0) then 1 else 0

/-- Sanity: the shift by one fails the rejection rule (NA1), since column `0` lands in row `1` of its own step. -/
theorem shiftR_not_admissible {k : ℕ} (hk : 16 ≤ k) : ¬ Admissible k 16 (shiftR k) := fun h => by
  have := (h.2.1 0 (by omega)).1 0 (by omega) (by omega) 1 (by omega) (by omega)
  simp [shiftR, show 0 < k by omega, show 1 < k by omega] at this

/-- The involution swapping `j` and `j + 24` in every period of `48`. -/
def swap24 (j : ℕ) : ℕ := if j % 48 < 24 then j + 24 else j - 24

/-- Sanity: the unsigned `swap24` is a signed permutation for `48 ∣ k` that passes the first two
clauses of the rejection rule (step `t`'s image is `24` away and straddles two steps), but `R² = I`, so the third
clause rejects it. -/
theorem swap24_not_admissible {k : ℕ} (h48 : 48 ∣ k) :
    SignedPerm k (permMat k swap24 fun _ => 1) ∧
    (∀ t < k / 16,
      (∀ j, t * 16 ≤ j → j < (t + 1) * 16 → ∀ i, t * 16 ≤ i → i < (t + 1) * 16 → permMat k swap24 (fun _ => 1) i j = 0) ∧
      ∀ t' < k / 16, ¬ ∀ j, t * 16 ≤ j → j < (t + 1) * 16 →
        ∃ i, t' * 16 ≤ i ∧ i < (t' + 1) * 16 ∧ permMat k swap24 (fun _ => 1) i j ≠ 0) ∧
    (∀ i < k, ∀ j < k, mm k (permMat k swap24 fun _ => 1) (permMat k swap24 fun _ => 1) i j = if i = j then 1 else 0) ∧
    ¬ Admissible k 16 (permMat k swap24 fun _ => 1) := by
  have hlt : ∀ j < k, swap24 j < k := by
    obtain ⟨m, rfl⟩ := h48
    intro j hj
    unfold swap24
    split_ifs <;> omega
  have hff : ∀ j, swap24 (swap24 j) = j := fun j => by
    unfold swap24
    split_ifs <;> omega
  have hsq : ∀ i < k, ∀ j < k, mm k (permMat k swap24 fun _ => 1) (permMat k swap24 fun _ => 1) i j
      = if i = j then 1 else 0 := fun i _ j hj => by
    rw [permMat_sq hlt hj, hff]
    simp
  refine ⟨permMat_signed hlt (fun j _ => hff j) (fun _ => Or.inl rfl),
    fun t ht => ⟨fun j hj1 hj2 i hi1 hi2 => ?_, fun t' ht' h => ?_⟩, hsq, fun h => h.2.2.1 hsq⟩
  · simp only [permMat, swap24]
    split_ifs <;> omega
  · obtain ⟨m, rfl⟩ := h48
    obtain ⟨i1, hi1, hi1', hr1⟩ := h (t * 16 + 7) (by omega) (by omega)
    obtain ⟨i2, hi2, hi2', hr2⟩ := h (t * 16 + 8) (by omega) (by omega)
    simp only [permMat, swap24] at hr1 hr2
    split_ifs at hr1 hr2 <;> omega

/-- The adjacent-step swap at depth `16`: steps `2u` and `2u + 1` trade places. -/
def stepSwap (j : ℕ) : ℕ := if j / 16 % 2 = 0 then j + 16 else j - 16

/-- The sign `−1` on step `0` and `+1` elsewhere, so that the signed step swap has `R² ≠ ±I`. -/
def stepSign (j : ℕ) : ℤ := if j < 16 then -1 else 1

/-- Sanity: the step swap with signs `stepSign` is a signed permutation for `32 ∣ k`, `k ≥ 64`, that
passes the first clause (no step meets its image) and the third (`R²` is `−1` at `(0, 0)` and `1` at `(32, 32)`), but
maps step `0` onto step `1`, so the second clause rejects it. -/
theorem stepSwap_not_admissible {k : ℕ} (hk : 64 ≤ k) (h32 : 32 ∣ k) :
    SignedPerm k (permMat k stepSwap stepSign) ∧
    (∀ t < k / 16, ∀ j, t * 16 ≤ j → j < (t + 1) * 16 → ∀ i, t * 16 ≤ i → i < (t + 1) * 16 →
      permMat k stepSwap stepSign i j = 0) ∧
    (¬ ∀ i < k, ∀ j < k, mm k (permMat k stepSwap stepSign) (permMat k stepSwap stepSign) i j
      = if i = j then 1 else 0) ∧
    (¬ ∀ i < k, ∀ j < k, mm k (permMat k stepSwap stepSign) (permMat k stepSwap stepSign) i j
      = if i = j then -1 else 0) ∧
    ¬ Admissible k 16 (permMat k stepSwap stepSign) := by
  have hlt : ∀ j < k, stepSwap j < k := by
    obtain ⟨m, rfl⟩ := h32
    intro j hj
    unfold stepSwap
    split_ifs <;> omega
  have hff : ∀ j, stepSwap (stepSwap j) = j := fun j => by
    unfold stepSwap
    split_ifs <;> omega
  have hσ : ∀ j, stepSign j = 1 ∨ stepSign j = -1 := fun j => by
    unfold stepSign
    split_ifs <;> simp
  refine ⟨permMat_signed hlt (fun j _ => hff j) hσ, fun t _ j hj1 hj2 i hi1 hi2 => ?_, fun h => ?_, fun h => ?_,
    fun h => (h.2.1 0 (by omega)).2 1 (by omega) fun j hj1 hj2 => ⟨j + 16, by omega, by omega, ?_⟩⟩
  · simp only [permMat, stepSwap]
    split_ifs <;> omega
  · have := h 0 (by omega) 0 (by omega)
    rw [permMat_sq hlt (show 0 < k by omega), hff] at this
    norm_num [stepSwap, stepSign] at this
  · have := h 32 (by omega) 32 (by omega)
    rw [permMat_sq hlt (show 32 < k by omega), hff] at this
    norm_num [stepSwap, stepSign] at this
  · simp only [permMat, stepSwap, stepSign]
    split_ifs <;> omega

/-- The witness instance: `d = 16`, zero noise, `R` the cyclic shift by `24`, no queries, `W_ref = 3·W_mm`. -/
def leadParams : Params Unit Unit Salt :=
  ⟨16, fun _ _ _ _ => [], fun _ _ => [], fun _ _ => 0, fun sh _ => (0, shiftBy 24 sh.k),
    fun L u => 3 * (L.shape u).Wmm⟩

/-- The witness layout: one unit of shape `(16, 1024, 1024)` on weight `0`. -/
def leadLayout : UnitLayout := ⟨1, fun _ => ⟨16, 1024, 1024⟩, fun _ => 0⟩

theorem leadLayout_inDomain : (⟨leadLayout, fun _ => 0⟩ : Workload).InDomain leadDomain := by
  refine ⟨by norm_num [leadLayout], ⟨fun u _ => ?_, fun w => ?_⟩, fun _ _ _ _ _ _ => by unfold InInt7; norm_num⟩
  · simp only [leadLayout, Shape.InDefaultDomain]
    norm_num
  · unfold UnitLayout.rowsOf
    refine le_trans (Finset.sum_le_sum_of_subset_of_nonneg (Finset.filter_subset _ _) fun _ _ _ => Nat.zero_le _) ?_
    simp [leadLayout]

/-- **The lead's instance, jointly.** -/
theorem leadWitness : GamePinned.LeadWitness := by
  have hN : leadParams.NoiseRange :=
    ⟨fun _ _ _ _ _ _ => by simp [leadParams, InNoise6], fun _ _ _ _ _ _ => by simp [leadParams, InNoise6]⟩
  have hadm : leadParams.RAdmissible := fun sh _ hsh =>
    shiftBy24_admissible (le_of_eq_of_le (by norm_num) hsh.2.2.1) (Nat.dvd_trans (by norm_num) hsh.2.2.2.2.1)
  have hS : leadParams.RSigned := fun sh b hsh => (hadm sh b hsh).1
  have hq : leadParams.QueryBound 0 := fun _ _ _ _ => by simp [leadParams]
  have hW : ∀ U : Workload, U.InDomain leadDomain → ∀ u < U.layout.N,
      3 * (U.layout.shape u).Wmm ≤ leadParams.Wref U.layout u := fun _ _ _ _ => le_rfl
  have hadmit := (ttncpWitness Unit Unit leadParams 0 0 le_rfl one_pos hq hW).2
  simp only [mul_zero] at hadmit
  exact ⟨leadParams, leadLayout, ⟨rfl, hN, hadm⟩, hS, hq, fun _ _ => rfl, leadComplete Unit Unit Salt leadParams,
    hadmit, fun γ₀ h0 h1 => (ttncpWitness Unit Unit leadParams 0 γ₀ h0 h1 hq hW).1, leadLayout_inDomain, one_pos,
    fun _ _ => ⟨64, rfl⟩⟩

/-! ## `Barrier.Segment`'s hypotheses with the lead's noise -/

theorem stackX_add (k : ℕ) (A E1 Rm : Mat) : stackX k A 0 0 + stackX k 0 E1 Rm = stackX k A E1 Rm := by
  funext i l
  simp only [stackX, hcat3, Pi.add_apply, Pi.neg_apply, E3, E2, mm, Pi.zero_apply, zero_mul, Finset.sum_const_zero]
  split_ifs <;> ring

theorem stackY_add (k : ℕ) (B Rm F1 : Mat) : stackY k B 0 0 + stackY k 0 Rm F1 = stackY k B Rm F1 := by
  funext l j
  simp only [stackY, vcat3, Pi.add_apply, F3, F2, mm, Pi.zero_apply, mul_zero, Finset.sum_const_zero, neg_zero]
  split_ifs <;> ring

theorem stackNoise_zero (k : ℕ) (Rm : Mat) (e f : (Q → R) → S → Workload → ℕ → Mat → Mat) (H : Q → R) (s : S)
    (U : Workload) (u : ℕ) (A : Mat) : Barrier.stackNoise k Rm e f H s U u A 0 = stackX k 0 (e H s U u A) Rm := by
  simp [Barrier.stackNoise]

theorem stackNoise_one (k : ℕ) (Rm : Mat) (e f : (Q → R) → S → Workload → ℕ → Mat → Mat) (H : Q → R) (s : S)
    (U : Workload) (u : ℕ) (A : Mat) : Barrier.stackNoise k Rm e f H s U u A 1 = stackY k 0 Rm (f H s U u A) := by
  simp [Barrier.stackNoise]

/-- `SegmentLead`'s layout: one unit of shape `(16, k, 1024)` on weight `0`. -/
def segLayout (k : ℕ) : UnitLayout := ⟨1, fun _ => ⟨16, k, 1024⟩, fun _ => 0⟩

theorem segLayout_inDomain {k : ℕ} (hk1 : 2 ^ 10 ≤ k) (hk2 : k ≤ 2 ^ 16) (hk3 : 64 ∣ k) :
    (⟨segLayout k, fun _ => 0⟩ : Workload).InDomain
      (fun L => leadDomain L ∧ ∀ u < L.N, (L.shape u).k = k) := by
  refine ⟨by norm_num [segLayout], ⟨⟨fun u _ => ?_, fun w => ?_⟩, fun _ _ => rfl⟩,
    fun _ _ _ _ _ _ => by unfold InInt7; norm_num⟩
  · exact ⟨by norm_num [segLayout], by norm_num [segLayout], hk1, hk2, hk3, by norm_num [segLayout],
      by norm_num [segLayout], by norm_num [segLayout]⟩
  · unfold UnitLayout.rowsOf
    refine le_trans (Finset.sum_le_sum_of_subset_of_nonneg (Finset.filter_subset _ _) fun _ _ _ => Nat.zero_le _) ?_
    simp [segLayout]

theorem segLayout_pos (k : ℕ) :
    0 < ∑ u ∈ range (segLayout k).N, ((((segLayout k).shape u).m * 16 * ((segLayout k).shape u).n : ℕ) : ℝ) := by
  simp only [segLayout, Finset.sum_range_one]
  norm_num

/-- **`Segment`'s hypotheses with the lead's noise.** -/
theorem segmentLead : GamePinned.SegmentLead := by
  intro Q R S e f k hk1 hk2 hk3
  have h316 : 3 * k / 16 * 16 = 3 * k := by omega
  have hk : 1024 ≤ k := le_of_eq_of_le (by norm_num) hk1
  refine ⟨shiftBy 24 k, segLayout k, shiftBy24_admissible hk (Nat.dvd_trans (by norm_num) hk3), by omega, h316, ?_, ?_, fun H s U u a _ b _ => ?_,
    segLayout_inDomain hk1 hk2 hk3, segLayout_pos k, fun u _ => ?_, ?_⟩
  · funext i l
    simp [stackX, hcat3, E2, E3, mm]
  · funext l j
    simp [stackY, vcat3, F2, F3, mm]
  · rw [h316, stackNoise_zero, stackNoise_one, (NCP.Proofs.stacked k 0 0 _ _ _).2]
    simp [mm]
  · rw [h316]
    simp only [segLayout, Shape.Wmm]
    push_cast
    exact le_of_eq (by ring)
  · intro U hU u hu H s A _
    refine ⟨fun x hx => ?_, fun i hi j hj => ?_⟩
    · simp only [Barrier.segProto, Finset.mem_product, Finset.mem_range] at hx ⊢
      obtain ⟨ht, hi, hj⟩ := hx
      simp [ht, Barrier.corner, hi, hj]
    · simp only [Barrier.segProto, Barrier.corner, hi, hj, and_self, ite_true]
      rw [h316, stackNoise_zero, stackNoise_one, stackX_add, stackY_add, (NCP.Proofs.stacked k _ _ _ _ _).2,
        hU.2.1.2 u hu]

end Pouw.NCP.GameProofs
