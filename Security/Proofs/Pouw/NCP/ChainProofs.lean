import Definitions.Pouw.Assumptions.NCP.Chain
import Proofs.Pouw.NCP.RelationWitness

/-!
# The chain end to end: proofs
-/

namespace Pouw.NCP.ChainProofs

open Finset Pouw.Proofs Pouw.NCP.Relation Pouw.NCP.Chain Pouw.NCP.GameProofs Pouw.NCP.RelationWitness

theorem paramsYLink : ParamsYLink := by
  intro Q R S P H s U u A t i j
  simp only [Params.Y, mmW, Params.opX, Params.opY, accW_eq]
  rfl

theorem leadDepth : LeadDepth := by
  intro k m n p h16 hp
  have hmn : 0 < m * n := by
    rcases Nat.eq_zero_or_pos (m * n) with h | h
    · simp [positions, h] at hp
    · exact h
  have hlt : p / (m * n) < 3 * k / 16 := (Nat.div_lt_iff_lt_mul hmn).2 hp
  simp only [D1.blockEnd]
  obtain ⟨c, rfl⟩ := h16
  generalize p / (m * n) = q at hlt ⊢
  omega

theorem wrapIdentity : WrapIdentity := by
  intro k m n N R A B adv u p hV hp hr
  have heq : ∀ ω q, q ≤ p → wrapped k m n N R A B ω u q = checked k m n N R A B ω u q :=
    fun ω q hq => wrap32_of_mem (hr ω q hq)
  have ht : ∀ ω, (adv.rule u p).target (wrapped k m n N R A B ω u) =
      (adv.rule u p).target (checked k m n N R A B ω u) :=
    fun ω => WordProofs.target_congr _ p (hV u p hp) _ _ fun q hq => heq ω q hq.le
  constructor
  · rintro ⟨hne, h⟩
    exact ⟨hne, fun ω => by rw [← heq ω p le_rfl, ← ht ω]; exact h ω⟩
  · rintro ⟨hne, h⟩
    exact ⟨hne, fun ω => by rw [heq ω p le_rfl, ht ω]; exact h ω⟩

theorem identCount_wrapped {k m n N : ℕ} {R : Mat} {A B : ℕ → Mat} {adv : Word.Adversary}
    (hV : Word.Valid (fun _ => positions k m n) adv)
    (hr : ∀ (ω : Noise N m n k) (u p : ℕ), u < N → p < positions k m n → InInt32 (checked k m n N R A B ω u p))
    {u : ℕ} (hu : u < N) :
    Word.identCount (fun _ => positions k m n) (wrapped k m n N R A B) adv u =
      Word.identCount (fun _ => positions k m n) (checked k m n N R A B) adv u := by
  classical
  unfold Word.identCount
  rw [filter_congr fun p hp => wrapIdentity k m n N R A B adv u p hV (mem_range.1 hp)
    fun ω q hq => hr ω u q hu (lt_of_le_of_lt hq (mem_range.1 hp))]

theorem wrapFinalOnlyInd : WrapFinalOnlyInd := by
  intro k m n N R A B adv h16 hI hV hr u hu
  rw [identCount_wrapped hV hr hu]
  exact ncpFinalOnlyInd k m n N R A B adv h16 hI hV u hu

theorem wrapWordFloorInd : WrapWordFloorInd := by
  intro k m n N R A B adv γ₀ h16 hI hγ0 hγ1 hV hr hNz
  refine (WordProofs.wordBound (Noise N m n k) (fun _ => positions k m n) (wrapped k m n N R A B) N adv γ₀
    hγ0 hγ1 hV hNz).trans ?_
  calc ∑ u ∈ range N, (2 : ℝ) ^ (-(γ₀ * ((fun _ => positions k m n) u : ℕ) -
        Word.identCount (fun _ => positions k m n) (wrapped k m n N R A B) adv u))
      ≤ ∑ _u ∈ range N, (2 : ℝ) ^ (-((m * n : ℕ) * (γ₀ * (3 * k / 16 : ℕ) - 1))) := sum_le_sum fun u hu => by
        apply Real.rpow_le_rpow_of_exponent_le (by norm_num)
        have h1 : (Word.identCount (fun _ => positions k m n) (wrapped k m n N R A B) adv u : ℝ) ≤ (m * n : ℕ) := by
          exact_mod_cast wrapFinalOnlyInd k m n N R A B adv h16 hI hV hr u (mem_range.1 hu)
        generalize Word.identCount (fun _ => positions k m n) (wrapped k m n N R A B) adv u = ic at h1 ⊢
        simp only [positions]
        push_cast at h1 ⊢
        nlinarith
    _ = N * (2 : ℝ) ^ (-((m * n : ℕ) * (γ₀ * (3 * k / 16 : ℕ) - 1))) := by
        rw [sum_const, card_range, nsmul_eq_mul]

theorem e1Of_noise {N m n k : ℕ} (ω : Noise N m n k) (u i l : ℕ) : InNoise6 (e1Of ω u i l) := by
  unfold e1Of InNoise6
  split_ifs with h
  · have := (ω.1 ⟨u, h.1⟩ ⟨i, h.2.1⟩ ⟨l, h.2.2⟩).isLt
    constructor <;> omega
  · norm_num

theorem f1Of_noise {N m n k : ℕ} (ω : Noise N m n k) (l j : ℕ) : InNoise6 (f1Of ω l j) := by
  unfold f1Of InNoise6
  split_ifs with h
  · have := (ω.2 ⟨l, h.1⟩ ⟨j, h.2⟩).isLt
    constructor <;> omega
  · norm_num

theorem abs32 {x : ℤ} (h : InNoise6 x) : |x| ≤ 32 := abs_le.2 ⟨by linarith [h.1], h.2⟩

theorem depth_le {k m n p : ℕ} (hp : p < positions k m n) : 16 * (p / (m * n) + 1) ≤ 3 * k := by
  have hmn : 0 < m * n := by
    rcases Nat.eq_zero_or_pos (m * n) with h | h
    · simp [positions, h] at hp
    · exact h
  have hlt : p / (m * n) < 3 * k / 16 := (Nat.div_lt_iff_lt_mul hmn).2 hp
  generalize p / (m * n) = q at hlt ⊢
  omega

theorem zeroInputInt32 : ZeroInputInt32 := by
  intro k m n N R ω u p hR hk hp
  exact zero_int32 hR (fun l _ => abs32 (e1Of_noise ω u _ l)) (fun l _ => abs32 (f1Of_noise ω l _)) hk
    (depth_le hp)

theorem abs_int7 {a : ℤ} (h : InInt7 a) : |a| ≤ 64 := abs_le.2 ⟨h.1, by linarith [h.2]⟩

theorem bnd {a c : ℤ} (h : |a| ≤ c) : -c ≤ a ∧ a ≤ c := abs_le.1 h

theorem int7Int32 : Int7Int32 := by
  intro k m n N R A B ω u p hR hk hA hB hp
  have hτ := depth_le hp
  show InInt32 (ck k (A u) (B u) (e1Of ω u) (f1Of ω) R (p % (m * n) / n) (p % (m * n) % n) (16 * (p / (m * n) + 1)))
  generalize 16 * (p / (m * n) + 1) = τ at hτ ⊢
  generalize p % (m * n) / n = i
  generalize p % (m * n) % n = j
  have hE : ∀ l < k, |e1Of ω u i l| ≤ 32 := fun l _ => abs32 (e1Of_noise ω u i l)
  have hF : ∀ l < k, |f1Of ω l j| ≤ 32 := fun l _ => abs32 (f1Of_noise ω l j)
  have x12 : ∀ l < 2 * k, -96 ≤ stackX k (A u) (e1Of ω u) R i l ∧ stackX k (A u) (e1Of ω u) R i l ≤ 96 := by
    intro l hl
    simp only [stackX, hcat3, Pi.add_apply]
    split_ifs with h1
    · have := bnd (abs_int7 (hA i l h1)); have := bnd (hE l h1); constructor <;> linarith
    · have := bnd (abs_int7 (hA i (l - k) (by omega))); have := bnd (e2_abs hR hE (show l - k < k by omega))
      constructor <;> linarith
  have x3 : ∀ l, 2 * k ≤ l → l < 3 * k →
      -128 ≤ stackX k (A u) (e1Of ω u) R i l ∧ stackX k (A u) (e1Of ω u) R i l ≤ 128 := by
    intro l hl1 hl
    simp only [stackX, hcat3, E3, Pi.add_apply, Pi.neg_apply, show ¬ l < k by omega, show ¬ l < 2 * k by omega,
      ite_false]
    have := bnd (abs_int7 (hA i (l - 2 * k) (by omega))); have := bnd (hE (l - 2 * k) (by omega))
    have := bnd (e2_abs hR hE (show l - 2 * k < k by omega))
    constructor <;> linarith
  have y12 : ∀ l < 2 * k, -96 ≤ stackY k (B u) R (f1Of ω) l j ∧ stackY k (B u) R (f1Of ω) l j ≤ 96 := by
    intro l hl
    simp only [stackY, vcat3, Pi.add_apply]
    split_ifs with h1
    · have := bnd (abs_int7 (hB l h1 j)); have := bnd (hF l h1); constructor <;> linarith
    · have := bnd (abs_int7 (hB (l - k) (by omega) j)); have := bnd (f2_abs hR hF (show l - k < k by omega))
      constructor <;> linarith
  have y3 : ∀ l, 2 * k ≤ l → l < 3 * k → -128 ≤ stackY k (B u) R (f1Of ω) l j ∧ stackY k (B u) R (f1Of ω) l j ≤ 128 := by
    intro l hl1 hl
    simp only [stackY, vcat3, F3, Pi.add_apply, show ¬ l < k by omega, show ¬ l < 2 * k by omega, ite_false]
    have := bnd (abs_int7 (hB (l - 2 * k) (by omega) j)); have := bnd (hF (l - 2 * k) (by omega))
    have := bnd (f2_abs hR hF (show l - 2 * k < k by omega))
    constructor <;> linarith
  have hk' : (k : ℤ) ≤ 65536 := by exact_mod_cast hk
  have h31 : (2 : ℤ) ^ 31 = 2147483648 := by norm_num
  show InInt32 (∑ l ∈ range τ, stackX k (A u) (e1Of ω u) R i l * stackY k (B u) R (f1Of ω) l j)
  unfold InInt32
  rw [h31]
  by_cases h2 : τ ≤ 2 * k
  · have hb := sum_bounds τ (fun l => stackX k (A u) (e1Of ω u) R i l * stackY k (B u) R (f1Of ω) l j)
      (-(96 * 96)) (96 * 96) fun l hl => mul_bound (x12 l (by omega)) (y12 l (by omega))
    have hτ' : (τ : ℤ) ≤ 2 * k := by exact_mod_cast h2
    constructor <;> nlinarith [hb.1, hb.2]
  · have hsplit := sum_range_add_sum_Ico
      (fun l => stackX k (A u) (e1Of ω u) R i l * stackY k (B u) R (f1Of ω) l j) hτ
    have htot : ∑ l ∈ range (3 * k), stackX k (A u) (e1Of ω u) R i l * stackY k (B u) R (f1Of ω) l j =
        ∑ l ∈ range k, A u i l * B u l j :=
      congrFun (congrFun (NCP.Proofs.stacked k (A u) (B u) (e1Of ω u) (f1Of ω) R).2 i) j
    have hAB := sum_bounds k (fun l => A u i l * B u l j) (-4032) 4096
      fun l hl => mul_bounds_int7 (hA i l hl) (hB l hl j)
    have hterm : ∀ l ∈ Ico τ (3 * k), -(128 * 128) ≤ stackX k (A u) (e1Of ω u) R i l * stackY k (B u) R (f1Of ω) l j ∧
        stackX k (A u) (e1Of ω u) R i l * stackY k (B u) R (f1Of ω) l j ≤ 128 * 128 := fun l hl => by
      obtain ⟨hl1, hl2⟩ := mem_Ico.1 hl
      exact mul_bound (x3 l (by omega) hl2) (y3 l (by omega) hl2)
    have hs1 := card_nsmul_le_sum (Ico τ (3 * k))
      (fun l => stackX k (A u) (e1Of ω u) R i l * stackY k (B u) R (f1Of ω) l j) (-(128 * 128)) fun l hl => (hterm l hl).1
    have hs2 := sum_le_card_nsmul (Ico τ (3 * k))
      (fun l => stackX k (A u) (e1Of ω u) R i l * stackY k (B u) R (f1Of ω) l j) (128 * 128) fun l hl => (hterm l hl).2
    rw [Nat.card_Ico, nsmul_eq_mul] at hs1 hs2
    have hc : ((3 * k - τ : ℕ) : ℤ) ≤ k := by exact_mod_cast (show 3 * k - τ ≤ k by omega)
    have hc0 : (0 : ℤ) ≤ ((3 * k - τ : ℕ) : ℤ) := by positivity
    constructor <;> nlinarith [hAB.1, hAB.2]

theorem leadIndWitness : LeadIndWitness := by
  have hN : leadParams.NoiseRange :=
    ⟨fun _ _ _ _ _ _ => by simp [leadParams, InNoise6], fun _ _ _ _ _ _ => by simp [leadParams, InNoise6]⟩
  have hadm : leadParams.RAdmissible := fun sh _ hsh =>
    shiftBy24_admissible (le_of_eq_of_le (by norm_num) hsh.2.2.1) (Nat.dvd_trans (by norm_num) hsh.2.2.2.2.1)
  have hind : RIndependent leadParams := fun sh _ hsh =>
    indep_of_fresh (fresh (le_of_eq_of_le (by norm_num) hsh.2.2.1) (Nat.dvd_trans (by norm_num) hsh.2.2.2.2.1))
  have hS : leadParams.RSigned := fun sh b hsh => (hadm sh b hsh).1
  have hq : leadParams.QueryBound 0 := fun _ _ _ _ => by simp [leadParams]
  have hW : ∀ U : Workload, U.InDomain leadDomain → ∀ u < U.layout.N,
      3 * (U.layout.shape u).Wmm ≤ leadParams.Wref U.layout u := fun _ _ _ _ => le_rfl
  have hadmit := (ttncpWitness Unit Unit leadParams 0 0 le_rfl one_pos hq hW).2
  simp only [mul_zero] at hadmit
  exact ⟨leadParams, leadLayout, ⟨⟨rfl, hN, hadm⟩, hind⟩, hS, hq, fun _ _ => rfl, leadComplete Unit Unit Salt leadParams,
    hadmit, fun γ₀ h0 h1 => (ttncpWitness Unit Unit leadParams 0 γ₀ h0 h1 hq hW).1, leadLayout_inDomain, one_pos,
    fun _ _ => ⟨64, rfl⟩⟩

end Pouw.NCP.ChainProofs
