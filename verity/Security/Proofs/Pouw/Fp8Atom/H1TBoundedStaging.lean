import Proofs.Pouw.Fp8Atom.H1TRunningProofs
import Proofs.Pouw.Fp8Atom.H1TGapStaging

/-!
# H-1T at bounded `k` (staging, unpinned)

`h1tGamma` reads `DistinctLiveH1T` only at the units it is given (`h1tUnitsDistinct`), so a bound on `k` passes
through. This file states the named assumptions and `H1TGamma` for the legal units with `k ≤ K`, and proves the
bounded forms of `h1tDistinctLive`, `h1tRunningOfDistinct`, `h1tGamma` and `h1tGammaRunning`. At `K = 32,768`, which
`Legal` already imposes, each bounded statement is the granted one (`*_iff`), so the family changes none of them.

* **`DistinctLiveH1TUpTo K`**, **`H1TRunningWordsUpTo K`** and **`H1TCrossLateUpTo K`** are `DistinctLiveH1T`,
  `H1TRunningWords` and `H1TCrossLate` on the legal units with `k ≤ K`.
* **`H1TGammaUpTo K`** is `H1TGamma` for families of legal units that all have `k ≤ K`. Both γ shapes (8,192³ and
  16,384³) are units with `k ≤ 16,384`, so `H1TGammaUpTo 16384` under `DistinctLiveH1TUpTo 16384` is γ₀ = 0 there.
* **`run_le`**: every running word of a legal unit is at most `T·6,881,792 − 256` in magnitude (`run_bound`). So
  `|R_p| < 2^31` (ulp at most 128) for `k ≤ 9,048` (`T ≤ 312`, which covers 8,192³), and `|R_p| < 2^32` (ulp at most
  256) for `k ≤ 18,096` (`T ≤ 624`, which covers 16,384³).
-/

namespace Pouw.Fp8Atom.H1T.Bounded

open Pouw.Fp8Atom Pouw.Fp8Atom.H1T Pouw.Fp8Atom.H1T.Proofs Pouw.Dimension.H100

/-- `DistinctLiveH1T` on the legal units with `k ≤ K`. -/
def DistinctLiveH1TUpTo (K : ℕ) : Prop :=
  ∀ U : Unit, U.Legal → U.k ≤ K → Function.Injective U.word ∧ ∀ c, U.word c ∉ U.free

/-- `H1TRunningWords` on the legal units with `k ≤ K`. -/
def H1TRunningWordsUpTo (K : ℕ) : Prop :=
  ∀ U : Unit, U.Legal → U.k ≤ K → ∀ (i : Fin U.m) (a a' : Fin U.n × Pos U.T),
    (a ≠ a' → Hard a.1.val a'.1.val a.2 a'.2 → U.word (i, a) ≠ U.word (i, a')) ∧
      (Stuck a.1.val a.2 → ∀ q : ℚ, U.word (i, a) ≠ fun _ => q)

/-- `H1TCrossLate` on the legal units with `k ≤ K`. -/
def H1TCrossLateUpTo (K : ℕ) : Prop :=
  ∀ U : Unit, U.Legal → U.k ≤ K → ∀ (i : Fin U.m) (a a' : Fin U.n × Pos U.T), a.1 ≠ a'.1 → Run2 a.2 →
    a'.2.isRight = true → U.T ≤ posIdx a'.2 + 1 → ¬ OppLater a.1.val a'.1.val a.2 a'.2 →
    U.word (i, a) ≠ U.word (i, a')

/-- `H1TGamma` for families of legal units that all have `k ≤ K`. -/
def H1TGammaUpTo (K : ℕ) : Prop :=
  ∀ pr : Prices100, H32 pr → ∀ (N : ℕ) (Us : Fin N → Unit), (∀ u, (Us u).Legal ∧ (Us u).k ≤ K) →
    ∀ (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (L : List (MSalt Us → ℚ)) (P : List (Op100 (MSalt Us))),
      (∀ f ∈ L, f ∈ mfree Us) → (∀ o ∈ P, o.WFw pr (mfree Us)) →
      (∀ x : Idx Us, ∃ r ∈ regs100 fpSem P L, ∀ G, r G = mword Us x G) →
      32 * ((∑ u, (Us u).m * (Us u).n * Fintype.card (Pos (Us u).T) : ℕ) : ℚ) ≤ (cost100 pr P : ℚ)

theorem k_le_of_legal {U : Unit} (hU : U.Legal) : U.k ≤ 32768 := hU.1

/-! ## At `K = 32,768` the bounded statements are the granted ones -/

theorem distinctLiveUpTo_iff : DistinctLiveH1TUpTo 32768 ↔ DistinctLiveH1T :=
  ⟨fun h U hU => h U hU (k_le_of_legal hU), fun h U hU _ => h U hU⟩

theorem runningWordsUpTo_iff : H1TRunningWordsUpTo 32768 ↔ H1TRunningWords :=
  ⟨fun h U hU => h U hU (k_le_of_legal hU), fun h U hU _ => h U hU⟩

theorem crossLateUpTo_iff : H1TCrossLateUpTo 32768 ↔ H1TCrossLate :=
  ⟨fun h U hU => h U hU (k_le_of_legal hU), fun h U hU _ => h U hU⟩

theorem gammaUpTo_iff : H1TGammaUpTo 32768 ↔ H1TGamma :=
  ⟨fun h pr hpr N Us hL => h pr hpr N Us fun u => ⟨hL u, k_le_of_legal (hL u)⟩,
   fun h pr hpr N Us hL => h pr hpr N Us fun u => (hL u).1⟩

theorem distinctLiveUpTo_of (h : DistinctLiveH1T) (K : ℕ) : DistinctLiveH1TUpTo K := fun U hU _ => h U hU

theorem runningWordsUpTo_of (h : H1TRunningWords) (K : ℕ) : H1TRunningWordsUpTo K := fun U hU _ => h U hU

theorem distinctLiveUpTo_mono {K K' : ℕ} (hK : K ≤ K') (h : DistinctLiveH1TUpTo K') : DistinctLiveH1TUpTo K :=
  fun U hU hk => h U hU (le_trans hk hK)

theorem runningWordsUpTo_mono {K K' : ℕ} (hK : K ≤ K') (h : H1TRunningWordsUpTo K') : H1TRunningWordsUpTo K :=
  fun U hU hk => h U hU (le_trans hk hK)

/-! ## The bounded forms of the M4 and γ theorems -/

theorem h1tDistinctLiveUpTo {K : ℕ} (hR : H1TRunningWordsUpTo K) : DistinctLiveH1TUpTo K := by
  intro U hU hk
  have hF := h1tOneFlip U hU
  have : Nonempty (Fin U.T → RawSlice) := ⟨fun _ => rawZero⟩
  have hnc : ∀ (i : Fin U.m) (a : Fin U.n × Pos U.T) (q : ℚ), U.word (i, a) ≠ fun _ => q := by
    intro i a q e
    by_cases hr : Stuck a.1.val a.2
    · exact (hR U hU hk i a a).2 hr q e
    · obtain ⟨s, hs⟩ := exists_of_half _ ((hF i).2 a hr q)
      exact hs (congrFun e _)
  have hsep : ∀ (i : Fin U.m) (a a' : Fin U.n × Pos U.T), a ≠ a' → U.word (i, a) ≠ U.word (i, a') := by
    intro i a a' hne e
    by_cases hH : Hard a.1.val a'.1.val a.2 a'.2
    · exact (hR U hU hk i a a').1 hne hH e
    · obtain ⟨s, hs⟩ := exists_of_half _ ((hF i).1 a a' hne hH)
      exact hs (congrFun e _)
  refine ⟨fun c c' e => ?_, fun c hc => ?_⟩
  · obtain ⟨i, a⟩ := c
    obtain ⟨i', a'⟩ := c'
    by_cases hi : i = i'
    · subst hi
      by_contra hne
      exact hsep i a a' (fun h => hne (by rw [h])) e
    · exfalso
      let g0 : U.Salt := fun _ _ => rawZero
      obtain ⟨g, hg⟩ : ∃ g, U.word (i, a) g ≠ U.word (i, a) g0 := by
        by_contra hc
        simp only [not_exists, not_not] at hc
        exact hnc i a (U.word (i, a) g0) (funext hc)
      let g1 : U.Salt := Function.update g0 i (g i)
      have h1 : U.word (i, a) g1 = U.word (i, a) g := word_row U (i, a) (by simp [g1])
      have h2 : U.word (i', a') g1 = U.word (i', a') g0 :=
        word_row U (i', a') (by simp [g1, Function.update_of_ne (Ne.symm hi)])
      apply hg
      rw [← h1, e, h2]
  · rcases hc with ⟨q, hq⟩ | ⟨i, τ, ℓ, hr⟩
    · exact hnc c.1 c.2 q hq
    · exact h1tNotRawWord U c i τ ℓ hr

theorem h1tRunningOfDistinctUpTo {K : ℕ} (hD : DistinctLiveH1TUpTo K) : H1TRunningWordsUpTo K := by
  intro U hU hk i a a'
  obtain ⟨hinj, hfree⟩ := hD U hU hk
  exact ⟨fun hne _ e => hne (congrArg Prod.snd (hinj e)), fun _ q e => hfree (i, a) (Or.inl ⟨q, e⟩)⟩

theorem unitsDistinctUpTo {K N : ℕ} {Us : Fin N → Unit} (hD : DistinctLiveH1TUpTo K)
    (hL : ∀ u, (Us u).Legal ∧ (Us u).k ≤ K) : Function.Injective (mword Us) ∧ ∀ x, mword Us x ∉ mfree Us := by
  have hD' : ∀ u, Function.Injective (Us u).word ∧ ∀ c, (Us u).word c ∉ (Us u).free := fun u =>
    hD (Us u) (hL u).1 (hL u).2
  have hconst : ∀ u c (q : ℚ), (Us u).word c ≠ fun _ => q := fun u c q e =>
    (hD' u).2 c (e ▸ Or.inl ⟨q, rfl⟩)
  refine ⟨?_, ?_⟩
  · rintro ⟨u, c⟩ ⟨u', c'⟩ e
    by_cases hu : u' = u
    · subst hu
      have : (Us u').word c = (Us u').word c' := by
        funext s
        rw [← mword_place_self, ← mword_place_self, e]
      rw [(hD' u').1 this]
    · refine absurd ?_ (hconst u c (mword Us ⟨u', c'⟩ (zeroSalt Us)))
      funext s
      rw [← mword_place_self, e, mword_place_ne hu]
  · rintro ⟨u, c⟩ (⟨q, e⟩ | ⟨u', i, τ, ℓ, e⟩)
    · refine hconst u c q (funext fun s => ?_)
      rw [← mword_place_self, e]
    · by_cases hu : u' = u
      · subst hu
        refine (hD' u').2 c (Or.inr ⟨i, τ, ℓ, funext fun s => ?_⟩)
        rw [← mword_place_self, e]
        simp [place, Function.update_self]
      · refine hconst u c 0 (funext fun s => ?_)
        rw [← mword_place_self, e]
        simp [place, Function.update_of_ne hu, zeroSalt]

theorem h1tGammaUpTo {K : ℕ} (hD : DistinctLiveH1TUpTo K) : H1TGammaUpTo K := by
  intro pr hpr N Us hL fpSem L P hLf hP hout
  obtain ⟨hinj, hfree⟩ := unitsDistinctUpTo hD hL
  let ts : List (MSalt Us → ℚ) := (Finset.univ : Finset (Idx Us)).toList.map (mword Us)
  have hlen : ts.length = ∑ u, (Us u).m * (Us u).n * Fintype.card (Pos (Us u).T) := by
    simp only [ts, List.length_map, Finset.length_toList, Finset.card_univ, Fintype.card_sigma, Fintype.card_prod,
      Fintype.card_fin, mul_assoc]
  have hDL : DistinctLiveAt 0 Set.univ (mfree Us) ts := by
    refine ⟨Finset.univ.image (mword Us), ?_, ⟨?_, ?_⟩, ?_⟩
    · intro t ht
      obtain ⟨x, -, rfl⟩ := Finset.mem_image.1 ht
      simp [ts]
    · intro t _ t' _ h
      exact funext fun g => h g trivial
    · intro t ht f hf h
      obtain ⟨x, -, rfl⟩ := Finset.mem_image.1 ht
      exact hfree x ((funext fun g => h g trivial : mword Us x = f) ▸ hf)
    · rw [Finset.card_image_of_injective _ hinj, Finset.card_univ]
      have : ts.length = Fintype.card (Idx Us) := by simp [ts]
      rw [this]; simp
  have hts : ∀ t ∈ ts, ∃ r ∈ regs100 fpSem P L, ∀ g ∈ (Set.univ : Set (MSalt Us)), r g = t g := by
    intro t ht
    obtain ⟨x, -, rfl⟩ := List.mem_map.1 ht
    obtain ⟨r, hr, e⟩ := hout x
    exact ⟨r, hr, fun g _ => e g⟩
  have h := distinctLiveMH100 pr hpr (MSalt Us) (mfree Us) (fun q => Or.inl ⟨q, rfl⟩) 0 Set.univ ts fpSem L P hDL
    hLf hP hts
  rw [hlen] at h
  simpa using h

theorem h1tGammaRunningUpTo {K : ℕ} (hR : H1TRunningWordsUpTo K) : H1TGammaUpTo K :=
  h1tGammaUpTo (h1tDistinctLiveUpTo hR)

/-! ## The running words' range at bounded `k` -/

theorem run_le {U : Unit} (hU : U.Legal) (i : Fin U.m) (j : Fin U.n) (g : U.Salt) {p : ℕ} (hp : p < 2 * U.T) :
    |runVal U.T (U.row i) (U.col j) j.val (U.rowSalt g i) p| ≤ (U.T : ℚ) * 6881792 - 256 := by
  obtain ⟨hk, -, hxq, hbq⟩ := hU
  have hx : ∀ r, (U.row i r).isNaN = false := fun r => by
    unfold Unit.row; split_ifs
    · exact hxq _ _
    · rfl
  have hb : ∀ r, (U.col j r).isNaN = false := fun r => by
    unfold Unit.col; split_ifs
    · exact hbq _ _
    · rfl
  have hT : U.T ≤ 1130 := by unfold Unit.T; omega
  have hF := fresh_facts (T := U.T) hx hb j.val (U.rowSalt g i)
  have hB := run_bound hT (fun p => (hF p).2) p hp
  split_ifs at hB with h
  · have hq : (p : ℚ) + 1 ≤ U.T := by exact_mod_cast h
    nlinarith
  · have hq : (p : ℚ) + 1 ≤ 2 * U.T := by exact_mod_cast hp
    linarith

/-- For `k ≤ 9,048` (`T ≤ 312`; 8,192³ has `T = 283`) every running word is below `2^31`, so its ulp is at most
128. -/
theorem run_lt_2pow31 {U : Unit} (hU : U.Legal) (hk : U.k ≤ 9048) (i : Fin U.m) (j : Fin U.n) (g : U.Salt) {p : ℕ}
    (hp : p < 2 * U.T) : |runVal U.T (U.row i) (U.col j) j.val (U.rowSalt g i) p| < 2 ^ 31 := by
  have hT : U.T ≤ 312 := by unfold Unit.T; omega
  have hTq : (U.T : ℚ) ≤ 312 := by exact_mod_cast hT
  have := run_le hU i j g hp
  norm_num; linarith

/-- For `k ≤ 18,096` (`T ≤ 624`; 16,384³ has `T = 565`) every running word is below `2^32`, so its ulp is at most
256. -/
theorem run_lt_2pow32 {U : Unit} (hU : U.Legal) (hk : U.k ≤ 18096) (i : Fin U.m) (j : Fin U.n) (g : U.Salt) {p : ℕ}
    (hp : p < 2 * U.T) : |runVal U.T (U.row i) (U.col j) j.val (U.rowSalt g i) p| < 2 ^ 32 := by
  have hT : U.T ≤ 624 := by unfold Unit.T; omega
  have hTq : (U.T : ℚ) ≤ 624 := by exact_mod_cast hT
  have := run_le hU i j g hp
  norm_num; linarith

end Pouw.Fp8Atom.H1T.Bounded
