import Proofs.Flock.Claims.Complete.Zk.Coins
import Proofs.Flock.Claims.KS.Range

/-!
# The rank bound at the verifier's coins

`MaskRankBound` (each table's rank checks refuse at most `4·m·2^-128` of the verifier's coins) from
`DeployedRankBound` (four points of ten coordinates, the `c` claims' first six pinned, rank-deficient at most `2^-122`
of the time; proved in `Zk.RankDeployed`) and, for a slot under `2^13` bits, the open `SmallSlotBound`
(`maskRankBound_of`).

At a table with `s = slot_log − 7 ≥ 6` and `K = k_log − 7`, the slot's words `b·2^K + pos0·2^s + j` (`b < 2^nbl`,
`j < 2^s`) hold every pattern on ten coordinates (`pos`: the slot's low `min s 10`, then the block index's), so the
words `word y` for `y : Fin 10 → Bool` are among them (`word_mem`). Each word's weight vector is the ten coordinates'
eq-vector times a factor `gammaOf` from the other `m − 17` (`weights_word`), so where every factor is nonzero, full
rank at the ten gives the prover's check (`maskRankOk_of_chosen`, by `rankFull_scale_iff`). A factor is zero only when
its coordinate's coin takes one value (`factor_ne_zero`), `4·(m − 17)·2^-128` together (`pr_list_le`). The ten
coordinates' coins are distinct coins of the verifier's (`tauN_inj`: the reps' rounds are apart; an `ab` claim reads
rounds from `k_log − 3` on, a `c` claim round `1`), and the pinned ones are the same constants as `deployedFix`'s
(`claimPoints_c`), so at uniform coins the ten are `DeployedRankBound`'s distribution (`Pr_coins_inj`):
`4·(m − 17)·2^-128 + 2^-122 ≤ 4·m·2^-128` (`rankBoundAt_of_deployed`).
-/

namespace FlockComplete.MaskRank

open FlockLevel3 Flock FlockDeploy
open FlockSoundness.Discharge.Composed (eqWord maskWordsOf)

/-- **Uniform coins seen through an injective map are uniform**: `f ↦ f ∘ τ` pushes the uniform distribution on
`γ → F` to the uniform one on `β → F`. -/
theorem Pr_comp_inj {β γ F : Type} [Fintype β] [Fintype γ] [Fintype F] [Nonempty F] [DecidableEq β] [DecidableEq γ]
    (τ : β → γ) (hτ : Function.Injective τ) (P : (β → F) → Prop) :
    Claims.Pr (fun f : γ → F => P (f ∘ τ)) = Claims.Pr P := by
  classical
  let e₁ := Equiv.piEquivPiSubtypeProd (fun c => c ∈ Set.range τ) (fun _ => F)
  let e₂ : ({c // c ∈ Set.range τ} → F) ≃ (β → F) :=
    Equiv.arrowCongr (Equiv.ofInjective τ hτ).symm (Equiv.refl F)
  let e := e₁.trans (Equiv.prodCongr e₂ (Equiv.refl _))
  have he : ∀ f : γ → F, (e f).1 = f ∘ τ := by
    intro f
    funext b
    simp [e, e₁, e₂, Equiv.arrowCongr_apply]
  have h1 : Claims.Pr (fun f : γ → F => P (f ∘ τ)) = Claims.Pr (fun f : γ → F => (fun s => P s.1) (e f)) :=
    Claims.Pr_congr fun f => by show P (f ∘ τ) ↔ P (e f).1; rw [he]
  rw [h1, Pr_equiv e (fun s => P s.1)]
  exact Claims.Pr_fst P

/-- **The verifier's coins seen at injectively chosen elements are uniform** in `GF(2^128)`. -/
theorem Pr_coins_inj {k a b : ℕ} (τ : Fin a × Fin b → Fin k × Fin 65536) (hτ : Function.Injective τ)
    (P : (Fin a → Fin b → GF128) → Prop) :
    Claims.Pr (fun r : Fin k → Coin => P fun i u => r (τ (i, u)).1 (τ (i, u)).2) = Claims.Pr P := by
  let e : (Fin k → Coin) ≃ (Fin k × Fin 65536 → GF128) := (Equiv.curry (Fin k) (Fin 65536) GF128).symm
  have h1 : Claims.Pr (fun r : Fin k → Coin => (fun g => P (Function.curry (g ∘ τ))) (e r)) =
      Claims.Pr (fun g : Fin k × Fin 65536 → GF128 => P (Function.curry (g ∘ τ))) :=
    Pr_equiv e (fun g => P (Function.curry (g ∘ τ)))
  have h2 := Pr_comp_inj τ hτ (fun g => P (Function.curry g))
  have h3 : Claims.Pr (fun g : Fin a × Fin b → GF128 => P (Equiv.curry (Fin a) (Fin b) GF128 g)) = Claims.Pr P :=
    Pr_equiv (Equiv.curry (Fin a) (Fin b) GF128) P
  exact h1.trans (h2.trans h3)


/-- The round of coordinate `t` of a rep's `ab` claim point: the lincheck's `t_{k_log − 8 − t}` below `k_log − 7`, else
the zerocheck's `ρ_{t + 1}`. -/
def abRound (m kLog t : ℕ) : ℕ := if t < kLog - 7 then 5 + (m - 6) + (kLog - 8 - t) else 4 + t

theorem pinned_size : Flock.pinned.size = 7 := rfl

theorem claimPoints_eq (m kLog : ℕ) (coin : ℕ → ℕ → F128) :
    claimPoints m kLog coin = #[((Array.range (kLog - 6)).map (fun j => coin (5 + (m - 6) + j) 0)).reverse ++
        ((Array.range (m - 6)).map fun j => coin (3 + j) 0).extract (kLog - 6) (m - 6) |>.toList.drop 1,
      (Flock.pinned ++ (Array.range (m - 13)).map (coin 1)).toList.drop 1] := rfl

/-- **A rep's `ab` claim point**: `m − 7` coordinates, coordinate `t` the coin element `0` of round `abRound t`. -/
theorem claimPoints_ab {m kLog : ℕ} (coin : ℕ → ℕ → F128) (hk : 7 ≤ kLog) (hm : kLog ≤ m) :
    ((claimPoints m kLog coin)[0]!).length = m - 7 ∧
      ∀ t < m - 7, ((claimPoints m kLog coin)[0]!)[t]! = coin (abRound m kLog t) 0 := by
  rw [claimPoints_eq, show ∀ a b : List F128, (#[a, b] : Array (List F128))[0]! = a from fun _ _ => rfl]
  refine ⟨by simp; omega, fun t ht => ?_⟩
  rw [getElem!_pos _ t (by simp; omega)]
  simp only [List.getElem_drop, Array.getElem_toList]
  unfold abRound
  split
  · rw [Array.getElem_append_left (by simp; omega)]
    simp only [Array.getElem_reverse, Array.getElem_map, Array.getElem_range, Array.size_map, Array.size_range]
    congr 2
    omega
  · rw [Array.getElem_append_right (by simp; omega)]
    simp only [Array.getElem_extract, Array.getElem_map, Array.getElem_range, Array.size_reverse, Array.size_map,
      Array.size_range]
    congr 1
    omega

/-- **A rep's `c` claim point**: `m − 7` coordinates, the pinned `pinned[1..7)` then round `1`'s elements. -/
theorem claimPoints_c {m kLog : ℕ} (coin : ℕ → ℕ → F128) (hm : 13 ≤ m) :
    ((claimPoints m kLog coin)[1]!).length = m - 7 ∧
      ∀ t < m - 7, ((claimPoints m kLog coin)[1]!)[t]! = if t < 6 then Flock.pinned[t + 1]! else coin 1 (t - 6) := by
  rw [claimPoints_eq, show ∀ a b : List F128, (#[a, b] : Array (List F128))[1]! = b from fun _ _ => rfl]
  refine ⟨by simp [pinned_size]; omega, fun t ht => ?_⟩
  rw [getElem!_pos _ t (by simp [pinned_size]; omega)]
  simp only [List.getElem_drop, Array.getElem_toList]
  split
  · rw [Array.getElem_append_left (by simp [pinned_size]; omega), getElem!_pos Flock.pinned (t + 1) (by
      rw [pinned_size]; omega)]
    simp only [Nat.add_comm 1 t]
  · rw [Array.getElem_append_right (by simp [pinned_size]; omega)]
    simp only [Array.getElem_map, Array.getElem_range, pinned_size]
    congr 1
    omega

theorem append_claims (st : Flock.Stmt) {k : ℕ} (r : Fin k → Coin) (n₀ n₁ : ℕ) :
    claimsAt st r n₀ ++ claimsAt st r n₁ = #[(claimsAt st r n₀)[0]!, (claimsAt st r n₀)[1]!,
      (claimsAt st r n₁)[0]!, (claimsAt st r n₁)[1]!] := rfl

/-- The number with bits `f 0, …, f (n − 1)`. -/
def bitsVal (f : ℕ → Bool) : ℕ → ℕ
  | 0 => 0
  | n + 1 => 2 ^ n * (if f n then 1 else 0) + bitsVal f n

theorem bitsVal_lt (f : ℕ → Bool) : ∀ n, bitsVal f n < 2 ^ n
  | 0 => by simp [bitsVal]
  | n + 1 => by
    have := bitsVal_lt f n
    rw [bitsVal, pow_succ]
    split <;> omega

theorem testBit_bitsVal (f : ℕ → Bool) : ∀ n t, (bitsVal f n).testBit t = (decide (t < n) && f t)
  | 0, t => by simp [bitsVal]
  | n + 1, t => by
    rw [bitsVal, Nat.testBit_two_pow_mul_add _ (bitsVal_lt f n), testBit_bitsVal f n]
    by_cases ht : t < n
    · simp [ht, show t < n + 1 by omega]
    · simp only [ht, ↓reduceIte]
      by_cases he : t = n
      · subst he
        split <;> simp_all
      · have : t - n ≠ 0 := by omega
        simp only [show ¬ t < n + 1 by omega, decide_false, Bool.false_and]
        split
        · rw [Bool.eq_false_iff]
          exact fun h => this (Nat.testBit_one_eq_true_iff_self_eq_zero.1 h)
        · exact Nat.zero_testBit _

theorem of_foldl_mul (g : ℕ → F128) :
    ∀ (l : List ℕ) (a : F128), GF128.of (l.foldl (fun acc i => acc * g i) a) =
      GF128.of a * (l.map fun i => GF128.of (g i)).prod
  | [], a => by simp
  | i :: l, a => by
    rw [List.foldl_cons, of_foldl_mul g l, List.map_cons, List.prod_cons, of_mul, mul_assoc]

theorem list_prod_range (f : ℕ → GF128) : ∀ n, ((List.range n).map f).prod = ∏ t ∈ Finset.range n, f t
  | 0 => by simp
  | n + 1 => by rw [List.range_succ, List.map_append, List.prod_append, list_prod_range f n,
      Finset.prod_range_succ]; simp

theorem of_eqWord (p : List F128) (x : ℕ) :
    GF128.of (eqWord p x) = ∏ t ∈ Finset.range p.length,
      if x.testBit t then GF128.of p[t]! else 1 + GF128.of p[t]! := by
  unfold eqWord
  rw [of_foldl_mul, list_prod_range, ← GF128.one_eq, one_mul]
  refine Finset.prod_congr rfl fun t _ => ?_
  split
  · rfl
  · rw [GF128.one_eq, GF128.add_eq]

/-! ## Ten of the slot's word coordinates -/

section Word

variable (s K : ℕ)

/-- The chosen bits `y` on `ℕ`, `false` past the tenth. -/
def yN (y : Fin 10 → Bool) (v : ℕ) : Bool := if h : v < 10 then y ⟨v, h⟩ else false

/-- **The `u`th chosen word coordinate**: the slot's low `min s 10` bits, then the block index's from bit `K` on. -/
def pos (u : Fin 10) : ℕ := if u.val < min s 10 then u.val else K + (u.val - min s 10)

/-- **The slot's word at the chosen bits `y`**: block `bitsVal` of `y`'s last `10 − min s 10`, word
`2^s·pos0 + bitsVal` of its first `min s 10` within the block (`K` its bits). -/
def word (pos0 : ℕ) (y : Fin 10 → Bool) : ℕ :=
  2 ^ K * bitsVal (fun v => yN y (min s 10 + v)) (10 - min s 10) + (2 ^ s * pos0 + bitsVal (yN y) (min s 10))

variable {s K}

theorem pos_injective (hsK : s ≤ K) : Function.Injective (pos s K) := by
  intro u v h
  unfold pos at h
  apply Fin.ext
  split_ifs at h <;> omega

theorem pos_lt {nbl : ℕ} (hsK : s ≤ K) (hD : 10 ≤ s + nbl) (u : Fin 10) : pos s K u < K + nbl := by
  unfold pos
  have := u.2
  split_ifs <;> omega

theorem pos_small (h6 : 6 ≤ s) (u : Fin 10) (hu : u.val < 6) : pos s K u = u.val := by
  unfold pos
  simp only [show u.val < min s 10 from lt_min (lt_of_lt_of_le hu h6) (by omega), ↓reduceIte]

theorem pos_ge (hK : 6 ≤ K) (u : Fin 10) (hu : 6 ≤ u.val) : 6 ≤ pos s K u := by
  unfold pos
  split_ifs <;> omega

theorem testBit_word {pos0 : ℕ} (hK : pos0 * 2 ^ s + 2 ^ s ≤ 2 ^ K) (y : Fin 10 → Bool) (t : ℕ) :
    (word s K pos0 y).testBit t =
      if t < K then (if t < s then (decide (t < min s 10) && yN y t) else pos0.testBit (t - s))
      else (decide (t - K < 10 - min s 10) && yN y (min s 10 + (t - K))) := by
  have hb : bitsVal (yN y) (min s 10) < 2 ^ s :=
    lt_of_lt_of_le (bitsVal_lt _ _) (Nat.pow_le_pow_right (by norm_num) (min_le_left _ _))
  have hlo : 2 ^ s * pos0 + bitsVal (yN y) (min s 10) < 2 ^ K := by
    rw [mul_comm]; omega
  rw [word, Nat.testBit_two_pow_mul_add _ hlo, Nat.testBit_two_pow_mul_add _ hb, testBit_bitsVal, testBit_bitsVal]

theorem testBit_word_pos {pos0 : ℕ} (hK : pos0 * 2 ^ s + 2 ^ s ≤ 2 ^ K) (hsK : s ≤ K) (y : Fin 10 → Bool)
    (u : Fin 10) : (word s K pos0 y).testBit (pos s K u) = y u := by
  rw [testBit_word hK]
  have hu := u.2
  unfold pos
  by_cases h1 : u.val < min s 10
  · simp only [h1, show u.val < K from (lt_of_lt_of_le h1 (min_le_left _ _)).trans_le hsK,
      show u.val < s by omega, ↓reduceIte, decide_true, Bool.true_and]
    simp [yN, hu]
  · have e : min s 10 + (K + (u.val - min s 10) - K) = u.val := by omega
    simp only [h1, show ¬ K + (u.val - min s 10) < K by omega, ↓reduceIte,
      show K + (u.val - min s 10) - K < 10 - min s 10 by omega, decide_true, Bool.true_and, e]
    simp [yN, hu]

theorem testBit_word_fixed {pos0 : ℕ} (hK : pos0 * 2 ^ s + 2 ^ s ≤ 2 ^ K) {t : ℕ}
    (ht : ∀ u, pos s K u ≠ t) (y y' : Fin 10 → Bool) :
    (word s K pos0 y).testBit t = (word s K pos0 y').testBit t := by
  rw [testBit_word hK, testBit_word hK]
  split_ifs with h1 h2
  · have : ¬ t < min s 10 := fun h => ht ⟨t, by omega⟩ (by unfold pos; simp [h])
    simp [this]
  · rfl
  · have : ¬ t - K < 10 - min s 10 := fun h =>
      ht ⟨min s 10 + (t - K), by omega⟩ (by
        unfold pos
        simp only [show ¬ min s 10 + (t - K) < min s 10 by omega, ↓reduceIte]
        omega)
    simp [this]

theorem word_mem {nbl pos0 : ℕ} (hD : 10 ≤ s + nbl) (y : Fin 10 → Bool) :
    word s K pos0 y ∈ (List.range (2 ^ nbl)).flatMap fun b =>
      (List.range (2 ^ s)).map fun j => b * 2 ^ K + pos0 * 2 ^ s + j := by
  rw [List.mem_flatMap]
  refine ⟨bitsVal (fun v => yN y (min s 10 + v)) (10 - min s 10), List.mem_range.2 ?_, List.mem_map.2
    ⟨bitsVal (yN y) (min s 10), List.mem_range.2 ?_, ?_⟩⟩
  · exact lt_of_lt_of_le (bitsVal_lt _ _) (Nat.pow_le_pow_right (by norm_num) (by omega))
  · exact lt_of_lt_of_le (bitsVal_lt _ _) (Nat.pow_le_pow_right (by norm_num) (min_le_left _ _))
  · unfold word; ring

/-- The word coordinates not chosen. -/
def others (nbl : ℕ) : Finset ℕ := Finset.range (K + nbl) \ Finset.univ.image (pos s K)

theorem others_card {nbl : ℕ} (hsK : s ≤ K) (hD : 10 ≤ s + nbl) :
    (others (s := s) (K := K) nbl).card = K + nbl - 10 := by
  unfold others
  rw [Finset.card_sdiff_of_subset (fun t ht => by
      obtain ⟨u, -, rfl⟩ := Finset.mem_image.1 ht
      exact Finset.mem_range.2 (pos_lt hsK hD u)),
    Finset.card_range, Finset.card_image_of_injective _ (pos_injective hsK), Finset.card_univ, Fintype.card_fin]

theorem mem_others {nbl t : ℕ} (h : t ∈ others (s := s) (K := K) nbl) : t < K + nbl ∧ ∀ u, pos s K u ≠ t := by
  unfold others at h
  rw [Finset.mem_sdiff, Finset.mem_range, Finset.mem_image] at h
  exact ⟨h.1, fun u hu => h.2 ⟨u, Finset.mem_univ _, hu⟩⟩

/-- **`eq(p, ·)` at the chosen words**: the chosen coordinates' `eq` factors times the others', fixed. -/
theorem eqWord_word {nbl pos0 : ℕ} (hK : pos0 * 2 ^ s + 2 ^ s ≤ 2 ^ K) (hsK : s ≤ K) (hD : 10 ≤ s + nbl)
    {p : List F128} (hp : p.length = K + nbl) (y : Fin 10 → Bool) :
    GF128.of (eqWord p (word s K pos0 y)) =
      (∏ t ∈ others (s := s) (K := K) nbl,
        if (word s K pos0 fun _ => false).testBit t then GF128.of p[t]! else 1 + GF128.of p[t]!) *
      ∏ u, (if y u then GF128.of p[pos s K u]! else 1 + GF128.of p[pos s K u]!) := by
  have hsub : Finset.univ.image (pos s K) ⊆ Finset.range (K + nbl) := fun t ht => by
    obtain ⟨u, -, rfl⟩ := Finset.mem_image.1 ht
    exact Finset.mem_range.2 (pos_lt hsK hD u)
  rw [of_eqWord, hp, ← Finset.prod_sdiff hsub, Finset.prod_image fun u _ v _ h => pos_injective hsK h]
  refine congrArg₂ (· * ·) (Finset.prod_congr rfl fun t ht => ?_) (Finset.prod_congr rfl fun u _ => ?_)
  · rw [testBit_word_fixed hK (mem_others ht).2 y fun _ => false]
  · rw [testBit_word_pos hK hsK]

/-- The others' factor at point `i`. -/
noncomputable def gammaOf (nbl pos0 : ℕ) (P : Array (List F128)) : Fin P.size → GF128 := fun i =>
  ∏ t ∈ others (s := s) (K := K) nbl,
    if (word s K pos0 fun _ => false).testBit t then GF128.of P[i][t]! else 1 + GF128.of P[i][t]!

/-- **The weights at the chosen words** are the chosen coordinates' eq-vectors, scaled by the others' factors. -/
theorem weights_word {nbl pos0 : ℕ} (hK : pos0 * 2 ^ s + 2 ^ s ≤ 2 ^ K) (hsK : s ≤ K) (hD : 10 ≤ s + nbl)
    {P : Array (List F128)} (hP : ∀ i : Fin P.size, P[i].length = K + nbl) (y : Fin 10 → Bool) :
    weights P (word s K pos0 y) =
      gammaOf (s := s) (K := K) nbl pos0 P * eqProd (fun u i => GF128.of P[i][pos s K u]!) y := by
  funext i
  rw [Pi.mul_apply, eqProd, Finset.prod_apply]
  show GF128.of (eqWord P[i] (word s K pos0 y)) = _
  rw [eqWord_word hK hsK hD (hP i)]
  refine congrArg (gammaOf (s := s) (K := K) nbl pos0 P i * ·) (Finset.prod_congr rfl fun u _ => ?_)
  by_cases hy : y u <;> simp only [hy, ↓reduceIte, Pi.add_apply, Pi.one_apply, Bool.false_eq_true]

/-- **The rank check passes** where the others' factors are nonzero and the chosen coordinates have full rank, over
any words that hold the chosen ones. -/
theorem maskRankOk_of_chosen {nbl pos0 : ℕ} (hK : pos0 * 2 ^ s + 2 ^ s ≤ 2 ^ K) (hsK : s ≤ K)
    (hD : 10 ≤ s + nbl) {P : Array (List F128)} (hP : ∀ i : Fin P.size, P[i].length = K + nbl) {words : List ℕ}
    (hw : ∀ y, word s K pos0 y ∈ words) (hγ : ∀ i, gammaOf (s := s) (K := K) nbl pos0 P i ≠ 0)
    (hQ : RankFull fun i u => GF128.of P[i][pos s K u]!) : maskRankOk P words = true := by
  rw [maskRankOk_iff, eq_top_iff, ← (rankFull_scale_iff _ hγ _).2 hQ]
  refine AddSubgroup.closure_mono ?_
  rintro _ ⟨y, rfl⟩
  obtain ⟨j, hj, hjy⟩ := List.mem_iff_getElem.1 (hw y)
  refine ⟨⟨j, hj⟩, ?_⟩
  simp only [Fin.getElem_fin, hjy]
  rw [weights_word hK hsK hD hP]
  simp only [Fin.getElem_fin]

end Word

/-! ## The deployed bound at the verifier's coins -/

section Bridge

/-- Rep `i / 2`'s first coin. -/
def repStart (n₀ n₁ : ℕ) (i : Fin 4) : ℕ := if i.val < 2 then n₀ else n₁

/-- **Where coordinate `t` of point `i` is read**: its coin's round and element. -/
def loc (m kLog n₀ n₁ : ℕ) (i : Fin 4) (t : ℕ) : ℕ × ℕ :=
  if i.val % 2 = 0 then (repStart n₀ n₁ i + abRound m kLog t, 0) else (repStart n₀ n₁ i + 1, t - 6)

/-- **Coordinate `t` of point `i`** at the coins `r`: a `c` claim's pinned coordinate, else the coin at `loc`. -/
def ptAt {k : ℕ} (r : Fin k → Coin) (m kLog n₀ n₁ : ℕ) (i : Fin 4) (t : ℕ) : F128 :=
  if i.val % 2 = 1 ∧ t < 6 then Flock.pinned[t + 1]!
  else (Claims.coinSeq r (loc m kLog n₀ n₁ i t).1).at (loc m kLog n₀ n₁ i t).2

/-- **The coin `DeployedRankBound`'s coordinate `(i, u)` reads**: the chosen coordinate's (`loc`), or for a pinned one
a coin no point reads (the rep's round `2`), so that the map is injective. -/
def tauN (m kLog s K n₀ n₁ : ℕ) (i : Fin 4) (u : Fin 10) : ℕ × ℕ :=
  if i.val % 2 = 1 ∧ u.val < 6 then (repStart n₀ n₁ i + 2, u.val) else loc m kLog n₀ n₁ i (pos s K u)

/-- The four points rep `1`'s check reads. -/
def fourPts (st : Flock.Stmt) {k : ℕ} (r : Fin k → Coin) (n₀ n₁ : ℕ) : Array (List F128) :=
  #[(claimsAt st r n₀)[0]!, (claimsAt st r n₀)[1]!, (claimsAt st r n₁)[0]!, (claimsAt st r n₁)[1]!]

theorem fourPts_size (st : Flock.Stmt) {k : ℕ} (r : Fin k → Coin) (n₀ n₁ : ℕ) : (fourPts st r n₀ n₁).size = 4 :=
  rfl

theorem maskRankAt_iff (st : Flock.Stmt) {k : ℕ} (r : Fin k → Coin) (n₀ n₁ : ℕ) :
    MaskRankAt st r n₀ n₁ ↔ maskRankOk (fourPts st r n₀ n₁) (maskWordsOf st) = true := by
  unfold MaskRankAt
  rw [append_claims]
  rfl

theorem fourPts_spec (st : Flock.Stmt) {k : ℕ} (r : Fin k → Coin) (n₀ n₁ : ℕ) (hk : 7 ≤ st.c.kLog)
    (hm : st.c.kLog ≤ st.m) (h13 : 13 ≤ st.m) (i : ℕ) (hi : i < 4) :
    ((fourPts st r n₀ n₁)[i]'hi).length = st.m - 7 ∧
      ∀ t < st.m - 7, ((fourPts st r n₀ n₁)[i]'hi)[t]! = ptAt r st.m st.c.kLog n₀ n₁ ⟨i, hi⟩ t := by
  interval_cases i
  · obtain ⟨h1, h2⟩ := claimPoints_ab (fun j e => (Claims.coinSeq r (n₀ + j)).at e) hk hm
    exact ⟨h1, fun t ht => (h2 t ht).trans (by simp [ptAt, loc, repStart])⟩
  · obtain ⟨h1, h2⟩ := claimPoints_c (kLog := st.c.kLog) (fun j e => (Claims.coinSeq r (n₀ + j)).at e) h13
    refine ⟨h1, fun t ht => (h2 t ht).trans ?_⟩
    simp only [ptAt, loc, repStart]
    split <;> simp_all
  · obtain ⟨h1, h2⟩ := claimPoints_ab (fun j e => (Claims.coinSeq r (n₁ + j)).at e) hk hm
    exact ⟨h1, fun t ht => (h2 t ht).trans (by simp [ptAt, loc, repStart])⟩
  · obtain ⟨h1, h2⟩ := claimPoints_c (kLog := st.c.kLog) (fun j e => (Claims.coinSeq r (n₁ + j)).at e) h13
    refine ⟨h1, fun t ht => (h2 t ht).trans ?_⟩
    simp only [ptAt, loc, repStart]
    split <;> simp_all

theorem abRound_le {m kLog t : ℕ} (hk : 7 ≤ kLog) (ht : t < m - 7) : abRound m kLog t ≤ m + kLog - 9 := by
  unfold abRound
  split <;> omega

theorem abRound_ge {m kLog t : ℕ} (ht : t < m - 7) : 4 + (kLog - 7) ≤ abRound m kLog t := by
  unfold abRound
  split <;> omega

theorem abRound_inj {m kLog t t' : ℕ} (hk : 7 ≤ kLog) (ht : t < m - 7) (ht' : t' < m - 7)
    (h : abRound m kLog t = abRound m kLog t') : t = t' := by
  unfold abRound at h
  split_ifs at h <;> omega

/-- **Every coordinate's coin is inside the verifier's** where rep `1`'s coins are. -/
theorem loc_lt {m kLog n₀ n₁ k : ℕ} (hk7 : 7 ≤ kLog) (hm : m ≤ 35) (hn : n₀ ≤ n₁)
    (hk : n₁ + (m + kLog) ≤ k) (i : Fin 4) {t : ℕ} (ht : t < m - 7) :
    (loc m kLog n₀ n₁ i t).1 < k ∧ (loc m kLog n₀ n₁ i t).2 < 65536 := by
  have := abRound_le hk7 ht
  unfold loc repStart
  split_ifs <;> simp only <;> omega

theorem tauN_lt {m kLog s K n₀ n₁ k : ℕ} (hk7 : 7 ≤ kLog) (hm : m ≤ 35) (hn : n₀ ≤ n₁)
    (hk : n₁ + (m + kLog) ≤ k) (hpos : ∀ u, pos s K u < m - 7) (i : Fin 4) (u : Fin 10) :
    (tauN m kLog s K n₀ n₁ i u).1 < k ∧ (tauN m kLog s K n₀ n₁ i u).2 < 65536 := by
  unfold tauN
  split
  · have := u.2
    unfold repStart
    split <;> simp only <;> omega
  · exact loc_lt hk7 hm hn hk i (hpos u)

/-- **The coin map is injective**: the reps' rounds are apart, a rep's `ab` claim reads rounds from `k_log − 3` on and
its `c` claim round `1` (the pinned coordinates round `2`), and within each the chosen coordinates are apart. -/
theorem tauN_inj {m kLog s K n₀ n₁ : ℕ} (hk13 : 13 ≤ kLog) (hn : n₀ + (m + kLog) ≤ n₁) (h6 : 6 ≤ s)
    (hsK : s ≤ K) (hK6 : 6 ≤ K) (hpos : ∀ u, pos s K u < m - 7) {i i' : Fin 4} {u u' : Fin 10}
    (h : tauN m kLog s K n₀ n₁ i u = tauN m kLog s K n₀ n₁ i' u') : i = i' ∧ u = u' := by
  have hu := hpos u
  have hu' := hpos u'
  have hp6 : 6 ≤ u.val → 6 ≤ pos s K u := pos_ge hK6 u
  have hp6' : 6 ≤ u'.val → 6 ≤ pos s K u' := pos_ge hK6 u'
  have hps : u.val < 6 → pos s K u = u.val := pos_small h6 u
  have hps' : u'.val < 6 → pos s K u' = u'.val := pos_small h6 u'
  have hr := abRound_ge (kLog := kLog) hu
  have hr' := abRound_ge (kLog := kLog) hu'
  have hl := abRound_le (kLog := kLog) (by omega) hu
  have hl' := abRound_le (kLog := kLog) (by omega) hu'
  have habi : abRound m kLog (pos s K u) = abRound m kLog (pos s K u') → pos s K u = pos s K u' :=
    abRound_inj (by omega) hu hu'
  have hu10 := u.2
  have hu10' := u'.2
  obtain ⟨i, hi⟩ := i
  obtain ⟨i', hi'⟩ := i'
  simp only [tauN, loc, repStart] at h
  have key : i = i' ∧ pos s K u = pos s K u' := by
    split_ifs at h <;> simp only [Prod.mk.injEq] at h <;> omega
  obtain ⟨rfl, hpu⟩ := key
  exact ⟨rfl, pos_injective hsK hpu⟩

/-- **A coordinate's factor is nonzero** off the one coin value that zeroes it. -/
theorem factor_ne_zero {b : Bool} {a : F128} (h : a ≠ if b then 0 else 1) :
    (if b then GF128.of a else 1 + GF128.of a) ≠ 0 := by
  cases b
  · simp only [Bool.false_eq_true, ↓reduceIte] at h ⊢
    intro h0
    have h2 : (2 : GF128) = 0 := CharTwo.two_eq_zero
    have : GF128.of a = 1 := by linear_combination h0 - h2
    exact h (this.trans GF128.one_eq)
  · simp only [↓reduceIte] at h ⊢
    intro h0
    exact h (h0.trans GF128.zero_eq)

/-- The coins at `f`'s positions inside the verifier's. -/
def tauFin {k : ℕ} (f : Fin 4 → Fin 10 → ℕ × ℕ) (hf : ∀ i u, (f i u).1 < k ∧ (f i u).2 < 65536) :
    Fin 4 × Fin 10 → Fin k × Fin 65536 := fun iu =>
  (⟨(f iu.1 iu.2).1, (hf iu.1 iu.2).1⟩, ⟨(f iu.1 iu.2).2, (hf iu.1 iu.2).2⟩)

theorem tauFin_injective {k : ℕ} (f : Fin 4 → Fin 10 → ℕ × ℕ) (hf : ∀ i u, (f i u).1 < k ∧ (f i u).2 < 65536)
    (hinj : ∀ i u i' u', f i u = f i' u' → i = i' ∧ u = u') : Function.Injective (tauFin f hf) := by
  rintro ⟨i, u⟩ ⟨i', u'⟩ h
  simp only [tauFin, Prod.mk.injEq, Fin.mk.injEq] at h
  obtain ⟨rfl, rfl⟩ := hinj i u i' u' (Prod.ext h.1 h.2)
  rfl

theorem coin_tauFin {k : ℕ} (r : Fin k → Coin) (f : Fin 4 → Fin 10 → ℕ × ℕ)
    (hf : ∀ i u, (f i u).1 < k ∧ (f i u).2 < 65536) (i : Fin 4) (u : Fin 10) :
    r (tauFin f hf (i, u)).1 (tauFin f hf (i, u)).2 = (Claims.coinSeq r (f i u).1).at (f i u).2 := by
  simp [tauFin, Claims.coinSeq, Coin.at, (hf i u).1, (hf i u).2]

open Classical in
theorem pr_rankBound {k D κ : ℕ} {fix : Fin k → Fin D → Option GF128} (h : RankBound k D fix κ) :
    Claims.Pr (fun R : Fin k → Fin D → GF128 => ¬ RankFull fun i t => (fix i t).getD (R i t)) ≤ 1 / 2 ^ κ := by
  unfold Claims.Pr
  unfold RankBound at h
  have hc : (0 : ℝ) < Fintype.card (Fin k → Fin D → GF128) := by exact_mod_cast Fintype.card_pos
  rw [div_le_div_iff₀ hc (by positivity), one_mul]
  have : ((Finset.univ.filter fun R : Fin k → Fin D → GF128 =>
      ¬ RankFull fun i t => (fix i t).getD (R i t)).card : ℝ) * 2 ^ κ ≤ Fintype.card (Fin k → Fin D → GF128) := by
    exact_mod_cast h
  convert this using 4
  congr 1

theorem kind_mask {kd : Flock.Kind} (h : (kd == Flock.Kind.mask) = true) : kd = Flock.Kind.mask := by
  cases kd
  · cases h
  · cases h
  · rfl

/-- **The deployed bound gives the rank bound at a table whose slot holds `2^13` or more bits**: its words hold ten
coordinates free (the slot's low `min s 10`, then the block index's), the four points' other `m − 17` coordinates
each zero their `eq` factor at one coin value (`4·(m − 17)·2^-128`), and at the ten the points are
`DeployedRankBound`'s, the `c` claims' first six pinned and every other one its own coin (`2^-122`). -/
theorem rankBoundAt_of_deployed (hD : DeployedRankBound) {st : Flock.Stmt} (hl : LayoutOk st)
    {rg : Flock.Range} (hrg : st.c.ranges.toList.find? (fun r => r.kind == Flock.Kind.mask) = some rg)
    (h13 : 13 ≤ rg.slotLog) : RankBoundAt st := by
  obtain ⟨hsh, hm, hmask⟩ := hl
  have hkb := List.find?_some hrg
  obtain ⟨h7, hin⟩ := hmask rg (List.mem_of_find?_eq_some hrg) (kind_mask hkb)
  have hm25 : 25 ≤ st.m := Readable.m25_of_shapeOk hsh
  have hm35 : st.m ≤ 35 := by
    unfold Flock.Zk.shapeOk at hsh
    obtain ⟨_, hq, -⟩ := FlockSoundness.except_bind_ok hsh
    exact (FlockDeploy.KS.queries0_range hq).2
  have hsl : rg.slotLog ≤ st.c.kLog := by
    by_contra h
    have : 2 ^ st.c.kLog < 2 ^ rg.slotLog := Nat.pow_lt_pow_right (by norm_num) (by omega)
    omega
  have e1 : 2 ^ rg.slotLog = 2 ^ (rg.slotLog - 7) * 128 := by
    rw [show (128 : ℕ) = 2 ^ 7 by norm_num, ← pow_add, Nat.sub_add_cancel h7]
  have e2 : 2 ^ st.c.kLog = 2 ^ (st.c.kLog - 7) * 128 := by
    rw [show (128 : ℕ) = 2 ^ 7 by norm_num, ← pow_add, Nat.sub_add_cancel (by omega)]
  obtain ⟨s, hs⟩ : ∃ s, s = rg.slotLog - 7 := ⟨_, rfl⟩
  obtain ⟨K, hKd⟩ : ∃ K, K = st.c.kLog - 7 := ⟨_, rfl⟩
  rw [← hs] at e1
  rw [← hKd] at e2
  have hsK : s ≤ K := by omega
  have h6 : 6 ≤ s := by omega
  have hK : rg.pos0 * 2 ^ s + 2 ^ s ≤ 2 ^ K := by
    unfold Flock.Range.start at hin
    rw [e1, e2] at hin
    have : (rg.pos0 * 2 ^ s + 2 ^ s) * 128 ≤ 2 ^ K * 128 := by linarith
    omega
  have hD10 : 10 ≤ s + st.nbl := by
    obtain ⟨D, hD10, hlen⟩ := maskWords_ge hsh
    rw [maskWordsOf_length st rg (by rwa [← Array.find?_toList]), ← pow_add] at hlen
    have := Nat.pow_right_injective (le_refl 2) hlen
    omega
  have hwords : ∀ y, word s K rg.pos0 y ∈ maskWordsOf st := by
    intro y
    have hst : rg.start / 128 = rg.pos0 * 2 ^ s := by
      unfold Flock.Range.start
      rw [e1, ← mul_assoc, Nat.mul_div_cancel _ (by norm_num)]
    unfold maskWordsOf
    rw [hrg]
    simp only [hst, ← hs, ← hKd]
    exact word_mem hD10 y
  have hMK : st.m - 7 = K + st.nbl := by omega
  have hposlt : ∀ u, pos s K u < st.m - 7 := fun u => hMK ▸ pos_lt hsK hD10 u
  intro k n₀ n₁ hn
  by_cases hk : n₁ + (st.m + st.c.kLog) ≤ k
  swap
  · refine le_trans (le_of_eq (Claims.Pr_congr (q := fun _ => False) fun r => ⟨fun h => hk h.1, False.elim⟩)) ?_
    rw [Claims.Pr_false]
    positivity
  have hτb := tauN_lt (m := st.m) (kLog := st.c.kLog) (s := s) (K := K) (n₀ := n₀) (n₁ := n₁) (by omega) hm35
    (by omega) hk hposlt
  have hτ : Function.Injective (tauFin _ hτb) :=
    tauFin_injective _ hτb fun _ _ _ _ h => tauN_inj (by omega) hn h6 hsK (by omega) hposlt h
  -- the others' bad values
  let fb : ℕ → Bool := fun t => (word s K rg.pos0 fun _ => false).testBit t
  let L : List (ℕ × ℕ × F128) := ((Finset.univ : Finset (Fin 4)) ×ˢ others (s := s) (K := K) st.nbl).toList.map
    fun it => ((loc st.m st.c.kLog n₀ n₁ it.1 it.2).1, (loc st.m st.c.kLog n₀ n₁ it.1 it.2).2,
      if fb it.2 then 0 else 1)
  have hLlen : L.length = 4 * (st.m - 17) := by
    simp only [L, List.length_map, Finset.length_toList, Finset.card_product, Finset.card_univ, Fintype.card_fin,
      others_card hsK hD10]
    omega
  have hothers : ∀ t ∈ others (s := s) (K := K) st.nbl, t < st.m - 7 ∧ 6 ≤ t := by
    intro t ht
    refine ⟨hMK ▸ (mem_others ht).1, ?_⟩
    by_contra h
    exact (mem_others ht).2 ⟨t, by omega⟩ (pos_small h6 _ (by simp; omega))
  have hLel : ∀ q ∈ L, q.2.1 < 65536 := by
    intro q hq
    obtain ⟨⟨i, t⟩, hit, rfl⟩ := List.mem_map.1 hq
    obtain ⟨-, ht⟩ := Finset.mem_product.1 (Finset.mem_toList.1 hit)
    exact (loc_lt (by omega) hm35 (by omega) hk i (hothers t ht).1).2
  -- the event's cover
  have hcover : ∀ r : Fin k → Coin, n₁ + (st.m + st.c.kLog) ≤ k ∧ ¬ MaskRankAt st r n₀ n₁ →
      (∃ q ∈ L, q.1 < k ∧ (Claims.coinSeq r q.1).at q.2.1 = q.2.2) ∨
        ¬ RankFull fun i u => (deployedFix i u).getD (r (tauFin _ hτb (i, u)).1 (tauFin _ hτb (i, u)).2) := by
    rintro r ⟨-, hbad⟩
    by_contra hc
    push Not at hc
    obtain ⟨hgood, hfull⟩ := hc
    apply hbad
    have spec := fourPts_spec st r n₀ n₁ (by omega) (by omega) (by omega)
    rw [maskRankAt_iff]
    refine maskRankOk_of_chosen (nbl := st.nbl) hK hsK hD10 (fun i => ?_) hwords (fun i => ?_) ?_
    · rw [← hMK]
      exact (spec i.1 i.2).1
    · unfold gammaOf
      rw [Finset.prod_ne_zero_iff]
      intro t ht
      obtain ⟨htm, h6t⟩ := hothers t ht
      rw [Fin.getElem_fin, (spec i.1 i.2).2 t htm]
      have hpt : ptAt r st.m st.c.kLog n₀ n₁ ⟨i.1, i.2⟩ t =
          (Claims.coinSeq r (loc st.m st.c.kLog n₀ n₁ ⟨i.1, i.2⟩ t).1).at
            (loc st.m st.c.kLog n₀ n₁ ⟨i.1, i.2⟩ t).2 := by
        unfold ptAt
        simp only [show ¬ t < 6 by omega, and_false, ↓reduceIte]
      rw [hpt]
      refine factor_ne_zero fun he => hgood _ (List.mem_map.2 ⟨(⟨i.1, i.2⟩, t), Finset.mem_toList.2
        (Finset.mem_product.2 ⟨Finset.mem_univ _, ht⟩), rfl⟩) ?_ he
      exact (loc_lt (by omega) hm35 (by omega) hk _ htm).1
    · have h2 := rankFull_sub (finCongr (fourPts_size st r n₀ n₁)).toEmbedding _ hfull
      refine cast (congrArg RankFull ?_) h2
      funext i u
      obtain ⟨i, hi⟩ := i
      have hi4 : i < 4 := fourPts_size st r n₀ n₁ ▸ hi
      show (deployedFix ⟨i, hi4⟩ u).getD (r (tauFin _ hτb (⟨i, hi4⟩, u)).1 (tauFin _ hτb (⟨i, hi4⟩, u)).2) =
        GF128.of ((fourPts st r n₀ n₁)[i]'hi4)[pos s K u]!
      rw [coin_tauFin r (tauN st.m st.c.kLog s K n₀ n₁) hτb, (spec i hi4).2 _ (hposlt u)]
      unfold deployedFix ptAt tauN
      by_cases hp : i % 2 = 1 ∧ u.val < 6
      · have hpu := pos_small (K := K) h6 u hp.2
        simp only [hp, hpu, and_self, ↓reduceIte]
        rfl
      · have hp' : ¬ (i % 2 = 1 ∧ pos s K u < 6) := fun h' =>
          hp ⟨h'.1, by by_contra h''; have := pos_ge (s := s) (K := K) (by omega) u (by omega); omega⟩
        simp only [hp, hp', ↓reduceIte]
        rfl
  -- the bound
  have hRB := Pr_coins_inj (tauFin _ hτb) hτ (fun R => ¬ RankFull fun i u => (deployedFix i u).getD (R i u))
  have hGB := pr_list_le
    (fun (q : ℕ × ℕ × F128) (r : Fin k → Coin) => q.1 < k ∧ (Claims.coinSeq r q.1).at q.2.1 = q.2.2)
    L (1 / 2 ^ 128) fun q hq => pr_slot_eq_le q.1 (hLel q hq) q.2.2
  calc Claims.Pr (fun r : Fin k → Coin => n₁ + (st.m + st.c.kLog) ≤ k ∧ ¬ MaskRankAt st r n₀ n₁)
      ≤ Claims.Pr (fun r : Fin k → Coin => (∃ q ∈ L, q.1 < k ∧ (Claims.coinSeq r q.1).at q.2.1 = q.2.2) ∨
          ¬ RankFull fun i u => (deployedFix i u).getD (r (tauFin _ hτb (i, u)).1 (tauFin _ hτb (i, u)).2)) :=
        Claims.Pr_mono hcover
    _ ≤ _ := Pr_or_le _ _
    _ ≤ L.length * (1 / 2 ^ 128) + 1 / 2 ^ 122 := add_le_add hGB (hRB.le.trans (pr_rankBound hD))
    _ ≤ 4 * st.m / 2 ^ 128 := by
      rw [hLlen, Nat.cast_mul, Nat.cast_sub (by omega)]
      have : (1 : ℝ) / 2 ^ 122 = 64 / 2 ^ 128 := by norm_num
      rw [this]
      have hp : (0 : ℝ) < 2 ^ 128 := by positivity
      rw [mul_one_div, ← add_div, div_le_div_iff_of_pos_right hp]
      push_cast
      linarith

/-- **The rank bound at a table whose slot holds fewer than `2^13` bits** (`slot_log < 13`, so `s < 6` of the ten
coordinates are the slot's and the `c` claims' pinned coordinates `s..5` sit off them): open. The deployed tables have
`slot_log = 14`; the verifier's refusing `slot_log < 13` at `Zk.shapeOk` would discharge it. -/
def SmallSlotBound : Prop :=
  ∀ (st : Flock.Stmt) (rg : Flock.Range), LayoutOk st →
    st.c.ranges.toList.find? (fun r => r.kind == Flock.Kind.mask) = some rg → rg.slotLog < 13 → RankBoundAt st

/-- **The rank bound at the verifier's coins** from the deployed bound and the small slots': a table the setup admits
has a mask slot (`Zk.shapeOk`), of `2^13` bits or more (`rankBoundAt_of_deployed`) or fewer (`SmallSlotBound`). -/
theorem maskRankBound_of (hD : DeployedRankBound) (hS : SmallSlotBound) : MaskRankBound := by
  intro st hl
  have hsh := hl.1
  unfold Flock.Zk.shapeOk at hsh
  obtain ⟨_, -, h⟩ := FlockSoundness.except_bind_ok hsh
  cases hr : st.c.ranges.find? (·.kind == Flock.Kind.mask) with
  | none => simp [hr] at h
  | some rg =>
    rw [← Array.find?_toList] at hr
    by_cases h13 : 13 ≤ rg.slotLog
    · exact rankBoundAt_of_deployed hD hl hr h13
    · exact hS st rg hl hr (by omega)

end Bridge

end FlockComplete.MaskRank
