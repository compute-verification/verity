import Proofs.Flock.Soundness.LigeritoPadQuery

/-!
# Ligerito with the padded level 0 is sound

`ligeritoPad_sound`: from a claim `Σ_y f(y)·b(y) = T` false for every family of rep `r`'s list,
`ligeritoPad` accepts with probability at most `padLigeritoError`:
* level 0's out-of-domain claim's batching coin `β₀` (`L₀'/|F|`), as in `ligerito_sound`;
* the lane rounds (`lane_rounds_pad`): the sumcheck and the folds' MCA events on the padded code;
* the extra lanes (`extras_pad`): one MCA line and one coin on the restarted claim per extra lane;
* `phaseA0Pad` into level 1, and `value_recursiveRounds_le` from there on, unchanged.
-/

namespace FlockSoundness

open Game Model Accounting
open CodingTheory CoreDefinitions
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

theorem padMcaA_nonneg (l : Level) (t : ℕ) (hr : 0 ≤ l.padRadius t)
    (hk : 1 < 2 ^ l.logCols + t) : 0 ≤ l.padMcaA t := by
  unfold Level.padMcaA
  have hn : (0 : ℝ) < 2 ^ l.logLen := by positivity
  have hρ : (0 : ℝ) < ((2 ^ l.logCols + t - 1 : ℕ) : ℝ) / 2 ^ l.logLen :=
    div_pos (by exact_mod_cast (by omega : 0 < 2 ^ l.logCols + t - 1)) hn
  set ρ : ℝ := ((2 ^ l.logCols + t - 1 : ℕ) : ℝ) / 2 ^ l.logLen
  set mm : ℝ := max (⌈ρ ^ ((1 : ℝ) / 2) / (1 - ρ ^ ((1 : ℝ) / 2) - l.padRadius t)⌉ : ℝ) 3
  have hmm : 3 ≤ mm := le_max_right _ _
  have h1 : 0 < ρ ^ ((3 : ℝ) / 2) := Real.rpow_pos_of_pos hρ _
  have h2 : 0 < ρ ^ ((1 : ℝ) / 2) := Real.rpow_pos_of_pos hρ _
  have hm : 0 < mm + 1 / 2 := by linarith
  have hnum : 0 ≤ 2 * (mm + 1 / 2) ^ 5 + 3 * (mm + 1 / 2) * l.padRadius t * ρ :=
    add_nonneg (by positivity) (mul_nonneg (mul_nonneg (mul_nonneg (by norm_num) hm.le) hr) hρ.le)
  exact add_nonneg (mul_nonneg (div_nonneg hnum (by positivity)) hn.le) (div_nonneg hm.le h2.le)

/-! ## The lane rounds -/

/-- **The padded level 0's lane rounds.** From a level-0 claim false for every family of the list
against `W0`, after the lane rounds the claim is false for every family's folded witness lanes, and
no round was an MCA event, except with probability `(2^k₀ − 1)·a'/|K| + 2k₀·|S|/|K|`. -/
theorem lane_rounds_pad (A : Arith F K) (hA : A.Correct)
    (n : ℕ) (b : ℕ → F) (l : Level) (t : ℕ) (hpad : PadOK l t) (c₀ : F) (k0 e r : ℕ)
    (hk0 : 1 ≤ k0) (hn : n = k0 + l.logCols) (C : Oracle F) (z₀ : List F) (hz : z₀.length = n)
    (β₀ : F) (s₀ : K) (q₀ : Quad K) (hq₀ : q₀.eval 0 + q₀.eval 1 = s₀)
    (hdoom : ∀ M ∈ padList A l t k0 e r C,
      s₀ ≠ ∑ y ∈ Finset.range (2 ^ n), A.embed (flat (padWit c₀ M) y) * W0 A b z₀ β₀ y) :
    value (fun out : List K × K => ¬ (out.1.length = k0 ∧
        MCAGood (codePad A hA l t (by have := hpad.hd; omega)) (l.padRadius t) k0 out.1
          (rowsK A k0 l.logLen C) ∧
        ∀ M ∈ padList A l t k0 e r C, out.2 ≠ ∑ x ∈ Finset.range (2 ^ l.logCols),
          foldMsg A out.1 (padWit c₀ M) x * resAt A n b out.1 [] [⟨z₀, β₀, none⟩] [] l.logCols x))
      (foldRoundsNL A k0 s₀ q₀) ≤
      ((2 ^ k0 - 1 : ℕ) : ℝ≥0∞) * ENNReal.ofReal (l.padMcaA t / qK) +
        ((2 * k0 * (padList A l t k0 e r C).card : ℕ) : ℝ≥0∞) / Fintype.card K := by
  classical
  haveI : CharP K 2 := hA.charTwoK
  have hd64 : l.logLen ≤ 64 := by have := hpad.hd; omega
  refine (value_foldRoundsNL_le A k0 _ s₀ q₀ hk0).trans ?_
  set S := padList A l t k0 e r C
  set code := codePad A hA l t hd64
  set u : (Fin (2 ^ k0 + e) → Fin (2 ^ l.logCols + t) → F) → ℕ → K :=
    fun M y => A.embed (flat (padWit c₀ M) y)
  set v := W0 A b z₀ β₀
  have himp : ∀ out : List K × K × Quad K, ¬ (out.1.length = k0 ∧
      MCAGood code (l.padRadius t) k0 out.1 (rowsK A k0 l.logLen C) ∧
      ∀ M ∈ S, out.2.1 ≠ ∑ x ∈ Finset.range (2 ^ l.logCols),
        foldMsg A out.1 (padWit c₀ M) x * resAt A n b out.1 [] [⟨z₀, β₀, none⟩] [] l.logCols x) →
      out.1.length ≠ k0 ∨ (¬ MCAGood code (l.padRadius t) k0 out.1 (rowsK A k0 l.logLen C) ∨
        ∃ M ∈ S, out.2.1 = ∑ y ∈ Finset.range (2 ^ l.logCols),
          foldLoL out.1 (u M) y * foldLoL out.1 v y) := by
    intro out hno
    by_contra hall
    simp only [not_or, not_not, not_exists, not_and] at hall
    obtain ⟨hlen, hmca, hdoom'⟩ := hall
    refine hno ⟨hlen, hmca, fun M hM heq => hdoom' M hM ?_⟩
    rw [heq]
    refine Finset.sum_congr rfl fun y hy => ?_
    rw [foldLoL_W0 A n b z₀ hz β₀ out.1 l.logCols (by omega) y (Finset.mem_range.1 hy)]
    rfl
  have hmcaε : ∀ U : Fin 2 → Fin (2 ^ l.logLen) → K,
      prCoin (fun γ : K => IsMCA (AffineLineGenerator K) code γ U (l.padRadius t)) ≤
        ENNReal.ofReal (l.padMcaA t / qK) :=
    prCoin_pad_mca_le A hA l t hd64 hpad.hk1 hpad.hr
  have hsumε := value_foldRounds_sum_le A hA S u v k0 l.logCols s₀ q₀ hq₀
    (fun M hM => by rw [← hn]; exact hdoom M hM)
  calc value _ (foldRounds A k0 s₀ q₀)
      ≤ value (fun out : List K × K × Quad K => out.1.length ≠ k0 ∨
          (¬ MCAGood code (l.padRadius t) k0 out.1 (rowsK A k0 l.logLen C) ∨
            ∃ M ∈ S, out.2.1 = ∑ y ∈ Finset.range (2 ^ l.logCols),
              foldLoL out.1 (u M) y * foldLoL out.1 v y)) (foldRounds A k0 s₀ q₀) :=
        value_mono himp _
    _ ≤ value (fun out : List K × K × Quad K => out.1.length ≠ k0) _ +
        (value (fun out : List K × K × Quad K =>
            ¬ MCAGood code (l.padRadius t) k0 out.1 (rowsK A k0 l.logLen C)) _ +
          value (fun out : List K × K × Quad K => ∃ M ∈ S, out.2.1 =
            ∑ y ∈ Finset.range (2 ^ l.logCols), foldLoL out.1 (u M) y * foldLoL out.1 v y) _) :=
        (value_or_le _ _ _).trans (add_le_add le_rfl (value_or_le _ _ _))
    _ ≤ 0 + (((2 ^ k0 - 1 : ℕ) : ℝ≥0∞) * ENNReal.ofReal (l.padMcaA t / qK) +
          ((2 * k0 * S.card : ℕ) : ℝ≥0∞) / Fintype.card K) :=
        add_le_add (value_foldRounds_length A hA _ _ _).le
          (add_le_add (value_foldRounds_mca_le A hA code (l.padRadius t) _ hmcaε k0 _ _ _) hsumε)
    _ = _ := zero_add _

/-! ## The extra lanes -/

/-- A family's next-level message against a weight: its folded witness lanes', plus each extra
lane's weighted by its coin. -/
theorem sum_padFold (A : Arith F K) {k0 e cols t : ℕ} (c₀ : F) (ch : List K) (ρ : Fin e → K)
    (M : Fin (2 ^ k0 + e) → Fin (2 ^ cols + t) → F) (res : ℕ → K) :
    ∑ x ∈ Finset.range (2 ^ cols), padFold A c₀ ch ρ M x * res x =
      ∑ x ∈ Finset.range (2 ^ cols), foldMsg A ch (padWit c₀ M) x * res x +
        ∑ i, ρ i * ∑ x ∈ Finset.range (2 ^ cols),
          (if h : x < 2 ^ cols then A.embed (padMsgAt c₀ (M (Fin.natAdd (2 ^ k0) i)) ⟨x, h⟩)
            else 0) * res x := by
  simp only [padFold, add_mul, Finset.sum_add_distrib, Finset.sum_mul, Finset.mul_sum]
  congr 1
  rw [Finset.sum_comm]
  exact Finset.sum_congr rfl fun _ _ => Finset.sum_congr rfl fun _ _ => by ring

/-- **The extra lanes.** After the lane rounds, the `e` extra coins keep the restarted claim false
for every family and are no MCA event, except with probability `e·a'/|K| + e·|S|/|K|`. -/
theorem extras_pad (A : Arith F K) (hA : A.Correct)
    (l : Level) (t : ℕ) (hpad : PadOK l t) (c₀ : F) (k0 e r : ℕ) (C : Oracle F) (ch : List K)
    (s : K) (ev : Fin e → K) (res : ℕ → K)
    (hdoom : ∀ M ∈ padList A l t k0 e r C,
      s ≠ ∑ x ∈ Finset.range (2 ^ l.logCols), foldMsg A ch (padWit c₀ M) x * res x) :
    value (fun ρ : Fin e → K => ¬ (ExtrasGood (codePad A hA l t (by have := hpad.hd; omega))
        (l.padRadius t) e ρ (xrowsK A k0 e r l.logLen C) (foldAllRows k0 ch (rowsK A k0 l.logLen C)) ∧
        ∀ M ∈ padList A l t k0 e r C, s + ∑ i, ρ i * ev i ≠
          ∑ x ∈ Finset.range (2 ^ l.logCols), padFold A c₀ ch ρ M x * res x))
      (drawKs A e) ≤
      (e : ℝ≥0∞) * ENNReal.ofReal (l.padMcaA t / qK) +
        ((e * (padList A l t k0 e r C).card : ℕ) : ℝ≥0∞) / Fintype.card K := by
  classical
  have hd64 : l.logLen ≤ 64 := by have := hpad.hd; omega
  set S := padList A l t k0 e r C
  set code := codePad A hA l t hd64
  set g : (Fin (2 ^ k0 + e) → Fin (2 ^ l.logCols + t) → F) → Fin e → K := fun M i =>
    ∑ x ∈ Finset.range (2 ^ l.logCols),
      (if h : x < 2 ^ l.logCols then A.embed (padMsgAt c₀ (M (Fin.natAdd (2 ^ k0) i)) ⟨x, h⟩)
        else 0) * res x
  set T0 : (Fin (2 ^ k0 + e) → Fin (2 ^ l.logCols + t) → F) → K := fun M =>
    ∑ x ∈ Finset.range (2 ^ l.logCols), foldMsg A ch (padWit c₀ M) x * res x
  refine (value_mono (q := fun ρ => ¬ ExtrasGood code (l.padRadius t) e ρ
      (xrowsK A k0 e r l.logLen C) (foldAllRows k0 ch (rowsK A k0 l.logLen C)) ∨
      ∃ M ∈ S, (s - T0 M) + ∑ i, ρ i * (ev i - g M i) = 0) (fun ρ h => ?_) _).trans
    ((value_or_le _ _ _).trans (add_le_add ?_ ?_))
  · by_contra hno
    push Not at hno
    obtain ⟨hx, hall⟩ := hno
    refine h ⟨hx, fun M hM heq => hall M hM ?_⟩
    rw [sum_padFold] at heq
    simp only [T0, g, mul_sub, Finset.sum_sub_distrib]
    linear_combination heq
  · exact value_drawKs_extras_le A hA code (l.padRadius t) _
      (prCoin_pad_mca_le A hA l t hd64 hpad.hk1 hpad.hr) e _ _
  · exact value_drawKs_affine_le A hA S e (fun M => s - T0 M) (fun M i => ev i - g M i)
      fun M hM => sub_ne_zero.2 (hdoom M hM)

/-- `phaseA0Pad` with the next level's variables named. -/
theorem phaseA0Pad' (A : Arith F K) (hA : A.Correct) (n : ℕ) (b : ℕ → F) (l l' : Level) (t : ℕ)
    (hd : l.logLen ≤ 40) (hd64 : l.logLen ≤ 64) (hcols : l.logCols ≤ l.logLen)
    (ht : t ≤ 2 ^ l.logCols) (hk : 2 ^ l.logCols + t ≤ 2 ^ l.logLen) (hf' : l'.Fits)
    (hQ : l.queries ≤ 2 ^ 64) (c₀ : F) (k0 e r : ℕ) (ch : List K) (ρ : Fin e → K)
    (ybar : Fin t → K) (st : LigState F K)
    (hpost : PostPad0 A hA n b l hd64 t c₀ k0 e r st.prev ch ρ st)
    (hdim : l.logCols + 1 = l'.lanes + l'.logCols) (dim : ℕ) (hdd : dim = l'.logCols + l'.lanes)
    (k : Oracle F → LigState F K → Game Bool) (εk : ℝ≥0∞)
    (hk' : ∀ C' st', PreFold A n b ch l' C' l.logCols st' →
      value (fun r => r = true) (k C' st') ≤ εk) :
    value (fun r => r = true) (midRoundV A l dim
      (padColumn A l.logLen l.logCols t c₀ ch ρ (extraLane k0 e r) ybar st.prev) st k) ≤
      ENNReal.ofReal (midAPad l l' t) + εk := by
  subst hdd
  exact phaseA0Pad A hA n b l l' t hd hd64 hcols ht hk hf' hQ c₀ k0 e r ch ρ ybar st hpost hdim k εk
    hk'

/-! ## Ligerito -/

/-- `ligeritoPad` on a schedule with at least two levels. -/
theorem ligeritoPad_eq (A : Arith F K) (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level) (ext : ℕ)
    (t : ℕ) (c₀ : F) (e r : ℕ) (T : F) (b : ℕ → F) (C : Oracle F) :
    ligeritoPad A ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ t c₀ e r T b C =
      Game.coin (Fin (m - 7) → F) fun z₀ => Game.send F fun y₀ => Game.coin F fun β₀ =>
        Game.send (K × K) fun msg =>
          (foldRoundsNL A k0 (A.embed T + A.embed β₀ * A.embed y₀)
              (fromMsg msg (A.embed T + A.embed β₀ * A.embed y₀))).bind fun out =>
            match out with
            | (initial, s) =>
                Game.send (Fin e → K) fun ev => (drawKs A e).bind fun ρ =>
                  Game.send (Fin t → K) fun ybar => Game.send (K × K) fun msg' =>
                    midRoundV A l₀ (m - 7 - k0 + 1)
                      (padColumn A l₀.logLen l₀.logCols t c₀ initial ρ (extraLane k0 e r) ybar C)
                      ⟨s + ∑ i, ρ i * ev i, fromMsg msg' (s + ∑ i, ρ i * ev i), m - 7 - k0 + 1, [],
                        [⟨List.ofFn z₀, β₀, none⟩], [], C, l₀⟩
                      fun C' st' => recursiveRounds A (m - 7) b initial ext (l₁ :: rest)
                        { st' with prev := C', prevLevel := l₁ } := rfl

/-- **Ligerito with the padded level 0 is sound.** For a schedule of the `fast100` shape whose
level 0 meets `PadOK`, if the claim `Σ_y f(y)·b(y) = T` is false for every family of rep `r`'s
list, `ligeritoPad` accepts with probability at most `padLigeritoError`. -/
theorem ligeritoPad_sound (A : Arith F K) (hA : A.Correct)
    (sch : Schedule)
    (hshape : 2 ≤ sch.levels.length ∧ (sch.levels.headD default).lanes = sch.k0 ∧
      sch.m - 7 = sch.k0 + (sch.levels.headD default).logCols ∧
        levelsChained sch.levels sch.ext = true)
    (t : ℕ) (hpad : PadOK sch.level₀ t) (c₀ : F) (e r : ℕ) (T : F) (b : ℕ → F) (C : Oracle F)
    (hdoom : ∀ M ∈ padList A sch.level₀ t sch.k0 e r C,
      (∑ y ∈ Finset.range (2 ^ (sch.m - 7)), flat (padWit c₀ M) y * b y) ≠ T) :
    value (fun r => r = true) (ligeritoPad A sch t c₀ e r T b C) ≤
      ENNReal.ofReal (padLigeritoError sch t e) := by
  classical
  haveI : CharP K 2 := hA.charTwoK
  obtain ⟨m, k0, levels, ext⟩ := sch
  obtain ⟨hlen, hk0, hn, hchain⟩ := hshape
  rcases levels with _ | ⟨l₀, _ | ⟨l₁, rest⟩⟩
  · simp at hlen
  · simp at hlen
  dsimp only [List.headD, Schedule.level₀] at hk0 hn hdoom hpad ⊢
  have h' := hchain
  simp only [levelsChained, Bool.and_eq_true, decide_eq_true_eq] at h'
  obtain ⟨⟨hgood₀, hdim₀⟩, hrest⟩ := h'
  obtain ⟨hl₀, hf₀, hlanes₀, hQ₀⟩ := levelGood_spec hgood₀
  obtain ⟨_, hf₁, _, _⟩ := levelGood_spec (levelsChained_head_good hrest)
  have hd64 : l₀.logLen ≤ 64 := by have := hpad.hd; omega
  have hcols : l₀.logCols ≤ l₀.logLen := by unfold Level.logLen; omega
  have hk1 : 1 ≤ k0 := by rw [← hk0]; exact hlanes₀
  set S₀ := padList A l₀ t k0 e r C
  have hS₀ : S₀.card ≤ l₀.padListBound t := by
    have := card_closeRows_le A hA l₀.logLen (2 ^ l₀.logCols + t) hpad.hd hd64 hpad.hk
      (by have := Nat.two_pow_pos l₀.logCols; omega) (s := 2 ^ k0 + e)
      (by have := Nat.two_pow_pos k0; omega)
      (padRows k0 e r l₀.logLen C)
    exact this
  -- the rest of the rounds, from level 1's `PreFold`
  set εrest : ℝ≥0∞ := ENNReal.ofReal (ligeritoError (l₁ :: rest))
  have hrec : ∀ (initial : List K) (C' : Oracle F) (st' : LigState F K),
      PreFold A (m - 7) b initial l₁ C' l₀.logCols st' →
      value (fun r => r = true) (recursiveRounds A (m - 7) b initial ext (l₁ :: rest)
        { st' with prev := C', prevLevel := l₁ }) ≤ εrest :=
    fun initial C' st' hpre' => value_recursiveRounds_le A hA (m - 7) b initial ext (l₁ :: rest)
      { st' with prev := C', prevLevel := l₁ } l₀.logCols hrest
      ⟨hpre'.sum, hpre'.ok, hpre'.dim, hpre'.doom⟩
  set εmid : ℝ≥0∞ := ENNReal.ofReal (midAPad l₀ l₁ t)
  set εlane : ℝ≥0∞ := ((2 ^ k0 - 1 : ℕ) : ℝ≥0∞) * ENNReal.ofReal (l₀.padMcaA t / qK) +
    ((2 * k0 * S₀.card : ℕ) : ℝ≥0∞) / Fintype.card K
  set εx : ℝ≥0∞ := (e : ℝ≥0∞) * ENNReal.ofReal (l₀.padMcaA t / qK) +
    ((e * S₀.card : ℕ) : ℝ≥0∞) / Fintype.card K
  rw [ligeritoPad_eq]
  refine (value_coin_le_of_forall _ _ _ (S₀.card / Fintype.card F + (εlane + (εx + (εmid + εrest))))
    fun z₀ => ?_).trans ?_
  · rw [value_send]
    refine iSup_le fun y₀ => ?_
    let s₀ : F → K := fun β₀ => A.embed T + A.embed β₀ * A.embed y₀
    refine value_coin_le _ F _ (fun β₀ => ∃ M ∈ S₀, s₀ β₀ = ∑ y ∈ Finset.range (2 ^ (m - 7)),
        A.embed (flat (padWit c₀ M) y) * W0 A b (List.ofFn z₀) β₀ y) _ _ ?_ fun β₀ hβ₀ => ?_
    · refine le_trans (prCoin_mono fun β₀ hβ => ?_) (prCoin_embed_affine_union_le A S₀
        (fun M => A.embed T - A.embed (∑ y ∈ Finset.range (2 ^ (m - 7)), flat (padWit c₀ M) y * b y))
        (fun M => A.embed y₀ - mleK A (m - 7) (flat (padWit c₀ M)) (List.ofFn z₀)))
      obtain ⟨M, hM, heq⟩ := hβ
      have htruth : ∑ y ∈ Finset.range (2 ^ (m - 7)), A.embed (flat (padWit c₀ M) y) *
          W0 A b (List.ofFn z₀) β₀ y =
          A.embed (∑ y ∈ Finset.range (2 ^ (m - 7)), flat (padWit c₀ M) y * b y) +
            A.embed β₀ * mleK A (m - 7) (flat (padWit c₀ M)) (List.ofFn z₀) := by
        simp only [W0, mleK, mul_add, Finset.sum_add_distrib, Finset.mul_sum, map_sum, map_mul]
        congr 1
        exact Finset.sum_congr rfl fun y _ => by ring
      refine ⟨M, hM, ?_, fun ⟨h0, _⟩ => hdoom M hM ?_⟩
      · have heq' : A.embed T + A.embed β₀ * A.embed y₀ = _ := heq
        linear_combination heq' + htruth
      · exact (A.embed.injective (sub_eq_zero.1 h0)).symm
    · rw [value_send]
      refine iSup_le fun msg => ?_
      have hq₀ := fromMsg_sum msg (s₀ β₀)
      push Not at hβ₀
      -- the lane rounds
      refine (value_bind_le _ _ _ (fun out : List K × K => out.1.length = k0 ∧
          MCAGood (codePad A hA l₀ t hd64) (l₀.padRadius t) k0 out.1 (rowsK A k0 l₀.logLen C) ∧
          ∀ M ∈ S₀, out.2 ≠ ∑ x ∈ Finset.range (2 ^ l₀.logCols),
            foldMsg A out.1 (padWit c₀ M) x *
              resAt A (m - 7) b out.1 [] [⟨List.ofFn z₀, β₀, none⟩] [] l₀.logCols x)
          (εx + (εmid + εrest)) fun out hpost => ?_).trans ?_
      · obtain ⟨initial, s⟩ := out
        obtain ⟨hlen', hmca, hdoom'⟩ := hpost
        show value _ (Game.send (Fin e → K) fun ev => _) ≤ _
        rw [value_send]
        refine iSup_le fun ev => ?_
        -- the extra lanes
        refine (value_bind_le _ _ _ (fun ρ : Fin e → K =>
            ExtrasGood (codePad A hA l₀ t hd64) (l₀.padRadius t) e ρ (xrowsK A k0 e r l₀.logLen C)
              (foldAllRows k0 initial (rowsK A k0 l₀.logLen C)) ∧
            ∀ M ∈ S₀, s + ∑ i, ρ i * ev i ≠ ∑ x ∈ Finset.range (2 ^ l₀.logCols),
              padFold A c₀ initial ρ M x *
                resAt A (m - 7) b initial [] [⟨List.ofFn z₀, β₀, none⟩] [] l₀.logCols x)
            (εmid + εrest) fun ρ hρ => ?_).trans ?_
        · obtain ⟨hxg, hdoomx⟩ := hρ
          rw [value_send]
          refine iSup_le fun ybar => ?_
          rw [value_send]
          refine iSup_le fun msg' => ?_
          set s' := s + ∑ i, ρ i * ev i
          have hpost0 : PostPad0 A hA (m - 7) b l₀ hd64 t c₀ k0 e r C initial ρ
              ⟨s', fromMsg msg' s', m - 7 - k0 + 1, [], [⟨List.ofFn z₀, β₀, none⟩], [], C, l₀⟩ :=
            { sum := fromMsg_sum msg' s'
              len := hlen'
              ok := ⟨fun o ho => by
                  simp only [List.mem_singleton] at ho
                  subst ho
                  exact ⟨Or.inl rfl, by simp [oodFixed, origOf, hlen', hn]⟩,
                fun c hc => by simp at hc⟩
              mca := hmca
              xgood := hxg
              doom := hdoomx }
          refine phaseA0Pad' A hA (m - 7) b l₀ l₁ t hpad.hd hd64 hcols hpad.ht hpad.hk hf₁ hQ₀ c₀ k0 e
            r initial ρ ybar
            ⟨s', fromMsg msg' s', m - 7 - k0 + 1, [], [⟨List.ofFn z₀, β₀, none⟩], [], C, l₀⟩ hpost0
            hdim₀ (m - 7 - k0 + 1) (by omega) _ εrest fun C' st' hpre' => hrec initial C' st' hpre'
        · refine add_le_add ?_ le_rfl
          refine extras_pad A hA l₀ t hpad c₀ k0 e r C initial s ev _ hdoom'
      · refine add_le_add ?_ le_rfl
        exact lane_rounds_pad A hA (m - 7) b l₀ t hpad c₀ k0 e r hk1 hn C (List.ofFn z₀)
          (by simp) β₀ (s₀ β₀) _ hq₀ fun M hM h => hβ₀ M hM h
  · -- the arithmetic
    have hq : (0 : ℝ) < qF := by unfold qF; positivity
    have hqK : (0 : ℝ) < qK := by unfold qK; positivity
    have hok := levelsChained_ok hchain
    have hr := ligeritoError_nonneg (l₁ :: rest) (fun l hl => hok l (List.mem_cons_of_mem _ hl))
    have hmca := padMcaA_nonneg l₀ t hpad.hr.le hpad.hk1
    have hm := midAPad_nonneg l₀ l₁ t
    have hL : (0 : ℝ) ≤ (l₀.padListBound t : ℝ) / qF := by positivity
    set L₀ := l₀.padListBound t
    have hmq : 0 ≤ l₀.padMcaA t / qK := div_nonneg hmca hqK.le
    have hfe : 0 ≤ l₀.padFoldError t e L₀ := by
      unfold Level.padFoldError
      have h2l : (1 : ℝ) ≤ 2 ^ l₀.lanes := one_le_pow₀ (by norm_num)
      have t1 : (0 : ℝ) ≤ l₀.lanes * (2 * L₀) / qK :=
        div_nonneg (mul_nonneg (Nat.cast_nonneg _) (mul_nonneg (by norm_num) (Nat.cast_nonneg _)))
          hqK.le
      have t2 : (0 : ℝ) ≤ (2 ^ l₀.lanes - 1) * l₀.padMcaA t / qK :=
        div_nonneg (mul_nonneg (by linarith) hmca) hqK.le
      have t3 : (0 : ℝ) ≤ e * l₀.padMcaA t / qK :=
        div_nonneg (mul_nonneg (Nat.cast_nonneg _) hmca) hqK.le
      have t4 : (0 : ℝ) ≤ e * L₀ / qK :=
        div_nonneg (mul_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _)) hqK.le
      linarith
    have hfold : εlane + εx ≤ ENNReal.ofReal (l₀.padFoldError t e L₀) := by
      have hK : Fintype.card K = 2 ^ 256 := hA.card_K
      have hKr : ((Fintype.card K : ℕ) : ℝ≥0∞) = ENNReal.ofReal qK := by
        rw [hK, show ((2 ^ 256 : ℕ) : ℝ≥0∞) = ENNReal.ofReal qK by
          unfold qK; rw [ENNReal.ofReal_pow (by norm_num)]; norm_num]
      have hdiv : ∀ N : ℕ, (N : ℝ≥0∞) / Fintype.card K = ENNReal.ofReal ((N : ℝ) / qK) := by
        intro N
        rw [hKr, ← ENNReal.ofReal_natCast, ← ENNReal.ofReal_div_of_pos hqK]
      have hmul : ∀ N : ℕ, (N : ℝ≥0∞) * ENNReal.ofReal (l₀.padMcaA t / qK) =
          ENNReal.ofReal (N * (l₀.padMcaA t / qK)) := by
        intro N
        rw [← ENNReal.ofReal_natCast, ← ENNReal.ofReal_mul (Nat.cast_nonneg N)]
      have h1 : 0 ≤ ((2 ^ k0 - 1 : ℕ) : ℝ) * (l₀.padMcaA t / qK) := mul_nonneg (Nat.cast_nonneg _) hmq
      have h2 : 0 ≤ ((2 * k0 * S₀.card : ℕ) : ℝ) / qK := div_nonneg (Nat.cast_nonneg _) hqK.le
      have h3 : 0 ≤ (e : ℝ) * (l₀.padMcaA t / qK) := mul_nonneg (Nat.cast_nonneg _) hmq
      have h4 : 0 ≤ ((e * S₀.card : ℕ) : ℝ) / qK := div_nonneg (Nat.cast_nonneg _) hqK.le
      have hεl : εlane = ENNReal.ofReal (((2 ^ k0 - 1 : ℕ) : ℝ) * (l₀.padMcaA t / qK) +
          ((2 * k0 * S₀.card : ℕ) : ℝ) / qK) := by
        simp only [εlane]; rw [hmul, hdiv, ENNReal.ofReal_add h1 h2]
      have hεx : εx = ENNReal.ofReal ((e : ℝ) * (l₀.padMcaA t / qK) +
          ((e * S₀.card : ℕ) : ℝ) / qK) := by
        simp only [εx]; rw [hmul, hdiv, ENNReal.ofReal_add h3 h4]
      rw [hεl, hεx, ← ENNReal.ofReal_add (add_nonneg h1 h2) (add_nonneg h3 h4)]
      refine ENNReal.ofReal_le_ofReal ?_
      have hcard : (S₀.card : ℝ) ≤ L₀ := by exact_mod_cast hS₀
      have e1 : ((2 ^ k0 - 1 : ℕ) : ℝ) * (l₀.padMcaA t / qK) =
          (2 ^ l₀.lanes - 1) * l₀.padMcaA t / qK := by
        rw [hk0, Nat.cast_sub Nat.one_le_two_pow]; push_cast; ring
      have e2 : ((2 * k0 * S₀.card : ℕ) : ℝ) / qK ≤ l₀.lanes * (2 * L₀) / qK := by
        rw [hk0]
        apply div_le_div_of_nonneg_right _ hqK.le
        push_cast
        nlinarith
      have e3 : (e : ℝ) * (l₀.padMcaA t / qK) = e * l₀.padMcaA t / qK := by ring
      have e4 : ((e * S₀.card : ℕ) : ℝ) / qK ≤ e * L₀ / qK := by
        apply div_le_div_of_nonneg_right _ hqK.le
        push_cast
        exact mul_le_mul_of_nonneg_left hcard (Nat.cast_nonneg _)
      unfold Level.padFoldError
      linarith
    calc (S₀.card : ℝ≥0∞) / Fintype.card F + (εlane + (εx + (εmid + εrest)))
        = (S₀.card : ℝ≥0∞) / Fintype.card F + ((εlane + εx) + (εmid + εrest)) := by ring
      _ ≤ ENNReal.ofReal ((L₀ : ℝ) / qF) +
            (ENNReal.ofReal (l₀.padFoldError t e L₀) + (εmid + εrest)) := by
          refine add_le_add ?_ (add_le_add hfold le_rfl)
          rw [ofReal_nat_div_qF', hA.card_F, Nat.cast_pow, Nat.cast_ofNat]
          gcongr
      _ = ENNReal.ofReal (padLigeritoError ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ t e) := by
          rw [show padLigeritoError ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ t e =
              (L₀ : ℝ) / qF + padMidLevelError l₀ l₁ t e + ligeritoError (l₁ :: rest) from rfl,
            padMidLevelError_eq, ENNReal.ofReal_add (add_nonneg hL (add_nonneg hfe hm)) hr,
            ENNReal.ofReal_add hL (add_nonneg hfe hm), ENNReal.ofReal_add hfe hm]
          ring

end FlockSoundness
