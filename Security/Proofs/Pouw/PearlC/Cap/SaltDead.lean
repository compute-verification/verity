import Proofs.Pouw.PearlC.FormingP
import Proofs.Pouw.PearlC.Cap.SaltDeadStep

/-!
# The salt-dead two-ends lemma

On a one-group step (`p.groups = [32]`: sm_120's `sm120E4m3K32 z` and Hopper's `hopperE4m3K32`), an element is
salt-dead (`saltDeadP`) iff the support's two ends agree (`EndsAgree`): every product `+|E′|·11/4` against every
product `−|E′|·11/4`. So RowA decides `saltDeadP` per element from two evaluations of the forming step instead of 33.

* **The structure** (`flat_structure`): on the support (`SaltDead.support`, the flat pairs) the code's value is
  `Φ(a·n)`, where `n = flatCount s s'` is the signed count (products `+` less products `−`), `Φ` is monotone and `a` has
  the sign of `β`. Every `E′` code is `±c_β` and every `F` code `±0x43`, so all 32 products have one magnitude and one
  exponent: the alignment exponent (`maxExp`) is the same at every pair, the aligned sum is `aligned(C) + n·a₀`, and the
  read (`codeW_mono`) is monotone. A non-finite accumulator makes the code constant (`pipeStep_nonfin`).
* **Monotone in the count for `β ≥ 0`, antitone for `β ≤ 0`** (`atomMixP_flat_mono`, `atomMixP_flat_anti`). Without a
  hypothesis on `β` the count is not monotone (`FlatMono`): `not_flatMono_sm120` and `not_flatMono_hopper` refute it at
  `α·x = 0`, `β = −1`, where count `−32` gives code `448` and count `32` gives `−448`.
* **(E)** `saltDeadP_iff_ends`, with no hypothesis on `α`, `x` or `β`; **(C)** `saltDeadP_card_ends`, the per-block
  count RowA's `sd` port carries; **(Z)** `codeVal_eq_iff`, the circuit's comparison of two codes' values.
-/

namespace Pouw.PearlC.Cap

open Pouw.Fp8Atom Pouw.PearlC Pouw.PearlC.FormingP Pouw.PearlC.SaltDead

/-! ## The ends and the signed count -/

/-- The support's ends: every product `+|E′|·11/4`. -/
def plusEnd : (Fin 32 → Code) × (Fin 32 → Code) := (flatLine fun _ => false, flatLineF fun _ => false)

/-- The support's ends: every product `−|E′|·11/4`. -/
def minusEnd : (Fin 32 → Code) × (Fin 32 → Code) := (flatLine fun _ => false, flatLineF fun _ => true)

/-- What RowA tests per element: the two ends' codes have one value. -/
def EndsAgree (p : Pipeline) (α y β : ℚ) : Prop :=
  codeVal ((atomMixP p).elem α y β plusEnd.1 plusEnd.2) = codeVal ((atomMixP p).elem α y β minusEnd.1 minusEnd.2)

/-- A support pair's signed count: products `+` less products `−` (`plusEnd` 32, `minusEnd` −32). -/
def flatCount (s s' : Fin 32 → Bool) : ℤ := ∑ t, if s t = s' t then 1 else -1

/-- cap-unit's (M) as first stated: on every flat pair, the code's value is monotone in the signed count, for every
`α`, `x` and `β`. False on sm_120 and Hopper (`not_flatMono_sm120`, `not_flatMono_hopper`); true for `0 ≤ β`
(`atomMixP_flat_mono`). -/
def FlatMono (p : Pipeline) : Prop :=
  ∀ (α y β : ℚ) (s₁ s₁' s₂ s₂' : Fin 32 → Bool), flatCount s₁ s₁' ≤ flatCount s₂ s₂' →
    codeVal ((atomMixP p).elem α y β (flatLine s₁) (flatLineF s₁')) ≤
      codeVal ((atomMixP p).elem α y β (flatLine s₂) (flatLineF s₂'))

theorem flatCount_plus : flatCount (fun _ => false) (fun _ => false) = 32 := by decide +kernel

theorem flatCount_minus : flatCount (fun _ => false) (fun _ => true) = -32 := by decide +kernel

theorem flatCount_le (s s' : Fin 32 → Bool) : flatCount s s' ≤ 32 := by
  unfold flatCount
  calc ∑ t, (if s t = s' t then 1 else -1 : ℤ) ≤ ∑ _t : Fin 32, (1 : ℤ) :=
        Finset.sum_le_sum fun t _ => by split_ifs <;> norm_num
    _ = 32 := by simp

theorem neg_le_flatCount (s s' : Fin 32 → Bool) : -32 ≤ flatCount s s' := by
  unfold flatCount
  calc (-32 : ℤ) = ∑ _t : Fin 32, (-1 : ℤ) := by simp
    _ ≤ ∑ t, (if s t = s' t then 1 else -1 : ℤ) := Finset.sum_le_sum fun t _ => by split_ifs <;> norm_num

/-! ## (M) as first stated is false -/

theorem not_flatMono_sm120 : ¬ FlatMono sm120E4m3 := by
  intro h
  have h1 := h 0 0 (-1) (fun _ => false) (fun _ => true) (fun _ => false) (fun _ => false)
    (by rw [flatCount_minus, flatCount_plus]; norm_num)
  have e1 : codeVal ((atomMixP sm120E4m3).elem 0 0 (-1) (flatLine fun _ => false) (flatLineF fun _ => true)) =
      448 := by decide +kernel
  have e2 : codeVal ((atomMixP sm120E4m3).elem 0 0 (-1) (flatLine fun _ => false) (flatLineF fun _ => false)) =
      -448 := by decide +kernel
  rw [e1, e2] at h1
  norm_num at h1

theorem not_flatMono_hopper : ¬ FlatMono hopperE4m3K32 := by
  intro h
  have h1 := h 0 0 (-1) (fun _ => false) (fun _ => true) (fun _ => false) (fun _ => false)
    (by rw [flatCount_minus, flatCount_plus]; norm_num)
  have e1 : codeVal ((atomMixP hopperE4m3K32).elem 0 0 (-1) (flatLine fun _ => false) (flatLineF fun _ => true)) =
      448 := by decide +kernel
  have e2 : codeVal ((atomMixP hopperE4m3K32).elem 0 0 (-1) (flatLine fun _ => false) (flatLineF fun _ => false)) =
      -448 := by decide +kernel
  rw [e1, e2] at h1
  norm_num at h1

/-! ## The products on the support -/

/-- A term with its sign set to `σ` (a zero term unchanged). -/
def sgnT (σ : Bool) (T : Term) : Term := if T.mag = 0 then T else ⟨σ, T.exp, T.mag⟩

theorem sgnT_mag (σ : Bool) (T : Term) : (sgnT σ T).mag = T.mag := by unfold sgnT; split_ifs <;> rfl

theorem sgnT_exp (σ : Bool) (T : Term) : (sgnT σ T).exp = T.exp := by unfold sgnT; split_ifs <;> rfl

theorem product_sgnT {c f c₀ f₀ : Code} (hs : c.sig = c₀.sig) (he : c.expOr1 = c₀.expOr1) (hs' : f.sig = f₀.sig)
    (he' : f.expOr1 = f₀.expOr1) : product c f = sgnT (c.sign ^^ f.sign) (product c₀ f₀) := by
  unfold product sgnT
  rw [hs, he, hs', he']
  by_cases h : (c₀.sig * f₀.sig) <<< 17 = 0 <;> simp [h, Term.zero]

/-- The two codes of a pair `(c₀, c₁)`: `c₁` is `c₀` with the sign bit set. -/
def CodePair (c₀ c₁ : Code) : Prop :=
  c₀.isNaN = false ∧ c₀.sign = false ∧ c₁.isNaN = false ∧ c₁.sign = true ∧ c₁.sig = c₀.sig ∧ c₁.expOr1 = c₀.expOr1

theorem product_pair {c₀ c₁ f₀ f₁ : Code} (hc : CodePair c₀ c₁) (hf : CodePair f₀ f₁) (σ₁ σ₂ : Bool) :
    product (if σ₁ then c₁ else c₀) (if σ₂ then f₁ else f₀) = sgnT (σ₁ ^^ σ₂) (product c₀ f₀) := by
  obtain ⟨-, hc0, -, hc1, hcs, hce⟩ := hc
  obtain ⟨-, hf0, -, hf1, hfs, hfe⟩ := hf
  have e1 : (if σ₁ then c₁ else c₀).sign = σ₁ := by cases σ₁ <;> simp [hc0, hc1]
  have e2 : (if σ₂ then f₁ else f₀).sign = σ₂ := by cases σ₂ <;> simp [hf0, hf1]
  rw [product_sgnT (c₀ := c₀) (f₀ := f₀) (by cases σ₁ <;> simp [hcs]) (by cases σ₁ <;> simp [hce])
    (by cases σ₂ <;> simp [hfs]) (by cases σ₂ <;> simp [hfe]), e1, e2]

/-- An `E′` entry's sign bit on a flat line: the line's sign, flipped when `β < 0`. At `β = 0` the zero product keeps
the line's sign (`mulCode`: `+0·(−44)` is `−0`). -/
def sgnB (β : ℚ) (b : Bool) : Bool := if β < 0 then !b else b

/-- Every `E′` code on a flat line is `c₀` or `c₁ = c₀` with the sign bit set, by `sgnB`; `c₀` has significand `0`
at `β = 0`. -/
theorem lineCodes_flat (β : ℚ) : ∃ c₀ c₁ : Code, CodePair c₀ c₁ ∧ (β = 0 → c₀.sig = 0) ∧
    ∀ (s : Fin 32 → Bool) (t : Fin 32), lineCodes β (flatLine s) t = if sgnB β (s t) then c₁ else c₀ := by
  obtain ⟨K, hK, hpos, hneg⟩ := e4m3Word_signs (β * 44)
  have hc := code_pair ⟨K, by omega⟩ ⟨128 + K, by omega⟩ hK rfl
  have hv : ∀ b : Bool, codeVal (e4m3 (if b then -44 else 44)) = if b then -44 else 44 := by
    intro b; cases b <;> decide +kernel
  have hsg : ∀ b : Bool, (e4m3 (if b then -44 else 44)).sign = b := by
    intro b; cases b <;> decide +kernel
  have hK0 : β = 0 → K = 0 := by
    rintro rfl
    have h0 : (e4m3Word (f32Word 0)).val = 0 := by decide +kernel
    simpa [h0] using hpos.symm
  have claim : ∀ b : Bool, e4m3Word (mulCode β (e4m3 (if b then -44 else 44))) =
      if sgnB β b then ⟨128 + K, by omega⟩ else ⟨K, by omega⟩ := by
    intro b
    rcases eq_or_ne β 0 with rfl | hβ0
    · obtain rfl := hK0 rfl
      have z0 : (e4m3Word 0).val = 0 := by decide +kernel
      have z1 : (e4m3Word 0x80000000).val = 128 := by decide +kernel
      unfold mulCode sgnB
      rw [hv, hsg]
      cases b <;> simp only [zero_mul, ↓reduceIte, lt_self_iff_false, decide_false, Bool.bne_true, Bool.bne_false,
        Bool.not_false, Bool.false_eq_true] <;> first | exact Fin.ext z0 | exact Fin.ext z1
    have hb : β * codeVal (e4m3 (if b then -44 else 44)) = if b then -(β * 44) else β * 44 := by
      rw [hv]; cases b <;> simp
    have hne : β * codeVal (e4m3 (if b then -44 else 44)) ≠ 0 := by
      rw [hb]; cases b <;> simp [hβ0]
    unfold mulCode
    rw [ite_eq_right hne, hb]
    apply Fin.ext
    rcases lt_or_gt_of_ne hβ0 with hβ | hβ
    · have h1 : β * 44 < 0 := by linarith
      have h2 : ¬ 0 < β * 44 := by linarith
      have hs : sgnB β b = !b := by simp [sgnB, hβ]
      rw [hs]
      cases b
      · simp only [Bool.false_eq_true, ite_false, Bool.not_false, ite_true]; rw [hpos]; simp [h1]
      · simp only [ite_true, Bool.not_true, Bool.false_eq_true, ite_false]; rw [hneg]; simp [h2]
    · have h1 : ¬ β * 44 < 0 := by linarith
      have h2 : 0 < β * 44 := by linarith
      have hs : sgnB β b = b := by simp [sgnB, not_lt.2 hβ.le]
      rw [hs]
      cases b
      · simp only [Bool.false_eq_true, ite_false]; rw [hpos]; simp [h1]
      · simp only [ite_true]; rw [hneg]; simp [h2]
  refine ⟨⟨K, by omega⟩, ⟨128 + K, by omega⟩, hc, fun hβ => ?_, fun s t => ?_⟩
  · obtain rfl := hK0 hβ
    show Code.sig (⟨0, by decide⟩ : Code) = 0
    decide
  · rw [← claim (s t)]; rfl

theorem codePair_F : CodePair ⟨67, by decide⟩ ⟨195, by decide⟩ := code_pair _ _ (by decide) rfl

theorem flatLineF_eq (s' : Fin 32 → Bool) (t : Fin 32) :
    flatLineF s' t = if s' t then (⟨195, by decide⟩ : Code) else ⟨67, by decide⟩ := by
  unfold flatLineF; cases s' t <;> decide +kernel

/-! ## The alignment on the support -/

theorem foldl_maxExp_congr (ts us : List Term)
    (h : ts.map (fun t => (t.mag, t.exp)) = us.map (fun t => (t.mag, t.exp))) (m : ℤ) :
    ts.foldl (fun m t => if t.mag ≠ 0 ∧ m < t.exp then t.exp else m) m =
      us.foldl (fun m t => if t.mag ≠ 0 ∧ m < t.exp then t.exp else m) m := by
  induction ts generalizing us m with
  | nil => cases us with
    | nil => rfl
    | cons => simp at h
  | cons t ts ih => cases us with
    | nil => simp at h
    | cons u us =>
      simp only [List.map_cons, List.cons.injEq, Prod.mk.injEq] at h
      simp only [List.foldl_cons, h.1.1, h.1.2]
      exact ih us h.2 _

theorem maxExp_sgnT (z : ℤ) (tc P : Term) (σ : Fin 32 → Bool) :
    maxExp z (tc :: List.ofFn fun t => sgnT (σ t) P) = maxExp z (tc :: List.ofFn fun _ : Fin 32 => P) := by
  unfold maxExp
  exact foldl_maxExp_congr _ _ (by simp [sgnT_mag, sgnT_exp]) z

theorem aligned_sgnT (w : ℕ) (E : ℤ) (σ : Bool) (T : Term) :
    aligned w E (sgnT σ T) = if σ then -aligned w E (sgnT false T) else aligned w E (sgnT false T) := by
  unfold aligned sgnT
  by_cases h : T.mag = 0 <;> cases σ <;> simp [h]

theorem aligned_sgnT_nonneg (w : ℕ) (E : ℤ) (T : Term) : 0 ≤ aligned w E (sgnT false T) := by
  unfold aligned sgnT
  by_cases h : T.mag = 0
  · simp [h]
  · simp only [h, ite_false, Bool.false_eq_true]
    exact Int.natCast_nonneg _

theorem aligned_sgnT_zero {w : ℕ} {E : ℤ} {σ : Bool} {T : Term} (hm : T.mag = 0) : aligned w E (sgnT σ T) = 0 := by
  unfold aligned sgnT
  simp [hm]

theorem alignedSum_sgnT (w : ℕ) (E : ℤ) (tc P : Term) (σ : Fin 32 → Bool) :
    alignedSum w E (tc :: List.ofFn fun t => sgnT (σ t) P) =
      aligned w E tc + aligned w E (sgnT false P) * ∑ t, (if σ t then -1 else 1 : ℤ) := by
  unfold alignedSum
  rw [List.map_cons, List.sum_cons, List.map_ofFn, List.sum_ofFn, Finset.mul_sum]
  congr 1
  refine Finset.sum_congr rfl fun t _ => ?_
  simp only [Function.comp_apply]
  rw [aligned_sgnT]
  split_ifs <;> ring

theorem sum_sgnB_pos {β : ℚ} (hβ : 0 < β) (s s' : Fin 32 → Bool) :
    ∑ t, (if (sgnB β (s t) ^^ s' t) then -1 else 1 : ℤ) = flatCount s s' := by
  unfold flatCount
  refine Finset.sum_congr rfl fun t _ => ?_
  simp only [sgnB, not_lt.2 hβ.le, ite_false]
  cases s t <;> cases s' t <;> rfl

theorem sum_sgnB_neg {β : ℚ} (hβ : β < 0) (s s' : Fin 32 → Bool) :
    ∑ t, (if (sgnB β (s t) ^^ s' t) then -1 else 1 : ℤ) = -flatCount s s' := by
  unfold flatCount
  rw [← Finset.sum_neg_distrib]
  refine Finset.sum_congr rfl fun t _ => ?_
  simp only [sgnB, hβ, ite_true]
  cases s t <;> cases s' t <;> rfl

/-! ## The step on the support -/

/-- The step of `⟨[n], w, z⟩` on a non-finite accumulator (NaN or an infinity) and finite codes writes the
accumulator's own value, whatever the products. -/
theorem pipeStep_nonfin {n w : ℕ} {z : ℤ} {acc : ℕ} (hacc : ∀ t, fp32Value acc ≠ .fin t) (a b : Fin n → Code)
    (ha : ∀ l, (a l).isNaN = false) (hb : ∀ l, (b l).isNaN = false) :
    (⟨[n], w, z⟩ : Pipeline).step acc (List.ofFn a) (List.ofFn b) = packValue (fp32Value acc) := by
  unfold Pipeline.step
  simp only [Pipeline.fold, zip_ofFnN]
  congr 1
  rw [List.take_of_length_le (by simp), List.map_ofFn]
  have hp : ((fun x : Code × Code => productTotal x.1 x.2) ∘ fun l => (a l, b l)) =
      fun l => Value.fin (product (a l) (b l)) := by
    funext l; simp [productTotal, ha l, hb l]
  rw [hp]
  have h1 : (List.ofFn fun l => Value.fin (product (a l) (b l))).any Value.isNaN = false := by
    rw [List.any_eq_false]; intro v hv
    obtain ⟨l, rfl⟩ := List.mem_ofFn.1 hv; simp [Value.isNaN]
  have h2 : ∀ s, (List.ofFn fun l => Value.fin (product (a l) (b l))).any (Value.isInf s) = false := by
    intro s; rw [List.any_eq_false]; intro v hv
    obtain ⟨l, rfl⟩ := List.mem_ofFn.1 hv; simp [Value.isInf]
  unfold groupStepTotal
  rcases h : fp32Value acc with _ | s | t
  · simp [Value.isNaN]
  · cases s <;> simp [List.any_cons, h1, h2, Value.isNaN, Value.isInf]
  · exact absurd h (hacc t)

/-- **The step on the support, finite accumulator.** At every flat pair the code's value is the read `codeW` at one
exponent `E` of `aligned(C) + A·S`, where `S` counts the products' signs and `A ≥ 0` is one product's aligned
magnitude, `0` at `β = 0`. -/
theorem flat_fin_eq {w : ℕ} (z : ℤ) (α y β : ℚ) {tc : Term} (hacc : fp32Value (f32Word (α * y)) = .fin tc)
    {c₀ c₁ : Code} (hc : CodePair c₀ c₁)
    (hline : ∀ (s : Fin 32 → Bool) (t : Fin 32), lineCodes β (flatLine s) t = if sgnB β (s t) then c₁ else c₀)
    (s s' : Fin 32 → Bool) :
    codeVal ((atomMixP ⟨[32], w, z⟩).elem α y β (flatLine s) (flatLineF s')) =
      codeW w (maxExp z (tc :: List.ofFn fun _ : Fin 32 => product c₀ ⟨67, by decide⟩))
        (alignedSum w (maxExp z (tc :: List.ofFn fun _ : Fin 32 => product c₀ ⟨67, by decide⟩))
          (tc :: List.ofFn fun t => sgnT (sgnB β (s t) ^^ s' t) (product c₀ ⟨67, by decide⟩))) := by
  have hprod : ∀ t, product (lineCodes β (flatLine s) t) (flatLineF s' t) =
      sgnT (sgnB β (s t) ^^ s' t) (product c₀ ⟨67, by decide⟩) := fun t => by
    rw [hline, flatLineF_eq]; exact product_pair hc codePair_F _ _
  have hna : ∀ t, (lineCodes β (flatLine s) t).isNaN = false := fun t => by
    rw [hline]; split
    · exact hc.2.2.1
    · exact hc.1
  have hnb : ∀ t, (flatLineF s' t).isNaN = false := fun t => by
    rw [flatLineF_eq]; split <;> decide
  have hstep := pipeStep_fin (w := w) (z := z) hacc _ _ hna hnb
  simp only [hprod, groupSum] at hstep
  rw [maxExp_sgnT] at hstep
  unfold codeW readW
  unfold FormChain.elem atomMixP
  dsimp only
  rw [hstep]
  rfl

theorem flat_fin {w : ℕ} (z : ℤ) (α y β : ℚ) {tc : Term} (hacc : fp32Value (f32Word (α * y)) = .fin tc) :
    ∃ E A : ℤ, 0 ≤ A ∧ (β = 0 → A = 0) ∧ ∀ s s' : Fin 32 → Bool,
      codeVal ((atomMixP ⟨[32], w, z⟩).elem α y β (flatLine s) (flatLineF s')) =
        codeW w E (aligned w E tc + A * ∑ t, (if (sgnB β (s t) ^^ s' t) then -1 else 1 : ℤ)) := by
  obtain ⟨c₀, c₁, hc, hc0, hline⟩ := lineCodes_flat β
  refine ⟨maxExp z (tc :: List.ofFn fun _ : Fin 32 => product c₀ ⟨67, by decide⟩),
    aligned w (maxExp z (tc :: List.ofFn fun _ : Fin 32 => product c₀ ⟨67, by decide⟩))
      (sgnT false (product c₀ ⟨67, by decide⟩)), aligned_sgnT_nonneg _ _ _, fun hβ => ?_, fun s s' => ?_⟩
  · refine aligned_sgnT_zero ?_
    unfold product; simp [hc0 hβ, Term.zero]
  · rw [flat_fin_eq z α y β hacc hc hline s s', alignedSum_sgnT]

/-- **The structure on the support.** On a one-group step the code's value at every flat pair is `Φ(a·n)`, with
`n = flatCount s s'`, `Φ` monotone and `a` of the sign of `β` (`a = 0` at `β = 0`), for every `α`, `x` and `β`. -/
theorem flat_structure (p : Pipeline) (hp : p.groups = [32]) (hw : 1 ≤ p.width) (α y β : ℚ) :
    ∃ Φ : ℤ → ℚ, Monotone Φ ∧ ∃ a : ℤ, (0 ≤ β → 0 ≤ a) ∧ (β ≤ 0 → a ≤ 0) ∧
      ∀ s s' : Fin 32 → Bool,
        codeVal ((atomMixP p).elem α y β (flatLine s) (flatLineF s')) = Φ (a * flatCount s s') := by
  obtain ⟨g, w, z⟩ := p
  change g = [32] at hp
  change 1 ≤ w at hw
  subst hp
  by_cases hfin : ∃ tc, fp32Value (f32Word (α * y)) = .fin tc
  · obtain ⟨tc, hacc⟩ := hfin
    obtain ⟨E, A, hA, hA0, hf⟩ := flat_fin (w := w) z α y β hacc
    refine ⟨fun m => codeW w E (aligned w E tc + m), ?_, ?_⟩
    · intro m m' h; exact codeW_mono hw E (by linarith)
    rcases lt_trichotomy β 0 with hβ | rfl | hβ
    · refine ⟨-A, fun h => absurd h (not_le.2 hβ), fun _ => by linarith, fun s s' => ?_⟩
      rw [hf, sum_sgnB_neg hβ]; ring_nf
    · refine ⟨0, fun _ => le_rfl, fun _ => le_rfl, fun s s' => ?_⟩
      rw [hf, hA0 rfl]; simp
    · refine ⟨A, fun _ => hA, fun h => absurd h (not_le.2 hβ), fun s s' => ?_⟩
      rw [hf, sum_sgnB_pos hβ]
  · simp only [not_exists] at hfin
    obtain ⟨c₀, c₁, hc, -, hline⟩ := lineCodes_flat β
    refine ⟨fun _ => codeVal (e4m3 (PearlC.wordVal (packValue (fp32Value (f32Word (α * y)))))),
      fun _ _ _ => le_rfl, 0, fun _ => le_rfl, fun _ => le_rfl, fun s s' => ?_⟩
    have hna : ∀ t, (lineCodes β (flatLine s) t).isNaN = false := fun t => by
      rw [hline]; split
      · exact hc.2.2.1
      · exact hc.1
    have hnb : ∀ t, (flatLineF s' t).isNaN = false := fun t => by
      rw [flatLineF_eq]; split <;> decide
    show codeVal (e4m3 (PearlC.wordVal ((⟨[32], w, z⟩ : Pipeline).step (f32Word (α * y))
      (List.ofFn (lineCodes β (flatLine s))) (List.ofFn (flatLineF s'))))) = _
    rw [pipeStep_nonfin hfin _ _ hna hnb]

/-! ## The lemma -/

/-- **(M⁺)** On a one-group step with `0 ≤ β`, an element's code value over the support is monotone in the signed
count. -/
theorem atomMixP_flat_mono (p : Pipeline) (hp : p.groups = [32]) (hw : 1 ≤ p.width) (α y : ℚ) {β : ℚ} (hβ : 0 ≤ β)
    {s₁ s₁' s₂ s₂' : Fin 32 → Bool} (h : flatCount s₁ s₁' ≤ flatCount s₂ s₂') :
    codeVal ((atomMixP p).elem α y β (flatLine s₁) (flatLineF s₁')) ≤
      codeVal ((atomMixP p).elem α y β (flatLine s₂) (flatLineF s₂')) := by
  obtain ⟨Φ, hΦ, a, ha, -, hf⟩ := flat_structure p hp hw α y β
  rw [hf, hf]
  exact hΦ (mul_le_mul_of_nonneg_left h (ha hβ))

/-- **(M⁻)** On a one-group step with `β ≤ 0`, an element's code value over the support is antitone in the signed
count. -/
theorem atomMixP_flat_anti (p : Pipeline) (hp : p.groups = [32]) (hw : 1 ≤ p.width) (α y : ℚ) {β : ℚ} (hβ : β ≤ 0)
    {s₁ s₁' s₂ s₂' : Fin 32 → Bool} (h : flatCount s₁ s₁' ≤ flatCount s₂ s₂') :
    codeVal ((atomMixP p).elem α y β (flatLine s₂) (flatLineF s₂')) ≤
      codeVal ((atomMixP p).elem α y β (flatLine s₁) (flatLineF s₁')) := by
  obtain ⟨Φ, hΦ, a, -, ha, hf⟩ := flat_structure p hp hw α y β
  rw [hf, hf]
  exact hΦ (mul_le_mul_of_nonpos_left h (ha hβ))

/-- **(E) The salt-dead lemma**: on a one-group step an element is salt-dead iff the support's two ends agree. -/
theorem saltDeadP_iff_ends (p : Pipeline) (hp : p.groups = [32]) (hw : 1 ≤ p.width)
    (alpha beta : (k : ℕ) → (Fin k → ℚ) → ℚ) (k : ℕ) (x : Fin k → ℚ) (l : Fin k) :
    saltDeadP p alpha beta k x l ↔ EndsAgree p (alpha k x) (x l) (beta k x) := by
  unfold saltDeadP EndsAgree
  constructor
  · intro h
    exact h plusEnd ⟨_, _, rfl⟩ minusEnd ⟨_, _, rfl⟩
  · intro h q hq q' hq'
    obtain ⟨Φ, hΦ, a, -, -, hf⟩ := flat_structure p hp hw (alpha k x) (x l) (beta k x)
    have hE : Φ (a * 32) = Φ (a * -32) := by
      have h' := h
      simp only [plusEnd, minusEnd] at h'
      rwa [hf, hf, flatCount_plus, flatCount_minus] at h'
    have key : ∀ s s' : Fin 32 → Bool,
        codeVal ((atomMixP p).elem (alpha k x) (x l) (beta k x) (flatLine s) (flatLineF s')) = Φ (a * 32) := by
      intro s s'
      rw [hf]
      have h1 := neg_le_flatCount s s'
      have h2 := flatCount_le s s'
      rcases le_total 0 a with ha | ha
      · refine le_antisymm (hΦ (mul_le_mul_of_nonneg_left h2 ha)) ?_
        rw [hE]; exact hΦ (mul_le_mul_of_nonneg_left h1 ha)
      · refine le_antisymm ?_ (hΦ (mul_le_mul_of_nonpos_left h2 ha))
        rw [hE]; exact hΦ (mul_le_mul_of_nonpos_left h1 ha)
    obtain ⟨s₁, s₁', rfl⟩ := hq
    obtain ⟨s₂, s₂', rfl⟩ := hq'
    rw [key, key]

open Classical in
/-- **(C)** The count RowA's `sd` port carries, per block `B` of elements. -/
theorem saltDeadP_card_ends (p : Pipeline) (hp : p.groups = [32]) (hw : 1 ≤ p.width)
    (alpha beta : (k : ℕ) → (Fin k → ℚ) → ℚ) (k : ℕ) (x : Fin k → ℚ) (B : Finset (Fin k)) :
    (B.filter (saltDeadP p alpha beta k x)).card =
      (B.filter fun l => EndsAgree p (alpha k x) (x l) (beta k x)).card := by
  congr 1
  exact Finset.filter_congr fun l _ => saltDeadP_iff_ends p hp hw alpha beta k x l

theorem codeVal_eq_iff_all : ∀ c d : Code, c.isNaN = false → d.isNaN = false →
    (codeVal c = codeVal d ↔ c = d ∨ (c.val % 128 = 0 ∧ d.val % 128 = 0)) := by
  decide +kernel

/-- **(Z)** The circuit's comparison: two non-NaN codes have one value iff they are equal or both zeros. -/
theorem codeVal_eq_iff (c d : Code) (hc : c.isNaN = false) (hd : d.isNaN = false) :
    codeVal c = codeVal d ↔ c = d ∨ (c.val % 128 = 0 ∧ d.val % 128 = 0) :=
  codeVal_eq_iff_all c d hc hd

/-- The lemma at sm_120's step. -/
theorem saltDeadP_iff_ends_sm120 (alpha beta : (k : ℕ) → (Fin k → ℚ) → ℚ) (k : ℕ) (x : Fin k → ℚ) (l : Fin k) :
    saltDeadP sm120E4m3 alpha beta k x l ↔ EndsAgree sm120E4m3 (alpha k x) (x l) (beta k x) :=
  saltDeadP_iff_ends sm120E4m3 rfl (by decide) alpha beta k x l

/-- The lemma at Hopper's step (Π's `SaltDead.saltDead`, `saltDeadP_hopper`). -/
theorem saltDeadP_iff_ends_hopper (alpha beta : (k : ℕ) → (Fin k → ℚ) → ℚ) (k : ℕ) (x : Fin k → ℚ) (l : Fin k) :
    saltDeadP hopperE4m3K32 alpha beta k x l ↔ EndsAgree hopperE4m3K32 (alpha k x) (x l) (beta k x) :=
  saltDeadP_iff_ends hopperE4m3K32 rfl (by decide) alpha beta k x l

end Pouw.PearlC.Cap
