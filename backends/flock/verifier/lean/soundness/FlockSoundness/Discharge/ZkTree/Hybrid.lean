import FlockSoundness.Discharge.ZkTree.Lazy

/-!
# The tree-coin hybrid

`treeCoins_le_model_of`: a committed prover of `g` (shown every leaf before its first message, and each round's block
and salt at its opening) is, on every event of its output and of the leaves it was shown, within `|I|·δ` of a mixture
of strategies of the model `G` (`ErD g G`), when no path opens a round twice and the leaf is two-vector `δ`-hiding
(`Hm96TwoVector`). The mixture draws for itself the leaves of independent blocks and salts, which is its `hello`, and,
at each opening, the rest of the block and a salt consistent with the leaf it showed. The mixture does not depend on
the event.

`treeCoins_le_model`: the same at hm96-sha512 under a uniform session key that the prover sees, with `δ = 2^-256` a
leaf and no hypothesis: the key and the leaves are the mixture's `hello`.

The proof is `ZkCoins.exp_step`'s hybrid, a round at a time from its real leaf to the leaf of an independent block and
salt (`fill_le`; the event reads the leaves through a first message, `TGame.CStrat.mapS`), then `lazy_mix` at the
all-ideal end, where the prover's view is fixed by the ideal leaves. A round's step costs `δ`: its leaf `x + M·y` and
an independent block's `x' + M·y` are `δ`-close (`hide_of_twoVector`). hm96 §3's one-vector bound `δ₁` gives the
two-vector case at `2·δ₁` (`twoVector_of_hiding`), so `2·2^-257 = 2^-256` a leaf at hm96-sha512.
-/

namespace FlockSoundness.Discharge.ZkTree

open FlockSoundness Game Discharge.ZkCoins Finset Function
open scoped ENNReal

namespace TGame

open Discharge.ZkCoins.TGame

variable {I Blk Salt α β : Type}

/-- A committed strategy of `g` as one of `g.map f`. -/
def CStrat.mapS (f : α → β) : (g : TGame I Blk α) → CStrat Salt g → CStrat Salt (g.map f)
  | .ret _, _ => PUnit.unit
  | .send _ k, s => ⟨s.1, CStrat.mapS f (k s.1) s.2⟩
  | .coin _ k, s => fun b y => CStrat.mapS f (k b) (s b y)
  | @TGame.fresh _ _ _ _ _ _ k, s => fun c => CStrat.mapS f (k c) (s c)

theorem pr_mapS (p : β → Prop) (f : α → β) :
    ∀ (g : TGame I Blk α) (s : CStrat Salt g) (ω : I → Blk × Salt),
      pr p (g.map f) (CStrat.mapS f g s) ω = pr (fun a => p (f a)) g s ω
  | .ret _, _, _ => rfl
  | .send _ k, s, ω => pr_mapS p f (k s.1) s.2 ω
  | .coin i k, s, ω => pr_mapS p f (k (ω i).1) _ ω
  | @TGame.fresh _ _ _ C iF _ k, s, ω => by
      show (@avg C iF fun c => pr p ((k c).map f) (CStrat.mapS f (k c) (s c)) ω) =
        @avg C iF fun c => pr _ (k c) (s c) ω
      exact congrArg _ (funext fun c => pr_mapS p f (k c) (s c) ω)

end TGame

/-! ## Uniform laws as product laws -/

section Uniform

variable {J Zt : Type} [Fintype J] [DecidableEq J] [Fintype Zt]

theorem avg_eq_W (F : (J → Zt) → ℝ≥0∞) :
    avg F = ∑ ω, W (fun _ _ => (Fintype.card Zt : ℝ≥0∞)⁻¹) ω * F ω := by
  unfold avg W
  simp only [Finset.prod_const, Finset.card_univ, Fintype.card_pi, Nat.cast_pow, ENNReal.inv_pow]

/-- **Resetting one coordinate of a uniform draw** to an independent uniform value leaves its average unchanged. -/
theorem avg_update [Nonempty Zt] (k : J) (F : (J → Zt) → ℝ≥0∞) :
    avg (fun z : Zt => avg fun ω : J → Zt => F (update ω k z)) = avg F := by
  have hμ : ∑ _z : Zt, (Fintype.card Zt : ℝ≥0∞)⁻¹ = 1 := by
    rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
    exact ENNReal.mul_inv_cancel (Nat.cast_ne_zero.2 Fintype.card_ne_zero) (ENNReal.natCast_ne_top _)
  rw [avg_eq_W F, resample (k := k) hμ F]
  simp only [avg_eq_W (fun ω => F (update ω k _))]
  rw [← sum_mul_avg]
  rfl

end Uniform

/-! ## hm96's two-vector case -/

theorem prCoin_le_one {C : Type} [Fintype C] [Nonempty C] (p : C → Prop) : prCoin p ≤ 1 :=
  calc prCoin p ≤ avg (fun _ : C => (1 : ℝ≥0∞)) := avg_mono fun c => by
          show (@ite _ (p c) (Classical.propDecidable _) (1 : ℝ≥0∞) 0) ≤ 1
          split_ifs <;> simp
    _ = 1 := avg_const 1

/-- **hm96 §3's two-vector case from its one-vector bound**: if `(M·y, c y)` is `δ`-close to `(U, c y)` on every
event, then `(x + M·y, c y)` and `(x' + M·y, c y)` are `2δ`-close, through `(U, c y)`, which a shift by `x` or by `x'`
leaves as it is. -/
theorem twoVector_of_hiding {Sl B C : Type} [Fintype Sl] [Nonempty Sl] [AddCommGroup B] [Fintype B]
    (My : Sl → B) (c : Sl → C) {δ : ℝ≥0∞} (h : Assumptions.Hm96Hiding My c δ) : Hm96TwoVector My c (δ + δ) := by
  intro x x' P
  have h1 : prCoin (fun y => P (x + My y, c y)) ≤ prCoin (fun q : B × Sl => P (x + q.1, c q.2)) + δ :=
    h fun v => P (x + v.1, v.2)
  have hsh : prCoin (fun q : B × Sl => P (x + q.1, c q.2)) = prCoin (fun q : B × Sl => P (x' + q.1, c q.2)) := by
    unfold prCoin avg
    have hx : ∀ q : B × Sl, x' + ((x - x') + q.1) = x + q.1 := fun q => by abel
    refine Fintype.sum_equiv ((Equiv.addLeft (x - x')).prodCongr (Equiv.refl Sl)) _ _ fun q => ?_
    show _ = (Fintype.card (B × Sl) : ℝ≥0∞)⁻¹ *
      @ite _ (P (x' + ((x - x') + q.1), c q.2)) (Classical.propDecidable _) 1 0
    rw [hx]
  have h3 : prCoin (fun y => ¬ P (x' + My y, c y)) ≤ prCoin (fun q : B × Sl => ¬ P (x' + q.1, c q.2)) + δ :=
    h fun v => ¬ P (x' + v.1, v.2)
  rw [Audit.Law.prCoin_not (fun y => P (x' + My y, c y)), Audit.Law.prCoin_not (fun q : B × Sl => P (x' + q.1, c q.2))]
    at h3
  have hU := prCoin_le_one (fun q : B × Sl => P (x' + q.1, c q.2))
  have h4 : prCoin (fun q : B × Sl => P (x' + q.1, c q.2)) ≤ prCoin (fun y => P (x' + My y, c y)) + δ := by
    rw [← ENNReal.sub_sub_cancel ENNReal.one_ne_top hU, tsub_le_iff_left]
    have := tsub_le_iff_right.1 h3
    calc (1 : ℝ≥0∞) ≤ 1 - prCoin (fun q : B × Sl => P (x' + q.1, c q.2)) + δ + prCoin (fun y => P (x' + My y, c y)) :=
          this
      _ = 1 - prCoin (fun q : B × Sl => P (x' + q.1, c q.2)) + (prCoin (fun y => P (x' + My y, c y)) + δ) := by
          rw [add_assoc, add_comm δ]
  calc prCoin (fun y => P (x + My y, c y)) ≤ prCoin (fun q : B × Sl => P (x + q.1, c q.2)) + δ := h1
    _ = prCoin (fun q : B × Sl => P (x' + q.1, c q.2)) + δ := by rw [hsh]
    _ ≤ prCoin (fun y => P (x' + My y, c y)) + δ + δ := add_le_add h4 le_rfl
    _ = prCoin (fun y => P (x' + My y, c y)) + (δ + δ) := add_assoc _ _ _

section Hybrid

variable {I Blk Salt B Cs D : Type} [Fintype I] [DecidableEq I] [Fintype Blk] [Nonempty Blk] [Fintype Salt]
  [Nonempty Salt]

omit [Fintype I] [DecidableEq I] in
/-- **A round's leaf against an independent block's**: under two-vector hiding, a test of round `k`'s leaf on block
`b` and a uniform salt moves by at most `δ` when the block is replaced by an independent uniform one. -/
theorem hide_of_twoVector [AddCommGroup B] [Fintype B] (dig : I → Blk → B) (My : Salt → B) (cs : Salt → Cs)
    (H : B × Cs → D) {δ : ℝ≥0∞} (hT : Hm96TwoVector My cs δ) (k : I) (b : Blk) (Γ : D → ℝ≥0∞)
    (hΓ : ∀ d, Γ d ≤ 1) :
    avg (fun y => Γ (leaf dig My cs H k (b, y))) ≤ avg (fun q : Blk × Salt => Γ (leaf dig My cs H k q)) + δ := by
  classical
  have h1 : ∀ b' : Blk, avg (fun y => Γ (H (dig k b + My y, cs y))) ≤
      avg (fun y => Γ (H (dig k b' + My y, cs y))) + δ := fun b' =>
    avg_le_of_prCoin (fun y => (dig k b + My y, cs y)) (fun y => (dig k b' + My y, cs y))
      ((univ : Finset B) ×ˢ univ.image cs)
      (fun y => mem_product.2 ⟨mem_univ _, mem_image_of_mem cs (mem_univ y)⟩)
      (fun y => mem_product.2 ⟨mem_univ _, mem_image_of_mem cs (mem_univ y)⟩)
      (fun P => hT _ _ P) (fun v => Γ (H v)) (fun v => hΓ _)
  calc avg (fun y => Γ (leaf dig My cs H k (b, y)))
      = avg (fun _ : Blk => avg fun y => Γ (H (dig k b + My y, cs y))) := (avg_const _).symm
    _ ≤ avg (fun b' : Blk => avg (fun y => Γ (H (dig k b' + My y, cs y))) + δ) := avg_mono h1
    _ = avg (fun q : Blk × Salt => Γ (leaf dig My cs H k q)) + δ := by
        rw [avg_add, avg_const, ZK.avg_prod_split]
        rfl

/-- Round `j` ideal with the leaf of `q j` where `m` leaves it real. -/
def fillM (ℓ : I → Blk × Salt → D) (m : I → Option D) (q : I → Blk × Salt) (j : I) : Option D :=
  some ((m j).getD (ℓ j (q j)))

omit [Fintype I] [Fintype Blk] [Nonempty Blk] [Fintype Salt] [Nonempty Salt] in
theorem fillM_update (ℓ : I → Blk × Salt → D) (m : I → Option D) {k : I} (hk : m k = none)
    (q : I → Blk × Salt) (q' : Blk × Salt) :
    fillM ℓ (update m k (some (ℓ k q'))) q = fillM ℓ m (update q k q') := by
  funext j
  by_cases hj : j = k
  · subst hj
    simp [fillM, hk]
  · simp [fillM, update_of_ne hj]

/-- **The hybrid, to the ideal end**: with `n` real rounds, the committed prover's success probability is at most
its average over independent blocks and salts `q` with every real round made ideal at `q`'s leaf, plus `n·δ`. -/
theorem fill_le {β : Type} (p : β → Prop) (g : TGame I Blk β) (ℓ : I → Blk × Salt → D)
    (τ : (I → D) → TGame.CStrat Salt g) {δ : ℝ≥0∞}
    (hide : ∀ (k : I) (b : Blk) (Γ : D → ℝ≥0∞), (∀ d, Γ d ≤ 1) →
      avg (fun y => Γ (ℓ k (b, y))) ≤ avg (fun q => Γ (ℓ k q)) + δ) :
    ∀ (n : ℕ) (m : I → Option D), (univ.filter fun j => m j = none).card = n →
      Exp p g ℓ τ m ≤ avg (fun q : I → Blk × Salt => Exp p g ℓ τ (fillM ℓ m q)) + n * δ := by
  intro n
  induction n with
  | zero =>
    intro m hm
    have hfill : ∀ q, fillM ℓ m q = m := by
      intro q
      funext j
      cases h : m j with
      | none =>
        have hj : j ∈ univ.filter fun j => m j = none := by simp [h]
        exact absurd (Finset.card_pos.2 ⟨j, hj⟩) (by omega)
      | some v => simp [fillM, h]
    simp only [hfill, avg_const, Nat.cast_zero, zero_mul, add_zero, le_refl]
  | succ n ih =>
    intro m hm
    obtain ⟨k, hk⟩ : ∃ k, m k = none := by
      obtain ⟨k, hk⟩ := Finset.card_pos.1 (show 0 < (univ.filter fun j => m j = none).card by omega)
      exact ⟨k, (mem_filter.1 hk).2⟩
    have hkm : k ∈ univ.filter fun j => m j = none := by simp [hk]
    calc Exp p g ℓ τ m ≤ avg (fun q' => Exp p g ℓ τ (update m k (some (ℓ k q')))) + δ :=
          exp_step p g ℓ τ (ℓ k) k (hide k) m hk
      _ ≤ avg (fun q' => avg (fun q => Exp p g ℓ τ (fillM ℓ (update m k (some (ℓ k q'))) q)) + n * δ) + δ := by
          refine add_le_add (avg_mono fun q' => ih _ ?_) le_rfl
          rw [card_none_update, Finset.card_erase_of_mem hkm, hm, Nat.add_sub_cancel]
      _ = avg (fun q => Exp p g ℓ τ (fillM ℓ m q)) + ((n + 1 : ℕ) : ℝ≥0∞) * δ := by
          simp only [fillM_update ℓ m hk]
          rw [avg_add, avg_const, avg_update k (fun q => Exp p g ℓ τ (fillM ℓ m q))]
          push_cast
          ring

variable [DecidableEq Blk] {α : Type} {g : TGame I Blk α} {G : Game α}

/-- The salt's posterior given the leaf of `q j`, for round `j`'s block `b`. -/
noncomputable def postK (ℓ : I → Blk × Salt → D) (q : I → Blk × Salt) (j : I) (b : Blk) (y : Salt) : ℝ≥0∞ :=
  kern (fun y => ℓ j (b, y)) (ℓ j (q j)) y

omit [Fintype I] [DecidableEq I] [Fintype Blk] [Nonempty Blk] [DecidableEq Blk] in
theorem sum_postK (ℓ : I → Blk × Salt → D) (q : I → Blk × Salt) (j : I) (b : Blk) : ∑ y, postK ℓ q j b y = 1 :=
  sum_kern _ _

/-- **The mixture of the hybrid's ideal end**: independent blocks and salts `q` (their leaves are its `hello`), and at
each opening of round `i` the rest of `(r i).1`'s block and a salt from the posterior of `q i`'s leaf given the block
it opens. -/
noncomputable def mixOf (hgG : ErD g G) (ℓ : I → Blk × Salt → D) (τ : (I → D) → TGame.CStrat Salt g) :
    Mix G (I → D) where
  Ρ := (I → Blk × Salt) × (I → Blk × (Blk → Salt))
  w x := (Fintype.card (I → Blk × Salt) : ℝ≥0∞)⁻¹ * W (wtR (postK ℓ x.1)) x.2
  sum_w := by
    rw [Fintype.sum_prod_type]
    simp only [← Finset.mul_sum, sum_W_one (sum_wtR (sum_postK ℓ _)), mul_one, Finset.sum_const, Finset.card_univ,
      nsmul_eq_mul]
    exact ENNReal.mul_inv_cancel (Nat.cast_ne_zero.2 Fintype.card_ne_zero) (ENNReal.natCast_ne_top _)
  σ x := toStrat hgG (τ fun j => ℓ j (x.1 j)) x.2
  hello x := fun j => ℓ j (x.1 j)

theorem mixOf_pr (hgG : ErD g G) (ℓ : I → Blk × Salt → D) (τ : (I → D) → TGame.CStrat Salt g)
    (E : (I → D) → α → Prop) :
    (mixOf hgG ℓ τ).pr E = avg fun q : I → Blk × Salt => ∑ r, W (wtR (postK ℓ q)) r *
      prob (E fun j => ℓ j (q j)) G (toStrat hgG (τ fun j => ℓ j (q j)) r) := by
  show ∑ x : (I → Blk × Salt) × (I → Blk × (Blk → Salt)), (Fintype.card (I → Blk × Salt) : ℝ≥0∞)⁻¹ *
      W (wtR (postK ℓ x.1)) x.2 * prob (E fun j => ℓ j (x.1 j)) G (toStrat hgG (τ fun j => ℓ j (x.1 j)) x.2) = _
  unfold avg
  rw [Fintype.sum_prod_type]
  refine Finset.sum_congr rfl fun q _ => ?_
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun r _ => ?_
  exact mul_assoc _ _ _

variable [AddCommGroup B] [Fintype B]

/-- **The tree-coin hybrid.** -/
theorem treeCoins_le_model_of (hgG : ErD g G) (hg : g.Distinct)
    (dig : I → Blk → B) (My : Salt → B) (cs : Salt → Cs) (H : B × Cs → D) {δ : ℝ≥0∞}
    (hT : Hm96TwoVector My cs δ) (τ : (I → D) → TGame.CStrat Salt g) :
    ∃ M : Mix G (I → D), ∀ E : (I → D) → α → Prop,
      avg (fun ω : I → Blk × Salt =>
        TGame.pr (E fun i => leaf dig My cs H i (ω i)) g (τ fun i => leaf dig My cs H i (ω i)) ω) ≤
        M.pr E + Fintype.card I * δ := by
  refine ⟨mixOf hgG (leaf dig My cs H) τ, fun E => ?_⟩
  set ℓ := leaf dig My cs H with hℓ
  let g' : TGame I Blk ((I → D) × α) := .send (I → D) fun v => g.map (Prod.mk v)
  let τ' : (I → D) → TGame.CStrat Salt g' := fun v => ⟨v, TGame.CStrat.mapS (Prod.mk v) g (τ v)⟩
  let p' : (I → D) × α → Prop := fun x => E x.1 x.2
  have hexp : ∀ m, Exp p' g' ℓ τ' m = ∑ ω, W (wt ℓ m) ω * TGame.pr (E (view ℓ m ω)) g (τ (view ℓ m ω)) ω :=
    fun m => Finset.sum_congr rfl fun ω _ => congrArg _ (TGame.pr_mapS p' (Prod.mk (view ℓ m ω)) g _ ω)
  have hnone : wt ℓ (fun _ => none) = fun _ _ => (Fintype.card (Blk × Salt) : ℝ≥0∞)⁻¹ :=
    funext fun _ => funext fun z => wt_none rfl z
  have hstart : avg (fun ω : I → Blk × Salt => TGame.pr (E fun i => ℓ i (ω i)) g (τ fun i => ℓ i (ω i)) ω) =
      Exp p' g' ℓ τ' (fun _ => none) := by
    rw [hexp, hnone, avg_eq_W]
    rfl
  have hend : ∀ q : I → Blk × Salt, Exp p' g' ℓ τ' (fillM ℓ (fun _ => none) q) =
      ∑ r, W (wtR (postK ℓ q)) r * prob (E fun j => ℓ j (q j)) G (toStrat hgG (τ fun j => ℓ j (q j)) r) := by
    intro q
    rw [hexp]
    exact lazy_mix (E fun j => ℓ j (q j)) (sum_postK ℓ q) hgG hg (τ fun j => ℓ j (q j))
  have hcard : (univ.filter fun j : I => (fun _ : I => (none : Option D)) j = none).card = Fintype.card I := by
    simp
  calc avg (fun ω : I → Blk × Salt => TGame.pr (E fun i => ℓ i (ω i)) g (τ fun i => ℓ i (ω i)) ω)
      = Exp p' g' ℓ τ' (fun _ => none) := hstart
    _ ≤ avg (fun q : I → Blk × Salt => Exp p' g' ℓ τ' (fillM ℓ (fun _ => none) q)) + Fintype.card I * δ :=
        fill_le p' g' ℓ τ' (fun k b Γ hΓ => hide_of_twoVector dig My cs H hT k b Γ hΓ) _ _ hcard
    _ = (mixOf hgG ℓ τ).pr E + Fintype.card I * δ := by
        rw [mixOf_pr]
        simp only [hend]

end Hybrid

/-- **A mixture over a uniform `κ`**: `κ`, then `M κ`'s randomness; `hello` is `κ` and `M κ`'s. -/
noncomputable def Mix.avgOver {α Hl K : Type} {G : Game α} [Fintype K] [Nonempty K] (M : K → Mix G Hl) :
    Mix G (K × Hl) where
  Ρ := (Σ κ : K, (M κ).Ρ)
  w x := (Fintype.card K : ℝ≥0∞)⁻¹ * (M x.1).w x.2
  sum_w := by
    rw [Fintype.sum_sigma]
    simp only [← Finset.mul_sum, Mix.sum_w, mul_one, Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
    exact ENNReal.mul_inv_cancel (Nat.cast_ne_zero.2 Fintype.card_ne_zero) (ENNReal.natCast_ne_top _)
  σ x := (M x.1).σ x.2
  hello x := (x.1, (M x.1).hello x.2)

theorem Mix.avgOver_pr {α Hl K : Type} {G : Game α} [Fintype K] [Nonempty K] (M : K → Mix G Hl)
    (E : K × Hl → α → Prop) : (Mix.avgOver M).pr E = avg fun κ => (M κ).pr fun v => E (κ, v) := by
  show ∑ x : (Σ κ : K, (M κ).Ρ), (Fintype.card K : ℝ≥0∞)⁻¹ * (M x.1).w x.2 *
      prob (E (x.1, (M x.1).hello x.2)) G ((M x.1).σ x.2) = _
  unfold avg
  rw [Fintype.sum_sigma]
  refine Finset.sum_congr rfl fun κ _ => ?_
  dsimp only
  unfold Mix.pr
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun ρ _ => ?_
  exact mul_assoc _ _ _

set_option exponentiation.threshold 300 in
/-- **The tree-coin hybrid at hm96-sha512.** -/
theorem treeCoins_le_model {E₂ I Blk Cs D α : Type} [Field E₂] [Fintype E₂] [DecidableEq E₂]
    (hE : Fintype.card E₂ = 2) [Fintype I] [DecidableEq I] [Fintype Blk] [Nonempty Blk] [DecidableEq Blk]
    [DecidableEq Cs] {g : TGame I Blk α} {G : Game α} (hgG : ErD g G) (hg : g.Distinct)
    (dig : I → Blk → (Fin 512 → E₂)) (cs : Salt E₂ → Cs) (hcs : (Finset.univ.image cs).card ≤ 2 ^ 512)
    (H : Key E₂ → (Fin 512 → E₂) × Cs → D) (τ : Key E₂ → (I → D) → TGame.CStrat (Salt E₂) g) :
    ∃ M : Mix G (Key E₂ × (I → D)), ∀ E : Key E₂ × (I → D) → α → Prop,
      avg (fun κ : Key E₂ => avg fun ω : I → Blk × Salt E₂ =>
        TGame.pr (E (κ, fun i => leaf dig (CROnly.hankel (n := 512) (l := 1536) κ) cs (H κ) i (ω i))) g
          (τ κ fun i => leaf dig (CROnly.hankel (n := 512) (l := 1536) κ) cs (H κ) i (ω i)) ω) ≤
        M.pr E + Fintype.card I * ENNReal.ofReal (1 / 2 ^ 256) := by
  have hT : ∀ κ : Key E₂, Hm96TwoVector (CROnly.hankel (n := 512) (l := 1536) κ) cs
      (ENNReal.ofReal (CROnly.hidingGap (CROnly.hankel (n := 512) (l := 1536) κ) cs) +
        ENNReal.ofReal (CROnly.hidingGap (CROnly.hankel (n := 512) (l := 1536) κ) cs)) :=
    fun κ => twoVector_of_hiding _ _ (CROnly.hm96Hiding_gap _ _)
  choose Mk hMk using fun κ => treeCoins_le_model_of hgG hg dig (CROnly.hankel (n := 512) (l := 1536) κ) cs (H κ)
    (hT κ) (τ κ)
  refine ⟨Mix.avgOver Mk, fun E => ?_⟩
  have h257 := CROnly.avg_ofReal_le _ (fun κ => CROnly.hidingGap_nonneg _ _) (CROnly.hm96_sha512_sum hE cs hcs)
  have hδ : avg (fun κ : Key E₂ => ENNReal.ofReal (CROnly.hidingGap (CROnly.hankel (n := 512) (l := 1536) κ) cs) +
      ENNReal.ofReal (CROnly.hidingGap (CROnly.hankel (n := 512) (l := 1536) κ) cs)) ≤
      ENNReal.ofReal (1 / 2 ^ 256) := by
    rw [avg_add]
    calc _ ≤ ENNReal.ofReal (1 / 2 ^ 257) + ENNReal.ofReal (1 / 2 ^ 257) := add_le_add h257 h257
      _ = ENNReal.ofReal (1 / 2 ^ 256) := by
          rw [← ENNReal.ofReal_add (by norm_num) (by norm_num)]
          norm_num
  rw [Mix.avgOver_pr]
  calc _ ≤ avg (fun κ : Key E₂ => (Mk κ).pr (fun v => E (κ, v)) + Fintype.card I *
          (ENNReal.ofReal (CROnly.hidingGap (CROnly.hankel (n := 512) (l := 1536) κ) cs) +
            ENNReal.ofReal (CROnly.hidingGap (CROnly.hankel (n := 512) (l := 1536) κ) cs))) :=
        avg_mono fun κ => hMk κ fun v => E (κ, v)
    _ = avg (fun κ => (Mk κ).pr fun v => E (κ, v)) + Fintype.card I * avg (fun κ : Key E₂ =>
          ENNReal.ofReal (CROnly.hidingGap (CROnly.hankel (n := 512) (l := 1536) κ) cs) +
            ENNReal.ofReal (CROnly.hidingGap (CROnly.hankel (n := 512) (l := 1536) κ) cs)) := by
        rw [avg_add, avg_const_mul]
    _ ≤ _ := add_le_add le_rfl (mul_le_mul_right hδ _)

end FlockSoundness.Discharge.ZkTree
