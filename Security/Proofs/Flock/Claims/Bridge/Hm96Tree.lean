import Proofs.Flock.Claims.Bridge.Committed
import Proofs.Flock.Verifier.Encode

/-!
# The coin tree's hiding from `Hm96Hiding`

`Committed.lean` bounds the committed-coins run by the coin tree's hiding distance `δ_hide` (`SimHides`, `TreeHides`).
This file derives it from the claims' hiding assumption `Hm96Hiding` (one hm96 leaf: `(Φy, c)` within `δ₁` of
`(U, c)` on every event), by a hybrid over the tree's leaves, at `δ_hide = 2 k δ₁` for a tree of `k` rounds' leaves.

* `hiding_abs`: one leaf hides two-sided. `Hm96Hiding` bounds each event one way; applied to its complement it bounds it
  the other, so `|Pr[E(Φy, c)] − Pr[E(U, c)]| ≤ δ₁`, and two leaves of different digests are within `2 δ₁`
  (`hiding_pair`).
* `exists_match`: for any `g h : α → β`, a permutation `σ` of `α` with `h ∘ σ = g` off a set whose probability is some
  event's advantage at telling `g`'s values from `h`'s (fibre by fibre, the first `min(|g⁻¹b|, |h⁻¹b|)` points matched).
* `Hm96Tree`: round `j`'s leaf is the hm96 leaf of its coins' digest under a uniform salt, the root reads the leaves and
  the tree's other randomness, round `j`'s opening its coin, its salt and the leaves. `Hm96Tree.sim` is the simulator:
  a root of fake coins under fake salts, and round `j`'s opening at the real coin with its fake salt rematched (`rematch`:
  `exists_match` between the leaf at the fake coin and at the real one).
* `Hm96Tree.simHides`, `Hm96Tree.treeHides`: the tree hides at `2 k δ₁` against `sim`. The rematched salts are uniform,
  and round `j`'s leaf at the real coin and rematched salt is the simulator's at the fake one but with probability `2 δ₁`.
  At the deployed tree (`coin-tree/hm96-sha512/v1`: at most `3 · 2^8` leaves) and `δ₁ = 2^-193`, `δ_hide < 2^-182`.
* `Pr_runC_le_hm96`: `Pr_runC_le` with `δ_hide = 2 · k(x) · δ₁`, so the named parameter is `Hm96Hiding`'s `δ₁`, the
  assumption the claims list; `Pr_runC_le_deploy` is it at `Deploy.V`, whose answer `Coin.bytes` determines its coin
  (`coin_bytes_injective`).
-/

namespace Claims.Bridge

open FlockSoundness.Game (ind)
open scoped ENNReal

/-! ## Uniform probabilities -/

theorem Pr_eq_card {α : Type} [Fintype α] (p : α → Prop) [DecidablePred p] :
    Pr p = (Fintype.card {a // p a} : ℝ) / Fintype.card α := by
  unfold Pr
  rw [Fintype.card_subtype]
  congr 3
  exact Finset.filter_congr_decidable _ _ _

theorem Pr_equiv {α β : Type} [Fintype α] [Fintype β] (e : α ≃ β) (p : β → Prop) :
    Pr (fun a => p (e a)) = Pr p := by
  rw [Pr_eq_Ex, Pr_eq_Ex]
  exact Ex_equiv e (ind p)

theorem Pr_prod {α β : Type} [Fintype α] [Fintype β] (P : α → β → Prop) :
    Pr (fun p : α × β => P p.1 p.2) = Ex fun a => Pr (P a) := by
  rw [Pr_eq_Ex, Ex_prod]
  simp only [Pr_eq_Ex]
  rfl

theorem Pr_le_of_forall {α β : Type} [Fintype α] [Nonempty α] [Fintype β] (P : α → β → Prop) {ε : ℝ}
    (h : ∀ a, Pr (P a) ≤ ε) : Pr (fun p : α × β => P p.1 p.2) ≤ ε := by
  rw [Pr_prod]
  calc _ ≤ Ex fun _ : α => ε := Ex_mono h
    _ = ε := Ex_const (α := α) ε

theorem Pr_not {α : Type} [Fintype α] [Nonempty α] (p : α → Prop) : Pr (fun a => ¬ p a) = 1 - Pr p := by
  have := Classical.decPred p
  rw [Pr_eq_card, Pr_eq_card, Fintype.card_subtype_compl]
  have : (0 : ℝ) < Fintype.card α := by exact_mod_cast Fintype.card_pos
  rw [Nat.cast_sub (Fintype.card_subtype_le p)]
  field_simp

theorem Pr_or_le {α : Type} [Fintype α] (p q : α → Prop) : Pr (fun a => p a ∨ q a) ≤ Pr p + Pr q := by
  rw [Pr_eq_Ex, Pr_eq_Ex, Pr_eq_Ex, ← Ex_add]
  refine Ex_mono fun a => ?_
  unfold ind
  split_ifs <;> simp_all

theorem Ex_sum {α : Type} [Fintype α] {k : ℕ} (f : Fin k → α → ℝ) :
    (Ex fun a => ∑ j, f j a) = ∑ j, Ex (f j) := by
  unfold Ex
  rw [Finset.sum_comm, Finset.sum_div]

theorem Pr_exists_le {α : Type} [Fintype α] {k : ℕ} (P : Fin k → α → Prop) :
    Pr (fun a => ∃ j, P j a) ≤ ∑ j, Pr (P j) := by
  simp only [Pr_eq_Ex]
  rw [← Ex_sum]
  refine Ex_mono fun a => ?_
  by_cases h : ∃ j, P j a
  · obtain ⟨j, hj⟩ := h
    have h' : ∃ j, P j a := ⟨j, hj⟩
    calc ind (fun a => ∃ j, P j a) a = ind (P j) a := by unfold ind; simp [hj, h']
      _ ≤ ∑ j, ind (P j) a :=
        Finset.single_le_sum (f := fun j => ind (P j) a) (fun i _ => by unfold ind; split_ifs <;> norm_num)
          (Finset.mem_univ j)
  · calc ind (fun a => ∃ j, P j a) a = 0 := by unfold ind; simp [h]
      _ ≤ ∑ j, ind (P j) a := Finset.sum_nonneg fun i _ => by unfold ind; split_ifs <;> norm_num

/-- Two events that agree off `D` have probabilities within `Pr D`. -/
theorem abs_Pr_sub_le {α : Type} [Fintype α] (A B D : α → Prop) (h : ∀ a, ¬ D a → (A a ↔ B a)) :
    |Pr A - Pr B| ≤ Pr D := by
  have h1 : Pr A ≤ Pr B + Pr D :=
    (Pr_mono fun a ha => by by_cases hd : D a; exacts [Or.inr hd, Or.inl ((h a hd).1 ha)]).trans (Pr_or_le B D)
  have h2 : Pr B ≤ Pr A + Pr D :=
    (Pr_mono fun a hb => by by_cases hd : D a; exacts [Or.inr hd, Or.inl ((h a hd).2 hb)]).trans (Pr_or_le A D)
  rw [abs_le]
  constructor <;> linarith

/-- The coordinate `j` of a uniform function is uniform. -/
theorem Pr_coord {k : ℕ} {Y : Type} [Fintype Y] [Nonempty Y] (j : Fin k) (P : Y → Prop) :
    Pr (fun f : Fin k → Y => P (f j)) = Pr P := by
  rw [← Pr_equiv (Equiv.piSplitAt j fun _ => Y).symm, ← Pr_fst (β := {i : Fin k // i ≠ j} → Y) P]
  congr 1
  funext a
  simp [Equiv.piSplitAt_symm_apply]

/-! ## `PrENNReal` as `Pr` -/

open Classical in
theorem PrENNReal_eq {α : Type} [Fintype α] (p : α → Prop) :
    PrENNReal p = (Fintype.card α : ℝ≥0∞)⁻¹ * ((Finset.univ.filter p).card : ℝ≥0∞) := by
  classical
  unfold PrENNReal
  rw [← Finset.mul_sum, Finset.sum_boole]

theorem PrENNReal_toReal {α : Type} [Fintype α] (p : α → Prop) : (PrENNReal p).toReal = Pr p := by
  classical
  rw [PrENNReal_eq, ENNReal.toReal_mul, ENNReal.toReal_inv, ENNReal.toReal_natCast, ENNReal.toReal_natCast, Pr,
    div_eq_inv_mul]

theorem PrENNReal_ne_top {α : Type} [Fintype α] (p : α → Prop) : PrENNReal p ≠ ⊤ := by
  classical
  rw [PrENNReal_eq]
  rcases isEmpty_or_nonempty α with hα | hα
  · simp
  · exact ENNReal.mul_ne_top (ENNReal.inv_ne_top.2 (by exact_mod_cast Fintype.card_ne_zero)) (ENNReal.natCast_ne_top _)

/-! ## The matching -/

section Match

variable {α β : Type} [Fintype α] [DecidableEq β] (g h : α → β)

/-- `α` as each value's fiber under `g`, indexed. -/
noncomputable def fibers : α ≃ Σ b, Fin (Fintype.card {a // g a = b}) :=
  (Equiv.sigmaFiberEquiv g).symm.trans (Equiv.sigmaCongrRight fun _ => Fintype.equivFin _)

theorem fibers_fst (a : α) : (fibers g a).1 = g a := rfl

theorem fibers_symm (x : Σ b, Fin (Fintype.card {a // g a = b})) : g ((fibers g).symm x) = x.1 :=
  ((Fintype.equivFin {a // g a = x.1}).symm x.2).2

/-- The matched indices: the first `min (G b) (H b)` of each fiber. -/
def matched : Type := {x : Σ b, Fin (Fintype.card {a // g a = b}) // x.2.val < Fintype.card {a // h a = x.1}}

/-- The matched indices of `g`'s fibers are those of `h`'s. -/
def matchedEquiv : matched g h ≃ matched h g where
  toFun x := ⟨⟨x.1.1, ⟨x.1.2.val, x.2⟩⟩, x.1.2.isLt⟩
  invFun y := ⟨⟨y.1.1, ⟨y.1.2.val, y.2⟩⟩, y.1.2.isLt⟩
  left_inv _ := rfl
  right_inv _ := rfl

/-- `a` is matched: its index in its fiber under `g` is below the size of the same value's fiber under `h`. -/
def InMatch (a : α) : Prop := ((fibers g a).2 : ℕ) < Fintype.card {a' // h a' = (fibers g a).1}

noncomputable instance instDecidablePredInMatch : DecidablePred (InMatch g h) := fun _ => Nat.decLt _ _

/-- The matched points of `α` under `g`, mapped to those under `h`. -/
noncomputable def matchSub : {a // InMatch g h a} ≃ {a // InMatch h g a} :=
  ((fibers g).subtypeEquiv fun _ => Iff.rfl).trans
    ((matchedEquiv g h).trans ((fibers h).subtypeEquiv fun _ => Iff.rfl).symm)

theorem matchSub_val (a : α) (ha : InMatch g h a) :
    h (matchSub g h ⟨a, ha⟩).val = g a :=
  fibers_symm h _

/-- The indices of `Fin n` from `m` on, as `Fin (n - m)`. -/
def finFrom (n m : ℕ) : {i : Fin n // ¬ i.val < m} ≃ Fin (n - m) where
  toFun i := ⟨i.1.val - m, by have := i.1.isLt; have := i.2; omega⟩
  invFun j := ⟨⟨j.val + m, by have := j.isLt; omega⟩, by simp⟩
  left_inv i := by ext; simp only; have := i.2; omega
  right_inv j := by ext; simp

theorem card_fin_not_lt (n m : ℕ) : Fintype.card {i : Fin n // ¬ i.val < m} = n - m := by
  rw [Fintype.card_congr (finFrom n m)]
  exact Fintype.card_fin _

/-- The unmatched indices, fiber by fiber. -/
def unmatchedSigma :
    {x : Σ b, Fin (Fintype.card {a // g a = b}) // ¬ (x.2 : ℕ) < Fintype.card {a' // h a' = x.1}} ≃
      Σ b, {i : Fin (Fintype.card {a // g a = b}) // ¬ i.val < Fintype.card {a // h a = b}} where
  toFun x := ⟨x.1.1, x.1.2, x.2⟩
  invFun y := ⟨⟨y.1, y.2.1⟩, y.2.2⟩
  left_inv _ := rfl
  right_inv _ := rfl

theorem card_unmatched [Fintype β] :
    Fintype.card {a // ¬ InMatch g h a} =
      ∑ b, (Fintype.card {a // g a = b} - Fintype.card {a // h a = b}) := by
  rw [Fintype.card_congr ((fibers g).subtypeEquiv (p := fun a => ¬ InMatch g h a) (q := fun x => ¬ (x.2 : ℕ) < Fintype.card {a' // h a' = x.1})
    fun _ => Iff.rfl)]
  rw [Fintype.card_congr (unmatchedSigma g h), Fintype.card_sigma]
  exact Finset.sum_congr rfl fun b _ => card_fin_not_lt _ _

open Classical in
theorem Pr_comp_eq [Fintype β] (E : β → Prop) :
    Pr (fun a => E (g a)) = (∑ b ∈ Finset.univ.filter E, (Fintype.card {a // g a = b} : ℝ)) / Fintype.card α := by
  unfold Pr
  congr 1
  rw [Finset.card_eq_sum_card_fiberwise (f := g) (t := Finset.univ.filter E) (fun a ha => by simpa using ha)]
  push_cast
  refine Finset.sum_congr rfl fun b hb => ?_
  rw [Fintype.card_subtype, Finset.filter_filter]
  congr 2
  refine Finset.filter_congr fun a _ => ?_
  simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hb
  exact ⟨fun h => h.2, fun h => ⟨h ▸ hb, h⟩⟩

/-- The event that the value `b` is no likelier under `h` than under `g`. -/
def NoLikelier (b : β) : Prop := Fintype.card {a // h a = b} ≤ Fintype.card {a // g a = b}

/-- **The matching**: some permutation `σ` of `α` has `h ∘ σ = g` off a set whose probability is the advantage of
`NoLikelier g h` at telling `g`'s values from `h`'s. -/
theorem exists_match_fin [Fintype β] :
    ∃ σ : α ≃ α, Pr (fun a => h (σ a) ≠ g a) ≤
      Pr (fun a => NoLikelier g h (g a)) - Pr (fun a => NoLikelier g h (h a)) := by
  classical
  refine ⟨Equiv.extendSubtype (p := InMatch g h) (q := InMatch h g) (matchSub g h), ?_⟩
  have hm : ∀ a, h (Equiv.extendSubtype (p := InMatch g h) (q := InMatch h g) (matchSub g h) a) ≠ g a →
      ¬ InMatch g h a := by
    intro a hne ha
    rw [Equiv.extendSubtype_apply_of_mem _ _ ha, matchSub_val g h a ha] at hne
    exact hne rfl
  refine (Pr_mono hm).trans ?_
  rw [Pr_eq_card, card_unmatched, Pr_comp_eq g, Pr_comp_eq h, ← sub_div, ← Finset.sum_sub_distrib]
  refine div_le_div_of_nonneg_right (le_of_eq ?_) (by positivity)
  rw [Nat.cast_sum, ← Finset.sum_filter_add_sum_filter_not Finset.univ (NoLikelier g h)]
  rw [Finset.sum_eq_zero (s := Finset.univ.filter fun b => ¬ NoLikelier g h b) fun b hb => by
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, NoLikelier, not_le] at hb
    rw [Nat.sub_eq_zero_of_le hb.le, Nat.cast_zero], add_zero]
  refine Finset.sum_congr rfl fun b hb => ?_
  simp only [Finset.mem_filter, Finset.mem_univ, true_and, NoLikelier] at hb
  rw [Nat.cast_sub hb]

end Match

/-- **The matching**, at values of any type: some permutation `σ` of `α` has `h ∘ σ = g` off a set whose probability
is some event's advantage at telling `g`'s values from `h`'s. -/
theorem exists_match {α β : Type} [Fintype α] (g h : α → β) :
    ∃ σ : α ≃ α, ∃ E : β → Prop, Pr (fun a => h (σ a) ≠ g a) ≤ Pr (fun a => E (g a)) - Pr (fun a => E (h a)) := by
  classical
  let S : Finset β := Finset.univ.image g ∪ Finset.univ.image h
  let g' : α → S := fun a => ⟨g a, Finset.mem_union_left _ (Finset.mem_image_of_mem g (Finset.mem_univ a))⟩
  let h' : α → S := fun a => ⟨h a, Finset.mem_union_right _ (Finset.mem_image_of_mem h (Finset.mem_univ a))⟩
  obtain ⟨σ, hσ⟩ := exists_match_fin g' h'
  refine ⟨σ, fun b => ∃ hb : b ∈ S, NoLikelier g' h' ⟨b, hb⟩, ?_⟩
  have e1 : ∀ a, h' (σ a) ≠ g' a ↔ h (σ a) ≠ g a := fun a => by simp [g', h', Subtype.ext_iff]
  have e2 : ∀ a, NoLikelier g' h' (g' a) ↔ ∃ hb : g a ∈ S, NoLikelier g' h' ⟨g a, hb⟩ :=
    fun a => ⟨fun h => ⟨_, h⟩, fun ⟨_, h⟩ => h⟩
  have e3 : ∀ a, NoLikelier g' h' (h' a) ↔ ∃ hb : h a ∈ S, NoLikelier g' h' ⟨h a, hb⟩ :=
    fun a => ⟨fun h => ⟨_, h⟩, fun ⟨_, h⟩ => h⟩
  rwa [Pr_congr e1, Pr_congr e2, Pr_congr e3] at hσ

/-! ## One leaf's hiding, two-sided -/

section Leaf

variable {Y B C : Type} [Fintype Y] [Fintype B] {My : Y → B} {c : Y → C} {δ₁ : ℝ≥0∞}

theorem hiding_toReal (h : Hm96Hiding My c δ₁) (hδ : δ₁ ≠ ⊤) (E : B × C → Prop) :
    Pr (fun y => E (My y, c y)) ≤ Pr (fun p : B × Y => E (p.1, c p.2)) + δ₁.toReal := by
  have := ENNReal.toReal_mono (ENNReal.add_ne_top.2 ⟨PrENNReal_ne_top _, hδ⟩) (h E)
  rwa [ENNReal.toReal_add (PrENNReal_ne_top _) hδ, PrENNReal_toReal, PrENNReal_toReal] at this

/-- **One leaf hides, two-sided**: `Hm96Hiding` bounds each event one way; its complement bounds it the other. -/
theorem hiding_abs [Nonempty Y] [Nonempty B] (h : Hm96Hiding My c δ₁) (hδ : δ₁ ≠ ⊤) (E : B × C → Prop) :
    |Pr (fun y => E (My y, c y)) - Pr (fun p : B × Y => E (p.1, c p.2))| ≤ δ₁.toReal := by
  have h1 := hiding_toReal h hδ E
  have h2 := hiding_toReal h hδ fun z => ¬ E z
  have n1 := Pr_not fun y => E (My y, c y)
  have n2 := Pr_not fun p : B × Y => E (p.1, c p.2)
  rw [abs_le]
  constructor <;> linarith

/-- A leaf of the digest `d`, `(d + My y, c y)`, against `(U, c y)`. -/
theorem hiding_shift [AddGroup B] [Nonempty Y] (h : Hm96Hiding My c δ₁) (hδ : δ₁ ≠ ⊤) (d : B)
    (E : B × C → Prop) :
    |Pr (fun y => E (d + My y, c y)) - Pr (fun p : B × Y => E (p.1, c p.2))| ≤ δ₁.toReal := by
  have e : Pr (fun p : B × Y => E (d + p.1, c p.2)) = Pr (fun p : B × Y => E (p.1, c p.2)) :=
    Pr_equiv ((Equiv.addLeft d).prodCongr (Equiv.refl Y)) fun p : B × Y => E (p.1, c p.2)
  rw [← e]
  exact hiding_abs h hδ fun z => E (d + z.1, z.2)

/-- The leaves of two digests are within `2 δ₁` on every event. -/
theorem hiding_pair [AddGroup B] [Nonempty Y] (h : Hm96Hiding My c δ₁) (hδ : δ₁ ≠ ⊤) (d d' : B)
    (E : B × C → Prop) :
    |Pr (fun y => E (d + My y, c y)) - Pr (fun y => E (d' + My y, c y))| ≤ 2 * δ₁.toReal := by
  have h1 := hiding_shift h hδ d E
  have h2 := hiding_shift h hδ d' E
  rw [abs_le] at h1 h2 ⊢
  constructor <;> linarith

end Leaf

/-! ## The hm96 coin tree -/

/-- **An hm96 coin tree** over `k` rounds (live `PROTOCOL.md` §9, `coin-tree/hm96-sha512`): round `j`'s leaf is the hm96
leaf of its coins' digest `dig j c` under a salt `y`, the pair `(b, c) = (dig j c + My y, cs y)` (`b = x ⊕ Φy`,
`c = H(p_salt ‖ y)`) whose hash `H(p_leaf ‖ b ‖ c)` the root reads; `root` is the root over the rounds' leaves and the
rest `t` of the tree's randomness (whatever no round opens: the leaves of rounds the run does not reach, the padding);
`opening` is round `j`'s opening, of its coin and its salt, with its path over the leaves. -/
structure Hm96Tree (Coin B Y C T₀ : Type) (k : ℕ) where
  dig : Fin k → Coin → B
  My : Y → B
  cs : Y → C
  root : (Fin k → B × C) → T₀ → Msg
  opening : (Fin k → B × C) → T₀ → Fin k → Coin → Y → Msg

namespace Hm96Tree

variable {Coin B Y C T₀ : Type} {k : ℕ} (T : Hm96Tree Coin B Y C T₀ k)

/-- Round `j`'s leaf at the coin `c` and the salt `y`. -/
def leaf [Add B] (j : Fin k) (c : Coin) (y : Y) : B × C := (T.dig j c + T.My y, T.cs y)

/-- The rounds' leaves at the coins `r` and the salts `y`. -/
def leaves [Add B] (r : Fin k → Coin) (y : Fin k → Y) : Fin k → B × C := fun j => T.leaf j (r j) (y j)

/-- **The tree as a `CoinTree`**: its randomness is the rounds' salts and the rest `t`. -/
def tree [Add B] : CoinTree Coin ((Fin k → Y) × T₀) k where
  hello r τ := T.root (T.leaves r τ.1) τ.2
  opn r τ j := T.opening (T.leaves r τ.1) τ.2 j (r j) (τ.1 j)

/-- **The rematch at leaf `j`**: a permutation of the salts that sends a salt's leaf at the coin `u` to the same leaf
at the coin `c`, but on few salts (`exists_match`). -/
noncomputable def rematch [Add B] [Fintype Y] (j : Fin k) (u c : Coin) : Y ≃ Y :=
  (exists_match (T.leaf j u) (T.leaf j c)).choose

/-- The rematch moves a leaf on at most `2 δ₁` of the salts. -/
theorem rematch_le [AddGroup B] [Fintype Y] [Nonempty Y] [Fintype B] {δ₁ : ℝ≥0∞} (hH : Hm96Hiding T.My T.cs δ₁)
    (hδ : δ₁ ≠ ⊤) (j : Fin k) (u c : Coin) :
    Pr (fun y => T.leaf j c (T.rematch j u c y) ≠ T.leaf j u y) ≤ 2 * δ₁.toReal := by
  obtain ⟨E, hE⟩ := (exists_match (T.leaf j u) (T.leaf j c)).choose_spec
  exact hE.trans ((le_abs_self _).trans (hiding_pair hH hδ (T.dig j u) (T.dig j c) E))

/-- Round `j`'s leaf at the coin `r j` and the salt rematched from `r' j` is the leaf at `r' j` and the salt but with
probability `2 δ₁`. -/
theorem Pr_mismatch_le [AddGroup B] [Fintype B] [Fintype Coin] [Nonempty Coin] [Fintype Y] [Nonempty Y]
    [Fintype T₀] [Nonempty T₀] {δ₁ : ℝ≥0∞} (hH : Hm96Hiding T.My T.cs δ₁) (hδ : δ₁ ≠ ⊤) (j : Fin k) :
    Pr (fun p : (Fin k → Coin) × (Fin k → Coin) × (Fin k → Y) × T₀ =>
      T.leaf j (p.1 j) (T.rematch j (p.2.1 j) (p.1 j) (p.2.2.1 j)) ≠ T.leaf j (p.2.1 j) (p.2.2.1 j)) ≤
        2 * δ₁.toReal := by
  rw [Pr_prod (fun (r : Fin k → Coin) (σ : (Fin k → Coin) × (Fin k → Y) × T₀) =>
    T.leaf j (r j) (T.rematch j (σ.1 j) (r j) (σ.2.1 j)) ≠ T.leaf j (σ.1 j) (σ.2.1 j))]
  refine (Ex_mono (g := fun _ => 2 * δ₁.toReal) fun r => ?_).trans (Ex_const _).le
  rw [Pr_prod (fun (r' : Fin k → Coin) (q : (Fin k → Y) × T₀) =>
    T.leaf j (r j) (T.rematch j (r' j) (r j) (q.1 j)) ≠ T.leaf j (r' j) (q.1 j))]
  refine (Ex_mono (g := fun _ => 2 * δ₁.toReal) fun r' => ?_).trans (Ex_const _).le
  rw [Pr_fst (fun y : Fin k → Y => T.leaf j (r j) (T.rematch j (r' j) (r j) (y j)) ≠ T.leaf j (r' j) (y j)),
    Pr_coord j fun y => T.leaf j (r j) (T.rematch j (r' j) (r j) y) ≠ T.leaf j (r' j) y]
  exact T.rematch_le hH hδ j (r' j) (r j)

/-- **The simulator**: its tape is fake coins `r'`, salts `y'` and the rest `t`. Its root is the tree's at `r'`, `y'`
and `t`; round `j`'s opening, once the answer to round `j` gives the coin `c`, is the tree's at `c` with the salt
rematched from `r' j` to `c`. -/
noncomputable def sim [Add B] [Fintype Coin] [Nonempty Coin] [Fintype Y] [Nonempty Y] [Fintype T₀] [Nonempty T₀]
    (dec : Msg → Coin) : OnlineSim where
  Tape := (Fin k → Coin) × (Fin k → Y) × T₀
  fin := inferInstance
  ne := inferInstance
  hello σ := T.root (T.leaves σ.1 σ.2.1) σ.2.2
  opn σ as :=
    if h : as ≠ [] ∧ as.length - 1 < k then
      T.opening (T.leaves σ.1 σ.2.1) σ.2.2 ⟨as.length - 1, h.2⟩ (dec (as.getLast h.1))
        (T.rematch ⟨as.length - 1, h.2⟩ (σ.1 ⟨as.length - 1, h.2⟩) (dec (as.getLast h.1)) (σ.2.1 ⟨as.length - 1, h.2⟩))
    else []

theorem sim_opn [Add B] [Fintype Coin] [Nonempty Coin] [Fintype Y] [Nonempty Y] [Fintype T₀] [Nonempty T₀]
    (dec : Msg → Coin) (σ : (T.sim dec).Tape) (as : List Msg) (j : Fin k) (hne : as ≠ [])
    (hj : as.length - 1 = j.val) :
    (T.sim dec).opn σ as = T.opening (T.leaves σ.1 σ.2.1) σ.2.2 j (dec (as.getLast hne))
      (T.rematch j (σ.1 j) (dec (as.getLast hne)) (σ.2.1 j)) := by
  have h : as ≠ [] ∧ as.length - 1 < k := ⟨hne, hj ▸ j.isLt⟩
  have hj' : (⟨as.length - 1, h.2⟩ : Fin k) = j := Fin.ext hj
  simp only [sim]
  split
  · rw [hj']
  · contradiction

/-- The simulator's openings against the coins `r`: round `j`'s is the tree's at `r j`, with the salt rematched. -/
theorem sim_opnSeq [Add B] [Fintype Coin] [Inhabited Coin] [Fintype Y] [Nonempty Y] [Fintype T₀] [Nonempty T₀]
    (ans : Coin → Msg) (dec : Msg → Coin) (hdec : ∀ c, dec (ans c) = c) (σ : (T.sim dec).Tape)
    (r : Fin k → Coin) (j : Fin k) :
    (T.sim dec).opnSeq ans σ r j =
      T.opening (T.leaves σ.1 σ.2.1) σ.2.2 j (r j) (T.rematch j (σ.1 j) (r j) (σ.2.1 j)) := by
  have hne : (List.range (j.val + 1)).map (ans ∘ coinSeq r) ≠ [] := by simp
  have hlast : ((List.range (j.val + 1)).map (ans ∘ coinSeq r)).getLast hne = ans (r j) := by
    rw [List.getLast_map, List.getLast_range]
    simp [coinSeq, j.isLt]
  unfold OnlineSim.opnSeq
  rw [T.sim_opn dec σ _ j hne (by simp), hlast, hdec]

/-- **The hm96 coin tree hides at distance `2 k δ₁`** against its simulator, where each leaf hides at `δ₁`
(`Hm96Hiding`) and the answers determine their coins. The hybrid runs over the leaves: draw the simulator's tape
`(r', y', t)`; the tree's salts `y_j = rematch j (r' j) (r j) (y' j)` are uniform for every `r'`, and round `j`'s leaf
at `(r j, y_j)` is the simulator's at `(r' j, y' j)` but with probability `2 δ₁` (`rematch_le`: `Hm96Hiding` one way,
its complement the other), so the two root-and-openings differ with probability at most `2 k δ₁`. -/
theorem simHides [AddGroup B] [Fintype B] [Fintype Coin] [Inhabited Coin] [Fintype Y] [Nonempty Y] [Fintype T₀]
    [Nonempty T₀] {δ₁ : ℝ≥0∞} (hH : Hm96Hiding T.My T.cs δ₁) (hδ : δ₁ ≠ ⊤) (ans : Coin → Msg) (dec : Msg → Coin)
    (hdec : ∀ c, dec (ans c) = c) :
    SimHides T.tree ans (T.sim dec) (2 * k * δ₁.toReal) := by
  intro Ev
  let Ψ : (Fin k → Coin) → (Fin k → Coin) → (Fin k → Y) ≃ (Fin k → Y) :=
    fun r r' => Equiv.piCongrRight fun j => T.rematch j (r' j) (r j)
  let F : (Fin k → Coin) → (Fin k → Y) → T₀ → Prop :=
    fun r y t => Ev r (T.root (T.leaves r y) t) fun j => T.opening (T.leaves r y) t j (r j) (y j)
  have hReal : Pr (fun p : (Fin k → Coin) × (Fin k → Y) × T₀ => Ev p.1 (T.tree.hello p.1 p.2) (T.tree.opn p.1 p.2)) =
      Pr (fun p : (Fin k → Coin) × (Fin k → Coin) × (Fin k → Y) × T₀ => F p.1 (Ψ p.1 p.2.1 p.2.2.1) p.2.2.2) := by
    show Pr (fun p : (Fin k → Coin) × (Fin k → Y) × T₀ => F p.1 p.2.1 p.2.2) = _
    rw [Pr_prod (fun r (τ : (Fin k → Y) × T₀) => F r τ.1 τ.2),
      Pr_prod (fun r (σ : (Fin k → Coin) × (Fin k → Y) × T₀) => F r (Ψ r σ.1 σ.2.1) σ.2.2)]
    congr 1
    funext r
    rw [Pr_prod (fun r' (q : (Fin k → Y) × T₀) => F r (Ψ r r' q.1) q.2)]
    have e : ∀ r', Pr (fun q : (Fin k → Y) × T₀ => F r (Ψ r r' q.1) q.2) =
        Pr (fun τ : (Fin k → Y) × T₀ => F r τ.1 τ.2) :=
      fun r' => Pr_equiv ((Ψ r r').prodCongr (Equiv.refl T₀)) fun τ => F r τ.1 τ.2
    simp only [e]
    rw [Ex_const]
  have hSim : Pr (fun p : (Fin k → Coin) × (T.sim dec).Tape =>
        Ev p.1 ((T.sim dec).hello p.2) ((T.sim dec).opnSeq ans p.2 p.1)) =
      Pr (fun p : (Fin k → Coin) × (Fin k → Coin) × (Fin k → Y) × T₀ => Ev p.1 (T.root (T.leaves p.2.1 p.2.2.1) p.2.2.2)
        fun j => T.opening (T.leaves p.2.1 p.2.2.1) p.2.2.2 j (p.1 j) (Ψ p.1 p.2.1 p.2.2.1 j)) := by
    congr 1
    funext p
    have : (T.sim dec).opnSeq ans p.2 p.1 =
        fun j => T.opening (T.leaves p.2.1 p.2.2.1) p.2.2.2 j (p.1 j) (Ψ p.1 p.2.1 p.2.2.1 j) :=
      funext fun j => T.sim_opnSeq ans dec hdec p.2 p.1 j
    rw [this]
    rfl
  rw [hReal, hSim]
  refine (abs_Pr_sub_le _ _ (fun p : (Fin k → Coin) × (Fin k → Coin) × (Fin k → Y) × T₀ =>
    ∃ j, T.leaves p.1 (Ψ p.1 p.2.1 p.2.2.1) j ≠ T.leaves p.2.1 p.2.2.1 j) fun p hp => ?_).trans ?_
  · push Not at hp
    have hL : T.leaves p.1 (Ψ p.1 p.2.1 p.2.2.1) = T.leaves p.2.1 p.2.2.1 := funext hp
    simp only [F]
    rw [hL]
  · refine (Pr_exists_le _).trans ?_
    calc ∑ j : Fin k, Pr (fun p : (Fin k → Coin) × (Fin k → Coin) × (Fin k → Y) × T₀ =>
          T.leaves p.1 (Ψ p.1 p.2.1 p.2.2.1) j ≠ T.leaves p.2.1 p.2.2.1 j)
        ≤ ∑ _j : Fin k, 2 * δ₁.toReal := Finset.sum_le_sum fun j _ => ?_
      _ = 2 * k * δ₁.toReal := by simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]; ring
    have hj : ∀ p : (Fin k → Coin) × (Fin k → Coin) × (Fin k → Y) × T₀,
        T.leaves p.1 (Ψ p.1 p.2.1 p.2.2.1) j ≠ T.leaves p.2.1 p.2.2.1 j ↔
          T.leaf j (p.1 j) (T.rematch j (p.2.1 j) (p.1 j) (p.2.2.1 j)) ≠ T.leaf j (p.2.1 j) (p.2.2.1 j) :=
      fun p => Iff.rfl
    rw [Pr_congr hj]
    exact T.Pr_mismatch_le hH hδ j

/-- **`TreeHides` from `Hm96Hiding`**: the hm96 coin tree's hiding distance is at most `2 k δ₁`, `k` its rounds'
leaves, where each leaf hides at `δ₁ ≠ ∞` and the verifier's answer determines its coin. -/
theorem treeHides [AddGroup B] [Fintype B] [Fintype Coin] [Inhabited Coin] [Fintype Y] [Nonempty Y] [Fintype T₀]
    [Nonempty T₀] {δ₁ : ℝ≥0∞} (hH : Hm96Hiding T.My T.cs δ₁) (hδ : δ₁ ≠ ⊤) (ans : Coin → Msg)
    (hans : Function.Injective ans) :
    TreeHides T.tree ans (2 * k * δ₁.toReal) :=
  ⟨T.sim (Function.invFun ans), T.simHides hH hδ ans _ (Function.leftInverse_invFun hans)⟩

end Hm96Tree

/-- **The bound a KS proof charges in `κ`, from `Hm96Hiding`**: against a verifier that answers each round with its
coin, through an injective `ans`, and an hm96 coin tree whose leaves hide at `δ₁ ≠ ∞`, an event of the committed run is
at most `2 · k(x) · δ₁` likelier than at `Claims.run` of a prover that reads a simulator's root and openings. -/
theorem Pr_runC_le_hm96 {Coin Ω B Y C T₀ : Type} [Fintype Coin] [Inhabited Coin] [Fintype Ω] [Nonempty Ω]
    [AddGroup B] [Fintype B] [Fintype Y] [Nonempty Y] [Fintype T₀] [Nonempty T₀] (H : Msg → Msg) (V : Verifier Coin)
    (x : Msg) (T : Hm96Tree Coin B Y C T₀ (V.k x)) {δ₁ : ℝ≥0∞} (hH : Hm96Hiding T.My T.cs δ₁) (hδ : δ₁ ≠ ⊤)
    (ans : Coin → Msg) (hans : Function.Injective ans) (hV : AnswersCoin H V x ans) (P : CProver Ω) :
    ∃ S : OnlineSim, ∀ Ev : Ω → (Fin (V.k x) → Coin) → Trace → Prop,
      Pr (fun p : Ω × (Fin (V.k x) → Coin) × (Fin (V.k x) → Y) × T₀ =>
          Ev p.1 p.2.1 (runC H P V x T.tree p.1 p.2.1 p.2.2)) ≤
        Pr (fun p : (Ω × S.Tape) × (Fin (V.k x) → Coin) => Ev p.1.1 p.2 (run H (simProver S V.k P) V x p.1 p.2)) +
          2 * V.k x * δ₁.toReal :=
  Pr_runC_le_of_treeHides H V x T.tree ans (T.treeHides hH hδ ans hans) hV P

/-! ## At Flock's deployed verifier -/

theorem flatMap_eq_of_length {α β : Type} {n : ℕ} {f g : α → List β} (hf : ∀ a, (f a).length = n)
    (hg : ∀ a, (g a).length = n) : ∀ {l : List α}, l.flatMap f = l.flatMap g → ∀ a ∈ l, f a = g a
  | [], _, a, ha => absurd ha List.not_mem_nil
  | b :: l, h, a, ha => by
    rw [List.flatMap_cons, List.flatMap_cons] at h
    obtain ⟨hb, hl⟩ := List.append_inj h (by rw [hf, hg])
    rcases List.mem_cons.mp ha with rfl | ha
    · exact hb
    · exact flatMap_eq_of_length hf hg hl a ha

/-- The deployed answer determines its coin: every element is 16 bytes, in order. -/
theorem coin_bytes_injective : Function.Injective FlockDeploy.Coin.bytes := by
  intro c d h
  have hlen : ∀ e : FlockDeploy.Coin, ∀ i, ((e i).toBytes.data.toList).length = 16 := fun e i => by
    rw [Array.length_toList, ByteArray.size_data, Flock.F128.toBytes_size]
  funext i
  have hi := flatMap_eq_of_length (hlen c) (hlen d) h i (List.mem_finRange i)
  exact Flock.F128.toBytes_inj (ByteArray.ext (Array.ext' hi))

/-- **`Pr_runC_le_hm96` at `Deploy.V`**: the deployed verifier answers each round with its coin's bytes, which determine
the coin, so an hm96 coin tree whose leaves hide at `δ₁ ≠ ∞` costs a committed run at most `2 · k(x) · δ₁`. -/
theorem Pr_runC_le_deploy {Ω B Y C T₀ : Type} [Fintype Ω] [Nonempty Ω] [AddGroup B] [Fintype B] [Fintype Y]
    [Nonempty Y] [Fintype T₀] [Nonempty T₀] (zk : Bool) (H : Msg → Msg) (x : Msg)
    (T : Hm96Tree FlockDeploy.Coin B Y C T₀ ((FlockDeploy.V zk).k x)) {δ₁ : ℝ≥0∞} (hH : Hm96Hiding T.My T.cs δ₁)
    (hδ : δ₁ ≠ ⊤) (P : CProver Ω) :
    ∃ S : OnlineSim, ∀ Ev : Ω → (Fin ((FlockDeploy.V zk).k x) → FlockDeploy.Coin) → Trace → Prop,
      Pr (fun p : Ω × (Fin ((FlockDeploy.V zk).k x) → FlockDeploy.Coin) × (Fin ((FlockDeploy.V zk).k x) → Y) × T₀ =>
          Ev p.1 p.2.1 (runC H P (FlockDeploy.V zk) x T.tree p.1 p.2.1 p.2.2)) ≤
        Pr (fun p : (Ω × S.Tape) × (Fin ((FlockDeploy.V zk).k x) → FlockDeploy.Coin) =>
          Ev p.1.1 p.2 (run H (simProver S (FlockDeploy.V zk).k P) (FlockDeploy.V zk) x p.1 p.2)) +
          2 * (FlockDeploy.V zk).k x * δ₁.toReal :=
  Pr_runC_le_hm96 H (FlockDeploy.V zk) x T hH hδ FlockDeploy.Coin.bytes coin_bytes_injective
    (answersCoin_deploy zk H x) P

end Claims.Bridge
