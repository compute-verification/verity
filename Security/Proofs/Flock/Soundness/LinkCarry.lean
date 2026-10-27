import Proofs.Flock.Soundness.LinkFinder

/-!
# Two trials behind one leaf: the carried-leaf finder

A leaf carried across a wipe is registered in two sessions' public files, and each session's trial reads it. Every
trial of either session that opens the leaf opens the same commitment, so two readings with different values give a
collision whichever session each came from. The finder for the pair is `LinkFinder.finderG` on the mixture of the two
trials (`mixG`: a fair coin picks which one runs), with the two sessions' values embedded in one type `V`.

Both sessions read the carried leaf as the mixture's plurality `Xm`, pulled back into each one's values (`pull`), and
neither pays more than twice the mixture's link mass for it:

* `massAt_le_two_mix`: each trial's link mass at any `X` is at most twice the mixture's at `X`, where the link mass is
  `L(X) = s + Σ_{v ≠ X} p v` (`massAt`, `p v` reading `v` alone and `s` reading two values).
* `carry_massAt`: the same with each trial's readings embedded injectively in `V`, at `pull e_i Xm`; the two pulled
  values agree in `V` whenever `Xm` is a value of both sessions (`pull_spec`).
* `outMix_spec`: every success of the mixture's finder outputs a collision, from each session's own collisions and one
  across the two.
* `massAt_le_coll`: at the plurality, a trial's link mass is at most the probability that its finder outputs a
  collision, plus `1/(eM)`: `link_mass_le` without its collision-resistance hypothesis, the break form.
-/

namespace FlockSoundness

namespace LinkFinder

open Game Extract

section Mass

variable {τ Val : Type} [Fintype Val] [DecidableEq Val] (T : Game τ) (sT : Strategy T) (rd : τ → Finset Val)

/-- **A trial's link mass at `X`**: the probability that it reads two values, or one value other than `X`. -/
noncomputable def massAt (X : Val) : ℝ := sRd T sT rd + ((∑ v, pRd T sT rd v) - pRd T sT rd X)

theorem massAt_eq (X : Val) : massAt T sT rd X = rRd T sT rd - pRd T sT rd X := by
  rw [massAt, rRd_eq]
  ring

/-- **A collision of `Hc` in a finder's output.** -/
def Coll {Dh : Type} (Hc : List UInt8 → Dh) (o : Option (List UInt8 × List UInt8)) : Prop :=
  ∃ a b, o = some (a, b) ∧ a ≠ b ∧ Hc a = Hc b

/-- **The break form of the link mass**: at the plurality `X`, a trial's link mass is at most the probability that its
finder outputs a collision, plus `1/(eM)`, when every success of the finder outputs one. -/
theorem massAt_le_coll {Dh : Type} (Hc : List UInt8 → Dh) (out : τ × List τ → Option (List UInt8 × List UInt8))
    (M : ℕ) (hM : 1 ≤ M) (hout : ∀ x, Succ rd x → Coll Hc (out x)) {X : Val}
    (hX : ∀ v, pRd T sT rd v ≤ pRd T sT rd X) :
    massAt T sT rd X ≤
      (prob (fun x => Coll Hc (out x)) (finderG T M) (finderStrat T sT M)).toReal + 1 / (Real.exp 1 * M) := by
  have h1 := succ_ge T sT rd hX M hM
  have h2 : expect (ind (Succ rd)) (finderG T M) (finderStrat T sT M) ≤
      (prob (fun x => Coll Hc (out x)) (finderG T M) (finderStrat T sT M)).toReal := by
    rw [prob_toReal]
    exact expect_mono (ind_mono hout) _ _
  unfold massAt
  linarith

end Mass

/-! ## The mixture of two trials -/

section Mix

variable {τ₁ τ₂ : Type} (T₁ : Game τ₁) (s₁ : Strategy T₁) (T₂ : Game τ₂) (s₂ : Strategy T₂)

/-- The mixture's continuation after its coin. -/
def mixK : Bool → Game (τ₁ ⊕ τ₂)
  | true => T₁.map Sum.inl
  | false => T₂.map Sum.inr

/-- **The mixture of two trials**: a fair coin, then the first trial on `true` and the second on `false`. -/
def mixG : Game (τ₁ ⊕ τ₂) := Game.coin Bool (mixK T₁ T₂)

/-- The two trials' strategies in the mixture. -/
def mixStrat : Strategy (mixG T₁ T₂) := fun b => match b with
  | true => (Strategy.map Sum.inl T₁ s₁ : Strategy (mixK T₁ T₂ true))
  | false => (Strategy.map Sum.inr T₂ s₂ : Strategy (mixK T₁ T₂ false))

theorem expect_mixG (F : τ₁ ⊕ τ₂ → ℝ) :
    expect F (mixG T₁ T₂) (mixStrat T₁ s₁ T₂ s₂) =
      (expect (fun e => F (.inl e)) T₁ s₁ + expect (fun e => F (.inr e)) T₂ s₂) / 2 := by
  show (∑ b : Bool, expect F (mixK T₁ T₂ b) (mixStrat T₁ s₁ T₂ s₂ b)) / (Fintype.card Bool : ℝ) = _
  rw [Fintype.sum_bool, Fintype.card_bool]
  show (expect F (T₁.map Sum.inl) (Strategy.map Sum.inl T₁ s₁) +
      expect F (T₂.map Sum.inr) (Strategy.map Sum.inr T₂ s₂)) / ((2 : ℕ) : ℝ) = _
  rw [expect_map, expect_map]
  norm_num

variable {Val : Type} [Fintype Val] (rd₁ : τ₁ → Finset Val) (rd₂ : τ₂ → Finset Val)

omit [Fintype Val] in
theorem pRd_mixG (v : Val) :
    pRd (mixG T₁ T₂) (mixStrat T₁ s₁ T₂ s₂) (Sum.elim rd₁ rd₂) v = (pRd T₁ s₁ rd₁ v + pRd T₂ s₂ rd₂ v) / 2 :=
  expect_mixG T₁ s₁ T₂ s₂ _

omit [Fintype Val] in
theorem sRd_mixG :
    sRd (mixG T₁ T₂) (mixStrat T₁ s₁ T₂ s₂) (Sum.elim rd₁ rd₂) = (sRd T₁ s₁ rd₁ + sRd T₂ s₂ rd₂) / 2 :=
  expect_mixG T₁ s₁ T₂ s₂ _

/-- **Each trial's link mass is at most twice the mixture's**, at any `X`. -/
theorem massAt_le_two_mix (X : Val) :
    massAt T₁ s₁ rd₁ X ≤ 2 * massAt (mixG T₁ T₂) (mixStrat T₁ s₁ T₂ s₂) (Sum.elim rd₁ rd₂) X ∧
      massAt T₂ s₂ rd₂ X ≤ 2 * massAt (mixG T₁ T₂) (mixStrat T₁ s₁ T₂ s₂) (Sum.elim rd₁ rd₂) X := by
  have hs : ∑ v, (pRd T₁ s₁ rd₁ v + pRd T₂ s₂ rd₂ v) / 2 =
      ((∑ v, pRd T₁ s₁ rd₁ v) + ∑ v, pRd T₂ s₂ rd₂ v) / 2 := by
    rw [← Finset.sum_add_distrib, div_eq_mul_inv, Finset.sum_mul]
    simp only [div_eq_mul_inv]
  have e₁ : pRd T₁ s₁ rd₁ X ≤ ∑ v, pRd T₁ s₁ rd₁ v :=
    Finset.single_le_sum (fun v _ => pRd_nonneg T₁ s₁ rd₁ v) (Finset.mem_univ X)
  have e₂ : pRd T₂ s₂ rd₂ X ≤ ∑ v, pRd T₂ s₂ rd₂ v :=
    Finset.single_le_sum (fun v _ => pRd_nonneg T₂ s₂ rd₂ v) (Finset.mem_univ X)
  have h₁ := sRd_nonneg T₁ s₁ rd₁
  have h₂ := sRd_nonneg T₂ s₂ rd₂
  simp only [massAt, sRd_mixG, pRd_mixG, hs]
  constructor <;> linarith

end Mix

/-! ## Readings embedded in a common type -/

section Image

variable {τ Val V : Type} [Fintype Val] [DecidableEq Val] [Inhabited Val] [Fintype V] [DecidableEq V] (T : Game τ)
  (sT : Strategy T) (rd : τ → Finset Val) {e : Val → V} (he : Function.Injective e)

open Classical in
/-- **A value of `V` pulled back into `Val`**: its preimage under `e`, else the default. -/
noncomputable def pull (e : Val → V) (w : V) : Val := if h : ∃ v, e v = w then Classical.choose h else default

omit [DecidableEq Val] [Fintype V] in
include he in
theorem pull_spec (v : Val) : pull e (e v) = v := by
  unfold pull
  rw [dite_eq_left_of_eq_true (eq_true ⟨v, rfl⟩)]
  exact he (Classical.choose_spec (⟨v, rfl⟩ : ∃ v', e v' = e v))

omit [Fintype Val] [DecidableEq Val] [Inhabited Val] [Fintype V] in
include he in
theorem image_eq_singleton_iff (S : Finset Val) (v : Val) : S.image e = {e v} ↔ S = {v} := by
  rw [← Finset.image_singleton e v]
  exact (Finset.image_injective he).eq_iff

omit [Fintype Val] [DecidableEq Val] [Inhabited Val] [Fintype V] in
include he in
theorem pRd_image (v : Val) : pRd T sT (fun x => (rd x).image e) (e v) = pRd T sT rd v := by
  unfold pRd
  refine congrArg (fun f => expect f T sT) (funext fun x => ?_)
  unfold ind
  split_ifs with h1 h2 h2
  · rfl
  · exact absurd ((image_eq_singleton_iff he _ _).1 h1) h2
  · exact absurd ((image_eq_singleton_iff he _ _).2 h2) h1
  · rfl

omit [Fintype Val] [DecidableEq Val] [Inhabited Val] [Fintype V] in
theorem pRd_image_out {w : V} (hw : ∀ v, e v ≠ w) : pRd T sT (fun x => (rd x).image e) w = 0 := by
  unfold pRd
  have h0 : ∀ x, ind (fun x => (rd x).image e = {w}) x = 0 := fun x => by
    have h : ¬ (rd x).image e = {w} := fun h => by
      have hm : w ∈ (rd x).image e := by rw [h]; exact Finset.mem_singleton_self w
      obtain ⟨v, -, hv⟩ := Finset.mem_image.1 hm
      exact hw v hv
    simp only [ind, h, ↓reduceIte]
  rw [show (ind fun x => (rd x).image e = {w}) = fun _ => (0 : ℝ) from funext h0, expect_const]

omit [Fintype Val] [DecidableEq Val] [Inhabited Val] [Fintype V] in
include he in
theorem rRd_image : rRd T sT (fun x => (rd x).image e) = rRd T sT rd := by
  unfold rRd
  refine congrArg (fun f => expect f T sT) (funext fun x => ?_)
  simp only [ind, ← Finset.card_pos, Finset.card_image_of_injective _ he]

include he in
/-- **The embedded link mass bounds the pulled-back one**: equal at a value of `Val`, and at any other `w` the
embedded readings never read `w` alone, so their link mass there is everything they read. -/
theorem massAt_pull_le (w : V) : massAt T sT rd (pull e w) ≤ massAt T sT (fun x => (rd x).image e) w := by
  rw [massAt_eq, massAt_eq, rRd_image T sT rd he]
  by_cases hw : ∃ v, e v = w
  · obtain ⟨v, rfl⟩ := hw
    rw [pull_spec he, pRd_image T sT rd he]
  · rw [pRd_image_out T sT rd fun v hv => hw ⟨v, hv⟩]
    linarith [pRd_nonneg T sT rd (pull e w)]

end Image

section Emb

variable {τ₁ τ₂ Val₁ Val₂ V : Type} [Fintype Val₁] [DecidableEq Val₁] [Inhabited Val₁] [Fintype Val₂]
  [DecidableEq Val₂] [Inhabited Val₂] [Fintype V] [DecidableEq V] (T₁ : Game τ₁) (s₁ : Strategy T₁) (T₂ : Game τ₂)
  (s₂ : Strategy T₂) (rd₁ : τ₁ → Finset Val₁) (rd₂ : τ₂ → Finset Val₂) (e₁ : Val₁ → V) (e₂ : Val₂ → V)

/-- The mixture's readings, each trial's embedded in `V`. -/
def rdMix : τ₁ ⊕ τ₂ → Finset V := Sum.elim (fun x => (rd₁ x).image e₁) (fun x => (rd₂ x).image e₂)

/-- **The carried leaf's link mass in each session**: read as the pulled-back value `pull e_i Xm`, each trial's link
mass is at most twice the mixture's at `Xm`. -/
theorem carry_massAt (he₁ : Function.Injective e₁) (he₂ : Function.Injective e₂) (Xm : V) :
    massAt T₁ s₁ rd₁ (pull e₁ Xm) ≤ 2 * massAt (mixG T₁ T₂) (mixStrat T₁ s₁ T₂ s₂) (rdMix rd₁ rd₂ e₁ e₂) Xm ∧
      massAt T₂ s₂ rd₂ (pull e₂ Xm) ≤ 2 * massAt (mixG T₁ T₂) (mixStrat T₁ s₁ T₂ s₂) (rdMix rd₁ rd₂ e₁ e₂) Xm := by
  have h := massAt_le_two_mix T₁ s₁ T₂ s₂ (fun x => (rd₁ x).image e₁) (fun x => (rd₂ x).image e₂) Xm
  exact ⟨(massAt_pull_le T₁ s₁ rd₁ he₁ Xm).trans h.1, (massAt_pull_le T₂ s₂ rd₂ he₂ Xm).trans h.2⟩

/-! ### The mixture's finder output -/

variable {O₁ O₂ : Type} (op₁ : τ₁ → Val₁ → O₁) (op₂ : τ₂ → Val₂ → O₂)
  (col₁ : O₁ → O₁ → List UInt8 × List UInt8) (col₂ : O₂ → O₂ → List UInt8 × List UInt8)
  (col₁₂ : O₁ → O₂ → List UInt8 × List UInt8)

/-- The mixture's opening of a value: the session's own opening of its preimage. -/
noncomputable def opMix : τ₁ ⊕ τ₂ → V → O₁ ⊕ O₂
  | .inl x, w => .inl (op₁ x (pull e₁ w))
  | .inr x, w => .inr (op₂ x (pull e₂ w))

/-- The mixture's collision from two openings: each session's own, and the cross one either way round. -/
def colMix : O₁ ⊕ O₂ → O₁ ⊕ O₂ → List UInt8 × List UInt8
  | .inl o, .inl o' => col₁ o o'
  | .inr o, .inr o' => col₂ o o'
  | .inl o, .inr o' => col₁₂ o o'
  | .inr o, .inl o' => ((col₁₂ o' o).2, (col₁₂ o' o).1)

omit [DecidableEq Val₁] [DecidableEq Val₂] [Fintype V] in
/-- **Every success of the mixture's finder outputs a collision**, given each session's openings and collisions and
the cross collision between openings of the carried leaf whose values differ in `V`. -/
theorem outMix_spec {Dh : Type} {Hc : List UInt8 → Dh} (he₁ : Function.Injective e₁) (he₂ : Function.Injective e₂)
    (G₁ : O₁ → Prop) (G₂ : O₂ → Prop) (val₁ : O₁ → Val₁) (val₂ : O₂ → Val₂)
    (hop₁ : ∀ x v, v ∈ rd₁ x → G₁ (op₁ x v) ∧ val₁ (op₁ x v) = v)
    (hop₂ : ∀ x v, v ∈ rd₂ x → G₂ (op₂ x v) ∧ val₂ (op₂ x v) = v)
    (hcol₁ : ∀ o o', G₁ o → G₁ o' → val₁ o ≠ val₁ o' → (col₁ o o').1 ≠ (col₁ o o').2 ∧ Hc (col₁ o o').1 = Hc (col₁ o o').2)
    (hcol₂ : ∀ o o', G₂ o → G₂ o' → val₂ o ≠ val₂ o' → (col₂ o o').1 ≠ (col₂ o o').2 ∧ Hc (col₂ o o').1 = Hc (col₂ o o').2)
    (hcol₁₂ : ∀ o o', G₁ o → G₂ o' → e₁ (val₁ o) ≠ e₂ (val₂ o') →
      (col₁₂ o o').1 ≠ (col₁₂ o o').2 ∧ Hc (col₁₂ o o').1 = Hc (col₁₂ o o').2)
    (x : (τ₁ ⊕ τ₂) × List (τ₁ ⊕ τ₂)) (hS : Succ (rdMix rd₁ rd₂ e₁ e₂) x) :
    Coll Hc (outOf (rdMix rd₁ rd₂ e₁ e₂) (opMix e₁ e₂ op₁ op₂) (colMix col₁ col₂ col₁₂) x) :=
  outOf_spec (rdMix rd₁ rd₂ e₁ e₂) (opMix e₁ e₂ op₁ op₂) (colMix col₁ col₂ col₁₂)
    (fun o => Sum.elim G₁ G₂ o) (fun o => Sum.elim (fun o => e₁ (val₁ o)) (fun o => e₂ (val₂ o)) o)
    (fun x w hw => match x, hw with
      | .inl x, hw => by
        obtain ⟨v, hv, rfl⟩ := Finset.mem_image.1 hw
        obtain ⟨hg, hval⟩ := hop₁ x v hv
        exact ⟨by simpa [opMix, pull_spec he₁] using hg, by simp [opMix, pull_spec he₁, hval]⟩
      | .inr x, hw => by
        obtain ⟨v, hv, rfl⟩ := Finset.mem_image.1 hw
        obtain ⟨hg, hval⟩ := hop₂ x v hv
        exact ⟨by simpa [opMix, pull_spec he₂] using hg, by simp [opMix, pull_spec he₂, hval]⟩)
    (fun o o' h h' hv => match o, o', h, h', hv with
      | .inl o, .inl o', h, h', hv => hcol₁ o o' h h' fun he => hv (congrArg e₁ he)
      | .inr o, .inr o', h, h', hv => hcol₂ o o' h h' fun he => hv (congrArg e₂ he)
      | .inl o, .inr o', h, h', hv => hcol₁₂ o o' h h' hv
      | .inr o, .inl o', h, h', hv => by
        obtain ⟨hne, heq⟩ := hcol₁₂ o' o h' h (Ne.symm hv)
        exact ⟨Ne.symm hne, heq.symm⟩)
    x hS

end Emb

end LinkFinder

end FlockSoundness
