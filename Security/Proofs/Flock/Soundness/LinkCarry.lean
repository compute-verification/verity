import Proofs.Flock.Soundness.LinkFinder

/-!
# Two trials behind one leaf: the carried-leaf finder

A leaf carried across a wipe is registered in two sessions' public files, and each session's value layer reads it as
the plurality of that session's own trials. Every trial of either session that opens the leaf opens the same commitment,
so two readings with different values give a collision whichever session each came from. The finder for the pair is
`LinkFinder.finderG` on the mixture of the two trials (`mixG`: a fair coin picks which one runs), with the two sessions'
values embedded in one type.

* `pRd_mixG`, `sRd_mixG`: the mixture reads `v` alone (reads two values) with the average of the two trials'
  probabilities.
* `carry_split`: if the two trials' values `X₁` and `X₂` differ, one of them reads anything at all with probability at
  most twice the mixture's link mass at any `Xm` plus its own link mass at `X_i`, `r_i ≤ 2·L_mix(Xm) + L_i(X_i)`, where
  `L(X) = s + Σ_{v ≠ X} p v` (`massAt`). No square root: a plurality `X_i` other than the mixture's is read alone at most
  as often as the mixture reads a value other than its own.
* `carry_split_emb`: the same with each trial's readings embedded injectively in a common type.
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

variable {Val : Type} [Fintype Val] [DecidableEq Val] (rd₁ : τ₁ → Finset Val) (rd₂ : τ₂ → Finset Val)

theorem pRd_mixG (v : Val) :
    pRd (mixG T₁ T₂) (mixStrat T₁ s₁ T₂ s₂) (Sum.elim rd₁ rd₂) v = (pRd T₁ s₁ rd₁ v + pRd T₂ s₂ rd₂ v) / 2 :=
  expect_mixG T₁ s₁ T₂ s₂ _

theorem sRd_mixG :
    sRd (mixG T₁ T₂) (mixStrat T₁ s₁ T₂ s₂) (Sum.elim rd₁ rd₂) = (sRd T₁ s₁ rd₁ + sRd T₂ s₂ rd₂) / 2 :=
  expect_mixG T₁ s₁ T₂ s₂ _

omit [DecidableEq Val] in
/-- Each trial's mass off `X` is at most twice the mixture's. -/
theorem sub_le_two_mul_mix {p₁ p₂ : Val → ℝ} (h₁ : ∀ v, 0 ≤ p₁ v) (h₂ : ∀ v, 0 ≤ p₂ v) (X : Val) :
    (∑ v, p₁ v) - p₁ X ≤ 2 * ((∑ v, (p₁ v + p₂ v) / 2) - (p₁ X + p₂ X) / 2) ∧
      (∑ v, p₂ v) - p₂ X ≤ 2 * ((∑ v, (p₁ v + p₂ v) / 2) - (p₁ X + p₂ X) / 2) := by
  have hs : ∑ v, (p₁ v + p₂ v) / 2 = ((∑ v, p₁ v) + ∑ v, p₂ v) / 2 := by
    rw [← Finset.sum_div, Finset.sum_add_distrib]
  have e₁ : p₁ X ≤ ∑ v, p₁ v := Finset.single_le_sum (fun v _ => h₁ v) (Finset.mem_univ X)
  have e₂ : p₂ X ≤ ∑ v, p₂ v := Finset.single_le_sum (fun v _ => h₂ v) (Finset.mem_univ X)
  rw [hs]
  constructor <;> linarith

/-- **A value other than `Xm` is read alone at most as often as anything but `Xm`**: if `X ≠ Xm`, the probability `Σ p + s`
of reading anything is at most `2(sm + Mm) + L(X)`, given the mass `Σ p − p Xm` off `Xm` is at most `2 Mm`. -/
theorem read_le_of_ne {p : Val → ℝ} (hp : ∀ v, 0 ≤ p v) {X Xm : Val} (hne : X ≠ Xm) (s : ℝ) {Mm sm : ℝ}
    (hsm : 0 ≤ sm) (hle : (∑ v, p v) - p Xm ≤ 2 * Mm) :
    (∑ v, p v) + s ≤ 2 * (sm + Mm) + (s + ((∑ v, p v) - p X)) := by
  have hpair : p X + p Xm ≤ ∑ v, p v := by
    have h := Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ {X, Xm}) (fun v _ _ => hp v)
    rwa [Finset.sum_pair hne] at h
  linarith

/-- **The carried-leaf split.** If two trials' values `X₁` and `X₂` at one leaf differ, then for any `Xm` one of the
trials reads anything at all with probability at most twice the mixture's link mass at `Xm` plus its own at `X_i`. -/
theorem carry_split {X₁ X₂ : Val} (hne : X₁ ≠ X₂) (Xm : Val) :
    rRd T₁ s₁ rd₁ ≤
        2 * massAt (mixG T₁ T₂) (mixStrat T₁ s₁ T₂ s₂) (Sum.elim rd₁ rd₂) Xm + massAt T₁ s₁ rd₁ X₁ ∨
      rRd T₂ s₂ rd₂ ≤
        2 * massAt (mixG T₁ T₂) (mixStrat T₁ s₁ T₂ s₂) (Sum.elim rd₁ rd₂) Xm + massAt T₂ s₂ rd₂ X₂ := by
  have hm : massAt (mixG T₁ T₂) (mixStrat T₁ s₁ T₂ s₂) (Sum.elim rd₁ rd₂) Xm =
      (sRd T₁ s₁ rd₁ + sRd T₂ s₂ rd₂) / 2 +
        ((∑ v, (pRd T₁ s₁ rd₁ v + pRd T₂ s₂ rd₂ v) / 2) - (pRd T₁ s₁ rd₁ Xm + pRd T₂ s₂ rd₂ Xm) / 2) := by
    simp only [massAt, sRd_mixG, pRd_mixG]
  have hsm : 0 ≤ (sRd T₁ s₁ rd₁ + sRd T₂ s₂ rd₂) / 2 :=
    div_nonneg (add_nonneg (sRd_nonneg T₁ s₁ rd₁) (sRd_nonneg T₂ s₂ rd₂)) zero_le_two
  have hle := sub_le_two_mul_mix (pRd_nonneg T₁ s₁ rd₁) (pRd_nonneg T₂ s₂ rd₂) Xm
  rw [hm, rRd_eq T₁ s₁ rd₁, rRd_eq T₂ s₂ rd₂]
  by_cases h1 : X₁ = Xm
  · exact Or.inr (read_le_of_ne (pRd_nonneg T₂ s₂ rd₂) (fun h => hne (h1.trans h.symm)) _ hsm hle.2)
  · exact Or.inl (read_le_of_ne (pRd_nonneg T₁ s₁ rd₁) h1 _ hsm hle.1)

end Mix

/-! ## Readings embedded in a common type -/

section Image

variable {τ Val V : Type} [Fintype Val] [DecidableEq Val] [Fintype V] [DecidableEq V] (T : Game τ) (sT : Strategy T)
  (rd : τ → Finset Val) {e : Val → V} (he : Function.Injective e)

include he in
theorem image_eq_singleton_iff (S : Finset Val) (v : Val) : S.image e = {e v} ↔ S = {v} := by
  rw [← Finset.image_singleton e v]
  exact (Finset.image_injective he).eq_iff

include he in
theorem pRd_image (v : Val) : pRd T sT (fun x => (rd x).image e) (e v) = pRd T sT rd v := by
  unfold pRd
  refine congrArg (fun f => expect f T sT) (funext fun x => ?_)
  simp only [ind, image_eq_singleton_iff he]

omit [DecidableEq Val] in
theorem pRd_image_out {w : V} (hw : ∀ v, e v ≠ w) : pRd T sT (fun x => (rd x).image e) w = 0 := by
  unfold pRd
  have h0 : ∀ x, ind (fun x => (rd x).image e = {w}) x = 0 := fun x => by
    have h : ¬ (rd x).image e = {w} := fun h => by
      have hm : w ∈ (rd x).image e := by rw [h]; exact Finset.mem_singleton_self w
      obtain ⟨v, -, hv⟩ := Finset.mem_image.1 hm
      exact hw v hv
    simp only [ind, h, ↓reduceIte]
  rw [show (ind fun x => (rd x).image e = {w}) = fun _ => (0 : ℝ) from funext h0, expect_const]

include he in
theorem sRd_image : sRd T sT (fun x => (rd x).image e) = sRd T sT rd := by
  unfold sRd
  refine congrArg (fun f => expect f T sT) (funext fun x => ?_)
  simp only [ind, Finset.card_image_of_injective _ he]

include he in
theorem rRd_image : rRd T sT (fun x => (rd x).image e) = rRd T sT rd := by
  unfold rRd
  refine congrArg (fun f => expect f T sT) (funext fun x => ?_)
  simp only [ind, ← Finset.card_pos, Finset.card_image_of_injective _ he]

include he in
theorem massAt_image (X : Val) : massAt T sT (fun x => (rd x).image e) (e X) = massAt T sT rd X := by
  have h1 := rRd_eq T sT (fun x => (rd x).image e)
  have h2 := rRd_eq T sT rd
  rw [rRd_image T sT rd he, sRd_image T sT rd he] at h1
  unfold massAt
  rw [sRd_image T sT rd he, pRd_image T sT rd he]
  linarith

include he in
/-- The embedded readings' plurality is the embedded plurality. -/
theorem plur_image {X : Val} (hX : ∀ v, pRd T sT rd v ≤ pRd T sT rd X) (w : V) :
    pRd T sT (fun x => (rd x).image e) w ≤ pRd T sT (fun x => (rd x).image e) (e X) := by
  rw [pRd_image T sT rd he X]
  by_cases hw : ∃ v, e v = w
  · obtain ⟨v, rfl⟩ := hw
    rw [pRd_image T sT rd he v]
    exact hX v
  · rw [pRd_image_out T sT rd fun v hv => hw ⟨v, hv⟩]
    exact pRd_nonneg T sT rd X

end Image

section Emb

variable {τ₁ τ₂ Val₁ Val₂ V : Type} [Fintype Val₁] [DecidableEq Val₁] [Fintype Val₂] [DecidableEq Val₂] [Fintype V]
  [DecidableEq V] (T₁ : Game τ₁) (s₁ : Strategy T₁) (T₂ : Game τ₂) (s₂ : Strategy T₂) (rd₁ : τ₁ → Finset Val₁)
  (rd₂ : τ₂ → Finset Val₂) {e₁ : Val₁ → V} {e₂ : Val₂ → V} (he₁ : Function.Injective e₁) (he₂ : Function.Injective e₂)

/-- The mixture's readings, each trial's embedded in `V`. -/
def rdMix : τ₁ ⊕ τ₂ → Finset V := Sum.elim (fun x => (rd₁ x).image e₁) (fun x => (rd₂ x).image e₂)

include he₁ he₂ in
/-- **The carried-leaf split, embedded**: if the two trials' values embed differently, one of them reads anything at
all with probability at most twice the mixture's link mass at any `Xm` plus its own link mass at its value. -/
theorem carry_split_emb {X₁ : Val₁} {X₂ : Val₂} (hne : e₁ X₁ ≠ e₂ X₂) (Xm : V) :
    rRd T₁ s₁ rd₁ ≤ 2 * massAt (mixG T₁ T₂) (mixStrat T₁ s₁ T₂ s₂) (rdMix rd₁ rd₂ (e₁ := e₁) (e₂ := e₂)) Xm +
        massAt T₁ s₁ rd₁ X₁ ∨
      rRd T₂ s₂ rd₂ ≤ 2 * massAt (mixG T₁ T₂) (mixStrat T₁ s₁ T₂ s₂) (rdMix rd₁ rd₂ (e₁ := e₁) (e₂ := e₂)) Xm +
        massAt T₂ s₂ rd₂ X₂ := by
  have h := carry_split T₁ s₁ T₂ s₂ (fun x => (rd₁ x).image e₁) (fun x => (rd₂ x).image e₂) hne Xm
  rwa [rRd_image T₁ s₁ rd₁ he₁, rRd_image T₂ s₂ rd₂ he₂, massAt_image T₁ s₁ rd₁ he₁,
    massAt_image T₂ s₂ rd₂ he₂] at h

end Emb

end LinkFinder

end FlockSoundness
