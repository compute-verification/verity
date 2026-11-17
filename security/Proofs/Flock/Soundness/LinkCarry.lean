module

public import Mathlib.Algebra.BigOperators.Field
public import Proofs.Flock.Soundness.LinkFinder
import Mathlib.Tactic.Positivity.Finset
import Proofs.Flock.Soundness.Game

/-! Carried state, one hiding leaf read in several sampling blocks, each of which pays at most `m` times the mixture's
binding error for it. -/

@[expose] public section

namespace FlockSoundness

namespace LinkFinder

open Game Extract

section Err

variable {τ Val : Type} [Fintype Val] [DecidableEq Val] (T : Game τ) (sT : Strategy T) (rd : τ → Finset Val)

noncomputable def errAt (X : Val) : ℝ := sRd T sT rd + ((∑ v, pRd T sT rd v) - pRd T sT rd X)

theorem errAt_eq (X : Val) : errAt T sT rd X = rRd T sT rd - pRd T sT rd X := by
  rw [errAt, rRd_eq]
  ring

theorem errAt_eq_expect (X : Val) :
    errAt T sT rd X = expect (fun e => ind (fun e => (rd e).Nonempty) e - ind (fun e => rd e = {X}) e) T sT :=
  (errAt_eq T sT rd X).trans (expect_sub _ _ T sT).symm

omit [DecidableEq Val] in
theorem errAt_nonneg (X : Val) : 0 ≤ errAt T sT rd X := by
  have h := Finset.single_le_sum (fun v _ => pRd_nonneg T sT rd v) (Finset.mem_univ X)
  have hs := sRd_nonneg T sT rd
  unfold errAt
  linarith

def Coll {Dh : Type} (Hc : List UInt8 → Dh) (o : Option (List UInt8 × List UInt8)) : Prop :=
  ∃ a b, o = some (a, b) ∧ a ≠ b ∧ Hc a = Hc b

theorem errAt_le_coll {Dh : Type} (Hc : List UInt8 → Dh) (out : τ × List τ → Option (List UInt8 × List UInt8))
    (M : ℕ) (hM : 1 ≤ M) (hout : ∀ x, Succ rd x → Coll Hc (out x)) {X : Val}
    (hX : ∀ v, pRd T sT rd v ≤ pRd T sT rd X) :
    errAt T sT rd X ≤
      (prob (fun x => Coll Hc (out x)) (finderG T M) (finderStrat T sT M)).toReal + 1 / (Real.exp 1 * M) := by
  have h1 := succ_ge T sT rd hX M hM
  have h2 : expect (ind (Succ rd)) (finderG T M) (finderStrat T sT M) ≤
      (prob (fun x => Coll Hc (out x)) (finderG T M) (finderStrat T sT M)).toReal := by
    rw [prob_toReal]
    exact expect_mono (ind_mono hout) _ _
  unfold errAt
  linarith

end Err

section Mix

variable {ι : Type} [Fintype ι] [Nonempty ι] {τ : ι → Type} (T : (i : ι) → Game (τ i))
  (s : (i : ι) → Strategy (T i))

/-- **The mixture of the trials**: a uniform coin `i`, then trial `i`. -/
def mixG : Game (Σ i, τ i) := Game.coin ι fun i => (T i).map (Sigma.mk i)

/-- The trials' strategies in the mixture. -/
def mixStrat : Strategy (mixG T) := fun i => Strategy.map (Sigma.mk i) (T i) (s i)

theorem expect_mixG (F : (Σ i, τ i) → ℝ) :
    expect F (mixG T) (mixStrat T s) = (∑ i, expect (fun e => F ⟨i, e⟩) (T i) (s i)) / Fintype.card ι := by
  show (∑ i, expect F ((T i).map (Sigma.mk i)) (Strategy.map (Sigma.mk i) (T i) (s i))) /
    (Fintype.card ι : ℝ) = _
  simp only [expect_map]

variable {V : Type} [Fintype V] [DecidableEq V] (rd : (i : ι) → τ i → Finset V)

/-- The mixture's readings: trial `i`'s. -/
def rdSig : (Σ i, τ i) → Finset V := fun x => rd x.1 x.2

omit [Fintype V] [DecidableEq V] in
theorem pRd_mixG (v : V) :
    pRd (mixG T) (mixStrat T s) (rdSig rd) v = (∑ i, pRd (T i) (s i) (rd i) v) / Fintype.card ι :=
  expect_mixG T s _

omit [Fintype V] [DecidableEq V] in
theorem sRd_mixG : sRd (mixG T) (mixStrat T s) (rdSig rd) = (∑ i, sRd (T i) (s i) (rd i)) / Fintype.card ι :=
  expect_mixG T s _

omit [DecidableEq V] in
theorem errAt_mixG (X : V) :
    errAt (mixG T) (mixStrat T s) (rdSig rd) X = (∑ i, errAt (T i) (s i) (rd i) X) / Fintype.card ι := by
  unfold errAt
  simp only [sRd_mixG, pRd_mixG]
  rw [← Finset.sum_div, Finset.sum_comm, Finset.sum_add_distrib, Finset.sum_sub_distrib]
  ring

theorem errAt_le_mix (i : ι) (X : V) :
    errAt (T i) (s i) (rd i) X ≤ Fintype.card ι * errAt (mixG T) (mixStrat T s) (rdSig rd) X := by
  have hm : (Fintype.card ι : ℝ) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
  rw [errAt_mixG, mul_div_assoc', mul_comm, mul_div_assoc, div_self hm, mul_one]
  exact Finset.single_le_sum (fun j _ => errAt_nonneg (T j) (s j) (rd j) X) (Finset.mem_univ i)

end Mix

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
theorem errAt_pull_le (w : V) : errAt T sT rd (pull e w) ≤ errAt T sT (fun x => (rd x).image e) w := by
  rw [errAt_eq, errAt_eq, rRd_image T sT rd he]
  by_cases hw : ∃ v, e v = w
  · obtain ⟨v, rfl⟩ := hw
    rw [pull_spec he, pRd_image T sT rd he]
  · rw [pRd_image_out T sT rd fun v hv => hw ⟨v, hv⟩]
    linarith [pRd_nonneg T sT rd (pull e w)]

end Image

section Emb

variable {ι : Type} [Fintype ι] [Nonempty ι] {τ : ι → Type} (T : (i : ι) → Game (τ i))
  (s : (i : ι) → Strategy (T i)) {Val : ι → Type} [∀ i, Fintype (Val i)] [∀ i, DecidableEq (Val i)]
  [∀ i, Inhabited (Val i)] {V : Type} [Fintype V] [DecidableEq V] (rd : (i : ι) → τ i → Finset (Val i))
  (e : (i : ι) → Val i → V)

def rdMix : (Σ i, τ i) → Finset V := rdSig fun i x => (rd i x).image (e i)

theorem carry_errAt (he : ∀ i, Function.Injective (e i)) (Xm : V) (i : ι) :
    errAt (T i) (s i) (rd i) (pull (e i) Xm) ≤ Fintype.card ι * errAt (mixG T) (mixStrat T s) (rdMix rd e) Xm :=
  (errAt_pull_le (T i) (s i) (rd i) (he i) Xm).trans (errAt_le_mix T s (fun i x => (rd i x).image (e i)) i Xm)

variable {O : ι → Type} (op : (i : ι) → τ i → Val i → O i) (col : (i j : ι) → O i → O j → List UInt8 × List UInt8)

noncomputable def opMix (x : Σ i, τ i) (w : V) : Σ i, O i := ⟨x.1, op x.1 x.2 (pull (e x.1) w)⟩

def colMix (o o' : Σ i, O i) : List UInt8 × List UInt8 := col o.1 o'.1 o.2 o'.2

omit [Fintype ι] [Nonempty ι] [(i : ι) → DecidableEq (Val i)] [Fintype V] in
theorem outMix_spec {Dh : Type} {Hc : List UInt8 → Dh} (he : ∀ i, Function.Injective (e i))
    (G : (i : ι) → O i → Prop) (val : (i : ι) → O i → Val i)
    (hop : ∀ i x v, v ∈ rd i x → G i (op i x v) ∧ val i (op i x v) = v)
    (hcol : ∀ i j o o', G i o → G j o' → e i (val i o) ≠ e j (val j o') →
      (col i j o o').1 ≠ (col i j o o').2 ∧ Hc (col i j o o').1 = Hc (col i j o o').2)
    (x : (Σ i, τ i) × List (Σ i, τ i)) (hS : Succ (rdMix rd e) x) :
    Coll Hc (outOf (rdMix rd e) (opMix e op) (colMix col) x) :=
  outOf_spec (rdMix rd e) (opMix e op) (colMix col) (fun o => G o.1 o.2) (fun o => e o.1 (val o.1 o.2))
    (fun x w hw => by
      obtain ⟨v, hv, rfl⟩ := Finset.mem_image.1 hw
      obtain ⟨hg, hval⟩ := hop x.1 x.2 v hv
      exact ⟨by simpa [opMix, pull_spec (he x.1)] using hg, by simp [opMix, pull_spec (he x.1), hval]⟩)
    (fun o o' h h' hv => hcol o.1 o'.1 o.2 o'.2 h h' hv) x hS

end Emb

end LinkFinder

end FlockSoundness
