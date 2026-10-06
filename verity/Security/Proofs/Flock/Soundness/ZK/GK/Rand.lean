import Proofs.Flock.Soundness.Game.Seq

/-!
# Zero knowledge: programs on finitely many uniform coins

The Goldreich–Kahan simulator (`GK/Sim.lean`) is an algorithm that draws fresh uniform randomness for each run it makes and
decides what to do next from what the runs returned. `Rand α` is such an algorithm: a `Game` without the prover's
messages, so a tree whose nodes draw a uniform value of a finite type and whose leaves return a value. `ev f r` is the
expectation of `f` at its output.

* `Rand.ev_bind`, `Rand.ev_map`: sequencing, as `Game.expect_bind`;
* `Rand.ev_mono`, `ev_add`, `ev_const_mul`, `ev_const`: the expectation is linear and monotone;
* `Rand.toGame`, `Rand.strat`, `Rand.expect_toGame`: a program is a game with nothing for the prover to play, which is how
  a collision finder built from it is a `StrictCR.Finder`;
* `avgR`, `avgR_ind_eq_prCoin`: one coin's average, and its link to `prCoin`;
* `avgR_le_of_events`: a bound on every event's probability bounds every test with values in `[0, 1]`.
-/

namespace FlockSoundness.ZK.GK

open Finset Game
open scoped ENNReal

/-- The average of a real function over a finite type, under the uniform distribution. -/
noncomputable def avgR {C : Type} [Fintype C] (f : C → ℝ) : ℝ := (∑ c : C, f c) / Fintype.card C

section Avg

variable {C : Type} [Fintype C]

theorem avgR_add (f g : C → ℝ) : avgR (fun c => f c + g c) = avgR f + avgR g := by
  simp only [avgR, Finset.sum_add_distrib, add_div]

theorem avgR_const_mul (a : ℝ) (f : C → ℝ) : avgR (fun c => a * f c) = a * avgR f := by
  simp only [avgR, ← Finset.mul_sum, mul_div_assoc]

theorem avgR_const [Nonempty C] (a : ℝ) : avgR (fun _ : C => a) = a := by
  simp only [avgR, Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
  have : (Fintype.card C : ℝ) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
  field_simp

theorem avgR_mono {f g : C → ℝ} (h : ∀ c, f c ≤ g c) : avgR f ≤ avgR g :=
  div_le_div_of_nonneg_right (Finset.sum_le_sum fun c _ => h c) (Nat.cast_nonneg _)

theorem avgR_sub (f g : C → ℝ) : avgR (fun c => f c - g c) = avgR f - avgR g := by
  simp only [avgR, Finset.sum_sub_distrib, sub_div]

theorem avgR_nonneg {f : C → ℝ} (h : ∀ c, 0 ≤ f c) : 0 ≤ avgR f :=
  div_nonneg (Finset.sum_nonneg fun c _ => h c) (Nat.cast_nonneg _)

theorem avgR_le_one [Nonempty C] {f : C → ℝ} (h : ∀ c, f c ≤ 1) : avgR f ≤ 1 :=
  (avgR_mono h).trans (avgR_const 1).le

theorem abs_avgR_le [Nonempty C] {f : C → ℝ} {a : ℝ} (h : ∀ c, |f c| ≤ a) : |avgR f| ≤ a := by
  rw [abs_le]
  constructor
  · have := avgR_mono (f := fun _ => -a) (g := f) fun c => (abs_le.1 (h c)).1
    rwa [avgR_const] at this
  · have := avgR_mono (f := f) (g := fun _ => a) fun c => (abs_le.1 (h c)).2
    rwa [avgR_const] at this

/-- A coin's average of a two-valued function: the probability of the condition weighs the two values. -/
theorem avgR_ite [Nonempty C] (p : C → Prop) [DecidablePred p] (a b : ℝ) :
    avgR (fun c => if p c then a else b) = avgR (Game.ind p) * a + (1 - avgR (Game.ind p)) * b := by
  classical
  have e : (fun c => if p c then a else b) = fun c => a * Game.ind p c + (b - b * Game.ind p c) := by
    funext c
    unfold Game.ind
    split_ifs <;> simp
  rw [e, avgR_add, avgR_sub (fun _ => b) fun c => b * Game.ind p c, avgR_const_mul, avgR_const_mul, avgR_const]
  ring

end Avg

open Classical in
/-- `prCoin` is the average of the indicator. -/
theorem avgR_ind_eq_prCoin {C : Type} [Fintype C] [Nonempty C] (p : C → Prop) :
    avgR (Game.ind p) = (prCoin p).toReal := by
  have h := prob_toReal p (Game.draw C) (drawStrat C)
  have e : prob p (Game.draw C) (drawStrat C) = prCoin p := by
    simp only [Game.draw, prob, prCoin, avg]
  rw [e] at h
  rw [h]
  rfl

/-- **A bound on every event bounds every test with values in `[0, 1]`.** If every event has probability under `f` at
most its probability under `g` plus `δ`, so has the average of every function `φ` with values in `[0, 1]`: `φ` is a
mixture of the events `{φ > λ}`; here, the finite sum over the values both sides take. -/
theorem avgR_le_of_events {Ω₁ Ω₂ V : Type} [Fintype Ω₁] [Fintype Ω₂] [Nonempty Ω₁] [Nonempty Ω₂] (f : Ω₁ → V)
    (g : Ω₂ → V) {δ : ℝ} (h : ∀ U : V → Prop, avgR (fun ω => Game.ind U (f ω)) ≤ avgR (fun ω => Game.ind U (g ω)) + δ)
    (φ : V → ℝ) (h0 : ∀ v, 0 ≤ φ v) (h1 : ∀ v, φ v ≤ 1) :
    avgR (fun ω => φ (f ω)) ≤ avgR (fun ω => φ (g ω)) + δ := by
  classical
  let S : Finset V := univ.image f ∪ univ.image g
  -- each side's average as a sum over the values in `S`, weighted by their probabilities
  let P : V → ℝ := fun v => avgR (fun ω => Game.ind (fun x => x = v) (f ω))
  let Q : V → ℝ := fun v => avgR (fun ω => Game.ind (fun x => x = v) (g ω))
  have decomp : ∀ {Ω : Type} [Fintype Ω] (k : Ω → V), (∀ ω, k ω ∈ S) → ∀ ψ : V → ℝ,
      avgR (fun ω => ψ (k ω)) = ∑ v ∈ S, ψ v * avgR (fun ω => Game.ind (fun x => x = v) (k ω)) := by
    intro Ω _ k hk ψ
    have hpt : ∀ ω, ψ (k ω) = ∑ v ∈ S, ψ v * Game.ind (fun x => x = v) (k ω) := fun ω => by
      rw [Finset.sum_eq_single (k ω)]
      · simp [Game.ind]
      · intro v _ hv
        simp [Game.ind, Ne.symm hv]
      · intro hn
        exact absurd (hk ω) hn
    simp only [hpt, avgR, div_eq_mul_inv, Finset.sum_mul]
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun v _ => ?_
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl fun _ _ => by ring
  have hf : ∀ ω, f ω ∈ S := fun ω => mem_union_left _ (mem_image_of_mem f (mem_univ ω))
  have hg : ∀ ω, g ω ∈ S := fun ω => mem_union_right _ (mem_image_of_mem g (mem_univ ω))
  rw [decomp f hf φ, decomp g hg φ]
  -- the event where `f` is likelier
  let U : V → Prop := fun v => v ∈ S ∧ Q v < P v
  have hU := h U
  rw [decomp f hf (Game.ind U), decomp g hg (Game.ind U)] at hU
  have key : ∑ v ∈ S, φ v * P v - ∑ v ∈ S, φ v * Q v ≤ ∑ v ∈ S, Game.ind U v * P v - ∑ v ∈ S, Game.ind U v * Q v := by
    rw [← Finset.sum_sub_distrib, ← Finset.sum_sub_distrib]
    refine Finset.sum_le_sum fun v hv => ?_
    by_cases hlt : Q v < P v
    · have hu : Game.ind U v = 1 := by simp [Game.ind, U, hv, hlt]
      rw [hu]
      nlinarith [h1 v]
    · have hu : Game.ind U v = 0 := by simp [Game.ind, U, hlt]
      rw [hu]
      push Not at hlt
      nlinarith [h0 v]
  show ∑ v ∈ S, φ v * P v ≤ ∑ v ∈ S, φ v * Q v + δ
  linarith

/-- Two-sided: if every event's probabilities differ by at most `δ`, so do the averages of every test with values in
`[0, 1]`. -/
theorem abs_avgR_sub_le_of_events {Ω₁ Ω₂ V : Type} [Fintype Ω₁] [Fintype Ω₂] [Nonempty Ω₁] [Nonempty Ω₂]
    (f : Ω₁ → V) (g : Ω₂ → V) {δ : ℝ}
    (h : ∀ U : V → Prop, |avgR (fun ω => Game.ind U (f ω)) - avgR (fun ω => Game.ind U (g ω))| ≤ δ)
    (φ : V → ℝ) (h0 : ∀ v, 0 ≤ φ v) (h1 : ∀ v, φ v ≤ 1) :
    |avgR (fun ω => φ (f ω)) - avgR (fun ω => φ (g ω))| ≤ δ := by
  have a := avgR_le_of_events f g (fun U => by linarith [(abs_le.1 (h U)).2]) φ h0 h1
  have b := avgR_le_of_events g f (fun U => by linarith [(abs_le.1 (h U)).1]) φ h0 h1
  rw [abs_le]
  constructor <;> linarith

theorem abs_avgR_sub_le {C : Type} [Fintype C] {f g k : C → ℝ} (h : ∀ c, |f c - g c| ≤ k c) :
    |avgR f - avgR g| ≤ avgR k := by
  rw [← avgR_sub, abs_le]
  constructor
  · have := avgR_mono (f := fun c => -1 * k c) (g := fun c => f c - g c) fun c => by
      linarith [(abs_le.1 (h c)).1]
    rw [avgR_const_mul] at this
    linarith
  · exact avgR_mono fun c => (abs_le.1 (h c)).2

/-- A finite uniform program: draws of uniform values of finite types, then a result. -/
inductive Rand (α : Type) : Type 1 where
  | ret : α → Rand α
  | coin (C : Type) [Fintype C] [Nonempty C] : (C → Rand α) → Rand α

namespace Rand

variable {α β γ : Type}

/-- Run `r`, then `f` of its result. -/
def bind : Rand α → (α → Rand β) → Rand β
  | .ret a, f => f a
  | @Rand.coin _ C iF iN k, f => @Rand.coin _ C iF iN fun c => bind (k c) f

/-- Relabel the result. -/
def map (f : α → β) (r : Rand α) : Rand β := r.bind fun a => .ret (f a)

/-- The expectation of `f` at the program's result, over its uniform coins. -/
noncomputable def ev (f : α → ℝ) : Rand α → ℝ
  | .ret a => f a
  | @Rand.coin _ C iF _ k => @avgR C iF fun c => ev f (k c)

@[simp] theorem ev_ret (f : α → ℝ) (a : α) : ev f (.ret a) = f a := rfl

theorem ev_coin (f : α → ℝ) (C : Type) [Fintype C] [Nonempty C] (k : C → Rand α) :
    ev f (.coin C k) = avgR fun c => ev f (k c) := rfl

theorem ev_bind (f : β → ℝ) : ∀ (r : Rand α) (g : α → Rand β), ev f (r.bind g) = ev (fun a => ev f (g a)) r
  | .ret _, _ => rfl
  | @Rand.coin _ C _ _ k, g => by
      show (avgR fun c => ev f ((k c).bind g)) = avgR fun c => ev (fun a => ev f (g a)) (k c)
      simp only [ev_bind f (k _) g]

theorem ev_map (f : β → ℝ) (g : α → β) (r : Rand α) : ev f (r.map g) = ev (fun a => f (g a)) r :=
  ev_bind f r _

theorem ev_mono {f g : α → ℝ} (h : ∀ a, f a ≤ g a) : ∀ r : Rand α, ev f r ≤ ev g r
  | .ret a => h a
  | @Rand.coin _ _ _ _ k => avgR_mono fun c => ev_mono h (k c)

theorem ev_add (f g : α → ℝ) : ∀ r : Rand α, ev (fun a => f a + g a) r = ev f r + ev g r
  | .ret _ => rfl
  | @Rand.coin _ C _ _ k => by
      simp only [ev_coin, ev_add f g (k _), avgR_add]

theorem ev_const_mul (c : ℝ) (f : α → ℝ) : ∀ r : Rand α, ev (fun a => c * f a) r = c * ev f r
  | .ret _ => rfl
  | @Rand.coin _ C _ _ k => by
      simp only [ev_coin, ev_const_mul c f (k _), avgR_const_mul]

theorem ev_const (c : ℝ) : ∀ r : Rand α, ev (fun _ => c) r = c
  | .ret _ => rfl
  | @Rand.coin _ C _ _ k => by
      simp only [ev_coin, ev_const c (k _), avgR_const]

theorem ev_sub (f g : α → ℝ) (r : Rand α) : ev (fun a => f a - g a) r = ev f r - ev g r := by
  have h := ev_add f (fun a => -g a) r
  have h' := ev_const_mul (-1) g r
  simp only [neg_one_mul] at h'
  simp only [sub_eq_add_neg, h, h']

theorem ev_nonneg {f : α → ℝ} (h : ∀ a, 0 ≤ f a) (r : Rand α) : 0 ≤ ev f r := by
  have := ev_mono h r
  rwa [ev_const] at this

theorem ev_le_one {f : α → ℝ} (h : ∀ a, f a ≤ 1) (r : Rand α) : ev f r ≤ 1 := by
  have := ev_mono h r
  rwa [ev_const] at this

theorem abs_ev_le {f : α → ℝ} {c : ℝ} (h : ∀ a, |f a| ≤ c) (r : Rand α) : |ev f r| ≤ c := by
  rw [abs_le]
  constructor
  · have := ev_mono (f := fun _ => -c) (fun a => (abs_le.1 (h a)).1) r
    rwa [ev_const] at this
  · have := ev_mono (g := fun _ => c) (fun a => (abs_le.1 (h a)).2) r
    rwa [ev_const] at this

theorem abs_ev_sub_le {f g k : α → ℝ} (h : ∀ a, |f a - g a| ≤ k a) : ∀ r : Rand α, |ev f r - ev g r| ≤ ev k r
  | .ret a => h a
  | @Rand.coin _ _ _ _ k' => abs_avgR_sub_le fun c => abs_ev_sub_le h (k' c)

/-- Every result of the program satisfies `P`. -/
def All (P : α → Prop) : Rand α → Prop
  | .ret a => P a
  | @Rand.coin _ _ _ _ k => ∀ c, All P (k c)

theorem All.bind {P : α → Prop} {Q : β → Prop} (g : α → Rand β) (hg : ∀ a, P a → All Q (g a)) :
    ∀ r : Rand α, All P r → All Q (r.bind g)
  | .ret a, h => hg a h
  | @Rand.coin _ _ _ _ k, h => fun c => All.bind g hg (k c) (h c)

theorem All.map {P : α → Prop} {Q : β → Prop} (f : α → β) (hf : ∀ a, P a → Q (f a)) (r : Rand α) (h : All P r) :
    All Q (r.map f) :=
  All.bind (P := P) (Q := Q) (fun a => .ret (f a)) (fun a ha => hf a ha) r h

theorem ev_le_of_All {P : α → Prop} {f g : α → ℝ} (hfg : ∀ a, P a → f a ≤ g a) : ∀ r : Rand α, All P r → ev f r ≤ ev g r
  | .ret a, h => hfg a h
  | @Rand.coin _ _ _ _ k, h => avgR_mono fun c => ev_le_of_All hfg (k c) (h c)

/-- The probability of an event: the expectation of its indicator. -/
noncomputable def pr (p : α → Prop) (r : Rand α) : ℝ := ev (Game.ind p) r

/-- A drawn value, returned. -/
theorem ev_draw (f : α → ℝ) (C : Type) [Fintype C] [Nonempty C] (x : C → α) :
    ev f (.coin C fun c => .ret (x c)) = avgR fun c => f (x c) := rfl

/-! ## A program is a game with nothing for the prover to play -/

/-- The program as a `Game`: only coins. -/
def toGame : Rand α → Game α
  | .ret a => .ret a
  | @Rand.coin _ C iF iN k => @Game.coin _ C iF iN fun c => toGame (k c)

/-- The one strategy of a game without messages. -/
def strat : (r : Rand α) → Strategy r.toGame
  | .ret _ => PUnit.unit
  | @Rand.coin _ _ _ _ k => fun c => strat (k c)

theorem expect_toGame (f : α → ℝ) : ∀ r : Rand α, expect f r.toGame r.strat = ev f r
  | .ret _ => rfl
  | @Rand.coin _ C _ _ k => by
      show (∑ c : C, expect f (k c).toGame ((k c).strat)) / _ = (∑ c : C, ev f (k c)) / _
      simp only [expect_toGame f (k _)]

theorem prob_toGame (p : α → Prop) (r : Rand α) : (prob p r.toGame r.strat).toReal = pr p r := by
  rw [prob_toReal, expect_toGame]
  rfl

end Rand

end FlockSoundness.ZK.GK
