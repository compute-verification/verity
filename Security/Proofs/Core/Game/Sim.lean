import Mathlib.Data.Fintype.Lattice
import Proofs.Core.Game.Tape
import Proofs.Pouw.PearlC.FlockTile.Bridge

/-!
# Running one game's strategy against another game's verifier

`Dominated p g s q G`: some strategy of `G` makes `q` at least as likely as `s` makes `p` in `g`. The rules build that
strategy node by node, as a reduction runs a prover of `g` against the verifier of `G`:

* a message of either game is the reduction's to choose (`dom_send_left`, `dom_send_right`);
* a coin of `g` that a coin of `G` supplies through a map keeping the uniform law (`Uniform`: `uniform_equiv`, or
  `uniform_of_prCoin` from a law stated on events) is that coin (`dom_coin_map`), and a part of it `G` doesn't supply is
  fixed at the value best for the reduction (`dom_coin_split`);
* a coin of `g` alone is fixed the same way (`dom_coin_left`), and a coin of `G` alone is averaged over
  (`dom_coin_right`);
* relabelled outputs on either side (`dom_map_left`, `dom_map_right`), and C-Flock's games through core's image
  (`Dominated.toFlock`).

A deterministic strategy of `G` can't toss the coins `g` reads beyond `G`'s, so the reduction fixes them at their best
value (`exists_avg_le`): the average over them is at most the largest.
-/

namespace Verity.Game

open scoped ENNReal

variable {α β γ : Type}

/-- **`s` in `g` is dominated by `G`**: some strategy of `G` makes `q` at least as likely as `s` makes `p` in `g`. -/
def Dominated (p : α → Prop) (g : Game α) (s : Strategy g) (q : β → Prop) (G : Game β) : Prop :=
  ∃ τ : Strategy G, prob p g s ≤ prob q G τ

/-- **A map of coins that keeps the uniform law**: every average over `C` is its composite's average over `C'`. -/
def Uniform {C' C : Type} [Fintype C'] [Fintype C] (f : C' → C) : Prop :=
  ∀ h : C → ℝ≥0∞, avg (h ∘ f) = avg h

/-- An average at most what some strategy of `G` reaches at each point is at most what one strategy reaches: the one
reaching the most. -/
theorem exists_avg_le {R : Type} [Fintype R] [Nonempty R] {q : β → Prop} {G : Game β} (x : R → ℝ≥0∞)
    (h : ∀ r, ∃ τ : Strategy G, x r ≤ prob q G τ) : ∃ τ : Strategy G, avg x ≤ prob q G τ := by
  choose τ hτ using h
  obtain ⟨r₀, hr₀⟩ := Finite.exists_max fun r => prob q G (τ r)
  refine ⟨τ r₀, ?_⟩
  calc avg x ≤ avg fun _ : R => prob q G (τ r₀) := avg_mono fun r => (hτ r).trans (hr₀ r)
    _ = prob q G (τ r₀) := avg_const _

theorem Dominated.mono {p p' : α → Prop} {q q' : β → Prop} {g : Game α} {s : Strategy g} {G : Game β}
    (h : Dominated p g s q G) (hp : ∀ a, p' a → p a) (hq : ∀ b, q b → q' b) : Dominated p' g s q' G := by
  obtain ⟨τ, hτ⟩ := h
  exact ⟨τ, (prob_mono hp g s).trans (hτ.trans (prob_mono hq G τ))⟩

theorem dom_ret {p : α → Prop} {q : β → Prop} {a : α} {b : β} (h : p a → q b) (s : Strategy (Game.ret a)) :
    Dominated p (.ret a) s q (.ret b) := by
  classical
  refine ⟨(PUnit.unit : PUnit), ?_⟩
  simp only [prob]
  by_cases hp : p a
  · simp [hp, h hp]
  · simp [hp]

/-- An outcome outside `p` is dominated by any game with a strategy. -/
theorem dom_ret_not {p : α → Prop} {q : β → Prop} {a : α} (h : ¬ p a) (s : Strategy (Game.ret a)) {G : Game β}
    (τ : Strategy G) : Dominated p (.ret a) s q G := by
  classical
  refine ⟨τ, ?_⟩
  simp only [prob, h, ite_false]
  exact zero_le

theorem dom_send_left {p : α → Prop} {q : β → Prop} {M : Type} {k : M → Game α} {s : Strategy (Game.send M k)}
    {G : Game β} (h : Dominated p (k s.1) s.2 q G) : Dominated p (.send M k) s q G := h

theorem dom_send_right {p : α → Prop} {q : β → Prop} {g : Game α} {s : Strategy g} {M : Type} {K : M → Game β}
    (m : M) (h : Dominated p g s q (K m)) : Dominated p g s q (.send M K) := by
  obtain ⟨τ, hτ⟩ := h
  exact ⟨(⟨m, τ⟩ : (m : M) × Strategy (K m)), hτ⟩

theorem dom_coin_left {p : α → Prop} {q : β → Prop} {C : Type} [Fintype C] [Nonempty C] {k : C → Game α}
    {s : Strategy (Game.coin C k)} {G : Game β} (h : ∀ c, Dominated p (k c) (s c) q G) :
    Dominated p (.coin C k) s q G :=
  exists_avg_le (fun c => prob p (k c) (s c)) h

theorem dom_coin_right {p : α → Prop} {q : β → Prop} {g : Game α} {s : Strategy g} {C : Type} [Fintype C]
    [Nonempty C] {K : C → Game β} (h : ∀ c, Dominated p g s q (K c)) : Dominated p g s q (.coin C K) := by
  choose τ hτ using h
  refine ⟨(τ : (c : C) → Strategy (K c)), ?_⟩
  show prob p g s ≤ avg fun c => prob q (K c) (τ c)
  calc prob p g s = avg fun _ : C => prob p g s := (avg_const _).symm
    _ ≤ avg fun c => prob q (K c) (τ c) := avg_mono hτ

theorem dom_coin_map {p : α → Prop} {q : β → Prop} {C C' : Type} [Fintype C] [Nonempty C] [Fintype C']
    [Nonempty C'] {k : C → Game α} {s : Strategy (Game.coin C k)} {K : C' → Game β} (f : C' → C) (hf : Uniform f)
    (h : ∀ c', Dominated p (k (f c')) (s (f c')) q (K c')) : Dominated p (.coin C k) s q (.coin C' K) := by
  choose τ hτ using h
  refine ⟨(τ : (c' : C') → Strategy (K c')), ?_⟩
  show (avg fun c => prob p (k c) (s c)) ≤ avg fun c' => prob q (K c') (τ c')
  calc (avg fun c => prob p (k c) (s c)) = avg ((fun c => prob p (k c) (s c)) ∘ f) := (hf _).symm
    _ ≤ avg fun c' => prob q (K c') (τ c') := avg_mono hτ

theorem dom_coin_split {p : α → Prop} {q : β → Prop} {C C' R : Type} [Fintype C] [Nonempty C] [Fintype C']
    [Nonempty C'] [Fintype R] [Nonempty R] {k : C → Game α} {s : Strategy (Game.coin C k)} {K : C' → Game β}
    (f : C' × R → C) (hf : Uniform f) (h : ∀ c' r, Dominated p (k (f (c', r))) (s (f (c', r))) q (K c')) :
    Dominated p (.coin C k) s q (.coin C' K) := by
  have h' : ∀ c', ∃ τ : Strategy (K c'),
      (avg fun r => prob p (k (f (c', r))) (s (f (c', r)))) ≤ prob q (K c') τ :=
    fun c' => exists_avg_le _ (h c')
  choose τ hτ using h'
  refine ⟨(τ : (c' : C') → Strategy (K c')), ?_⟩
  show (avg fun c => prob p (k c) (s c)) ≤ avg fun c' => prob q (K c') (τ c')
  calc (avg fun c => prob p (k c) (s c)) = avg ((fun c => prob p (k c) (s c)) ∘ f) := (hf _).symm
    _ = avg fun c' => avg fun r => prob p (k (f (c', r))) (s (f (c', r))) := avg_prod _
    _ ≤ avg fun c' => prob q (K c') (τ c') := avg_mono hτ

theorem dom_map_left {p : β → Prop} {q : γ → Prop} (f : α → β) {g : Game α} {s : Strategy (g.map f)} {G : Game γ}
    (h : Dominated (fun a => p (f a)) g (Strategy.ofMap f g s) q G) : Dominated p (g.map f) s q G := by
  obtain ⟨τ, hτ⟩ := h
  exact ⟨τ, (prob_map p f g s).trans_le hτ⟩

theorem dom_map_right {p : α → Prop} {q : γ → Prop} (f : β → γ) {g : Game α} {s : Strategy g} {G : Game β}
    (h : Dominated p g s (fun b => q (f b)) G) : Dominated p g s q (G.map f) := by
  obtain ⟨τ, hτ⟩ := h
  refine ⟨Strategy.toMap f G τ, ?_⟩
  rw [prob_map, Pouw.PearlC.FlockTile.ofMap_toMap]
  exact hτ

/-- **Into C-Flock's game**: domination by core's image of `G` gives a strategy of `G` itself. -/
theorem Dominated.toFlock {p : α → Prop} {g : Game α} {s : Strategy g} {q : β → Prop} {G : FlockSoundness.Game β}
    (h : Dominated p g s q (Pouw.PearlC.FlockTile.toVerity G)) :
    ∃ τ : FlockSoundness.Game.Strategy G, prob p g s ≤ FlockSoundness.Game.prob q G τ := by
  obtain ⟨τ, hτ⟩ := h
  refine ⟨Pouw.PearlC.FlockTile.stratOfVerity G τ, ?_⟩
  rw [Pouw.PearlC.FlockTile.prob_ofVerity]
  exact hτ

/-! ## Maps that keep the uniform law -/

theorem uniform_equiv {C' C : Type} [Fintype C'] [Fintype C] (e : C' ≃ C) : Uniform e := fun h => by
  unfold avg
  rw [Fintype.card_congr e]
  exact Fintype.sum_equiv e _ _ fun _ => rfl

theorem uniform_of_bijective {C' C : Type} [Fintype C'] [Fintype C] {f : C' → C} (hf : Function.Bijective f) :
    Uniform f :=
  uniform_equiv (Equiv.ofBijective f hf)

theorem Uniform.comp {A B C : Type} [Fintype A] [Fintype B] [Fintype C] {f : A → B} {g : B → C} (hg : Uniform g)
    (hf : Uniform f) : Uniform (g ∘ f) := fun h => (hf (h ∘ g)).trans (hg h)

/-- An average is the sum over the values `h` takes of each value times the probability `h` takes it. -/
theorem avg_eq_sum_level {C : Type} [Fintype C] (h : C → ℝ≥0∞) (S : Finset ℝ≥0∞) (hS : ∀ c, h c ∈ S) :
    avg h = ∑ v ∈ S, v * prCoin fun c => h c = v := by
  classical
  simp only [prCoin, avg, Finset.mul_sum]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun c _ => ?_
  rw [Finset.sum_eq_single (h c)]
  · rw [if_pos rfl, mul_one]
    exact mul_comm _ _
  · intro b _ hb
    rw [if_neg (Ne.symm hb)]
    simp
  · intro hn
    exact absurd (hS c) hn

/-- **The uniform law, stated on events, keeps every average** (A3's form, `prCoin (E ∘ f) = prCoin E`). -/
theorem uniform_of_prCoin {C' C : Type} [Fintype C'] [Fintype C] (f : C' → C)
    (hf : ∀ E : C → Prop, prCoin (E ∘ f) = prCoin E) : Uniform f := fun h => by
  classical
  have hS : ∀ c, h c ∈ Finset.univ.image h := fun c => Finset.mem_image_of_mem h (Finset.mem_univ c)
  rw [avg_eq_sum_level (h ∘ f) _ fun c' => hS (f c'), avg_eq_sum_level h _ hS]
  refine Finset.sum_congr rfl fun v _ => ?_
  congr 1
  exact hf fun c => h c = v

end Verity.Game
