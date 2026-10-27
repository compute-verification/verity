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
  (`Dominated.toFlock`);
* a coin of `G` read through a map keeping the uniform law, or a factor at a time (`dom_coin_right_map`,
  `dom_coin_prod`, `dom_coin_equiv`), and drawn in chunks (`Split`, `dom_split`, through `Covers`).

A deterministic strategy of `G` can't toss the coins `g` reads beyond `G`'s, so the reduction fixes them at their best
value (`exists_avg_le`): the average over them is at most the largest.

`dom_reach` runs the rule over rounds of released bits (`roundsThen`): at each round a plan either leaves the round to
`g`, its bits at their best value (`Move.skip`), or drives `G` to a coin those bits are an image of (`Move.step`), and
the domination at the end is asked only at the live states and continuations the plan reaches (`Reach`).
-/

universe u

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

/-! ## Live coins in rounds and in chunks -/

/-- **A coin of `G` drawn through a map that keeps the uniform law**: `G`'s coin may be read as `t`'s image. -/
theorem dom_coin_right_map {β : Type} {p : α → Prop} {q : β → Prop} {g : Game α} {s : Strategy g} {C C' : Type}
    [Fintype C] [Nonempty C] [Fintype C'] [Nonempty C'] {K : C' → Game β} (t : C → C') (ht : Uniform t)
    (h : Dominated p g s q (.coin C' K)) : Dominated p g s q (.coin C fun c => K (t c)) := by
  obtain ⟨τ, hτ⟩ := h
  refine ⟨(fun c => τ (t c) : (c : C) → Strategy (K (t c))), ?_⟩
  exact hτ.trans_eq (ht fun c' => prob q (K c') (τ c')).symm

/-- **A coin of `G` on a product, drawn a factor at a time.** -/
theorem dom_coin_prod {β : Type} {p : α → Prop} {q : β → Prop} {g : Game α} {s : Strategy g} {C₁ C₂ : Type}
    [Fintype C₁] [Nonempty C₁] [Fintype C₂] [Nonempty C₂] {K : C₁ × C₂ → Game β}
    (h : Dominated p g s q (.coin C₁ fun a => .coin C₂ fun b => K (a, b))) :
    Dominated p g s q (.coin (C₁ × C₂) K) := by
  obtain ⟨τ, hτ⟩ := h
  refine ⟨(fun x => τ x.1 x.2 : (x : C₁ × C₂) → Strategy (K x)), ?_⟩
  show prob p g s ≤ avg fun x : C₁ × C₂ => prob q (K x) (τ x.1 x.2)
  rw [avg_prod]
  exact hτ

/-- **A coin of `G` relabelled by an equivalence.** -/
theorem dom_coin_equiv {β : Type} {p : α → Prop} {q : β → Prop} {g : Game α} {s : Strategy g} {C C' : Type}
    [Fintype C] [Nonempty C] [Fintype C'] [Nonempty C'] {K : C → Game β} (e : C' ≃ C)
    (h : Dominated p g s q (.coin C' fun c' => K (e c'))) : Dominated p g s q (.coin C K) := by
  have h' := dom_coin_right_map (K := fun c' => K (e c')) e.symm (uniform_equiv e.symm) h
  simpa only [Equiv.apply_symm_apply] using h'

/-! ## Driving a live game through rounds of released bits -/

/-- **Rounds of released bits, then `k`**: in each round the prover sends a message, then the verifier draws `n`
uniform bits; `mk` records the round, and `k` goes on from the record of the rounds. -/
def roundsThen {M ρ : Type} (mk : M → List Bool → ρ) (k : List ρ → Game α) : List ℕ → Game α
  | [] => k []
  | n :: ns => .send M fun b => .coin (Fin n → Bool) fun x =>
      roundsThen mk (fun log => k (mk b (List.ofFn x) :: log)) ns

/-- **One round driven through the live game**: from the live state `st`, the reduction's messages bring the live game
to a coin `C` (`reach`); the round's `m` bits are that coin and extra bits `R` through a map that keeps the uniform law
(`f`), and the live game goes on at `next c`. -/
structure Step {β : Type} {St : Type u} (q : β → Prop) (G : St → Game β) (st : St) (m : ℕ) where
  C : Type
  [fC : Fintype C]
  [nC : Nonempty C]
  R : Type
  [fR : Fintype R]
  [nR : Nonempty R]
  f : C × R → (Fin m → Bool)
  hf : Uniform f
  next : C → St
  reach : ∀ {γ : Type} (p : γ → Prop) (g : Game γ) (s : Strategy g),
    Dominated p g s q (.coin C fun c => G (next c)) → Dominated p g s q (G st)

attribute [instance] Step.fC Step.nC Step.fR Step.nR

/-- What the reduction does with a round of the play at the live state `st`: leave it to the play, its bits fixed at
their best value, at a state of the same live game (`skip`), or drive it through the live game (`step`). -/
inductive Move {β : Type} {St : Type u} (q : β → Prop) (G : St → Game β) (st : St) (m : ℕ) where
  | skip (st' : St) (h : G st' = G st)
  | step (S : Step q G st m)

section Drive

variable {M ρ β : Type} {St : Type u} {q : β → Prop} (mk : M → List Bool → ρ) (G : St → Game β)
  (plan : (st : St) → (m : ℕ) → M → (Σ g : Game α, Strategy g) → Move q G st m) (st₀ : St) (k₀ : List ρ → Game α)

/-- **The live states and play continuations the reduction reaches** from `st₀` and `k₀`, one round at a time; the plan
sees the round's message and the play's strategy from there on. -/
inductive Reach : St → (List ρ → Game α) → Prop
  | start : Reach st₀ k₀
  | skip {st : St} {K : List ρ → Game α} {m : ℕ} {b : M} {gs : Σ g : Game α, Strategy g} {st' : St}
      {h : G st' = G st} (x : Fin m → Bool) :
      Reach st K → plan st m b gs = .skip st' h → Reach st' fun l => K (mk b (List.ofFn x) :: l)
  | step {st : St} {K : List ρ → Game α} {m : ℕ} {b : M} {gs : Σ g : Game α, Strategy g} {S : Step q G st m}
      (c : S.C) (r : S.R) :
      Reach st K → plan st m b gs = .step S → Reach (S.next c) fun l => K (mk b (List.ofFn (S.f (c, r))) :: l)

/-- A reached continuation is `k₀` after the rounds played so far. -/
theorem Reach.hist {st : St} {K : List ρ → Game α} (h : Reach mk G plan st₀ k₀ st K) :
    ∃ hs : List ρ, K = fun l => k₀ (hs ++ l) := by
  induction h with
  | start => exact ⟨[], rfl⟩
  | @skip _ _ _ b _ _ _ x _ _ ih =>
    obtain ⟨hs, rfl⟩ := ih
    exact ⟨hs ++ [mk b (List.ofFn x)], funext fun l => by simp⟩
  | @step _ _ _ b _ S c r _ _ ih =>
    obtain ⟨hs, rfl⟩ := ih
    exact ⟨hs ++ [mk b (List.ofFn (S.f (c, r)))], funext fun l => by simp⟩

variable {p : α → Prop}

theorem dom_reach_aux
    (hfin : ∀ st K (s : Strategy (K [])), Reach mk G plan st₀ k₀ st K → Dominated p (K []) s q (G st)) :
    ∀ (ns : List ℕ) (st : St) (K : List ρ → Game α), Reach mk G plan st₀ k₀ st K →
      ∀ s : Strategy (roundsThen mk K ns), Dominated p (roundsThen mk K ns) s q (G st)
  | [], st, K, hR, s => hfin st K s hR
  | n :: ns, st, K, hR, s => by
    refine dom_send_left ?_
    cases hp : plan st n s.1 ⟨_, s.2⟩ with
    | skip st' hG =>
      refine dom_coin_left fun x => ?_
      rw [← hG]
      exact dom_reach_aux hfin ns st' _ (Reach.skip x hR hp) _
    | step S =>
      refine S.reach _ _ _ (dom_coin_split S.f S.hf fun c r => ?_)
      exact dom_reach_aux hfin ns (S.next c) _ (Reach.step c r hR hp) _

/-- **The play's rounds driven through the live game**: the live game at `st₀` dominates the rounds then `k₀`, given
the domination at every live state and continuation the reduction reaches. -/
theorem dom_reach
    (hfin : ∀ st K (s : Strategy (K [])), Reach mk G plan st₀ k₀ st K → Dominated p (K []) s q (G st))
    (ns : List ℕ) (s : Strategy (roundsThen mk k₀ ns)) : Dominated p (roundsThen mk k₀ ns) s q (G st₀) :=
  dom_reach_aux mk G plan st₀ k₀ hfin ns st₀ k₀ .start s

end Drive

/-- Domination carries across an equation of games. -/
theorem Dominated.of_eq {β : Type} {p : α → Prop} {q : β → Prop} {g g' : Game α} (h : g = g') {G : Game β}
    {s : Strategy g} (hd : Dominated p g' (h ▸ s) q G) : Dominated p g s q G := by
  subst h
  exact hd

/-! ## Covering one game of the verifier by another -/

/-- **`G'` is covered by `G`**: every strategy of `G'` is matched, in the probability of `q`, by one of `G`. -/
def Covers {β : Type} (q : β → Prop) (G' G : Game β) : Prop :=
  ∀ τ' : Strategy G', ∃ τ : Strategy G, prob q G' τ' ≤ prob q G τ

theorem Dominated.covers {α β : Type} {p : α → Prop} {q : β → Prop} {g : Game α} {s : Strategy g} {G' G : Game β}
    (h : Dominated p g s q G') (hc : Covers q G' G) : Dominated p g s q G := by
  obtain ⟨τ', hτ'⟩ := h
  obtain ⟨τ, hτ⟩ := hc τ'
  exact ⟨τ, hτ'.trans hτ⟩

theorem covers_coin {β : Type} {q : β → Prop} {C : Type} [Fintype C] [Nonempty C] {A B : C → Game β}
    (h : ∀ c, Covers q (A c) (B c)) : Covers q (.coin C A) (.coin C B) := fun τ' => by
  choose τ hτ using fun c => h c (τ' c)
  exact ⟨(τ : (c : C) → Strategy (B c)), avg_mono hτ⟩

theorem covers_coin_unit {β : Type} {q : β → Prop} {K : Unit → Game β} : Covers q (K ()) (.coin Unit K) :=
  fun τ' => ⟨(fun _ => τ' : (u : Unit) → Strategy (K u)), (avg_const (prob q (K ()) τ')).symm.le⟩

theorem covers_coin_prod {β : Type} {q : β → Prop} {C₁ C₂ : Type} [Fintype C₁] [Nonempty C₁] [Fintype C₂]
    [Nonempty C₂] {K : C₁ × C₂ → Game β} :
    Covers q (.coin C₁ fun a => .coin C₂ fun b => K (a, b)) (.coin (C₁ × C₂) K) := fun τ' => by
  refine ⟨(fun x => τ' x.1 x.2 : (x : C₁ × C₂) → Strategy (K x)), ?_⟩
  show (avg fun a => avg fun b => prob q (K (a, b)) (τ' a b)) ≤ avg fun x : C₁ × C₂ => prob q (K x) (τ' x.1 x.2)
  rw [avg_prod]

/-! ## A coin drawn in chunks -/

/-- **One chunk of a coin, read as a round of `m` released bits**: the chunk `P` and extra bits `R` through a map that
keeps the uniform law. -/
structure Chunk (m : ℕ) where
  P : Type
  [fP : Fintype P]
  [nP : Nonempty P]
  R : Type
  [fR : Fintype R]
  [nR : Nonempty R]
  f : P × R → (Fin m → Bool)
  hf : Uniform f

attribute [instance] Chunk.fP Chunk.nP Chunk.fR Chunk.nR

/-- The chunks' values together. -/
def Chunks {m : ℕ} : List (Chunk m) → Type
  | [] => Unit
  | c :: cs => c.P × Chunks cs

instance Chunks.fintype {m : ℕ} : (cs : List (Chunk m)) → Fintype (Chunks cs)
  | [] => inferInstanceAs (Fintype Unit)
  | c :: cs => @instFintypeProd c.P (Chunks cs) _ (Chunks.fintype cs)

instance Chunks.nonempty {m : ℕ} : (cs : List (Chunk m)) → Nonempty (Chunks cs)
  | [] => inferInstanceAs (Nonempty Unit)
  | c :: cs => @instNonemptyProd c.P (Chunks cs) _ (Chunks.nonempty cs)

/-- The chunks drawn one after the other, then `K` at their values. -/
def nested {β : Type} {m : ℕ} : (cs : List (Chunk m)) → (Chunks cs → Game β) → Game β
  | [], K => K ()
  | c :: cs, K => .coin c.P fun a => nested cs fun r => K (a, r)

theorem covers_nested {β : Type} {q : β → Prop} {m : ℕ} :
    ∀ (cs : List (Chunk m)) (K : Chunks cs → Game β), Covers q (nested cs K) (.coin (Chunks cs) K)
  | [], _ => covers_coin_unit
  | c :: cs, K => fun τ' => by
    obtain ⟨τ₁, h₁⟩ := covers_coin (B := fun a => Game.coin (Chunks cs) fun r => K (a, r))
      (fun a => covers_nested cs fun r => K (a, r)) τ'
    obtain ⟨τ, h⟩ := covers_coin_prod (K := K) τ₁
    exact ⟨τ, h₁.trans h⟩

/-- **A coin split into chunks**: the first chunk, the rest, and the coin as their values. -/
structure Split (X : Type) (m : ℕ) where
  first : Chunk m
  rest : List (Chunk m)
  e : Chunks (first :: rest) ≃ X

/-- The coin `X`, drawn chunk by chunk. -/
theorem dom_split {α β : Type} {p : α → Prop} {q : β → Prop} {g : Game α} {s : Strategy g} {X : Type} [Fintype X]
    [Nonempty X] {m : ℕ} (sp : Split X m) {K : X → Game β}
    (h : Dominated p g s q (nested (sp.first :: sp.rest) fun y => K (sp.e y))) : Dominated p g s q (.coin X K) :=
  dom_coin_equiv sp.e (h.covers (covers_nested _ _))

end Verity.Game
