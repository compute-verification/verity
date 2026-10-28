import Definitions.Accounting.Guarantees.CoinFlow
import Proofs.Accounting.HMu.Luck

/-!
# The coin rules, proved on the dataflow model

`viewIndependent` and `coinsThroughNoise` are strong inductions over the node order: a node's value is its `op` on what
it reads (`eval_congr`), so two draws give it one value when they agree on its coins and on its earlier reads. Under rule
(b) no node outside a noise injection's forward cone reads its coins; under rule (a) and sink confinement a non-fold
node's coins are its own fresh coins, and its reads are non-fold. `pointMass` moves the run's coins to
`Ω₀ × (J → Bool)`, the coins a noise injection doesn't make fresh and the ones it does, through an equivalence that
keeps the uniform measure (`prob_comp_equiv`). The view is then a function of `Ω₀` (`viewIndependent`), and
`Accounting.HMuProofs.coinPointMass` applies.
-/

namespace Accounting.CoinFlowProofs

open Accounting.CoinFlow Accounting.HMu Pouw.RelativeSecurity Finset

variable {m : ℕ} {C V : Type}

theorem admits_spec [DecidableEq C] {P : Dataflow m C V} (h : P.admits = true) :
    P.Topological ∧ P.CoinFlowRule ∧ P.OneUse ∧ P.SinkConfined :=
  of_decide_eq_true h

theorem eval_eq [Inhabited V] (P : Dataflow m C V) (σ : C → Bool) (n : Fin m) :
    P.eval σ n = P.op n (fun k => if _h : k.1 < n then P.eval σ k.1 else default) (fun c => σ c.1) := by
  rw [Dataflow.eval]

/-- Two draws give node `g` one value when they agree on its coins and give its earlier reads one value. -/
theorem eval_congr [Inhabited V] (P : Dataflow m C V) {σ σ' : C → Bool} {g : Fin m}
    (hc : ∀ c ∈ P.coins g, σ c = σ' c) (hr : ∀ k ∈ P.reads g, k < g → P.eval σ k = P.eval σ' k) :
    P.eval σ g = P.eval σ' g := by
  rw [eval_eq, eval_eq]
  congr 1
  · funext k
    split_ifs with h
    · exact hr k.1 k.2 h
    · rfl
  · funext c
    exact hc c.1 c.2

/-- Strong induction over the node order. -/
theorem fin_strong {Q : Fin m → Prop} (h : ∀ g, (∀ k, k < g → Q k) → Q g) (g : Fin m) : Q g := by
  suffices ∀ n (g : Fin m), g.val = n → Q g from this g.val g rfl
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    intro g hg
    exact h g fun k hk => ih k.val (hg ▸ hk) k rfl

theorem mem_noiseCoins {P : Dataflow m C V} {k : Fin m} {c : C} :
    c ∈ P.noiseCoins k ↔ P.noise k = true ∧ c ∈ P.coins k := by
  unfold Dataflow.noiseCoins
  by_cases hk : P.noise k = true
  · simp [hk]
  · simp [hk]

theorem viewIndependent : Accounting.Guarantees.CoinFlow.ViewIndependent := by
  intro m C V _ _ P hP k σ σ' hσ
  obtain ⟨-, -, hone, -⟩ := admits_spec hP
  refine fin_strong fun g ih hd => eval_congr P (fun c hc => hσ c fun hck => ?_) fun r hr hlt =>
    ih r hlt fun hdr => hd (hdr.tail hr)
  obtain ⟨hk, hck⟩ := mem_noiseCoins.mp hck
  have : g = k := (hone k hk).1 c hck g hc
  exact hd (this ▸ Relation.ReflTransGen.refl)

theorem coinsThroughNoise : Accounting.Guarantees.CoinFlow.CoinsThroughNoise := by
  intro m C V _ _ P hP g σ σ'
  obtain ⟨-, hflow, -, hsink⟩ := admits_spec hP
  revert g
  refine fin_strong fun g ih hg hagree => eval_congr P (fun c hc => ?_) fun r hr hlt => ?_
  · have hn : P.noise g = true := by
      rcases hflow g ⟨c, hc⟩ with h | h
      · exact h
      · rw [hg] at h; exact absurd h (by decide)
    exact hagree g Relation.ReflTransGen.refl c (mem_noiseCoins.mpr ⟨hn, hc⟩)
  · have hr' : P.fold r = false := by
      cases h : P.fold r with
      | false => rfl
      | true => rw [hsink g r hr h] at hg; exact absurd hg (by decide)
    exact ih r hlt hr' fun k hk => hagree k (hk.tail hr)

theorem oneReader : Accounting.Guarantees.CoinFlow.OneReader := by
  intro m C V _ P hP k r g hk hr hd
  obtain ⟨-, -, hone, -⟩ := admits_spec hP
  rcases Relation.ReflTransGen.cases_head hd with h | ⟨y, hy, hyg⟩
  · exact Or.inl h.symm
  · exact Or.inr ((hone k hk).2 y r hy hr ▸ hyg)

section Prob

theorem prob_comp_equiv {α β : Type} [Fintype α] [Fintype β] (e : α ≃ β) (p : β → Prop) :
    prob (fun a => p (e a)) = prob p := by
  classical
  rw [HMuProofs.prob_eq_card, HMuProofs.prob_eq_card, Fintype.card_congr e]
  congr 1
  rw [← Fintype.card_subtype, ← Fintype.card_subtype]
  exact_mod_cast Fintype.card_congr (e.subtypeEquiv fun _ => Iff.rfl)

variable [DecidableEq C] (S : Finset C)

/-- A draw of every coin from the coins outside `S` and the coins in it. -/
def glue (ω : ({c // c ∉ S} → Bool) × ({c // c ∈ S} → Bool)) (c : C) : Bool :=
  if h : c ∈ S then ω.2 ⟨c, h⟩ else ω.1 ⟨c, h⟩

/-- The run's coins as the coins outside `S` and the coins in it. -/
def split : (({c // c ∉ S} → Bool) × ({c // c ∈ S} → Bool)) ≃ (C → Bool) where
  toFun := glue S
  invFun σ := (fun c => σ c.1, fun c => σ c.1)
  left_inv ω := by
    refine Prod.ext (funext fun c => ?_) (funext fun c => ?_)
    · simp [glue, c.2]
    · simp [glue, c.2]
  right_inv σ := by
    funext c
    by_cases h : c ∈ S
    · simp [glue, h]
    · simp [glue, h]

theorem glue_out (ω : ({c // c ∉ S} → Bool) × ({c // c ∈ S} → Bool)) (c : {c // c ∉ S}) : glue S ω c.1 = ω.1 c := by
  simp [glue, c.2]

theorem glue_in (ω : ({c // c ∉ S} → Bool) × ({c // c ∈ S} → Bool)) (c : {c // c ∈ S}) : glue S ω c.1 = ω.2 c := by
  simp [glue, c.2]

end Prob

theorem pointMass : Accounting.Guarantees.CoinFlow.PointMass := by
  intro m C V _ _ _ P hP k O F a v o t
  classical
  let V₀ : ({c // c ∉ P.noiseCoins k} → Bool) → P.View k := fun ω₀ =>
    P.view k (glue (P.noiseCoins k) (ω₀, fun _ => false))
  have hview : ∀ ω, P.view k (split (P.noiseCoins k) ω) = V₀ ω.1 := by
    intro ω
    refine Prod.ext (funext fun g => ?_) (funext fun c => ?_)
    · exact viewIndependent m C V P hP k _ _ (fun c hc => by
        rw [split, Equiv.coe_fn_mk, glue_out _ ω ⟨c, hc⟩, glue_out _ (ω.1, fun _ => false) ⟨c, hc⟩]) g.1 g.2
    · change glue _ ω c.1 = glue _ (ω.1, fun _ => false) c.1
      rw [glue_out, glue_out]
  have hcoins : ∀ ω, (fun j : {c // c ∈ P.noiseCoins k} => split (P.noiseCoins k) ω j.1) = ω.2 :=
    fun ω => funext (glue_in _ ω)
  have key := HMuProofs.coinPointMass ({c // c ∉ P.noiseCoins k} → Bool) O {c // c ∈ P.noiseCoins k}
    (fun ω₀ => F (V₀ ω₀)) (fun ω₀ => a (V₀ ω₀)) (fun ω₀ => v (V₀ ω₀)) o t
  rw [Fintype.card_coe] at key
  rw [← prob_comp_equiv (split (P.noiseCoins k)), ← prob_comp_equiv (split (P.noiseCoins k))]
  simp only [hview, hcoins]
  exact key

end Accounting.CoinFlowProofs
