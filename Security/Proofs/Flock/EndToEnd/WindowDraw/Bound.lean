import Proofs.Flock.EndToEnd.WindowDraw.Law
import Proofs.Flock.EndToEnd.Window
import Definitions.Flock.Game.Seq

/-!
# The window's bound from its members' session bounds

A window strategy, at a draw whose member share has `κ` units, plays the member's session at each resampled share
(`exists_resample`): the window's draws with that share are its draws outside the member with the member's units drawn
by `Law.subset size κ` (`stratified_share`), so the member's event, which needs a nonempty share, has probability at most
`∑_κ Pr[|share| = κ] · β κ` (`member_prob_le`), and the union bound over the members gives the window's
(`window_bound_prob`). At `k = 1` that is `∑_j size_j / n_{s_j} · β j 1` (`window_bound_one`).

`Flock.SecurityProofs.WindowBound : Flock.Guarantees.WindowBound`.
-/

namespace FlockSoundness.Audit.Window

open Finset Game
open scoped ENNReal

theorem prob_map_strategy {α β : Type} (p : β → Prop) (f : α → β) : ∀ (g : Game α) (s : Strategy g),
    prob p (g.map f) (Strategy.map f g s) = prob (fun a => p (f a)) g s
  | .ret _, _ => rfl
  | .send _ k, s => prob_map_strategy p f (k s.1) s.2
  | @Game.coin _ _ _ _ k, s => congrArg avg (funext fun c => prob_map_strategy p f (k c) (s c))

theorem prob_cast_body {Ix β : Type} (G : Ix → Game β) (p : Ix → β → Prop) {x y : Ix} (h : x = y)
    (s : Strategy (G x)) :
    prob (p y) (G y) (cast (congrArg (fun z => Strategy (G z)) h) s) = prob (p x) (G x) s := by
  subst h
  rfl

theorem prob_audit {α Reg : Type} {N : ℕ} (p : Reg × Finset (Fin N) × α → Prop) (L : Law N)
    (body : Finset (Fin N) → Reg → Game α) (τ : Strategy (audit L Reg body)) :
    prob p (audit L Reg body) τ =
      avg fun ω => prob p ((body (L.draw ω) τ.1).map fun a => (τ.1, L.draw ω, a)) (τ.2 ω) :=
  rfl

theorem avg_ite_const {C : Type} [Fintype C] (p : C → Prop) [DecidablePred p] (x : ℝ≥0∞) :
    avg (fun c => if p c then x else 0) = prCoin p * x := by
  rw [← avg_ite_eq_prCoin, ← avg_mul_const]
  exact congrArg avg (funext fun c => by by_cases h : p c <;> simp [h])

/-- **One member's bound inside the window**, against every prover strategy. -/
theorem member_prob_le {α Reg : Type} {N m : ℕ} (σ : Fin N → Fin m) (k : Fin m → ℕ)
    (hk : ∀ s, k s ≤ (Law.stratum σ s).card) (body : Finset (Fin N) → Reg → Game α) (mb : Member) (s₀ : Fin m)
    (hfit : mb.base + mb.size ≤ N) (hin : ∀ u : Fin N, mb.base ≤ u.val → u.val < mb.base + mb.size → σ u = s₀)
    (E : Reg × Finset (Fin N) × α → Prop) (hrun : ∀ o, E o → (mb.share o.2.1).Nonempty) (β : ℕ → ℝ≥0∞)
    (hsess : ∀ κ (hκ : κ ≤ mb.size), 1 ≤ κ → κ ≤ k s₀ → ∀ ctx : Finset (Fin N), Disjoint ctx mb.range →
      ∀ τ : Strategy (audit (Law.subset mb.size κ hκ) Reg fun S R => body (ctx ∪ mb.unshift S) R),
        prob (fun o => E (o.1, ctx ∪ mb.unshift o.2.1, o.2.2))
          (audit (Law.subset mb.size κ hκ) Reg fun S R => body (ctx ∪ mb.unshift S) R) τ ≤ β κ)
    (τ : Strategy (audit (Law.stratified σ k hk) Reg body)) :
    prob E (audit (Law.stratified σ k hk) Reg body) τ ≤
      ∑ κ ∈ Icc 1 (k s₀), prCoin (fun ω => (mb.share ((Law.stratified σ k hk).draw ω)).card = κ) * β κ := by
  classical
  obtain ⟨R, t⟩ := τ
  have hRT := range_subset_stratum mb s₀ hin
  let P : (Law.stratified σ k hk).Ω → ℝ≥0∞ := fun ω =>
    prob (fun a => E (R, (Law.stratified σ k hk).draw ω, a)) (body ((Law.stratified σ k hk).draw ω) R)
      (Strategy.ofMap (fun a => (R, (Law.stratified σ k hk).draw ω, a)) _ (t ω))
  let g : Finset (Fin N) → ℝ≥0∞ := fun D =>
    if h : ∃ ω, (Law.stratified σ k hk).draw ω = D then P h.choose else 0
  have hPg : ∀ ω, P ω = g ((Law.stratified σ k hk).draw ω) := fun ω => by
    have h : ∃ ω', (Law.stratified σ k hk).draw ω' = (Law.stratified σ k hk).draw ω := ⟨ω, rfl⟩
    show _ = dite _ _ _
    rw [dite_eq_left h]
    exact congrArg P (draw_injective h.choose_spec).symm
  have hwin : prob E (audit (Law.stratified σ k hk) Reg body) ⟨R, t⟩ =
      avg fun ω => g ((Law.stratified σ k hk).draw ω) :=
    (prob_audit _ _ _ _).trans (congrArg avg (funext fun ω => (prob_map _ _ _ _).trans (hPg ω)))
  have hg0 : ∀ D, (mb.share D).card = 0 → g D = 0 := fun D hD => by
    show dite _ _ _ = 0
    split_ifs with h
    · refine le_antisymm ((prob_mono (q := fun _ => False) (fun a ha => ?_) _ _).trans (prob_false _ _).le)
        zero_le
      have hne := hrun _ ha
      rw [h.choose_spec] at hne
      exact card_ne_zero.2 hne hD
    · rfl
  have hsplit : ∀ ω, g ((Law.stratified σ k hk).draw ω) ≤
      ∑ κ ∈ Icc 1 (k s₀), if (mb.share ((Law.stratified σ k hk).draw ω)).card = κ then
        g ((Law.stratified σ k hk).draw ω) else 0 := fun ω => by
    by_cases h0 : (mb.share ((Law.stratified σ k hk).draw ω)).card = 0
    · rw [hg0 _ h0]
      exact zero_le
    · rw [sum_ite_eq, ite_eq_left (mem_Icc.2 ⟨Nat.one_le_iff_ne_zero.2 h0, share_card_le mb s₀ hfit hin ω⟩)]
  calc prob E (audit (Law.stratified σ k hk) Reg body) ⟨R, t⟩
      = avg fun ω => g ((Law.stratified σ k hk).draw ω) := hwin
    _ ≤ avg fun ω => ∑ κ ∈ Icc 1 (k s₀), if (mb.share ((Law.stratified σ k hk).draw ω)).card = κ then
          g ((Law.stratified σ k hk).draw ω) else 0 := avg_mono hsplit
    _ = ∑ κ ∈ Icc 1 (k s₀), avg fun ω => if (mb.share ((Law.stratified σ k hk).draw ω)).card = κ then
          g ((Law.stratified σ k hk).draw ω) else 0 := Law.avg_sum _ _
    _ ≤ ∑ κ ∈ Icc 1 (k s₀), prCoin (fun ω => (mb.share ((Law.stratified σ k hk).draw ω)).card = κ) * β κ := by
        refine sum_le_sum fun κ hκI => ?_
        obtain ⟨h1, hk₀⟩ := mem_Icc.1 hκI
        rw [← avg_ite_const]
        by_cases hκ : κ ≤ mb.size
        · rw [stratified_share mb s₀ hfit hin κ hκ g]
          refine avg_mono fun ω => ?_
          split_ifs with hc
          · have hc' : ((ω s₀).1 ∩ mb.range).card = κ := by
              rwa [Member.card_share hfit, draw_inter_range mb s₀ hRT] at hc
            let ctx := (Law.stratified σ k hk).draw ω \ mb.range
            have hex : ∀ S : (Law.subset mb.size κ hκ).Ω,
                ∃ ω', (Law.stratified σ k hk).draw ω' = ctx ∪ mb.unshift ((Law.subset mb.size κ hκ).draw S) :=
              fun S => exists_resample mb s₀ hfit hRT ω S.1 (hc'.trans S.2.symm)
            let sS : ∀ S : (Law.subset mb.size κ hκ).Ω,
                Strategy (body (ctx ∪ mb.unshift ((Law.subset mb.size κ hκ).draw S)) R) := fun S =>
              cast (congrArg (fun D => Strategy (body D R)) (hex S).choose_spec)
                (Strategy.ofMap (fun a => (R, (Law.stratified σ k hk).draw (hex S).choose, a)) _
                  (t (hex S).choose))
            let τ' : Strategy (audit (Law.subset mb.size κ hκ) Reg fun S R => body (ctx ∪ mb.unshift S) R) :=
              ⟨R, fun S => Strategy.map (fun a => (R, (Law.subset mb.size κ hκ).draw S, a)) _ (sS S)⟩
            have hS : ∀ S : (Law.subset mb.size κ hκ).Ω, g (ctx ∪ mb.unshift ((Law.subset mb.size κ hκ).draw S)) =
                prob (fun o => E (o.1, ctx ∪ mb.unshift o.2.1, o.2.2))
                  ((body (ctx ∪ mb.unshift ((Law.subset mb.size κ hκ).draw S)) R).map
                    fun a => (R, (Law.subset mb.size κ hκ).draw S, a))
                  (Strategy.map (fun a => (R, (Law.subset mb.size κ hκ).draw S, a)) _ (sS S)) := fun S => by
              rw [prob_map_strategy]
              refine Eq.trans ?_
                (prob_cast_body (fun D => body D R) (fun D a => E (R, D, a)) (hex S).choose_spec _).symm
              exact dite_eq_left (hex S)
            exact le_trans (le_of_eq (congrArg avg (funext hS))) (hsess κ hκ h1 hk₀ ctx sdiff_disjoint τ')
          · exact le_rfl
        · refine avg_mono fun ω => ?_
          have hne : (mb.share ((Law.stratified σ k hk).draw ω)).card ≠ κ := fun h => hκ (by
            rw [← h]
            simpa using card_le_univ (mb.share ((Law.stratified σ k hk).draw ω)))
          rw [ite_eq_right hne]
          exact zero_le

/-- **The composed window bound**, against every prover strategy: the union bound over the members. -/
theorem window_bound_prob {α Reg ι : Type} {N m : ℕ} (σ : Fin N → Fin m) (k : Fin m → ℕ)
    (hk : ∀ s, k s ≤ (Law.stratum σ s).card) (body : Finset (Fin N) → Reg → Game α) (ms : Finset ι)
    (mem : ι → Member) (str : ι → Fin m) (hfit : ∀ j ∈ ms, (mem j).base + (mem j).size ≤ N)
    (hin : ∀ j ∈ ms, ∀ u : Fin N, (mem j).base ≤ u.val → u.val < (mem j).base + (mem j).size → σ u = str j)
    (E : ι → Reg × Finset (Fin N) × α → Prop) (hrun : ∀ j ∈ ms, ∀ o, E j o → ((mem j).share o.2.1).Nonempty)
    (β : ι → ℕ → ℝ≥0∞)
    (hsess : ∀ j ∈ ms, ∀ κ (hκ : κ ≤ (mem j).size), 1 ≤ κ → κ ≤ k (str j) →
      ∀ ctx : Finset (Fin N), Disjoint ctx (mem j).range →
      ∀ τ : Strategy (audit (Law.subset (mem j).size κ hκ) Reg fun S R => body (ctx ∪ (mem j).unshift S) R),
        prob (fun o => E j (o.1, ctx ∪ (mem j).unshift o.2.1, o.2.2))
          (audit (Law.subset (mem j).size κ hκ) Reg fun S R => body (ctx ∪ (mem j).unshift S) R) τ ≤ β j κ)
    (τ : Strategy (audit (Law.stratified σ k hk) Reg body)) :
    prob (fun o => ∃ j ∈ ms, E j o) (audit (Law.stratified σ k hk) Reg body) τ ≤
      ∑ j ∈ ms, ∑ κ ∈ Icc 1 (k (str j)),
        prCoin (fun ω => ((mem j).share ((Law.stratified σ k hk).draw ω)).card = κ) * β j κ :=
  (Flock.SecurityProofs.window_prob_exists_le ms E _ τ).trans (sum_le_sum fun j hj =>
    member_prob_le σ k hk body (mem j) (str j) (hfit j hj) (hin j hj) (E j) (hrun j hj) (β j) (hsess j hj) τ)

/-- At `k = 1` the composed bound is `∑_j size_j / n_{s_j} · β j 1`. -/
theorem window_bound_one {ι : Type} {N m : ℕ} (σ : Fin N → Fin m) (k : Fin m → ℕ)
    (hk : ∀ s, k s ≤ (Law.stratum σ s).card) (ms : Finset ι) (mem : ι → Member) (str : ι → Fin m)
    (hfit : ∀ j ∈ ms, (mem j).base + (mem j).size ≤ N)
    (hin : ∀ j ∈ ms, ∀ u : Fin N, (mem j).base ≤ u.val → u.val < (mem j).base + (mem j).size → σ u = str j)
    (h1 : ∀ j ∈ ms, k (str j) = 1) (β : ι → ℕ → ℝ≥0∞) :
    ∑ j ∈ ms, ∑ κ ∈ Icc 1 (k (str j)),
        prCoin (fun ω => ((mem j).share ((Law.stratified σ k hk).draw ω)).card = κ) * β j κ =
      ∑ j ∈ ms, ((mem j).size : ℝ≥0∞) / ((Law.stratum σ (str j)).card : ℝ≥0∞) * β j 1 :=
  sum_congr rfl fun j hj => by
    rw [h1 j hj, Icc_self, sum_singleton,
      prCoin_share_one (hk := hk) (mem j) (str j) (hfit j hj) (hin j hj) (h1 j hj)]

end FlockSoundness.Audit.Window

namespace Flock.SecurityProofs

open FlockSoundness.Audit.Window

/-- **The window's bound from its members' session bounds.** -/
theorem WindowBound : Flock.Guarantees.WindowBound :=
  ⟨fun _ _ _ _ _ σ k hk body ms mem str E β hfit hin hrun hsess τ =>
      window_bound_prob σ k hk body ms mem str hfit hin E hrun β hsess τ,
    fun _ _ _ σ k hk ms mem str β hfit hin h1 => window_bound_one σ k hk ms mem str hfit hin h1 β⟩

end Flock.SecurityProofs
