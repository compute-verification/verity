import Proofs.Flock.Soundness.Game.Lock

/-!
# a1l, step 1: games under relabelling, and a lockstep beside another

`holoTable_value_le_oracle` (`Coupling.lean`) couples the compiled holographic table with an oracle game in which R's
stream is a third lockstep beside the two reps. This file has the game-level pieces it needs.

* `value_bind_mono`: the value is monotone through a bind's continuation.
* `interleave_map`: interleaving two relabelled games is relabelling their interleaving.
* `getEO`, `liftL`, `liftR`: two locksteps' tables side by side, the first's slot `s` at `2s` and the second's at
  `2s + 1`; `liftL` runs the first while the second's index stays, `liftR` the second while the first's stays.
-/

namespace FlockC3.Fork

open FlockSoundness FlockSoundness.Game
open scoped ENNReal

variable {α β γ δ : Type}

/-- **The value is monotone through a bind's continuation.** -/
theorem value_bind_mono {p : β → Prop} {q : γ → Prop} {f : α → Game β} {f' : α → Game γ}
    (h : ∀ a, value p (f a) ≤ value q (f' a)) : ∀ g : Game α, value p (g.bind f) ≤ value q (g.bind f')
  | .ret a => by simpa only [bind_ret] using h a
  | .send M k => by
      simp only [bind_send, value_send]
      exact iSup_mono fun m => value_bind_mono h (k m)
  | @Game.coin _ C _ _ k => by
      simp only [bind_coin, value_coin]
      exact Finset.sum_le_sum fun c _ => mul_le_mul_right (value_bind_mono h (k c)) _

theorem map_map (f : α → β) (f' : β → γ) (g : Game α) : (g.map f).map f' = g.map fun a => f' (f a) := by
  simp only [Game.map, Game.bind_assoc, Game.bind_ret]

theorem map_id' (g : Game α) : g.map id = g := Game.bind_ret_right g

theorem interleaveWith_map {adv : Game β → Game (α × β)} {fin : β → Game (α × β)}
    {adv' : Game δ → Game (γ × δ)} {fin' : δ → Game (γ × δ)} (F : α × β → γ × δ) (f₂ : β → δ)
    (hadv : ∀ g, adv' (g.map f₂) = (adv g).map F) (hfin : ∀ b, fin' (f₂ b) = (fin b).map F) :
    ∀ g : Game β, interleaveWith adv' fin' (g.map f₂) = (interleaveWith adv fin g).map F
  | .ret b => hfin b
  | .send M k => congrArg (Game.send Bool) (funext fun first => by
      cases first
      · exact congrArg (Game.send M) (funext fun m => interleaveWith_map F f₂ hadv hfin (k m))
      · exact hadv (.send M k))
  | @Game.coin _ C iF iN k => congrArg (Game.send Bool) (funext fun first => by
      cases first
      · exact congrArg (@Game.coin _ C iF iN) (funext fun c => interleaveWith_map F f₂ hadv hfin (k c))
      · exact hadv (@Game.coin _ C iF iN k))

/-- **Interleaving relabelled games** is relabelling their interleaving. -/
theorem interleave_map (f₁ : α → γ) (f₂ : β → δ) : ∀ (g₁ : Game α) (g₂ : Game β),
    interleave (g₁.map f₁) (g₂.map f₂) = (interleave g₁ g₂).map (Prod.map f₁ f₂)
  | .ret a, g₂ => by
      show (g₂.map f₂).map (fun b => (f₁ a, b)) = (g₂.map fun b => (a, b)).map (Prod.map f₁ f₂)
      rw [map_map, map_map]
      rfl
  | .send M k, g₂ => by
      refine interleaveWith_map (adv := fun g₂ => .send M fun m => interleave (k m) g₂)
        (fin := fun b => (Game.send M k).map fun a => (a, b)) (Prod.map f₁ f₂) f₂ (fun g => ?_) (fun b => ?_) g₂
      · exact congrArg (Game.send M) (funext fun m => interleave_map f₁ f₂ (k m) g)
      · show ((Game.send M k).map f₁).map (fun a => (a, f₂ b)) =
          ((Game.send M k).map fun a => (a, b)).map (Prod.map f₁ f₂)
        rw [map_map, map_map]
        rfl
  | @Game.coin _ C iF iN k, g₂ => by
      refine interleaveWith_map (adv := fun g₂ => @Game.coin _ C iF iN fun c => interleave (k c) g₂)
        (fin := fun b => (@Game.coin _ C iF iN k).map fun a => (a, b)) (Prod.map f₁ f₂) f₂ (fun g => ?_)
        (fun b => ?_) g₂
      · exact congrArg (@Game.coin _ C iF iN) (funext fun c => interleave_map f₁ f₂ (k c) g)
      · show ((@Game.coin _ C iF iN k).map f₁).map (fun a => (a, f₂ b)) =
          ((@Game.coin _ C iF iN k).map fun a => (a, b)).map (Prod.map f₁ f₂)
        rw [map_map, map_map]
        rfl

/-! ## Two locksteps' tables side by side -/

section Lift

variable {T D I₁ I₂ : Type} {get₁ : I₁ → ℕ → Option (T × D)} {get₂ : I₂ → ℕ → Option (T × D)}

/-- Two locksteps' tables side by side: the first's slot `s` at `2s`, the second's at `2s + 1`. -/
def getEO (get₁ : I₁ → ℕ → Option (T × D)) (get₂ : I₂ → ℕ → Option (T × D)) (i : I₁ × I₂)
    (s : ℕ) : Option (T × D) :=
  if s % 2 = 0 then get₁ i.1 (s / 2) else get₂ i.2 (s / 2)

theorem getEO_even (i : I₁ × I₂) (s : ℕ) : getEO get₁ get₂ i (2 * s) = get₁ i.1 s := by
  have h1 : 2 * s % 2 = 0 := by omega
  have h2 : 2 * s / 2 = s := by omega
  simp only [getEO, h1, h2, ite_true]

theorem getEO_odd (i : I₁ × I₂) (s : ℕ) : getEO get₁ get₂ i (2 * s + 1) = get₂ i.2 s := by
  have h1 : ¬ (2 * s + 1) % 2 = 0 := by omega
  have h2 : (2 * s + 1) / 2 = s := by omega
  simp only [getEO, h1, h2, ite_false]

/-- **The first lockstep, with the second's index fixed.** -/
def liftL {R₁ : I₁ → α → β → Prop} (i₂ : I₂) :
    {i₁ : I₁} → {g : Game α} → {g' : Game β} → Lock T D get₁ R₁ i₁ g g' →
      Lock T D (getEO get₁ get₂) (fun i a b => i.2 = i₂ ∧ R₁ i.1 a b) (i₁, i₂) g g'
  | _, _, _, .ret h => .ret ⟨rfl, h⟩
  | _, _, _, @Lock.coin _ _ _ _ _ _ _ _ C iF iN _ _ L =>
      @Lock.coin _ _ _ _ _ _ _ _ C iF iN _ _ fun c => liftL i₂ (L c)
  | _, _, _, .send M L => .send M fun m => liftL i₂ (L m)
  | _, _, _, .table slot upd hf hs hk L =>
      .table (2 * slot) (fun t d => (upd t d, i₂)) (by rw [getEO_even]; exact hf)
        (fun t d => by rw [getEO_even]; exact hs t d)
        (fun t d s h => by
          unfold getEO
          split_ifs with he
          · exact hk t d (s / 2) (by omega)
          · rfl)
        fun t d => liftL i₂ (L t d)

/-- **The second lockstep, with the first's index fixed.** -/
def liftR {R₂ : I₂ → α → β → Prop} (i₁ : I₁) :
    {i₂ : I₂} → {g : Game α} → {g' : Game β} → Lock T D get₂ R₂ i₂ g g' →
      Lock T D (getEO get₁ get₂) (fun i a b => i.1 = i₁ ∧ R₂ i.2 a b) (i₁, i₂) g g'
  | _, _, _, .ret h => .ret ⟨rfl, h⟩
  | _, _, _, @Lock.coin _ _ _ _ _ _ _ _ C iF iN _ _ L =>
      @Lock.coin _ _ _ _ _ _ _ _ C iF iN _ _ fun c => liftR i₁ (L c)
  | _, _, _, .send M L => .send M fun m => liftR i₁ (L m)
  | _, _, _, .table slot upd hf hs hk L =>
      .table (2 * slot + 1) (fun t d => (i₁, upd t d)) (by rw [getEO_odd]; exact hf)
        (fun t d => by rw [getEO_odd]; exact hs t d)
        (fun t d s h => by
          unfold getEO
          split_ifs with he
          · rfl
          · exact hk t d (s / 2) (by omega))
        fun t d => liftR i₁ (L t d)

end Lift

end FlockC3.Fork
