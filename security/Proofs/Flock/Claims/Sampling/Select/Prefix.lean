module

public import Proofs.Flock.Claims.Sampling.Select.Take

/-! # A draw from bytes draws the same from any extension of them

`Draw.drawUnits` reads its source in order (`Draw.take`) and stops when its law's draw ends, so a draw that ends on
`ξ`'s bytes reads only `ξ` and ends the same on `ξ ++ t` (`drawUnits_append`). -/

@[expose] public section

open Spot

namespace Properties.Spot.SecurityProofs

/-- A bind whose first step and continuation each return on `y` and `g` what they return on `x` and `f` returns the same. -/
theorem run_bind_of {σ α β : Type} {x y : StateT σ (Except String) α} {f g : α → StateT σ (Except String) β}
    {s : σ} {out : β × σ} (h : (x >>= f).run s = .ok out)
    (hxy : ∀ r, x.run s = .ok r → y.run s = .ok r)
    (hfg : ∀ r, x.run s = .ok r → (f r.1).run r.2 = .ok out → (g r.1).run r.2 = .ok out) :
    (y >>= g).run s = .ok out := by
  rw [StateT.run_bind] at h ⊢
  cases hx : x.run s with
  | error e => rw [hx] at h; exact absurd h (by rw [bind_error]; simp)
  | ok r =>
    rw [hx, bind_ok] at h
    rw [hxy r hx, bind_ok]
    exact hfg r hx h

theorem extract_append_of_le {a b : ByteArray} {i j : ℕ} (hj : j ≤ a.size) :
    (a ++ b).extract i j = a.extract i j := by
  have he : b.extract (i - a.size) (j - a.size) = ByteArray.empty :=
    ByteArray.extract_eq_empty_iff.2 (by omega)
  rw [ByteArray.extract_append, he, ByteArray.append_empty]

theorem take_append {ξ t : ByteArray} {need p : ℕ} {x : ByteArray × ℕ}
    (h : (Draw.take ξ need).run p = .ok x) : (Draw.take (ξ ++ t) need).run p = .ok x := by
  rw [take_run] at h ⊢
  split at h
  · cases h
  · rename_i hle
    have hle' : ¬ (ξ ++ t).size < p + need := by rw [ByteArray.size_append]; omega
    rw [ite_eq_right hle', extract_append_of_le (show p + need ≤ ξ.size by omega)]
    exact h

/-- A `repeat` loop whose body returns on `g` what it returns on `f`, and whose every `yield` on `f` moves the state up,
to at most `B`, returns the same on `g`. -/
theorem loop_append {β : Type} (f g : Unit → β → StateT ℕ (Except String) (ForInStep β)) (B : ℕ)
    (hfg : ∀ b s x, (f () b).run s = .ok x → (g () b).run s = .ok x)
    (hy : ∀ b s b' s', (f () b).run s = .ok (.yield b', s') → s < s' ∧ s' ≤ B) :
    ∀ N b s, B + 1 - s < N → ∀ x, (Lean.Loop.forIn Lean.Loop.mk b f).run s = .ok x →
      (Lean.Loop.forIn Lean.Loop.mk b g).run s = .ok x := by
  intro N
  induction N with
  | zero => intro b s h; omega
  | succ N ih =>
    intro b s hμ x h
    rw [Lean.Loop.forIn_eq_of_monadTail, StateT.run_bind] at h ⊢
    cases hb : (f () b).run s with
    | error e => rw [hb] at h; exact absurd h (by simp [bind, Except.bind])
    | ok y =>
      obtain ⟨step, s₁⟩ := y
      rw [hb] at h
      rw [hfg b s _ hb]
      cases step with
      | done b' =>
        change (pure b' : StateT ℕ (Except String) β).run s₁ = .ok x at h
        change (pure b' : StateT ℕ (Except String) β).run s₁ = .ok x
        exact h
      | yield b' =>
        obtain ⟨hlt, hle⟩ := hy b s b' s₁ hb
        change (Lean.Loop.forIn Lean.Loop.mk b' f).run s₁ = .ok x at h
        change (Lean.Loop.forIn Lean.Loop.mk b' g).run s₁ = .ok x
        exact ih b' s₁ (by omega) x h

/-- A `for` loop over a list whose body returns on `g` what it returns on `f` returns the same on `g`. -/
theorem forIn_append {α β : Type} (f g : α → β → StateT ℕ (Except String) (ForInStep β))
    (hfg : ∀ a b s x, (f a b).run s = .ok x → (g a b).run s = .ok x) :
    ∀ (l : List α) (b : β) (s : ℕ) (x : β × ℕ), (forIn l b f).run s = .ok x → (forIn l b g).run s = .ok x := by
  intro l
  induction l with
  | nil => intro b s x h; rw [List.forIn_nil] at h ⊢; exact h
  | cons a as ih =>
    intro b s x h
    rw [List.forIn_cons] at h ⊢
    refine run_bind_of h (fun r hr => hfg a b s r hr) (fun r _ hk => ?_)
    obtain ⟨step, s₁⟩ := r
    cases step with
    | done b' => exact hk
    | yield b' => exact ih b' s₁ x hk

theorem onWith_append {ξ t : ByteArray} {N : ℕ} (hN : 0 < N) {run : ByteArray → Option (Array ℕ)} {p : ℕ}
    {x : Option (Array ℕ) × ℕ}
    (h : (Flock.Draw.onWith (fun need => Draw.take ξ need) N run).run p = .ok x) :
    (Flock.Draw.onWith (fun need => Draw.take (ξ ++ t) need) N run).run p = .ok x := by
  unfold Flock.Draw.onWith at h ⊢
  try dsimp only at h ⊢
  refine run_bind_of h (fun r hr => take_append hr) (fun r _ hr => ?_)
  refine run_bind_of hr (fun r' hr' => ?_) (fun _ _ hk => hk)
  refine loop_append _ _ ξ.size (fun b s z hz => ?_) (fun b s b' s' hz => ?_) _ _ _ (Nat.lt_succ_self _) r' hr'
  · split at hz
    · exact hz
    · exact run_bind_of hz (fun r hr => take_append hr) (fun _ _ hk => hk)
  · split at hz
    · rw [StateT.run_pure] at hz
      obtain ⟨h1, -⟩ := Prod.mk.inj (Except.ok.inj hz)
      exact absurd h1 (by simp)
    · rw [StateT.run_bind, take_run] at hz
      split at hz
      · exact absurd hz (by rw [bind_error]; simp)
      · rename_i hle
        rw [bind_ok, StateT.run_pure] at hz
        obtain ⟨-, rfl⟩ := Prod.mk.inj (Except.ok.inj hz)
        exact ⟨by omega, by omega⟩

theorem stratifiedWith_append {ξ t : ByteArray} {strata : Array Flock.Draw.Stratum} {p : ℕ}
    {x : Option (Array ℕ) × ℕ}
    (h : (Flock.Draw.stratifiedWith (fun _ need => Draw.take ξ need) strata).run p = .ok x) :
    (Flock.Draw.stratifiedWith (fun _ need => Draw.take (ξ ++ t) need) strata).run p = .ok x := by
  unfold Flock.Draw.stratifiedWith at h ⊢
  dsimp only at h ⊢
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at h ⊢
  refine run_bind_of h (fun r hr => forIn_append _ _ (fun i b s z hz => ?_) _ _ _ _ hr) (fun _ _ hk => hk)
  exact run_bind_of hz (fun r hr => onWith_append (by unfold Flock.Draw.Stratum.need; omega) hr) (fun _ _ hk => hk)

theorem drawWith_append {ξ t : ByteArray} (law : Flock.Draw.Law) (n : ℕ) {p : ℕ} {x : Option Flock.Draw.UnitDraw × ℕ}
    (h : (Flock.Draw.drawWith (fun _ need => Draw.take ξ need) law n).run p = .ok x) :
    (Flock.Draw.drawWith (fun _ need => Draw.take (ξ ++ t) need) law n).run p = .ok x := by
  cases law with
  | subset k =>
    unfold Flock.Draw.drawWith at h ⊢
    dsimp only at h ⊢
    exact run_bind_of h (fun r hr => onWith_append (by omega) hr) (fun _ _ hk => hk)
  | bernoulli a b =>
    unfold Flock.Draw.drawWith at h ⊢
    dsimp only at h ⊢
    exact run_bind_of h (fun r hr => onWith_append (by omega) hr) (fun _ _ hk => hk)
  | stratified k strata =>
    unfold Flock.Draw.drawWith at h ⊢
    dsimp only at h ⊢
    exact run_bind_of h (fun r hr => stratifiedWith_append hr) (fun _ _ hk => hk)
  | work k strata =>
    unfold Flock.Draw.drawWith at h ⊢
    dsimp only at h ⊢
    exact run_bind_of h (fun r hr => stratifiedWith_append hr) (fun _ _ hk => hk)

/-- **A draw reads only what it needs.** The units a law draws from `ξ`'s bytes, and the bytes it read, are the same
from `ξ ++ t`. -/
theorem drawUnits_append {ξ t : ByteArray} {law : Flock.Draw.Law} {n : ℕ} {r : Array ℕ × ℕ}
    (h : Draw.drawUnits law n ξ = .ok r) : Draw.drawUnits law n (ξ ++ t) = .ok r := by
  unfold Draw.drawUnits at h ⊢
  cases hw : (Flock.Draw.drawWith (fun _ need => Draw.take ξ need) law n).run 0 with
  | error e => rw [hw] at h; exact absurd h (by rw [bind_error]; simp)
  | ok y =>
    rw [hw] at h
    rw [drawWith_append (t := t) law n hw]
    exact h

end Properties.Spot.SecurityProofs
