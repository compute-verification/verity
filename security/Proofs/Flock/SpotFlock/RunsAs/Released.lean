module

public import Proofs.Flock.SpotFlock.RunsAs.Statement
public import Proofs.Flock.Claims.Sampling.Select.Prefix
import Proofs.Flock.Claims.Sampling.Select.Take
import Proofs.Flock.Claims.Sampling.Select.Steps

/-!
# A sample reads only the bytes it reports reading

A sample reads its source in order (`Draw.take`), from position `0`, and throws where the source runs out, so the
position only grows and a sample that ends at position `R` read only the source's first `R` bytes. It samples the same from
any source that starts with those `R` bytes (`AgreesTo`): `drawUnits_read`, and through the stages' samples,
`select_read` and `selectSecond_read`. The `R` a stage reports is the `read` `Run.coinsCode` holds to the challenger's
released coins, so a stage's sample depends on its coins only through the bytes the challenger released.

Each step is stated as one fact of a successful run from position `p` to `x.2`: `p ≤ x.2`, and where `x.2 ≤ R`, the
run on a source agreeing up to `R` is the same.
-/

@[expose] public section

namespace SpotFlock.RunsAs

open Lean (Json)
open Properties.Spot.SecurityProofs (take_run bind_ok bind_error bind_eq_ok throw_bind_ne)

theorem pure_le {α : Type} {a : α} {s : ℕ} {x : α × ℕ}
    (h : (pure a : StateT ℕ (Except String) α).run s = .ok x) : s ≤ x.2 := by
  rw [StateT.run_pure] at h
  obtain rfl := Except.ok.inj h
  exact Nat.le_refl _

theorem run_pure_eq {α : Type} {a : α} {s : ℕ} {x : α × ℕ}
    (h : (pure a : StateT ℕ (Except String) α).run s = .ok x) : x = (a, s) := by
  rw [StateT.run_pure] at h
  exact (Except.ok.inj h).symm

theorem bind_read {α β : Type} {x y : StateT ℕ (Except String) α} {f g : α → StateT ℕ (Except String) β}
    {s R : ℕ} {out : β × ℕ} (h : (x >>= f).run s = .ok out)
    (hx : ∀ r, x.run s = .ok r → s ≤ r.2 ∧ (r.2 ≤ R → y.run s = .ok r))
    (hf : ∀ r, x.run s = .ok r → (f r.1).run r.2 = .ok out →
      r.2 ≤ out.2 ∧ (out.2 ≤ R → (g r.1).run r.2 = .ok out)) :
    s ≤ out.2 ∧ (out.2 ≤ R → (y >>= g).run s = .ok out) := by
  rw [StateT.run_bind] at h
  cases hxr : x.run s with
  | error e => rw [hxr] at h; exact absurd h (by rw [bind_error]; simp)
  | ok r =>
    rw [hxr, bind_ok] at h
    obtain ⟨h1, h2⟩ := hx r hxr
    obtain ⟨h3, h4⟩ := hf r hxr h
    refine ⟨Nat.le_trans h1 h3, fun hR => ?_⟩
    rw [StateT.run_bind, h2 (Nat.le_trans h3 hR), bind_ok]
    exact h4 hR

theorem take_read {ξ ξ' : ByteArray} {R need p : ℕ} (hA : AgreesTo ξ ξ' R) {x : ByteArray × ℕ}
    (h : (Spot.Draw.take ξ need).run p = .ok x) :
    p ≤ x.2 ∧ (x.2 ≤ R → (Spot.Draw.take ξ' need).run p = .ok x) := by
  rw [take_run] at h
  split at h
  · cases h
  · obtain rfl := (Except.ok.inj h).symm
    refine ⟨Nat.le_add_right _ _, fun hR => ?_⟩
    have hR' : p + need ≤ R := hR
    rw [take_run, if_neg (by have := hA.size_le; omega)]
    have e : ∀ ζ : ByteArray, (ζ.extract 0 R).extract p (p + need) = ζ.extract p (p + need) := fun ζ => by
      rw [ByteArray.extract_extract, Nat.zero_add, Nat.zero_add, Nat.min_eq_left hR']
    rw [← e ξ', hA.extract_eq, e ξ]

/-- A `repeat` loop whose body, on `f`, moves the position up and returns the same on `g` where it ends by `R`, and whose
every `yield` on `f` moves the position up, to at most `B`, does so too. -/
theorem loop_read {β : Type} (f g : Unit → β → StateT ℕ (Except String) (ForInStep β)) (B R : ℕ)
    (hfg : ∀ b s x, (f () b).run s = .ok x → s ≤ x.2 ∧ (x.2 ≤ R → (g () b).run s = .ok x))
    (hy : ∀ b s b' s', (f () b).run s = .ok (.yield b', s') → s < s' ∧ s' ≤ B) :
    ∀ N b s, B + 1 - s < N → ∀ x, (Lean.Loop.forIn Lean.Loop.mk b f).run s = .ok x →
      s ≤ x.2 ∧ (x.2 ≤ R → (Lean.Loop.forIn Lean.Loop.mk b g).run s = .ok x) := by
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
      obtain ⟨hs₁, hg⟩ := hfg b s _ hb
      have hs : s ≤ s₁ := hs₁
      cases step with
      | done b' =>
        change (pure b' : StateT ℕ (Except String) β).run s₁ = .ok x at h
        obtain rfl := run_pure_eq h
        refine ⟨hs, fun hR => ?_⟩
        rw [hg hR]
        rfl
      | yield b' =>
        obtain ⟨hlt, hle⟩ := hy b s b' s₁ hb
        change (Lean.Loop.forIn Lean.Loop.mk b' f).run s₁ = .ok x at h
        obtain ⟨h1, h2⟩ := ih b' s₁ (by omega) x h
        refine ⟨Nat.le_trans hs h1, fun hR => ?_⟩
        rw [hg (show s₁ ≤ R from Nat.le_trans h1 hR)]
        exact h2 hR

/-- A `for` loop over a list whose body does as `loop_read`'s does so too. -/
theorem forIn_read {α β : Type} (f g : α → β → StateT ℕ (Except String) (ForInStep β)) (R : ℕ)
    (hfg : ∀ a b s x, (f a b).run s = .ok x → s ≤ x.2 ∧ (x.2 ≤ R → (g a b).run s = .ok x)) :
    ∀ (l : List α) (b : β) (s : ℕ) (x : β × ℕ), (forIn l b f).run s = .ok x →
      s ≤ x.2 ∧ (x.2 ≤ R → (forIn l b g).run s = .ok x) := by
  intro l
  induction l with
  | nil =>
    intro b s x h
    rw [List.forIn_nil] at h ⊢
    obtain rfl := run_pure_eq h
    exact ⟨Nat.le_refl _, fun _ => rfl⟩
  | cons a as ih =>
    intro b s x h
    rw [List.forIn_cons] at h ⊢
    refine bind_read h (fun r hr => hfg a b s r hr) (fun r _ hk => ?_)
    obtain ⟨step, s₁⟩ := r
    cases step with
    | done b' =>
      obtain rfl := run_pure_eq hk
      exact ⟨Nat.le_refl _, fun _ => hk⟩
    | yield b' => exact ih b' s₁ x hk

theorem onWith_read {ξ ξ' : ByteArray} {R : ℕ} (hA : AgreesTo ξ ξ' R) {N : ℕ} (hN : 0 < N)
    {run : ByteArray → Option (Array ℕ)} {p : ℕ} {x : Option (Array ℕ) × ℕ}
    (h : (Flock.Draw.onWith (fun need => Spot.Draw.take ξ need) N run).run p = .ok x) :
    p ≤ x.2 ∧ (x.2 ≤ R → (Flock.Draw.onWith (fun need => Spot.Draw.take ξ' need) N run).run p = .ok x) := by
  unfold Flock.Draw.onWith at h ⊢
  try dsimp only at h ⊢
  refine bind_read h (fun r hr => take_read hA hr) (fun r _ hr => ?_)
  refine bind_read hr (fun r' hr' => ?_) (fun r'' _ hk => ⟨by first | exact pure_le hk | (revert hk; (repeat' split) <;> intro hk <;> exact pure_le hk), fun _ => hk⟩)
  refine loop_read _ _ ξ.size R (fun b s z hz => ?_) (fun b s b' s' hz => ?_) _ _ _ (Nat.lt_succ_self _) r' hr'
  · split at hz
    · exact ⟨pure_le hz, fun _ => hz⟩
    · exact bind_read hz (fun r hr => take_read hA hr) (fun r _ hk => ⟨by first | exact pure_le hk | (revert hk; (repeat' split) <;> intro hk <;> exact pure_le hk), fun _ => hk⟩)
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

theorem stratifiedWith_read {ξ ξ' : ByteArray} {R : ℕ} (hA : AgreesTo ξ ξ' R) {strata : Array Flock.Draw.Stratum}
    {p : ℕ} {x : Option (Array ℕ) × ℕ}
    (h : (Flock.Draw.stratifiedWith (fun _ need => Spot.Draw.take ξ need) strata).run p = .ok x) :
    p ≤ x.2 ∧ (x.2 ≤ R → (Flock.Draw.stratifiedWith (fun _ need => Spot.Draw.take ξ' need) strata).run p = .ok x) := by
  unfold Flock.Draw.stratifiedWith at h ⊢
  dsimp only at h ⊢
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at h ⊢
  refine bind_read h (fun r hr => forIn_read _ _ R (fun i b s z hz => ?_) _ _ _ _ hr)
    (fun r _ hk => ⟨by first | exact pure_le hk | (revert hk; (repeat' split) <;> intro hk <;> exact pure_le hk), fun _ => hk⟩)
  exact bind_read hz (fun r hr => onWith_read hA (by unfold Flock.Draw.Stratum.need; omega) hr)
    (fun r _ hk => ⟨by first | exact pure_le hk | (revert hk; (repeat' split) <;> intro hk <;> exact pure_le hk), fun _ => hk⟩)

theorem drawWith_read {ξ ξ' : ByteArray} {R : ℕ} (hA : AgreesTo ξ ξ' R) (law : Flock.Draw.Law) (n : ℕ) {p : ℕ}
    {x : Option Flock.Draw.UnitDraw × ℕ}
    (h : (Flock.Draw.drawWith (fun _ need => Spot.Draw.take ξ need) law n).run p = .ok x) :
    p ≤ x.2 ∧ (x.2 ≤ R → (Flock.Draw.drawWith (fun _ need => Spot.Draw.take ξ' need) law n).run p = .ok x) := by
  cases law with
  | subset k =>
    unfold Flock.Draw.drawWith at h ⊢
    dsimp only at h ⊢
    exact bind_read h (fun r hr => onWith_read hA (by omega) hr) (fun r _ hk => ⟨by first | exact pure_le hk | (revert hk; (repeat' split) <;> intro hk <;> exact pure_le hk), fun _ => hk⟩)
  | bernoulli a b =>
    unfold Flock.Draw.drawWith at h ⊢
    dsimp only at h ⊢
    exact bind_read h (fun r hr => onWith_read hA (by omega) hr) (fun r _ hk => ⟨by first | exact pure_le hk | (revert hk; (repeat' split) <;> intro hk <;> exact pure_le hk), fun _ => hk⟩)
  | stratified k strata =>
    unfold Flock.Draw.drawWith at h ⊢
    dsimp only at h ⊢
    exact bind_read h (fun r hr => stratifiedWith_read hA hr) (fun r _ hk => ⟨by first | exact pure_le hk | (revert hk; (repeat' split) <;> intro hk <;> exact pure_le hk), fun _ => hk⟩)
  | work k strata =>
    unfold Flock.Draw.drawWith at h ⊢
    dsimp only at h ⊢
    exact bind_read h (fun r hr => stratifiedWith_read hA hr) (fun r _ hk => ⟨by first | exact pure_le hk | (revert hk; (repeat' split) <;> intro hk <;> exact pure_le hk), fun _ => hk⟩)

/-- **A sample reads only the bytes it reports reading**: the units a sampling design samples from `ξ`, having read `r.2` bytes, are
the same from any source that starts with `ξ`'s first `r.2` bytes. -/
theorem drawUnits_read {ξ ξ' : ByteArray} {law : Flock.Draw.Law} {n : ℕ} {r : Array ℕ × ℕ}
    (h : Spot.Draw.drawUnits law n ξ = .ok r) (hA : AgreesTo ξ ξ' r.2) : Spot.Draw.drawUnits law n ξ' = .ok r := by
  unfold Spot.Draw.drawUnits at h ⊢
  cases hw : (Flock.Draw.drawWith (fun _ need => Spot.Draw.take ξ need) law n).run 0 with
  | error e => rw [hw] at h; exact absurd h (by rw [bind_error]; simp)
  | ok y =>
    obtain ⟨o, rd⟩ := y
    rw [hw] at h
    cases o with
    | none => cases h
    | some d =>
      obtain rfl := (Except.ok.inj h).symm
      rw [(drawWith_read hA law n hw).2 (Nat.le_refl _)]
      rfl

theorem drawFirstRead_read {law : Json} {n : ℕ} {ξ ξ' : ByteArray} {r : Json × ℕ}
    (h : Spot.Draw.drawFirstRead law n ξ = .ok r) (hA : AgreesTo ξ ξ' r.2) :
    Spot.Draw.drawFirstRead law n ξ' = .ok r := by
  unfold Spot.Draw.drawFirstRead at h ⊢
  obtain ⟨parsed, hp, h⟩ := bind_eq_ok.1 h
  obtain ⟨ur, hu, h⟩ := bind_eq_ok.1 h
  obtain ⟨u, rd⟩ := ur
  have hr : r.2 = rd := congrArg Prod.snd (Except.ok.inj h).symm
  have hA' : AgreesTo ξ ξ' rd := by rw [← hr]; exact hA
  rw [hp, bind_ok, drawUnits_read hu hA', bind_ok]
  exact h

/-- **The second stage reads only the bytes it reports reading.** -/
theorem selectSecond_read {sample ns ks : Array ℕ} {ξ ξ' : ByteArray} {r : Json × ℕ}
    (h : Spot.Select.selectSecond sample ns ks ξ = .ok r) (hA : AgreesTo ξ ξ' r.2) :
    Spot.Select.selectSecond sample ns ks ξ' = .ok r := by
  unfold Spot.Select.selectSecond at h ⊢
  cases hq : Spot.Select.secondProblem sample ns ks with
  | some e =>
    try rw [hq] at h
    dsimp only at h
    exact absurd h (throw_bind_ne _ _ _)
  | none =>
    try rw [hq] at h
    try rw [hq]
    dsimp only at h ⊢
    obtain ⟨dr, hd, h⟩ := bind_eq_ok.1 h
    obtain ⟨d, rd⟩ := dr
    have hr : r.2 = rd := by
      revert h
      try dsimp only
      repeat' split
      all_goals intro h
      all_goals first
        | exact absurd h (throw_bind_ne _ _ _)
        | (cases h; done)
        | exact congrArg Prod.snd (Except.ok.inj h).symm
    have hA' : AgreesTo ξ ξ' rd := by rw [← hr]; exact hA
    rw [drawFirstRead_read hd hA', bind_ok]
    exact h

/-- **The first stage reads only the bytes it reports reading**, with no closure map. -/
theorem select_read {rule : Spot.Select.Rule} {p : Spot.Program} {query : Json} {table : Option Json}
    {ξ ξ' : ByteArray} {r : Json × Array ℕ × ℕ}
    (h : Spot.Select.select rule p query table none ξ = .ok r) (hA : AgreesTo ξ ξ' r.2.2) :
    Spot.Select.select rule p query table none ξ' = .ok r := by
  unfold Spot.Select.select at h ⊢
  obtain ⟨n, hn, h⟩ := bind_eq_ok.1 h
  obtain ⟨named, hnamed, h⟩ := bind_eq_ok.1 h
  obtain ⟨derived, hderived, h⟩ := bind_eq_ok.1 h
  rw [hn, bind_ok, hnamed, bind_ok, hderived, bind_ok]
  dsimp only at h ⊢
  cases rule with
  | all => exact h
  | law text =>
    dsimp only at h ⊢
    obtain ⟨parsed, hp, h⟩ := bind_eq_ok.1 h
    obtain ⟨ur, hu, h⟩ := bind_eq_ok.1 h
    obtain ⟨u, rd⟩ := ur
    have hr : r.2.2 = rd := by
      revert h
      try dsimp only
      repeat' split
      all_goals intro h
      all_goals first
        | exact absurd h (throw_bind_ne _ _ _)
        | (cases h; done)
        | exact congrArg (fun q => q.2.2) (Except.ok.inj h).symm
    have hA' : AgreesTo ξ ξ' rd := by rw [← hr]; exact hA
    rw [hp, bind_ok, drawUnits_read hu hA', bind_ok]
    exact h
  | twoStage grain replay k =>
    dsimp only at h ⊢
    obtain ⟨parsed, hp, h⟩ := bind_eq_ok.1 h
    obtain ⟨ur, hu, h⟩ := bind_eq_ok.1 h
    obtain ⟨u, rd⟩ := ur
    have hr : r.2.2 = rd := by
      revert h
      try dsimp only
      repeat' split
      all_goals intro h
      all_goals first
        | exact absurd h (throw_bind_ne _ _ _)
        | (cases h; done)
        | exact congrArg (fun q => q.2.2) (Except.ok.inj h).symm
    have hA' : AgreesTo ξ ξ' rd := by rw [← hr]; exact hA
    rw [hp, bind_ok, drawUnits_read hu hA', bind_ok]
    exact h

/-- **Part 1, at the released bytes**: where the registration has no closure map and the first stage accepted on `ξ`
having read `read` bytes, the replay sample on any source that starts with those bytes is the stage's units. -/
theorem sample_read {g : SamplingDeploy.RegTwo} {ξ ξ' : ByteArray} {draw : Json} {proved : Array ℕ} {read : ℕ}
    (hc : g.first.closure = none)
    (h : Spot.Select.select g.first.rule g.first.program g.first.query g.first.table g.first.closure ξ =
      .ok (draw, proved, read)) (hA : AgreesTo ξ ξ' read) :
    g.sample ξ' = proved := by
  rw [hc] at h
  simp only [SamplingDeploy.RegTwo.sample, hc, select_read h hA]

/-- **Part 2, at the released bytes**: where the second stage accepted on `ξ` over the model's replay sample, having
read `read` bytes, and the second stage's coins start with those bytes, the model's second sample is the run's. -/
theorem secondDraw_read (h : Held) (c₀ : FlockDeploy.Coin) (c₁ : Stage h) {ξ : ByteArray} {draw : Json} {read : ℕ}
    (hs : Spot.Select.selectSecond (h.reg.sample (src c₀)) h.reg.ns h.reg.ks ξ = .ok (draw, read))
    (hA : AgreesTo ξ (stageSrc c₁) read) :
    secondDraw h c₀ c₁ = some draw := by
  simp only [secondDraw, selectSecond_read hs hA]

end SpotFlock.RunsAs
