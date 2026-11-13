module

public import Proofs.Flock.SpotFlock.RunsAs.Released
import Proofs.Flock.Claims.Sampling.Select.Take
import Proofs.Flock.Claims.Sampling.Select.Steps

/-!
# A sample reads no more bytes than its source holds

A draw reads its source in order (`Draw.take`), and `Draw.take` throws where fewer bytes are left than it is asked for.
So a draw that ends, ends at a position inside its source: the number of bytes a stage's sample reports reading is at
most its source's size (`drawUnits_within`, and through the stages' samples `select_within` and `selectSecond_within`).
Both hold of every rule, program, query, table, closure map and source, with no hypothesis on the run: `reads_within`
states them at a held registration.

`Released` shows that a sample depends on its source only through the bytes it reports reading; this file bounds that
number. Each step is one fact of a successful run from a position within the source: the position it ends at is within
the source.
-/

@[expose] public section

namespace SpotFlock.RunsAs

open Lean (Json)
open Properties.Spot.SecurityProofs (take_run bind_ok bind_error bind_eq_ok throw_bind_ne)

theorem pure_within {α : Type} {a : α} {s B : ℕ} {x : α × ℕ}
    (h : (pure a : StateT ℕ (Except String) α).run s = .ok x) (hs : s ≤ B) : x.2 ≤ B := by
  obtain rfl := run_pure_eq h
  exact hs

theorem bind_within {α β : Type} {x : StateT ℕ (Except String) α} {f : α → StateT ℕ (Except String) β}
    {s B : ℕ} {out : β × ℕ} (h : (x >>= f).run s = .ok out) (hx : ∀ r : α × ℕ, x.run s = .ok r → r.2 ≤ B)
    (hf : ∀ r : α × ℕ, r.2 ≤ B → (f r.1).run r.2 = .ok out → out.2 ≤ B) : out.2 ≤ B := by
  rw [StateT.run_bind] at h
  cases hxr : x.run s with
  | error e => rw [hxr] at h; exact absurd h (by rw [bind_error]; simp)
  | ok r =>
    rw [hxr, bind_ok] at h
    exact hf r (hx r hxr) h

/-- `Draw.take` ends inside its source: it throws where the source runs out. -/
theorem take_within {ξ : ByteArray} {need p : ℕ} {x : ByteArray × ℕ}
    (h : (Spot.Draw.take ξ need).run p = .ok x) : x.2 ≤ ξ.size := by
  rw [take_run] at h
  split at h
  · cases h
  · rename_i hle
    obtain rfl := (Except.ok.inj h).symm
    exact Nat.le_of_not_lt hle

/-- A `repeat` loop whose body keeps the position within `B` from a position within `B`, and whose every `yield` moves
the position up, ends within `B`. -/
theorem loop_within {β : Type} (f : Unit → β → StateT ℕ (Except String) (ForInStep β)) (B : ℕ)
    (hf : ∀ b s x, s ≤ B → (f () b).run s = .ok x → x.2 ≤ B)
    (hy : ∀ b s b' s', (f () b).run s = .ok (.yield b', s') → s < s') :
    ∀ N b s, B + 1 - s < N → s ≤ B → ∀ x, (Lean.Loop.forIn Lean.Loop.mk b f).run s = .ok x → x.2 ≤ B := by
  intro N
  induction N with
  | zero => intro b s h; exact absurd h (Nat.not_lt_zero _)
  | succ N ih =>
    intro b s hμ hs x h
    rw [Lean.Loop.forIn_eq_of_monadTail, StateT.run_bind] at h
    cases hb : (f () b).run s with
    | error e => rw [hb] at h; exact absurd h (by simp [bind, Except.bind])
    | ok y =>
      obtain ⟨step, s₁⟩ := y
      rw [hb] at h
      have hs₁ : s₁ ≤ B := hf b s _ hs hb
      cases step with
      | done b' =>
        change (pure b' : StateT ℕ (Except String) β).run s₁ = .ok x at h
        obtain rfl := run_pure_eq h
        exact hs₁
      | yield b' =>
        have hlt := hy b s b' s₁ hb
        change (Lean.Loop.forIn Lean.Loop.mk b' f).run s₁ = .ok x at h
        exact ih b' s₁ (by omega) hs₁ x h

/-- A `for` loop over a list whose body keeps the position within `B` ends within `B`. -/
theorem forIn_within {α β : Type} (f : α → β → StateT ℕ (Except String) (ForInStep β)) (B : ℕ)
    (hf : ∀ a b s x, s ≤ B → (f a b).run s = .ok x → x.2 ≤ B) :
    ∀ (l : List α) (b : β) (s : ℕ) (x : β × ℕ), s ≤ B → (forIn l b f).run s = .ok x → x.2 ≤ B := by
  intro l
  induction l with
  | nil =>
    intro b s x hs h
    rw [List.forIn_nil] at h
    obtain rfl := run_pure_eq h
    exact hs
  | cons a as ih =>
    intro b s x hs h
    rw [List.forIn_cons] at h
    refine bind_within h (fun r hr => hf a b s r hs hr) (fun r hr hk => ?_)
    obtain ⟨step, s₁⟩ := r
    cases step with
    | done b' =>
      obtain rfl := run_pure_eq hk
      exact hr
    | yield b' => exact ih b' s₁ x hr hk

theorem onWith_within {ξ : ByteArray} {N : ℕ} (hN : 0 < N) {run : ByteArray → Option (Array ℕ)} {p : ℕ}
    {x : Option (Array ℕ) × ℕ}
    (h : (Flock.Draw.onWith (fun need => Spot.Draw.take ξ need) N run).run p = .ok x) : x.2 ≤ ξ.size := by
  unfold Flock.Draw.onWith at h
  try dsimp only at h
  refine bind_within h (fun r hr => take_within hr) (fun r hs hr => ?_)
  refine bind_within hr (fun r' hr' => ?_) (fun r'' hs' hk => by
    first
    | exact pure_within hk hs'
    | (revert hk; (repeat' split) <;> intro hk <;> exact pure_within hk hs'))
  refine loop_within _ ξ.size (fun b s z hsz hz => ?_) (fun b s b' s' hz => ?_) _ _ _ (Nat.lt_succ_self _) hs r' hr'
  · split at hz
    · exact pure_within hz hsz
    · exact bind_within hz (fun r hr => take_within hr) (fun r hs'' hk => by
        first
        | exact pure_within hk hs''
        | (revert hk; (repeat' split) <;> intro hk <;> exact pure_within hk hs''))
  · split at hz
    · rw [StateT.run_pure] at hz
      obtain ⟨h1, -⟩ := Prod.mk.inj (Except.ok.inj hz)
      exact absurd h1 (by simp)
    · rw [StateT.run_bind, take_run] at hz
      split at hz
      · exact absurd hz (by rw [bind_error]; simp)
      · rw [bind_ok, StateT.run_pure] at hz
        obtain ⟨-, rfl⟩ := Prod.mk.inj (Except.ok.inj hz)
        omega

theorem stratifiedWith_within {ξ : ByteArray} {strata : Array Flock.Draw.Stratum} {p : ℕ}
    {x : Option (Array ℕ) × ℕ} (hp : p ≤ ξ.size)
    (h : (Flock.Draw.stratifiedWith (fun _ need => Spot.Draw.take ξ need) strata).run p = .ok x) :
    x.2 ≤ ξ.size := by
  unfold Flock.Draw.stratifiedWith at h
  dsimp only at h
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at h
  refine bind_within h (fun r hr => forIn_within _ ξ.size (fun i b s z hsz hz => ?_) _ _ _ _ hp hr)
    (fun r hs hk => by
      first
      | exact pure_within hk hs
      | (revert hk; (repeat' split) <;> intro hk <;> exact pure_within hk hs))
  exact bind_within hz (fun r hr => onWith_within (by unfold Flock.Draw.Stratum.need; omega) hr)
    (fun r hs hk => by
      first
      | exact pure_within hk hs
      | (revert hk; (repeat' split) <;> intro hk <;> exact pure_within hk hs))

theorem drawWith_within {ξ : ByteArray} (law : Flock.Draw.Law) (n : ℕ) {p : ℕ}
    {x : Option Flock.Draw.UnitDraw × ℕ} (hp : p ≤ ξ.size)
    (h : (Flock.Draw.drawWith (fun _ need => Spot.Draw.take ξ need) law n).run p = .ok x) : x.2 ≤ ξ.size := by
  cases law with
  | subset k =>
    unfold Flock.Draw.drawWith at h
    dsimp only at h
    exact bind_within h (fun r hr => onWith_within (by omega) hr) (fun r hs hk => by
      first
      | exact pure_within hk hs
      | (revert hk; (repeat' split) <;> intro hk <;> exact pure_within hk hs))
  | bernoulli a b =>
    unfold Flock.Draw.drawWith at h
    dsimp only at h
    exact bind_within h (fun r hr => onWith_within (by omega) hr) (fun r hs hk => by
      first
      | exact pure_within hk hs
      | (revert hk; (repeat' split) <;> intro hk <;> exact pure_within hk hs))
  | stratified k strata =>
    unfold Flock.Draw.drawWith at h
    dsimp only at h
    exact bind_within h (fun r hr => stratifiedWith_within hp hr) (fun r hs hk => by
      first
      | exact pure_within hk hs
      | (revert hk; (repeat' split) <;> intro hk <;> exact pure_within hk hs))
  | work k strata =>
    unfold Flock.Draw.drawWith at h
    dsimp only at h
    exact bind_within h (fun r hr => stratifiedWith_within hp hr) (fun r hs hk => by
      first
      | exact pure_within hk hs
      | (revert hk; (repeat' split) <;> intro hk <;> exact pure_within hk hs))

/-- **A draw ends inside its source**: the bytes a law's draw from `ξ` reports reading are at most `ξ`'s. -/
theorem drawUnits_within {ξ : ByteArray} {law : Flock.Draw.Law} {n : ℕ} {r : Array ℕ × ℕ}
    (h : Spot.Draw.drawUnits law n ξ = .ok r) : r.2 ≤ ξ.size := by
  unfold Spot.Draw.drawUnits at h
  cases hw : (Flock.Draw.drawWith (fun _ need => Spot.Draw.take ξ need) law n).run 0 with
  | error e => rw [hw] at h; exact absurd h (by rw [bind_error]; simp)
  | ok y =>
    obtain ⟨o, rd⟩ := y
    rw [hw] at h
    cases o with
    | none => cases h
    | some d =>
      obtain rfl := (Except.ok.inj h).symm
      exact drawWith_within law n (Nat.zero_le _) hw

theorem drawFirstRead_within {law : Json} {n : ℕ} {ξ : ByteArray} {r : Json × ℕ}
    (h : Spot.Draw.drawFirstRead law n ξ = .ok r) : r.2 ≤ ξ.size := by
  unfold Spot.Draw.drawFirstRead at h
  obtain ⟨parsed, hp, h⟩ := bind_eq_ok.1 h
  obtain ⟨ur, hu, h⟩ := bind_eq_ok.1 h
  obtain ⟨u, rd⟩ := ur
  have hr : r.2 = rd := congrArg Prod.snd (Except.ok.inj h).symm
  rw [hr]
  exact drawUnits_within hu

/-- **The second stage's sample ends inside its source.** -/
theorem selectSecond_within {sample ns ks : Array ℕ} {ξ : ByteArray} {r : Json × ℕ}
    (h : Spot.Select.selectSecond sample ns ks ξ = .ok r) : r.2 ≤ ξ.size := by
  unfold Spot.Select.selectSecond at h
  cases hq : Spot.Select.secondProblem sample ns ks with
  | some e =>
    try rw [hq] at h
    dsimp only at h
    exact absurd h (throw_bind_ne _ _ _)
  | none =>
    try rw [hq] at h
    dsimp only at h
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
    rw [hr]
    exact drawFirstRead_within hd

/-- **The first stage's sample ends inside its source**, under any rule and any closure map: a rule that samples every
unit reads nothing, and any other reads what its draw reads (`drawUnits_within`). -/
theorem select_within {rule : Spot.Select.Rule} {p : Spot.Program} {query : Json} {table : Option Json}
    {cm : Option (Array (Int × Array Int))} {ξ : ByteArray} {r : Json × Array ℕ × ℕ}
    (h : Spot.Select.select rule p query table cm ξ = .ok r) : r.2.2 ≤ ξ.size := by
  unfold Spot.Select.select at h
  obtain ⟨n, hn, h⟩ := bind_eq_ok.1 h
  obtain ⟨named, hnamed, h⟩ := bind_eq_ok.1 h
  obtain ⟨derived, hderived, h⟩ := bind_eq_ok.1 h
  have tail : ∀ {x : Except String (Array ℕ × ℕ)} {k : Array ℕ × ℕ → Except String (Json × Array ℕ × ℕ)},
      (∀ ur, x = .ok ur → ur.2 ≤ ξ.size) → (∀ u rd, k (u, rd) = .ok r → r.2.2 = rd) → (x >>= k) = .ok r →
      r.2.2 ≤ ξ.size := by
    intro x k hx hk h
    obtain ⟨ur, hu, h⟩ := bind_eq_ok.1 h
    rw [hk ur.1 ur.2 h]
    exact hx ur hu
  have zero : ∀ ur : Array ℕ × ℕ, (pure (Array.range n, 0) : Except String (Array ℕ × ℕ)) = .ok ur →
      ur.2 ≤ ξ.size := by
    intro ur hu
    have h0 : 0 = ur.2 := congrArg Prod.snd (Except.ok.inj hu)
    rw [← h0]
    exact Nat.zero_le _
  have rest : ∃ (x : Except String (Array ℕ × ℕ)) (k : Array ℕ × ℕ → Except String (Json × Array ℕ × ℕ)),
      (∀ ur, x = .ok ur → ur.2 ≤ ξ.size) ∧ (∀ u rd, k (u, rd) = .ok r → r.2.2 = rd) ∧ (x >>= k) = .ok r := by
    cases cm with
    | none =>
      dsimp only at h
      cases rule <;> dsimp only at h <;>
        first
        | (obtain ⟨parsed, hp, h⟩ := bind_eq_ok.1 h
           refine ⟨_, _, fun ur hu => drawUnits_within hu, fun u rd hk => ?_, h⟩)
        | refine ⟨_, _, zero, fun u rd hk => ?_, h⟩
      all_goals
        revert hk
        try dsimp only
        repeat' split
        all_goals intro h
        all_goals first
          | exact absurd h (throw_bind_ne _ _ _)
          | (cases h; done)
          | exact congrArg (fun q => q.2.2) (Except.ok.inj h).symm
    | some cm' =>
      dsimp only at h
      split at h
      · exact absurd h (throw_bind_ne _ _ _)
      · cases rule <;> dsimp only at h <;>
          first
          | (obtain ⟨parsed, hp, h⟩ := bind_eq_ok.1 h
             refine ⟨_, _, fun ur hu => drawUnits_within hu, fun u rd hk => ?_, h⟩)
          | refine ⟨_, _, zero, fun u rd hk => ?_, h⟩
        all_goals
          revert hk
          try dsimp only
          repeat' split
          all_goals intro h
          all_goals first
            | exact absurd h (throw_bind_ne _ _ _)
            | (cases h; done)
            | exact congrArg (fun q => q.2.2) (Except.ok.inj h).symm
  obtain ⟨x, k, hx, hk, h⟩ := rest
  exact tail hx hk h

/-- **Both stages' samples read within their sources**, at a held registration `h`: whatever source the first stage's
sample is drawn from, under `h`'s rule, program, query, table and closure map, the bytes it reports reading are at most
the source's, and so are the second stage's over any replay sample. -/
theorem reads_within (h : Held) :
    (∀ ξ draw proved read, Spot.Select.select h.reg.first.rule h.reg.first.program h.reg.first.query
        h.reg.first.table h.reg.first.closure ξ = .ok (draw, proved, read) → read ≤ ξ.size) ∧
      ∀ (sample : Array ℕ) ξ draw read,
        Spot.Select.selectSecond sample h.reg.ns h.reg.ks ξ = .ok (draw, read) → read ≤ ξ.size :=
  ⟨fun _ _ _ _ hs => select_within hs, fun _ _ _ _ hs => selectSecond_within hs⟩

end SpotFlock.RunsAs
