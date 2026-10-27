import Proofs.Flock.Claims.Sampling.Select.Steps
import Proofs.Flock.Claims.Sampling.Select.Take
import Proofs.Flock.Soundness.ExecDrawOS

/-!
# `Select.select`'s subset draw is the live draw's (`Law.runOS`) on the same bytes

`Select.select` draws a law's units with `Draw.drawUnits`, which is `Flock.Draw.drawWith` reading the coins in order
through `Draw.take`, one position for every stream, in `StateT ℕ (Except String)`. `Law.runOS` is the same `drawWith`
reading through `Law.tapeGet`, one position per stream, in `StateT (ℕ → ℕ) Option`. A subset draw reads stream `0` only,
so on stream `0`'s bytes the two runs agree step for step (`TapeSim`), the code erring where the model reads past the
tape.

* `drawUnits_subset_runOS`: the code's subset draw on the bytes `f 0` is `runOS`'s on `f`, and fails exactly where it is
  `none`.
* `select_subset_runOS`: an accepted `Select.select` at a rule whose law over the population C-Flock reads as
  `.subset k` proves every unit `runOS` draws.
* `select_subset_escape_le`: so on bytes uniform by A3 (`Assumptions.UniformRandomBytes`) it misses `B` no more often
  than the uniform `k`-subset does (`Law.drawOS_subset_escape_le`).

A stratified or work draw reads one stream per stratum, which the code serves from the one sequence of coins and `runOS`
from a tape each, so the agreement here is the subset law's, which a rate law resolves to (`Select.Rule.over`).
-/

open SampledProofs

namespace Properties.SampledProofs.SecurityProofs

open Lean (Json)
open FlockSoundness FlockSoundness.Game FlockSoundness.Audit

/-- A code run's outcome as a model run's on stream `0`: stream `0` moved to where the code stopped, `none` where it
erred. -/
def onTape {α : Type} (pos : ℕ → ℕ) : Except String (α × ℕ) → Option (α × (ℕ → ℕ))
  | .ok (a, p) => some (a, Function.update pos 0 p)
  | .error _ => none

/-- A model run (`Law.tapeGet` on stream `0`) agrees with a code run (`Draw.take`) from every state. -/
def TapeSim {α : Type} (x₁ : StateT ℕ (Except String) α) (x₂ : StateT (ℕ → ℕ) Option α) : Prop :=
  ∀ pos, x₂.run pos = onTape pos (x₁.run (pos 0))

theorem onTape_update {α : Type} (pos : ℕ → ℕ) (p : ℕ) (x : Except String (α × ℕ)) :
    onTape (Function.update pos 0 p) x = onTape pos x := by
  cases x with
  | error e => rfl
  | ok y =>
    obtain ⟨a, q⟩ := y
    show some (a, Function.update (Function.update pos 0 p) 0 q) = some (a, Function.update pos 0 q)
    rw [Function.update_idem]

theorem tapeSim_pure {α : Type} (a : α) :
    TapeSim (pure a : StateT ℕ (Except String) α) (pure a : StateT (ℕ → ℕ) Option α) := by
  intro pos
  show some (a, pos) = some (a, Function.update pos 0 (pos 0))
  rw [Function.update_eq_self]

theorem tapeSim_bind {α β : Type} {x₁ : StateT ℕ (Except String) α} {x₂ : StateT (ℕ → ℕ) Option α}
    {g₁ : α → StateT ℕ (Except String) β} {g₂ : α → StateT (ℕ → ℕ) Option β}
    (hx : TapeSim x₁ x₂) (hg : ∀ a, TapeSim (g₁ a) (g₂ a)) : TapeSim (x₁ >>= g₁) (x₂ >>= g₂) := by
  intro pos
  rw [StateT.run_bind, StateT.run_bind, hx pos]
  cases x₁.run (pos 0) with
  | error e => rfl
  | ok y =>
    obtain ⟨a, q⟩ := y
    show (g₂ a).run (Function.update pos 0 q) = onTape pos ((g₁ a).run q)
    rw [hg a (Function.update pos 0 q), Function.update_self, onTape_update]

/-- The code's read is the model's on stream `0` of tapes whose stream `0` is the code's bytes. -/
theorem tapeSim_take (T : ℕ → ByteArray) (need : ℕ) : TapeSim (Draw.take (T 0) need) (Law.tapeGet T 0 need) := by
  intro pos
  rw [take_run]
  show (if pos 0 + need ≤ (T 0).size then
      some ((T 0).extract (pos 0) (pos 0 + need), Function.update pos 0 (pos 0 + need)) else none) = _
  by_cases h : pos 0 + need ≤ (T 0).size
  · simp only [h, show ¬ (T 0).size < pos 0 + need by omega, ↓reduceIte]
    rfl
  · simp only [h, show (T 0).size < pos 0 + need by omega, ↓reduceIte]
    rfl

/-- A `repeat` loop whose bodies agree, each code `yield` lowering a measure of the position, agrees. -/
theorem tapeSim_loop {β : Type} (f₁ : Unit → β → StateT ℕ (Except String) (ForInStep β))
    (f₂ : Unit → β → StateT (ℕ → ℕ) Option (ForInStep β)) (μ : ℕ → ℕ) (hf : ∀ b, TapeSim (f₁ () b) (f₂ () b))
    (hμ : ∀ b p b' p', (f₁ () b).run p = .ok (.yield b', p') → μ p' < μ p) (b : β) :
    TapeSim (Lean.Loop.forIn Lean.Loop.mk b f₁) (Lean.Loop.forIn Lean.Loop.mk b f₂) := by
  suffices h : ∀ N b pos, μ (pos 0) < N → (Lean.Loop.forIn Lean.Loop.mk b f₂).run pos =
      onTape pos ((Lean.Loop.forIn Lean.Loop.mk b f₁).run (pos 0)) from
    fun pos => h _ b pos (Nat.lt_succ_self _)
  intro N
  induction N with
  | zero => intro b pos h; omega
  | succ N ih =>
    intro b pos hN
    rw [Lean.Loop.forIn_eq_of_monadTail, Lean.Loop.forIn_eq_of_monadTail, StateT.run_bind, StateT.run_bind,
      hf b pos]
    cases h₁ : (f₁ () b).run (pos 0) with
    | error e => rfl
    | ok y =>
      obtain ⟨step, q⟩ := y
      cases step with
      | done b' => rfl
      | yield b' =>
        show (Lean.Loop.forIn Lean.Loop.mk b' f₂).run (Function.update pos 0 q) =
          onTape pos ((Lean.Loop.forIn Lean.Loop.mk b' f₁).run q)
        have hq := hμ b (pos 0) b' q h₁
        rw [ih b' (Function.update pos 0 q) (by rw [Function.update_self]; omega), Function.update_self,
          onTape_update]

/-- **`onWith` on the code's reader is `onWith` on stream `0` of the tapes.** -/
theorem tapeSim_onWith (T : ℕ → ByteArray) (N : ℕ) (hN : 0 < N) (run : ByteArray → Option (Array ℕ)) :
    TapeSim (Flock.Draw.onWith (fun need => Draw.take (T 0) need) N run)
      (Flock.Draw.onWith (Law.tapeGet T 0) N run) := by
  unfold Flock.Draw.onWith
  refine tapeSim_bind (tapeSim_take T N) fun src =>
    tapeSim_bind (tapeSim_loop _ _ (fun p => (T 0).size + 1 - p) (fun b => ?_) ?_ _) fun r => ?_
  · obtain ⟨r, s⟩ := b
    try dsimp only
    cases hrun : run s with
    | some units => exact tapeSim_pure _
    | none => exact tapeSim_bind (tapeSim_take T N) fun x => tapeSim_pure _
  · intro b p b' p' h
    obtain ⟨r, s⟩ := b
    cases hrun : run s with
    | some units =>
      simp only [hrun, StateT.run_pure] at h
      obtain ⟨h1, _⟩ := Prod.mk.inj (Except.ok.inj h)
      exact absurd h1 (by simp)
    | none =>
      simp only [hrun] at h
      rw [StateT.run_bind, take_run] at h
      split at h
      · exact absurd h (by rw [bind_error]; simp)
      · rw [bind_ok, StateT.run_pure] at h
        obtain ⟨_, rfl⟩ := Prod.mk.inj (Except.ok.inj h)
        omega
  · obtain ⟨r, s⟩ := r
    cases r <;> exact tapeSim_pure _

/-- **The subset draw on the code's reader is the one on stream `0` of the tapes.** -/
theorem tapeSim_drawWith (T : ℕ → ByteArray) (k n : ℕ) :
    TapeSim (Flock.Draw.drawWith (fun _ need => Draw.take (T 0) need) (.subset k) n)
      (Flock.Draw.drawWith (Law.tapeGet T) (.subset k) n) := by
  unfold Flock.Draw.drawWith
  exact tapeSim_bind (tapeSim_onWith T _ (by omega) _) fun u => tapeSim_pure _

/-- **The code's subset draw is the live draw's** (`Law.runOS`, `Flock.Draw.drawOS`'s `drawWith`): on the bytes `f 0`,
`Draw.drawUnits` draws the units `runOS` draws on `f`, and fails exactly where `runOS` is `none`. -/
theorem drawUnits_subset_runOS (k n : ℕ) {L : Fin 1 → ℕ} (f : (i : Fin 1) → Fin (L i) → Fin 256) :
    (Draw.drawUnits (.subset k) n (Law.streamOf (f 0))).toOption.map Prod.fst =
      (Law.runOS f (.subset k) n).map (·.units) := by
  have hT : Law.tapesOf f 0 = Law.streamOf (f 0) := by simp [Law.tapesOf, Array.getD_eq_getD_getElem?]
  have hsim : (Flock.Draw.drawWith (Law.tapeGet (Law.tapesOf f)) (.subset k) n).run (fun _ => 0) =
      onTape (fun _ => 0)
        ((Flock.Draw.drawWith (fun _ need => Draw.take (Law.streamOf (f 0)) need) (.subset k) n).run 0) := by
    have h := tapeSim_drawWith (Law.tapesOf f) k n (fun _ => 0)
    rw [hT] at h
    exact h
  unfold Law.runOS Draw.drawUnits
  rw [hsim]
  cases (Flock.Draw.drawWith (fun _ need => Draw.take (Law.streamOf (f 0)) need) (.subset k) n).run 0 with
  | error e => rfl
  | ok x =>
    obtain ⟨r, read⟩ := x
    cases r <;> rfl

/-- A law object `flockLaw` reads as a subset law is of kind `subset`, drawing no more than the population. -/
theorem flockLaw_subset (law : Json) (n k : ℕ) (h : Draw.flockLaw law n = .ok (.subset k)) :
    kindOf law = "subset" ∧ k ≤ n := by
  unfold Draw.flockLaw at h
  split at h
  next s hs =>
    have hk : kindOf law = s := by simp [kindOf, hs]
    dsimp only at h
    split at h
    · split at h
      · exact absurd h (throw_bind_ne _ _ _)
      split at h
      · split at h
        · exact absurd h (throw_bind_ne _ _ _)
        · obtain rfl := Flock.Draw.Law.subset.inj (Except.ok.inj h)
          exact ⟨by rwa [hk], by omega⟩
      · exact absurd h (throw_ne _ _)
    all_goals exfalso
    all_goals repeat' (first
      | exact absurd h (throw_bind_ne _ _ _)
      | exact absurd h (throw_ne _ _)
      | exact absurd (Except.ok.inj h) (by split <;> simp)
      | exact absurd (Except.ok.inj h) (by simp)
      | split at h)
  next => exact absurd h (throw_ne _ _)

/-- `deriveLaw` expands no law of kind `subset`. -/
theorem deriveLaw_subset (d : Program) (query law : Json) (table : Option Json) (derived : Option Flock.Draw.Law)
    (hk : kindOf law = "subset") (h : Draw.deriveLaw d query law table = .ok derived) : derived = none := by
  unfold Draw.deriveLaw at h
  split at h
  next s hs =>
    have hs' : kindOf law = s := by simp [kindOf, hs]
    rw [hk] at hs'
    subst hs'
    exact (Except.ok.inj h).symm
  next hne =>
    exfalso
    have h0 : kindOf law = "" := by
      unfold kindOf; split
      · rename_i s hs; exact absurd hs (hne s)
      · rfl
    rw [hk] at h0
    exact absurd h0 (by decide)

/-- **An accepted subset draw of `Select.select` proves `runOS`'s draw**: at a rule other than `All` whose law over the
population C-Flock reads as `.subset k`, with no closure map, an accepted draw on the bytes `f 0` proves every unit that
the live draw (`Law.runOS`) draws on `f`, which therefore runs. -/
theorem select_subset_runOS (rule : Select.Rule) (hne : rule ≠ .all) (d : Program) (query : Json)
    (table : Option Json) (n : ℕ) (named : Json) (k : ℕ) (hn : Partition.units d query = .ok n)
    (hnamed : rule.lawOver n = .ok named) (hk : Draw.flockLaw named n = .ok (.subset k))
    {L : Fin 1 → ℕ} (f : (i : Fin 1) → Fin (L i) → Fin 256) (draw : Json) (proved : Array ℕ) (read : ℕ)
    (h : Select.select rule d query table none (Law.streamOf (f 0)) = .ok (draw, proved, read)) :
    ∃ dr, Law.runOS f (.subset k) n = some dr ∧ ∀ x ∈ dr.units, x ∈ proved := by
  obtain ⟨n', named', derived, hn', hnamed', hder, units, hsub, hcase⟩ :=
    select_ok rule d query table none _ draw proved read h
  have e1 : n = n' := by rw [hn] at hn'; exact Except.ok.inj hn'
  subst e1
  have e2 : named = named' := by rw [hnamed] at hnamed'; exact Except.ok.inj hnamed'
  subst e2
  have hd : derived = none := deriveLaw_subset d query named table derived (flockLaw_subset named n k hk).1 hder
  subst hd
  rcases hcase with ⟨hall, -⟩ | ⟨-, parsed, read', hp, hu⟩
  · exact absurd hall hne
  · have hp' : Draw.flockLaw named n = .ok parsed := hp
    rw [hk] at hp'
    have e3 : parsed = .subset k := (Except.ok.inj hp').symm
    subst e3
    have hu' : Draw.drawUnits (.subset k) n (Law.streamOf (f 0)) = .ok (units, read') := hu
    have heq := drawUnits_subset_runOS k n f
    rw [hu'] at heq
    cases hr : Law.runOS f (.subset k) n with
    | none =>
      rw [hr] at heq
      exact absurd heq (by simp [Except.toOption])
    | some dr =>
      rw [hr] at heq
      have hud : units = dr.units := by simpa [Except.toOption] using heq
      exact ⟨dr, rfl, fun x hx => hsub x (by rw [hud]; exact hx)⟩

/-- **`Select.select`'s subset draw misses `B` no more often than the uniform `k`-subset**, on coins uniform by A3
(`Assumptions.UniformRandomBytes`), through the live draw's bound (`Law.drawOS_subset_escape_le`). -/
theorem select_subset_escape_le (rule : Select.Rule) (hne : rule ≠ .all) (d : Program) (query : Json)
    (table : Option Json) (n : ℕ) (named : Json) (k : ℕ) (hn : Partition.units d query = .ok n)
    (hnamed : rule.lawOver n = .ok named) (hk : Draw.flockLaw named n = .ok (.subset k)) (hkn : k ≤ n)
    {Ω : Type} [Fintype Ω] (L : Fin 1 → ℕ) (t : Ω → (i : Fin 1) → Fin (L i) → Fin 256)
    (ht : FlockSoundness.Assumptions.UniformRandomBytes L t) (B : Finset (Fin n)) :
    prCoin (fun ω => ∃ draw proved read,
      Select.select rule d query table none (Law.streamOf (t ω 0)) = .ok (draw, proved, read) ∧
        ∀ u ∈ B, u.val ∉ proved) ≤ (Law.subset n k hkn).escape B := by
  refine le_trans (prCoin_mono fun ω hω => ?_) (Law.drawOS_subset_escape_le n k hkn L t ht B)
  obtain ⟨draw, proved, read, hs, hm⟩ := hω
  obtain ⟨dr, hdr, hsub⟩ := select_subset_runOS rule hne d query table n named k hn hnamed hk (t ω) draw proved read hs
  exact ⟨dr, hdr, fun u hu hud => hm u hu (hsub _ hud)⟩

end Properties.SampledProofs.SecurityProofs
