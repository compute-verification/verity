import Proofs.Flock.Recursive.Schedule

/-!
# Recursion: a fork at round 0 is any two runs

A strategy of `Inner.game` fixes its round-0 message, so the fork at round 0 (`forkG` at `k = 0`) continues it twice from
the root: any two transcripts the strategy reaches. `Inner.fork_of_outcomes` reads that as an `Inner.Fork`, which is how
the compile lemma turns a clash of two runs into a fork.

* `Game.Outcome.bind_of`, `Game.Outcome.map_of`: the outcomes of a sequenced game.
* `Game.prob_mono_outcome`: `prob_mono` with the implication needed only at outcomes the strategy reaches.
* `Game.PlaysT.det`, `Game.PlaysT.ne_nil_of_send`: a run on one tape ends at one outcome, and a game that starts with a
  message takes at least one step.
* `outcome_contF`, `outcome_rounds_length`: an outcome of `r` rounds is a transcript of length `r`, and an outcome of the
  rest of the run from the root.
* `Inner.fork_of_outcomes`: two outcomes of `s` that `pick` reads as a collision are a fork of `s`.
-/

namespace FlockSoundness.Game

variable {α β : Type}

theorem Outcome.bind_of : ∀ (g : Game α) (f : α → Game β) (s : Strategy g) (t : (a : α) → Strategy (f a)) {a : α}
    {b : β}, Outcome g s a → Outcome (f a) (t a) b → Outcome (g.bind f) (Strategy.bind g f s t) b
  | .ret _, _, _, _, _, _, ha, hb => by cases ha; exact hb
  | .send _ k, f, s, t, _, _, ha, hb => Outcome.bind_of (k s.1) f s.2 t ha hb
  | @Game.coin _ _ _ _ k, f, s, t, _, _, ha, hb => by
      obtain ⟨c, hc⟩ := ha
      exact ⟨c, Outcome.bind_of (k c) f (s c) t hc hb⟩

theorem Outcome.map_of (f : α → β) (g : Game α) (s : Strategy g) {a : α} (ha : Outcome g s a) :
    Outcome (g.map f) (Strategy.map f g s) (f a) :=
  Outcome.bind_of g (fun a => .ret (f a)) s (fun _ => PUnit.unit) ha (show f a = f a from rfl)

/-- **`prob_mono` at the reached outcomes**: the implication is needed only at outcomes `s` reaches. -/
theorem prob_mono_outcome {p q : α → Prop} :
    ∀ (g : Game α) (s : Strategy g), (∀ a, Outcome g s a → p a → q a) → prob p g s ≤ prob q g s
  | .ret a, _, h => by
      classical
      simp only [prob]
      by_cases hp : p a
      · simp [hp, h a rfl hp]
      · simp [hp]
  | .send _ k, s, h => prob_mono_outcome (k s.1) s.2 h
  | @Game.coin _ C _ _ k, s, h =>
      Finset.sum_le_sum fun c _ => mul_le_mul_right (prob_mono_outcome (k c) (s c) fun a ha => h a ⟨c, ha⟩) _

/-- **A run on one tape ends at one outcome.** -/
theorem PlaysT.det {g : Game α} {ts : List Step} {a b : α} (ha : PlaysT g ts a) (hb : PlaysT g ts b) : a = b := by
  induction ha generalizing b with
  | ret _ => exact hb.ret_inv.2.symm
  | send _ _ ih => exact ih hb.send_msg
  | coin _ _ ih => exact ih hb.coin_val

/-- A game that starts by receiving a message takes at least one step. -/
theorem PlaysT.ne_nil_of_send {M : Type} {k : M → Game α} {ts : List Step} {a : α} (h : PlaysT (.send M k) ts a) :
    ts ≠ [] := by
  rintro rfl
  cases h

end FlockSoundness.Game

namespace FlockSoundness.Discharge.Recursion

open Game

section Fork

variable {M C : Type} [Fintype C] [Nonempty C]

/-- **The rest of the run from a node reaches what the rounds reach**: an outcome of `r` rounds ending in the
transcript `h t` is an outcome of `contF` from the same node, its trial the leaf's transcript. -/
theorem outcome_contF : ∀ (r : ℕ) (h : List (M × C) → List (M × C))
    (s : Strategy (rounds M C r fun t => .ret (h t))) {b : List (M × C)},
    Outcome (rounds M C r fun t => .ret (h t)) s b →
      Outcome (contF r (fun t => .ret (h t)) (fun t _ => ⟨.ret (h t), PUnit.unit⟩) s).g
        (contF r (fun t => .ret (h t)) (fun t _ => ⟨.ret (h t), PUnit.unit⟩) s).s b
  | 0, _, _, _, hb => hb
  | r + 1, h, s, _, hb => by
      obtain ⟨ρ, hρ⟩ := hb
      exact ⟨ρ, outcome_contF r (fun t => h ((s.1, ρ) :: t)) (s.2 ρ) hρ⟩

/-- **An outcome of `r` rounds is a transcript of length `r`**, read by the leaf's `h`. -/
theorem outcome_rounds_length : ∀ (r : ℕ) (h : List (M × C) → List (M × C))
    (s : Strategy (rounds M C r fun t => .ret (h t))) {b : List (M × C)},
    Outcome (rounds M C r fun t => .ret (h t)) s b → ∃ t : List (M × C), t.length = r ∧ h t = b
  | 0, _, _, _, hb => ⟨[], rfl, hb⟩
  | r + 1, h, s, _, hb => by
      obtain ⟨ρ, hρ⟩ := hb
      obtain ⟨t, ht, rfl⟩ := outcome_rounds_length r (fun t => h ((s.1, ρ) :: t)) (s.2 ρ) hρ
      exact ⟨(s.1, ρ) :: t, by simp [ht], rfl⟩

/-- **The fork at round 0 finds what two runs find**: two outcomes of `s` that `pick` reads as a collision. -/
theorem forkG_zero_finds {D : Type} (H : List UInt8 → D) (pick : List (M × C) → List (M × C) → StrictCR.Pair)
    (r : ℕ) (s : Strategy (rounds M C (r + 1) .ret)) {t t' : List (M × C)}
    (ht : Outcome (rounds M C (r + 1) .ret) s t) (ht' : Outcome (rounds M C (r + 1) .ret) s t')
    (hc : StrictCR.IsColl H (pick t t')) :
    (forkG pick 0 (r + 1) .ret (fun t _ => ⟨.ret t, PUnit.unit⟩) s).Finds H := by
  have hT := outcome_contF (r + 1) (fun t => t) s ht
  have hT' := outcome_contF (r + 1) (fun t => t) s ht'
  show (StrictCR.twoRuns (contF (r + 1) (fun t => .ret t) (fun t _ => ⟨.ret t, PUnit.unit⟩) s).g
    (contF (r + 1) (fun t => .ret t) (fun t _ => ⟨.ret t, PUnit.unit⟩) s).s pick).Finds H
  exact ⟨pick t t', Outcome.bind_of _ _ _ _ hT (Outcome.map_of _ _ _ hT'), hc⟩

end Fork

/-- **A transcript an inner prover reaches has `I.r` rounds.** -/
theorem Inner.length_of_outcome (I : Inner) (s : Strategy I.game) {t : List (I.Msg × I.Coin)}
    (ht : Outcome I.game s t) : t.length = I.r := by
  obtain ⟨t', h, rfl⟩ := outcome_rounds_length I.r (fun t => t) s ht
  exact h

/-- **Two runs are a fork.** Two transcripts `s` reaches, which `pick` reads as a collision of `H`, are a fork of `s` at
round 0: `s` fixes its round-0 message, and the fork continues it twice from the root. -/
theorem Inner.fork_of_outcomes {D : Type} (H : List UInt8 → D) (I : Inner) (hr : 0 < I.r)
    (pick : List (I.Msg × I.Coin) → List (I.Msg × I.Coin) → StrictCR.Pair) (s : Strategy I.game)
    {t t' : List (I.Msg × I.Coin)} (ht : Outcome I.game s t) (ht' : Outcome I.game s t')
    (hc : StrictCR.IsColl H (pick t t')) : I.Fork H pick s := by
  refine ⟨0, hr, ?_⟩
  have key : ∀ (n : ℕ), 0 < n → ∀ (s : Strategy (rounds I.Msg I.Coin n .ret)),
      Outcome (rounds I.Msg I.Coin n .ret) s t → Outcome (rounds I.Msg I.Coin n .ret) s t' →
      (forkG pick 0 n .ret (fun t _ => ⟨.ret t, PUnit.unit⟩) s).Finds H := by
    intro n hn
    cases n with
    | zero => exact absurd hn (lt_irrefl 0)
    | succ n => exact fun s ht ht' => forkG_zero_finds H pick n s ht ht' hc
  exact key I.r hr s ht ht'

theorem pickOn_of {τ : Type} {G : τ → Prop} {pick : τ → τ → StrictCR.Pair} {t t' : τ} (h : G t) (h' : G t') :
    pickOn G pick t t' = pick t t' := by
  unfold pickOn
  rw [if_pos ⟨h, h'⟩]

end FlockSoundness.Discharge.Recursion
