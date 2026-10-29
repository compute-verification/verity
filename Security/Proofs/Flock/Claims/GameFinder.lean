import Proofs.Flock.Claims.ERepCost
import Proofs.Flock.Soundness.Game.Expect

/-!
# A game finder as a `Claims.Finder`, and the uniform mixture of a finite family

C-Flock's collision finders are games played by a fixed strategy (`FlockSoundness.Game`); a `Claims.Finder` is a uniform
coin space with an output and a cost on each coin. Two constructions relate them:

* `Game.tapeFinder g s out cost`: one tape holds a coin for every coin node of the tree `s` plays, drawn whether or not
  the run reaches it (`Game.Tape`), and the run is read off it (`Game.run`). Its success is the game's
  (`tapeFinder_Pr`), with no loss: averages over the tape are the game's expectations (`Game.Ex_run`).
* `Claims.Finder.mix fam`: a uniform member of a finite family of `k`, run on its own coins. Each member succeeds at
  most `k` times as often (`le_mix`), and it costs what its members do (`mix_cost_le`).
-/

namespace Claims

theorem Ex_inst {β : Type} (i j : Fintype β) (f : β → ℝ) : @Ex β i f = @Ex β j f := by
  obtain rfl := Subsingleton.elim i j
  rfl

theorem Ex_equiv {β γ : Type} [Fintype β] [Fintype γ] (e : β ≃ γ) (f : γ → ℝ) :
    Ex (fun b => f (e b)) = Ex f := by
  unfold Ex
  rw [Fintype.sum_equiv e (fun b => f (e b)) f (fun _ => rfl), Fintype.card_congr e]

/-- An average over independent coordinates of a function of one coordinate is its average over that coordinate. -/
theorem Ex_pi_eval {C : Type} [Fintype C] [DecidableEq C] {T : C → Type} [∀ c, Fintype (T c)]
    [∀ c, Nonempty (T c)] (c₀ : C) (h : T c₀ → ℝ) :
    Ex (fun t : (c : C) → T c => h (t c₀)) = Ex h := by
  calc Ex (fun t : (c : C) → T c => h (t c₀))
      = Ex (fun t : (c : C) → T c => (fun p : T c₀ × ((j : { j // j ≠ c₀ }) → T j) => h p.1)
          (Equiv.piSplitAt c₀ T t)) := rfl
    _ = Ex (fun p : T c₀ × ((j : { j // j ≠ c₀ }) → T j) => h p.1) :=
        Ex_equiv (Equiv.piSplitAt c₀ T) (fun p : T c₀ × ((j : { j // j ≠ c₀ }) → T j) => h p.1)
    _ = Ex (fun a : T c₀ => Ex fun _ : (j : { j // j ≠ c₀ }) → T j => h a) := Ex_prod _
    _ = Ex h := by simp only [Ex_const]

/-- **The uniform mixture of a finite family of finders**: a uniform member, run on its own coins. Its cost is that
member's. -/
noncomputable def Finder.mix {ι : Type} [Fintype ι] [Nonempty ι] (fam : ι → Finder) : Finder :=
  haveI := Classical.decEq ι
  { S := ι × ((i : ι) → (fam i).S)
    out := fun x => (fam x.1).out (x.2 x.1)
    cost := fun x => (fam x.1).cost (x.2 x.1) }

/-- **The mixture succeeds with the members' average success.** -/
theorem Finder.mix_Pr {ι : Type} [Fintype ι] [Nonempty ι] (H : Msg → Msg) (fam : ι → Finder) :
    (Finder.mix fam).Pr H = Ex fun i => (fam i).Pr H := by
  classical
  unfold Finder.Pr
  show Claims.Pr (fun x : ι × ((i : ι) → (fam i).S) =>
      ∃ u v, (fam x.1).out (x.2 x.1) = some (u, v) ∧ Collision H u v) =
    Ex fun i => Claims.Pr fun s : (fam i).S => ∃ u v, (fam i).out s = some (u, v) ∧ Collision H u v
  rw [← Ex_ind, Ex_prod]
  congr 1
  funext i
  rw [← Ex_ind]
  exact Ex_pi_eval (T := fun j => (fam j).S) i
    (fun s => if ∃ u v, (fam i).out s = some (u, v) ∧ Collision H u v then (1 : ℝ) else 0)

/-- **Each member succeeds at most `k` times as often as the mixture**, `k` the family's size. -/
theorem Finder.le_mix {ι : Type} [Fintype ι] [Nonempty ι] (H : Msg → Msg) (fam : ι → Finder) (i : ι) :
    (fam i).Pr H ≤ Fintype.card ι * (Finder.mix fam).Pr H := by
  have hk : (Fintype.card ι : ℝ) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
  have hi : (fam i).Pr H ≤ ∑ j, (fam j).Pr H :=
    Finset.single_le_sum (f := fun j => (fam j).Pr H) (fun j _ => Pr_nonneg _) (Finset.mem_univ i)
  rw [Finder.mix_Pr]
  unfold Ex
  calc (fam i).Pr H ≤ ∑ j, (fam j).Pr H := hi
    _ = Fintype.card ι * ((∑ j, (fam j).Pr H) / Fintype.card ι) := by
      first
        | (field_simp; done)
        | (field_simp; ring)
        | exact (mul_div_cancel₀ _ hk).symm

theorem Finder.mix_cost_le {ι : Type} [Fintype ι] [Nonempty ι] (fam : ι → Finder) (b : ℕ)
    (h : ∀ i (x : (fam i).S), (fam i).cost x ≤ b) (x : (Finder.mix fam).S) : (Finder.mix fam).cost x ≤ b :=
  h _ _

/-- The finder that outputs nothing. -/
def Finder.null : Finder where
  S := Unit
  out _ := none
  cost _ := 0

end Claims

namespace FlockSoundness.Game

variable {α : Type}

/-- **The strategy's coin tape**: a coin for every coin node of the tree `s` plays, drawn whether or not the run reaches
it. -/
def Tape : (g : Game α) → Strategy g → Type
  | .ret _, _ => Unit
  | .send _ k, s => Tape (k s.1) s.2
  | @Game.coin _ C _ _ k, s => C × ((c : C) → Tape (k c) (s c))

noncomputable instance instFintypeTape : (g : Game α) → (s : Strategy g) → Fintype (Tape g s)
  | .ret _, _ => inferInstanceAs (Fintype Unit)
  | .send _ k, s => instFintypeTape (k s.1) s.2
  | @Game.coin _ C _ _ k, s =>
      haveI := fun c => instFintypeTape (k c) (s c)
      haveI := Classical.decEq C
      inferInstanceAs (Fintype (C × ((c : C) → Tape (k c) (s c))))

instance instNonemptyTape : (g : Game α) → (s : Strategy g) → Nonempty (Tape g s)
  | .ret _, _ => inferInstanceAs (Nonempty Unit)
  | .send _ k, s => instNonemptyTape (k s.1) s.2
  | @Game.coin _ C _ _ k, s =>
      haveI := fun c => instNonemptyTape (k c) (s c)
      inferInstanceAs (Nonempty (C × ((c : C) → Tape (k c) (s c))))

/-- **The run of `s` on a tape**: at each coin node, the tape's coin there. -/
def run : (g : Game α) → (s : Strategy g) → Tape g s → α
  | .ret a, _, _ => a
  | .send _ k, s, t => run (k s.1) s.2 t
  | @Game.coin _ C _ _ k, s, t =>
      let t' : C × ((c : C) → Tape (k c) (s c)) := t
      run (k t'.1) (s t'.1) (t'.2 t'.1)

/-- **Averages over the tape are the game's expectations.** -/
theorem Ex_run (f : α → ℝ) (g : Game α) :
    ∀ s : Strategy g, Claims.Ex (fun t : Tape g s => f (run g s t)) = expect f g s := by
  induction g with
  | ret a =>
    intro s
    exact (Claims.Ex_inst (β := Unit) (instFintypeTape (.ret a) s) inferInstance (fun _ => f a)).trans
      (Claims.Ex_const (f a))
  | send M k ih =>
    intro s
    exact ih s.1 s.2
  | @coin C iF iN k ih =>
    intro s
    classical
    haveI : ∀ c, Fintype (Tape (k c) (s c)) := fun c => instFintypeTape (k c) (s c)
    haveI : ∀ c, Nonempty (Tape (k c) (s c)) := fun c => instNonemptyTape (k c) (s c)
    calc Claims.Ex (fun t : Tape (Game.coin C k) s => f (run (Game.coin C k) s t))
        = @Claims.Ex (C × ((c : C) → Tape (k c) (s c))) inferInstance
            (fun t : C × ((c : C) → Tape (k c) (s c)) => f (run (k t.1) (s t.1) (t.2 t.1))) :=
          Claims.Ex_inst (β := C × ((c : C) → Tape (k c) (s c))) (instFintypeTape (Game.coin C k) s) _ _
      _ = Claims.Ex (fun c : C =>
            Claims.Ex (fun t : (c : C) → Tape (k c) (s c) => f (run (k c) (s c) (t c)))) :=
          Claims.Ex_prod _
      _ = Claims.Ex (fun c : C => expect f (k c) (s c)) := by
          congr 1
          funext c
          exact (Claims.Ex_pi_eval (T := fun c => Tape (k c) (s c)) c (fun x => f (run (k c) (s c) x))).trans
            ((Claims.Ex_inst _ _ _).trans (ih c (s c)))
      _ = expect f (Game.coin C k) s := rfl

/-- **A game finder as a `Claims.Finder`**: `s`'s run on a tape, with output `out` and cost `cost` there. -/
noncomputable def tapeFinder {β : Type} (g : Game β) (s : Strategy g) (out : β → Option (Claims.Msg × Claims.Msg))
    (cost : β → ℕ) : Claims.Finder where
  S := Tape g s
  out t := out (run g s t)
  cost t := cost (run g s t)

/-- **The tape finder succeeds as the game does**: its probability of outputting a collision of `H` is the game's. -/
theorem tapeFinder_Pr {β : Type} (H : Claims.Msg → Claims.Msg) (g : Game β) (s : Strategy g)
    (out : β → Option (Claims.Msg × Claims.Msg)) (cost : β → ℕ) :
    (tapeFinder g s out cost).Pr H =
      (prob (fun b => ∃ u v, out b = some (u, v) ∧ Claims.Collision H u v) g s).toReal := by
  classical
  rw [prob_toReal, ← Ex_run]
  show Claims.Pr (fun t : Tape g s => ∃ u v, out (run g s t) = some (u, v) ∧ Claims.Collision H u v) = _
  rw [← Claims.Ex_ind]
  all_goals congr 1
  all_goals (funext t; unfold ind; exact if_congr Iff.rfl rfl rfl)

theorem tapeFinder_cost_le {β : Type} (g : Game β) (s : Strategy g) (out : β → Option (Claims.Msg × Claims.Msg))
    (cost : β → ℕ) (b : ℕ) (h : ∀ x, cost x ≤ b) (x : (tapeFinder g s out cost).S) :
    (tapeFinder g s out cost).cost x ≤ b :=
  h _

end FlockSoundness.Game
