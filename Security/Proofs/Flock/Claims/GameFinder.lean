import Proofs.Flock.Claims.ERepCost
import Proofs.Flock.Soundness.Game.Expect

/-!
# A game finder as a `Claims.Finder`, diluted, and the worst of a finite family

C-Flock's collision finders are games played by a fixed strategy (`FlockSoundness.Game`); a `Claims.Finder` is a uniform
coin space with an output and a cost on each coin. Three constructions relate them:

* `Game.tapeFinder g s out cost`: one tape holds a coin for every coin node of the tree `s` plays, drawn whether or not
  the run reaches it (`Game.Tape`), and the run is read off it (`Game.run`). Its success is the game's
  (`tapeFinder_Pr`), with no loss: averages over the tape are the game's expectations (`Game.Ex_run`).
* `Claims.Finder.dilute m N F`: `F` beside a uniform `u : Fin (N + 1)`, run where `u < m` and outputting nothing
  elsewhere. Its success is `m / (N + 1)` times `F`'s (`dilute_Pr`).
* `Claims.Finder.worst H fam`: a member of a finite family with the largest success at `H` (`le_worst`).
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
    _ = Ex (fun p : T c₀ × ((j : { j // j ≠ c₀ }) → T j) => h p.1) := Ex_equiv _ _
    _ = Ex (fun a : T c₀ => Ex fun _ : (j : { j // j ≠ c₀ }) → T j => h a) := Ex_prod _
    _ = Ex h := by simp only [Ex_const]

/-- Independent events of two coordinates. -/
theorem Pr_and_prod {α β : Type} [Fintype α] [Fintype β] (p : α → Prop) (q : β → Prop) :
    Pr (fun x : α × β => p x.1 ∧ q x.2) = Pr p * Pr q := by
  classical
  rw [Pr_eq_sum (fun x : α × β => p x.1 ∧ q x.2), Pr_eq_sum p, Pr_eq_sum q, div_mul_div_comm, Finset.sum_mul_sum,
    Fintype.sum_prod_type, Fintype.card_prod]
  push_cast
  congr 1
  refine Finset.sum_congr rfl fun a _ => Finset.sum_congr rfl fun b _ => ?_
  split_ifs <;> simp_all

theorem Pr_fin_lt {m N : ℕ} (hm : m ≤ N + 1) : Pr (fun u : Fin (N + 1) => u.val < m) = m / (N + 1) := by
  rw [Pr_eq_sum, Finset.sum_boole, Fin.card_filter_val_lt, min_eq_right hm, Fintype.card_fin]
  push_cast
  rfl

/-- **`F` diluted to `m` in `N + 1`**: `F` beside a uniform `u : Fin (N + 1)`, with `F`'s output and cost where
`u < m`, and no output and no cost elsewhere. -/
def Finder.dilute (m N : ℕ) (F : Finder) : Finder where
  S := Fin (N + 1) × F.S
  out x := if x.1.val < m then F.out x.2 else none
  cost x := if x.1.val < m then F.cost x.2 else 0

theorem Finder.dilute_Pr (H : Msg → Msg) (F : Finder) {m N : ℕ} (hm : m ≤ N + 1) :
    (F.dilute m N).Pr H = m / (N + 1) * F.Pr H := by
  unfold Finder.Pr
  show Pr (fun x : Fin (N + 1) × F.S =>
    ∃ u v, (if x.1.val < m then F.out x.2 else none) = some (u, v) ∧ Collision H u v) = _
  have e : ∀ x : Fin (N + 1) × F.S,
      (∃ u v, (if x.1.val < m then F.out x.2 else none) = some (u, v) ∧ Collision H u v) ↔
        (x.1.val < m ∧ ∃ u v, F.out x.2 = some (u, v) ∧ Collision H u v) := by
    intro x
    by_cases h : x.1.val < m <;> simp [h]
  rw [Pr_congr e, Pr_and_prod (fun u : Fin (N + 1) => u.val < m), Pr_fin_lt hm]

theorem Finder.dilute_cost_le (F : Finder) (m N b : ℕ) (h : ∀ x, F.cost x ≤ b) (x : (F.dilute m N).S) :
    (F.dilute m N).cost x ≤ b := by
  show (if x.1.val < m then F.cost x.2 else 0) ≤ b
  split_ifs
  · exact h _
  · exact Nat.zero_le _

/-- **The worst of a finite family**: a member with the largest probability of outputting a collision of `H`. -/
noncomputable def Finder.worst {ι : Type} [Finite ι] [Nonempty ι] (H : Msg → Msg) (fam : ι → Finder) : Finder :=
  fam (Classical.choose (Finite.exists_max fun i => (fam i).Pr H))

theorem Finder.le_worst {ι : Type} [Finite ι] [Nonempty ι] (H : Msg → Msg) (fam : ι → Finder) (i : ι) :
    (fam i).Pr H ≤ (Finder.worst H fam).Pr H :=
  Classical.choose_spec (Finite.exists_max fun i => (fam i).Pr H) i

theorem Finder.worst_cost_le {ι : Type} [Finite ι] [Nonempty ι] (H : Msg → Msg) (fam : ι → Finder) (b : ℕ)
    (h : ∀ i (x : (fam i).S), (fam i).cost x ≤ b) (x : (Finder.worst H fam).S) :
    (Finder.worst H fam).cost x ≤ b :=
  h _ x

/-- The finder that outputs nothing. -/
def Finder.null : Finder where
  S := Unit
  out _ := none
  cost _ := 0

theorem Finder.null_Pr (H : Msg → Msg) : Finder.null.Pr H = 0 := by
  unfold Finder.Pr
  exact (Pr_congr fun _ => by simp [Finder.null]).trans Pr_false

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
            (ih c (s c))
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
  congr 1
  funext t
  unfold ind
  exact if_congr Iff.rfl rfl rfl

theorem tapeFinder_cost_le {β : Type} (g : Game β) (s : Strategy g) (out : β → Option (Claims.Msg × Claims.Msg))
    (cost : β → ℕ) (b : ℕ) (h : ∀ x, cost x ≤ b) (x : (tapeFinder g s out cost).S) :
    (tapeFinder g s out cost).cost x ≤ b :=
  h _

end FlockSoundness.Game
