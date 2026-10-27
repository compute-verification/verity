import Proofs.Flock.Claims.ERepCost
import Proofs.Flock.Soundness.Game.Expect

/-!
# A game finder as a `Claims.Finder`, and the sequential composition of a finite family

C-Flock's collision finders are games played by a fixed strategy (`FlockSoundness.Game`); a `Claims.Finder` is a uniform
coin space with an output and a cost on each coin. Two constructions relate them:

* `Game.tapeFinder g s out cost`: one tape holds a coin for every coin node of the tree `s` plays, drawn whether or not
  the run reaches it (`Game.Tape`), and the run is read off it (`Game.run`). Its success is the game's
  (`tapeFinder_Pr`), with no loss: averages over the tape are the game's expectations (`Game.Ex_run`).
* `Claims.Finder.seq H fam`: every member of a finite family on its own coins, in a fixed order, outputting the first
  collision of `H` found. It succeeds at least as often as each member (`le_seq`) and costs the members' sum plus
  two calls of `H` a check (`seq_cost_le`).
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

/-- An event of one coordinate of independent coordinates has that coordinate's probability. -/
theorem Pr_pi_eval {C : Type} [Fintype C] [DecidableEq C] {T : C → Type} [∀ c, Fintype (T c)]
    [∀ c, Nonempty (T c)] (c₀ : C) (p : T c₀ → Prop) :
    Pr (fun t : (c : C) → T c => p (t c₀)) = Pr p := by
  classical
  rw [← Ex_ind, ← Ex_ind]
  exact Ex_pi_eval c₀ (fun a => if p a then (1 : ℝ) else 0)

/-- The first entry of a list at which `f` returns something, where `f` returns something somewhere, satisfies
whatever every value of `f` on the list satisfies. -/
theorem findSome?_spec {β γ : Type} (f : β → Option γ) (P : γ → Prop) :
    ∀ l : List β, (∀ a ∈ l, ∀ b, f a = some b → P b) → ∀ a ∈ l, (f a).isSome →
      ∃ b, l.findSome? f = some b ∧ P b
  | [], _, a, ha, _ => by simp at ha
  | x :: xs, hP, a, ha, hs => by
    cases hx : f x with
    | some b => exact ⟨b, by simp [List.findSome?, hx], hP x (by simp) b hx⟩
    | none =>
      have hxs : a ∈ xs := by
        rcases List.mem_cons.1 ha with rfl | h
        · simp [hx] at hs
        · exact h
      obtain ⟨b, hb, hPb⟩ :=
        findSome?_spec f P xs (fun a' h' => hP a' (List.mem_cons_of_mem _ h')) a hxs hs
      exact ⟨b, by simp [List.findSome?, hx, hb], hPb⟩

open Classical in
/-- A finder's output where it is a collision of `H`, and nothing elsewhere. -/
noncomputable def collisionOf (H : Msg → Msg) : Option (Msg × Msg) → Option (Msg × Msg)
  | some (u, v) => if Collision H u v then some (u, v) else none
  | none => none

theorem collisionOf_spec (H : Msg → Msg) (o : Option (Msg × Msg)) (w : Msg × Msg) (h : collisionOf H o = some w) :
    Collision H w.1 w.2 := by
  rcases o with _ | ⟨u, v⟩
  · simp [collisionOf] at h
  · simp only [collisionOf] at h
    by_cases hc : Collision H u v
    · rw [if_pos hc] at h
      cases h
      exact hc
    · rw [if_neg hc] at h
      cases h

/-- **The sequential composition of a finite family of finders**: every member runs on its own coins, in the order of
`Fintype.equivFin`, and the first output that is a collision of `H` is output. Its cost is every member's plus two
calls of `H` for each check. -/
noncomputable def Finder.seq {ι : Type} [Fintype ι] (H : Msg → Msg) (fam : ι → Finder) : Finder :=
  haveI := Classical.decEq ι
  { S := (i : ι) → (fam i).S
    out := fun x => (List.finRange (Fintype.card ι)).findSome? fun k =>
      collisionOf H ((fam ((Fintype.equivFin ι).symm k)).out (x ((Fintype.equivFin ι).symm k)))
    cost := fun x => ∑ i, ((fam i).cost (x i) + 2) }

/-- **The sequential composition succeeds at least as often as each member.** -/
theorem Finder.le_seq {ι : Type} [Fintype ι] (H : Msg → Msg) (fam : ι → Finder) (i : ι) :
    (fam i).Pr H ≤ (Finder.seq H fam).Pr H := by
  haveI := Classical.decEq ι
  unfold Finder.Pr
  rw [← Pr_pi_eval (T := fun j => (fam j).S) i (fun s => ∃ u v, (fam i).out s = some (u, v) ∧ Collision H u v)]
  refine Pr_mono fun x hx => ?_
  obtain ⟨k, rfl⟩ : ∃ k, (Fintype.equivFin ι).symm k = i := ⟨_, (Fintype.equivFin ι).symm_apply_apply i⟩
  obtain ⟨u, v, hout, hc⟩ := hx
  have hsome : (collisionOf H ((fam ((Fintype.equivFin ι).symm k)).out (x ((Fintype.equivFin ι).symm k)))).isSome := by
    rw [hout]
    simp only [collisionOf, if_pos hc, Option.isSome_some]
  obtain ⟨⟨u', v'⟩, hb, hPb⟩ := findSome?_spec
    (fun k => collisionOf H ((fam ((Fintype.equivFin ι).symm k)).out (x ((Fintype.equivFin ι).symm k))))
    (fun w => Collision H w.1 w.2) (List.finRange (Fintype.card ι))
    (fun _ _ w hw => collisionOf_spec H _ w hw) k (List.mem_finRange _) hsome
  exact ⟨u', v', hb, hPb⟩

theorem Finder.seq_cost_le {ι : Type} [Fintype ι] (H : Msg → Msg) (fam : ι → Finder) (b : ℕ)
    (h : ∀ i (x : (fam i).S), (fam i).cost x ≤ b) (x : (Finder.seq H fam).S) :
    (Finder.seq H fam).cost x ≤ Fintype.card ι * (b + 2) := by
  show ∑ i, ((fam i).cost (x i) + 2) ≤ _
  calc ∑ i, ((fam i).cost (x i) + 2) ≤ ∑ _i : ι, (b + 2) :=
        Finset.sum_le_sum fun i _ => Nat.add_le_add_right (h i (x i)) 2
    _ = Fintype.card ι * (b + 2) := by simp [Finset.sum_const, Finset.card_univ]

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
  all_goals congr 1
  all_goals (funext t; unfold ind; exact if_congr Iff.rfl rfl rfl)

theorem tapeFinder_cost_le {β : Type} (g : Game β) (s : Strategy g) (out : β → Option (Claims.Msg × Claims.Msg))
    (cost : β → ℕ) (b : ℕ) (h : ∀ x, cost x ≤ b) (x : (tapeFinder g s out cost).S) :
    (tapeFinder g s out cost).cost x ≤ b :=
  h _

end FlockSoundness.Game
