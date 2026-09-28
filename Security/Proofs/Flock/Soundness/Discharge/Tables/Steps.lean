import Proofs.Flock.Soundness.Refine.LiveSim
import Proofs.Flock.Soundness.Game.Batch

/-!
# The model's moves inside a batch

A session of `J` tables runs `batch` over the tables, each table the `interleave` of its two reps. The simulation of a
live session moves one rep of one table at a time, so it needs the model's moves at a position deep inside that
structure as moves of the whole game.

* `Steps g g'`: a path of the model's messages from `g` to `g'` (data: the messages), and `Sim.ofSteps` plays it.
* `Ctx C`: a context `C` in which a message at the hole is a path of the whole game's messages, and a coin at the hole
  is, after such a path, a coin of the whole game. Contexts compose (`Ctx.comp`), and `bind`, either side of
  `interleave`, and a position of `batch` are contexts.
* `steps_batch_ofFn`: a batch whose every position has a path to an end has a path to the end with their outputs.
-/

namespace FlockSoundness.Discharge.Tables

open Game Refine

variable {α β γ : Type}

/-- **A path of the model's messages** from `g` to `g'`. -/
inductive Steps : Game α → Game α → Type 1
  | refl (g : Game α) : Steps g g
  | send {M : Type} {k : M → Game α} {g' : Game α} (m : M) : Steps (k m) g' → Steps (.send M k) g'

namespace Steps

def trans {g₁ g₂ g₃ : Game α} : Steps g₁ g₂ → Steps g₂ g₃ → Steps g₁ g₃
  | .refl _, s => s
  | .send m s, s' => .send m (s.trans s')

/-- Equal games: the empty path. -/
def ofEq {g g' : Game α} (h : g = g') : Steps g g' := h ▸ .refl g

end Steps

/-- The model plays a path of its messages, then continues. -/
def Sim.ofSteps {pL : β → Prop} {pg : α → Prop} {L : Game β} :
    {g g' : Game α} → Steps g g' → Sim pL pg L g' → Sim pL pg L g
  | _, _, .refl _, d => d
  | _, _, .send m s, d => .model m (Sim.ofSteps s d)

/-- After a context's path at a coin: the whole game's coin node and its branches, which are the context's at the
hole's branches. -/
structure CoinStep {A : Type} (C : Game A → Game γ) (Cm : Type) [Fintype Cm] [Nonempty Cm] (k : Cm → Game A) :
    Type 1 where
  next : Cm → Game γ
  steps : Steps (C (.coin Cm k)) (.coin Cm next)
  eq : ∀ c, next c = C (k c)

/-- **A context**: a message at the hole is a path of the whole game's messages, and a coin at the hole is a coin of
the whole game after such a path. -/
structure Ctx {A : Type} (C : Game A → Game γ) : Type 1 where
  send : ∀ {M : Type} (k : M → Game A) (m : M), Steps (C (.send M k)) (C (k m))
  coin : ∀ {Cm : Type} [Fintype Cm] [Nonempty Cm] (k : Cm → Game A), CoinStep C Cm k

namespace Ctx

variable {A B : Type}

/-- A path at the hole is a path of the whole game. -/
def lift {C : Game A → Game γ} (c : Ctx C) : {g g' : Game A} → Steps g g' → Steps (C g) (C g')
  | _, _, .refl _ => .refl _
  | _, _, .send m s => (c.send _ m).trans (c.lift s)

/-- Contexts compose. -/
def comp {C : Game A → Game B} {D : Game B → Game γ} (c : Ctx C) (d : Ctx D) : Ctx fun g => D (C g) where
  send k m := d.lift (c.send k m)
  coin k :=
    let s := c.coin k
    let t := d.coin s.next
    ⟨t.next, (d.lift s.steps).trans t.steps, fun x => by rw [t.eq, s.eq]⟩

/-- A context is one up to pointwise equality. -/
def congr {C C' : Game A → Game γ} (c : Ctx C) (h : ∀ g, C g = C' g) : Ctx C' where
  send k m := Steps.ofEq (h _).symm |>.trans ((c.send k m).trans (Steps.ofEq (h _)))
  coin k :=
    let s := c.coin k
    ⟨s.next, (Steps.ofEq (h _).symm).trans s.steps, fun x => by rw [s.eq, h]⟩

/-- The hole. -/
def id : Ctx (fun g : Game A => g) where
  send _ m := .send m (.refl _)
  coin k := ⟨k, .refl _, fun _ => rfl⟩

/-- `bind`'s first interaction is a context. -/
def bind (K : A → Game γ) : Ctx fun g : Game A => g.bind K where
  send _ m := .send m (.refl _)
  coin k := ⟨fun c => (k c).bind K, .refl _, fun _ => rfl⟩

/-- `interleave`'s first interaction is a context. -/
def left : (h : Game B) → Ctx fun g : Game A => interleave g h
  | .ret b => {
      send := fun k m => .send m (Steps.ofEq (Plays.interleave_ret (k m) b).symm)
      coin := fun k => ⟨fun c => (k c).map fun a => (a, b), .refl _, fun c => (Plays.interleave_ret (k c) b).symm⟩ }
  | .send _ _ => {
      send := fun _ m => .send true (.send m (.refl _))
      coin := fun k => ⟨fun c => interleave (k c) _, .send true (.refl _), fun _ => rfl⟩ }
  | @Game.coin _ _ _ _ _ => {
      send := fun _ m => .send true (.send m (.refl _))
      coin := fun k => ⟨fun c => interleave (k c) _, .send true (.refl _), fun _ => rfl⟩ }

/-- `interleave`'s second interaction is a context. -/
def right : (g : Game A) → Ctx fun h : Game B => interleave g h
  | .ret _ => {
      send := fun _ m => .send m (.refl _)
      coin := fun _ => ⟨_, .refl _, fun _ => rfl⟩ }
  | .send _ _ => {
      send := fun _ m => .send false (.send m (.refl _))
      coin := fun _ => ⟨_, .send false (.refl _), fun _ => rfl⟩ }
  | @Game.coin _ _ _ _ _ => {
      send := fun _ m => .send false (.send m (.refl _))
      coin := fun _ => ⟨_, .send false (.refl _), fun _ => rfl⟩ }

/-- A position of `batch` is a context. -/
def batchAt : (pre post : List (Game A)) → Ctx fun g => batch (pre ++ g :: post)
  | [], post => (left (batch post)).comp (bind _)
  | p :: pre, post => ((batchAt pre post).comp (right p)).comp (bind _)

end Ctx

theorem ofFn_update {A : Type*} {J : ℕ} (f : Fin J → A) (j : Fin J) (a : A) :
    List.ofFn (Function.update f j a) = (List.ofFn f).take j ++ a :: (List.ofFn f).drop (j + 1) := by
  apply List.ext_getElem
  · simp only [List.length_ofFn, List.length_append, List.length_take, List.length_cons, List.length_drop]
    omega
  · intro i h₁ h₂
    simp only [List.length_ofFn] at h₁
    rw [List.getElem_ofFn]
    rcases lt_trichotomy i j with hlt | rfl | hgt
    · rw [List.getElem_append_left (by simp only [List.length_take, List.length_ofFn]; omega), List.getElem_take, List.getElem_ofFn,
        Function.update_of_ne (Fin.ne_of_lt (by exact hlt))]
    · rw [List.getElem_append_right (by simp only [List.length_take, List.length_ofFn]; omega)]
      simp
    · rw [List.getElem_append_right (by simp only [List.length_take, List.length_ofFn]; omega), Function.update_of_ne (Fin.ne_of_gt (by exact hgt))]
      simp only [List.length_take, List.length_ofFn, List.getElem_cons, List.getElem_drop, List.getElem_ofFn]
      split
      · omega
      congr 1
      ext
      simp only
      omega

/-- **A table of a batch over `Fin J`** is a context. -/
def Ctx.batchFn {A : Type} {J : ℕ} (f : Fin J → Game A) (j : Fin J) :
    Ctx fun g => batch (List.ofFn (Function.update f j g)) :=
  (Ctx.batchAt ((List.ofFn f).take j) ((List.ofFn f).drop (j + 1))).congr fun g =>
    congrArg batch (ofFn_update f j g).symm

/-- **A batch whose every position has a path to an end** has a path to the end with their outputs. -/
def steps_batch_ofFn {A : Type} : {J : ℕ} → (f : Fin J → Game A) → (a : Fin J → A) →
    (∀ j, Steps (f j) (.ret (a j))) → Steps (batch (List.ofFn f)) (.ret (List.ofFn a))
  | 0, _, _, _ => .refl _
  | _ + 1, f, a, s =>
      (Steps.ofEq (congrArg batch (List.ofFn_succ (f := f)))).trans
        ((((Ctx.batchAt [] (List.ofFn fun i => f i.succ)).lift (s 0)).trans
          (((Ctx.right (.ret (a 0))).comp (Ctx.bind _)).lift
            (steps_batch_ofFn (fun i => f i.succ) (fun i => a i.succ) fun i => s i.succ))).trans
          (Steps.ofEq (by rw [List.ofFn_succ (f := a)]; rfl)))

end FlockSoundness.Discharge.Tables
