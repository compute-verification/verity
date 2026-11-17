module

public import Flock.Transcript
public import Definitions.Flock.Game
import Mathlib.Algebra.Order.Module.Field
import Mathlib.Basic.ENNReal.Real

/-! A run of a game (`Game.Plays`), and the rounds a run of the executable consumes (`consumed`). -/

@[expose] public section

namespace FlockSoundness.Game

/-- A message: its type and its value. -/
abbrev Msg := Σ M : Type, M

variable {W F : Type} [Inhabited W] (ι : W → F)

/-- A coin node of type `Fin n → F` with `n ≠ 1` takes an array of `n` words read through `ι`, and a node of type `F`
an array of one word, so one word is always the scalar. -/
inductive Plays : {α : Type} → Game α → List Msg → List (Array W) → α → Prop
  | ret {α : Type} (a : α) : Plays (.ret a) [] [] a
  | send {α M : Type} {k : M → Game α} (m : M) {ms : List Msg} {cs : List (Array W)} {a : α} :
      Plays (k m) ms cs a → Plays (.send M k) (⟨M, m⟩ :: ms) cs a
  | coinVec {α : Type} {n : ℕ} {iF : Fintype (Fin n → F)} {iN : Nonempty (Fin n → F)} {k : (Fin n → F) → Game α}
      (c : Array W) (hc : c.size = n) (h1 : n ≠ 1) {ms : List Msg} {cs : List (Array W)} {a : α} :
      Plays (k fun i => ι c[i.val]!) ms cs a → Plays (@Game.coin α (Fin n → F) iF iN k) ms (c :: cs) a
  | coinOne {α : Type} {iF : Fintype F} {iN : Nonempty F} {k : F → Game α}
      (c : Array W) (hc : c.size = 1) {ms : List Msg} {cs : List (Array W)} {a : α} :
      Plays (k (ι c[0]!)) ms cs a → Plays (@Game.coin α F iF iN k) ms (c :: cs) a

end FlockSoundness.Game

namespace FlockSoundness.Refine

open Flock

def consumed (rs : Array Round) (k k' : ℕ) : List (Array F128) :=
  (List.range' k (k' - k)).map fun i => (rs[i]!).coins

def RoundsWF (rs : Array Round) : Prop := ∀ i < rs.size, (rs[i]!).coins.size = (rs[i]!).n

end FlockSoundness.Refine
