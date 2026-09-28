import Proofs.Flock.Soundness.Game.Basic
import Flock.Transcript

/-!
# A run of a game, and the rounds a run of the executable consumes

A `Game` is the verifier's side of an interaction, as a tree; nothing in it names a single run. `Game.Plays ι g ms cs a`
is one: `g`, played with the prover's messages `ms` and the verifier's coins `cs`, ends with `a`.

* **Messages** are typed: each entry of `ms` is a type and a value of it, and a send node takes the next entry only when
  its type is the node's.
* **Coins** are arrays of words, as a challenger records them. A coin node of type `Fin n → F` with `n ≠ 1` takes the
  next array when it has `n` words and reads word `i` through `ι` as the coin's `i`-th value; a node of type `F` takes
  an array of one word. So a run's coin step is determined by its array: one word is always the scalar. The refinement
  fixes `ι` to the executable's field embedding, so the coins are the record's.

On the executable's side, `consumed rs k k'` is the coins of rounds `k, …, k' − 1` of a replay transcript (PROTOCOL.md
§6), and `RoundsWF` says every round's coin array has the round's coin count, as the record's check S12 ensures.
-/

namespace FlockSoundness.Game

/-- A message: its type and its value. -/
abbrev Msg := Σ M : Type, M

variable {W F : Type} [Inhabited W] (ι : W → F)

/-- **A run.** `g`, played with the messages `ms` and the coins `cs`, ends with `a`. -/
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

/-- The coins of rounds `k, …, k' − 1`. -/
def consumed (rs : Array Round) (k k' : ℕ) : List (Array F128) :=
  (List.range' k (k' - k)).map fun i => (rs[i]!).coins

/-- Every round's coin array has the round's coin count (S12). -/
def RoundsWF (rs : Array Round) : Prop := ∀ i < rs.size, (rs[i]!).coins.size = (rs[i]!).n

end FlockSoundness.Refine
