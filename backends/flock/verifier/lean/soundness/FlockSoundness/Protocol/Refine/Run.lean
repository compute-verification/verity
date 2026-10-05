import Flock.Transcript
import FlockSoundness.Protocol.Game.Basic

/-!
# A run of a game, and the rounds a run of the executable consumes

A `Game` is the verifier's side of an interaction, as a tree; nothing in it names a single run. `Game.Plays ι g ms cs a`
is one: `g`, played with the prover's messages `ms` and the verifier's coins `cs`, ends with `a`.

* **Messages** are typed: each entry of `ms` is a type and a value of it, and a send node takes the next entry only when
  its type is the node's.
* **Coins** are arrays of words, as a coin server records them. A coin node of type `Fin n → F` with `n ≠ 1` takes the
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

end FlockSoundness.Game

namespace FlockSoundness.Refine

open Flock

end FlockSoundness.Refine
