module

import Proofs.Flock.Soundness.Game
public import Proofs.Flock.Soundness.StrictCR
import Mathlib.Combinatorics.Matroid.Init

/-! # What V*'s bridge obligation reads of `Recursive.Defs`, without its recursion imports. -/

@[expose] public section

namespace FlockSoundness.Game

variable {α : Type}

/-- **An outcome the strategy reaches**: `s` plays `g` to `a` at some coins. -/
def Outcome : (g : Game α) → Strategy g → α → Prop
  | .ret b, _, a => b = a
  | .send _ k, s, a => Outcome (k s.1) s.2 a
  | @Game.coin _ _ _ _ k, s, a => ∃ c, Outcome (k c) (s c) a

end FlockSoundness.Game

namespace FlockSoundness.StrictCR

open Game

variable {D β : Type} (H : List UInt8 → D)

/-- **A run of `g` against `s` finds a collision of `H`**: some outcome `s` reaches outputs, by `out`, two different
strings that `H` maps to one digest. -/
def FindsColl (g : Game β) (s : Strategy g) (out : β → Pair) : Prop :=
  ∃ b, Game.Outcome g s b ∧ IsColl H (out b)

/-- **A finder finds a collision of `H`** at an outcome its strategy reaches. -/
def Finder.Finds (Fd : Finder Pair) : Prop := FindsColl H Fd.g Fd.s id

end FlockSoundness.StrictCR

namespace FlockSoundness.Discharge.Recursion

open Game Audit

open scoped ENNReal

/-- **`r` committed rounds**: in each, the prover sends a message of type `M`, and then the verifier flips a live coin from
`C`, independently of everything before. The continuation reads the transcript, first round first. -/
def rounds (M C : Type) [Fintype C] [Nonempty C] {β : Type} : ℕ → (List (M × C) → Game β) → Game β
  | 0, f => f []
  | r + 1, f => .send M fun m => .coin C fun ρ => rounds M C r fun t => f ((m, ρ) :: t)

/-- **An inner protocol**: an interactive verifier with `r` rounds of public coins, messages committed before coins. In
each round the prover sends a message of type `Msg` and the verifier flips a uniform coin from `Coin`; after the last
round it decides `accepts x t` from the statement `x` and the transcript `t`, first round first. `holds x` is the
statement's truth. -/
structure Inner : Type 1 where
  Stmt : Type
  Msg : Type
  Coin : Type
  [fintype : Fintype Coin]
  [nonempty : Nonempty Coin]
  r : ℕ
  accepts : Stmt → List (Msg × Coin) → Prop
  holds : Stmt → Prop

attribute [instance] Inner.fintype Inner.nonempty

/-- The inner verifier's side of the interaction, its output the transcript. -/
def Inner.game (I : Inner) : Game (List (I.Msg × I.Coin)) := rounds I.Msg I.Coin I.r .ret

section Fork

variable {M C β γ : Type} [Fintype C] [Nonempty C]

/-- **The rest of the run from a node**: the remaining round coins, the prover's messages read from its strategy, and then
the finder `T` at the leaf reached. -/
noncomputable def contF : (r : ℕ) → (f : List (M × C) → Game β) →
    ((t : List (M × C)) → Strategy (f t) → StrictCR.Finder γ) → Strategy (rounds M C r f) → StrictCR.Finder γ
  | 0, _, T, s => T [] s
  | r + 1, f, T, s =>
      ⟨.coin C fun ρ => (contF r (fun t => f ((s.1, ρ) :: t)) (fun t => T ((s.1, ρ) :: t)) (s.2 ρ)).g,
        fun ρ => (contF r (fun t => f ((s.1, ρ) :: t)) (fun t => T ((s.1, ρ) :: t)) (s.2 ρ)).s⟩

/-- **The fork at round `k`**: the round coins to the node where the prover sends its round-`k` message, then the rest of
the run twice, independently (`contF`, the trial `T` at each leaf), the two trials' outputs read by `pick`. With no round
`k` it finds nothing. -/
noncomputable def forkG (pick : γ → γ → StrictCR.Pair) : (k r : ℕ) → (f : List (M × C) → Game β) →
    ((t : List (M × C)) → Strategy (f t) → StrictCR.Finder γ) → Strategy (rounds M C r f) →
    StrictCR.Finder StrictCR.Pair
  | _, 0, _, _, _ => ⟨.ret none, PUnit.unit⟩
  | 0, r + 1, f, T, s => StrictCR.twoRuns (contF (r + 1) f T s).g (contF (r + 1) f T s).s pick
  | k + 1, r + 1, f, T, s =>
      ⟨.coin C fun ρ => (forkG pick k r (fun t => f ((s.1, ρ) :: t)) (fun t => T ((s.1, ρ) :: t)) (s.2 ρ)).g,
        fun ρ => (forkG pick k r (fun t => f ((s.1, ρ) :: t)) (fun t => T ((s.1, ρ) :: t)) (s.2 ρ)).s⟩

end Fork

/-- **A fork of an inner prover's own runs**: at some round `k < r`, two continuations of `s` after its round-`k` message,
each run to the end, and `pick` of their two transcripts outputs a collision of `H` at an outcome the fork reaches. -/
def Inner.Fork {D : Type} (H : List UInt8 → D) (I : Inner)
    (pick : List (I.Msg × I.Coin) → List (I.Msg × I.Coin) → StrictCR.Pair) (s : Strategy I.game) : Prop :=
  ∃ k < I.r, (forkG pick k I.r .ret (fun t _ => ⟨.ret t, PUnit.unit⟩) s).Finds H

end FlockSoundness.Discharge.Recursion
