import Definitions.Accounting.Program
import Definitions.Pouw.Receipt.Run
import Mathlib.Data.Set.Card

/-!
# The fan-in floor for lane programs (trusted)

What any straight-line lane program (`Definitions/Accounting/Program.lean`) must spend, whatever its instructions
compute:

* `FanInFloor`: a program whose `m` instructions each read `f` earlier nodes, with `c` outputs, has at most
  `c + (f − 1)·m` essential inputs.
* `FoldFanIn`: so a program that computes a fold of `N` words through binding steps, on registers of at least one bit,
  has `N ≤ c + (f − 1)·m`: at least `(N − c)/(f − 1)` instructions, `½` a word at `f = 3`.
* `DuoFoldFanIn`, `Fx1FoldFanIn`: duo's and fx1's runs (`Definitions/Pouw/Receipt/Run.lean`), whose steps bind word
  and state (`Pouw.Guarantees.Receipt.DuoRunWordInjective`), so `N ≤ 2 + (f − 1)·m` for a program that outputs `(y, z)`.

This is the across-words part of FOLD-OPT (`Accounting.Assumptions.TS.FoldOpt`) for a class-free count of lane
instructions. It is a floor of `1/(f − 1)` a word, not the per-word fold's price.
-/

namespace Accounting.Guarantees

open Accounting.Program

/-- **The fan-in floor.** For every value type, a straight-line program with `m` instructions, each reading `f`
earlier nodes, and `c` output nodes has at most `c + (f − 1)·m` essential inputs. -/
def FanInFloor : Prop :=
  ∀ (α : Type) (n f m c : ℕ) (P : Prog α n f m) (out : Fin c → Fin (n + m)),
    {i | Essential (P.run out) i}.ncard ≤ c + (f - 1) * m

/-- **A fold of binding steps costs `(N − c)/(f − 1)` instructions.** For registers of `w + 1` bits, `N` words each
absorbed by a step that binds its word and its state, any start state, and any program with `m` instructions each
reading `f` earlier nodes: either the program's `c` outputs differ from the fold's final state on some run of words, or
`N ≤ c + (f − 1)·m`. -/
def FoldFanIn : Prop :=
  ∀ (w N f m c : ℕ) (g : Fin N → BindingStep c (BitVec (w + 1))) (s : Fin c → BitVec (w + 1))
    (P : Prog (BitVec (w + 1)) N f m) (out : Fin c → Fin (N + m)),
    (∃ d, P.run out d ≠ foldRun (fun i => (g i).1) s d) ∨ N ≤ c + (f - 1) * m

/-- **duo across words costs `(N − 2)/(f − 1)` instructions.** For registers of `w + 1` bits, any keys for each of `N`
words, any start state, and any program with `m` instructions each reading `f` earlier nodes: either its two outputs
differ from duo's final state `(y, z)` on some run of words, or `N ≤ 2 + (f − 1)·m`. -/
def DuoFoldFanIn : Prop :=
  ∀ (w N f m : ℕ) (k a : Fin N → BitVec (w + 1)) (r : Fin N → ℕ) (s : BitVec (w + 1) × BitVec (w + 1))
    (P : Prog (BitVec (w + 1)) N f m) (out : Fin 2 → Fin (N + m)),
    (∃ d, (P.run out d 0, P.run out d 1) ≠ Pouw.Receipt.duoRun k a r s d) ∨ N ≤ 2 + (f - 1) * m

/-- **fx1 across words costs `(N − 2)/(f − 1)` instructions.** As `DuoFoldFanIn`, for fx1 with an odd IMAD key for
each word. -/
def Fx1FoldFanIn : Prop :=
  ∀ (w N f m : ℕ) (m₁ : Fin N → Pouw.Receipt.OddKey (w + 1)) (k a : Fin N → BitVec (w + 1)) (r : Fin N → ℕ)
    (s : BitVec (w + 1) × BitVec (w + 1)) (P : Prog (BitVec (w + 1)) N f m) (out : Fin 2 → Fin (N + m)),
    (∃ d, (P.run out d 0, P.run out d 1) ≠ Pouw.Receipt.fx1Run (fun j => (m₁ j).1) k a r s d) ∨
      N ≤ 2 + (f - 1) * m

end Accounting.Guarantees
