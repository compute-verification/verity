import Definitions.Accounting.TS.FanIn
import Definitions.Accounting.TS.Fold
import Definitions.Pouw.Receipt.Fx1

/-!
# The fan-in floor of a fold (trusted)

`FanInCount` (`Definitions/Accounting/Guarantees/TS.lean`) read for a fold of words, with every node of the dataflow
reading at most `f` sources, each `def G : Prop` proved as `Accounting.SecurityProofs.TS.G` in
`Proofs.Accounting.TS.FoldFanIn`:

* `FoldFanIn`: a dataflow whose `c` outputs hold the final state of a fold of binding steps over `N` words has at
  least `(N − c)/(f − 1)` nodes.
* `DuoFoldFanIn`, `Fx1FoldFanIn`: the same for duo's and fx1's runs (`c = 2`), with odd multipliers.

`f` counts every value a node's value depends on: its registers, predicates and carry-ins, and for a guarded
instruction its guard and everything it writes, which it keeps when the guard is false (its destination, carry-outs and
predicate destinations). Three registers give `f = 3`, ½ a node a word. A guarded three-register instruction gives
`f = 5`, ¼ a word; a guarded IADD3.X (three registers, two carry-ins, the guard, the destination and two carry-outs)
`f = 9`, ⅛ a word, and `f = 10` on sm_80, sm_89 and sm_90, whose `cx[UR][imm]` operand reads a uniform register and a
constant bank where a register was; the widest lane forms, a guarded P2R with that operand there and sm_120's IMNMX.U64
(three pairs, two predicate operands, the guard and two predicate destinations), `f = 11`, ⅒ a word. A warp-wide
node counts its lanes together, so its `f` is far larger: SHFL's 192, IMMA.16832's 481, and sm_100's LDTM.32x32b.x128,
which keeps its 128 registers on 32 lanes under one uniform guard, 4099 (`Accounting.Assumptions.TS.FanIn`).

No count of this kind gives more than `1/(f − 1)` a word. A tree of `f`-input XORs sees `N` words with
`⌈(N − 1)/(f − 1)⌉` nodes. A duo run is a dataflow of two 3-source nodes a word, since `y′` and `z′` are each a
function of `y`, `z` and the word. So duo's 3 instructions a word (`FoldOpt`) are out of reach of any such count.
-/

namespace Accounting.Guarantees.TS

open Accounting.TS

/-- **A fold of binding steps costs `(N − c)/(f − 1)` nodes.** For registers of `w + 1` bits, `N` words each absorbed
by a step that binds its word and its state, any start state, and any dataflow over the words with `c` output
vertices `out`: either some run's outputs are not the fold's final state, or some node reads more than `f` sources,
or the dataflow's `n` nodes have `N ≤ c + (f − 1)·n`. -/
def FoldFanIn : Prop :=
  ∀ (w N c f : ℕ) (g : Fin N → BindingStep c (BitVec (w + 1))) (s : Fin c → BitVec (w + 1))
    (D : Dataflow (Fin N) (BitVec (w + 1))) (out : Fin c → Fin N ⊕ Fin D.n),
    (∃ x v, D.Runs x v ∧ (fun o => v (out o)) ≠ foldRun (fun i => (g i).1) s x) ∨ (∃ i, f < (D.src i).card) ∨
      N ≤ c + (f - 1) * D.n

/-- **A duo run costs `(L − 2)/(f − 1)` nodes.** For registers of `n + 1` bits, any XOR keys, odd multipliers and
rotations for each of `L` words, any start state, and any dataflow over the words with two output vertices `out`:
either some run's outputs are not duo's final state `(y, z)`, or some node reads more than `f` sources, or the
dataflow's nodes number at least `(L − 2)/(f − 1)`. -/
def DuoFoldFanIn : Prop :=
  ∀ (n L f : ℕ) (k : Fin L → BitVec (n + 1)) (a : Fin L → Pouw.Receipt.OddKey (n + 1)) (r : Fin L → ℕ)
    (s : BitVec (n + 1) × BitVec (n + 1)) (D : Dataflow (Fin L) (BitVec (n + 1))) (out : Fin 2 → Fin L ⊕ Fin D.n),
    (∃ x v, D.Runs x v ∧ (v (out 0), v (out 1)) ≠ Pouw.Receipt.duoRun k (fun j => (a j).1) r x s) ∨
      (∃ i, f < (D.src i).card) ∨ L ≤ 2 + (f - 1) * D.n

/-- **An fx1 run costs `(L − 2)/(f − 1)` nodes.** As `DuoFoldFanIn`, for fx1 with an odd IMAD multiplier for each
word. -/
def Fx1FoldFanIn : Prop :=
  ∀ (n L f : ℕ) (k : Fin L → BitVec (n + 1)) (a : Fin L → Pouw.Receipt.OddKey (n + 1)) (r : Fin L → ℕ)
    (m : Fin L → Pouw.Receipt.OddKey (n + 1)) (s : BitVec (n + 1) × BitVec (n + 1))
    (D : Dataflow (Fin L) (BitVec (n + 1))) (out : Fin 2 → Fin L ⊕ Fin D.n),
    (∃ x v, D.Runs x v ∧
        (v (out 0), v (out 1)) ≠ Pouw.Receipt.fx1Run k (fun j => (a j).1) r (fun j => (m j).1) x s) ∨
      (∃ i, f < (D.src i).card) ∨ L ≤ 2 + (f - 1) * D.n

end Accounting.Guarantees.TS
