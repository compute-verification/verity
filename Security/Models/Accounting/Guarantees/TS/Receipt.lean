import Models.Accounting.Assumptions.TS
import Definitions.Accounting.TS.Receipt

/-!
# (I7) from the receipt (trusted)

The luck premise (I7) `LuckAt` split at the receipt (LEDGER §1.3–§1.5, note:20261017T1723Z-draft-ledger), each
`def G : Prop` proved as `Accounting.SecurityProofs.TS.G` in `Proofs.Accounting.TS.Receipt`:

* `LuckOfReceipt`: with the openings pinned, the operand luck at `p` and the receipt luck at `pr ≤ 1` give (I7) at
  `p + (1 − p)·pr`.
* `ReceiptLuckCaseA`: Lemma B in case (a), the receipt luck is 0 when every round's committed value is the full state
  of duo or fx1 units, one deficient word a unit (`Pouw.Guarantees.Receipt.DuoRunWordInjective`,
  `Fx1RunWordInjective`), or li units' checkpoint a segment, one deficient word a segment (`LiSegmentsBind`).
* `LuckCaseA`: the two together, (I7) at the operand luck `p` in case (a).
* `OperandLuckCube`: the operand luck at `luck_quad`'s ¾ for rounds that draw uniform cube bits against a degree-2
  operand error.
* `LuckCaseACube`: case (a) with such rounds, (I7) at ¾ outright.
* `LiSegmentsBind`: li units (LI-CP-W) bind their words: when every segment's warp checkpoint is the honest one, every
  segment's deficient word is right (`Pouw.Guarantees.Receipt.LiCheckpointWordInjective`, segment by segment).
* `LiWindowUnbound`: without the checkpoint a segment it fails: the window's last checkpoint alone keeps the honest
  value when a persistent skip flips the top bit of a word in two segments, whatever the keys.
-/

namespace Accounting.Guarantees.TS

open Accounting.TS Accounting.Assumptions.TS

/-- **(I7) from the operand and receipt lucks.** For every ideal game split at the receipt whose openings are pinned,
the operand luck at `p` and the receipt luck at `pr ≤ 1` give (L) at `p + (1 − p)·pr`: a deficient round passes only
if its committed value is the honest one, which happens with every deficient word right (probability at most `p`) or
with one wrong, kept by the receipt with probability at most `pr`. -/
def LuckOfReceipt : Prop :=
  ∀ {Ω : Type} [Fintype Ω] {S : ℕ} (G : ReceiptGame Ω S) (p pr : ℝ),
    G.Pinned → G.OperandLuck p → G.ReceiptLuck pr → pr ≤ 1 → LuckAt (p + (1 - p) * pr) G.game

/-- **Lemma B in case (a): the receipt luck is 0.** For every ideal game split at the receipt whose rounds commit the
full state of duo or fx1 units from pinned starts, one deficient word a unit, or li units' warp checkpoint a segment
from a pinned window start, one deficient word a segment, a round whose deficient words are not all right never keeps
the honest committed value: each unit's final state, or each segment's checkpoint, determines its deficient word. -/
def ReceiptLuckCaseA : Prop :=
  ∀ {Ω : Type} [Fintype Ω] {S : ℕ} (G : ReceiptGame Ω S), G.CaseA → G.ReceiptLuck 0

/-- **(I7) in case (a) at the operand luck.** For every ideal game split at the receipt whose openings are pinned and
whose rounds commit the full state of duo or fx1 units, one deficient word a unit, or li units' warp checkpoint a
segment, one deficient word a segment, the operand luck at `p` gives (L) at `p`. -/
def LuckCaseA : Prop :=
  ∀ {Ω : Type} [Fintype Ω] {S : ℕ} (G : ReceiptGame Ω S) (p : ℝ),
    G.Pinned → G.CaseA → G.OperandLuck p → LuckAt p G.game

/-- **The operand luck at ¾ for degree-2 operands** (`luck_quad`). For every ideal game split at the receipt whose
rounds draw uniform cube bits after fixing their share and a degree-2 operand error, nonzero when the round is
deficient, the operand luck holds at ¾. -/
def OperandLuckCube : Prop :=
  ∀ (R : Type) [CommRing R] {α V : Type} [Fintype α] [Fintype V] [DecidableEq V] {S : ℕ}
    (G : ReceiptGame (α × Finset V) S), G.CubeOperands R → G.OperandLuck (3 / 4)

/-- **(I7) at ¾ in case (a) for degree-2 operands.** For every ideal game split at the receipt whose openings are
pinned, whose rounds commit the full state of duo or fx1 units, one deficient word a unit, or li units' warp checkpoint
a segment, one deficient word a segment, and draw uniform cube bits after fixing their share and a degree-2 operand
error, (L) holds at ¾, with no luck hypothesis. -/
def LuckCaseACube : Prop :=
  ∀ (R : Type) [CommRing R] {α V : Type} [Fintype α] [Fintype V] [DecidableEq V] {S : ℕ}
    (G : ReceiptGame (α × Finset V) S), G.Pinned → G.CaseA → G.CubeOperands R → LuckAt (3 / 4) G.game

/-- **li units bind their words, a checkpoint a segment** (LI-CP-W). For every width, every li unit's window of segments
from a pinned start, each segment absorbed by any number of lanes, chains and half steps with odd multipliers, and every
coins: when every segment's warp checkpoint (each lane's XOR of its chains' states, XORed across the lanes) is the honest
words' checkpoint, every segment's deficient word is right. The first segment whose word is wrong starts from the
honest states, so its checkpoint moves. -/
def LiSegmentsBind : Prop :=
  ∀ {Ω U : Type} {n S W C H : ℕ} (D : LiUnits Ω U n S W C H) (ω : Ω), D.same ω → D.right ω

/-- **One receipt word a window does not bind li's words, even pinned.** For every width `n + 1` and every li unit of
one chain absorbing one half step a segment over two segments, whatever its keys, odd multipliers, pinned start and
honest words: when the run flips the top bit of each segment's first word (a persistent skip's two copies), the
window's last checkpoint is the honest one though both words are wrong, and the first segment's checkpoint is not. -/
def LiWindowUnbound : Prop :=
  ∀ {Ω U : Type} (n : ℕ) (D : LiUnits Ω U (n + 1) 2 1 1 1) (ω : Ω),
    (∀ u, ∀ s < 2, D.pos ω u s = 0 ∧ D.value ω u s = D.honest ω u s 0 ^^^ BitVec.twoPow (n + 1) n) →
      D.sameWindow ω ∧ ∀ u, (∀ s < 2, D.value ω u s ≠ D.honest ω u s (D.pos ω u s)) ∧
        D.checkpoint ω u (D.word ω u) 0 ≠ D.checkpoint ω u (D.honest ω u) 0

end Accounting.Guarantees.TS
