import Definitions.Accounting.TS.Play
import Definitions.Pouw.Receipt.Fx1
import Definitions.Pouw.Receipt.Li

/-!
# The ideal game split at the receipt (trusted)

LEDGER §1.3: with every absorbed word pinned in the proof, a round's ideal check
recomputes the honest committed value from committed data and passes only if the committed value equals it. A round
then carries two more events on its coins:

* `right ω`: every deficient word's value is right, the operand luck's event (`luck_quad`'s ¾ for affine operands);
* `same ω`: the committed value is the honest one.

* `ReceiptGame`: the ideal game with each round split so; `ReceiptGame.game` forgets the split.
* `ReceiptGame.Pinned`: a round passes only if `same` (the openings are pinned).
* `ReceiptGame.OperandLuck p`: (L) at `p` for the event `right`: every round's share lies in `[0, 1]`, and given the
  past a deficient round's deficient words are all right with probability at most `p`.
* `ReceiptGame.ReceiptLuck pr`: given the past, a deficient round whose deficient words are not all right keeps the
  honest committed value with probability at most `pr`.
* `DuoUnits`, `Fx1Units`: case (a)'s committed units: each commits the full state of a duo (or fx1) chain after its
  words, from a start state the proof pins, and the run's words in a unit are honest but for one position.
* `LiUnits`: case (a)'s committed units for li (LI-CP-W): a warp's window of segments from a start the proof pins, one
  checkpoint a segment (each lane's XOR of its chains' states, XORed across the warp by `redux.sync.xor`), and the
  run's words honest but for one position a segment. `LiUnits.sameWindow` is the window's last checkpoint alone.
* `DuoCPUnits`, `Fx1CPUnits`: case (a)'s committed units for duo and fx1 a checkpoint a segment (DUO-CP-W): a warp's
  window of segments from a start the proof pins, one checkpoint a segment (the XOR of every chain's full state `(y, z)`
  across the warp), and the run's words honest but for one position a segment.
  `ReceiptGame.CaseA`: every round's split is that of duo, fx1, li or duo and fx1 checkpoint units.
* `ReceiptGame.CubeOperands R`: every round draws uniform cube bits after fixing its share and a degree-2 operand
  error, nonzero when the round is deficient (`luck_quad`'s rounds).
* `ReceiptGame.WrongWords`: every round's share lies in `[0, 1]` and is positive only where a deficient word is wrong
  (the keyed rounds' share).
-/

namespace Accounting.TS

open Pouw.RelativeSecurity Pouw.Receipt

/-- **Case (a)'s committed units** on a round's coins (LEDGER §1.3, case (a)): unit `u` commits the full state of a
duo chain after its `L` words, word `j` with keys `(k u j, a u j, r u j)` and an odd multiplier, from the start state
`start ω u` that the proof pins. On the coins `ω` the run folds the honest words `honest ω u` but at the one position
`pos ω u`, where it folds `value ω u`: each deficient value sits on one word of each unit. With `L = 1`, a full-state
checkpoint after every word, every run has this form. -/
structure DuoUnits (Ω U : Type) (n L : ℕ) where
  k : U → Fin L → BitVec n
  a : U → Fin L → OddKey n
  r : U → Fin L → ℕ
  start : Ω → U → BitVec n × BitVec n
  honest : Ω → U → Fin L → BitVec n
  pos : Ω → U → Fin L
  value : Ω → U → BitVec n

namespace DuoUnits

variable {Ω U : Type} {n L : ℕ}

/-- The run's words in unit `u`: the honest words, with `value ω u` at `pos ω u`. -/
def word (D : DuoUnits Ω U n L) (ω : Ω) (u : U) : Fin L → BitVec n :=
  Function.update (D.honest ω u) (D.pos ω u) (D.value ω u)

/-- Every deficient word's value is right. -/
def right (D : DuoUnits Ω U n L) (ω : Ω) : Prop := ∀ u, D.value ω u = D.honest ω u (D.pos ω u)

/-- Every unit's committed state is the honest words' state from its start. -/
def same (D : DuoUnits Ω U n L) (ω : Ω) : Prop :=
  ∀ u, duoRun (D.k u) (fun j => (D.a u j).1) (D.r u) (D.word ω u) (D.start ω u) =
    duoRun (D.k u) (fun j => (D.a u j).1) (D.r u) (D.honest ω u) (D.start ω u)

end DuoUnits

/-- **Case (a)'s committed units for fx1**: `DuoUnits` whose chains fold with fx1, word `j` of unit `u` multiplied by
the odd IMAD multiplier `m u j` before duo's step. -/
structure Fx1Units (Ω U : Type) (n L : ℕ) extends DuoUnits Ω U n L where
  m : U → Fin L → OddKey n

/-- Every unit's committed fx1 state is the honest words' state from its start. -/
def Fx1Units.same {Ω U : Type} {n L : ℕ} (D : Fx1Units Ω U n L) (ω : Ω) : Prop :=
  ∀ u, fx1Run (D.k u) (fun j => (D.a u j).1) (D.r u) (fun j => (D.m u j).1) (D.word ω u) (D.start ω u) =
    fx1Run (D.k u) (fun j => (D.a u j).1) (D.r u) (fun j => (D.m u j).1) (D.honest ω u) (D.start ω u)

/-- A word's place in a segment of li units: its lane, its chain in the lane, its half step in the chain, and which word
of the half step's pair (`0`: `d₀`, XORed into the state; `1`: `d₁`, added). -/
abbrev LiPlace (W C H : ℕ) : Type := Fin W × Fin C × Fin H × Fin 2

/-- **Case (a)'s committed units for li, a checkpoint a segment** (LI-CP-W): unit `u`
is a warp's window. Each of its `W` lanes runs `C` chains of li's half steps (`Pouw.Receipt.liHalf`). In segment `s`,
chain `c` of lane `ℓ` absorbs `H` half steps, half step `j` with XOR key `k ω u s (ℓ, c, j)` and odd multiplier
`a ω u s (ℓ, c, j)`, from the state it ended segment `s − 1` in; segment 0 starts from `start ω u`, which the proof
pins. At the end of each segment every lane XORs its chains' states and the warp XORs its lanes' words
(`redux.sync.xor`): that word is the segment's checkpoint, and the unit commits one a segment for its `S` segments. On
the coins `ω` the run absorbs the honest words `honest ω u s`, but at the one place `pos ω u s` of each segment, where it
absorbs `value ω u s`. A persistent skip, which moves one word of every segment from its own to the window's end, has
this form; a keyed round (`y ← (y ⊕ k)·a + c`, absorbing no word) is a half step whose honest pair is `(0, c)`. -/
structure LiUnits (Ω U : Type) (n S W C H : ℕ) where
  k : Ω → U → ℕ → Fin W × Fin C × Fin H → BitVec n
  a : Ω → U → ℕ → Fin W × Fin C × Fin H → OddKey n
  start : Ω → U → Fin W → Fin C → BitVec n
  honest : Ω → U → ℕ → LiPlace W C H → BitVec n
  pos : Ω → U → ℕ → LiPlace W C H
  value : Ω → U → ℕ → BitVec n

namespace LiUnits

variable {Ω U : Type} {n S W C H : ℕ}

/-- The run's words in segment `s` of unit `u`: the honest words, with `value ω u s` at `pos ω u s`. -/
def word (D : LiUnits Ω U n S W C H) (ω : Ω) (u : U) (s : ℕ) : LiPlace W C H → BitVec n :=
  Function.update (D.honest ω u s) (D.pos ω u s) (D.value ω u s)

/-- Chain `c` of lane `ℓ`'s half steps in segment `s` of unit `u`, absorbing the words `w`. -/
def steps (D : LiUnits Ω U n S W C H) (ω : Ω) (u : U) (s : ℕ) (w : LiPlace W C H → BitVec n) (ℓ : Fin W) (c : Fin C) :
    List (LiHalfStep n) :=
  List.ofFn fun j => (D.k ω u s (ℓ, c, j), (D.a ω u s (ℓ, c, j)).1, (w (ℓ, c, j, 0), w (ℓ, c, j, 1)))

/-- Every chain's state at the start of segment `s` of unit `u` when segment `t` absorbs the words `w t`: the pinned
start, then each earlier segment's half steps. -/
def state (D : LiUnits Ω U n S W C H) (ω : Ω) (u : U) (w : ℕ → LiPlace W C H → BitVec n) :
    ℕ → Fin W → Fin C → BitVec n
  | 0 => D.start ω u
  | s + 1 => fun ℓ c => liChain (D.steps ω u s (w s) ℓ c) (D.state ω u w s ℓ c)

/-- Segment `s`'s checkpoint when segment `t` absorbs the words `w t`: each lane's word, the XOR of its chains' states at
the segment's end (`liCheckpoint`), XORed across the warp's lanes (`redux.sync.xor`). -/
def checkpoint (D : LiUnits Ω U n S W C H) (ω : Ω) (u : U) (w : ℕ → LiPlace W C H → BitVec n) (s : ℕ) : BitVec n :=
  (List.ofFn fun ℓ => liCheckpoint (List.ofFn fun c => (D.steps ω u s (w s) ℓ c, D.state ω u w s ℓ c))).foldl
    (· ^^^ ·) 0

/-- Every segment's deficient word is right. -/
def right (D : LiUnits Ω U n S W C H) (ω : Ω) : Prop := ∀ u, ∀ s < S, D.value ω u s = D.honest ω u s (D.pos ω u s)

/-- Every segment's committed checkpoint is the honest words' checkpoint. -/
def same (D : LiUnits Ω U n S W C H) (ω : Ω) : Prop :=
  ∀ u, ∀ s < S, D.checkpoint ω u (D.word ω u) s = D.checkpoint ω u (D.honest ω u) s

/-- The window's one receipt word alone (li without LI-CP's checkpoint a segment): every unit's last checkpoint is the
honest words' one. -/
def sameWindow (D : LiUnits Ω U n S W C H) (ω : Ω) : Prop :=
  ∀ u, D.checkpoint ω u (D.word ω u) (S - 1) = D.checkpoint ω u (D.honest ω u) (S - 1)

end LiUnits

/-- The XOR of `m` words. -/
def xorAll {n m : ℕ} (f : Fin m → BitVec n) : BitVec n := (List.ofFn f).foldl (· ^^^ ·) 0

/-- A word's place in a segment of duo checkpoint units: its lane, its chain in the lane and its position in the
chain's run. -/
abbrev CPPlace (W C H : ℕ) : Type := Fin W × Fin C × Fin H

/-- **Case (a)'s committed units for duo, a full-state checkpoint a segment** (DUO-CP-W):
unit `u` is a warp's window. Each of its `W` lanes runs `C` duo chains. In segment
`s`, chain `c` of lane `ℓ` absorbs `H` words, word `j` with XOR key `k ω u s (ℓ, c, j)`, odd multiplier
`a ω u s (ℓ, c, j)` and rotation `r ω u s (ℓ, c, j)`, from the state it ended segment `s − 1` in; segment 0 starts from
`start ω u`, which the proof pins. At the end of each segment every lane XORs its chains' states `(y, z)` and the warp
XORs its lanes' pairs (two `redux.sync.xor`): that pair is the segment's checkpoint, and the unit commits one a segment
for its `S` segments. On the coins `ω` the run absorbs the honest words `honest ω u s`, but at the one place
`pos ω u s` of each segment, where it absorbs `value ω u s`. The served GEMM's layout (`W = 32`, `C = 8`, `H = 4`, one
segment a k32 step) has this form for a skip that lands on one word of a segment; `W = 1` is a lane's checkpoint
without the warp's reduction. The served receipt, `y ⊕ rotl(z, 16)` a thread a window, is not such a unit. -/
structure DuoCPUnits (Ω U : Type) (n S W C H : ℕ) where
  k : Ω → U → ℕ → CPPlace W C H → BitVec n
  a : Ω → U → ℕ → CPPlace W C H → OddKey n
  r : Ω → U → ℕ → CPPlace W C H → ℕ
  start : Ω → U → Fin W → Fin C → BitVec n × BitVec n
  honest : Ω → U → ℕ → CPPlace W C H → BitVec n
  pos : Ω → U → ℕ → CPPlace W C H
  value : Ω → U → ℕ → BitVec n

namespace DuoCPUnits

variable {Ω U : Type} {n S W C H : ℕ}

/-- The run's words in segment `s` of unit `u`: the honest words, with `value ω u s` at `pos ω u s`. -/
def word (D : DuoCPUnits Ω U n S W C H) (ω : Ω) (u : U) (s : ℕ) : CPPlace W C H → BitVec n :=
  Function.update (D.honest ω u s) (D.pos ω u s) (D.value ω u s)

/-- Every chain's state at the start of segment `s` of unit `u` when segment `t` absorbs the words `w t`: the pinned
start, then each earlier segment's duo run. -/
def state (D : DuoCPUnits Ω U n S W C H) (ω : Ω) (u : U) (w : ℕ → CPPlace W C H → BitVec n) :
    ℕ → Fin W → Fin C → BitVec n × BitVec n
  | 0 => D.start ω u
  | s + 1 => fun ℓ c =>
      duoRun (fun j => D.k ω u s (ℓ, c, j)) (fun j => (D.a ω u s (ℓ, c, j)).1) (fun j => D.r ω u s (ℓ, c, j))
        (fun j => w s (ℓ, c, j)) (D.state ω u w s ℓ c)

/-- Segment `s`'s checkpoint when segment `t` absorbs the words `w t`: the XOR of every chain's `y` and of every chain's
`z` at the segment's end, over the lane's chains and then the warp's lanes. -/
def checkpoint (D : DuoCPUnits Ω U n S W C H) (ω : Ω) (u : U) (w : ℕ → CPPlace W C H → BitVec n) (s : ℕ) :
    BitVec n × BitVec n :=
  (xorAll fun ℓ => xorAll fun c => (D.state ω u w (s + 1) ℓ c).1,
    xorAll fun ℓ => xorAll fun c => (D.state ω u w (s + 1) ℓ c).2)

/-- Every segment's deficient word is right. -/
def right (D : DuoCPUnits Ω U n S W C H) (ω : Ω) : Prop := ∀ u, ∀ s < S, D.value ω u s = D.honest ω u s (D.pos ω u s)

/-- Every segment's committed checkpoint is the honest words' checkpoint. -/
def same (D : DuoCPUnits Ω U n S W C H) (ω : Ω) : Prop :=
  ∀ u, ∀ s < S, D.checkpoint ω u (D.word ω u) s = D.checkpoint ω u (D.honest ω u) s

end DuoCPUnits

/-- **Case (a)'s committed units for fx1, a full-state checkpoint a segment**: `DuoCPUnits` whose chains fold with fx1,
the word at each place multiplied by the odd IMAD multiplier `m ω u s p` before duo's step (`fx1Word`). -/
structure Fx1CPUnits (Ω U : Type) (n S W C H : ℕ) extends DuoCPUnits Ω U n S W C H where
  m : Ω → U → ℕ → CPPlace W C H → OddKey n

namespace Fx1CPUnits

variable {Ω U : Type} {n S W C H : ℕ}

/-- The words duo absorbs when fx1 folds the words `w`: each times its IMAD multiplier. -/
def imad (D : Fx1CPUnits Ω U n S W C H) (ω : Ω) (u : U) (w : ℕ → CPPlace W C H → BitVec n) :
    ℕ → CPPlace W C H → BitVec n :=
  fun s p => w s p * (D.m ω u s p).1

/-- Every segment's committed fx1 checkpoint is the honest words' checkpoint. -/
def same (D : Fx1CPUnits Ω U n S W C H) (ω : Ω) : Prop :=
  ∀ u, ∀ s < S, D.checkpoint ω u (D.imad ω u (D.word ω u)) s = D.checkpoint ω u (D.imad ω u (D.honest ω u)) s

end Fx1CPUnits

/-- **The ideal game split at the receipt** over `n` rounds of coins on `Ω`: per round, its share `d`, its check
`pass`, and the events `right` (every deficient word's value is right) and `same` (the committed value is the honest
one), then the rest of the play, which may depend on the round's coins. -/
inductive ReceiptGame (Ω : Type) : ℕ → Type
  | done : ReceiptGame Ω 0
  | round {n : ℕ} (d : Ω → ℝ) (pass right same : Ω → Prop) (next : Ω → ReceiptGame Ω n) : ReceiptGame Ω (n + 1)

namespace ReceiptGame

variable {Ω : Type}

/-- The ideal game: each round's share and check. -/
def game : {n : ℕ} → ReceiptGame Ω n → RoundGame Ω n
  | _, .done => .done
  | _, .round d pass _ _ next => .round d pass fun ω => (next ω).game

/-- **The openings are pinned**: at every round, whatever the past, the round passes only if its committed value is
the honest one. -/
def Pinned : {n : ℕ} → ReceiptGame Ω n → Prop
  | _, .done => True
  | _, .round _ pass _ same next => (∀ ω, pass ω → same ω) ∧ ∀ ω, (next ω).Pinned

/-- **The operand luck at `p`**: every round's share lies in `[0, 1]`, and at every round, whatever the past, a
deficient round's deficient words are all right with probability at most `p`:
`Pr[right ∧ d_w > 0] ≤ p·Pr[d_w > 0]` over the round's coins. -/
def OperandLuck [Fintype Ω] (p : ℝ) : {n : ℕ} → ReceiptGame Ω n → Prop
  | _, .done => True
  | _, .round d _ right _ next =>
      (∀ ω, 0 ≤ d ω ∧ d ω ≤ 1) ∧ prob (fun ω => right ω ∧ 0 < d ω) ≤ p * prob (fun ω => 0 < d ω) ∧
        ∀ ω, OperandLuck p (next ω)

/-- **The receipt luck at `pr`**: at every round, whatever the past, a deficient round whose deficient words are not
all right keeps the honest committed value with probability at most `pr`:
`Pr[same ∧ ¬right ∧ d_w > 0] ≤ pr·Pr[¬right ∧ d_w > 0]` over the round's coins. -/
def ReceiptLuck [Fintype Ω] (pr : ℝ) : {n : ℕ} → ReceiptGame Ω n → Prop
  | _, .done => True
  | _, .round d _ right same next =>
      prob (fun ω => same ω ∧ ¬right ω ∧ 0 < d ω) ≤ pr * prob (fun ω => ¬right ω ∧ 0 < d ω) ∧
        ∀ ω, ReceiptLuck pr (next ω)

/-- **Case (a)** (LEDGER §1.3): at every round, whatever the past, the round's `right` and `same` are those of
committed duo or fx1 units (`DuoUnits`, `Fx1Units`: each unit's full final state from a pinned start, the run's words
honest but for one position a unit), of committed li units (`LiUnits`: each segment's warp checkpoint from a window
start the proof pins, the run's words honest but for one position a segment) or of committed duo or fx1 checkpoint
units (`DuoCPUnits`, `Fx1CPUnits`: each segment's warp checkpoint of the chains' full states from a window start the
proof pins, the run's words honest but for one position a segment). -/
def CaseA : {n : ℕ} → ReceiptGame Ω n → Prop
  | _, .done => True
  | _, .round _ _ right same next =>
      ((∃ (U : Type) (w L : ℕ) (D : DuoUnits Ω U w L), ∀ ω, (right ω ↔ D.right ω) ∧ (same ω ↔ D.same ω)) ∨
        (∃ (U : Type) (w L : ℕ) (D : Fx1Units Ω U w L), ∀ ω, (right ω ↔ D.right ω) ∧ (same ω ↔ D.same ω)) ∨
        (∃ (U : Type) (w S W C H : ℕ) (D : LiUnits Ω U w S W C H),
          ∀ ω, (right ω ↔ D.right ω) ∧ (same ω ↔ D.same ω)) ∨
        (∃ (U : Type) (w S W C H : ℕ) (D : DuoCPUnits Ω U w S W C H),
          ∀ ω, (right ω ↔ D.right ω) ∧ (same ω ↔ D.same ω)) ∨
        ∃ (U : Type) (w S W C H : ℕ) (D : Fx1CPUnits Ω U w S W C H),
          ∀ ω, (right ω ↔ D.right ω) ∧ (same ω ↔ D.same ω)) ∧
        ∀ ω, (next ω).CaseA

/-- **Degree-2 operands on cube bits** (`luck_quad`'s rounds, full checking): a round's coins are other coins `a`
and uniform cube bits `S`; at every round, whatever the past, its share lies in `[0, 1]` and is fixed before the bits
(`δ a`), the prover's operand error `e a` is a function of the bits of degree at most 2 (`CubeQuad`), a deficient
share means a nonzero error, and the deficient words are all right only where the error vanishes at the drawn bits. -/
def CubeOperands (R : Type) [CommRing R] {α V : Type} [Fintype V] [DecidableEq V] :
    {n : ℕ} → ReceiptGame (α × Finset V) n → Prop
  | _, .done => True
  | _, .round d _ right _ next =>
      (∀ ω, 0 ≤ d ω ∧ d ω ≤ 1) ∧
        (∃ (δ : α → ℝ) (e : α → (V → R) → R), (∀ a, CubeQuad (e a)) ∧ (∀ ω, d ω = δ ω.1) ∧
          (∀ a, 0 < δ a → ∃ S : Finset V, e a (bitInd S) ≠ 0) ∧ ∀ ω, right ω → e ω.1 (bitInd ω.2) = 0) ∧
        ∀ ω, (next ω).CubeOperands R

/-- **A share of wrong words** (the keyed rounds' share): at every round, whatever the past, its share lies in `[0, 1]`
and is positive only where some deficient word is wrong. For keyed rounds, a deficient word is a round whose state the
run did not compute, and FOLD-OPT prices a right one at the round's work, so the share counts the wrong ones. -/
def WrongWords : {n : ℕ} → ReceiptGame Ω n → Prop
  | _, .done => True
  | _, .round d _ right _ next => (∀ ω, 0 ≤ d ω ∧ d ω ≤ 1) ∧ (∀ ω, 0 < d ω → ¬right ω) ∧ ∀ ω, (next ω).WrongWords

end ReceiptGame

end Accounting.TS
