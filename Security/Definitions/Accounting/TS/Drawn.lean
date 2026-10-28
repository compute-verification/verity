import Definitions.Accounting.TS.ProofLayer

/-!
# A sampled window's proofs: what (I8) reads of its drawn units (trusted)

`ProofLayer`'s counterpart for 𝒫_samp. A round's coins are `Paths C m × B`: on `Paths C m` the `m` coins released before
the draws, in order (the round's σ, or a window's call sites' coins, each after the values it follows are committed),
and on `B` the live draw coin, which names `s` units (`DrawnLayer.draw`), with the proofs' coins. Each drawn unit is
proved alone, with its own noise's generation; a unit is one output coordinate, or a K-chunk committing its running sum.

Values are per committed position, not per unit: a unit reads its slots' positions (`DrawnLayer.pos`: its operand rows,
its seed's inputs, a running sum, its outputs), and a position one unit writes is the position another reads, so one
committed value serves both. Each position is bound before the coin that follows its commitment (`DrawnLayer.cut`):
round `w`'s coin `k < m` for an operand committed before it, the draw (`k = m`) for a unit's output. Its committed value
(`DrawnLayer.committed`) is the value its drawn readers' accepted proofs open on the most continuations from that coin,
each opening it to that value alone, so it is a function of the coins before the cut. Two drawn readers that open one
position differently on one path move it (`DrawnLayer.Moved`): equivocation is a binding failure, counted in `δ_bind`.

Proofs exist for drawn units only: round `w`'s proof `i` is drawn unit `drawnAt x w i`'s (`DrawnLayer.accepted`,
`DrawnLayer.opened`), so an accepted proof is a drawn unit's.

* `coinAt x w`: round `w`'s coins on `x`.
* `spliceAt w k x z`: `x`'s coins before round `w`'s coin `k` (`k = m`: before its draw), then `z`'s.
* `Alone`, `tally`, `plural`: a path that opens one value only; how many continuations of `x` open `v` alone; the value
  the most do.
* `DrawnLayer`: the draw; each unit's slots' positions and noise coin, each position's cut and writer; per path, round
  and draw, whether the drawn unit's proof was accepted and the values it opened its slots to; each unit's relation and
  rightness, on its slots' values and the public coins.
* `Opens`, `committed`, `view`: a path's accepted openings of a position; its committed value; a unit's committed
  slots.
* `Moved`, `MovedIn`, `BindingAt`, `Forked`: binding's failure, per position and per round, its bound, and its fork
  form; `Unsound`,
  `JointSound`: proof soundness, jointly over a round's draws; `Pins`: each unit's relation pins its rightness.
* `WriterReads`: a position's writer has it among its slots, checked for each layer.
* `Mode`: a unit's mode in the run: paid (computed), folded (skipped, its committed words folded) or whole (skipped
  with its fold).
* `Late`, `Good`, `coneWt`: a unit writes a position another reads, committed after the coin the reader's noise reads; a
  unit whose late cone (itself included) the run paid for, under the run's modes; a unit's late cone's weight, which the
  share charges a skipped unit.
* `Of`: the play's tie to its drawn units' proofs; `Covers`: the ideal game's share counts every unit that isn't good,
  a property of the share's definition, proved for each game.

`Good` reads the run's modes, not rightness: a unit's check can pin less than its words (`PouwWarpCheck_v2` compares
its receipt's checkpoints), so a skipped unit whose receipt is lucky is right and its words still wrong.

Two conditions are protocol structure, outside this vocabulary, and `UnitFixed` (E1(c) per unit) is claimed only where
they hold: a layer's `cut` is no later than the protocol's real commitment of each position (a later cut binds less),
and the modes are the run's (a unit is paid only if the run computed it, from the values it committed, under its
noise).
-/

namespace Accounting.TS

open Pouw.RelativeSecurity

/-- Round `w`'s coins on the path `x`. -/
def coinAt {Ω : Type} : {n : ℕ} → Paths Ω n → Fin n → Ω
  | 0, _, w => w.elim0
  | _ + 1, x, ⟨0, _⟩ => x.1
  | _ + 1, x, ⟨w + 1, h⟩ => coinAt x.2 ⟨w, Nat.lt_of_succ_lt_succ h⟩

/-- `x`'s coins before round `w`'s coin `k` (its rounds before `w`, and round `w`'s coins before `k`; `k = m`: all of
round `w`'s coins before its draw), then `z`'s: the run that shares what `x` committed before that coin and continues on
fresh coins. -/
def spliceAt {C B : Type} {m : ℕ} : {n : ℕ} → ℕ → ℕ → Paths (Paths C m × B) n → Paths (Paths C m × B) n →
    Paths (Paths C m × B) n
  | 0, _, _, _, _ => ()
  | _ + 1, 0, k, x, z => ((splice k x.1.1 z.1.1, z.1.2), z.2)
  | _ + 1, w + 1, k, x, z => (x.1, spliceAt w k x.2 z.2)

section Plurality

variable {X V : Type}

/-- `y` opens `v` and no other value. -/
def Alone (opens : X → V → Prop) (y : X) (v : V) : Prop := opens y v ∧ ∀ v', opens y v' → v' = v

open Classical in
/-- How many continuations `sp x z` of `x` open `v` alone. -/
noncomputable def tally [Fintype X] (sp : X → X → X) (opens : X → V → Prop) (x : X) (v : V) : ℕ :=
  (Finset.univ.filter fun z => Alone opens (sp x z) v).card

/-- **The plurality value**: a value that the most continuations of `x` open alone. -/
noncomputable def plural [Fintype X] [Nonempty V] (sp : X → X → X) (opens : X → V → Prop) (x : X) : V :=
  Classical.epsilon fun v => ∀ u, tally sp opens x u ≤ tally sp opens x v

end Plurality

/-- A unit's mode in a cheater's run: paid (the run computed it: its outputs are its own, from the operands it
committed, under its noise); folded (skipped and its words guessed, its committed words folded into the receipts); or
whole (skipped with its fold). -/
inductive Mode
  | paid
  | folded
  | whole
  deriving DecidableEq

/-- **A sampled window's proofs** over `S` rounds of coins on `Paths C m × B`, `s` draws a round of units `J`, each
unit reading its slots `Q`'s positions `Pos`, each position's value in `V`:
* `draw`: the units the draw coin names;
* `pos w j k`: the position unit `j` of round `w` reads at slot `k`;
* `coin w j`: the coin of round `w` before its draw that unit `j`'s noise reads;
* `cut ℓ = (w, k)`: `ℓ` is committed before round `w`'s coin `k` (`k = m`: before its draw), after every coin before it,
  and no later than the protocol commits it;
* `src ℓ`: the unit whose output `ℓ` is, none for the circuit's inputs (its writer reads it, `WriterReads`);
* `accepted x w i`, `opened x w i`: on `x`, whether round `w`'s proof of its `i`-th drawn unit was accepted, and the
  values its opening gives that unit's slots;
* `rel x w j`: unit `j`'s relation in round `w` on its slots' values, the public Program its proof passes, at the
  public coins of `x` its seed reads;
* `right x w j`: its rightness: its outputs are its own, from its operands, under the noise its seed derives. -/
structure DrawnLayer (C B : Type) (S m s : ℕ) (J Q Pos V : Type) where
  draw : B → Fin s → J
  pos : Fin S → J → Q → Pos
  coin : Fin S → J → Fin m
  cut : Pos → Fin S × Fin (m + 1)
  src : Pos → Option (Fin S × J)
  accepted : Paths (Paths C m × B) S → Fin S → Fin s → Prop
  opened : Paths (Paths C m × B) S → Fin S → Fin s → Q → V
  rel : Paths (Paths C m × B) S → Fin S → J → (Q → V) → Prop
  right : Paths (Paths C m × B) S → Fin S → J → (Q → V) → Prop

namespace DrawnLayer

variable {C B : Type} {S m s : ℕ} {J Q Pos V : Type} (pf : DrawnLayer C B S m s J Q Pos V)

/-- The unit round `w`'s draw `i` names on `x`. -/
def drawnAt (x : Paths (Paths C m × B) S) (w : Fin S) (i : Fin s) : J := pf.draw (coinAt x w).2 i

/-- **`x` opens position `ℓ` to `v`**: in some round, an accepted proof of a drawn unit opens one of its slots, at `ℓ`,
to `v`. -/
def Opens (x : Paths (Paths C m × B) S) (ℓ : Pos) (v : V) : Prop :=
  ∃ w i k, pf.pos w (pf.drawnAt x w i) k = ℓ ∧ pf.accepted x w i ∧ pf.opened x w i k = v

/-- The continuations from position `ℓ`'s cut. -/
def cont (ℓ : Pos) : Paths (Paths C m × B) S → Paths (Paths C m × B) S → Paths (Paths C m × B) S :=
  spliceAt (pf.cut ℓ).1 (pf.cut ℓ).2

/-- **Position `ℓ`'s value as committed before its cut**: the value its drawn readers' accepted proofs open alone on the
most continuations of `x` from `ℓ`'s cut. -/
noncomputable def committed [Fintype C] [Fintype B] [Nonempty V] (x : Paths (Paths C m × B) S) (ℓ : Pos) : V :=
  plural (pf.cont ℓ) (fun y v => pf.Opens y ℓ v) x

/-- Unit `j` of round `w`'s slots at their committed values. -/
noncomputable def view [Fintype C] [Fintype B] [Nonempty V] (x : Paths (Paths C m × B) S) (w : Fin S) (j : J) :
    Q → V :=
  fun k => pf.committed x (pf.pos w j k)

/-- **Binding fails at position `ℓ`**: an accepted proof of a drawn unit opens `ℓ` to a value other than its committed
one. Two that open it differently on `x` move it. -/
def Moved [Fintype C] [Fintype B] [Nonempty V] (x : Paths (Paths C m × B) S) (ℓ : Pos) : Prop :=
  ∃ v, pf.Opens x ℓ v ∧ v ≠ pf.committed x ℓ

/-- **Binding fails in round `w`**: an accepted proof of one of its drawn units opens the unit's slots to values other
than their committed ones. -/
def MovedIn [Fintype C] [Fintype B] [Nonempty V] (x : Paths (Paths C m × B) S) (w : Fin S) : Prop :=
  ∃ i, pf.accepted x w i ∧ pf.opened x w i ≠ pf.view x w (pf.drawnAt x w i)

/-- **Binding at `δ_bind`**: in every round, binding fails (`MovedIn`) with probability at most `δ_bind` over the
window's coins. -/
def BindingAt [Fintype C] [Fintype B] [Nonempty V] (δbind : ℝ) : Prop := ∀ w, prob (fun x => pf.MovedIn x w) ≤ δbind

/-- **Two runs off one commitment open a position differently** (the fork form): on `x` and on its continuation from
`ℓ`'s cut on fresh coins `z`, which share every value committed before that cut, accepted proofs of drawn units open
`ℓ` to two different values (on one run or across the two). Where `ℓ` is an hm96 row of a frame-v3-sha512 tree whose
root is committed before the cut, two openings are two inputs hashed to one SHA-512 digest along their paths. The
openings are witnesses of zero-knowledge proofs, so a finder that runs the window twice from the cut takes them with
C-Flock's knowledge extractor, at `ε_ks` a proof, before `cr/sha-512` bounds the collision. -/
def Forked (ℓ : Pos) (x z : Paths (Paths C m × B) S) : Prop :=
  ∃ v v', pf.Opens x ℓ v ∧ pf.Opens (pf.cont ℓ x z) ℓ v' ∧ v ≠ v'

/-- Round `w`'s proof of its `i`-th drawn unit is accepted, and its opening fails the unit's relation. -/
def Unsound (x : Paths (Paths C m × B) S) (w : Fin S) (i : Fin s) : Prop :=
  pf.accepted x w i ∧ ¬ pf.rel x w (pf.drawnAt x w i) (pf.opened x w i)

/-- **Proof soundness at `δ_ks`, jointly over a round's draws**: in each round, some drawn unit's proof is accepted with
an opening that fails its relation with probability at most `δ_ks` over the window's coins. -/
def JointSound [Fintype C] [Fintype B] (δks : ℝ) : Prop := ∀ w, prob (fun x => ∃ i, pf.Unsound x w i) ≤ δks

/-- **Each unit's relation pins its rightness**: values of its slots that satisfy unit `j`'s relation are right. A
check with no absorbed word as witness (`PouwCallCheck_v1`, `PouwWarpCheck_v1`, `PouwSegCheck_v3`) recomputes
every word it absorbs from the unit's operands, its seed and its noise. -/
def Pins : Prop := ∀ x w j v, pf.rel x w j v → pf.right x w j v

/-- **A position's writer reads it**: the unit whose output a position is has it among its slots, so the writer's
rightness speaks of it. Protocol structure, checked for each layer. -/
def WriterReads : Prop := ∀ ℓ u, pf.src ℓ = some u → ∃ k, pf.pos u.1 u.2 k = ℓ

/-- **Unit `u` writes, after `v`'s coin, a position that unit `v` reads**: the position's cut is after the coin `v`'s
noise reads, so its value can be chosen after that coin (such as a K-chunk's running sum). A position committed before
`v`'s coin is fixed when `v`'s noise is drawn, whether its writer is right or not, so it is no edge. -/
def Late (u v : Fin S × J) : Prop :=
  ∃ k, pf.src (pf.pos v.1 v.2 k) = some u ∧
    (v.1 < (pf.cut (pf.pos v.1 v.2 k)).1 ∨
      v.1 = (pf.cut (pf.pos v.1 v.2 k)).1 ∧ (pf.coin v.1 v.2).castSucc < (pf.cut (pf.pos v.1 v.2 k)).2)

/-- **Unit `j` of round `w` is good on `x`** under the run's modes `mode`: the run paid for every unit of its late cone
(itself, and every unit that writes it a value after its coin, transitively). A skipped unit isn't good even when its
check passes by luck. -/
def Good (mode : Paths (Paths C m × B) S → Fin S × J → Mode) (x : Paths (Paths C m × B) S) (w : Fin S) (j : J) :
    Prop :=
  ∀ u : Fin S × J, Relation.ReflTransGen pf.Late u (w, j) → mode x u = .paid

open Classical in
/-- **Unit `u`'s late cone's weight**: the weights `wt` of every unit `u` reaches along `Late`, itself included. A share
that charges each skipped unit this covers every unit that isn't good; for whole-coordinate units there are no late
edges and it is `wt u`. -/
noncomputable def coneWt [Fintype J] (wt : Fin S → J → ℝ) (u : Fin S × J) : ℝ :=
  ∑ v : Fin S × J, if Relation.ReflTransGen pf.Late u v then wt v.1 v.2 else 0

/-- **The play is its drawn units' proofs'** (𝒫_samp): the window is accepted only if every drawn unit's proof is, and
the ideal game's check is the drawn units' committed values, so drawn units right at them are accepted by the ideal
game. -/
def Of [Fintype C] [Fintype B] [Nonempty V] {K T L U : Type} (P : Play (Paths C m × B) S K T L U) : Prop :=
  (∀ x, P.accepts x → ∀ w i, pf.accepted x w i) ∧
    ∀ x, (∀ w i, pf.right x w (pf.drawnAt x w i) (pf.view x w (pf.drawnAt x w i))) → P.game.accepts x

open Classical in
/-- **The share covers every unit that isn't good**: on an accepted path, the weight `wt w j` (unit `j`'s words in round
`w`) of the units that aren't good under the run's modes is at most the ideal game's deficiency times `N_w`. A property
of the share's definition, proved for each game, not assumed: a share that charges each unit the run didn't pay for its
`coneWt` covers (`Accounting.SecurityProofs.TS.DrawnLayer.covers_of_cone`). -/
def Covers [Fintype J] {K T L U : Type} (P : Play (Paths C m × B) S K T L U)
    (mode : Paths (Paths C m × B) S → Fin S × J → Mode) (wt : Fin S → J → ℝ) (Nw : ℝ) : Prop :=
  ∀ x, P.accepts x → (∑ w, ∑ j, if pf.Good mode x w j then 0 else wt w j) ≤ P.game.deficiency x * Nw

end DrawnLayer

end Accounting.TS
