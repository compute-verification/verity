module

public import Definitions.Core.Game
public import Definitions.WeightSecurity

/-!
# Weight security at an audited deployment: the windows it serves, the halt, and the channel they leave through

A deployment serves the outsider one query per window. In window `n` it serves the outsider's `n`-th query with an
advice `a ∈ Adv` (the run-time choices a registered compiler reads, which the deployment picks), releases the outputs'
values on the window's egress slots `O` at a timing bucket `Tick`, and then the window's audit runs on live coins: it
accepts or rejects, and its sample holds one of the window's incorrect proof blocks or misses them all. A release leaves
before its window's verdict (detection, not prevention), and the deployment halts once a window is rejected, after a lag
of `L` further windows. The network certifier is the layer that makes the release the whole of what leaves: the
outputs the trace commits, at its bucket, and nothing else.

* `Served`: what a run records of a window;
* `Served.count L l`: how many of the run `l`'s windows release before the halt;
* `Served.reply L l n` and `Served.view`: what the outsider receives in its `n`-th round, and its view of the run;
* `Served.lead l`: the incorrect proof blocks of the windows before the first whose sample holds one of its own;
* `Served.Undetected M l`: the audit leaves `M` incorrect proof blocks undetected: an accepted window's sample held one
  of its incorrect proof blocks, or the windows before the first sampled one hold at least `M`;
* `Served.UndetectedLe M δ run`: on every run, the audit leaves `M` incorrect proof blocks undetected with probability
  at most `δ`;
* `Served.Local C E Ou L θ r l`: each window that releases changes at most `E` of its slots per incorrect proof block,
  against the served function's honest answer to the outsider's query at the window's advice. It is what the layers
  below the composition give: the network certifier ties the release to the trace's outputs, and the trace differs
  from the honest run only in its incorrect proof blocks' egress cones (`Population.Dataflow.card_changed_le`);
* `Served.Isolated C F Ou L θ r l`: through each window accepted before the first rejection, the releases change at
  most `F` outputs per incorrect proof block of the windows so far, so a block's changes may fall in later windows,
  through state they read;
* `servedChannel T K`: the covert channel of such a deployment over `T` windows: each window's advice and timing bucket,
  at most `K` of the `T · |O|` released values overwritten, and how many windows release before the halt;
* `servedBits T K`: that channel's covert capacity in bits.
-/

@[expose] public section

namespace WeightSecurity

/-- **What a run records of a served window**: whether its audit accepted it, whether its sample holds one of its
incorrect proof blocks, how many of its proof blocks are incorrect, and what the deployment released in it before the
audit ran: the values on the egress slots `O`, the timing bucket, and the advice it served the query with. -/
structure Served (O V Tick Adv : Type) where
  accepted : Prop
  sampledIncorrect : Prop
  incorrect : ℕ
  egress : O → V
  tick : Tick
  adv : Adv

namespace Served

variable {O V Tick Adv R Q Θ Θ' : Type}

open Classical in
/-- **How many windows release before the halt**: each window through the first rejected one, and `L` more after it
(the lag before the deployment halts), as far as the run goes. -/
noncomputable def count (L : ℕ) : List (Served O V Tick Adv) → ℕ
  | [] => 0
  | w :: l => if w.accepted then count L l + 1 else min l.length L + 1

open Classical in
/-- **The windows accepted before the first rejection**: how many there are. -/
noncomputable def accLen : List (Served O V Tick Adv) → ℕ
  | [] => 0
  | w :: l => if w.accepted then accLen l + 1 else 0

/-- **What the outsider receives in round `n`**: window `n`'s release, its values and its timing bucket, when the
window releases, and nothing after the halt. -/
noncomputable def reply (L : ℕ) (l : List (Served O V Tick Adv)) (n : ℕ) : Option ((O → V) × Tick) :=
  if n < count L l then l[n]?.map fun w => (w.egress, w.tick) else none

/-- **The outsider's view of the run `l`** after `n` rounds: its queries and what it received. -/
noncomputable def view (Ou : Outsider R (Q × Adv) (Option ((O → V) × Tick)) Θ') (L : ℕ) (r : R)
    (l : List (Served O V Tick Adv)) (n : ℕ) : List ((Q × Adv) × Option ((O → V) × Tick)) :=
  Ou.view r (fun h _ => reply L l h.length) n

open Classical in
/-- **The incorrect proof blocks before the first sampled one**: those of each window, up to the first window whose
sample holds one of its own incorrect proof blocks. -/
noncomputable def lead : List (Served O V Tick Adv) → ℕ
  | [] => 0
  | w :: l => if w.sampledIncorrect then 0 else w.incorrect + lead l

/-- **The audit leaves `M` incorrect proof blocks undetected**: an accepted window's sample held one of its incorrect
proof blocks, or the windows before the first sampled one hold at least `M`. Outside it, every window accepted before
the first rejection is one whose sample missed its incorrect proof blocks, and they hold fewer than `M`. -/
def Undetected (M : ℕ) (l : List (Served O V Tick Adv)) : Prop :=
  (∃ w ∈ l, w.accepted ∧ w.sampledIncorrect) ∨ M ≤ lead l

open scoped ENNReal in
/-- **The audit leaves `M` incorrect proof blocks undetected with probability at most `δ`** on every run `run θ r`. -/
def UndetectedLe (M : ℕ) (δ : ℝ≥0∞) (run : Θ → R → Verity.Game (List (Served O V Tick Adv))) : Prop :=
  ∀ θ r, Verity.Game.value (Undetected M) (run θ r) ≤ δ

open Classical in
/-- **The slots window `n`'s release changes**: those where it differs from the served function `C`'s honest answer at
the weights `θ` to the outsider's `n`-th query, served with the window's advice. -/
noncomputable def changed [Fintype O] (C : Θ → Q × Adv → O → V) (Ou : Outsider R (Q × Adv) (Option ((O → V) × Tick)) Θ')
    (L : ℕ) (θ : Θ) (r : R) (l : List (Served O V Tick Adv)) (n : ℕ) : Finset O :=
  match l[n]? with
  | some w => Finset.univ.filter fun o => w.egress o ≠ C θ ((Ou.query r (view Ou L r l n)).1, w.adv) o
  | none => ∅

/-- **The releases are local**: each window that releases changes at most `E` of its slots per incorrect proof block of
its own. The network certifier gives that the release is the trace's outputs, and the dataflow that the trace differs
from the honest run at the window's query and advice only within its incorrect proof blocks' egress cones, each at most
`E` slots. -/
def Local [Fintype O] (C : Θ → Q × Adv → O → V) (E : ℕ) (Ou : Outsider R (Q × Adv) (Option ((O → V) × Tick)) Θ')
    (L : ℕ) (θ : Θ) (r : R) (l : List (Served O V Tick Adv)) : Prop :=
  ∀ n < count L l, ∀ w, l[n]? = some w → (changed C Ou L θ r l n).card ≤ E * w.incorrect

/-- **Window `n`'s number of incorrect proof blocks**, `0` past the run. -/
def incorrectAt (l : List (Served O V Tick Adv)) (n : ℕ) : ℕ := (l[n]?.map Served.incorrect).getD 0

/-- **The accepted releases are isolated**: through each window accepted before the first rejection, the outputs the
releases changed number at most `F` per incorrect proof block of the windows so far. A block changes outputs of its own
window and, through state a later window reads (a carried prefix's key and value rows), of later ones, never of earlier
ones; so a change counts against the incorrect proof blocks up to its window, and each block's changes, wherever they
fall, number at most `F`. The windows from the first rejection to the halt are charged whole. With `O` a window's
charge units (its requests, or each request with the requests that read its state) and `V` a unit's whole output, a
block changes at most `F` units. `Local` at `E = F` gives it. -/
def Isolated [Fintype O] (C : Θ → Q × Adv → O → V) (F : ℕ) (Ou : Outsider R (Q × Adv) (Option ((O → V) × Tick)) Θ')
    (L : ℕ) (θ : Θ) (r : R) (l : List (Served O V Tick Adv)) : Prop :=
  ∀ k ≤ accLen l, ∑ n ∈ Finset.range k, (changed C Ou L θ r l n).card ≤ F * ∑ n ∈ Finset.range k, incorrectAt l n

end Served

/-- **The served channel** over `T` windows. A covert string is each window's advice and timing bucket, an overwrite of
at most `K` of the `T · |O|` released values, and how many windows release before the halt. With it the datacenter
serves the outsider's `n`-th query with the `n`-th advice and, for the windows before the halt, tells it the honest
answer with the overwrite applied, at the `n`-th bucket; after the halt it tells it nothing. -/
noncomputable def servedChannel (R Q Adv O V Tick : Type) [Fintype Adv] [Fintype O] [DecidableEq O] [Fintype V]
    [Fintype Tick] (T K : ℕ) : Channel R (Q × Adv) (O → V) (Option ((O → V) × Tick)) where
  S := (Fin T → Adv) × (Fin T → Tick) ×
    {d : Fin T × O → Option V // (Finset.univ.filter fun p => d p ≠ none).card ≤ K} × Fin (T + 1)
  ask s _ h x := (x.1, if hn : h.length < T then s.1 ⟨h.length, hn⟩ else x.2)
  tell s _ h _ y := if hn : h.length < s.2.2.2.val ∧ h.length < T then
      some (fun o => (s.2.2.1.val (⟨h.length, hn.2⟩, o)).getD (y o), s.2.1 ⟨h.length, hn.2⟩)
    else none

/-- **The served channel's covert capacity in bits**: each of the `T` windows' advice and timing bucket, each of the `K`
overwritten values with its name among the `T · |O|` released ones, and the halt. Two to its power is at least the
channel's number of covert strings. -/
noncomputable def servedBits (Adv O V Tick : Type) [Fintype Adv] [Fintype O] [Fintype V] [Fintype Tick] (T K : ℕ) : ℝ :=
  T * (Real.logb 2 (Fintype.card Adv) + Real.logb 2 (Fintype.card Tick)) +
    K * (Real.logb 2 (Fintype.card V) + Real.logb 2 (T * Fintype.card O + 1)) + Real.logb 2 (T + 1)

end WeightSecurity
