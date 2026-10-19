import Definitions.Accounting.Compose

/-!
# The served window's layout, its rounds, and the coin each served value is committed before (`Accounting.Served`)

A served int8 window (SERVE-H1's static form, `benchmarks/pouw/served_h1/serve.py`) is a fixed function of public
parameters (`Layout`): `steps` engine steps, `rows` slot rows a linear call, square `tile`s, and the model's linears in
call order, each its `(K, N)`. Call `(s, w)` is step `s`'s call of linear `w`; its units are its `rows/tile × N_w/tile`
tiles at positions `base s w + u`, `base s w = s·U + off w`, `U` the units of one step (`Layout.perStep`). Every one of
the window's `steps·U` units exists whatever the requests are (`Layout.population`).

Two forms (`Form`), which differ only in their rounds:
* `beside`, today's SERVE-H1: the tile proves receipts over the call's committed A rows and the registered B, beside
  cuBLASLt's served `y`, one live round a step (`window.live_challenger`, `sampled_proofs.challenger.LiveRecord`);
* `served`, J11 (kernels-dither-rtx): the served `y` is the tile's own `z`, out of one noised GEMM, so the noise exists
  before the call runs and call `(s, w + 1)`'s A rows are a function of call `(s, w)`'s `z`; one live round a call.

`commitCoin` is the coin before which the served code commits a value of each class (`Cls`) of call `(s, w)`, as the
served code is to do it: the registration (the weights' root over every linear's root(B), and the request queue's
root) before the salt coin and so before round `0`'s; a call's A rows and admit words under its round's root, before
its coin; its receipts (and, in the `served` form, its `z`) under the next round's root, the last round's in the
manifest before the draw. Today's code commits the receipts at the draw (`serve.py`'s receipt trees, built after
serving), which this schedule is the fix of; `z` it never commits.

`Tie` is what a declared circuit (`Accounting.Compose.Declared`) must be to be the served window's: its call sites and
rounds the layout's, each value's commit coin `commitCoin` of its class, and the IR facts that put each class's value
at the cut `commitCoin` names (a weight registered, an A row a run input read only by its own call, a receipt reading
its round's coin). `TieCommits` states that the tie gives `Accounting.Compose.LayerCommitted`, and `TieLive` that it
gives `Declared.Live` under any scheme: the two `Scope` conjuncts that are protocol structure. They are stated here,
to be proved in `Proofs`. The units, the population and the draw, which read
the verifier's partition of each call's Program, are `Properties.Served`'s.

What the verifier must check at run time for each class, so that a `Declared` meeting `Tie` is the run's (the
statement is over the declared circuit; these are the checks that make its `commitRound` the code's):
* `registered`: the salt coin opened against the registration root, recorded before every round's coin; the weights'
  root equal to the served model's published one; each drawn unit's B rows opened under its linear's root(B), and
  root(B) under the weights' root;
* `act`, `admit`: round `r`'s root, recorded before round `r`'s coin, over the A table roots and admit roots of exactly
  round `r`'s calls; each drawn unit's A rows and admit words opened under them;
* `receipt`, `out`: round `r + 1`'s root over round `r`'s receipt (and `z`) roots, the last round's in the manifest
  before the draw coin; each drawn unit's receipt (and `z`) opened under them;
* the unit's position `base s w + u`, computed by the verifier from the layout, never read; the unit's salt
  `window.round_key(commitment, coin_r, r, root_r)`'s stream, derived by the verifier;
* the draw: `K_d` units of `population` from the draw coin opened after the manifest.

The `served` form's `z` reads no coin in today's tile (`h1tile`: "z is each cell's exact product ... which no noise
reaches"), so its IR cut is its own round while J11's kernel writes it after that round's coin: `Tie`'s `noised` is then
false for `z` and the tie has no model. J11 needs either a tile whose `z` is computed off the noised product (its cut
then moves to the next round, where `commitCoin` puts it) or `LayerCommitted` relaxed for a word no luck premise reads.
-/

namespace Accounting.Served

open Accounting.Compose

/-- **A served window's public shape**: its engine `steps`, the slot `rows` of every linear call, the square `tile`
(rows and columns of a unit), and the model's `linears` in call order within a step, each its `(K, N)`. -/
structure Layout where
  steps : ℕ
  rows : ℕ
  tile : ℕ
  linears : List (ℕ × ℕ)
  deriving DecidableEq, Repr

namespace Layout

variable (L : Layout)

/-- The linear calls of one step. -/
def calls : ℕ := L.linears.length

/-- Linear `w`'s `K`, and `0` past the linears. -/
def kOf (w : ℕ) : ℕ := (L.linears.getD w (0, 0)).1

/-- Linear `w`'s `N`, and `0` past the linears. -/
def nOf (w : ℕ) : ℕ := (L.linears.getD w (0, 0)).2

/-- The units of one call of linear `w`: its row tiles by its column tiles. -/
def unitsOf (w : ℕ) : ℕ := (L.rows / L.tile) * (L.nOf w / L.tile)

/-- Linear `w`'s offset in a step: the units of the linears before it. -/
def off (w : ℕ) : ℕ := ((List.range w).map L.unitsOf).sum

/-- The units of one step, `U`. -/
def perStep : ℕ := L.off L.calls

/-- Call `(s, w)`'s first position, `s·U + off w`. -/
def base (s w : ℕ) : ℕ := s * L.perStep + L.off w

/-- The window's units, `steps·U`, every one present whatever the requests are. -/
def population : ℕ := L.steps * L.perStep

/-- Call site `c`'s step, numbering the call sites step by step, each step's in linear order. -/
def stepOf (c : ℕ) : ℕ := c / L.calls

/-- Call site `c`'s linear. -/
def linearOf (c : ℕ) : ℕ := c % L.calls

/-- **A layout is well formed**: a step, a linear, a tile dividing the rows and every linear's `N`, and a `K` of at
least one 32-byte segment. -/
def Wf : Prop :=
  0 < L.steps ∧ 0 < L.calls ∧ 0 < L.tile ∧ L.tile ∣ L.rows ∧
    ∀ w, w < L.calls → 0 < L.nOf w ∧ L.tile ∣ L.nOf w ∧ 32 ∣ L.kOf w ∧ 0 < L.kOf w

end Layout

/-- One decoder layer of Llama-3.1-8B's linears in call order: `qkv` (`K` 4096, `N` 4096 + 2·1024), `o`, `gate_up`
(`N` 2·14336) and `down`. -/
def llamaLayer : List (ℕ × ℕ) := [(4096, 6144), (4096, 4096), (4096, 28672), (14336, 4096)]

/-- **The served Llama-3.1-8B window** (SERVE-H1 round 4, r20261019-012935-2c52): 18 steps, 64 slot rows a call, 32 × 32
units, and 32 decoder layers of `llamaLayer`, 128 calls a step; `U` is 86,016 and the population 1,548,288. -/
def llama31_8b : Layout where
  steps := 18
  rows := 64
  tile := 32
  linears := (List.replicate 32 llamaLayer).flatten

/-- **A served window's form**: `beside`, receipts over the committed operands beside the served kernel, one round a
step (today's SERVE-H1); `served`, the tile's `z` is the served output, one round a call (J11). -/
inductive Form
  | beside
  | served
  deriving DecidableEq, Repr

/-- The window's live rounds: a step each (`beside`), a call each (`served`). -/
def Form.rounds : Form → Layout → ℕ
  | .beside, L => L.steps
  | .served, L => L.steps * L.calls

/-- Call `(s, w)`'s round. -/
def Form.round : Form → Layout → ℕ → ℕ → ℕ
  | .beside, _, s, _ => s
  | .served, L, s, w => s * L.calls + w

/-- **A served value's class**:
* `registered`: a row of a linear's B table (under the weights' root) or an entry of the request queue (under the
  queue's root), both under the registration root;
* `act`: a row of a call's A table, its slot's s8 row and scale (the empty row in a slot the queue leaves unassigned);
* `admit`: an admit word of a call, written by its admit unit from the queue;
* `receipt`: a receipt word of a call's unit;
* `out`: a served product word of a call's unit, its `z` (the `served` form only);
* `inner`: a value no opening reads, a proof unit's own wire (in the `beside` form, `z` too);
* `coin`: a round's coin input, the unit's salt port. -/
inductive Cls
  | registered
  | act
  | admit
  | receipt
  | out
  | inner
  | coin
  deriving DecidableEq, Repr

/-- **The coin before which the served code commits a value** of class `k` of call `(s, w)`, of the window's rounds
(the draw is `Form.rounds`): the registration before round `0`'s; A rows and admit words before their call's round's;
receipts and `z` before the next round's. `inner` and `coin` values are committed nowhere (`Tie.inner` sets an inner
value's commit coin to its cut; a coin is no value the prover writes). -/
def commitCoin (F : Form) (L : Layout) : Cls → ℕ → ℕ → ℕ
  | .registered, _, _ => 0
  | .act, s, w => F.round L s w
  | .admit, s, w => F.round L s w
  | .receipt, s, w => F.round L s w + 1
  | .out, s, w => F.round L s w + 1
  | .inner, _, _ => 0
  | .coin, _, _ => 0

/-- **Declared circuit `Cd`, its gates classed by `cls`, is served window `L` in form `F`**:
* `calls`, `rounds`, `round`: its call sites are the window's `steps × calls` calls, numbered step by step, and its rounds
  and each call site's round are the form's;
* `inRange`: every gate's call site is one of them;
* `live`: every round's coin is live (the live challenger's);
* `coin`, `registered`, `act`, `computed`: a gate is a coin input exactly when it is of class `coin`; a `registered`
  gate is a registered input; an `act` gate is a run input read only by gates of its own call; an `admit`, `receipt`,
  `out` or `inner` gate is no input;
* `noised`: a `receipt` or `out` gate depends on its round's coin (`coinMark` past its round);
* `beside`: the `beside` form has no `out` gate;
* `commit`, `inner`: a value of an opened class is committed at `commitCoin` of its class and call, and an inner value at
  its own cut.
The `commit` and `inner` fields are how the declared circuit is built from the layout; the others are facts of each
call's Program, which its compile checks once; the commit coins are the code's only if the verifier checks the run's
round log (the module docstring's list). -/
structure Tie {ι : Type} (L : Layout) (F : Form) (Cd : Declared ι) (cls : ℕ → Cls) : Prop where
  calls : Cd.calls = L.steps * L.calls
  rounds : Cd.rounds = F.rounds L
  round : ∀ c, c < Cd.calls → Cd.round c = F.round L (L.stepOf c) (L.linearOf c)
  inRange : ∀ i, i < Cd.dag.length → Cd.call i < Cd.calls
  live : ∀ r, r < Cd.rounds → Cd.coinMode r = .live
  coin : ∀ i, i < Cd.dag.length → (cls i = .coin ↔ Cd.binding i = some .coin)
  registered : ∀ i, i < Cd.dag.length → cls i = .registered → Cd.binding i = some .registered
  act : ∀ i, i < Cd.dag.length → cls i = .act →
    Cd.binding i = some .run ∧ ∀ j, j < Cd.dag.length → Cd.Reads j i → Cd.call j = Cd.call i
  computed : ∀ i, i < Cd.dag.length →
    (cls i = .admit ∨ cls i = .receipt ∨ cls i = .out ∨ cls i = .inner) → Cd.binding i = none
  noised : ∀ i, i < Cd.dag.length → (cls i = .receipt ∨ cls i = .out) →
    Cd.round (Cd.call i) + 1 ≤ (coinMark Cd).getD i 0
  beside : F = .beside → ∀ i, i < Cd.dag.length → cls i ≠ .out
  commit : ∀ i, i < Cd.dag.length → cls i ≠ .inner → cls i ≠ .coin →
    Cd.commitRound i = commitCoin F L (cls i) (L.stepOf (Cd.call i)) (L.linearOf (Cd.call i))
  inner : ∀ i, i < Cd.dag.length → cls i = .inner → Cd.commitRound i = Cd.cutOf i

/-- **The tie gives the service's commit order**: a declared circuit that is a well-formed served window, in either
form, meets `LayerCommitted`. The commit coins are then no later than the IR's cuts: a weight's `0`; an A row's, its
call's round, its first reader's; a receipt's, the round after its own, which its coin read reaches. -/
def TieCommits : Prop :=
  ∀ (ι : Type) (L : Layout) (F : Form) (Cd : Declared ι) (cls : ℕ → Cls),
    L.Wf → Tie L F Cd cls → LayerCommitted Cd

/-- **The tie's credited tiles are in live rounds**: under any scheme, every unit of a tied circuit with a credited word
is in a round of the window (its first gate's call is a call of the layout), and every such round is live, so the tied
circuit meets `Declared.Live`. -/
def TieLive : Prop :=
  ∀ (ι K : Type) (Sc : Scheme K) (L : Layout) (F : Form) (Cd : Declared ι) (cls : ℕ → Cls),
    L.Wf → Tie L F Cd cls → Cd.Live Sc

/-- **A served call's round is a round of the window**: call `(s, w)` of a well-formed layout, `s < steps` and
`w < calls`, is in round `F.round L s w < F.rounds L`, so its A rows' commit coin is a coin and its receipts' is at most
the draw. -/
def RoundInWindow : Prop :=
  ∀ (L : Layout) (F : Form) (s w : ℕ), L.Wf → s < L.steps → w < L.calls → F.round L s w < F.rounds L

/-- **The served Llama-3.1-8B window's shape**: 128 linear calls a step, 86,016 units a step, 1,548,288 a window, and
`llama31_8b` well formed. -/
def Llama31Shape : Prop :=
  llama31_8b.calls = 128 ∧ llama31_8b.perStep = 86016 ∧ llama31_8b.population = 1548288 ∧ llama31_8b.Wf

end Accounting.Served
