import Definitions.Accounting.Compose

/-!
# The served window's layout, its rounds, and the coin each served value is committed before (`Accounting.Served`)

A served int8 window (SERVE-H1's static form, `benchmarks/pouw/served_h1/serve.py`) is a fixed function of public
parameters (`Layout`): `steps` engine steps, `rows` slot rows a linear call, `tileRows × tileCols` units, and the
model's linears in call order, each its `(K, N)`. Call `(s, w)` is step `s`'s call of linear `w`; its units are its
`rows/tileRows × N_w/tileCols` tiles at positions `base s w + u`, `base s w = s·U + off w`, `U` the units of one step
(`Layout.perStep`). Every one of the window's `steps·U` units exists whatever the requests are (`Layout.population`).
The unit is the proof unit: SERVE-H1 receipts 32 × 32 units, which C-Flock can't prove at `K` = 4096 (about 5 G ANDs),
while W2 (#2657) proves 4 × 8 ones; the layout takes the size as a parameter, and the receipts must be per proof unit.

Every form has one live round a step (`window.live_challenger`, `sampled_proofs.challenger.LiveRecord`), and they differ
in the served output and in what a receipt unit reads of it (`Form`):
* `beside`, today's SERVE-H1: the tile proves receipts over the call's committed A rows and the registered B, beside
  the served linear's own GEMM (`int8_ops.scaled_mm`: `y = dequantize(torch._int_mm(A, B), s_a, s_w)`, cuBLASLt's s8
  GEMM, at the round-4 run's commit 14e4c23a), whose `y` no unit reads;
* `zTree`, J11's first iteration as kernels-dither-rtx describes it (its coin schedule unread here): one fused GEMM
  serves `y = dequant(z)` from its exact product `z` and commits `z` as an AROW row table under the step's round root;
  each receipt unit opens its tile's `z`;
* `yOpen`, the alternative kernels prices: no second tree; each receipt unit opens the Commit's own `y` rows of its tile
  and its Program checks `dequant(z) = y`, `z` an inner wire.

Integrity of the Commit's `y` is its served partition's sampled units' (proofs' framing, not read here), so it never
reads the form. The link from that `y` to the proved product does (`Form.ServedLink`): `beside` has none (no unit reads the served `y`);
`zTree` has it under a premise on the prover's fused kernel, that the `y` it serves is `dequant` of the `z` it commits,
which nothing checks and the design rules exclude (nothing the prover runs is trusted); `yOpen` has it from the tie,
whose `linked` field puts the Commit's `y` in every receipt unit, and the Program's check, with no premise. So `yOpen`
needs the less trusted surface.

The link is an equality of values, which useful-work credit needs and which doesn't give it. In every form the served
value is committed before its step's coin and every receipt reads that coin (`Tie.noised`), and a receipt's H1 sets
are over the noised operands `A + b` and `B + c + U`, which no function of `z` gives: the receipts' GEMM runs after the
served one, in every form. A schedule whose served GEMM is the noised one draws a call's coin after its A rows are
committed and before its GEMM, and a step's later calls read its earlier calls' outputs, so it has about a round a
call, not a step; this module doesn't state one. Today's code computes `y` as `dequant` of the exact s32 product
(`int8_ops.scaled_mm` at 14e4c23a: `RN_bf16(RN32(RN32(RN32(z)·s_a)·s_w))`), so `yOpen`'s check would hold of its `y`
once those rows are committed, which round 4 doesn't do.

Between the forms, `LayerCommitted` gains one committed class at its step's round (`z`, a computed
word that reads no coin, or the Commit's `y`, a run input only its own call reads) and the round root one table root
(`z`'s or the Commit's `y` rows'); the units keep their number, each gaining its tile's `z` or `y` words.

`commitCoin` is the coin before which the served code commits a value of each class (`Cls`) of step `s`, as the served
code is to do it: the registration (the weights' root over every linear's root(B), and the request queue's root) before
the salt coin and so before round `0`'s; a call's A rows, admit words, and its `z` or `y` rows under its step's round
root, before its coin; its receipts under the next round's root, the last round's in the manifest before the draw.
Today's code commits the receipts at the draw (`serve.py`'s receipt trees, built after serving), which this schedule is
the fix of, and its round root covers the A and admit roots only.

`Tie` is what a declared circuit (`Accounting.Compose.Declared`) must be to be the served window's: its call sites and
rounds the layout's, each value's commit coin `commitCoin` of its class, and the IR facts that put each class's value
at the cut `commitCoin` names (a weight registered, an A row or a `y` row a run input read only by its own call, a
receipt reading its round's coin). `TieCommits` states that the tie gives `Accounting.Compose.LayerCommitted`, and
`TieLive` that it gives `Declared.Live` under any scheme: the two `Scope` conjuncts that are protocol structure. The
units, the population and the draw, which read the verifier's partition of each call's Program, are
`Properties.Served`'s.

What the verifier must check at run time for each class, so that a `Declared` meeting `Tie` is the run's (the
statement is over the declared circuit; these are the checks that make its `commitRound` the code's):
* `registered`: the salt coin opened against the registration root, recorded before every round's coin; the weights'
  root equal to the served model's published one; each drawn unit's B rows opened under its linear's root(B), and
  root(B) under the weights' root;
* `act`, `admit`, `out`, `served`: round `s`'s root, recorded before round `s`'s coin, over the A table, admit, and `z`
  or Commit `y` roots of exactly step `s`'s calls; each drawn unit's rows of them opened under it;
* `receipt`: round `s + 1`'s root over step `s`'s receipt roots, the last step's in the manifest before the draw coin;
  each drawn unit's receipt opened under it;
* the unit's position `base s w + u`, computed by the verifier from the layout, never read; the unit's salt
  `window.round_key(commitment, coin_s, s, root_s)`'s stream, derived by the verifier;
* the draw: `K_d` units of `population` from the draw coin opened after the manifest.
-/

namespace Accounting.Served

open Accounting.Compose

/-- **A served window's public shape**: its engine `steps`, the slot `rows` of every linear call, a unit's
`tileRows` slot rows by `tileCols` weight rows, and the model's `linears` in call order within a step, each its
`(K, N)`. -/
structure Layout where
  steps : ℕ
  rows : ℕ
  tileRows : ℕ
  tileCols : ℕ
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
def unitsOf (w : ℕ) : ℕ := (L.rows / L.tileRows) * (L.nOf w / L.tileCols)

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

/-- **A layout is well formed**: a step, a linear, a unit's rows dividing the slot rows and its columns every
linear's `N`, as the H1 tile's lanes need (`4 ∣ tileRows`, `8 ∣ tileCols`), and a `K` of at least one 32-byte
segment. -/
def Wf : Prop :=
  0 < L.steps ∧ 0 < L.calls ∧ 0 < L.tileRows ∧ 0 < L.tileCols ∧ 4 ∣ L.tileRows ∧ 8 ∣ L.tileCols ∧
    L.tileRows ∣ L.rows ∧
    ∀ w, w < L.calls → 0 < L.nOf w ∧ L.tileCols ∣ L.nOf w ∧ 32 ∣ L.kOf w ∧ 0 < L.kOf w

end Layout

/-- One decoder layer of Llama-3.1-8B's linears in call order: `qkv` (`K` 4096, `N` 4096 + 2·1024), `o`, `gate_up`
(`N` 2·14336) and `down`. -/
def llamaLayer : List (ℕ × ℕ) := [(4096, 6144), (4096, 4096), (4096, 28672), (14336, 4096)]

/-- **The served Llama-3.1-8B window** (SERVE-H1 round 4, r20261019-012935-2c52): 18 steps, 64 slot rows a call, 32 × 32
units, and 32 decoder layers of `llamaLayer`, 128 calls a step; `U` is 86,016 and the population 1,548,288. -/
def llama31_8b : Layout where
  steps := 18
  rows := 64
  tileRows := 32
  tileCols := 32
  linears := (List.replicate 32 llamaLayer).flatten

/-- **A served window's form**: `beside`, receipts over the committed operands beside cuBLASLt's served `y` (today's
SERVE-H1); `zTree`, the fused kernel serves `dequant(z)` and commits `z`, which receipt units open (J11's first
iteration, as kernels describes it); `yOpen`, receipt units open the Commit's `y` and check `dequant(z) = y` (the
alternative). In all three the step's receipts are computed after its served values are committed. -/
inductive Form
  | beside
  | zTree
  | yOpen
  deriving DecidableEq, Repr

/-- **A served value's class**:
* `registered`: a row of a linear's B table (under the weights' root) or an entry of the request queue (under the
  queue's root), both under the registration root;
* `act`: a row of a call's A table, its slot's s8 row and scale (the empty row in a slot the queue leaves unassigned);
* `admit`: an admit word of a call, written by its admit unit from the queue;
* `receipt`: a receipt word of a call's unit;
* `out`: a word of a call's exact product `z`, committed in its AROW row table (`zTree` only);
* `served`: a word of the Commit's served output `y` of a call, which its unit opens (`yOpen` only);
* `inner`: a value no opening reads, a proof unit's own wire (`z` in `beside` and `yOpen`);
* `coin`: a round's coin input, the unit's salt port. -/
inductive Cls
  | registered
  | act
  | admit
  | receipt
  | out
  | served
  | inner
  | coin
  deriving DecidableEq, Repr

/-- **The coin before which the served code commits a value** of class `k` of step `s`, of the window's `steps` rounds
(the draw is `steps`): the registration before round `0`'s; A rows, admit words, `z` and the Commit's `y` before their
step's round's; receipts before the next round's. `inner` and `coin` values are committed nowhere (`Tie.inner` sets an
inner value's commit coin to its cut; a coin is no value the prover writes). -/
def commitCoin : Cls → ℕ → ℕ
  | .registered, _ => 0
  | .act, s => s
  | .admit, s => s
  | .out, s => s
  | .served, s => s
  | .receipt, s => s + 1
  | .inner, _ => 0
  | .coin, _ => 0

/-- **What ties receipt `i`'s unit to the served output**, by form: nothing in `beside`; in `zTree`, a `z` word of the
unit; in `yOpen`, a gate of the unit that reads a word of the Commit's `y`. -/
def Form.Links {ι : Type} (Cd : Declared ι) (cls : ℕ → Cls) : Form → ℕ → Prop
  | .beside, _ => True
  | .zTree, i => ∃ j, j < Cd.dag.length ∧ cls j = .out ∧ Cd.unit j = Cd.unit i
  | .yOpen, i => ∃ g j, g < Cd.dag.length ∧ Cd.unit g = Cd.unit i ∧ Cd.Reads g j ∧ cls j = .served

/-- **The served `y` is the proved product's**, by form, where `fused` is the fused kernel's premise (the `y` it serves
is `dequant` of the `z` rows it commits): `beside` has no link (`False`); `zTree` has it exactly under `fused`, a
premise on code the prover runs, which no run-time check reaches; `yOpen` has it from `Tie.linked` and the receipt
Program's `dequant(z) = y` check over the opened `y` (a fact of that Program's Definition, which doesn't exist yet),
with no premise, given the run-time check that the unit's `served` words open under the Commit's `y` root: `Tie`
doesn't tell a `served` run input from an `act` one. It is an equality of values: useful-work credit needs it, and it doesn't give that credit, since in
every form the receipts' GEMM follows the served one (the module docstring). Integrity reads none of it. -/
def Form.ServedLink (fused : Prop) : Form → Prop
  | .beside => False
  | .zTree => fused
  | .yOpen => True

/-- **Declared circuit `Cd`, its gates classed by `cls`, is served window `L` in form `F`**:
* `calls`, `rounds`, `round`: its call sites are the window's `steps × calls` calls, numbered step by step, its rounds
  the window's steps, and each call site's round its step;
* `inRange`: every gate's call site is one of them;
* `live`: every round's coin is live (the live challenger's);
* `coin`, `registered`, `act`, `computed`: a gate is a coin input exactly when it is of class `coin`; a `registered`
  gate is a registered input; an `act` or `served` gate is a run input read only by gates of its own call; an `admit`,
  `receipt`, `out` or `inner` gate is no input; so every input's class is its binding's;
* `committed`: a gate that is no input is `inner` exactly when the IR doesn't mark it a committed output
  (`Gates.Gate.commit`), so no opened value is declared inner and committed at its cut by fiat;
* `noised`: a `receipt` gate depends on its round's coin (`coinMark` past its round);
* `forms`, `linked`: an `out` gate is `zTree`'s and a `served` gate `yOpen`'s, and every receipt's unit has its form's
  link to the served output (`Form.Links`);
* `commit`, `inner`: a value of an opened class is committed at `commitCoin` of its class and step, and an inner value
  at its own cut.
The `commit` and `inner` fields are how the declared circuit is built from the layout; the others are facts of each
call's Program, which no code checks yet (an extractor building `Cd` from the compiled Programs is to); the commit
coins are the code's only if the verifier checks the run's round log (the module docstring's list). -/
structure Tie {ι : Type} (L : Layout) (F : Form) (Cd : Declared ι) (cls : ℕ → Cls) : Prop where
  calls : Cd.calls = L.steps * L.calls
  rounds : Cd.rounds = L.steps
  round : ∀ c, c < Cd.calls → Cd.round c = L.stepOf c
  inRange : ∀ i, i < Cd.dag.length → Cd.call i < Cd.calls
  live : ∀ r, r < Cd.rounds → Cd.coinMode r = .live
  coin : ∀ i, i < Cd.dag.length → (cls i = .coin ↔ Cd.binding i = some .coin)
  registered : ∀ i, i < Cd.dag.length → cls i = .registered → Cd.binding i = some .registered
  act : ∀ i, i < Cd.dag.length → (cls i = .act ∨ cls i = .served) →
    Cd.binding i = some .run ∧ ∀ j, j < Cd.dag.length → Cd.Reads j i → Cd.call j = Cd.call i
  computed : ∀ i, i < Cd.dag.length →
    (cls i = .admit ∨ cls i = .receipt ∨ cls i = .out ∨ cls i = .inner) → Cd.binding i = none
  committed : ∀ i, i < Cd.dag.length → Cd.binding i = none →
    (cls i = .inner ↔ (Cd.dag[i]?).any (fun g => g.commit) = false)
  noised : ∀ i, i < Cd.dag.length → cls i = .receipt → Cd.round (Cd.call i) + 1 ≤ (coinMark Cd).getD i 0
  forms : ∀ i, i < Cd.dag.length → (cls i = .out → F = .zTree) ∧ (cls i = .served → F = .yOpen)
  linked : ∀ i, i < Cd.dag.length → cls i = .receipt → F.Links Cd cls i
  commit : ∀ i, i < Cd.dag.length → cls i ≠ .inner → cls i ≠ .coin →
    Cd.commitRound i = commitCoin (cls i) (L.stepOf (Cd.call i))
  inner : ∀ i, i < Cd.dag.length → cls i = .inner → Cd.commitRound i = Cd.cutOf i

/-- **The tie gives the service's commit order**: a declared circuit that is a served window, in any form, meets
`LayerCommitted`. The commit coins are then no later than the IR's cuts: a weight's `0`; an A row's or a `y` row's,
its step's round, its first reader's; an admit word's or a `z` word's, its step's round, the floor of its cut; a
receipt's, the round after its own, which its coin read reaches. -/
def TieCommits : Prop :=
  ∀ (ι : Type) (L : Layout) (F : Form) (Cd : Declared ι) (cls : ℕ → Cls), Tie L F Cd cls → LayerCommitted Cd

/-- **The tie's credited tiles are in live rounds**: under any scheme, every unit of a tied circuit with a credited word
is in a round of the window (its first gate's call is a call of the layout), and every such round is live, so the tied
circuit meets `Declared.Live`. -/
def TieLive : Prop :=
  ∀ (ι K : Type) (Sc : Scheme K) (L : Layout) (F : Form) (Cd : Declared ι) (cls : ℕ → Cls),
    Tie L F Cd cls → Cd.Live Sc

/-- **The served Llama-3.1-8B window's shape**: 128 linear calls a step, 86,016 units a step, 1,548,288 a window, and
`llama31_8b` well formed. -/
def Llama31Shape : Prop :=
  llama31_8b.calls = 128 ∧ llama31_8b.perStep = 86016 ∧ llama31_8b.population = 1548288 ∧ llama31_8b.Wf

end Accounting.Served
