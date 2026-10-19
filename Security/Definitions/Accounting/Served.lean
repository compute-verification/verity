import Definitions.Accounting.Compose
import Definitions.Accounting.Gates.Declared
import Definitions.Pouw.Game.Deadline

/-!
# The served window's layout, its rounds, and the coin each served value is committed before (`Accounting.Served`)

A served int8 window (SERVE-H1's static form, `benchmarks/pouw/served_h1/serve.py`) is a fixed function of public
parameters (`Layout`): `steps` engine steps, `rows` slot rows a linear call, `tileRows × tileCols` units, and the
model's linears in call order, each its `(K, N)`. Call `(s, w)` is step `s`'s call of linear `w`; its units are its
`rows/tileRows × N_w/tileCols` tiles at positions `base s w + u`, `base s w = s·U + off w`, `U` the units of one step
(`Layout.perStep`). Every one of the window's `steps·U` units exists whatever the requests are (`Layout.population`).
The unit is the proof unit: SERVE-H1 receipts 32 × 32 units, which C-Flock can't prove at `K` = 4096 (about 5 G ANDs),
while W2 (#2657) proves 4 × 8 ones; the layout takes the size as a parameter, and the receipts must be per proof unit.

Three forms have one live round a step (`window.live_challenger`, `sampled_proofs.challenger.LiveRecord`) and one has a
live round a call; they differ in the served output and in what a receipt unit reads of it (`Form`):
* `beside`, today's SERVE-H1: the tile proves receipts over the call's committed A rows and the registered B, beside
  the served linear's own GEMM (`int8_ops.scaled_mm`: `y = dequantize(torch._int_mm(A, B), s_a, s_w)`, cuBLASLt's s8
  GEMM, at the round-4 run's commit 14e4c23a), whose `y` no unit reads;
* `zTree`, J11's first iteration as kernels-dither-rtx describes it (its coin schedule unread here): one fused GEMM
  serves `y = dequant(z)` from its exact product `z` and commits `z` as an AROW row table under the step's round root;
  each receipt unit opens its tile's `z`;
* `yOpen`, the alternative kernels prices: no second tree; each receipt unit opens the Commit's own `y` rows of its tile
  and its Program checks `dequant(z) = y`, `z` an inner wire;
* `perCall`, a round a call: round `c` is call `c`'s (`steps × calls` rounds, 2,304 for Llama-3.1-8B); its root
  covers the call's A rows, its coin noises the call's GEMM, and the call's receipts and its served `y = dequant(z)`
  go under round `c + 1`'s root, `y` as each unit's own output words, so the Commit's `y` tree is the units' (one
  tree, compared by root). Its `y` reads the coin (`Tie.noised`, reachability: the coin is in `y`'s cone), which is
  what puts Compose's cut of `y` after the coin; `h1tile`'s `z`, its own dot, reads none, and Compose's cut would
  then put `y` before the coin its GEMM needs. It was the schedule a path whose served product is the noised one
  needs, which credit no longer asks (below).

Integrity of the Commit's `y` is its served partition's sampled units' (proofs' framing, not read here), so it never
reads the form. The link from that `y` to the proved product does (`Form.ServedLink`): `beside` has none (no unit
reads the served `y`); `zTree` has it under a premise on the prover's fused kernel, that the `y` it serves is
`dequant` of the `z` it commits, which nothing checks and the design rules exclude (nothing the prover runs is
trusted); `yOpen` has it from the tie, whose `linked` field puts a reader of the Commit's `y` in every receipt unit
(a read, not a dependence), and the Program's check, with no premise; `perCall` has it from the tie and the unit's own
`dequant(z)`, under the run-time check that the Commit's `y` root of the call is the root of those output words. So
`yOpen` and `perCall` need the less trusted surface. Under `ValueChecked` (below) a `zTree` unit commits its
`y = dequant(z)` beside its `z`, and its link is then `perCall`'s run-time `y`-root check, with no premise on the
kernel.

The link is an equality of values, which useful-work credit needs and which doesn't give it. Useful-work credit is
Daniel's ASK-71 (19 Oct, 22:09Z): the credited noised work was done on the operands `A`, `B` of a served product that
is value-checked on every drawn unit, `z_u = A_u B_uᵀ` proved and committed before the draw coin, with the served `y`
computed from that committed `z_u`. `ValueChecked` states it of the circuit, and `CreditTie` is `Tie` with it. Each
receipt's unit has an `out` word `z` whose value, on every typed input, is the dot of its call's committed A row with
the registered B column over the linear's whole `K` (`ProductOf`), and an `out` word `y` whose value is the Program's
public dequant `dq` of `z`'s and the scales' (`DequantOf`); every `out` word is one or the other. Both are
function-of statements: a `y` that reads `z` and takes another product's value (`y = 0·z + 1·p`, RT3-INT-3) fails
`DequantOf`. The receipt reads `z`'s operands (`Feeds`), which is reachability; that it is their noised product is its
public Program's. The credited work is computation on the served operands, not necessarily the computation that
produced the served bytes: a prover may serve from a plain GEMM beside the noised one. No check can tell a `z` derived
from the noised accumulators (kernels' `z = O − X cᵀ − b Bᵀ + 256·xa Bᵀ`, `X = A + b`, `xa` its wrap) from one
computed beside them (RT3-INT-11), so `ValueChecked` asks neither: a `z` of either kind meets it if its value is
`A_u B_uᵀ`'s. `CreditTie` reads no form: `zTree` meets it at one round a step (`Properties.Served.witCreditTie`), so
credit needs neither `perCall`'s schedule nor its rounds, and `perCall` meets it with a `z` peeled from the noised
product (`Properties.Served.witCallCredit`); `beside` and `yOpen` have no `out` word, and a window of theirs with a
receipt fails it (`Properties.Served.wit_notValueChecked`). Where `z` is committed is the form's: under its round's
root before the coin in `zTree`, where it reads no coin; with the receipts in `perCall`, where `Tie.noised` asks its
cone, an `out` word's, to hold the call's coin, which a peeled `z` meets and a dot beside the noised product doesn't
(its cut is then its own round, which a commit after the coin is past). No credit statement is stated here. Today's
code computes `y` as `dequant` of the exact s32 product (`int8_ops.scaled_mm` at 14e4c23a:
`RN_bf16(RN32(RN32(RN32(z)·s_a)·s_w))`, the `dq` a Definition would fix), so `yOpen`'s check and `DequantOf` would
hold of its `y` once its units commit those words, which round 4 doesn't do.

Between the forms, `LayerCommitted` gains one committed class: at its step's round in `zTree` and `yOpen` (`z`, a
computed word that reads no coin, and under `ValueChecked` its `y`, or the Commit's `y`, a run input only its own call
reads), and at the round after its call's in `perCall` (`y`, a computed word that reads the call's coin, and under
`ValueChecked` its `z`). The round root gains one table root (`z`'s or `y`'s; both under `ValueChecked`), and the
units keep their number, each gaining its tile's `z` or `y` words.

`commitCoin` is the coin before which the served code commits a value of each class (`Cls`) of round `r`, as the
served code is to do it: the registration (the weights' root over every linear's root(B)) before the salt coin and so
before round `0`'s; a call's A rows, the request entries its round admits and its `z` or `y` rows under its round's
root, before its coin (`perCall`'s `y` under the next round's); its receipts under the next round's root, the last
round's in the manifest before the draw. A request entry is a run input (`act`), not a registered one: certified
serving releases a window's requests after its salt frame, so no queue root can precede the salt (RT3-INT-9). Today's
code commits the receipts at the draw (`serve.py`'s receipt trees, built after serving), which this schedule is the
fix of, and its round root covers the A and admit roots only. The admit words and the tally have no class: their
always-proved units are not in the per-call Program `Properties.Served.ServedUnits` names, so no gate of a served
circuit is one (RT2-INT-5). They are PADCREDIT's (R8, R9), and the round root's admit root stays a run-time check
below.

`Tie` is what a declared circuit (`Accounting.Compose.Declared`) must be to be the served window's: its call sites and
rounds the layout's, each value's commit coin `commitCoin` of its class, and the IR facts that put each class's value
at the cut `commitCoin` names (a weight registered, an A row or a `y` row a run input read only by its own call, a
receipt reading its round's coin). `TieCommits` states that the tie gives `Accounting.Compose.LayerCommitted`, and
`TieLive` that it gives `Declared.Live` under any scheme: the two `Scope` conjuncts that are protocol structure. The
units, the population and the draw, which read the verifier's partition of each call's Program, are
`Properties.Served`'s. `EpochTie` ties an epoch's windows: each one's `Tie`, and each timed from the one salt's release,
`Pouw.TimedRun.ts`, the start of the interval `Pouw.Guarantees.WorkBetweenSaltAndDeadline` counts work in. `CreditTie`
is `Tie` with what useful-work credit needs of the circuit (`ValueChecked`): a necessary condition, which no credit
statement reads yet.

What the verifier must check at run time for each class, so that a `Declared` meeting `Tie` is the run's (the
statement is over the declared circuit; these are the checks that make its `commitRound` the code's):
* `registered`: the salt coin opened against the registration root, recorded before every round's coin; the weights'
  root equal to the served model's published one; each drawn unit's B rows opened under its linear's root(B), and
  root(B) under the weights' root;
* `act`, `out`, `served`: round `r`'s root, recorded before round `r`'s coin, over the A table, the request entries
  the round admits, admit (PADCREDIT's), and `z` or Commit `y` roots of exactly round `r`'s calls (`perCall`'s `y`
  roots in round `r + 1`'s; both under `ValueChecked`); each drawn unit's rows of them opened under it; in `perCall`,
  and in any form under `ValueChecked`, each call's Commit `y` root equal to its units' `y` root;
* the challenger: every round's coin opened by the live challenger, the auditor's device (AGENTS.md, auditor-trusted
  devices; `trusted-nebius-host`), on a machine the prover doesn't control and never a process on the prover's node.
  A challenger in the prover's process gives the prover its coins as they are drawn, so its coins are seeded ones,
  whatever the record calls them. `serve.py` builds `live_challenger` in the prover's process today (#2639 l.446–454,
  `window.py`'s `entropy=os.urandom`), so its runs establish only the order of roots and coins (RT2-INT-6, SG-11);
* `receipt`: round `r + 1`'s root over round `r`'s receipt roots, the last round's in the manifest before the draw
  coin; each drawn unit's receipt opened under it;
* the unit's position `base s w + u`, computed by the verifier from the layout, never read; the unit's salt
  `window.round_key(commitment, coin_r, r, root_r)`'s stream, derived by the verifier from the challenger's opening of
  round `r`'s coin and the recorded root, never read from the record (`serve.py`'s reference recomputation reads the
  record's `salt` today);
* every window of a salt timed from the salt's release (`EpochTie.salted`): the challenger's one `salted_at` an epoch,
  never a window's own start;
* the window's `F.rounds L` live rounds (`Llama31Rounds`). Compose's bound reads the largest round's share of the
  words, not the count, which is `1/F.rounds L` only when the rounds are equal (`Llama31Rounds` has the numbers);
* the draw: `K_d` units of `population` from the draw coin opened after the manifest;
* for credit (`CreditTie`), `ValueChecked` of each call's compiled Program at the Program's `dq`, once per Program
  as `ServedUnits` is. No gate checks it yet: SERVE-GATE's (#2731) has no such check. A fact of the Program's IR, it
  says nothing of the prover's kernel, which may serve from a GEMM beside the noised one; the credited work is the
  receipts' noised GEMM on the served operands. `ProductOf` and `DequantOf` bear on a run whose inputs are typed
  (`Gates.Typed`): each drawn unit's opened input word in the declared range of every input gate of its slot.
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
alternative); `perCall`, a round a call, whose units output the served `y = dequant(z)`, committed the round after
their own. In the first three a step's receipts are computed after its served values are committed. -/
inductive Form
  | beside
  | zTree
  | yOpen
  | perCall
  deriving DecidableEq, Repr

/-- Form `F`'s rounds on layout `L`: one a call in `perCall`, one a step in the others. -/
def Form.rounds (L : Layout) : Form → ℕ
  | .perCall => L.steps * L.calls
  | _ => L.steps

/-- Call site `c`'s round in form `F`: the call's own in `perCall`, its step's in the others. -/
def Form.roundOf (L : Layout) : Form → ℕ → ℕ
  | .perCall, c => c
  | _, c => L.stepOf c

/-- The rounds past its own after which form `F` commits an `out` word: one in `perCall`, whose `y` reads its round's
coin, none in the others. -/
def Form.outLag : Form → ℕ
  | .perCall => 1
  | _ => 0

/-- **A served value's class**:
* `registered`: a row of a linear's B table, or its scale, under the weights' root, the registration root;
* `act`: a row of a call's A table, its slot's s8 row and scale (the empty row in a slot no request is assigned to),
  or a request entry its round admits: a run input under its round's root. Certified serving releases a window's
  requests after its salt frame, so a request entry can't be committed before the salt (RT3-INT-9);
* `receipt`: a receipt word of a call's unit;
* `out`: a word a call's unit computes and commits in a row table: the exact product `z` in `zTree`, the served
  `y = dequant(z)` in `perCall`, and under `ValueChecked` both in either;
* `served`: a word of the Commit's served output `y` of a call, which its unit opens (`yOpen` only);
* `inner`: a value no opening reads, a proof unit's own wire (`z` in `beside` and `yOpen`);
* `coin`: a round's coin input, the unit's salt port.
A call's admit words have no class: their always-proved units are outside the per-call Program (the module
docstring). -/
inductive Cls
  | registered
  | act
  | receipt
  | out
  | served
  | inner
  | coin
  deriving DecidableEq, Repr

/-- **The coin before which the served code commits a value** of class `k` of round `r` in form `F`, of the window's
`F.rounds L` rounds (the draw is the last): the registration before round `0`'s; A rows, `zTree`'s `z` and the
Commit's `y` before their round's; receipts and `perCall`'s `y`, which read their round's coin, before the
next round's. `inner` and `coin` values are committed nowhere (`Tie.inner` sets an inner value's commit coin to its
cut; a coin is no value the prover writes). -/
def commitCoin (F : Form) : Cls → ℕ → ℕ
  | .registered, _ => 0
  | .act, r => r
  | .out, r => r + F.outLag
  | .served, r => r
  | .receipt, r => r + 1
  | .inner, _ => 0
  | .coin, _ => 0

/-- **What ties receipt `i`'s unit to the served output**, by form: nothing in `beside`; in `zTree` and `perCall`, an
`out` word of the unit (`z`, or the served `y`); in `yOpen`, a gate of the unit that reads a word of the Commit's
`y`. A read, not a dependence: what ties the `out` word's value to the product is `ValueChecked`. -/
def Form.Links {ι : Type} (Cd : Declared ι) (cls : ℕ → Cls) : Form → ℕ → Prop
  | .beside, _ => True
  | .zTree, i => ∃ j, j < Cd.dag.length ∧ cls j = .out ∧ Cd.unit j = Cd.unit i
  | .yOpen, i => ∃ g j, g < Cd.dag.length ∧ Cd.unit g = Cd.unit i ∧ Cd.Reads g j ∧ cls j = .served
  | .perCall, i => ∃ j, j < Cd.dag.length ∧ cls j = .out ∧ Cd.unit j = Cd.unit i

/-- **The served `y` is the proved product's**, by form, where `fused` is the fused kernel's premise (the `y` it serves
is `dequant` of the `z` rows it commits): `beside` has no link (`False`); `zTree` has it exactly under `fused`, a
premise on code the prover runs, which no run-time check reaches; `yOpen` has it from `Tie.linked` and the receipt
Program's `dequant(z) = y` check over the opened `y` (a fact of that Program's Definition, which doesn't exist yet),
with no premise, given the run-time check that the unit's `served` words open under the Commit's `y` root: `Tie`
doesn't tell a `served` run input from an `act` one. `perCall` has it from `Tie.linked` and the unit's own
`dequant(z)` output, given the run-time check that the call's Commit `y` root is its units' `y` root. Under
`ValueChecked` a `zTree` unit outputs its `y` too, and its link is that run-time check, not `fused`. It is an equality
of values: useful-work credit needs it, and it doesn't give that credit (`ValueChecked`, the module docstring).
Integrity reads none of it. -/
def Form.ServedLink (fused : Prop) : Form → Prop
  | .beside => False
  | .zTree => fused
  | .yOpen => True
  | .perCall => True

/-- **Declared circuit `Cd`, its gates classed by `cls`, is served window `L` in form `F`**:
* `calls`, `rounds`, `round`: its call sites are the window's `steps × calls` calls, numbered step by step, its rounds
  the form's (`Form.rounds`), and each call site's round its form's (`Form.roundOf`: its step's, or `perCall`'s own);
* `inRange`: every gate's call site is one of them;
* `live`: every round's coin is live: the live challenger's, which runs off the prover's node, on the auditor's device
  (AGENTS.md, auditor-trusted devices; `trusted-nebius-host`). A challenger in the prover's process gives seeded coins,
  however `coinMode` records them, so a `Declared` with `.live` rounds is the run's only under the module docstring's
  challenger check, which today's `serve.py` doesn't meet;
* `coin`, `registered`, `act`, `computed`: a gate is a coin input exactly when it is of class `coin`; a `registered`
  gate is a registered input; an `act` or `served` gate is a run input read only by gates of its own call; a
  `receipt`, `out` or `inner` gate is no input; so every input's class is its binding's;
* `committed`: a gate that is no input is `inner` exactly when the IR doesn't mark it a committed output
  (`Gates.Gate.commit`), so no opened value is declared inner and committed at its cut by fiat;
* `noised`: a `receipt` gate, and in `perCall` an `out` gate, reads its round's coin (`coinMark` past its round).
  This is reachability: the coin input is in the gate's cone, not that the gate's value depends on the coin. It is
  what Compose's cut reads (`Declared.cutOf`), so it is what the commit order needs; that a receipt's value is the
  noised product is its public Program's (`Properties.Served.ServedUnits`);
* `forms`, `linked`: an `out` gate is `zTree`'s or `perCall`'s and a `served` gate `yOpen`'s, and every receipt's unit
  has its form's link to the served output (`Form.Links`);
* `commit`, `inner`: a value of an opened class is committed at `commitCoin` of its form, class and round, and an
  inner value at its own cut.
The `commit` and `inner` fields are how the declared circuit is built from the layout; the others are facts of each
call's Program, which no code checks yet (an extractor building `Cd` from the compiled Programs is to); the commit
coins are the code's only if the verifier checks the run's round log (the module docstring's list). -/
structure Tie {ι : Type} (L : Layout) (F : Form) (Cd : Declared ι) (cls : ℕ → Cls) : Prop where
  calls : Cd.calls = L.steps * L.calls
  rounds : Cd.rounds = F.rounds L
  round : ∀ c, c < Cd.calls → Cd.round c = F.roundOf L c
  inRange : ∀ i, i < Cd.dag.length → Cd.call i < Cd.calls
  live : ∀ r, r < Cd.rounds → Cd.coinMode r = .live
  coin : ∀ i, i < Cd.dag.length → (cls i = .coin ↔ Cd.binding i = some .coin)
  registered : ∀ i, i < Cd.dag.length → cls i = .registered → Cd.binding i = some .registered
  act : ∀ i, i < Cd.dag.length → (cls i = .act ∨ cls i = .served) →
    Cd.binding i = some .run ∧ ∀ j, j < Cd.dag.length → Cd.Reads j i → Cd.call j = Cd.call i
  computed : ∀ i, i < Cd.dag.length →
    (cls i = .receipt ∨ cls i = .out ∨ cls i = .inner) → Cd.binding i = none
  committed : ∀ i, i < Cd.dag.length → Cd.binding i = none →
    (cls i = .inner ↔ (Cd.dag[i]?).any (fun g => g.commit) = false)
  noised : ∀ i, i < Cd.dag.length → (cls i = .receipt ∨ (cls i = .out ∧ F = .perCall)) →
    Cd.round (Cd.call i) + 1 ≤ (coinMark Cd).getD i 0
  forms : ∀ i, i < Cd.dag.length → (cls i = .out → F = .zTree ∨ F = .perCall) ∧ (cls i = .served → F = .yOpen)
  linked : ∀ i, i < Cd.dag.length → cls i = .receipt → F.Links Cd cls i
  commit : ∀ i, i < Cd.dag.length → cls i ≠ .inner → cls i ≠ .coin →
    Cd.commitRound i = commitCoin F (cls i) (F.roundOf L (Cd.call i))
  inner : ∀ i, i < Cd.dag.length → cls i = .inner → Cd.commitRound i = Cd.cutOf i

/-- **The tie gives the service's commit order**: a declared circuit that is a served window, in any form, meets
`LayerCommitted`. The commit coins are then no later than the IR's cuts: a weight's `0`; an A row's or a `y` row's,
its round, its first reader's; a `zTree` `z` word's, its round, the floor of its cut; a receipt's
or `perCall`'s `y` word's, the round after its own, which its coin read reaches. -/
def TieCommits : Prop :=
  ∀ (ι : Type) (L : Layout) (F : Form) (Cd : Declared ι) (cls : ℕ → Cls), Tie L F Cd cls → LayerCommitted Cd

/-- **The tie's credited tiles are in live rounds**: under any scheme, every unit of a tied circuit with a credited word
is in a round of the window (its first gate's call is a call of the layout), and every such round is live, so the tied
circuit meets `Declared.Live`. -/
def TieLive : Prop :=
  ∀ (ι K : Type) (Sc : Scheme K) (L : Layout) (F : Form) (Cd : Declared ι) (cls : ℕ → Cls),
    Tie L F Cd cls → Cd.Live Sc

/-- **Gate `i` is in gate `j`'s cone** along reads: `j` is `i`, or reads a gate whose cone holds `i`. This is
reachability, not dependence: `j`'s value may ignore `i`'s (`0·i`). -/
def Feeds {ι : Type} (Cd : Declared ι) (i j : ℕ) : Prop := Relation.ReflTransGen (fun a b => Cd.Reads b a) i j

/-- Gate `i` is declared an s8 word, `[−128, 127]`: the type of an A row's and a B row's words. -/
def S8 {ι : Type} (Cd : Declared ι) (i : ℕ) : Prop :=
  (Cd.dag[i]?).any (fun g => decide (g.lo = -128 ∧ g.hi = 127)) = true

/-- **Gate `z` is the product of its call's committed A row and a registered B column** (ASK-71's value check): its
operands `as`, run inputs of `z`'s call (`act`), and `bs` (`registered`), each declared an s8 word (`S8`), are as many
as its linear's `K`, and on every typed input (`Gates.Typed`: each input slot a word of its type) `z`'s value is
`Σ_k a_k·b_k`, a coordinate of `A_u B_uᵀ`. A function-of statement: it holds of any gates with that value (`h1tile`'s
dot, cuBLASLt's, or a peel of the noised accumulators if its corrections cancel the noise on every typed input), and of
no `z` whose value differs from it on some typed input. `S8` keeps the typed inputs the served rows' words: an operand
declared `[0, 0]` would leave only `a = 0` typed. Which row and column `as` and `bs` are is the call's Program's
(`Properties.Served.ServedUnits`). -/
def ProductOf {ι : Type} (L : Layout) (Cd : Declared ι) (cls : ℕ → Cls) (z : ℕ) (as bs : List ℕ) : Prop :=
  as.length = L.kOf (L.linearOf (Cd.call z)) ∧ bs.length = as.length ∧
    (∀ a ∈ as, a < Cd.dag.length ∧ cls a = .act ∧ Cd.call a = Cd.call z ∧ S8 Cd a) ∧
    (∀ b ∈ bs, b < Cd.dag.length ∧ cls b = .registered ∧ S8 Cd b) ∧
    ∀ x, Gates.Typed Cd.dag x →
      Gates.val Cd.dag x z = (List.zipWith (· * ·) (as.map (Gates.val Cd.dag x)) (bs.map (Gates.val Cd.dag x))).sum

/-- **Gate `y` is the public dequant `dq` of gate `z`**: on every typed input `y`'s value is `dq` of `z`'s and of the
scales `ss` (the A row's, `act`, and the weight's, `registered`), so no other value reaches `y`'s. A function-of
statement: a `y` that reads `z` and takes another product's value (`y = 0·z + 1·p`, RT3-INT-3) fails it unless that
value is `dq` of `z`'s on every typed input. That `dq` is the dequant the served code computes is the Definition's
review. -/
def DequantOf {ι : Type} (dq : List ℤ → ℤ) (Cd : Declared ι) (cls : ℕ → Cls) (y z : ℕ) (ss : List ℕ) : Prop :=
  (∀ s ∈ ss, s < Cd.dag.length ∧ (cls s = .act ∨ cls s = .registered)) ∧
    ∀ x, Gates.Typed Cd.dag x → Gates.val Cd.dag x y = dq (Gates.val Cd.dag x z :: ss.map (Gates.val Cd.dag x))

/-- **Every drawn unit's served product is value-checked** (ASK-71, Daniel, 19 Oct, 22:09Z), what useful-work credit
needs of the circuit:
* each receipt `r`'s unit has an `out` word `z` that is the product of its call's A row and a B column (`ProductOf`),
  and an `out` word `y` that is `dq` of `z` (`DequantOf`), both committed before the draw coin; `r` reads each of
  `z`'s operands (`Feeds`);
* every `out` word is such a product, or `dq` of one of its unit's, so the served `y` words, the Commit's under the
  run-time `y`-root check, are each `dq` of a value-checked `z`.
`ProductOf` and `DequantOf` are function-of statements. `Feeds` is reachability: `z`'s operands are in the receipt's
cone, which doesn't say the receipt is their noised product; that is its public Program's
(`Properties.Served.ServedUnits`). The credited work is the receipts' noised GEMM on the served operands, not
necessarily the computation that produced the served bytes: a prover may serve from a plain GEMM beside the noised
one, which no check can tell from a `z` peeled from the noised accumulators (RT3-INT-11), and this asks neither. `dq`
is the Program's public dequant, `RN_bf16(RN32(RN32(RN32(z)·s_a)·s_w))` in today's code (`int8_ops.scaled_mm` at
14e4c23a). -/
def ValueChecked {ι : Type} (L : Layout) (dq : List ℤ → ℤ) (Cd : Declared ι) (cls : ℕ → Cls) : Prop :=
  (∀ r, r < Cd.dag.length → cls r = .receipt →
    ∃ z y as bs ss, z < Cd.dag.length ∧ y < Cd.dag.length ∧ Cd.unit z = Cd.unit r ∧ Cd.unit y = Cd.unit r ∧
      cls z = .out ∧ cls y = .out ∧ Cd.commitRound z ≤ Cd.rounds ∧ Cd.commitRound y ≤ Cd.rounds ∧
      ProductOf L Cd cls z as bs ∧ DequantOf dq Cd cls y z ss ∧ ∀ t ∈ as ++ bs, Feeds Cd t r) ∧
    ∀ i, i < Cd.dag.length → cls i = .out →
      (∃ as bs, ProductOf L Cd cls i as bs) ∨
        ∃ z as bs ss, Cd.unit z = Cd.unit i ∧ cls z = .out ∧ ProductOf L Cd cls z as bs ∧ DequantOf dq Cd cls i z ss

/-- **A tied window meets what useful-work credit needs of its circuit** (ASK-71): the tie, in any form, with every
drawn unit's served product value-checked at the Program's public dequant `dq` (`ValueChecked`). It is necessary, not
sufficient: no credit statement reads it yet, and it is of the declared circuit, not of the prover's kernel. No gate
checks it yet (SERVE-GATE's, #2731, has no such check). -/
def CreditTie {ι : Type} (L : Layout) (F : Form) (dq : List ℤ → ℤ) (Cd : Declared ι) (cls : ℕ → Cls) : Prop :=
  Tie L F Cd cls ∧ ValueChecked L dq Cd cls

/-- **The served Llama-3.1-8B window's shape**: 128 linear calls a step, 86,016 units a step, 1,548,288 a window,
2,304 rounds in `perCall`, and `llama31_8b` well formed. -/
def Llama31Shape : Prop :=
  llama31_8b.calls = 128 ∧ llama31_8b.perStep = 86016 ∧ llama31_8b.population = 1548288 ∧
    Form.rounds llama31_8b .perCall = 2304 ∧ llama31_8b.Wf

/-- **An epoch of `W` served windows of one salt is tied**: each window's declared circuit is layout `L`'s in form `F`
(`Tie`, which fixes its live rounds at `F.rounds L`), and each window's timed run
starts at the salt's release `release` (`Pouw.TimedRun.ts`; the service's `salted_at`, `sampled_proofs.service`), so
every window of the salt is accounted in one `(t_s, t_d]`. One salt spans an epoch's windows (`SaltWindows`,
`PouwCheck.epochSalt`), and a call's noise key reads that salt and its call index only, so a later window's noise can
be computed once the salt is out; that work is spent after `release`, inside the interval, so it is counted work. A
window timed from its own start after `release` under the same salt would leave what it spends from `release` to that
start uncounted (`Properties.Served.spentIn_split`), and that reopens the precompute. For an actual circuit's run
(`Accounting.Actual.timedRun`, whose `ts` is `CM_A`'s `tₛ`), `salted` says every window of the salt is costed in
`CM_A` at the one `tₛ`. -/
structure EpochTie {ι Q R S : Type} (L : Layout) (F : Form) (W : ℕ) (Cd : ℕ → Declared ι) (cls : ℕ → ℕ → Cls)
    (release : ℝ) (run : ℕ → Pouw.TimedRun Q R S) : Prop where
  tie : ∀ w, w < W → Tie L F (Cd w) (cls w)
  salted : ∀ w, w < W → (run w).ts = release

/-- **The served Llama-3.1-8B window's live rounds**: 18 in each form of a round a step and 2,304 in `perCall`.
Compose's bound reads the largest round's share of the words (`CircuitWork`'s `∀ j, stageShare … j ≤ wmax`), not the
count. The count enters only because the shares sum to `1`, so some share is at least `1/M`, and `1/M` bounds every
share only when the rounds are equal. RT-INT-1's 10,552 rounds for `ε < 1%` (`eps.py`, and r20261019-055920-6d6f's
`rounds_floor_live_scope`, a statement not on main) are equal rounds. `perCall`'s are not: gate_up's call (`K` 4096,
`N` 28,672) holds 1.687% of a step's words, 2.16 times an equal 1/128. So `perCall` is at `ε ≥ 4.80%` at 18 steps
and needs at least 178 steps (22,784 rounds) for `ε < 1%` (RT2-INT-2, from `CircuitWork`'s premises at
`ε_♭ = 0.571%`, its share lemma checked by r20261019-112735-692e and not stated here). No 18-step form reaches 1%. -/
def Llama31Rounds : Prop :=
  (∀ F : Form, F ≠ .perCall → Form.rounds llama31_8b F = 18) ∧ Form.rounds llama31_8b .perCall = 2304

end Accounting.Served
