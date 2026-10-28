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
while W2 proves 4 × 8 ones; the layout takes the size as a parameter, and the receipts must be per proof unit.

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
  covers the call's A rows, its coin noises the call's GEMM, its served `y = dequant(z)` goes under round `c + 1`'s
  root, as each unit's own output words, so the Commit's `y` tree is the units' (one tree, compared by root), and the
  call's receipts under a later round's root, from `c + 1`'s to the draw's. Its `y` reads the coin (`Tie.noised`,
  reachability: the coin is in `y`'s cone), which is what puts Compose's cut of `y` after the coin; `h1tile`'s `z`, its own dot, reads none, and Compose's cut would
  then put `y` before the coin its GEMM needs. It was the schedule a path whose served product is the noised one
  needs, which credit no longer asks (below).

Integrity of the Commit's `y` is its served partition's sampled units' (proofs' framing, not read here), so it never
reads the form. The link from that `y` to the proved product does (`Form.ServedLink`): `beside` has none (no unit
reads the served `y`); `zTree` outside `ValueChecked` has it only under a premise on the prover's fused kernel, that
the `y` it serves is `dequant` of the `z` it commits, which nothing checks and the design rules exclude (nothing the
prover runs is trusted); `yOpen` has it from the tie, whose `linked` field puts a reader of the Commit's `y` in every
receipt unit (a read, not a dependence), and the Program's check, with no premise; `perCall` has it from the tie and
the unit's own `dequant(z)`, under the run-time check that the Commit's `y` root of the call is the root of those
output words. Under `ValueChecked` (below) a `zTree` unit's `z` and `y` are the words it computes, or its record: the
`z` and `y` the fused kernel commits under its round's root, run inputs of the unit (`PouwH1TileZy_v1`'s `zy` port,
`Tie.outWords`), which a `check` word of the unit (its `ok`) holds to the unit's product and its `dq`. Its link is
then `perCall`'s run-time `y`-root check, and `zTree` needs no premise on the kernel: what the kernel serves is held,
on every drawn unit, to what its proof checks. The input side is ROOTS1's run-time check: each call's A table root is
the Commit's root of its linear's input. Without it a credited product could be over an A that isn't the
served input, and on the drawn units the served `y` would be `dq` of that A's product, the forward pass broken at the
linear with nothing to see it.

The link is an equality of values, which useful-work credit needs and which doesn't give it. Useful-work credit
means the credited noised work was done on the operands `A`, `B` of a served product that is
value-checked on every drawn unit, `z_u = A_u B_uᵀ` proved and committed before the draw coin, with the served `y`
computed from that committed `z_u`. `ValueChecked` states it of the circuit, and `CreditTie` is `Tie` with it. Each
receipt's unit has an `outY` word `y` whose value is the Program's public dequant `dq` of an `outZ` word `z` of the unit
and the call's two scales, `s_a` with the A row and `s_w` registered (`DequantOf`), and every `outZ` word's value, on
every typed input, is the dot of its call's committed A row with the registered B column over the linear's whole `K`
(`ProductOf`), no two of them with one value. `z` and `y` have their own classes, so a served `y` is
never a raw product, and a scale is no product's operand. Both are function-of statements:
a `y` that reads `z` and takes another product's value (`y = 0·z + 1·p`) fails `DequantOf`. On a unit
with `check` words both read the inputs on which its checks hold (`Held`), so a record `z` or `y` meets them when a
check of its unit holds it to the dot or to `dq`, and `ValueChecked` asks that the records alone can meet every check
on every input, so that no check no record meets holds them vacuously, and that a `check`, `outZ` or `outY` word read
no other unit's computed word, which its unit's proof would see as that unit's commitment. The
receipt reads `z`'s operands (`Feeds`), which is reachability; that it is their noised product is its public
Program's. The credited work is computation on the served operands, not necessarily the computation that produced the
served bytes: a prover may serve from a plain GEMM beside the noised one. No check can tell a `z` derived from the noised accumulators (kernels'
`z = O − X cᵀ − b Bᵀ + 256·xa Bᵀ`, `X = A + b`, `xa` its wrap) from one computed beside them, so
`ValueChecked` asks neither: a `z` of either kind meets it if its value is `A_u B_uᵀ`'s. `CreditTie` reads no form:
`zTree` meets it at one round a step (`Properties.Served.witCreditTie`), so credit needs neither `perCall`'s schedule
nor its rounds, and with a checked record (`Properties.Served.witCheckedCredit`), and `perCall` meets it with a `z`
peeled from the noised product (`Properties.Served.witCallCredit`); `beside` and `yOpen` have no `outZ` or `outY`
word, and a window of theirs with a receipt fails it (`Properties.Served.wit_notValueChecked`). Where `z` is committed
is the form's: under its round's root before the coin in `zTree`, where it reads no coin, computed or as the unit's
record; with the receipts in `perCall`, where `Tie.noised` asks its cone, an `outZ` word's,
to hold the call's coin, which a peeled `z` meets and a dot beside the noised product doesn't (its cut is then its own
round, which a commit after the coin is past). No credit statement is stated here. Today's code computes `y` as
`dequant` of the exact s32 product (`int8_ops.scaled_mm` at 14e4c23a: `RN_bf16(RN32(RN32(RN32(z)·s_a)·s_w))`, the `dq` a
Definition would fix), so `yOpen`'s check and `DequantOf` would hold of its `y` once its units commit those words, which
round 4 doesn't do.

Between the forms, `LayerCommitted` gains one committed class: at its step's round in `zTree` and `yOpen` (`z`, a
computed word that reads no coin or a run input only its own call and unit read, and under `ValueChecked` its `y`, or
the Commit's `y`, a run input only its own call reads), and at the round after its call's in `perCall` (`y`, a
computed word that reads the call's coin, and under `ValueChecked` its `z`). The round root gains one table root
(`z`'s or `y`'s; both under `ValueChecked`), and the units keep their number, each gaining its tile's `z` or `y`
words. A unit that checks a record also commits its `check` words, with its receipts (`LayerCommitted`'s second case),
and reads the public inputs (`pub`: its position, its round's roots), each committed before the coin of every round
that reads it (the first case).

`commitCoin` is the coin before which the served code commits a value of each class (`Cls`) of round `r`, as the
served code is to do it: the registration (the weights' root over every linear's root(B)) before the salt coin and so
before round `0`'s; a call's A rows and their scales, the request entries its round admits and its `z` or `y` rows
under its round's root, before its coin (`perCall`'s `z` and `y` under the next round's); its receipts under the root
of any later round, up to the draw's: the next round's at the earliest (`commitCoin`'s), round `r + D − 1`'s in a ring
of depth `D`, and the last rounds' in the manifest before the draw coin (`Tie.receipt`), and a unit's check words
with them (`Tie.checked`); a public input before the coin of each round that reads it (`Tie.pub`). Every served arm is
to commit in this order, each round's coin live and drawn after its root.

A receipt's commit coin is a range because only gates of its own unit read it (`Tie.readers`), and so is a check
word's. It is then in its own
unit's late cone and in no other unit's, so committing it after later coins leaves no other unit's proof reading a
value chosen after that unit's noise, and no reader whose skip goes uncharged: its own unit's proof checks it when the
unit is drawn, and the unit's cone pays for it when it isn't. The draw coin still follows every commitment. That
is `Accounting.Compose.LayerCommitted`'s second case. The ring hides the live coins' round trip off the prover's node,
about 100 ms, behind `D` steps: about 104 ms a token at `D = 2` (the next round's root), 0.09 ms at `D = 6`
(`Properties.Served.witTieRing`). An A row gets no such range: written after its reader's coin, `A = −b` cancels a
one-bit dither. An arm that forms a call's A rows
after the coin or salt its noise reads is in no form here: dither-a, dither-one and `serve.py`'s seeded mode do today,
and dither-a is to move to this order. A request entry is a run input (`act`), not a registered one:
certified serving releases a window's requests after its salt frame, so no queue root can precede the salt.
Today's code commits the receipts at the draw (`serve.py`'s receipt trees, built after serving), which this schedule is
the fix of, and its round root covers the A and admit roots only. The admit words and the tally have no class: their
always-proved units are not in the per-call Program `Properties.Served.ServedUnits` names, so no gate of a served
circuit is one. They are PADCREDIT's (R8, R9), and the round root's admit root stays a run-time check below.

`Tie` is what a declared circuit (`Accounting.Compose.Declared`) must be to be the served window's: its call sites and
rounds the layout's, each value's commit coin `commitCoin` of its class (a receipt's any coin after its round's, by
the draw), and the IR facts that put each class's value at the cut `commitCoin` names (a weight registered, an A row or
a `y` row a run input read only by its own call, a record a run input read only by its own call and unit, a public
input committed before every coin that reads it, a receipt reading its round's coin and read only by its own unit, a
check word read only by its own unit).
`TieCommits` states that the tie gives `Accounting.Compose.LayerCommitted`, and `TieLive` that it gives
`Declared.Live` under any scheme: the two `Scope` conjuncts that are protocol structure. The units, the population and
the draw, which read the verifier's partition of each call's Program, are `Properties.Served`'s. `EpochTie` ties an epoch's windows: each one's `Tie`, and each timed from the one salt's release,
`Pouw.TimedRun.ts`, the start of the interval `Pouw.Guarantees.WorkBetweenSaltAndDeadline` counts work in. `CreditTie`
is `Tie` with what useful-work credit needs of the circuit (`ValueChecked`): a necessary condition, which no credit
statement reads yet.

What the verifier must check at run time for each class, so that a `Declared` meeting `Tie` is the run's (the
statement is over the declared circuit; these are the checks that make its `commitRound` the code's):
* `registered`: the salt coin opened against the registration root, recorded before every round's coin; the weights'
  root equal to the served model's published one; each drawn unit's B rows opened under its linear's root(B), and
  root(B) under the weights' root;
* `act`, `outZ`, `outY`, `served`: round `r`'s root, recorded before round `r`'s coin, over the A table, the request
  entries the round admits, admit (PADCREDIT's), and `z` or Commit `y` roots of exactly round `r`'s calls (`perCall`'s
  `y` roots in round `r + 1`'s; both under `ValueChecked`); each drawn unit's rows of them opened under it; in
  `perCall`, and in any form under `ValueChecked`, each call's Commit `y` root equal to its units' `y` root; and each
  call's A table root equal to the Commit's root of its linear's input (ROOTS1), so the A rows a call
  commits are the linear's input as the Commit commits it. Layer 0's input alike: round `r`'s input-ids table (slot
  i's input token id, `ids@<r>`) under round `r`'s root before its coin, its root, domain and leaf count the Commit's
  `input-ids/<r>`, so the token ids the Commit's first layer reads are the ones fixed under that root; round 0's
  input-ids table also inside round 0's request table, under the window's open root before the salt coin
  (`credit.requests_root`), its domain binding the window's session in place of that root; the Commit
  record's digest under the manifest's `calls_root`, before the draw coin (`credit._arow`). In `zTree` a unit's record (`PouwH1TileZy_v1`'s `zy`, which
  its proof reads at a committed table, `--registered`) is among them: its call's record table (h1_prove's `zy@<c>`),
  under round `r`'s root before its coin;
* the challenger: every round's coin opened by the live challenger, the auditor's device (AGENTS.md, auditor-trusted
  devices; `trusted-host`), on a machine the prover doesn't control and never a process on the prover's node.
  A challenger in the prover's process gives the prover its coins as they are drawn, so its coins are seeded ones,
  whatever the record calls them. `serve.py` builds `live_challenger` in the prover's process today
  (`window.py`'s `entropy=os.urandom`), so its runs establish only the order of roots and coins;
* `receipt`: round `r`'s receipt roots under the root of one later round, recorded before that round's coin (round
  `r + D − 1`'s in a ring of depth `D`), or under the manifest before the draw coin, at the commit coin the declared
  circuit gives them; each drawn unit's receipt opened under it;
* `check`: each drawn unit's check words `1` in its accepted proof, a row of known content (`PouwH1TileZy_v1`'s `ok`,
  the one word `1` under the public `ok_salt`), committed with the unit's receipts;
* `pub`: the unit's position `base s w + u`, computed by the verifier from the layout, and its round's roots, from the
  recorded round log, each recorded before the coin of every round whose units read it, never read from the prover's
  record (`PouwH1TileZy_v1`'s `pub` row, whose salt is public, so the verifier computes it); the unit's salt
  `window.round_key(commitment, coin_r, r, root_r)`'s stream, derived by the verifier from the challenger's opening of
  round `r`'s coin and the recorded root, never read from the record (`serve.py`'s reference recomputation reads the
  record's `salt` today, and `h1_prove`'s units PoUW's window salt `pouw_salt`, which no round's coin keys);
* every window of a salt timed from the salt's release (`EpochTie.salted`): the challenger's one `salted_at` an epoch,
  never a window's own start;
* the window's `F.rounds L` live rounds (`Llama31Rounds`). Compose's bound reads the largest round's share of the
  words, not the count, which is `1/F.rounds L` only when the rounds are equal (`Llama31Rounds` has the numbers);
* the draw: `K_d` units of `population` from the draw coin opened after the manifest;
* for credit (`CreditTie`), `ValueChecked` of each call's compiled Program at the Program's `dq`, once per Program
  as `ServedUnits` is. No gate checks it yet: SERVE-GATE's has no such check. A fact of the Program's IR, it
  says nothing of the prover's kernel, which may serve from a GEMM beside the noised one; the credited work is the
  receipts' noised GEMM on the served operands. `ProductOf` and `DequantOf` bear on a run whose inputs are typed
  (`Gates.Typed`): each drawn unit's opened input word in the declared range of every input gate of its slot.

A checked record asks nothing more of this statement: `Tie`, `ValueChecked` and `CreditTie` take no premise on it. What
a proof layer (`Accounting.TS.ProofLayer`) realizing a value-checked `zTree` window must supply at a unit's record port
is that realization's to prove, at the premises `Accounting.Guarantees.TS.SessionSoundAt` takes
(`Proofs.Accounting.TS.Session.Assumptions`):
* `pf.accepted` holds of a proof only where its unit's check words are `1`: flock-verify `--zk --registered` reads the
  `ok` row at its known content (`AcceptedLive`);
* `pf.rel` holds its record to the product and its `dq`, as `Held` reads them, and the layer pins each committed value's
  derivation (`ProofLayer.Pins`; `ServedChecksHold` at the session's value layer). The unit's Definition with its check
  words fixed to `1` (`PouwH1TileZy_v1` with `ok = 1`) is `Accounting.Guarantees.TS.TileSessionSound`'s relation
  (`TileRel`), under its premise `Accounting.TS.Tile.TileComputes` (the circuit computes its Definition at its
  registered reads; that the Definition's `ok = 1` gives the cells' checks is proved, at a `K` its EXACT binds at).
  `CheckedSessionSound`'s relation (`CheckedRel`) is only that the opened rows are an assignment the circuit's
  evaluation agrees with, which says nothing of the product or `dq`;
* `pf.opened` holds the leaves of the round's record tables, the records committed under its root before σ_w;
* `ServedRootsLinked`'s `rows` read the unit's `zy` and A ports at its round root's tables, which
  `Flock.SecurityProofs.ServedRowsLinked` gives at the drawn units under `RowsSession.ServedRegistered`.

A leaf proved as several statements is one unit: option A's 33 statements (32 lanes
`PouwH1LaneZy_v1` and their combine `PouwH1Combine_v1`) are one `Cd` unit, the leaf's, and the lanes' partials are
values of the proof layer that realizes it, not gates of `Cd`. They take no class, `ValueChecked`'s fifth clause ranges
over the unit's records only, and no gate of the unit reads another unit's computed word through them. That the 33
statements, accepted over shared committed values, realize the one unit's checks is that linked realization's to
prove: BRIDGE2's `Accounting.Guarantees.TS.LeafLanesSound`, from `CheckedSessionSound` at each statement, under its
premises `LaneLowers` and `CombineLowers` (each statement's pieces at their lowerings, whose composition computing its
Definition is proved), with the leaf and the
receipt the ones the statements' committed rows decode to (the known ports' are checked, and each A, B and partials
row is one row of one registered tree that every statement reading it reads in full).
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

/-- **The served Llama-3.1-8B window**: 18 steps, 64 slot rows a call, 32 × 32
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

/-- The rounds past its own after which form `F` commits an `outZ` or `outY` word: one in `perCall`, whose `y` reads its
round's coin, none in the others. -/
def Form.outLag : Form → ℕ
  | .perCall => 1
  | _ => 0

/-- **A served value's class**:
* `registered`: a row of a linear's B table, or its scale, under the weights' root, the registration root;
* `act`: a row of a call's A table, its slot's s8 row and scale (the empty row in a slot no request is assigned to),
  or a request entry its round admits: a run input under its round's root. Certified serving releases a window's
  requests after its salt frame, so a request entry can't be committed before the salt;
* `pub`: a public input of a unit's Program that no request or product brings, which the verifier derives (the unit's
  position, from the layout; its round's roots, from the round log): a run input committed before the coin of every
  round whose gates read it (`Tie.pub`), never a word Compose credits (`isWord`), and no product's operand or scale,
  since `ProductOf` and `DequantOf` take `act` and `registered` words only. An H1 unit's `position` and `roots` ports
  are of it. Its salt is not: the salt is its round's `coin`, drawn after the root it is keyed by;
* `receipt`: a receipt word of a call's unit;
* `check`: a word that its unit's proof holds to `1` (`PouwH1TileZy_v1`'s `ok`, which the verifier reads through a row
  of known content): computed, committed after its round's coin and by the draw (`Tie.checked`), and read by no gate
  of another unit. `ProductOf`, `DequantOf` and `ValueChecked`'s distinctness read the inputs on which every check
  word of the unit is `1` (`Held`). Under `ValueChecked` it reads no computed gate of another unit, directly or
  through computed gates, and neither does an `outZ` or `outY` word;
* `outZ`: a word of the exact product `z` of a call's unit, committed in a row table (`zTree`, and under `ValueChecked`
  `perCall`): a word the unit computes, or in `zTree` a word of the unit's record, the run input the served kernel
  commits under its round's root, which a `check` word of the unit holds to the unit's product (`Tie.outWords`);
* `outY`: a word of the served `y = dequant(z)` of a call's unit, committed in a row table, the words the Commit's `y`
  root is over (`perCall`, and under `ValueChecked` `zTree`): computed by the unit, or in `zTree` a word of its record,
  as `outZ`. Under `ValueChecked` each is `dq` of an `outZ` word, never a raw product;
* `served`: a word of the Commit's served output `y` of a call, which its unit opens (`yOpen` only);
* `inner`: a value no opening reads, a proof unit's own wire (`z` in `beside` and `yOpen`, and the unit's own `z` and
  `y` beside a record it checks);
* `coin`: a round's coin input, the unit's salt port.
A call's admit words have no class: their always-proved units are outside the per-call Program (the module
docstring). -/
inductive Cls
  | registered
  | act
  | pub
  | receipt
  | check
  | outZ
  | outY
  | served
  | inner
  | coin
  deriving DecidableEq, Repr

/-- **The coin before which the served code commits a value** of class `k` of round `r` in form `F`, of the window's
`F.rounds L` rounds (the draw is the last): the registration before round `0`'s; A rows, `zTree`'s `z` and `y` (words
the unit computes or its record's) and the Commit's `y` before their round's; `perCall`'s `z` and `y`, which read their
round's coin, before the next round's. A receipt's is the earliest it may be committed at, the next round's:
`Tie.receipt` lets it go later, by the draw. A check word's is the next round's, beside its unit's receipts:
`Tie.checked` lets it go anywhere by the draw. A public input's is at the latest its round's: `Tie.pub` asks instead
that it precede the coin of each round that reads it. `inner` and `coin` values are committed nowhere (`Tie.inner` sets
an inner value's commit coin to its cut; a coin is no value the prover writes). -/
def commitCoin (F : Form) : Cls → ℕ → ℕ
  | .registered, _ => 0
  | .act, r => r
  | .pub, r => r
  | .outZ, r => r + F.outLag
  | .outY, r => r + F.outLag
  | .served, r => r
  | .receipt, r => r + 1
  | .check, r => r + 1
  | .inner, _ => 0
  | .coin, _ => 0

/-- **What ties receipt `i`'s unit to the served output**, by form: nothing in `beside`; in `zTree` and `perCall`, an
`outZ` or `outY` word of the unit (`z`, or the served `y`, computed or in `zTree` its record's); in `yOpen`, a gate of
the unit that reads a word of the Commit's `y`. A read, not a dependence: what ties the word's value to the product is
`ValueChecked`. -/
def Form.Links {ι : Type} (Cd : Declared ι) (cls : ℕ → Cls) : Form → ℕ → Prop
  | .beside, _ => True
  | .zTree, i => ∃ j, j < Cd.dag.length ∧ (cls j = .outZ ∨ cls j = .outY) ∧ Cd.unit j = Cd.unit i
  | .yOpen, i => ∃ g j, g < Cd.dag.length ∧ Cd.unit g = Cd.unit i ∧ Cd.Reads g j ∧ cls j = .served
  | .perCall, i => ∃ j, j < Cd.dag.length ∧ (cls j = .outZ ∨ cls j = .outY) ∧ Cd.unit j = Cd.unit i

/-- **The served `y` is the proved product's**, by form, where `fused` is the fused kernel's premise (the `y` it serves
is `dequant` of the `z` rows it commits): `beside` has no link (`False`); `zTree` has it exactly under `fused`, a
premise on code the prover runs, which no run-time check reaches; `yOpen` has it from `Tie.linked` and the receipt
Program's `dequant(z) = y` check over the opened `y` (a fact of that Program's Definition, which doesn't exist yet),
with no premise, given the run-time check that the unit's `served` words open under the Commit's `y` root: `Tie`
doesn't tell a `served` run input from an `act` one. `perCall` has it from `Tie.linked` and the unit's own
`dequant(z)` output, given the run-time check that the call's Commit `y` root is its units' `y` root. Under
`ValueChecked` a `zTree` unit commits its `y` too, a word it computes or its record's, which its `check` words hold to
`dq` of its product. Its link is then that run-time check, and `zTree` needs no premise on the kernel: `fused` is the
link of a `zTree` window that isn't value-checked, and `CreditTie` doesn't read `ServedLink`. It is an equality of
values: useful-work credit needs it, and it doesn't give that credit (`ValueChecked`, the module docstring). Integrity
reads none of it. -/
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
  (AGENTS.md, auditor-trusted devices; `trusted-host`). A challenger in the prover's process gives seeded coins,
  however `coinMode` records them, so a `Declared` with `.live` rounds is the run's only under the module docstring's
  challenger check, which today's `serve.py` doesn't meet;
* `coin`, `registered`, `act`, `pub`, `computed`, `outWords`: a gate is a coin input exactly when it is of class
  `coin`; a `registered` gate is a registered input; an `act` or `served` gate is a run input read only by gates of
  its own call; a `pub` gate is a run input that no credited word is (`isWord`), committed by the draw and before the
  coin of every round with a gate that reads it; a `receipt`, `check` or `inner` gate is no input; an `outZ` or
  `outY` gate is no input, or in `zTree` a run input (its unit's record) read only by gates of its own call and unit;
  so every input's class is its binding's. A record committed after its reader's coin fits no case of
  `LayerCommitted` (an input has no second case), so `perCall`'s `z` and `y`, committed the round after,
  are words the unit computes;
* `committed`: a gate that is no input is `inner` exactly when the IR doesn't mark it a committed output
  (`Gates.IntGate.commit`), so no opened value is declared inner and committed at its cut by fiat;
* `noised`: a `receipt` gate, and in `perCall` an `outZ` or `outY` gate, reads its round's coin (`coinMark` past its
  round). This is reachability: the coin input is in the gate's cone, not that the gate's value depends on the coin. It
  is what Compose's cut reads (`Declared.cutOf`), so it is what the commit order needs; that a receipt's value is the
  noised product is its public Program's (`Properties.Served.ServedUnits`);
* `forms`, `linked`: an `outZ` or `outY` gate is `zTree`'s or `perCall`'s and a `served` gate `yOpen`'s, and every
  receipt's unit has its form's link to the served output (`Form.Links`);
* `commit`, `receipt`, `checked`, `inner`: a value of an opened class other than a receipt, a check word or a public
  input is committed at `commitCoin` of its form, class and round; a receipt after its round's coin and by the draw,
  under any later round's root or the manifest (a ring of depth `D` puts step `s`'s under round `s + D − 1`'s); a
  check word by the draw; an inner value at its own cut;
* `readers`: only gates of a receipt's or a check word's own unit read it, so it is in no other unit's late cone, and
  committing it after later coins leaves no reader in another unit (the module docstring; Compose's
  `LayerCommitted`, second case).
The `commit`, `receipt` and `inner` fields are how the declared circuit is built from the layout; the others are facts
of each call's Program, which no code checks yet (an extractor building `Cd` from the compiled Programs is to); the
commit coins are the code's only if the verifier checks the run's round log (the module docstring's list). -/
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
  pub : ∀ i, i < Cd.dag.length → cls i = .pub →
    Cd.binding i = some .run ∧ (Cd.dag[i]?).any isWord = false ∧ Cd.commitRound i ≤ Cd.rounds ∧
      ∀ j, j < Cd.dag.length → Cd.Reads j i → Cd.commitRound i ≤ Cd.round (Cd.call j)
  computed : ∀ i, i < Cd.dag.length →
    (cls i = .receipt ∨ cls i = .check ∨ cls i = .inner) → Cd.binding i = none
  outWords : ∀ i, i < Cd.dag.length → (cls i = .outZ ∨ cls i = .outY) →
    Cd.binding i = none ∨ (F = .zTree ∧ Cd.binding i = some .run ∧
      ∀ j, j < Cd.dag.length → Cd.Reads j i → Cd.call j = Cd.call i ∧ Cd.unit j = Cd.unit i)
  committed : ∀ i, i < Cd.dag.length → Cd.binding i = none →
    (cls i = .inner ↔ (Cd.dag[i]?).any (fun g => g.commit) = false)
  noised : ∀ i, i < Cd.dag.length → (cls i = .receipt ∨ ((cls i = .outZ ∨ cls i = .outY) ∧ F = .perCall)) →
    Cd.round (Cd.call i) + 1 ≤ (coinMark Cd).getD i 0
  forms : ∀ i, i < Cd.dag.length → ((cls i = .outZ ∨ cls i = .outY) → F = .zTree ∨ F = .perCall) ∧
    (cls i = .served → F = .yOpen)
  linked : ∀ i, i < Cd.dag.length → cls i = .receipt → F.Links Cd cls i
  commit : ∀ i, i < Cd.dag.length → cls i ≠ .inner → cls i ≠ .coin → cls i ≠ .receipt → cls i ≠ .check →
    cls i ≠ .pub → Cd.commitRound i = commitCoin F (cls i) (F.roundOf L (Cd.call i))
  receipt : ∀ i, i < Cd.dag.length → cls i = .receipt →
    F.roundOf L (Cd.call i) + 1 ≤ Cd.commitRound i ∧ Cd.commitRound i ≤ Cd.rounds
  checked : ∀ i, i < Cd.dag.length → cls i = .check → Cd.commitRound i ≤ Cd.rounds
  readers : ∀ i, i < Cd.dag.length → (cls i = .receipt ∨ cls i = .check) →
    ∀ j, j < Cd.dag.length → Cd.Reads j i → Cd.unit j = Cd.unit i
  inner : ∀ i, i < Cd.dag.length → cls i = .inner → Cd.commitRound i = Cd.cutOf i

/-- **The tie gives the service's commit order**: a declared circuit that is a served window, in any form, meets
`LayerCommitted`. Every commit coin but a receipt's is then no later than its IR cut (the first case): a weight's `0`;
an A row's or a `y` row's, its round, its first reader's; a `zTree` `z` word's, its round, the floor of its cut;
`perCall`'s `z` or `y` word's, the round after its own, which its coin read reaches; an inner value's, its cut. A
receipt is a computed value only its own unit reads, committed by the draw (the second case). -/
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

/-- **Gate `i` is in gate `j`'s computed cone**: `j` is `i`, or `j` is a computed gate (no input: `Declared.binding` is
`none`) that reads a computed gate of the circuit whose computed cone holds `i`. Inputs stop the walk: no input is in
another gate's computed cone, and an input's is itself, whatever its operand list (an input's value is its slot's). -/
def ComputedFeeds {ι : Type} (Cd : Declared ι) (i j : ℕ) : Prop :=
  Relation.ReflTransGen
    (fun a b => Cd.Reads b a ∧ a < Cd.dag.length ∧ Cd.binding a = none ∧ Cd.binding b = none) i j

/-- Gate `i` is declared an s8 word, `[−128, 127]`: the type of an A row's and a B row's words. -/
def S8 {ι : Type} (Cd : Declared ι) (i : ℕ) : Prop :=
  (Cd.dag[i]?).any (fun g => decide (g.lo = -128 ∧ g.hi = 127)) = true

/-- **Unit `u`'s checks hold on input `x`**: every `check` word of the unit is `1`, as its proof holds it when the unit
is drawn (an accepted proof of its Program, whose `ok` row the verifier reads at known content). With no check word in
the unit it holds of every input. -/
def Held {ι : Type} (Cd : Declared ι) (cls : ℕ → Cls) (u : ℕ) (x : ι → ℤ) : Prop :=
  ∀ k, k < Cd.dag.length → cls k = .check → Cd.unit k = u → Gates.val Cd.dag x k = 1

/-- **Gate `z` is the product of its call's committed A row and a registered B column** (the value check): its
operands `as`, run inputs of `z`'s call (`act`), and `bs` (`registered`), each declared an s8 word (`S8`), are as many
as its linear's `K`, each list without repeats, and on every typed input (`Gates.Typed`: each input slot a word of its
type) on which its unit's checks hold (`Held`) `z`'s value is `Σ_k a_k·b_k`, a coordinate of `A_u B_uᵀ`. A function-of
statement: it holds of any gates with that value (`h1tile`'s dot, cuBLASLt's, or a peel of the noised accumulators if
its corrections cancel the noise on every typed input), of a record `z` that a check word of its unit holds to such a
dot, and of no `z` whose value differs from it on some typed input its unit's checks hold on. A constant check, or one
with an escape (`ok = (z = Σ) ∨ e`), holds on an input with any record, and fails it
(`Properties.Served.witConst_notValueChecked`, `witHatch_notValueChecked`). `S8` keeps the typed inputs the served
rows' words: an operand declared `[0, 0]` would leave only `a = 0` typed. `Nodup` keeps the `K`
products distinct: an A word read `K` times would give `a·Σ_k b_k`, one product's work credited as `K`. Which row and
column `as` and `bs` are is the call's Program's (`Properties.Served.ServedUnits`). -/
def ProductOf {ι : Type} (L : Layout) (Cd : Declared ι) (cls : ℕ → Cls) (z : ℕ) (as bs : List ℕ) : Prop :=
  as.length = L.kOf (L.linearOf (Cd.call z)) ∧ bs.length = as.length ∧ as.Nodup ∧ bs.Nodup ∧
    (∀ a ∈ as, a < Cd.dag.length ∧ cls a = .act ∧ Cd.call a = Cd.call z ∧ S8 Cd a) ∧
    (∀ b ∈ bs, b < Cd.dag.length ∧ cls b = .registered ∧ S8 Cd b) ∧
    ∀ x, Gates.Typed Cd.dag x → Held Cd cls (Cd.unit z) x →
      Gates.val Cd.dag x z = (List.zipWith (· * ·) (as.map (Gates.val Cd.dag x)) (bs.map (Gates.val Cd.dag x))).sum

/-- **Gate `y` is the public dequant `dq` of gate `z` at the scales `sa`, `sw`**: `sa` is a run input of `z`'s call
(`act`, so under its round's root before its coin, as the A row's scale is) and `sw` a registered word (under the
weights' root), neither declared an s8 word (`S8`), and on every typed input on which `y`'s unit's checks hold
(`Held`) `y`'s value is `dq` of `z`'s, `sa`'s and `sw`'s, so no other value reaches `y`'s: a word `y` computes, or a
record `y` a check word of its unit holds to it. Since `ProductOf` declares every operand s8, no A or B word of a
product is a scale (`y = dq(z, A_1, B_1)` fails it). Which of the call's other run inputs is its row's
scale is the call's Program's (`Properties.Served.ServedUnits`), as which row and column a product's operands are. A
function-of statement: a `y` that reads `z` and takes another product's value (`y = 0·z + 1·p`) fails it
unless that value is `dq` of `z`'s on every typed input. It is `dq` at every `outY` word, an empty slot's included,
with no case for zero: an empty slot's `s_a` is `+0` and its `y` is `dq(0, +0, s_w)`, which is `+0` only at a finite
positive `s_w` (0x8000 at a negative one, 0x7FFF at an infinite or NaN one). That `dq` is the dequant
the served code computes is the Definition's review. -/
def DequantOf {ι : Type} (dq : List ℤ → ℤ) (Cd : Declared ι) (cls : ℕ → Cls) (y z sa sw : ℕ) : Prop :=
  sa < Cd.dag.length ∧ cls sa = .act ∧ Cd.call sa = Cd.call z ∧ ¬ S8 Cd sa ∧
    sw < Cd.dag.length ∧ cls sw = .registered ∧ ¬ S8 Cd sw ∧
    ∀ x, Gates.Typed Cd.dag x → Held Cd cls (Cd.unit y) x →
      Gates.val Cd.dag x y = dq [Gates.val Cd.dag x z, Gates.val Cd.dag x sa, Gates.val Cd.dag x sw]

/-- **Every drawn unit's served product is value-checked**, what useful-work credit
needs of the circuit. With `ops z` the operands of `outZ` word `z` and `src y` the `outZ` word and scales `outY` word
`y` is dequantized from:
* every `outZ` word is the product of its call's A row and a B column (`ProductOf`);
* no two `outZ` words have one value on every typed input on which both their units' checks hold, so each credited
  unit has its own products, and a unit's receipt is credited for no product another unit's is (a
  `z' = 0 + 1·z` in a second unit fails it);
* every `outY` word is `dq` of an `outZ` word of its unit at its scales (`DequantOf`), so the served `y` words, the
  Commit's under the run-time `y`-root check, are each `dq` of a value-checked `z` and never a raw product;
* each receipt `r`'s unit has an `outY` word, which with its `outZ` word is committed before the draw coin, and `r`
  reads each of that `z`'s operands (`Feeds`);
* the checks can be met: on every typed input some typed input that differs from it only in the `outZ` and `outY`
  records holds every unit's checks. A check no record meets (`ok = 0`, or `ok` reading a coin or an A word to `1`
  on some inputs only) would hold on no input, or on too few, and make `ProductOf` and `DequantOf` hold vacuously;
  this keeps them about every run. Since they then fix each record on such an input to its unit's `Σ` and `dq`, the
  records that meet the checks are those values, which the served code has before its round's coin;
* every computed gate that a `check`, `outZ` or `outY` word reads, directly or through computed gates
  (`ComputedFeeds`), is of the word's unit. `Held`, `ProductOf` and `DequantOf` evaluate every gate
  honestly, but a unit's proof sees another unit's computed word as whatever that unit's prover committed, which
  nothing checks when that unit isn't drawn, and which `Declared.Late` doesn't charge, being committed before the
  reader's coin. A check `ok₀ = (z₀ = Σ₀ ∧ y₀ = dq₀) ∨ z₁ ≠ Σ₁′` over another unit's computed `z₁`, `Σ₁′` its own
  recomputation of it, never escapes under `Gates.val` and holds with any record at run time, and so does a computed
  `z₀ = Σ₀ + (z₁ − Σ₁′)` (`Properties.Served.witConeCheck_notValueChecked`, `witConeZ_notValueChecked`). Inputs stop
  the walk: a record is read only in its own unit (`Tie.outWords`), an A row or a weight is opened against its root,
  a public input is the verifier's and a coin the challenger's, each the same word in every unit that reads it.
`ProductOf` and `DequantOf` are function-of statements, and the distinct values are too: two `outZ` words over the
same operand pairs, in any order, have one value. A record is an input that its unit's proof holds to its product, so
`zTree`'s fused kernel, which commits its `z` and `y` and serves the `y`, is held to them on every drawn unit with no
premise on its code (`Properties.Served.witCheckedCredit`), and a check no record meets fails the last clause
(`Properties.Served.witNever_notValueChecked`). `Feeds` is reachability: `z`'s operands are in the receipt's cone, which
doesn't say the receipt is their noised product; that is its public Program's (`Properties.Served.ServedUnits`).
The credited work is the receipts' noised GEMM on the served operands, not necessarily the computation that produced
the served bytes: a prover may serve from a plain GEMM beside the noised one, which no check can tell from a `z` peeled
from the noised accumulators, and this asks neither. `dq` is the Program's public dequant,
`RN_bf16(RN32(RN32(RN32(z)·s_a)·s_w))` in today's code (`int8_ops.scaled_mm` at 14e4c23a). -/
def ValueChecked {ι : Type} (L : Layout) (dq : List ℤ → ℤ) (Cd : Declared ι) (cls : ℕ → Cls) : Prop :=
  ∃ (ops : ℕ → List ℕ × List ℕ) (src : ℕ → ℕ × ℕ × ℕ),
    (∀ z, z < Cd.dag.length → cls z = .outZ → ProductOf L Cd cls z (ops z).1 (ops z).2) ∧
    (∀ z z', z < Cd.dag.length → z' < Cd.dag.length → cls z = .outZ → cls z' = .outZ →
      (∀ x, Gates.Typed Cd.dag x → Held Cd cls (Cd.unit z) x → Held Cd cls (Cd.unit z') x →
        Gates.val Cd.dag x z = Gates.val Cd.dag x z') → z = z') ∧
    (∀ y, y < Cd.dag.length → cls y = .outY →
      (src y).1 < Cd.dag.length ∧ cls (src y).1 = .outZ ∧ Cd.unit (src y).1 = Cd.unit y ∧
        DequantOf dq Cd cls y (src y).1 (src y).2.1 (src y).2.2) ∧
    (∀ r, r < Cd.dag.length → cls r = .receipt →
      ∃ y, y < Cd.dag.length ∧ cls y = .outY ∧ Cd.unit y = Cd.unit r ∧
        Cd.commitRound (src y).1 ≤ Cd.rounds ∧ Cd.commitRound y ≤ Cd.rounds ∧
        ∀ t ∈ (ops (src y).1).1 ++ (ops (src y).1).2, Feeds Cd t r) ∧
    (∀ x, Gates.Typed Cd.dag x → ∃ x', Gates.Typed Cd.dag x' ∧ (∀ u, Held Cd cls u x') ∧
      ∀ i g b s, Cd.dag[i]? = some g → g.op = .input b s → cls i ≠ .outZ → cls i ≠ .outY → x' s = x s) ∧
    ∀ w, w < Cd.dag.length → (cls w = .check ∨ cls w = .outZ ∨ cls w = .outY) →
      ∀ g, ComputedFeeds Cd g w → Cd.unit g = Cd.unit w

/-- **A tied window meets what useful-work credit needs of its circuit**: the tie, in any form, with every
drawn unit's served product value-checked at the Program's public dequant `dq` (`ValueChecked`). It is necessary, not
sufficient: no credit statement reads it yet, and it is of the declared circuit, not of the prover's kernel. No gate
checks it yet (SERVE-GATE's has no such check). -/
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
`PouwCheck.epochSalt`). Every window is in `Tie`'s live order, which every served arm is to use:
* a call's A rows and their scales go under its round's root before that round's live coin, and so do its `z` and `y`
  leaves in a form of a round a step;
* its receipts, which read the coin, follow it under a later round's root, by the draw (the next round's at the
  earliest, round `s + D − 1`'s in a ring of depth `D`), and `perCall`'s `z` and `y` under the next round's root.
A unit's noise reads its round's coin and its round's root, which is over its call's A table (the unit's salt in the
module docstring's run-time list), so no noise exists before the A rows it noises are committed. A key that read only
the salt and the call index would let a prover that holds the salt write A after its noise: `A = −b` cancels a one-bit
dither, and the receipt is then the same for every B. There is no seeded form: `Tie.live` makes
every round's coin live and `Tie.noised` has each receipt read its round's coin, so a window whose noise the salt fixes
before its A rows are committed isn't tied. What a window spends between `release` and its own start (its requests come
after the salt frame, and its A rows from them) is inside the interval. A window timed from its own start would leave
that spending outside its interval (`Properties.Served.spentIn_split`). For an actual circuit's run
(`Accounting.Actual.timedRun`, whose `ts` is `CM_A`'s `tₛ`), `salted` says every window of the salt is costed in `CM_A`
at the one `tₛ`. -/
structure EpochTie {ι Q R S : Type} (L : Layout) (F : Form) (W : ℕ) (Cd : ℕ → Declared ι) (cls : ℕ → ℕ → Cls)
    (release : ℝ) (run : ℕ → Pouw.TimedRun Q R S) : Prop where
  tie : ∀ w, w < W → Tie L F (Cd w) (cls w)
  salted : ∀ w, w < W → (run w).ts = release

/-- **The served Llama-3.1-8B window's live rounds**: 18 in each form of a round a step and 2,304 in `perCall`.
Compose's bound reads the largest round's share of the words (`CircuitWork`'s `∀ j, stageShare … j ≤ wmax`), not the
count. The count enters only because the shares sum to `1`, so some share is at least `1/M`, and `1/M` bounds every
share only when the rounds are equal. The 10,552 rounds for `ε < 1%` (`eps.py`) are equal
rounds. `perCall`'s are not: gate_up's call (`K` 4096,
`N` 28,672) holds 1.687% of a step's words, 2.16 times an equal 1/128. So `perCall` is at `ε ≥ 4.80%` at 18 steps
and needs at least 178 steps (22,784 rounds) for `ε < 1%` (from `CircuitWork`'s premises at
`ε_♭ = 0.571%`, its share lemma checked numerically and not stated here). No 18-step form reaches 1%. -/
def Llama31Rounds : Prop :=
  (∀ F : Form, F ≠ .perCall → Form.rounds llama31_8b F = 18) ∧ Form.rounds llama31_8b .perCall = 2304

end Accounting.Served
