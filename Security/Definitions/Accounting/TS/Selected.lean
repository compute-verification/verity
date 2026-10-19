import Definitions.Accounting.TS.Drawn

/-!
# A padded window's selections: credit that depends on committed values (trusted)

`DrawnLayer`'s wiring, cuts, coins and weights don't depend on the path, so its credit is static: every unit is
credited its words. A padded static window (Daniel, 18 Oct, 20:52Z) credits only its selected slots: its Program is one
circuit shaped by public parameters, a static superset of units, and which of them are credited is a value, a selection
(a mask, a forming's `ok`, a done flag) that a unit computes from committed values, or that the weights' registration
proved once, and that is committed like any value. A wrong unit can emit a wrong selection; it can't rewire. This is
the credit rule of note:20261018T2200Z-draft-padded-credit-rule (round 5, after
note:20261018T2310Z-finding-padded-credit-r3, note:20261018T2207Z-report-redteam-padded-credit and
note:proofs/20261018T2230Z-report-rt10-padded-credit), on the conditions of
note:20261018T2125Z-report-format-table-r4-r5 (C4's ✓):

* credit only selected units, a unit being selected when every selection value gating it is committed as one;
* each selection value is a unit's output, or a certificate the weights' registration proved (`SelectionsSourced`);
  in the first version every in-window selection unit is always proved (`SelectionsProved`);
* every operand a credited unit's floor depends on is written by a forming, always proved or weighted, whose piece
  chain reads the raw row one of its certificates certifies, or is the salt's epoch record of that row
  (`CertifiedOperands`), and every forming reads only its raw row, its noise and formings (`FormingClosed`), so a mask
  or a mux applied after the forming, or an input moved before it, is refused, and a wrong forming is an unsound proof,
  a weighted unit's charged cone, or the epoch's failure (`EpochFormedRight`);
* no selection is committed after the coin its unit's noise reads (`NoLateSelection`): `ok_A` is committed with the
  raw activation row, before the noise is derived, and `ok_B` with the raw weight row at registration, before the
  salt, each a function of its raw row alone; so there is no grinding;
* every unit writing a value the noise of more than one credited unit depends on is always proved or weighted, every
  noise slot is a unit's output or an anchor the verifier recomputes (`AnchorsPinned`), and a credited unit with a
  floor reads noise or derives it inside itself (`NoiseProved`, round 3.1, Daniel's 22:39Z condition (e)): a noise step
  is a unit of fan-out `F`, the credited units its noise reaches, and a wrong one chosen to cancel makes each of them
  right and cheap;
* each selection unit outside the always-proved set, and each weighted unit, is charged the cone it reaches: a unit is
  good only if the run paid for every unit that feeds it, along late edges, gates from units outside that set and
  operand edges from weighted units (`Feeds`, `SelGood`), and
  a share charging each unpaid unit its cone (`selConeWt`) covers every selected unit that isn't good (`SelCovers`);
* the always-proved set (`proved`: the A rows' ok units and formings, the B rows' per-epoch formings, the tally's limb
  units, any chained selection such as a done flag) is in every draw of its round (`AlwaysProved`); the tally is in it,
  reads every gate position or its registered sum (`GatesTallied`) and only gate positions, registered values and its
  own limbs and carries (`TallyReads`), takes no advice (`TallyNoAdvice`), and right tally units publish at most the
  selected units' weight (`TallyComputes`).

The draw stays value-free (`DrawnLayer.draw` reads only the draw coin): a draw weighted by committed selections would
publish, through its drawn indices, which slots are live and which weight rows the forming refuses, a private
partition.

* `SelectLayer`: a `DrawnLayer` with each unit's gating positions and the slots its floor depends on, the reading of a
  committed selection value as selected, the always-proved set, the tally's units, each position's registration record
  and whether that record's certificate is right, the tally's output position and the credit its value reads.
* `Selected`, `credited`, `published`: a unit selected at the committed values; the selected units' weight; the credit
  the window publishes, its tally's output at its committed value.
* `Gates`, `Feeds`, `SelGood`, `selConeWt`, `SelFan`: a unit writing a selection value that gates another; a late edge,
  a gate from a unit outside the always-proved set, or an operand edge from a weighted unit; a unit whose cone the run
  paid for; an unpaid unit's charge; the cone factor of the selection units outside the always-proved set and of the
  weighted units, at their draw weights.
* `AfterCoin`, `NoLateSelection`: a position committed after a unit's coin; no gate position is.
* `ReadsPos`: a unit has a position among its slots.
* `NoiseIn`, `OperandIn`, `FormEdge`: a unit writes a value another unit's noise reads; a floor or noise slot of
  another; a slot of another forming (a piece chain).
* `SelectionsSourced`, `SelectionsProved`, `CertifiedOperands`, `FormingClosed`, `NoiseProved`, `AnchorsPinned`,
  `AlwaysProved`, `TallyIn`, `GatesTallied`, `TallyReads`, `TallyNoAdvice`: protocol structure, checked by the window's
  shape check and the verifier's public input regions.
* `EpochFormedRight`: every epoch record is its raw row's forming under the salt's noise, the per-salt forming
  proofs' own guarantee.
* `RegisteredRight`: every registration record's certificate is right, the registration's own guarantee.
* `TallyComputes`: a property of the tally's Definition, proved for each window.
* `SelCovers`: the share covers every selected unit that isn't good, a property of the share's definition, proved for
  each game.
* `CreditSound`: (I8) at the published credit: outside an event of probability at most `δ`, an accepted window is
  accepted by the ideal game and its fixed words per round are at least the published credit less the deficiency's
  words, over `S`.

The hardness premise's condition is outside this vocabulary, and `Accounting.Assumptions.TS.SelectedFixed` is claimed
only where it holds: each unit's weight is its floor at every value of its slots outside `floor` and at every value of
its `floor` slots that is the forming, under its noise, of a raw row its certificates admit (its branches' minimum).
-/

namespace Accounting.TS

open Pouw.RelativeSecurity

/-- **A padded window's proofs with selections**: a sampled window's proofs (`DrawnLayer`) and
* `gates w j`: the positions of the selection values gating unit `j` of round `w` (its certificates), static: the
  forming's `ok` of each operand row, a done flag;
* `floor w j`: the slots its hardness floor depends on (a tile's operand rows; empty for a unit whose noise makes it
  hard at every operand);
* `certRow ℓ`: the raw row the certificate at gate position `ℓ` certifies (`ok_A,i`: activation row `x_i`; `ok_B,j`:
  weight row `B_j`);
* `forms`: the units whose Definition is a forming (`Pc8RowA`'s, `Pc8RowB`'s): their output is the noised forming of the
  raw row they read;
* `one v`: the committed value `v` of a selection position reads as selected;
* `proved`: the always-proved units (the A rows' ok units and formings, the B rows' per-epoch formings, the tally's limb
  units, chained selections, any noise unit whose noise reaches more than one credited unit), each in every draw of its
  round;
* `tally ⊆ proved`: the units computing the published credit, each output at most 16 bits (limbs, with carries
  committed between them);
* `regd ℓ`: the weights' registration record `ℓ` belongs to, if any (a raw weight row `B_j` and its `ok_B,j` share
  one), committed and proved once at registration, before every salt; the formed B row is not in it (its noise reads
  the epoch's salt);
* `regRight x r`: on `x`, record `r`'s certificate is its raw row's forming verdict (finite, `ρ·α ≥ 1`, at most `k/64`
  dead lanes), at the record's committed values, and a weight's registered `S_B` is `Σ_j ok_B,j` over its rows. The
  window's coins carry the registration's as their earliest coordinates, so whether it is right is an event on the
  window's paths, of probability at most `δ_reg` on accepted paths (`Accounting.Guarantees.TS.RegisteredRightOff`);
* `out`, `credOf`: the tally's output position, and the credit its value reads (`point(level)` tiles' words, at most
  16 bits of output);
* `noise w j`: the slots carrying unit `j` of round `w`'s noise or the values it is derived from, read off its
  Definition's ports by id: a forming's line and factors (`Pc8RowA`'s `e`, `fa`, `fb`; `Pc8RowB`'s `e`, `fb`, `fa`,
  `g`), a tile's noised operands (`Pc8Tile`'s `ac`, `pa`, `bc`, `pb`), a noise unit's seed;
* `anchor ℓ`: position `ℓ` is a public input region the verifier recomputes from public seeds under its derivation's
  id and refuses otherwise (`pc8.DERIVATIONS`: `verity/pouw/anchor/pearl-c/{ea,eb,fa,fb,g}/v1`, from `seed_a`,
  `seed_b`, each a hash of the salt and hiding roots);
* `noiseSelf`: the units whose Definition derives its noise in its own body from its coin and its public position,
  told apart by Definition id (under h3, the A forming: E_A,i's seed hashes `row_digest(x_i)`, a digest of a private
  row, so it is derived inside the always-proved forming, `F = 1`; under h0 none);
* `derive ℓ x`: the verifier's recomputation of anchor `ℓ` on `x`, from the public coins, the hiding roots and `ℓ`'s
  public position (`TileAnchors.check_public_inputs`);
* `weighted`: the units drawn in proportion to their public static fan-out rather than always proved (a forming, a
  noise step: Daniel's 22:42Z third option, `F ≈ 3`), each its `dw w j`, the public static draw weight its index is
  replicated in, never a committed value;
* `epoch ℓ`, `epochRow e`, `epochRight x e`: the epoch record position `ℓ` belongs to, if any (a formed B row
  `B̃_j ‖ P_B,j` in the salt-bound epoch tree, whose domain binds `(salt, root_W)` and is a public input of every tile
  instance), the raw row record `e` forms, and whether on `x` record `e` is that raw row's forming under the salt's
  noise (an event on the window's paths, the epoch's coins among their earliest coordinates, of probability at most
  `δ_epoch` on accepted paths: `Accounting.Guarantees.TS.EpochFormedRightOff`);
* `advised`: the units with C-Flock advice ports (parameters in the unit's io and in no read set, which nothing
  binds). -/
structure SelectLayer (C B : Type) (S m s : ℕ) (J Q Pos V : Type) extends DrawnLayer C B S m s J Q Pos V where
  gates : Fin S → J → Finset Pos
  floor : Fin S → J → Finset Q
  certRow : Pos → Pos
  forms : Finset (Fin S × J)
  one : V → Prop
  proved : Finset (Fin S × J)
  tally : Finset (Fin S × J)
  regd : Pos → Option ℕ
  regRight : Paths (Paths C m × B) S → ℕ → Prop
  out : Pos
  credOf : V → ℝ
  noise : Fin S → J → Finset Q
  anchor : Pos → Prop
  noiseSelf : Finset (Fin S × J)
  derive : Pos → Paths (Paths C m × B) S → V
  weighted : Finset (Fin S × J)
  dw : Fin S → J → ℝ
  epoch : Pos → Option ℕ
  epochRow : ℕ → Pos
  epochRight : Paths (Paths C m × B) S → ℕ → Prop
  advised : Finset (Fin S × J)

namespace SelectLayer

variable {C B : Type} {S m s : ℕ} {J Q Pos V : Type} (sl : SelectLayer C B S m s J Q Pos V)

/-- **Unit `j` of round `w` is selected on `x`**: every selection value gating it is committed as selected. -/
def Selected [Fintype C] [Fintype B] [Nonempty V] (x : Paths (Paths C m × B) S) (w : Fin S) (j : J) : Prop :=
  ∀ ℓ ∈ sl.gates w j, sl.one (sl.toDrawnLayer.committed x ℓ)

open Classical in
/-- **The selected units' weight** on `x`: what a right tally credits at the committed selections. Unselected units,
the padding, count nothing. -/
noncomputable def credited [Fintype C] [Fintype B] [Nonempty V] [Fintype J] (wt : Fin S → J → ℝ)
    (x : Paths (Paths C m × B) S) : ℝ :=
  ∑ w, ∑ j, if sl.Selected x w j then wt w j else 0

/-- **The published credit** on `x`: the tally's output at its committed value. -/
noncomputable def published [Fintype C] [Fintype B] [Nonempty V] (x : Paths (Paths C m × B) S) : ℝ :=
  sl.credOf (sl.toDrawnLayer.committed x sl.out)

/-- **Unit `u` writes a selection value gating unit `v`.** -/
def Gates (u v : Fin S × J) : Prop := ∃ ℓ ∈ sl.gates v.1 v.2, sl.src ℓ = some u

/-- **Unit `u` writes a floor or noise slot of unit `v`.** -/
def OperandIn (u v : Fin S × J) : Prop :=
  ∃ k, (k ∈ sl.floor v.1 v.2 ∨ k ∈ sl.noise v.1 v.2) ∧ sl.src (sl.pos v.1 v.2 k) = some u

/-- **`u` feeds `v`**: `u` writes `v` a value after `v`'s coin (`DrawnLayer.Late`), or `u` is outside the always-proved
set and writes a selection value gating `v`, whenever it is committed. A gate is an edge before the coin too: a
selection is a hardness certificate (the forming's `ok` admits a row only when it is finite, above its noise floor and
with few dead lanes), so a wrong one fixed before the noise still credits work that isn't hard. A gate from an
always-proved unit is no edge (RT W5's R7): it is right outside the proofs' failures, so it gates honestly whether or
not the run computed it; its own words still need paying, and its late edges stay. A weighted unit writing a floor or
noise slot (`OperandIn`) is an edge whenever it is committed: a per-salt B forming is cut before the window's coin,
so it is no late edge, and a wrong one is caught only at its draw weight. -/
def Feeds (u v : Fin S × J) : Prop :=
  sl.toDrawnLayer.Late u v ∨ (sl.Gates u v ∧ u ∉ sl.proved) ∨ (sl.OperandIn u v ∧ u ∈ sl.weighted)

/-- **Unit `j` of round `w` is good on `x` with its selections** under the run's modes: the run paid for every unit
that feeds it, transitively (itself included). -/
def SelGood (mode : Paths (Paths C m × B) S → Fin S × J → Mode) (x : Paths (Paths C m × B) S) (w : Fin S) (j : J) :
    Prop :=
  ∀ u : Fin S × J, Relation.ReflTransGen sl.Feeds u (w, j) → mode x u = .paid

open Classical in
/-- **Unit `u`'s cone over gates and late edges**: the weights of every unit `u` feeds, transitively, itself included.
A selection unit's, outside the always-proved set, is the cone it gates. -/
noncomputable def selConeWt [Fintype J] (wt : Fin S → J → ℝ) (u : Fin S × J) : ℝ :=
  ∑ v : Fin S × J, if Relation.ReflTransGen sl.Feeds u v then wt v.1 v.2 else 0

open Classical in
/-- **The cone factor of the selection and weighted units is at most `F`**: every unit outside the always-proved set
that gates another, and every weighted unit, has a cone of at most `F` times its draw weight: its own weight at uniform
draws, `dw` for a weighted unit, the public static weight its index is replicated in. A selection unit is caught only
at its own draw rate, so at uniform draws its cone multiplies the round's saving per unit of detection (RT W5's R2: ×17
for an `ok_B`, vacuous for an
`ok_A`). With every selection proved or registered (`SelectionsProved`) no unit outside the set gates, and it holds at
`F = 1`; cone-proportional draws (the large-batch option, a weighted forming at `dw` its fan-out plus one) bring the
factor back to about 3. Lemma C(a), that the draw law realizes `dw`, is not stated. -/
def SelFan [Fintype J] (wt : Fin S → J → ℝ) (F : ℝ) : Prop :=
  ∀ g : Fin S × J, ((g ∉ sl.proved ∧ ∃ v, sl.Gates g v) ∨ g ∈ sl.weighted) →
    sl.selConeWt wt g ≤ F * (if g ∈ sl.weighted then sl.dw g.1 g.2 else wt g.1 g.2)

/-- **Position `ℓ` is committed after unit `j` of round `w`'s coin** (the coin its noise reads). -/
def AfterCoin (ℓ : Pos) (w : Fin S) (j : J) : Prop :=
  w < (sl.cut ℓ).1 ∨ w = (sl.cut ℓ).1 ∧ (sl.coin w j).castSucc < (sl.cut ℓ).2

/-- **No selection is committed after its unit's coin**: every gate position is committed before the coin the gated
unit's noise reads (`coin`, a coin before the draw: for PoUW's seeded noise, the derivation from the salt and the roots
it hashes), so a selection is fixed before the noise and buys no grinding. STATIC-WIN's side row, whose `ok` sits
beside codes that read the noise, fails it (RT10 (e)); the corrected layout commits `ok_A,i` with the raw row `x_i` in
the activation commitment the noise's seed hashes, and `ok_B,j` with `B_j` at registration, before the salt. Protocol
structure. -/
def NoLateSelection : Prop := ∀ w j, ∀ ℓ ∈ sl.gates w j, ¬ sl.AfterCoin ℓ w j

/-- **Every selection value has a source**: a unit's output, or a certificate of the weights' registration. Never a
circuit input the prover fills, nor a host flag. Protocol structure. -/
def SelectionsSourced : Prop :=
  ∀ w j, ∀ ℓ ∈ sl.gates w j, (∃ g, sl.src ℓ = some g) ∨ (sl.src ℓ = none ∧ ∃ r, sl.regd ℓ = some r)

/-- **Every selection value is proved**: an always-proved unit's output, or a registered certificate. The first
version's window (the A rows' ok units always proved, `ok_B` proved at registration). Protocol structure. -/
def SelectionsProved : Prop :=
  ∀ w j, ∀ ℓ ∈ sl.gates w j, (∃ g ∈ sl.proved, sl.src ℓ = some g) ∨ (sl.src ℓ = none ∧ ∃ r, sl.regd ℓ = some r)

/-- **Unit `u` has position `ℓ` among its slots.** -/
def ReadsPos (u : Fin S × J) (ℓ : Pos) : Prop := ∃ k, sl.pos u.1 u.2 k = ℓ

/-- **`u` writes a slot of `v`, both formings**: an edge of a forming's piece chain (rowk's head, first, form segments
and tail; the cap's row pieces). -/
def FormEdge (u v : Fin S × J) : Prop :=
  u ∈ sl.forms ∧ v ∈ sl.forms ∧ ∃ k, sl.src (sl.pos v.1 v.2 k) = some u

/-- **Every operand a unit's floor depends on is a forming of a certified raw row**: for every slot `k` in unit `j`'s
`floor`, some gate position `ℓ` of `j` certifies a raw row (`certRow ℓ`) such that
* the operand's writer is a forming unit (`forms`) that is always proved or weighted, and some forming reaching it
  along the piece chain (`FormEdge`, itself included) reads that raw row; or the operand has no writer in the window
  and is the epoch record of that raw row (`epoch`, `epochRow`: the salt's formed B row, proved once per salt), and
* the certificate is computed from that raw row: its writer reads it, or the raw row and the certificate share a
  registration record.

With `FormingClosed` (every piece reads only its raw row, its noise and pieces), the operand is the noised forming of
the very row the certificate admits, and a wrong forming (zero codes, a zero `P`: RT10 (c)) is an accepted unsound
proof, a weighted unit the run didn't pay for (its operand edge), or the epoch's failure. A mask or a mux applied by
arithmetic after the forming (`codes' = codes·(1 − done)`, RT W5's R1) is no forming of the certified row and is
refused. At the served `k` the writer is rowk's tail (`Pc8RowATail`), which reads no raw row; the head and the form
segments read it (W5 r3 1b). Protocol structure. -/
def CertifiedOperands : Prop :=
  ∀ w j, ∀ k ∈ sl.floor w j, ∃ ℓ ∈ sl.gates w j,
    ((∃ g, g ∈ sl.forms ∧ (g ∈ sl.proved ∨ g ∈ sl.weighted) ∧ sl.src (sl.pos w j k) = some g ∧
        ∃ c, Relation.ReflTransGen sl.FormEdge c g ∧ sl.ReadsPos c (sl.certRow ℓ)) ∨
      (sl.src (sl.pos w j k) = none ∧ ∃ e, sl.epoch (sl.pos w j k) = some e ∧ sl.epochRow e = sl.certRow ℓ)) ∧
    ((∃ c, sl.src ℓ = some c ∧ sl.ReadsPos c (sl.certRow ℓ)) ∨
      (∃ r, sl.regd ℓ = some r ∧ sl.regd (sl.certRow ℓ) = some r))

/-- **A forming reads only its raw row, its noise and formings** (W5 r3 1a): every slot of a forming unit is a certified
raw row, one of its noise slots, or the output of a forming that is always proved or weighted (a piece of its chain,
its own output). So its output is a function of its raw row and its noise, and no value outside the forming (a row's
`α` from a stats unit outside `proved`, committed before the coin, which no edge charges) moves it. The shape check's
forming table lists rowk's four pieces (`Pc8RowAHead`, `Pc8RowAFirst`, `Pc8RowAForm`, `Pc8RowATail`, and `Pc8RowB`'s)
as one forming, every piece always proved or weighted. Protocol structure. -/
def FormingClosed : Prop :=
  ∀ g ∈ sl.forms, ∀ k, (∃ ℓ, sl.pos g.1 g.2 k = sl.certRow ℓ) ∨ k ∈ sl.noise g.1 g.2 ∨
    ∃ u ∈ sl.forms, (u ∈ sl.proved ∨ u ∈ sl.weighted) ∧ sl.src (sl.pos g.1 g.2 k) = some u

/-- **Unit `u` writes a value unit `v`'s noise reads**: a noise slot of `v` (a line, a factor, a seed, a noised operand)
whose writer is `u`. -/
def NoiseIn (u v : Fin S × J) : Prop := ∃ k ∈ sl.noise v.1 v.2, sl.src (sl.pos v.1 v.2 k) = some u

/-- **Every noise step that fans out is proved** (Daniel's 22:39Z condition (e)):
* every noise slot is a unit's output, or an anchor (`anchor`) the verifier recomputes: never a value the prover fills;
* every unit of positive weight with a hardness floor reads noise, or derives it in its own body (`noiseSelf`);
* every unit that writes a value the noise of more than one unit of positive weight depends on, along noise slots
  (`NoiseIn`, transitively: a seed, a line, a forming), is always proved or weighted (charged its cone at its public
  draw weight, `SelFan`).

A noise step's late cone is every credited unit its noise reaches, so at uniform draws its factor is that fan-out: a
wrong line, factor or seed chosen to cancel (a forming of zero codes and zero `P`) makes each tile it reaches right and
`+0` at one unit's draw rate. Always proved, it is right outside the proofs' failures; derived in-unit, it is the unit's
own proof (`F = 1`); an anchor no unit writes. A noise writer reaching one credited unit may stay outside `proved`: its
late edge charges that unit's cone, as any late edge. The shape check reads the noise slots off each Definition's ports
by id and refuses a noise port that is neither an anchor nor an always-proved unit's output when its writer reaches
more than one credited unit (`window-form: unproved-noise`), and a credited Definition with a floor, no noise port and
no in-body derivation (`window-form: unnoised-unit`). Under h0 every Pc8 line and factor is an anchor (`seed_a`,
`seed_b` hash the salt and hiding roots); under h3 E_A,i is derived in the A forming's body (`noiseSelf`: its row seed
hashes
`row_digest(x_i)`, a digest of a private row), and E_B, F_A, F_B, G stay anchors. Protocol structure. -/
def NoiseProved (wt : Fin S → J → ℝ) : Prop :=
  (∀ w j, ∀ k ∈ sl.noise w j, (∃ g, sl.src (sl.pos w j k) = some g) ∨
      (sl.src (sl.pos w j k) = none ∧ sl.anchor (sl.pos w j k))) ∧
    (∀ w j, 0 < wt w j → (sl.floor w j).Nonempty → (sl.noise w j).Nonempty ∨ (w, j) ∈ sl.noiseSelf) ∧
    ∀ g v v' : Fin S × J, 0 < wt v.1 v.2 → 0 < wt v'.1 v'.2 → v ≠ v' →
      Relation.TransGen sl.NoiseIn g v → Relation.TransGen sl.NoiseIn g v' → g ∈ sl.proved ∨ g ∈ sl.weighted

/-- **Every anchor is pinned**: every unit reading an anchor is always proved, and every accepted proof of a drawn unit
opens each anchor slot to the verifier's recomputation (`derive`) from the public coins, the hiding roots and the
slot's public position: a public input region the verifier builds itself (`TileAnchors.check_public_inputs`, C-Flock's
`HmIn.checkOwn`). So an anchor is read every round at its derivation, and an anchor that isn't one (`anchor := True`
on a slot the prover fills) fails it. Under h0 the Pc8 lines and factors are anchors; under h3 E_A,i is not (its seed
hashes `row_digest(x_i)`), and it is derived in the A forming's body (`noiseSelf`). Protocol structure. -/
def AnchorsPinned : Prop :=
  (∀ w j k, sl.anchor (sl.pos w j k) → (w, j) ∈ sl.proved) ∧
    ∀ x w i k, sl.accepted x w i → sl.anchor (sl.pos w (sl.toDrawnLayer.drawnAt x w i) k) →
      sl.opened x w i k = sl.derive (sl.pos w (sl.toDrawnLayer.drawnAt x w i) k) x

/-- **The epoch's formed rows are right on `x`**: every epoch record is its raw row's forming under the salt's noise.
The per-salt forming proofs' own guarantee (each B forming proved once per salt, opened by the window's tiles against
the epoch tree whose domain binds `(salt, root_W)`, so a stale salt's row has the wrong domain), failing on accepted
paths with probability at most `δ_epoch` (`Accounting.Guarantees.TS.EpochFormedRightOff`), which a composition adds as
it adds `δ_reg`. -/
def EpochFormedRight (x : Paths (Paths C m × B) S) : Prop := ∀ e, sl.epochRight x e

/-- **The always-proved units are in every draw**: each is named by every draw of its round. Protocol structure. -/
def AlwaysProved : Prop := ∀ t ∈ sl.proved, ∀ b : B, ∃ i, sl.draw b i = t.2

/-- **The tally is always proved.** Protocol structure. -/
def TallyIn : Prop := sl.tally ⊆ sl.proved

/-- **Every gate position is a tally slot**: some tally unit reads it, so it is opened in every round and its committed
value is its opening, not an arbitrary one; or a registered certificate, which the tally reads through the
registered sum (`S_B = Σ_j ok_B,j`, `RegisteredRight`). Protocol structure. -/
def GatesTallied : Prop :=
  ∀ w j, ∀ ℓ ∈ sl.gates w j, (∃ t ∈ sl.tally, ∃ k, sl.pos t.1 t.2 k = ℓ) ∨ (sl.src ℓ = none ∧ (sl.regd ℓ).isSome)

/-- **The registration's certificates are right on `x`**: every record's certificate is its raw row's forming verdict
(`ok_B,j` over `B_j` alone, which no salt changes). The registration's own guarantee (each `ok_B` proved once, at
registration, before every salt), failing on accepted paths with probability at most `δ_reg`
(`Accounting.Guarantees.TS.RegisteredRightOff`). -/
def RegisteredRight (x : Paths (Paths C m × B) S) : Prop := ∀ r, sl.regRight x r

/-- **The tally reads only the selections and itself**: every slot of a tally unit is a gate position, or written by
a tally unit (a limb, a carry between limbs, a partial sum). A tally reading a value a data unit writes could be moved
by that unit; a registered value (the weight's `S_B`) is read as registered. Each tally unit's outputs are at most 16
bits, so `real ≤ 2^40` is carried in limbs, each limb unit in
`proved`. Protocol structure; with `TallyComputes`, a property of the tally's Program. -/
def TallyReads : Prop :=
  ∀ t ∈ sl.tally, ∀ k, (∃ w j, sl.pos t.1 t.2 k ∈ sl.gates w j) ∨
    (∃ u ∈ sl.tally, sl.src (sl.pos t.1 t.2 k) = some u) ∨
      (sl.src (sl.pos t.1 t.2 k) = none ∧ (sl.regd (sl.pos t.1 t.2 k)).isSome)

/-- **No tally unit takes advice**: a C-Flock advice port is in no read set, so `TallyReads` can't see it, and a limb
taking its carry as advice would satisfy it with a carry nothing binds (W5 r3 3a). Protocol structure. -/
def TallyNoAdvice : Prop := ∀ t ∈ sl.tally, t ∉ sl.advised

/-- **A right tally publishes at most the selected units' weight**: on every path where each tally unit is right at
its committed slots, the published credit is at most the selected units' weight at the committed selections. A
property of the tally's Definition (STATIC-WIN's L4: per call `(Σ_i ok_A,i)·(Σ_j ok_B,j)`, the filler's `ok` zero,
rounded down to `point(level)`), proved for each window. It holds only if the tally reads each ok through `one`
(`Σ_c [one ok_A]·[one ok_B]`): a wrong ok writer can commit a word other than 0 or 1. With `S_B` registered it reads
the registration's sum, so it is stated under `RegisteredRight`; the filler's rows are constant zero, not committed
rows (a junk row is admitted at 1). -/
def TallyComputes [Fintype C] [Fintype B] [Nonempty V] [Fintype J] (wt : Fin S → J → ℝ) : Prop :=
  ∀ x, sl.RegisteredRight x → (∀ t ∈ sl.tally, sl.right x t.1 t.2 (sl.toDrawnLayer.view x t.1 t.2)) →
    sl.published x ≤ sl.credited wt x

open Classical in
/-- **The share covers every selected unit that isn't good with its selections**: on an accepted path, the weight of
the selected units the run didn't pay for, or whose feeding units it didn't, is at most the ideal game's deficiency
times `N_w`, the round's padded words. A property of the share's definition, proved for each game: a share charging
each unpaid unit its `selConeWt` covers (`DrawnLayer.covers_of_cone`'s argument over `Feeds`). -/
def SelCovers [Fintype C] [Fintype B] [Nonempty V] [Fintype J] {K T L U : Type}
    (P : Play (Paths C m × B) S K T L U) (mode : Paths (Paths C m × B) S → Fin S × J → Mode) (wt : Fin S → J → ℝ)
    (Nw : ℝ) : Prop :=
  ∀ x, P.accepts x →
    (∑ w, ∑ j, if sl.Selected x w j ∧ ¬ sl.SelGood mode x w j then wt w j else 0) ≤ P.game.deficiency x * Nw

end SelectLayer

/-- **(I8) at a value-dependent credit**: outside an event of probability at most `δ`, an accepted window is accepted
by the ideal game, and its owned right fixed words per round are at least its credit less its deficiency's words, over
`S`. `Accounting.Assumptions.TS.ProofsSound` is the static case, `credit = S·N_w`. -/
def CreditSound {Ω : Type} [Fintype Ω] {S : ℕ} {K T L U : Type} (P : Play Ω S K T L U) (credit : Paths Ω S → ℝ)
    (Nw δ : ℝ) : Prop :=
  ∃ E : Paths Ω S → Prop, prob E ≤ δ ∧
    ∀ x, P.accepts x → ¬ E x → P.game.accepts x ∧ (credit x - P.game.deficiency x * Nw) / S ≤ (P.run x).fixed

end Accounting.TS
