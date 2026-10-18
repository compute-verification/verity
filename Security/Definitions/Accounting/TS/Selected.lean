import Definitions.Accounting.TS.Drawn

/-!
# A padded window's selections: credit that depends on committed values (trusted)

`DrawnLayer`'s wiring, cuts, coins and weights don't depend on the path, so its credit is static: every unit is
credited its words. A padded static window (Daniel, 18 Oct, 20:52Z) credits only its selected slots: its Program is one
circuit shaped by public parameters, a static superset of units, and which of them are credited is a value, a selection
(a mask, a forming's `ok`, a done flag) that a unit computes from committed values, or that the weights' registration
proved once, and that is committed like any value. A wrong unit can emit a wrong selection; it can't rewire. This is
the credit rule of note:20261018T2200Z-draft-padded-credit-rule (round 2, after
note:20261018T2207Z-report-redteam-padded-credit), on the conditions of note:20261018T2125Z-report-format-table-r4-r5
(C4's ✓):

* credit only selected units, a unit being selected when every selection value gating it is committed as one;
* each selection value is a unit's output, or a certificate the weights' registration proved (`SelectionsSourced`);
  in the first version every in-window selection unit is always proved (`SelectionsProved`);
* the writer of every operand a credited unit's floor depends on writes one of its certificates (`CertifiedOperands`),
  so a mask or a mux applied after the forming is refused: the certificate is computed from the position the unit
  reads;
* no selection is committed after its unit's coin (`NoLateSelection`), so there is no grinding;
* each selection unit outside the always-proved set is charged the cone it gates: a unit is good only if the run paid
  for every unit that feeds it, along late edges and along gates from units outside that set (`Feeds`, `SelGood`), and
  a share charging each unpaid unit its cone (`selConeWt`) covers every selected unit that isn't good (`SelCovers`);
* the always-proved set (`proved`: the A rows' formings, the tally, any chained selection such as a done flag) is in
  every round's draw (`AlwaysProved`); the tally is in it, reads every gate position (`GatesTallied`), and right tally
  units publish at most the selected units' weight (`TallyComputes`).

* `SelectLayer`: a `DrawnLayer` with each unit's gating positions and the slots its floor depends on, the reading of a
  committed selection value as selected, the always-proved set, the tally's units, each position's registration record
  and whether that record's certificate is right, the tally's output position and the credit its value reads.
* `Selected`, `credited`, `published`: a unit selected at the committed values; the selected units' weight; the credit
  the window publishes, its tally's output at its committed value.
* `Gates`, `Feeds`, `SelGood`, `selConeWt`, `SelFan`: a unit writing a selection value that gates another; a late edge,
  or a gate from a unit outside the always-proved set; a unit whose cone over both the run paid for; an unpaid unit's
  charge; the cone factor of the selection units outside the always-proved set.
* `AfterCoin`, `NoLateSelection`: a position committed after a unit's coin; no gate position is.
* `SelectionsSourced`, `SelectionsProved`, `CertifiedOperands`, `AlwaysProved`, `TallyIn`, `GatesTallied`: protocol
  structure, checked by the window's shape check.
* `RegisteredRight`: every registration record's certificate is right, the registration's own guarantee.
* `TallyComputes`: a property of the tally's Definition, proved for each window.
* `SelCovers`: the share covers every selected unit that isn't good, a property of the share's definition, proved for
  each game.
* `CreditSound`: (I8) at the published credit: outside an event of probability at most `δ`, an accepted window is
  accepted by the ideal game and its fixed words per round are at least the published credit less the deficiency's
  words, over `S`.

The hardness premise's condition is outside this vocabulary, and `Accounting.Assumptions.TS.SelectedFixed` is claimed
only where it holds: each unit's weight is its floor at every value of its slots outside `floor` and at every value of
its `floor` slots that its certificates admit (its branches' minimum, its noise included).
-/

namespace Accounting.TS

open Pouw.RelativeSecurity

/-- **A padded window's proofs with selections**: a sampled window's proofs (`DrawnLayer`) and
* `gates w j`: the positions of the selection values gating unit `j` of round `w` (its certificates), static: the
  forming's `ok` of each operand row, a done flag;
* `floor w j`: the slots its hardness floor depends on (a tile's operand rows; empty for a unit whose noise makes it
  hard at every operand);
* `one v`: the committed value `v` of a selection position reads as selected;
* `proved`: the always-proved units (the A rows' formings, the tally, chained selections), each in every draw of its
  round;
* `tally ⊆ proved`: the units computing the published credit;
* `regd ℓ`: the weights' registration record `ℓ` belongs to, if any (a B row's operand words and its `ok_B` share
  one), committed and proved once at registration, before every window's salt;
* `regRight r`: record `r`'s certificate is its row's forming verdict, at the record's committed values;
* `out`, `credOf`: the tally's output position, and the credit its value reads (`point(level)` tiles' words, at most
  16 bits of output). -/
structure SelectLayer (C B : Type) (S m s : ℕ) (J Q Pos V : Type) extends DrawnLayer C B S m s J Q Pos V where
  gates : Fin S → J → Finset Pos
  floor : Fin S → J → Finset Q
  one : V → Prop
  proved : Finset (Fin S × J)
  tally : Finset (Fin S × J)
  regd : Pos → Option ℕ
  regRight : ℕ → Prop
  out : Pos
  credOf : V → ℝ

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

/-- **`u` feeds `v`**: `u` writes `v` a value after `v`'s coin (`DrawnLayer.Late`), or `u` is outside the always-proved
set and writes a selection value gating `v`, whenever it is committed. A gate is an edge before the coin too: a
selection is a hardness certificate (the forming's `ok` admits a row only when it is finite, above its noise floor and
with few dead lanes), so a wrong one fixed before the noise still credits work that isn't hard. A gate from an
always-proved unit is no edge (RT W5's R7): it is right outside the proofs' failures, so it gates honestly whether or
not the run computed it; its own words still need paying, and its late edges stay. -/
def Feeds (u v : Fin S × J) : Prop := sl.toDrawnLayer.Late u v ∨ (sl.Gates u v ∧ u ∉ sl.proved)

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

/-- **The selection units' cone factor is at most `F`**: every unit outside the always-proved set that gates another
has a cone of at most `F` times its own weight. A selection unit is caught only at its own draw rate, so at uniform
draws its cone multiplies the round's saving per unit of detection (RT W5's R2: ×17 for an `ok_B`, vacuous for an
`ok_A`). With every selection proved or registered (`SelectionsProved`) no unit outside the set gates, and it holds at
`F = 1`; cone-proportional draws (the large-batch option) bring the factor back to about 3. -/
def SelFan [Fintype J] (wt : Fin S → J → ℝ) (F : ℝ) : Prop :=
  ∀ g : Fin S × J, g ∉ sl.proved → (∃ v, sl.Gates g v) → sl.selConeWt wt g ≤ F * wt g.1 g.2

/-- **Position `ℓ` is committed after unit `j` of round `w`'s coin** (the coin its noise reads). -/
def AfterCoin (ℓ : Pos) (w : Fin S) (j : J) : Prop :=
  w < (sl.cut ℓ).1 ∨ w = (sl.cut ℓ).1 ∧ (sl.coin w j).castSucc < (sl.cut ℓ).2

/-- **No selection is committed after its unit's coin**: every gate position is committed before the coin the gated
unit's noise reads, so a selection is fixed before the noise and buys no grinding. STATIC-WIN meets it (`ok` is in the
side row, committed before the window's coin). Protocol structure. -/
def NoLateSelection : Prop := ∀ w j, ∀ ℓ ∈ sl.gates w j, ¬ sl.AfterCoin ℓ w j

/-- **Every selection value has a source**: a unit's output, or a certificate of the weights' registration. Never a
circuit input the prover fills, nor a host flag. Protocol structure. -/
def SelectionsSourced : Prop :=
  ∀ w j, ∀ ℓ ∈ sl.gates w j, (∃ g, sl.src ℓ = some g) ∨ (sl.src ℓ = none ∧ ∃ r, sl.regd ℓ = some r)

/-- **Every selection value is proved**: an always-proved unit's output, or a registered certificate. The first
version's window (A rows always proved, B rows' formings proved at registration). Protocol structure. -/
def SelectionsProved : Prop :=
  ∀ w j, ∀ ℓ ∈ sl.gates w j, (∃ g ∈ sl.proved, sl.src ℓ = some g) ∨ (sl.src ℓ = none ∧ ∃ r, sl.regd ℓ = some r)

/-- **The writer of every operand a unit's floor depends on writes one of its certificates**: for every slot `k` in
unit `j`'s `floor`, some gate position of `j` has the same writer (`src`), or the same registration record (`regd`).
The certificate is then computed by the unit that wrote the operand, from the position the unit reads: a mask or a
mux applied by arithmetic after the forming (`codes' = codes·(1 − done)`, RT W5's R1) writes an operand no certificate's
writer writes, and is refused. A unit with no gates has an empty floor. STATIC-WIN meets it (`Pc8RowA` writes the row
and its `ok` together; a B row and its `ok_B` share a registration record). Protocol structure. -/
def CertifiedOperands : Prop :=
  ∀ w j, ∀ k ∈ sl.floor w j, ∃ ℓ ∈ sl.gates w j,
    (∃ g, sl.src (sl.pos w j k) = some g ∧ sl.src ℓ = some g) ∨
      (∃ r, sl.regd (sl.pos w j k) = some r ∧ sl.regd ℓ = some r)

/-- **The always-proved units are in every draw**: each is named by every draw of its round. Protocol structure. -/
def AlwaysProved : Prop := ∀ t ∈ sl.proved, ∀ b : B, ∃ i, sl.draw b i = t.2

/-- **The tally is always proved.** Protocol structure. -/
def TallyIn : Prop := sl.tally ⊆ sl.proved

/-- **Every gate position is a tally slot**: some tally unit reads it, so it is opened in every round and its committed
value is its opening, not an arbitrary one. Protocol structure. -/
def GatesTallied : Prop := ∀ w j, ∀ ℓ ∈ sl.gates w j, ∃ t ∈ sl.tally, ∃ k, sl.pos t.1 t.2 k = ℓ

/-- **The registration's certificates are right**: every record's certificate is its row's forming verdict. The
registration's own guarantee (its formings proved once, at registration), at its own `δ_reg`. -/
def RegisteredRight : Prop := ∀ r, sl.regRight r

/-- **A right tally publishes at most the selected units' weight**: on every path where each tally unit is right at
its committed slots, the published credit is at most the selected units' weight at the committed selections. A
property of the tally's Definition (STATIC-WIN's L4: per call `(Σ_i ok_A,i)·(Σ_j ok_B,j)`, the filler's `ok` zero,
rounded down to `point(level)`), proved for each window. -/
def TallyComputes [Fintype C] [Fintype B] [Nonempty V] [Fintype J] (wt : Fin S → J → ℝ) : Prop :=
  ∀ x, (∀ t ∈ sl.tally, sl.right x t.1 t.2 (sl.toDrawnLayer.view x t.1 t.2)) → sl.published x ≤ sl.credited wt x

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
