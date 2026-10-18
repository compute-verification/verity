import Definitions.Accounting.TS.Drawn

/-!
# A padded window's selections: credit that depends on committed values (trusted)

`DrawnLayer`'s wiring, cuts, coins and weights don't depend on the path, so its credit is static: every unit is
credited its words. A padded static window (Daniel, 18 Oct, 20:52Z) credits only its selected slots: its Program is one
circuit shaped by public parameters, a static superset of units, and which of them are credited is a value, a selection
(a mask, a forming's `ok`, a mux select, a done flag) that a unit of the circuit computes from committed values and that
is committed and sampled like any value. A wrong unit can emit a wrong selection; it can't rewire. This is the credit
rule of note:20261018T2200Z-draft-padded-credit-rule, on the conditions of
note:20261018T2125Z-report-format-table-r4-r5 (C4's ✓):

* credit only selected units, a unit being selected when every selection value gating it is committed as one;
* each selection value is a unit's output (`SelectionsComputed`), never a host flag or a circuit input;
* each selection unit is charged the cone it gates: a unit is good only if the run paid for every unit that feeds it,
  along late edges and along gates (`Feeds`, `SelGood`), and a share charging each unpaid unit its cone over both
  (`selConeWt`) covers every selected unit that isn't good (`SelCovers`);
* the tally, the units computing the window's published credit from the committed selections, is in every round's draw
  (`AlwaysProved`), and right tally units publish at most the selected units' weight (`TallyComputes`);
* a unit whose selection is committed after its coin (`Exposed`) has every candidate operand committed before that
  coin (`CandidatesEarly`); its grinding, a selection that is an honest function of the noise among `k` candidates,
  costs `k·μ` at H-μ's point mass `μ` (`Accounting.Assumptions.TS.SelectedFixed`).

* `SelectLayer`: a `DrawnLayer` with each unit's gating positions, the reading of a committed selection value as
  selected, each unit's number of candidates, the tally's units, its output position and the credit its value reads.
* `Selected`, `credited`, `published`: a unit selected at the committed values; the selected units' weight; the
  credit the window publishes, its tally's output at its committed value.
* `Gates`, `Feeds`, `SelGood`, `selConeWt`: a unit writing a selection value that gates another; a gate or a late edge;
  a unit whose cone over both the run paid for; an unpaid unit's charge.
* `AfterCoin`, `Exposed`, `CandidatesEarly`: a position committed after a unit's coin; a unit gated by such a
  selection; its candidates committed before its coin.
* `SelectionsComputed`, `AlwaysProved`, `TallyComputes`: protocol structure and a property of the tally's
  Definition, checked for each window.
* `SelCovers`: the share covers every selected unit that isn't good, a property of the share's definition, proved for
  each game.
* `CreditSound`: (I8) at the published credit: outside an event of probability at most `δ`, an accepted window is
  accepted by the ideal game and its fixed words per round are at least the published credit less the deficiency's
  words, over `S`.

Two conditions are the hardness premise's, outside this vocabulary, and `SelectedFixed` is claimed only where they
hold: each unit's weight is its hardness floor at every input its selections admit (its branches' minimum, noise
included), and its noise is downstream of its selections in the dataflow, so the floor holds at whichever candidate is
selected.
-/

namespace Accounting.TS

open Pouw.RelativeSecurity

/-- **A padded window's proofs with selections**: a sampled window's proofs (`DrawnLayer`) and
* `gates w j`: the positions of the selection values gating unit `j` of round `w`, static (public wiring): every
  selection on the path of its operands or its credit (the forming's `ok` of each operand row, a mux's select, a done
  flag);
* `one v`: the committed value `v` of a selection position reads as selected;
* `cands w j`: the number of candidates its selections choose among (`1` when they choose none);
* `tally`: the units computing the published credit, each in its round's draw;
* `out`, `credOf`: the tally's output position, and the credit its value reads (`point(level)` tiles' words, at most
  16 bits of output). -/
structure SelectLayer (C B : Type) (S m s : ℕ) (J Q Pos V : Type) extends DrawnLayer C B S m s J Q Pos V where
  gates : Fin S → J → Finset Pos
  one : V → Prop
  cands : Fin S → J → ℕ
  tally : Finset (Fin S × J)
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

/-- **`u` feeds `v`**: `u` writes `v` a value after `v`'s coin (`DrawnLayer.Late`), or a selection value gating `v`
whenever it is committed. A gate is an edge before the coin too: a selection is a hardness certificate (the forming's
`ok` admits a row only when it is finite, above its noise floor and with few dead lanes), so a wrong one fixed before
the noise still credits work that isn't hard. -/
def Feeds (u v : Fin S × J) : Prop := sl.toDrawnLayer.Late u v ∨ sl.Gates u v

/-- **Unit `j` of round `w` is good on `x` with its selections** under the run's modes: the run paid for every unit
that feeds it, transitively (itself included). -/
def SelGood (mode : Paths (Paths C m × B) S → Fin S × J → Mode) (x : Paths (Paths C m × B) S) (w : Fin S) (j : J) :
    Prop :=
  ∀ u : Fin S × J, Relation.ReflTransGen sl.Feeds u (w, j) → mode x u = .paid

open Classical in
/-- **Unit `u`'s cone over gates and late edges**: the weights of every unit `u` feeds, transitively, itself included.
A selection unit's is the cone it gates, `F_sel = selConeWt g / wt g` (the row unit of a call: its `m_c` or `n_c`
tiles). -/
noncomputable def selConeWt [Fintype J] (wt : Fin S → J → ℝ) (u : Fin S × J) : ℝ :=
  ∑ v : Fin S × J, if Relation.ReflTransGen sl.Feeds u v then wt v.1 v.2 else 0

/-- **Position `ℓ` is committed after unit `j` of round `w`'s coin** (the coin its noise reads). -/
def AfterCoin (ℓ : Pos) (w : Fin S) (j : J) : Prop :=
  w < (sl.cut ℓ).1 ∨ w = (sl.cut ℓ).1 ∧ (sl.coin w j).castSucc < (sl.cut ℓ).2

/-- **Unit `j` of round `w` is exposed**: a selection gating it is committed after its coin, so the selection can be a
function of its noise. -/
def Exposed (w : Fin S) (j : J) : Prop := ∃ ℓ ∈ sl.gates w j, sl.AfterCoin ℓ w j

/-- **An exposed unit's candidates are committed before its coin**: every slot of an exposed unit that isn't a
selection is committed before the coin its noise reads. A candidate written after the coin is the unrolled
interpreter's attack (an operand `−b` chosen after the salt) and the window is refused. -/
def CandidatesEarly : Prop :=
  ∀ w j, sl.Exposed w j → ∀ k, sl.pos w j k ∉ sl.gates w j → ¬ sl.AfterCoin (sl.pos w j k) w j

/-- **Every selection value is a unit's output**: computed in the circuit from committed values and sampled like any
value, never a circuit input or a host flag. Protocol structure, checked for each window. -/
def SelectionsComputed : Prop := ∀ w j, ∀ ℓ ∈ sl.gates w j, ∃ g, sl.src ℓ = some g

/-- **The tally is fully proved**: each of its units is named by every draw of its round. Protocol structure. -/
def AlwaysProved : Prop := ∀ t ∈ sl.tally, ∀ b : B, ∃ i, sl.draw b i = t.2

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
