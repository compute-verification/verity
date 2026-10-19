import Definitions.Accounting.TS.SharedNoise

/-!
# One noise seed a call: the luck premises a tile takes (trusted; a draft for the red team)

J10 (note:compute-accounting/20261019T0825Z-draft-j10-shared-noise): H1 + fx1's noise derived once a call row and once
a call column from one seed a call (`Accounting.TS.CallCoin`), each tile's fold keys still its own. The luck premises
the composed statements take per tile (`PerTile` over a stage's coin, #2638) are stated over the whole coin the
cheater's strategy reads, so their Lean form already allows tiles whose coins correlate; what one seed a call changes
is which of them a tile's own coins can carry:

* B′ (`ReceiptLuckAt`, `r_B = ½`) is the fold keys' luck, and the keys stay a tile's own: `KeyReceiptLuckAt` takes it
  given every block's noise and every other tile's keys, over the tile's keys alone, and
  `Guarantees.TS.CondReceiptLuckMarginal` gives the marginal `ReceiptLuck` that `PerTile` reads.
* (H-μ) (`OperandHMu`, `μ = ¾`) is the noise's luck, and no part of the noise is a tile's own: `CallOperandLuckAt` takes
  it marginally over the call's coin, whose strategy may read every neighbour's noise and every word it computed.
  Its new attack is a neighbour's words: a tile whose rows (or columns) a computed tile shares reads them through
  that tile's words (`benchmarks/rate_redteam/shared_noise.py`, A1–A3).
* The guess bound (`GuessLuckAt`, `p0 = ½`) is the fold state's, from the keys: `CallGuessLuckAt` is it per tile over
  the call's coin, unchanged.

The changes to the TS and composed statements' premises (#2482, #2638): `CircuitWork`'s and `CircuitCredit`'s
`PerTile … OperandHMu`, `PerTile … ReceiptLuckAt` and `PerTile … GuessLuckAt` keep their form, the stage's coin `C`
now one seed a call's coin; `ReceiptLuckAt` is then met from `KeyReceiptLuckAt` (proved bridge, to write), and
`OperandHMu` is this file's `CallOperandLuckAt`, a stronger claim than one seed a tile's, since its strategy reads
correlated coins. `ServiceBudget`'s bound reads only each tile's marginal luck (its stage factor is affine in the
lucky shares), so neither costs `ε` a factor. `UnitFixed` and `WholeEntropyAt` (tiles read disjoint key blocks) are
unchanged. The seeded statements' `SeedLuckAt` folded half reads `TemplateMoment`'s rate, whose `PrivateMoment`
step needs each tile's noise its own: under one seed a call it is `Guarantees.TS.GridTemplateMoment`'s, per seed
group at its grid.
-/

namespace Accounting.Assumptions.SharedNoise

open Accounting.TS Pouw.RelativeSecurity

/-- `accounting-shared-noise-receipt-luck` (proposed, J10). **B′ from a tile's own keys, under one seed a call**: for
every tile `t` of an `m × n` call, its receipt game (`G t`, over the call's coin) keeps its honest committed value
while its deficient words are not all right with probability at most `pr` over its fold keys, at every value of the
call's noise and of every other tile's keys (`CallCoin.keySplit t`). It is one seed a tile's B′ read with the rest of
the call fixed: fx1's word step is a bijection of the state, the keys are σ_R's stream at the tile's own position, and
no other tile's receipt reads them. It fails if a tile's keys are a function of the call's seed or of a neighbour's
position. -/
def KeyReceiptLuckAt {m n S : ℕ} {Rw Cl Ky : Type} [Fintype Ky] (pr : ℝ)
    (G : Fin m × Fin n → ReceiptGame (CallCoin m n Rw Cl Ky) S) : Prop :=
  ∀ t, (G t).CondReceiptLuck (CallCoin.keySplit t) pr

/-- `accounting-shared-noise-operand-luck` (proposed, J10). **(H-μ) per tile over the call's coin, under one seed a
call**: every tile's operand luck at `μ` over the whole coin, its strategy reading every block's noise and every word
it computed of the tiles that share its rows or columns. Neighbour words are the new channel: with `B_J1 = B_J0`
(`A_I1 = A_I0`) the skipped tile's word differs from the computed neighbour's by ``X·Δc`` (``ΔX·W'``) beyond what a
row's or a column's work gives, a ternary product as costly as the tile's own (A1, A2 of
`benchmarks/rate_redteam/shared_noise.py`: word point mass 0.142 against one seed a tile's 0.115 at `A = B = 0`, at
most 0.001 on uniform int8, for `μ = ¾`). It fails if a cheaper product than the tile's own reproduces a skipped
word from a neighbour's. -/
def CallOperandLuckAt {m n S : ℕ} {Rw Cl Ky : Type} [Fintype Rw] [Fintype Cl] [Fintype Ky] (μ : ℝ)
    (G : Fin m × Fin n → ReceiptGame (CallCoin m n Rw Cl Ky) S) : Prop :=
  ∀ t, (G t).OperandLuck μ

/-- `accounting-shared-noise-guess-luck` (proposed, J10). **The whole-unit guess bound per tile over the call's coin**:
a tile skipped whole keeps its checked values with probability at most `p0` a round (`RoundGame.LuckAt` of its game),
its strategy reading the call's coin whole. The fold state starts from the tile's own keys, so one seed a call leaves
it as one seed a tile has it. -/
def CallGuessLuckAt {m n S : ℕ} {Rw Cl Ky : Type} [Fintype Rw] [Fintype Cl] [Fintype Ky] (p0 : ℝ)
    (G : Fin m × Fin n → ReceiptGame (CallCoin m n Rw Cl Ky) S) : Prop :=
  ∀ t, RoundGame.LuckAt p0 (G t).game

end Accounting.Assumptions.SharedNoise
