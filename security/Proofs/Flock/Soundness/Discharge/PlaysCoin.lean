module

public import Proofs.Flock.Soundness.Discharge.PlaysCoin.Statements
public import Proofs.Flock.Soundness.Discharge.PlaysCoin.Coins
public import Proofs.Flock.Soundness.Discharge.PlaysCoin.Tape
public import Proofs.Flock.Soundness.Discharge.PlaysCoin.Decodes
import Mathlib.Combinatorics.Matroid.Init

/-!
# The coin half of `DecodesAll` and `DecodesOn`

Statements in `Statements.lean`: `ToTExact`, `CoinsOfRounds`, `TapeOfGroups`, `CutOfPat`, `DecodesOfRuns`,
`DecodesOfRunsAt`, `DecodesAllOfRuns`, `DecodesOnOfRuns`. Proofs: `Coins.lean` (`toT_exact`, `coinsOfRounds`),
`Tape.lean` (`tapeOfGroups`, `cutOfPat`), `Decodes.lean` (`decodesOfRuns`, `decodesOfRunsAt`, `decodesAllOfRuns`,
`decodesOnOfRuns`).
-/

@[expose] public section
