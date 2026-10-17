import Proofs.Flock.Claims.Complete.ZkTables.Level0
import Proofs.Flock.Claims.Complete.ZkTables.Table

/-!
# Masked completeness from its parts

`complete_jsonT` takes four table-level premises (`RepShapedT`, `ProverRunsT`, `LigeritoT`, `InnerT`). Each splits into
level 0's masked Ligerito, stated once over `ligeritoPZk` and `Zk.ligeritoWith` (`Level0`: `ProverRuns`, `ProverShape`,
`Lanes`, `Verifier`), and the rest, which the plain proofs and the closed forms of the masked PIOP cover: what each
T-Prop asks beyond `Level0` (`ShapeRest`, `RunsRest`, `LigeritoRest`, `InnerRest`). `complete_jsonT_of_parts` composes
them.

The mask-rank bound and the prover's rank refusal enter none of the four: the masked prover `repPZk` has no rank
refusal, so its error is `BadZk`'s, which `δc` already counts.
-/

namespace FlockComplete.ZkTables

open Flock Transcript FlockComplete FlockDeploy JsonRT

section Rest

variable (H : Claims.Msg → Claims.Msg)

/-- **`RepShapedT` beyond level 0's shape**: the masked PIOP's and the inner proof's lengths, and the encoding's bound,
given `Level0.ProverShape`. -/
def ShapeRest : Prop := Level0.ProverShape → RepShapedT H

/-- **`ProverRunsT` beyond level 0**: the masked PIOP returns in closed form, the rep's constraints vanish at its pads
(level 0's lane rows from `Level0.Lanes`), so the inner proof returns, and the rep records `repRoundsT` rounds, given
`Level0.ProverRuns`, `Level0.Lanes` and `Level0.Verifier` (whose final position gives the Ligerito's round count). -/
def RunsRest : Prop := Level0.ProverRuns → Level0.Lanes → Level0.Verifier → ProverRunsT H

/-- **`LigeritoT` beyond level 0**: the verifier's replay and ring switch return the prover's values at the prover's
position, so its masked Ligerito runs on the prover's inputs, given `Level0.Verifier`. -/
def LigeritoRest : Prop := Level0.Verifier → LigeritoT H

/-- **`InnerT` beyond level 0**: as `LigeritoRest`, and the verifier's constraints are the ones the prover's inner
proof checked before it returned, given `Level0.Verifier`. -/
def InnerRest : Prop := Level0.Verifier → InnerT H

end Rest

/-- **Masked completeness from its parts**: `complete_jsonT` from level 0's four premises (`Level0`) and the rest of
each table-level premise. -/
theorem complete_jsonT_of_parts (H : Claims.Msg → Claims.Msg) (S : Claims.JsonSpec printJ Flock.canon)
    (h0 : Level0.ProverRuns) (h1 : Level0.ProverShape) (h2 : Level0.Lanes) (h3 : Level0.Verifier)
    (gS : ShapeRest H) (gR : RunsRest H) (gL : LigeritoRest H) (gI : InnerRest H) :
    Claims.Flock.Complete H (deployZk Readable.relation) :=
  complete_jsonT H S (gS h1) (gR h0 h2 h3) (gL h3) (gI h3)

end FlockComplete.ZkTables
