import Proofs.Flock.Claims.Recursion.KSLin2E
import Proofs.Flock.Claims.KS.PlainRConflict
import Proofs.Flock.Claims.KS.PlainRCapPrefix

/-!
# `KSFlockLin` at the deployed reader's bound record (U3)

The bound record whose steps are the deployed reader's (`FlockDeploy.nextR rd`) and whose columns are the ones it
opens, trimmed (`FlockDeploy.deployPlainTrimR`), has `DE` = `deployPlainTrimR` up to a weight `KSFlockLin` doesn't
read (`RecDeploy.deE_R_trim`). So `KSFlockLin` there is `plainR_KSFlockLin`, from conflicts tied to collisions
(`plainR_conflictTied`, proved) and the table bound at the abort law at `deployPlainR`
(`ksFlockLin_R_of_table`). That table bound is `plainR_tableBoundAbortLin_R`, from the reader's bound wherever some
run accepts (`PlainRdInR`) alone, the level caps' prefix being proved (`capPrefixR`), so `ksFlockLin_R` takes exactly
that one.
-/

namespace RecDeploy

open Claims (Msg)
open FlockDeploy (Coin FinalP nextR)

/-- **`KSFlockLin` at `DE` at the deployed reader's steps and trimmed columns, from the table bound at the abort law
at `deployPlainR`**: `plainR_KSFlockLin` with `plainR_conflictTied`, as `DE` is `deployPlainTrimR` up to a weight
`KSFlockLin` doesn't read (`deE_R_trim`). -/
theorem ksFlockLin_R_of_table (rd : Msg → Option FinalP) (wE : Msg → ℝ) (join : List Msg → Msg)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (htab : Claims.Recursion.TableBoundAbortLin FlockDeploy.sha512
      (FlockDeploy.deployPlainR _root_.Flock.Readable.relation rd) (FlockDeploy.KS.wordLen false)
      (FlockDeploy.KS.wide false) FlockDeploy.KS.CpDep) :
    Claims.Recursion.KSFlockLin FlockDeploy.sha512
      (bdDeployed _root_.Flock.Readable.relation (nextR rd)
        (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation rd).cols wE join rowInDeployed final w).DE
      FlockDeploy.KS.CpDep :=
  fun decode hdec => FlockDeploy.KS.plainR_KSFlockLin _root_.Flock.Readable.relation rd
    (FlockDeploy.KS.plainR_conflictTied _ rd) htab decode hdec

/-- **`KSFlockLin` at `DE` at the deployed reader's steps and trimmed columns** (U3), from the reader's bound wherever
some run accepts (`PlainRdInR`) alone, through the table bound `plainR_tableBoundAbortLin_R`. -/
theorem ksFlockLin_R (rd : Msg → Option FinalP) (wE : Msg → ℝ) (join : List Msg → Msg)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (hIn : ∀ x, ¬ Claims.Flock.NoAcc FlockDeploy.sha512 (FlockDeploy.VR rd) x → FlockDeploy.KS.PlainRdInR rd x) :
    Claims.Recursion.KSFlockLin FlockDeploy.sha512
      (bdDeployed _root_.Flock.Readable.relation (nextR rd)
        (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation rd).cols wE join rowInDeployed final w).DE
      FlockDeploy.KS.CpDep :=
  ksFlockLin_R_of_table rd wE join final w (FlockDeploy.KS.plainR_tableBoundAbortLin_R rd hIn)

end RecDeploy
