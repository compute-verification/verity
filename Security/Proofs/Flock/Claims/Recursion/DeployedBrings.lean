import Proofs.Flock.Claims.Recursion.RegSetup
import Proofs.Flock.Claims.Recursion.ResLowerDeployed

/-!
# The deployed brings' setup (`deployedB`)

`Brings.setup xO` lists the SHA-512 evaluations the compiled verifier makes itself at the V\* statement `xO`, outside
the oracle, which `DecompBV`'s and `JointV`'s escape (`setupCalls`) searches. The deployed verifier makes two kinds:

* check 5's (`ResLower.resSetup`): `checkValue`'s and `checkZero`'s commit inputs at each `rec-residuals` statement,
  `checkOpen`'s at each `rec-open` statement;
* the registered reads' (`regIns`): for the table row each instance's ref names in each registered port, its tree leaf,
  its frame-v3 leaf and each climb node, as `checkRegistered` opens them. `checkRegistered` also opens rows no instance
  references, and those evaluations are left out: no conclusion reads an unreferenced row, so its climb needs no
  escape.

`deployedSetup xO` is the two, and `deployedB b` is `b` with that setup. Both read `xO` alone (its recursive setup,
`setupO xO`), never a decision, a witness or the prover's run, so whatever the prover sends, the setup at an accepted
decision is these evaluations. With RECE's staging round the shared rows are in each part's staged statement, so
`setupO xO` reads them there, and the definition is unchanged. `resSetup_deployedB` is `recursiveKSAt_of_leaves`' `hset`
and `regSetup_deployedB` is `RegSetup`.
-/

namespace RecDeploy

/-- **The compiled verifier's own SHA-512 evaluations at the V\* statement `xO`**: check 5's (`ResLower.resSetup`),
then the registered reads' leaves and climbs (`regIns`). -/
def deployedSetup (xO : Claims.Msg) : List Claims.Msg := ResLower.resSetup xO ++ regIns xO

/-- **The deployed brings** at `b`: `b` with the verifier's own evaluations as its setup. -/
def deployedB (b : Brings) : Brings := { b with setup := deployedSetup }

theorem deployedB_setup (b : Brings) : (deployedB b).setup = deployedSetup := rfl

/-- **`hset` at the deployed brings**: their setup lists check 5's evaluations at every V\* statement. -/
theorem resSetup_deployedB (b : Brings) : ∀ xO, ∀ u ∈ ResLower.resSetup xO, u ∈ (deployedB b).setup xO :=
  fun _ _ hu => List.mem_append_left _ hu

/-- **`RegSetup` at the deployed brings**: their setup lists every registered read's leaf and climb. -/
theorem regSetup_deployedB (b : Brings) : RegSetup (deployedB b) := fun _ _ hu => List.mem_append_right _ hu

/-- The record `recBound` compiles at the deployed brings lists the registered reads (`regListed_recBound`). -/
theorem regListed_deployedB (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
    (L : Claims.Recursion.VStarB.LayoutB) (IV : Claims.Verifier FlockDeploy.Coin) :
    RegListed (recBound rel (deployedB b) L IV) :=
  regListed_recBound rel (regSetup_deployedB b) L IV

/-- The coin server's record keeps the deployed setup: `deployedB` commutes with replacing `tr`. -/
theorem deployedB_tr (b : Brings) (t : Claims.Msg → List (Claims.Msg × Claims.Out) → ByteArray) :
    { deployedB b with tr := t } = deployedB { b with tr := t } := rfl

end RecDeploy
