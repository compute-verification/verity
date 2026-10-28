import Proofs.Flock.Claims.Recursion.VStarFinal
import Proofs.Flock.Claims.Recursion.HpEarlyStaged
import Proofs.Flock.Claims.Recursion.HhHeld
import Proofs.Flock.Claims.Recursion.HmsgRun
import Proofs.Flock.Claims.Recursion.HonestRun
import Proofs.Flock.Claims.Recursion.MoreSlotsAll
import Proofs.Flock.Claims.Recursion.HonestPartsLayR
import Proofs.Flock.Claims.Recursion.SchedDense
import Proofs.Flock.Claims.Recursion.DeployedBrings
import Proofs.Flock.Claims.Recursion.VStarDense
import Proofs.Flock.Claims.Recursion.HonestWitR
import Proofs.Flock.Claims.Recursion.NumsFixedR
import Proofs.Flock.Claims.Recursion.VStarTopsOk
import Proofs.Flock.Claims.Recursion.RoundWords
import Proofs.Flock.Claims.Recursion.VStarPortNames

/-!
# The honest half's run and its named premises

`RecursiveKS` reads, at the brings `deployedB b` for every `b`, the premise `DecompB` at `PartAccepts`: wherever every V\*
part's circuit accepts at a dense compiled statement, `IVStar` accepts or two recorded calls collide. It is non-vacuous
where some run has every part accepting. `honest_half_tops` gives that run: the compiled verifier's run of the reference
prover at `deployedB` of the deployed brings at `x` (`bringsAt`: the deployed assigner `A0`, the deployed inner prover
`IrDep` and the fitted records), and every part `s < nS x` accepts on it.

This file has the run's records (`bLAt`, `IAt`, `DAt`) and the premises witnessed outside Lean, all on one run
(`HonestFixture`): the run accepting (`HonestRunAccepted`), and no collision among a part's recorded calls (`NoCollAt`,
SHA-512's collision event, which passed on that run's tree). Each part's staging setting up follows from the run
accepting (`stagedSetsUp_accepted`).
`bringsAt_deployedB`: the brings here are `RecursiveKS`'s `deployedB` at `bringsAt mask Ir b0 x`.
-/

namespace RecDeploy.HonestHalf

open _root_.Flock
open Claims (Msg Out)
open FlockDeploy (Coin)
open Claims.Recursion.VStarB (readForms)
open RecDeploy.VStarBuild
open RecDeploy.VStarHonest

/-- **The brings `honest_half_tops` runs at are `RecursiveKS`'s**: `deployedB` of the deployed brings at `x`. -/
theorem bringsAt_deployedB (mask : ℕ → ℕ → Bool) (Ir : InnerRec) (b0 : Brings) (x : Msg) :
    bringsAt mask Ir (deployedB b0) x = deployedB (bringsAt mask Ir b0 x) := rfl

/-- A part with a statement and a circuit is one of `x`'s parts. -/
theorem lt_nS_of_partAt {x : Msg} {s : ℕ} {sv : _root_.VStar.Stmt} {c : Circuit}
    (h : partAt x s = some (sv, c)) : s < nS x := by
  cases hr : readX x with
  | none => simp [partAt, hr] at h
  | some r =>
    simp only [nS, hr]
    by_contra hs
    have hn : r.w.parts[s]? = none := by simp; omega
    simp [partAt, hr, hn] at h

/-! ## The run and its named premises -/

section Premises

variable (mask : ℕ → ℕ → Bool) (Ir : InnerRec) (b0 : Brings) (x : Msg) (aux : Msg → ℕ → Flock.RecRecord.Round)
  (nOf : Msg → ℕ → ℕ) (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
  (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace)

/-- The deployed brings at `x` under `deployedB`, with the coin server's record and V\*'s layout's readers. -/
noncomputable abbrev bLAt : Brings :=
  { { bringsAt mask Ir (deployedB b0) x with tr := recTr aux nOf hd recRest } with
    rd := (DeployedR.layR OpenRows.elemR).readers }

/-- The inner verifier the compiled record runs. -/
noncomputable abbrev IAt : Claims.Recursion.Inner Coin Claims.Flock.Wit := Claims.Recursion.Inner.ofFlock DeployedR.bdR.deploy decI

/-- `RecursiveKSAt`'s compiled record at the deployed brings at `x`. -/
noncomputable abbrev DAt : Claims.Recursion.Deploy Coin :=
  recBound _root_.Flock.Readable.relation
    { bringsAt mask Ir (deployedB b0) x with tr := recTr aux nOf hd recRest } (DeployedR.layR OpenRows.elemR)
    DeployedR.bdR.IVStar

/-- **The verifier accepts the reference prover's run, at an instance that exists** (witnessed outside Lean). The
statement `x` with witness `w` is in the compiled record's relation, parses to `st` at the dense form, and the compiled
verifier accepts the reference prover's run on tape `ω` and coins `r`. A concrete statement can't be evaluated in Lean
(its circuit is 22 MB and `Json.parse` is not kernel-evaluable), so this is witnessed outside Lean: every V\* part's
session of the honest deployed recursive run is accepted by the compiled `flock-verify` (composed per part; no single
compiled verdict exists).

The witness is a run of `benchmarks/recursion_e2e --e2e --cpu` (K = 64, N = 16, under the firewall, at
16a44e3b86). The compiled verifier printed `VERDICT {"accepted":true}` on each of the 10 V\* parts (L0–L3,
alg-p0, alg-p1, cap, cap-L1–L3), and the inner session was accepted. With a challenge or query coin flipped, the parts
that read it refused.

Only Python joins the 10 per-part verdicts, so what the run witnesses is that every part's session accepted, not that
one compiled verifier accepted the whole outer statement. -/
def HonestRunAccepted (w : Claims.Flock.Wit) (st : Claims.Recursion.Stmt)
    (ω : Tape (bLAt mask Ir b0 x aux nOf hd recRest))
    (r : Fin ((DAt mask Ir b0 x aux nOf hd recRest).V.k x) → Coin) : Prop :=
  Claims.Recursion.R (DAt mask Ir b0 x aux nOf hd recRest) (IAt decI) x w ∧
    (DAt mask Ir b0 x aux nOf hd recRest).parse x = some st ∧ Claims.Recursion.Form.Dense st.form ∧
    (Claims.run FlockDeploy.sha512 (refP (IAt decI).V (bLAt mask Ir b0 x aux nOf hd recRest) w)
      (DAt mask Ir b0 x aux nOf hd recRest).V x ω r).out = .acc

/-- **No collision among a part's recorded calls** on the reference prover's run: at each part `s < nS x`, the hm96
calls `layR` records at its outer witness and the calls the compiled verifier makes at its V\* statement hold no two
distinct inputs with one SHA-512 output (`findColl`). At SHA-512 modelled as a random function with 512-bit outputs,
each pair of distinct inputs collides with probability `2^-512`, so the run has a collision with probability at most
`C²·2^-513` for `C` calls; this is the run's collision event, not a fact of the honest prover. -/
def NoCollAt (w : Claims.Flock.Wit) (ω : Tape (bLAt mask Ir b0 x aux nOf hd recRest))
    (r : Fin ((DAt mask Ir b0 x aux nOf hd recRest).V.k x) → Coin) : Prop :=
  ∀ s < nS x, (Claims.findColl ((DeployedR.layR OpenRows.elemR).calls x s
    ((bLAt mask Ir b0 x aux nOf hd recRest).ref.wit w ω.1
      (((Claims.run FlockDeploy.sha512 (refP (IAt decI).V (bLAt mask Ir b0 x aux nOf hd recRest) w)
        (DAt mask Ir b0 x aux nOf hd recRest).V x ω r).msgs.take ((IAt decI).V.k (xIOf x))).map histOf) s) ++
    Claims.Recursion.VStarB.vcallsAt FlockDeploy.sha512 (DAt mask Ir b0 x aux nOf hd recRest)
      ((Claims.Recursion.runVstar FlockDeploy.sha512 (DAt mask Ir b0 x aux nOf hd recRest) (IAt decI)
        (refP (IAt decI).V (bLAt mask Ir b0 x aux nOf hd recRest) w) x ω r).getD s []))).isSome = false

/-- **The honest run's staging sets up** (`StagedSetsUp` at the honest run's statement `x`), from
`HonestRunAccepted`: the compiled verifier accepting the reference prover's run means each part's recursive setup passed
(`stagedSetsUp_run`). -/
theorem stagedSetsUp_accepted {w : Claims.Flock.Wit} {st : Claims.Recursion.Stmt}
    {ω : Tape (bLAt mask Ir b0 x aux nOf hd recRest)} {r : Fin ((DAt mask Ir b0 x aux nOf hd recRest).V.k x) → Coin}
    (h : HonestRunAccepted mask Ir b0 x aux nOf hd recRest decI w st ω r) :
    ∀ s < nS x, (setupO (staged (bLAt mask Ir b0 x aux nOf hd recRest).st x
      ((Claims.run FlockDeploy.sha512 (refP (IAt decI).V (bLAt mask Ir b0 x aux nOf hd recRest) w)
        (DAt mask Ir b0 x aux nOf hd recRest).V x ω r).msgs.take ((IAt decI).V.k (xIOf x) + kS x)) s)).isSome :=
  stagedSetsUp_run FlockDeploy.sha512 (IAt decI).V _ x _ ω r h.2.2.2

end Premises

end RecDeploy.HonestHalf
