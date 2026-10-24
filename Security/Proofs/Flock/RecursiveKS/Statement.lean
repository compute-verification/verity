import Proofs.Flock.Claims.Recursion.PartAccepts

/-!
# The recursion's knowledge soundness at the compiled V\* (trusted)

`Flock.Guarantees.RecursiveKS` is a `def G : Prop` in the form of `Flock.Guarantees.MaskedKS`: route 2C's knowledge
soundness (`Claims.Recursion.KSLin2`, at the dense statements: `KSLin2On` at `RecDeploy.DenseAt`) at the compiled
recursive verifier, at SHA-512. Its proof is `Flock.SecurityProofs.RecursiveKS : Flock.Guarantees.RecursiveKS`
(`Proofs.Flock.RecursiveKS`).

It takes three open obligations, each a named `Prop` of this module that the lock lists under `assumptions`: the coin
server's transcript reads back (`Flock.Assumptions.RecKSReadsBack`), V\*'s parts decompose the inner verifier
(`Flock.Assumptions.RecKSDecomp`) and V\*'s witnesses record only true evaluations (`Flock.Assumptions.RecKSReadersTrue`).
Each is a fact about our own code (V\*'s circuits, the readers and the coin server's record), so each is an
obligation, not an assumption about the world.

The compiled verifier, its bound layout and the deployed records (`RecDeploy.bdDeployed`, `recBoundAt`, `deployZkRec`,
`layoutBTops`, `recRead`) are still in C-Flock's proofs package, so this module imports them from
`Proofs.Flock.Claims`, and `Models.lean`, which the `security` package builds without `security_proofs`, does not
import it: `Proofs.Flock.RecursiveKS` does.
-/

namespace RecDeploy

open Flock
open Claims (Msg)
open FlockDeploy (Coin)

/-- Obligation: discharged wherever the brings' transcript is the model of the coin server's record
(`RecDeploy.readsBack_recTr`, at `recTr`); `test_lean_rec_record.py` ties `recTr` to the coin server's bytes on the
honest fixtures.

**`rec-ks-reads-back`: the coin server's transcript reads back.** At the compiled verifier whose inner view runs the
deployed bound record's `IVStar` (`bdDeployed` at the deployed relation, the deployed plain verifier's steps and
columns, `rowInDeployed`, and the code side's `wE`, `join`, `final`, `w`), the brings `b` with V\*'s bound layout's
readers (`layoutBTops (finOpen rest)`): at every compiled statement and every accepted run, the record's reader
`recRead` reads the brings' transcript `b.tr` of the run's inner rounds back to those rounds' prover messages before the
last, each with the values its round's coin drew (`ReadsBackB`). -/
def _root_.Flock.Assumptions.RecKSReadsBack (b : Brings)
    (rest : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) : Prop :=
  ReadsBackB FlockDeploy.sha512
    (bdDeployed _root_.Flock.Readable.relation (FlockDeploy.V false).next
      (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).cols wE join rowInDeployed final w).IVStar
    { b with rd := (layoutBTops (finOpen rest)).readers } (fun _ => True) recRead

/-- Obligation: open. No Lean lemma proves it, and no honest witness of it is in Lean: it is what V\*'s units must
compute, each its share of the inner verifier's checks.

**`rec-ks-decomp`: V\*'s parts decompose the inner verifier.** At the compiled record `recBound` of the brings `b`, V\*'s
bound layout `L = layoutBTops (finOpen rest)` and the deployed bound record's `IVStar` (as in
`Flock.Assumptions.RecKSReadsBack`), at every dense compiled statement (`Form.Dense`: neither Holo nor sparse):
wherever every part's circuit accepts at a witness that agrees with the readers' messages (`VStarB.PartAccepts`, at
the deployed outer `--zk` record `deployZkRec` and the coins the staged V\* statements show through `recRead`,
`showsPart recRead b`), `IVStar` accepts those messages at those coins, or two of the parts' recorded calls collide
(`VStarB.DecompB`). It is the pair `PartChecksB` and `DecompB` at its least inner check (`VStarB.exists_chk_iff`). -/
def _root_.Flock.Assumptions.RecKSDecomp (b : Brings)
    (rest : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) : Prop :=
  Claims.Recursion.VStarB.DecompB FlockDeploy.sha512
    (recBound _root_.Flock.Readable.relation b (layoutBTops (finOpen rest))
      (bdDeployed _root_.Flock.Readable.relation (FlockDeploy.V false).next
        (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).cols wE join rowInDeployed final w).IVStar)
    (bdDeployed _root_.Flock.Readable.relation (FlockDeploy.V false).next
      (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).cols wE join rowInDeployed final w).IVStar
    (layoutBTops (finOpen rest)) Claims.Recursion.Form.Dense
    (Claims.Recursion.VStarB.PartAccepts FlockDeploy.sha512
      (recBound _root_.Flock.Readable.relation b (layoutBTops (finOpen rest))
        (bdDeployed _root_.Flock.Readable.relation (FlockDeploy.V false).next
          (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).cols wE join rowInDeployed final w).IVStar)
      (deployZkRec _root_.Flock.Readable.relation) (layoutBTops (finOpen rest)) Claims.Recursion.Form.Dense
      (showsPart recRead b))

/-- Obligation: open. No Lean lemma proves it at `IVStar`; its companion `ReadersLogCollS` is proved there
(`readersLogCollS_ivstar`).

**`rec-ks-readers-true`: V\*'s witnesses record only true evaluations.** At the deployed bound record's `IVStar` and
the brings `b` with V\*'s bound layout's readers (as in `Flock.Assumptions.RecKSReadsBack`): outer witnesses in the
deployed outer `--zk` record's relation (`deployZkRec`) at the V\* statements staged at any inner and staging messages
and coins, each part staged as V\* stages it, record only true evaluations of SHA-512 (`ReadersTrue`). -/
def _root_.Flock.Assumptions.RecKSReadersTrue (b : Brings)
    (rest : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) : Prop :=
  ReadersTrue FlockDeploy.sha512
    (bdDeployed _root_.Flock.Readable.relation (FlockDeploy.V false).next
      (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).cols wE join rowInDeployed final w).IVStar
    { b with rd := (layoutBTops (finOpen rest)).readers } (deployZkRec _root_.Flock.Readable.relation)

/-- **The recursion is knowledge sound at the compiled V\*, at its dense statements** (route 2C, `KSLin2`), at SHA-512.
For every brings `b` (the coin server's transcript, the stager, the reference prover's pieces), every layout of the
last round's pieces `rest`, and every choice of the code side's `wE`, `join`, `final` and `w`, take the compiled
record whose inner view runs `IVStar` of the deployed bound record (`bdDeployed` at the deployed relation
`Flock.Readable.relation`, the deployed plain verifier's steps and columns and `rowInDeployed`), on V\*'s bound layout
`layoutBTops (finOpen rest)` (`recBoundAt`), with the deployed outer `--zk` record (`deployZkRec`) and the deployed
inner caps (`FlockDeploy.KS.CpDep`). Under the three named obligations at those records (`RecKSReadsBack`,
`RecKSDecomp`, `RecKSReadersTrue`), at every inner and outer decoder that meets the list-decoding spec, every dense
compiled statement `x` (`DenseAt`: every parse of `x` has a dense inner form, neither Holo nor sparse), every prover
and every `M`, `R`:
$$\Pr[\text{run} = \text{acc} \wedge \neg R(x, E^*(x))] \le \kappa(x) + \sqrt{\gamma(x)\cdot\Pr[B]} +
  (4k_I + 1)\Pr[B_I] + 4W_I\big(\Pr[F_{CI}] + \tfrac{1}{M+1}\big) + 4\big(\Pr[F_{0I}] + \tfrac{k_I+1}{R+1}\big) +
  \beta_2\big(\Pr[F_{dev}] + \tfrac{1}{M+1}\big),$$
with `E*` the recursion's extractor (`Claims.Recursion.E`: the outer extractor, then the inner Flock's `E^rep`), `R`
the recursion's relation (`x` parses, with no circuit commitment, and the inner Flock's relation holds of the witness at
its inner statement), `κ`, `γ` the compiled record's, `k_I` the inner verifier's rounds,
`W_I` the inner caps' positions, `β₂ = βdev2`, `B` the outer collision finder and `B_I`, `F_CI`, `F_0I`, `F_dev` the
finders over the inner verifier's own calls with the recorded ones (`Deploy.withSetup`). Every `Pr[·]` of a finder is
the probability that it outputs a SHA-512 collision; it assumes no hash.

What it does not say: nothing at a Holo statement (candidate 3's private inner circuits) or a sparse one, where
`DenseAt` fails; no bits (`BitsLin2On` follows, `bitsLin2On_of_ksLin2On`, reading `StrictBirthday` and `ExpectedCR`,
and route 2C's `λ₂` is at least 100 at the deployed records, `lamLin2_deployed_ge_dep`); and nothing where an
obligation fails. `RecKSDecomp` and `RecKSReadersTrue` are open, with no honest witness in Lean. -/
def _root_.Flock.Guarantees.RecursiveKS : Prop :=
  ∀ (b : Brings) (rest : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ),
    Flock.Assumptions.RecKSReadsBack b rest wE join final w →
    Flock.Assumptions.RecKSDecomp b rest wE join final w →
    Flock.Assumptions.RecKSReadersTrue b rest wE join final w →
    Claims.Recursion.KSLin2On FlockDeploy.sha512
      (recBoundAt _root_.Flock.Readable.relation b (layoutBTops (finOpen rest)))
      (bdDeployed _root_.Flock.Readable.relation (FlockDeploy.V false).next
        (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).cols wE join rowInDeployed final w)
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt

end RecDeploy
