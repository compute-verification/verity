import Proofs.Flock.Claims.Recursion.TopsOpens
import Proofs.Flock.Claims.Recursion.LastColsE
import Proofs.Flock.Claims.Recursion.ConflictTiedE
import Proofs.Flock.Claims.Recursion.LastBindsEC
import Proofs.Flock.Claims.Recursion.CallsAgree
import Proofs.Flock.Claims.Recursion.VStarCross
import Proofs.Flock.Claims.Recursion.SetupCalls
import Proofs.Flock.Claims.Recursion.PartAccepts

/-!
# `KSLin2` and `BitsLin2` at the element-level inner verifier's steps (`FlockDeploy.VE`)

`ksLin2_route_staged` reads three facts of the bound record's steps `nextE` (`StepsBase`): `KSFlockLin` at `DE`, (P4)
modulo what `E*` reads (`LastBindsEC`) and `LastCoinFree`. `ksLin2_steps_staged`, `ksLin2_steps_open`
and `ksLin2_steps_recRead` are `ksLin2_deployed_staged`, `ksLin2_deployed_open` and `ksLin2_deployed_recRead` at any
`nextE` meeting `StepsBase`, and `bitsLin2_steps_recRead` is the last's `BitsLin2On` form. At the deployed plain
verifier's steps `StepsBase` holds (`stepsBase_V`).

At `VE`'s steps (`stepsBase_VE`):

* `LastCoinFree` holds (`lastStepFree_of_VE`), and `LastBindsEC` at SHA-512 with no premise
  (`lastBindsEC_deployedE_sha512`);
* `KSFlockLin` at `DE` holds from the plain record's table bound at the abort law, linearly, at `VE`
  (`TableBoundAbortLin` at `deployPlainE`), as at the deployed plain verifier (`plainE_KSFlockLin`, with `VE`'s code
  facts and `ConflictTied` at `VE`). That table bound is proved at `deployPlain` (`plain_tableBoundAbortLin`) and not
  yet at `VE`: it is the premise `htab`.

So `ksLin2_deployedE_recRead` and `bitsLin2_deployedE_recRead` are the B corollaries at `VE`'s steps, reopened on
`htab`, and `ksLin2_deployedE_partAccepts` and `bitsLin2_deployedE_partAccepts` are those at the coin server's record
(`recTr`), as `ksLin2_deployed_partAccepts` and `bitsLin2_deployed_partAccepts` are at the deployed plain verifier's
steps: they take `DecompB` at `PartAccepts` at `VE`'s bound and `htab`, and nothing else. The transfer: `VE` accepts
every run the deployed plain verifier accepts (`FlockDeploy.run_out_VE`), so the bound record at `VE`'s steps accepts
at least the runs the one at the deployed plain verifier's accepts, and the honest runs inhabit it. That is what moves
the B corollaries to `VE`; no theorem here derives the bound at the deployed plain verifier's steps from the one at
`VE`'s.
-/

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Refine FlockSoundness.Discharge
  FlockSoundness.Discharge.Exec FlockSoundness.Discharge.Tables
open Claims.Bridge.CustodyPlain Claims.Bridge.DecodePlain Claims.Bridge.Custody Claims.Bridge.PlainDraw
open Claims.Bridge.Live

/-- **`KSFlockLin` at the trimmed plain record at `VE`**, at SHA-512 and the deployed caps `CpDep`, from the table
bound at the abort law, linearly, at the untrimmed record at `VE`: `plain_KSFlockLin`'s assembly with `VE`'s code
facts (`plainE_kSplits`, `plainE_hone`, `plainE_colsOk`) and `ConflictTied` at `VE` (`plainE_conflictTied`). -/
theorem plainE_KSFlockLin (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (htab : Claims.Recursion.TableBoundAbortLin sha512 (deployPlainE rel) (wordLen false) (wide false) CpDep) :
    Claims.Recursion.KSFlockLin sha512 ((deployPlainE rel).mapCols trimCol) CpDep :=
  Claims.Recursion.KSFlockLin_of_tableBoundAbortLin sha512 ((deployPlainE rel).mapCols trimCol) (wordLen false)
    (trimWide false) CpDep (wordLen_le_N₀ false) (fun x => εtable_nonneg false x) (fun _ => cap_pos)
    (plainE_kSplits rel) (plainE_hone rel) (trimWide_nil false)
    (Claims.Flock.colsOk_mapCols sha512 (deployPlainE rel) (wordLen false) trimCol (wide false)
      (plainE_colsOk sha512 rel))
    (Claims.Flock.conflictTied_mapCols sha512 (deployPlainE rel) (wordLen false) trimCol (plainE_conflictTied rel))
    (Claims.Recursion.tableBoundAbortLin_mapCols sha512 (deployPlainE rel) (wordLen false) trimCol rfl
      (wide false) CpDep (plainE_colsOk sha512 rel) (wide_trimCol_inj false) htab)

end FlockDeploy.KS

namespace RecDeploy

open Flock
open Claims (Msg Out OC)
open Claims.Recursion (Bound)
open FlockDeploy (Coin)

/-! ## `StepsBase` at `VE` -/

/-- **`StepsBase` at `VE`'s steps**, from the plain record's table bound at the abort law, linearly, at `VE`
(`htab`): `KSFlockLin` at `DE` from `plainE_KSFlockLin`, as `DE` differs from the trimmed plain record at `VE` only in
a weight `KSFlockLin` doesn't read (`deE_VE_trim`); the other two with no premise. -/
theorem stepsBase_VE (wE : Msg → ℝ) (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (w : Msg → ℝ)
    (htab : Claims.Recursion.TableBoundAbortLin FlockDeploy.sha512
      (FlockDeploy.deployPlainE _root_.Flock.Readable.relation) (FlockDeploy.KS.wordLen false)
      (FlockDeploy.KS.wide false) FlockDeploy.KS.CpDep) :
    StepsBase FlockDeploy.VE.next wE join final w :=
  ⟨fun decode hdec => FlockDeploy.KS.plainE_KSFlockLin _root_.Flock.Readable.relation htab decode hdec,
    lastBindsEC_deployedE_sha512 _ wE join final w,
    Bound.lastCoinFree_of_step (lastStepFree_of_VE _ _ rfl)⟩

/-! ## The layers at any steps meeting `StepsBase` -/

/-- **`ksLin2_deployed_staged` at any steps `nextE` meeting `StepsBase`**: `ksLin2_steps_parts` at `layoutBTops`, with
`HeldB` from `heldB_layoutBTops` and `HeldWhole` from `heldWhole_layoutBTops`. -/
theorem ksLin2_steps_staged (nextE : Msg → List Msg → List Coin → Claims.OC Out) (b : Brings)
    (fin : Msg → List (ℕ × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ) (join : List Msg → Msg)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) (hB : StepsBase nextE wE join final w)
    (untr : TrReader) :
    let L := layoutBTops fin
    let Bd := bdDeployed _root_.Flock.Readable.relation nextE
      (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).cols wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation b L Bd.IVStar
    let bL : Brings := { b with rd := L.readers }
    ReadsBackB FlockDeploy.sha512 Bd.IVStar bL (fun _ => True) untr →
    (∀ shape ∈ [Claims.Recursion.Form.Dense],
      FinPieces fin shape ∧
      ∃ chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop,
        (∀ s, Claims.Recursion.VStarB.PartChecksBV FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
          L shape (showsPart untr b) (cmtTr untr) chk s) ∧
        (∀ s, Claims.Recursion.VStarB.PartOpensB FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
          L shape (showsPart untr b) (cmtTr untr) s) ∧
        Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape chk) →
    ReadersTrue FlockDeploy.sha512 Bd.IVStar bL (deployZkRec _root_.Flock.Readable.relation) →
    Claims.Recursion.KSLin2On FlockDeploy.sha512 (recBoundAt _root_.Flock.Readable.relation b L) Bd
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt := by
  intro L Bd D bL hrb hP hRT
  refine ksLin2_steps_parts nextE b L (Claims.Recursion.VStarB.LayoutB.readsRows_of_rowIn fun _ _ => rfl)
    (heldWhole_layoutBTops fin) wE join final w hB untr hrb ?_ hRT
  intro shape hsh
  obtain rfl := List.mem_singleton.1 hsh
  obtain ⟨hF, chk, hPC, hPO, hD⟩ := hP _ hsh
  exact ⟨chk, fun decI => heldB_layoutBTops _ _ _ _ (fun _ => rfl) fin _ schedMatches_dense (capPorts _)
    (rowsCover _) hF, hPC, hPO, hD⟩

/-- **`ksLin2_deployed_open` at any steps `nextE` meeting `StepsBase`**: `ksLin2_steps_staged` with `FinPieces` from
`finPieces_finOpen`. -/
theorem ksLin2_steps_open (nextE : Msg → List Msg → List Coin → Claims.OC Out) (b : Brings)
    (rest : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (hB : StepsBase nextE wE join final w) (untr : TrReader) :
    let L := layoutBTops (finOpen rest)
    let Bd := bdDeployed _root_.Flock.Readable.relation nextE
      (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).cols wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation b L Bd.IVStar
    let bL : Brings := { b with rd := L.readers }
    ReadsBackB FlockDeploy.sha512 Bd.IVStar bL (fun _ => True) untr →
    (∀ shape ∈ [Claims.Recursion.Form.Dense],
      ∃ chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop,
        (∀ s, Claims.Recursion.VStarB.PartChecksBV FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
          L shape (showsPart untr b) (cmtTr untr) chk s) ∧
        (∀ s, Claims.Recursion.VStarB.PartOpensB FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
          L shape (showsPart untr b) (cmtTr untr) s) ∧
        Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape chk) →
    ReadersTrue FlockDeploy.sha512 Bd.IVStar bL (deployZkRec _root_.Flock.Readable.relation) →
    Claims.Recursion.KSLin2On FlockDeploy.sha512 (recBoundAt _root_.Flock.Readable.relation b L) Bd
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt := by
  intro L Bd D bL hrb hP hRT
  exact ksLin2_steps_staged nextE b (finOpen rest) wE join final w hB untr hrb
    (fun shape hsh => ⟨finPieces_finOpen rest shape, hP shape hsh⟩) hRT

/-- **`ksLin2_deployed_recRead` at any steps `nextE` meeting `StepsBase`**: `ksLin2_steps_open` at `recRead`, with
`PartOpensB` from `partOpensB_recRead`. -/
theorem ksLin2_steps_recRead (nextE : Msg → List Msg → List Coin → Claims.OC Out) (b : Brings)
    (rest : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (hB : StepsBase nextE wE join final w) :
    let L := layoutBTops (finOpen rest)
    let Bd := bdDeployed _root_.Flock.Readable.relation nextE
      (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).cols wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation b L Bd.IVStar
    let bL : Brings := { b with rd := L.readers }
    ReadsBackB FlockDeploy.sha512 Bd.IVStar bL (fun _ => True) recRead →
    (∀ shape ∈ [Claims.Recursion.Form.Dense],
      ∃ chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop,
        (∀ s, Claims.Recursion.VStarB.PartChecksBV FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
          L shape (showsPart recRead b) (cmtTr recRead) chk s) ∧
        Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape chk) →
    ReadersTrue FlockDeploy.sha512 Bd.IVStar bL (deployZkRec _root_.Flock.Readable.relation) →
    Claims.Recursion.KSLin2On FlockDeploy.sha512 (recBoundAt _root_.Flock.Readable.relation b L) Bd
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt := by
  intro L Bd D bL hrb hP hRT
  refine ksLin2_steps_open nextE b rest wE join final w hB recRead hrb (fun shape hsh => ?_) hRT
  obtain rfl := List.mem_singleton.1 hsh
  obtain ⟨chk, hPC, hDc⟩ := hP _ hsh
  exact ⟨chk, hPC, fun _ => partOpensB_recRead (finOpen rest) b (fun _ _ h => h) schedMatches_dense, hDc⟩

/-- **`bitsLin2_deployed_recRead` at any steps `nextE` meeting `StepsBase`**: `ksLin2_steps_recRead` through
`bitsLin2On_of_ksLin2On` (`sessionAccepts`). -/
theorem bitsLin2_steps_recRead (nextE : Msg → List Msg → List Coin → Claims.OC Out) (b : Brings)
    (rest : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (hB : StepsBase nextE wE join final w) :
    let L := layoutBTops (finOpen rest)
    let Bd := bdDeployed _root_.Flock.Readable.relation nextE
      (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).cols wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation b L Bd.IVStar
    let bL : Brings := { b with rd := L.readers }
    ReadsBackB FlockDeploy.sha512 Bd.IVStar bL (fun _ => True) recRead →
    (∀ shape ∈ [Claims.Recursion.Form.Dense],
      ∃ chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop,
        (∀ s, Claims.Recursion.VStarB.PartChecksBV FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
          L shape (showsPart recRead b) (cmtTr recRead) chk s) ∧
        Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape chk) →
    ReadersTrue FlockDeploy.sha512 Bd.IVStar bL (deployZkRec _root_.Flock.Readable.relation) →
    Claims.Recursion.BitsLin2On FlockDeploy.sha512 (recBoundAt _root_.Flock.Readable.relation b L) Bd
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt := by
  intro L Bd D bL hrb hP hRT
  exact Claims.Recursion.bitsLin2On_of_ksLin2On
    (sessionAccepts Bd.IVStar (FlockDeploy.deployPlain _root_.Flock.Readable.relation).scopeCheck bL)
    (ksLin2_steps_recRead nextE b rest wE join final w hB hrb hP hRT)

/-! ## The B corollaries at `VE` -/

/-- **`KSLin2` at the deployment's dense statements, at `layoutBTops (finOpen rest)`, the coin server's record and
`VE`'s steps**: `ksLin2_steps_recRead` at `stepsBase_VE`. Beside `ksLin2_deployed_recRead`'s open premises
(`PartChecksBV` at `showsPart recRead`, `DecompB`, the readers' `ReadersTrue` and `ReadsBackB`) it takes the plain
record's table bound at the abort law at `VE` (`htab`). `VE` accepts every run the deployed plain verifier accepts
(`FlockDeploy.run_out_VE`), so the inner verifier here accepts the honest runs; this is not a theorem at the deployed
plain verifier's steps. This bounds no number by itself: its premises are open. -/
theorem ksLin2_deployedE_recRead (b : Brings)
    (rest : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (htab : Claims.Recursion.TableBoundAbortLin FlockDeploy.sha512
      (FlockDeploy.deployPlainE _root_.Flock.Readable.relation) (FlockDeploy.KS.wordLen false)
      (FlockDeploy.KS.wide false) FlockDeploy.KS.CpDep) :
    let L := layoutBTops (finOpen rest)
    let Bd := bdDeployed _root_.Flock.Readable.relation FlockDeploy.VE.next
      (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).cols wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation b L Bd.IVStar
    let bL : Brings := { b with rd := L.readers }
    ReadsBackB FlockDeploy.sha512 Bd.IVStar bL (fun _ => True) recRead →
    (∀ shape ∈ [Claims.Recursion.Form.Dense],
      ∃ chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop,
        (∀ s, Claims.Recursion.VStarB.PartChecksBV FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
          L shape (showsPart recRead b) (cmtTr recRead) chk s) ∧
        Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape chk) →
    ReadersTrue FlockDeploy.sha512 Bd.IVStar bL (deployZkRec _root_.Flock.Readable.relation) →
    Claims.Recursion.KSLin2On FlockDeploy.sha512 (recBoundAt _root_.Flock.Readable.relation b L) Bd
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt :=
  ksLin2_steps_recRead _ b rest wE join final w (stepsBase_VE wE join final w htab)

/-- **`BitsLin2` at the deployment's dense statements, at `layoutBTops (finOpen rest)`, the coin server's record and
`VE`'s steps**: `bitsLin2_steps_recRead` at `stepsBase_VE`. It reads `StrictBirthday` and `ExpectedCR`, the named
SHA-512 collision assumptions, and `ksLin2_deployedE_recRead`'s open premises, `htab` among them: a bound on a number
under named premises, not a proved security level. -/
theorem bitsLin2_deployedE_recRead (b : Brings)
    (rest : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (htab : Claims.Recursion.TableBoundAbortLin FlockDeploy.sha512
      (FlockDeploy.deployPlainE _root_.Flock.Readable.relation) (FlockDeploy.KS.wordLen false)
      (FlockDeploy.KS.wide false) FlockDeploy.KS.CpDep) :
    let L := layoutBTops (finOpen rest)
    let Bd := bdDeployed _root_.Flock.Readable.relation FlockDeploy.VE.next
      (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).cols wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation b L Bd.IVStar
    let bL : Brings := { b with rd := L.readers }
    ReadsBackB FlockDeploy.sha512 Bd.IVStar bL (fun _ => True) recRead →
    (∀ shape ∈ [Claims.Recursion.Form.Dense],
      ∃ chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop,
        (∀ s, Claims.Recursion.VStarB.PartChecksBV FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
          L shape (showsPart recRead b) (cmtTr recRead) chk s) ∧
        Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape chk) →
    ReadersTrue FlockDeploy.sha512 Bd.IVStar bL (deployZkRec _root_.Flock.Readable.relation) →
    Claims.Recursion.BitsLin2On FlockDeploy.sha512 (recBoundAt _root_.Flock.Readable.relation b L) Bd
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt :=
  bitsLin2_steps_recRead _ b rest wE join final w (stepsBase_VE wE join final w htab)

/-! ## The B corollaries at `VE`, from `DecompB` and `htab` -/

/-- **`KSLin2` at the deployment's dense statements, at `layoutBTops (finOpen rest)`, the coin server's record
(`recTr`) and `VE`'s steps, from `DecompB` and `htab`**: `ksLin2_deployedE_recRead` at the brings whose transcript is
`recTr aux nOf hd recRest`, with its inner check `PartAccepts` (`exists_chk_iff`), `ReadsBackB` from `readsBack_recTr`
and `ReadersTrue` from `readersTrue_layoutBTops`, as `ksLin2_deployed_partAccepts` is at the deployed plain
verifier's steps. What stays open: `DecompB` at what each part's circuit accepts, at `VE`'s bound, and the plain
record's table bound at the abort law at `VE` (`htab`). This bounds no number by itself: those premises are open. -/
theorem ksLin2_deployedE_partAccepts (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (rest : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (htab : Claims.Recursion.TableBoundAbortLin FlockDeploy.sha512
      (FlockDeploy.deployPlainE _root_.Flock.Readable.relation) (FlockDeploy.KS.wordLen false)
      (FlockDeploy.KS.wide false) FlockDeploy.KS.CpDep) :
    let bR : Brings := { b with tr := recTr aux nOf hd recRest }
    let L := layoutBTops (finOpen rest)
    let Bd := bdDeployed _root_.Flock.Readable.relation FlockDeploy.VE.next
      (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).cols wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation bR L Bd.IVStar
    (∀ shape ∈ [Claims.Recursion.Form.Dense],
      Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape
        (Claims.Recursion.VStarB.PartAccepts FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation) L
          shape (showsPart recRead bR) (cmtTr recRead))) →
    Claims.Recursion.KSLin2On FlockDeploy.sha512 (recBoundAt _root_.Flock.Readable.relation bR L) Bd
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt := by
  intro bR L Bd D hD
  exact ksLin2_deployedE_recRead bR rest wE join final w htab
    (readsBack_recTr _ _ { bR with rd := L.readers } _ aux nOf hd recRest rfl)
    (fun shape hsh => (Claims.Recursion.VStarB.exists_chk_iff _ _ _ _ _ _ _ _).2 (hD shape hsh))
    (readersTrue_layoutBTops (finOpen rest) Bd.IVStar bR _ hmSlotsCommit_of_full)

/-- **`BitsLin2` at the deployment's dense statements, at `layoutBTops (finOpen rest)`, the coin server's record
(`recTr`) and `VE`'s steps, from `DecompB` and `htab`**: `bitsLin2_deployedE_recRead` at `recTr aux nOf hd recRest`,
with its inner check `PartAccepts`, `ReadsBackB` from `readsBack_recTr` and `ReadersTrue` from
`readersTrue_layoutBTops`. It reads `StrictBirthday` and `ExpectedCR`, the named SHA-512 collision assumptions, and
`ksLin2_deployedE_partAccepts`'s open premises: a bound on a number under named premises, not a proved security
level. -/
theorem bitsLin2_deployedE_partAccepts (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (rest : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (htab : Claims.Recursion.TableBoundAbortLin FlockDeploy.sha512
      (FlockDeploy.deployPlainE _root_.Flock.Readable.relation) (FlockDeploy.KS.wordLen false)
      (FlockDeploy.KS.wide false) FlockDeploy.KS.CpDep) :
    let bR : Brings := { b with tr := recTr aux nOf hd recRest }
    let L := layoutBTops (finOpen rest)
    let Bd := bdDeployed _root_.Flock.Readable.relation FlockDeploy.VE.next
      (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).cols wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation bR L Bd.IVStar
    (∀ shape ∈ [Claims.Recursion.Form.Dense],
      Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape
        (Claims.Recursion.VStarB.PartAccepts FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation) L
          shape (showsPart recRead bR) (cmtTr recRead))) →
    Claims.Recursion.BitsLin2On FlockDeploy.sha512 (recBoundAt _root_.Flock.Readable.relation bR L) Bd
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt := by
  intro bR L Bd D hD
  exact bitsLin2_deployedE_recRead bR rest wE join final w htab
    (readsBack_recTr _ _ { bR with rd := L.readers } _ aux nOf hd recRest rfl)
    (fun shape hsh => (Claims.Recursion.VStarB.exists_chk_iff _ _ _ _ _ _ _ _).2 (hD shape hsh))
    (readersTrue_layoutBTops (finOpen rest) Bd.IVStar bR _ hmSlotsCommit_of_full)

end RecDeploy
