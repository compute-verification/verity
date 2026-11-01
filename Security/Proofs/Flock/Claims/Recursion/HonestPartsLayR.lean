import Proofs.Flock.Claims.Recursion.HonestPartsR
import Proofs.Flock.Claims.Recursion.DeployHonest
import Proofs.Flock.Claims.Recursion.HonestWitR
import Proofs.Flock.Claims.Recursion.DeployedR
import Proofs.Flock.Claims.Recursion.SlotsCheck
import Proofs.Flock.Claims.Recursion.CapPorts
import Proofs.Flock.Claims.Recursion.RowsCover
import Proofs.Flock.Claims.Recursion.SchedDense

/-!
# What each V\* part's circuit accepts on the honest outer run, at the layout of record (`DeployedR.layR`)

`RecursiveKSAt`'s premise is `DecompB` at `PartAccepts`, over V\*'s layout of record `DeployedR.layR elem`.
 It is not vacuous only if `PartAccepts` holds where the honest recursive prover runs. Here the honest run is
the compiled record's own reference prover (`refP`, the record's `P`) at the deployed recursive record
`recBound Readable.relation bR (layR elem) bdR.IVStar` (`bR` the brings whose transcript is the coin server's record,
`recTr`), on a statement `x` in the recursion's relation at the inner witness `w`, and the part's outer witness is the
one `refP` hands its masked session (`RefParts.wit`, at the inner rounds as the prover saw them).

`honest_partAccepts_layR` is `partAccepts_run_deployRec` there, with these of its premises proved:

* `ReadsBackB` (`hrb`): `readsBack_recTr`, the coin server's record read back by `recRead`;
* `HeldB` (`hHeld`): `heldB_layoutBTops` at `finOpen (restR elem)` (`heldB_layRH`: `layR` changes only the calls);
* the run's messages (`hms`): `ms` is the run's;
* `Flock.check` at the part's V\* statement (`hck`): the honest outer witness is in the V\* relation there
  (`vstar_complete_run_setup_at`, `check_of_R`);
* the recorded calls are true (`hT`), at every call `layoutBTops` records: the hm96 slots output their commit strings at
  the outer relation (`hmSlotsCommit_of_full`, `layoutBTops_calls_true`) at the run's staged statement
  (`stagedPart_run`, `runVstar_getD_staged`).

What it still takes is about the honest run itself: the builder's facts (`NumsFixed`, and `HonestWitSatisfies` at the
run, as in `vstar_complete_run_setup_at`; the staging setting up follows from the run accepting), the statement (in
the relation, dense), the run (accepted), the calls `layR` records beyond `layoutBTops`' (`moreCalls`: the output rows' and
check 5's) being true at the honest witness, no collision among the honest run's calls, and the honest witness holding
the run's messages where the layout places them (held rounds, pieces, shared rows).
-/

namespace RecDeploy

open Flock
open Claims (Msg Out)
open FlockDeploy (Coin)
open Claims.Recursion.VStarB (readForms)

/-- **Every call `layoutBTops` records at part `s` is a true SHA-512 evaluation**, at a witness `Flock.check` accepts
at the part's staged statement, wherever every hm96 slot outputs its commit string there (`HmSlotsCommit`): the
per-part form of `readersTrue_layoutBTops`. -/
theorem layoutBTops_calls_true {Col T : Type} (fin : Msg → List (ℕ × List (Msg × List HmNets.Lin)))
    {DO : Claims.Flock.Deploy Coin Col T} {st : Stage} (h : HmSlotsCommit DO st) {x : Msg} {ms : List (Msg × Out)}
    {s : ℕ} {w : Claims.Flock.Wit} (hsg : StagedPart st s x (staged st x ms s))
    (hck : Claims.Flock.check DO.parse (staged st x ms s) w = true) :
    ∀ e ∈ (layoutBTops fin).calls x s w, FlockDeploy.sha512 e.1 = e.2 := by
  intro e he
  obtain ⟨p, hp, rfl⟩ := List.mem_map.1 he
  cases hr : readX x with
  | none => simp [layoutBTops, layoutBDeployed, hr] at hp
  | some r =>
    have hp' : p ∈ layoutBDeployed.shaAt x s ∨ p ∈ capsCalls r s := by
      simpa [layoutBTops, hr] using hp
    rcases hp' with hp | hp
    · cases hpt : r.w.parts[s]? with
      | none => simp [layoutBDeployed, hr, hpt] at hp
      | some pt =>
        cases hc : partCircuit pt with
        | none => simp [layoutBDeployed, hr, hpt, hc] at hp
        | some c =>
          simp only [layoutBDeployed, hr, hpt, hc, Option.bind_some, Option.map_some, Option.getD_some,
            List.mem_flatMap] at hp
          obtain ⟨hh, hmem, hp⟩ := hp
          have hc' : r.w.parts[s]?.bind partCircuit = some c := by rw [hpt]; exact hc
          exact hmCalls_true ((h x _ s w r c hsg hck hr hc').1 hh hmem) p hp
    · obtain ⟨l, N, c, q, n, hst, hc, hq, hn, hp⟩ := mem_capsCalls hp
      exact hmCalls_true ((h x _ s w r c hsg hck hr hc).2 l N hst q hq n hn) p hp

/-- **`HeldB` at the layout of record**: `heldB_layoutBTops` at `finOpen (restR elem)`, `layR` changing only the calls
it records. -/
theorem heldB_layRH {W : Type} (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (IV : Claims.Verifier Coin) (ivScope : Msg → Bool) (b : Brings) (I : Claims.Recursion.Inner Coin W)
    (hk : ∀ x, I.V.k x = (FlockDeploy.V false).k x) :
    Claims.Recursion.VStarB.HeldB (deployRec IV ivScope b) I (DeployedR.layR elem) Claims.Recursion.Form.Dense :=
  fun x st hst hf =>
    heldB_layoutBTops IV ivScope b I hk (finOpen (DeployedR.restR elem)) _ schedMatches_dense (capPorts _)
      (rowsCover _) (finPieces_finOpen (DeployedR.restR elem) _) x st hst hf

/-- **The run's V\* statement is the staged one**, at an accepted run, where the stager's numbers are fixed: the run
stages part `s` (`stagedPart_run`), so its statement sets up and `vstarAt` takes it (`vstarAt_staged`). -/
theorem runVstar_getD_staged {W : Type} (H : Msg → Msg) (I : Claims.Recursion.Inner Coin W) (ivScope : Msg → Bool)
    (b : Brings) (hN : NumsFixed b.st) {x : Msg} {Ω : Type} (P : Claims.Prover Ω) (ω : Ω)
    (r : Fin ((V I.V b.st).k x) → Coin) (hacc : (Claims.run H P (V I.V b.st) x ω r).out = .acc) {s : ℕ}
    (hs : s < nS x) :
    (Claims.Recursion.runVstar H (deployRec I.V ivScope b) I P x ω r).getD s [] =
      staged b.st x ((Claims.run H P (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x) + kS x)) s := by
  obtain ⟨_, i, spec, sts, -, hsu, -⟩ := stagedPart_run H I.V b.st x P ω r hacc hs
  show (vstar b.st x ((Claims.run H P (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x) + kS x))).getD s [] = _
  rw [vstar_getD _ _ _ hs]
  exact vstarAt_staged hN (by rw [hsu]; rfl)

/-- **Part `s`'s circuit accepts on the honest outer run, at the layout of record** (non-vacuity of `RecursiveKSAt`'s
`DecompB` premise, the parts' half). At the deployed recursive record over `DeployedR.layR elem` with the coin server's
record (`recTr`), on the record's reference prover's run (`refP`) at a statement `x` in the recursion's relation read
at the dense form, where that run accepts, `PartAccepts` holds at the
run's messages and coins, its outer witness the one the reference prover hands part `s`'s session. It takes what is
about the honest run and its witness: the builder's facts (`hN`, and `hW` at the run), the statement (`hR`, `hx`,
`hf`), the run (`hacc`), the calls `layR` records beyond `layoutBTops`' (`hTm`), no collision among the run's calls
(`hnc`), and the witness holding the run's messages where the layout places them (`hh`, `hp`, `hr`). -/
theorem honest_partAccepts_layR (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ) (hd : Msg → Flock.RecRecord.Record)
    (recRest : Msg → List (Msg × Out) → List UInt8)
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace) :
    let bR : Brings := { b with tr := recTr aux nOf hd recRest }
    let L := DeployedR.layR elem
    let bL : Brings := { bR with rd := L.readers }
    let I := Claims.Recursion.Inner.ofFlock DeployedR.bdR.deploy decI
    let D := recBound _root_.Flock.Readable.relation bR L DeployedR.bdR.IVStar
    NumsFixed bL.st →
    ∀ {x : Msg} {w : Claims.Flock.Wit} {st : Claims.Recursion.Stmt}, Claims.Recursion.R D I x w →
      D.parse x = some st → Claims.Recursion.Form.Dense st.form →
    ∀ (ω : Tape bL) (r : Fin (D.V.k x) → Coin),
      (Claims.run FlockDeploy.sha512 (refP I.V bL w) D.V x ω r).out = .acc →
      HonestWitSatisfies FlockDeploy.sha512 I.V bL x w ω r →
    ∀ s < nS x,
      let wO := bL.ref.wit w ω.1
        (((Claims.run FlockDeploy.sha512 (refP I.V bL w) D.V x ω r).msgs.take (I.V.k (xIOf x))).map histOf) s
      let ms := (Claims.run FlockDeploy.sha512 (refP I.V bL w) D.V x ω r).msgs.map Prod.fst
      (∀ xr, readX x = some xr → ∀ p ∈ DeployedR.moreCalls xr s,
        FlockDeploy.sha512 (p.1 ++ readForms wO p.2.1) = readForms wO p.2.2) →
      (Claims.findColl (L.calls x s wO ++ Claims.Recursion.VStarB.vcallsAt FlockDeploy.sha512 D
        ((Claims.Recursion.runVstar FlockDeploy.sha512 D I (refP I.V bL w) x ω r).getD s []))).isSome = false →
      (∀ i < L.rounds x, ∀ m, L.held x s wO i = some m → ms.getD i [] = m) →
      (∀ i < L.rounds x, ∀ o m, (o, m) ∈ L.piecesOf x s wO i → ((ms.getD i []).drop o).take m.length = m) →
      L.RowsRead x s wO ms →
      Claims.Recursion.VStarB.PartAccepts FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation) L
        Claims.Recursion.Form.Dense (showsPart recRead bR) (cmtTr recRead) s x (Claims.coinSeq r) ms := by
  intro bR L bL I D hN x w st hR hx hf ω r hacc hW s hs wO ms hTm hnc hh hp hr
  have hrb : ReadsBackB FlockDeploy.sha512 I.V bL Claims.Recursion.Form.Dense recRead :=
    readsBack_recTr _ _ bL _ aux nOf hd recRest rfl
  have hHeld : Claims.Recursion.VStarB.HeldB (deployRec I.V (FlockDeploy.deployPlain
      _root_.Flock.Readable.relation).scopeCheck bL) I L Claims.Recursion.Form.Dense :=
    heldB_layRH elem I.V _ bL I fun _ => rfl
  obtain ⟨-, -, -, -, hrel, -⟩ := vstar_complete_run_setup_at FlockDeploy.sha512 I
    (FlockDeploy.deployPlain _root_.Flock.Readable.relation).scopeCheck bL hN hR ω r hacc hW hs
  have hck := check_of_R hrel
  have hsg := stagedPart_run FlockDeploy.sha512 I.V bL.st x (refP I.V bL w) ω r hacc hs
  have heq := runVstar_getD_staged FlockDeploy.sha512 I
    (FlockDeploy.deployPlain _root_.Flock.Readable.relation).scopeCheck bL hN (refP I.V bL w) ω r hacc hs
  have hT : ∀ e ∈ L.calls x s wO, FlockDeploy.sha512 e.1 = e.2 := by
    intro e he
    have he' : e ∈ (DeployedR.layR elem).calls x s wO := he
    rw [DeployedR.layR_calls] at he'
    rcases List.mem_append.1 he' with he | he
    · exact layoutBTops_calls_true _ hmSlotsCommit_of_full hsg (by rw [← heq]; exact hck) e he
    · obtain ⟨p, hp, rfl⟩ := List.mem_map.1 he
      cases hxr : readX x with
      | none => simp [hxr] at hp
      | some xr =>
        rw [hxr, Option.map_some, Option.getD_some] at hp
        exact hTm xr hxr p hp
  exact partAccepts_run_deployRec FlockDeploy.sha512 I
    (FlockDeploy.deployPlain _root_.Flock.Readable.relation).scopeCheck bL recRead
    (deployZkRec _root_.Flock.Readable.relation) L Claims.Recursion.Form.Dense hrb hHeld hx hf hacc hs
    (fun _ _ => rfl) hck hT hnc hh hp hr

end RecDeploy
