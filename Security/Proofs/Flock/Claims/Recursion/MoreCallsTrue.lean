import Proofs.Flock.Claims.Recursion.MoreSlotsAll
import Proofs.Flock.Claims.Recursion.HonestPartsLayR

/-!
# The extra calls `layR` records are true at any staged part (`hTm`)

`DeployedR.layR` records, beside `layoutBTops`' calls, each output row's (`outsCalls`), check 5's (`valCalls`) and the
`rec-acc` chain's (`accCalls`) hm96 calls. At any witness `Flock.check` accepts at a staged part, each is a true SHA-512
evaluation (`moreCalls_true`): CAPLOWER's `MoreSlotsTrue` at `deployZkRec` (`moreSlotsTrue_deployed`, off `pastIn` and
`accFits`), at one part. No witness builder enters. So `honest_partAccepts_layR` holds without its `hTm`
(`honest_partAccepts_layR_of`).
-/

namespace RecDeploy

open Flock
open Claims (Msg Out)
open FlockDeploy (Coin)
open Claims.Recursion.VStarB (readForms)

/-- **The calls `layR` records beyond `layoutBTops`' are true at a staged part**, at any witness `Flock.check` accepts
there: `MoreSlotsTrue` at `deployZkRec`, at part `s`. -/
theorem moreCalls_true {st : Stage} {s : ℕ} {x : Msg} {ms : List (Msg × Out)} {w : Claims.Flock.Wit}
    (hsg : StagedPart st s x (staged st x ms s))
    (hck : Claims.Flock.check (deployZkRec _root_.Flock.Readable.relation).parse (staged st x ms s) w = true) :
    ∀ xr, readX x = some xr → ∀ p ∈ DeployedR.moreCalls xr s,
      FlockDeploy.sha512 (p.1 ++ readForms w p.2.1) = readForms w p.2.2 :=
  fun xr hxr => MoreSlots.moreSlotsTrue_deployed st x ms s w xr hsg hck hxr

/-- **`honest_partAccepts_layR` without `hTm`**: the extra calls are true at the outer witness of the honest run
(`moreCalls_true` at the run's staged part, `stagedPart_run`, where `Flock.check` accepts its outer witness). -/
theorem honest_partAccepts_layR_of (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ) (hd : Msg → Flock.RecRecord.Record)
    (recRest : Msg → List (Msg × Out) → List UInt8)
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace) :
    let bR : Brings := { b with tr := recTr aux nOf hd recRest }
    let L := DeployedR.layR elem
    let bL : Brings := { bR with rd := L.readers }
    let I := Claims.Recursion.Inner.ofFlock DeployedR.bdR.deploy decI
    let D := recBound _root_.Flock.Readable.relation bR L DeployedR.bdR.IVStar
    NumsFixed bL.st → StagedSetsUp FlockDeploy.sha512 I.V bL → WitSatisfies FlockDeploy.sha512 I.V bL →
    ∀ {x : Msg} {w : Claims.Flock.Wit} {st : Claims.Recursion.Stmt}, Claims.Recursion.R D I x w →
      D.parse x = some st → Claims.Recursion.Form.Dense st.form →
    ∀ (ω : Tape bL) (r : Fin (D.V.k x) → Coin),
      (Claims.run FlockDeploy.sha512 (refP I.V bL w) D.V x ω r).out = .acc →
      InnerAccepts FlockDeploy.sha512 I.V x (runCs (I.V.k (xIOf x) + kS x) r)
        (((Claims.run FlockDeploy.sha512 (refP I.V bL w) D.V x ω r).msgs.take (I.V.k (xIOf x) + kS x)).map
          histOf) →
    ∀ s < nS x,
      let wO := bL.ref.wit w ω.1
        (((Claims.run FlockDeploy.sha512 (refP I.V bL w) D.V x ω r).msgs.take (I.V.k (xIOf x))).map histOf) s
      let ms := (Claims.run FlockDeploy.sha512 (refP I.V bL w) D.V x ω r).msgs.map Prod.fst
      (Claims.findColl (L.calls x s wO ++ Claims.Recursion.VStarB.vcallsAt FlockDeploy.sha512 D
        ((Claims.Recursion.runVstar FlockDeploy.sha512 D I (refP I.V bL w) x ω r).getD s []))).isSome = false →
      (∀ i < L.rounds x, ∀ m, L.held x s wO i = some m → ms.getD i [] = m) →
      (∀ i < L.rounds x, ∀ o m, (o, m) ∈ L.piecesOf x s wO i → ((ms.getD i []).drop o).take m.length = m) →
      L.RowsRead x s wO ms →
      Claims.Recursion.VStarB.PartAccepts FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation) L
        Claims.Recursion.Form.Dense (showsPart recRead bR) (cmtTr recRead) s x (Claims.coinSeq r) ms := by
  intro bR L bL I D hN hS hW x w st hR hx hf ω r hacc hI s hs wO ms hnc hh hp hr
  obtain ⟨-, -, -, -, hrel, -⟩ := vstar_complete_run_setup FlockDeploy.sha512 I
    (FlockDeploy.deployPlain _root_.Flock.Readable.relation).scopeCheck bL hN hS hW hR ω r hI hs
  have hck := check_of_R hrel
  have hsg := stagedPart_run FlockDeploy.sha512 I.V bL.st x (refP I.V bL w) ω r hacc hs
  have heq := runVstar_getD_staged FlockDeploy.sha512 I
    (FlockDeploy.deployPlain _root_.Flock.Readable.relation).scopeCheck bL hN (refP I.V bL w) ω r hacc hs
  exact honest_partAccepts_layR elem b aux nOf hd recRest decI hN hS hW hR hx hf ω r hacc hI s hs
    (moreCalls_true hsg (by rw [← heq]; exact hck)) hnc hh hp hr

end RecDeploy
