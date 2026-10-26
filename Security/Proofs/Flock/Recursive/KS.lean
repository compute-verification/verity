import Proofs.Flock.Recursive.KSStatement
import Proofs.Flock.Claims.Recursion.NonVacuous
import Proofs.Flock.Claims.KS.PlainRConflict
import Proofs.Flock.Claims.KS.PlainAccPrR
import Proofs.Flock.Claims.KS.PlainRCapPrefix
import Proofs.Flock.Claims.Recursion.LastCallsR

/-!
# `RecursiveKSAt` from its open premises

`recursiveKSAt_of` is `ksLin2_deployedR_partAccepts` at the deployed join (`FlockDeploy.joinR`). At the plain record at
the deployed reader, conflicts are tied to collisions (`plainR_conflictTied`) and the table bound at the abort law is
`plainR_tableBoundAbortLin_R`, from `PlainRdInR` alone (`hIn`; the cap prefix is `capPrefixR`, proved with no
hypotheses). Its last-message binding is `rColsBind_sha512` and
`rCallsBind_sha512`. It also takes `StepsLogFree` at its bound record (`hLog`), and `DecompB` at what each
part's circuit accepts (`hD`). `recursiveKSAt_of_decomp` takes that last one as `RecKSDecompR`, and
`recursiveKSAt_of_targets` proves it from the verdict's targets (`recKSDecompR_of_targets`).
-/

namespace Flock.SecurityProofs

open Claims (Msg Out)
open FlockDeploy (Coin FinalP)

theorem recursiveKSAt_of (rd : Msg → Option FinalP)
    (final : Msg → Option (List (List (Msg × RecDeploy.Salt)) × Msg))
    (rest : (r : RecDeploy.XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE w : Msg → ℝ)
    (b : RecDeploy.Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (hIn : ∀ x, ¬ Claims.Flock.NoAcc FlockDeploy.sha512 (FlockDeploy.VR rd) x → FlockDeploy.KS.PlainRdInR rd x)
    (hLog : (RecDeploy.bdDeployed _root_.Flock.Readable.relation (FlockDeploy.nextR rd)
      (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation rd).cols wE FlockDeploy.joinR
      RecDeploy.rowInDeployed final w).StepsLogFree FlockDeploy.sha512)
    (hD : ∀ shape ∈ [Claims.Recursion.Form.Dense],
      Claims.Recursion.VStarB.DecompB FlockDeploy.sha512
        (RecDeploy.recBound _root_.Flock.Readable.relation { b with tr := RecDeploy.recTr aux nOf hd recRest }
          (RecDeploy.layoutBTops (RecDeploy.finOpen rest))
          (RecDeploy.bdDeployed _root_.Flock.Readable.relation (FlockDeploy.nextR rd)
            (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation rd).cols wE FlockDeploy.joinR
            RecDeploy.rowInDeployed final w).IVStar)
        (RecDeploy.bdDeployed _root_.Flock.Readable.relation (FlockDeploy.nextR rd)
          (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation rd).cols wE FlockDeploy.joinR
          RecDeploy.rowInDeployed final w).IVStar
        (RecDeploy.layoutBTops (RecDeploy.finOpen rest)) shape
        (Claims.Recursion.VStarB.PartAccepts FlockDeploy.sha512
          (RecDeploy.recBound _root_.Flock.Readable.relation { b with tr := RecDeploy.recTr aux nOf hd recRest }
            (RecDeploy.layoutBTops (RecDeploy.finOpen rest))
            (RecDeploy.bdDeployed _root_.Flock.Readable.relation (FlockDeploy.nextR rd)
              (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation rd).cols wE FlockDeploy.joinR
              RecDeploy.rowInDeployed final w).IVStar)
          (RecDeploy.deployZkRec _root_.Flock.Readable.relation) (RecDeploy.layoutBTops (RecDeploy.finOpen rest)) shape
          (RecDeploy.showsPart RecDeploy.recRead { b with tr := RecDeploy.recTr aux nOf hd recRest })
          (RecDeploy.cmtTr RecDeploy.recRead))) :
    Flock.Guarantees.RecursiveKSAt rd final rest wE w b aux nOf hd recRest :=
  RecDeploy.ksLin2_deployedR_partAccepts rd b aux nOf hd recRest rest wE FlockDeploy.joinR final w
    (FlockDeploy.KS.plainR_conflictTied _ rd)
    (FlockDeploy.KS.plainR_tableBoundAbortLin_R rd hIn)
    (RecDeploy.rColsBind_sha512 rd) (RecDeploy.rCallsBind_sha512 rd) hLog hD

/-- **`RecursiveKSAt` from `RecKSDecompR`**: `recursiveKSAt_of` with `DecompB` at the dense form as
`RecDeploy.RecKSDecompR`, V\*'s layout at the last part's pieces `rest` and the record past its link points
`recRest`. -/
theorem recursiveKSAt_of_decomp (rd : Msg → Option FinalP)
    (final : Msg → Option (List (List (Msg × RecDeploy.Salt)) × Msg))
    (rest : (r : RecDeploy.XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE w : Msg → ℝ)
    (b : RecDeploy.Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (hIn : ∀ x, ¬ Claims.Flock.NoAcc FlockDeploy.sha512 (FlockDeploy.VR rd) x → FlockDeploy.KS.PlainRdInR rd x)
    (hLog : (RecDeploy.bdR rd wE final w).StepsLogFree FlockDeploy.sha512)
    (hD : RecDeploy.RecKSDecompR rd b aux nOf hd recRest rest wE final w) :
    Flock.Guarantees.RecursiveKSAt rd final rest wE w b aux nOf hd recRest :=
  recursiveKSAt_of rd final rest wE w b aux nOf hd recRest hIn hLog fun shape hs => by
    obtain rfl := List.mem_singleton.1 hs
    exact hD

section Targets

open RecDeploy Claims.Recursion.VStarB
open FlockDeploy (nextR nextR_coin VerdictR)

/-- **`RecursiveKSAt` from the verdict's targets**: `recursiveKSAt_of_decomp` with `RecKSDecompR` from
`recKSDecompR_of_targets`. At the parts' pieces `rows` (what `holds` says of each) and the reps' accumulators `acc`,
where the targets hold (`h`, which `Top.targets_of_open` gives from the rows still open), it takes the owners (`hR`),
the checks no part holds (`hrest`) and each part's lowering (`hlowR`, `hlowP`). -/
theorem recursiveKSAt_of_targets (rd : Msg → Option FinalP)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (rest : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE w : Msg → ℝ)
    (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (hIn : ∀ x, ¬ Claims.Flock.NoAcc FlockDeploy.sha512 (FlockDeploy.VR rd) x → FlockDeploy.KS.PlainRdInR rd x)
    (hLog : (bdR rd wE final w).StepsLogFree FlockDeploy.sha512)
    {Q : Type} (rows : Msg → List Q) (holds : Q → Msg → List Msg → List Coin → Prop)
    (acc : Msg → List Msg → List Coin → ℕ → Array F128) (h : Top.Targets rd rows holds acc)
    (ownB : Msg → VerdictR → Option ℕ) (ownQ : Msg → Top.Piece Q → Option ℕ)
    (hR : ∀ (x : Msg) (st : Claims.Recursion.Stmt), (recR rd b aux nOf hd recRest rest wE final w).parse x = some st →
      Claims.Recursion.Form.Dense st.form →
      ∀ r ∈ (splitR rd rows holds acc h).checks (Claims.Recursion.xI (recR rd b aux nOf hd recRest rest wE final w) x),
        ∀ s, ownP ownB ownQ x r = some s → s < (recR rd b aux nOf hd recRest rest wE final w).nS x)
    (hrest : ∀ (x : Msg) (st : Claims.Recursion.Stmt), (recR rd b aux nOf hd recRest rest wE final w).parse x = some st →
      Claims.Recursion.Form.Dense st.form →
      ∀ q ∈ ivChecks (bdR rd wE final w)
          (deploySplit FlockDeploy.sha512 _root_.Flock.Readable.relation (nextR rd)
            (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation rd).cols wE FlockDeploy.joinR rowInDeployed
            final w (nextR_coin rd FlockDeploy.sha512) (splitR rd rows holds acc h).toLast)
          (Claims.Recursion.xI (recR rd b aux nOf hd recRest rest wE final w) x),
        holdersR rd b aux nOf hd recRest rest wE final w ownB ownQ x q = none →
        (∀ r, q = .elem r → ∃ r', r = .last r') →
        ∀ (c : ℕ → Coin) (ws : List Claims.Flock.Wit), ws.length = (recR rd b aux nOf hd recRest rest wE final w).nS x →
          (∀ e ∈ (layR rest).allCalls x ws, FlockDeploy.sha512 e.1 = e.2) →
          q.Holds FlockDeploy.sha512 (bdR rd wE final w)
            (deploySplit FlockDeploy.sha512 _root_.Flock.Readable.relation (nextR rd)
              (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation rd).cols wE FlockDeploy.joinR rowInDeployed
              final w (nextR_coin rd FlockDeploy.sha512) (splitR rd rows holds acc h).toLast)
            (Claims.Recursion.xI (recR rd b aux nOf hd recRest rest wE final w) x) c ((layR rest).msgs x ws))
    (hlowR : ∀ s x c ms, acceptsR rd b aux nOf hd recRest rest wE final w s x c ms →
      RestComputes FlockDeploy.sha512 (recR rd b aux nOf hd recRest rest wE final w) (bdR rd wE final w)
        (deploySplit FlockDeploy.sha512 _root_.Flock.Readable.relation (nextR rd)
          (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation rd).cols wE FlockDeploy.joinR rowInDeployed
          final w (nextR_coin rd FlockDeploy.sha512) (splitR rd rows holds acc h).toLast)
        (holdersR rd b aux nOf hd recRest rest wE final w ownB ownQ) s x c ms)
    (hlowP : ∀ s x c ms, acceptsR rd b aux nOf hd recRest rest wE final w s x c ms →
      PiecesComputes (bdR rd wE final w) (recR rd b aux nOf hd recRest rest wE final w) (Top.pieces rows)
        (Top.pholds holds) ownQ s x c ms) :
    Flock.Guarantees.RecursiveKSAt rd final rest wE w b aux nOf hd recRest :=
  recursiveKSAt_of_decomp rd final rest wE w b aux nOf hd recRest hIn hLog
    (recKSDecompR_of_targets rd rows holds acc b aux nOf hd recRest rest wE final w h ownB ownQ hR hrest hlowR hlowP)

end Targets

end Flock.SecurityProofs
