import Proofs.Flock.Claims.Recursion.HonestHalf
import Proofs.Flock.Claims.Recursion.VStarRowsFit
import Proofs.Flock.Claims.Recursion.IrDep

/-!
# The honest half of `RecursiveKSAt`'s non-vacuity at the deployed V\* (`honest_half_tops`)

`honest_half_tops`: at the deployed brings at `x` with the deployed inner prover `IrDep x`, where the compiled verifier
accepts the reference prover's run, every V\* part `s < nS x` accepts on it (`PartAccepts` at
`DeployedR.layR OpenRows.elemR`, `RecursiveKSAt`'s layout). It is `honest_partAccepts_final` with its pieces discharged:

* `hA`, the run lemma (`run_msg_inner`);
* `hpE`, hp before the last round: the rounds without pieces (round 0, and every stream round that commits no tops,
  where `layR elemR`'s `pieceAt` is empty while the round has rows) hold none, so hp there is vacuous
  (`hpEarly_tops`), and at the rounds with pieces `rowsFit_tops` proves the fitted rows (`RowsFit`);
* `hhFit`, hh from the fitting (`hh_A0` at the dense schedule `schedMatches_dense`, the trace `traceAt_staged` and the
  messages `hmsg_run`);
* `hTm`, the extra calls `layR` records (`moreSlotsTrue_deployed` at the run's staged part, whose outer witness
  `Flock.check` accepts: `vstar_complete_run_setup_at`);
* the stager accepting every part (`hok_honest`: the level tops and the caps from `hTops`, at the history
  `hist.take (kI x - 1)`, which holds the schedule's `2R` rounds since `kI x = 2 + 2R` (`hH`); the inner parts' port
  names from `VStar.parts`, `innerPorts_named`; the inner prover's openings fitting V\*'s levels, `irShaped_dep`);
* the holders of one round's row agreeing on its port's words (`heldRows_words`, the verifier's `checkRoundWords`);
* the schedule (`schedMatches_dense`) and each part's staging (`stagedPart_run`, from the run accepting).

What it takes is the run accepting (`HonestRunAccepted`), V\*'s gate-level completeness at the run
(`HonestWitSatisfies`), no collision among a part's recorded calls (`NoCollAt`) and the tape exception (`hTops`: the
reps' tops differ at every level above 0).
-/

namespace RecDeploy.HonestHalfTops

open _root_.Flock
open Claims (Msg Out)
open FlockDeploy (Coin)
open Claims.Recursion.VStarB (readForms)
open RecDeploy.VStarBuild
open RecDeploy.VStarHonest
open RecDeploy.HonestHalf

/-- **Every V\* part accepts at an honest run**: at the deployed brings at `x` with the deployed inner prover `IrDep x`,
where the reference prover `refP`'s run is accepted and satisfies V\*, no recorded calls collide and the reps' tops
differ (`TopsAt`), `PartAccepts` holds at every part `s < nS x`.

The premises are about Lean's `refP`; the run witnessing them (`HonestFixture`) used Python's builder and C-Flock's CPU
prover. `HonestFixture`'s docstring says which difftest ties the two: `test_vstar_trace_difftest`, with
`vstar-trace --sat`. -/
theorem honest_half_tops (mask : ℕ → ℕ → Bool) (b0 : Brings) (x : Msg)
    (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ) (hd : Msg → Flock.RecRecord.Record)
    (recRest : Msg → List (Msg × Out) → List UInt8)
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace) :
    let bR : Brings := { bringsAt mask (IrDep x) (deployedB b0) x with tr := recTr aux nOf hd recRest }
    let L := DeployedR.layR OpenRows.elemR
    let bL := bLAt mask (IrDep x) b0 x aux nOf hd recRest
    let I := IAt decI
    let D := DAt mask (IrDep x) b0 x aux nOf hd recRest
    ∀ {w : Claims.Flock.Wit} {st : Claims.Recursion.Stmt} (ω : Tape bL) (r : Fin (D.V.k x) → Coin),
    HonestRunAccepted mask (IrDep x) b0 x aux nOf hd recRest decI w st ω r →
    HonestWitSatisfies FlockDeploy.sha512 I.V bL x w ω r →
    NoCollAt mask (IrDep x) b0 x aux nOf hd recRest decI w ω r →
    let hist := ((Claims.run FlockDeploy.sha512 (refP I.V bL w) D.V x ω r).msgs.take (I.V.k (xIOf x))).map histOf
    let ms := (Claims.run FlockDeploy.sha512 (refP I.V bL w) D.V x ω r).msgs.map Prod.fst
    (∀ src, srcOf ((IrDep x).fit x) x w ω.1 (hist.take (RestRead.kI x - 1)) = some src → ∀ l, 0 < l → ∀ a b,
      (src.tops[0]!)[l]? = some a → (src.tops[1]!)[l]? = some b → _root_.VStar.In.sameTop a b = false) →
    ∀ s < nS x,
      Claims.Recursion.VStarB.PartAccepts FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation) L
        Claims.Recursion.Form.Dense (showsPart recRead bR) (cmtTr recRead) s x (Claims.coinSeq r) ms := by
  intro bR L bL I D w st ω r hRun hW hnc hist ms hTops s hs
  have hN := numsFixed bL.st
  obtain ⟨hR, hx, hf, hacc⟩ := hRun
  obtain ⟨xr, hxr⟩ := readX_of_lt_nS hs
  have hk : I.V.k (xIOf x) = RestRead.kI x := rfl
  have hSall : ∀ s' sv c, partAt x s' = some (sv, c) → ∃ xO, StagedPart bL.st s' x xO := fun s' _ _ hp =>
    ⟨_, stagedPart_run FlockDeploy.sha512 I.V bL.st x (refP I.V bL w) ω r hacc (lt_nS_of_partAt hp)⟩
  have hTm : ∀ xr, readX x = some xr → ∀ p ∈ DeployedR.moreCalls xr s,
      FlockDeploy.sha512 (p.1 ++ readForms (bL.ref.wit w ω.1 hist s) p.2.1) =
        readForms (bL.ref.wit w ω.1 hist s) p.2.2 := by
    obtain ⟨-, -, -, -, hrel, -⟩ := vstar_complete_run_setup_at FlockDeploy.sha512 I
      (FlockDeploy.deployPlain _root_.Flock.Readable.relation).scopeCheck bL hN hR ω r hacc hW hs
    have hck := check_of_R hrel
    have hsg := stagedPart_run FlockDeploy.sha512 I.V bL.st x (refP I.V bL w) ω r hacc hs
    have heq := runVstar_getD_staged FlockDeploy.sha512 I
      (FlockDeploy.deployPlain _root_.Flock.Readable.relation).scopeCheck bL hN (refP I.V bL w) ω r hacc hs
    exact fun xr' hxr' => MoreSlots.moreSlotsTrue_deployed bL.st x _ s _ xr' hsg (by rw [← heq]; exact hck) hxr'
  have hA : ∀ i < I.V.k (xIOf x), ms.getD i [] =
      (((bL.ref.PI w).next (xIOf x) (hist.take i) ω.1).eval FlockDeploy.sha512).1 :=
    run_msg_inner FlockDeploy.sha512 I.V bL w hxr ω r
  have hmsg : ∀ i < RestRead.kI x - 1,
      ms.getD i [] = (roundRows ((IrDep x).fit x) w ω.1 hist i).flatMap HpEarly.rowCommit :=
    fun i hi => hmsg_run (A0 mask) ((IrDep x).fit x) I.V bL w hxr hk ω r hi
  obtain ⟨xr₁, rs, hxr₁, hrs, hkR⟩ := schedMatches_dense x st hx hf
  have hH : ∀ r' rs', readX x = some r' → repSched r'.shape = some rs' →
      2 * rs'.size ≤ (hist.take (RestRead.kI x - 1)).length := by
    intro r' rs' hr' hrs'
    have hrr : rs' = rs := by
      rw [hxr₁] at hr'
      rw [← Option.some.inj hr', hrs] at hrs'
      exact (Option.some.inj hrs').symm
    have e : (hist.take (RestRead.kI x - 1)).length = RestRead.kI x - 1 := by
      rw [List.length_take, Nat.min_eq_left]
      show RestRead.kI x - 1 ≤
        (((Claims.run FlockDeploy.sha512 (refP I.V bL w) D.V x ω r).msgs.take (I.V.k (xIOf x))).map histOf).length
      rw [List.length_map, List.length_take]
      have hlen : I.V.k (xIOf x) + kS x ≤ (Claims.run FlockDeploy.sha512 (refP I.V bL w) D.V x ω r).msgs.length :=
        run_length_ge FlockDeploy.sha512 I.V bL.st hxr (refP I.V bL w) ω r
      have hk2 : I.V.k (xIOf x) = RestRead.kI x := rfl
      omega
    have hkI : RestRead.kI x = 2 + 2 * rs.size := hkR
    rw [e, hrr]
    omega
  have hIr : IrShaped (IrDep x) x w ω.1 (hist.take (RestRead.kI x - 1)) := irShaped_dep hx hf hH hTops
  have hok : ∀ s' sv c src, partAt x s' = some (sv, c) →
      srcOf ((IrDep x).fit x) x w ω.1 (hist.take (RestRead.kI x - 1)) = some src →
      ∃ pin, _root_.VStar.In.partIn src (progOf c) sv = .ok pin := fun s' =>
    hok_honest (s := s') hx hf hIr (fun _ _ _ _ hp _ _ hr hrs => innerPorts_named hp hr hrs) hH hTops
  have hfit := rowsFit_tops (Ir := IrDep x) (w := w) (ρ := ω.1) (stg := bL.st) hx hf hH
    (fun s' hs' => ⟨_, stagedPart_run FlockDeploy.sha512 I.V bL.st x (refP I.V bL w) ω r hacc hs'⟩) hok hTops
  have hpE : ∀ i < RestRead.kI x - 1, ∀ o m, (o, m) ∈ L.piecesOf x s (bL.ref.wit w ω.1 hist s) i →
      ((ms.getD i []).drop o).take m.length = m := fun i hi =>
    HpEarly.hpEarly_tops mask ((IrDep x).fit x) I.V bL w hxr hk ω r hs hi (hfit i hi) hSall
  exact honest_partAccepts_final mask (IrDep x) (deployedB b0) x aux nOf hd recRest decI hN hR hx hf ω r hacc hW s hs
    hTm (hnc s hs) hA hpE ⟨xr₁, rs, hxr₁, hrs⟩
    ⟨bL.st, _, stagedPart_run FlockDeploy.sha512 I.V bL.st x (refP I.V bL w) ω r hacc hs⟩ hTops
    (fun _ => hok s)
    (fun _ hxr' => heldRows_words hxr')
    (fun hfitH => HhHeld.hh_A0 (mask := mask) (I := (IrDep x).fit x) (x := x) (w := w) (ρ := ω.1) (hist := hist)
      (s := s) (ms := ms) (schedMatches_dense x st hx hf)
      (fun sv c src pin hp hsrc hpi => VStarTraceAt.traceAt_staged hSall mask s sv c src pin hp hsrc hpi) hfitH hmsg)

end RecDeploy.HonestHalfTops
