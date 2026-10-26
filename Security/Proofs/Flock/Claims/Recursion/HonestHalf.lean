import Proofs.Flock.Claims.Recursion.VStarFinal
import Proofs.Flock.Claims.Recursion.HpEarlyStaged
import Proofs.Flock.Claims.Recursion.HhHeld
import Proofs.Flock.Claims.Recursion.HmsgRun
import Proofs.Flock.Claims.Recursion.HonestRun
import Proofs.Flock.Claims.Recursion.MoreSlotsAll
import Proofs.Flock.Claims.Recursion.HonestPartsLayR
import Proofs.Flock.Claims.Recursion.SchedDense
import Proofs.Flock.Claims.Recursion.DeployedBrings

/-!
# The honest half of `RecursiveKSAt`'s non-vacuity at the deployed V\* (`honest_half`)

`RecursiveKS` reads, at the brings `deployedB b` for every `b`, the premise `DecompB` at `PartAccepts`: wherever every V\*
part's circuit accepts at a dense compiled statement, `IVStar` accepts or two recorded calls collide. It is non-vacuous
where some run has every part accepting. `honest_half` gives that run: the compiled verifier's run of the reference
prover at `deployedB` of the deployed brings at `x` (`bringsAt`: the deployed assigner `A0` and the fitted records), and
every part `s < nS x` accepts on it (`PartAccepts` at `DeployedR.layR OpenRows.elemR`, `RecursiveKSAt`'s layout).

`honest_partAccepts_final` with its pieces discharged:

* `hA`, the run lemma (`run_msg_inner`);
* `hpE`, hp before the last round (`hpEarly_staged`, every part staged, given `RowsFit`);
* `hhFit`, hh from the fitting (`hh_A0` at the dense schedule `schedMatches_dense`, the trace `traceAt_staged` and the
  messages `hmsg_run`);
* `hTm`, the extra calls `layR` records (`moreSlotsTrue_deployed` at the run's staged part, whose outer witness
  `Flock.check` accepts: `vstar_complete_run_setup`);
* the schedule (`schedMatches_dense`) and each part's staging (`stagedPart_run`, from the run accepting).

What is left is a hypothesis: the builder's facts (`NumsFixed`, `StagedSetsUp`, `WitSatisfies`), the statement (in the
relation, parsed, dense), the outer run accepting and its inner proof accepted, no collision at each part, the fitted
rows (`RowsFit`), the tape exception (`hTops`), the stager accepting (`hok`) and the shared rows' widths (`hWd`).
`bringsAt_deployedB`: the brings here are `RecursiveKS`'s `deployedB` at `bringsAt mask Ir b0 x`.
-/

namespace RecDeploy.HonestHalf

open _root_.Flock
open Claims (Msg Out)
open FlockDeploy (Coin)
open Claims.Recursion.VStarB (readForms)
open RecDeploy.VStarBuild
open RecDeploy.VStarHonest

/-- **The brings `honest_half` runs at are `RecursiveKS`'s**: `deployedB` of the deployed brings at `x`. -/
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

/-- **Every V\* part accepts on the honest run, at `RecursiveKSAt`'s record** (the honest half of its non-vacuity). At
the deployed brings at `x` under `deployedB` (`bringsAt_deployedB`), V\*'s layout of record `layR elemR` and the deployed
bound record, on the compiled verifier's run of the reference prover, each part `s < nS x`'s circuit accepts. -/
theorem honest_half (mask : ℕ → ℕ → Bool) (Ir : InnerRec) (b0 : Brings) (x : Msg)
    (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ) (hd : Msg → Flock.RecRecord.Record)
    (recRest : Msg → List (Msg × Out) → List UInt8)
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace) :
    let b := bringsAt mask Ir (deployedB b0) x
    let bR : Brings := { b with tr := recTr aux nOf hd recRest }
    let L := DeployedR.layR OpenRows.elemR
    let bL : Brings := { bR with rd := L.readers }
    let I := Claims.Recursion.Inner.ofFlock DeployedR.bdR.deploy decI
    let D := recBound _root_.Flock.Readable.relation bR L DeployedR.bdR.IVStar
    NumsFixed bL.st → StagedSetsUp FlockDeploy.sha512 I.V bL → WitSatisfies FlockDeploy.sha512 I.V bL →
    ∀ {w : Claims.Flock.Wit} {st : Claims.Recursion.Stmt}, Claims.Recursion.R D I x w →
      D.parse x = some st → Claims.Recursion.Form.Dense st.form →
    ∀ (ω : Tape bL) (r : Fin (D.V.k x) → Coin),
      (Claims.run FlockDeploy.sha512 (refP I.V bL w) D.V x ω r).out = .acc →
      InnerAccepts FlockDeploy.sha512 I.V x (runCs (I.V.k (xIOf x) + kS x) r)
        (((Claims.run FlockDeploy.sha512 (refP I.V bL w) D.V x ω r).msgs.take (I.V.k (xIOf x) + kS x)).map
          histOf) →
    let hist := ((Claims.run FlockDeploy.sha512 (refP I.V bL w) D.V x ω r).msgs.take (I.V.k (xIOf x))).map histOf
    let ms := (Claims.run FlockDeploy.sha512 (refP I.V bL w) D.V x ω r).msgs.map Prod.fst
    (∀ s < nS x, (Claims.findColl (L.calls x s (bL.ref.wit w ω.1 hist s) ++
      Claims.Recursion.VStarB.vcallsAt FlockDeploy.sha512 D
        ((Claims.Recursion.runVstar FlockDeploy.sha512 D I (refP I.V bL w) x ω r).getD s []))).isSome = false) →
    (∀ i < RestRead.kI x - 1, HpEarly.RowsFit (Ir.fit x) x w ω.1 (hist.take (RestRead.kI x - 1)) i) →
    (∀ src, srcOf (Ir.fit x) x w ω.1 (hist.take (RestRead.kI x - 1)) = some src → ∀ l, 0 < l → ∀ a b,
      (src.tops[0]!)[l]? = some a → (src.tops[1]!)[l]? = some b → _root_.VStar.In.sameTop a b = false) →
    ((∀ src, srcOf (Ir.fit x) x w ω.1 (hist.take (RestRead.kI x - 1)) = some src → ∀ l, 0 < l → ∀ a b,
        (src.tops[0]!)[l]? = some a → (src.tops[1]!)[l]? = some b → _root_.VStar.In.sameTop a b = false) →
      ∀ s st c src, partAt x s = some (st, c) →
        srcOf (Ir.fit x) x w ω.1 (hist.take (RestRead.kI x - 1)) = some src →
        ∃ pin, _root_.VStar.In.partIn src (progOf c) st = .ok pin) →
    (∀ xr, readX x = some xr → ∀ s₁ c₁ h₁ s₂ c₂ h₂, xr.w.parts[s₁]?.bind partCircuit = some c₁ →
      h₁ ∈ heldRows xr s₁ → xr.w.parts[s₂]?.bind partCircuit = some c₂ → h₂ ∈ heldRows xr s₂ → h₁.1 = h₂.1 →
      (c₁.ports[h₁.2.2.1]?).map (·.words) = (c₂.ports[h₂.2.2.1]?).map (·.words)) →
    ∀ s < nS x,
      Claims.Recursion.VStarB.PartAccepts FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation) L
        Claims.Recursion.Form.Dense (showsPart recRead bR) (cmtTr recRead) s x (Claims.coinSeq r) ms := by
  intro b bR L bL I D hN hS hW w st hR hx hf ω r hacc hI hist ms hnc hfit hTops hok hWd s hs
  obtain ⟨xr, hxr⟩ := readX_of_lt_nS hs
  have hk : I.V.k (xIOf x) = RestRead.kI x := rfl
  have hK := schedMatches_dense x st hx hf
  have hSall : ∀ s' sv c, partAt x s' = some (sv, c) → ∃ xO, StagedPart bL.st s' x xO := fun s' _ _ hp =>
    ⟨_, stagedPart_run FlockDeploy.sha512 I.V bL.st x (refP I.V bL w) ω r hacc (lt_nS_of_partAt hp)⟩
  have hTm : ∀ xr, readX x = some xr → ∀ p ∈ DeployedR.moreCalls xr s,
      FlockDeploy.sha512 (p.1 ++ readForms (bL.ref.wit w ω.1 hist s) p.2.1) =
        readForms (bL.ref.wit w ω.1 hist s) p.2.2 := by
    obtain ⟨-, -, -, -, hrel, -⟩ := vstar_complete_run_setup FlockDeploy.sha512 I
      (FlockDeploy.deployPlain _root_.Flock.Readable.relation).scopeCheck bL hN hS hW hR ω r hI hs
    have hck := check_of_R hrel
    have hsg := stagedPart_run FlockDeploy.sha512 I.V bL.st x (refP I.V bL w) ω r hacc hs
    have heq := runVstar_getD_staged FlockDeploy.sha512 I
      (FlockDeploy.deployPlain _root_.Flock.Readable.relation).scopeCheck bL hN (refP I.V bL w) ω r hacc hs
    exact fun xr' hxr' => MoreSlots.moreSlotsTrue_deployed bL.st x _ s _ xr' hsg (by rw [← heq]; exact hck) hxr'
  have hA : ∀ i < I.V.k (xIOf x), ms.getD i [] =
      (((bL.ref.PI w).next (xIOf x) (hist.take i) ω.1).eval FlockDeploy.sha512).1 :=
    run_msg_inner FlockDeploy.sha512 I.V bL w hxr ω r
  have hpE : ∀ i < RestRead.kI x - 1, ∀ o m, (o, m) ∈ L.piecesOf x s (bL.ref.wit w ω.1 hist s) i →
      ((ms.getD i []).drop o).take m.length = m := fun i hi =>
    HpEarly.hpEarly_staged mask (Ir.fit x) I.V bL w hxr hk ω r hs hi (hfit i hi) hSall
  have hmsg : ∀ i < RestRead.kI x - 1, ms.getD i [] = (roundRows (Ir.fit x) w ω.1 hist i).flatMap HpEarly.rowCommit :=
    fun i hi => hmsg_run (A0 mask) (Ir.fit x) I.V bL w hxr hk ω r hi
  obtain ⟨xr₁, rs, hxr₁, hrs, -⟩ := hK
  exact honest_partAccepts_final mask Ir (deployedB b0) x aux nOf hd recRest decI hN hS hW hR hx hf ω r hacc hI s hs
    hTm (hnc s hs) hA hpE ⟨xr₁, rs, hxr₁, hrs⟩
    ⟨bL.st, _, stagedPart_run FlockDeploy.sha512 I.V bL.st x (refP I.V bL w) ω r hacc hs⟩ hTops
    (fun h => hok h s) hWd
    (fun hfitH => HhHeld.hh_A0 (mask := mask) (I := Ir.fit x) (x := x) (w := w) (ρ := ω.1) (hist := hist) (s := s)
      (ms := ms) (schedMatches_dense x st hx hf)
      (fun sv c src pin hp hsrc hpi => VStarTraceAt.traceAt_staged hSall mask s sv c src pin hp hsrc hpi) hfitH hmsg)

end RecDeploy.HonestHalf
