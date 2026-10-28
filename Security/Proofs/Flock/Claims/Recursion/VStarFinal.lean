import Proofs.Flock.Claims.Recursion.VStarHonest
import Proofs.Flock.Claims.Recursion.VStarFitHeld
import Proofs.Flock.Claims.Recursion.VStarFitPin

/-!
# What each V\* part's circuit accepts on the honest run, with hh's fitting discharged (`honest_partAccepts_final`)

`honest_partAccepts_deployed` at the deployed brings (`bringsAt`, `refPartsFin` at `A0` and the fitted records), with
`hh`'s fitting proved: each round before the last that part `s` holds whole is as many hm96 slots as the round has rows,
the `k`-th slot's port reading the round's `k`-th row at the fitted port inputs (`hfit_A0`). The rest are hypotheses,
each one piece:

* `hA`, the run lemma: each inner round's message is the inner prover's at the run's earlier rounds;
* `hpE`, `hp` at the rounds before the last;
* `hhFit`, `hh` from the fitting: `hh_A0` at the schedule (`schedMatches_dense`), the trace at the fitted port inputs
  (`TraceOk` and `SlotsOk`) and the run's messages being their rows' commit strings (`hmsg_run`);
* the schedule `x` reads, and part `s`'s staging (`StagedPart`);
* the tape exception (`hTops`): at each level above 0 the two reps' tops differ (`sameTop` false), as they do
  wherever a salt differs (`sameTop_false_of_salt`); they coincide with probability at most `2^-1536` a pair, a
  completeness error, and a non-vacuity instance picks a tape off it;
* the stager accepts at the honest source where the tape is off the exception (`hok`);
* the parts holding one round's row agree on its port's words (`hW`).
-/

namespace RecDeploy.VStarHonest

open _root_.Flock
open Claims (Msg Out)
open FlockDeploy (Coin)
open Claims.Recursion.VStarB (readForms)
open RecDeploy.VStarBuild

theorem honest_partAccepts_final (mask : ℕ → ℕ → Bool) (Ir : InnerRec) (b0 : Brings) (x : Msg)
    (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ) (hd : Msg → Flock.RecRecord.Record)
    (recRest : Msg → List (Msg × Out) → List UInt8)
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace) :
    let b := bringsAt mask Ir b0 x
    let bR : Brings := { b with tr := recTr aux nOf hd recRest }
    let L := DeployedR.layR OpenRows.elemR
    let bL : Brings := { bR with rd := L.readers }
    let I := Claims.Recursion.Inner.ofFlock DeployedR.bdR.deploy decI
    let D := recBound _root_.Flock.Readable.relation bR L DeployedR.bdR.IVStar
    NumsFixed bL.st →
    ∀ {w : Claims.Flock.Wit} {st : Claims.Recursion.CompiledStatement}, Claims.Recursion.R D I x w →
      D.parse x = some st → Claims.Recursion.Form.Dense st.form →
    ∀ (ω : Tape bL) (r : Fin (D.V.k x) → Coin),
      (Claims.run FlockDeploy.sha512 (refP I.V bL w) D.V x ω r).out = .acc →
      HonestWitSatisfies FlockDeploy.sha512 I.V bL x w ω r →
    ∀ s < nS x,
      let hist := ((Claims.run FlockDeploy.sha512 (refP I.V bL w) D.V x ω r).msgs.take (I.V.k (xIOf x))).map histOf
      let wO := bL.ref.wit w ω.1 hist s
      let ms := (Claims.run FlockDeploy.sha512 (refP I.V bL w) D.V x ω r).msgs.map Prod.fst
      (∀ xr, readX x = some xr → ∀ p ∈ DeployedR.moreCalls xr s,
        FlockDeploy.sha512 (p.1 ++ readForms wO p.2.1) = readForms wO p.2.2) →
      (Claims.findColl (L.calls x s wO ++ Claims.Recursion.VStarB.vcallsAt FlockDeploy.sha512 D
        ((Claims.Recursion.runVstar FlockDeploy.sha512 D I (refP I.V bL w) x ω r).getD s []))).isSome = false →
      (∀ i < I.V.k (xIOf x), ms.getD i [] =
        (((bL.ref.PI w).next (xIOf x) (hist.take i) ω.1).eval FlockDeploy.sha512).1) →
      (∀ i < RestRead.kI x - 1, ∀ o m, (o, m) ∈ L.piecesOf x s wO i → ((ms.getD i []).drop o).take m.length = m) →
      (∃ (xr : XR) (rs : Array RecVStar.SchedRound), readX x = some xr ∧ repSched xr.shape = some rs) →
      (∃ (stg : Stage) (xO : Msg), StagedPart stg s x xO) →
      (∀ src, srcOf (Ir.fit x) x w ω.1 (hist.take (RestRead.kI x - 1)) = some src → ∀ l, 0 < l → ∀ a b,
        (src.tops[0]!)[l]? = some a → (src.tops[1]!)[l]? = some b → _root_.VStar.In.sameTop a b = false) →
      ((∀ src, srcOf (Ir.fit x) x w ω.1 (hist.take (RestRead.kI x - 1)) = some src → ∀ l, 0 < l → ∀ a b,
          (src.tops[0]!)[l]? = some a → (src.tops[1]!)[l]? = some b → _root_.VStar.In.sameTop a b = false) →
        ∀ st c src, partAt x s = some (st, c) →
          srcOf (Ir.fit x) x w ω.1 (hist.take (RestRead.kI x - 1)) = some src →
          ∃ pin, _root_.VStar.In.partIn src (progOf c) st = .ok pin) →
      (∀ xr, readX x = some xr → ∀ s₁ c₁ h₁ s₂ c₂ h₂, xr.w.parts[s₁]?.bind partCircuit = some c₁ →
        h₁ ∈ heldRows xr s₁ → xr.w.parts[s₂]?.bind partCircuit = some c₂ → h₂ ∈ heldRows xr s₂ → h₁.1 = h₂.1 →
        (c₁.ports[h₁.2.2.1]?).map (·.words) = (c₂.ports[h₂.2.2.1]?).map (·.words)) →
      ((∀ i < RestRead.kI x - 1, ∀ segs, (DeployedR.layR OpenRows.elemR).msgAt x s i = some segs →
          ∃ (st : _root_.VStar.Stmt) (c : Circuit) (src : _root_.VStar.In.Src) (pin : _root_.VStar.In.PartIn),
            partAt x s = some (st, c) ∧ srcOf (Ir.fit x) x w ω.1 (hist.take (RestRead.kI x - 1)) = some src ∧
            _root_.VStar.In.partIn src (progOf c) st = .ok pin ∧
            segs.length = (roundRows (Ir.fit x) w ω.1 hist i).length ∧
            ∀ k (hk : k < segs.length), ∃ (nm : String) (q n : ℕ) (ps : List ℕ) (rw : _root_.VStar.In.Row),
              segs[k] = (([] : Msg), hmCommit c q n) ∧ c.ports.findIdx? (·.name == nm) = some q ∧
              portRow (fitPin c pin) nm n = some rw ∧ rowHidden c nm n = some ps ∧
              (roundRows (Ir.fit x) w ω.1 hist i)[k]? = some rw) →
        ∀ i < L.rounds x, ∀ m, L.held x s wO i = some m → ms.getD i [] = m) →
      Claims.Recursion.VStarB.PartAccepts FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation) L
        Claims.Recursion.Form.Dense (showsPart recRead bR) (cmtTr recRead) s x (Claims.coinSeq r) ms := by
  intro b bR L bL I D hN w st hR hx hf ω r hacc hW s hs hist wO ms hTm hnc hA hpE hsch hSt hTops hok hWd hhFit
  obtain ⟨xr, rs, hxr, hrs⟩ := hsch
  obtain ⟨stg, xO, hSP⟩ := hSt
  have hrun : I.V.k (xIOf x) + kS x ≤ (Claims.run FlockDeploy.sha512 (refP I.V bL w) D.V x ω r).msgs.length :=
    run_length_ge FlockDeploy.sha512 I.V bL.st hxr (refP I.V bL w) ω r
  have hk : I.V.k (xIOf x) = RestRead.kI x := rfl
  have hlen : RestRead.kI x - 1 ≤ hist.length := by
    simp only [hist, List.length_map, List.length_take]
    omega
  exact honest_partAccepts_deployed mask Ir b0 x aux nOf hd recRest decI hN hR hx hf ω r hacc hW s hs hTm hnc
    hA hpE (hhFit (hfit_A0 OpenRows.elemR hxr hrs hSP hlen (hok hTops) (hWd xr hxr)))

end RecDeploy.VStarHonest
