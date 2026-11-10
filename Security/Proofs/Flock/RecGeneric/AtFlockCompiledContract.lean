module

public import Proofs.Flock.RecGeneric.AtFlockCompiled
public import Proofs.Flock.RecGeneric.AtFlockContract

/-!
# L2 at Flock: the compiler's contract at the deployment's dense statements

`Compiled.EO` gives witnesses only on an accepted run, one per session, each paired with the session's staged
statement and accepted by the outer argument's check there (`Compiled.eo_some_staged`). At today's compiled record over
Flock `--zk` (`Compiled.ofRec`) the staged statements are the run's V\* statements, so the witnesses' second components
are in the outer relation there (`compiled_outerHolds`), which is all `RecFacts` reads: the facts that give L1's
contract at `compAt` give L2's at `Compiled.ofRec` (`contractCompiled_of`), each field at the witnesses' second
components. At the server's `Open(DE, hm96)` and the element-level Flock's openings it is the contract at the bound
record's `IVStar` and opened views (`contractCompiled_ofBound`), and at the deployment `RecDeploy.facts_deployed` gives
the facts at each dense statement (`contractCompiledDense`, `ContractCompiledDense`), from F1's `IVStarOfBound` and
`OpenViewOfBound` at the deployed bound.
-/

@[expose] public section

namespace Claims.RecGeneric

open Claims Claims.Recursion
open Claims.SamplingGeneric (Arg)

namespace Compiled

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-- **`E_O`'s witnesses**: where `Compiled.EO` gives `ws`, the judged run accepted, there is one per session, and each
is paired with its session's staged statement and accepted by `A`'s check there (`firstWait_some`, through the shape
test and the serving). -/
theorem eo_some_staged (L : Compiled Coin A) (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (r : Fin (L.V.k x) → Coin) (c : L.EO.C x) {ws : List (Msg × A.W)} (h : (L.EO.eval H P x ω r c).1 = some ws) :
    (run H P L.V x ω r).out = .acc ∧ ws.length = L.nS x ∧ ∀ (s : ℕ) (hs : s < ws.length),
      ws[s].1 = (L.stmts x (run H P L.V x ω r).msgs).getD s [] ∧ A.check ws[s].1 ws[s].2 = true := by
  have hacc : (run H P L.V x ω r).out = .acc := by
    by_contra hn
    simp [Extractor.eval, Compiled.EO, RP.eval, Transcript.seen, hn] at h
  simp only [Extractor.eval, Compiled.EO, RP.eval, Transcript.seen, hacc, ↓reduceIte] at h
  have hf := collect_some H _ _ ws h
  have hlen := hf.length_eq
  rw [List.length_map, List.length_finRange] at hlen
  refine ⟨hacc, hlen.symm, fun s hs => ?_⟩
  have h1 := (List.forall₂_iff_get.1 hf).2 s (by rw [List.length_map, List.length_finRange, hlen]; exact hs) hs
  simp only [List.get_eq_getElem, List.getElem_map, List.getElem_finRange] at h1
  unfold sessionProg at h1
  split_ifs at h1
  · rw [RP.eval_bind_ret] at h1
    obtain ⟨w, hw, hws⟩ := Option.map_eq_some_iff.1 h1
    rw [← hws]
    refine ⟨rfl, ?_⟩
    rw [serve_eq, (RP.serve_eval H _ _ _).1] at hw
    simp only [Claims.SamplingGeneric.Arg.ext, ERep, RP.eval] at hw
    split at hw
    · exact firstWait_some H _ _ _ _ _ _ _ hw
    · simp [RP.eval] at hw
  · simp [RP.eval] at h1

end Compiled

section Contract

variable {Coin Col T Col' T' : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
  [DecidableEq Col'] (D : RecAt Coin) (Bd : Bound Coin Col T) (decI : List (ℕ × Col) → List T)
  (DO : Claims.Flock.Deploy Coin Col' T') {H : Msg → Msg} {x : Msg} {pk : ℕ → Msg → Msg}

/-- **`Compiled.EO`'s witnesses at today's compiled record are in the outer relation at the run's V\* statements**:
the sessions' staged statements are the V\* statements, and the outer argument's check is Flock's. -/
theorem compiled_outerHolds {Ω : Type} {P : Prover Ω} {ω : Ω} {r : Fin ((D.record Bd.IVStar).V.k x) → Coin}
    {c : (Compiled.ofRec D DO Bd.IVStar).EO.C x} {ws : List (Msg × Claims.Flock.Wit)}
    (h : ((Compiled.ofRec D DO Bd.IVStar).EO.eval H P x ω r c).1 = some ws) :
    (run H P (D.record Bd.IVStar).V x ω r).out = .acc ∧
      OuterHolds DO (runVstar H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) P x ω r)
        ((D.record Bd.IVStar).nS x) (ws.map Prod.snd) := by
  obtain ⟨hacc, hlen, hs⟩ := Compiled.eo_some_staged (Compiled.ofRec D DO Bd.IVStar) H P x ω r c h
  refine ⟨hacc, by rw [List.length_map]; exact hlen, fun s hs' => ?_⟩
  obtain ⟨h1, h2⟩ := hs s (by rw [List.length_map] at hs'; exact hs')
  simp only [List.getElem_map]
  rw [h1] at h2
  exact h2

/-- **L2's contract at today's compiled record over the bound record**, at the statement `x`, from the record's facts
there (`RecFacts`), each field at the witnesses' second components: `Compiled.EO` gives witnesses only on an accepted
run, in the outer relation at its V\* statements (`compiled_outerHolds`). -/
theorem contractCompiled_of (hF : RecFacts H D Bd decI DO pk x) :
    (Compiled.ofRec D DO Bd.IVStar).toComp.Contract H Bd.IVStar (fun v => [Bd.DE.cols (Bd.openView v)]) x where
  replays _ P ω r c _ h := (Compiled.eo_some_staged (Compiled.ofRec D DO Bd.IVStar) H P x ω r c h).1
  refuses Ω P ω r hr := hF.refuses x Ω P ω r hr
  rounds := hF.rounds x
  margin Ω P ω r hr := hF.margin x Ω P ω r hr
  callsTrue Ω P ω r _ w hg :=
    callsAt_true_of (hF.callsTrue Ω P ω r (w.map Prod.snd) hg.1 (compiled_outerHolds D Bd decI DO hg.2).2)
  ownTrue _ _ _ _ := setupCalls_true H _ _
  callsLog _ _ _ r _ w _ _ := (hF.callsLog r (w.map Prod.snd)).imp_left fun h e he => h e
    ((congrArg (fun t => e ∈ t.vLog)
      (innerRun_ofRec D (Inner.ofFlock Bd.deploy decI) hF.xI_eq r (w.map Prod.snd))).mpr he)
  forks _ _ P j ω r r' _ _ w w' hj hg hg' _ _ hne hcond :=
    forks_of D Bd decI DO hF P j ω r r' (w.map Prod.snd) (w'.map Prod.snd) hj hg.1 hg'.1
      (compiled_outerHolds D Bd decI DO hg.2).2 (compiled_outerHolds D Bd decI DO hg'.2).2 hne hcond
  accepts _ Ω P ω r _ w hg _ :=
    (hF.accepts Ω P ω r (w.map Prod.snd) hg.1 (compiled_outerHolds D Bd decI DO hg.2).2).imp
      (fun h => (congrArg (·.out) (view_ofRec D Bd.IVStar hF.xI_eq r (w.map Prod.snd))).trans h)
      (coll_lift (fun e he => (List.mem_append.1 he).elim
        (callsAt_true_of (hF.callsTrue Ω P ω r (w.map Prod.snd) hg.1 (compiled_outerHolds D Bd decI DO hg.2).2) e)
        (setupCalls_true H _ _ e))
        (fun _ hp => mem_callsAt_append hp))

/-- **L2's contract at the server's `Open(DE, hm96)` and its reading of opened views** (`ContractCompiledOn`'s) is the
contract at the bound record's `IVStar` and opened views (`IVStarOfBound`, `OpenViewOfBound`): `compiledAt` is
`Compiled.ofRec` at the server's `IVStar`, and the element-level Flock's opening of a view is `DE`'s columns of it. -/
theorem contractCompiled_ofBound (Cp : Caps Coin) (hIV : IVStarOfBound Bd) (hOV : OpenViewOfBound Bd)
    (h : (Compiled.ofRec D DO Bd.IVStar).toComp.Contract H Bd.IVStar (fun v => [Bd.DE.cols (Bd.openView v)]) x) :
    (compiledAt D Bd DO Cp).toComp.Contract H ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)
      (fun v => (Proto.ofFlock Bd Bd.deploy.decode Cp).opens ((Server.ofBound Bd).openView v)) x := by
  show (Compiled.ofRec D DO ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)).toComp.Contract H
    ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans) (fun v => [Bd.DE.cols ((Server.ofBound Bd).openView v)]) x
  rw [hIV, hOV]
  exact h

end Contract

/-- **A1d: the compiler's contract at the deployment** (`ContractCompiledDense`): at each dense statement the record's
facts (`RecDeploy.facts_deployed`, at the deployed inner decoder) give the contract at `Compiled.ofRec`
(`contractCompiled_of`), hence at `compiledAt` (`contractCompiled_ofBound`). It takes F1's `IVStarOfBound` and
`OpenViewOfBound` at the deployed bound. -/
theorem contractCompiledDense
    (hIV : IVStarOfBound (RecDeploy.bdDeployed _root_.Flock.Readable.relation
      (FlockDeploy.nextR RecDeploy.DeployedR.rdR)
      (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation RecDeploy.DeployedR.rdR).cols
      RecDeploy.DeployedR.wER FlockDeploy.joinR RecDeploy.rowInDeployed RecDeploy.DeployedR.finalR
      RecDeploy.DeployedR.wR))
    (hOV : OpenViewOfBound (RecDeploy.bdDeployed _root_.Flock.Readable.relation
      (FlockDeploy.nextR RecDeploy.DeployedR.rdR)
      (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation RecDeploy.DeployedR.rdR).cols
      RecDeploy.DeployedR.wER FlockDeploy.joinR RecDeploy.rowInDeployed RecDeploy.DeployedR.finalR
      RecDeploy.DeployedR.wR)) :
    ContractCompiledDense := by
  intro b aux nOf hd recRest x hx
  exact contractCompiled_ofBound _ _ _ FlockDeploy.KS.CpDep hIV hOV (contractCompiled_of _ _ (fun _ => []) _
    ((RecDeploy.facts_deployed b aux nOf hd recRest (fun _ _ _ _ => (0, 0, 0, 0)) x hx).2 _))

end Claims.RecGeneric
