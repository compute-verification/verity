import Proofs.Flock.Claims.ZK.Sent
import Proofs.Flock.Claims.ZK.Split
import Proofs.Flock.Claims.ZK.SimCommit
import Proofs.Flock.Claims.ZK.PrefixLive
import Proofs.Flock.Claims.ZK.SimSession
import Proofs.Flock.Claims.ZK.SplitInner
import Proofs.Flock.Claims.ZK.SimPiop
import Proofs.Flock.Claims.ZK.InnerSlotTriple
import Proofs.Flock.Claims.ZK.InnerLeaf
import Proofs.Flock.Claims.ZK.InnerY
import Proofs.Flock.Claims.ZK.InnerBatched
import Proofs.Flock.Claims.ZK.SimLig
import Proofs.Flock.Claims.ZK.LiveEnd
import Proofs.Flock.Claims.ZK.InnerTauLig
import Proofs.Flock.Claims.ZK.FinInner
import Proofs.Flock.Claims.ZK.CausalExe

/-!
# The deployed masked session's zero knowledge

`Flock.zk_deployZk_masked_exe_sent_gap` at the deployed plan (`Split`: `PE`, `czExe`, `encExe`, `finExe`, `UE`), with
every premise discharged. Its round `0`, `OuterReads`, `CoinReads` and `PointReads` hold by construction (`ExePlan`);
`PrefixLive`, `SimCommit`, `SimPiop`, `SimLig`, `SimSession` and `CausalExe` are proved (`prefixLive`, `simCommit`,
`simPiop`, `simLig`, `simSession_of`, `causalExe`); `SimInner` from `SplitInner`'s statements, all proved
(`simInner_of_split`: `innerSlot_holds`, `innerTriple_holds`, `innerBatched`, `innerTauVal`, `innerTauLeaf`, `innerY`);
and `FinExe` from `liveEnd`, `simCommit`, the hybrids and `innerTauVal` (`finExe_of_inner`, through `repTau_of` and
`repOpenRest`).

* `hstep_exe`: the gap's `hstep` at the deployed plan, from `PrefixLive` and `SimSession`. Past the first round, the
  deployed encoding is `sessionCut`'s body on the cut, which `SimSession` makes the masked session's stopped right after
  the round, and `PrefixLive` that answered by the openings.
* `Flock.zk_deployZk_masked_exe_split`: Flock's zero knowledge at the deployed masked session, given only that `H` is
  SHA-512 and `0 < t`.
-/

namespace Claims.ZKMal

open Claims FlockSoundness.Discharge
open CoinLayer

namespace Masked

/-- **The deployed encoding past the first round is the session's body**: at a stop `n ≠ 0` where `setupOf` gives the
spec, `encExe`'s round `n` on the session's cut is `0 ::` body `n` of the masked session stopped right after round
`n`, where it sends one, else `[]` (`SimSession`). -/
theorem encExe_of_sim (hss : SimSession) (nr : Msg → ℕ → ℕ) {x : Msg} (hx : Live x) {w : Claims.Flock.Wit}
    (hw : RZk x w) (τ : UE.Tape) (c : Fin (refK x) → FlockDeploy.Coin) {n : ℕ} (h0 : n ≠ 0) (hnr : ¬ Refused x c n)
    {s : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec}
    (hsx : FlockDeploy.setupOf true x = some (s, spec, stsOf x)) :
    encExe PE x c n ((session nr (tableData PE QE UE)).cutOf x c (wtOf PE x w) τ) =
      if n < (runExe x w (UE.tapeZk τ) c n spec).2.size then 0 :: (runExe x w (UE.tapeZk τ) c n spec).2[n]!
      else [] := by
  have hS : (runCut x c n spec (tabSrcCut PE x (czExe PE x c)
      ((session nr (tableData PE QE UE)).cutOf x c (wtOf PE x w) τ))).2 = (runExe x w (UE.tapeZk τ) c n spec).2 :=
    hss x hx w hw τ c n hnr s spec hsx
  rw [← hS]
  unfold encExe
  rw [ite_eq_right h0]
  simp only [hsx]

/-- **The sent gap's `hstep` at the deployed plan**: at a round after the first where the openings show no refusal,
either the masked session answered by the openings sends past them and the deployed encoding is its next body, or it
throws within them and the encoding is `[]`. -/
theorem hstep_exe (hpl : PrefixLive) (hss : SimSession) (nr : Msg → ℕ → ℕ) :
    ∀ x, Live x → ∀ w, RZk x w → ∀ (τ : UE.Tape) (ops : List (Msg × Msg)), 0 < ops.length →
      ops.length < refK x → ¬ Refused x (padOps ops) ops.length →
      ∀ s spec, FlockDeploy.setupOf true x = some (s, spec, stsOf x) →
        (ops.length < (liveRun x w (UE.tapeZk τ) ops spec).2.size ∧
          encExe PE x (padOps ops) ops.length
              ((session nr (tableData PE QE UE)).cutOf x (padOps ops) (wtOf PE x w) τ) =
            0 :: (liveRun x w (UE.tapeZk τ) ops spec).2[ops.length]!) ∨
        ((∃ e, (liveRun x w (UE.tapeZk τ) ops spec).1 = .error e) ∧
          (liveRun x w (UE.tapeZk τ) ops spec).2.size ≤ ops.length ∧
          encExe PE x (padOps ops) ops.length
              ((session nr (tableData PE QE UE)).cutOf x (padOps ops) (wtOf PE x w) τ) = []) := by
  intro x hx w hw τ ops hpos hlt hnr s spec hsx
  obtain ⟨hiff, hbody, herr⟩ :
      (ops.length < (liveRun x w (UE.tapeZk τ) ops spec).2.size ↔
        ops.length < (runExe x w (UE.tapeZk τ) (padOps ops) ops.length spec).2.size) ∧
      (ops.length < (liveRun x w (UE.tapeZk τ) ops spec).2.size →
        (liveRun x w (UE.tapeZk τ) ops spec).2[ops.length]! =
          (runExe x w (UE.tapeZk τ) (padOps ops) ops.length spec).2[ops.length]!) ∧
      ((liveRun x w (UE.tapeZk τ) ops spec).2.size ≤ ops.length →
        ∃ e, (liveRun x w (UE.tapeZk τ) ops spec).1 = .error e) :=
    hpl x hx w hw (UE.tapeZk τ) ops hpos hlt s spec hsx
  rw [encExe_of_sim hss nr hx hw τ (padOps ops) (Nat.pos_iff_ne_zero.mp hpos) hnr hsx]
  by_cases hn : ops.length < (liveRun x w (UE.tapeZk τ) ops spec).2.size
  · refine .inl ⟨hn, ?_⟩
    rw [ite_eq_left (hiff.mp hn), ← hbody hn]
  · have hle : (liveRun x w (UE.tapeZk τ) ops spec).2.size ≤ ops.length := Nat.le_of_not_lt hn
    exact .inr ⟨herr hle, hle, ite_eq_right fun h => hn (hiff.mpr h)⟩

end Masked

open Masked

open Classical in
/-- **Flock's zero knowledge at the deployed masked session**: `Flock.zk_deployZk_masked_exe_sent_gap` at the deployed
plan (`Split`), its round `0` and coin reads by construction, `hstep` from `prefixLive` and `SimSession` (`hstep_exe`),
`SimSession` from `simCommit` and the hybrids `simPiop`, `simLig`, `SimInner` (`simSession_of`), `SimInner` from its
statements (`simInner_of_split`, `innerTau_of`), its `Causal` `causalExe` and its last round from `liveEnd` and
`innerTauVal` (`finExe_of_inner`). -/
theorem Flock.zk_deployZk_masked_exe_split (H : Msg → Msg) (hH : Claims.hOf H = _root_.Flock.Sha512.hash)
    (nr : Msg → ℕ → ℕ) (t : ℕ) (ht : 0 < t) :
    Claims.Flock.ZK H (zkDeployed _root_.Flock.Readable.relation nr
        (tableData PE (RoundPlan.stop (czExe PE) (encStop (encExe PE)) (finExe PE)) UE) t)
      (fun x => if Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x then
        8 * nL PE x * FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash +
          8 * t * nL PE x * FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash + 2 * (1 / 3) ^ t else 1)
      fun _ => 4 * t * 3 ^ t :=
  have hin : SimInner :=
    simInner_of_split innerSlot_holds innerTriple_holds innerBatched (innerTau_of innerTauVal innerTauLeaf) innerY
  Flock.zk_deployZk_masked_exe_sent_gap H hH nr PE (encExe_zero PE) t ht causalExe (outerReads_czExe PE)
    (coinReads_czExe PE) (pointReads_czExe PE) (hstep_exe prefixLive (simSession_of simCommit simPiop simLig hin) nr)
    (finExe_of_inner liveEnd simCommit simPiop simLig hin innerTauVal H hH nr)

end Claims.ZKMal
