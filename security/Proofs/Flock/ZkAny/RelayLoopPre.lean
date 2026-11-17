module

public import Proofs.Flock.ZkAny.RelayLoopDeployed
public import Proofs.Flock.ZkAny.RelayLoopFinish
public import Proofs.Flock.ZkAny.RelayLoopOwnAt
public import Proofs.Flock.ZkAny.RelayFreshStart
import Proofs.Flock.Claims.ZK.Reference

/-! `EnvPre` at the deployed loop, from `LoopP` at `LoopJ` on the loop's own tables (`envPre_deployed`, `envPre_withInner`). -/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock.ZkServer.Session (State Placed planOf)
open _root_.Flock (V SessionSpec Stmt)

theorem LoopModel.Deployed.last_eq {T : ZKMal.CoinLayer.TableData}
    {M : LoopModel Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand →
      Nat → V (_root_.Flock.ZkServer.Finish.Tab × Array ByteArray)}
    {s0 : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt} (hD : LoopModel.Deployed M L₀ real s0 spec sts) :
    M.L.last M.x = (planOf spec sts).size + 2 := by
  rw [hD.rel]
  show Loop.lastOf M.x = _
  rw [Loop.lastOf_eq, Claims.ZKMal.refK_of_flock hD.setup, Loop.rounds_eq hD.sizes, ← Loop.plan_size]
  omega

theorem envPre_deployed {M : LoopModel Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Nonce ×
        (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Rand)}
    {L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Nonce ×
        (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Rand)}
    {src : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Nonce ×
        (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Rand →
      Nat → V (Array _root_.Flock.F128 × ByteArray × Array ByteArray)}
    {s0 : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hD : LoopModel.Deployed M L₀ (Loop.realOf src) s0 spec sts) (hsrc : M.src = src M.x M.w M.ρ)
    (hS : Loop.Level0Src src M.x M.w) (hlive : ZKMal.Masked.Live M.x) (hOwn : M.Own) {sid : Nat}
    {nonce : ByteArray} (hn : nonce = ⟨(ZKMal.CoinLayer.nb M.ρ.1).toArray⟩)
    (h0 : M.msgs.headD [] = _root_.Flock.ZkServer.Hello.msg spec nonce)
    (hwk : M.wk M.x M.w M.ρ (M.hist 1) 0 = some (M.L.ownMsg M.x M.w M.ρ (M.hist 1), false))
    (hRF : ∀ {stream : String} {n : Nat}, ∀ (seen : List String) (st : State) (pl : Placed) (e : Env) (σ σ' : LoopAt)
      (i r : Nat) (dom : ByteArray) (v : Claims.Msg), slot st pl = some (.round stream n) →
      LoopFacts sid spec seen st pl σ → LoopJ M sid spec sts seen st σ → σ.c < M.τ.length →
      M.wk M.x M.w M.ρ (M.hist σ.c) 0 = some (v, false) →
      RoundRun sid st pl stream n i r dom ⟨v.toArray⟩ e σ.R σ.L0 σ'.R →
      (∀ p, frameAt σ.R sid i r = some p → p.cfg = M.L.cfg M.x M.ρ (M.hist σ.c)) → SaltsSrc M sid σ →
      InnerSrc M sid i r σ → StartOk sts sid i r σ → (∃ o, e.round = .ok o) →
      RoundFacts M sid spec seen stream n i r dom v σ) :
    ∀ (es : List Env) (seen : List String) (st : State) (pl : Placed) (ms : List Claims.Msg),
      LoopP M (LoopJ M sid spec sts) sid spec sts nonce seen st pl es ms →
        EnvPre spec seen (schedSteps true st pl es) ms := by
  have hlast := hD.last_eq
  have hnext := hD.next
  exact envPre_loop loopKept_ctl_all (slotCase_hello h0 (loopKept_hello (hD.helloChk hn)))
    (slotCase_commit hOwn (by omega) (loopKept_commit hD.setup (by omega) hwk hD.req hD.frameAnswer_one))
    (fun _ _ => slotCase_round hOwn hRF (loopKept_round_deployed hD hsrc hS hlive hRF))
    (slotCase_finish hOwn fun _ _ _ _ hsl _ hJ => by rw [hJ.stop, slot_finish hsl, Bool.true_and])

theorem ownSched_deployed
    {L₁ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Nonce ×
        (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Rand)}
    {src : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Nonce ×
        (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Rand →
      Nat → V (Array _root_.Flock.F128 × ByteArray × Array ByteArray)}
    {fin : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Nonce ×
        (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Rand →
      List (Msg × Msg) → Array ByteArray}
    {M : LoopModel Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Nonce ×
        (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Rand)}
    {s0 : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hD : LoopModel.Deployed M (LeanRelease.withFiles (LeanRelease.withLink
      (LeanRelease.withInner ZKMal.Masked.dataE L₁ (Loop.realOf src) fin) src) src) (Loop.realOf src) s0 spec sts)
    (hsrc : M.src = src M.x M.w M.ρ) (hfin : M.fin = fin M.x M.w M.ρ) (hS : Loop.Level0Src src M.x M.w)
    (hlive : ZKMal.Masked.Live M.x) {sid : Nat} {stream : String} {n : Nat} :
    OwnSched M sid spec sts stream n := by
  intro seen st pl e σ σ' i r dom v hsl hF hJ hlt _ hr _ hsal hin _ _ hsn q pc f nd hrp
  obtain ⟨hsent, -, -, -⟩ := slot_round hsl
  obtain ⟨C, hC⟩ := Option.isSome_iff_exists.mp (hF.reach.sess hsent)
  obtain ⟨-, hiff⟩ := rep_seen hD hsl hF hJ hr.rep_eq hlt
  obtain ⟨h3, hs⟩ := hiff.1 hsn
  have htk := hJ.took hlt
  exact placeSched_own hD (Loop.realTabs_realOf src M.x M.w M.ρ hD.setup) hsl hF hJ hr.rep_eq hlt
    (realAt_slot hD hsrc hS hlive hsl hF hJ hsal) (LeanRelease.answeredR_same (htk.opens hC) htk.2.1 htk.2.2 h3 hs)
    hin hfin hrp hsn

theorem envPre_withInner
    {L₁ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Nonce ×
        (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Rand)}
    {src : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Nonce ×
        (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Rand →
      Nat → V (Array _root_.Flock.F128 × ByteArray × Array ByteArray)}
    {fin : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Nonce ×
        (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Rand →
      List (Msg × Msg) → Array ByteArray}
    {M : LoopModel Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Nonce ×
        (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Rand)}
    {s0 : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hD : LoopModel.Deployed M (LeanRelease.withFiles (LeanRelease.withLink
      (LeanRelease.withInner ZKMal.Masked.dataE L₁ (Loop.realOf src) fin) src) src) (Loop.realOf src) s0 spec sts)
    (hsrc : M.src = src M.x M.w M.ρ) (hfin : M.fin = fin M.x M.w M.ρ) (hS : Loop.Level0Src src M.x M.w)
    (hlive : ZKMal.Masked.Live M.x) (hcp : M.L.CrossPasses) (hOwn : M.Own) {sid : Nat}
    {nonce : ByteArray} (hn : nonce = ⟨(ZKMal.CoinLayer.nb M.ρ.1).toArray⟩)
    (h0 : M.msgs.headD [] = _root_.Flock.ZkServer.Hello.msg spec nonce)
    (hwk : M.wk M.x M.w M.ρ (M.hist 1) 0 = some (M.L.ownMsg M.x M.w M.ρ (M.hist 1), false)) :
    ∀ (es : List Env) (seen : List String) (st : State) (pl : Placed) (ms : List Claims.Msg),
      LoopP M (LoopJ M sid spec sts) sid spec sts nonce seen st pl es ms →
        EnvPre spec seen (schedSteps true st pl es) ms :=
  envPre_deployed hD hsrc hS hlive hOwn hn h0 hwk fun {_ _} =>
    roundFacts_deployed hD hsrc hS hlive hcp (ownSched_deployed hD hsrc hfin hS hlive) (freshStart_deployed hD hlive)

end Claims.ZKServer.Relay
