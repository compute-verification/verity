module

public import Proofs.Flock.ZkAny.RelayLoopPre
import Proofs.Flock.Claims.ZK.Reference
public import Proofs.Flock.ZkAny.RelayAuditor
public import Proofs.Flock.ZkAny.RelayWorker

/-! The relay's model frames are the model's messages, encoded, at the model's run from the session's first request, over `withoutCross`, where `CrossPasses` holds by definition (`envPre_run`, `relayPre_run`). -/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock.ZkServer.Session (State Placed)
open _root_.Flock (V SessionSpec Stmt)

theorem loopP_start {W Rr : Type} {M : LoopModel W Rr} {sid : Nat} {spec : SessionSpec} {sts : Array Stmt}
    {nonce : ByteArray} {σ : LoopAt} {es : List Env} (hc : σ.c = 0) (hnew : ∀ p ∈ σ.R.reps, p.sid ≠ sid)
    (hR : LoopReads M sid spec sts nonce σ {} {} es) (hes : ∀ e ∈ es, e.setup = .ok (spec, sts)) :
    LoopP M (LoopJ M sid spec sts) sid spec sts nonce [] {} {} es M.msgs := by
  have hkey : ∀ i r p, frameAt σ.R sid i r = some p → False := by
    intro i r p hp
    have hk := Array.find?_some hp
    simp only [keyIs, Bool.and_eq_true, beq_iff_eq] at hk
    exact hnew p (Array.mem_of_find?_eq_some hp) hk.1.1
  have hstop : σ.R.stopped sid = false := by
    unfold _root_.Flock.ZkServer.Frames.Sessions.stopped
    exact Array.any_eq_false'.mpr fun p hp => by simp [hnew p hp]
  have hF : LoopFacts sid spec [] {} {} σ :=
    { setup := fun h => by cases h
      fresh := fun i r _ _ _ p hp => (hkey i r p hp).elim
      live := rfl
      reach := reach_init
      start := fun _ => hc
      count := fun h => by cases h }
  have hJ : LoopJ M sid spec sts [] {} σ :=
    { stop := hstop
      unsent := fun _ => rfl
      reps := fun _ _ _ _ h => by simp at h
      plan := fun h => by cases h
      seenAt := seenAt_nil (by rw [hc, LoopModel.hist, length_histOf_take]; omega)
      early := fun _ => rfl
      built := fun h => absurd h (by omega)
      sess := fun h => by cases h
      links := linksOk_empty spec
      link := fun h => absurd h (by omega)
      sizes := fun h => by cases h
      took := fun _ => by
        rw [hc]
        exact LoopModel.TookAt.zero M rfl }
  exact ⟨σ, by rw [hc, List.drop_zero], hF, hJ, hR, hes⟩

theorem pick_setup {spec : SessionSpec} {sts : Array Stmt} {ins : List (ByteArray × Env)}
    (hs : ∀ p ∈ ins, p.2.setup = .ok (spec, sts)) (hb : ∀ p ∈ ins, ∀ why, (p.2.barrier (some why)).isSome) :
    ∀ e ∈ pick true {} {} ins, e.setup = .ok (spec, sts) := by
  obtain ⟨hsub, -⟩ := out_sched true {} {} ins (.inr ⟨rfl, reach_init⟩) hb
  intro e he
  obtain ⟨p, hp, rfl⟩ := List.mem_map.mp (hsub.subset he)
  exact hs p hp

open Claims.ZKMal in
theorem LoopModel.own_of_runStar {W Rr Θ : Type} {M : LoopModel W Rr} {H : Msg → Msg} {Vs : VStar Θ} {θ : Θ}
    (hτ : M.τ = (runStar H (M.L.toSrv.proverD M.L.fillOf M.nN M.wk M.w) Vs M.x (M.ρ, M.κ) θ).msgs) : M.Own := by
  intro j hj
  have e : M.τ = interact (fun as => ((Vs.next M.x as θ).eval H).1)
      (fun h => (((M.L.toSrv.proverD M.L.fillOf M.nN M.wk M.w).next M.x h (M.ρ, M.κ)).eval H).1) (Vs.k M.x) [] :=
    hτ.trans (ZKMal.runStar_msgs _ _ _ _ _ _)
  have h := interact_getElem_fst _ _ _ [] j (by rw [← e]; exact hj)
  rw [← e] at h
  simp only [List.getElem?_eq_getElem hj, Option.map_some, List.nil_append, Option.some.injEq] at h
  exact h

theorem LoopModel.Deployed.head_hello {T : ZKMal.CoinLayer.TableData}
    {M : LoopModel Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand →
      Nat → V (_root_.Flock.ZkServer.Finish.Tab × Array ByteArray)}
    {s0 : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt} (hD : LoopModel.Deployed M L₀ real s0 spec sts)
    (hOwn : M.Own) (hne : 0 < M.τ.length) :
    M.msgs.headD [] = _root_.Flock.ZkServer.Hello.msg spec ⟨(ZKMal.CoinLayer.nb M.ρ.1).toArray⟩ := by
  have hlast := hD.last_eq
  have hnext := hD.next
  have hm : M.msgs.headD [] = (M.τ[0]'hne).1 := by
    obtain ⟨a, τ', hτ⟩ := List.exists_cons_of_length_pos hne
    simp only [LoopModel.msgs, hτ, List.map_cons, List.headD_cons, List.getElem_cons_zero]
  have h0 := hOwn 0 hne
  rw [show M.hist 0 = [] from rfl] at h0
  unfold ZkServer.nextD at h0
  rw [ite_eq_left (Or.inl (by simp only [List.length_nil]; omega)),
    LeanRelease.frameD_own _ _ _ _ _ _ _ _ (LeanRelease.kindAt_zero _ _), hD.rel] at h0
  rw [hm, h0, show (LeanRelease.ofLoop T L₀ real).hello M.x M.ρ = ZKMal.CoinLayer.helloMsg M.x M.ρ.1 from
    LeanRelease.loop_helloIs _ T _ M.x M.ρ.1 M.ρ.2]
  unfold ZKMal.CoinLayer.helloMsg
  rw [Claims.ZKMal.sessOf_flock hD.setup]

theorem envPre_run
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
      (LeanRelease.withInner ZKMal.Masked.dataE L₁.withoutCross (Loop.realOf src) fin) src) src) (Loop.realOf src) s0
      spec sts)
    (hsrc : M.src = src M.x M.w M.ρ) (hfin : M.fin = fin M.x M.w M.ρ) (hS : Loop.Level0Src src M.x M.w)
    (hlive : ZKMal.Masked.Live M.x) {Θ : Type} {H : Msg → Msg}
    {k : Msg → ℕ} {A : Θ → List ByteArray → Option ByteArray} {θ : Θ}
    (hτ : M.τ = (runStar H (M.L.toSrv.proverD M.L.fillOf M.nN M.wk M.w) (auditorV k spec sts A) M.x (M.ρ, M.κ)
      θ).msgs)
    (hK : (tmplOf (.ok (spec, sts))).countP isModel ≤ k M.x)
    {dv : M.L.toSrv.Worker} (hdw : M.wk = M.L.deployedWk dv)
    {rv : List (Msg × Msg) → Option Msg} (hwd : WorkerDeterministic M rv)
    {sid : Nat} {nonce : ByteArray} (hn : nonce = ⟨(ZKMal.CoinLayer.nb M.ρ.1).toArray⟩)
    {ins : List (ByteArray × Env)} (hs : ∀ p ∈ ins, p.2.setup = .ok (spec, sts))
    (hb : ∀ p ∈ ins, ∀ why, (p.2.barrier (some why)).isSome) {σ : LoopAt} (hc : σ.c = 0)
    (hnew : ∀ p ∈ σ.R.reps, p.sid ≠ sid) (hR : RelayServes M rv (A θ) sid spec sts nonce σ ins) :
    EnvPre spec [] (relaySteps true ins) M.msgs := by
  have hOwn := LoopModel.own_of_runStar hτ
  have hk : 0 < k M.x := lt_of_lt_of_le (audK_pos spec sts M.x) hK
  have hne : 0 < M.τ.length := (runStar_auditorV hτ).2 0 hk fun _ hi => absurd hi (Nat.not_lt_zero _)
  have h0 : M.msgs.headD [] = _root_.Flock.ZkServer.Hello.msg spec nonce := by
    rw [hn]
    exact hD.head_hello hOwn hne
  have hwk := LoopModel.link_deployed hdw (M.hist 1) (by rw [LoopModel.hist, length_histOf_take]; omega)
  have hF : ∀ es, (∀ e ∈ es, e.setup = .ok (spec, sts)) → LoopReads M sid spec sts nonce σ {} {} es →
      EnvPre spec [] (schedSteps true {} {} es) M.msgs := fun es hes hLR =>
    envPre_withInner hD hsrc hfin hS hlive hD.crossPasses hOwn hn h0 hwk es [] {} {} M.msgs
      (loopP_start hc hnew hLR hes)
  exact hF _ (pick_setup hs hb)
    (loopReads_auditorV hτ hK
      (fun _ _ _ _ _ _ hv hp => loopRead_of_serve hv hwd (LoopModel.finish_deployed hdw) hp)
      (fun _ _ _ _ _ _ hv => hv.ans_none) hc hF (pick_setup hs hb) (pick_barrier hb) hR.2 hR.1)

theorem relayPre_run
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
      (LeanRelease.withInner ZKMal.Masked.dataE L₁.withoutCross (Loop.realOf src) fin) src) src) (Loop.realOf src) s0
      spec sts)
    (hsrc : M.src = src M.x M.w M.ρ) (hfin : M.fin = fin M.x M.w M.ρ) (hS : Loop.Level0Src src M.x M.w)
    (hlive : ZKMal.Masked.Live M.x) {Θ : Type} {H : Msg → Msg}
    {k : Msg → ℕ} {A : Θ → List ByteArray → Option ByteArray} {θ : Θ}
    (hτ : M.τ = (runStar H (M.L.toSrv.proverD M.L.fillOf M.nN M.wk M.w) (auditorV k spec sts A) M.x (M.ρ, M.κ)
      θ).msgs)
    (hK : (tmplOf (.ok (spec, sts))).countP isModel ≤ k M.x)
    {dv : M.L.toSrv.Worker} (hdw : M.wk = M.L.deployedWk dv)
    {rv : List (Msg × Msg) → Option Msg} (hwd : WorkerDeterministic M rv)
    {sid : Nat} {nonce : ByteArray} (hn : nonce = ⟨(ZKMal.CoinLayer.nb M.ρ.1).toArray⟩)
    {ins : List (ByteArray × Env)} (hs : ∀ p ∈ ins, p.2.setup = .ok (spec, sts))
    (hb : ∀ p ∈ ins, ∀ why, (p.2.barrier (some why)).isSome) {σ : LoopAt} (hc : σ.c = 0)
    (hnew : ∀ p ∈ σ.R.reps, p.sid ≠ sid) (hR : RelayServes M rv (A θ) sid spec sts nonce σ ins) :
    RelayPre (relayTrace true ins) (encOf spec sts M.msgs) :=
  relayPre_of_env ins M.msgs hs hb
    (envPre_run hD hsrc hfin hS hlive hτ hK hdw hwd hn hs hb hc hnew hR)

end Claims.ZKServer.Relay
