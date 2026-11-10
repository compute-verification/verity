module

public import Proofs.Flock.ZkAny.RelayLoopPre
import Proofs.Flock.Claims.ZK.Reference
public import Proofs.Flock.ZkAny.RelayAuditor
public import Proofs.Flock.ZkAny.RelayWorker

/-!
# The relay's model frames at the model's run, from the session's first request

`envPre_withInner` gives `EnvPre` wherever `LoopP` at `LoopJ` holds, given that each of the run's messages is the
deferred release's (`LoopModel.Own`), its first the `Hello` at the relay's nonce and the worker's link the ZK server's
own. Here `LoopP` holds at the session's first request, the first two are read off the model's run of the deferred
server against `V*`, and the link off the deployed worker (`LoopModel.link_deployed`):

* `loopP_start`: at the session's first request, where the loop's frames hold no rep of the session yet and the
  session's slots took none of the model's messages, `LoopP` at `LoopJ` holds wherever the relay's requests to come are
  read as the loop answers them (`LoopReads`) and each reads the setup.
* `pick_setup`: each request that reaches a slot (`pick`) is one the relay read, so reads the relay's setup.
* `LoopModel.own_of_runStar`: the model's messages of a run of the deferred server against `V*` are its own.
* `LoopModel.Deployed.head_hello`: at the deployed loop, such a run's first message is the `Hello` at the tape's nonce
  (`frameD_own`, `loop_helloIs`).
* **`envPre_run`**: `EnvPre` at the relay's requests from its start, at the release with the session loop's own rounds
  of the inner proof over `withoutCross` (Rust's checks are the worker's, so `CrossPasses` holds by definition), at the
  model's run of the deferred server against the relay's auditor as `V*` (`auditorV`, with at least the schedule's
  model slots as rounds) whose worker is the deployed worker (`deployedWk`), which answers the link and the `Finish`
  with the ZK server's own message, where the relay's nonce is the tape's, the relay serves its requests as its IO code
  reads with the auditor's answers (`RelayServes`) and the model's worker gives the worker's round values
  (`WorkerDeterministic`): what `relay_zkOut_env` reads at each randomness.
* **`relayPre_run`**: so the relay's model frames are the model's messages, encoded (`relayPre_of_env`).
-/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock.Firewall.Session (State Placed)
open _root_.Flock (V SessionSpec Stmt)

/-- **`LoopP` at `LoopJ` at the session's first request**: with no session state, no placement and no stream seen,
where the loop's frames hold no rep of the session yet (so it is not stopped and each of its reps' frames is absent)
and the session's slots took none of the model's messages, the relay's requests to come read as the loop answers them
and each reading the setup. -/
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
    unfold _root_.Flock.Firewall.Frames.Sessions.stopped
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

/-- **Each request that reaches a slot reads the relay's setup**: it is one the relay read (`out_sched`). -/
theorem pick_setup {spec : SessionSpec} {sts : Array Stmt} {ins : List (ByteArray × Env)}
    (hs : ∀ p ∈ ins, p.2.setup = .ok (spec, sts)) (hb : ∀ p ∈ ins, ∀ why, (p.2.barrier (some why)).isSome) :
    ∀ e ∈ pick true {} {} ins, e.setup = .ok (spec, sts) := by
  obtain ⟨hsub, -⟩ := out_sched true {} {} ins (.inr ⟨rfl, reach_init⟩) hb
  intro e he
  obtain ⟨p, hp, rfl⟩ := List.mem_map.mp (hsub.subset he)
  exact hs p hp

open Claims.ZKMal in
/-- **A run of the deferred server against `V*` is its own**: where the model's run is the transcript of the deferred
server with the model's worker, statement, witness, tape and check coins against `V*` (`ZkServer.proverD`), each of its
messages is the deferred release's at the history before it (`interact_getElem_fst`). -/
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

/-- **At the deployed loop the run's first message is the `Hello` at the tape's nonce**: where each of the run's
messages is the deferred release's and it has one, the first is the server's own `Hello` (`frameD_own`; the run has a
message past the `Hello`, so the release defers no stop there), the loop's `flock-firewall hello` message
(`loop_helloIs`) at the statement's setup. -/
theorem LoopModel.Deployed.head_hello {T : ZKMal.CoinLayer.TableData}
    {M : LoopModel Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand →
      Nat → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray)}
    {s0 : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt} (hD : LoopModel.Deployed M L₀ real s0 spec sts)
    (hOwn : M.Own) (hne : 0 < M.τ.length) :
    M.msgs.headD [] = _root_.Flock.Firewall.Hello.msg spec ⟨(ZKMal.CoinLayer.nb M.ρ.1).toArray⟩ := by
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

/-- **`EnvPre` at the relay's requests from its start, at the model's run**: at the release
`withFiles (withLink (withInner Masked.dataE L₁.withoutCross (Loop.realOf src) fin) src) src` on the loop's own tables
on `src`, whose `cross` passes every value as Rust's checks are the worker's (`LoopModel.Deployed.crossPasses`), with
`envPre_withInner`'s premises on the loop (`hsrc`, `hfin`, `hS`, `hlive`), where the model's run is the deferred
server's against the relay's auditor `A` as `V*` (`auditorV`, `hτ`), of at least as many rounds as the schedule has
model slots (`hK`; `audK`'s rounds are those), its worker the deployed worker (`hdw`, `LeanRelease.deployedWk`), which
answers the link and the `Finish` with the ZK server's own message, giving the worker's round values `rv`
(`WorkerDeterministic`), and the relay's nonce is the tape's (`hn`): at every requests the worker sends and
every answer of the auditor at the setup `spec`, `sts` (`hs`, `hb`), where at the session's first request the loop's
frames hold no rep of the session and the session's slots took none of the model's messages, and the relay serves its
requests as its IO code reads with `A`'s answers at its tape `θ` (`RelayServes`), they are read as the loop answers
them (`loopReads_auditorV`), what the relay read at each model slot is the model's message there. -/
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
  have h0 : M.msgs.headD [] = _root_.Flock.Firewall.Hello.msg spec nonce := by
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

/-- **The relay's model frames are the model's messages, encoded, at the model's run**: `envPre_run`'s premises give
`EnvPre`, so `relayPre_of_env` gives `RelayPre` at `encOf`. -/
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
