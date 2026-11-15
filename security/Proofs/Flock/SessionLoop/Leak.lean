module

public import Proofs.Flock.SessionLoop.ZkAny
public import Proofs.Flock.Claims.ZK.NonVacuous

/-! # `SessionLoopZKAny`'s conclusion fails at a release whose link tells two witnesses apart

`Flock.Guarantees.SessionLoopZKAny` concludes `Claims.ZKOut` only where the transport's trusts hold
(`LeanRelease.TrustsStop`), the link's among them. Without them its conclusion fails: at the session loop's own release
(`LeanRelease.ofLoop`) with a link an event tells apart at two witnesses, against the verifier that answers the `Hello`
and rejects the link (`Loop.helloRej`), at a statement the deployed setup parses with a reference round, no simulator
given the outcome meets `ZKOut` at errors whose sum at the two witnesses is below `1` (`Loop.not_zkOut_link`), the
loop's own `Loop.anySim` included.

* `Claims.not_zkOut_two`: two witnesses of one statement whose transcripts, at one verifier and one outcome, an event
  tells apart surely fail `ZKOut` at every simulator, at errors summing below `1` there.
* `Loop.helloRej`: the verifier.
* `Loop.nextD_link`, `Loop.msgs_link`: the release's first message past the `Hello` is its link, whatever the worker
  sends, and the transcript is the `Hello` and the link.
* **`Loop.not_zkOut_link`**.
-/

@[expose] public section

namespace Claims

/-- **Two witnesses an event tells apart fail `ZKOut`**: where at one verifier every run at `w₁` shows the outcome `o`
and the event, and every run at `w₂` shows `o` and not the event, no simulator meets `ZKOut` at errors summing below `1`
there. -/
theorem not_zkOut_two {W St : Type} {N Ω : Msg → Type} [∀ x, Fintype (N x)] [∀ x, Nonempty (N x)]
    [∀ x, Fintype (Ω x)] [∀ x, Nonempty (Ω x)] {H : Msg → Msg} {k : Msg → ℕ} {P : W → (x : Msg) → Prover (N x × Ω x)}
    {R : Rel W} {out : Msg → List (Msg × Out) → St} (S : SimOut St)
    (ε : (x : Msg) → W → (Θ : Type) → VStar Θ → Θ → St → ℝ) {x : Msg} {w₁ w₂ : W} (h₁ : R x w₁) (h₂ : R x w₂)
    {Θ : Type} (Vs : VStar Θ) (θ : Θ) (hk : Vs.k x = k x) (o : St) (Ev : List (Msg × Out) → Prop)
    (e₁ : ∀ ω, out x (runStar H (P w₁ x) Vs x ω θ).msgs = o ∧ Ev (runStar H (P w₁ x) Vs x ω θ).msgs)
    (e₂ : ∀ ω, out x (runStar H (P w₂ x) Vs x ω θ).msgs = o ∧ ¬ Ev (runStar H (P w₂ x) Vs x ω θ).msgs)
    (hε : ε x w₁ Θ Vs θ o + ε x w₂ Θ Vs θ o < 1) :
    ¬ ZKOut N Ω H k P R out S ε := by
  intro h
  have a₁ := h x w₁ h₁ Θ Vs θ hk o Ev
  have a₂ := h x w₂ h₂ Θ Vs θ hk o fun τ => ¬ Ev τ
  beta_reduce at a₂
  rw [Pr_congr (q := fun _ => True) fun ω => iff_true_intro (e₁ ω), ZKMal.Pr_true] at a₁
  rw [Pr_congr (q := fun _ => True) fun ω => iff_true_intro (e₂ ω), ZKMal.Pr_true, ZKMal.Pr_compl] at a₂
  linarith

end Claims

namespace Claims.ZKServer

open Claims Claims.ZKMal
open FlockSoundness.Discharge

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Loop

/-- **The verifier that answers the `Hello` and rejects the link**, of the session's `refK x + 2` rounds. -/
def helloRej : VStar Unit :=
  ⟨fun x => refK x + 2, fun _ as _ => .ret (if as.length ≤ 1 then .msg [] else .rej)⟩

/-- **The release's message past the `Hello` is its link** at `L₁`, whatever the worker sends, at a statement the
deployed setup parses with a reference round: the coin check takes the `Hello`'s answer, and the link is not the
`Finish`. -/
theorem nextD_link (T : CoinLayer.TableData)
    (L₁ : LeanRelease Flock.Wit ((CoinLayer.session loopNr T).refP.Nonce × (CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Flock.Wit → (CoinLayer.session loopNr T).refP.Nonce × (CoinLayer.session loopNr T).refP.Rand →
      ℕ → _root_.Flock.V (_root_.Flock.ZkServer.Finish.Tab × Array ByteArray))
    (wk : (LeanRelease.ofLoop T L₁ real).toSrv.Worker) (w : Flock.Wit) {x : Msg} {s : FlockDeploy.Ins}
    {spec : _root_.Flock.SessionSpec} {sts : Array _root_.Flock.Stmt}
    (hs : FlockDeploy.setupOf true x = some (s, spec, sts)) (hK : 0 < refK x)
    (ρ : (CoinLayer.session loopNr T).refP.Nonce × (CoinLayer.session loopNr T).refP.Rand)
    (κ : (LeanRelease.ofLoop T L₁ real).toSrv.Chk) (m a : Msg) :
    (LeanRelease.ofLoop T L₁ real).toSrv.nextD (LeanRelease.ofLoop T L₁ real).fillOf
      (fun x => refK x + 2) wk x w ρ κ [(m, a)] = L₁.link x w ρ [(m, a)] := by
  have hfa : (LeanRelease.ofLoop T L₁ real).frameAnswer x ρ [(m, a)] = .ok {} := by
    show freshOf x [] = _
    unfold freshOf
    rw [hs]
    rfl
  have hr : (LeanRelease.ofLoop T L₁ real).refusesCoins x ρ [(m, a)] = false := by
    refine (LeanRelease.refusesCoins_eq_false _ x ρ _).2 ⟨rfl, fun j hj => ?_⟩
    rcases Nat.le_one_iff_eq_zero_or_eq_one.1 hj with rfl | rfl
    · rfl
    · show ((LeanRelease.ofLoop T L₁ real).frameAnswer x ρ [(m, a)]).isOk = true
      rw [hfa]
      rfl
  have hk : (LeanRelease.ofLoop T L₁ real).kindAt x [(m, a)].length = .recomputed := by
    unfold LeanRelease.kindAt
    rw [ite_eq_right (show ¬ [(m, a)].length = 0 by simp),
      ite_eq_left (Or.inl (show [(m, a)].length = 1 from rfl))]
  unfold ZkServer.nextD
  rw [ite_eq_left (Or.inl (show [(m, a)].length + 1 < refK x + 2 by simp only [List.length_singleton]; omega)),
    LeanRelease.frameD_recomp _ _ wk x w ρ κ _ hk]
  unfold LeanRelease.ownMsg
  rw [ite_eq_right (show ¬ (LeanRelease.ofLoop T L₁ real).refusesCoins x ρ [(m, a)] = true by
      rw [hr]; exact Bool.false_ne_true),
    ite_eq_right (show ¬ [(m, a)].length = (LeanRelease.ofLoop T L₁ real).last x by
      show ¬ [(m, a)].length = lastOf x
      rw [lastOf_eq]
      simp only [List.length_singleton]
      omega)]
  rfl

/-- **The transcript against `helloRej`** at `L₁`: the `Hello`, answered, then the link, rejected. -/
theorem msgs_link (T : CoinLayer.TableData)
    (L₁ : LeanRelease Flock.Wit ((CoinLayer.session loopNr T).refP.Nonce × (CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Flock.Wit → (CoinLayer.session loopNr T).refP.Nonce × (CoinLayer.session loopNr T).refP.Rand →
      ℕ → _root_.Flock.V (_root_.Flock.ZkServer.Finish.Tab × Array ByteArray))
    (wk : (LeanRelease.ofLoop T L₁ real).toSrv.Worker) (w : Flock.Wit) {x : Msg} {s : FlockDeploy.Ins}
    {spec : _root_.Flock.SessionSpec} {sts : Array _root_.Flock.Stmt}
    (hs : FlockDeploy.setupOf true x = some (s, spec, sts)) (hK : 0 < refK x)
    (ω : ((CoinLayer.session loopNr T).refP.Nonce × (CoinLayer.session loopNr T).refP.Rand) ×
      (LeanRelease.ofLoop T L₁ real).toSrv.Chk) :
    (runStar CoinOpen.shaH ((LeanRelease.ofLoop T L₁ real).toSrv.proverD
        (LeanRelease.ofLoop T L₁ real).fillOf (fun x => refK x + 2) wk w) helloRej x ω ()).msgs =
      [((LeanRelease.ofLoop T L₁ real).hello x ω.1, .msg []),
        (L₁.link x w ω.1 [((LeanRelease.ofLoop T L₁ real).hello x ω.1, [])], .rej)] := by
  obtain ⟨n, hn⟩ : ∃ n, refK x = n + 1 := ⟨refK x - 1, by omega⟩
  have h0 : (LeanRelease.ofLoop T L₁ real).toSrv.nextD (LeanRelease.ofLoop T L₁ real).fillOf
      (fun x => refK x + 2) wk x w ω.1 ω.2 [] = (LeanRelease.ofLoop T L₁ real).hello x ω.1 := by
    unfold ZkServer.nextD
    rw [ite_eq_left (Or.inl (show ([] : List (Msg × Msg)).length + 1 < refK x + 2 by simp))]
    exact LeanRelease.frameD_own _ _ wk x w ω.1 ω.2 [] (LeanRelease.kindAt_zero _ x)
  have h1 := nextD_link T L₁ real wk w hs hK ω.1 ω.2 ((LeanRelease.ofLoop T L₁ real).hello x ω.1) []
  simp [runStar, helloRej, hn, stepWith, ZkServer.proverD, OC.eval, h0, h1]

/-- **`SessionLoopZKAny`'s conclusion fails where its link tells two witnesses apart.** At the session loop's release
on `L₁`, every tables `real` and worker `wk`, at a statement the deployed setup parses with a reference round and two
witnesses of a relation whose links an event `Ev` tells apart, no simulator given the outcome meets `ZKOut` at errors
whose sum at the two witnesses against `helloRej` is below `1`: against it every run shows the outcome "stopped", and its
link shows `Ev` at the one witness and not at the other. -/
theorem not_zkOut_link (T : CoinLayer.TableData)
    (L₁ : LeanRelease Flock.Wit ((CoinLayer.session loopNr T).refP.Nonce × (CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Flock.Wit → (CoinLayer.session loopNr T).refP.Nonce × (CoinLayer.session loopNr T).refP.Rand →
      ℕ → _root_.Flock.V (_root_.Flock.ZkServer.Finish.Tab × Array ByteArray))
    (wk : (LeanRelease.ofLoop T L₁ real).toSrv.Worker) {x : Msg} {s : FlockDeploy.Ins}
    {spec : _root_.Flock.SessionSpec} {sts : Array _root_.Flock.Stmt}
    (hs : FlockDeploy.setupOf true x = some (s, spec, sts)) (hK : 0 < refK x) {w₁ w₂ : Flock.Wit} (Ev : Msg → Prop)
    (e₁ : ∀ ρ h, Ev (L₁.link x w₁ ρ h)) (e₂ : ∀ ρ h, ¬ Ev (L₁.link x w₂ ρ h)) (R : Rel Flock.Wit) (h₁ : R x w₁)
    (h₂ : R x w₂) (S : SimOut Bool) (ε : (x : Msg) → Flock.Wit → (Θ : Type) → VStar Θ → Θ → Bool → ℝ)
    (hε : ε x w₁ Unit helloRej () false + ε x w₂ Unit helloRej () false < 1) :
    ¬ ZKOut (fun _ => (CoinLayer.session loopNr T).refP.Nonce × (CoinLayer.session loopNr T).refP.Rand)
        (fun _ => (LeanRelease.ofLoop T L₁ real).toSrv.Chk) CoinOpen.shaH (fun x => refK x + 2)
        (fun w _ => (LeanRelease.ofLoop T L₁ real).toSrv.proverD (LeanRelease.ofLoop T L₁ real).fillOf
          (fun x => refK x + 2) wk w) R
        (fun x τ => goesOnAt (LeanRelease.ofLoop T L₁ real).stop (refK x + 1) τ) S ε := by
  have hgo : ∀ w ω, goesOnAt (LeanRelease.ofLoop T L₁ real).stop (refK x + 1)
      (runStar CoinOpen.shaH ((LeanRelease.ofLoop T L₁ real).toSrv.proverD
        (LeanRelease.ofLoop T L₁ real).fillOf (fun x => refK x + 2) wk w) helloRej x ω ()).msgs = false := by
    intro w ω
    obtain ⟨n, hn⟩ : ∃ n, refK x = n + 1 := ⟨refK x - 1, by omega⟩
    rw [msgs_link T L₁ real wk w hs hK ω, hn]
    rfl
  have hev : ∀ w ω, ((runStar CoinOpen.shaH ((LeanRelease.ofLoop T L₁ real).toSrv.proverD
      (LeanRelease.ofLoop T L₁ real).fillOf (fun x => refK x + 2) wk w) helloRej x ω ()).msgs[1]?).map Prod.fst =
        some (L₁.link x w ω.1 [((LeanRelease.ofLoop T L₁ real).hello x ω.1, [])]) := by
    intro w ω
    rw [msgs_link T L₁ real wk w hs hK ω]
    rfl
  exact Claims.not_zkOut_two S ε h₁ h₂ helloRej () rfl false
    (fun τ => ∃ m, (τ[1]?).map Prod.fst = some m ∧ Ev m)
    (fun ω => ⟨hgo w₁ ω, _, hev w₁ ω, e₁ _ _⟩)
    (fun ω => ⟨hgo w₂ ω, fun ⟨m, hm, hE⟩ => by
      obtain rfl := Option.some.inj ((hev w₂ ω).symm.trans hm)
      exact e₂ _ _ hE⟩) hε

end Loop

end Claims.ZKServer
