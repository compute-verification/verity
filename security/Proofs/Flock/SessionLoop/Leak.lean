module

public import Proofs.Flock.SessionLoop.ZkAny
public import Proofs.Flock.Claims.ZK.NonVacuous

/-! # `SessionLoopZKAny`'s conclusion fails at a release whose link carries the witness

`Flock.Guarantees.SessionLoopZKAny` concludes `Claims.ZKOut` only where the transport's trusts hold
(`LeanRelease.TrustsStop`), the link's among them. Without them its conclusion fails: at the session loop's own release
(`LeanRelease.ofLoop`) with a link that is the witness's hidden values (`Loop.leakL`), against the verifier that answers
the `Hello` and rejects the link (`Loop.helloRej`), at a statement the deployed setup parses with a reference round and
two witnesses whose hidden values differ, no simulator given the outcome meets `ZKOut` at errors whose sum at the two
witnesses is below `1` (`Loop.not_zkOut_leakL`), the loop's own `Loop.anySim` included.

* `Claims.not_zkOut_two`: two witnesses of one statement whose transcripts, at one verifier and one outcome, an event
  tells apart surely fail `ZKOut` at every simulator, at errors summing below `1` there.
* `Loop.leakL`, `Loop.helloRej`: the link that carries the witness, and the verifier.
* `Loop.nextD_leakL`, `Loop.msgs_leakL`: the release's link there is the witness's hidden values, whatever the worker
  sends, and the transcript is the `Hello` and the link.
* **`Loop.not_zkOut_leakL`**.
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

/-- **The link that carries the witness**: `L₀` with the witness's hidden values as its link. -/
def leakL {R : Type} (L₀ : LeanRelease Flock.Wit R) : LeanRelease Flock.Wit R :=
  { L₀ with link := fun _ w _ _ => witMsg w }

/-- **The verifier that answers the `Hello` and rejects the link**, of the session's `refK x + 2` rounds. -/
def helloRej : VStar Unit :=
  ⟨fun x => refK x + 2, fun _ as _ => .ret (if as.length ≤ 1 then .msg [] else .rej)⟩

/-- **The release's link is the witness's hidden values** at `leakL`, whatever the worker sends, at a statement the
deployed setup parses with a reference round: the coin check takes the `Hello`'s answer, and the link is not the
`Finish`. -/
theorem nextD_leakL (T : CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit ((CoinLayer.session loopNr T).refP.Nonce × (CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Flock.Wit → (CoinLayer.session loopNr T).refP.Nonce × (CoinLayer.session loopNr T).refP.Rand →
      ℕ → _root_.Flock.V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (wk : (LeanRelease.ofLoop T (leakL L₀) real).toSrv.Worker) (w : Flock.Wit) {x : Msg} {s : FlockDeploy.Ins}
    {spec : _root_.Flock.SessionSpec} {sts : Array _root_.Flock.Stmt}
    (hs : FlockDeploy.setupOf true x = some (s, spec, sts)) (hK : 0 < refK x)
    (ρ : (CoinLayer.session loopNr T).refP.Nonce × (CoinLayer.session loopNr T).refP.Rand)
    (κ : (LeanRelease.ofLoop T (leakL L₀) real).toSrv.Chk) (m a : Msg) :
    (LeanRelease.ofLoop T (leakL L₀) real).toSrv.nextD (LeanRelease.ofLoop T (leakL L₀) real).fillOf
      (fun x => refK x + 2) wk x w ρ κ [(m, a)] = witMsg w := by
  have hfa : (LeanRelease.ofLoop T (leakL L₀) real).frameAnswer x ρ [(m, a)] = .ok {} := by
    show freshOf x [] = _
    unfold freshOf
    rw [hs]
    rfl
  have hr : (LeanRelease.ofLoop T (leakL L₀) real).refusesCoins x ρ [(m, a)] = false := by
    refine (LeanRelease.refusesCoins_eq_false _ x ρ _).2 ⟨rfl, fun j hj => ?_⟩
    rcases Nat.le_one_iff_eq_zero_or_eq_one.1 hj with rfl | rfl
    · rfl
    · show ((LeanRelease.ofLoop T (leakL L₀) real).frameAnswer x ρ [(m, a)]).isOk = true
      rw [hfa]
      rfl
  have hk : (LeanRelease.ofLoop T (leakL L₀) real).kindAt x [(m, a)].length = .recomputed := by
    unfold LeanRelease.kindAt
    rw [ite_eq_right (show ¬ [(m, a)].length = 0 by simp),
      ite_eq_left (Or.inl (show [(m, a)].length = 1 from rfl))]
  unfold ZkServer.nextD
  rw [ite_eq_left (Or.inl (show [(m, a)].length + 1 < refK x + 2 by simp only [List.length_singleton]; omega)),
    LeanRelease.frameD_recomp _ _ wk x w ρ κ _ hk]
  unfold LeanRelease.ownMsg
  rw [ite_eq_right (show ¬ (LeanRelease.ofLoop T (leakL L₀) real).refusesCoins x ρ [(m, a)] = true by
      rw [hr]; exact Bool.false_ne_true),
    ite_eq_right (show ¬ [(m, a)].length = (LeanRelease.ofLoop T (leakL L₀) real).last x by
      show ¬ [(m, a)].length = lastOf x
      rw [lastOf_eq]
      simp only [List.length_singleton]
      omega)]
  rfl

/-- **The transcript against `helloRej`** at `leakL`: the `Hello`, answered, then the link, the witness's hidden values,
rejected. -/
theorem msgs_leakL (T : CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit ((CoinLayer.session loopNr T).refP.Nonce × (CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Flock.Wit → (CoinLayer.session loopNr T).refP.Nonce × (CoinLayer.session loopNr T).refP.Rand →
      ℕ → _root_.Flock.V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (wk : (LeanRelease.ofLoop T (leakL L₀) real).toSrv.Worker) (w : Flock.Wit) {x : Msg} {s : FlockDeploy.Ins}
    {spec : _root_.Flock.SessionSpec} {sts : Array _root_.Flock.Stmt}
    (hs : FlockDeploy.setupOf true x = some (s, spec, sts)) (hK : 0 < refK x)
    (ω : ((CoinLayer.session loopNr T).refP.Nonce × (CoinLayer.session loopNr T).refP.Rand) ×
      (LeanRelease.ofLoop T (leakL L₀) real).toSrv.Chk) :
    (runStar CoinOpen.shaH ((LeanRelease.ofLoop T (leakL L₀) real).toSrv.proverD
        (LeanRelease.ofLoop T (leakL L₀) real).fillOf (fun x => refK x + 2) wk w) helloRej x ω ()).msgs =
      [((LeanRelease.ofLoop T (leakL L₀) real).hello x ω.1, .msg []), (witMsg w, .rej)] := by
  obtain ⟨n, hn⟩ : ∃ n, refK x = n + 1 := ⟨refK x - 1, by omega⟩
  have h0 : (LeanRelease.ofLoop T (leakL L₀) real).toSrv.nextD (LeanRelease.ofLoop T (leakL L₀) real).fillOf
      (fun x => refK x + 2) wk x w ω.1 ω.2 [] = (LeanRelease.ofLoop T (leakL L₀) real).hello x ω.1 := by
    unfold ZkServer.nextD
    rw [ite_eq_left (Or.inl (show ([] : List (Msg × Msg)).length + 1 < refK x + 2 by simp))]
    exact LeanRelease.frameD_own _ _ wk x w ω.1 ω.2 [] (LeanRelease.kindAt_zero _ x)
  have h1 := nextD_leakL T L₀ real wk w hs hK ω.1 ω.2 ((LeanRelease.ofLoop T (leakL L₀) real).hello x ω.1) []
  simp [runStar, helloRej, hn, stepWith, ZkServer.proverD, OC.eval, h0, h1]

/-- **`SessionLoopZKAny`'s conclusion fails where its link carries the witness.** At the session loop's release with
the link `leakL L₀`, every tables `real` and worker `wk`, at a statement the deployed setup parses with a reference round
and two witnesses of a relation whose hidden values differ, no simulator given the outcome meets `ZKOut` at errors whose
sum at the two witnesses against `helloRej` is below `1`: against it every run shows the outcome "stopped" and its link
tells the two witnesses apart. -/
theorem not_zkOut_leakL (T : CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit ((CoinLayer.session loopNr T).refP.Nonce × (CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Flock.Wit → (CoinLayer.session loopNr T).refP.Nonce × (CoinLayer.session loopNr T).refP.Rand →
      ℕ → _root_.Flock.V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (wk : (LeanRelease.ofLoop T (leakL L₀) real).toSrv.Worker) {x : Msg} {s : FlockDeploy.Ins}
    {spec : _root_.Flock.SessionSpec} {sts : Array _root_.Flock.Stmt}
    (hs : FlockDeploy.setupOf true x = some (s, spec, sts)) (hK : 0 < refK x) {w₁ w₂ : Flock.Wit}
    (hw : w₁.hid ≠ w₂.hid) (R : Rel Flock.Wit) (h₁ : R x w₁) (h₂ : R x w₂) (S : SimOut Bool)
    (ε : (x : Msg) → Flock.Wit → (Θ : Type) → VStar Θ → Θ → Bool → ℝ)
    (hε : ε x w₁ Unit helloRej () false + ε x w₂ Unit helloRej () false < 1) :
    ¬ ZKOut (fun _ => (CoinLayer.session loopNr T).refP.Nonce × (CoinLayer.session loopNr T).refP.Rand)
        (fun _ => (LeanRelease.ofLoop T (leakL L₀) real).toSrv.Chk) CoinOpen.shaH (fun x => refK x + 2)
        (fun w _ => (LeanRelease.ofLoop T (leakL L₀) real).toSrv.proverD (LeanRelease.ofLoop T (leakL L₀) real).fillOf
          (fun x => refK x + 2) wk w) R
        (fun x τ => goesOnAt (LeanRelease.ofLoop T (leakL L₀) real).stop (refK x + 1) τ) S ε := by
  have hgo : ∀ w ω, goesOnAt (LeanRelease.ofLoop T (leakL L₀) real).stop (refK x + 1)
      (runStar CoinOpen.shaH ((LeanRelease.ofLoop T (leakL L₀) real).toSrv.proverD
        (LeanRelease.ofLoop T (leakL L₀) real).fillOf (fun x => refK x + 2) wk w) helloRej x ω ()).msgs = false := by
    intro w ω
    obtain ⟨n, hn⟩ : ∃ n, refK x = n + 1 := ⟨refK x - 1, by omega⟩
    rw [msgs_leakL T L₀ real wk w hs hK ω, hn]
    rfl
  have hev : ∀ w ω, ((runStar CoinOpen.shaH ((LeanRelease.ofLoop T (leakL L₀) real).toSrv.proverD
      (LeanRelease.ofLoop T (leakL L₀) real).fillOf (fun x => refK x + 2) wk w) helloRej x ω ()).msgs[1]?).map
        Prod.fst = some (witMsg w) := by
    intro w ω
    rw [msgs_leakL T L₀ real wk w hs hK ω]
    rfl
  exact Claims.not_zkOut_two S ε h₁ h₂ helloRej () rfl false
    (fun τ => (τ[1]?).map Prod.fst = some (witMsg w₁))
    (fun ω => ⟨hgo w₁ ω, hev w₁ ω⟩)
    (fun ω => ⟨hgo w₂ ω, fun e => witMsg_ne hw (Option.some.inj ((hev w₂ ω).symm.trans e)).symm⟩) hε

end Loop

end Claims.ZKServer
