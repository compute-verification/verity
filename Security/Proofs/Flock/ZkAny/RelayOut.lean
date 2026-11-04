module

public import Definitions.Claims.ZKOut
public import Proofs.Flock.Claims.Sampling.Draws
public import Proofs.Flock.ZkAny.Generic
public import Proofs.Flock.ZkAny.Relay
public import Proofs.Flock.ZkAny.Unswitched

/-!
# The deployed relay's release, per outcome

`Relay.relay_release_cut` holds the deployed relay's release to the schedule: the setup's control frames at their places
and the model's frames (`relayFrames`) at theirs, over the schedule cut where the release ended. So where the model's
frames are a release that is zero knowledge per outcome, so is the relay's, at the model's outcome and where the
release ended, at the model's error: the outcome is each verdict and where the release ended.

* `PerOutcome.refine`: released bytes per outcome, read through a public function of a second outcome, are per outcome
  at the pair, at the same error.
* `prCoin_le_of_Pr`, `perOutcome_of_zkOut`: `ZKOut` at one statement, witness and verifier is `PerOutcome` over the
  protocol's randomness, its error read in `ℝ≥0∞`.
* `relay_perOutcome`: where the relay's model frames are the model's release encoded by a public `enc`, and the auditor
  answers each coin request with coins, the relay's release is per outcome at the model's outcome and `relayEnd`, one
  of the schedule's `(tmplOf s).length + 1` cuts. Its simulator is the model's, encoded and woven with the setup's
  control frames; its error is the model's. `PerOutcome.le_card` then costs `|St| · ((tmplOf s).length + 1)` outcomes.
* `relay_perOutcome_slots`: without the auditor's premise, the same at the slots the release reached (`relaySlots`, cut
  at `relayCut`). A coin request the auditor answered without coins comes again as often as the worker sends it, so
  this outcome counts the worker's repeats and has no bound but the number of requests.
* `relay_zkOut`: `relay_perOutcome` at a model that is `ZKOut`.
* `relay_preFinish`: at the deferred release, where the relay's model frames are its messages each encoded by a public
  `e`, the relay's release before the `Finish` is the unswitched prover's run (`LeanRelease.release_preFinish_runStar`),
  encoded and woven with the control frames at the schedule's cut, on either branch.
* `relay_stop`: at the deferred release, on its stop branch, the relay's release is the unswitched prover's run cut at
  its last round, then the stop (`LeanRelease.release_stop_runStar`), encoded and woven with the control frames at the
  schedule's cut.

The premise that the relay's model frames are the model's release (`hfr`) is read here, not proved: what the relay's
`Env` holds at the model's run (its rep frames' rounds, `Level0.commit`'s roots, its `Hello`) against the model's
messages. Scope: the released bytes and their order, not their times.
-/

@[expose] public section

namespace FlockSoundness.ZK

open Verity.Game
open scoped ENNReal

/-- **Released bytes per outcome, read through a public function of a second outcome**: where the bytes are `g` at the
second outcome `pos` of bytes `rel` simulated per outcome, they are simulated per outcome at the pair of outcomes, by
`g` at the second of the first's simulator, at the same error. -/
theorem PerOutcome.refine {Ω St Ωs V Q V' : Type} [Fintype Ω] [Fintype Ωs] {out : Ω → St} {rel : Ω → V}
    {S : St → Ωs → V} {ε : St → ℝ≥0∞} (h : PerOutcome out rel S ε) (pos : Ω → Q) (g : Q → V → V') :
    PerOutcome (fun ω => (out ω, pos ω)) (fun ω => g (pos ω) (rel ω)) (fun o σ => g o.2 (S o.1 σ))
      (fun o => ε o.1) := by
  rintro ⟨s, q⟩ E
  have hm : prCoin (fun ω => (out ω, pos ω) = (s, q) ∧ E (g (pos ω) (rel ω))) ≤
      prCoin (fun ω => out ω = s ∧ E (g q (rel ω))) := prCoin_mono fun ω hω => by
    obtain ⟨h1, h2⟩ := Prod.mk.inj hω.1
    refine ⟨h1, ?_⟩
    rw [← h2]
    exact hω.2
  exact hm.trans (h s fun v => E (g q v))

/-- **A bound in `Claims.Pr` is one in `prCoin`**, its error read in `ℝ≥0∞`. -/
theorem prCoin_le_of_Pr {α β : Type} [Fintype α] [Nonempty α] [Fintype β] [Nonempty β] {p : α → Prop}
    {q : β → Prop} {e : ℝ} (h : Claims.Pr p ≤ Claims.Pr q + e) : prCoin p ≤ prCoin q + ENNReal.ofReal e :=
  calc prCoin p = ENNReal.ofReal (Claims.Pr p) := by
        rw [Claims.Sampling.Pr_eq_toReal_prCoin, ENNReal.ofReal_toReal (prCoin_ne_top p)]
    _ ≤ ENNReal.ofReal (Claims.Pr q + e) := ENNReal.ofReal_le_ofReal h
    _ ≤ ENNReal.ofReal (Claims.Pr q) + ENNReal.ofReal e := ENNReal.ofReal_add_le
    _ = prCoin q + ENNReal.ofReal e := by
        rw [Claims.Sampling.Pr_eq_toReal_prCoin, ENNReal.ofReal_toReal (prCoin_ne_top q)]

/-- **`ZKOut` at one statement, witness and verifier is `PerOutcome`** over the protocol's randomness: the transcript
against `V*`, its outcome read off it, the simulator given the outcome, the error read in `ℝ≥0∞`. -/
theorem perOutcome_of_zkOut {W St : Type} {N Ω : Claims.Msg → Type} [∀ x, Fintype (N x)] [∀ x, Nonempty (N x)]
    [∀ x, Fintype (Ω x)] [∀ x, Nonempty (Ω x)] {H : Claims.Msg → Claims.Msg} {k : Claims.Msg → ℕ}
    {P : W → (x : Claims.Msg) → Claims.Prover (N x × Ω x)} {R : Claims.Rel W}
    {out : Claims.Msg → List (Claims.Msg × Claims.Out) → St} {S : Claims.SimOut St}
    {ε : (x : Claims.Msg) → W → (Θ : Type) → Claims.VStar Θ → Θ → St → ℝ} (h : Claims.ZKOut N Ω H k P R out S ε)
    {x : Claims.Msg} {w : W} (hR : R x w) {Θ : Type} (Vs : Claims.VStar Θ) (θ : Θ) (hk : Vs.k x = k x) :
    PerOutcome (fun ω : N x × Ω x => out x (Claims.runStar H (P w x) Vs x ω θ).msgs)
      (fun ω => (Claims.runStar H (P w x) Vs x ω θ).msgs) (fun o (c : S.C x) => (S.given o).out H Vs θ x c)
      (fun o => ENNReal.ofReal (ε x w Θ Vs θ o)) :=
  fun o Ev => prCoin_le_of_Pr (h x w hR Θ Vs θ hk o Ev)

end FlockSoundness.ZK

namespace Claims.ZKServer.Relay

open Claims Claims.ZKMal FlockSoundness.ZK
open Claims.ZKServer.LeanRelease (StopsAt CheckIs HelloEq LinkRoundEqStop LiveRoundEqStop FillRoundEqStop)
open _root_.Flock (V SessionSpec Stmt)
open scoped ENNReal

/-- **The deployed relay's release, per outcome.** At any randomness `ω` the relay reads the requests `ins ω`, the
worker's and the auditor's answers at the setup `s`. Where the model's frames it releases are the model's release
`mrel ω`, encoded by `enc`, and the model's release is simulated per outcome, the relay's release is simulated per
outcome at the model's outcome and where the release ended (`relayEnd`), by the model's simulator encoded and woven
with the setup's control frames over the schedule's cut, at the model's error. -/
theorem relay_perOutcome {Ω St Ωs M : Type} [Fintype Ω] [Fintype Ωs] (hold : Bool) (s : V (SessionSpec × Array Stmt))
    (ins : Ω → List (ByteArray × Env)) {mout : Ω → St} {mrel : Ω → M} {S : St → Ωs → M} {ε : St → ℝ≥0∞}
    (enc : M → List (List ByteArray)) (h : PerOutcome mout mrel S ε) (hs : ∀ ω, ∀ p ∈ ins ω, p.2.setup = s)
    (hb : ∀ ω, ∀ p ∈ ins ω, ∀ why, (p.2.barrier (some why)).isSome) (hc : ∀ ω, CoinReqsAnswered (ins ω))
    (hfr : ∀ ω, relayFrames hold (ins ω) = enc (mrel ω)) :
    PerOutcome (fun ω => (mout ω, relayEnd hold s (ins ω))) (fun ω => out hold {} {} (ins ω))
      (fun o σ => weave (ctlOf s) ((tmplOf s).take o.2) (enc (S o.1 σ))) (fun o => ε o.1) := by
  have key : (fun ω => out hold {} {} (ins ω)) =
      fun ω => weave (ctlOf s) ((tmplOf s).take (relayEnd hold s (ins ω))) (enc (mrel ω)) := by
    funext ω
    obtain ⟨hle, hout⟩ := relay_release_cut hold s (ins ω) (hs ω) (hb ω) (hc ω)
    rw [hout, relayEnd_val hle, hfr ω]
  rw [key]
  exact h.refine (fun ω => relayEnd hold s (ins ω)) fun c m => weave (ctlOf s) ((tmplOf s).take c) (enc m)

/-- **The deployed relay's release, per outcome, at any auditor**: as `relay_perOutcome`, at the slots the release
reached up to where it ended, a coin slot the auditor answered without coins coming again as often as the worker sends
its request. -/
theorem relay_perOutcome_slots {Ω St Ωs M : Type} [Fintype Ω] [Fintype Ωs] (hold : Bool)
    (s : V (SessionSpec × Array Stmt)) (ins : Ω → List (ByteArray × Env)) {mout : Ω → St} {mrel : Ω → M}
    {S : St → Ωs → M} {ε : St → ℝ≥0∞} (enc : M → List (List ByteArray)) (h : PerOutcome mout mrel S ε)
    (hs : ∀ ω, ∀ p ∈ ins ω, p.2.setup = s) (hb : ∀ ω, ∀ p ∈ ins ω, ∀ why, (p.2.barrier (some why)).isSome)
    (hfr : ∀ ω, relayFrames hold (ins ω) = enc (mrel ω)) :
    PerOutcome (fun ω => (mout ω, (relaySlots hold (ins ω)).take (relayCut hold (ins ω))))
      (fun ω => out hold {} {} (ins ω)) (fun o σ => weave (ctlOf s) o.2 (enc (S o.1 σ))) (fun o => ε o.1) := by
  have key : (fun ω => out hold {} {} (ins ω)) =
      fun ω => weave (ctlOf s) ((relaySlots hold (ins ω)).take (relayCut hold (ins ω))) (enc (mrel ω)) := by
    funext ω
    rw [(relay_release hold s (ins ω) (hs ω) (hb ω)).2.2.2, hfr ω]
  rw [key]
  exact h.refine (fun ω => (relaySlots hold (ins ω)).take (relayCut hold (ins ω)))
    fun ks m => weave (ctlOf s) ks (enc m)

/-- **The deployed relay's release, read through `ZKOut`**: where the model is zero knowledge per outcome (`ZKOut`) and
the relay's model frames at each of the model's runs against `V*` are its transcript encoded by `enc`, the relay's
release is simulated per outcome at the model's outcome and where the release ended, by the model's simulator given its
outcome, encoded and woven with the setup's control frames, at the model's error. -/
theorem relay_zkOut {W St : Type} {N Ω : Msg → Type} [∀ x, Fintype (N x)] [∀ x, Nonempty (N x)]
    [∀ x, Fintype (Ω x)] [∀ x, Nonempty (Ω x)] {H : Msg → Msg} {k : Msg → ℕ} {P : W → (x : Msg) → Prover (N x × Ω x)}
    {R : Rel W} {mout : Msg → List (Msg × Out) → St} {S : SimOut St}
    {ε : (x : Msg) → W → (Θ : Type) → VStar Θ → Θ → St → ℝ} (h : ZKOut N Ω H k P R mout S ε) {x : Msg} {w : W}
    (hR : R x w) {Θ : Type} (Vs : VStar Θ) (θ : Θ) (hk : Vs.k x = k x) (hold : Bool)
    (s : V (SessionSpec × Array Stmt)) (ins : N x × Ω x → List (ByteArray × Env))
    (enc : List (Msg × Out) → List (List ByteArray)) (hs : ∀ ω, ∀ p ∈ ins ω, p.2.setup = s)
    (hb : ∀ ω, ∀ p ∈ ins ω, ∀ why, (p.2.barrier (some why)).isSome) (hc : ∀ ω, CoinReqsAnswered (ins ω))
    (hfr : ∀ ω, relayFrames hold (ins ω) = enc (runStar H (P w x) Vs x ω θ).msgs) :
    PerOutcome (fun ω => (mout x (runStar H (P w x) Vs x ω θ).msgs, relayEnd hold s (ins ω)))
      (fun ω => out hold {} {} (ins ω))
      (fun o (c : S.C x) => weave (ctlOf s) ((tmplOf s).take o.2) (enc ((S.given o.1).out H Vs θ x c)))
      (fun o => ENNReal.ofReal (ε x w Θ Vs θ o.1)) :=
  relay_perOutcome hold s ins enc (perOutcome_of_zkOut h hR Vs θ hk) hs hb hc hfr

/-- **Before the `Finish` the deployed relay's release is the unswitched prover's**: at the deferred release against
`V*` (`LeanRelease.release_preFinish_runStar`'s premises), where the relay's model frames are the deferred release's
messages each encoded by a public `e`, the relay's release is the unswitched prover's first `n x - 1` messages, then the
deferred release's last, encoded and woven with the setup's control frames over the schedule's cut where the release
ended, on either branch. -/
theorem relay_preFinish {Θ : Type} (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool)
    (vs : S.SwVals) (L : LeanRelease Claims.Flock.Wit (S.refP.Nonce × S.refP.Rand)) (H : Msg → Msg) (n : Msg → ℕ)
    (wk : L.toSrv.Worker) (x : Msg) (w : Claims.Flock.Wit) (hn : n x = refK x + 2) (hlast : L.last x = refK x + 1)
    (hR : StopsAt S Rf L H x) (hC : CheckIs S L H x) (hH : HelloEq S L x) (hLk : LinkRoundEqStop S vs L x w)
    (hE : LiveRoundEqStop S vs L H wk x w) (hEf : FillRoundEqStop S vs L H wk x w) (Vs : VStar Θ) (θ : Θ)
    (hk : Vs.k x = n x) (ω : (S.refP.Nonce × S.refP.Rand) × L.toSrv.Chk) (hold : Bool)
    (s : V (SessionSpec × Array Stmt)) (ins : List (ByteArray × Env)) (e : Msg × Out → List ByteArray)
    (hs : ∀ p ∈ ins, p.2.setup = s) (hb : ∀ p ∈ ins, ∀ why, (p.2.barrier (some why)).isSome)
    (hc : CoinReqsAnswered ins)
    (hfr : relayFrames hold ins = (runStar H (L.toSrv.proverD L.fillOf n wk w) Vs x ω θ).msgs.map e) :
    out hold {} {} ins = weave (ctlOf s) ((tmplOf s).take (relayCut hold ins))
      (((runStar H (((S.withStops Rf L.stop).unP vs).prover w) Vs x ω.1 θ).msgs.take (n x - 1)).map e ++
        ((runStar H (L.toSrv.proverD L.fillOf n wk w) Vs x ω θ).msgs.drop (n x - 1)).map e) := by
  rw [(relay_release_cut hold s ins hs hb hc).2, hfr,
    ← L.release_preFinish_runStar S Rf vs H n wk x w hn hlast hR hC hH hLk hE hEf Vs θ hk ω, ← List.map_append,
    List.take_append_drop]

/-- **On the stop branch the deployed relay's release is the unswitched cut, then the stop**: at the deferred release
against `V*`, where the session stops at its `Finish` (`LeanRelease.release_stop_runStar`'s premises) and the relay's
model frames are the deferred release's transcript encoded by `enc`, the relay's release is the unswitched prover's run
with its last message the stop, encoded and woven with the setup's control frames over the schedule's cut where the
release ended. -/
theorem relay_stop {Θ : Type} (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool)
    (vs : S.SwVals) (L : LeanRelease Claims.Flock.Wit (S.refP.Nonce × S.refP.Rand)) (H : Msg → Msg) (n : Msg → ℕ)
    (wk : L.toSrv.Worker) (x : Msg) (w : Claims.Flock.Wit) (hn : n x = refK x + 2) (hlast : L.last x = refK x + 1)
    (hR : StopsAt S Rf L H x) (hC : CheckIs S L H x) (hH : HelloEq S L x) (hLk : LinkRoundEqStop S vs L x w)
    (hE : LiveRoundEqStop S vs L H wk x w) (hEf : FillRoundEqStop S vs L H wk x w) (Vs : VStar Θ) (θ : Θ)
    (hk : Vs.k x = n x) (ω : (S.refP.Nonce × S.refP.Rand) × L.toSrv.Chk)
    (hd : L.toSrv.live wk x w ω.1 ω.2
      (histOf ((runStar H (((S.withStops Rf L.stop).unP vs).prover w) Vs x ω.1 θ).msgs.take (n x - 1))) = false)
    (hold : Bool) (s : V (SessionSpec × Array Stmt)) (ins : List (ByteArray × Env))
    (enc : List (Msg × Out) → List (List ByteArray)) (hs : ∀ p ∈ ins, p.2.setup = s)
    (hb : ∀ p ∈ ins, ∀ why, (p.2.barrier (some why)).isSome) (hc : CoinReqsAnswered ins)
    (hfr : relayFrames hold ins = enc (runStar H (L.toSrv.proverD L.fillOf n wk w) Vs x ω θ).msgs) :
    out hold {} {} ins = weave (ctlOf s) ((tmplOf s).take (relayCut hold ins))
      (enc (cutT (fun as => ((Vs.next x as θ).eval H).1) L.stop (n x - 1) (n x)
        (runStar H (((S.withStops Rf L.stop).unP vs).prover w) Vs x ω.1 θ).msgs)) := by
  rw [(relay_release_cut hold s ins hs hb hc).2, hfr,
    L.release_stop_runStar S Rf vs H n wk x w hn hlast hR hC hH hLk hE hEf Vs θ hk ω hd]

end Claims.ZKServer.Relay
