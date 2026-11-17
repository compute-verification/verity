module

public import Definitions.Claims.ZKOut
public import Proofs.Flock.Claims.Sampling.Draws
public import Proofs.Flock.ZkAny.Generic
public import Proofs.Flock.ZkAny.Relay
public import Proofs.Flock.ZkAny.RelayFrames
public import Proofs.Flock.ZkAny.Unswitched

/-! The deployed relay's release is zero knowledge per outcome where the model's frames are (`relay_perOutcome`, `relay_zkOut_env`); this covers the released bytes and their order, not their times. -/

@[expose] public section

namespace FlockSoundness.ZK

open Verity.Game
open scoped ENNReal

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

theorem prCoin_le_of_Pr {α β : Type} [Fintype α] [Nonempty α] [Fintype β] [Nonempty β] {p : α → Prop}
    {q : β → Prop} {e : ℝ} (h : Claims.Pr p ≤ Claims.Pr q + e) : prCoin p ≤ prCoin q + ENNReal.ofReal e :=
  calc prCoin p = ENNReal.ofReal (Claims.Pr p) := by
        rw [Claims.Sampling.Pr_eq_toReal_prCoin, ENNReal.ofReal_toReal (prCoin_ne_top p)]
    _ ≤ ENNReal.ofReal (Claims.Pr q + e) := ENNReal.ofReal_le_ofReal h
    _ ≤ ENNReal.ofReal (Claims.Pr q) + ENNReal.ofReal e := ENNReal.ofReal_add_le
    _ = prCoin q + ENNReal.ofReal e := by
        rw [Claims.Sampling.Pr_eq_toReal_prCoin, ENNReal.ofReal_toReal (prCoin_ne_top q)]

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

theorem relay_perOutcome {Ω St Ωs M : Type} [Fintype Ω] [Fintype Ωs] (hold : Bool) (s : V (SessionSpec × Array Stmt))
    (ins : Ω → List (ByteArray × Env)) {mout : Ω → St} {mrel : Ω → M} {S : St → Ωs → M} {ε : St → ℝ≥0∞}
    (enc : M → List (List ByteArray)) (B : ℕ) (hB : ∀ m, (weave (ctlOf s) (tmplOf s) (enc m)).length ≤ B)
    (h : PerOutcome mout mrel S ε) (hs : ∀ ω, ∀ p ∈ ins ω, p.2.setup = s)
    (hb : ∀ ω, ∀ p ∈ ins ω, ∀ why, (p.2.barrier (some why)).isSome)
    (hfr : ∀ ω, RelayPre (relayTrace hold (ins ω)) (enc (mrel ω))) :
    PerOutcome (fun ω => (mout ω, relayLen hold B (ins ω))) (fun ω => out hold {} {} (ins ω))
      (fun o σ => (weave (ctlOf s) (tmplOf s) (enc (S o.1 σ))).take o.2) (fun o => ε o.1) := by
  have key : (fun ω => out hold {} {} (ins ω)) =
      fun ω => (weave (ctlOf s) (tmplOf s) (enc (mrel ω))).take (relayLen hold B (ins ω)) :=
    funext fun ω => relay_release_take (hB _) (hs ω) (hb ω) (hfr ω)
  rw [key]
  exact h.refine (fun ω => relayLen hold B (ins ω)) fun c m => (weave (ctlOf s) (tmplOf s) (enc m)).take c

theorem relay_zkOut {W St : Type} {N Ω : Msg → Type} [∀ x, Fintype (N x)] [∀ x, Nonempty (N x)]
    [∀ x, Fintype (Ω x)] [∀ x, Nonempty (Ω x)] {H : Msg → Msg} {k : Msg → ℕ} {P : W → (x : Msg) → Prover (N x × Ω x)}
    {R : Rel W} {mout : Msg → List (Msg × Out) → St} {S : SimOut St}
    {ε : (x : Msg) → W → (Θ : Type) → VStar Θ → Θ → St → ℝ} (h : ZKOut N Ω H k P R mout S ε) {x : Msg} {w : W}
    (hR : R x w) {Θ : Type} (Vs : VStar Θ) (θ : Θ) (hk : Vs.k x = k x) (hold : Bool)
    (s : V (SessionSpec × Array Stmt)) (ins : N x × Ω x → List (ByteArray × Env))
    (enc : List (Msg × Out) → List (List ByteArray)) (B : ℕ)
    (hB : ∀ m, (weave (ctlOf s) (tmplOf s) (enc m)).length ≤ B) (hs : ∀ ω, ∀ p ∈ ins ω, p.2.setup = s)
    (hb : ∀ ω, ∀ p ∈ ins ω, ∀ why, (p.2.barrier (some why)).isSome)
    (hfr : ∀ ω, RelayPre (relayTrace hold (ins ω)) (enc (runStar H (P w x) Vs x ω θ).msgs)) :
    PerOutcome (fun ω => (mout x (runStar H (P w x) Vs x ω θ).msgs, relayLen hold B (ins ω)))
      (fun ω => out hold {} {} (ins ω))
      (fun o (c : S.C x) => (weave (ctlOf s) (tmplOf s) (enc ((S.given o.1).out H Vs θ x c))).take o.2)
      (fun o => ENNReal.ofReal (ε x w Θ Vs θ o.1)) :=
  relay_perOutcome hold s ins enc B hB (perOutcome_of_zkOut h hR Vs θ hk) hs hb hfr

theorem relay_zkOut_env {W St : Type} {N Ω : Msg → Type} [∀ x, Fintype (N x)] [∀ x, Nonempty (N x)]
    [∀ x, Fintype (Ω x)] [∀ x, Nonempty (Ω x)] {H : Msg → Msg} {k : Msg → ℕ} {P : W → (x : Msg) → Prover (N x × Ω x)}
    {R : Rel W} {mout : Msg → List (Msg × Out) → St} {S : SimOut St}
    {ε : (x : Msg) → W → (Θ : Type) → VStar Θ → Θ → St → ℝ} (h : ZKOut N Ω H k P R mout S ε) {x : Msg} {w : W}
    (hR : R x w) {Θ : Type} (Vs : VStar Θ) (θ : Θ) (hk : Vs.k x = k x) (spec : SessionSpec) (sts : Array Stmt)
    (ins : N x × Ω x → List (ByteArray × Env))
    (hs : ∀ ω, ∀ p ∈ ins ω, p.2.setup = .ok (spec, sts))
    (hb : ∀ ω, ∀ p ∈ ins ω, ∀ why, (p.2.barrier (some why)).isSome)
    (he : ∀ ω, EnvPre spec [] (relaySteps true (ins ω)) ((runStar H (P w x) Vs x ω θ).msgs.map (·.1))) :
    PerOutcome (fun ω => (mout x (runStar H (P w x) Vs x ω θ).msgs, relayLen true (relayB spec sts) (ins ω)))
      (fun ω => out true {} {} (ins ω))
      (fun o (c : S.C x) => (weave (ctlOf (.ok (spec, sts))) (tmplOf (.ok (spec, sts)))
        (encOf spec sts (((S.given o.1).out H Vs θ x c).map (·.1)))).take o.2)
      (fun o => ENNReal.ofReal (ε x w Θ Vs θ o.1)) := by
  have hB : ∀ m : List (Msg × Out), (weave (ctlOf (.ok (spec, sts))) (tmplOf (.ok (spec, sts)))
      (encOf spec sts (m.map (·.1)))).length ≤ relayB spec sts := fun m => weave_encOf_length spec sts _
  have hfr : ∀ ω, RelayPre (relayTrace true (ins ω))
      (encOf spec sts ((runStar H (P w x) Vs x ω θ).msgs.map (·.1))) := fun ω =>
    relayPre_of_env (ins ω) _ (hs ω) (hb ω) (he ω)
  have key := relay_zkOut (enc := fun m => encOf spec sts (m.map (·.1))) (B := relayB spec sts) h hR Vs θ hk true
    (.ok (spec, sts)) ins hB hs hb hfr
  exact key

theorem relay_preFinish {Θ : Type} (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool)
    (vs : S.SwVals) (L : LeanRelease Claims.Flock.Wit (S.refP.Nonce × S.refP.Rand)) (H : Msg → Msg) (n : Msg → ℕ)
    (wk : L.toSrv.Worker) (x : Msg) (w : Claims.Flock.Wit) (hn : n x = refK x + 2) (hlast : L.last x = refK x + 1)
    (hR : StopsAt S Rf L H x) (hC : CheckIs S L H x) (hH : HelloEq S L x) (hLk : LinkRoundEqStop S vs L x w)
    (hE : LiveRoundEqStop S vs L H wk x w) (hEf : FillRoundEqStop S vs L H wk x w) (Vs : VStar Θ) (θ : Θ)
    (hk : Vs.k x = n x) (ω : (S.refP.Nonce × S.refP.Rand) × L.toSrv.Chk) (hold : Bool)
    (s : V (SessionSpec × Array Stmt)) (ins : List (ByteArray × Env))
    (enc : List (Msg × Out) → List (List ByteArray)) (hs : ∀ p ∈ ins, p.2.setup = s)
    (hb : ∀ p ∈ ins, ∀ why, (p.2.barrier (some why)).isSome)
    (hfr : RelayPre (relayTrace hold ins) (enc (runStar H (L.toSrv.proverD L.fillOf n wk w) Vs x ω θ).msgs)) :
    out hold {} {} ins <+: weave (ctlOf s) (tmplOf s)
      (enc ((runStar H (((S.withStops Rf L.stop).unP vs).prover w) Vs x ω.1 θ).msgs.take (n x - 1) ++
        (runStar H (L.toSrv.proverD L.fillOf n wk w) Vs x ω θ).msgs.drop (n x - 1))) := by
  rw [← L.release_preFinish_runStar S Rf vs H n wk x w hn hlast hR hC hH hLk hE hEf Vs θ hk ω, List.take_append_drop]
  exact relay_release_pre hold s ins _ hs hb hfr

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
    (hb : ∀ p ∈ ins, ∀ why, (p.2.barrier (some why)).isSome)
    (hfr : RelayPre (relayTrace hold ins) (enc (runStar H (L.toSrv.proverD L.fillOf n wk w) Vs x ω θ).msgs)) :
    out hold {} {} ins <+: weave (ctlOf s) (tmplOf s)
      (enc (cutT (fun as => ((Vs.next x as θ).eval H).1) L.stop (n x - 1) (n x)
        (runStar H (((S.withStops Rf L.stop).unP vs).prover w) Vs x ω.1 θ).msgs)) := by
  rw [← L.release_stop_runStar S Rf vs H n wk x w hn hlast hR hC hH hLk hE hEf Vs θ hk ω hd]
  exact relay_release_pre hold s ins _ hs hb hfr

end Claims.ZKServer.Relay
