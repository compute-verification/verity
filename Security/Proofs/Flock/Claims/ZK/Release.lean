import Proofs.Flock.Claims.ZK.Blind
import Flock.Firewall.Frame
import Flock.Firewall.Assumptions

/-!
# The ZK server's release is the Lean frame

The outer firewall releases only what `Flock.Firewall.Frame.round` returns (`flock-firewall frame`, PROTOCOL.md §10.1 and
§10.4): the device's unmasked round, once the firewall passed it, masked under the rep's pads, with the pads' label and
root inserted after the zerocheck's label. `LeanRelease` is the ZK server whose frame is that function, and
`LeanRelease.toSrv` is it as a `ZkServer` with one masked slot per round: the worker's value is its device round, the mask
is the Lean frame's round, and the check is the firewall's pass decision and the frame's own refusals. Its frame at honest
workers is, by definition, the Lean frame of the honest device round (`LeanRelease.refMsg_toSrv`).

The firewall stops a session only at its `Finish` (§10.1): once a round fails, it masks its own scheduled round in the
device's place (`LeanRelease.scheduled`: the skeleton's round with zero values in the algebraic phase, the shadow's in
the opening), and the last round is the stop. That is the deferred server `ZkServer.nextD` at `L.toSrv` with the stand-in
`LeanRelease.fillOf`. The device round and the scheduled round read, by name, the values the firewall fixes by its own
computation: its level-0 recheck, its opening, `final_c'` and the inner proof, as its transcript reports them
(`computedOf`). Two Props give `ZkServer.RefD` at the deferred release (`LeanRelease.refD`):
* `FirewallComputes` (D8 (a), still a named assumption): each reported value is the specified one;
* `LeanRelease.ItemsRefine`, the item-stream refinement: the transcript reports every specified value (`Reports`; with
  D8 (a) it then reports exactly those, `computedOf_eq`), and wherever the deferred release releases a frame, with the
  specified values in its scheduled rounds, that frame is the prover's message.

So `Flock.zk_server_blind`'s bound holds at the deployed release, at any worker: one bit a session
(`Flock.zk_server_lean`).
-/

namespace Claims.ZKServer

open Claims Claims.ZKMal

/-- The value the firewall's transcript `t` reports fixing for `what` by its own computation (`By.recheck`, `By.shadow`,
`By.own`): the first such item's. -/
def computedOf (t : _root_.Flock.Firewall.Contract.Transcript) (what : String) : Option ByteArray :=
  t.items.toList.findSome? fun
    | .fixed w .recheck want _ _ | .fixed w .shadow want _ _ | .fixed w .own want _ _ =>
      if w = what then some want else none
    | _ => none

/-- Under D8 (a), every value the transcript reports fixing by the firewall's own computation is `spec`'s. -/
theorem computedOf_spec {spec : String → Option ByteArray} {t : _root_.Flock.Firewall.Contract.Transcript}
    (hc : _root_.Flock.Firewall.Contract.FirewallComputes spec t) {what : String} {v : ByteArray}
    (hv : computedOf t what = some v) : spec what = some v := by
  obtain ⟨a, ha, hf⟩ := List.exists_of_findSome?_eq_some hv
  cases a with
  | fixed w src want sent ps =>
    cases src
    case recheck | shadow | own =>
      dsimp only at hf
      split at hf
      · cases hf
        subst_vars
        exact hc _ _ _ sent ps ha (by decide)
      · cases hf
    all_goals cases hf
  | _ => cases hf

/-- The transcript `t` reports a value for every name `spec` fixes. -/
def Reports (spec : String → Option ByteArray) (t : _root_.Flock.Firewall.Contract.Transcript) : Prop :=
  ∀ what v, spec what = some v → (computedOf t what).isSome

/-- Under D8 (a), a transcript that reports every name `spec` fixes reports exactly `spec`. -/
theorem computedOf_eq {spec : String → Option ByteArray} {t : _root_.Flock.Firewall.Contract.Transcript}
    (hc : _root_.Flock.Firewall.Contract.FirewallComputes spec t) (hr : Reports spec t) : computedOf t = spec := by
  funext what
  cases hs : spec what with
  | none =>
    cases hv : computedOf t what with
    | none => rfl
    | some v =>
      have := computedOf_spec hc hv
      rw [hs] at this
      cases this
  | some v =>
    obtain ⟨v', hv'⟩ := Option.isSome_iff_exists.mp (hr what v hs)
    have := computedOf_spec hc hv'
    rw [hs] at this
    rw [hv', this]

/-- **The Lean frame's release** of the device round `v` (transcript-v2 frames, as the device frames them) with `n` coins
from the masker `m`: the bytes `Flock.Firewall.Frame.round` returns, or `stop` where it refuses. -/
def frameOut (stop : Msg) (m : _root_.Flock.Firewall.Frame.Masker) (v : Msg) (n : ℕ) : Msg :=
  match _root_.Flock.Firewall.Frame.round m ⟨v.toArray⟩ n with
  | .ok (out, _) => out.data.toList
  | .error _ => stop

/-- Whether the Lean frame takes the device round `v`. -/
def frameTakes (m : _root_.Flock.Firewall.Frame.Masker) (v : Msg) (n : ℕ) : Bool :=
  match _root_.Flock.Firewall.Frame.round m ⟨v.toArray⟩ n with
  | .ok _ => true
  | .error _ => false

/-- **The deployed release**, for witnesses `W` on the server's tape `R`: at statement `x`, after the history `h`, round
`h.length` leaves as the Lean frame's round of the device round, from the rep's masker. -/
structure LeanRelease (W R : Type) where
  /-- the rep's masker after the rounds of `h`: `Masker.new` of the layout, the pads and the pads' root on the server's
  tape, advanced by `Frame.round` over the rounds so far -/
  masker : Msg → R → List (Msg × Msg) → _root_.Flock.Firewall.Frame.Masker
  /-- the device round at honest worker values, given the values the firewall fixes by its own computation, by name -/
  device : Msg → W → R → List (Msg × Msg) → (String → Option ByteArray) → Msg
  /-- the firewall's transcript after the rounds of `h`, whose own values the device round reads -/
  computed : Msg → W → R → List (Msg × Msg) → _root_.Flock.Firewall.Contract.Transcript
  /-- the firewall's pass decision on a worker's device round (the algebraic phase's skeleton, the opening's shadow) -/
  pass : Msg → ℕ → W → R → List (Msg × Msg) → Msg → Bool
  /-- the device round the firewall masks in the device's place once a round failed: the skeleton's round with zero
  values in the algebraic phase, the shadow's round in the opening, given the values the firewall fixes by its own
  computation, by name -/
  scheduled : Msg → W → R → List (Msg × Msg) → (String → Option ByteArray) → Msg
  /-- round `j`'s coin count -/
  coins : Msg → ℕ → ℕ
  /-- round `j`'s release time -/
  sched : Msg → ℕ → ℕ
  stop : Msg

namespace LeanRelease

/-- **The deployed release as a ZK server**: one masked slot per round, whose worker value is the device round, whose
mask is the Lean frame's round and whose check is the firewall's pass decision and the frame's own refusals. -/
def toSrv {W R : Type} (L : LeanRelease W R) : ZkServer W R where
  Chk := Unit
  Val := Msg
  nS := fun _ _ => 1
  kind := fun _ _ _ => .masked
  cst := fun _ _ _ => []
  own := fun _ _ _ _ _ => []
  recomp := fun _ _ _ _ _ _ _ => []
  mask := fun x j _ ρ h v => frameOut L.stop (L.masker x ρ h) v (L.coins x j)
  rel := fun x j _ w ρ h _ _ v => L.pass x j w ρ h v && frameTakes (L.masker x ρ h) v (L.coins x j)
  honest := fun x _ _ w ρ h => L.device x w ρ h (computedOf (L.computed x w ρ h))
  frame := fun _ _ vs => vs ⟨0, Nat.zero_lt_one⟩
  stop := L.stop
  sched := L.sched

/-- **The server's frame at honest workers is the Lean frame** of the honest device round, by definition. -/
theorem refMsg_toSrv {W R : Type} (L : LeanRelease W R) (x : Msg) (w : W) (ρ : R) (h : List (Msg × Msg)) :
    L.toSrv.refMsg x w ρ h =
      frameOut L.stop (L.masker x ρ h) (L.device x w ρ h (computedOf (L.computed x w ρ h))) (L.coins x h.length) :=
  rfl

/-- **The deployed release's stand-in**: once a round failed, the scheduled round, with the values the firewall's
transcript reports. -/
def fillOf {W R : Type} (L : LeanRelease W R) : Msg → ℕ → ℕ → W → R → List (Msg × Msg) → Msg :=
  fun x _ _ w ρ h => L.scheduled x w ρ h (computedOf (L.computed x w ρ h))

/-- **The item-stream refinement** at `(x, w)`, the worker `wk`, the last round `n` and the specified values `spec`:
after every history, the firewall's transcript reports every value `spec` fixes, and wherever the deferred release
releases a frame, with `spec`'s values in its scheduled rounds, it is `P`'s message. -/
def ItemsRefine {W R : Type} (L : LeanRelease W R) (H : Msg → Msg) (P : W → Prover R) (n : Msg → ℕ)
    (wk : L.toSrv.Worker) (x : Msg) (w : W) (spec : String → Option ByteArray) : Prop :=
  (∀ ρ h, Reports spec (L.computed x w ρ h)) ∧
    L.toSrv.RefD H P (fun x _ _ w ρ h => L.scheduled x w ρ h spec) n wk x w

/-- **`hRef` at the deferred Lean release**: D8 (a) on every transcript the firewall keeps, and the item-stream
refinement, give `ZkServer.RefD` at the stand-in `fillOf`. -/
theorem refD {W R : Type} (L : LeanRelease W R) (H : Msg → Msg) (P : W → Prover R) (n : Msg → ℕ)
    (wk : L.toSrv.Worker) (x : Msg) (w : W) (spec : String → Option ByteArray)
    (hc : ∀ ρ h, _root_.Flock.Firewall.Contract.FirewallComputes spec (L.computed x w ρ h))
    (hI : L.ItemsRefine H P n wk x w spec) : L.toSrv.RefD H P L.fillOf n wk x w := fun ρ κ h hl => by
  rw [hI.2 ρ κ h hl]
  refine L.toSrv.frameD_congr wk x w (fun j s ρ' h' => ?_) ρ κ h
  show L.scheduled x w ρ' h' spec = L.scheduled x w ρ' h' (computedOf (L.computed x w ρ' h'))
  rw [computedOf_eq (hc ρ' h') (hI.1 ρ' h')]

end LeanRelease

open FlockSoundness.Discharge

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- **The ZK server at `flock-circuit --zk`'s reference session, released by the Lean frame**: the deployed release
`L.toSrv`, which stops a session only at its `Finish` (`ZkServer.nextD` with the stand-in `LeanRelease.fillOf`), at any
worker `wk` and any values `vs` whose round encoding is causal, releases, on every event, at most twice the probability
at the simulator that guesses whether the session stops (`leaky1`, at `Flock.zk_server_ref`'s simulator, `refP`'s), plus
`ε_zk + γ·Fork`, with no term for the firewall's checks, under `Claims.Hm96Hiding`, the session's five Props, D8 (a) on
the firewall's transcripts (`FirewallComputes`) and the item-stream refinement at `swP vs` (`LeanRelease.ItemsRefine`).
The stop leaks one bit a session; over a stream of `m` sessions with at most `b` stops, `log₂ ∑_{i ≤ b} C(m, i)` bits
(`stream_leak_deferred`). -/
theorem Flock.zk_server_lean (H : Msg → Msg) (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (S : RefSession) (vs : S.SwVals) (t : ℕ) (ht : 0 < t) {δ₁ : ENNReal}
    (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤) (hS : S.Tables)
    (hV : S.Complete (Flock.R (FlockDeploy.deployZk rel).parse)) (hD : S.DummyReg) (hc : S.Causal)
    (hsc : S.SwCausal vs) (hT : S.TapeOK) (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand))
    (wk : L.toSrv.Worker) (x : Msg) (w : Flock.Wit) (hR : Flock.R (FlockDeploy.deployZk rel).parse x w)
    (spec : String → Option ByteArray)
    (hFC : ∀ ρ h, _root_.Flock.Firewall.Contract.FirewallComputes spec (L.computed x w ρ h))
    (hI : L.ItemsRefine H (S.swP vs).prover (fun x => (FlockDeploy.deployZk rel).V.k x + 1) wk x w spec) {Θ : Type}
    (Vs : VStar Θ) (θ : Θ) (hk : Vs.k x = (FlockDeploy.deployZk rel).V.k x + 1)
    (Ev : List ((Msg × Out) × ℕ) → Prop) :
    Pr (fun ω : (S.refP.Nonce × S.refP.Rand) × L.toSrv.Chk =>
        Ev (L.toSrv.timed x (runStar H (L.toSrv.proverD L.fillOf (fun x => (FlockDeploy.deployZk rel).V.k x + 1) wk w) Vs x ω θ).msgs)) ≤
      2 * (Pr (fun c : (leaky1 ((S.refP.withSim fun x _ c => S.coinModel x c).simulatorE t (t * 3 ^ t))
            (fun x => (FlockDeploy.deployZk rel).V.k x + 1) L.toSrv.stop).C x =>
          Ev (L.toSrv.timed x ((leaky1 ((S.refP.withSim fun x _ c => S.coinModel x c).simulatorE t (t * 3 ^ t))
            (fun x => (FlockDeploy.deployZk rel).V.k x + 1) L.toSrv.stop).out H Vs θ x c))) +
          (8 * S.nLeaves x * δ₁.toReal + 8 * t * S.nLeaves x * δ₁.toReal + 2 * (1 / 3) ^ t) +
          4 * t * 3 ^ t * (Fork H ((S.swP vs).prover w) Vs θ x).Pr H) := by
  have h := L.toSrv.zk_deferred L.fillOf H (fun x => (FlockDeploy.deployZk rel).V.k x + 1) (S.swP vs).prover _ _ _ _
    (Flock.zk_deployZk_sw H rel S vs t ht hT1 hδ₁ hS hV hD hc hsc hT).2 x w hR wk
    (L.refD H _ _ wk x w spec hFC hI) Vs θ hk Ev
  exact h

end Claims.ZKServer
