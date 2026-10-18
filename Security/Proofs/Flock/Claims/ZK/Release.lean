import Proofs.Flock.Claims.ZK.Server
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

The honest device round reads, by name, the values the firewall fixes by its own computation: its level-0 recheck, its
opening, `final_c'` and the inner proof, as its transcript reports them (`computedOf`). Two Props give `ZkServer.Ref`
(`LeanRelease.ref`):
* `FirewallComputes` (D8 (a), still a named assumption): each reported value is the specified one;
* `LeanRelease.ItemsRefine`, the item-stream refinement: at honest worker values, with any reported values that agree
  with the specified ones, the Lean frame of the device round is the reference prover's message.

So `Flock.zk_server_ref`'s `hRef` at the deployed release is those two (`Flock.zk_server_lean`).
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

/-- **The Lean frame's release** of the device round `v` (transcript-v2 frames, as the device frames them) with `n` coins
at the rep's configuration `c` and masking state `m`: the bytes `Flock.Firewall.Frame.round` returns, or `stop` where it
refuses. -/
def frameOut (stop : Msg) (c : _root_.Flock.Firewall.Frame.Cfg) (m : _root_.Flock.Firewall.Frame.Masker) (v : Msg)
    (n : ℕ) : Msg :=
  match _root_.Flock.Firewall.Frame.round c m ⟨v.toArray⟩ n with
  | .ok (out, _) => out.data.toList
  | .error _ => stop

/-- Whether the Lean frame takes the device round `v`. -/
def frameTakes (c : _root_.Flock.Firewall.Frame.Cfg) (m : _root_.Flock.Firewall.Frame.Masker) (v : Msg) (n : ℕ) :
    Bool :=
  match _root_.Flock.Firewall.Frame.round c m ⟨v.toArray⟩ n with
  | .ok _ => true
  | .error _ => false

/-- **The deployed release**, for witnesses `W` on the server's tape `R`: at statement `x`, after the history `h`, round
`h.length` leaves as the Lean frame's round of the device round, from the rep's masker. -/
structure LeanRelease (W R : Type) where
  /-- the rep's masking configuration, fixed at its start: `Cfg.ofDraws` of the layout, the draws on the server's tape and
  the statement's mask slot -/
  cfg : Msg → R → _root_.Flock.Firewall.Frame.Cfg
  /-- the rep's masking state after the rounds of `h`: `{}`, advanced by `Frame.round` over the rounds so far -/
  masker : Msg → R → List (Msg × Msg) → _root_.Flock.Firewall.Frame.Masker
  /-- the device round at honest worker values, given the values the firewall fixes by its own computation, by name -/
  device : Msg → W → R → List (Msg × Msg) → (String → Option ByteArray) → Msg
  /-- the firewall's transcript after the rounds of `h`, whose own values the device round reads -/
  computed : Msg → W → R → List (Msg × Msg) → _root_.Flock.Firewall.Contract.Transcript
  /-- the firewall's pass decision on a worker's device round (the algebraic phase's skeleton, the opening's shadow) -/
  pass : Msg → ℕ → W → R → List (Msg × Msg) → Msg → Bool
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
  mask := fun x j _ ρ h v => frameOut L.stop (L.cfg x ρ) (L.masker x ρ h) v (L.coins x j)
  rel := fun x j _ w ρ h _ _ v => L.pass x j w ρ h v && frameTakes (L.cfg x ρ) (L.masker x ρ h) v (L.coins x j)
  honest := fun x _ _ w ρ h => L.device x w ρ h (computedOf (L.computed x w ρ h))
  frame := fun _ _ vs => vs ⟨0, Nat.zero_lt_one⟩
  stop := L.stop
  sched := L.sched

/-- **The server's frame at honest workers is the Lean frame** of the honest device round, by definition. -/
theorem refMsg_toSrv {W R : Type} (L : LeanRelease W R) (x : Msg) (w : W) (ρ : R) (h : List (Msg × Msg)) :
    L.toSrv.refMsg x w ρ h =
      frameOut L.stop (L.cfg x ρ) (L.masker x ρ h) (L.device x w ρ h (computedOf (L.computed x w ρ h)))
        (L.coins x h.length) :=
  rfl

/-- **The item-stream refinement** at `(x, w)` and the specified values `spec`: after every history, at honest worker
values, with any values for the firewall's own computation that agree with `spec` wherever they are given, the Lean
frame of the device round is `P`'s message. -/
def ItemsRefine {W R : Type} (L : LeanRelease W R) (H : Msg → Msg) (P : W → Prover R) (x : Msg) (w : W)
    (spec : String → Option ByteArray) : Prop :=
  ∀ ρ h (fix : String → Option ByteArray), (∀ what v, fix what = some v → spec what = some v) →
    (((P w).next x h ρ).eval H).1 =
      frameOut L.stop (L.cfg x ρ) (L.masker x ρ h) (L.device x w ρ h fix) (L.coins x h.length)

/-- **`hRef` at the Lean release**: D8 (a) on every transcript the firewall keeps, and the item-stream refinement, give
`ZkServer.Ref`. -/
theorem ref {W R : Type} (L : LeanRelease W R) (H : Msg → Msg) (P : W → Prover R) (x : Msg) (w : W)
    (spec : String → Option ByteArray)
    (hc : ∀ ρ h, _root_.Flock.Firewall.Contract.FirewallComputes spec (L.computed x w ρ h))
    (hI : L.ItemsRefine H P x w spec) : L.toSrv.Ref H P x w := fun ρ h => by
  rw [refMsg_toSrv]
  exact hI ρ h _ fun _ _ hv => computedOf_spec (hc ρ h) hv

end LeanRelease

open FlockSoundness.Discharge

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- **The ZK server at `flock-circuit --zk`'s masked record, released by the Lean frame**: `Flock.zk_server_ref` at the
deployed release `L.toSrv`, its `hRef` given as D8 (a) on the firewall's transcripts (`FirewallComputes`) and the
item-stream refinement at the record's reference prover (`LeanRelease.ItemsRefine`). -/
theorem Flock.zk_server_lean (H : Msg → Msg) (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (S : RefSession) (t : ℕ) (ht : 0 < t) {δ₁ : ENNReal} (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁)
    (hδ₁ : δ₁ ≠ ⊤) (hS : S.Tables) (hV : S.Complete (Flock.R (FlockDeploy.deployZk rel).parse)) (hD : S.DummyReg)
    (hc : S.Causal) (hT : S.TapeOK) (coinOf : Msg → Msg)
    (tapeOf : S.refP.Nonce × S.refP.Rand → FlockComplete.TapeZk) (hH : S.HonestMsgs H coinOf tapeOf)
    (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand)) (x : Msg) (w : Flock.Wit)
    (hR : Flock.R (FlockDeploy.deployZk rel).parse x w) (spec : String → Option ByteArray)
    (hFC : ∀ ρ h, _root_.Flock.Firewall.Contract.FirewallComputes spec (L.computed x w ρ h))
    (hI : L.ItemsRefine H (S.zkD rel coinOf tapeOf t).prover x w spec)
    (ε : Msg → ℕ → ℕ → ℝ) (hε : ∀ j s, 0 ≤ ε x j s) (hCS : L.toSrv.CheckSound ε) (wk : L.toSrv.Worker) {Θ : Type}
    (Vs : VStar Θ) (θ : Θ) (hk : Vs.k x = (FlockDeploy.deployZk rel).V.k x + 1)
    (Ev : List ((Msg × Out) × ℕ) → Prop) :
    Pr (fun ω : (S.refP.Nonce × S.refP.Rand) × L.toSrv.Chk =>
        Ev (L.toSrv.timed x (runStar H (L.toSrv.prover wk w) Vs x ω θ).msgs)) ≤
      ((FlockDeploy.deployZk rel).V.k x + 1 + 1) *
        (Pr (fun c : (leaky ((S.refP.withSim fun x _ c => S.coinModel x c).simulatorE t (t * 3 ^ t))
            (fun x => (FlockDeploy.deployZk rel).V.k x + 1) L.toSrv.stop).C x =>
          Ev (L.toSrv.timed x ((leaky ((S.refP.withSim fun x _ c => S.coinModel x c).simulatorE t (t * 3 ^ t))
            (fun x => (FlockDeploy.deployZk rel).V.k x + 1) L.toSrv.stop).out H Vs θ x c))) +
          (8 * S.nLeaves x * δ₁.toReal + 8 * t * S.nLeaves x * δ₁.toReal + 2 * (1 / 3) ^ t) +
          4 * t * 3 ^ t * (Fork H ((S.zkD rel coinOf tapeOf t).prover w) Vs θ x).Pr H) +
      ∑ i ∈ Finset.range ((FlockDeploy.deployZk rel).V.k x + 1),
        ∑ s ∈ (Finset.range (L.toSrv.nS x i)).filter (fun s => L.toSrv.kind x i s = .masked), ε x i s :=
  Flock.zk_server_ref H rel S t ht hT1 hδ₁ hS hV hD hc hT coinOf tapeOf hH L.toSrv x w hR
    (L.ref H _ x w spec hFC hI) ε hε hCS wk Vs θ hk Ev

end Claims.ZKServer
