module

public import Flock.Firewall.Session
public import Proofs.Flock.Claims.ZK.FrameInner
public import Proofs.Flock.Claims.ZK.FrameReal
public import Proofs.Flock.Claims.ZK.FrameState
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init

/-!
# `Flock.zk_server_lean` at the session loop's own frames

`Flock.zk_server_lean_loop` took the transport's frame state as given (`ProgReads`, at a `pc` of the caller's). At
`LeanRelease.ofLoop` that state is a Lean function, so `ProgReads` holds by definition (`ofLoop_progReads`) and what it
took as given is `ProgChecks`: that the deployed frame's state is the loop's but for `stopped` (`ofLoop_progChecks`).
This file states the bound there with `ProgChecks` as a conclusion, its premises the guarantee's own:

* `Session.sizesOk` (the code's): the sizes the loop's frames read, a check on the setup, one pass over the rep streams:
  each table's link points the session's, nonzero and within `Frames.COIN_MAX` (`SessionSpec.mPts`, `points`); the
  stream's domain (`SessionSpec.domain`, `Drive`'s domain frame) and the extra claims at the link points (`Drive.extras`:
  the table's regions at each link point) within a `u64`'s count; and the placeholder's pad row, `Drive.start`'s layout
  plus `Zk.K_PAD` (`Cfg.ofDraws`'s `rowH`), within a `u64`'s count. The loop refuses a session whose setup fails it
  (`Session.State.sendHello`: no `Hello` leaves).
* `Loop.Serves`, `Loop.sizesOk_of_serves`: the loop serves `x` where its `Hello` leaves at `x`'s setup
  (`FlockDeploy.setupOf`), and then the setup passes `sizesOk` (`sizesOk_of_sendHello`).
* `Loop.SwOk T`: the coin vectors with no zerocheck outer coin `1` at a `Live` statement the loop serves, or that every
  table admits, where `RefSession.SwCausalOn` asks the switched round encoding to be causal; `SwCausal` gives it there
  (`Loop.swCausalOn_swOk`).
* `LeanRelease.ofLoop_progChecks_sizes`: `ofLoop_progChecks` with its size premises (`hS`, `hP`, `hRow`) from that check.
* **`Flock.zk_server_lean_ofLoop`**: `Flock.zk_server_lean_loop` at `ofLoop`, at `Loop.progOf` and `Loop.freshOf`,
  with no `ProgReads`, and `ProgChecks` at `ofLoop`, at every `x` the loop serves. Beyond `zk_server_lean_loop`'s other premises it reads the
  tables `real` answers being the session loop's own (`Count.RealTabs`: `Finish`'s `level0` request, `tabOfLevel0` at a
  level 0 `Level0.ofDraws` drew) and `L₀`'s round at the inner proof's round 0 being the ZK server's own at some
  `final_c'` (`hL0`: the loop's frame releases that round itself, `Frame.innerSched` on its finish's round 0, at the inner
  proof's label).
-/

@[expose] public section

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec)
open _root_.Flock.Transcript (header)
open _root_.Flock.Firewall.Frame (Cfg Masker labelFrame)
open _root_.Flock.Firewall.Frames (COIN_MAX)
open _root_.Flock.Firewall.Session (sizesOk)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- as in the ZK files this builds on; its own need is unmeasured

namespace Loop

/-- **The loop serves `x`**: at `x`'s setup, the session loop's `Hello` leaves (`Session.State.sendHello` from a fresh
session, at some nonce). Where it does not, the loop refuses the `Hello` and stops the session: a setup whose sizes are
past its frames' (`Session.sizesOk`) is refused there. -/
def Serves (x : Msg) : Prop :=
  ∀ (s0 : FlockDeploy.Ins) (spec : SessionSpec) (sts : Array Stmt), FlockDeploy.setupOf true x = some (s0, spec, sts) →
    ∃ (nonce : ByteArray) (r : _root_.Flock.Firewall.Session.State × ByteArray),
      ({} : _root_.Flock.Firewall.Session.State).sendHello (.ok (spec, sts)) nonce = .ok r

/-- Where the loop's `Hello` leaves, the setup passes `sizesOk`. -/
theorem sizesOk_of_sendHello {S : _root_.Flock.Firewall.Session.State} {spec : SessionSpec} {sts : Array Stmt}
    {nonce : ByteArray} {r : _root_.Flock.Firewall.Session.State × ByteArray}
    (h : S.sendHello (.ok (spec, sts)) nonce = .ok r) : sizesOk spec sts = true := by
  unfold _root_.Flock.Firewall.Session.State.sendHello at h
  cases hs : S.stop with
  | some why => simp [hs] at h
  | none =>
    cases hz : sizesOk spec sts with
    | true => rfl
    | false => simp [hs, hz] at h

/-- **Where the loop serves `x`, its setup passes `sizesOk`**. -/
theorem sizesOk_of_serves {x : Msg} (h : Serves x) :
    ∀ (s0 : FlockDeploy.Ins) (spec : SessionSpec) (sts : Array Stmt),
      FlockDeploy.setupOf true x = some (s0, spec, sts) → sizesOk spec sts = true :=
  fun s0 spec sts hx =>
    let ⟨_, _, he⟩ := h s0 spec sts hx
    sizesOk_of_sendHello he

theorem sizesOk_at {spec : SessionSpec} {sts : Array Stmt} (h : sizesOk spec sts = true) {k : ℕ}
    (hk : k < spec.repStreams.size) :
    (sts[k / 2]!.mPts = spec.mPts ∧ 0 < spec.mPts ∧ spec.points * spec.mPts ≤ COIN_MAX) ∧
      (spec.domain (spec.tables[k / 2]!) (k % 2)).toUTF8.size < 2 ^ 64 ∧
      sts[k / 2]!.regions.size * COIN_MAX + 2 < 2 ^ 64 ∧
      ∀ lay mask, _root_.Flock.Firewall.Drive.start sts[k / 2]! = .ok (lay, mask) →
        lay.len + _root_.Flock.Zk.K_PAD < 2 ^ 64 := by
  have h' := List.all_eq_true.1 h k (List.mem_range.2 hk)
  simp only [Bool.and_eq_true, decide_eq_true_eq] at h'
  obtain ⟨⟨⟨h1, h2⟩, h3⟩, h4⟩ := h'
  refine ⟨h1, h2, h3, fun lay mask hs => ?_⟩
  rw [hs] at h4
  simpa using h4

/-- The extra claims at the link points `pts`: the table's regions at each point. -/
theorem extras_size (st : Stmt) (pts : Array (Array F128)) :
    (_root_.Flock.Firewall.Drive.extras st pts).size = st.regions.size * pts.size := by
  unfold _root_.Flock.Firewall.Drive.extras
  rw [Array.size_flatMap]
  rcases st.regions with ⟨l⟩
  simp only [List.map_toArray, List.sum_toArray, List.size_toArray]
  induction l with
  | nil => simp
  | cons a l ih => simp [Nat.succ_mul]; omega

theorem linkPoints_size_le (st : Stmt) (c : Array F128) :
    (_root_.Flock.Firewall.Drive.linkPoints st c).size ≤ c.size := by
  unfold _root_.Flock.Firewall.Drive.linkPoints
  rw [Array.size_map, Array.size_range]
  exact Nat.div_le_self _ _

/-- **The coin vectors the one-tape switched causality covers** (`RefSession.SwCausalOn`): at a `Live` statement the
loop serves, no table's rep's zerocheck outer coin is `1`; or every table admits them. -/
def SwOk (T : ZKMal.CoinLayer.TableData) (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) : Prop :=
  (Masked.Live x → Serves x → ∀ j < (Masked.stsOf x).size, ∀ r < 2, ∀ e < (Masked.stsOf x)[j]!.m - 13,
      Masked.outerAt c j r e ≠ 1) ∨
    (ZKMal.CoinLayer.session loopNr T).Admits x c

/-- Coin vectors every table admits are `SwOk`'s. -/
theorem swOk_of_admits (T : ZKMal.CoinLayer.TableData) {x : Msg} {c : Fin (refK x) → FlockDeploy.Coin}
    (h : (ZKMal.CoinLayer.session loopNr T).Admits x c) : SwOk T x c :=
  .inr h

/-- Where the statement is not `Live` or the loop doesn't serve it, every coin vector is `SwOk`'s, so `SwCausalOn` at
`SwOk` is one-tape `SwCausal` there. -/
theorem swOk_of_not_served (T : ZKMal.CoinLayer.TableData) {x : Msg} (h : ¬ (Masked.Live x ∧ Serves x))
    (c : Fin (refK x) → FlockDeploy.Coin) : SwOk T x c :=
  .inl fun hx hS => absurd ⟨hx, hS⟩ h

/-- **`SwCausal vs` gives `SwCausalOn (SwOk T) vs`**: the restricted premise asks for less. -/
theorem swCausalOn_swOk (T : ZKMal.CoinLayer.TableData) (vs : (ZKMal.CoinLayer.session loopNr T).SwVals)
    (h : (ZKMal.CoinLayer.session loopNr T).SwCausal vs) :
    (ZKMal.CoinLayer.session loopNr T).SwCausalOn (SwOk T) vs :=
  (ZKMal.CoinLayer.session loopNr T).swCausalOn_of_swCausal vs _ h

end Loop

namespace LeanRelease

open Loop in
/-- **`ProgChecks` at the loop's own frames, its sizes checked** (`ofLoop_progChecks`, its premises `hS`, `hP` and
`hRow` from `Session.sizesOk` at the setup): from the check, the rep's own tables where `real` answers the loop's (`hR`)
and `L₀`'s round at the inner proof's round 0 the ZK server's own at some `final_c'` (`hL0`). -/
theorem ofLoop_progChecks_sizes (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (x : Msg) (w : Flock.Wit)
    (hSz : ∀ (s0 : _) (spec : SessionSpec) (sts : Array Stmt),
      FlockDeploy.setupOf true x = some (s0, spec, sts) → sizesOk spec sts = true)
    (hR : ∀ (ρ : (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)
      (s0 : _) (spec : SessionSpec) (sts : Array Stmt),
      FlockDeploy.setupOf true x = some (s0, spec, sts) → Count.RealTabs sts (real x w ρ))
    (hL0 : ∀ (ρ : (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)
      (h : List (Msg × Msg)) (vals : _) (s0 : _) (spec : SessionSpec) (sts : Array Stmt) (q k j : ℕ)
      (tr : _root_.Flock.Firewall.Finish.Tab) (hr : Array ByteArray) (rsched : Array (ByteArray × ℕ)),
      FlockDeploy.setupOf true x = some (s0, spec, sts) → h.length = q + 2 →
      ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q = (k, j + 1) → real x w ρ k = .ok (tr, hr) →
      schedSoFar tr (k % 2) (spec.domain (spec.tables[k / 2]!) (k % 2)).toUTF8
        (_root_.Flock.Firewall.Drive.extras sts[k / 2]! (_root_.Flock.Firewall.Drive.linkPoints sts[k / 2]!
          (wordsAt x (h.tail.map fun p => ZKMal.CoinLayer.rd p.2) 0))) hr
        ((Array.range (j + 1)).map
          fun i => wordsAt x (h.tail.map fun p => ZKMal.CoinLayer.rd p.2) (q - (j + 1) + i + 1)) = .ok rsched →
      rsched.size = j + 1 →
      ∃ w' : ByteArray, w'.size = 16 ∧ L₀.scheduled x w ρ h vals =
        (labelFrame _root_.Flock.Zk.LABEL_INNER.toUTF8 ++ header 3 1 1 ++ w' ++ header 4 1 1).data.toList) :
    ProgChecks (ofLoop T L₀ real) x w := by
  refine ofLoop_progChecks T L₀ real x w ?_ hR ?_ ?_ hL0
  · intro s0 spec sts k link c' t dom xs hi hx hk hl hph
    obtain ⟨-, hd, hx2, -⟩ := sizesOk_at (hSz s0 spec sts hx) hk
    obtain ⟨-, -, -, -, hdm, hxe⟩ := placeholder_ok hph
    subst hdm hxe
    refine ⟨hd, ?_⟩
    rw [extras_size]
    have := Nat.mul_le_mul_left sts[k / 2]!.regions.size ((linkPoints_size_le sts[k / 2]! link).trans hl)
    omega
  · intro s0 spec sts k hx hk
    exact (sizesOk_at (hSz s0 spec sts hx) hk).1
  · intro s0 spec sts k link c' t dom xs hi hx hk hph
    obtain ⟨lay, mask, hs, hc, -, -⟩ := placeholder_ok hph
    subst hc
    rw [ofDraws_rowH_size]
    exact (sizesOk_at (hSz s0 spec sts hx) hk).2.2.2 lay mask hs

end LeanRelease

open FlockSoundness.Discharge

/-- **`Flock.zk_server_lean_loop` at the session loop's own frames** (`LeanRelease.ofLoop`): the transport's frame
states are `Loop.progOf` and `Loop.freshOf` of the decoded coins, so `ProgReads` is a theorem (`ofLoop_progReads`),
and the deployed frame's state is that one but for `stopped` (`ProgChecks`, `ofLoop_progChecks_sizes`), a conclusion
here, at every `x` the loop
serves (`Loop.Serves`: its `Hello` leaves, so the setup's sizes pass `Session.sizesOk`). In `ProgReads`' place it reads
the tables `real` answers being the session loop's own
(`hRt`, `Count.RealTabs`: `Finish`'s `level0` request) and `L₀`'s round at the inner proof's round 0 being the
ZK server's own at some `final_c'` (`hL0`: the loop's frame releases that round itself, `Frame.innerSched`). The
transport's other trusts
(`LinkRoundEq`, `FinalRoundEq`, `LiveRoundEqOwn`, `FillRoundEq`) and the tables' side are `zk_server_lean_loop`'s. -/
theorem Flock.zk_server_lean_ofLoop
    (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (T : CoinLayer.TableData)
    (vs : (CoinLayer.session Loop.loopNr T).SwVals) (t : ℕ) (ht : 0 < t)
    {δ₁ : ENNReal} (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤)
    (hS : (CoinLayer.session Loop.loopNr T).Tables)
    (hV : (CoinLayer.session Loop.loopNr T).Complete (Flock.R (FlockDeploy.deployZk rel).parse))
    (hD : (CoinLayer.session Loop.loopNr T).DummyReg) (hc : (CoinLayer.session Loop.loopNr T).Causal)
    (hsc : (CoinLayer.session Loop.loopNr T).SwCausalOn (Loop.SwOk T) vs) (hT : (CoinLayer.session Loop.loopNr T).TapeOK)
    (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr T).refP.Nonce × (CoinLayer.session Loop.loopNr T).refP.Rand))
    (real : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr T).refP.Nonce × (CoinLayer.session Loop.loopNr T).refP.Rand →
      ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (hOk : (CoinLayer.session Loop.loopNr T).StopsOk (LeanRelease.rfOf (CoinLayer.session Loop.loopNr T)
      (LeanRelease.ofLoop T L₀ real) Loop.progOf Loop.freshOf Loop.readCoins) (Loop.SwOk T))
    (wk : (LeanRelease.ofLoop T L₀ real).toSrv.Worker) (x : Msg) (w : Flock.Wit)
    (hR : Flock.R (FlockDeploy.deployZk rel).parse x w) (hServe : Loop.Serves x)
    (hRt : ∀ (ρ : (CoinLayer.session Loop.loopNr T).refP.Nonce × (CoinLayer.session Loop.loopNr T).refP.Rand)
      (s0 : _) (spec : SessionSpec) (sts : Array Stmt),
      FlockDeploy.setupOf true x = some (s0, spec, sts) → Count.RealTabs sts (real x w ρ))
    (hL0 : ∀ (ρ : (CoinLayer.session Loop.loopNr T).refP.Nonce × (CoinLayer.session Loop.loopNr T).refP.Rand)
      (h : List (Msg × Msg)) (vals : _) (s0 : _) (spec : SessionSpec) (sts : Array Stmt) (q k j : ℕ)
      (tr : _root_.Flock.Firewall.Finish.Tab) (hr : Array ByteArray) (rsched : Array (ByteArray × ℕ)),
      FlockDeploy.setupOf true x = some (s0, spec, sts) → h.length = q + 2 →
      CoinLayer.locate (Loop.roundsOf spec sts) 0 q = (k, j + 1) → real x w ρ k = .ok (tr, hr) →
      Loop.schedSoFar tr (k % 2) (spec.domain (spec.tables[k / 2]!) (k % 2)).toUTF8
        (_root_.Flock.Firewall.Drive.extras sts[k / 2]! (_root_.Flock.Firewall.Drive.linkPoints sts[k / 2]!
          (Loop.wordsAt x (h.tail.map fun p => CoinLayer.rd p.2) 0))) hr
        ((Array.range (j + 1)).map
          fun i => Loop.wordsAt x (h.tail.map fun p => CoinLayer.rd p.2) (q - (j + 1) + i + 1)) = .ok rsched →
      rsched.size = j + 1 →
      ∃ w' : ByteArray, w'.size = 16 ∧ L₀.scheduled x w ρ h vals =
        (labelFrame _root_.Flock.Zk.LABEL_INNER.toUTF8 ++ header 3 1 1 ++ w' ++ header 4 1 1).data.toList)
    (hLk : LeanRelease.LinkRoundEq (CoinLayer.session Loop.loopNr T) vs (LeanRelease.ofLoop T L₀ real) x w)
    (hF : LeanRelease.FinalRoundEq (CoinLayer.session Loop.loopNr T) vs (LeanRelease.ofLoop T L₀ real) CoinOpen.shaH
      wk x w)
    (hE : LeanRelease.LiveRoundEqOwn (CoinLayer.session Loop.loopNr T) vs (LeanRelease.ofLoop T L₀ real) CoinOpen.shaH
      wk x w)
    (hEf : LeanRelease.FillRoundEq (CoinLayer.session Loop.loopNr T) vs (LeanRelease.ofLoop T L₀ real) CoinOpen.shaH
      wk x w)
    {Θ : Type} (Vs : VStar Θ) (θ : Θ) (hk : Vs.k x = (FlockDeploy.deployZk rel).V.k x + 1)
    (Ev : List ((Msg × Out) × ℕ) → Prop) :
    (LeanRelease.ofLoop T L₀ real).ProgChecks x w ∧
    Pr (fun ω : ((CoinLayer.session Loop.loopNr T).refP.Nonce × (CoinLayer.session Loop.loopNr T).refP.Rand) ×
          (LeanRelease.ofLoop T L₀ real).toSrv.Chk =>
        Ev ((LeanRelease.ofLoop T L₀ real).toSrv.timed x
          (runStar CoinOpen.shaH ((LeanRelease.ofLoop T L₀ real).toSrv.proverD (LeanRelease.ofLoop T L₀ real).fillOf
            (fun x => (FlockDeploy.deployZk rel).V.k x + 1) wk w) Vs x ω θ).msgs)) ≤
      2 * (Pr (fun c : (leaky1 ((((CoinLayer.session Loop.loopNr T).withStops
                (LeanRelease.rfOf (CoinLayer.session Loop.loopNr T) (LeanRelease.ofLoop T L₀ real) Loop.progOf
                  Loop.freshOf Loop.readCoins) (LeanRelease.ofLoop T L₀ real).stop).refP.withSim fun x _ c =>
              ((CoinLayer.session Loop.loopNr T).withStops
                (LeanRelease.rfOf (CoinLayer.session Loop.loopNr T) (LeanRelease.ofLoop T L₀ real) Loop.progOf
                  Loop.freshOf Loop.readCoins) (LeanRelease.ofLoop T L₀ real).stop).coinModel x c).simulatorE t (t * 3 ^ t))
            (fun x => (FlockDeploy.deployZk rel).V.k x + 1) (LeanRelease.ofLoop T L₀ real).toSrv.stop).C x =>
          Ev ((LeanRelease.ofLoop T L₀ real).toSrv.timed x ((leaky1 ((((CoinLayer.session Loop.loopNr T).withStops
                (LeanRelease.rfOf (CoinLayer.session Loop.loopNr T) (LeanRelease.ofLoop T L₀ real) Loop.progOf
                  Loop.freshOf Loop.readCoins) (LeanRelease.ofLoop T L₀ real).stop).refP.withSim fun x _ c =>
              ((CoinLayer.session Loop.loopNr T).withStops
                (LeanRelease.rfOf (CoinLayer.session Loop.loopNr T) (LeanRelease.ofLoop T L₀ real) Loop.progOf
                  Loop.freshOf Loop.readCoins) (LeanRelease.ofLoop T L₀ real).stop).coinModel x c).simulatorE t (t * 3 ^ t))
            (fun x => (FlockDeploy.deployZk rel).V.k x + 1) (LeanRelease.ofLoop T L₀ real).toSrv.stop).out
              CoinOpen.shaH Vs θ x c))) +
          (8 * ((CoinLayer.session Loop.loopNr T).withStops (LeanRelease.rfOf (CoinLayer.session Loop.loopNr T)
              (LeanRelease.ofLoop T L₀ real) Loop.progOf Loop.freshOf Loop.readCoins)
              (LeanRelease.ofLoop T L₀ real).stop).nLeaves x * δ₁.toReal +
            8 * t * ((CoinLayer.session Loop.loopNr T).withStops (LeanRelease.rfOf (CoinLayer.session Loop.loopNr T)
              (LeanRelease.ofLoop T L₀ real) Loop.progOf Loop.freshOf Loop.readCoins)
              (LeanRelease.ofLoop T L₀ real).stop).nLeaves x * δ₁.toReal + 2 * (1 / 3) ^ t) +
          4 * t * 3 ^ t *
            (Fork CoinOpen.shaH ((((CoinLayer.session Loop.loopNr T).withStops
                (LeanRelease.rfOf (CoinLayer.session Loop.loopNr T) (LeanRelease.ofLoop T L₀ real) Loop.progOf
                  Loop.freshOf Loop.readCoins) (LeanRelease.ofLoop T L₀ real).stop).swP vs).prover w) Vs θ x).Pr
              CoinOpen.shaH) :=
  ⟨LeanRelease.ofLoop_progChecks_sizes T L₀ real x w (Loop.sizesOk_of_serves hServe) hRt hL0,
    Flock.zk_server_lean_loop rel T vs t ht hT1 hδ₁ hS hV hD hc (Loop.SwOk T) hsc hT (LeanRelease.withFrames T L₀ real)
      (LeanRelease.ofLoop T L₀ real) rfl wk x w hR Loop.progOf Loop.freshOf
      (LeanRelease.ofLoop_progReads T L₀ real x) hOk
      hLk hF hE hEf Vs θ hk Ev⟩

end Claims.ZKServer
