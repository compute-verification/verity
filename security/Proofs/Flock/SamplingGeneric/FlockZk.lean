module

public import Proofs.Flock.Claims.ZK.SessionLean
public import Proofs.Flock.Claims.ZK.DeferredOut
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
# Flock's ZK server per outcome

Flock's ZK server (`Claims.ZKServer.Flock.zk_server_lean_loop`) releases the session's transcript within a factor `2` of
`leaky1`'s, the simulator that guesses whether the session stops: one bit a session. Where the session stopped is public
(`ZKServer.outD`), so given the outcome the simulator guesses nothing, and the server is zero knowledge per outcome in
the claims core's `ZKOut` at the factor `1`: the instance of `ZkArg.ZK`'s notion at Flock's masked session.

* `Sess`, `Tape`, `Release`, `PC`, `FC`, `Stops`, `rounds`: the server's session, tape, release, frame states and rounds.
* `Live`, `LiveRel`: the live path's premises at a statement, each `zk_server_lean_loop`'s or one that gives it, and the
  relation the server is zero knowledge at.
* `simOut`, `errOut`: the server's simulator given the outcome, and its error per outcome.
* `zkOut_stops`, `server_zkOut`: `zk_server_lean_loop` per outcome.

Each holds at any masked record `D` whose rounds on its relation are the session's, `refK x + 1`
(`roundsAt`, `LiveRelAt`, `simOutAt`, `server_zkOutAt`); `rounds`, `LiveRel`, `simOut` and `server_zkOut` are those at
Flock's masked deployment (`FlockDeploy.deployZk rel`, `ZKMal.V_k_of_R`). The session reads the setup of the masked
session a statement makes, Flock's or the recursive one's (`ZKMal.sessOf`), so the same server is zero knowledge at the
recursive outer record too (`RecGeneric.server_zkOutRec`).
-/

@[expose] public section

namespace Claims.SamplingGeneric.FlockZk

open scoped ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- The relation a parsed statement names, as the masked deployment reads it. -/
abbrev Parser : Type := _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation

/-- The hash the server's runs call: SHA-512 on messages. -/
abbrev sha : Claims.Msg → Claims.Msg := Claims.ZKMal.CoinOpen.shaH

/-- The ZK server's session at the tables `T`: the coin layer's, at the coin counts the session loop requests. -/
abbrev Sess (T : Claims.ZKMal.CoinLayer.TableData) : Claims.ZKMal.RefSession :=
  Claims.ZKMal.CoinLayer.session Claims.ZKServer.Loop.loopNr T

/-- The server's tape: the session's nonce and the rest of the reference prover's tape. -/
abbrev Tape (T : Claims.ZKMal.CoinLayer.TableData) : Type := (Sess T).refP.Nonce × (Sess T).refP.Rand

/-- The release at the server's tape. -/
abbrev Release (T : Claims.ZKMal.CoinLayer.TableData) : Type := Claims.ZKServer.LeanRelease Claims.Flock.Wit (Tape T)

/-- The transport's frame state at each statement and coins (`zk_server_lean_loop`'s `pc`). -/
abbrev PC (T : Claims.ZKMal.CoinLayer.TableData) : Type :=
  Claims.Msg → List (Sess T).Coin → _root_.Flock.Firewall.Frame.Masker

/-- The transport's checked frame state (`zk_server_lean_loop`'s `fc`). -/
abbrev FC (T : Claims.ZKMal.CoinLayer.TableData) : Type :=
  Claims.Msg → List (Sess T).Coin → Except String _root_.Flock.Firewall.Frame.Masker

/-- The session with the release's stops, which the server's simulator and its error read. -/
abbrev Stops (T : Claims.ZKMal.CoinLayer.TableData) (L : Release T) (pc : PC T) (fc : FC T) :
    Claims.ZKMal.RefSession :=
  (Sess T).withStops (Claims.ZKServer.LeanRelease.rfOf (Sess T) L pc fc Claims.ZKServer.Loop.readCoins) L.stop

/-- The server's rounds at `x` at a masked record `D`: its verifier's and the `Hello`. -/
abbrev roundsAt {Coin Col Tr : Type} (D : Claims.Flock.Deploy Coin Col Tr) : Claims.Msg → ℕ := fun x => D.V.k x + 1

/-- The server's rounds at `x`: the masked verifier's and the `Hello`. -/
noncomputable abbrev rounds (rel : Parser) : Claims.Msg → ℕ := roundsAt (FlockDeploy.deployZk rel)

/-- **The live path's premises at `(x, w)`**, each `zk_server_lean_loop`'s or one that gives it: the transport's frame
states (`ProgReads`), and the link, final, live and fill rounds' releases being the reference's (`LinkRoundEq`,
`FinalRoundEq`, `LiveRoundEq`, `FillRoundEq`). `LiveRoundEq` holds at every history and gives `LiveRoundEqOwn`, the
worker's own (`LiveRoundEq.own`). -/
abbrev Live (T : Claims.ZKMal.CoinLayer.TableData) (vs : (Sess T).SwVals) (L : Release T) (wk : L.toSrv.Worker)
    (pc : PC T) (fc : FC T) (x : Claims.Msg) (w : Claims.Flock.Wit) : Prop :=
  Claims.ZKServer.LeanRelease.ProgReads (Sess T) L pc fc x ∧
    Claims.ZKServer.LeanRelease.LinkRoundEq (Sess T) vs L x w ∧
    Claims.ZKServer.LeanRelease.FinalRoundEq (Sess T) vs L sha wk x w ∧
    Claims.ZKServer.LeanRelease.LiveRoundEq (Sess T) vs L sha wk x w ∧
    Claims.ZKServer.LeanRelease.FillRoundEq (Sess T) vs L sha wk x w

/-- **The relation the server is zero knowledge at**, at a masked record `D`: the statement's (`Claims.Flock.R` at
`D`'s parser) with the live path's premises at the statement (`Live`). -/
abbrev LiveRelAt {Coin Col Tr : Type} (D : Claims.Flock.Deploy Coin Col Tr) (T : Claims.ZKMal.CoinLayer.TableData)
    (vs : (Sess T).SwVals) (L : Release T) (wk : L.toSrv.Worker) (pc : PC T) (fc : FC T) (x : Claims.Msg)
    (w : Claims.Flock.Wit) : Prop :=
  Claims.Flock.R D.parse x w ∧ Live T vs L wk pc fc x w

/-- **The relation the server is zero knowledge at**: the statement's (`Claims.Flock.R` at the masked deployment's
parser) with the live path's premises at the statement (`Live`). -/
abbrev LiveRel (rel : Parser) (T : Claims.ZKMal.CoinLayer.TableData) (vs : (Sess T).SwVals) (L : Release T)
    (wk : L.toSrv.Worker) (pc : PC T) (fc : FC T) (x : Claims.Msg) (w : Claims.Flock.Wit) : Prop :=
  LiveRelAt (FlockDeploy.deployZk rel) T vs L wk pc fc x w

/-! ## Per outcome -/

/-- **The server's simulator given the outcome** at a masked record `D`: `sim` with its guess replaced by the outcome's
bit (`ZKServer.simOutD`), at `D`'s rounds. -/
noncomputable abbrev simOutAt {Coin Col Tr : Type} (D : Claims.Flock.Deploy Coin Col Tr)
    (T : Claims.ZKMal.CoinLayer.TableData) (L : Release T) (pc : PC T) (fc : FC T) (t : ℕ) :
    Claims.SimOut (Option Bool × Bool) :=
  Claims.ZKServer.simOutD
    (((Stops T L pc fc).refP.withSim fun x _ c => (Stops T L pc fc).coinModel x c).simulatorE t (t * 3 ^ t))
    (roundsAt D) L.toSrv.stop

/-- **The server's simulator given the outcome**: `sim` with its guess replaced by the outcome's bit
(`ZKServer.simOutD`). -/
noncomputable abbrev simOut (rel : Parser) (T : Claims.ZKMal.CoinLayer.TableData) (L : Release T) (pc : PC T)
    (fc : FC T) (t : ℕ) : Claims.SimOut (Option Bool × Bool) :=
  simOutAt (FlockDeploy.deployZk rel) T L pc fc t

/-- **The server's error per outcome** at `(x, w)` against `V*` at `θ`, at every outcome:
`8·N·δ₁ + 8t·N·δ₁ + 2·3^{-t} + 4t·3^t·Pr[Fork]`, `err` without its factor `2`. -/
noncomputable def errOut (T : Claims.ZKMal.CoinLayer.TableData) (vs : (Sess T).SwVals) (L : Release T) (pc : PC T)
    (fc : FC T) (t : ℕ) (δ₁ : ℝ≥0∞) (x : Claims.Msg) (w : Claims.Flock.Wit) (Θ : Type) (Vs : Claims.VStar Θ) (θ : Θ)
    (_ : Option Bool × Bool) : ℝ :=
  (8 * (Stops T L pc fc).nLeaves x * δ₁.toReal + 8 * t * (Stops T L pc fc).nLeaves x * δ₁.toReal + 2 * (1 / 3) ^ t) +
    4 * t * 3 ^ t * (Claims.Fork sha (((Stops T L pc fc).swP vs).prover w) Vs θ x).Pr sha

/-- **`Flock.zk_server_lean_stops` per outcome**, at a masked record `D` whose rounds on its relation are the session's
(`hkD`): at a session `S`, a stop rule `Rf` and the premises `Flock.zk_server_lean_stops` takes, the deferred server's
transcript shows an outcome `o` (`ZKServer.outD`) and an event with probability at most the simulator's given `o`
(`ZKServer.simOutD`, at the session with the frame's stops) of the event, plus the server's error without its factor `2`.
`ZkServer.zkOut_deferred` where `Flock.zk_server_lean` takes `ZkServer.zk_deferred`, at the item-stream refinement
`Flock.zk_server_lean_stops` proves. -/
theorem zkOut_stops (H : Claims.Msg → Claims.Msg) {Coin Col Tr : Type} (D : Claims.Flock.Deploy Coin Col Tr)
    (hzk : D.zk = true) (hkD : ∀ x w, Claims.Flock.R D.parse x w → D.V.k x = Claims.ZKMal.refK x + 1)
    (S : Claims.ZKMal.RefSession)
    (Rf : (x : Claims.Msg) → (Fin (Claims.ZKMal.refK x) → S.Coin) → ℕ → Bool) (hRf : S.StopsCausal Rf)
    (vs : S.SwVals) (t : ℕ) (ht : 0 < t) {δ₁ : ℝ≥0∞}
    (hT1 : Claims.Hm96Hiding FlockSoundness.Discharge.ZkView.My FlockSoundness.Discharge.ZkView.saltHash δ₁)
    (hδ₁ : δ₁ ≠ ⊤) (hS : S.Tables) (hV : S.Complete (Claims.Flock.R D.parse))
    (hD : S.DummyReg) (hc : S.Causal) (hsc : S.SwCausal vs) (hT : S.TapeOK)
    (L : Claims.ZKServer.LeanRelease Claims.Flock.Wit (S.refP.Nonce × S.refP.Rand)) (wk : L.toSrv.Worker)
    (x : Claims.Msg) (w : Claims.Flock.Wit) (hR : Claims.Flock.R D.parse x w)
    (hlast : L.last x = Claims.ZKMal.refK x + 1) (hst : Claims.ZKServer.LeanRelease.StopsAt S Rf L H x)
    (hC : Claims.ZKServer.LeanRelease.CheckIs S L H x) (hH : Claims.ZKServer.LeanRelease.HelloEq S L x)
    (hLk : Claims.ZKServer.LeanRelease.LinkRoundEq S vs L x w)
    (hF : Claims.ZKServer.LeanRelease.FinalRoundEq S vs L H wk x w)
    (hE : Claims.ZKServer.LeanRelease.LiveRoundEq S vs L H wk x w)
    (hEf : Claims.ZKServer.LeanRelease.FillRoundEq S vs L H wk x w)
    {Θ : Type} (Vs : Claims.VStar Θ) (θ : Θ) (hk : Vs.k x = roundsAt D x) (o : Option Bool × Bool)
    (Ev : List (Claims.Msg × Claims.Out) → Prop) :
    Claims.Pr (fun ω : (S.refP.Nonce × S.refP.Rand) × L.toSrv.Chk =>
        Claims.ZKServer.outD L.toSrv.stop (roundsAt D) x
            (Claims.runStar H (L.toSrv.proverD L.fillOf (roundsAt D) wk w) Vs x ω θ).msgs = o ∧
          Ev (Claims.runStar H (L.toSrv.proverD L.fillOf (roundsAt D) wk w) Vs x ω θ).msgs) ≤
      Claims.Pr (fun c : (((S.withStops Rf L.stop).refP.withSim fun x _ c =>
            (S.withStops Rf L.stop).coinModel x c).simulatorE t (t * 3 ^ t)).C x =>
          Ev (((Claims.ZKServer.simOutD (((S.withStops Rf L.stop).refP.withSim fun x _ c =>
            (S.withStops Rf L.stop).coinModel x c).simulatorE t (t * 3 ^ t)) (roundsAt D) L.toSrv.stop).given o).out
              H Vs θ x c)) +
        ((8 * (S.withStops Rf L.stop).nLeaves x * δ₁.toReal + 8 * t * (S.withStops Rf L.stop).nLeaves x * δ₁.toReal +
            2 * (1 / 3) ^ t) +
          4 * t * 3 ^ t * (Claims.Fork H (((S.withStops Rf L.stop).swP vs).prover w) Vs θ x).Pr H) :=
  have hO := Claims.ZKServer.LeanRelease.openingsRefused_of_checkIs S L H x hC
  have hK : roundsAt D x = Claims.ZKMal.refK x + 2 := by
    have := hkD x w hR
    show D.V.k x + 1 = _
    omega
  L.toSrv.zkOut_deferred L.fillOf H (roundsAt D) ((S.withStops Rf L.stop).swP vs).prover _ _ _ _
    (Claims.ZKMal.Flock.zkAt_sw_stops H D hzk hkD S Rf L.stop vs (fun _ _ => True) t ht hT1 hδ₁ hS hV hD hc hRf
      (S.swCausalOn_of_swCausal vs _ hsc) (S.stopsOk_true Rf) hT).2 x w hR wk
    (L.refD H _ _ wk x w
      ⟨L.helloRefine_withStops S Rf vs H x w hH,
        L.ownRefine_withStops S Rf vs H (roundsAt D) wk x w hK hlast hst hO hLk hF,
        L.liveRefine_withStops S Rf vs H wk x w hlast hst hO hE.own,
        L.fillRefine_withStops S Rf vs H (roundsAt D) wk x w hK.le hlast hst hO hEf,
        L.stopRefine_withStops S Rf vs H (roundsAt D) wk x w hK.le hst⟩)
    Vs θ hk o Ev

/-- **Flock's ZK server is zero knowledge per outcome at a masked record `D`** whose rounds on its relation are the
session's (`hkD`): `server_zkOut`'s statement at `D`'s rounds (`roundsAt`), relation (`LiveRelAt`) and simulator
(`simOutAt`), from its premises with the session's completeness on `D`'s relation. -/
theorem server_zkOutAt {Coin Col Tr : Type} (D : Claims.Flock.Deploy Coin Col Tr) (hzk : D.zk = true)
    (hkD : ∀ x w, Claims.Flock.R D.parse x w → D.V.k x = Claims.ZKMal.refK x + 1)
    (T : Claims.ZKMal.CoinLayer.TableData) (vs : (Sess T).SwVals) (t : ℕ) (ht : 0 < t) {δ₁ : ℝ≥0∞}
    (hT1 : Claims.Hm96Hiding FlockSoundness.Discharge.ZkView.My FlockSoundness.Discharge.ZkView.saltHash δ₁)
    (hδ₁ : δ₁ ≠ ⊤) (hS : (Sess T).Tables) (hV : (Sess T).Complete (Claims.Flock.R D.parse))
    (hD : (Sess T).DummyReg) (hc : (Sess T).Causal) (hsc : (Sess T).SwCausal vs) (hT : (Sess T).TapeOK)
    (L₀ L : Release T) (hL : L = Claims.ZKServer.LeanRelease.loop Claims.ZKServer.Loop.loopNr T L₀)
    (wk : L.toSrv.Worker) (pc : PC T) (fc : FC T) :
    Claims.ZKOut (fun _ => Tape T) (fun _ => L.toSrv.Chk) sha (roundsAt D)
      (fun w _ => L.toSrv.proverD L.fillOf (roundsAt D) wk w) (LiveRelAt D T vs L wk pc fc)
      (Claims.ZKServer.outD L.toSrv.stop (roundsAt D)) (simOutAt D T L pc fc t) (errOut T vs L pc fc t δ₁) := by
  intro x w hR Θ Vs θ hk o Ev
  obtain ⟨hR, hPr, hLk, hF, hE, hEf⟩ := hR
  have hlast := Claims.ZKServer.LeanRelease.loop_last Claims.ZKServer.Loop.loopNr T L₀ x
  have hCo := Claims.ZKServer.LeanRelease.loop_coinsRead Claims.ZKServer.Loop.loopNr T L₀ x
  have hH := Claims.ZKServer.LeanRelease.loop_helloIs Claims.ZKServer.Loop.loopNr T L₀ x
  have hP := Claims.ZKServer.LeanRelease.loop_helloParses Claims.ZKServer.Loop.loopNr T L₀ x
  have hFw := Claims.ZKServer.LeanRelease.loop_reqsFollow T L₀ x
  rw [← hL] at hlast hCo hH hP hFw
  have hC := Claims.ZKServer.LeanRelease.checkIs_coinLayer Claims.ZKServer.Loop.loopNr T L x hP hFw
  exact zkOut_stops sha D hzk hkD (Sess T) _
    (Claims.ZKServer.LeanRelease.stopsCausal_rfOf (Sess T) L pc fc Claims.ZKServer.Loop.readCoins) vs t ht hT1 hδ₁
    hS hV hD hc hsc hT L wk x w hR hlast
    (Claims.ZKServer.LeanRelease.stopsAt_of_readsCoins (Sess T) L sha pc fc Claims.ZKServer.Loop.readCoins x
      ⟨hPr, hCo⟩ hC) hC
    (Claims.ZKServer.LeanRelease.helloEq_of_helloIs Claims.ZKServer.Loop.loopNr T L x hH) hLk hF hE hEf Vs θ hk o Ev

/-- **Flock's ZK server is zero knowledge per outcome** (`Claims.ZKOut`): at a statement in the relation with its live
path's premises (`LiveRel`), against every verifier of the server's rounds, the deferred server's transcript shows an
outcome `o` (`ZKServer.outD`: the verdict, and whether the session stopped at its `Finish`) and an event with
probability at most the simulator's given `o` (`simOut`) of the event, plus `errOut`, which reads the statement, the
witness and the verifier through `Fork`. `zk_server_lean_loop`'s factor `2` is the simulator's guess of the outcome's bit,
which the simulator given `o` doesn't make. The premises are `zk_server_lean_loop`'s or give them: `SwCausal` gives
`SwCausalOn` at every coin vector (`swCausalOn_of_swCausal`), which every stop rule's `StopsOk` takes in
(`stopsOk_true`). The release's facts at the session loop are `Flock.zk_server_lean_coinLayer`'s and
`zk_server_lean_loop`'s, then `zkOut_stops`. `server_zkOutAt` at Flock's masked deployment, whose rounds on its relation
are the session's (`ZKMal.V_k_of_R`). -/
theorem server_zkOut (rel : Parser) (T : Claims.ZKMal.CoinLayer.TableData) (vs : (Sess T).SwVals) (t : ℕ)
    (ht : 0 < t) {δ₁ : ℝ≥0∞}
    (hT1 : Claims.Hm96Hiding FlockSoundness.Discharge.ZkView.My FlockSoundness.Discharge.ZkView.saltHash δ₁)
    (hδ₁ : δ₁ ≠ ⊤) (hS : (Sess T).Tables) (hV : (Sess T).Complete (Claims.Flock.R (FlockDeploy.deployZk rel).parse))
    (hD : (Sess T).DummyReg) (hc : (Sess T).Causal) (hsc : (Sess T).SwCausal vs) (hT : (Sess T).TapeOK)
    (L₀ L : Release T) (hL : L = Claims.ZKServer.LeanRelease.loop Claims.ZKServer.Loop.loopNr T L₀)
    (wk : L.toSrv.Worker) (pc : PC T) (fc : FC T) :
    Claims.ZKOut (fun _ => Tape T) (fun _ => L.toSrv.Chk) sha (rounds rel)
      (fun w _ => L.toSrv.proverD L.fillOf (rounds rel) wk w) (LiveRel rel T vs L wk pc fc)
      (Claims.ZKServer.outD L.toSrv.stop (rounds rel)) (simOut rel T L pc fc t) (errOut T vs L pc fc t δ₁) :=
  server_zkOutAt (FlockDeploy.deployZk rel) rfl (fun _ _ hR => Claims.ZKMal.V_k_of_R hR) T vs t ht hT1 hδ₁ hS hV hD
    hc hsc hT L₀ L hL wk pc fc

end Claims.SamplingGeneric.FlockZk
