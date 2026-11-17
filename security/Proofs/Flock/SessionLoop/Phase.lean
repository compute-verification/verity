module

public import Proofs.Flock.SessionLoop.Statement
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

/-! The worker by phase: `SessionLoopZKDeployed`'s premises on the worker read it only in the algebraic phase
(`Loop.liveRoundEq_phased`, `Loop.fillRoundEq_phased`). -/

@[expose] public section

namespace Claims.ZKServer.ZkServer

theorem copy_congr {W R : Type} (S : ZkServer W R) {wk wk' : S.Worker} {x : Msg} {w : W} {ρ : R}
    {h : List (Msg × Msg)} (he : ∀ h' : List (Msg × Msg), h'.length ≤ h.length → wk x w ρ h' = wk' x w ρ h') :
    S.copy wk x w ρ h = S.copy wk' x w ρ h := by
  funext i s
  simp only [copy, he (h.take i) (List.length_take_le' _ _)]

theorem pass_congr {W R : Type} (S : ZkServer W R) {wk wk' : S.Worker} {x : Msg} {w : W} {ρ : R} (κ : S.Chk)
    {h : List (Msg × Msg)} (he : ∀ h' : List (Msg × Msg), h'.length ≤ h.length → wk x w ρ h' = wk' x w ρ h') :
    S.pass wk x w ρ κ h = S.pass wk' x w ρ κ h := by
  simp only [pass, S.copy_congr he, he h le_rfl]

theorem live_congr {W R : Type} (S : ZkServer W R) {wk wk' : S.Worker} {x : Msg} {w : W} {ρ : R} (κ : S.Chk)
    {h : List (Msg × Msg)} (he : ∀ h' : List (Msg × Msg), h'.length ≤ h.length → wk x w ρ h' = wk' x w ρ h') :
    S.live wk x w ρ κ h = S.live wk' x w ρ κ h := by
  unfold live
  congr 1
  funext i
  exact S.pass_congr κ fun h' hl => he h' (hl.trans (List.length_take_le' _ _))

end Claims.ZKServer.ZkServer

namespace Claims.ZKServer

open Claims Claims.ZKMal
open FlockSoundness.Discharge

namespace LeanRelease

theorem liveRoundEq_congr (S : RefSession) (vs : S.SwVals) (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand))
    (H : Msg → Msg) {wk wk' : L.toSrv.Worker} (x : Msg) (w : Flock.Wit)
    (he : ∀ ρ (h : List (Msg × Msg)), h.length ≤ refK x → wk x w ρ h = wk' x w ρ h) :
    LiveRoundEq S vs L H wk x w ↔ LiveRoundEq S vs L H wk' x w := by
  have key : ∀ {wk wk' : L.toSrv.Worker}, (∀ ρ (h : List (Msg × Msg)), h.length ≤ refK x → wk x w ρ h = wk' x w ρ h) →
      LiveRoundEq S vs L H wk x w → LiveRoundEq S vs L H wk' x w := by
    intro wk wk' he hE ν t κ m₀ ρ₀ ops v h0 hlt hl hok hr hv
    have hle : ((m₀, ρ₀) :: ops).length ≤ refK x := by
      simp only [List.length_cons]
      omega
    refine hE ν t κ m₀ ρ₀ ops v h0 hlt ?_ hok hr ?_
    · rwa [L.toSrv.live_congr κ fun h' hl' => he _ h' (hl'.trans hle)]
    · rwa [he _ _ hle]
  exact ⟨key he, key fun ρ h hl => (he ρ h hl).symm⟩

theorem fillRoundEq_congr (S : RefSession) (vs : S.SwVals) (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand))
    (H : Msg → Msg) {wk wk' : L.toSrv.Worker} (x : Msg) (w : Flock.Wit)
    (he : ∀ ρ (h : List (Msg × Msg)), h.length ≤ refK x → wk x w ρ h = wk' x w ρ h) :
    FillRoundEq S vs L H wk x w ↔ FillRoundEq S vs L H wk' x w := by
  have key : ∀ {wk wk' : L.toSrv.Worker}, (∀ ρ (h : List (Msg × Msg)), h.length ≤ refK x → wk x w ρ h = wk' x w ρ h) →
      FillRoundEq S vs L H wk x w → FillRoundEq S vs L H wk' x w := by
    intro wk wk' he hE ν t κ m₀ ρ₀ ops h0 hlt hl hok hr
    have hle : ((m₀, ρ₀) :: ops).length ≤ refK x := by
      simp only [List.length_cons]
      omega
    refine hE ν t κ m₀ ρ₀ ops h0 hlt ?_ hok hr
    rwa [L.toSrv.live_congr κ fun h' hl' => he _ h' (hl'.trans hle)]
  exact ⟨key he, key fun ρ h hl => (he ρ h hl).symm⟩

end LeanRelease

namespace Loop

noncomputable def seenOfExe (x : Msg) (e : Fin Claims.Flock.zkTableCap → Masked.ExeSeen) :
    ∀ j, ZkView.Seen ((CoinLayer.session loopNr Masked.dataE).C x j).z := fun j =>
  (Masked.cut (Masked.fits_zkTab Masked.PE (Masked.zkTab x j)).nMask (e (Fin.castLE (Masked.jOf_le_cap x) j)).1,
    Masked.readLanes (Masked.fits_zkTab Masked.PE (Masked.zkTab x j)) (e (Fin.castLE (Masked.jOf_le_cap x) j)).2.1,
    Masked.cut (Nat.add_le_add (Masked.fits_zkTab Masked.PE (Masked.zkTab x j)).nS
      (Masked.fits_zkTab Masked.PE (Masked.zkTab x j)).nU) (e (Fin.castLE (Masked.jOf_le_cap x) j)).2.2)

theorem seenOf_exe (x : Msg) (ρ : (CoinLayer.session loopNr Masked.dataE).refP.Nonce ×
    (CoinLayer.session loopNr Masked.dataE).refP.Rand) :
    (CoinLayer.session loopNr Masked.dataE).seenOf x ρ.2 = seenOfExe x (Masked.seenOf ρ).2 :=
  rfl

variable (L : LeanRelease Flock.Wit
  ((CoinLayer.session loopNr Masked.dataE).refP.Nonce × (CoinLayer.session loopNr Masked.dataE).refP.Rand))

abbrev AlgWorker : Type :=
  (x : Msg) → Flock.Wit → (CoinLayer.session loopNr Masked.dataE).Nonce →
    (∀ j, ZkView.Seen ((CoinLayer.session loopNr Masked.dataE).C x j).z) → Option (List FlockDeploy.Coin) → ℕ →
      Option L.toSrv.Val

noncomputable def bwOf (bx : AlgWorker L) : L.toSrv.BWorker Seen FlockDeploy.Coin :=
  fun x w r cs s => bx x w r.1 (seenOfExe x r.2) cs.tail? s

noncomputable def phased (bx : AlgWorker L) (post : L.toSrv.Worker) : L.toSrv.Worker :=
  fun x w ρ h => if h.length ≤ refK x then blind L (bwOf L bx) x w ρ h else post x w ρ h

theorem phased_alg (bx : AlgWorker L) (post : L.toSrv.Worker) (x : Msg) (w : Flock.Wit) (ρ : _)
    (h : List (Msg × Msg)) (hl : h.length ≤ refK x) : phased L bx post x w ρ h = blind L (bwOf L bx) x w ρ h :=
  ite_eq_left hl

theorem phased_eq (bx : AlgWorker L) (post : L.toSrv.Worker) (x : Msg) (w : Flock.Wit)
    (ν : (CoinLayer.session loopNr Masked.dataE).Nonce) (t : (CoinLayer.session loopNr Masked.dataE).Tape)
    (h : List (Msg × Msg)) (hl : h.length ≤ refK x) (s : ℕ) :
    phased L bx post x w (ν, t) h s =
      bx x w ν ((CoinLayer.session loopNr Masked.dataE).seenOf x t) (h.map fun m => CoinLayer.rd m.2).tail? s := by
  rw [phased_alg L bx post x w _ h hl]
  rfl

def PhaseBlind (wk : L.toSrv.Worker) : Prop :=
  ∀ x w (ρ ρ' : (CoinLayer.session loopNr Masked.dataE).refP.Nonce × (CoinLayer.session loopNr Masked.dataE).refP.Rand)
    (h h' : List (Msg × Msg)) s, h.length ≤ refK x → ρ.1 = ρ'.1 →
    (CoinLayer.session loopNr Masked.dataE).seenOf x ρ.2 = (CoinLayer.session loopNr Masked.dataE).seenOf x ρ'.2 →
    (h.map fun m => CoinLayer.rd m.2).tail? = (h'.map fun m => CoinLayer.rd m.2).tail? → wk x w ρ h s = wk x w ρ' h' s

theorem length_eq_of_tail? {α : Type} {l l' : List α} (h : l.tail? = l'.tail?) : l.length = l'.length := by
  cases l <;> cases l' <;> simp_all

theorem phaseBlind_phased (bx : AlgWorker L) (post : L.toSrv.Worker) : PhaseBlind L (phased L bx post) := by
  intro x w ρ ρ' h h' s hl h1 hs hh
  have hl' : h'.length ≤ refK x := by
    have := length_eq_of_tail? hh
    simp only [List.length_map] at this
    omega
  refine (phased_eq L bx post x w ρ.1 ρ.2 h hl s).trans (Eq.trans ?_ (phased_eq L bx post x w ρ'.1 ρ'.2 h' hl' s).symm)
  rw [h1, hs, hh]

noncomputable def coinsHist (oc : Option (List FlockDeploy.Coin)) : List (Msg × Msg) :=
  match oc with
  | none => []
  | some cs => (([] : Msg), ([] : Msg)) :: cs.map fun a => (([] : Msg), Function.invFun CoinLayer.rd a)

theorem PhaseBlind.exists_phased {wk : L.toSrv.Worker} (hb : PhaseBlind L wk) :
    ∃ bx : AlgWorker L, phased L bx wk = wk := by
  refine ⟨fun x w ν s oc i => wk x w (ν, Function.invFun ((CoinLayer.session loopNr Masked.dataE).seenOf x) s)
    (coinsHist oc) i, ?_⟩
  funext x w ρ h i
  by_cases hl : h.length ≤ refK x
  · refine (phased_eq L _ wk x w ρ.1 ρ.2 h hl i).trans ?_
    cases h with
    | nil => exact hb x w _ _ _ [] i (Nat.zero_le _) rfl (Function.invFun_eq ⟨ρ.2, rfl⟩) rfl
    | cons e t =>
      refine hb x w _ _ _ (e :: t) i ?_ rfl (Function.invFun_eq ⟨ρ.2, rfl⟩) ?_
      · simpa only [coinsHist, List.map_cons, List.tail?_cons, List.length_cons, List.length_map] using hl
      · simp only [coinsHist, List.map_cons, List.tail?_cons, List.map_map, Function.comp_def, Option.some.injEq]
        exact List.map_congr_left fun m _ => Function.invFun_eq ⟨m.2, rfl⟩
  · exact congrFun (ite_eq_right hl) i

theorem liveRoundEq_phased (vs : (CoinLayer.session loopNr Masked.dataE).SwVals) (H : Msg → Msg) (bx : AlgWorker L)
    (post : L.toSrv.Worker) (x : Msg) (w : Flock.Wit) :
    LeanRelease.LiveRoundEq (CoinLayer.session loopNr Masked.dataE) vs L H (phased L bx post) x w ↔
      LeanRelease.LiveRoundEq (CoinLayer.session loopNr Masked.dataE) vs L H (blind L (bwOf L bx)) x w :=
  LeanRelease.liveRoundEq_congr _ vs L H x w fun ρ h hl => phased_alg L bx post x w ρ h hl

theorem fillRoundEq_phased (vs : (CoinLayer.session loopNr Masked.dataE).SwVals) (H : Msg → Msg) (bx : AlgWorker L)
    (post : L.toSrv.Worker) (x : Msg) (w : Flock.Wit) :
    LeanRelease.FillRoundEq (CoinLayer.session loopNr Masked.dataE) vs L H (phased L bx post) x w ↔
      LeanRelease.FillRoundEq (CoinLayer.session loopNr Masked.dataE) vs L H (blind L (bwOf L bx)) x w :=
  LeanRelease.fillRoundEq_congr _ vs L H x w fun ρ h hl => phased_alg L bx post x w ρ h hl

end Loop

end Claims.ZKServer
