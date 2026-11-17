module

public import Proofs.Flock.SessionLoop.TakenVals
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
import Proofs.Flock.Claims.ZK.FramePass

/-! `Loop.TakesTau`, the worker premise that its `τ` commitment is `vs`'s, met by `Loop.valsWk` at every `vs`. -/

@[expose] public section

namespace Claims.ZKServer

open Claims Claims.ZKMal
open Claims.ZKMal.CoinLayer Claims.ZKMal.Masked
open _root_.Flock.ZkServer.Frame (Masker)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoopLink`'s session types, as in `Statement`

namespace Loop

def TakesTau (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (wk : L.toSrv.Worker) (x : Msg) (w : Flock.Wit) : Prop :=
  ∀ (ν : (CoinLayer.session Loop.loopNr Masked.dataE).Nonce) (t : (CoinLayer.session Loop.loopNr Masked.dataE).Tape)
    (κ : L.toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) (v : Msg) (m : Masker) (w' : ByteArray)
    (j : Fin (jOf x)) (r : Fin 2),
    0 < ops.length → ops.length < refK x → L.toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true →
    ((allOk ((CoinLayer.session Loop.loopNr Masked.dataE).refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval
      CoinOpen.shaH).1 = true →
    wk x w (ν, t) ((m₀, ρ₀) :: ops) 0 = some (v, false) →
    L.answered x (ν, t) ((m₀, ρ₀) :: ops) = .ok m → m.inner = some 2 →
    w'.size = 64 →
    v = (_root_.Flock.Transcript.header 5 0 64 ++ w' ++ _root_.Flock.Transcript.header 4 1 1).data.toList →
    L.rep x (ops.length + 1) = 2 * j.val + r.val →
    w' = vsTau vs x w ν ((CoinLayer.session Loop.loopNr Masked.dataE).pad
      (ops.map fun p => (CoinLayer.session Loop.loopNr Masked.dataE).decOf p.2)) t j r

theorem setTau_inner2 {m : Masker} {t w' : ByteArray} {u : List UInt8} (hm : m.inner = some 2) (hw : w'.size = 64)
    (h : setTau m t u = (inner2 w').data.toList) : (inner2 t).data.toList = (inner2 w').data.toList := by
  unfold setTau at h
  split_ifs at h with hc
  · exact h
  · exact absurd ⟨hm, w', hw, h⟩ hc

theorem inner2_inj {t t' : ByteArray} (h : (inner2 t).data.toList = (inner2 t').data.toList) : t = t' := by
  simp only [inner2, ByteArray.toList_data_append, List.append_assoc, List.append_cancel_left_eq,
    List.append_cancel_right_eq] at h
  have e : t.data = t'.data := Array.ext' h
  cases t
  cases t'
  exact congrArg ByteArray.mk e

theorem takesTau_valsWk (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    {x : Msg} {w : Flock.Wit} (hlast : refK x < L.last x) : TakesTau vs L (valsWk vs L) x w := by
  intro ν t κ m₀ ρ₀ ops v m w' j r hpos hlt hl _ hv hm hi hw' hshape hrep
  have hkd : L.kindAt x ((m₀, ρ₀) :: ops).length = .masked :=
    L.kindAt_masked_of (by simp only [List.length_cons]; omega) (by simp only [List.length_cons]; omega)
  have hv' := hv
  rw [valsWk_masked vs L hkd] at hv'
  obtain ⟨m', out', s', hm', hc⟩ := LeanRelease.check_of_live L _ x w (ν, t) κ _ v hl hkd hv
  rw [hm] at hm'
  obtain rfl := Except.ok.inj hm'
  obtain ⟨qr, hqr, -, -⟩ := passes_none (check_none_passes hc)
  obtain ⟨o₁, s₁, frr⟩ := qr
  have hvd : v = valsDev vs L x w (ν, t) ((m₀, ρ₀) :: ops)
      (computedOf (L.computed x w (ν, t) ((m₀, ρ₀) :: ops))) := (congrArg Prod.fst (Option.some.inj hv')).symm
  unfold valsDev at hvd
  rw [hm] at hvd
  simp only at hvd
  rw [hqr] at hvd
  simp only at hvd
  have hs2 : v = (inner2 w').data.toList := hshape
  split at hvd
  · exact (inner2_inj (setTau_inner2 hi hw' (hvd.symm.trans hs2)).symm).trans
      (vsTauAt_eq vs L x w (ν, t) ((m₀, ρ₀) :: ops) hrep)
  · have e := congrArg List.length (hvd.symm.trans hshape)
    rw [ByteArray.toList_data_append, List.length_append, FlockSoundness.Refine.Frame.header_length,
      List.length_nil] at e
    omega

theorem takesTau_honestVals_ofLoopLink (L₀ src real) {x : Msg} {w : Flock.Wit} :
    TakesTau (CoinLayer.session Loop.loopNr Masked.dataE).honestVals (LeanRelease.ofLoopLink L₀ src real)
      (valsWk (CoinLayer.session Loop.loopNr Masked.dataE).honestVals (LeanRelease.ofLoopLink L₀ src real)) x w := by
  have he : (LeanRelease.ofLoopLink L₀ src real).last x = refK x + 1 :=
    LeanRelease.loop_last Loop.loopNr Masked.dataE _ x
  have hlast : refK x < (LeanRelease.ofLoopLink L₀ src real).last x := by omega
  exact takesTau_valsWk _ _ hlast

end Loop

end Claims.ZKServer
