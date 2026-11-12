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

/-!
# The worker's `τ` commitment

`Loop.TakesTau` is a worker premise of `TakesVals`'s and `TakesFinalC`'s kind: at the inner proof's round 2 (the frame's
state counts that round, and the worker's round is `τ`'s 64-byte commitment between its frame's header and `β`'s squeeze,
the only shape `Frame.innerOwn` takes there), the commitment is `vs`'s (`Loop.vsTau`, the switched source's at the rep
stream's table and rep), at the coins the openings open and the tape. Where the rows hold that commitment is the hiding
leaf of the worker's view's `τ` under the rep's salt (`Masked.swSrcE_tauCom`).

`Loop.takesTau_valsWk`: `Loop.valsWk vs` meets it at every `vs` and every release whose `Finish` comes after the scheduled
rounds, because `valsDev` sets round 2's commitment to `vs`'s (`Loop.setTau`); at the honest values at the session loop
among them (`Loop.takesTau_honestVals_ofLoopLink`), beside `TakesVals` and `TakesFinalC` at the same worker.
-/

@[expose] public section

namespace Claims.ZKServer

open Claims Claims.ZKMal
open Claims.ZKMal.CoinLayer Claims.ZKMal.Masked
open _root_.Flock.Firewall.Frame (Masker)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoopLink`'s session types, as in `Statement`

namespace Loop

/-- **The worker's `τ` commitment is the switched source's**: after every history `LiveRoundEq` reads (`0 < ops.length <
refK x` openings, live, whose openings pass under the tape's nonce), where the frame takes the stream's coins at the
inner proof's round 2 (`answered`, `Masker.inner`) and the worker's round `v` of rep stream `2j + r` is `τ`'s 64-byte
commitment `w'` between its frame's header and `β`'s squeeze, `w'` is `vs`'s `τ` commitment there (`vsTau`), at the
coins the openings open and the tape. -/
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

/-- At the inner proof's round 2 a round of that round's shape comes out of `setTau` with its commitment set. -/
theorem setTau_inner2 {m : Masker} {t w' : ByteArray} {u : List UInt8} (hm : m.inner = some 2) (hw : w'.size = 64)
    (h : setTau m t u = (inner2 w').data.toList) : (inner2 t).data.toList = (inner2 w').data.toList := by
  unfold setTau at h
  split_ifs at h with hc
  · exact h
  · exact absurd ⟨hm, w', hw, h⟩ hc

/-- Round 2's bytes determine its commitment. -/
theorem inner2_inj {t t' : ByteArray} (h : (inner2 t).data.toList = (inner2 t').data.toList) : t = t' := by
  simp only [inner2, ByteArray.toList_data_append, List.append_assoc, List.append_cancel_left_eq,
    List.append_cancel_right_eq] at h
  have e : t.data = t'.data := Array.ext' h
  cases t
  cases t'
  exact congrArg ByteArray.mk e

/-- **`valsWk` takes `vs`'s `τ` commitment**: at every `vs` and every release whose `Finish` comes after the scheduled
rounds (`refK x < L.last x`), `valsWk vs L` meets `TakesTau`: where its round is the inner proof's round 2, `setTau`
wrote `vs`'s commitment there (`vsTauAt`). -/
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

/-- **`TakesTau` holds at the honest values** at the session loop: `valsWk honestVals`, which meets `TakesVals` and
`TakesFinalC` (`takes_honestVals_ofLoopLink`), meets it too. -/
theorem takesTau_honestVals_ofLoopLink (L₀ src real) {x : Msg} {w : Flock.Wit} :
    TakesTau (CoinLayer.session Loop.loopNr Masked.dataE).honestVals (LeanRelease.ofLoopLink L₀ src real)
      (valsWk (CoinLayer.session Loop.loopNr Masked.dataE).honestVals (LeanRelease.ofLoopLink L₀ src real)) x w := by
  have he : (LeanRelease.ofLoopLink L₀ src real).last x = refK x + 1 :=
    LeanRelease.loop_last Loop.loopNr Masked.dataE _ x
  have hlast : refK x < (LeanRelease.ofLoopLink L₀ src real).last x := by omega
  exact takesTau_valsWk _ _ hlast

end Loop

end Claims.ZKServer
