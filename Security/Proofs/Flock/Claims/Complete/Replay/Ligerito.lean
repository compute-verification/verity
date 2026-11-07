module

public import Proofs.Flock.Claims.Complete.Replay.RingSwitch
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
# Ligerito replays the honest prover's rounds

`Flock.Ligerito`, `Flock.Merkle` at the honest run: the verifier's Ligerito (`ligeritoWith`) on the rounds the session recorded.
-/

@[expose] public section

namespace FlockComplete.Replay

open Flock Transcript

/-! ## Where the ring switch leaves the rep -/

/-- Table `i`'s rep `r`'s transcript after the prover's ring switch. -/
def _root_.FlockComplete.Run.rsEnd {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}
    (o : Run H rel) (i r : ℕ) : Transcript :=
  Rs.tEnd (o.trace i) (o.claims i r) (o.rsCoins i r) (o.lcEnd i r)

theorem _root_.FlockComplete.Run.rsEnd_k {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}
    (o : Run H rel) (i r : ℕ) : (o.rsEnd i r).k = (o.rsEnd i r).rounds.size := by
  simp only [Run.rsEnd, Rs.tEnd_k, Rs.tEnd_size, o.lcEnd_k]

/-- **The ring switch's value at a run**: the prover's `T` and switched claims, and the transcript where the prover's
ring switch stopped. -/
theorem _root_.FlockComplete.Run.sw_eq {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}
    (hR1 : TraceR1CS H rel) (o : Run H rel) {i : ℕ} (hi : i < o.tabs.size) {r : ℕ} (hr : r < 2) (hzc : ZcAt o i r)
    (hlc : LcAt o i r) (hal : LcAlgebra H rel) (hrs : RsAt o i r) (hra : RsAlgebra H rel) :
    o.sw i r = ((Rs.T (o.trace i) (o.claims i r) (o.rsCoins i r), Rs.sws (o.claims i r) (o.rsCoins i r)),
      ⟨(o.stream i r).rounds, (o.rsEnd i r).k, (o.rsEnd i r).pending⟩) := by
  obtain ⟨hp, hag⟩ := hrs
  have h := ringSwitchWith_of o.hs o.tabs[i]!.m (o.trace i) (o.claims i r) (o.rsCoins i r) (o.lcEnd i r)
    (o.lcEnd_k i r) (o.claims_size_ne i r) (hag.mono (Nat.zero_le _)) (hra o i hi r hr)
  unfold Run.sw
  rw [o.lc_eq hR1 hi hr hzc hlc hal, o.zc_eq hR1 hi hr hzc, hp]
  exact congrArg val h

end FlockComplete.Replay
