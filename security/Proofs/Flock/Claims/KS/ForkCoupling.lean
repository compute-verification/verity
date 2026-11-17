module

public import Proofs.Flock.Claims.KS.ForkSlack
public import Proofs.Flock.Claims.KS.FnFork
public import Proofs.Flock.Claims.KS.LiveMix
public import Proofs.Flock.Claims.DeployVK
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
import Proofs.Flock.Claims.KS.ConflictTied

/-! The fork's coin coupling at a simulator of uniform functions, in `ForkSlack`'s closure and at the deployed plain verifier. -/

@[expose] public section

namespace Claims.Flock

open FnFork

theorem forkCoinCoupling_fnAt {M : ℕ} {A D : Type} [Fintype A] [DecidableEq A] [Nonempty A]
    [hI : Inhabited (Fin M → A)] [Fintype D] [DecidableEq D] {Ω : Type} (H : Msg → Msg) (V : Verifier (Fin M → A))
    (P : Prover Ω) (x : Msg) (ω : Ω) {L : ℕ} (h : L = V.k x) (n : Fin (V.k x) → ℕ)
    {key : Fin (V.k x) → (Fin (V.k x) → Fin M → A) → D} (hk : PrefixKey n key) :
    ForkCoinCoupling H V P x (δN A M n)
      (simForkPr H V P x (fun q => fnAt h n key q.1 q.2) (fun t => forkN t) ω) ω := by
  obtain ⟨hstep, hlast⟩ := fnAt_coupling h hk (pairEv H V P x ω) (oneEv H V P x ω)
    fun s hs => ⟨hs.1, FlockDeploy.KS.tiedColl_of_append_self hs.2.2⟩
  refine ⟨fun t ht => ?_, ?_⟩
  · unfold forkPrAt
    rw [dite_eq_left_of_eq_true (eq_true ht)]
    exact hstep t ht
  · unfold forkPrAt
    rw [dite_eq_right_of_eq_false (eq_false (lt_irrefl _))]
    exact hlast

theorem slack_fnAt {M : ℕ} {A D : Type} [Fintype A] [DecidableEq A] [Nonempty A] [Inhabited (Fin M → A)]
    [Fintype D] [DecidableEq D] {Ω : Type} (H : Msg → Msg) (V : Verifier (Fin M → A)) (P : Prover Ω) (x : Msg)
    (ω : Ω) {L : ℕ} (h : L = V.k x) (n : Fin (V.k x) → ℕ) {key : Fin (V.k x) → (Fin (V.k x) → Fin M → A) → D}
    (hk : PrefixKey n key) (j : Fin (V.k x)) :
    simForkPr H V P x (fun q => fnAt h n key q.1 q.2) (fun t => forkN t) ω j ≤
      (Pr fun s : (Fin (V.k x) → Fin M → A) × (Fin (V.k x) → Fin M → A) => forkEv H V P x j ω s.1 s.2) +
        forkSlack H V P x (δN A M n) j ω :=
  slack_of_coupling H V P x (fun t => δN_nonneg («X» := A) (M := M) n t) (forkCoinCoupling_fnAt H V P x ω h n hk) j

end Claims.Flock

namespace FlockDeploy.KS

theorem live_fnAt_eq {K L M : ℕ} {A D : Type} (h : L = K) (n : Fin K → ℕ) (key : Fin K → (Fin K → Fin M → A) → D)
    (f : Fin K → D → Fin M → A) (r : Fin L → Fin M → A) :
    Claims.Bridge.Live.fnAt h n key f r = Claims.Flock.FnFork.fnAt h n key f r := rfl

theorem forkCoinCoupling_plainK (keep : Bool → Bool) {D : Type} [Fintype D] [DecidableEq D] {Ω : Type} (P : Claims.Prover Ω)
    (x : Claims.Msg) (ω : Ω) {L : ℕ} (h : L = (V false).k x) (n : Fin ((V false).k x) → ℕ)
    {key : Fin ((V false).k x) → (Fin ((V false).k x) → Claims.Bridge.Live.LCoin 65536) → D}
    (hk : Claims.Flock.FnFork.PrefixKey n key) :
    Claims.Flock.ForkCoinCoupling sha512 (VK keep) P x (Claims.Flock.FnFork.δN FlockLevel3.GF128 65536 n)
      (Claims.Flock.simForkPr sha512 (VK keep) P x (τ := Fin ((V false).k x) → D → Claims.Bridge.Live.LCoin 65536)
        (LC := Claims.Bridge.Live.LCoin 65536) (fun q => Claims.Bridge.Live.fnAt h n key q.1 q.2)
        (fun t => Claims.Flock.FnFork.forkN t) ω) ω :=
  Claims.Flock.forkCoinCoupling_fnAt (A := FlockLevel3.GF128) (M := 65536) (hI := instInhabitedCoin) sha512
    (VK keep) P x ω h n hk

theorem forkCoinCoupling_plain {D : Type} [Fintype D] [DecidableEq D] {Ω : Type} (P : Claims.Prover Ω)
    (x : Claims.Msg) (ω : Ω) {L : ℕ} (h : L = (V false).k x) (n : Fin ((V false).k x) → ℕ)
    {key : Fin ((V false).k x) → (Fin ((V false).k x) → Claims.Bridge.Live.LCoin 65536) → D}
    (hk : Claims.Flock.FnFork.PrefixKey n key) :
    Claims.Flock.ForkCoinCoupling sha512 (V false) P x (Claims.Flock.FnFork.δN FlockLevel3.GF128 65536 n)
      (Claims.Flock.simForkPr sha512 (V false) P x (τ := Fin ((V false).k x) → D → Claims.Bridge.Live.LCoin 65536)
        (LC := Claims.Bridge.Live.LCoin 65536) (fun q => Claims.Bridge.Live.fnAt h n key q.1 q.2)
        (fun t => Claims.Flock.FnFork.forkN t) ω) ω :=
  forkCoinCoupling_plainK id (D := D) (Ω := Ω) (P := P) (x := x) (ω := ω) (L := L) (h := h) (n := n) (key := key) (hk := hk)

end FlockDeploy.KS
