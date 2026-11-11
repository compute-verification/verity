module

public import Proofs.Flock.Claims.Recursion.CallsAgree
public import Proofs.Flock.Claims.Recursion.CallsBridge
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
# The deployed plain verifier binds its last message's calls, up to the decoder's slack

`RecDeploy.PlainCallsBindOrSlack` holds at SHA-512 (`RecDeploy.plainCallsBindOrSlack_sha512`). Where a proof file
differs, `RecDeploy.plainCallsBindOrSlack_of_proofs` charges it to a collision among the calls or to `ProofSlack`.
Where the files agree, custody pins both records' rounds on every rep stream (`FlockDeploy.KS.sess_rounds_eq`), and
two accepted plain decisions with the same files and rep rounds make one set of calls
(`FlockDeploy.KS.plain_calls_agree`): that is `RecDeploy.plainCallsOfProofs_sha512`. So (P4) at the deployed bound
record, at SHA-512, rests on the decoder's canonicity alone (`RecDeploy.lastBindsEC_deployed_sha512_of_canonical`).
-/

@[expose] public section

namespace RecDeploy

open Claims (Msg)

/-- **The calls half where the proof files agree, at SHA-512.** -/
theorem plainCallsOfProofs_sha512 : PlainCallsOfProofs FlockDeploy.sha512 := by
  intro x ms l l' cs f f' _ hc _ hacc hacc' hfl hfl' hpf
  obtain ⟨s, spec, sts, link, c₁, mids, hact, hact'⟩ := FlockDeploy.KS.plain_decide_shared hc hacc hacc' hfl hfl'
  refine FlockDeploy.KS.plain_calls_agree hact hact' hacc hacc' hpf fun sess sess' hd hd' q hq y y' hy hy' => ?_
  obtain ⟨hV, hck⟩ := FlockDeploy.KS.acc_verifyLogged hact hacc
  obtain ⟨hV', hck'⟩ := FlockDeploy.KS.acc_verifyLogged hact' hacc'
  obtain ⟨-, lv, hlv, hcu⟩ := Claims.Bridge.Custody.checks_ok hck
  obtain ⟨-, lv', hlv', hcu'⟩ := Claims.Bridge.Custody.checks_ok hck'
  have hv : Flock.verifyWith Flock.Sha512.hash spec (FlockDeploy.tabsOf sts) f.record f.proofs = .ok () :=
    (Flock.verifyLogged_fst _ _ _ _ _ _).symm.trans (congrArg Prod.fst hV)
  have hv' : Flock.verifyWith Flock.Sha512.hash spec (FlockDeploy.tabsOf sts) f'.record f'.proofs = .ok () :=
    (Flock.verifyLogged_fst _ _ _ _ _ _).symm.trans (congrArg Prod.fst hV')
  obtain ⟨z, z', hz, hz', hzz, -⟩ := FlockDeploy.KS.sess_rounds_eq _ _ hv hlv hv' hlv' hcu hcu' hd hd' hq
  rw [hy] at hz
  rw [hy'] at hz'
  cases hz
  cases hz'
  exact hzz

/-- **The calls half at SHA-512.** -/
theorem plainCallsBindOrSlack_sha512 : PlainCallsBindOrSlack FlockDeploy.sha512 :=
  plainCallsBindOrSlack_of_proofs plainCallsOfProofs_sha512

/-- **`PlainLastBinds` at SHA-512 from the decoder's canonicity alone.** -/
theorem plainLastBinds_sha512_of_canonical (hC : CanonicalProofFiles) : PlainLastBinds FlockDeploy.sha512 :=
  plainLastBinds_sha512_of_calls hC plainCallsBindOrSlack_sha512

/-- **(P4) modulo what `E*` reads at the deployed bound record, at SHA-512, from the decoder's canonicity alone.** -/
theorem lastBindsEC_deployed_sha512_of_canonical
    (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (wE : Msg → ℝ) (join : List Msg → Msg)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) (hC : CanonicalProofFiles) :
    (bdDeployed rel (FlockDeploy.V false).next (FlockDeploy.deployPlainTrim rel).cols wE join rowInDeployed final
      w).LastBindsEC FlockDeploy.sha512 :=
  lastBindsEC_deployed_sha512 rel wE join final w hC plainCallsBindOrSlack_sha512

end RecDeploy
