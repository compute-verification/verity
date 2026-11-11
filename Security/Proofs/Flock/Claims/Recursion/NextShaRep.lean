module

public import Proofs.Flock.Claims.Recursion.NextP
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
import Proofs.Flock.Claims.Recursion.NextSha

/-!
# A rep's verdict at the proofs' files (`verifyRepWith_filesAt`)

At the proofs' files (`filesAt`), `verifyRepWith` is `verifyRepPWith` at the rep's decoded proof, in any verdict monad
whose lift of a returning `V` computation returns (`liftV_ok_ocv` for `OCV`): the rep's stream (S11) and the statement's
extra claims as `verifyRepWith_split` reads them, the file's shape is `shapeAt` (`repFileShape_shapeAt`), and a shaped
proof's file decodes to it (`decode_filesAt`). `verifyRepWith_filesAt_oc` is the oracle monad's case.
-/

@[expose] public section

namespace FlockDeploy

open Flock

/-- **`verifyRepWith` at the proofs' files is `verifyRepPWith` at the decoded proof**, rep `r` of table `i`, at the
rep's stream and the statement's extra claims at the session's link points. -/
theorem verifyRepWith_filesAt {M : Type → Type} [Monad M] [LawfulMonad M] [MonadExceptOf String M]
    [MonadLiftT Flock.V M] (hlift : ∀ {α : Type} (a : α), (monadLift (Except.ok a : Flock.V α) : M α) = pure a)
    (hs : Flock.Hashes M) (spec : SessionSpec) (sess : RunRecord) {tabs : Array Flock.Setup}
    {pts : Array (Array F128)} {ps : Array Flock.Proof} {i r : ℕ} (hr : r < 2) (hk : 2 * i + r < ps.size)
    {sch : Flock.Schedule} (hsc : fast100 tabs[i]!.m = some sch) (hpts : sess.points = pts)
    {extras : Array Flock.Claim} (he : tabs[i]!.extra pts = .ok extras) {stream : StreamRec}
    (hst : sess.streams.find? (·.name == s!"{spec.tables[i]!}/rep{r}") = some stream)
    (hsh : FlockVBridge.Family.ProofShaped (shapeAt tabs pts (2 * i + r)) ps[2 * i + r]) :
    verifyRepWith hs spec tabs[i]! sch sess i r (filesAt tabs pts ps)[2 * i + r]! =
      verifyRepPWith hs spec tabs[i]! sch sess i r stream extras ps[2 * i + r] := by
  have hk' : 2 * i + r < (filesAt tabs pts ps).size := by simpa only [filesAt, Array.size_mapIdx] using hk
  rw [verifyRepWith_split, getElem!_pos (filesAt tabs pts ps) (2 * i + r) hk']
  simp only [hst, hpts, he, hlift, pure_bind, repFileShape_shapeAt hr hsc he, decode_filesAt hk hsh]

/-- In the oracle monad, a returning `V` computation lifts to `pure`. -/
theorem liftV_ok_ocv {α : Type} (a : α) : (monadLift (Except.ok a : Flock.V α) : Claims.OCV α) = pure a := rfl

/-- **`verifyRepWith_filesAt` in the oracle monad.** -/
theorem verifyRepWith_filesAt_oc (hs : Flock.Hashes Claims.OCV) (spec : SessionSpec) (sess : RunRecord)
    {tabs : Array Flock.Setup} {pts : Array (Array F128)} {ps : Array Flock.Proof} {i r : ℕ} (hr : r < 2)
    (hk : 2 * i + r < ps.size) {sch : Flock.Schedule} (hsc : fast100 tabs[i]!.m = some sch)
    (hpts : sess.points = pts) {extras : Array Flock.Claim} (he : tabs[i]!.extra pts = .ok extras)
    {stream : StreamRec} (hst : sess.streams.find? (·.name == s!"{spec.tables[i]!}/rep{r}") = some stream)
    (hsh : FlockVBridge.Family.ProofShaped (shapeAt tabs pts (2 * i + r)) ps[2 * i + r]) :
    verifyRepWith hs spec tabs[i]! sch sess i r (filesAt tabs pts ps)[2 * i + r]! =
      verifyRepPWith hs spec tabs[i]! sch sess i r stream extras ps[2 * i + r] :=
  verifyRepWith_filesAt (fun a => liftV_ok_ocv a) hs spec sess hr hk hsc hpts he hst hsh

end FlockDeploy
