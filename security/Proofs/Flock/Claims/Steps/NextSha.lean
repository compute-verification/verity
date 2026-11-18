module

public import Proofs.Flock.C3.Reps.AtList
public import Proofs.Flock.C3.Reps.RoundLinks
public import Proofs.Flock.C3.VStarList.Accepts
public import Proofs.Flock.Claims.Complete.Json.Record
public import Proofs.Flock.Claims.Complete.Json.UnitSample
public import Proofs.Flock.Claims.Complete.Walk
public import Proofs.Flock.Claims.Complete.ZkTables.Scope
public import Proofs.Flock.Claims.KS.PlainKSLin
public import Proofs.Flock.Claims.KS.ZkOuter
public import Proofs.Flock.Claims.Sampling.Bext
public import Proofs.Flock.Claims.Steps.NextCs
public import Proofs.Flock.Claims.Steps.NextP
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

/-! # What the plain KS proofs read of `Claims.Recursion.NextSha`, without its recursion imports. -/

@[expose] public section

namespace FlockDeploy

open Flock

open Lean (Json)

/-- `x` run against `H` returns `a`. -/
def EvOk {α : Type} (H : Claims.Msg → Claims.Msg) (x : Claims.OCV α) (a : α) : Prop :=
  ((x.run).eval H).1 = .ok a

theorem evOk_pure {α : Type} (H : Claims.Msg → Claims.Msg) (a : α) : EvOk H (pure a : Claims.OCV α) a := rfl

/-- A bind of a run that returns `a` runs as its continuation at `a`. -/
theorem evOk_bind_fst {α β : Type} {H : Claims.Msg → Claims.Msg} {x : Claims.OCV α} {f : α → Claims.OCV β} {a : α}
    (hx : EvOk H x a) : (((x >>= f).run).eval H).1 = (((f a).run).eval H).1 := by
  unfold EvOk at hx
  simp only [ExceptT.run_bind, Claims.OC.bind_eq, Claims.OC.eval_bind, hx]

theorem evOk_bind {α β : Type} {H : Claims.Msg → Claims.Msg} {x : Claims.OCV α} {f : α → Claims.OCV β} {a : α} {b : β}
    (hx : EvOk H x a) (hf : EvOk H (f a) b) : EvOk H (x >>= f) b := by
  unfold EvOk at hf ⊢
  rw [evOk_bind_fst hx]
  exact hf

theorem evOk_mapM_list {α β : Type} {H : Claims.Msg → Claims.Msg} {g : α → Claims.OCV β} {G : α → β}
    (hg : ∀ a, EvOk H (g a) (G a)) : ∀ l : List α, EvOk H (l.mapM g) (l.map G)
  | [] => by
    simp only [List.mapM_nil, List.map_nil]
    exact evOk_pure H []
  | a :: l => by
    rw [List.mapM_cons]
    exact evOk_bind (hg a) (evOk_bind (evOk_mapM_list hg l) (evOk_pure H _))

theorem evOk_mapM_array {α β : Type} {H : Claims.Msg → Claims.Msg} {g : α → Claims.OCV β} {G : α → β}
    (hg : ∀ a, EvOk H (g a) (G a)) (xs : Array α) : EvOk H (xs.mapM g) (xs.map G) := by
  rw [Array.mapM_eq_mapM_toList, map_eq_pure_bind]
  refine evOk_bind (evOk_mapM_list hg xs.toList) ?_
  have e : (xs.toList.map G).toArray = xs.map G := by rw [← Array.toList_map, Array.toArray_toList]
  rw [← e]
  exact evOk_pure H _

/-- The record digest through the oracle returns `dgOf` at `H`. -/
theorem evOk_dg (H : Claims.Msg → Claims.Msg) (spec : SessionSpec) (b : ByteArray) :
    EvOk H (spec.dgWith Claims.Hashes.oc b) (FlockComplete.dgOf spec (Claims.hOf H) b) := by
  unfold EvOk SessionSpec.dgWith FlockComplete.dgOf
  cases spec.sha512Digests <;> rfl

/-- `proof_{dk}` at the digests `ds`: rep stream `w`'s entry `{"stream": w, dk: hex d}`, in `spec.repStreams`' order. -/
def shaArr (spec : SessionSpec) (ds : Array ByteArray) : Json :=
  .arr ((spec.repStreams.zip ds).map fun q =>
    Json.mkObj [("stream", Json.str q.1.2.2), (spec.dk, FlockComplete.hexJ q.2)])

/-- **The record's value with its proof digests `ds`'s**: `proof_{dk}`'s value replaced in place by `shaArr` (`mapKey`,
so a value without the key keeps none). -/
def shaSubJ (spec : SessionSpec) (ds : Array ByteArray) (j : Json) : Json :=
  mapKey s!"proof_{spec.dk}" (fun _ => shaArr spec ds) j

/-- **The decision with the record's proof digests the files'**: the files' digests through the oracle, then
`decisionAtP` at the record's value with `proof_{dk}` written from them. -/
def decisionAtS (s : Ins) (spec : SessionSpec) (sts : Array Stmt) (x : Claims.Msg)
    (link : List ByteArray × List ByteArray) (c₁ : Coin) (mids : List (Claims.Msg × Coin)) (f : FinalP) :
    Claims.OCV Unit := do
  let ds ← (filesAt (tabsOf sts) (pointsOf spec c₁) f.ps).mapM (spec.dgWith Claims.Hashes.oc)
  decisionAtP s spec sts x link c₁ mids ⟨f.stmt, shaSubJ spec ds f.j, f.ps⟩

/-- The decision with the proof digests substituted: the statement's setup over `OC`, then `decisionAtS`. -/
def decisionSF (s : Ins) (x : Claims.Msg) (link : List ByteArray × List ByteArray) (c₁ : Coin)
    (mids : List (Claims.Msg × Coin)) (f : FinalP) : Claims.OCV Unit := do
  let (spec, sts) ← setupM Claims.Hashes.oc s.read
  decisionAtS s spec sts x link c₁ mids f

end FlockDeploy
