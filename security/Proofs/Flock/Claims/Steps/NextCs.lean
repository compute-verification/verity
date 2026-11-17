module

public import Proofs.Flock.C3.Reps.AtList
public import Proofs.Flock.C3.Reps.RoundLinks
public import Proofs.Flock.C3.VStarList.Accepts
import Proofs.Flock.Claims.Complete.Json.Decode
public import Proofs.Flock.Claims.Complete.Json.UnitSample
public import Proofs.Flock.Claims.Complete.Walk
public import Proofs.Flock.Claims.Complete.ZkTables.Scope
public import Proofs.Flock.Claims.KS.PlainKSLin
public import Proofs.Flock.Claims.KS.ZkOuter
public import Proofs.Flock.Claims.Sampling.Bext
import all Lean.Data.Json.Basic
import all Lean.Data.Json.FromToJson.Basic
import all Init.Data.Array.Basic
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
import Mathlib.Tactic.ReduceModChar
import Mathlib.Topology.Sheaves.Init

/-! # What the plain KS proofs read of `Claims.Recursion.NextCs`, without its recursion imports. -/

@[expose] public section

namespace FlockDeploy

open Flock

open Lean (Json)

/-- A recorded round as `liveOf` reads it. -/
def roundJ (r : Json) : Flock.V Round := do
  let n ← Record.nat r "n"
  let coins ← Record.f128s (← Record.arr r "coins")
  let bytes ← if Record.isNull r "msg" then pure none else some <$> unhex (← Record.str r "msg")
  pure (⟨n, ByteArray.empty, coins, bytes⟩ : Round)

/-- Rep stream `name`'s rounds as `liveOf` reads them: the first stream of that name's. -/
def streamJ (recs : Array Json) (name : String) : Flock.V (Array Round) := do
  let some s := recs.find? (fun s => (s.getObjValAs? String "stream").toOption == some name)
    | throw s!"S11: no stream {name}"
  (← Record.arr s "rounds").mapM fun r => roundJ r

/-- **`liveOf` at the record's parsed value `j`.** -/
def liveOfJ (spec : SessionSpec) (j : Json) : Flock.V Live := do
  let l ← Record.fld j "link"
  let rootB ← spec.tables.mapM fun t => do unhex (← Record.str (← Record.fld l "root_b") t)
  let pubs ← Record.fld l "publics"
  let publics ← spec.tables.mapM fun t =>
    if Record.isNull pubs t then pure ByteArray.empty else do unhex (← Record.str pubs t)
  let points ← (← Record.arr l "points").mapM fun p => match p.getArr? with
    | .ok a => Record.f128s a
    | .error _ => .error "record: a link point is not a list"
  let recs ← Record.arr j "streams"
  let streams ← spec.repStreams.mapM fun (_, _, name) => streamJ recs name
  pure ⟨rootB, publics, points, streams⟩

/-- `j`'s value at key `k` mapped by `g`, its keys and tree kept; a value that is not an object is kept. -/
def mapKey (k : String) (g : Json → Json) : Json → Json
  | .obj m => .obj (m.map fun k' v => if k' = k then g v else v)
  | j => j

/-- An array's elements mapped by `g`; a value that is not an array is kept. -/
def mapArr (g : Json → Json) : Json → Json
  | .arr a => .arr (a.map g)
  | j => j

/-- An array's elements mapped by `g` with their indices. -/
def mapArrIdx (g : ℕ → Json → Json) : Json → Json
  | .arr a => .arr (a.mapIdx g)
  | j => j

theorem impl_get?_map (f : String → Json → Json) :
    ∀ (t : Std.DTreeMap.Internal.Impl String (fun _ => Json)) (k : String),
      Std.DTreeMap.Internal.Impl.Const.get? (t.map f) k = (Std.DTreeMap.Internal.Impl.Const.get? t k).map (f k)
  | .leaf, _ => rfl
  | .inner _ k' v l r, k => by
    simp only [Std.DTreeMap.Internal.Impl.map, Std.DTreeMap.Internal.Impl.Const.get?]
    split
    · exact impl_get?_map f l k
    · exact impl_get?_map f r k
    · rename_i h
      rw [Std.LawfulEqOrd.eq_of_compare h]
      rfl

/-- **A lookup reads through a value map**, with no hypothesis on the tree. -/
theorem get?_map (f : String → Json → Json) (m : Std.TreeMap.Raw String Json compare) (k : String) :
    (m.map f).get? k = (m.get? k).map (f k) :=
  impl_get?_map f m.inner.inner k

theorem getObjVal?_mapKey_ne {k k' : String} (g : Json → Json) (j : Json) (h : k' ≠ k) :
    (mapKey k g j).getObjVal? k' = j.getObjVal? k' := by
  cases j with
  | obj m =>
    simp only [mapKey, Json.getObjVal?, get?_map]
    cases m.get? k' <;> simp [h]
  | _ => rfl

/-- A stream's name as the record writes it, `""` if none. -/
def nameOf (s : Json) : String := (s.getObjValAs? String "stream").toOption.getD ""

/-- A round's count as `liveOf` reads it, `0` if none. -/
def nOf (r : Json) : ℕ := match Record.nat r "n" with
  | .ok n => n
  | .error _ => 0

/-- Each round of a stream, `g name k round` at its round `k`. -/
def mapStream (g : String → ℕ → Json → Json) (s : Json) : Json := mapKey "rounds" (mapArrIdx (g (nameOf s))) s

/-- Each round of each stream of the record. -/
def mapRounds (g : String → ℕ → Json → Json) : Json → Json := mapKey "streams" (mapArr (mapStream g))

/-- The record with the link's points `v`. -/
def setPoints (v : Json) : Json → Json := mapKey "link" (mapKey "points" fun _ => v)

/-- **The record with its coins erased**: the link's points and every round's coins `null`. -/
def eraseCoins (j : Json) : Json := mapRounds (fun _ _ => mapKey "coins" fun _ => .null) (setPoints .null j)

/-- Field elements as the record writes them, hex strings. -/
def hexArr (a : Array F128) : Json := .arr (a.map FlockComplete.f128J)

/-- Stream `name`'s recorded rounds as `liveOf` finds the stream, `0` if none. -/
def roundsOf (j : Json) (name : String) : ℕ :=
  match (Record.arr j "streams").toOption.bind
      (·.find? fun s => (s.getObjValAs? String "stream").toOption == some name) with
  | some s => ((Record.arr s "rounds").toOption.map (·.size)).getD 0
  | none => 0

/-- Where stream `name`'s rounds start among the round coins: the rounds of the rep streams before it. -/
def streamOffset (spec : SessionSpec) (j : Json) (name : String) : ℕ :=
  (((spec.repStreams.toList.map (·.2.2)).takeWhile (· ≠ name)).map (roundsOf j)).sum

/-- **The record's coins filled from the live coins**: the link's points the first coin's (`pointsOf`), round `k` of
stream `name` the first `n` of its coin at `streamOffset + k`. -/
def fillCoins (spec : SessionSpec) (c₁ : Coin) (mids : List (Claims.Msg × Coin)) (j : Json) : Json :=
  mapRounds (fun name k r => match mids[streamOffset spec j name + k]? with
      | some (_, c) => mapKey "coins" (fun _ => hexArr (c.vals (nOf r))) r
      | none => r)
    (setPoints (.arr ((pointsOf spec c₁).map hexArr)) j)

/-- **The record's parsed value with its coins the live coins**: erased, then filled. -/
def coinSubJ (spec : SessionSpec) (c₁ : Coin) (mids : List (Claims.Msg × Coin)) (j : Json) : Json :=
  fillCoins spec c₁ mids (eraseCoins j)

end FlockDeploy
