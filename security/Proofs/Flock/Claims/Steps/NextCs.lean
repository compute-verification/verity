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

theorem getObjVal?_mapKey_self (k : String) (g : Json → Json) (j : Json) :
    (mapKey k g j).getObjVal? k = g <$> j.getObjVal? k := by
  cases j with
  | obj m =>
    simp only [mapKey, Json.getObjVal?, get?_map]
    cases m.get? k <;> simp <;> rfl
  | _ => rfl

section Custody

open FlockComplete.JsonRT (f128s_map array_mapM_ok_of)

theorem bind_ok {α β : Type} {x : Flock.V α} {f : α → Flock.V β} {b : β} (h : (x >>= f) = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x with
  | error e => cases h
  | ok a => exact ⟨a, rfl, h⟩

theorem fld_mapKey_eq (k : String) (g : Json → Json) (j : Json) :
    Record.fld (mapKey k g j) k = g <$> Record.fld j k := by
  unfold Record.fld
  rw [getObjVal?_mapKey_self]
  cases j.getObjVal? k <;> rfl

theorem fld_mapKey_self {k : String} {g : Json → Json} {j v : Json} (h : Record.fld (mapKey k g j) k = .ok v) :
    ∃ v₀, Record.fld j k = .ok v₀ ∧ v = g v₀ := by
  rw [fld_mapKey_eq] at h
  cases hj : Record.fld j k with
  | error e => rw [hj] at h; cases h
  | ok v₀ => rw [hj] at h; cases h; exact ⟨v₀, rfl, rfl⟩

theorem fld_mapKey_ne {k k' : String} (g : Json → Json) (j : Json) (hk : k' ≠ k) :
    Record.fld (mapKey k g j) k' = Record.fld j k' := by
  unfold Record.fld
  rw [getObjVal?_mapKey_ne g j hk]

theorem arr_mapKey_const {k : String} {a b : Array Json} {j : Json}
    (h : Record.arr (mapKey k (fun _ => .arr a) j) k = .ok b) : b = a := by
  unfold Record.arr at h
  cases hf : Record.fld (mapKey k (fun _ => .arr a) j) k with
  | error e => rw [hf] at h; cases h
  | ok v =>
    obtain ⟨v₀, -, rfl⟩ := fld_mapKey_self hf
    rw [hf] at h
    cases h
    rfl

theorem arr_mapKey_mapArr (k : String) (g : Json → Json) (j : Json) :
    Record.arr (mapKey k (mapArr g) j) k = (·.map g) <$> Record.arr j k := by
  unfold Record.arr
  rw [fld_mapKey_eq]
  cases Record.fld j k with
  | error e => rfl
  | ok v => cases v <;> rfl

theorem arr_mapKey_mapArrIdx (k : String) (g : ℕ → Json → Json) (j : Json) :
    Record.arr (mapKey k (mapArrIdx g) j) k = (·.mapIdx g) <$> Record.arr j k := by
  unfold Record.arr
  rw [fld_mapKey_eq]
  cases Record.fld j k with
  | error e => rfl
  | ok v => cases v <;> rfl

theorem arr_mapKey_ne {k k' : String} (g : Json → Json) (j : Json) (hk : k' ≠ k) :
    Record.arr (mapKey k g j) k' = Record.arr j k' := by
  unfold Record.arr
  rw [fld_mapKey_ne g j hk]

theorem nat_mapKey_ne {k k' : String} (g : Json → Json) (j : Json) (hk : k' ≠ k) :
    Record.nat (mapKey k g j) k' = Record.nat j k' := by
  unfold Record.nat
  rw [fld_mapKey_ne g j hk]

theorem getObjValAs_mapKey_ne {k k' : String} (g : Json → Json) (j : Json) (hk : k' ≠ k) :
    (mapKey k g j).getObjValAs? String k' = j.getObjValAs? String k' := by
  unfold Json.getObjValAs? Json.getObjValD
  rw [getObjVal?_mapKey_ne g j hk]

theorem list_mapM_ok_inv {α β : Type} (f : α → Flock.V β) :
    ∀ (l : List α) (ys : List β), l.mapM f = .ok ys →
      ∃ hl : ys.length = l.length, ∀ (i : ℕ) (h : i < l.length), f l[i] = .ok (ys[i]'(hl ▸ h))
  | [], ys, h => by
    simp only [List.mapM_nil] at h
    cases h
    exact ⟨rfl, fun i h => absurd h (by simp)⟩
  | a :: l, ys, h => by
    rw [List.mapM_cons] at h
    cases ha : f a with
    | error e => rw [ha] at h; cases h
    | ok b =>
      rw [ha] at h
      cases hl : l.mapM f with
      | error e => rw [hl] at h; cases h
      | ok bs =>
        rw [hl] at h
        cases h
        obtain ⟨hlen, hf⟩ := list_mapM_ok_inv f l bs hl
        refine ⟨by simp [hlen], fun i hi => ?_⟩
        cases i with
        | zero => exact ha
        | succ i => exact hf i (by simpa using hi)

theorem array_mapM_ok_inv {α β : Type} (f : α → Flock.V β) (a : Array α) (ys : Array β) (h : a.mapM f = .ok ys) :
    ∃ hl : ys.size = a.size, ∀ (i : ℕ) (h : i < a.size), f a[i] = .ok (ys[i]'(hl ▸ h)) := by
  rw [Array.mapM_eq_mapM_toList] at h
  cases hl : a.toList.mapM f with
  | error e => rw [hl] at h; cases h
  | ok bs =>
    rw [hl] at h
    cases h
    obtain ⟨hlen, hf⟩ := list_mapM_ok_inv f a.toList bs hl
    refine ⟨by simpa using hlen, fun i hi => ?_⟩
    have := hf i (by simpa using hi)
    simp only [Array.getElem_toList] at this
    exact this

theorem flatten_getElem {α : Type} :
    ∀ (L : List (List α)) (t : ℕ) (ht : t < L.flatten.length),
      ∃ (i : ℕ) (hi : i < L.length) (k : ℕ) (hk : k < L[i].length),
        L.flatten[t] = L[i][k] ∧ t = ((L.take i).map List.length).sum + k
  | [], t, ht => by simp at ht
  | s :: L, t, ht => by
    by_cases hs : t < s.length
    · refine ⟨0, by simp, t, hs, ?_, by simp⟩
      simp only [List.flatten_cons]
      rw [List.getElem_append_left hs]
      rfl
    · have ht' : t - s.length < L.flatten.length := by
        simp only [List.flatten_cons, List.length_append] at ht
        omega
      obtain ⟨i, hi, k, hk, he, hsum⟩ := flatten_getElem L (t - s.length) ht'
      refine ⟨i + 1, by simpa using hi, k, by simpa using hk, ?_, ?_⟩
      · simp only [List.flatten_cons]
        rw [List.getElem_append_right (by omega)]
        simpa using he
      · simp only [List.take_succ_cons, List.map_cons, List.sum_cons]
        omega

theorem takeWhile_ne_nodup {α : Type} [DecidableEq α] :
    ∀ (l : List α) (_ : l.Nodup) (i : ℕ) (hi : i < l.length), l.takeWhile (· ≠ l[i]) = l.take i
  | [], _, i, hi => by simp at hi
  | a :: l, hn, i, hi => by
    cases i with
    | zero => simp
    | succ i =>
      have hl : i < l.length := by simpa using hi
      have hne : a ≠ l[i] := by
        intro h
        exact (List.nodup_cons.1 hn).1 (h ▸ List.getElem_mem hl)
      simp only [List.getElem_cons_succ, List.takeWhile_cons, List.take_succ_cons]
      simp only [ne_eq, hne, not_false_eq_true, decide_true]
      rw [takeWhile_ne_nodup l (List.nodup_cons.1 hn).2 i hl]
      simp

/-- The rep-stream test is blind to the rounds. -/
theorem find_mapStream (g : String → ℕ → Json → Json) (recs : Array Json) (name : String) :
    (recs.map (mapStream g)).find? (fun s => (s.getObjValAs? String "stream").toOption == some name) =
      (recs.find? fun s => (s.getObjValAs? String "stream").toOption == some name).map (mapStream g) := by
  rw [Array.find?_map]
  congr 2
  funext s
  simp only [Function.comp, mapStream, getObjValAs_mapKey_ne _ _ (by decide : "stream" ≠ "rounds")]

theorem nameOf_of_find {recs : Array Json} {name : String} {s : Json}
    (h : recs.find? (fun s => (s.getObjValAs? String "stream").toOption == some name) = some s) :
    nameOf s = name := by
  have := Array.find?_some h
  simp only [beq_iff_eq] at this
  simp only [nameOf, this, Option.getD_some]

/-- **A rep stream read at substituted rounds**: the stream the record's test finds, its rounds mapped. -/
theorem streamJ_map {g : String → ℕ → Json → Json} {recs : Array Json} {name : String} {rs : Array Round}
    (h : streamJ (recs.map (mapStream g)) name = .ok rs) :
    ∃ s rs₀, recs.find? (fun s => (s.getObjValAs? String "stream").toOption == some name) = some s ∧
      Record.arr s "rounds" = .ok rs₀ ∧ (rs₀.mapIdx (g name)).mapM (fun r => roundJ r) = .ok rs := by
  unfold streamJ at h
  rw [find_mapStream] at h
  cases hf : recs.find? (fun s => (s.getObjValAs? String "stream").toOption == some name) with
  | none => rw [hf] at h; cases h
  | some s =>
    rw [hf] at h
    obtain ⟨rs', hrs', h⟩ := bind_ok h
    simp only [mapStream] at hrs'
    rw [arr_mapKey_mapArrIdx, nameOf_of_find hf] at hrs'
    cases ha : Record.arr s "rounds" with
    | error e => rw [ha] at hrs'; cases hrs'
    | ok rs₀ =>
      rw [ha] at hrs'
      cases hrs'
      exact ⟨s, rs₀, rfl, ha, h⟩

/-- `roundJ`'s round count and coins are the record's `n` and `coins`. -/
theorem roundJ_ok {r : Json} {rd : Round} (h : roundJ r = .ok rd) :
    Record.nat r "n" = .ok rd.n ∧ (Record.arr r "coins" >>= Record.f128s) = .ok rd.coins := by
  unfold roundJ at h
  obtain ⟨n, hn, h⟩ := bind_ok h
  obtain ⟨cs, hcs, h⟩ := bind_ok h
  obtain ⟨coins, hcoins, h⟩ := bind_ok h
  have hc : (Record.arr r "coins" >>= Record.f128s) = .ok coins := by rw [hcs]; exact hcoins
  split at h
  · cases h
    exact ⟨hn, hc⟩
  · obtain ⟨_, -, h⟩ := bind_ok h
    obtain ⟨_, -, h⟩ := bind_ok h
    cases h
    exact ⟨hn, hc⟩

/-- **A round with its coins replaced reads them** (`f128s_map`). -/
theorem roundJ_coins (c : Coin) (r : Json) {rd : Round}
    (h : roundJ (mapKey "coins" (fun _ => hexArr (c.vals (nOf r))) r) = .ok rd) : rd.coins = c.vals rd.n := by
  obtain ⟨hn, hc⟩ := roundJ_ok h
  rw [nat_mapKey_ne _ _ (by decide)] at hn
  have hnr : nOf r = rd.n := by simp only [nOf, hn]
  rw [hnr] at hc
  obtain ⟨cs, hcs, hc⟩ := bind_ok hc
  obtain rfl := arr_mapKey_const hcs
  rw [f128s_map] at hc
  injection hc with hc
  exact hc.symm

/-- `roundsOf` at a stream `streamJ` reads: its rounds. -/
theorem roundsOf_of {j : Json} {recs : Array Json} {name : String} {s : Json} {rs : Array Json}
    (hr : Record.arr j "streams" = .ok recs)
    (hf : recs.find? (fun s => (s.getObjValAs? String "stream").toOption == some name) = some s)
    (ha : Record.arr s "rounds" = .ok rs) : roundsOf j name = rs.size := by
  unfold roundsOf
  have hro : (Record.arr j "streams").toOption = some recs := by rw [hr]; rfl
  have hsa : (Record.arr s "rounds").toOption = some rs := by rw [ha]; rfl
  rw [hro, Option.bind_some, hf]
  dsimp only
  rw [hsa]
  rfl

/-- **Custody's points at the substituted value are the first coin's.** -/
theorem liveOfJ_points (spec : SessionSpec) (c₁ : Coin) (mids : List (Claims.Msg × Coin)) (j : Json) {lv : Live}
    (h : liveOfJ spec (coinSubJ spec c₁ mids j) = .ok lv) : lv.points = pointsOf spec c₁ := by
  unfold liveOfJ at h
  obtain ⟨l, hl, h⟩ := bind_ok h
  obtain ⟨_, -, h⟩ := bind_ok h
  obtain ⟨_, -, h⟩ := bind_ok h
  obtain ⟨_, -, h⟩ := bind_ok h
  obtain ⟨ps, hps, h⟩ := bind_ok h
  obtain ⟨points, hpoints, h⟩ := bind_ok h
  obtain ⟨_, -, h⟩ := bind_ok h
  obtain ⟨_, -, h⟩ := bind_ok h
  cases h
  show points = _
  simp only [coinSubJ, fillCoins, mapRounds, setPoints] at hl
  rw [fld_mapKey_ne _ _ (by decide)] at hl
  obtain ⟨l₀, -, rfl⟩ := fld_mapKey_self hl
  obtain rfl := arr_mapKey_const hps
  have := array_mapM_ok_of (fun p => match p.getArr? with
    | .ok a => Record.f128s a
    | .error _ => .error "record: a link point is not a list") ((pointsOf spec c₁).map hexArr) (pointsOf spec c₁)
    (by simp) fun i h₁ h₂ => by
      simp only [Array.getElem_map, hexArr, Json.getArr?]
      exact f128s_map _
  rw [this] at hpoints
  cases hpoints
  rfl

/-- **Custody's round coins at the substituted value are the live coins**: at rep streams of distinct names, every
recorded round custody pairs with a round message carries the first `n` of that message's coin. -/
theorem liveOfJ_coins (spec : SessionSpec) (c₁ : Coin) (mids : List (Claims.Msg × Coin)) (j : Json) {lv : Live}
    (hnd : (spec.repStreams.toList.map (·.2.2)).Nodup) (h : liveOfJ spec (coinSubJ spec c₁ mids j) = .ok lv) :
    ∀ (t : ℕ) (h₁ : t < (lv.streams.toList.flatMap fun s : Array Round => s.toList).length) (h₂ : t < mids.length),
      (lv.streams.toList.flatMap fun s : Array Round => s.toList)[t].coins =
        mids[t].2.vals (lv.streams.toList.flatMap fun s : Array Round => s.toList)[t].n := by
  unfold liveOfJ at h
  obtain ⟨_, -, h⟩ := bind_ok h
  obtain ⟨_, -, h⟩ := bind_ok h
  obtain ⟨_, -, h⟩ := bind_ok h
  obtain ⟨_, -, h⟩ := bind_ok h
  obtain ⟨_, -, h⟩ := bind_ok h
  obtain ⟨_, -, h⟩ := bind_ok h
  obtain ⟨recs, hrecs, h⟩ := bind_ok h
  obtain ⟨streams, hstreams, h⟩ := bind_ok h
  cases h
  simp only [coinSubJ, fillCoins, mapRounds, setPoints] at hrecs
  rw [arr_mapKey_mapArr, arr_mapKey_ne _ _ (by decide : "streams" ≠ "link")] at hrecs
  cases hE : Record.arr (eraseCoins j) "streams" with
  | error e => rw [hE] at hrecs; cases hrecs
  | ok aE =>
  rw [hE] at hrecs
  injection hrecs with hrecs
  subst hrecs
  obtain ⟨hsz, hf⟩ := array_mapM_ok_inv _ _ _ hstreams
  have hS : ∀ (i : ℕ) (hi : i < spec.repStreams.size), ∃ rs₁ : Array Json,
      roundsOf (eraseCoins j) spec.repStreams[i].2.2 = rs₁.size ∧
      (rs₁.mapIdx fun k r => match mids[streamOffset spec (eraseCoins j) spec.repStreams[i].2.2 + k]? with
          | some (_, c) => mapKey "coins" (fun _ => hexArr (c.vals (nOf r))) r
          | none => r).mapM (fun r => roundJ r) = .ok (streams[i]'(hsz ▸ hi)) := by
    intro i hi
    obtain ⟨s₁, rs₁, hfind, ha, hm⟩ := streamJ_map (hf i hi)
    exact ⟨rs₁, roundsOf_of hE hfind ha, hm⟩
  have hsize : ∀ (i : ℕ) (hi : i < spec.repStreams.size),
      roundsOf (eraseCoins j) spec.repStreams[i].2.2 = (streams[i]'(hsz ▸ hi)).size := by
    intro i hi
    obtain ⟨rs₁, hr, hm⟩ := hS i hi
    obtain ⟨hl, -⟩ := array_mapM_ok_inv _ _ _ hm
    rw [hr, hl, Array.size_mapIdx]
  intro t h₁ h₂
  have hfl : (streams.toList.flatMap fun s : Array Round => s.toList) =
      (streams.toList.map fun s : Array Round => s.toList).flatten := rfl
  simp only [hfl] at h₁ ⊢
  obtain ⟨i, hi, k, hk, he, ht⟩ := flatten_getElem _ t h₁
  simp only [he]
  have hi' : i < spec.repStreams.size := by simpa [hsz] using hi
  have hk' : k < (streams[i]'(hsz ▸ hi')).size := by simpa using hk
  have hget : (streams.toList.map fun s : Array Round => s.toList)[i][k] = (streams[i]'(hsz ▸ hi'))[k] := by simp
  rw [hget]
  have hoff : streamOffset spec (eraseCoins j) spec.repStreams[i].2.2 + k = t := by
    rw [ht]
    congr 1
    have hnames : spec.repStreams[i].2.2 = (spec.repStreams.toList.map (·.2.2))[i]'(by simpa using hi') := by simp
    unfold streamOffset
    rw [hnames, takeWhile_ne_nodup _ hnd i (by simpa using hi'), List.map_take, List.map_take]
    congr 2
    apply List.ext_getElem (by simp [hsz])
    intro i' h₁' h₂'
    have hi'' : i' < spec.repStreams.size := by simpa using h₁'
    simp only [List.getElem_map, Array.getElem_toList]
    exact hsize i' hi''
  obtain ⟨rs₁, -, hm⟩ := hS i hi'
  obtain ⟨hl, hr⟩ := array_mapM_ok_inv _ _ _ hm
  have hk₁ : k < rs₁.size := by
    have := hl ▸ hk'
    simpa using this
  have hrk := hr k (by simpa using hk₁)
  simp only [Array.getElem_mapIdx, hoff, List.getElem?_eq_getElem h₂] at hrk
  exact roundJ_coins _ _ hrk

end Custody

end FlockDeploy
