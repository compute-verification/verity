module

public import Flock.Canon
public import Proofs.Core.QSort

/-!
# `Flock.canonUtf8` of a flat object, in a form the kernel evaluates

`Flock.canonUtf8` is well founded and sorts with core's `Array.qsort`, neither of which the kernel unfolds. On a flat
object, every value `null`, a Boolean, a number or a string (`jsonFlat`), it is `canonFlat`, which sorts only items
whose keys are out of order (`canonUtf8_flat`); `Json.mkObj` already keeps them in order, so `decide +kernel` evaluates
`canonFlat` on its objects, given the `import all`s of core's string order, arrays and `Repr` that its caller needs.
-/

@[expose] public section

namespace Verity.CanonUtf8

open Lean (Json)
open Flock (V canonStrUtf8)

def jsonLeaf : Json → Bool
  | .null | .bool _ | .num _ | .str _ => true
  | .arr _ | .obj _ => false

/-- `Flock.canonUtf8` of a leaf. -/
def canonLeaf : Json → V String
  | .null => pure "null"
  | .bool b => pure (if b then "true" else "false")
  | .num n => if n.exponent == 0 then pure (toString n.mantissa) else throw "canonical JSON: a non-integer number"
  | .str s => pure (canonStrUtf8 s)
  | .arr _ | .obj _ => throw "canonical JSON: not a leaf"

/-- `Flock.canonUtf8` of an object of leaves. -/
def canonObj (kvs : Std.TreeMap.Raw String Json) : V String := do
  let vals ← kvs.toArray.attach.mapM fun p => do pure (p.1.1, ← canonLeaf p.1.2)
  let items := vals.qsort (fun a b => a.1 < b.1)
  pure ("{" ++ ",".intercalate (items.toList.map fun kv => canonStrUtf8 kv.1 ++ ":" ++ kv.2) ++ "}")

def KeysSorted (as : Array (String × String)) : Prop :=
  ∀ i (hi : i < as.size) j (hj : j < as.size), decide (as[i].1 < as[j].1) = decide (i < j)

instance instDecidableKeysSorted (as : Array (String × String)) : Decidable (KeysSorted as) := by
  unfold KeysSorted; infer_instance

/-- `canonObj`, sorting only items whose keys are out of order, so that the kernel evaluates it on sorted keys. -/
def canonObjInOrder (kvs : Std.TreeMap.Raw String Json) : V String := do
  let vals ← kvs.toArray.attach.mapM fun p => do pure (p.1.1, ← canonLeaf p.1.2)
  let items := if KeysSorted vals then vals else vals.qsort (fun a b => a.1 < b.1)
  pure ("{" ++ ",".intercalate (items.toList.map fun kv => canonStrUtf8 kv.1 ++ ":" ++ kv.2) ++ "}")

theorem canonObj_inOrder (kvs : Std.TreeMap.Raw String Json) : canonObj kvs = canonObjInOrder kvs := by
  unfold canonObj canonObjInOrder
  congr 1
  funext vals
  split
  · rename_i h; rw [Verity.QSort.qsort_sorted vals (fun a b => decide (a.1 < b.1)) h]
  · rfl

def jsonFlat (j : Json) : Bool := match j with
  | .obj kvs => kvs.toArray.all fun p => jsonLeaf p.2
  | _ => false

def canonFlat (j : Json) : V String := match j with
  | .obj kvs => canonObjInOrder kvs
  | _ => throw "not an object"

theorem canonUtf8_leaf (j : Json) (h : jsonLeaf j = true) : Flock.canonUtf8 j = canonLeaf j := by
  cases j <;> first
    | (simp [jsonLeaf] at h; done)
    | simp only [Flock.canonUtf8, canonLeaf]

theorem canonUtf8_obj (kvs : Std.TreeMap.Raw String Json)
    (h : (kvs.toArray.all fun p => jsonLeaf p.2) = true) : Flock.canonUtf8 (.obj kvs) = canonObj kvs := by
  have hc : ∀ p : {x // x ∈ kvs.toArray}, Flock.canonUtf8 p.1.2 = canonLeaf p.1.2 :=
    fun p => canonUtf8_leaf _ (Array.all_eq_true_iff_forall_mem.1 h p.1 p.2)
  rw [Flock.canonUtf8, canonObj]
  simp only [hc]

theorem canonUtf8_flat (j : Json) (h : jsonFlat j = true) : Flock.canonUtf8 j = canonFlat j := by
  cases j
  case obj kvs => exact (canonUtf8_obj kvs h).trans (canonObj_inOrder kvs)
  all_goals simp [jsonFlat] at h

/-- An `Except` the kernel decides is `.ok a` is `.ok a`. -/
theorem eq_ok_of {α : Type} [DecidableEq α] {x : Except String α} {a : α}
    (h : (match x with | .ok b => decide (b = a) | .error _ => false) = true) : x = .ok a := by
  cases x with
  | error e => simp at h
  | ok b =>
    simp only [decide_eq_true_eq] at h
    rw [h]

end Verity.CanonUtf8
