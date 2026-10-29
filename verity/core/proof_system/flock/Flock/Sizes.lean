/-
Parsed JSON shrinks inward: a member of an array or an object is smaller than it, what recursion over parsed JSON
terminates by.
-/
import Lean.Data.Json

namespace Flock

open Lean (Json)

private theorem impl_foldl_push_mem {β : Type} [SizeOf β] (i : Std.DTreeMap.Internal.Impl String (fun _ => β))
    (init : Array (String × β)) (p : String × β) (h : p ∈ i.foldl (fun l k v => l.push (k, v)) init) :
    p ∈ init ∨ sizeOf p.2 < sizeOf i := by
  induction i generalizing init with
  | leaf => exact .inl (by simpa [Std.DTreeMap.Internal.Impl.foldl, Std.DTreeMap.Internal.Impl.foldlM] using h)
  | inner sz k v l r ihl ihr =>
    have hu : (Std.DTreeMap.Internal.Impl.inner sz k v l r).foldl (fun l k v => l.push (k, v)) init =
        r.foldl (fun l k v => l.push (k, v)) ((l.foldl (fun l k v => l.push (k, v)) init).push (k, v)) := by
      simp [Std.DTreeMap.Internal.Impl.foldl, Std.DTreeMap.Internal.Impl.foldlM]
    rw [hu] at h
    simp only [Std.DTreeMap.Internal.Impl.inner.sizeOf_spec]
    rcases ihr _ h with h | h
    · rcases Array.mem_push.1 h with h | h
      · rcases ihl _ h with h | h
        · exact .inl h
        · right; omega
      · subst h; right; simp only; omega
    · right; omega

theorem sizeOf_lt_of_mem_obj {kvs : Std.TreeMap.Raw String Json} {p : String × Json} (h : p ∈ kvs.toArray) :
    sizeOf p.2 < sizeOf (Json.obj kvs) := by
  have h1 : sizeOf kvs.inner.inner < sizeOf kvs.inner + 1 := by
    cases kvs.inner; simp only [Std.DTreeMap.Raw.mk.sizeOf_spec]; omega
  have h2 : sizeOf kvs.inner < sizeOf kvs + 1 := by
    cases kvs; simp only [Std.TreeMap.Raw.mk.sizeOf_spec]; omega
  simp only [Json.obj.sizeOf_spec]
  rcases impl_foldl_push_mem kvs.inner.inner _ p h with h | h
  · simp at h
  · omega

theorem sizeOf_lt_of_mem_arr {a : Array Json} {x : Json} (h : x ∈ a) : sizeOf x < sizeOf (Json.arr a) := by
  have := Array.sizeOf_lt_of_mem h
  simp only [Json.arr.sizeOf_spec]
  omega

theorem sizeOf_lt_of_getElem?_arr {a : Array Json} {i : Nat} {x : Json} (h : a[i]? = some x) :
    sizeOf x < sizeOf (Json.arr a) :=
  sizeOf_lt_of_mem_arr (Array.mem_of_getElem? h)

end Flock
