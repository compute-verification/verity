module

public import Properties.Pous.WipeCount
public import Proofs.Core.CanonUtf8
public import Proofs.Core.QSort
import all Init.Data.Array.QSort.Basic
import all Init.Data.String.Defs
import all Lean.Data.Json.Basic
import all Lean.Data.Json.FromToJson.Basic
import all Init.Data.Ord.String
import all Init.Data.Array.Basic
import all Init.Data.Array.DecidableEq
import all Init.Data.List.Control
import all Init.Data.Range.Basic
import all Init.Data.Int.Repr
import all Init.Data.Repr

/-!
# The digests of the honest span `WipeCountTied`'s witnesses read

The default schedule's digest, and each execution phase's rows domain and head under it in the witnesses' session
(`witnessDomain`, `witnessHead`), are defined (`domain_eq`, `head_eq`), the head 64 bytes (`witnessHead_size`): the
kernel evaluates them, their SHA-512s included, once `Flock.canonUtf8` is in a form it unfolds (`canonUtf8_vals`).
-/

@[expose] public section

namespace MemoryChallenger.Properties.SecurityProofs

/-! ## `Flock.canonUtf8` is well founded and sorts with core's `Array.qsort`, which the kernel does not unfold. On an object
whose values are leaves or arrays of leaves (`objVals`), as the default schedule's digest and a rows domain hash, it is
`canonC`, which sorts only keys out of order (`canonUtf8_vals`), so `decide +kernel` evaluates the default schedule's
digest, each execution phase's domain and head, and their SHA-512s. -/

open Lean (Json) in
def canonVal : Json → Flock.V String
  | .arr a => do
    let parts ← a.attach.mapM fun x => Verity.CanonUtf8.canonLeaf x.1
    pure ("[" ++ ",".intercalate parts.toList ++ "]")
  | j => Verity.CanonUtf8.canonLeaf j

open Lean (Json) in
def jsonVal : Json → Bool
  | .arr a => a.all Verity.CanonUtf8.jsonLeaf
  | j => Verity.CanonUtf8.jsonLeaf j

open Lean (Json) in
theorem canonUtf8_val (j : Json) (h : jsonVal j = true) : Flock.canonUtf8 j = canonVal j := by
  cases j with
  | arr a =>
    have hc : ∀ x : {x // x ∈ a}, Flock.canonUtf8 x.1 = Verity.CanonUtf8.canonLeaf x.1 :=
      fun x => Verity.CanonUtf8.canonUtf8_leaf _ (Array.all_eq_true_iff_forall_mem.1 h x.1 x.2)
    rw [Flock.canonUtf8, canonVal]
    simp only [hc]
  | obj kvs => simp [jsonVal, Verity.CanonUtf8.jsonLeaf] at h
  | null => exact Verity.CanonUtf8.canonUtf8_leaf _ h
  | bool b => exact Verity.CanonUtf8.canonUtf8_leaf _ h
  | num n => exact Verity.CanonUtf8.canonUtf8_leaf _ h
  | str s => exact Verity.CanonUtf8.canonUtf8_leaf _ h

open Lean (Json) in
def canonObjVal (kvs : Std.TreeMap.Raw String Json) : Flock.V String := do
  let vals ← kvs.toArray.attach.mapM fun p => do pure (p.1.1, ← canonVal p.1.2)
  let items := if Verity.CanonUtf8.KeysSorted vals then vals else vals.qsort (fun a b => a.1 < b.1)
  pure ("{" ++ ",".intercalate (items.toList.map fun kv => Flock.canonStrUtf8 kv.1 ++ ":" ++ kv.2) ++ "}")

open Lean (Json) in
def objVals (j : Json) : Bool := match j with
  | .obj kvs => kvs.toArray.all fun p => jsonVal p.2
  | _ => false

open Lean (Json) in
def canonC (j : Json) : Flock.V String := match j with
  | .obj kvs => canonObjVal kvs
  | _ => throw "not an object"

open Lean (Json) in
theorem canonUtf8_vals (j : Json) (h : objVals j = true) : Flock.canonUtf8 j = canonC j := by
  cases j
  case obj kvs =>
    have hc : ∀ p : {x // x ∈ kvs.toArray}, Flock.canonUtf8 p.1.2 = canonVal p.1.2 :=
      fun p => canonUtf8_val _ (Array.all_eq_true_iff_forall_mem.1 h p.1 p.2)
    rw [Flock.canonUtf8]
    simp only [hc, canonC, canonObjVal]
    congr 1
    funext vals
    split
    · rename_i hs; rw [Verity.QSort.qsort_sorted vals (fun a b => decide (a.1 < b.1)) hs]
    · rfl
  all_goals simp [objVals] at h

theorem ok_of_isSome {α : Type} {x : Except String α} {d : α} (h : x.toOption.isSome = true) :
    x = .ok (x.toOption.getD d) := by
  cases x with
  | ok a => rfl
  | error e => simp [Except.toOption] at h

-- the kernel evaluates the default schedule's digest and three domains, their SHA-512s included
set_option maxHeartbeats 4000000 in
theorem domain_isSome (w : Nat) (hw : w < 3) :
    (NetworkCheck.Period.rowsDomain NetworkCheck.Period.defaultSchedule witnessScheduleDigest witnessSession "eth0"
      .egress w).toOption.isSome = true := by
  obtain rfl | rfl | rfl : w = 0 ∨ w = 1 ∨ w = 2 := by omega
  all_goals
    unfold NetworkCheck.Period.rowsDomain witnessScheduleDigest NetworkCheck.Period.Schedule.digest
      Identity.identityDigestSha512
    rw [canonUtf8_vals, canonUtf8_vals]
    all_goals decide +kernel

-- the kernel evaluates the default schedule's digest and three domains and heads, their SHA-512s included
set_option maxHeartbeats 4000000 in
theorem head_size (w : Nat) (hw : w < 3) :
    (NetworkCheck.Period.headDigest "eth0" .egress w witnessScheduleDigest NetworkCheck.Period.defaultSchedule
      (witnessZeros 64) (witnessDomain w) (witnessZeros 64)).toOption.map ByteArray.size = some 64 := by
  obtain rfl | rfl | rfl : w = 0 ∨ w = 1 ∨ w = 2 := by omega
  all_goals
    unfold witnessDomain NetworkCheck.Period.rowsDomain witnessScheduleDigest NetworkCheck.Period.Schedule.digest
      Identity.identityDigestSha512
    rw [canonUtf8_vals, canonUtf8_vals]
    all_goals decide +kernel

theorem domain_eq (w : Nat) (hw : w < 3) :
    NetworkCheck.Period.rowsDomain NetworkCheck.Period.defaultSchedule witnessScheduleDigest witnessSession "eth0"
      .egress w = .ok (witnessDomain w) :=
  ok_of_isSome (domain_isSome w hw)

theorem head_eq (w : Nat) (hw : w < 3) :
    NetworkCheck.Period.headDigest "eth0" .egress w witnessScheduleDigest NetworkCheck.Period.defaultSchedule
      (witnessZeros 64) (witnessDomain w) (witnessZeros 64) = .ok (witnessHead w) := by
  refine ok_of_isSome ?_
  have h := head_size w hw
  revert h
  cases NetworkCheck.Period.headDigest "eth0" .egress w witnessScheduleDigest NetworkCheck.Period.defaultSchedule
    (witnessZeros 64) (witnessDomain w) (witnessZeros 64) <;> simp [Except.toOption]

theorem witnessHead_size (w : Nat) (hw : w < 3) : (witnessHead w).size = 64 := by
  have h := head_size w hw
  rw [head_eq w hw] at h
  simpa [Except.toOption] using h

end MemoryChallenger.Properties.SecurityProofs
