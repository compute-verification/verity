module

public import Proofs.Flock.SpotFlock.InstanceStmt.Bytes
public import Proofs.Flock.SessionLoop.Concrete.Public
public import Proofs.Flock.Recursive.FirewallObj
import Flock.Firewall
import Flock.HmRow

/-! # The statement's values are plain

`plain` reads each string through `plainStr`, which is `String.all`, and the kernel does not run `String.all`, so no
value's `plain` is a `kernel_rfl`. `plain_json` proves it one object at a time (`FirewallObj.plain_mkObj`): literal
strings by `simp`, the hex strings by `plainStr_hexOf` before `plainStr` unfolds, and META's rows by `plain_rowJ`.
-/

@[expose] public section

namespace SpotFlock.InstanceStmt

open Flock Flock.Firewall
open FlockDeploy.Concrete (plainStr_hexOf)
open Lean (Json)

theorem plainStr_keyHex : plainStr keyHex = true := plainStr_hexOf _
theorem plainStr_hex16 (w : UInt64) : plainStr (HmRow.hex16 w) = true := plainStr_hexOf _

theorem plainList_of {l : List Json} (h : ∀ j ∈ l, plain j = true) : plainList l = true := by
  induction l with
  | nil => rfl
  | cons j l ih => simp only [plainList, h j (by simp), ih fun x hx => h x (by simp [hx]), Bool.and_self]

theorem plain_arr_map {α : Type} (A : Array α) (f : α → Json) (h : ∀ x, plain (f x) = true) :
    plain (.arr (A.map f)) = true := by
  rcases hA : A.map f with ⟨l⟩
  have hl : A.toList.map f = l := by rw [← Array.toList_map, hA]
  show plainList l = true
  exact plainList_of fun j hj => by
    rw [← hl] at hj
    obtain ⟨x, _, rfl⟩ := List.mem_map.1 hj
    exact h x

theorem plain_rowJ (p : Port) : plain (rowJ p) = true := by
  unfold rowJ
  refine FlockSoundness.Discharge.Recursion.FirewallObj.plain_mkObj (by simp) ?_
  simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq]
  exact ⟨⟨by simp [plainStr, String.all_bool_eq], plain_arr_map _ _ fun w => plainStr_hex16 w⟩,
    ⟨by simp [plainStr, String.all_bool_eq], plainStr_hexOf _⟩,
    ⟨by simp [plainStr, String.all_bool_eq], plainStr_hexOf _⟩⟩

/-- `plain` of a value written with `Json.mkObj`, `ja`, `js` and `jn`, once its other definitions are unfolded. Each
object's members are its side goals, so the discharge depth bounds how deep objects nest. -/
macro "plain_json" : tactic =>
  `(tactic| simp (maxDischargeDepth := 16) [FlockSoundness.Discharge.Recursion.FirewallObj.plain_mkObj, plain,
    plainList, plainStr, String.all_bool_eq, jn, js, ja, plain_rowJ, ↓plainStr_hexOf, ↓plainStr_keyHex,
    ↓plainStr_hex16])

end SpotFlock.InstanceStmt
