module

public import Proofs.Flock.Claims.Sampling.Select.Design
public import Proofs.Flock.VBridge.Family.UnitName
public import Std.Data.String.ToNat
import all Lean.Data.Json.Basic
import all Init.Data.Ord.String
import all Init.Data.Array.Basic
import all Init.Data.String.Search
import all Init.Data.String.Slice

/-!
# `parseDesign` reads a design's text back

* **A number's digits** (`not_mem_repr`, `digits_repr`, `toNat!_repr`): `Nat.repr K` holds only digits, `Select`'s
  `digits` accepts it, and `String.toNat!` reads `K` back.
* **`Select`'s `split1`** (`split1_colon`, `split1_slash`): `a ++ ":" ++ b` splits into `a` and `b` where neither holds
  a colon, and likewise at a slash.
* **`parseDesign` at a text** (`parseDesign_count`, `parseDesign_bernoulli`): `kind:K` at `K ≥ 1`, for a `subset`,
  `stratified` or `work` design, is `{law: kind, k: K}`, and `bernoulli:NUM/DEN` at `0 < DEN` and `NUM ≤ DEN` is
  `{law: bernoulli, p: "NUM/DEN"}`, each number in `Nat.repr`'s digits.
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace Properties.Spot.SecurityProofs

open Lean (Json)
open private split1 digits from Spot.Select

theorem isDigit_of_mem_repr {c : Char} {K : ℕ} (h : c ∈ (Nat.repr K).toList) : c.isDigit = true := by
  rw [Nat.toList_repr] at h
  exact Nat.isDigit_of_mem_toDigits (by decide) (by decide) h

theorem not_mem_repr {c : Char} (hc : c.isDigit = false) (K : ℕ) : c ∉ (Nat.repr K).toList := fun h => by
  rw [isDigit_of_mem_repr h] at hc
  exact absurd hc (by decide)

/-- `Select`'s `split1` at the separator `sep` of one character `c`, where `String.splitOn sep` is `List.splitOn c` on
the characters: `a ++ sep ++ b` is `a` and `b` where neither holds `c`. -/
theorem split1_sep (sep : String) (c : Char) (hsp : sep.toList = [c])
    (hsplit : ∀ s : String, s.splitOn sep = (s.toList.splitOn c).map String.ofList)
    (a b : String) (ha : c ∉ a.toList) (hb : c ∉ b.toList) : split1 (a ++ sep ++ b) sep = (a, b) := by
  have hne : ∀ l : List Char, c ∉ l → ∀ x ∈ l, (x == c) = false := fun l hl x hx => by
    simp only [beq_eq_false_iff_ne, ne_eq]
    rintro rfl
    exact hl hx
  unfold split1
  rw [hsplit, String.toList_append, String.toList_append, hsp, List.append_assoc, List.singleton_append,
    List.splitOn_eq_splitOnP, List.splitOnP_append_cons_of_forall_mem (hne _ ha) c (by simp),
    List.splitOnP_eq_singleton (hne _ hb)]
  simp

theorem split1_colon (a b : String) (ha : ':' ∉ a.toList) (hb : ':' ∉ b.toList) : split1 (a ++ ":" ++ b) ":" = (a, b) :=
  split1_sep ":" ':' (by decide) (FlockVBridge.Names.splitOn_sep ":" ':' rfl rfl rfl rfl (by decide)) a b ha hb

theorem split1_slash (a b : String) (ha : '/' ∉ a.toList) (hb : '/' ∉ b.toList) : split1 (a ++ "/" ++ b) "/" = (a, b) :=
  split1_sep "/" '/' (by decide) (FlockVBridge.Names.splitOn_sep "/" '/' rfl rfl rfl rfl (by decide)) a b ha hb

theorem digits_repr (K : ℕ) : digits (Nat.repr K) = true := by
  unfold digits
  rw [String.all_bool_eq, Bool.and_eq_true, Bool.not_eq_true', List.all_eq_true]
  exact ⟨String.isEmpty_eq_false_iff.2 Nat.repr_ne_empty, fun c hc => isDigit_of_mem_repr hc⟩

theorem toNat!_repr (K : ℕ) : (Nat.repr K).toNat! = K := by
  have h := Nat.toNat?_repr K
  unfold String.toNat? String.Slice.toNat? at h
  unfold String.toNat! String.Slice.toNat!
  split at h
  · rename_i hi
    rw [if_pos hi]
    exact Option.some.inj h
  · exact absurd h (by simp)

/-- **`parseDesign` reads `kind:K` back**: a `subset`, `stratified` or `work` design of `K ≥ 1`, its count in
`Nat.repr`'s digits, is `{law: kind, k: K}`. -/
theorem parseDesign_count (kind : String) (hk : kind = "subset" ∨ kind = "stratified" ∨ kind = "work") (K : ℕ)
    (h1 : 1 ≤ K) :
    _root_.Spot.Select.parseDesign (kind ++ ":" ++ Nat.repr K) =
      .ok (Json.mkObj [("law", Json.str kind), ("k", Lean.toJson K)]) := by
  have hka : ':' ∉ kind.toList := by rcases hk with rfl | rfl | rfl <;> decide
  unfold _root_.Spot.Select.parseDesign
  rw [split1_colon kind (Nat.repr K) hka (not_mem_repr (by decide) K)]
  simp only [digits_repr, toNat!_repr, Bool.and_true, decide_eq_true h1]
  rcases hk with rfl | rfl | rfl <;> rfl

/-- **`parseDesign` reads `bernoulli:NUM/DEN` back**: at `0 < DEN` and `NUM ≤ DEN`, each in `Nat.repr`'s digits, it is
`{law: bernoulli, p: "NUM/DEN"}`. -/
theorem parseDesign_bernoulli (a b : ℕ) (hb : 0 < b) (hab : a ≤ b) :
    _root_.Spot.Select.parseDesign ("bernoulli:" ++ (Nat.repr a ++ "/" ++ Nat.repr b)) =
      .ok (Json.mkObj [("law", "bernoulli"), ("p", Json.str s!"{a}/{b}")]) := by
  unfold _root_.Spot.Select.parseDesign
  rw [show "bernoulli:" ++ (Nat.repr a ++ "/" ++ Nat.repr b) =
    "bernoulli" ++ ":" ++ (Nat.repr a ++ "/" ++ Nat.repr b) from rfl, split1_colon _ _ (by decide) ?_]
  · dsimp only
    rw [split1_slash _ _ (not_mem_repr (by decide) a) (not_mem_repr (by decide) b)]
    simp only [digits_repr, toNat!_repr, decide_eq_true hb, decide_eq_true hab]
    rfl
  · rw [String.toList_append, String.toList_append]
    simp only [List.mem_append, not_or]
    exact ⟨⟨not_mem_repr (by decide) a, by decide⟩, not_mem_repr (by decide) b⟩

end Properties.Spot.SecurityProofs
