/-
The forms `verify --zk` refuses so that the sampled audit's knowledge soundness (`Flock.Guarantees.SampledKSLiveV4`) takes
none of `SampledKSLiveV3`'s file preconditions as a hypothesis: no form goes ahead of its proof. Under `--zk` every session
of a statement outside them is refused, beside `ProvedScope.check` and `PortRoots.check`:

* `tagsWhy tags`: a typed statement, one with public outputs, one without retained rounds, or one whose leaves are not
  `hm96-sha512/v1` (V3's `I.tags.typed = false` and `TagsOkZHJ`).
* `filesWhy c pub`: a circuit outside `ProvedScope.scopeOkH` (the unit reads one input group and returns the outputs,
  V3's `scopeOkH`; `ProvedScope.check` holds a hidden-output circuit only to the weaker `scopeOkI`), and a shared-row
  public file (header `shared_rows`, V3's `pub.tables = none`), with or without `--registered F`.
* `drawWhy count draw`: at more than one table, a session whose record holds no draw that parses, or whose subset draw's
  `k` the tables don't divide (V3's `kd % nTab = 0`).

`checkOf parsed tags publicFile count draw` runs them on the circuit as `HmOut.parse` reads it and the public file as
`HmOut.loadPublic` loads it; `check_ok` says what passing it establishes. V3's sixth, that the unit is a built net, follows
from the parse of an untyped statement. A V* session (`--rec-inner`) is the recursion's, whose guarantee is its own, so
`verify` doesn't run the check on it. `flock-verify ks-form` runs `checkOf` alone on a statement's files, and the Python
mirror (`one_stage.proofs.ks_form_why`, `filesWhy` on the files' heads) is held to it.
-/
import Flock.ProvedScope

namespace Flock.KsScope

open Lean (Json)

/-- What `--zk`'s tags must be for the sampled audit's knowledge soundness: untyped, hidden outputs, retained rounds, and
`hm96-sha512/v1` leaves. -/
def tagsWhy (tags : Tags) : Option String :=
  if tags.typed then some s!"{tags.name}, a typed statement"
  else if !tags.hiddenOutputs then some s!"{tags.name}, a statement with public outputs under --zk"
  else if !tags.retainRounds then some s!"{tags.name}, a statement without retained rounds"
  else if tags.merkleLeaf != "hm96-sha512/v1" then some s!"{tags.name}, a statement with {tags.merkleLeaf} leaves"
  else none

def SCOPE_WHY : String :=
  "a circuit outside the unit's form under --zk (scopeOkH: the unit reads its one input group and returns the outputs)"

def TABLES_WHY : String := "a shared-row public file (header shared_rows) under --zk"

/-- The circuit and public file: the unit's form (`ProvedScope.scopeOkH`), and no shared rows. -/
def filesWhy (c : Circuit) (pub : Public) : Option String :=
  if !ProvedScope.scopeOkH c then some SCOPE_WHY
  else if pub.tables.isSome then some TABLES_WHY
  else none

/-- The session's draw (`draw`, the record's `unit_draw`) at `count` tables: none to check at one table; at more, a draw
that parses, and a subset draw's `k` a multiple of `count`, so every table proves `k / count` units. -/
def drawWhy (count : Nat) (draw : Option Json) : Option String :=
  if count ≤ 1 then none
  else match draw with
    | none => some s!"a session of {count} tables without a draw"
    | some j => match Draw.UnitDraw.ofJson j with
      | .error e => some s!"a draw that does not parse ({e})"
      | .ok d => match d.law with
        | .subset k => if k % count = 0 then none
          else some s!"a subset draw of {k} units over {count} tables, which do not divide it"
        | _ => none

/-- **`verify --zk`'s files are in the form the sampled audit's knowledge soundness covers**: the tags (`tagsWhy`), on the
circuit `parsed` (`HmOut.parse`'s) and the public file as `HmOut.loadPublic` loads it the unit's form and no shared rows
(`filesWhy`), and the session's draw (`drawWhy`). -/
def checkOf (parsed : V Circuit) (tags : Tags) (publicFile : ByteArray) (count : Nat) (draw : Option Json) :
    Except String Unit :=
  match tagsWhy tags with
  | some why => .error (ProvedScope.refusal why)
  | none => match parsed with
    | .error e => .error (ProvedScope.refusal s!"a circuit that does not parse ({e})")
    | .ok c => match HmOut.loadPublic tags c publicFile with
      | .error e => .error (ProvedScope.refusal s!"a public file that does not load ({e})")
      | .ok pub => match filesWhy c pub with
        | some why => .error (ProvedScope.refusal why)
        | none => match drawWhy count draw with
          | some why => .error (ProvedScope.refusal why)
          | none => .ok ()

/-- `checkOf` on the circuit file as `HmOut.parse` reads it. -/
def check (tags : Tags) (circuitFile publicFile : ByteArray) (tables : List (String × ByteArray)) (count : Nat)
    (draw : Option Json) : Except String Unit :=
  checkOf (HmOut.parse tags circuitFile tables) tags publicFile count draw

theorem tagsWhy_none {tags : Tags} (h : tagsWhy tags = none) :
    tags.typed = false ∧ tags.hiddenOutputs = true ∧ tags.retainRounds = true ∧ tags.merkleLeaf = "hm96-sha512/v1" := by
  unfold tagsWhy at h
  split at h
  next => cases h
  next h1 =>
    split at h
    next => cases h
    next h2 =>
      split at h
      next => cases h
      next h3 =>
        split at h
        next => cases h
        next h4 => exact ⟨by simpa using h1, by simpa using h2, by simpa using h3, by simpa using h4⟩

theorem filesWhy_none {c : Circuit} {pub : Public} (h : filesWhy c pub = none) :
    ProvedScope.scopeOkH c = true ∧ pub.tables = none := by
  unfold filesWhy at h
  split at h
  next => cases h
  next h1 =>
    split at h
    next => cases h
    next h2 => exact ⟨by simpa using h1, by simpa using h2⟩

/-- **At more than one table, a passing draw parses, and its subset `k` is a multiple of the tables.** -/
theorem drawWhy_none {count : Nat} {draw : Option Json} (h : drawWhy count draw = none) (hc : ¬ count ≤ 1) :
    ∃ j d, draw = some j ∧ Draw.UnitDraw.ofJson j = .ok d ∧ ∀ k, d.law = .subset k → k % count = 0 := by
  unfold drawWhy at h
  rw [if_neg hc] at h
  cases draw with
  | none => simp at h
  | some j =>
    cases hj : Draw.UnitDraw.ofJson j with
    | error e => simp [hj] at h
    | ok d =>
      refine ⟨j, d, rfl, hj, fun k hk => ?_⟩
      simp only [hj, hk] at h
      split at h
      next hm => exact hm
      next => cases h

/-- **What `check` passing says of the files**: the tags pass `tagsWhy`, and at the circuit `HmOut.parse` reads and the
public file `HmOut.loadPublic` loads, the circuit is in `scopeOkH`, the public file has no shared rows, and the draw passes
`drawWhy`. -/
theorem check_ok {tags : Tags} {cf pf : ByteArray} {tables : List (String × ByteArray)} {count : Nat}
    {draw : Option Json} {c : Circuit} {pub : Public} (h : check tags cf pf tables count draw = .ok ())
    (hc : HmOut.parse tags cf tables = .ok c) (hpub : HmOut.loadPublic tags c pf = .ok pub) :
    tagsWhy tags = none ∧ ProvedScope.scopeOkH c = true ∧ pub.tables = none ∧ drawWhy count draw = none := by
  unfold check checkOf at h
  rw [hc] at h
  cases ht : tagsWhy tags with
  | some w => simp [ht] at h
  | none =>
    simp only [ht, hpub] at h
    cases hf : filesWhy c pub with
    | some w => simp [hf] at h
    | none =>
      simp only [hf] at h
      cases hd : drawWhy count draw with
      | some w => simp [hd] at h
      | none =>
        obtain ⟨hs, hp⟩ := filesWhy_none hf
        exact ⟨rfl, hs, hp, rfl⟩

end Flock.KsScope
