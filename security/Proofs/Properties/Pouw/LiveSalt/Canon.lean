module

public import Batteries.Tactic.OpenPrivate
public import Proofs.Flock.Claims.CanonText
public import Properties.Pouw.LiveSalt
public import Spot.Challenger
public import Std.Data.String.ToNat
import all Init.Data.Array.QSort.Basic
import all Init.Data.Int.Repr
import all Init.Data.Ord.String
import all Lean.Data.Json.Basic

/-!
# The coins' binding has one canonical form per layout

`liveDomain` hashes the identity bytes (`Flock.canonUtf8`) of the binding `{epoch, families, scheme}`. Its form is
`{"epoch":E,"families":F,"scheme":S}` (`canon_binding`): `canon` sorts the three keys, and they are already in order
(`qsort3`). That string parses back uniquely (`binding_inj`): a decimal numeral ends at the first non-digit
(`digits_split`), an escaped string at its first unescaped quote, each escape decoding to one character (`unesc_esc`,
`canonStr_split`), and the families' list item by item, each a string and a numeral (`fam_split`). What holds of
`canonUtf8`'s text alone (the numerals, the escapes) is `Proofs.Flock.Claims.CanonText`'s, in this namespace.
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

open private Array.qsort.sort Array.qpartition.loop from Init.Data.Array.QSort.Basic

namespace Properties.Pouw.LiveSalt.Canon

open Lean (Json)
open Spot.Challenger (familiesJson)

/-! ## Sorting three keys in order -/

private theorem loop02 {β : Type} (lt : β → β → Bool) (x y z : β) (hxy : lt x y = true) (hzy : lt z y = false) :
    Array.qpartition.loop (n := 3) lt 0 2 (by omega) y #v[x, z, y] 0 0 = (⟨1, by omega⟩, #v[x, y, z]) := by
  rw [Array.qpartition.loop.eq_def]
  simp (config := {decide := true}) [hxy]
  rw [Array.qpartition.loop.eq_def]
  simp (config := {decide := true}) [hzy]
  rw [Array.qpartition.loop.eq_def]
  simp (config := {decide := true})

private theorem loop01 {β : Type} (lt : β → β → Bool) (x y z : β) (hyx : lt y x = false) :
    Array.qpartition.loop (n := 3) lt 0 1 (by omega) x #v[y, x, z] 0 0 = (⟨0, by omega⟩, #v[x, y, z]) := by
  rw [Array.qpartition.loop.eq_def]
  simp (config := {decide := true}) [hyx]
  rw [Array.qpartition.loop.eq_def]
  simp (config := {decide := true})

private theorem sort_self {β : Type} (lt : β → β → Bool) {n : Nat} (v : Vector β n) (i : Nat) (h : i < n) :
    Array.qsort.sort lt v i i (by omega) h h = v := by
  rw [Array.qsort.sort.eq_def]
  simp

private theorem sort01 {β : Type} (lt : β → β → Bool) (x y z : β) (hxx : lt x x = false) (hyx : lt y x = false)
    (hxy : lt x y = true) :
    Array.qsort.sort (n := 3) lt #v[x, y, z] 0 1 = #v[x, y, z] := by
  rw [Array.qsort.sort.eq_def]
  simp only [Array.qpartition]
  simp (config := {decide := true}) [hxx, hyx, hxy]
  simp (config := {decide := true}) only [loop01 lt x y z hyx]
  simp [sort_self]

/-- `qsort` leaves three elements that are already in order where they are. -/
theorem qsort3 {β : Type} (lt : β → β → Bool) (x y z : β) (hxx : lt x x = false) (hyx : lt y x = false)
    (hzx : lt z x = false) (hyz : lt y z = true) (hxy : lt x y = true) (hzy : lt z y = false) :
    #[x, y, z].qsort lt = #[x, y, z] := by
  unfold Array.qsort
  show (Array.qsort.sort lt (#[x, y, z].toVector) 0 2 (by omega) (by simp) (by simp)).toArray = _
  rw [Array.qsort.sort.eq_def]
  simp only [Array.qpartition]
  simp (config := {decide := true}) [hyx, hzx, hyz]
  simp (config := {decide := true}) only [loop02 lt x y z hxy hzy]
  simp [sort01 lt x y z hxx hyx hxy, sort_self]

theorem binding_sort {α : Type} (a b c : α) :
    #[("epoch", a), ("families", b), ("scheme", c)].qsort (fun x y => x.1 < y.1) =
      #[("epoch", a), ("families", b), ("scheme", c)] :=
  qsort3 _ _ _ _ (by exact (by decide : decide ("epoch" < "epoch") = false))
    (by exact (by decide : decide ("families" < "epoch") = false))
    (by exact (by decide : decide ("scheme" < "epoch") = false))
    (by exact (by decide : decide ("families" < "scheme") = true))
    (by exact (by decide : decide ("epoch" < "families") = true))
    (by exact (by decide : decide ("scheme" < "families") = false))

/-! ## `canon`, unfolded -/

theorem canon_str (s : String) : Flock.canonUtf8 (.str s) = pure (Flock.canonStrUtf8 s) := by
  rw [Flock.canonUtf8]

theorem canon_nat (n : Nat) : Flock.canonUtf8 (Lean.toJson n) = pure (toString n) := by
  show Flock.canonUtf8 (.num ⟨n, 0⟩) = _
  rw [Flock.canonUtf8]; rfl

/-- A family's canonical form: `[label,count]`. -/
def itemStr (p : String × Nat) : String := "[" ++ ",".intercalate [Flock.canonStrUtf8 p.1, toString p.2] ++ "]"

/-- The families' canonical form: their items, comma-separated, in brackets. -/
def famStr (fs : List (String × Nat)) : String := "[" ++ ",".intercalate (fs.map itemStr) ++ "]"

theorem canon_families (fs : List (String × Nat)) : Flock.canonUtf8 (familiesJson fs) = pure (famStr fs) := by
  unfold familiesJson
  rw [canon_arr, List.map_toArray, List.mapM_toArray]
  suffices h : ∀ gs : List (String × Nat), (gs.map fun (l, n) => Json.arr #[Json.str l, Lean.toJson n]).mapM
      Flock.canonUtf8 = pure (gs.map itemStr) by
    rw [h]; simp [famStr]
  intro gs
  induction gs with
  | nil => rfl
  | cons p gs ih =>
    obtain ⟨l, n⟩ := p
    rw [List.map_cons, List.mapM_cons, ih, canon_arr, List.mapM_toArray]
    simp [canon_str, canon_nat, itemStr]

/-- The binding object `liveDomain` hashes. -/
def binding (e : Nat) (fs : List (String × Nat)) (s : String) : Json :=
  Json.mkObj [("epoch", Lean.toJson e), ("families", familiesJson fs), ("scheme", Json.str s)]

/-- The binding's canonical form. -/
def bindStr (e : Nat) (fs : List (String × Nat)) (s : String) : String :=
  "{" ++ ",".intercalate [Flock.canonStrUtf8 "epoch" ++ ":" ++ toString e, Flock.canonStrUtf8 "families" ++ ":" ++ famStr fs,
    Flock.canonStrUtf8 "scheme" ++ ":" ++ Flock.canonStrUtf8 s] ++ "}"

theorem canon_binding (e : Nat) (fs : List (String × Nat)) (s : String) :
    Flock.canonUtf8 (binding e fs s) = .ok (bindStr e fs s) := by
  show Flock.canonUtf8 (Json.obj _) = _
  rw [canon_obj]
  have ht : (Std.TreeMap.Raw.ofList [("epoch", Lean.toJson e), ("families", familiesJson fs), ("scheme", Json.str s)] :
      Std.TreeMap.Raw String Json).toArray =
      #[("epoch", Lean.toJson e), ("families", familiesJson fs), ("scheme", Json.str s)] := rfl
  rw [ht]
  simp only [List.mapM_toArray, List.mapM_cons, List.mapM_nil, canon_nat, canon_families, canon_str, pure_bind,
    map_pure]
  rw [binding_sort]
  rfl

/-! ## The families -/

theorem item_eq (p : String × Nat) (r : List Char) :
    (itemStr p).toList ++ r = '[' :: ((Flock.canonStrUtf8 p.1).toList ++ ',' :: ((toString p.2).toList ++ ']' :: r)) := by
  simp [itemStr]

theorem item_split {p p' : String × Nat} {r r' : List Char} (h : (itemStr p).toList ++ r = (itemStr p').toList ++ r') :
    p = p' ∧ r = r' := by
  rw [item_eq, item_eq, List.cons.injEq] at h
  obtain ⟨hl, h⟩ := canonStr_split h.2
  simp only [List.cons.injEq, true_and] at h
  obtain ⟨hn, -, hr⟩ := nat_split (by decide) (by decide) h
  exact ⟨Prod.ext hl hn, hr⟩

/-- What follows the first family's item: each further item after a comma, then the closing bracket and `X`. -/
def rest : List (String × Nat) → List Char → List Char
  | [], X => ']' :: X
  | p :: ps, X => ',' :: ((itemStr p).toList ++ rest ps X)

theorem rest_inj : ∀ (ps ps' : List (String × Nat)) (X X' : List Char), rest ps X = rest ps' X' → ps = ps' ∧ X = X' := by
  intro ps
  induction ps with
  | nil =>
    intro ps' X X' h
    cases ps' with
    | nil => simpa [rest] using h
    | cons p' ps' => simp [rest] at h
  | cons p ps ih =>
    intro ps' X X' h
    cases ps' with
    | nil => simp [rest] at h
    | cons p' ps' =>
      simp only [rest, List.cons.injEq, true_and] at h
      obtain ⟨rfl, h2⟩ := item_split h
      obtain ⟨rfl, rfl⟩ := ih ps' X X' h2
      exact ⟨rfl, rfl⟩

theorem inter_eq (p : String × Nat) (ps : List (String × Nat)) (X : List Char) :
    (",".intercalate ((p :: ps).map itemStr)).toList ++ ']' :: X = (itemStr p).toList ++ rest ps X := by
  induction ps generalizing p with
  | nil => simp [rest]
  | cons q ps ih =>
    rw [List.map_cons, List.map_cons, String.intercalate_cons_cons, String.toList_append, String.toList_append,
      List.append_assoc, List.append_assoc, ← List.map_cons, ih]
    simp [rest]

theorem fam_split {fs fs' : List (String × Nat)} {X X' : List Char}
    (h : (famStr fs).toList ++ X = (famStr fs').toList ++ X') : fs = fs' ∧ X = X' := by
  simp only [famStr, String.toList_append, List.append_assoc] at h
  simp only [show "[".toList = ['['] from rfl, show "]".toList = [']'] from rfl, List.cons_append, List.nil_append,
    List.cons.injEq, true_and] at h
  cases fs with
  | nil =>
    cases fs' with
    | nil => simpa using h
    | cons p' ps' =>
      rw [inter_eq, item_eq] at h
      simp at h
  | cons p ps =>
    cases fs' with
    | nil =>
      rw [inter_eq, item_eq] at h
      simp at h
    | cons p' ps' =>
      rw [inter_eq, inter_eq] at h
      obtain ⟨rfl, h2⟩ := item_split h
      obtain ⟨rfl, rfl⟩ := rest_inj ps ps' X X' h2
      exact ⟨rfl, rfl⟩

/-! ## The binding -/

/-- The binding's canonical form names its epoch, its families and its scheme. -/
theorem binding_inj {e e' : Nat} {fs fs' : List (String × Nat)} {s s' : String}
    (h : bindStr e fs s = bindStr e' fs' s') : e = e' ∧ fs = fs' ∧ s = s' := by
  have h := congrArg String.toList h
  simp only [bindStr, String.toList_append, String.toList_intercalate, List.map_cons, List.map_nil,
    List.append_assoc] at h
  simp only [show "{".toList = ['{'] from rfl, show "}".toList = ['}'] from rfl, show ":".toList = [':'] from rfl,
    show ",".toList = [','] from rfl, List.intercalate_cons_cons, List.intercalate_singleton, List.cons_append,
    List.nil_append, List.append_assoc, List.cons.injEq, true_and, List.append_cancel_left_eq] at h
  obtain ⟨he, -, h⟩ := nat_split (by decide) (by decide) h
  simp only [List.append_cancel_left_eq, List.cons.injEq, true_and] at h
  obtain ⟨hfs, h⟩ := fam_split h
  simp only [List.cons.injEq, true_and, List.append_cancel_left_eq] at h
  obtain ⟨hs, -⟩ := canonStr_split h
  exact ⟨he, hfs, hs⟩

end Properties.Pouw.LiveSalt.Canon
