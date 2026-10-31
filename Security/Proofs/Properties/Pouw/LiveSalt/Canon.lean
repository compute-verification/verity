import Batteries.Tactic.OpenPrivate
import Properties.Pouw.LiveSalt
import Std.Data.String.ToNat

/-!
# The coins' binding has one canonical form per layout

`liveDomain` hashes the identity bytes (`Flock.canonUtf8`) of the binding `{epoch, families, scheme}`. Its form is
`{"epoch":E,"families":F,"scheme":S}` (`canon_binding`): `canon` sorts the three keys, and they are already in order
(`qsort3`). That string parses back uniquely (`binding_inj`): a decimal numeral ends at the first non-digit
(`digits_split`), an escaped string at its first unescaped quote, each escape decoding to one character (`unesc_esc`,
`canonStr_split`), and the families' list item by item, each a string and a numeral (`fam_split`).
-/

open private hex4 from Flock.Canon
open private Array.qsort.sort Array.qpartition.loop from Init.Data.Array.QSort.Basic

namespace Properties.Pouw.LiveSalt.Canon

open Lean (Json)
open SampledProofs.Challenger (familiesJson)

/-! ## Sorting three keys in order -/

theorem loop02 {β : Type} (lt : β → β → Bool) (x y z : β) (hxy : lt x y = true) (hzy : lt z y = false) :
    Array.qpartition.loop (n := 3) lt 0 2 (by omega) y #v[x, z, y] 0 0 = (⟨1, by omega⟩, #v[x, y, z]) := by
  rw [Array.qpartition.loop.eq_def]
  simp (config := {decide := true}) [hxy]
  rw [Array.qpartition.loop.eq_def]
  simp (config := {decide := true}) [hzy]
  rw [Array.qpartition.loop.eq_def]
  simp (config := {decide := true})

theorem loop01 {β : Type} (lt : β → β → Bool) (x y z : β) (hyx : lt y x = false) :
    Array.qpartition.loop (n := 3) lt 0 1 (by omega) x #v[y, x, z] 0 0 = (⟨0, by omega⟩, #v[x, y, z]) := by
  rw [Array.qpartition.loop.eq_def]
  simp (config := {decide := true}) [hyx]
  rw [Array.qpartition.loop.eq_def]
  simp (config := {decide := true})

theorem sort_self {β : Type} (lt : β → β → Bool) {n : Nat} (v : Vector β n) (i : Nat) (h : i < n) :
    Array.qsort.sort lt v i i (by omega) h h = v := by
  rw [Array.qsort.sort.eq_def]
  simp

theorem sort01 {β : Type} (lt : β → β → Bool) (x y z : β) (hxx : lt x x = false) (hyx : lt y x = false)
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

theorem canon_arr (a : Array Json) : Flock.canonUtf8 (.arr a) =
    (do let parts ← a.mapM Flock.canonUtf8; pure ("[" ++ ",".intercalate parts.toList ++ "]")) := by
  rw [Flock.canonUtf8]
  simp only [Array.mapM_subtype, Array.unattach_attach]

theorem canon_obj (kvs : Std.TreeMap.Raw String Json) : Flock.canonUtf8 (.obj kvs) =
    (do
      let vals ← kvs.toArray.mapM fun p => do pure (p.1, ← Flock.canonUtf8 p.2)
      let items := vals.qsort (fun a b => a.1 < b.1)
      pure ("{" ++ ",".intercalate (items.toList.map fun kv => Flock.canonStrUtf8 kv.1 ++ ":" ++ kv.2) ++ "}")) := by
  rw [Flock.canonUtf8]
  simp only [Array.mapM_subtype, Array.unattach_attach]

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

/-! ## Escaped strings -/

/-- What `canonStrUtf8` writes for one character. -/
def esc (c : Char) : List Char :=
  match c with
  | '"' => ['\\', '"']
  | '\\' => ['\\', '\\']
  | '\n' => ['\\', 'n']
  | '\r' => ['\\', 'r']
  | '\t' => ['\\', 't']
  | c => if c.toNat == 8 then ['\\', 'b'] else if c.toNat == 12 then ['\\', 'f']
    else if c.toNat < 32 then '\\' :: 'u' :: (hex4 c.toNat).toList else [c]

theorem foldl_esc (f : String → Char → String) (hf : ∀ b a, (f b a).toList = b.toList ++ esc a) :
    ∀ (cs : List Char) (out : String), (cs.foldl f out).toList = out.toList ++ cs.flatMap esc := by
  intro cs
  induction cs with
  | nil => intro out; simp
  | cons c cs ih => intro out; rw [List.foldl_cons, ih, hf, List.flatMap_cons, List.append_assoc]

theorem canonStr_toList (s : String) : (Flock.canonStrUtf8 s).toList = '"' :: s.toList.flatMap esc ++ ['"'] := by
  unfold Flock.canonStrUtf8
  simp only [Id.run, List.forIn_pure_yield_eq_foldl]
  show (List.foldl _ "\"" s.toList ++ "\"").toList = _
  rw [String.toList_append, foldl_esc]
  · simp
  · intro b a
    unfold esc
    split
    all_goals first
      | (simp; done)
      | ((repeat' split) <;> simp_all)

/-- A hex digit's value. -/
def nib (c : Char) : Nat := if c.toNat < 87 then c.toNat - 48 else c.toNat - 87

/-- A hex numeral's value. -/
def unhexL (l : List Char) : Nat := l.foldl (fun acc c => 16 * acc + nib c) 0

theorem hex4_ok : ∀ n, n < 32 → (hex4 n).toList.length = 4 ∧ unhexL (hex4 n).toList = n := by
  decide

/-- Reads one escape's character after its backslash. -/
def unesc2 : List Char → Option (Char × List Char)
  | '"' :: r => some ('"', r)
  | '\\' :: r => some ('\\', r)
  | 'n' :: r => some ('\n', r)
  | 'r' :: r => some ('\r', r)
  | 't' :: r => some ('\t', r)
  | 'b' :: r => some (Char.ofNat 8, r)
  | 'f' :: r => some (Char.ofNat 12, r)
  | 'u' :: a :: b :: c :: d :: r => some (Char.ofNat (unhexL [a, b, c, d]), r)
  | _ => none

/-- Reads one character of an escaped string. -/
def unesc : List Char → Option (Char × List Char)
  | [] => none
  | c :: r => if c = '\\' then unesc2 r else some (c, r)

theorem unesc_esc (c : Char) (r : List Char) : unesc (esc c ++ r) = some (c, r) := by
  unfold esc
  split
  all_goals first
    | (simp [unesc, unesc2]; done)
    | skip
  rename_i h1 h2 h3 h4 h5
  by_cases h8 : c.toNat = 8
  · obtain rfl : c = Char.ofNat 8 := by rw [← h8, Char.ofNat_toNat]
    simp (config := {decide := true}) [unesc, unesc2]
  by_cases h12 : c.toNat = 12
  · obtain rfl : c = Char.ofNat 12 := by rw [← h12, Char.ofNat_toNat]
    simp (config := {decide := true}) [unesc, unesc2]
  by_cases h32 : c.toNat < 32
  · simp only [beq_iff_eq, h8, h12, h32, ↓reduceIte]
    obtain ⟨hl, hv⟩ := hex4_ok _ h32
    generalize (hex4 c.toNat).toList = xs at hl hv ⊢
    match xs, hl with
    | [a1, a2, a3, a4], _ => simp [unesc, unesc2, hv, Char.ofNat_toNat]
  · simp only [beq_iff_eq, h8, h12, h32, ↓reduceIte, List.singleton_append, unesc]
    exact ite_eq_right h2

theorem escs_inj : ∀ (cs cs' r r' : List Char), cs.flatMap esc ++ '"' :: r = cs'.flatMap esc ++ '"' :: r' →
    cs = cs' ∧ r = r' := by
  intro cs
  induction cs with
  | nil =>
    intro cs' r r' h
    cases cs' with
    | nil => simpa using h
    | cons c' cs' =>
      exfalso
      simp only [List.flatMap_nil, List.nil_append, List.flatMap_cons, List.append_assoc] at h
      have hu := congrArg unesc h
      rw [unesc_esc] at hu
      simp only [unesc] at hu
      obtain ⟨rfl, -⟩ := Prod.mk.inj (Option.some.inj hu)
      simp [esc] at h
  | cons c cs ih =>
    intro cs' r r' h
    cases cs' with
    | nil =>
      exfalso
      simp only [List.flatMap_nil, List.nil_append, List.flatMap_cons, List.append_assoc] at h
      have hu := congrArg unesc h
      rw [unesc_esc] at hu
      simp only [unesc] at hu
      obtain ⟨rfl, -⟩ := Prod.mk.inj (Option.some.inj hu)
      simp [esc] at h
    | cons c' cs' =>
      simp only [List.flatMap_cons, List.append_assoc] at h
      have hu := congrArg unesc h
      rw [unesc_esc, unesc_esc] at hu
      obtain ⟨rfl, h2⟩ := Prod.mk.inj (Option.some.inj hu)
      obtain ⟨rfl, rfl⟩ := ih cs' r r' h2
      exact ⟨rfl, rfl⟩

/-- An escaped string ends where it ends: what follows it doesn't change which string it is. -/
theorem canonStr_split {s s' : String} {r r' : List Char}
    (h : (Flock.canonStrUtf8 s).toList ++ r = (Flock.canonStrUtf8 s').toList ++ r') : s = s' ∧ r = r' := by
  simp only [canonStr_toList, List.cons_append, List.append_assoc, List.cons.injEq, true_and] at h
  obtain ⟨h1, h2⟩ := escs_inj _ _ _ _ h
  exact ⟨String.toList_inj.mp h1, h2⟩

/-! ## Numbers -/

theorem digits_split : ∀ (D D' : List Char) (c c' : Char) (r r' : List Char), (∀ x ∈ D, x.isDigit = true) →
    (∀ x ∈ D', x.isDigit = true) → c.isDigit = false → c'.isDigit = false → D ++ c :: r = D' ++ c' :: r' →
    D = D' ∧ c = c' ∧ r = r' := by
  intro D
  induction D with
  | nil =>
    intro D' c c' r r' _ hD' hc _ h
    cases D' with
    | nil => simpa using h
    | cons d D' =>
      simp only [List.nil_append, List.cons_append, List.cons.injEq] at h
      have := hD' d (by simp)
      rw [← h.1] at this
      simp [hc] at this
  | cons d D ih =>
    intro D' c c' r r' hD hD' hc hc' h
    cases D' with
    | nil =>
      simp only [List.nil_append, List.cons_append, List.cons.injEq] at h
      have := hD d (by simp)
      rw [h.1] at this
      simp [hc'] at this
    | cons d' D' =>
      simp only [List.cons_append, List.cons.injEq] at h
      obtain ⟨rfl, h2⟩ := h
      obtain ⟨rfl, rfl, rfl⟩ := ih D' c c' r r' (fun x hx => hD x (by simp [hx]))
        (fun x hx => hD' x (by simp [hx])) hc hc' h2
      exact ⟨rfl, rfl, rfl⟩

theorem toString_toList (n : Nat) : (toString n).toList = Nat.toDigits 10 n := Nat.toList_repr

/-- A numeral ends at the first non-digit. -/
theorem nat_split {n n' : Nat} {c c' : Char} {r r' : List Char} (hc : c.isDigit = false) (hc' : c'.isDigit = false)
    (h : (toString n).toList ++ c :: r = (toString n').toList ++ c' :: r') : n = n' ∧ c = c' ∧ r = r' := by
  rw [toString_toList, toString_toList] at h
  obtain ⟨h1, h2, h3⟩ := digits_split _ _ _ _ _ _ (fun _ hx => Nat.isDigit_of_mem_toDigits (by decide) (by decide) hx)
    (fun _ hx => Nat.isDigit_of_mem_toDigits (by decide) (by decide) hx) hc hc' h
  refine ⟨Nat.repr_injective ?_, h2, h3⟩
  rw [← String.toList_inj, Nat.toList_repr, Nat.toList_repr, h1]

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
