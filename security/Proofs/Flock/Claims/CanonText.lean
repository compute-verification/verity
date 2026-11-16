module

public import Batteries.Tactic.OpenPrivate
public import Base.Canon
public import Std.Data.String.ToNat
import all Init.Data.Int.Repr
import all Init.Data.Ord.String
import all Lean.Data.Json.Basic

/-!
# What `Flock.canonUtf8` prints

A list and an object, unfolded (`canon_arr`, `canon_obj`). A string: each character is itself or one escape (`esc`),
each escape decodes to its character (`unesc_esc`), so an escaped string ends at its first unescaped quote
(`canonStr_split`). A numeral: a decimal numeral ends at its first non-digit (`digits_split`, `nat_split`).
-/

@[expose] public section

open private hex4 from Base.Canon

namespace Properties.Pouw.LiveSalt.Canon

open Lean (Json)

/-! ## `canonUtf8`, unfolded -/

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

end Properties.Pouw.LiveSalt.Canon
