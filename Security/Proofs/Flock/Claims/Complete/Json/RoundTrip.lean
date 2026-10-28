import Flock.JsonParse
import Flock.Canon
import Proofs.Flock.Claims.Complete.Json.Value
import Proofs.Flock.Claims.Complete.Prover.Print
import Std.Data.TreeMap.Raw.Lemmas
import Std.Data.TreeMap.Raw.WF
import Batteries.Tactic.OpenPrivate

/-!
# The verifier's parser reads back what the printers print

`Flock.JsonParse.parse`, the verifier's total parser, reads back an alike value (`Same`) from the honest prover's printer
`FlockComplete.printJ` of a printable value (`Good`), and from the verifier's canonical printer `Flock.canon`
(`parse_print`, `parse_canon`).

The proof is one relation, `ReadBack.Pr j t`: a text `t` that `value` reads as a value alike to `j`, whatever follows it
up to a separator (`Stop`). Both printers print a text in it (`print_pr`, `canon_pr`), and `pr_value` reads it, by
induction on `Pr`. A string's text is a run of escapes `str` reads back one character at a time (`Esc`, `esc_of`); a
number's is `Int.repr` (`num_int`); an object's fields are `Flock.canon`'s keys in `Array.qsort`'s order, a permutation
of the tree's own (`qsort_perm`), and `object` inserts them into a tree alike at every key (`Inv`, `obj_loop`).
-/

open private Flock.hex4 from Flock.Canon
open private Array.qsort.sort Array.qpartition.loop from Init.Data.Array.QSort.Basic

namespace FlockComplete.JsonRT.ReadBack

open Lean (Json JsonNumber)
open Flock.JsonParse
open Claims.Json

/-! ## `Array.qsort` permutes -/

theorem loop_perm {α} {n} (lt : α → α → Bool) (lo hi : Nat) (hhi : hi < n) (pivot : α) :
    ∀ (as : Vector α n) (i k : Nat) (ilo : lo ≤ i) (ik : i ≤ k) (w : k ≤ hi),
    (Array.qpartition.loop lt lo hi hhi pivot as i k ilo ik w).2.Perm as := by
  intro as i k ilo ik w
  induction h : hi - k generalizing as i k with
  | zero =>
    rw [Array.qpartition.loop.eq_def]
    split
    · omega
    · exact Vector.swap_perm (by omega) hhi
  | succ d ih =>
    rw [Array.qpartition.loop.eq_def]
    split
    · split
      · exact (ih _ _ _ _ _ _ (by omega)).trans (Vector.swap_perm (by omega) (by omega))
      · exact ih _ _ _ _ _ _ (by omega)
    · omega

theorem qpartition_perm {α} {n} (as : Vector α n) (lt : α → α → Bool) (lo hi : Nat) (w : lo ≤ hi)
    (hlo : lo < n) (hhi : hi < n) : (Array.qpartition as lt lo hi w hlo hhi).2.Perm as := by
  unfold Array.qpartition
  refine (loop_perm _ _ _ _ _ _ _ _ _ _ _).trans ?_
  have sw : ∀ (bs : Vector α n) (c : Bool) (i j : Nat) (hi : i < n) (hj : j < n),
      (if c then bs.swap i j hi hj else bs).Perm bs := fun bs c i j hi hj => by
    cases c
    · exact Vector.Perm.refl _
    · exact Vector.swap_perm hi hj
  exact ((sw _ _ _ _ _ _).trans (sw _ _ _ _ _ _)).trans (sw _ _ _ _ _ _)

theorem sort_perm {α} (lt : α → α → Bool) {n} :
    ∀ (as : Vector α n) (lo hi : Nat) (w : lo ≤ hi) (hlo : lo < n) (hhi : hi < n),
    (Array.qsort.sort lt as lo hi w hlo hhi).Perm as := by
  intro as lo hi w hlo hhi
  suffices ∀ F, hi - lo < F → (Array.qsort.sort lt as lo hi w hlo hhi).Perm as from this _ (Nat.lt_succ_self _)
  intro F hF
  induction F generalizing as lo hi with
  | zero => omega
  | succ F ih =>
    rw [Array.qsort.sort.eq_def]
    split
    · split
      rename_i mid hmid bs hp
      have hb : bs.Perm as := by
        have := qpartition_perm as lt lo hi w hlo hhi
        rw [hp] at this
        exact this
      split
      · exact hb
      · exact ((ih _ _ _ _ _ _ (by omega)).trans (ih _ _ _ _ _ _ (by omega))).trans hb
    · exact Vector.Perm.refl _

theorem qsort_perm {α} (as : Array α) (lt : α → α → Bool) : (as.qsort lt).Perm as := by
  unfold Array.qsort
  split
  · exact Array.Perm.refl _
  · rename_i h
    have := sort_perm lt as.toVector (min 0 (as.size - 1)) (max (min 0 (as.size - 1)) (min (as.size - 1) (as.size - 1)))
      (by omega) (by omega) (by omega)
    exact Vector.perm_iff_toArray_perm.1 this

/-! ## Texts the parser reads back -/

/-- A string's text between its quotes, which `str` reads back as the string. -/
def Esc (s : String) (t : List Char) : Prop :=
  ∀ k acc rest o, t.length < k → ∃ o', str k acc (t ++ '"' :: rest) o = .ok rest o' (acc ++ s)

/-- One character's text, which one step of `str` reads back as the character. -/
def EscC (c : Char) (t : List Char) : Prop :=
  t ≠ [] ∧ ∀ k acc tail o, ∃ o', str (k + 1) acc (t ++ tail) o = str k (acc.push c) tail o'

/-- Characters' texts, joined, read back as the characters. -/
theorem esc_join (e : Char → List Char) (he : ∀ c, EscC c (e c)) :
    ∀ (l : List Char) k (acc : String) rest o, (l.flatMap e).length < k →
      ∃ o', str k acc (l.flatMap e ++ '"' :: rest) o = .ok rest o' (acc ++ String.ofList l)
  | [], k + 1, acc, rest, o, _ => ⟨o + 1, by simp [str]⟩
  | c :: l, k + 1, acc, rest, o, hk => by
    obtain ⟨hne, hs⟩ := he c
    have hl : (e c).length ≥ 1 := by
      cases h : e c with
      | nil => exact absurd h hne
      | cons _ _ => simp
    simp only [List.flatMap_cons, List.length_append] at hk
    obtain ⟨o₁, h₁⟩ := hs k acc (l.flatMap e ++ '"' :: rest) o
    obtain ⟨o₂, h₂⟩ := esc_join e he l k (acc.push c) rest o₁ (by omega)
    refine ⟨o₂, ?_⟩
    rw [List.flatMap_cons, List.append_assoc, h₁, h₂]
    congr 1
    apply String.ext
    simp

theorem esc_of (e : Char → List Char) (he : ∀ c, EscC c (e c)) (s : String) : Esc s (s.toList.flatMap e) := by
  intro k acc rest o hk
  have := esc_join e he s.toList k acc rest o hk
  simpa using this

theorem hexChar_digit (d : Nat) (hd : d < 16) (cs : List Char) (o : Nat) :
    hexChar (Nat.digitChar d :: cs) o = .ok cs (o + 1) d := by
  match d, hd with
  | 0, _ | 1, _ | 2, _ | 3, _ | 4, _ | 5, _ | 6, _ | 7, _ | 8, _ | 9, _ | 10, _ | 11, _ | 12, _ | 13, _ | 14, _
  | 15, _ => rfl
  | _ + 16, h => exact absurd h (by omega)

theorem escC_esc (c : Char) : EscC c (FlockComplete.escC c).toList := by
  unfold FlockComplete.escC
  split
  · subst_vars; exact ⟨by simp, fun k acc tail o => ⟨_, rfl⟩⟩
  split
  · subst_vars; exact ⟨by simp, fun k acc tail o => ⟨_, rfl⟩⟩
  split
  · subst_vars; exact ⟨by simp, fun k acc tail o => ⟨_, rfl⟩⟩
  split
  · subst_vars; exact ⟨by simp, fun k acc tail o => ⟨_, rfl⟩⟩
  split
  · rename_i h1 h2 h3 h4 h5
    refine ⟨by simp, fun k acc tail o => ⟨o + c.utf8Size, ?_⟩⟩
    simp only [String.toList_singleton, List.singleton_append]
    rw [str]
    simp only [h1, h2, h5, ↓reduceIte, and_self]
  · rename_i h1 h2 h3 h4 h5
    have hc : c.toNat < 0x20 := by
      have := c.valid
      simp only [Char.toNat, UInt32.isValidChar, Nat.isValidChar] at this ⊢
      have h5' : ¬ (0x0020 ≤ c.val.toNat ∧ c.val.toNat ≤ 0x10ffff) := by
        intro ⟨a, b⟩; exact h5 ⟨UInt32.le_iff_toNat_le.2 (by simpa using a), UInt32.le_iff_toNat_le.2 (by simpa using b)⟩
      omega
    refine ⟨by simp, fun k acc tail o => ⟨o + 6, ?_⟩⟩
    simp only [String.toList_append, String.toList_ofList]
    have e : "\\u".toList = ['\\', 'u'] := rfl
    rw [e]
    simp only [List.cons_append, List.nil_append]
    have hx : ∀ x, x < 16 → ∀ cs o, hexChar (Nat.digitChar x :: cs) o = .ok cs (o + 1) x := hexChar_digit
    rw [str]
    simp only [escapedChar, hexN, hx _ (by omega : c.toNat / 4096 < 16), hx _ (by omega : c.toNat % 4096 / 256 < 16),
      hx _ (by omega : c.toNat % 256 / 16 < 16), hx _ (by omega : c.toNat % 16 < 16)]
    have hv : 16 * (16 * (16 * (16 * 0 + c.toNat / 4096) + c.toNat % 4096 / 256) + c.toNat % 256 / 16) +
        c.toNat % 16 = c.toNat := by omega
    rw [hv]
    simp only [show c.toNat < 55296 by omega, ↓reduceIte, Char.ofNat_toNat]
    rw [if_neg (by decide)]

/-- One character's text in `Flock.canonStr`. -/
def cesc (c : Char) : String :=
  match c with
  | '"' => "\\\""
  | '\\' => "\\\\"
  | '\n' => "\\n"
  | '\r' => "\\r"
  | '\t' => "\\t"
  | c => if c.toNat == 8 then "\\b" else if c.toNat == 12 then "\\f"
         else if c.toNat < 0x20 then "\\u" ++ Flock.hex4 c.toNat else String.singleton c

theorem foldl_esc (e : Char → List Char) (f : String → Char → String)
    (hf : ∀ b c, (f b c).toList = b.toList ++ e c) :
    ∀ (l : List Char) (b : String), (l.foldl f b).toList = b.toList ++ l.flatMap e
  | [], b => by simp
  | c :: l, b => by rw [List.foldl_cons, foldl_esc e f hf l, hf, List.flatMap_cons, List.append_assoc]

theorem canonStr_toList (s : String) :
    (Flock.canonStr s).toList = '"' :: s.toList.flatMap (fun c => (cesc c).toList) ++ ['"'] := by
  unfold Flock.canonStr
  simp only [List.forIn_pure_yield_eq_foldl, Id.run, pure_bind]
  show (_ ++ _ : String).toList = _
  rw [String.toList_append, foldl_esc (fun c => (cesc c).toList)]
  · rfl
  · intro b c
    unfold cesc
    split
    all_goals first
      | (simp_all; done)
      | (by_cases h8 : c.toNat = 8 <;> by_cases h12 : c.toNat = 12 <;> by_cases h32 : c.toNat < 32 <;>
           simp [h8, h12, h32])

theorem hexChar_canon (x : Nat) (hx : x < 16) (cs : List Char) (o : Nat) :
    hexChar ((if x < 10 then Char.ofNat (48 + x) else Char.ofNat (87 + x)) :: cs) o = .ok cs (o + 1) x := by
  match x, hx with
  | 0, _ | 1, _ | 2, _ | 3, _ | 4, _ | 5, _ | 6, _ | 7, _ | 8, _ | 9, _ | 10, _ | 11, _ | 12, _ | 13, _ | 14, _
  | 15, _ => rfl
  | _ + 16, h => exact absurd h (by omega)

theorem cesc_esc (c : Char) : EscC c (cesc c).toList := by
  unfold cesc
  split
  all_goals first
    | exact ⟨by simp, fun k acc tail o => ⟨_, rfl⟩⟩
    | skip
  rename_i h1 h2 h3 h4 h5
  have hcv : c.toNat < 0xd800 ∨ (0xdfff < c.toNat ∧ c.toNat < 0x110000) := c.valid
  by_cases h8 : c.toNat = 8
  · have : c = '\x08' := by rw [← Char.ofNat_toNat c, h8]
    subst this
    exact ⟨by simp, fun k acc tail o => ⟨_, rfl⟩⟩
  by_cases h12 : c.toNat = 12
  · have : c = '\x0c' := by rw [← Char.ofNat_toNat c, h12]
    subst this
    exact ⟨by simp, fun k acc tail o => ⟨_, rfl⟩⟩
  by_cases h32 : c.toNat < 32
  · simp only [h8, h12, h32, beq_iff_eq, ↓reduceIte]
    refine ⟨by simp, fun k acc tail o => ⟨o + 6, ?_⟩⟩
    unfold Flock.hex4
    simp only [String.toList_append, String.toList_ofList]
    have e : "\\u".toList = ['\\', 'u'] := rfl
    rw [e]
    simp only [List.cons_append, List.nil_append]
    have hx : ∀ x, x < 16 → ∀ cs o, hexChar ((if x < 10 then Char.ofNat (48 + x) else Char.ofNat (87 + x)) :: cs) o =
      .ok cs (o + 1) x := hexChar_canon
    rw [str]
    simp only [escapedChar, hexN, Nat.shiftRight_eq_div_pow,
      hx _ (Nat.mod_lt _ (by omega) : c.toNat / 2 ^ (4 * 3) % 16 < 16),
      hx _ (Nat.mod_lt _ (by omega) : c.toNat / 2 ^ (4 * 2) % 16 < 16),
      hx _ (Nat.mod_lt _ (by omega) : c.toNat / 2 ^ (4 * 1) % 16 < 16),
      hx _ (Nat.mod_lt _ (by omega) : c.toNat / 2 ^ (4 * 0) % 16 < 16)]
    simp only [Nat.reducePow, Nat.reduceMul]
    have hv : 16 * (16 * (16 * (16 * 0 + c.toNat / 4096 % 16) + c.toNat / 256 % 16) + c.toNat / 16 % 16) +
        c.toNat / 1 % 16 = c.toNat := by omega
    rw [hv]
    simp only [show c.toNat < 55296 by omega, ↓reduceIte, Char.ofNat_toNat]
    rw [if_neg (by decide)]
  · have h1' : c ≠ '"' := h1
    have h2' : c ≠ '\\' := h2
    have hv : 0x0020 ≤ c.val ∧ c.val ≤ 0x10ffff := by
      have e : c.toNat = c.val.toNat := rfl
      rw [e] at h32 hcv
      exact ⟨UInt32.le_iff_toNat_le.2 (by simp; omega), UInt32.le_iff_toNat_le.2 (by simp; omega)⟩
    simp only [h8, h12, h32, beq_iff_eq, ↓reduceIte]
    refine ⟨by simp, fun k acc tail o => ⟨o + c.utf8Size, ?_⟩⟩
    simp only [String.toList_singleton, List.singleton_append]
    rw [str]
    simp only [h1', h2', hv, ↓reduceIte, and_self]

/-- What can follow a value: nothing, or the comma or bracket after it. -/
def Stop (rest : List Char) : Prop := rest = [] ∨ ∃ c cs, rest = c :: cs ∧ (c = ',' ∨ c = ']' ∨ c = '}')

def dval (a : Nat) (c : Char) : Nat := 10 * a + (c.val - '0'.val).toNat

theorem digitChar_dig (d : Nat) (hd : d < 10) :
    ('0' ≤ Nat.digitChar d ∧ Nat.digitChar d ≤ '9') ∧ ((Nat.digitChar d).val - '0'.val).toNat = d := by
  match d, hd with
  | 0, _ | 1, _ | 2, _ | 3, _ | 4, _ | 5, _ | 6, _ | 7, _ | 8, _ | 9, _ => decide
  | _ + 10, h => exact absurd h (by omega)

theorem toDigits_succ (n : Nat) (h : ¬ n < 10) :
    Nat.toDigits 10 n = Nat.toDigits 10 (n / 10) ++ [Nat.digitChar (n % 10)] := by
  rw [Nat.toDigits_eq_ite (by decide), if_neg h]

theorem toDigits_lt (n : Nat) (h : n < 10) : Nat.toDigits 10 n = [Nat.digitChar n] := by
  rw [Nat.toDigits_eq_ite (by decide), if_pos h]

/-- A number's digits: digits, their value the number, the first nonzero unless the number is. -/
theorem toDigits_facts : ∀ F n, n < F →
    (∀ c ∈ Nat.toDigits 10 n, '0' ≤ c ∧ c ≤ '9') ∧ (Nat.toDigits 10 n).foldl dval 0 = n ∧
    (n ≠ 0 → ∃ d cs, 0 < d ∧ d < 10 ∧ Nat.toDigits 10 n = Nat.digitChar d :: cs)
  | 0, _, h => absurd h (Nat.not_lt_zero _)
  | F + 1, n, hF => by
    by_cases h : n < 10
    · rw [toDigits_lt n h]
      refine ⟨fun c hc => ?_, ?_, fun hn => ⟨n, [], by omega, h, rfl⟩⟩
      · simp only [List.mem_singleton] at hc; subst hc; exact (digitChar_dig n h).1
      · have := (digitChar_dig n h).2
        simp only [List.foldl_cons, List.foldl_nil, dval]
        omega
    · obtain ⟨h1, h2, h3⟩ := toDigits_facts F (n / 10) (by omega)
      rw [toDigits_succ n h]
      refine ⟨fun c hc => ?_, ?_, fun _ => ?_⟩
      · simp only [List.mem_append, List.mem_singleton] at hc
        rcases hc with hc | hc
        · exact h1 c hc
        · subst hc; exact (digitChar_dig _ (Nat.mod_lt _ (by omega))).1
      · rw [List.foldl_append, h2]
        simp only [List.foldl_cons, List.foldl_nil, dval, (digitChar_dig _ (Nat.mod_lt _ (by omega))).2]
        omega
      · obtain ⟨d, cs, hd0, hd, he⟩ := h3 (by omega)
        exact ⟨d, cs ++ [Nat.digitChar (n % 10)], hd0, hd, by rw [he]; rfl⟩

theorem stop_not_digit {rest : List Char} (h : Stop rest) :
    rest = [] ∨ ∃ c cs, rest = c :: cs ∧ ¬ ('0' ≤ c ∧ c ≤ '9') ∧ c ≠ '.' ∧ c ≠ 'e' ∧ c ≠ 'E' := by
  rcases h with h | ⟨c, cs, h, hc⟩
  · exact .inl h
  · refine .inr ⟨c, cs, h, ?_⟩
    rcases hc with rfl | rfl | rfl <;> decide

theorem digits_app : ∀ (l rest : List Char) o acc n, (∀ c ∈ l, '0' ≤ c ∧ c ≤ '9') →
    (rest = [] ∨ ∃ c cs, rest = c :: cs ∧ ¬ ('0' ≤ c ∧ c ≤ '9')) →
    digits (l ++ rest) o acc n = (rest, o + l.length, l.foldl dval acc, n + l.length)
  | [], rest, o, acc, n, _, hr => by
    rcases hr with rfl | ⟨c, cs, rfl, hc⟩
    · rfl
    · simp [digits, hc]
  | c :: l, rest, o, acc, n, hl, hr => by
    simp only [List.cons_append, digits, hl c (List.mem_cons_self ..), and_self, ↓reduceIte]
    rw [digits_app l rest _ _ _ (fun c hc => hl c (List.mem_cons_of_mem _ hc)) hr]
    simp only [List.length_cons, List.foldl_cons, dval, Prod.mk.injEq]
    exact ⟨trivial, by omega, trivial, by omega⟩

theorem natAfter_digits (p : Char → Bool) (why : String) (l rest : List Char) (o : Nat)
    (hl : ∀ c ∈ l, '0' ≤ c ∧ c ≤ '9') (hp : ∀ c cs, l = c :: cs → p c = true) (hne : l ≠ [])
    (hr : rest = [] ∨ ∃ c cs, rest = c :: cs ∧ ¬ ('0' ≤ c ∧ c ≤ '9')) :
    natAfter p why (l ++ rest) o = .ok rest (o + l.length) (l.foldl dval 0, l.length) := by
  cases l with
  | nil => exact absurd rfl hne
  | cons c cs =>
    rw [List.cons_append, natAfter, if_pos (hp c cs rfl), ← List.cons_append, digits_app _ _ _ _ _ hl hr]
    simp

theorem stop_digits {rest : List Char} (hs : Stop rest) :
    rest = [] ∨ ∃ c cs, rest = c :: cs ∧ ¬ ('0' ≤ c ∧ c ≤ '9') := by
  rcases hs with h | ⟨c, cs, h, hc⟩
  · exact .inl h
  · exact .inr ⟨c, cs, h, by rcases hc with rfl | rfl | rfl <;> decide⟩

/-- A nonzero number's digits, read by `natNonZero`. -/
theorem natAfter_pos (n : Nat) (hn : n ≠ 0) (rest : List Char) (hs : Stop rest) (o : Nat) :
    ∃ d cs, 0 < d ∧ d < 10 ∧ Nat.toDigits 10 n = Nat.digitChar d :: cs ∧
      natAfter (fun c => decide ('1' ≤ c ∧ c ≤ '9')) "1-9" (Nat.digitChar d :: (cs ++ rest)) o =
        .ok rest (o + (Nat.toDigits 10 n).length) (n, (Nat.toDigits 10 n).length) := by
  obtain ⟨hd, hv, hh⟩ := toDigits_facts (n + 1) n (by omega)
  obtain ⟨d, cs, hd0, hd10, he⟩ := hh hn
  refine ⟨d, cs, hd0, hd10, he, ?_⟩
  have hna := natAfter_digits (fun c => decide ('1' ≤ c ∧ c ≤ '9')) "1-9" _ rest o hd
    (fun c' cs' h => by
      rw [he] at h; injection h with h; subst h
      match d, hd0, hd10 with
      | 1, _, _ | 2, _, _ | 3, _, _ | 4, _, _ | 5, _, _ | 6, _, _ | 7, _, _ | 8, _, _ | 9, _, _ => decide
      | 0, h, _ => exact absurd h (by omega)
      | _ + 10, _, h => exact absurd h (by omega))
    (by rw [he]; simp) (stop_digits hs)
  rw [hv] at hna
  rw [← List.cons_append, ← he]
  exact hna

theorem digitChar_ne_zero (d : Nat) (hd0 : 0 < d) (hd10 : d < 10) : Nat.digitChar d ≠ '0' := by
  match d, hd0, hd10 with
  | 1, _, _ | 2, _, _ | 3, _, _ | 4, _, _ | 5, _, _ | 6, _, _ | 7, _, _ | 8, _, _ | 9, _, _ => decide
  | 0, h, _ => exact absurd h (by omega)
  | _ + 10, _, h => exact absurd h (by omega)

theorem num_nat (n : Nat) (rest : List Char) (hs : Stop rest) (o : Nat) :
    ∃ o', num (Nat.toDigits 10 n ++ rest) o = .ok rest o' ⟨n, 0⟩ := by
  by_cases hn : n = 0
  · subst hn
    refine ⟨o + 1, ?_⟩
    rw [show Nat.toDigits 10 0 = ['0'] from rfl, List.cons_append, List.nil_append]
    rcases hs with rfl | ⟨c, cs, rfl, rfl | rfl | rfl⟩ <;> rfl
  · obtain ⟨d, cs, hd0, hd10, he, hna⟩ := natAfter_pos n hn rest hs o
    have hc := (digitChar_dig d hd10).1
    refine ⟨o + (Nat.toDigits 10 n).length, ?_⟩
    generalize (Nat.toDigits 10 n).length = L at hna ⊢
    rw [he, List.cons_append]
    unfold num
    split
    rename_i h
    split at h
    · rename_i h'; injection h' with h'; rw [h'] at hc; exact absurd hc (by decide)
    injection h with h3 h4; injection h4 with h4 h5; subst h3 h4 h5
    split
    · rename_i h'; injection h'
    · rename_i h'; injection h' with h'; exact absurd h' (digitChar_ne_zero d hd0 hd10)
    · rw [hna]
      simp only [Int.one_mul]
      rcases hs with rfl | ⟨c, cs', rfl, rfl | rfl | rfl⟩ <;> rfl

theorem num_neg (k : Nat) (rest : List Char) (hs : Stop rest) (o : Nat) :
    ∃ o', num ('-' :: Nat.toDigits 10 (k + 1) ++ rest) o = .ok rest o' ⟨Int.negSucc k, 0⟩ := by
  obtain ⟨d, cs, hd0, hd10, he, hna⟩ := natAfter_pos (k + 1) (by omega) rest hs (o + 1)
  have hc := (digitChar_dig d hd10).1
  refine ⟨o + 1 + (Nat.toDigits 10 (k + 1)).length, ?_⟩
  generalize (Nat.toDigits 10 (k + 1)).length = L at hna ⊢
  rw [he]
  unfold num
  split
  rename_i h
  split at h
  · injection h with h3 h4; injection h4 with h4 h5; subst h3 h4 h5
    rename_i h'
    injection h' with _ h'
    subst h'
    split
    · rename_i h'; injection h'
    · rename_i h'; injection h' with h'; exact absurd h' (digitChar_ne_zero d hd0 hd10)
    · simp only [List.append_eq, List.cons_append] at hna ⊢
      rw [hna]
      simp only [Int.neg_one_mul]
      rcases hs with rfl | ⟨c, cs', rfl, rfl | rfl | rfl⟩ <;> rfl
  · rename_i h'; exact absurd rfl (h' _)

theorem num_int (m : Int) (rest : List Char) (hs : Stop rest) (o : Nat) :
    ∃ o', num ((Int.repr m).toList ++ rest) o = .ok rest o' ⟨m, 0⟩ := by
  cases m with
  | ofNat n => simpa [Int.repr, Nat.repr] using num_nat n rest hs o
  | negSucc k =>
    have := num_neg k rest hs o
    simp only [Int.repr, Nat.repr, String.toList_append, String.toList_ofList] at this ⊢
    exact this

/-- A field's text: its key between quotes, a colon, its value. -/
def fieldT (f : String × Json × List Char × List Char) : List Char := '"' :: f.2.2.1 ++ '"' :: ':' :: f.2.2.2

/-- **`t` is a text of `j`**: the forms both printers print, which the parser reads back alike. -/
inductive Pr : Json → List Char → Prop
  | null : Pr .null ['n', 'u', 'l', 'l']
  | tru : Pr (.bool true) ['t', 'r', 'u', 'e']
  | fls : Pr (.bool false) ['f', 'a', 'l', 's', 'e']
  | num (m : Int) : Pr (.num ⟨m, 0⟩) (Int.repr m).toList
  | str (s : String) (t : List Char) (h : Esc s t) : Pr (.str s) ('"' :: t ++ ['"'])
  | arr (l : List Json) (ts : List (List Char)) (hl : ts.length = l.length) (h : ∀ p ∈ l.zip ts, Pr p.1 p.2) :
      Pr (.arr ⟨l⟩) ('[' :: [','].intercalate ts ++ [']'])
  | obj (m : Std.TreeMap.Raw String Json) (hm : m.WF) (fs : List (String × Json × List Char × List Char))
      (hget : ∀ f ∈ fs, m.get? f.1 = some f.2.1) (hkeys : (fs.map (·.1)).Perm (m.toList.map (·.1)))
      (hk : ∀ f ∈ fs, Esc f.1 f.2.2.1) (hv : ∀ f ∈ fs, Pr f.2.1 f.2.2.2) :
      Pr (.obj m) ('{' :: [','].intercalate (fs.map fieldT) ++ ['}'])

/-- A value's first character. -/
def Lead (c : Char) : Prop :=
  c = 'n' ∨ c = 't' ∨ c = 'f' ∨ c = '"' ∨ c = '[' ∨ c = '{' ∨ c = '-' ∨ ('0' ≤ c ∧ c ≤ '9')

theorem toDigits_ne (n : Nat) : ∃ c cs, Nat.toDigits 10 n = c :: cs ∧ '0' ≤ c ∧ c ≤ '9' := by
  obtain ⟨hd, -, -⟩ := toDigits_facts (n + 1) n (by omega)
  cases h : Nat.toDigits 10 n with
  | nil => exact absurd h (Nat.toDigits_ne_nil)
  | cons c cs => exact ⟨c, cs, rfl, hd c (by rw [h]; exact List.mem_cons_self ..)⟩

theorem pr_lead {j : Json} {t : List Char} (h : Pr j t) : ∃ c cs, t = c :: cs ∧ Lead c := by
  cases h with
  | null => exact ⟨_, _, rfl, by unfold Lead; decide⟩
  | tru => exact ⟨_, _, rfl, by unfold Lead; decide⟩
  | fls => exact ⟨_, _, rfl, by unfold Lead; decide⟩
  | num m =>
    cases m with
    | ofNat n =>
      obtain ⟨c, cs, he, hc⟩ := toDigits_ne n
      exact ⟨c, cs, by simp [Int.repr, Nat.repr, he], by unfold Lead; simp only [hc, and_self, or_true]⟩
    | negSucc n =>
      exact ⟨'-', Nat.toDigits 10 (n + 1), by simp [Int.repr, Nat.repr], by unfold Lead; decide⟩
  | str => exact ⟨_, _, rfl, by unfold Lead; decide⟩
  | arr => exact ⟨_, _, rfl, by unfold Lead; decide⟩
  | obj => exact ⟨_, _, rfl, by unfold Lead; decide⟩

theorem lead_ws {c : Char} (h : Lead c) (cs : List Char) (o : Nat) : ws (c :: cs) o = (c :: cs, o) := by
  unfold Lead at h
  have : ¬ (c = '\t' ∨ c = '\n' ∨ c = '\r' ∨ c = ' ') := by
    rcases h with rfl | rfl | rfl | rfl | rfl | rfl | rfl | h <;> try decide
    intro h'; rcases h' with rfl | rfl | rfl | rfl <;> exact absurd h (by decide)
  rw [ws, if_neg this]

theorem lead_ne {c : Char} (h : Lead c) : c ≠ ']' ∧ c ≠ '}' := by
  unfold Lead at h
  rcases h with rfl | rfl | rfl | rfl | rfl | rfl | rfl | h <;> try decide
  constructor <;> (rintro rfl; exact absurd h (by decide))

theorem stop_ws {rest : List Char} (hs : Stop rest) (o : Nat) : ws rest o = (rest, o) := by
  rcases hs with rfl | ⟨c, cs, rfl, rfl | rfl | rfl⟩ <;> rfl

/-- `value` reads `t` back as a value alike to `j`, whatever stops it. -/
def VP (j : Json) (t : List Char) : Prop :=
  ∀ k rest o, Stop rest → t.length < k → ∃ o' j', value k (t ++ rest) o = .ok rest o' j' ∧ Same j j'

theorem arr_loop : ∀ (l : List Json) (ts : List (List Char)), ts.length = l.length → l ≠ [] →
    (∀ p ∈ l.zip ts, VP p.1 p.2 ∧ ∃ c cs, p.2 = c :: cs ∧ Lead c) →
    ∀ k acc rest o, Stop rest → ([','].intercalate ts ++ [']']).length < k →
    ∃ o' ys, array k acc ([','].intercalate ts ++ ']' :: rest) o = .ok rest o' (acc ++ ys.toArray) ∧
      ys.length = l.length ∧ ∀ p ∈ l.zip ys, Same p.1 p.2
  | [], _, _, hne, _, _, _, _, _, _, _ => absurd rfl hne
  | [x], [t], _, _, hp, k + 1, acc, rest, o, hs, hk => by
    obtain ⟨hv, -⟩ := hp (x, t) (by simp)
    dsimp only at hv
    simp only [List.intercalate_singleton, List.length_append, List.length_singleton] at hk
    obtain ⟨o₁, y, hy, hs'⟩ := hv k (']' :: rest) o (.inr ⟨_, _, rfl, by simp⟩) (by omega)
    refine ⟨o₁ + 1, [y], ?_, rfl, by simpa using hs'⟩
    simp only [List.intercalate_singleton]
    rw [array, hy]
    simp only [↓reduceIte, stop_ws hs]
    congr 1
  | x :: x' :: l, t :: t' :: ts, hl, _, hp, k + 1, acc, rest, o, hs, hk => by
    obtain ⟨hv, -⟩ := hp (x, t) (by simp)
    obtain ⟨-, c, cs, hc, hlead⟩ := hp (x', t') (by simp)
    dsimp only at hv hc
    rw [List.intercalate_cons_cons] at hk ⊢
    simp only [List.length_append, List.length_singleton] at hk
    obtain ⟨X, hX⟩ : ∃ X, [','].intercalate (t' :: ts) = t' ++ X := by
      cases ts with
      | nil => exact ⟨[], by simp⟩
      | cons _ _ => exact ⟨_, by rw [List.intercalate_cons_cons, List.append_assoc]⟩
    obtain ⟨o₁, y, hy, hs'⟩ := hv k (',' :: ([','].intercalate (t' :: ts) ++ ']' :: rest)) o
      (.inr ⟨_, _, rfl, by simp⟩) (by omega)
    obtain ⟨o', ys, hys, hyl, hya⟩ := arr_loop (x' :: l) (t' :: ts) (by simpa using hl) (by simp)
      (fun p hp' => hp p (List.mem_cons_of_mem _ hp')) k (acc.push y) rest (o₁ + 1) hs
      (by simp only [List.length_append, List.length_singleton]; omega)
    refine ⟨o', y :: ys, ?_, by simpa using hyl, ?_⟩
    · rw [List.append_assoc, List.append_assoc, List.singleton_append, array, hy]
      have hw : ws ([','].intercalate (t' :: ts) ++ ']' :: rest) (o₁ + 1) =
          ([','].intercalate (t' :: ts) ++ ']' :: rest, o₁ + 1) := by
        rw [hX, hc, List.cons_append, List.cons_append, lead_ws hlead]
      simp only [show (',' : Char) ≠ ']' by decide, ↓reduceIte, hw, hys]
      congr 1
      apply Array.ext'
      simp
    · intro p hp'
      simp only [List.zip_cons_cons, List.mem_cons] at hp'
      rcases hp' with rfl | hp'
      · exact hs'
      · exact hya p hp'
  | [_], [], hl, _, _, _, _, _, _, _, _ => by simp at hl
  | [_], _ :: _ :: _, hl, _, _, _, _, _, _, _, _ => by simp at hl
  | _ :: _ :: _, [], hl, _, _, _, _, _, _, _, _ => by simp at hl
  | _ :: _ :: _, [_], hl, _, _, _, _, _, _, _, _ => by simp at hl
  | _, _, _, _, _, 0, _, _, _, _, hk => absurd hk (Nat.not_lt_zero _)

theorem same_arr (l ys : List Json) (hl : ys.length = l.length) (h : ∀ p ∈ l.zip ys, Same p.1 p.2) :
    Same (.arr ⟨l⟩) (.arr ⟨ys⟩) := by
  refine .arr (by simp [hl]) fun i h₁ h₂ => ?_
  have hi : (l.zip ys)[i]'(by simp [hl]; simpa using h₁) ∈ l.zip ys := List.getElem_mem _
  simpa [List.getElem_zip] using h _ hi

/-- What the parser has built of an object alike to `m`: a well-formed tree, with the keys `K`, its value at each
alike to `m`'s. -/
def Inv (m n : Std.TreeMap.Raw String Json) (K : List String) : Prop :=
  n.WF ∧ (∀ k, (n.get? k).isSome = true ↔ k ∈ K) ∧ ∀ k b, n.get? k = some b → ∃ a, m.get? k = some a ∧ Same a b

theorem inv_insert {m n : Std.TreeMap.Raw String Json} {K : List String} (h : Inv m n K) {key : String}
    {v v' : Json} (hv : m.get? key = some v) (hs : Same v v') : Inv m (n.insert key v') (K ++ [key]) := by
  obtain ⟨hn, hk, hb⟩ := h
  have hg : ∀ k, (n.insert key v').get? k = if key = k then some v' else n.get? k := by
    intro k
    rw [Std.TreeMap.Raw.get?_eq_getElem?, Std.TreeMap.Raw.getElem?_insert hn, Std.TreeMap.Raw.get?_eq_getElem?]
    simp only [Std.compare_eq_iff_eq]
  refine ⟨hn.insert, fun k => ?_, fun k b hb' => ?_⟩
  · rw [hg]
    by_cases e : key = k
    · subst e; simp
    · simp only [e, ↓reduceIte, hk k, List.mem_append, List.mem_singleton, Ne.symm e, or_false]
  · rw [hg] at hb'
    by_cases e : key = k
    · subst e; simp at hb'; subst hb'; exact ⟨v, hv, hs⟩
    · simp only [e, ↓reduceIte] at hb'; exact hb k b hb'

theorem obj_loop (m : Std.TreeMap.Raw String Json) : ∀ (fs : List (String × Json × List Char × List Char))
    (K : List String) (n : Std.TreeMap.Raw String Json) k o rest, fs ≠ [] → Inv m n K →
    (K ++ fs.map (·.1)).Nodup →
    (∀ f ∈ fs, m.get? f.1 = some f.2.1 ∧ Esc f.1 f.2.2.1 ∧ VP f.2.1 f.2.2.2 ∧ ∃ c cs, f.2.2.2 = c :: cs ∧ Lead c) →
    Stop rest → ([','].intercalate (fs.map fieldT) ++ ['}']).length < k →
    ∃ o' n', object k n ([','].intercalate (fs.map fieldT) ++ '}' :: rest) o = .ok rest o' n' ∧
      Inv m n' (K ++ fs.map (·.1))
  | [], _, _, _, _, _, hne, _, _, _, _, _ => absurd rfl hne
  | f :: fs, K, n, k + 1, o, rest, _, hi, hnd, hf, hs, hk => by
    obtain ⟨key, v, kt, vt⟩ := f
    obtain ⟨hg, hke, hvp, c, cs, hc, hlead⟩ := hf _ (List.mem_cons_self ..)
    dsimp only at hg hke hvp hc
    have hcol : ∀ X o, ws (':' :: X) o = (':' :: X, o) := fun _ _ => rfl
    have hvt : ∀ X o, ws (vt ++ X) o = (vt ++ X, o) := fun X o => by rw [hc, List.cons_append, lead_ws hlead]
    cases fs with
    | nil =>
      simp only [List.map_cons, List.map_nil, List.intercalate_singleton, fieldT, List.cons_append,
        List.append_assoc, List.length_cons, List.length_append, List.length_nil] at hk ⊢
      obtain ⟨o₁, ho⟩ := hke k "" (':' :: (vt ++ '}' :: rest)) (o + 1) (by omega)
      obtain ⟨o₂, v', hv', hsv⟩ := hvp k ('}' :: rest) (o₁ + 1) (.inr ⟨_, _, rfl, by simp⟩) (by omega)
      refine ⟨o₂ + 1, n.insert key v', ?_, by simpa using inv_insert hi hg hsv⟩
      rw [object.eq_3, ho, String.empty_append]
      simp only [hcol, hvt, hv', ↓reduceIte, stop_ws hs]
    | cons g gs =>
      obtain ⟨-, -, -, -⟩ := hf g (by simp)
      have hgt : ∃ Y, fieldT g = '"' :: Y := ⟨_, rfl⟩
      obtain ⟨Y, hY⟩ := hgt
      obtain ⟨Z, hZ⟩ : ∃ Z, [','].intercalate (fieldT g :: gs.map fieldT) = fieldT g ++ Z := by
        cases gs with
        | nil => exact ⟨[], by simp⟩
        | cons _ _ => exact ⟨_, by rw [List.map_cons, List.intercalate_cons_cons, List.append_assoc]⟩
      have hnext : ∀ o, ws ([','].intercalate (fieldT g :: gs.map fieldT) ++ '}' :: rest) o =
          ([','].intercalate (fieldT g :: gs.map fieldT) ++ '}' :: rest, o) := fun o => by
        rw [hZ, hY]; rfl
      simp only [List.map_cons, List.intercalate_cons_cons, List.length_append, List.length_singleton] at hk ⊢
      generalize hT : [','].intercalate (fieldT g :: gs.map fieldT) = T at hk hnext ⊢
      have hft : fieldT (key, v, kt, vt) = '"' :: (kt ++ '"' :: ':' :: vt) := rfl
      rw [hft] at hk ⊢
      simp only [List.cons_append, List.append_assoc, List.length_cons, List.length_append,
        List.nil_append] at hk ⊢
      obtain ⟨o₁, ho⟩ := hke k "" (':' :: (vt ++ ',' :: (T ++ '}' :: rest))) (o + 1) (by omega)
      obtain ⟨o₂, v', hv', hsv⟩ := hvp k (',' :: (T ++ '}' :: rest)) (o₁ + 1) (.inr ⟨_, _, rfl, by simp⟩) (by omega)
      subst hT
      obtain ⟨o', n', hn', hi'⟩ := obj_loop m (g :: gs) (K ++ [key]) (n.insert key v') k (o₂ + 1) rest (by simp)
        (inv_insert hi hg hsv) (by simpa using hnd) (fun f hf' => hf f (List.mem_cons_of_mem _ hf')) hs
        (by simp only [List.map_cons, List.length_append, List.length_singleton] at hk ⊢; omega)
      refine ⟨o', n', ?_, by simpa using hi'⟩
      rw [object.eq_3, ho, String.empty_append]
      simp only [List.map_cons] at hn'
      simp only [hcol, hvt, hv', ↓reduceIte, show (',' : Char) ≠ '}' by decide, hnext, hn']
  | _, _, _, 0, _, _, _, _, _, _, _, hk => absurd hk (Nat.not_lt_zero _)

theorem mem_keys {m : Std.TreeMap.Raw String Json} (hm : m.WF) (k : String) :
    k ∈ m.toList.map (·.1) ↔ (m.get? k).isSome = true := by
  rw [Std.TreeMap.Raw.get?_eq_getElem?, Option.isSome_iff_exists]
  constructor
  · intro h
    obtain ⟨⟨k', v⟩, hp, rfl⟩ := List.mem_map.1 h
    exact ⟨v, (Std.TreeMap.Raw.mem_toList_iff_getElem?_eq_some hm).1 hp⟩
  · intro ⟨v, hv⟩
    exact List.mem_map.2 ⟨(k, v), (Std.TreeMap.Raw.mem_toList_iff_getElem?_eq_some hm).2 hv, rfl⟩

theorem keys_nodup {m : Std.TreeMap.Raw String Json} (hm : m.WF) : (m.toList.map (·.1)).Nodup := by
  have := Std.TreeMap.Raw.distinct_keys_toList hm
  unfold List.Nodup
  rw [List.pairwise_map]
  exact this.imp fun h e => h (Std.compare_eq_iff_eq.2 e)

theorem inv_same {m n : Std.TreeMap.Raw String Json} (hm : m.WF) {K : List String}
    (hK : K.Perm (m.toList.map (·.1))) (h : Inv m n K) : Same (.obj m) (.obj n) := by
  obtain ⟨hn, hk, hb⟩ := h
  refine .obj hm hn (fun k => ?_) (fun k a b ha hb' => ?_)
  · apply Bool.eq_iff_iff.2
    rw [hk k, hK.mem_iff, mem_keys hm]
  · obtain ⟨a', ha', hs⟩ := hb k b hb'
    rw [ha] at ha'
    injection ha' with e
    exact e ▸ hs

/-- **The parser reads a value's text back**: `value` reads `t` and stops before `rest`, with a value alike. -/
theorem pr_value {j : Json} {t : List Char} (h : Pr j t) :
    ∀ k rest o, Stop rest → t.length < k → ∃ o' j', value k (t ++ rest) o = .ok rest o' j' ∧ Same j j' := by
  induction h with
  | null =>
    intro k rest o hs hk
    obtain ⟨k, rfl⟩ : ∃ k', k = k' + 1 := ⟨k - 1, by simp at hk; omega⟩
    rcases hs with rfl | ⟨c, cs, rfl, rfl | rfl | rfl⟩ <;> exact ⟨_, _, rfl, .null⟩
  | tru =>
    intro k rest o hs hk
    obtain ⟨k, rfl⟩ : ∃ k', k = k' + 1 := ⟨k - 1, by simp at hk; omega⟩
    rcases hs with rfl | ⟨c, cs, rfl, rfl | rfl | rfl⟩ <;> exact ⟨_, _, rfl, .bool _⟩
  | fls =>
    intro k rest o hs hk
    obtain ⟨k, rfl⟩ : ∃ k', k = k' + 1 := ⟨k - 1, by simp at hk; omega⟩
    rcases hs with rfl | ⟨c, cs, rfl, rfl | rfl | rfl⟩ <;> exact ⟨_, _, rfl, .bool _⟩
  | num m =>
    intro k rest o hs hk
    obtain ⟨k, rfl⟩ : ∃ k', k = k' + 1 := ⟨k - 1, by omega⟩
    obtain ⟨o', ho⟩ := num_int m rest hs o
    have hl := pr_lead (Pr.num m)
    obtain ⟨c, cs, hct, hc⟩ := hl
    have hn : c = '-' ∨ ('0' ≤ c ∧ c ≤ '9') := by
      cases m with
      | ofNat n =>
        obtain ⟨c', cs', he, hc'⟩ := toDigits_ne n
        simp [Int.repr, Nat.repr, he] at hct
        obtain ⟨rfl, -⟩ := hct
        exact .inr hc'
      | negSucc n =>
        simp [Int.repr] at hct
        exact .inl hct.1.symm
    refine ⟨o', .num ⟨m, 0⟩, ?_, .num _⟩
    rw [hct] at ho ⊢
    rw [List.cons_append] at ho ⊢
    have ne : ∀ d : Char, (d = '[' ∨ d = '{' ∨ d = '"' ∨ d = 'f' ∨ d = 't' ∨ d = 'n') → c ≠ d := by
      intro d hd hcd
      subst hcd
      rcases hn with h | h <;> rcases hd with rfl | rfl | rfl | rfl | rfl | rfl <;> first | exact absurd h (by decide)
    rw [value]
    simp only [ne '[' (by simp), ne '{' (by simp), ne '"' (by simp), ne 'f' (by simp), ne 't' (by simp),
      ne 'n' (by simp), hn, ↓reduceIte, ho, stop_ws hs]
  | str s t he =>
    intro k rest o hs hk
    obtain ⟨k, rfl⟩ : ∃ k', k = k' + 1 := ⟨k - 1, by simp at hk; omega⟩
    obtain ⟨o', ho⟩ := he k "" rest (o + 1) (by simp at hk; omega)
    refine ⟨o', .str s, ?_, .str s⟩
    simp only [List.cons_append, List.append_assoc, List.nil_append]
    rw [value]
    simp only [show ('"' : Char) ≠ '[' by decide, show ('"' : Char) ≠ '{' by decide, ↓reduceIte]
    rw [ho, String.empty_append]
    simp only [stop_ws hs]
  | arr l ts hl hp ih =>
    intro k rest o hs hk
    obtain ⟨k, rfl⟩ : ∃ k', k = k' + 1 := ⟨k - 1, by omega⟩
    cases ts with
    | nil =>
      have : l = [] := List.eq_nil_of_length_eq_zero (by simpa using hl.symm)
      subst this
      refine ⟨o + 2, .arr #[], ?_, .arr rfl fun i h => absurd h (by simp)⟩
      rcases hs with rfl | ⟨c, cs, rfl, rfl | rfl | rfl⟩ <;> rfl
    | cons t ts =>
      have hne : l ≠ [] := by rintro rfl; simp at hl
      obtain ⟨o', ys, hys, hyl, hya⟩ := arr_loop l (t :: ts) hl hne
        (fun p hp' => ⟨ih p hp', pr_lead (hp p hp')⟩) k #[] rest (o + 1) hs (by simp at hk ⊢; omega)
      obtain ⟨c, cs, hc, hlead⟩ : ∃ c cs, t = c :: cs ∧ Lead c := by
        cases l with
        | nil => exact absurd rfl hne
        | cons x l => exact pr_lead (hp (x, t) (by simp))
      obtain ⟨X, hX⟩ : ∃ X, [','].intercalate (t :: ts) = t ++ X := by
        cases ts with
        | nil => exact ⟨[], by simp⟩
        | cons _ _ => exact ⟨_, by rw [List.intercalate_cons_cons, List.append_assoc]⟩
      refine ⟨o', .arr ⟨ys⟩, ?_, same_arr l ys hyl hya⟩
      have hw : ws ([','].intercalate (t :: ts) ++ ']' :: rest) (o + 1) =
          ([','].intercalate (t :: ts) ++ ']' :: rest, o + 1) := by
        rw [hX, hc, List.cons_append, List.cons_append, lead_ws hlead]
      simp only [List.cons_append, List.append_assoc, List.nil_append]
      rw [value]
      simp only [↓reduceIte]
      split
      · rename_i h; rw [hw, hX, hc] at h; simp at h
      · rename_i h; rw [hw, hX, hc] at h; simp at h; exact absurd h.1.1 (lead_ne hlead).1
      · rename_i h
        rw [hw] at h
        injection h with h1 h2
        rw [← h1, ← h2, hys]
        simp
  | obj m hm fs hget hkeys hk' hv ih =>
    intro k rest o hs hk
    obtain ⟨k, rfl⟩ : ∃ k', k = k' + 1 := ⟨k - 1, by omega⟩
    cases fs with
    | nil =>
      have hm0 : ∀ k, m.get? k = none := fun k => by
        cases h : m.get? k with
        | none => rfl
        | some v =>
          have := (mem_keys hm k).2 (by rw [h]; rfl)
          rw [← hkeys.mem_iff] at this
          simp at this
      refine ⟨o + 2, .obj ∅, ?_, .obj hm Std.TreeMap.Raw.WF.emptyc (fun k => ?_) (fun k a b ha _ => ?_)⟩
      · rcases hs with rfl | ⟨c, cs, rfl, rfl | rfl | rfl⟩ <;> rfl
      · rw [hm0, Std.TreeMap.Raw.get?_eq_getElem?, Std.TreeMap.Raw.getElem?_emptyc]
      · rw [hm0] at ha; cases ha
    | cons f fs =>
      obtain ⟨o', n', hn', hi'⟩ := obj_loop m (f :: fs) [] ∅ k (o + 1) rest (by simp)
        ⟨Std.TreeMap.Raw.WF.emptyc, fun k => by simp [Std.TreeMap.Raw.get?_eq_getElem?,
          Std.TreeMap.Raw.getElem?_emptyc], fun k b h => by
            simp [Std.TreeMap.Raw.get?_eq_getElem?, Std.TreeMap.Raw.getElem?_emptyc] at h⟩
        (by simpa using hkeys.nodup_iff.2 (keys_nodup hm))
        (fun g hg => ⟨hget g hg, hk' g hg, ih g hg, pr_lead (hv g hg)⟩) hs (by simp at hk ⊢; omega)
      refine ⟨o', .obj n', ?_, inv_same hm (by simpa using hkeys) hi'⟩
      obtain ⟨Z, hZ⟩ : ∃ Z, [','].intercalate ((f :: fs).map fieldT) = fieldT f ++ Z := by
        cases fs with
        | nil => exact ⟨[], by simp⟩
        | cons g gs =>
          refine ⟨[','] ++ [','].intercalate ((g :: gs).map fieldT), ?_⟩
          rw [List.map_cons, List.map_cons, List.intercalate_cons_cons, List.append_assoc]
      have hw : ws ([','].intercalate ((f :: fs).map fieldT) ++ '}' :: rest) (o + 1) =
          ([','].intercalate ((f :: fs).map fieldT) ++ '}' :: rest, o + 1) := by
        rw [hZ]; rfl
      simp only [List.cons_append, List.append_assoc, List.nil_append]
      rw [value]
      simp only [show ('{' : Char) ≠ '[' by decide, ↓reduceIte]
      split
      · rename_i h; rw [hw] at h; simp [fieldT] at h
      · rename_i h; rw [hw] at h; simp [fieldT] at h
      · rename_i h
        rw [hw] at h
        injection h with h1 h2
        rw [← h1, ← h2, hn']

/-- **A text in `Pr` parses**, as a value alike. -/
theorem pr_parse {j : Json} {s : String} (h : Pr j s.toList) : ∃ j', parse s = .ok j' ∧ Same j j' := by
  obtain ⟨c, cs, hc, hl⟩ := pr_lead h
  obtain ⟨o', j', hv, hs⟩ := pr_value h (2 * s.toList.length + 2) [] 0 (.inl rfl) (by omega)
  refine ⟨j', ?_, hs⟩
  rw [parse_eq, parseList]
  rw [List.append_nil] at hv
  simp only [hc, lead_ws hl] at hv ⊢
  rw [hv]

theorem itemsJ_eq : ∀ l : List Json, FlockComplete.itemsJ l = l.map FlockComplete.printJ
  | [] => by simp [FlockComplete.itemsJ]
  | j :: l => by simp [FlockComplete.itemsJ, itemsJ_eq l]

theorem fieldsJ_eq : ∀ t : Std.DTreeMap.Internal.Impl String (fun _ => Json),
    FlockComplete.fieldsJ t = t.toListModel.map fun p => FlockComplete.strJ p.1 ++ ":" ++ FlockComplete.printJ p.2
  | .leaf => by simp [FlockComplete.fieldsJ, Std.DTreeMap.Internal.Impl.toListModel_leaf]
  | .inner _ k v l r => by
    simp [FlockComplete.fieldsJ, fieldsJ_eq l, fieldsJ_eq r, Std.DTreeMap.Internal.Impl.toListModel_inner]

theorem zip_map_self {α β} (f : α → β) : ∀ (l : List α) p, p ∈ l.zip (l.map f) → p.1 ∈ l ∧ p.2 = f p.1
  | [], _, h => by simp at h
  | x :: l, p, h => by
    simp only [List.map_cons, List.zip_cons_cons, List.mem_cons] at h
    rcases h with rfl | h
    · simp
    · obtain ⟨h1, h2⟩ := zip_map_self f l p h
      exact ⟨List.mem_cons_of_mem _ h1, h2⟩

/-- A string's text as `printJ` prints it, between its quotes. -/
def escP (s : String) : List Char := s.toList.flatMap fun c => (FlockComplete.escC c).toList

theorem strJ_toList (s : String) : (FlockComplete.strJ s).toList = '"' :: escP s ++ ['"'] := by
  simp [FlockComplete.strJ, escP, String.toList_join, List.flatMap_map]

/-- **`printJ`'s texts are in `Pr`.** -/
theorem print_pr {j : Json} (h : Good j) : Pr j (FlockComplete.printJ j).toList := by
  induction h with
  | null => exact .null
  | bool b => cases b <;> first | exact .tru | exact .fls
  | num m => exact .num m
  | str s =>
    rw [show FlockComplete.printJ (.str s) = FlockComplete.strJ s from rfl, strJ_toList]
    exact .str s _ (esc_of _ escC_esc s)
  | arr h ih =>
    rename_i a
    obtain ⟨l⟩ := a
    rw [show FlockComplete.printJ (.arr ⟨l⟩) = "[" ++ ",".intercalate (FlockComplete.itemsJ l) ++ "]" from rfl,
      itemsJ_eq]
    have e : ("[" ++ ",".intercalate (l.map FlockComplete.printJ) ++ "]").toList =
        '[' :: [','].intercalate (l.map fun x => (FlockComplete.printJ x).toList) ++ [']'] := by
      simp [String.toList_intercalate, List.map_map]; rfl
    rw [e]
    refine .arr l _ (by simp) fun p hp => ?_
    obtain ⟨h1, h2⟩ := zip_map_self _ l p hp
    rw [h2]
    exact ih p.1 (by simpa using h1)
  | obj hm h ih =>
    rename_i m
    have hl : m.toList = m.inner.inner.toListModel.map fun e => (e.1, e.2) := by
      simp only [Std.TreeMap.Raw.toList, Std.DTreeMap.Raw.Const.toList,
        Std.DTreeMap.Internal.Impl.Const.toList_eq_toListModel_map]
    have hp : FlockComplete.printJ (.obj m) = "{" ++ ",".intercalate (m.toList.map fun p =>
        FlockComplete.strJ p.1 ++ ":" ++ FlockComplete.printJ p.2) ++ "}" := by
      obtain ⟨⟨t⟩⟩ := m
      rw [hl, List.map_map]
      show "{" ++ ",".intercalate (FlockComplete.fieldsJ t) ++ "}" = _
      rw [fieldsJ_eq]
      rfl
    let fs := m.toList.map fun p => (p.1, p.2, escP p.1, (FlockComplete.printJ p.2).toList)
    have e : (FlockComplete.printJ (.obj m)).toList = '{' :: [','].intercalate (fs.map fieldT) ++ ['}'] := by
      rw [hp]
      simp only [fs, String.toList_append, String.toList_intercalate, List.map_map]
      show '{' :: _ ++ ['}'] = _
      simp only [List.cons_append]
      congr 2
      rw [show [].append _ = _ from List.nil_append _, show ",".toList = [','] from rfl]
      congr 1
      apply List.map_congr_left
      intro p _
      simp [fieldT, strJ_toList]
    have hmem : ∀ f ∈ fs, (f.1, f.2.1) ∈ m.toList ∧ f.2.2.1 = escP f.1 ∧
        f.2.2.2 = (FlockComplete.printJ f.2.1).toList := by
      intro f hf
      obtain ⟨⟨k, v⟩, hkv, rfl⟩ := List.mem_map.1 hf
      exact ⟨hkv, rfl, rfl⟩
    have hget : ∀ f ∈ fs, m.get? f.1 = some f.2.1 := fun f hf => by
      rw [Std.TreeMap.Raw.get?_eq_getElem?]
      exact (Std.TreeMap.Raw.mem_toList_iff_getElem?_eq_some hm).1 (hmem f hf).1
    rw [e]
    refine .obj m hm fs hget (by simp only [fs, List.map_map]; exact .refl _) (fun f hf => ?_) (fun f hf => ?_)
    · rw [(hmem f hf).2.1]; exact esc_of _ escC_esc _
    · rw [(hmem f hf).2.2]; exact ih _ _ (hget f hf)

theorem mapM_zip {α β ε : Type} (f : α → Except ε β) :
    ∀ (l : List α) (r : List β), l.mapM f = .ok r → r.length = l.length ∧ ∀ p ∈ l.zip r, f p.1 = .ok p.2
  | [], r, h => by
    simp only [List.mapM_nil] at h
    cases h; simp
  | a :: l, r, h => by
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
        obtain ⟨h1, h2⟩ := mapM_zip f l bs hl
        refine ⟨by simp [h1], fun p hp => ?_⟩
        simp only [List.zip_cons_cons, List.mem_cons] at hp
        rcases hp with rfl | hp
        · exact ha
        · exact h2 p hp

theorem array_mapM_zip {α β ε : Type} (f : α → Except ε β) (a : Array α) (r : Array β) (h : a.mapM f = .ok r) :
    r.toList.length = a.toList.length ∧ ∀ p ∈ a.toList.zip r.toList, f p.1 = .ok p.2 := by
  rw [Array.mapM_eq_mapM_toList] at h
  cases hl : a.toList.mapM f with
  | error e => rw [hl] at h; cases h
  | ok bs =>
    rw [hl] at h
    cases h
    simpa using mapM_zip f _ bs hl

/-- A string's text as `canon` prints it, between its quotes. -/
def escK (s : String) : List Char := s.toList.flatMap fun c => (cesc c).toList

theorem toString_int (m : Int) : toString m = Int.repr m := rfl

theorem zip_val {α β γ : Type} {P : α → Prop} (l : List {x // P x}) (r : List β) (g : β → γ) :
    ∀ p ∈ (l.map Subtype.val).zip (r.map g), ∃ q ∈ l.zip r, p = (q.1.1, g q.2) := by
  intro p hp
  rw [List.zip_map] at hp
  obtain ⟨q, hq, rfl⟩ := List.mem_map.1 hp
  exact ⟨q, hq, rfl⟩

theorem map_eq_of_zip {α β γ : Type} (f : α → γ) (g : β → γ) :
    ∀ (l : List α) (r : List β), l.length = r.length → (∀ p ∈ l.zip r, f p.1 = g p.2) → l.map f = r.map g
  | [], [], _, _ => rfl
  | a :: l, b :: r, hl, h => by
    simp only [List.map_cons, List.cons.injEq]
    exact ⟨h (a, b) (by simp), map_eq_of_zip f g l r (by simpa using hl)
      fun p hp => h p (by simp [List.zip_cons_cons, hp])⟩
  | [], _ :: _, hl, _ => by simp at hl
  | _ :: _, [], hl, _ => by simp at hl

/-- **`canon`'s texts are in `Pr`.** -/
theorem canon_pr {j : Json} (h : Good j) : ∀ s, Flock.canon j = .ok s → Pr j s.toList := by
  induction h with
  | null => intro s h; rw [Flock.canon] at h; cases h; exact .null
  | bool b => intro s h; rw [Flock.canon] at h; cases h; cases b <;> first | exact .tru | exact .fls
  | num m => intro s h; rw [Flock.canon] at h; cases h; exact .num m
  | str s =>
    intro s' h; rw [Flock.canon] at h
    by_cases ht : Flock.ascii s = true
    · rw [if_pos ht] at h; cases h
      rw [canonStr_toList]
      exact .str s _ (esc_of _ cesc_esc s)
    · rw [if_neg ht] at h; cases h
  | arr h ih =>
    rename_i a
    intro s hs
    rw [Flock.canon] at hs
    cases hp : a.attach.mapM (fun x => Flock.canon x.1) with
    | error e => rw [hp] at hs; cases hs
    | ok parts =>
      rw [hp] at hs
      cases hs
      obtain ⟨hlen, hz⟩ := array_mapM_zip _ _ _ hp
      have ha : a.attach.toList.map Subtype.val = a.toList := by simp
      have e : ("[" ++ ",".intercalate parts.toList ++ "]").toList =
          '[' :: [','].intercalate (parts.toList.map String.toList) ++ [']'] := by
        simp [String.toList_intercalate]
      rw [e]
      show Pr (.arr ⟨a.toList⟩) _
      refine .arr a.toList _ (by simp at hlen ⊢; try omega) fun p hp' => ?_
      rw [← ha] at hp'
      obtain ⟨q, hq, rfl⟩ := zip_val _ _ _ p hp'
      exact ih q.1.1 q.1.2 q.2 (hz q hq)
  | obj hm h ih =>
    rename_i m
    intro s hs
    rw [Flock.canon] at hs
    cases hp : m.toArray.attach.mapM (fun p =>
        if Flock.ascii p.1.1 then do pure (p.1.1, ← Flock.canon p.1.2) else throw Flock.notAscii) with
    | error e => rw [hp] at hs; cases hs
    | ok vals =>
      rw [hp] at hs
      cases hs
      obtain ⟨hlen, hz⟩ := array_mapM_zip _ _ _ hp
      have hpair : ∀ q ∈ m.toArray.attach.toList.zip vals.toList,
          q.2.1 = q.1.1.1 ∧ Flock.canon q.1.1.2 = .ok q.2.2 := by
        intro q hq
        have := hz q hq
        try dsimp only at this
        by_cases ht : Flock.ascii q.1.1.1 = true
        · rw [if_pos ht] at this
          cases hc : Flock.canon q.1.1.2 with
          | error e => rw [hc] at this; cases this
          | ok sv =>
            rw [hc] at this
            have e : Except.ok (q.1.1.1, sv) = (Except.ok q.2 : Except String (String × String)) := this
            injection e with e
            rw [← e]
            exact ⟨rfl, rfl⟩
        · rw [if_neg ht] at this; cases this
      have hw : ∀ w ∈ vals.toList, ∃ v, m.get? w.1 = some v ∧ Flock.canon v = .ok w.2 := by
        intro w hw'
        obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem hw'
        have hi' : i < m.toArray.attach.toList.length := hlen ▸ hi
        have hq : (m.toArray.attach.toList[i], vals.toList[i]) ∈ m.toArray.attach.toList.zip vals.toList := by
          have := List.getElem_mem (l := m.toArray.attach.toList.zip vals.toList) (n := i)
            (by rw [List.length_zip]; exact Nat.lt_min.2 ⟨hi', hi⟩)
          simpa [List.getElem_zip] using this
        obtain ⟨e1, e2⟩ := hpair _ hq
        refine ⟨_, ?_, e2⟩
        rw [e1, Std.TreeMap.Raw.get?_eq_getElem?]
        apply (Std.TreeMap.Raw.mem_toList_iff_getElem?_eq_some hm).1
        rw [← Std.TreeMap.Raw.toList_toArray]
        exact Array.mem_toList_iff.2 (m.toArray.attach.toList[i]).2
      have hkv : vals.toList.map (·.1) = m.toList.map (·.1) := by
        rw [← Std.TreeMap.Raw.toList_toArray]
        have ha : m.toArray.attach.toList.map Subtype.val = m.toArray.toList := by
          simpa using List.attach_map_subtype_val m.toArray.toList
        rw [← ha, List.map_map]
        exact (map_eq_of_zip (fun p => p.1.1) (·.1) _ _ (by omega) fun q hq => (hpair q hq).1.symm).symm
      let items := vals.qsort (fun a b => a.1 < b.1)
      have hperm : items.toList.Perm vals.toList := Array.perm_iff_toList_perm.1 (qsort_perm _ _)
      let fs := items.toList.map fun w => (w.1, (m.get? w.1).getD .null, escK w.1, w.2.toList)
      have hfs : ∀ f ∈ fs, ∃ w ∈ vals.toList, f = (w.1, (m.get? w.1).getD .null, escK w.1, w.2.toList) := by
        intro f hf
        obtain ⟨w, hw', rfl⟩ := List.mem_map.1 hf
        exact ⟨w, hperm.mem_iff.1 hw', rfl⟩
      have e : ("{" ++ ",".intercalate (items.toList.map fun kv => Flock.canonStr kv.1 ++ ":" ++ kv.2) ++
          "}").toList = '{' :: [','].intercalate (fs.map fieldT) ++ ['}'] := by
        simp only [fs, String.toList_append, String.toList_intercalate, List.map_map]
        show '{' :: _ ++ ['}'] = _
        simp only [List.cons_append]
        congr 2
        rw [show [].append _ = _ from List.nil_append _, show ",".toList = [','] from rfl]
        congr 1
        apply List.map_congr_left
        intro p _
        simp [fieldT, canonStr_toList, escK]
      rw [e]
      refine .obj m hm fs (fun f hf => ?_) ?_ (fun f hf => ?_) (fun f hf => ?_)
      · obtain ⟨w, hw', rfl⟩ := hfs f hf
        obtain ⟨v, hv, -⟩ := hw w hw'
        show m.get? w.1 = some ((m.get? w.1).getD .null)
        rw [hv]; rfl
      · rw [← hkv]
        simp only [fs, List.map_map]
        exact hperm.map _
      · obtain ⟨w, hw', rfl⟩ := hfs f hf
        exact esc_of _ cesc_esc _
      · obtain ⟨w, hw', rfl⟩ := hfs f hf
        obtain ⟨v, hv, hc⟩ := hw w hw'
        simp only [hv, Option.getD_some]
        exact ih w.1 v hv w.2 hc

end FlockComplete.JsonRT.ReadBack

namespace FlockComplete.JsonRT

open Lean (Json)
open Claims.Json (Same Good)

/-- **The verifier's parser reads back the prover's printer**: `parse` of `printJ j`, for a printable `j`, is a value
alike to `j`. -/
theorem parse_print (j : Json) (h : Good j) :
    ∃ j', Flock.JsonParse.parse (FlockComplete.printJ j) = .ok j' ∧ Same j j' :=
  ReadBack.pr_parse (ReadBack.print_pr h)

/-- **The verifier's parser reads back `canon`**: `parse` of what `Flock.canon` printed of a printable `j` is a value
alike to `j`. -/
theorem parse_canon (j : Json) (s : String) (h : Good j) (hc : Flock.canon j = .ok s) :
    ∃ j', Flock.JsonParse.parse s = .ok j' ∧ Same j j' :=
  ReadBack.pr_parse (ReadBack.canon_pr h s hc)

end FlockComplete.JsonRT
