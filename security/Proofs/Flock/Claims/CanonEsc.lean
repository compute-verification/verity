module

public import Batteries.Tactic.OpenPrivate
public import Proofs.Flock.Claims.CanonText

/-!
# `Flock.canonStr`'s strings, read back

`canonStr` writes each character as itself, a short escape, or a `\u` escape (two, a UTF-16 surrogate pair, above
`0xFFFF`): `unesc` reads one back (`unesc_esc`).
-/

@[expose] public section

open private hex4 from Base.Canon

namespace Properties.Composition

namespace Esc

open Properties.Pouw.LiveSalt.Canon (unhexL nib)

/-- What `Flock.canonStr` writes for one character. -/
def esc (c : Char) : List Char :=
  match c with
  | '"' => ['\\', '"']
  | '\\' => ['\\', '\\']
  | '\n' => ['\\', 'n']
  | '\r' => ['\\', 'r']
  | '\t' => ['\\', 't']
  | c => if c.toNat == 8 then ['\\', 'b'] else if c.toNat == 12 then ['\\', 'f']
    else if c.toNat < 0x20 || 0x7f ≤ c.toNat then (Flock.escU c.toNat).toList else [c]

theorem foldl_esc (f : String → Char → String) (hf : ∀ b a, (f b a).toList = b.toList ++ esc a) :
    ∀ (cs : List Char) (out : String), (cs.foldl f out).toList = out.toList ++ cs.flatMap esc := by
  intro cs
  induction cs with
  | nil => intro out; simp
  | cons c cs ih => intro out; rw [List.foldl_cons, ih, hf, List.flatMap_cons, List.append_assoc]

theorem canonStr_toList (s : String) : (Flock.canonStr s).toList = '"' :: s.toList.flatMap esc ++ ['"'] := by
  unfold Flock.canonStr
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

/-- `hex4`'s digit of `x`. -/
def dg (x : Nat) : Char := if x < 10 then Char.ofNat (48 + x) else Char.ofNat (87 + x)

theorem nib_dg : ∀ x, x < 16 → nib (dg x) = x := by decide

theorem hex4_toList (n : Nat) :
    (hex4 n).toList = [dg ((n >>> 12) % 16), dg ((n >>> 8) % 16), dg ((n >>> 4) % 16), dg ((n >>> 0) % 16)] := by
  simp only [hex4, String.toList_ofList, dg] <;> rfl

/-- `hex4` of a number below `0x10000` is four hex digits that read back to it. -/
theorem hex4_ok (n : Nat) (hn : n < 0x10000) : (hex4 n).toList.length = 4 ∧ unhexL (hex4 n).toList = n := by
  rw [hex4_toList]
  refine ⟨rfl, ?_⟩
  have e12 : n >>> 12 = n / 4096 := by rw [Nat.shiftRight_eq_div_pow]
  have e8 : n >>> 8 = n / 256 := by rw [Nat.shiftRight_eq_div_pow]
  have e4 : n >>> 4 = n / 16 := by rw [Nat.shiftRight_eq_div_pow]
  have e0 : n >>> 0 = n := Nat.shiftRight_zero
  simp only [unhexL, List.foldl_cons, List.foldl_nil, nib_dg _ (Nat.mod_lt _ (by decide : 16 > 0)), e12, e8, e4, e0]
  omega

/-- Whether a `\u` escape's number is a high surrogate, the first of a pair. -/
def high (n : Nat) : Bool := 0xD800 ≤ n && n < 0xDC00

/-- The character a surrogate pair `h`, `l` stands for. -/
def pair (h l : Nat) : Nat := 0x10000 + (h - 0xD800) * 0x400 + (l - 0xDC00)

/-- Reads a surrogate pair's second `\u` escape. -/
def unescLow : List Char → Option (Nat × List Char)
  | '\\' :: 'u' :: a :: b :: c :: d :: r => some (unhexL [a, b, c, d], r)
  | _ => none

/-- Reads one escape's character after its backslash. -/
def unesc2 : List Char → Option (Char × List Char)
  | '"' :: r => some ('"', r)
  | '\\' :: r => some ('\\', r)
  | 'n' :: r => some ('\n', r)
  | 'r' :: r => some ('\r', r)
  | 't' :: r => some ('\t', r)
  | 'b' :: r => some (Char.ofNat 8, r)
  | 'f' :: r => some (Char.ofNat 12, r)
  | 'u' :: a :: b :: c :: d :: r =>
    if high (unhexL [a, b, c, d]) then (unescLow r).map fun (l, r') => (Char.ofNat (pair (unhexL [a, b, c, d]) l), r')
    else some (Char.ofNat (unhexL [a, b, c, d]), r)
  | _ => none

/-- Reads one character of an escaped string. -/
def unesc : List Char → Option (Char × List Char)
  | [] => none
  | c :: r => if c = '\\' then unesc2 r else some (c, r)

theorem u_toList : ("\\u" : String).toList = ['\\', 'u'] := by decide

theorem unesc_escU (c : Char) (r : List Char) : unesc ((Flock.escU c.toNat).toList ++ r) = some (c, r) := by
  have hv : c.toNat < 0xd800 ∨ (0xdfff < c.toNat ∧ c.toNat < 0x110000) := c.valid
  unfold Flock.escU
  by_cases hlt : c.toNat < 0x10000
  · simp only [hlt, ↓reduceIte]
    have hc : high c.toNat = false := by
      unfold high
      rcases hv with h | ⟨h, -⟩
      · simp [show ¬ 0xD800 ≤ c.toNat by omega]
      · simp [show ¬ c.toNat < 0xDC00 by omega]
    obtain ⟨hl, hx⟩ := hex4_ok _ hlt
    rw [String.toList_append]
    generalize (hex4 c.toNat).toList = xs at hl hx ⊢
    match xs, hl with
    | [a1, a2, a3, a4], _ => simp [u_toList, unesc, unesc2, hx, hc, Char.ofNat_toNat]
  · simp only [hlt, ↓reduceIte]
    have hH : 0xD800 + (c.toNat - 0x10000) / 0x400 < 0x10000 := by omega
    have hL : 0xDC00 + (c.toNat - 0x10000) % 0x400 < 0x10000 := by omega
    have hp : pair (0xD800 + (c.toNat - 0x10000) / 0x400) (0xDC00 + (c.toNat - 0x10000) % 0x400) = c.toNat := by
      unfold pair; omega
    have hc : high (0xD800 + (c.toNat - 0x10000) / 0x400) = true := by
      unfold high
      simp [show 0xD800 ≤ 0xD800 + (c.toNat - 0x10000) / 0x400 by omega,
        show 0xD800 + (c.toNat - 0x10000) / 0x400 < 0xDC00 by omega]
    obtain ⟨hl, hx⟩ := hex4_ok _ hH
    obtain ⟨hl', hy⟩ := hex4_ok _ hL
    simp only [String.toList_append, List.append_assoc]
    generalize 0xD800 + (c.toNat - 0x10000) / 0x400 = H at hl hx hc hp ⊢
    generalize 0xDC00 + (c.toNat - 0x10000) % 0x400 = L at hl' hy hp ⊢
    generalize (hex4 H).toList = xs at hl hx ⊢
    generalize (hex4 L).toList = ys at hl' hy ⊢
    match xs, hl, ys, hl' with
    | [a1, a2, a3, a4], _, [b1, b2, b3, b4], _ =>
      simp [u_toList, unesc, unesc2, unescLow, hx, hy, hc, hp, Char.ofNat_toNat]

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
  by_cases hu : c.toNat < 0x20 ∨ 0x7f ≤ c.toNat
  · have hb : (decide (c.toNat < 0x20) || decide (0x7f ≤ c.toNat)) = true := by simpa using hu
    simp only [beq_iff_eq, h8, h12, hb, ↓reduceIte]
    exact unesc_escU c r
  · have hb : (decide (c.toNat < 0x20) || decide (0x7f ≤ c.toNat)) = false := by simpa using hu
    simp only [beq_iff_eq, h8, h12, hb, ↓reduceIte, Bool.false_eq_true, List.singleton_append, unesc]
    exact ite_eq_right h2

end Esc

end Properties.Composition
