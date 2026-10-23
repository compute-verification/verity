import Proofs.Flock.Claims.Recursion.VStarCross
import Flock.RecRecord

/-!
# The coin server's record reads back

`Flock.RecRecord.readFull`, the record format's own total reader, reads back what `recordFullW` writes
(`readFull_recordFullW`): the rounds, the record's kind and tops' format, the prover's Hello (`readHello_helloW`) and
the link's level-0 tops and points (`readObj_objW`), piece by piece: literals, hex strings, escaped strings, decimal
numbers, lists and objects at their indentation and coins. No JSON library is involved, so no property of one is
assumed (`Claims.JsonSpec` stays a completeness premise).

At the deployment's model (`recTr`, `recRead`) the coin server's transcript of the inner rounds is that record, as
`rec_live`'s coin server writes it (`recOf`): round 0's message, the link, as its table's level-0 top (one node a
128-byte commit string, `nodesOf`) and its coin's values as the link points; each later round before the last as a
recorded round, its message's first commit strings as its tops (as many a level as `aux` says) and the rest, where
there is any, as its one `c_k`, with its coin's values; and no last round, whose openings the coin server never sees.
Everything else (stream, index, level-0 count, kind, Hello, the record past the link points) is whatever `aux`, `hd`
and `rest` say. `recRead` reads the transcript back to each round's message and drawn values before the last round,
at every accepted run (`readsBack_recTr`, `ReadsBackB`). The difftest (`verity/ml/flock/tests/test_lean_rec_record.py`)
holds the model to the honest records: `recordFullW` writes `rec_live.record_bytes`' bytes up to the link points, and
the rounds `Record.msgs` shows are the link's and the recorded rounds', one fewer than the inner verifier's, each the
schedule's count of commit strings. The model before it, which listed every inner round, set up no staged statement
(`Flock.Test.RecSchedule`: `RecSetup.commits` holds the staged record to the schedule's rounds).
-/

namespace Flock.RecRecord

/-- A literal reads off where it's written. -/
@[simp] theorem lit_append (l r : List UInt8) : lit l (l ++ r) = some r := by
  induction l with
  | nil => rfl
  | cons a l ih => simp [lit, ih]

@[simp] theorem lit_nil (s : List UInt8) : lit [] s = some s := rfl

@[simp] theorem lit_cons_cons (a : UInt8) (l s : List UInt8) : lit (a :: l) (a :: s) = lit l s := by simp [lit]

@[simp] theorem lit_append_left (l l' s : List UInt8) : lit (l ++ l') (l ++ s) = lit l' s := by
  induction l with
  | nil => rfl
  | cons a l ih => simp [ih]

/-- Every byte's two hex digits read back, and the first is no quote. -/
theorem hexB_nat : ∀ n < 256, unhexPair (hexDigit (n / 16)) (hexDigit (n % 16)) = some (UInt8.ofNat n) ∧
    hexDigit (n / 16) ≠ 34 := by
  decide +kernel

theorem hexB_spec (b : UInt8) : unhexPair (hexDigit (b.toNat / 16)) (hexDigit (b.toNat % 16)) = some b ∧
    hexDigit (b.toNat / 16) ≠ 34 := by
  have h := hexB_nat b.toNat b.toNat_lt
  rwa [UInt8.ofNat_toNat] at h

/-- Hex bytes read back up to their closing quote. -/
theorem hexBytes_hexW (bs r : List UInt8) : hexBytes (hexW bs ++ 34 :: r) = some (bs, r) := by
  induction bs with
  | nil => cases r <;> simp [hexW, hexBytes]
  | cons x bs ih =>
    obtain ⟨h1, h2⟩ := hexB_spec x
    simp only [hexW, List.flatMap_cons, hexB] at ih ⊢
    simp [hexBytes, h1, h2, ih]

/-- **A hex string reads back.** -/
theorem readHex_hexStr (bs r : List UInt8) : readHex (hexStr bs ++ r) = some (bs, r) := by
  simp [readHex, hexStr, lit, hexBytes_hexW]

/-- An escaped string's bytes read back up to its closing quote. -/
theorem unesc_escW (bs r : List UInt8) : unesc (escW bs ++ 34 :: r) = some (bs, r) := by
  induction bs with
  | nil => cases r <;> simp [escW, unesc]
  | cons x bs ih =>
    simp only [escW, List.flatMap_cons] at ih ⊢
    by_cases h : x = 34 ∨ x = 92
    · simp only [esc, h, ↓reduceIte, List.cons_append, List.nil_append]
      simp [unesc, ih]
    · simp only [esc, h, ↓reduceIte, List.cons_append, List.nil_append]
      have h34 : x ≠ 34 := fun e => h (Or.inl e)
      have h92 : x ≠ 92 := fun e => h (Or.inr e)
      cases hw : List.flatMap esc bs ++ 34 :: r with
      | nil => simp at hw
      | cons y t => rw [hw] at ih; simp [unesc, h34, h92, ih]

/-- **A stream's name reads back.** -/
theorem readStr_strStr (bs r : List UInt8) : readStr (strStr bs ++ r) = some (bs, r) := by
  simp [readStr, strStr, lit, unesc_escW]

theorem digit_spec (n : Nat) : isDigit (digit n) = true ∧ (digit n).toNat - 48 = n % 10 := by
  have h : ∀ m < 10, isDigit (digit m) = true ∧ (digit m).toNat - 48 = m := by decide
  have := h (n % 10) (Nat.mod_lt _ (by decide))
  simpa [digit, Nat.mod_mod] using this

/-- A number's decimal digits read on from any value before them. -/
theorem digits_decW (n : Nat) : ∀ acc r, digits acc (decW n ++ r) = digits (acc * 10 ^ (decW n).length + n) r := by
  induction n using Nat.strongRecOn with
  | _ n ih =>
    intro acc r
    obtain ⟨hd, hv⟩ := digit_spec n
    by_cases h : n < 10
    · rw [decW]
      simp only [h, ↓reduceIte]
      simp [digits, hd, hv, Nat.mod_eq_of_lt h]
    · rw [decW]
      simp only [h, ↓reduceIte]
      rw [List.append_assoc, ih (n / 10) (Nat.div_lt_self (by omega) (by decide))]
      simp only [List.singleton_append, digits, hd, ↓reduceIte, hv, List.length_append, List.length_singleton,
        Nat.pow_succ]
      congr 1
      have := Nat.div_add_mod n 10
      rw [Nat.add_mul, Nat.mul_assoc]
      omega

/-- **A number reads back** where no digit follows it. -/
theorem readNat_decW (n : Nat) (r : List UInt8) (hr : ∀ c t, r = c :: t → isDigit c = false) :
    readNat (decW n ++ r) = some (n, r) := by
  have hne : decW n ≠ [] := by
    unfold decW; split <;> simp
  obtain ⟨c, t, hct⟩ := List.exists_cons_of_ne_nil hne
  have hc : isDigit c = true := by
    have key : ∀ m, ∃ t, decW m = digit m :: t ∨ ∃ k, decW m = digit k :: t := by
      intro m
      induction m using Nat.strongRecOn with
      | _ m ih =>
        by_cases h : m < 10
        · exact ⟨[], Or.inl (by rw [decW]; simp [h])⟩
        · obtain ⟨t', ht'⟩ := ih (m / 10) (Nat.div_lt_self (by omega) (by decide))
          rcases ht' with ht' | ⟨k, ht'⟩
          · exact ⟨t' ++ [digit m], Or.inr ⟨m / 10, by rw [decW]; simp only [h, ↓reduceIte]; rw [ht']; rfl⟩⟩
          · exact ⟨t' ++ [digit m], Or.inr ⟨k, by rw [decW]; simp only [h, ↓reduceIte]; rw [ht']; rfl⟩⟩
    obtain ⟨t', ht'⟩ := key n
    rcases ht' with ht' | ⟨k, ht'⟩
    · rw [ht'] at hct; cases hct; exact (digit_spec n).1
    · rw [ht'] at hct; cases hct; exact (digit_spec k).1
  have hd := digits_decW n 0 r
  simp only [Nat.zero_mul, Nat.zero_add] at hd
  have hend : digits n r = (n, r) := by
    cases r with
    | nil => rfl
    | cons c' t' => simp [digits, hr c' t' rfl]
  rw [readNat.eq_def]
  simp only [hct, List.cons_append, hc, ↓reduceIte]
  rw [← List.cons_append, ← hct, hd, hend]

section Lists

variable {α : Type}

theorem length_tailW (d : Nat) (w : α → List UInt8) (es : List α) : es.length + 1 ≤ (tailW d w es).length := by
  induction es with
  | nil => simp [tailW]
  | cons e es ih => simp only [tailW, List.length_cons, List.length_append]; omega

/-- A list's elements read back, given enough fuel, where `p` reads each element back. -/
theorem elems_tailW (d : Nat) (w : α → List UInt8) (p : P α) (hp : ∀ e r, p (w e ++ r) = some (e, r))
    (es : List α) : ∀ (e : α) (fuel : Nat) (r : List UInt8), es.length + 1 ≤ fuel →
      elems d p fuel (w e ++ (tailW d w es ++ r)) = some (e :: es, r) := by
  induction es with
  | nil =>
    intro e fuel r hf
    obtain ⟨f, rfl⟩ : ∃ f, fuel = f + 1 := ⟨fuel - 1, by simp at hf; omega⟩
    simp [elems, hp, tailW]
  | cons e' es ih =>
    intro e fuel r hf
    obtain ⟨f, rfl⟩ : ∃ f, fuel = f + 1 := ⟨fuel - 1, by simp at hf; omega⟩
    have h := ih e' f r (by simp at hf; omega)
    simp [elems, hp, tailW, h]

/-- **A list reads back** where `p` reads each element back. -/
theorem readList_listW (d : Nat) (w : α → List UInt8) (p : P α) (hp : ∀ e r, p (w e ++ r) = some (e, r))
    (es : List α) (r : List UInt8) : readList d p (listW d w es ++ r) = some (es, r) := by
  cases es with
  | nil => simp [readList, listW, lit]
  | cons e es =>
    have h93 : (10 : UInt8) ≠ 93 := by decide
    have hl := length_tailW d w es
    have h := elems_tailW d w p hp es e ((w e).length + ((tailW d w es).length + r.length)) r (by omega)
    simp [readList, listW, h93, h]

end Lists

theorem length_be (w v : Nat) : (be w v).length = w := by
  induction w generalizing v with
  | zero => rfl
  | succ w ih => simp [be, ih]

/-- A number's bytes read back to it, after any value before them. -/
theorem foldl_be (w : Nat) : ∀ v acc, v < 256 ^ w →
    (be w v).foldl (fun a x => a * 256 + x.toNat) acc = acc * 256 ^ w + v := by
  induction w with
  | zero => intro v acc h; simp [be]; simp at h; omega
  | succ w ih =>
    intro v acc h
    have hq : v / 256 ^ w < 256 := by
      rw [Nat.div_lt_iff_lt_mul (Nat.pow_pos (by decide))]
      rw [Nat.pow_succ, Nat.mul_comm] at h
      exact h
    have hx : (UInt8.ofNat (v / 256 ^ w)).toNat = v / 256 ^ w := by simp [Nat.mod_eq_of_lt hq]
    simp only [be, List.foldl_cons]
    rw [ih _ _ (Nat.mod_lt _ (Nat.pow_pos (by decide))), Nat.pow_succ, hx]
    have := Nat.div_add_mod v (256 ^ w)
    rw [Nat.add_mul, Nat.mul_assoc, Nat.add_assoc, Nat.mul_comm (v / 256 ^ w), this, Nat.mul_comm 256]

theorem unbe_be (v : UInt64) : UInt64.ofNat (unbe (be 8 v.toNat)) = v := by
  rw [unbe, foldl_be 8 _ 0 v.toNat_lt_size]
  simp

/-- **A coin reads back.** -/
theorem readCoin_coinStr (c : F128) (r : List UInt8) : readCoin (coinStr c ++ r) = some (c, r) := by
  have h8 := length_be 8 c.hi.toNat
  simp [readCoin, coinStr, readHex_hexStr, coinB, length_be, List.drop_left' h8, List.take_left' h8, unbe_be]

theorem readNat_fld (n d : Nat) (s : String) (r : List UInt8) :
    readNat (decW n ++ (fld d s ++ r)) = some (n, fld d s ++ r) :=
  readNat_decW n _ fun c t h => by simp [fld] at h; rw [← h.1]; decide

/-- **A round reads back.** -/
theorem readRound_roundW (x : Round) (r : List UInt8) : readRound (roundW x ++ r) = some (x, r) := by
  have hh : ∀ (bs : List UInt8) r, readHex (hexStr bs ++ r) = some (bs, r) := readHex_hexStr
  have ht : ∀ (t : List (List UInt8)) r, readList 4 readHex (listW 4 hexStr t ++ r) = some (t, r) :=
    readList_listW 4 hexStr readHex hh
  have hb : ∀ {α β : Type} (a : α) (f : α → Option β), (some a >>= f) = f a := fun _ _ => rfl
  have hw : roundW x ++ r = (123 :: 10 :: (sp 3 ++ key "stream")) ++ (strStr x.stream ++ (fld 3 "k" ++ (decW x.k ++
      (fld 3 "tops" ++ (listW 3 (listW 4 hexStr) x.tops ++ (fld 3 "level0" ++ (decW x.level0 ++ (fld 3 "commits" ++
      (listW 3 hexStr x.commits ++ (fld 3 "n" ++ (decW x.n ++ (fld 3 "coins" ++ (listW 3 coinStr x.coins ++
      ((10 :: (sp 2 ++ [125])) ++ r)))))))))))))) := by
    simp only [roundW, List.cons_append, List.nil_append, List.append_assoc]
  rw [hw]
  dsimp only [readRound]
  -- Not `simp`: under Mathlib it descends into each bind through `Option.bind_congr'`, exponentially in their number.
  rw [lit_append, hb, readStr_strStr, hb, lit_append, hb, readNat_fld, hb, lit_append, hb,
    readList_listW 3 _ _ ht, hb, lit_append, hb, readNat_fld, hb, lit_append, hb,
    readList_listW 3 hexStr readHex hh, hb, lit_append, hb, readNat_fld, hb, lit_append, hb,
    readList_listW 3 coinStr readCoin readCoin_coinStr, hb, lit_append, hb]
  rfl

/-- **The record reads back**: `readRecord` reads the rounds `recordW` wrote, whatever the record's other fields. -/
theorem readRecord_recordW (xs : List Round) (rest : List UInt8) : readRecord (recordW xs rest) = some xs := by
  simp [readRecord, recordW, readList_listW 1 roundW readRound readRound_roundW]

section Objects

variable {α : Type}

/-- An entry reads back where `p` reads its value back. -/
theorem readEnt_entW (w : α → List UInt8) (p : P α) (hp : ∀ e r, p (w e ++ r) = some (e, r))
    (e : List UInt8 × α) (r : List UInt8) : readEnt p (entW w e ++ r) = some (e, r) := by
  simp [readEnt, entW, readStr_strStr, lit, hp]

theorem length_objTailW (d : Nat) (w : α → List UInt8) (es : List (List UInt8 × α)) :
    es.length + 1 ≤ (objTailW d w es).length := by
  induction es with
  | nil => simp [objTailW]
  | cons e es ih => simp only [objTailW, List.length_cons, List.length_append]; omega

theorem objElems_objTailW (d : Nat) (w : α → List UInt8) (p : P α) (hp : ∀ e r, p (w e ++ r) = some (e, r))
    (es : List (List UInt8 × α)) : ∀ (e : List UInt8 × α) (fuel : Nat) (r : List UInt8), es.length + 1 ≤ fuel →
      objElems d p fuel (entW w e ++ (objTailW d w es ++ r)) = some (e :: es, r) := by
  have he := readEnt_entW w p hp
  induction es with
  | nil =>
    intro e fuel r hf
    obtain ⟨f, rfl⟩ : ∃ f, fuel = f + 1 := ⟨fuel - 1, by simp at hf; omega⟩
    simp [objElems, he, objTailW]
  | cons e' es ih =>
    intro e fuel r hf
    obtain ⟨f, rfl⟩ : ∃ f, fuel = f + 1 := ⟨fuel - 1, by simp at hf; omega⟩
    have h := ih e' f r (by simp at hf; omega)
    simp [objElems, he, objTailW, h]

/-- **An object reads back** where `p` reads each value back. -/
theorem readObj_objW (d : Nat) (w : α → List UInt8) (p : P α) (hp : ∀ e r, p (w e ++ r) = some (e, r))
    (es : List (List UInt8 × α)) (r : List UInt8) : readObj d p (objW d w es ++ r) = some (es, r) := by
  cases es with
  | nil => simp [readObj, objW, lit]
  | cons e es =>
    have h125 : (10 : UInt8) ≠ 125 := by decide
    have hl := length_objTailW d w es
    have h := objElems_objTailW d w p hp es e ((entW w e).length + ((objTailW d w es).length + r.length)) r
      (by omega)
    simp [readObj, objW, h125, h]

end Objects

theorem readNat_nl (n : Nat) (l r : List UInt8) : readNat (decW n ++ ((10 :: l) ++ r)) = some (n, (10 :: l) ++ r) :=
  readNat_decW n _ fun c t h => by simp at h; rw [← h.1]; decide

/-- **Hello reads back.** -/
theorem readHello_helloW (h : Hello) (r : List UInt8) : readHello (helloW h ++ r) = some (h, r) := by
  have hb : ∀ {α β : Type} (a : α) (f : α → Option β), (some a >>= f) = f a := fun _ _ => rfl
  have hw : helloW h ++ r = (123 :: 10 :: (sp 2 ++ key "digests")) ++ (strStr h.digests ++ (fld 2 "flavor" ++
      (strStr h.flavor ++ (fld 2 "link" ++ ((123 :: 10 :: (sp 3 ++ key "m")) ++ (decW h.m ++ (fld 3 "points" ++
      (decW h.points ++ ((10 :: (sp 2 ++ [125])) ++ (fld 2 "profile" ++ (strStr h.profile ++ (fld 2 "reps" ++
      (decW h.reps ++ (fld 2 "rounds" ++ (strStr h.rounds ++ (fld 2 "tables" ++ (listW 2 strStr h.tables ++
      ((10 :: (sp 1 ++ [125])) ++ r)))))))))))))))))) := by
    unfold helloW
    simp only [List.cons_append, List.nil_append, List.append_assoc]
  rw [hw]
  dsimp only [readHello]
  rw [lit_append, hb, readStr_strStr, hb, lit_append, hb, readStr_strStr, hb, lit_append, hb, lit_append, hb,
    readNat_fld, hb, lit_append, hb, readNat_nl, hb, lit_append, hb, lit_append, hb, readStr_strStr, hb,
    lit_append, hb, readNat_fld, hb, lit_append, hb, readStr_strStr, hb, lit_append, hb,
    readList_listW 2 strStr readStr readStr_strStr, hb, lit_append, hb]
  rfl

/-- **The record reads back up to its link points**, whatever follows them. -/
theorem readFull_recordFullW (x : Record) (rest : List UInt8) : readFull (recordFullW x rest) = some x := by
  have hb : ∀ {α β : Type} (a : α) (f : α → Option β), (some a >>= f) = f a := fun _ _ => rfl
  have ht : ∀ (t : List (List UInt8)) r, readList 3 readHex (listW 3 hexStr t ++ r) = some (t, r) :=
    readList_listW 3 hexStr readHex readHex_hexStr
  have hw : recordFullW x rest = (123 :: 10 :: (sp 1 ++ key "rounds")) ++ (listW 1 roundW x.rounds ++
      (fld 1 "kind" ++ (strStr x.kind ++ (fld 1 "tops" ++ (strStr x.tops ++ (fld 1 "hello" ++ (helloW x.hello ++
      (fld 1 "link" ++ ((123 :: 10 :: (sp 2 ++ key "roots")) ++ (objW 2 (listW 3 hexStr) x.roots ++
      (fld 2 "points" ++ (listW 2 coinStr x.points ++ rest)))))))))))) := by
    simp only [recordFullW, recordW, List.cons_append, List.nil_append, List.append_assoc]
  rw [hw]
  dsimp only [readFull]
  rw [lit_append, hb, readList_listW 1 roundW readRound readRound_roundW, hb, lit_append, hb, readStr_strStr, hb,
    lit_append, hb, readStr_strStr, hb, lit_append, hb, readHello_helloW, hb, lit_append, hb, lit_append, hb,
    readObj_objW 2 _ _ ht, hb, lit_append, hb, readList_listW 2 coinStr readCoin readCoin_coinStr, hb]
  rfl


/-! ## The model of the coin server's record -/

/-- **A message's commit strings**: its 128-byte pieces in order, the last shorter where its length is no multiple
of 128. -/
def nodesOf (m : List UInt8) : List (List UInt8) :=
  if m = [] then [] else m.take 128 :: nodesOf (m.drop 128)
termination_by m.length
decreasing_by
  have : 0 < m.length := List.length_pos_iff.2 ‹¬m = []›
  simp only [List.length_drop]
  omega

theorem flatten_nodesOf (m : List UInt8) : (nodesOf m).flatten = m := by
  induction h : m.length using Nat.strongRecOn generalizing m with
  | _ n ih =>
    rw [nodesOf]
    split
    · simp_all
    · rename_i hm
      have : 0 < m.length := List.length_pos_iff.2 hm
      rw [List.flatten_cons, ih _ (by simp only [List.length_drop]; omega) _ rfl, List.take_append_drop]

theorem length_nodesOf (m : List UInt8) (n : Nat) (h : m.length ≤ 128 * n) : (nodesOf m).length ≤ n := by
  induction hl : m.length using Nat.strongRecOn generalizing m n with
  | _ l ih =>
    rw [nodesOf]
    split
    · simp
    · rename_i hm
      have : 0 < m.length := List.length_pos_iff.2 hm
      have h' := ih _ (by simp only [List.length_drop]; omega) (m.drop 128) (n - 1)
        (by simp only [List.length_drop]; omega) rfl
      simp only [List.length_cons]
      omega

section Group
variable {α : Type}

/-- `l` in consecutive pieces of the lengths `ns`. -/
def group : List Nat → List α → List (List α)
  | [], _ => []
  | n :: ns, l => l.take n :: group ns (l.drop n)

theorem flatten_group : ∀ (ns : List Nat) (l : List α), l.length ≤ ns.sum → (group ns l).flatten = l
  | [], l, h => by
    simp only [List.sum_nil, Nat.le_zero, List.length_eq_zero_iff] at h
    simp [group, h]
  | n :: ns, l, h => by
    simp only [List.sum_cons] at h
    rw [group, List.flatten_cons, flatten_group ns (l.drop n) (by simp only [List.length_drop]; omega),
      List.take_append_drop]

end Group

section Model
variable {β : Type}

/-- **The record's round of a message round** `(m, o)`, whose coin drew `n` values (`cv o n`): `m`'s first commit
strings as tops of the levels' node counts `a`'s tops have, the rest, where there is any, as its one `c_k`, and
everything else `a`'s. -/
def roundD (cv : β → Nat → List F128) (a : Round) (n : Nat) (mo : List UInt8 × β) : Round :=
  let ls := a.tops.map List.length
  { a with
    tops := group ls (nodesOf (mo.1.take (128 * ls.sum)))
    commits := if mo.1.drop (128 * ls.sum) = [] then [] else [mo.1.drop (128 * ls.sum)]
    n
    coins := cv mo.2 n }

theorem msg_roundD (cv : β → Nat → List F128) (a : Round) (n : Nat) (mo : List UInt8 × β) :
    ((roundD cv a n mo).tops.map List.flatten).flatten ++ (roundD cv a n mo).commits.flatten = mo.1 := by
  have hn := length_nodesOf (mo.1.take (128 * (a.tops.map List.length).sum)) (a.tops.map List.length).sum
    (by simp only [List.length_take]; omega)
  have hg := flatten_group _ _ hn
  simp only [roundD]
  rw [← List.flatten_flatten, hg, flatten_nodesOf]
  split
  · rename_i h
    rw [List.flatten_nil, List.append_nil]
    have := List.take_append_drop (128 * (a.tops.map List.length).sum) mo.1
    rw [h, List.append_nil] at this
    exact this
  · rw [List.flatten_singleton, List.take_append_drop]

theorem coins_roundD (cv : β → Nat → List F128) (a : Round) (n : Nat) (mo : List UInt8 × β) :
    (roundD cv a n mo).coins = cv mo.2 n := rfl

/-- **The model of the coin server's record of the message rounds `ms`** (`rec_live`'s `commit`, `round` and
`record`), each answered by `o` whose coin drew `cv o n` at its count `nOf i`: round 0's message as the link's one
table's level-0 top (that table Hello's first) and its coin as the link points; each later round before the last as
a recorded round (`roundD`), whose other fields are `aux`'s; nothing of the last round; and the record's kind, tops'
format and Hello `hd`'s. -/
def recOf (cv : β → Nat → List F128) (aux : Nat → Round) (nOf : Nat → Nat) (hd : Record)
    (ms : List (List UInt8 × β)) : Record :=
  { hd with
    rounds := ms.tail.dropLast.mapIdx fun j mo => roundD cv (aux j) (nOf (j + 1)) mo
    roots := [(hd.hello.tables.headD [], nodesOf ((ms.head?.map Prod.fst).getD []))]
    points := (ms.head?.map fun mo => cv mo.2 (nOf 0)).getD [] }

theorem length_msgs_recOf (cv : β → Nat → List F128) (aux : Nat → Round) (nOf : Nat → Nat) (hd : Record)
    (ms : List (List UInt8 × β)) : (recOf cv aux nOf hd ms).msgs.length = ms.length - 2 + 1 := by
  simp only [recOf, Record.msgs, List.length_cons, List.length_map, List.length_mapIdx, List.length_dropLast,
    List.length_tail]
  omega

/-- **The model shows each round it records as sent**: its `i`-th round is the `i`-th message round's message and
its coin's values. -/
theorem getD_msgs_recOf (cv : β → Nat → List F128) (aux : Nat → Round) (nOf : Nat → Nat) (hd : Record)
    (ms : List (List UInt8 × β)) (i : Nat) (hi : i < (recOf cv aux nOf hd ms).msgs.length) :
    (recOf cv aux nOf hd ms).msgs.getD i ([], []) =
      (((ms[i]?).map Prod.fst).getD [], ((ms[i]?).map fun mo => cv mo.2 (nOf i)).getD []) := by
  rw [length_msgs_recOf] at hi
  cases i with
  | zero =>
    cases ms with
    | nil => simp [recOf, Record.msgs, nodesOf]
    | cons mo ms => simp [recOf, Record.msgs, flatten_nodesOf]
  | succ j =>
    have hj : j < ms.tail.dropLast.length := by simp; omega
    have hm : ms[j + 1]? = some (ms.tail.dropLast[j]'hj) := by
      cases ms with
      | nil => simp at hj
      | cons mo ms => simp [List.getElem_dropLast]
    rw [hm]
    simp [recOf, Record.msgs, List.getD_eq_getElem?_getD, List.getElem?_mapIdx, List.getElem?_eq_getElem hj,
      msg_roundD, coins_roundD]

/-- **The model's rounds' messages are the message rounds' before the last.** -/
theorem fst_recOf (cv : β → Nat → List F128) (aux : Nat → Round) (nOf : Nat → Nat) (hd : Record)
    (ms : List (List UInt8 × β)) (i : Nat) (hi : i + 1 < ms.length) :
    (((recOf cv aux nOf hd ms).msgs.map fun mc => (mc.1, mc.2.toArray)).map Prod.fst).getD i [] =
      (ms.map Prod.fst).getD i [] := by
  have hl : i < (recOf cv aux nOf hd ms).msgs.length := by rw [length_msgs_recOf]; omega
  have h := getD_msgs_recOf cv aux nOf hd ms i hl
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hl, Option.getD_some] at h
  rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_map,
    List.getElem?_map, List.getElem?_eq_getElem hl, List.getElem?_eq_getElem (show i < ms.length by omega)]
  simp [h, List.getElem?_eq_getElem (show i < ms.length by omega)]

/-- **The model's rounds' coins are the message rounds' answers' values**, where there is a message round. -/
theorem snd_recOf (cv : β → Nat → List F128) (aux : Nat → Round) (nOf : Nat → Nat) (hd : Record)
    (ms : List (List UInt8 × β)) (i : Nat)
    (hi : i < ((recOf cv aux nOf hd ms).msgs.map fun mc => (mc.1, mc.2.toArray)).length) :
    (((recOf cv aux nOf hd ms).msgs.map fun mc => (mc.1, mc.2.toArray)).getD i ([], #[])).2 =
      ((ms[i]?).map fun mo => (cv mo.2 (nOf i)).toArray).getD #[] := by
  rw [List.length_map] at hi
  have h := getD_msgs_recOf cv aux nOf hd ms i hi
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hi, Option.getD_some] at h
  rw [List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_eq_getElem hi]
  cases hm : ms[i]? <;> simp_all

end Model


end Flock.RecRecord

namespace RecDeploy

open Claims (Msg Out Prover run coinSeq)
open FlockDeploy (Coin)

/-- **The coin a round's answer is**: the coin whose bytes it is (`coinOf`), where it is a message. -/
noncomputable def outCoin : Out → Coin
  | .msg bs => coinOf bs
  | _ => default

/-- **The values a round's answer drew**: its coin's first `n`. -/
noncomputable def coinVals (o : Out) (n : ℕ) : List Flock.F128 := ((outCoin o).vals n).toList

/-- **The coin server's transcript as its record**: at a compiled statement `x`, `rec_live`'s record of the inner
rounds as `recOf` models it, round `i` drawing `nOf x i` values, each recorded round's other fields `aux x j`, the
record's kind, tops' format and Hello `hd x`'s, and the record past its link points `rest x ms`. -/
noncomputable def recTr (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (rest : Msg → List (Msg × Out) → List UInt8) (x : Msg)
    (ms : List (Msg × Out)) : ByteArray :=
  ⟨(Flock.RecRecord.recordFullW (Flock.RecRecord.recOf coinVals (aux x) (nOf x) (hd x) ms) (rest x ms)).toArray⟩

/-- **The record's reader**: the rounds the record shows (`Record.msgs`), read in its own format (`readFull`). -/
def recRead : TrReader := fun _ t =>
  (Flock.RecRecord.readFull t.data.toList).map fun r => r.msgs.map fun mc => (mc.1, mc.2.toArray)

/-- **The record reads back at every accepted run, before the last round**: wherever the coin server's transcript is
its record (`recTr`, whatever `aux`, `nOf`, `hd` and `rest`), `recRead` reads it back to the run's inner messages
before the last, and every round it shows with the values its coin drew (`run_inner`: each inner round is answered
by its coin's bytes). -/
theorem readsBack_recTr (H : Msg → Msg) (IV : Claims.Verifier Coin) (b : Brings)
    (shape : Claims.Recursion.Form → Prop) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (rest : Msg → List (Msg × Out) → List UInt8)
    (hb : b.tr = recTr aux nOf hd rest) : ReadsBackB H IV b shape recRead := by
  intro x _ _ _ Ω P ω r hacc
  obtain ⟨xr, hx⟩ := run_acc_readX H IV b.st x P ω r hacc
  have hK := run_acc_length H IV b.st x P ω r hacc
  have hl : IV.k (xIOf x) ≤ (run H P (V IV b.st) x ω r).msgs.length := by
    rw [hK]
    show IV.k (xIOf x) ≤ IV.k (xIOf x) + kS x + _
    omega
  refine ⟨(Flock.RecRecord.recOf coinVals (aux x) (nOf x) (hd x)
    ((run H P (V IV b.st) x ω r).msgs.take (IV.k (xIOf x)))).msgs.map fun mc => (mc.1, mc.2.toArray), ?_, ?_, ?_⟩
  · rw [hb]
    simp only [recRead, recTr, List.toList_toArray, Flock.RecRecord.readFull_recordFullW, Option.map_some]
  · intro i hi
    rw [Flock.RecRecord.fst_recOf _ _ _ _ _ i (by rw [List.length_take]; omega), List.getD_eq_getElem?_getD,
      List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_map, List.getElem?_take_of_lt (by omega)]
  · intro i hi
    rw [Flock.RecRecord.snd_recOf _ _ _ _ _ i hi]
    by_cases hi' : i < IV.k (xIOf x)
    · obtain ⟨a, ha⟩ := run_inner H IV b.st x P ω r hx (i := i) (by simp only [kS]; omega) (by omega)
      have hm : ((run H P (V IV b.st) x ω r).msgs.take (IV.k (xIOf x)))[i]? =
          some (a, .msg (coinSeq r i).bytes) := by
        rw [List.getElem?_take_of_lt hi', ha]
      have hc : outCoin (.msg (coinSeq r i).bytes) = coinSeq r i := coinOf_bytes _
      rw [hm]
      simp only [Option.map_some, Option.getD_some, coinVals, hc]
      unfold Drawn
      simp [FlockDeploy.Coin.vals]
      apply Array.toList_inj.mp
      simp
    · rw [List.getElem?_eq_none (by rw [List.length_take]; omega)]
      simp [Drawn, FlockDeploy.Coin.vals]

end RecDeploy
