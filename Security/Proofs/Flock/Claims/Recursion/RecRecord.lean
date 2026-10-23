import Proofs.Flock.Claims.Recursion.VStarCross
import Flock.RecRecord

/-!
# The coin server's record reads back

`Flock.RecRecord.readRecord`, the record format's own total reader, reads back the rounds `recordW` writes
(`readRecord_recordW`), piece by piece: literals, hex strings, the escaped stream name, decimal numbers, lists at their
indentation and coins. No JSON library is involved, so no property of one is assumed (`Claims.JsonSpec` stays a
completeness premise).

At the deployment's model (`recTr`, `recRead`) the coin server's transcript of the inner rounds is that record: each
round's message as its one commit string (none where the message is empty: `RecVStar.innerCommits` allows at most one
a round), the `nOf x i` values its coin drew, and everything else (stream, index, tops, level-0 count, the record's
other fields) whatever `aux` and `rest` say. `recRead` reads the transcript back to each round's message and drawn
values, at every accepted run (`readsBack_recTr`). The difftest
(`verity/ml/flock/tests/test_lean_rec_record.py`: `recordW` writes `rec_live.record_bytes`' bytes of the honest
records, and `readRecord` reads them) ties the format to the deployed writer, not the round list. The coin server's
record (`rec_live`'s `round`) lists only the rep streams' rounds, each with its tops and level-0 copies in their own
fields and only its `c_k` under `commits`. Round 0's roots and coins are in its `link` field, and the last round is
absent. `recTr` lists every inner round, with each whole message as its one commit string. So `recTr` is not yet the
deployed record, and a premise read at `cmtTr recRead` is about the model's record, not the coin server's.
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

end Flock.RecRecord

namespace RecDeploy

open Claims (Msg Out Prover run coinSeq)
open FlockDeploy (Coin)

/-- **The coin a round's answer is**: the coin whose bytes it is (`coinOf`), where it is a message. -/
noncomputable def outCoin : Out → Coin
  | .msg bs => coinOf bs
  | _ => default

/-- **The record's round of an inner round** `(m, o)`, which drew `n` values: the message as its one commit string
(none where it is empty), the coin's first `n` values, and the rest `a`'s. -/
noncomputable def roundOf (a : Flock.RecRecord.Round) (n : ℕ) (mo : Msg × Out) : Flock.RecRecord.Round :=
  { a with commits := if mo.1 = [] then [] else [mo.1], n, coins := ((outCoin mo.2).vals n).toList }

/-- **The coin server's transcript as its record**: at a compiled statement `x`, the inner rounds as the record's
rounds (`roundOf`), round `i` drawing `nOf x i` values with its other fields `aux x i`, then the record's other
fields `rest x ms`. -/
noncomputable def recTr (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (rest : Msg → List (Msg × Out) → List UInt8) (x : Msg) (ms : List (Msg × Out)) : ByteArray :=
  ⟨(Flock.RecRecord.recordW (ms.mapIdx fun i mo => roundOf (aux x i) (nOf x i) mo) (rest x ms)).toArray⟩

/-- **The record's reader**: each round's commit strings, joined, and its coins. -/
def recRead : TrReader := fun _ t =>
  (Flock.RecRecord.readRecord t.data.toList).map fun xs => xs.map fun r => (r.commits.flatten, r.coins.toArray)

/-- A round's message reads back from its commit strings. -/
theorem flatten_commits (m : Msg) : (if m = [] then [] else [m]).flatten = m := by
  split <;> simp_all

/-- **The record reads back at every accepted run**: wherever the coin server's transcript is its record (`recTr`,
whatever `aux`, `nOf` and `rest`), `recRead` reads it back to the run's inner messages, each with the values its coin
drew (`run_inner`: each inner round is answered by its coin's bytes). -/
theorem readsBack_recTr (H : Msg → Msg) (IV : Claims.Verifier Coin) (b : Brings)
    (shape : Claims.Recursion.Form → Prop) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (rest : Msg → List (Msg × Out) → List UInt8) (hb : b.tr = recTr aux nOf rest) :
    ReadsBack H IV b shape recRead := by
  intro x _ _ _ Ω P ω r hacc
  obtain ⟨xr, hx⟩ := run_acc_readX H IV b.st x P ω r hacc
  refine ⟨((run H P (V IV b.st) x ω r).msgs.take (IV.k (xIOf x))).mapIdx fun i mo =>
    (mo.1, (outCoin mo.2).vals (nOf x i)), ?_, ?_, ?_⟩
  · rw [hb]
    simp only [recRead, recTr, List.toList_toArray, Flock.RecRecord.readRecord_recordW, Option.map_some]
    congr 1
    apply List.ext_getElem (by simp)
    intro i h₁ h₂
    simp [roundOf, flatten_commits]
  · apply List.ext_getElem (by simp)
    intro i h₁ h₂
    simp
  · intro i hi
    have hl : i < (run H P (V IV b.st) x ω r).msgs.length ∧ i < IV.k (xIOf x) := by
      simp only [List.length_mapIdx, List.length_take] at hi
      omega
    obtain ⟨a, ha⟩ := run_inner H IV b.st x P ω r hx (i := i) (by simp only [kS]; omega) hl.1
    have hm : ((run H P (V IV b.st) x ω r).msgs.take (IV.k (xIOf x)))[i]? = some (a, .msg (coinSeq r i).bytes) := by
      rw [List.getElem?_take_of_lt hl.2, ha]
    have hc : outCoin (.msg (coinSeq r i).bytes) = coinSeq r i := coinOf_bytes _
    rw [List.getD_eq_getElem?_getD, List.getElem?_mapIdx, hm]
    simp only [Option.map_some, Option.getD_some, hc]
    unfold Drawn
    simp [FlockDeploy.Coin.vals]

end RecDeploy
