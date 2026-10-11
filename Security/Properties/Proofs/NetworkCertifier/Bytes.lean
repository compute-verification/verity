import Properties.NetworkCertifier
import Properties.Proofs.Identity
import Mathlib.Data.Set.Card

/-!
# The record's encodings are injective

Each string the check hashes determines what it encodes: a record head's encoding its head (`encode_inj`), a tagged
input its payload (`taggedInput_inj`), a certificate message's input its key identifier, link-window and digest
(`messageInput_inj`), a leaf's input its frame (`leafInput_inj`), a node's input its two children (`nodeInput_inj`);
and a payload's bytes its bits, for whole bytes of bits (`bitsOfBytes_bytesOfBits`). Byte strings are read as their
lists of bytes (`bs`), and an encoding is parsed field by field, each of a width it fixes.
-/

open NetworkCheck hiding Params
open NetworkCheck.Check
open Flock

set_option maxRecDepth 20000

namespace Properties.NetworkCertifier.Bytes

/-- A byte string's bytes. -/
def bs (b : ByteArray) : List UInt8 := b.data.toList

theorem bs_append (a b : ByteArray) : bs (a ++ b) = bs a ++ bs b := by
  simp [bs, ByteArray.data_append]

theorem bs_length (b : ByteArray) : (bs b).length = b.size := by
  simp [bs]

theorem bs_inj {a b : ByteArray} (h : bs a = bs b) : a = b := by
  apply ByteArray.ext
  simpa [bs] using h

/-- Two lists that agree after a prefix of one length agree on the prefix and on the rest. -/
theorem split {a a' r r' : List UInt8} (h : a ++ r = a' ++ r') (hl : a.length = a'.length) : a = a' ∧ r = r' :=
  List.append_inj h hl

theorem split_bytes {a a' : ByteArray} {r r' : List UInt8} (h : bs a ++ r = bs a' ++ r') (hl : a.size = a'.size) :
    a = a' ∧ r = r' := by
  obtain ⟨h1, h2⟩ := split h (by rw [bs_length, bs_length, hl])
  exact ⟨bs_inj h1, h2⟩

theorem error_ne_ok {ε α : Type} {s : ε} {x : α} (h : (Except.error s : Except ε α) = .ok x) : False := by
  cases h

/-! ## Big-endian integers -/

/-- The number big-endian bytes encode. -/
def ofBytes (l : List UInt8) : Nat := l.foldl (fun a b => a * 256 + b.toNat) 0

theorem ofBytes_append (l : List UInt8) (b : UInt8) : ofBytes (l ++ [b]) = ofBytes l * 256 + b.toNat := by
  simp [ofBytes, List.foldl_append]

theorem toNat_toUInt8 (n : Nat) : n.toUInt8.toNat = n % 256 := by
  simp [Nat.toUInt8]

theorem ofBytes_be : ∀ (w n : Nat), n < 2 ^ (8 * w) →
    ofBytes ((List.range w).map fun i => (n >>> (8 * (w - 1 - i))).toUInt8) = n
  | 0, n, h => by simp at h; simp [ofBytes, h]
  | w + 1, n, h => by
    rw [List.range_succ, List.map_append, List.map_singleton, ofBytes_append]
    have hm : (List.range w).map (fun i => (n >>> (8 * (w + 1 - 1 - i))).toUInt8) =
        (List.range w).map (fun i => ((n / 256) >>> (8 * (w - 1 - i))).toUInt8) := by
      apply List.map_congr_left
      intro i hi
      rw [List.mem_range] at hi
      congr 1
      rw [Nat.shiftRight_eq_div_pow, Nat.shiftRight_eq_div_pow, Nat.div_div_eq_div_mul,
        show 8 * (w + 1 - 1 - i) = 8 + 8 * (w - 1 - i) by omega, Nat.pow_add]
    have h' : n / 256 < 2 ^ (8 * w) := by
      rw [Nat.div_lt_iff_lt_mul (by decide), show 2 ^ (8 * w) * 256 = 2 ^ (8 * (w + 1)) by
        rw [show 8 * (w + 1) = 8 * w + 8 by omega, Nat.pow_add]]
      exact h
    rw [hm, ofBytes_be w (n / 256) h', toNat_toUInt8]
    simp
    omega

theorem uBE_ok {w : Nat} {x : Int} {n : String} {b : ByteArray} (h : uBE w x n = .ok b) :
    0 ≤ x ∧ x < 2 ^ (8 * w) ∧ bs b = (List.range w).map fun i => (x.toNat >>> (8 * (w - 1 - i))).toUInt8 := by
  unfold uBE at h
  split at h
  · rename_i hx
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact ⟨hx.1, hx.2, by simp [bs]⟩
  · simp [throw, throwThe, MonadExceptOf.throw] at h

theorem uBE_size {w : Nat} {x : Int} {n : String} {b : ByteArray} (h : uBE w x n = .ok b) : b.size = w := by
  rw [← bs_length, (uBE_ok h).2.2]
  simp

theorem uBE_value {w : Nat} {x : Int} {n : String} {b : ByteArray} (h : uBE w x n = .ok b) : ofBytes (bs b) = x := by
  obtain ⟨h0, h1, h2⟩ := uBE_ok h
  rw [h2, ofBytes_be w x.toNat ((Int.toNat_lt h0).mpr (by exact_mod_cast h1))]
  omega

/-- A big-endian field and what follows it determine the field's value and what follows. -/
theorem uBE_split {w : Nat} {x y : Int} {n n' : String} {b b' : ByteArray} {r r' : List UInt8}
    (h : uBE w x n = .ok b) (h' : uBE w y n' = .ok b') (e : bs b ++ r = bs b' ++ r') : x = y ∧ r = r' := by
  obtain ⟨hb, hr⟩ := split_bytes e (by rw [uBE_size h, uBE_size h'])
  subst hb
  exact ⟨by rw [← uBE_value h, ← uBE_value h'], hr⟩

theorem be64_bs (x : UInt64) : bs (be64 x) =
    [(x >>> 56).toUInt8, (x >>> 48).toUInt8, (x >>> 40).toUInt8, (x >>> 32).toUInt8,
     (x >>> 24).toUInt8, (x >>> 16).toUInt8, (x >>> 8).toUInt8, (x >>> 0).toUInt8] := by
  simp [be64, bs, List.range', ByteArray.push, ByteArray.emptyWithCapacity]
  rfl

theorem be64_size (x : UInt64) : (be64 x).size = 8 := by
  rw [← bs_length, be64_bs]
  rfl

theorem be32_size (x : UInt32) : (be32 x).size = 4 := by
  simp [be32]
  rfl

theorem byte_toNat (x : UInt64) (k : Nat) (hk : k < 64) :
    ((x >>> (UInt64.ofNat k)).toUInt8).toNat = x.toNat / 2 ^ k % 256 := by
  simp [UInt64.toNat_shiftRight, Nat.shiftRight_eq_div_pow, Nat.mod_eq_of_lt hk]

theorem be64_inj {x y : UInt64} (h : be64 x = be64 y) : x = y := by
  have h' := congrArg bs h
  rw [be64_bs, be64_bs] at h'
  simp only [List.cons.injEq] at h'
  obtain ⟨h0, h1, h2, h3, h4, h5, h6, h7, -⟩ := h'
  have e : ∀ k, k < 64 → (x >>> (UInt64.ofNat k)).toUInt8 = (y >>> (UInt64.ofNat k)).toUInt8 →
      x.toNat / 2 ^ k % 256 = y.toNat / 2 ^ k % 256 := fun k hk he => by
    rw [← byte_toNat x k hk, ← byte_toNat y k hk, he]
  have a0 := e 56 (by omega) h0
  have a1 := e 48 (by omega) h1
  have a2 := e 40 (by omega) h2
  have a3 := e 32 (by omega) h3
  have a4 := e 24 (by omega) h4
  have a5 := e 16 (by omega) h5
  have a6 := e 8 (by omega) h6
  have a7 := e 0 (by omega) h7
  have hx := x.toNat_lt
  have hy := y.toNat_lt
  apply UInt64.toNat_inj.mp
  simp only [Nat.reducePow] at a0 a1 a2 a3 a4 a5 a6 a7 hx hy ⊢
  omega

/-- `field`'s length prefix of a string under `2^64` bytes determines its length. -/
theorem be64_size_inj {n m : Nat} (hn : n < 2 ^ 64) (hm : m < 2 ^ 64) (h : be64 n.toUInt64 = be64 m.toUInt64) :
    n = m := by
  have := congrArg UInt64.toNat (be64_inj h)
  simp only [Nat.toUInt64, UInt64.toNat_ofNat'] at this
  rwa [Nat.mod_eq_of_lt hn, Nat.mod_eq_of_lt hm] at this

/-! ## Text and status -/

theorem text_ok {s n : String} {b : ByteArray} (h : text s n = .ok b) :
    ∃ m lb, uBE 2 s.toUTF8.size m = .ok lb ∧ bs b = bs lb ++ bs s.toUTF8 := by
  unfold text at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  split at h
  · simp at h
  · rename_i lb hlb
    simp only [Except.ok.injEq] at h
    subst h
    exact ⟨_, lb, hlb, bs_append _ _⟩

theorem text_short {s n : String} {b : ByteArray} (h : text s n = .ok b) : s.toUTF8.size < 2 ^ 16 := by
  obtain ⟨m, lb, hlb, -⟩ := text_ok h
  have := (uBE_ok hlb).2.1
  omega

/-- A text field and what follows it determine the text and what follows. -/
theorem text_split {s s' n n' : String} {b b' : ByteArray} {r r' : List UInt8}
    (h : text s n = .ok b) (h' : text s' n' = .ok b') (e : bs b ++ r = bs b' ++ r') : s = s' ∧ r = r' := by
  obtain ⟨m, lb, hlb, hb⟩ := text_ok h
  obtain ⟨m', lb', hlb', hb'⟩ := text_ok h'
  rw [hb, hb', List.append_assoc, List.append_assoc] at e
  obtain ⟨hsz, e⟩ := uBE_split hlb hlb' e
  obtain ⟨hs, hr⟩ := split_bytes e (by exact_mod_cast hsz)
  exact ⟨String.toByteArray_inj.mp hs, hr⟩

theorem statusCode_inj {a b : Option String} {n : Nat} (ha : statusCode a = .ok n) (hb : statusCode b = .ok n) :
    a = b := by
  unfold statusCode at ha hb
  split at ha <;> split at hb <;> simp_all [pure, Except.pure, throw, throwThe, MonadExceptOf.throw] <;> omega

theorem code_inj {d d' : Direction} (h : (d.code : Int) = d'.code) : d = d' := by
  cases d <;> cases d' <;> simp_all [Direction.code]

/-! ## The record head -/

/-- The bytes of the counts, four per count. -/
def CountsEnc : List Int → List UInt8 → Prop
  | [], l => l = []
  | n :: ns, l => ∃ b rest, uBE 4 n "count" = .ok b ∧ l = bs b ++ rest ∧ CountsEnc ns rest

theorem countsEnc_length : ∀ {ns : List Int} {l : List UInt8}, CountsEnc ns l → l.length = 4 * ns.length
  | [], l, h => by simp [CountsEnc] at h; simp [h]
  | n :: ns, l, ⟨b, rest, hb, hl, hr⟩ => by
    rw [hl, List.length_append, countsEnc_length hr, bs_length, uBE_size hb]
    simp
    omega

theorem countsEnc_inj : ∀ {ns ns' : List Int} {l : List UInt8}, CountsEnc ns l → CountsEnc ns' l → ns = ns'
  | [], [], _, _, _ => rfl
  | [], _ :: _, l, h, ⟨b, _, hb, hl, _⟩ => by
    simp only [CountsEnc] at h
    subst h
    have := congrArg List.length hl
    rw [List.length_append, bs_length, uBE_size hb] at this
    simp at this
    omega
  | _ :: _, [], l, ⟨b, _, hb, hl, _⟩, h => by
    simp only [CountsEnc] at h
    subst h
    have := congrArg List.length hl
    rw [List.length_append, bs_length, uBE_size hb] at this
    simp at this
    omega
  | n :: ns, n' :: ns', l, ⟨b, rest, hb, hl, hr⟩, ⟨b', rest', hb', hl', hr'⟩ => by
    rw [hl] at hl'
    obtain ⟨hn, hrest⟩ := uBE_split hb hb' hl'
    subst hn hrest
    rw [countsEnc_inj hr hr']

theorem forIn_counts (f : Int → ByteArray → Except String (ForInStep ByteArray))
    (hf : ∀ n s y, f n s = .ok y → ∃ v, uBE 4 n "count" = .ok v ∧ y = .yield (s ++ v)) :
    ∀ (ns : List Int) (a y : ByteArray), forIn ns a f = .ok y → ∃ rest, bs y = bs a ++ rest ∧ CountsEnc ns rest
  | [], a, y, h => by
    simp only [List.forIn_nil, pure, Except.pure, Except.ok.injEq] at h
    exact ⟨[], by simp [h], rfl⟩
  | n :: ns, a, y, h => by
    rw [List.forIn_cons] at h
    simp only [bind, Except.bind] at h
    split at h
    · simp at h
    · rename_i r hb
      obtain ⟨b, hu, rfl⟩ := hf n a r hb
      obtain ⟨rest, h1, h2⟩ := forIn_counts f hf ns (a ++ b) y h
      exact ⟨bs b ++ rest, by rw [h1, bs_append, List.append_assoc], b, rest, hu, rfl, h2⟩

/-- What `RecordHead.encode` checks and writes, field by field. -/
structure EncodeView (h : RecordHead) (x : ByteArray) : Prop where
  counts : h.counts.size = h.buckets
  params : h.parameters.size = 64
  root : h.root.size = 64
  parts : ∃ lk dir win bk al fb bm sc st dl rest,
    text h.link "link" = .ok lk ∧ uBE 1 h.direction.code "direction" = .ok dir ∧ uBE 8 h.window "window" = .ok win ∧
    uBE 4 h.buckets "T" = .ok bk ∧ uBE 4 h.allocation "r" = .ok al ∧ uBE 4 h.frameBytes "B" = .ok fb ∧
    uBE 4 h.bucketMs "τ_b" = .ok bm ∧ statusCode h.status = .ok sc ∧ uBE 1 sc "status" = .ok st ∧
    uBE 1 (if h.deadline then 1 else 0) "deadline" = .ok dl ∧ CountsEnc h.counts.toList rest ∧
    bs x = bs lk ++ (bs dir ++ (bs win ++ (bs bk ++ (bs al ++ (bs fb ++ (bs bm ++ (bs st ++ (bs dl ++
      (bs h.parameters ++ (rest ++ bs h.root))))))))))

theorem encode_view {h : RecordHead} {x : ByteArray} (he : h.encode = .ok x) : EncodeView h x := by
  unfold RecordHead.encode at he
  simp only [bind, Except.bind, pure, Except.pure] at he
  split at he
  · simp [throw, throwThe, MonadExceptOf.throw] at he
  rename_i hc
  split at he
  · simp [throw, throwThe, MonadExceptOf.throw] at he
  rename_i hp
  repeat' (split at he; · exact absurd he (by simp))
  rename_i _ lk hlk _ dir hdir _ win hwin _ bk hbk _ al hal _ fb hfb _ bm hbm _ sc hsc _ st hst _ dl hdl _ y hy
  simp only [Except.ok.injEq] at he
  subst he
  rw [← Array.forIn_toList] at hy
  have hcs := forIn_counts _ ?hf _ _ _ hy
  case hf =>
    intro n s y h
    split at h
    · simp at h
    · rename_i v hv; simp only [Except.ok.injEq] at h; exact ⟨v, hv, h.symm⟩
  obtain ⟨rest, h1, h2⟩ := hcs
  refine ⟨by omega, by simp [digestBytes] at hp; omega, by simp [digestBytes] at hp; omega,
    _, _, _, _, _, _, _, _, _, _, rest, hlk, hdir, hwin, hbk, hal, hfb, hbm, hsc, hst, hdl, h2, ?_⟩
  simp only [bs_append, h1, List.append_assoc]

theorem encode_link {h : RecordHead} {x : ByteArray} (he : h.encode = .ok x) : h.link.toUTF8.size < 2 ^ 16 := by
  obtain ⟨-, -, -, ⟨lk, -, -, -, -, -, -, -, -, -, -, hlk, -⟩⟩ := encode_view he
  exact text_short hlk

theorem encode_size {h : RecordHead} {x : ByteArray} (he : h.encode = .ok x) : x.size < 2 ^ 64 := by
  obtain ⟨hc, hpa, hro, ⟨lk, dir, win, bk, al, fb, bm, sc, st, dl, rest, hlk, hdir, hwin, hbk, hal, hfb, hbm, hsc,
    hst, hdl, hrest, hx⟩⟩ := encode_view he
  obtain ⟨m, lb, hlb, hlk'⟩ := text_ok hlk
  have h1 := congrArg List.length hx
  have h2 := congrArg List.length hlk'
  simp only [List.length_append, bs_length, countsEnc_length hrest, Array.length_toList] at h1 h2
  rw [uBE_size hdir, uBE_size hwin, uBE_size hbk, uBE_size hal, uBE_size hfb, uBE_size hbm, uBE_size hst,
    uBE_size hdl, hpa, hro] at h1
  rw [uBE_size hlb] at h2
  have h3 := text_short hlk
  have h4 : (h.counts.size : Int) < 4294967296 := by rw [hc]; exact (uBE_ok hbk).2.1
  omega

theorem encode_inj {h h' : RecordHead} {x : ByteArray} (he : h.encode = .ok x) (he' : h'.encode = .ok x) :
    h = h' := by
  obtain ⟨hc, hpa, hro, ⟨lk, dir, win, bk, al, fb, bm, sc, st, dl, rest, hlk, hdir, hwin, hbk, hal, hfb, hbm, hsc,
    hst, hdl, hrest, hx⟩⟩ := encode_view he
  obtain ⟨hc', hpa', hro', ⟨lk', dir', win', bk', al', fb', bm', sc', st', dl', rest', hlk', hdir', hwin', hbk', hal',
    hfb', hbm', hsc', hst', hdl', hrest', hx'⟩⟩ := encode_view he'
  have e := hx.symm.trans hx'
  obtain ⟨e1, e⟩ := text_split hlk hlk' e
  obtain ⟨e2, e⟩ := uBE_split hdir hdir' e
  obtain ⟨e3, e⟩ := uBE_split hwin hwin' e
  obtain ⟨e4, e⟩ := uBE_split hbk hbk' e
  obtain ⟨e5, e⟩ := uBE_split hal hal' e
  obtain ⟨e6, e⟩ := uBE_split hfb hfb' e
  obtain ⟨e7, e⟩ := uBE_split hbm hbm' e
  obtain ⟨e8, e⟩ := uBE_split hst hst' e
  obtain ⟨e9, e⟩ := uBE_split hdl hdl' e
  obtain ⟨e10, e⟩ := split_bytes e (by rw [hpa, hpa'])
  have hl : rest.length = rest'.length := by
    rw [countsEnc_length hrest, countsEnc_length hrest', Array.length_toList, Array.length_toList]
    omega
  obtain ⟨e11, eroot⟩ := split e hl
  subst e11
  have e12 := countsEnc_inj hrest hrest'
  have e13 := bs_inj eroot
  have e14 := statusCode_inj hsc (by rw [show sc = sc' by exact_mod_cast e8]; exact hsc')
  have e15 : h.direction = h'.direction := code_inj e2
  have e16 : h.deadline = h'.deadline := by
    cases hd : h.deadline <;> cases hd' : h'.deadline <;> simp_all
  have e17 : h.counts = h'.counts := Array.toList_inj.mp e12
  cases h; cases h'
  simp_all

/-! ## The SHA-512 inputs -/

theorem taggedInput_inj {tag : String} {a b x : ByteArray} (h : Properties.Identity.taggedInput tag a = .ok x)
    (h' : Properties.Identity.taggedInput tag b = .ok x) : a = b := by
  obtain ⟨-, rfl⟩ := Properties.Identity.taggedInput_ok.mp h
  obtain ⟨-, e⟩ := Properties.Identity.taggedInput_ok.mp h'
  have h' := congrArg bs e
  simp only [bs_append, List.append_assoc] at h'
  have h'' := List.append_cancel_left (List.append_cancel_left (List.append_cancel_left h'))
  exact (split_bytes h'' (by rw [be64_size, be64_size])).2 |> bs_inj

theorem field_bs (b : ByteArray) : bs (field b) = bs (be64 b.size.toUInt64) ++ bs b := bs_append _ _

theorem messageInput_ok {k : ByteArray} {l : String} {d : Direction} {w : Int} {g x : ByteArray}
    (h : messageInput k l d w g = .ok x) :
    g.size = 64 ∧ k.size = 16 ∧ ∃ wb, uBE 8 w "window" = .ok wb ∧
      x = field "verity/network-certificate/v1".toUTF8 ++ k ++
        field (field l.toUTF8 ++ field d.value.toUTF8 ++ wb) ++ field g := by
  unfold messageInput at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  split at h
  · simp [throw, throwThe, MonadExceptOf.throw] at h
  rename_i hg
  split at h
  · simp at h
  rename_i wb hwb
  split at h
  · simp [throw, throwThe, MonadExceptOf.throw] at h
  rename_i hk
  simp only [Except.ok.injEq] at h
  exact ⟨by simpa [digestBytes] using hg, by simpa using hk, wb, hwb, h.symm⟩

theorem value_inj {d d' : Direction} (h : d.value.toUTF8 = d'.value.toUTF8) : d = d' := by
  cases d <;> cases d' <;> first | rfl | (exfalso; revert h; decide)

/-- A certificate message's input determines its key identifier, link (of fewer than `2^16` bytes, as every encodable
head's is), direction, window and digest. -/
theorem messageInput_inj {k k' : ByteArray} {l l' : String} {d d' : Direction} {w w' : Int} {g g' x : ByteArray}
    (h : messageInput k l d w g = .ok x) (h' : messageInput k' l' d' w' g' = .ok x)
    (hl : l.toUTF8.size < 2 ^ 16) (hl' : l'.toUTF8.size < 2 ^ 16) :
    k = k' ∧ l = l' ∧ d = d' ∧ w = w' ∧ g = g' := by
  obtain ⟨hg, hk, wb, hwb, rfl⟩ := messageInput_ok h
  obtain ⟨hg', hk', wb', hwb', hx⟩ := messageInput_ok h'
  have e := congrArg bs hx
  simp only [bs_append, field_bs, List.append_assoc] at e
  have e := List.append_cancel_left (List.append_cancel_left e)
  obtain ⟨ek, e⟩ := split_bytes e (by rw [hk, hk'])
  obtain ⟨-, e⟩ := split_bytes e (by rw [be64_size, be64_size])
  obtain ⟨el, e⟩ := split_bytes e (by rw [be64_size, be64_size])
  have hsz := be64_size_inj (by omega) (by omega) el
  obtain ⟨els, e2⟩ := split_bytes e hsz
  obtain ⟨ed, e2⟩ := split_bytes e2 (by rw [be64_size, be64_size])
  have hdsz := be64_size_inj (by cases d <;> decide) (by cases d' <;> decide) ed
  obtain ⟨edv, e2⟩ := split_bytes e2 hdsz
  obtain ⟨ew, e2⟩ := uBE_split hwb hwb' e2
  obtain ⟨-, e2⟩ := split_bytes e2 (by rw [be64_size, be64_size])
  exact ⟨ek, String.toByteArray_inj.mp els, value_inj edv, ew, bs_inj e2⟩

theorem block_ok {x b : ByteArray} (h : block x = .ok b) : bs b = bs x ++ List.replicate (128 - x.size) 0 := by
  unfold block at h
  split at h
  · simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    simp [bs]
  · simp [throw, throwThe, MonadExceptOf.throw] at h

theorem leafInput_ok {f x : ByteArray} {B : Nat} (h : leafInput f B = .ok x) :
    f.size = B ∧ ∃ u blk, uBE 4 B "B" = .ok u ∧ block (frameTag ++ u) = .ok blk ∧ x = blk ++ f := by
  unfold leafInput at h
  by_cases hf : f.size ≠ B
  · rw [ite_eq_left hf] at h
    exact (error_ne_ok h).elim
  rw [ite_eq_right hf] at h
  refine ⟨not_not.mp hf, ?_⟩
  cases hu : uBE 4 B "B" with
  | error _ => rw [hu] at h; exact (error_ne_ok h).elim
  | ok u =>
    rw [hu] at h
    have h' : (block (frameTag ++ u) >>= fun b => pure (b ++ f)) = Except.ok x := h
    cases hb : block (frameTag ++ u) with
    | error _ => rw [hb] at h'; exact (error_ne_ok h').elim
    | ok blk =>
      rw [hb] at h'
      exact ⟨u, blk, rfl, hb, (Except.ok.inj h').symm⟩

theorem leafInput_inj {f f' x : ByteArray} {B : Nat} (h : leafInput f B = .ok x) (h' : leafInput f' B = .ok x) :
    f = f' := by
  obtain ⟨-, u, blk, hu, hblk, rfl⟩ := leafInput_ok h
  obtain ⟨-, u', blk', hu', hblk', hx⟩ := leafInput_ok h'
  rw [hu] at hu'
  cases hu'
  rw [hblk] at hblk'
  cases hblk'
  have e := congrArg bs hx
  simp only [bs_append] at e
  exact bs_inj (List.append_cancel_left e)

theorem leafInput_ne_emptyInput {f x : ByteArray} {B : Nat} (h : leafInput f B = .ok x) : x ≠ emptyInput := by
  obtain ⟨-, u, blk, -, hblk, rfl⟩ := leafInput_ok h
  intro he
  have e := congrArg (fun b => (bs b)[25]?) he
  simp only [bs_append, block_ok hblk, List.append_assoc] at e
  rw [List.getElem?_append_left (by rw [bs_length]; decide), emptyInput, bs_append,
    List.getElem?_append_left (by rw [bs_length]; decide)] at e
  revert e
  decide

theorem nodeInput_inj {a b a' b' : ByteArray} (ha : a.size = 64) (ha' : a'.size = 64)
    (h : nodeInput a b = nodeInput a' b') : a = a' ∧ b = b' := by
  have e := congrArg bs h
  simp only [nodeInput, bs_append, List.append_assoc] at e
  obtain ⟨e1, e2⟩ := split_bytes (List.append_cancel_left e) (by rw [ha, ha'])
  exact ⟨e1, bs_inj e2⟩

theorem hash_size (m : ByteArray) : (Sha512.hash m).size = 64 := by
  unfold Sha512.hash
  exact Array.size_ofFn

theorem empty_size (l : Nat) : (empty l).size = 64 := by
  cases l <;> simp only [empty, node] <;> exact hash_size _

/-! ## Bits and bytes -/

theorem bitsOfBytes_cons (x : UInt8) (l : List UInt8) :
    bitsOfBytes ⟨(x :: l).toArray⟩ = bitsOfBytes ⟨#[x]⟩ ++ bitsOfBytes ⟨l.toArray⟩ := by
  simp [bitsOfBytes]

theorem bitsOfByte (b0 b1 b2 b3 b4 b5 b6 b7 : Bool) :
    bitsOfBytes ⟨#[byteOfBits [b0, b1, b2, b3, b4, b5, b6, b7]]⟩ = [b0, b1, b2, b3, b4, b5, b6, b7] := by
  revert b0 b1 b2 b3 b4 b5 b6 b7
  decide

theorem bitsOfBytes_bytesOfBitsList : ∀ (x : List Bool), x.length % 8 = 0 →
    bitsOfBytes ⟨(bytesOfBitsList x).toArray⟩ = x
  | [], _ => by simp [bytesOfBitsList, bitsOfBytes]
  | b0 :: b1 :: b2 :: b3 :: b4 :: b5 :: b6 :: b7 :: rest, h => by
    simp only [bytesOfBitsList]
    rw [bitsOfBytes_cons, bitsOfByte, bitsOfBytes_bytesOfBitsList rest (by simp at h; omega)]
    rfl
  | [_], h | [_, _], h | [_, _, _], h | [_, _, _, _], h | [_, _, _, _, _], h | [_, _, _, _, _, _], h
  | [_, _, _, _, _, _, _], h => by simp at h

/-- Whole bytes of bits come back from their bytes. -/
theorem bitsOfBytes_bytesOfBits {x : List Bool} (h : x.length % 8 = 0) : bitsOfBytes (bytesOfBits x) = x :=
  bitsOfBytes_bytesOfBitsList x h

theorem bytesOfBits_inj {x y : List Bool} (hx : x.length % 8 = 0) (hy : y.length % 8 = 0)
    (h : bytesOfBits x = bytesOfBits y) : x = y := by
  rw [← bitsOfBytes_bytesOfBits hx, ← bitsOfBytes_bytesOfBits hy, h]

end Properties.NetworkCertifier.Bytes
