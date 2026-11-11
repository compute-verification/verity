module

public import Spot.OneStageZk
public import Proofs.Properties.NetworkCertifier.Check.Bytes

/-!
# A member's session's evidence decodes one way

The bytes a member's session's closing round is opened after the SHA-512 of (`Spot.OneStageZk.evidenceBytes`) hold the
number of the evidence's parts, then each part's length and its bytes, each count in an 8-byte big-endian word. Read
as lists of bytes (`bs`):

* **The encoding** (`bs_evidenceBytes`): the count's word, then each part's length's word and its bytes (`enc`).
* **A word read back** (`word_value`, `beNat_bs`): `Flock.Draw.beNat` at a word `be64 x` is `x`.
* **The decoder inverts the encoder** (`evidenceParts_evidenceBytes`): `Spot.OneStageZk.evidenceParts` of the bytes of
  parts under `2^64` many, each under `2^64` bytes, is those parts. So two such lists of parts with the same bytes are
  one list (`evidenceBytes_inj`): a digest of the bytes fixes every part's bytes, up to one SHA-512 collision.
-/

@[expose] public section

namespace Verity.Sampled.MemberForm

open Flock (be64)
open Properties.NetworkCertifier.Bytes (bs bs_append bs_length bs_inj be64_bs be64_size byte_toNat toNat_toUInt8 ofBytes
  ofBytes_be)

/-! ## Lists -/

theorem take_append_length {α : Type} : ∀ (a b : List α), (a ++ b).take a.length = a
  | [], _ => rfl
  | x :: a, b => congrArg (x :: ·) (take_append_length a b)

theorem drop_append_length {α : Type} : ∀ (a b : List α), (a ++ b).drop a.length = b
  | [], _ => rfl
  | _ :: a, b => drop_append_length a b

theorem take_of_length {α : Type} {a b : List α} {n : ℕ} (h : a.length = n) : (a ++ b).take n = a := by
  subst h
  exact take_append_length a b

theorem drop_of_length {α : Type} {a b : List α} {n : ℕ} (h : a.length = n) : (a ++ b).drop n = b := by
  subst h
  exact drop_append_length a b

theorem drop_drop_of {α : Type} : ∀ (l : List α) (i j : ℕ), (l.drop i).drop j = l.drop (i + j)
  | l, 0, j => by
    rw [Nat.zero_add]
    rfl
  | [], i + 1, j => by simp
  | _ :: l, i + 1, j => by
    rw [Nat.add_right_comm]
    exact drop_drop_of l i j

/-! ## Bytes -/

theorem get!_data (b : ByteArray) (j : ℕ) (hj : j < b.size) :
    b.get! j = b.data.toList[j]'(by rw [Array.length_toList]; exact hj) := by
  show b.data[j]! = _
  rw [getElem!_pos b.data j hj]
  simp

/-- **A big-endian read is its bytes' fold**: `Flock.Draw.beNat` of `len` bytes of `b` from `off`, over `b`'s list
of bytes. -/
theorem beNat_bs (b : ByteArray) (off len : ℕ) (h : off + len ≤ b.size) :
    Flock.Draw.beNat b off len = (((bs b).drop off).take len).foldl (fun acc x => acc * 256 + x.toNat) 0 := by
  have e : ((bs b).drop off).take len = (List.range len).map fun i => b.get! (off + i) := by
    apply List.ext_getElem
    · simp [bs_length]
      omega
    · intro i h1 h2
      simp only [List.getElem_take, List.getElem_drop, List.getElem_map, List.getElem_range]
      exact (get!_data b (off + i) (by simp at h2; omega)).symm
  rw [e, List.foldl_map]
  rfl

/-- A byte of a word is that byte of its value. -/
theorem byte_eq (x : UInt64) (k : ℕ) (hk : k < 64) : (x >>> (UInt64.ofNat k)).toUInt8 = (x.toNat >>> k).toUInt8 := by
  apply UInt8.toNat_inj.mp
  rw [byte_toNat x k hk, toNat_toUInt8, Nat.shiftRight_eq_div_pow]

/-- **A word's bytes fold to its value.** -/
theorem word_value (x : UInt64) : (bs (be64 x)).foldl (fun acc y => acc * 256 + y.toNat) 0 = x.toNat := by
  have b0 : (x >>> 56).toUInt8 = (x.toNat >>> 56).toUInt8 := byte_eq x 56 (by omega)
  have b1 : (x >>> 48).toUInt8 = (x.toNat >>> 48).toUInt8 := byte_eq x 48 (by omega)
  have b2 : (x >>> 40).toUInt8 = (x.toNat >>> 40).toUInt8 := byte_eq x 40 (by omega)
  have b3 : (x >>> 32).toUInt8 = (x.toNat >>> 32).toUInt8 := byte_eq x 32 (by omega)
  have b4 : (x >>> 24).toUInt8 = (x.toNat >>> 24).toUInt8 := byte_eq x 24 (by omega)
  have b5 : (x >>> 16).toUInt8 = (x.toNat >>> 16).toUInt8 := byte_eq x 16 (by omega)
  have b6 : (x >>> 8).toUInt8 = (x.toNat >>> 8).toUInt8 := byte_eq x 8 (by omega)
  have b7 : (x >>> 0).toUInt8 = (x.toNat >>> 0).toUInt8 := byte_eq x 0 (by omega)
  have l : [(x >>> 56).toUInt8, (x >>> 48).toUInt8, (x >>> 40).toUInt8, (x >>> 32).toUInt8,
      (x >>> 24).toUInt8, (x >>> 16).toUInt8, (x >>> 8).toUInt8, (x >>> 0).toUInt8] =
      [(x.toNat >>> 56).toUInt8, (x.toNat >>> 48).toUInt8, (x.toNat >>> 40).toUInt8, (x.toNat >>> 32).toUInt8,
        (x.toNat >>> 24).toUInt8, (x.toNat >>> 16).toUInt8, (x.toNat >>> 8).toUInt8, (x.toNat >>> 0).toUInt8] := by
    rw [b0, b1, b2, b3, b4, b5, b6, b7]
  have e : bs (be64 x) = (List.range 8).map fun i => (x.toNat >>> (8 * (8 - 1 - i))).toUInt8 :=
    (be64_bs x).trans (l.trans rfl)
  have hx : x.toNat < 2 ^ (8 * 8) := x.toNat_lt
  rw [e]
  exact ofBytes_be 8 x.toNat hx

/-- A length under `2^64` is its word's value. -/
theorem toNat_size {n : ℕ} (h : n < 2 ^ 64) : n.toUInt64.toNat = n := by
  simp only [Nat.toUInt64, UInt64.toNat_ofNat']
  exact Nat.mod_eq_of_lt h

theorem bs_extract (b : ByteArray) (s e : ℕ) : bs (b.extract s e) = ((bs b).drop s).take (e - s) := by
  simp only [bs, ByteArray.data_extract, Array.toList_extract, List.extract_eq_take_drop]

/-! ## The encoding -/

/-- The parts `l`'s bytes as `Spot.OneStageZk.evidenceBytes` lays them out after their count: each part's length's
word, then its bytes. -/
def enc : List ByteArray → List UInt8
  | [] => []
  | p :: l => bs (be64 p.size.toUInt64) ++ (bs p ++ enc l)

theorem bs_foldl : ∀ (l : List ByteArray) (acc : ByteArray),
    bs (l.foldl (fun acc p => acc ++ be64 p.size.toUInt64 ++ p) acc) = bs acc ++ enc l
  | [], acc => by simp [enc]
  | p :: l, acc => by
    show bs (l.foldl (fun acc p => acc ++ be64 p.size.toUInt64 ++ p) (acc ++ be64 p.size.toUInt64 ++ p)) =
      bs acc ++ (bs (be64 p.size.toUInt64) ++ (bs p ++ enc l))
    rw [bs_foldl l, bs_append, bs_append]
    simp only [List.append_assoc]

/-- **The evidence's bytes**: the count's word, then the parts' (`enc`). -/
theorem bs_evidenceBytes (ps : Array ByteArray) :
    bs (_root_.Spot.OneStageZk.evidenceBytes ps) = bs (be64 ps.size.toUInt64) ++ enc ps.toList := by
  unfold _root_.Spot.OneStageZk.evidenceBytes
  rw [← Array.foldl_toList, bs_foldl]

/-! ## The decoder -/

/-- **The decoder's scan reads the parts back**: where `b`'s bytes from `o` are the parts `l`'s, each part under
`2^64` bytes, `partsFrom` at `l`'s count from `o` is `l`. -/
theorem partsFrom_enc : ∀ (l : List ByteArray) (b : ByteArray) (o : ℕ), (∀ p ∈ l, p.size < 2 ^ 64) →
    (bs b).drop o = enc l → o ≤ b.size → _root_.Spot.OneStageZk.partsFrom b l.length o = some l
  | [], b, o, _, h, ho => by
    have hlen := congrArg List.length h
    rw [List.length_drop, bs_length] at hlen
    have e : o = b.size := by
      have h0 : b.size - o = 0 := hlen
      omega
    subst e
    simp [_root_.Spot.OneStageZk.partsFrom]
  | p :: l, b, o, hp, h, ho => by
    have hps : p.size < 2 ^ 64 := hp p List.mem_cons_self
    have hA : (bs (be64 p.size.toUInt64)).length = 8 := by rw [bs_length, be64_size]
    have h' : (bs b).drop o = bs (be64 p.size.toUInt64) ++ (bs p ++ enc l) := h
    have hlen := congrArg List.length h'
    rw [List.length_drop, bs_length, List.length_append, List.length_append, hA, bs_length] at hlen
    have h8 : o + 8 ≤ b.size := by omega
    have hbe : Flock.Draw.beNat b o 8 = p.size := by
      rw [beNat_bs b o 8 h8, h', take_of_length hA, word_value, toNat_size hps]
    have hd : (bs b).drop (o + 8) = bs p ++ enc l := by
      rw [← drop_drop_of, h', drop_of_length hA]
    have hstop : o + 8 + p.size ≤ b.size := by omega
    have hext : b.extract (o + 8) (o + 8 + p.size) = p := by
      apply bs_inj
      rw [bs_extract, hd, Nat.add_sub_cancel_left, take_of_length (bs_length p)]
    have hd' : (bs b).drop (o + 8 + p.size) = enc l := by
      rw [← drop_drop_of, hd, drop_of_length (bs_length p)]
    have ih := partsFrom_enc l b (o + 8 + p.size) (fun q hq => hp q (List.mem_cons_of_mem _ hq)) hd' hstop
    show _root_.Spot.OneStageZk.partsFrom b (l.length + 1) o = some (p :: l)
    simp only [_root_.Spot.OneStageZk.partsFrom]
    rw [ite_eq_left h8, hbe, ite_eq_left hstop, ih, hext]
    rfl

/-- **The decoder inverts the encoder**: `evidenceParts` of the bytes of the parts `ps`, under `2^64` many and each
under `2^64` bytes, is `ps`. -/
theorem evidenceParts_evidenceBytes (ps : Array ByteArray) (hn : ps.size < 2 ^ 64)
    (hp : ∀ p ∈ ps, p.size < 2 ^ 64) :
    _root_.Spot.OneStageZk.evidenceParts (_root_.Spot.OneStageZk.evidenceBytes ps) = some ps := by
  have hb := bs_evidenceBytes ps
  have hA : (bs (be64 ps.size.toUInt64)).length = 8 := by rw [bs_length, be64_size]
  have hlen := congrArg List.length hb
  rw [bs_length, List.length_append, hA] at hlen
  have h8 : 8 ≤ (_root_.Spot.OneStageZk.evidenceBytes ps).size := by omega
  have hbe : Flock.Draw.beNat (_root_.Spot.OneStageZk.evidenceBytes ps) 0 8 = ps.size := by
    rw [beNat_bs (_root_.Spot.OneStageZk.evidenceBytes ps) 0 8 (by omega), hb, List.drop_zero,
      take_of_length hA, word_value, toNat_size hn]
  have hd : (bs (_root_.Spot.OneStageZk.evidenceBytes ps)).drop 8 = enc ps.toList := by
    rw [hb, drop_of_length hA]
  have hl := partsFrom_enc ps.toList (_root_.Spot.OneStageZk.evidenceBytes ps) 8
    (fun p h => hp p (by simpa using h)) hd h8
  rw [Array.length_toList] at hl
  unfold _root_.Spot.OneStageZk.evidenceParts
  rw [ite_eq_left h8, hbe, hl]
  simp

/-- **Parts with the same bytes are the same parts**, each list under `2^64` parts and each part under `2^64` bytes:
the encoding decodes one way. -/
theorem evidenceBytes_inj {ps qs : Array ByteArray} (hn : ps.size < 2 ^ 64) (hp : ∀ p ∈ ps, p.size < 2 ^ 64)
    (hm : qs.size < 2 ^ 64) (hq : ∀ q ∈ qs, q.size < 2 ^ 64)
    (h : _root_.Spot.OneStageZk.evidenceBytes ps = _root_.Spot.OneStageZk.evidenceBytes qs) : ps = qs := by
  have h1 := evidenceParts_evidenceBytes ps hn hp
  rw [h, evidenceParts_evidenceBytes qs hm hq] at h1
  exact (Option.some.inj h1).symm

end Verity.Sampled.MemberForm
