module

public import Properties.Pous
public import Proofs.Properties.Hash.Identity
public import Proofs.Flock.Bytes
public import Proofs.Properties.Pous
public import Proofs.Properties.Pous.Setup
public import Proofs.Pous.P2.RingTight
public import Proofs.Core.CanonUtf8
public import Proofs.Properties.Pous.RunAccepts
import all Init.Data.String.Defs
import all Lean.Data.Json.Basic
import all Lean.Data.Json.FromToJson.Basic
import all Init.Data.Ord.String
import all Init.Data.Array.Basic
import all Init.Data.Array.DecidableEq
import all Init.Data.List.Control
import all Init.Data.Range.Basic
import all Init.Data.Int.Repr
import all Init.Data.Repr
import all Init.Data.Nat.Power2.Basic

/-! The pieces of setup's accepted play that an honest `C*` needs before the setup check's loops, none of which
evaluates a hash. -/

@[expose] public section

namespace PousCheck.Properties.SecurityProofs.Accepts

open Flock PousCheck PousCheck.Prim Identity.Sha512
open Properties.Identity (L L_length L_append Rep rep_init rep_update chain chain_succ chain_congr finalize_eq
  L_padding zeroPad lenBytes getElem?_L_append_left update_update)
open Properties.Certifier.Bytes (be64_size be32_size)

theorem byte_lt (y : UInt8) : y.toNat < 256 := y.toNat_lt_size

def beStep (b : ByteArray) (acc i : Nat) : Nat := acc * 256 + (b.get! i).toNat

def beVal (b : ByteArray) (lo n : Nat) : Nat := (List.range' lo n).foldl (beStep b) 0

theorem foldl_beStep_lt (b : ByteArray) : ∀ (l : List Nat) (v m : Nat), v < 256 ^ m →
    l.foldl (beStep b) v < 256 ^ (m + l.length)
  | [], v, m, h => by simpa using h
  | i :: l, v, m, h => by
    rw [List.foldl_cons, List.length_cons, show m + (l.length + 1) = (m + 1) + l.length by omega]
    refine foldl_beStep_lt b l _ (m + 1) ?_
    have hy : (b.get! i).toNat < 256 := byte_lt _
    have hle : (v + 1) * 256 ≤ 256 ^ m * 256 := Nat.mul_le_mul_right _ h
    rw [Nat.pow_succ]
    unfold beStep
    generalize 256 ^ m = P at hle ⊢
    omega

theorem foldl_beStep_shift (b : ByteArray) : ∀ (l : List Nat) (v : Nat),
    l.foldl (beStep b) v = v * 256 ^ l.length + l.foldl (beStep b) 0
  | [], v => by simp
  | i :: l, v => by
    rw [List.foldl_cons, List.foldl_cons, foldl_beStep_shift b l (beStep b v i),
      foldl_beStep_shift b l (beStep b 0 i), List.length_cons, Nat.pow_succ]
    unfold beStep
    generalize 256 ^ l.length = P
    rw [Nat.zero_mul, Nat.zero_add, Nat.add_mul]
    ac_rfl

theorem beVal_lt (b : ByteArray) (lo n : Nat) : beVal b lo n < 256 ^ n := by
  have := foldl_beStep_lt b (List.range' lo n) 0 0 (by decide)
  rw [List.length_range', Nat.zero_add] at this
  exact this

theorem beVal_add (b : ByteArray) (lo m n : Nat) :
    beVal b lo (m + n) = beVal b lo m * 256 ^ n + beVal b (lo + m) n := by
  unfold beVal
  rw [← List.range'_append_1, List.foldl_append,
    foldl_beStep_shift b (List.range' (lo + m) n) (List.foldl (beStep b) 0 (List.range' lo m)), List.length_range']

theorem toNat_shift_or (x : UInt64) (y : UInt8) (hx : x.toNat < 2 ^ 56) :
    ((x <<< 8) ||| y.toUInt64).toNat = x.toNat * 256 + y.toNat := by
  have hy : y.toNat < 2 ^ 8 := lt_of_lt_of_eq (byte_lt y) (by decide)
  have hlt : x.toNat <<< 8 < 2 ^ 64 := by
    rw [Nat.shiftLeft_eq, show (2 : Nat) ^ 64 = 2 ^ 56 * 2 ^ 8 by decide]
    exact Nat.mul_lt_mul_of_pos_right hx (by decide)
  rw [UInt64.toNat_or, UInt64.toNat_shiftLeft, UInt8.toNat_toUInt64, show (8 : UInt64).toNat % 64 = 8 from rfl,
    Nat.mod_eq_of_lt hlt, ← Nat.shiftLeft_add_eq_or_of_lt hy, Nat.shiftLeft_eq]

theorem foldl_uint64 (b : ByteArray) : ∀ (l : List Nat) (x : UInt64) (m : Nat), x.toNat < 256 ^ m →
    m + l.length ≤ 8 →
    (l.foldl (fun (x : UInt64) (i : Nat) => (x <<< 8) ||| (b.get! i).toUInt64) x).toNat =
      l.foldl (beStep b) x.toNat
  | [], _, _, _, _ => rfl
  | i :: l, x, m, hx, hl => by
    rw [List.length_cons] at hl
    have h56 : x.toNat < 2 ^ 56 := by
      have : 256 ^ m ≤ 2 ^ 56 :=
        calc 256 ^ m ≤ 256 ^ 7 := Nat.pow_le_pow_right (by decide) (by omega)
          _ = 2 ^ 56 := by decide
      exact lt_of_lt_of_le hx this
    have hstep := toNat_shift_or x (b.get! i) h56
    have hy : (b.get! i).toNat < 256 := byte_lt _
    have hle : (x.toNat + 1) * 256 ≤ 256 ^ m * 256 := Nat.mul_le_mul_right _ hx
    have hnext : ((x <<< 8) ||| (b.get! i).toUInt64).toNat < 256 ^ (m + 1) := by
      rw [hstep, Nat.pow_succ]
      generalize 256 ^ m = P at hle ⊢
      omega
    rw [List.foldl_cons, List.foldl_cons, foldl_uint64 b l _ (m + 1) hnext (by omega), hstep]
    all_goals rfl

theorem natOfBERange_eq (b : ByteArray) (n : Nat) : ∀ lo, natOfBERange b lo (lo + n) = beVal b lo n := by
  induction n using Nat.strong_induction_on with
  | _ n ih =>
  intro lo
  rw [natOfBERange]
  by_cases h : n ≤ 8
  · have hc : lo + n - lo ≤ 8 := by omega
    rw [ite_cond_eq_true _ _ (eq_true hc)]
    simp only [Id.run, Std.Legacy.Range.forIn_eq_forIn_range', List.forIn_pure_yield_eq_foldl, bind_pure_comp,
      map_pure, Std.Legacy.Range.size, Nat.add_sub_cancel_left, Nat.add_sub_cancel, Nat.div_one]
    exact foldl_uint64 b (List.range' lo n) 0 0 (by decide)
      (by simp only [List.length_range', Nat.zero_add]; exact h)
  · have hc : ¬ lo + n - lo ≤ 8 := by omega
    rw [ite_cond_eq_false _ _ (eq_false hc)]
    dsimp only
    rw [Nat.add_sub_cancel_left, show lo + n - (lo + n / 2) = n - n / 2 by omega, ih (n / 2) (by omega) lo,
      show lo + n = lo + n / 2 + (n - n / 2) by omega, ih (n - n / 2) (by omega) (lo + n / 2)]
    have hlt : beVal b (lo + n / 2) (n - n / 2) < 2 ^ (8 * (n - n / 2)) := by
      rw [Nat.pow_mul]
      exact beVal_lt b _ _
    have hadd := beVal_add b lo (n / 2) (n - n / 2)
    rw [show n / 2 + (n - n / 2) = n by omega] at hadd
    rw [hadd, ← Nat.shiftLeft_add_eq_or_of_lt hlt, Nat.shiftLeft_eq, Nat.pow_mul]

theorem natOfBE_eq (b : ByteArray) : natOfBE b = beVal b 0 b.size := by
  have := natOfBERange_eq b b.size 0
  rwa [Nat.zero_add] at this

def suffix (s j : Nat) (purpose : UInt8) (i : Nat) : ByteArray :=
  Flock.be64 s.toUInt64 ++ Flock.be64 j.toUInt64 ++ ByteArray.mk #[purpose] ++ Flock.be32 i.toUInt32

def suffixHead (s j : Nat) (purpose : UInt8) : List UInt8 :=
  L (Flock.be64 s.toUInt64) ++ L (Flock.be64 j.toUInt64) ++ [purpose]

theorem suffixHead_length (s j : Nat) (purpose : UInt8) : (suffixHead s j purpose).length = 17 := by
  simp [suffixHead, be64_size]

def padTail (n : Nat) : List UInt8 := 128 :: List.replicate (zeroPad 21) 0 ++ lenBytes (n + 21)

theorem padTail_length (n : Nat) : (padTail n).length = 107 := by
  simp [padTail, zeroPad, lenBytes]

theorem L_be32 (i : Nat) :
    L (Flock.be32 i.toUInt32) = [(i >>> 24).toUInt8, (i >>> 16).toUInt8, (i >>> 8).toUInt8, i.toUInt8] := by
  have hb : Flock.be32 i.toUInt32 = ⟨#[(i.toUInt32 >>> 24).toUInt8, (i.toUInt32 >>> 16).toUInt8,
      (i.toUInt32 >>> 8).toUInt8, i.toUInt32.toUInt8]⟩ := by
    simp only [Flock.be32, Std.Legacy.Range.forIn_eq_forIn_range', List.forIn_pure_yield_eq_foldl,
      Std.Legacy.Range.size]
    rfl
  rw [hb]
  show [(i.toUInt32 >>> 24).toUInt8, (i.toUInt32 >>> 16).toUInt8, (i.toUInt32 >>> 8).toUInt8, i.toUInt32.toUInt8] = _
  simp only [List.cons.injEq, and_true]
  refine ⟨?_, ?_, ?_, ?_⟩ <;> apply UInt8.toNat_inj.1 <;>
    simp [UInt32.toNat_shiftRight, Nat.shiftRight_eq_div_pow] <;> omega

theorem L_suffix (s j : Nat) (purpose : UInt8) (i : Nat) :
    L (suffix s j purpose i) =
      suffixHead s j purpose ++ [(i >>> 24).toUInt8, (i >>> 16).toUInt8, (i >>> 8).toUInt8, i.toUInt8] := by
  rw [suffix, L_append, L_append, L_append, L_be32]
  all_goals rfl

theorem suffix_size (s j : Nat) (purpose : UInt8) (i : Nat) : (suffix s j purpose i).size = 21 := by
  rw [← L_length, L_suffix, List.length_append, suffixHead_length]
  all_goals rfl

theorem rep_suffix (pre : ByteArray) (s j : Nat) (purpose : UInt8) (i : Nat) :
    Rep ((Keys.Expander.ofPrefix pre).mid.update (suffix s j purpose i)) (pre ++ suffix s j purpose i) := by
  have := rep_update (rep_update rep_init pre) (suffix s j purpose i)
  rwa [ByteArray.empty_append] at this

theorem h_suffix (pre : ByteArray) (hpre : pre.size % 128 = 0) (s j : Nat) (purpose : UInt8) (i : Nat) :
    ((Keys.Expander.ofPrefix pre).mid.update (suffix s j purpose i)).h = chain Sha512.iv pre 0 (pre.size / 128) := by
  rw [(rep_suffix pre s j purpose i).h, ByteArray.size_append, suffix_size,
    show (pre.size + 21) / 128 = pre.size / 128 by omega]
  exact chain_congr _ _ fun k hk => by rw [Nat.zero_add]; exact getElem?_L_append_left pre _ (by omega)

theorem buf_suffix (pre : ByteArray) (hpre : pre.size % 128 = 0) (s j : Nat) (purpose : UInt8) (i : Nat) :
    L ((Keys.Expander.ofPrefix pre).mid.update (suffix s j purpose i)).buf = L (suffix s j purpose i) := by
  rw [(rep_suffix pre s j purpose i).buf, ByteArray.size_append, suffix_size,
    show 128 * ((pre.size + 21) / 128) = pre.size by omega, L_append]
  exact List.drop_left' (L_length pre)

theorem L_padding_suffix (pre : ByteArray) (hpre : pre.size % 128 = 0) (s j : Nat) (purpose : UInt8) (i : Nat) :
    L ((Keys.Expander.ofPrefix pre).mid.update (suffix s j purpose i)).padding =
      suffixHead s j purpose ++ (i >>> 24).toUInt8 :: (i >>> 16).toUInt8 :: (i >>> 8).toUInt8 :: i.toUInt8 ::
        padTail pre.size := by
  have hb := buf_suffix pre hpre s j purpose i
  have hbs : ((Keys.Expander.ofPrefix pre).mid.update (suffix s j purpose i)).buf.size = 21 := by
    rw [← L_length, hb, L_length, suffix_size]
  rw [L_padding, hb, hbs, (rep_suffix pre s j purpose i).len, ByteArray.size_append, suffix_size, L_suffix]
  simp [padTail]

theorem padding_suffix_size (pre : ByteArray) (hpre : pre.size % 128 = 0) (s j : Nat) (purpose : UInt8)
    (i : Nat) : ((Keys.Expander.ofPrefix pre).mid.update (suffix s j purpose i)).padding.size = 128 := by
  rw [← L_length, L_padding_suffix pre hpre]
  simp [suffixHead_length, padTail_length]

def setCounter (B : ByteArray) (i : Nat) : ByteArray :=
  (List.range' 0 4).foldl (fun blk b => blk.set! (17 + b) (i >>> (8 * (3 - b))).toUInt8) B

theorem L_set! (b : ByteArray) (k : Nat) (v : UInt8) : L (b.set! k v) = (L b).set k v := by
  obtain ⟨bs⟩ := b
  show (bs.set! k v).toList = bs.toList.set k v
  simp

theorem L_setCounter (P T : List UInt8) (hP : P.length = 17) (a b c d : UInt8) (B : ByteArray)
    (hB : L B = P ++ a :: b :: c :: d :: T) (i : Nat) :
    L (setCounter B i) = P ++ (i >>> 24).toUInt8 :: (i >>> 16).toUInt8 :: (i >>> 8).toUInt8 :: i.toUInt8 :: T := by
  show L ((((B.set! (17 + 0) (i >>> (8 * (3 - 0))).toUInt8).set! (17 + 1) (i >>> (8 * (3 - 1))).toUInt8).set!
    (17 + 2) (i >>> (8 * (3 - 2))).toUInt8).set! (17 + 3) (i >>> (8 * (3 - 3))).toUInt8) = _
  rw [L_set!, L_set!, L_set!, L_set!, hB]
  simp [hP]

theorem L_inj {a b : ByteArray} (h : L a = L b) : a = b := ByteArray.ext (Array.toList_inj.1 h)

theorem chain_zero_blocks (h : Array UInt64) (d : ByteArray) (off : Nat) : chain h d off 0 = h := by
  simp only [chain, List.range'_zero, List.foldl_nil]

theorem compress_setCounter (pre : ByteArray) (hpre : pre.size % 128 = 0) (s j : Nat) (purpose : UInt8)
    (a b c d : UInt8) (B : ByteArray) (hB : L B = suffixHead s j purpose ++ a :: b :: c :: d :: padTail pre.size)
    (i : Nat) :
    digest (Flock.Sha512.compress ((Keys.Expander.ofPrefix pre).mid.update (suffix s j purpose 0)).h
        (setCounter B i) 0) = (Keys.Expander.ofPrefix pre).chunk s j purpose i ∧
      L (setCounter B i) = suffixHead s j purpose ++ (i >>> 24).toUInt8 :: (i >>> 16).toUInt8 :: (i >>> 8).toUInt8 ::
        i.toUInt8 :: padTail pre.size := by
  have hL := L_setCounter _ _ (suffixHead_length s j purpose) a b c d B hB i
  refine ⟨?_, hL⟩
  have heq : setCounter B i = ((Keys.Expander.ofPrefix pre).mid.update (suffix s j purpose i)).padding :=
    L_inj (hL.trans (L_padding_suffix pre hpre s j purpose i).symm)
  show _ = State.finalize ((Keys.Expander.ofPrefix pre).mid.update (suffix s j purpose i))
  rw [finalize_eq, padding_suffix_size pre hpre, heq, h_suffix pre hpre s j purpose 0,
    h_suffix pre hpre s j purpose i]
  show _ = digest (chain _ _ 0 (0 + 1))
  rw [chain_succ, chain_zero_blocks, Nat.mul_zero, Nat.add_zero]

theorem chunk_eq_hash (pre : ByteArray) (s j : Nat) (purpose : UInt8) (i : Nat) :
    (Keys.Expander.ofPrefix pre).chunk s j purpose i = Flock.Sha512.hash (pre ++ suffix s j purpose i) :=
  update_update pre _

def chunks (e : Keys.Expander) (s j : Nat) (purpose : UInt8) (first n : Nat) : ByteArray :=
  (List.range' first n).foldl (fun out i => out ++ e.chunk s j purpose i) .empty

theorem beVal_one (b : ByteArray) (k : Nat) : beVal b k 1 = (b.get! k).toNat := by
  show 0 * 256 + (b.get! k).toNat = _
  rw [Nat.zero_mul, Nat.zero_add]

theorem ff_of_ge {x : ByteArray} {k c : Nat} (hk : k + 2 < x.size) (hc : c ≤ 65535)
    (h : 256 ^ x.size - c ≤ natOfBE x) : x.get! k = 255 := by
  obtain ⟨r, hr⟩ : ∃ r, x.size = k + (1 + r) := ⟨x.size - 1 - k, by omega⟩
  have hr2 : 2 ≤ r := by omega
  have hv : natOfBE x = beVal x 0 k * (256 * 256 ^ r) + ((x.get! k).toNat * 256 ^ r + beVal x (k + 1) r) := by
    rw [natOfBE_eq, hr, beVal_add, Nat.zero_add, beVal_add, beVal_one, Nat.pow_add, Nat.pow_one]
  have hT : 256 ^ x.size = 256 ^ k * (256 * 256 ^ r) := by rw [hr, Nat.pow_add, Nat.pow_add, Nat.pow_one]
  have hA := beVal_lt x 0 k
  have hC := beVal_lt x (k + 1) r
  have hQ : 65536 ≤ 256 ^ r :=
    calc 65536 = 256 ^ 2 := by decide
      _ ≤ 256 ^ r := Nat.pow_le_pow_right (by decide) hr2
  have hy := byte_lt (x.get! k)
  have hbyte : (x.get! k).toNat = 255 := by
    apply Classical.byContradiction
    intro hne
    have h1 : (beVal x 0 k + 1) * (256 * 256 ^ r) ≤ 256 ^ k * (256 * 256 ^ r) := Nat.mul_le_mul_right _ hA
    have h2 : ((x.get! k).toNat + 1) * 256 ^ r ≤ 255 * 256 ^ r := Nat.mul_le_mul_right _ (by omega)
    rw [Nat.add_mul, Nat.one_mul] at h1 h2
    rw [hv, hT] at h
    generalize beVal x 0 k * (256 * 256 ^ r) = M at h h1
    generalize 256 ^ k * (256 * 256 ^ r) = T at h h1
    generalize (x.get! k).toNat * 256 ^ r = N at h h2
    generalize 256 ^ r = Q at h h1 h2 hQ hC
    omega
  exact UInt8.toNat_inj.1 (hbyte.trans rfl)

theorem fold_range (pre : ByteArray) (hpre : pre.size % 128 = 0) (s j : Nat) (purpose : UInt8) :
    ∀ (l : List Nat) (out B : ByteArray) (a b c d : UInt8),
      L B = suffixHead s j purpose ++ a :: b :: c :: d :: padTail pre.size →
      (l.foldl (fun r i => (r.1 ++ digest (Flock.Sha512.compress
          ((Keys.Expander.ofPrefix pre).mid.update (suffix s j purpose 0)).h (setCounter r.2 i) 0),
          setCounter r.2 i)) (out, B)).1 =
        l.foldl (fun o i => o ++ (Keys.Expander.ofPrefix pre).chunk s j purpose i) out
  | [], _, _, _, _, _, _, _ => rfl
  | i :: l, out, B, a, b, c, d, hB => by
    obtain ⟨h1, h2⟩ := compress_setCounter pre hpre s j purpose a b c d B hB i
    rw [List.foldl_cons, List.foldl_cons]
    dsimp only
    rw [h1]
    exact fold_range pre hpre s j purpose l _ _ _ _ _ _ h2

theorem range_eq (pre : ByteArray) (hpre : pre.size % 128 = 0) (s j : Nat) (purpose : UInt8) (lo hi : Nat) :
    (Keys.Expander.ofPrefix pre).range s j purpose lo hi =
      (chunks (Keys.Expander.ofPrefix pre) s j purpose (lo / 64) ((hi + 63) / 64 - lo / 64)).extract
        (lo - 64 * (lo / 64)) (hi - 64 * (lo / 64)) := by
  have hsz : ((Keys.Expander.ofPrefix pre).mid.update (Flock.be64 s.toUInt64 ++ Flock.be64 j.toUInt64 ++
      ByteArray.mk #[purpose] ++ Flock.be32 0)).padding.size = 128 := padding_suffix_size pre hpre s j purpose 0
  have hbuf : ((Keys.Expander.ofPrefix pre).mid.update (Flock.be64 s.toUInt64 ++ Flock.be64 j.toUInt64 ++
      ByteArray.mk #[purpose] ++ Flock.be32 0)).buf.size = 21 := by
    rw [← L_length]
    exact (congrArg List.length (buf_suffix pre hpre s j purpose 0)).trans
      ((L_length _).trans (suffix_size s j purpose 0))
  unfold Keys.Expander.range
  simp only [Id.run, Std.Legacy.Range.forIn_eq_forIn_range', List.forIn_pure_yield_eq_foldl,
    bind_pure_comp, map_pure, Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one, hsz, hbuf,
    bne_self_eq_false, Bool.false_eq_true, ↓reduceIte]
  exact congrArg (fun o => ByteArray.extract o (lo - 64 * (lo / 64)) (hi - 64 * (lo / 64)))
    (fold_range pre hpre s j purpose _ _ _ _ _ _ _ (L_padding_suffix pre hpre s j purpose 0))

theorem digest_size (h : Array UInt64) : (digest h).size = 64 := Array.size_ofFn

theorem chunk_size (e : Keys.Expander) (s j : Nat) (purpose : UInt8) (i : Nat) : (e.chunk s j purpose i).size = 64 := by
  show (State.finalize _).size = 64
  rw [finalize_eq]
  exact digest_size _

theorem L_foldl_append (c : Nat → ByteArray) : ∀ (l : List Nat) (init : ByteArray),
    L (l.foldl (fun out i => out ++ c i) init) = L init ++ (l.map fun i => L (c i)).flatten
  | [], init => by simp
  | i :: l, init => by
    rw [List.foldl_cons, L_foldl_append c l, L_append, List.map_cons, List.flatten_cons, List.append_assoc]

theorem chunks_size (e : Keys.Expander) (s j : Nat) (purpose : UInt8) (first : Nat) :
    ∀ n, (chunks e s j purpose first n).size = 64 * n
  | 0 => rfl
  | n + 1 => by
    have ih := chunks_size e s j purpose first n
    unfold chunks at ih ⊢
    rw [List.range'_concat, List.foldl_append, List.foldl_cons, List.foldl_nil, ByteArray.size_append, chunk_size,
      ih]
    omega

theorem L_chunks_two (e : Keys.Expander) (s j : Nat) (purpose : UInt8) (n : Nat) :
    L (chunks e s j purpose 0 (n + 2)) = L (e.chunk s j purpose 0) ++ L (e.chunk s j purpose 1) ++
      ((List.range' 2 n).map fun i => L (e.chunk s j purpose i)).flatten := by
  unfold chunks
  rw [L_foldl_append, show n + 2 = (n + 1) + 1 by omega, List.range'_succ, List.range'_succ]
  simp [List.append_assoc]

theorem getElem?_of_get! {b : ByteArray} {k : Nat} (hk : k < b.size) {v : UInt8} (h : b.get! k = v) :
    (L b)[k]? = some v := by
  have hk' : k < (L b).length := by rw [L_length]; exact hk
  rw [Properties.Identity.get!_eq, List.getElem?_eq_getElem hk'] at h
  rw [List.getElem?_eq_getElem hk', ← h]
  all_goals rfl

set_option exponentiation.threshold 20000 in
theorem pow_256_2056 : (256 : Nat) ^ 2056 = 2 ^ 16448 := by
  rw [show (256 : Nat) = 2 ^ 8 from rfl, ← Nat.pow_mul]

theorem chunk01_of_ge (pre : ByteArray) (hpre : pre.size % 128 = 0) (s j : Nat) (purpose : UInt8)
    (h : 2 ^ 16448 - 21065 ≤ natOfBE ((Keys.Expander.ofPrefix pre).range s j purpose 0 2056)) :
    (Keys.Expander.ofPrefix pre).chunk s j purpose 0 = (Keys.Expander.ofPrefix pre).chunk s j purpose 1 := by
  rw [range_eq pre hpre, show (2056 + 63) / 64 - 0 / 64 = 31 + 2 from rfl, show 0 / 64 = 0 from rfl,
    Nat.mul_zero, Nat.sub_zero, Nat.sub_zero] at h
  generalize hx : (chunks (Keys.Expander.ofPrefix pre) s j purpose 0 (31 + 2)).extract 0 2056 = x at h
  have hLx : L x = (L ((Keys.Expander.ofPrefix pre).chunk s j purpose 0) ++
      L ((Keys.Expander.ofPrefix pre).chunk s j purpose 1) ++
      ((List.range' 2 31).map fun i => L ((Keys.Expander.ofPrefix pre).chunk s j purpose i)).flatten).take 2056 := by
    rw [← hx, Properties.Identity.L_extract_zero, L_chunks_two]
  have h0 := L_length ((Keys.Expander.ofPrefix pre).chunk s j purpose 0)
  have h1 := L_length ((Keys.Expander.ofPrefix pre).chunk s j purpose 1)
  rw [chunk_size] at h0 h1
  have hxs : x.size = 2056 := by
    rw [← hx, ByteArray.size_extract, chunks_size]
    all_goals decide
  rw [← pow_256_2056, ← hxs] at h
  apply L_inj
  apply List.ext_getElem?
  intro k
  by_cases hk : k < 64
  · have e0 : (L x)[k]? = (L ((Keys.Expander.ofPrefix pre).chunk s j purpose 0))[k]? := by
      rw [hLx, List.getElem?_take_of_lt (by omega), List.append_assoc, List.getElem?_append_left (by omega)]
    have e1 : (L x)[64 + k]? = (L ((Keys.Expander.ofPrefix pre).chunk s j purpose 1))[k]? := by
      rw [hLx, List.getElem?_take_of_lt (by omega), List.append_assoc, List.getElem?_append_right (by omega), h0,
        Nat.add_sub_cancel_left, List.getElem?_append_left (by omega)]
    rw [← e0, ← e1, getElem?_of_get! (by omega) (ff_of_ge (k := k) (by omega) (by decide) h),
      getElem?_of_get! (by omega) (ff_of_ge (k := 64 + k) (by omega) (by decide) h)]
  · rw [List.getElem?_eq_none (by omega), List.getElem?_eq_none (by omega)]

theorem suffix_ne (pre : ByteArray) (s j : Nat) (purpose : UInt8) :
    pre ++ suffix s j purpose 0 ≠ pre ++ suffix s j purpose 1 := by
  intro heq
  have h1 := congrArg (fun b => (L b)[pre.size + 20]?) heq
  simp only [Properties.Identity.getElem?_L_append_right, L_suffix] at h1
  rw [List.getElem?_append_right (by rw [suffixHead_length]; omega),
    List.getElem?_append_right (by rw [suffixHead_length]; omega), suffixHead_length] at h1
  exact absurd h1 (by decide)

def storeStrings (pre : ByteArray) : List ByteArray :=
  (List.range 8).flatMap fun s => (List.range 65536).flatMap fun j =>
    [pre ++ suffix s j Keys.PURPOSE_T 0, pre ++ suffix s j Keys.PURPOSE_T 1]

theorem mem_storeStrings {pre : ByteArray} {s j : Nat} (hs : s < 8) (hj : j < 65536) :
    pre ++ suffix s j Keys.PURPOSE_T 0 ∈ storeStrings pre ∧ pre ++ suffix s j Keys.PURPOSE_T 1 ∈ storeStrings pre := by
  have h : ∀ x, x ∈ [pre ++ suffix s j Keys.PURPOSE_T 0, pre ++ suffix s j Keys.PURPOSE_T 1] → x ∈ storeStrings pre :=
    fun x hx => List.mem_flatMap.2 ⟨s, List.mem_range.2 hs, List.mem_flatMap.2 ⟨j, List.mem_range.2 hj, hx⟩⟩
  exact ⟨h _ (.head _), h _ (.tail _ (.head _))⟩

theorem collision_of_ge (pre : ByteArray) (hpre : pre.size % 128 = 0) {s j : Nat} (hs : s < 8) (hj : j < 65536)
    (h : 2 ^ 16448 - 21065 ≤ natOfBE ((Keys.Expander.ofPrefix pre).range s j Keys.PURPOSE_T 0 2056)) :
    ∃ x ∈ storeStrings pre, ∃ y ∈ storeStrings pre, x ≠ y ∧ Flock.Sha512.hash x = Flock.Sha512.hash y :=
  ⟨_, (mem_storeStrings hs hj).1, _, (mem_storeStrings hs hj).2, suffix_ne pre s j Keys.PURPOSE_T, by
    rw [← chunk_eq_hash, ← chunk_eq_hash]
    exact chunk01_of_ge pre hpre s j Keys.PURPOSE_T h⟩

theorem validate_deployed : P2.Params.deployed.validate = .ok P2.Params.deployed := by rfl

theorem deployed_blockBytes : P2.Params.deployed.blockBytes = 2056 := by rfl

theorem deployed_payloadBytes : P2.Params.deployed.payloadBytes = 2054 := by rfl

theorem deployed_p : P2.Params.deployed.p = 2 ^ 16448 - 21065 := by
  simp only [P2.Params.p, P2.Params.deployed, W_BITS]

theorem pad128_size (body : ByteArray) : (Keys.pad128 body).size % 128 = 0 := by
  have hz : ∀ n, (zeros n).size = n := fun _ => Array.size_replicate
  rw [Keys.pad128, ByteArray.size_append, hz]
  omega

theorem packagePrefix_ok (q : P2.Params) {salt : ByteArray} (h : salt.size = 24) :
    Keys.packagePrefix q salt = .ok (Keys.pad128 (Keys.KEYS_TAG ++ FlockSoundness.Randomness.frame (.str Keys.PROTOCOL_ID) ++
      FlockSoundness.Randomness.frame (.str q.primeId) ++ FlockSoundness.Randomness.frame (.bytes salt))) := by
  simp only [Keys.packagePrefix, h]
  rfl

theorem keys_ok_or_collision (pre : ByteArray) (hpre : pre.size % 128 = 0) {s j : Nat} (hs : s < 8)
    (hj : j < 65536) :
    (∃ t, t < P2.Params.deployed.p ∧ (Keys.Expander.ofPrefix pre).keys P2.Params.deployed s j =
        .ok (natOfBE ((Keys.Expander.ofPrefix pre).range s j Keys.PURPOSE_K 0 2056), t, 0)) ∨
      ∃ x ∈ storeStrings pre, ∃ y ∈ storeStrings pre, x ≠ y ∧ Flock.Sha512.hash x = Flock.Sha512.hash y := by
  by_cases h : natOfBE ((Keys.Expander.ofPrefix pre).range s j Keys.PURPOSE_T 0 2056) < P2.Params.deployed.p
  · refine Or.inl ⟨_, h, ?_⟩
    have hfuel : Keys.STREAM_BYTES / 2056 = 133695479 + 1 := by decide
    unfold Keys.Expander.keys
    simp only [deployed_blockBytes]
    rw [hfuel, Keys.Expander.below.eq_2]
    simp only [Nat.zero_mul, Nat.zero_add, Nat.one_mul, h, ↓reduceIte, pure_bind]
    all_goals rfl
  · exact Or.inr (collision_of_ge pre hpre hs hj (by rw [← deployed_p]; exact Nat.le_of_not_lt h))

theorem idLabeling_val {p : Nat} [NeZero p] (x : ZMod p) : ((idLabeling p).1 x : Nat) = x.val := by
  cases p with
  | zero => exact absurd rfl (NeZero.ne 0)
  | succ n => rfl

theorem d_eq_sqE {p a : Nat} [NeZero p] (ha : a < p) :
    P2.d p a = (Pous.sqE (idLabeling p) (a : ZMod p)).val := by
  have hv : ((idLabeling p).1 (a : ZMod p) : Nat) = a := by
    rw [idLabeling_val, ZMod.val_natCast, Nat.mod_eq_of_lt ha]
  simp only [P2.d, P2.square, Pous.sqE]
  rw [hv, ← Nat.cast_mul]
  by_cases h : a % 2 = 1
  · have hb : (a % 2 == 1) = true := beq_iff_eq.mpr h
    simp only [h, beq_self_eq_true, ↓reduceIte]
    rw [ZMod.neg_val', ZMod.val_natCast]
  · have hb : ¬(a % 2 == 1) = true := fun e => h (beq_iff_eq.mp e)
    simp only [hb, h, Bool.false_eq_true, ↓reduceIte]
    rw [ZMod.val_natCast]

theorem d_surj {p y : Nat} (hp : p.Prime) (h4 : p % 4 = 3) (hy : y < p) : ∃ c, c < p ∧ P2.d p c = y := by
  have : NeZero p := ⟨hp.ne_zero⟩
  obtain ⟨x, hx⟩ := PousP2Ring.sqE_surjective hp h4 (idLabeling p) (y : ZMod p)
  refine ⟨x.val, ZMod.val_lt x, ?_⟩
  rw [d_eq_sqE (ZMod.val_lt x), ZMod.natCast_zmod_val, hx, ZMod.val_natCast, Nat.mod_eq_of_lt hy]

noncomputable def dInv (p y : Nat) : Nat :=
  if h : ∃ c, c < p ∧ P2.d p c = y then Classical.choose h else 0

theorem dInv_spec {p y : Nat} (hp : p.Prime) (h4 : p % 4 = 3) (hy : y < p) :
    dInv p y < p ∧ P2.d p (dInv p y) = y := by
  have h := d_surj hp h4 hy
  simp only [dInv, h, ↓reduceDIte]
  exact Classical.choose_spec h

theorem sigma_lt {p r : Nat} (k : Nat) (hr : r < p) : P2.sigma p r k < p := by
  simp only [P2.sigma]
  split <;> assumption

theorem sigma_sigma {p r : Nat} (k : Nat) (hr : r < p) : P2.sigma p (P2.sigma p r k) k = r := by
  by_cases h : r ^^^ k < p
  · have hx : r ^^^ k ^^^ k = r := by rw [Nat.xor_assoc, Nat.xor_self, Nat.xor_zero]
    simp only [P2.sigma, h, ↓reduceIte, hx, hr]
  · simp only [P2.sigma, h, ↓reduceIte]

noncomputable def honestC (p k t : Nat) : Nat := dInv p (P2.sigma p (dInv p t) k)

theorem honestC_spec {p : Nat} (hp : p.Prime) (h4 : p % 4 = 3) (k : Nat) {t : Nat} (ht : t < p) :
    honestC p k t < p ∧ P2.decodeValue p k t (honestC p k t) = 0 := by
  obtain ⟨h1, h2⟩ := dInv_spec hp h4 ht
  obtain ⟨h3, h5⟩ := dInv_spec hp h4 (sigma_lt k h1)
  refine ⟨h3, ?_⟩
  simp only [P2.decodeValue, honestC]
  rw [h5, sigma_sigma k h1, h2, Nat.add_sub_cancel_left, Nat.mod_self]

theorem honestC_deployed (hP : Pous.P2.Assumptions.PrimeP16448) (k : Nat) {t : Nat}
    (ht : t < P2.Params.deployed.p) :
    honestC P2.Params.deployed.p k t < P2.Params.deployed.p ∧
      P2.decodeValue P2.Params.deployed.p k t (honestC P2.Params.deployed.p k t) = 0 := by
  rw [deployed_p] at ht ⊢
  exact honestC_spec hP PousP2Ring.p16448_mod_four k ht

theorem forIn'_ok {σ ε : Type} (g : Nat → σ → σ) :
    ∀ (l : List Nat) (init : σ) (f : (s : Nat) → s ∈ l → σ → Except ε (ForInStep σ)),
      (∀ s (hs : s ∈ l) r, f s hs r = .ok (.yield (g s r))) →
        forIn' l init f = .ok (l.foldl (fun r s => g s r) init)
  | [], _, _, _ => rfl
  | a :: l, init, f, hf => by
    rw [List.forIn'_cons, hf a (by simp) init]
    simp only [bind, Except.bind, List.foldl_cons]
    exact forIn'_ok g l (g a init) _ (fun s hs r => hf s (List.mem_cons_of_mem a hs) r)

theorem L_foldl_push (c : Nat → UInt8) : ∀ (l : List Nat) (init : ByteArray),
    L (l.foldl (fun out i => out.push (c i)) init) = L init ++ l.map c
  | [], init => by simp
  | i :: l, init => by
    rw [List.foldl_cons, L_foldl_push c l, List.map_cons]
    simp [L, ByteArray.data_push]

theorem get!_append_left (a b : ByteArray) {i : Nat} (hi : i < a.size) : (a ++ b).get! i = a.get! i := by
  rw [Properties.Identity.get!_eq, Properties.Identity.get!_eq, getElem?_L_append_left a b hi]

theorem get!_append_right (a b : ByteArray) (k : Nat) : (a ++ b).get! (a.size + k) = b.get! k := by
  rw [Properties.Identity.get!_eq, Properties.Identity.get!_eq, Properties.Identity.getElem?_L_append_right]

theorem byte_uint64 (n e : Nat) (he : e < 8) :
    ((n.toUInt64 >>> (8 * e).toUInt64).toUInt8).toNat = n / 256 ^ e % 256 := by
  have h8 : (8 * e).toUInt64.toNat % 64 = 8 * e := by
    rw [Nat.toUInt64_eq, UInt64.toNat_ofNat', Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by omega)]
  have h64 : (2 : Nat) ^ 64 = 256 ^ e * 256 ^ (8 - e) := by
    rw [← Nat.pow_add, Nat.add_sub_cancel' (by omega : e ≤ 8)]
    rfl
  rw [UInt64.toNat_toUInt8, UInt64.toNat_shiftRight, h8, Nat.toUInt64_eq, UInt64.toNat_ofNat',
    Nat.shiftRight_eq_div_pow, Nat.pow_mul, show (2 : Nat) ^ 8 = 256 from rfl, h64, Nat.mod_mul_right_div_self,
    Nat.mod_mod_of_dvd _ (dvd_pow_self 256 (by omega : 8 - e ≠ 0))]

theorem beVal_digits (b : ByteArray) (lo : Nat) : ∀ (m n : Nat),
    (∀ i, i < m → (b.get! (lo + i)).toNat = n / 256 ^ (m - 1 - i) % 256) → beVal b lo m = n % 256 ^ m
  | 0, n, _ => by rw [Nat.pow_zero, Nat.mod_one]; rfl
  | m + 1, n, h => by
    have hm := beVal_digits b lo m (n / 256) (fun i hi => by
      have he : 256 * 256 ^ (m - 1 - i) = 256 ^ (m + 1 - 1 - i) := by
        rw [← Nat.pow_succ']
        congr 1
        omega
      rw [h i (by omega), Nat.div_div_eq_div_mul, he])
    have h1 := Nat.div_add_mod (n % (256 * 256 ^ m)) 256
    rw [Nat.mod_mul_right_div_self, Nat.mod_mul_right_mod] at h1
    rw [beVal_add, Nat.pow_one, beVal_one, hm, h m (by omega), show m + 1 - 1 - m = 0 by omega, Nat.pow_zero,
      Nat.div_one, Nat.pow_succ', ← h1]
    generalize n / 256 % 256 ^ m = X
    omega

theorem beOfNat_get (w : Nat) :
    ∀ n i, i < w → ((Prim.beOfNat n w).get! i).toNat = n / 256 ^ (w - 1 - i) % 256 := by
  induction w using Nat.strong_induction_on with
  | _ w ih =>
  intro n i hi
  rw [Prim.beOfNat]
  split
  · rename_i hw
    simp only [Id.run, Std.Legacy.Range.forIn_eq_forIn_range', List.forIn_pure_yield_eq_foldl]
    change ((List.foldl _ (ByteArray.emptyWithCapacity w) (List.range' 0 [:w].size)).get! i).toNat = _
    have hsz : [:w].size = w := by simp [Std.Legacy.Range.size]
    rw [Properties.Identity.get!_eq, L_foldl_push, hsz, show L (ByteArray.emptyWithCapacity w) = [] from rfl,
      List.nil_append, List.getElem?_map, List.getElem?_eq_getElem (by rw [List.length_range']; exact hi),
      List.getElem_range', Option.map_some, Option.getD_some, Nat.zero_add, Nat.one_mul]
    exact byte_uint64 n (w - 1 - i) (by omega)
  · rename_i hw
    dsimp only
    by_cases hlo : i < w - w / 2
    · rw [get!_append_left _ _ (by rw [beOfNat_size]; exact hlo), ih (w - w / 2) (by omega) _ i hlo,
        Nat.shiftRight_eq_div_pow, Nat.pow_mul, show (2 : Nat) ^ 8 = 256 from rfl, Nat.div_div_eq_div_mul,
        ← Nat.pow_add, show w / 2 + (w - w / 2 - 1 - i) = w - 1 - i by omega]
    · obtain ⟨k, rfl⟩ : ∃ k, i = (w - w / 2) + k := ⟨i - (w - w / 2), by omega⟩
      have hr := get!_append_right (Prim.beOfNat (n >>> (8 * (w / 2))) (w - w / 2))
        (Prim.beOfNat (n &&& (2 ^ (8 * (w / 2)) - 1)) (w / 2)) k
      rw [beOfNat_size] at hr
      have hsplit : (256 : Nat) ^ (w / 2) = 256 ^ (w / 2 - 1 - k) * 256 ^ (k + 1) := by
        rw [← Nat.pow_add]
        congr 1
        omega
      rw [hr, ih (w / 2) (by omega) _ k (by omega), Nat.and_two_pow_sub_one_eq_mod, Nat.pow_mul,
        show (2 : Nat) ^ 8 = 256 from rfl, show w - 1 - (w - w / 2 + k) = w / 2 - 1 - k by omega, hsplit,
        Nat.mod_mul_right_div_self, Nat.mod_mod_of_dvd _ (dvd_pow_self 256 (by omega : k + 1 ≠ 0))]

theorem natOfBE_beOfNat {n w : Nat} (h : n < 256 ^ w) : natOfBE (Prim.beOfNat n w) = n := by
  rw [natOfBE_eq, beOfNat_size, beVal_digits _ 0 w n (fun i hi => by rw [Nat.zero_add]; exact beOfNat_get w n i hi),
    Nat.mod_eq_of_lt h]

theorem honestC_lt (hP : Pous.P2.Assumptions.PrimeP16448) (k : Nat) {t : Nat} (ht : t < P2.Params.deployed.p) :
    honestC P2.Params.deployed.p k t < 256 ^ 2056 := by
  have hp : P2.Params.deployed.p < 256 ^ 2056 := by
    rw [deployed_p, pow_256_2056]
    have h0 : 0 < (2 : Nat) ^ 16448 := Nat.two_pow_pos _
    generalize (2 : Nat) ^ 16448 = P at h0 ⊢
    omega
  exact Nat.lt_trans (honestC_deployed hP k ht).1 hp

theorem decodeBlock_honest (hP : Pous.P2.Assumptions.PrimeP16448) (j k : Nat) {t : Nat}
    (ht : t < P2.Params.deployed.p) :
    P2.decodeBlock P2.Params.deployed j k t (Prim.beOfNat (honestC P2.Params.deployed.p k t) 2056) =
      .ok (Prim.beOfNat 0 2054) := by
  obtain ⟨hc, hd⟩ := honestC_deployed hP k ht
  have hge : ¬(P2.Params.deployed.p ≤ honestC P2.Params.deployed.p k t) := Nat.not_le.mpr hc
  unfold P2.decodeBlock
  simp only [beOfNat_size, deployed_blockBytes, deployed_payloadBytes, natOfBE_beOfNat (honestC_lt hP k ht),
    ge_iff_le, hge, hd, Nat.zero_shiftRight, bne_self_eq_false, Bool.false_eq_true, ↓reduceIte]
  all_goals rfl

def keyPrefix (salt : ByteArray) : ByteArray :=
  Keys.pad128 (Keys.KEYS_TAG ++ FlockSoundness.Randomness.frame (.str Keys.PROTOCOL_ID) ++
    FlockSoundness.Randomness.frame (.str P2.Params.deployed.primeId) ++ FlockSoundness.Randomness.frame (.bytes salt))

theorem packagePrefix_deployed {salt : ByteArray} (h : salt.size = 24) :
    Keys.packagePrefix P2.Params.deployed salt = .ok (keyPrefix salt) :=
  packagePrefix_ok _ h

def KeysOk (pre : ByteArray) : Prop :=
  ∀ s j, s < 8 → j < 65536 →
    ∃ k t, t < P2.Params.deployed.p ∧ (Keys.Expander.ofPrefix pre).keys P2.Params.deployed s j = .ok (k, t, 0)

theorem keysOk_or_collision (pre : ByteArray) (hpre : pre.size % 128 = 0) :
    KeysOk pre ∨
      ∃ x ∈ storeStrings pre, ∃ y ∈ storeStrings pre, x ≠ y ∧ Flock.Sha512.hash x = Flock.Sha512.hash y := by
  by_cases h : ∃ x ∈ storeStrings pre, ∃ y ∈ storeStrings pre, x ≠ y ∧ Flock.Sha512.hash x = Flock.Sha512.hash y
  · exact Or.inr h
  · refine Or.inl fun s j hs hj => ?_
    obtain ⟨t, ht, hk⟩ := (keys_ok_or_collision pre hpre hs hj).resolve_right h
    exact ⟨_, t, ht, hk⟩

noncomputable def honestBlock (pre : ByteArray) (s j : Nat) : ByteArray :=
  match (Keys.Expander.ofPrefix pre).keys P2.Params.deployed s j with
  | .ok (k, t, _) => Prim.beOfNat (honestC P2.Params.deployed.p k t) 2056
  | .error _ => .empty

noncomputable def honestSegment (pre : ByteArray) (s : Nat) : Array ByteArray :=
  Array.ofFn (n := 65536) fun j => honestBlock pre s j.1

noncomputable def honestEncoding (pre : ByteArray) : Array (Array ByteArray) :=
  Array.ofFn (n := 8) fun s => honestSegment pre s.1

def segPayload : ByteArray :=
  (List.range' 0 65536).foldl (fun out _ => out ++ Prim.beOfNat 0 2054)
    (ByteArray.emptyWithCapacity P2.Params.deployed.segmentBytes)

theorem honestBlock_eq {pre : ByteArray} {s j k t : Nat}
    (h : (Keys.Expander.ofPrefix pre).keys P2.Params.deployed s j = .ok (k, t, 0)) :
    honestBlock pre s j = Prim.beOfNat (honestC P2.Params.deployed.p k t) 2056 := by
  unfold honestBlock
  rw [h]

theorem honestSegment_size (pre : ByteArray) (s : Nat) : (honestSegment pre s).size = 65536 := Array.size_ofFn

theorem honestSegment_get (pre : ByteArray) (s j : Nat) (h : j < (honestSegment pre s).size) :
    (honestSegment pre s)[j] = honestBlock pre s j := by
  simp only [honestSegment, Array.getElem_ofFn]

theorem decodeSegment_honest (hP : Pous.P2.Assumptions.PrimeP16448) {salt : ByteArray} (h24 : salt.size = 24)
    (hk : KeysOk (keyPrefix salt)) {s : Nat} (hs : s < 8) {blocks : Array ByteArray} (hsz : blocks.size = 65536)
    (hblk : ∀ j (h : j < blocks.size), blocks[j] = honestBlock (keyPrefix salt) s j) :
    Setup.decodeSegment P2.Params.deployed salt s blocks = .ok segPayload := by
  have hseg : ¬(P2.Params.deployed.segments ≤ s) := by show ¬8 ≤ s; omega
  have hne : (blocks.size != P2.Params.deployed.blocksPerSegment) = false := by rw [hsz]; rfl
  unfold Setup.decodeSegment
  simp only [ge_iff_le, hseg, ↓reduceIte, hne, Bool.false_eq_true, packagePrefix_deployed h24, Except.mapError, bind,
    Except.bind, pure, Except.pure]
  rw [Std.Legacy.Range.forIn'_eq_forIn'_range', forIn'_ok (fun _ out => out ++ Prim.beOfNat 0 2054)]
  · simp only [hsz, Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one, segPayload]
  · intro j hj out
    have hj' : j < 65536 := by
      have h := hj
      simp only [List.mem_range'] at h
      obtain ⟨i, hi, he⟩ := h
      simp only [Std.Legacy.Range.size, hsz] at hi
      omega
    obtain ⟨k, t, ht, hkeq⟩ := hk s j hs hj'
    have hc := (honestC_deployed hP k ht).1
    have hge : ¬(P2.Params.deployed.p ≤ natOfBE (Prim.beOfNat (honestC P2.Params.deployed.p k t) 2056)) := by
      rw [natOfBE_beOfNat (honestC_lt hP k ht)]
      exact Nat.not_le.mpr hc
    simp only [hblk, honestBlock_eq hkeq, beOfNat_size, deployed_blockBytes, bne_self_eq_false,
      Bool.false_eq_true, ↓reduceIte, hge, hkeq, decodeBlock_honest hP j k ht]

theorem manifest_deployed : ∃ m, Commit.manifest P2.Params.deployed = .ok m ∧ m.size < 1000 := by
  have h : (match Verity.CanonUtf8.canonFlat (Commit.manifestJson P2.Params.deployed) with
      | .ok s => decide (s.toUTF8.size < 1000) | .error _ => false) = true := by decide +kernel
  unfold Commit.manifest
  rw [Verity.CanonUtf8.canonUtf8_flat _ (by decide +kernel)]
  revert h
  cases Verity.CanonUtf8.canonFlat (Commit.manifestJson P2.Params.deployed) with
  | error e => intro h; cases h
  | ok s => intro h; exact ⟨s.toUTF8, rfl, of_decide_eq_true h⟩

local instance instLawfulBEqByteArray_accepts : LawfulBEq ByteArray where
  eq_of_beq h := ByteArray.ext (beq_iff_eq.1 h)
  rfl {a} := show (a.data == a.data) = true from beq_iff_eq.2 rfl

theorem commitTag_size : 0 < Commit.COMMIT_TAG.toUTF8.size ∧ Commit.COMMIT_TAG.toUTF8.size < 2 ^ 32 := by
  decide +kernel

theorem wBytes_deployed : P2.Params.deployed.wBytes = 8 * (65536 * 2054) := rfl

theorem header_deployed {m : ByteArray} (hsz : m.size < 1000) :
    Identity.taggedHeader Commit.COMMIT_TAG (8 + m.size + P2.Params.deployed.wBytes) =
      .ok (Identity.TAGGED_SHA512 ++ Flock.be32 Commit.COMMIT_TAG.toUTF8.size.toUInt32 ++ Commit.COMMIT_TAG.toUTF8 ++
        Flock.be64 (8 + m.size + P2.Params.deployed.wBytes).toUInt64) :=
  (Properties.Identity.taggedHeader_ok).2
    ⟨⟨commitTag_size.1, commitTag_size.2, by rw [wBytes_deployed]; omega⟩, rfl⟩

theorem commitState_ok {q : P2.Params} {m hd : ByteArray} (hm : Commit.manifest q = .ok m)
    (hh : Identity.taggedHeader Commit.COMMIT_TAG (8 + m.size + q.wBytes) = .ok hd) :
    Commit.commitState q = .ok ((Identity.Sha512.init.update hd).update (Flock.be64 m.size.toUInt64 ++ m)) := by
  unfold Commit.commitState Identity.taggedState
  simp only [hm, hh, bind, Except.bind, pure, Except.pure]

theorem commitState_deployed : ∃ st0, Commit.commitState P2.Params.deployed = .ok st0 := by
  obtain ⟨m, hm, hsz⟩ := manifest_deployed
  exact ⟨_, commitState_ok hm (header_deployed hsz)⟩

def vkRun (salt : ByteArray) (st0 : Identity.Sha512.State) (seg : Nat → Array ByteArray) (pay : Nat → ByteArray)
    (n : Nat) : Identity.Sha512.State × Array ByteArray :=
  (List.range' 0 n).foldl (fun r s => (r.1.update (pay s), r.2 ++ Setup.segmentLeaves salt s (seg s))) (st0, #[])

theorem vk_generic {q : P2.Params} {commitment salt : ByteArray} {C : Array (Array ByteArray)}
    {st0 : Identity.Sha512.State} {seg : Nat → Array ByteArray} {pay : Nat → ByteArray}
    (hst : Commit.commitState q = .ok st0) (hC : C.size = q.segments)
    (hCs : ∀ s (h : s < C.size), C[s] = seg s)
    (hsz : ∀ s, s < q.segments → (seg s).size = q.blocksPerSegment)
    (hany : ∀ s, s < q.segments →
      (seg s).any (fun b => b.size != q.blockBytes) 0 q.blocksPerSegment = false)
    (hdec : ∀ s, s < q.segments → Setup.decodeSegment q salt s (seg s) = .ok (pay s))
    (hcom : (vkRun salt st0 seg pay q.segments).1.finalize = commitment) :
    Setup.verifierKeyFromEncoding q commitment salt C =
      .ok ⟨(Merkle.root (vkRun salt st0 seg pay q.segments).2).getD .empty, (vkRun salt st0 seg pay q.segments).2⟩ := by
  unfold Setup.verifierKeyFromEncoding
  simp only [hst, Except.mapError, bind, Except.bind, pure, Except.pure]
  rw [Std.Legacy.Range.forIn'_eq_forIn'_range', forIn'_ok (fun s r =>
    (r.1.update (pay s), r.2 ++ Setup.segmentLeaves salt s (seg s)))]
  · unfold vkRun at hcom ⊢
    simp only [hC, Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one, bne_self_eq_false,
      Bool.false_eq_true, ↓reduceIte, hcom]
  · rintro s hs ⟨h0, l0⟩
    have hs' : s < q.segments := by
      simp only [List.mem_range'] at hs
      obtain ⟨i, hi, rfl⟩ := hs
      simp only [Std.Legacy.Range.size, hC] at hi
      omega
    have hseg : ¬(q.segments ≤ s) := by omega
    simp only [hCs, ge_iff_le, hseg, ↓reduceIte, hsz s hs', bne_self_eq_false, Bool.false_or, hany s hs',
      Bool.false_eq_true, hdec s hs']

theorem any_honest {pre : ByteArray} (hk : KeysOk pre) {s : Nat} (hs : s < 8) :
    (honestSegment pre s).any (fun b => b.size != P2.Params.deployed.blockBytes) 0
      P2.Params.deployed.blocksPerSegment = false := by
  rw [show P2.Params.deployed.blocksPerSegment = (honestSegment pre s).size from (honestSegment_size pre s).symm,
    Bool.eq_false_iff]
  intro h
  obtain ⟨i, hi, hp⟩ := Array.any_eq_true.1 h
  obtain ⟨k, t, _, hkeq⟩ := hk s i hs (by have := honestSegment_size pre s; omega)
  simp only [honestSegment_get, honestBlock_eq hkeq, beOfNat_size, deployed_blockBytes, bne_self_eq_false,
    Bool.false_eq_true] at hp

noncomputable def honestW (salt : ByteArray) (st0 : Identity.Sha512.State) : ByteArray :=
  (vkRun salt st0 (honestSegment (keyPrefix salt)) (fun _ => segPayload) P2.Params.deployed.segments).1.finalize

theorem verifierKey_honest (hP : Pous.P2.Assumptions.PrimeP16448) {salt : ByteArray} (h24 : salt.size = 24)
    (hk : KeysOk (keyPrefix salt)) {st0 : Identity.Sha512.State}
    (hst : Commit.commitState P2.Params.deployed = .ok st0) {C : Array (Array ByteArray)}
    (hC : C.size = P2.Params.deployed.segments) (hCs : ∀ s (h : s < C.size), C[s] = honestSegment (keyPrefix salt) s) :
    Setup.verifierKeyFromEncoding P2.Params.deployed (honestW salt st0) salt C =
      .ok ⟨(Merkle.root (vkRun salt st0 (honestSegment (keyPrefix salt)) (fun _ => segPayload)
          P2.Params.deployed.segments).2).getD .empty,
        (vkRun salt st0 (honestSegment (keyPrefix salt)) (fun _ => segPayload) P2.Params.deployed.segments).2⟩ :=
  vk_generic hst hC hCs (fun s _ => honestSegment_size _ s) (fun _ hs => any_honest hk hs)
    (fun _ hs => decodeSegment_honest hP h24 hk hs (honestSegment_size _ _) (honestSegment_get _ _)) rfl

theorem honestEncoding_size (pre : ByteArray) : (honestEncoding pre).size = 8 := Array.size_ofFn

theorem honestEncoding_get (pre : ByteArray) (s : Nat) (h : s < (honestEncoding pre).size) :
    (honestEncoding pre)[s] = honestSegment pre s := by
  simp only [honestEncoding, Array.getElem_ofFn]

theorem check_of_vk {q q' : P2.Params} {w salt root : ByteArray} {leaves : Array ByteArray}
    {C : Array (Array ByteArray)} (hq : q.validate = .ok q')
    (h : Setup.verifierKeyFromEncoding q w salt C = .ok ⟨root, leaves⟩) :
    SetupProtocol.check q ⟨w, salt⟩ C = .accept ⟨q.blocks, w, root, salt⟩ () := by
  unfold SetupProtocol.check
  rw [hq]
  simp only [h]

theorem check_honest (hP : Pous.P2.Assumptions.PrimeP16448) {salt : ByteArray} (h24 : salt.size = 24)
    (hk : KeysOk (keyPrefix salt)) {st0 : Identity.Sha512.State}
    (hst : Commit.commitState P2.Params.deployed = .ok st0) :
    SetupProtocol.check P2.Params.deployed ⟨honestW salt st0, salt⟩ (honestEncoding (keyPrefix salt)) =
      .accept ⟨P2.Params.deployed.blocks, honestW salt st0,
        (Merkle.root (vkRun salt st0 (honestSegment (keyPrefix salt)) (fun _ => segPayload)
          P2.Params.deployed.segments).2).getD .empty, salt⟩ () := by
  exact check_of_vk validate_deployed
    (verifierKey_honest hP h24 hk hst (honestEncoding_size _) (honestEncoding_get _))

theorem setup_accepts_or_collision (hP : Pous.P2.Assumptions.PrimeP16448) {salt : ByteArray} (h24 : salt.size = 24) :
    (∃ (c : SetupProtocol.Commitment) (C : Array (Array ByteArray)) (pub : SetupProtocol.Public),
        c.salt = salt ∧ SetupProtocol.check P2.Params.deployed c C = .accept pub ()) ∨
      ∃ x ∈ storeStrings (keyPrefix salt), ∃ y ∈ storeStrings (keyPrefix salt),
        x ≠ y ∧ Flock.Sha512.hash x = Flock.Sha512.hash y := by
  obtain ⟨st0, hst⟩ := commitState_deployed
  rcases keysOk_or_collision (keyPrefix salt) (pad128_size _) with hk | hcol
  · exact Or.inl ⟨⟨honestW salt st0, salt⟩, honestEncoding (keyPrefix salt), _, rfl, check_honest hP h24 hk hst⟩
  · exact Or.inr hcol

def digestTable (f : Nat → ByteArray) : Nat → ByteArray
  | 0 => .empty
  | n + 1 => digestTable f n ++ f n

theorem rowDigest_size (b : ByteArray) : (Merkle.rowDigest b).size = Merkle.DIGEST :=
  Properties.CertifierDevice.Proofs.Lms.sha512_size _

theorem extract_zero_right (b : ByteArray) (i : Nat) : b.extract i 0 = .empty :=
  ByteArray.extract_eq_empty_iff.2 (by simp)

theorem extract_of_size_le {b : ByteArray} {i : Nat} (j : Nat) (h : b.size ≤ i) : b.extract i j = .empty :=
  ByteArray.extract_eq_empty_iff.2 (by omega)

theorem digestTable_size {f : Nat → ByteArray} (hf : ∀ i, (f i).size = Merkle.DIGEST) :
    ∀ n, (digestTable f n).size = n * Merkle.DIGEST
  | 0 => by simp [digestTable]
  | n + 1 => by
    show (digestTable f n ++ f n).size = _
    rw [ByteArray.size_append, digestTable_size hf n, hf, Nat.add_one_mul]

theorem digestTable_row {f : Nat → ByteArray} (hf : ∀ i, (f i).size = Merkle.DIGEST) :
    ∀ n i, i < n → PousCheck.Run.row (digestTable f n) i = f i
  | 0, _, h => absurd h (Nat.not_lt_zero _)
  | n + 1, i, h => by
    have hs := digestTable_size hf n
    unfold PousCheck.Run.row
    show (digestTable f n ++ f n).extract _ _ = f i
    rw [ByteArray.extract_append, hs]
    by_cases hi : i < n
    · rw [show Merkle.DIGEST * (i + 1) - n * Merkle.DIGEST = 0 by unfold Merkle.DIGEST; omega,
        extract_zero_right, ByteArray.append_empty]
      exact digestTable_row hf n i hi
    · have hin : i = n := by omega
      rw [hin, extract_of_size_le (b := digestTable f n) (i := Merkle.DIGEST * n) (Merkle.DIGEST * (n + 1))
          (by rw [hs]; unfold Merkle.DIGEST; omega), ByteArray.empty_append,
        show Merkle.DIGEST * n - n * Merkle.DIGEST = 0 by unfold Merkle.DIGEST; omega,
        show Merkle.DIGEST * (n + 1) - n * Merkle.DIGEST = (f n).size by rw [hf]; unfold Merkle.DIGEST; omega,
        ByteArray.extract_zero_size]

theorem rowLeaves_digestTable (q : P2.Params) (salt : ByteArray) (blk : Nat → ByteArray) :
    PousCheck.Run.rowLeaves q salt (digestTable (fun i => Merkle.rowDigest (blk i)) q.blocks) =
      (Array.range q.blocks).map fun i =>
        Merkle.packageLeaf salt (i / q.blocksPerSegment) (i % q.blocksPerSegment) (blk i) := by
  unfold PousCheck.Run.rowLeaves
  refine Array.ext (by simp) fun i h1 _ => ?_
  have hi : i < q.blocks := by simpa using h1
  simp only [Array.getElem_map, Array.getElem_range]
  rw [digestTable_row (f := fun i => Merkle.rowDigest (blk i)) (fun _ => rowDigest_size _) q.blocks i hi]
  rfl

theorem leaves_flat (salt : ByteArray) {m : Nat} {seg : Nat → Array ByteArray} {B : Nat → Nat → ByteArray}
    (hsz : ∀ s, (seg s).size = m) (hget : ∀ s v (h : v < (seg s).size), (seg s)[v] = B s v) :
    ∀ n, (List.range' 0 n).foldl (fun acc s => acc ++ Setup.segmentLeaves salt s (seg s)) #[] =
      (Array.range (n * m)).map fun i => Merkle.packageLeaf salt (i / m) (i % m) (B (i / m) (i % m))
  | 0 => by simp
  | n + 1 => by
    rw [List.range'_concat, List.foldl_append, leaves_flat salt hsz hget n]
    simp only [List.foldl_cons, List.foldl_nil, Nat.zero_add, Nat.one_mul]
    refine Array.ext (by simp [Setup.segmentLeaves, hsz, Nat.add_one_mul]) fun i _ h2 => ?_
    have hi : i < (n + 1) * m := by simpa using h2
    by_cases hl : i < n * m
    · simp [hl]
    · have hle : n * m ≤ i := Nat.le_of_not_lt hl
      have hdiv : i / m = n := Nat.div_eq_of_lt_le hle hi
      have hmod : i % m = i - n * m := by
        have h3 := Nat.div_add_mod i m
        rw [hdiv] at h3
        have h4 := Nat.mul_comm m n
        omega
      simp [Array.getElem_append, hl, Setup.segmentLeaves, hdiv, hmod, hget]

noncomputable def setupDigests (salt : ByteArray) : ByteArray :=
  digestTable (fun i => Merkle.rowDigest (honestBlock (keyPrefix salt) (i / 65536) (i % 65536)))
    P2.Params.deployed.blocks

theorem rowLeaves_setup (salt : ByteArray) :
    PousCheck.Run.rowLeaves P2.Params.deployed salt (setupDigests salt) =
      (List.range' 0 8).foldl (fun acc s => acc ++ Setup.segmentLeaves salt s (honestSegment (keyPrefix salt) s)) #[] := by
  have hbps : P2.Params.deployed.blocksPerSegment = 65536 := rfl
  have hbl : P2.Params.deployed.blocks = 8 * 65536 := rfl
  rw [leaves_flat salt (honestSegment_size (keyPrefix salt)) (honestSegment_get (keyPrefix salt)) 8, setupDigests,
    rowLeaves_digestTable, hbps, hbl]

theorem runCheck_accepts_setup {salt : ByteArray} (h24 : salt.size = 24) (pub : SetupProtocol.Public)
    (hb : pub.blocks = P2.Params.deployed.blocks) (hs : pub.salt = salt)
    (hr : Merkle.root ((List.range' 0 8).foldl
      (fun acc s => acc ++ Setup.segmentLeaves salt s (honestSegment (keyPrefix salt) s)) #[]) = some pub.root) :
    ∃ (trusted : List ByteArray) (r : PousCheck.Run.ChallengePhaseRecord) (sid : ByteArray),
      PousCheck.Run.check trusted pub (setupDigests salt) r sid = .ok () := by
  refine Pous.EndToEnd.SecurityProofs.Honest.runCheck_acceptsAt pub _ hb (by simp [hs, h24, PousCheck.Run.SALT])
    (by rw [setupDigests]; exact digestTable_size (fun _ => rowDigest_size _) _) ?_
  rw [hs, rowLeaves_setup, hr]
  exact beq_self_eq_true _

theorem vkRun_snd (salt : ByteArray) (st0 : Identity.Sha512.State) (seg : Nat → Array ByteArray)
    (pay : Nat → ByteArray) : ∀ n, (vkRun salt st0 seg pay n).2 =
      (List.range' 0 n).foldl (fun acc s => acc ++ Setup.segmentLeaves salt s (seg s)) #[]
  | 0 => rfl
  | n + 1 => by
    have ih := vkRun_snd salt st0 seg pay n
    unfold vkRun at ih ⊢
    simp only [List.range'_concat, List.foldl_append, List.foldl_cons, List.foldl_nil]
    rw [ih]

theorem honest_leaves (salt : ByteArray) (st0 : Identity.Sha512.State) :
    (vkRun salt st0 (honestSegment (keyPrefix salt)) (fun _ => segPayload) P2.Params.deployed.segments).2 =
      PousCheck.Run.rowLeaves P2.Params.deployed salt (setupDigests salt) := by
  have h8 : P2.Params.deployed.segments = 8 := rfl
  rw [vkRun_snd, h8, rowLeaves_setup]

theorem rowLeaves_size (q : P2.Params) (salt digests : ByteArray) :
    (PousCheck.Run.rowLeaves q salt digests).size = q.blocks := by
  simp [PousCheck.Run.rowLeaves]

theorem root_of_size {L : Array ByteArray} (h : L.size ≠ 0) :
    Merkle.root L = some ((Merkle.root L).getD .empty) := by
  have hne : L.isEmpty = false := by
    rw [Bool.eq_false_iff]
    intro he
    rw [Array.isEmpty_iff_size_eq_zero] at he
    exact h he
  unfold Merkle.root
  simp only [hne, Bool.false_eq_true, ↓reduceIte, Option.getD_some]

theorem root_setup (salt : ByteArray) :
    Merkle.root (PousCheck.Run.rowLeaves P2.Params.deployed salt (setupDigests salt)) =
      some ((Merkle.root (PousCheck.Run.rowLeaves P2.Params.deployed salt (setupDigests salt))).getD .empty) :=
  root_of_size (by rw [rowLeaves_size]; decide)

theorem runCheck_accepts_setup' {salt : ByteArray} (h24 : salt.size = 24) (w R : ByteArray)
    (hr : Merkle.root ((List.range' 0 8).foldl
      (fun acc s => acc ++ Setup.segmentLeaves salt s (honestSegment (keyPrefix salt) s)) #[]) = some R) :
    ∃ (key : ByteArray) (r : PousCheck.Run.ChallengePhaseRecord),
      PousCheck.Run.check [key] ⟨P2.Params.deployed.blocks, w, R, salt⟩ (setupDigests salt) r
        Pous.EndToEnd.SecurityProofs.Honest.honestSid = .ok () := by
  refine Pous.EndToEnd.SecurityProofs.Honest.runCheck_acceptsSid _ _ rfl (by simp [h24, PousCheck.Run.SALT])
    (by rw [setupDigests]; exact digestTable_size (fun _ => rowDigest_size _) _) ?_
  show (Merkle.root (PousCheck.Run.rowLeaves P2.Params.deployed salt (setupDigests salt)) == some R) = true
  rw [rowLeaves_setup, hr]
  exact beq_self_eq_true _

theorem setup_run_accept_sid_or_collision (hP : Pous.P2.Assumptions.PrimeP16448) {salt : ByteArray}
    (h24 : salt.size = 24) :
    (∃ (c : SetupProtocol.Commitment) (C : Array (Array ByteArray)) (pub : SetupProtocol.Public),
        c.salt = salt ∧ C.size = 8 ∧ SetupProtocol.check P2.Params.deployed c C = .accept pub () ∧
        ∃ (key : ByteArray) (r : PousCheck.Run.ChallengePhaseRecord),
          PousCheck.Run.check [key] pub (setupDigests salt) r Pous.EndToEnd.SecurityProofs.Honest.honestSid =
            .ok ()) ∨
      ∃ x ∈ storeStrings (keyPrefix salt), ∃ y ∈ storeStrings (keyPrefix salt),
        x ≠ y ∧ Flock.Sha512.hash x = Flock.Sha512.hash y := by
  obtain ⟨st0, hst⟩ := commitState_deployed
  rcases keysOk_or_collision (keyPrefix salt) (pad128_size _) with hk | hcol
  · have hr : Merkle.root ((List.range' 0 8).foldl
        (fun acc s => acc ++ Setup.segmentLeaves salt s (honestSegment (keyPrefix salt) s)) #[]) =
        some ((Merkle.root (vkRun salt st0 (honestSegment (keyPrefix salt)) (fun _ => segPayload)
          P2.Params.deployed.segments).2).getD .empty) := by
      rw [honest_leaves, ← rowLeaves_setup salt]
      exact root_setup salt
    exact Or.inl ⟨⟨honestW salt st0, salt⟩, honestEncoding (keyPrefix salt),
      ⟨P2.Params.deployed.blocks, honestW salt st0, (Merkle.root (vkRun salt st0 (honestSegment (keyPrefix salt))
        (fun _ => segPayload) P2.Params.deployed.segments).2).getD .empty, salt⟩, rfl,
      honestEncoding_size _, check_honest hP h24 hk hst, runCheck_accepts_setup' h24 _ _ hr⟩
  · exact Or.inr hcol

theorem setup_run_accept_or_collision (hP : Pous.P2.Assumptions.PrimeP16448) {salt : ByteArray}
    (h24 : salt.size = 24) :
    (∃ (c : SetupProtocol.Commitment) (C : Array (Array ByteArray)) (pub : SetupProtocol.Public),
        c.salt = salt ∧ SetupProtocol.check P2.Params.deployed c C = .accept pub () ∧
        ∃ (trusted : List ByteArray) (r : PousCheck.Run.ChallengePhaseRecord) (sid : ByteArray),
          PousCheck.Run.check trusted pub (setupDigests salt) r sid = .ok ()) ∨
      ∃ x ∈ storeStrings (keyPrefix salt), ∃ y ∈ storeStrings (keyPrefix salt),
        x ≠ y ∧ Flock.Sha512.hash x = Flock.Sha512.hash y := by
  rcases setup_run_accept_sid_or_collision hP h24 with ⟨c, C, pub, hc, -, hs, key, r, h⟩ | hcol
  · exact Or.inl ⟨c, C, pub, hc, hs, [key], r, _, h⟩
  · exact Or.inr hcol

end PousCheck.Properties.SecurityProofs.Accepts
