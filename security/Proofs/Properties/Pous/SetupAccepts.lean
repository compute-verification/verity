module

public import Properties.Pous
public import Proofs.Properties.Hash.Identity
public import Proofs.Flock.Bytes

/-!
# Setup's accepted play, without evaluating SHA-512: the bytes and the key stream

The pieces an honest `C*` needs before the setup check's loops, none of which evaluates a hash:

* a byte string's value: `natOfBERange`, which reads by halves, is the big-endian fold `beVal`
  (`natOfBERange_eq`, `natOfBE_eq`);
* the key stream: with the prefix whole 128-byte blocks, `Expander.range`'s compression branch, which sets the counter
  into one padded block, compresses exactly chunk `i`'s padding (`L_setCounter`, `compress_setCounter`), and a chunk is
  SHA-512 of the prefix and its 21 bytes (`chunk_eq_hash`, from `Properties.Identity.update_update`);
* a candidate at or above `p`: within `2^16` of `256^n`, every byte but the last two is `0xFF` (`ff_of_ge`);
* the deployed parameters: `validate` accepts them by reduction (`validate_deployed`) and `packagePrefix` accepts any
  24-byte salt (`packagePrefix_ok`), its prefix whole blocks (`pad128_size`).
-/

@[expose] public section

namespace PousCheck.Properties.SecurityProofs.Accepts

open Flock PousCheck PousCheck.Prim Identity.Sha512
open Properties.Identity (L L_length L_append Rep rep_init rep_update chain chain_succ chain_congr finalize_eq
  L_padding zeroPad lenBytes getElem?_L_append_left update_update)
open Properties.NetworkCertifier.Bytes (be64_size be32_size)

/-! ## A byte string's big-endian value -/

theorem byte_lt (y : UInt8) : y.toNat < 256 := y.toNat_lt_size

/-- One big-endian digit more: `acc · 256 + b[i]`. -/
def beStep (b : ByteArray) (acc i : Nat) : Nat := acc * 256 + (b.get! i).toNat

/-- `int.from_bytes(b[lo : lo + n], "big")`, one byte at a time. -/
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
    first | ac_rfl | grind

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
  all_goals rfl

/-- `natOfBERange`'s loop of at most eight bytes, in `UInt64`, is the fold while it fits. -/
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

/-- **`natOfBERange` is the big-endian value** of its bytes, both ways it reads them: one `UInt64` for eight bytes or
fewer, and the high half shifted over the low half above that. -/
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
    all_goals first | rfl | simp

theorem natOfBE_eq (b : ByteArray) : natOfBE b = beVal b 0 b.size := by
  have := natOfBERange_eq b b.size 0
  rwa [Nat.zero_add] at this

/-! ## A chunk of the key stream is one compression after the prefix -/

/-- A chunk's 21 bytes after the prefix's midstate: the segment, the block, the purpose and the counter. -/
def suffix (s j : Nat) (purpose : UInt8) (i : Nat) : ByteArray :=
  Flock.be64 s.toUInt64 ++ Flock.be64 j.toUInt64 ++ ByteArray.mk #[purpose] ++ Flock.be32 i.toUInt32

/-- The suffix's first 17 bytes, the same for every counter. -/
def suffixHead (s j : Nat) (purpose : UInt8) : List UInt8 :=
  L (Flock.be64 s.toUInt64) ++ L (Flock.be64 j.toUInt64) ++ [purpose]

theorem suffixHead_length (s j : Nat) (purpose : UInt8) : (suffixHead s j purpose).length = 17 := by
  simp [suffixHead, be64_size]

/-- SHA-512's padding after a 21-byte tail of a message of `n + 21` bytes, `n` a multiple of 128. -/
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

/-- `Expander.range`'s write of counter `i` into the padded block: its last four bytes before the padding. -/
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

/-- **`Expander.range`'s compression branch is chunk `i`**: on a prefix of whole blocks, setting counter `i` into any block
laid out as chunk 0's padding and compressing it once after chunk 0's chaining value gives chunk `i`, and the block it
leaves is laid out alike. -/
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

/-- A chunk is SHA-512 of the prefix and its 21 bytes. -/
theorem chunk_eq_hash (pre : ByteArray) (s j : Nat) (purpose : UInt8) (i : Nat) :
    (Keys.Expander.ofPrefix pre).chunk s j purpose i = Flock.Sha512.hash (pre ++ suffix s j purpose i) :=
  update_update pre _

/-- Chunks `first, first + 1, …` of a stream, `n` of them, in order. -/
def chunks (e : Keys.Expander) (s j : Nat) (purpose : UInt8) (first n : Nat) : ByteArray :=
  (List.range' first n).foldl (fun out i => out ++ e.chunk s j purpose i) .empty

/-! ## A key candidate at or above `p` -/

theorem beVal_one (b : ByteArray) (k : Nat) : beVal b k 1 = (b.get! k).toNat := by
  show 0 * 256 + (b.get! k).toNat = _
  rw [Nat.zero_mul, Nat.zero_add]

/-- **A value within `2^16` of `256^n` has every byte but its last two `0xFF`**: if byte `k < n − 2` were below `0xFF`,
the value would be at most `256^n − 256^(n−1−k) − 1`, and `256^(n−1−k) ≥ 2^16`. -/
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

/-! ## The deployed parameters -/

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

end PousCheck.Properties.SecurityProofs.Accepts
