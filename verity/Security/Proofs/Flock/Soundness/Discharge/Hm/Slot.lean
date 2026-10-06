/-
The hm96 slot from the empty state: in a satisfying assignment of its rows, when its `cv` inputs hold the chaining value
of a message before its last block, its `pad` inputs the last block and its salt inputs a 192-byte salt's bits, its
outputs are the bits of `SHA-512(m) ⊕ M·y` and of `SHA-512(salt_prefix ‖ y)`. The chaining value before each block is
`chainV` (`chainV_succ`: one compression per block, the form the sha512x3 slots compute).
-/
import Proofs.Flock.Soundness.Discharge.Hm.Salt
import Proofs.Flock.Soundness.Discharge.Hm.Top

namespace FlockSoundness.Discharge.Hm

open Flock.HmNets hiding and

open Flock

/-- Words from bits: 64-bit words whose concatenation has `ws'`'s bits are `ws'`, word for word. -/
theorem map_bv_of_bits (z : Var → Bool) {ws : List Word} (hw : Words64 ws) {ws' : List (BitVec 64)}
    (hl : ws.length = ws'.length) (h : ∀ q < 64 * ws.length, bit z ws.flatten q = wbit ws' q) :
    ws.map (bv z) = ws' := by
  apply List.ext_getElem
  · simp [hl]
  · intro i h1 h2
    have hi : i < ws.length := by simpa using h1
    rw [List.getElem_map]
    apply BitVec.eq_of_getLsbD_eq
    intro j hj
    have e := h (64 * i + j) (by omega)
    rw [bit_flatten z hw, wbit, show (64 * i + j) / 64 = i by omega, show (64 * i + j) % 64 = j by omega,
      List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hi, Option.getD_some, List.getD_eq_getElem?_getD,
      List.getElem?_eq_getElem h2, Option.getD_some] at e
    exact e

theorem bit_words_flatten (z : Var → Bool) {bits : List Lin} {n : ℕ} (hb : bits.length = 64 * n) {q : ℕ}
    (hq : q < 64 * n) : bit z (words bits n).flatten q = bit z bits q := by
  have hw : ∀ w ∈ words bits n, w.length = 64 := words64_words hb
  rw [show q = 64 * (q / 64) + q % 64 by omega, bit_flatten_of_len z hw (Nat.mod_lt _ (by norm_num))]
  have hk : q / 64 < n := by omega
  simp only [words, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range hk, Option.map_some,
    Option.getD_some, bit, List.getElem?_take, List.getElem?_drop, show q % 64 < 64 by omega, ↓reduceIte]

theorem bit_vars_eq_bitL (z : Var → Bool) {s : List UInt8} (hs : s.length = 192)
    (hy : ∀ q < 1536, z (1536 + q) = bitL s q) : bit z (vars 1536 1536) = bitL s := by
  funext q
  by_cases hq : q < 1536
  · rw [bit_vars z _ _ hq, hy q hq]
  · have h1 : bit z (vars 1536 1536) q = false := by
      simp [bit, vars, show ¬ q < 1536 from hq, ev_zero]
    have h2 : s.getD (q / 8) 0 = 0 := by
      rw [List.getD_eq_getElem?_getD, List.getElem?_eq_none (by omega), Option.getD_none]
    rw [h1, bitL, h2]
    simp

/-- **The hm96 slot computes `b ‖ c`'s bits.** In a satisfying assignment `z` of the hm96 circuit's rows, if its `cv`
inputs hold the chaining value of a message `m` before its last block (`lastCv m`), its `pad` inputs `m`'s last block
(`lastBlock m`) and its salt inputs the bits of a 192-byte salt `s` (bit `q % 8` of byte `q / 8`), then bit `t < 512` of
its output `b` is bit `t` of `SHA-512(m)` plus bit `t` of `M·s` (`maskV`), and bit `t` of `c` is bit `t` of
`SHA-512(salt_prefix ‖ s)`. -/
theorem hm96_slot (key : ByteArray) (z : Var → Bool) (h : stHolds z ((hm96Row key).run St.empty).2) (m : ByteArray)
    {s : List UInt8} (hs : s.length = 192) (hcv : ∀ q < 512, z q = wbit (lastCv m) q)
    (hpad : ∀ q < 1024, z (512 + q) = wbit (lastBlock m) q) (hy : ∀ q < 1536, z (1536 + q) = bitL s q) :
    ((hm96Row key).run St.empty).1.2.2.2.length = 1024 ∧ ∀ t < 512,
      bit z ((hm96Row key).run St.empty).1.2.2.2 t =
          (bitL (Sha512.hash m).data.toList t ^^ maskV key (bitL s) t) ∧
        bit z ((hm96Row key).run St.empty).1.2.2.2 (512 + t) = bitL (Sha512.hash (saltMsg s)).data.toList t := by
  obtain ⟨-, -, -, hlen, hb⟩ := hm96Row_spec key z h
  have l1 : (vars 0 512).length = 64 * 8 := length_vars 0 512
  have l2 : (vars 512 1024).length = 64 * 16 := length_vars 512 1024
  have e1 : (words (vars 0 512) 8).map (bv z) = lastCv m := by
    refine map_bv_of_bits z (words64_words l1) (by rw [length_words, lastCv, List.length_map, List.length_range])
      fun q hq => ?_
    rw [length_words] at hq
    rw [bit_words_flatten z l1 hq, bit_vars z 0 512 (show q < 512 by omega), Nat.zero_add, hcv q (by omega)]
  have e2 : (words (vars 512 1024) 16).map (bv z) = lastBlock m := by
    refine map_bv_of_bits z (words64_words l2) (by rw [length_words, lastBlock, List.length_map, List.length_range])
      fun q hq => ?_
    rw [length_words] at hq
    rw [bit_words_flatten z l2 hq, bit_vars z 512 1024 (show q < 1024 by omega), hpad q (by omega)]
  have ey := bit_vars_eq_bitL z hs hy
  have es : (saltWords (vars 1536 1536)).map (bv z) = saltWordsV s :=
    map_bv_saltWords z _ s fun q _ => by rw [ey]
  refine ⟨hlen, fun t ht => ⟨?_, ?_⟩⟩
  · rw [(hb t ht).1, e1, e2, ey, bitL_hash m ht]
  · rw [(hb t ht).2, es, bitL_saltHash hs ht]

/-- The chaining value after the first `k` blocks of `d`, as words. -/
def chainV (d : ByteArray) (k : ℕ) : List (BitVec 64) := (List.range 8).map fun i => (chain d k)[i]!.toBitVec

/-- The `k`-th block of `d`, as its sixteen big-endian words. -/
def blockV (d : ByteArray) (k : ℕ) : List (BitVec 64) := (List.range 16).map fun i => (wordBE d (128 * k) i).toBitVec

theorem chainV_zero (d : ByteArray) : chainV d 0 = (List.range 8).map fun i => Sha512.iv[i]!.toBitVec := rfl

/-- **One compression per block**: the chaining value after `k + 1` blocks is `compressV` of the one after `k` and
block `k`. -/
theorem chainV_succ (d : ByteArray) (k : ℕ) : chainV d (k + 1) = compressV (chainV d k) (blockV d k) := by
  rw [chainV, chain_succ, map_range_eq_toBV _ (size_compress _ _ _), compress_eq]
  rfl

/-- `lastCv` and `lastBlock` are the chain before the last block and that block. -/
theorem lastCv_eq (m : ByteArray) : lastCv m = chainV (padData m) ((padData m).size / 128 - 1) := rfl

theorem lastBlock_eq (m : ByteArray) : lastBlock m = blockV (padData m) ((padData m).size / 128 - 1) := rfl

end FlockSoundness.Discharge.Hm
