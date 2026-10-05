import FlockSoundness.Discharge.Hidden.Wide.RowCommit

set_option Elab.async false

/-!
# A row's commit string at any length: the partial last block

`Hidden.row_commitP` hashes a row whose `B` data bytes are whole blocks (`B = 128 n`): `n` `sha512x3` compressions read
the data, and the `hm96` slot reads the padding block `HmRow.padWords`, all constants. A canonical row (`PROTOCOL.md`
§16.16) has `B = ⌈ℓ / 8⌉` bytes for any `ℓ < 2^64`. After the 128-byte prefix `P(ℓ)` its message is
`data ‖ padTail(128 + B)` (SHA-512's own padding), `n + 1 = ⌈(B + 17) / 128⌉` blocks: the first `n = chunks` are the
`sha512x3` slots', the last the `hm96` slot's. The data ends in one of the last two blocks (in block `n − 1` when
`B > 0` and `B mod 128` is 0 or at least 112), so either may be partial, its rest padding. `row_commitC` is `row_commitP` at that message: the same
chain, every block read bit by bit from `data ‖ padTail` (the last instead of as `padWords`), and `n = 0` allowed (a row
of at most 111 bytes is hashed by its `hm96` slot alone, `row_commit0P`'s case with data).

This is the one genuinely new hashing fact the canonical format needs; everything else generalizes existing lemmas.

**Proof work** (the bulk of the format's new proof): `padData (rowMsgP pre data)` is `pre ‖ data ‖ padTail(128 + B)`
(`Hm.padData_data` and `padTail`'s unfolding: the same `0x80`, `zeroPad` zeros and 16 length bytes); then
`blockV_rowP`'s block lemma for an unaligned `data ‖ tail` (block `k + 1` is bytes `128 k …` of `data ‖ tail`, by
`getD_append` on either side of `B`), `chainV_one_rowP` unchanged, and `lastIdx_rowMsgP` at `n + 1` blocks from the
`zeroPad` arithmetic (`(B + 17 + 127) / 128`, an `omega` over `B % 128`). The compression induction is `row_commitP`'s
verbatim with `hmsgW` from the new block lemma; the last step is `hm96_commit` with `hpad` read through `wordBit` like
`hmsgW`, and `hcv` from `hcvh0` (`n = 0`, `chainV_one_rowP`) or `hcvh` (`n > 0`).
-/

namespace FlockSoundness.Discharge.Canonical

open Flock.HmNets hiding and
open Flock Hm

/-- **A row's message after its prefix**: its `B` bytes, then SHA-512's padding of a message of `128 + B` bytes. -/
def rowTail (data : List UInt8) : List UInt8 := data ++ (HmRow.padTail (128 + data.length)).data.toList

theorem padTail_data (total : ℕ) :
    (HmRow.padTail total).data.toList =
      [128] ++ List.replicate ((239 - total % 128) % 128) 0 ++ lenBytes total := by
  simp only [HmRow.padTail, Std.Legacy.Range.forIn_eq_forIn_range', List.forIn_pure_yield_eq_foldl, pure_bind,
    Id.run_pure, Std.Legacy.Range.size, Nat.reduceSub, Nat.reduceAdd, Nat.sub_zero, Nat.div_one]
  rw [foldl_push_data (fun i => ((8 * total) >>> (8 * (15 - i)) % 256).toUInt8), lenBytes]
  simp only [Nat.mul_comm 8 total]
  rfl

/-- `padTail(128 + B)` is `Hm.padData`'s padding of a message of `128 + B` bytes. -/
theorem rowTail_eq (data : List UInt8) :
    rowTail data =
      data ++ [128] ++ List.replicate (zeroPad (128 + data.length)) 0 ++ lenBytes (128 + data.length) := by
  rw [rowTail, padTail_data,
    show (239 - (128 + data.length) % 128) % 128 = zeroPad (128 + data.length) by unfold zeroPad; omega]
  simp only [List.append_assoc]

theorem length_rowTail {data : List UInt8} {n : ℕ} (hn : n + 1 = (data.length + 17 + 127) / 128) :
    (rowTail data).length = 128 * (n + 1) := by
  rw [rowTail_eq]
  simp only [List.length_append, List.length_cons, List.length_nil, List.length_replicate, lenBytes, List.length_map,
    List.length_range']
  unfold zeroPad
  omega

section Pre

variable {pre : ByteArray} (hpre : pre.size = 128)
include hpre

theorem padData_rowTail (data : List UInt8) :
    (padData (Hidden.rowMsgP pre data)).data.toList = pre.data.toList ++ rowTail data := by
  have hs : (Hidden.rowMsgP pre data).size = 128 + data.length := by
    rw [← ByteArray.size_data, ← Array.length_toList]
    show (pre.data.toList ++ data).length = _
    rw [List.length_append, Hidden.length_pre hpre]
  rw [padData_data, hs, rowTail_eq]
  show (pre.data.toList ++ data) ++ _ ++ _ ++ _ = _
  simp only [List.append_assoc]

theorem lastIdx_rowTail {data : List UInt8} {n : ℕ} (hn : n + 1 = (data.length + 17 + 127) / 128) :
    (padData (Hidden.rowMsgP pre data)).size / 128 - 1 = n + 1 := by
  rw [← ByteArray.size_data, ← Array.length_toList, padData_rowTail hpre, List.length_append, Hidden.length_pre hpre,
    length_rowTail hn]
  omega

/-- After the prefix block, the chain is the prefix's midstate. -/
theorem chainV_one_rowC (data : List UInt8) :
    chainV (padData (Hidden.rowMsgP pre data)) 1 =
      (List.range 8).map fun i => (Sha512.compress Sha512.iv pre 0)[i]!.toBitVec := by
  have h1 : Sha512.compress Sha512.iv (padData (Hidden.rowMsgP pre data)) 0 = Sha512.compress Sha512.iv pre 0 := by
    refine compress_congr _ fun k hk => ?_
    rw [get!_eq_getD, get!_eq_getD, padData_rowTail hpre, Nat.zero_add,
      List.getD_append _ _ _ _ (by rw [Hidden.length_pre hpre]; omega)]
  rw [chainV, show chain (padData (Hidden.rowMsgP pre data)) 1 =
    Sha512.compress Sha512.iv (padData (Hidden.rowMsgP pre data)) 0 from rfl, h1]

/-- Block `k + 1` is bytes `128 k …` of `data ‖ padTail(128 + B)`, on either side of `B`. -/
theorem blockV_rowC (data : List UInt8) (k : ℕ) :
    blockV (padData (Hidden.rowMsgP pre data)) (k + 1) =
      (List.range 16).map fun i => (wordL (rowTail data) (128 * k) i).toBitVec := by
  rw [blockV]
  refine List.map_congr_left fun i _ => ?_
  rw [wordBE_eq_wordL, wordL_congr (l' := rowTail data) (off' := 128 * k) (i' := i) fun j _ => ?_]
  rw [padData_rowTail hpre, List.getD_append_right _ _ _ _ (by rw [Hidden.length_pre hpre]; omega),
    Hidden.length_pre hpre, show 128 * (k + 1) + 8 * i + j - 128 = 128 * k + 8 * i + j by omega]

/-- A block's message bits, read through `wordBit`, are `rowTail`'s bits. -/
theorem msgW_rowC (data : List UInt8) (f : ℕ → Bool) (k : ℕ)
    (hf : ∀ o < 128, ∀ t < 8, f (wordBit (8 * o + t)) = bitL (rowTail data) (8 * (128 * k + o) + t)) :
    ∀ q < 1024, f q = wbit (blockV (padData (Hidden.rowMsgP pre data)) (k + 1)) q := by
  intro q hq
  have ho : 8 * (q / 64) + (7 - q % 64 / 8) < 128 := by omega
  have hq' : wordBit (8 * (8 * (q / 64) + (7 - q % 64 / 8)) + q % 8) = q := by unfold wordBit; omega
  have e := hf _ ho (q % 8) (Nat.mod_lt _ (by norm_num))
  rw [hq'] at e
  rw [e, blockV_rowC hpre, wbit, List.getD_eq_getElem?_getD, List.getElem?_map,
    List.getElem?_range (show q / 64 < 16 by omega), Option.map_some, Option.getD_some,
    getLsbD_wordL _ _ _ (Nat.mod_lt _ (by norm_num)), bitL,
    show (8 * (128 * k + (8 * (q / 64) + (7 - q % 64 / 8))) + q % 8) / 8 = 128 * k + 8 * (q / 64) + (7 - q % 64 / 8)
      by omega, show (8 * (128 * k + (8 * (q / 64) + (7 - q % 64 / 8))) + q % 8) % 8 = q % 64 % 8 by omega]

end Pre

/-- **`row_commitP` at any length**: a row of `B` data bytes in `n + 1 = ⌈(B + 17) / 128⌉` message blocks after the
128-byte prefix `pre`, whose `n` `sha512x3` slots hold Δ's copies (the first `cv` the prefix's midstate, each next `cv`
the previous output, `msg` block `k` of `rowTail data`) and whose `hm96` slot reads the last output (or, at `n = 0`, the
prefix's midstate) and the last block of `rowTail data`, outputs the commit string of `pre ‖ data` under the salt `s`.

*Route*: the module docstring's. -/
theorem row_commitC {pre : ByteArray} (hpre : pre.size = 128) {data : List UInt8} {n : ℕ}
    (hn : n + 1 = (data.length + 17 + 127) / 128) (zs : ℕ → Var → Bool) (sub : ℕ → ℕ) (hsub : ∀ k < n, sub k < 3)
    (hrows : ∀ k < n, stHolds (zs k) (compression3.run St.empty).2)
    (hcv0 : 0 < n → ∀ q : ℕ, q < 512 → zs 0 (1536 * sub 0 + q) = wordsBit (Sha512.compress Sha512.iv pre 0) q)
    (hchain : ∀ k, k + 1 < n → ∀ q : ℕ, q < 512 →
      zs (k + 1) (1536 * sub (k + 1) + q) = Hm.bit (zs k) ((compression3.run St.empty).1.2.getD (sub k) []) q)
    (hmsg : ∀ k < n, ∀ o < 128, ∀ t < 8,
      zs k (1536 * sub k + 512 + wordBit (8 * o + t)) = bitL (rowTail data) (8 * (128 * k + o) + t))
    (zh : Var → Bool) (hh : stHolds zh ((hm96Row Hm96.Default512.key).run St.empty).2)
    (hcvh0 : n = 0 → ∀ q : ℕ, q < 512 → zh q = wordsBit (Sha512.compress Sha512.iv pre 0) q)
    (hcvh : 0 < n → ∀ q : ℕ, q < 512 →
      zh q = Hm.bit (zs (n - 1)) ((compression3.run St.empty).1.2.getD (sub (n - 1)) []) q)
    (hpadh : ∀ o < 128, ∀ t < 8, zh (512 + wordBit (8 * o + t)) = bitL (rowTail data) (8 * (128 * n + o) + t))
    {s : List UInt8} (hs : s.length = 192) (hy : ∀ q : ℕ, q < 1536 → zh (1536 + q) = bitL s q) :
    ∀ t < 1024, Hm.bit zh ((hm96Row Hm96.Default512.key).run St.empty).1.2.2.2 t =
      bitL (Binding.commitString (pre.data.toList ++ data) s) t := by
  set P := padData (Hidden.rowMsgP pre data) with hP
  have hmsgW : ∀ k < n, ∀ q : ℕ, q < 1024 → zs k (1536 * sub k + 512 + q) = wbit (blockV P (k + 1)) q :=
    fun k hk => msgW_rowC hpre data (fun x => zs k (1536 * sub k + 512 + x)) k (hmsg k hk)
  have hlen8 : ∀ k, (chainV P k).length = 8 := fun k => by rw [chainV, List.length_map, List.length_range]
  have hlen16 : ∀ k, (blockV P k).length = 16 := fun k => by rw [blockV, List.length_map, List.length_range]
  have hout : ∀ k < n, ∀ q < 512,
      Hm.bit (zs k) ((compression3.run St.empty).1.2.getD (sub k) []) q = wbit (chainV P (k + 2)) q := by
    intro k
    induction k with
    | zero =>
      intro hk q hq
      refine (compression3_bits (zs 0) (hrows 0 hk) (hsub 0 hk) (hlen8 1) (hlen16 1) (fun q hq => ?_)
        (hmsgW 0 hk) q hq).trans (by rw [show 0 + 2 = 1 + 1 from rfl, chainV_succ P 1])
      rw [hcv0 hk q hq, wordsBit_eq _ (show q < 64 * 8 by omega), chainV_one_rowC hpre]
    | succ k ih =>
      intro hk q hq
      refine (compression3_bits (zs (k + 1)) (hrows _ hk) (hsub _ hk) (hlen8 (k + 2)) (hlen16 (k + 2))
        (fun q hq => ?_) (hmsgW _ hk) q hq).trans (by rw [chainV_succ P (k + 1 + 1)])
      rw [hchain k hk q hq, ih (by omega) q hq]
  have hlast := lastIdx_rowTail hpre hn
  have hcv : ∀ q < 512, zh q = wbit (lastCv (Hidden.rowMsgP pre data)) q := by
    intro q hq
    rw [lastCv_eq, hlast]
    rcases Nat.eq_zero_or_pos n with h0 | h0
    · rw [hcvh0 h0 q hq, wordsBit_eq _ (show q < 64 * 8 by omega), show n + 1 = 1 by omega, chainV_one_rowC hpre]
    · rw [hcvh h0 q hq, hout (n - 1) (by omega) q hq, show n - 1 + 2 = n + 1 by omega]
  have hpad : ∀ q < 1024, zh (512 + q) = wbit (lastBlock (Hidden.rowMsgP pre data)) q := by
    intro q hq
    rw [lastBlock_eq, hlast]
    exact msgW_rowC hpre data (fun x => zh (512 + x)) n hpadh q hq
  exact (hm96_commit zh hh _ hs hcv hpad hy).2

end FlockSoundness.Discharge.Canonical
