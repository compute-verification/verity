import FlockSoundness.Discharge.Hm.Nets
import FlockSoundness.Protocol.Discharge.Hidden.Wide.RowCommit
import FlockSoundness.Discharge.Hm.Port

/-!
# A row's commit string under any 128-byte prefix

`Hm.row_commit` and `Hm.port_commit_rows` read `HmRow.rowPrefix words` only through its 128 bytes and its midstate.
These are theirs for any 128-byte prefix `pre` (`row_commitP`, `port_commit_rowsP`), so for `HmRow.prefixOf` of a
`sha512/row/v1` or `sha512/row/v2` row (`size_prefixOf`), whose bytes are whole blocks. A row of no compression
(`port_commit_rows0`) is hashed by its `hm96` slot alone: its `cv` the prefix's midstate, its `pad` the padding block of
the prefix.
-/

set_option Elab.async false

namespace FlockSoundness.Discharge.Hidden

open Flock.HmNets hiding and

open Flock Hm

/-! ## The prefix -/

theorem size_rowPrefixSeg (l : ℕ) : (HmRow.rowPrefixSeg l).size = 128 := by
  have h0 : ("verity/sha512-row-seg/v1".toUTF8.push 0).size = 25 := by decide +kernel
  have h8 : ∀ x, (HmRow.be64 x).size = 8 := fun x => by simp [HmRow.be64, ByteArray.size]
  have h4 : (be32 1024).size = 4 := by decide +kernel
  refine size_pad ?_
  rw [ByteArray.size_append, ByteArray.size_append, ByteArray.size_append, h0, h8, h8, h4]
  omega

theorem size_prefixOf (p : Port) : (HmRow.prefixOf p).size = 128 := by
  unfold HmRow.prefixOf
  split
  · exact size_rowPrefixSeg _
  · split
    · exact size_rowPrefixV2 _
    · exact size_rowPrefix _

/-! ## The row's bytes after a prefix -/

section Pre

variable {pre : ByteArray} (hpre : pre.size = 128)
include hpre

theorem rowMsgP_pad {n : ℕ} {data : List UInt8} (hd : data.length = 128 * n) :
    (padData (rowMsgP pre data)).data.toList = pre.data.toList ++ data ++ rowPadT n := by
  have hs : (rowMsgP pre data).size = 128 + 128 * n := by
    rw [← ByteArray.size_data, ← Array.length_toList]
    show (pre.data.toList ++ data).length = _
    rw [List.length_append, length_pre hpre, hd]
  rw [padData_data, hs, rowPadT, show zeroPad (128 + 128 * n) = 111 by unfold zeroPad; omega]
  show (pre.data.toList ++ data) ++ _ ++ _ ++ _ = _
  simp only [List.append_assoc]

theorem size_padData_rowMsgP {n : ℕ} {data : List UInt8} (hd : data.length = 128 * n) :
    (padData (rowMsgP pre data)).size = 128 * (n + 2) := by
  rw [← ByteArray.size_data, ← Array.length_toList, rowMsgP_pad hpre hd]
  simp only [List.length_append, length_pre hpre, hd, rowPadT, lenBytes, List.length_cons, List.length_nil,
    List.length_replicate, List.length_map, List.length_range']
  ring

theorem getD_rowPadP_pre {n : ℕ} {data : List UInt8} (hd : data.length = 128 * n) {i : ℕ} (hi : i < 128) :
    (pre.data.toList ++ data ++ rowPadT n).getD i 0 = pre.data.toList.getD i 0 := by
  rw [List.getD_append _ _ _ _ (by simp [hd, length_pre hpre]; omega),
    List.getD_append _ _ _ _ (by rw [length_pre hpre]; omega)]

theorem getD_rowPadP_data {n : ℕ} {data : List UInt8} (hd : data.length = 128 * n) {i : ℕ} (hi1 : 128 ≤ i)
    (hi2 : i < 128 + 128 * n) :
    (pre.data.toList ++ data ++ rowPadT n).getD i 0 = data.getD (i - 128) 0 := by
  rw [List.getD_append _ _ _ _ (by simp [hd, length_pre hpre]; omega),
    List.getD_append_right _ _ _ _ (by rw [length_pre hpre]; omega), length_pre hpre]

theorem getD_rowPadP_tail {n : ℕ} {data : List UInt8} (hd : data.length = 128 * n) {i : ℕ}
    (hi : 128 + 128 * n ≤ i) :
    (pre.data.toList ++ data ++ rowPadT n).getD i 0 = (rowPadT n).getD (i - (128 + 128 * n)) 0 := by
  rw [List.getD_append_right _ _ _ _ (by simp [hd, length_pre hpre]; omega)]
  simp [hd, length_pre hpre]

/-- After the prefix block, the chain is the prefix's midstate. -/
theorem chainV_one_rowP {n : ℕ} {data : List UInt8} (hd : data.length = 128 * n) :
    chainV (padData (rowMsgP pre data)) 1 =
      (List.range 8).map fun i => (Sha512.compress Sha512.iv pre 0)[i]!.toBitVec := by
  have h1 : Sha512.compress Sha512.iv (padData (rowMsgP pre data)) 0 = Sha512.compress Sha512.iv pre 0 := by
    refine compress_congr _ fun k hk => ?_
    rw [get!_eq_getD, get!_eq_getD, rowMsgP_pad hpre hd, Nat.zero_add, getD_rowPadP_pre hpre hd hk]
  rw [chainV, show chain (padData (rowMsgP pre data)) 1 = Sha512.compress Sha512.iv (padData (rowMsgP pre data)) 0
    from rfl, h1]

/-- Block `k + 1` (`k < n`) is chunk `k` of the data. -/
theorem blockV_rowP {n : ℕ} {data : List UInt8} (hd : data.length = 128 * n) {k : ℕ} (hk : k < n) :
    blockV (padData (rowMsgP pre data)) (k + 1) = (List.range 16).map fun i => (wordL data (128 * k) i).toBitVec := by
  rw [blockV]
  refine List.map_congr_left fun i hi => ?_
  simp only [List.mem_range] at hi
  rw [wordBE_eq_wordL, wordL_congr (l' := data) (off' := 128 * k) (i' := i) fun j hj => ?_]
  rw [rowMsgP_pad hpre hd, getD_rowPadP_data hpre hd (by omega) (by nlinarith),
    show 128 * (k + 1) + 8 * i + j - 128 = 128 * k + 8 * i + j by omega]

/-- The last block (`n + 1`) is the padding block's words when the row has `2 · words = 128 n` data bytes. -/
theorem blockV_rowP_last {words n : ℕ} (hw : 2 * words = 128 * n) {data : List UInt8} (hd : data.length = 128 * n) :
    blockV (padData (rowMsgP pre data)) (n + 1) =
      (List.range 16).map fun i => (HmRow.padWords words)[i]!.toBitVec := by
  rw [blockV]
  refine List.map_congr_left fun i hi => ?_
  simp only [List.mem_range] at hi
  rw [HmRow.padWords, wordsBE_getElem! hi, wordBE_eq_wordL, padBlock_data, hw,
    wordL_congr (l' := [128] ++ List.replicate 111 0 ++ lenBytes (128 + 128 * n)) (off' := 0) (i' := i) fun j hj => ?_]
  rw [rowMsgP_pad hpre hd, getD_rowPadP_tail hpre hd (by nlinarith), rowPadT,
    show 128 * (n + 1) + 8 * i + j - (128 + 128 * n) = 0 + 8 * i + j by ring_nf; omega]

theorem lastIdx_rowMsgP {n : ℕ} {data : List UInt8} (hd : data.length = 128 * n) :
    (padData (rowMsgP pre data)).size / 128 - 1 = n + 1 := by
  rw [size_padData_rowMsgP hpre hd, Nat.mul_div_cancel_left _ (by norm_num)]
  rfl

/-- **`Hm.row_commit` under the prefix `pre`**: a row of `2 · words = 128 n` data bytes (`n > 0`) whose slots hold Δ's
copies, the first `cv` the prefix's midstate, outputs the commit string of `pre ‖ data`. -/
theorem row_commitP {words n : ℕ} (hn : 0 < n) (hw : 2 * words = 128 * n) {data : List UInt8}
    (hd : data.length = 128 * n) (zs : ℕ → Var → Bool) (sub : ℕ → ℕ) (hsub : ∀ k < n, sub k < 3)
    (hrows : ∀ k < n, stHolds (zs k) (compression3.run St.empty).2)
    (hcv0 : ∀ q : ℕ, q < 512 → zs 0 (1536 * sub 0 + q) = wordsBit (Sha512.compress Sha512.iv pre 0) q)
    (hchain : ∀ k, k + 1 < n → ∀ q : ℕ, q < 512 →
      zs (k + 1) (1536 * sub (k + 1) + q) = Hm.bit (zs k) ((compression3.run St.empty).1.2.getD (sub k) []) q)
    (hmsg : ∀ k < n, ∀ o < 128, ∀ t < 8,
      zs k (1536 * sub k + 512 + wordBit (8 * o + t)) = bitL data (8 * (128 * k + o) + t))
    (zh : Var → Bool) (hh : stHolds zh ((hm96Row Hm96.Default512.key).run St.empty).2)
    (hcvh : ∀ q : ℕ, q < 512 → zh q = Hm.bit (zs (n - 1)) ((compression3.run St.empty).1.2.getD (sub (n - 1)) []) q)
    (hpadh : ∀ q : ℕ, q < 1024 → zh (512 + q) = wordsBit (HmRow.padWords words) q)
    {s : List UInt8} (hs : s.length = 192) (hy : ∀ q : ℕ, q < 1536 → zh (1536 + q) = bitL s q) :
    ∀ t < 1024, Hm.bit zh ((hm96Row Hm96.Default512.key).run St.empty).1.2.2.2 t =
      bitL (Binding.commitString (pre.data.toList ++ data) s) t := by
  set P := padData (rowMsgP pre data) with hP
  have hmsgW : ∀ k < n, ∀ q : ℕ, q < 1024 → zs k (1536 * sub k + 512 + q) = wbit (blockV P (k + 1)) q := by
    intro k hk q hq
    have ho : 8 * (q / 64) + (7 - q % 64 / 8) < 128 := by omega
    have hq' : wordBit (8 * (8 * (q / 64) + (7 - q % 64 / 8)) + q % 8) = q := by unfold wordBit; omega
    have e := hmsg k hk (8 * (q / 64) + (7 - q % 64 / 8)) ho (q % 8) (Nat.mod_lt _ (by norm_num))
    rw [hq'] at e
    rw [e, blockV_rowP hpre hd hk, wbit, List.getD_eq_getElem?_getD, List.getElem?_map,
      List.getElem?_range (show q / 64 < 16 by omega), Option.map_some, Option.getD_some,
      getLsbD_wordL _ _ _ (Nat.mod_lt _ (by norm_num)), bitL,
      show (8 * (128 * k + (8 * (q / 64) + (7 - q % 64 / 8))) + q % 8) / 8 = 128 * k + 8 * (q / 64) + (7 - q % 64 / 8)
        by omega, show (8 * (128 * k + (8 * (q / 64) + (7 - q % 64 / 8))) + q % 8) % 8 = q % 64 % 8 by omega]
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
      rw [hcv0 q hq, wordsBit_eq _ (show q < 64 * 8 by omega), chainV_one_rowP hpre hd]
    | succ k ih =>
      intro hk q hq
      refine (compression3_bits (zs (k + 1)) (hrows _ hk) (hsub _ hk) (hlen8 (k + 2)) (hlen16 (k + 2))
        (fun q hq => ?_) (hmsgW _ hk) q hq).trans (by rw [chainV_succ P (k + 1 + 1)])
      rw [hchain k hk q hq, ih (by omega) q hq]
  have hlast := lastIdx_rowMsgP hpre (n := n) hd
  have hcv : ∀ q < 512, zh q = wbit (lastCv (rowMsgP pre data)) q := by
    intro q hq
    rw [hcvh q hq, hout (n - 1) (by omega) q hq, lastCv_eq, hlast, show n - 1 + 2 = n + 1 by omega]
  have hpad : ∀ q < 1024, zh (512 + q) = wbit (lastBlock (rowMsgP pre data)) q := by
    intro q hq
    rw [hpadh q hq, wordsBit_eq _ (show q < 64 * 16 by omega), lastBlock_eq, hlast, blockV_rowP_last hpre hw hd]
  exact (hm96_commit zh hh _ hs hcv hpad hy).2

/-- **A row of no compression**: its `hm96` slot, its `cv` the prefix's midstate and its `pad` the padding block after
the prefix, outputs the commit string of `pre`. -/
theorem row_commit0P (zh : Var → Bool) (hh : stHolds zh ((hm96Row Hm96.Default512.key).run St.empty).2)
    (hcvh : ∀ q : ℕ, q < 512 → zh q = wordsBit (Sha512.compress Sha512.iv pre 0) q)
    (hpadh : ∀ q : ℕ, q < 1024 → zh (512 + q) = wordsBit (HmRow.padWords 0) q)
    {s : List UInt8} (hs : s.length = 192) (hy : ∀ q : ℕ, q < 1536 → zh (1536 + q) = bitL s q) :
    ∀ t < 1024, Hm.bit zh ((hm96Row Hm96.Default512.key).run St.empty).1.2.2.2 t =
      bitL (Binding.commitString (pre.data.toList ++ []) s) t := by
  have hd : ([] : List UInt8).length = 128 * 0 := rfl
  have hlast := lastIdx_rowMsgP hpre hd
  have hcv : ∀ q < 512, zh q = wbit (lastCv (rowMsgP pre [])) q := by
    intro q hq
    rw [hcvh q hq, wordsBit_eq _ (show q < 64 * 8 by omega), lastCv_eq, hlast, chainV_one_rowP hpre hd]
  have hpad : ∀ q < 1024, zh (512 + q) = wbit (lastBlock (rowMsgP pre [])) q := by
    intro q hq
    rw [hpadh q hq, wordsBit_eq _ (show q < 64 * 16 by omega), lastBlock_eq, hlast,
      blockV_rowP_last hpre (words := 0) (n := 0) rfl hd]
  exact (hm96_commit zh hh _ hs hcv hpad hy).2

end Pre

/-! ## The port -/

/-- **`Hm.port_commit` under the prefix `pre`.** -/
theorem port_commitP (c : Circuit) (w : ℕ → Bool) (g p : ℕ) {pre : ByteArray} (hpre : pre.size = 128) {words n : ℕ}
    (hs : c.ports[p]!.segs = none) (hn : 0 < n) (hw : 2 * words = 128 * n)
    (hslots : n ≤ ((HmRow.rowSlots c.ports).2[p]!).size) (hsub : ∀ k < n, shaSub c p k < 3)
    (hsin : ∀ s < 3, shaIn c (2 * s) = 1536 * s ∧ shaIn c (2 * s + 1) = 1536 * s + 512)
    (hhin : hmIn c 0 = 0 ∧ hmIn c 1 = 512 ∧ hmIn c 2 = 1536)
    (zs : ℕ → Var → Bool) (hzs : ∀ k < n, ∀ v : ℕ, v < 4608 → zs k v = w (shaBase c g p k + v))
    (hrows : ∀ k < n, stHolds (zs k) (compression3.run St.empty).2)
    (hout : ∀ k < n, ∀ t < 512, w (shaBase c g p k + shaOut c (shaSub c p k) + t) =
      Hm.bit (zs k) ((compression3.run St.empty).1.2.getD (shaSub c p k) []) t)
    (zh : Var → Bool) (hzh : ∀ v : ℕ, v < 3072 → zh v = w (c.slotCol c.hmNet g p + v))
    (hh : stHolds zh ((hm96Row Hm96.Default512.key).run St.empty).2)
    (hbc : ∀ t < 1024, w (c.slotCol c.hmNet g p + hmOut c + t) = Hm.bit zh ((hm96Row Hm96.Default512.key).run St.empty).1.2.2.2 t)
    (hcv0 : ∀ t < 512, w (shaBase c g p 0 + shaIn c (2 * shaSub c p 0) + t) =
      wordsBit (Sha512.compress Sha512.iv pre 0) t)
    (hchain : ∀ k, k + 1 < n → ∀ t < 512, w (shaBase c g p (k + 1) + shaIn c (2 * shaSub c p (k + 1)) + t) =
      w (shaBase c g p k + shaOut c (shaSub c p k) + t))
    (hcvh : ∀ t < 512, w (c.slotCol c.hmNet g p + hmIn c 0 + t) =
      w (shaBase c g p (n - 1) + shaOut c (shaSub c p (n - 1)) + t))
    (hpad : ∀ t < 1024, w (c.slotCol c.hmNet g p + hmIn c 1 + t) = wordsBit (HmRow.padWords words) t) :
    bytesOf w 128 (bcCol c g p) = Binding.commitString
      (pre.data.toList ++ bytesOf w (2 * words) (Typed.msgCol c g p)) (bytesOf w 192 (saltCol c g p)) := by
  have hd : (bytesOf w (2 * words) (Typed.msgCol c g p)).length = 128 * n := by rw [length_bytesOf, hw]
  have hrow := row_commitP hpre hn hw hd zs (shaSub c p) hsub hrows
    (fun q hq => by
      rw [hzs 0 hn _ (by have := hsub 0 hn; omega), ← hcv0 q hq, (hsin _ (hsub 0 hn)).1]
      exact congrArg w (by omega))
    (fun k hk q hq => by
      rw [hzs (k + 1) hk _ (by have := hsub (k + 1) hk; omega), ← hout k (by omega) q hq, ← hchain k hk q hq,
        (hsin _ (hsub (k + 1) hk)).1]
      exact congrArg w (by omega))
    (fun k hk o ho t ht => by
      have hwb : wordBit (8 * o + t) < 1024 := by unfold wordBit; omega
      rw [hzs k hk _ (by have := hsub k hk; omega), bitL_bytesOf _ _ (by omega),
        show (8 * (128 * k + o) + t) / 8 = 128 * k + o by omega, show (8 * (128 * k + o) + t) % 8 = t by omega,
        msgCol_eq c g p k hs (by omega) ho ht, (hsin _ (hsub k hk)).2]
      exact congrArg w (by omega))
    zh hh
    (fun q hq => by
      rw [hzh q (by omega), ← hout (n - 1) (by omega) q hq, ← hcvh q hq, hhin.1, Nat.add_zero])
    (fun q hq => by
      rw [hzh _ (by omega), ← hpad q hq, hhin.2.1]
      exact congrArg w (by omega))
    (length_bytesOf w 192 (saltCol c g p))
    (fun q hq => by
      rw [hzh _ (by omega), bitL_bytesOf _ _ (by omega), saltCol, hhin.2.2]
      exact congrArg w (by omega))
  refine eq_of_bitL (by rw [length_bytesOf, length_commitString]) fun t ht => ?_
  rw [length_bytesOf] at ht
  rw [bitL_bytesOf _ _ ht, ← hrow t ht, ← hbc t ht, bcCol]
  exact congrArg w (by omega)

section Nets

variable (c : Circuit) (w : ℕ → Bool) (g p : ℕ)

end Nets

/-- **`Hm.port_commit_rows` under the prefix `pre`**: the slots hold their nets' rows and the block holds Δ's copies,
the first `cv` the prefix's midstate. -/
theorem port_commit_rowsP (c : Circuit) (w : ℕ → Bool) (g p : ℕ) {pre : ByteArray} (hpre : pre.size = 128)
    {words n : ℕ} (hs : c.ports[p]!.segs = none) (hn : 0 < n)
    (hw : 2 * words = 128 * n) (hslots : n ≤ ((HmRow.rowSlots c.ports).2[p]!).size) (hck : HmNets.check c = .ok ())
    (hsub : ∀ k < n, shaSub c p k < 3)
    (hsha : ∀ k < n, LaidRows (shaLaid ()) 8192 fun j => w (shaBase c g p k + j))
    (hhm : LaidRows (hmLaid ()) 4096 fun j => w (c.slotCol c.hmNet g p + j))
    (hcv0 : ∀ t < 512, w (shaBase c g p 0 + shaIn c (2 * shaSub c p 0) + t) =
      wordsBit (Sha512.compress Sha512.iv pre 0) t)
    (hchain : ∀ k, k + 1 < n → ∀ t < 512, w (shaBase c g p (k + 1) + shaIn c (2 * shaSub c p (k + 1)) + t) =
      w (shaBase c g p k + shaOut c (shaSub c p k) + t))
    (hcvh : ∀ t < 512, w (c.slotCol c.hmNet g p + hmIn c 0 + t) =
      w (shaBase c g p (n - 1) + shaOut c (shaSub c p (n - 1)) + t))
    (hpad : ∀ t < 1024, w (c.slotCol c.hmNet g p + hmIn c 1 + t) = wordsBit (HmRow.padWords words) t) :
    bytesOf w 128 (bcCol c g p) = Binding.commitString
      (pre.data.toList ++ bytesOf w (2 * words) (Typed.msgCol c g p)) (bytesOf w 192 (saltCol c g p)) := by
  have hshaL : ∀ k < n, LaidRows (layout (compression3.run St.empty).2 SHA_IN SHA_OUT (compression3.run St.empty).1.2.flatten.toArray)
      (inRowsOf SHA_IN) fun j => w (shaBase c g p k + j) := by
    intro k hk
    have := hsha k hk
    rw [shaLaid_eq] at this
    rw [inRows_SHA_IN]
    exact this
  have hgs : Good (SHA_IN.foldl (· + ·) 0) (compression3.run St.empty).2 := by rw [sum_SHA_IN]; exact good_compression3
  let zs : ℕ → Var → Bool := fun k x => w (shaBase c g p k + col (SHA_IN.foldl (· + ·) 0) (inRowsOf SHA_IN) x)
  have hrows : ∀ k < n, stHolds (zs k) (compression3.run St.empty).2 := fun k hk => laid_rows _ _ _ _ _ hgs (hshaL k hk)
  have hzs : ∀ k < n, ∀ v : ℕ, v < 4608 → zs k v = w (shaBase c g p k + v) := by
    intro k _ v hv
    show w (shaBase c g p k + col _ _ v) = _
    rw [col_input (by rw [sum_SHA_IN]; exact hv)]
  have hout : ∀ k < n, ∀ t < 512, w (shaBase c g p k + shaOut c (shaSub c p k) + t) =
      Hm.bit (zs k) ((compression3.run St.empty).1.2.getD (shaSub c p k) []) t := by
    intro k hk t ht
    have hs := hsub k hk
    obtain ⟨-, hl3, -⟩ := compression3_spec (zs k) (hrows k hk)
    have hlens := compression3_lens (zs k) (hrows k hk)
    have hflat : (compression3.run St.empty).1.2.flatten.length = 1536 := by
      rw [List.length_flatten, List.map_congr_left (g := fun _ => 512) hlens, List.map_const', List.sum_replicate, hl3]
      rfl
    have hj : 512 * shaSub c p k + t < (compression3.run St.empty).1.2.flatten.toArray.size := by
      rw [List.size_toArray, hflat]; omega
    have := laid_outs (compression3.run St.empty).2 SHA_IN SHA_OUT _ (fun j => w (shaBase c g p k + j))
      (by rw [List.size_toArray, hflat]; decide) (hshaL k hk) _ hj
    rw [shaOut_eq hck hs, Nat.add_assoc, Nat.add_assoc, this, List.getElem_toArray,
      ← bit_of_lt (by rw [hflat]; omega), bit_flatten_of_len (zs k) hlens ht]
  exact port_commitP c w g p hpre hs hn hw hslots hsub (shaIn_eq hck) (hmIn_eq hck) zs hzs hrows hout _
    (hm_zh_in c w g p) (hm_zh c w g p hhm) (hm_bc c w g p hck hhm) hcv0 hchain hcvh hpad

/-- **A row of no compression at its columns**: its `hm96` slot holds its net's rows, its `cv` the prefix's midstate
and its `pad` the padding block after the prefix, so the bytes at `bcCol` are the commit string of the prefix. -/
theorem port_commit_rows0 (c : Circuit) (w : ℕ → Bool) (g p : ℕ) {pre : ByteArray} (hpre : pre.size = 128)
    (hck : HmNets.check c = .ok ()) (hhm : LaidRows (hmLaid ()) 4096 fun j => w (c.slotCol c.hmNet g p + j))
    (hcvh : ∀ t < 512, w (c.slotCol c.hmNet g p + hmIn c 0 + t) = wordsBit (Sha512.compress Sha512.iv pre 0) t)
    (hpad : ∀ t < 1024, w (c.slotCol c.hmNet g p + hmIn c 1 + t) = wordsBit (HmRow.padWords 0) t) :
    bytesOf w 128 (bcCol c g p) = Binding.commitString
      (pre.data.toList ++ bytesOf w (2 * 0) (Typed.msgCol c g p)) (bytesOf w 192 (saltCol c g p)) := by
  have hhin := hmIn_eq hck
  have hzh := hm_zh_in c w g p
  beta_reduce at hzh
  have hrow := row_commit0P hpre _ (hm_zh c w g p hhm)
    (fun q hq => by rw [hzh q (by omega), ← hcvh q hq, hhin.1, Nat.add_zero])
    (fun q hq => by
      rw [hzh _ (by omega), ← hpad q hq, hhin.2.1, Nat.add_assoc])
    (length_bytesOf w 192 (saltCol c g p))
    (fun q hq => by
      rw [hzh _ (by omega), bitL_bytesOf _ _ (by omega), saltCol, hhin.2.2]
      exact congrArg w (by omega))
  refine eq_of_bitL (by rw [length_bytesOf, length_commitString]) fun t ht => ?_
  rw [length_bytesOf] at ht
  rw [bitL_bytesOf _ _ ht, show bytesOf w (2 * 0) (Typed.msgCol c g p) = [] from rfl, ← hrow t ht,
    ← hm_bc c w g p hck hhm t ht, bcCol]
  exact congrArg w (by omega)

end FlockSoundness.Discharge.Hidden
