import Proofs.Flock.Soundness.Discharge.Layout.OutPin

set_option Elab.async false

/-!
# The bytes `sha512/row/v1` hashes for a row

`Binding.HmRows.enc` is the byte string `HmRowComputes` says a row's `hm96` slot commits to: for port `p`'s row of
`words` u16 words, the circuit's `sha512x3` slots hash `rowPrefix words` and the row's `2 · words` bytes at
`Typed.msgCol` ([e2e-hm]'s `Discharge.Hm.port_commit`). The layout's values carry their width (`BitVal`), so one
injective `enc` recovers each port's words from its value:

* `rowEnc`: `rowPrefix w ‖ bytes` for a value of `16 w` bits with nothing set past them (`RowShaped`), and a longer code
  of every other value; `rowEnc_inj`;
* `rowEnc_row`: at a row of `16 k` bits it is `rowPrefix k` and the row's `2 k` bytes;
* `keyRow_enc`: at `keyProg`'s places, row port `q` of `16 · words` bits (`rowW q`) holds a value whose `rowEnc` is
  `rowPrefix words` and the `2 · words` bytes the witness holds at the drawn unit's message columns, `port_commit`'s
  row with the block's bits at the site;
* `keyRowsEnc`: `rs` there (`keyRowsOut`) with `enc := rowEnc` and `bitCode` for the other commit strings.
-/

namespace FlockSoundness.Discharge.Layout

open Flock Binding

/-! ## Bytes and their bits -/

/-- **The bits of a sum of bits' powers** are the bits. -/
theorem sumBits_testBit : ∀ (k : ℕ) (b : Fin k → Bool) (i : ℕ),
    (∑ t : Fin k, if b t then 2 ^ t.val else 0).testBit i = if h : i < k then b ⟨i, h⟩ else false
  | 0, b, i => by simp
  | k + 1, b, i => by
    rw [Fin.sum_univ_succ]
    have hrest : (∑ t : Fin k, if b t.succ then 2 ^ (t.succ : Fin (k + 1)).val else 0) =
        2 * ∑ t : Fin k, if b t.succ then 2 ^ t.val else 0 := by
      rw [Finset.mul_sum]
      refine Finset.sum_congr rfl fun t _ => ?_
      simp only [Fin.val_succ, pow_succ]
      split_ifs <;> ring
    rw [hrest]
    have ih := sumBits_testBit k (fun t => b t.succ)
    generalize (∑ t : Fin k, if b t.succ then 2 ^ t.val else 0) = S at ih ⊢
    cases i with
    | zero =>
      rw [Nat.testBit_zero, dif_pos (Nat.zero_lt_succ k)]
      cases h : b 0 <;> simp [Fin.zero_eta, h]
    | succ i =>
      have hdiv : ((if b 0 then 2 ^ (0 : Fin (k + 1)).val else 0) + 2 * S) / 2 = S := by
        simp only [Fin.val_zero, pow_zero]
        split_ifs <;> omega
      rw [Nat.testBit_succ, hdiv, ih i]
      by_cases hi : i < k
      · rw [dif_pos hi, dif_pos (Nat.succ_lt_succ hi)]
        rfl
      · rw [dif_neg hi, dif_neg (by omega)]

/-- **A byte's bits**: bit `t` of `byteOf b` is `b t`. -/
theorem byteOf_bit (b : Fin 8 → Bool) (t : Fin 8) : (byteOf b).toNat.testBit t.val = b t := by
  have hlt : (∑ t : Fin 8, if b t then 2 ^ t.val else 0) < 2 ^ 8 :=
    Nat.lt_pow_two_of_testBit _ fun i hi => by rw [sumBits_testBit, dif_neg (by omega)]
  have e : (byteOf b).toNat = ∑ t : Fin 8, if b t then 2 ^ t.val else 0 := by
    unfold byteOf
    rw [UInt8.toNat_ofNat', Nat.mod_eq_of_lt (by simpa using hlt)]
  rw [e, sumBits_testBit, dif_pos t.isLt]

theorem byteOf_inj {a b : Fin 8 → Bool} (h : byteOf a = byteOf b) : a = b :=
  funext fun t => by rw [← byteOf_bit a t, h, byteOf_bit]

/-! ## The row prefix -/

theorem be32_size (x : UInt32) : (be32 x).size = 4 := by
  simp only [be32, Id.run, Std.Legacy.Range.forIn_eq_forIn_range', List.forIn_pure_yield_eq_foldl, pure_bind,
    Std.Legacy.Range.size]
  simp only [List.range', List.foldl]
  rfl

theorem rowLabel_size : ("verity/sha512-row/v1".toUTF8.push 0 ++ ByteArray.mk #[1, 16]).size = 23 := by
  decide

/-- **`sha512/row/v1`'s first block is one SHA-512 block**, 128 bytes, at every word count. -/
theorem rowPrefix_size (w : ℕ) : (HmRow.rowPrefix w).size = 128 := by
  unfold HmRow.rowPrefix Hm96.pad
  rw [ByteArray.size_append, ByteArray.size_append, rowLabel_size, be32_size]
  rfl

theorem rowPrefix_length (w : ℕ) : (HmRow.rowPrefix w).data.toList.length = 128 := by
  rw [Array.length_toList]
  exact rowPrefix_size w

/-! ## The encoding -/

variable {W : ℕ}

/-- Bit `i` of a value, `false` past `W`. -/
def BitVal.get (v : BitVal W) (i : ℕ) : Bool := if h : i < W then v.2 ⟨i, h⟩ else false

/-- **A row's value**: a width of `16 w` bits (`w` u16 words), and no bit set past it. -/
def RowShaped (v : BitVal W) : Prop := v.1.val % 16 = 0 ∧ ∀ i : Fin W, v.1.val ≤ i.val → v.2 i = false

instance instDecidablePredBitValRowShaped : DecidablePred (RowShaped (W := W)) := fun _ => inferInstanceAs (Decidable (_ ∧ _))

/-- One byte per width and per bit: a code of fixed length `2 W + 1`. -/
def bitCode (v : BitVal W) : List UInt8 :=
  List.ofFn (fun i : Fin (W + 1) => if i = v.1 then (1 : UInt8) else 0) ++
    List.ofFn (fun i : Fin W => if v.2 i then (1 : UInt8) else 0)

/-- The `2 w` bytes of a row of `w` words: byte `j`'s bit `t` is bit `8 j + t`. -/
def rowBytes (v : BitVal W) : List UInt8 :=
  List.ofFn fun j : Fin (2 * (v.1.val / 16)) => byteOf fun t => v.get (8 * j.val + t.val)

/-- **The bytes `sha512/row/v1` hashes for a value**: a row of `w` words is `rowPrefix w` and its `2 w` bytes, as the
circuit's `sha512x3` slots hash it; any other value is `128 + W` zero bytes and `bitCode`, longer than every row. -/
def rowEnc (v : BitVal W) : List UInt8 :=
  if RowShaped v then (HmRow.rowPrefix (v.1.val / 16)).data.toList ++ rowBytes v
  else List.replicate (128 + W) 0 ++ bitCode v

theorem bitCode_length (v : BitVal W) : (bitCode v).length = 2 * W + 1 := by
  simp only [bitCode, List.length_append, List.length_ofFn]
  omega

theorem bitCode_inj : Function.Injective (bitCode (W := W)) := by
  intro v w h
  obtain ⟨h1, h2⟩ := List.append_inj h (by simp)
  have e1 := List.ofFn_injective h1
  have e2 := List.ofFn_injective h2
  have hw : v.1 = w.1 := by
    have := congrFun e1 v.1
    simp only [↓reduceIte] at this
    by_contra hne
    rw [if_neg hne] at this
    exact absurd this (by decide)
  refine Prod.ext hw (funext fun i => ?_)
  have := congrFun e2 i
  cases hv : v.2 i <;> cases hw' : w.2 i <;> simp only [hv, hw'] at this <;> first | rfl | exact absurd this (by decide)

theorem rowEnc_row_length {v : BitVal W} (h : RowShaped v) : (rowEnc v).length = 128 + 2 * (v.1.val / 16) := by
  simp only [rowEnc, h, ↓reduceIte, List.length_append, rowPrefix_length, rowBytes, List.length_ofFn]

theorem rowEnc_other_length {v : BitVal W} (h : ¬ RowShaped v) : (rowEnc v).length = 128 + W + (2 * W + 1) := by
  simp only [rowEnc, h, ↓reduceIte, List.length_append, List.length_replicate, bitCode_length]

theorem get_of_rowBytes {v w : BitVal W} (hv : RowShaped v) (hk : v.1 = w.1) (h : rowBytes v = rowBytes w) (i : Fin W)
    (hi : i.val < v.1.val) : v.2 i = w.2 i := by
  have hj : i.val / 8 < 2 * (v.1.val / 16) := by have := hv.1; omega
  have hl : i.val / 8 < (rowBytes v).length := by simp only [rowBytes, List.length_ofFn]; exact hj
  have e := List.getElem_of_eq h hl
  simp only [rowBytes, List.getElem_ofFn] at e
  have := congrFun (byteOf_inj e) ⟨i.val % 8, Nat.mod_lt _ (by norm_num)⟩
  simp only [BitVal.get, show 8 * (i.val / 8) + i.val % 8 = i.val by omega, i.isLt, ↓reduceDIte] at this
  exact this

/-- **`rowEnc` is injective**: a row's length fixes its words, its bytes its bits; every other value is longer, and
`bitCode` is injective. -/
theorem rowEnc_inj : Function.Injective (rowEnc (W := W)) := by
  intro v w h
  have hv1 := v.1.isLt
  have hw1 := w.1.isLt
  by_cases hv : RowShaped v <;> by_cases hw : RowShaped w
  · have hl := congrArg List.length h
    rw [rowEnc_row_length hv, rowEnc_row_length hw] at hl
    have hk : v.1 = w.1 := Fin.ext (by have := hv.1; have := hw.1; omega)
    simp only [rowEnc, hv, hw, ↓reduceIte] at h
    obtain ⟨-, hb⟩ := List.append_inj h (by rw [rowPrefix_length, rowPrefix_length])
    refine Prod.ext hk (funext fun i => ?_)
    by_cases hi : i.val < v.1.val
    · exact get_of_rowBytes hv hk hb i hi
    · rw [hv.2 i (by omega), hw.2 i (by rw [← hk]; omega)]
  · have hl := congrArg List.length h
    rw [rowEnc_row_length hv, rowEnc_other_length hw] at hl
    omega
  · have hl := congrArg List.length h
    rw [rowEnc_other_length hv, rowEnc_row_length hw] at hl
    omega
  · simp only [rowEnc, hv, hw, ↓reduceIte] at h
    exact bitCode_inj (List.append_inj h rfl).2

/-- **`rowEnc` of a row of `16 k` bits**: `rowPrefix k` and the row's `2 k` bytes, byte `j`'s bit `t` its bit
`8 j + t`. -/
theorem rowEnc_row {K k : ℕ} (hK : K = 16 * k) (hKW : K ≤ W) (b : Fin K → Bool) :
    rowEnc (widthOf W K, (padBits b : Fin W → Bool)) =
      (HmRow.rowPrefix k).data.toList ++
        List.ofFn fun j : Fin (2 * k) => byteOf fun t => b ⟨8 * j.val + t.val, by omega⟩ := by
  have hw : (widthOf W K).val = K := by simp only [widthOf]; omega
  have hs : RowShaped (widthOf W K, (padBits b : Fin W → Bool)) :=
    ⟨by rw [hw, hK]; omega, fun i hi => by
      change (widthOf W K).val ≤ i.val at hi
      rw [hw] at hi
      simp only [padBits, show ¬ i.val < K by omega, ↓reduceDIte]⟩
  have hk : (widthOf W K).val / 16 = k := by rw [hw, hK]; omega
  simp only [rowEnc, hs, ↓reduceIte]
  rw [hk]
  congr 1
  refine List.ext_getElem (by simp [rowBytes, hk]) fun j h1 h2 => ?_
  simp only [rowBytes, List.getElem_ofFn]
  congr 1
  funext t
  have hj : 8 * j + t.val < K := by simp at h2; omega
  simp only [BitVal.get, show 8 * j + t.val < W by omega, ↓reduceDIte, padBits, hj]

/-! ## Under any 128-byte prefix -/

/-- **`rowEnc` under the first block `pre w`** of a row of `w` words, for any `pre` of 128 bytes: a `sha512/row/v2`
row's (`Hidden.rowPre`) as well as `sha512/row/v1`'s. -/
def rowEncP (pre : ℕ → ByteArray) (v : BitVal W) : List UInt8 :=
  if RowShaped v then (pre (v.1.val / 16)).data.toList ++ rowBytes v
  else List.replicate (128 + W) 0 ++ bitCode v

theorem rowEncP_inj {pre : ℕ → ByteArray} (hpre : ∀ w, (pre w).size = 128) : Function.Injective (rowEncP pre (W := W)) := by
  have hl : ∀ w, (pre w).data.toList.length = 128 := fun w => by rw [Array.length_toList]; exact hpre w
  have hrow : ∀ {v : BitVal W}, RowShaped v → (rowEncP pre v).length = 128 + 2 * (v.1.val / 16) := fun h => by
    simp only [rowEncP, h, ↓reduceIte, List.length_append, hl, rowBytes, List.length_ofFn]
  have hoth : ∀ {v : BitVal W}, ¬ RowShaped v → (rowEncP pre v).length = 128 + W + (2 * W + 1) := fun h => by
    simp only [rowEncP, h, ↓reduceIte, List.length_append, List.length_replicate, bitCode_length]
  intro v w h
  have hv1 := v.1.isLt
  have hw1 := w.1.isLt
  by_cases hv : RowShaped v <;> by_cases hw : RowShaped w
  · have hk : v.1 = w.1 := by
      have e := congrArg List.length h
      rw [hrow hv, hrow hw] at e
      exact Fin.ext (by have := hv.1; have := hw.1; omega)
    simp only [rowEncP, hv, hw, ↓reduceIte] at h
    obtain ⟨-, hb⟩ := List.append_inj h (by rw [hl, hl])
    refine Prod.ext hk (funext fun i => ?_)
    by_cases hi : i.val < v.1.val
    · exact get_of_rowBytes hv hk hb i hi
    · rw [hv.2 i (by omega), hw.2 i (by rw [← hk]; omega)]
  · have e := congrArg List.length h
    rw [hrow hv, hoth hw] at e
    omega
  · have e := congrArg List.length h
    rw [hoth hv, hrow hw] at e
    omega
  · simp only [rowEncP, hv, hw, ↓reduceIte] at h
    exact bitCode_inj (List.append_inj h rfl).2

/-- **`rowEncP` of a row of `16 k` bits**: `pre k` and the row's `2 k` bytes. -/
theorem rowEncP_row (pre : ℕ → ByteArray) {K k : ℕ} (hK : K = 16 * k) (hKW : K ≤ W) (b : Fin K → Bool) :
    rowEncP pre (widthOf W K, (padBits b : Fin W → Bool)) =
      (pre k).data.toList ++ List.ofFn fun j : Fin (2 * k) => byteOf fun t => b ⟨8 * j.val + t.val, by omega⟩ := by
  have hw : (widthOf W K).val = K := by simp only [widthOf]; omega
  have hs : RowShaped (widthOf W K, (padBits b : Fin W → Bool)) :=
    ⟨by rw [hw, hK]; omega, fun i hi => by
      change (widthOf W K).val ≤ i.val at hi
      rw [hw] at hi
      simp only [padBits, show ¬ i.val < K by omega, ↓reduceDIte]⟩
  have hk : (widthOf W K).val / 16 = k := by rw [hw, hK]; omega
  simp only [rowEncP, hs, ↓reduceIte]
  rw [hk]
  congr 1
  refine List.ext_getElem (by simp [rowBytes, hk]) fun j h1 h2 => ?_
  simp only [rowBytes, List.getElem_ofFn]
  congr 1
  funext t
  have hj : 8 * j + t.val < K := by simp at h2; omega
  simp only [BitVal.get, show 8 * j + t.val < W by omega, ↓reduceDIte, padBits, hj]

/-! ## At `keyProg`'s rows -/

open Placed

variable {c : Flock.Circuit} {hU : Built c.unit} {n : ℕ} {rowW : Fin c.ports.size → ℕ} {Reg : Type} {mPts : ℕ}
  {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  {si : Placed.Sites plan tab}
  (hparse : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
    HmRow.parse (si.acc S R u h).tags (si.acc S R u h).cf (si.acc S R u h).tables = .ok c)
  (outs : Finset (Fin (keyN c hU n))) (saltCol : Flock.Circuit → ℕ → ℕ → ℕ → ℕ → ℕ) (popOf : Reg → Public)
  (outBit : Reg → Fin (keyN c hU n) → Bool)

/-- **The bytes `sha512/row/v1` hashes for a row at `keyProg`'s places**: at a drawn unit `u`'s site, the value a
message holds at row port `q` of unit `u'`, `16 · words` bits (`rowW q`), has `rowEnc` the row prefix of `words` and the
`2 · words` bytes the witness holds at the unit's VU's message columns (`Typed.msgCol`) in its block: [e2e-hm]'s
`port_commit` row, with the block's bits at the site. -/
theorem keyRow_enc (encO : BitVal (Finset.univ.sup rowW + 1) → List UInt8) {S : Finset (Fin n)} {R : Reg} {u : Fin n}
    (hu : u ∈ S) (M : Binding.Msg plan tab S R u) (u' : Fin n) (q : Fin c.ports.size) {words : ℕ}
    (hrw : rowW q = 16 * words) :
    rowEnc ((keyLayout rowW hparse outs saltCol popOf outBit encO).row S R u M (.inl (u', q))) =
      (HmRow.rowPrefix words).data.toList ++
        List.ofFn fun j : Fin (2 * words) => byteOf fun t => bit (Binding.wit plan tab S R u M
          (siteAddr (si.acc S R u hu) (si.o S R u hu)
            (Typed.msgCol (si.acc S R u hu).st.c (si.vu S R u hu) q.val j.val t.val))) := by
  have hle : rowW q ≤ Finset.univ.sup rowW + 1 :=
    (Finset.le_sup (Finset.mem_univ q)).trans (Nat.le_succ _)
  show rowEnc (W := Finset.univ.sup rowW + 1) (widthOf (Finset.univ.sup rowW + 1) (rowW q),
    padBits (bitsAt (Binding.wit plan tab S R u M) fun k : Fin (rowW q) => keyRowAddr si S R u q k.val)) = _
  rw [rowEnc_row hrw hle]
  congr 1
  refine List.ext_getElem (by simp) fun j h1 h2 => ?_
  simp only [List.getElem_ofFn]
  congr 1
  funext t
  have hj : (8 * j + t.val) / 8 = j := by omega
  have ht : (8 * j + t.val) % 8 = t.val := by omega
  show bit (Binding.wit plan tab S R u M (keyRowAddr si S R u q (8 * j + t.val))) = _
  simp only [keyRowAddr, hu, ↓reduceDIte, hj, ht]

variable {outs popOf} (rowW) in
/-- **`rs` at `keyProg`'s places, with the row bytes `sha512/row/v1` hashes**: `keyRowsOut` with `enc := rowEnc`
(`keyRow_enc`) and `bitCode` for the other commit strings. -/
noncomputable def keyRowsEnc (ho : c.outNet = c.unitNet)
    (hsite : ∀ S R u (h : u ∈ S) M, Binding.Sat plan tab S R u M →
      BcSite (si.acc S R u h) (popOf R) (si.o S R u h) (si.vu S R u h) u.val)
    (hkeys : KeysRow c hU rowW) (houts : OutsOut c hU n outs)
    (hc1 : ((keyProg c hU n).partition outs).ConstCols plan tab (keyPlace hparse outs)
      {oneG (c := c) (hU := hU) (n := n)})
    (hz1 : ((keyProg c hU n).partition outs).ZeroCols plan tab (keyPlace hparse outs)
      {zeroG (c := c) (hU := hU) (n := n)}) :
    Binding.HmRows (keyLayout rowW hparse outs saltCol popOf (outBitOf c hU n outs popOf) bitCode) :=
  keyRowsOut (outs := outs) (popOf := popOf) (hparse := hparse) (rowW := rowW) (saltCol := saltCol) rowEnc
    rowEnc_inj bitCode bitCode_inj ho hsite hkeys houts hc1 hz1

/-- `keyRowsEnc`'s `enc` is `rowEnc`. -/
theorem keyRowsEnc_enc (ho : c.outNet = c.unitNet)
    (hsite : ∀ S R u (h : u ∈ S) M, Binding.Sat plan tab S R u M →
      BcSite (si.acc S R u h) (popOf R) (si.o S R u h) (si.vu S R u h) u.val)
    (hkeys : KeysRow c hU rowW) (houts : OutsOut c hU n outs)
    (hc1 : ((keyProg c hU n).partition outs).ConstCols plan tab (keyPlace hparse outs)
      {oneG (c := c) (hU := hU) (n := n)})
    (hz1 : ((keyProg c hU n).partition outs).ZeroCols plan tab (keyPlace hparse outs)
      {zeroG (c := c) (hU := hU) (n := n)}) :
    (keyRowsEnc (rowW := rowW) (hparse := hparse) (saltCol := saltCol) ho hsite hkeys houts hc1 hz1).enc =
      rowEnc := rfl

end FlockSoundness.Discharge.Layout
