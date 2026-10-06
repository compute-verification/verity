import Proofs.Flock.Soundness.Discharge.Layout.Keyed

set_option Elab.async false

/-!
# The bytes a row's digest hashes, under any 128-byte prefix

`Layout.rowEnc` with `HmRow.rowPrefix` replaced by any prefix `pre` of 128 bytes per word count (`rowEncP`), so for a
`sha512/row/v2` row of `16 · words` bits, `rowPrefixV2 (16 · words)`: `rowEncP_inj`, `rowEncP_row`, `keyRow_encP`, and
`keyRowsEncP` … `keyedRowsOutsP`, `Keyed.lean`'s `rs` with `enc := rowEncP pre`.
-/

namespace FlockSoundness.Discharge.Layout

open Flock Binding

variable {W : ℕ}

/-- **The bytes a row's digest hashes for a value under `pre`**: a row of `w` words is `pre w` and its `2 w` bytes; any
other value is `128 + W` zero bytes and `bitCode`, longer than every row. -/
def rowEncP (pre : ℕ → ByteArray) (v : BitVal W) : List UInt8 :=
  if RowShaped v then (pre (v.1.val / 16)).data.toList ++ rowBytes v
  else List.replicate (128 + W) 0 ++ bitCode v

theorem rowEncP_other_length (pre : ℕ → ByteArray) {v : BitVal W} (h : ¬ RowShaped v) :
    (rowEncP pre v).length = 128 + W + (2 * W + 1) := by
  simp only [rowEncP, h, ↓reduceIte, List.length_append, List.length_replicate, bitCode_length]

section Pre

variable {pre : ℕ → ByteArray} (hpre : ∀ w, (pre w).size = 128)
include hpre

theorem pre_length (w : ℕ) : (pre w).data.toList.length = 128 := by
  rw [Array.length_toList]
  exact hpre w

theorem rowEncP_row_length {v : BitVal W} (h : RowShaped v) :
    (rowEncP pre v).length = 128 + 2 * (v.1.val / 16) := by
  simp only [rowEncP, h, ↓reduceIte, List.length_append, pre_length hpre, rowBytes, List.length_ofFn]

/-- **`rowEncP` is injective**: `rowEnc_inj`'s argument, which reads the prefix only through its 128 bytes. -/
theorem rowEncP_inj : Function.Injective (rowEncP (W := W) pre) := by
  intro v w h
  have hv1 := v.1.isLt
  have hw1 := w.1.isLt
  by_cases hv : RowShaped v <;> by_cases hw : RowShaped w
  · have hl := congrArg List.length h
    rw [rowEncP_row_length hpre hv, rowEncP_row_length hpre hw] at hl
    have hk : v.1 = w.1 := Fin.ext (by have := hv.1; have := hw.1; omega)
    simp only [rowEncP, hv, hw, ↓reduceIte] at h
    obtain ⟨-, hb⟩ := List.append_inj h (by rw [pre_length hpre, pre_length hpre])
    refine Prod.ext hk (funext fun i => ?_)
    by_cases hi : i.val < v.1.val
    · exact get_of_rowBytes hv hk hb i hi
    · rw [hv.2 i (by omega), hw.2 i (by rw [← hk]; omega)]
  · have hl := congrArg List.length h
    rw [rowEncP_row_length hpre hv, rowEncP_other_length pre hw] at hl
    omega
  · have hl := congrArg List.length h
    rw [rowEncP_other_length pre hv, rowEncP_row_length hpre hw] at hl
    omega
  · simp only [rowEncP, hv, hw, ↓reduceIte] at h
    exact bitCode_inj (List.append_inj h rfl).2

end Pre

/-- **`rowEncP` of a row of `16 k` bits**: `pre k` and the row's `2 k` bytes, byte `j`'s bit `t` its bit `8 j + t`. -/
theorem rowEncP_row (pre : ℕ → ByteArray) {K k : ℕ} (hK : K = 16 * k) (hKW : K ≤ W) (b : Fin K → Bool) :
    rowEncP pre (widthOf W K, (padBits b : Fin W → Bool)) =
      (pre k).data.toList ++
        List.ofFn fun j : Fin (2 * k) => byteOf fun t => b ⟨8 * j.val + t.val, by omega⟩ := by
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

/-- **`keyRow_enc` under `pre`**: at a drawn unit `u`'s site, the value a message holds at row port `q` of unit `u'`,
`16 · words` bits (`rowW q`), has `rowEncP pre` the prefix `pre words` and the `2 · words` bytes the witness holds at
the unit's VU's message columns. -/
theorem keyRow_encP (pre : ℕ → ByteArray) (encO : BitVal (Finset.univ.sup rowW + 1) → List UInt8) {S : Finset (Fin n)}
    {R : Reg} {u : Fin n} (hu : u ∈ S) (M : Binding.Msg plan tab S R u) (u' : Fin n) (q : Fin c.ports.size)
    {words : ℕ} (hrw : rowW q = 16 * words) :
    rowEncP pre ((keyLayout rowW hparse outs saltCol popOf outBit encO).row S R u M (.inl (u', q))) =
      (pre words).data.toList ++
        List.ofFn fun j : Fin (2 * words) => byteOf fun t => bit (Binding.wit plan tab S R u M
          (siteAddr (si.acc S R u hu) (si.o S R u hu)
            (Typed.msgCol (si.acc S R u hu).st.c (si.vu S R u hu) q.val j.val t.val))) := by
  have hle : rowW q ≤ Finset.univ.sup rowW + 1 :=
    (Finset.le_sup (Finset.mem_univ q)).trans (Nat.le_succ _)
  show rowEncP (W := Finset.univ.sup rowW + 1) pre (widthOf (Finset.univ.sup rowW + 1) (rowW q),
    padBits (bitsAt (Binding.wit plan tab S R u M) fun k : Fin (rowW q) => keyRowAddr si S R u q k.val)) = _
  rw [rowEncP_row pre hrw hle]
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
/-- **`keyRowsEnc` under `pre`**: `keyRowsOut` with `enc := rowEncP pre` and `bitCode` for the other commit strings. -/
noncomputable def keyRowsEncP (pre : ℕ → ByteArray) (hpre : ∀ w, (pre w).size = 128) (ho : c.outNet = c.unitNet)
    (hsite : ∀ S R u (h : u ∈ S) M, Binding.Sat plan tab S R u M →
      BcSite (si.acc S R u h) (popOf R) (si.o S R u h) (si.vu S R u h) u.val)
    (hkeys : KeysRow c hU rowW) (houts : OutsOut c hU n outs)
    (hc1 : ((keyProg c hU n).partition outs).ConstCols plan tab (keyPlace hparse outs)
      {oneG (c := c) (hU := hU) (n := n)})
    (hz1 : ((keyProg c hU n).partition outs).ZeroCols plan tab (keyPlace hparse outs)
      {zeroG (c := c) (hU := hU) (n := n)}) :
    Binding.HmRows (keyLayout rowW hparse outs saltCol popOf (outBitOf c hU n outs popOf) bitCode) :=
  keyRowsOut (outs := outs) (popOf := popOf) (hparse := hparse) (rowW := rowW) (saltCol := saltCol) (rowEncP pre)
    (rowEncP_inj hpre) bitCode bitCode_inj ho hsite hkeys houts hc1 hz1

variable {popOf} in
/-- **`keyedRowsOuts` under `pre`**: `rs` at the production program's places, at row widths `16 · words`, with `KeysRow`
from the parse and the returned outputs' bits for the program's outputs, and `enc := rowEncP pre`. -/
noncomputable def keyedRowsOutsP (pre : ℕ → ByteArray) (hpre : ∀ w, (pre w).size = 128) {tags : Tags}
    {bytes : ByteArray} {tables : List (String × ByteArray)}
    (hc : HmRow.parse tags bytes tables = .ok c) (h1 : c.unit.inGroups.size = 1) (hpr : PortReads c hU)
    (ho : c.outNet = c.unitNet)
    (hsite : ∀ S R u (h : u ∈ S) M, Binding.Sat plan tab S R u M →
      BcSite (si.acc S R u h) (popOf R) (si.o S R u h) (si.vu S R u h) u.val) :
    Binding.HmRows (keyedLayout hparse (outsOf c hU n) saltCol popOf) :=
  keyRowsEncP (portRowW c) (hparse := hparse) (saltCol := saltCol) pre hpre ho hsite (keysRow_of_parse hc h1 hpr)
    (outsOut_outsOf c hU n) (Const.constCols_keyProg hU hparse (outsOf c hU n))
    (Zero.zeroCols_keyProg hU hparse (outsOf c hU n))

/-- `keyedRowsOutsP`' `enc` is `rowEncP pre`. -/
theorem keyedRowsOutsP_enc (pre : ℕ → ByteArray) (hpre : ∀ w, (pre w).size = 128) {tags : Tags}
    {bytes : ByteArray} {tables : List (String × ByteArray)}
    (hc : HmRow.parse tags bytes tables = .ok c) (h1 : c.unit.inGroups.size = 1) (hpr : PortReads c hU)
    (ho : c.outNet = c.unitNet)
    (hsite : ∀ S R u (h : u ∈ S) M, Binding.Sat plan tab S R u M →
      BcSite (si.acc S R u h) (popOf R) (si.o S R u h) (si.vu S R u h) u.val) :
    (keyedRowsOutsP hparse saltCol pre hpre hc h1 hpr ho hsite).enc = rowEncP pre := rfl

end FlockSoundness.Discharge.Layout
