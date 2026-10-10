import Proofs.Flock.Soundness.Discharge.Hidden.Wide.SegCommit
import Proofs.Flock.Soundness.Discharge.Hidden.Wide.SegCopies

/-!
# A segmented row's commit string at its columns

`port_commit_seg` is `port_commit_rowsP` for a `sha512/row-seg/v1` row of `l` bits: when its compressions' and its
`hm96` slot's columns hold their nets' rows, each chain start's `cv` the midstate Δ pins (`segMidstate`, then the outer
`midstateOf`), every other `cv` the previous compression's output, and each message bit the constant or the digest bit
`HmRow.msgBit` names, the bytes at `bcCol` are the commit string of `RowSeg.enc l` of the row's bytes at `Typed.msgCol`.
Segment `j`'s compressions are `firstOf j … firstOf (j + 1) - 1` (`seg_digest`), the outer message's from
`firstOf s` on, ending in the `hm96` slot (`seg_commit`).
-/

set_option Elab.async false

namespace FlockSoundness.Discharge.Hidden

open Flock.HmNets hiding and
open Flock Hm Refine
open Binding (digBytes)
open Binding.RowSeg (count segBits segBytes segMsg digests enc segShape_get size_segShape)
open SegShape

namespace Any

/-! ## The slots' rows -/

section Slots

variable (c : Circuit) (w : ℕ → Bool) (g p : ℕ)

/-- Compression `k`'s variables at its `sha512x3` slot. -/
def zsOf (k : ℕ) (x : Var) : Bool := w (shaBase c g p k + col (SHA_IN.foldl (· + ·) 0) (inRowsOf SHA_IN) x)

variable {c w g p} {n : ℕ} (hck : HmNets.check c = .ok ()) (hsub : ∀ k < n, shaSub c p k < 3)
  (hsha : ∀ k < n, LaidRows (shaLaid ()) 8192 fun j => w (shaBase c g p k + j))
include hsha

theorem shaL {k : ℕ} (hk : k < n) :
    LaidRows (layout (compression3.run St.empty).2 SHA_IN SHA_OUT (compression3.run St.empty).1.2.flatten.toArray)
      (inRowsOf SHA_IN) fun j => w (shaBase c g p k + j) := by
  have := hsha k hk
  rw [shaLaid_eq] at this
  rw [inRows_SHA_IN]
  exact this

theorem zs_rows {k : ℕ} (hk : k < n) : stHolds (zsOf c w g p k) (compression3.run St.empty).2 := by
  have hgs : Good (SHA_IN.foldl (· + ·) 0) (compression3.run St.empty).2 := by rw [sum_SHA_IN]; exact good_compression3
  exact laid_rows _ _ _ _ _ hgs (shaL hsha hk)

omit hsha in
theorem zs_in {k : ℕ} {v : ℕ} (hv : v < 4608) : zsOf c w g p k v = w (shaBase c g p k + v) := by
  show w (shaBase c g p k + col _ _ v) = _
  rw [col_input (by rw [sum_SHA_IN]; exact hv)]

include hck hsub in
theorem zs_out {k : ℕ} (hk : k < n) {t : ℕ} (ht : t < 512) :
    w (shaBase c g p k + shaOut c (shaSub c p k) + t) =
      Hm.bit (zsOf c w g p k) ((compression3.run St.empty).1.2.getD (shaSub c p k) []) t := by
  have hs := hsub k hk
  obtain ⟨-, hl3, -⟩ := compression3_spec (zsOf c w g p k) (zs_rows hsha hk)
  have hlens := compression3_lens (zsOf c w g p k) (zs_rows hsha hk)
  have hflat : (compression3.run St.empty).1.2.flatten.length = 1536 := by
    rw [List.length_flatten, List.map_congr_left (g := fun _ => 512) hlens, List.map_const', List.sum_replicate, hl3]
    rfl
  have hj : 512 * shaSub c p k + t < (compression3.run St.empty).1.2.flatten.toArray.size := by
    rw [List.size_toArray, hflat]; omega
  have := laid_outs (compression3.run St.empty).2 SHA_IN SHA_OUT _ (fun j => w (shaBase c g p k + j))
    (by rw [List.size_toArray, hflat]; decide) (shaL hsha hk) _ hj
  rw [shaOut_eq hck hs, Nat.add_assoc, Nat.add_assoc, this, List.getElem_toArray,
    ← bit_of_lt (by rw [hflat]; omega)]
  exact bit_flatten_of_len (zsOf c w g p k) hlens ht

end Slots

/-! ## Bits -/

theorem blockBit_wordBit {o t : ℕ} (ho : o < 128) (ht : t < 8) : HmRow.blockBit (wordBit (8 * o + t)) = (o, t) := by
  unfold HmRow.blockBit wordBit
  ext <;> dsimp only <;> omega

theorem wordBit_lt {o t : ℕ} (ho : o < 128) (ht : t < 8) : wordBit (8 * o + t) < 1024 := by
  unfold wordBit; omega

/-- Bit `t` of byte `b` of a padding tail, read as `HmRow.msgBit` reads it. -/
theorem tail_bit (tl : ByteArray) (b t : ℕ) (ht : t < 8) :
    ((tl.get! b).toNat >>> t % 2 == 1) = bitL tl.data.toList (8 * b + t) := by
  unfold bitL
  rw [show (8 * b + t) / 8 = b by omega, show (8 * b + t) % 8 = t by omega, get!_eq_getD]
  generalize (tl.data.toList.getD b 0).toNat = x
  rw [Nat.testBit, Nat.one_and_eq_mod_two]
  rcases Nat.mod_two_eq_zero_or_one (x >>> t) with h | h <;> simp [h]

/-- Segment `j`'s bit `8 i + t` inside its bytes is the row's bit `8 (1024 j + i) + t`. -/
theorem bitL_segBytes {l : ℕ} {row : List UInt8} (hrow : row.length = (l + 7) / 8) {j : ℕ} (hj : j < count l)
    (tl : List UInt8) {i t : ℕ} (hi : i < (segBits l j + 7) / 8) (ht : t < 8) :
    bitL (segBytes l row j ++ tl) (8 * i + t) = bitL row (8 * (1024 * j + i) + t) := by
  have hd := length_segBytes hrow hj
  unfold bitL
  rw [show (8 * i + t) / 8 = i by omega, show (8 * (1024 * j + i) + t) / 8 = 1024 * j + i by omega,
    show (8 * i + t) % 8 = t by omega, show (8 * (1024 * j + i) + t) % 8 = t by omega,
    List.getD_append _ _ _ _ (by rw [hd]; exact hi), segBytes, List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD,
    List.getElem?_take_of_lt hi, List.getElem?_drop]

/-- Segment `j`'s bit past its bytes is its tail's. -/
theorem bitL_segTail {l : ℕ} {row : List UInt8} (hrow : row.length = (l + 7) / 8) {j : ℕ} (hj : j < count l)
    (tl : List UInt8) {i t : ℕ} (hi : (segBits l j + 7) / 8 ≤ i) (ht : t < 8) :
    bitL (segBytes l row j ++ tl) (8 * i + t) = bitL tl (8 * (i - (segBits l j + 7) / 8) + t) := by
  have hd := length_segBytes hrow hj
  unfold bitL
  rw [show (8 * i + t) / 8 = i by omega, show (8 * (i - (segBits l j + 7) / 8) + t) / 8 = i - (segBits l j + 7) / 8 by omega,
    show (8 * i + t) % 8 = t by omega, show (8 * (i - (segBits l j + 7) / 8) + t) % 8 = t by omega,
    List.getD_append_right _ _ _ _ (by rw [hd]; exact hi), hd]

/-! ## The port -/

variable {c : Circuit} (R : RowRanges c)
include R

omit R in
/-- A segmented row's byte `byte`, in a compression, is at its compression's message column. -/
theorem msgCol_seg {segs : Array (ℕ × ℕ)} {g p byte t : ℕ} (hp : p < c.ports.size) (hs : c.ports[p]!.segs = some segs)
    (hk : firstOf segs (byte / 1024) + byte % 1024 / 128 < c.ports[p]!.chunks) (ht : t < 8) :
    Typed.msgCol c g p byte t =
      segMsgCol c g p (firstOf segs (byte / 1024) + byte % 1024 / 128) + wordBit (8 * (byte % 128) + t) := by
  have hk' : firstOf segs (byte / 1024) + byte % 1024 / 128 < ((HmRow.rowSlots c.ports).2[p]!).size := by
    rw [rowSlots_size hp]; exact hk
  simp only [Typed.msgCol, byteAt_seg hs, hk', ↓reduceIte, segMsgCol, segCols]
  unfold wordBit
  omega

/-- **A segmented row's `b ‖ c` at its columns**: a row of `l` bits shaped `HmRow.segShape l`, whose segments' bytes
reach their last blocks (`SegFits`), its slots holding their nets' rows, its chain starts' `cv`s the midstates
(`hcvS`), its other `cv`s the previous compressions' outputs (`hcvC`) and its message bits the constants (`hmC`) and
digest bits (`hmD`) `HmRow.msgBit` names: the bytes at `bcCol` are the commit string of `RowSeg.enc l` of the row's
`(l + 7) / 8` bytes at `Typed.msgCol`, under the salt at `saltCol`. -/
theorem port_commit_seg (w : ℕ → Bool) {g p l : ℕ} (hp : p < c.ports.size)
    (hs : c.ports[p]!.segs = some (HmRow.segShape l).1) (hbits : c.ports[p]!.bits = l)
    (hn : c.ports[p]!.chunks = firstOf (HmRow.segShape l).1 (count l) + outerComps l) (hfit : SegFits l)
    (hsha : ∀ k < c.ports[p]!.chunks, LaidRows (shaLaid ()) 8192 fun j => w (shaBase c g p k + j))
    (hhm : LaidRows (hmLaid ()) 4096 fun j => w (c.slotCol c.hmNet g p + j))
    (hcvS : ∀ j ≤ count l, ∀ t < 512, w (segCvCol c g p (firstOf (HmRow.segShape l).1 j) + t) =
      wordsBit (if j < count l then HmRow.segMidstate c.ports[p]! j else HmRow.midstateOf c.ports[p]!) t)
    (hcvC : ∀ k ≤ c.ports[p]!.chunks, 0 < k → HmRow.chainStart c.ports[p]! k = none → ∀ t < 512,
      w (segCvCol c g p k + t) = w (shaBase c g p (k - 1) + shaOut c (shaSub c p (k - 1)) + t))
    (hmC : ∀ k ≤ c.ports[p]!.chunks, ∀ t < 1024, ∀ v, HmRow.msgBit c.ports[p]! k t = .const v →
      w (segMsgCol c g p k + t) = v)
    (hmD : ∀ k ≤ c.ports[p]!.chunks, ∀ t < 1024, ∀ kk col, HmRow.msgBit c.ports[p]! k t = .digest kk col →
      w (segMsgCol c g p k + t) = w (digestSrc c g p kk col)) :
    bytesOf w 128 (bcCol c g p) = Binding.commitString
      (enc l (bytesOf w ((l + 7) / 8) (Typed.msgCol c g p))) (bytesOf w 192 (saltCol c g p)) := by
  have hck := R.rowNets
  set S := (HmRow.segShape l).1 with hS
  set N := c.ports[p]!.chunks with hN
  set F := firstOf S (count l) with hF
  set row := bytesOf w ((l + 7) / 8) (Typed.msgCol c g p) with hrowdef
  have hrow : row.length = (l + 7) / 8 := length_bytesOf _ _ _
  have hsS : S.size = count l := size_segShape l
  have hpos : ∀ j < S.size, 0 < S[j]!.2 := segShape_pos l
  have hsub : ∀ k < N, shaSub c p k < 3 := fun k hk => (rowSlot_lt hp hk).2
  have hsin := shaIn_eq hck
  have hhin := hmIn_eq hck
  have hfs : ∀ j < count l, firstOf S (j + 1) = firstOf S j + segComps (segBits l j) := fun j hj => by
    rw [firstOf_succ (by rw [hsS]; exact hj), segShape_get hj]; rfl
  have hmono : ∀ {i j : ℕ}, i ≤ j → j ≤ count l → firstOf S i ≤ firstOf S j := fun hij hj =>
    firstOf_mono hpos hij (by rw [hsS]; exact hj)
  have hc0 : ∀ j < count l, 0 < segComps (segBits l j) := fun j hj => by
    have := hpos j (by rw [hsS]; exact hj)
    rw [segShape_get hj] at this
    exact this
  have hsegN : ∀ j < count l, firstOf S (j + 1) ≤ N := fun j hj => by
    have := hmono (show j + 1 ≤ count l by omega) le_rfl
    omega
  have hgetD : ∀ j < count l, S.getD j default = (segBits l j, segComps (segBits l j)) := fun j hj => by
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem (by rw [hsS]; exact hj), Option.getD_some,
      ← getElem!_pos S j (by rw [hsS]; exact hj), segShape_get hj]
    rfl
  -- a compression strictly inside a chain is no chain start
  have hnone : ∀ k, (∀ j ≤ count l, k ≠ firstOf S j) → HmRow.chainStart c.ports[p]! k = none := fun k hk =>
    chainStart_none hs fun j hj => hk j (by rw [← hsS]; exact hj)
  have hin_seg : ∀ j < count l, ∀ k, firstOf S j < k → k < firstOf S (j + 1) →
      HmRow.chainStart c.ports[p]! k = none := by
    intro j hj k h1 h2
    refine hnone k fun j' hj' he => ?_
    subst he
    rcases Nat.lt_or_ge j j' with h | h
    · have := hmono (show j + 1 ≤ j' by omega) hj'
      omega
    · have := hmono h (Nat.le_of_lt hj)
      omega
  have hin_out : ∀ k, F < k → HmRow.chainStart c.ports[p]! k = none := by
    intro k hk
    refine hnone k fun j' hj' he => ?_
    subst he
    have := hmono hj' le_rfl
    omega
  have hzcv : ∀ k : ℕ, k < N → ∀ q : ℕ, q < 512 → zsOf c w g p k (1536 * shaSub c p k + q) = w (segCvCol c g p k + q) := by
    intro k hk q hq
    have := hsub k hk
    rw [zs_in (c := c) (by omega), (segCols_sha R hp hk q).1, shaPos]
    exact congrArg w (by omega)
  have hzmsg : ∀ k : ℕ, k < N → ∀ q : ℕ, q < 1024 →
      zsOf c w g p k (1536 * shaSub c p k + 512 + q) = w (segMsgCol c g p k + q) := by
    intro k hk q hq
    have := hsub k hk
    rw [zs_in (c := c) (by omega), (segCols_sha R hp hk q).2, shaPos]
    exact congrArg w (by omega)
  have hout := fun {k : ℕ} (hk : k < N) {t : ℕ} (ht : t < 512) => zs_out hck hsub hsha hk ht
  have hdig : ∀ kk < N, ∀ col < 512, w (digestSrc c g p kk col) =
      Hm.bit (zsOf c w g p kk) ((compression3.run St.empty).1.2.getD (shaSub c p kk) []) col := by
    intro kk hkk col hcol
    rw [← hout hkk hcol]
    unfold digestSrc shaBase shaOut shaSub
    rw [Array.getD_eq_getD_getElem?, getElem!_def]
    rfl
  -- each segment's chain
  have hD : ∀ j < count l, ∀ t < 512,
      Hm.bit (zsOf c w g p (firstOf S j + (segComps (segBits l j) - 1)))
        ((compression3.run St.empty).1.2.getD (shaSub c p (firstOf S j + (segComps (segBits l j) - 1))) [])
        (wordBit t) = bitL (digBytes (H512 (segMsg l row j))) t := by
    intro j hj
    have hfj := hfs j hj
    have hNj := hsegN j hj
    have hc := hc0 j hj
    refine seg_digest hrow hj (hfit j hj) (fun k' => zsOf c w g p (firstOf S j + k'))
      (fun k' => shaSub c p (firstOf S j + k')) (fun k' hk' => hsub _ (by omega))
      (fun k' hk' => zs_rows hsha (by omega)) (fun q hq => ?_) (fun k' hk' q hq => ?_) (fun k' hk' o ho t ht => ?_)
    · try dsimp only
      rw [Nat.add_zero, hzcv _ (by omega) q hq, hcvS j (Nat.le_of_lt hj) q hq, if_pos hj]
      simp only [HmRow.segMidstate, hs, Option.getD_some, hsS, hgetD j hj]
    · try dsimp only
      rw [hzcv _ (by omega) q hq, show firstOf S j + (k' + 1) = firstOf S j + k' + 1 by omega,
        hcvC _ (by omega) (by omega) (hin_seg j hj _ (by omega) (by omega)) q hq, Nat.add_sub_cancel,
        hout (by omega) hq]
    · try dsimp only
      have hwb := wordBit_lt ho ht
      rw [hzmsg _ (by omega) _ hwb]
      have hk1 : firstOf S j ≤ firstOf S j + k' := Nat.le_add_right _ _
      have hk2 : firstOf S j + k' < firstOf S (j + 1) := by omega
      have hmb := msgBit_seg hs (by rw [hsS]; exact hj) hk1 hk2 (wordBit (8 * o + t))
      rw [blockBit_wordBit ho ht] at hmb
      dsimp only at hmb
      have hm1 : S[j]!.1 = segBits l j := by rw [segShape_get hj]
      rw [hm1] at hmb
      by_cases hi : 128 * k' + o < (segBits l j + 7) / 8
      · rw [bitL_segBytes hrow hj _ hi ht, bitL_bytesOf _ _ (by
          have : segBits l j ≤ l - 8192 * j := min_le_left _ _
          have : 8192 * j < l := by unfold count HmRow.ROW_SEG_BITS at hj; omega
          omega)]
        rw [show (8 * (1024 * j + (128 * k' + o)) + t) / 8 = 1024 * j + (128 * k' + o) by omega,
          show (8 * (1024 * j + (128 * k' + o)) + t) % 8 = t by omega]
        have hc9 : segComps (segBits l j) ≤ 9 := by
          have : segBits l j ≤ 8192 := min_le_right _ _
          unfold segComps; omega
        have hlt : 128 * k' + o < 1024 := by
          have : segBits l j ≤ 8192 := min_le_right _ _
          omega
        rw [msgCol_seg hp hs (by
            rw [show (1024 * j + (128 * k' + o)) / 1024 = j by omega,
              show (1024 * j + (128 * k' + o)) % 1024 / 128 = k' by omega]
            omega) ht,
          show (1024 * j + (128 * k' + o)) / 1024 = j by omega,
          show (1024 * j + (128 * k' + o)) % 1024 / 128 = k' by omega,
          show (1024 * j + (128 * k' + o)) % 128 = o by omega]
      · have hv : HmRow.msgBit c.ports[p]! (firstOf S j + k') (wordBit (8 * o + t)) =
            .const (((HmRow.segTail c.ports[p]! j).get! (128 * k' + o - (segBits l j + 7) / 8)).toNat >>> t % 2 == 1) := by
          rw [hmb, segVal, Nat.add_sub_cancel_left, if_neg (by omega), decide_eq_true (by omega), Bool.true_and]
        rw [hmC _ (by omega) _ (wordBit_lt ho ht) _ hv, bitL_segTail hrow hj _ (by omega) ht, tail_bit _ _ _ ht]
        simp only [HmRow.segTail, hs, Option.getD_some, hgetD j hj]
  -- the outer chain
  have hdigOut : ∀ k ≤ N, F ≤ k → ∀ o < 128, ∀ t < 8, 128 * (k - F) + o < 64 * count l →
      w (segMsgCol c g p k + wordBit (8 * o + t)) =
        Hm.bit (zsOf c w g p (firstOf S ((128 * (k - F) + o) / 64) + (segComps (segBits l ((128 * (k - F) + o) / 64)) - 1)))
          ((compression3.run St.empty).1.2.getD
            (shaSub c p (firstOf S ((128 * (k - F) + o) / 64) + (segComps (segBits l ((128 * (k - F) + o) / 64)) - 1))) [])
          (wordBit (8 * ((128 * (k - F) + o) % 64) + t)) := by
    intro k hk hFk o ho t ht ha
    have hmb := msgBit_outer hs (k := k) (by rw [hsS]; exact hFk) (wordBit (8 * o + t))
    rw [blockBit_wordBit ho ht] at hmb
    dsimp only at hmb
    rw [hsS, if_pos ha] at hmb
    have hj : (128 * (k - F) + o) / 64 < count l := by omega
    have hfj := hfs _ hj
    have hc := hc0 _ hj
    have hNj := hsegN _ hj
    rw [hmD k hk _ (wordBit_lt ho ht) _ _ hmb, hfj, show firstOf S ((128 * (k - F) + o) / 64) +
        segComps (segBits l ((128 * (k - F) + o) / 64)) - 1 = firstOf S ((128 * (k - F) + o) / 64) +
        (segComps (segBits l ((128 * (k - F) + o) / 64)) - 1) by omega,
      hdig _ (by omega) _ (by omega)]
    unfold wordBit
    congr 1
    omega
  have hD' : ∀ j < count l, ∀ t < 512,
      (fun j x => Hm.bit (zsOf c w g p (firstOf S j + (segComps (segBits l j) - 1)))
        ((compression3.run St.empty).1.2.getD (shaSub c p (firstOf S j + (segComps (segBits l j) - 1))) []) x) j
        (wordBit t) = bitL (digBytes (H512 (segMsg l row j))) t := hD
  have hFN : N = F + outerComps l := hn
  have hzh := hm_zh_in c w g p
  beta_reduce at hzh
  have hcom := seg_commit (row := row) _ hD' (fun k' => zsOf c w g p (F + k')) (fun k' => shaSub c p (F + k'))
    (fun k' hk' => hsub _ (by omega)) (fun k' hk' => zs_rows hsha (by omega))
    (fun h0 q hq => by
      try dsimp only
      rw [Nat.add_zero, hzcv _ (by omega) q hq, hcvS _ le_rfl q hq, if_neg (lt_irrefl _)]
      simp only [HmRow.midstateOf, HmRow.prefixOf, hs, Option.isSome_some, ↓reduceIte, hbits])
    (fun k' hk' q hq => by
      try dsimp only
      rw [hzcv _ (by omega) q hq, show F + (k' + 1) = F + k' + 1 by omega,
        hcvC _ (by omega) (by omega) (hin_out _ (by omega)) q hq, Nat.add_sub_cancel, hout (by omega) hq])
    (fun k' hk' o ho t ht => by
      try dsimp only
      have h1 : 128 * outerComps l ≤ 64 * count l := by unfold outerComps; omega
      rw [hzmsg _ (by omega) _ (wordBit_lt ho ht), hdigOut _ (by omega) (by omega) o ho t ht (by rw [Nat.add_sub_cancel_left]; omega),
        Nat.add_sub_cancel_left])
    _ (hm_zh c w g p hhm)
    (fun q hq => by
      rw [hzh q (by omega)]
      have e := (segCols_hm R hp (k := N) (g := g) (lt_irrefl _) q).1
      rw [← e]
      split
      · rename_i h0
        have hNF : N = F := by omega
        rw [hNF, hcvS _ le_rfl q hq, if_neg (lt_irrefl _)]
        simp only [HmRow.midstateOf, HmRow.prefixOf, hs, Option.isSome_some, ↓reduceIte, hbits]
      · rename_i h0
        have e2 : N - 1 = F + (outerComps l - 1) := by omega
        rw [hcvC _ le_rfl (by omega) (hin_out _ (by omega)) q hq, e2, hout (by omega) hq])
    (fun o ho t ht => by
      have hwb := wordBit_lt ho ht
      rw [hzh _ (by omega)]
      have e := (segCols_hm R hp (k := N) (g := g) (lt_irrefl _) (wordBit (8 * o + t))).2
      rw [← e]
      split
      · rename_i ha
        rw [hdigOut N le_rfl (by omega) o ho t ht (by rw [hFN, Nat.add_sub_cancel_left]; exact ha), hFN,
          Nat.add_sub_cancel_left]
      · rename_i ha
        have hmb := msgBit_outer hs (k := N) (by rw [hsS]; omega) (wordBit (8 * o + t))
        rw [blockBit_wordBit ho ht] at hmb
        dsimp only at hmb
        have hNF' : N - F = outerComps l := by omega
        rw [hsS, hNF', if_neg ha] at hmb
        rw [hmC N le_rfl _ hwb _ hmb, tail_bit _ _ _ ht]
        simp only [HmRow.tailOf, hs, hsS])
    (length_bytesOf w 192 (saltCol c g p))
    (fun q hq => by
      rw [hzh _ (by omega), bitL_bytesOf _ _ (by omega), saltCol, hhin.2.2]
      exact congrArg w (by omega))
  refine eq_of_bitL (by rw [length_bytesOf, length_commitString]) fun t ht => ?_
  rw [length_bytesOf] at ht
  rw [bitL_bytesOf _ _ ht, ← hcom t ht, ← hm_bc c w g p hck hhm t ht, bcCol]
  exact congrArg w (by omega)

end Any

end FlockSoundness.Discharge.Hidden
